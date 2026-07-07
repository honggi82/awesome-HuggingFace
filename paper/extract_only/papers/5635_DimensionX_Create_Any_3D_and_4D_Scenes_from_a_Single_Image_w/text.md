## DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion

Wenqiang Sun*1,3, Shuo Chen*2, Fangfu Liu*2, Zilong Chen2,3, Yueqi Duan2, Jun Zhang†1, Yikai Wang†2 1HKUST 2Tsinghua University 3ShengShu

wsunap@connect.ust.hk, {chenshuo20,liuff23,chenz122}@mails.tsinghua.edu.cn, duanyueqi@tsinghua.edu.cn, eejzhang@ust.hk, yikaiw@outlook.com,

# arXiv:2411.04928v1[cs.CV]7Nov2024

[Figure 1]

Input Image

[Figure 2]

[Figure 3]

Input Image

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

3D Scene Optim.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Spatial Director

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

Image to 3D Scene: Renderings of Novel Views

Image to Video: Motion Fix, Only Camera Changes

[Figure 26]

[Figure 27]

[Figure 28]

Input Image

Input Image

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

4D Scene Optim.

[Figure 36]

[Figure 37]

[Figure 38]

Temporal Director

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

Image to 4D Scene: Renderings of Novel View

Image to Video: Camera Fix, Only Motion Changes

Figure 1. With just a single image as input, our proposed DimensionX can generate highly realistic videos and 3D/4D environments that are aware of spatial and temporal dimensions.

### Abstract

manipulation of spatial structure and temporal dynamics, allowing us to reconstruct both 3D and 4D representations from sequential frames with the combination of spatial and temporal dimensions. Additionally, to bridge the gap between generated videos and real-world scenes, we introduce a trajectory-aware mechanism for 3D generation and an identity-preserving denoising strategy for 4D generation. Extensive experiments on various real-world and synthetic datasets demonstrate that DimensionX achieves superior results in controllable video generation, as well as in 3D and 4D scene generation, compared with previous methods. Project Page: https://chenshuo20.github. io/DimensionX/ .

In this paper, we introduce DimensionX, a framework designed to generate photorealistic 3D and 4D scenes from just a single image with video diffusion. Our approach begins with the insight that both the spatial structure of a 3D scene and the temporal evolution of a 4D scene can be effectively represented through sequences of video frames. While recent video diffusion models have shown remarkable success in producing vivid visuals, they face limitations in directly recovering 3D/4D scenes due to limited spatial and temporal controllability during generation. To overcome this, we propose ST-Director, which decouples spatial and temporal factors in video diffusion by learning dimension-aware LoRAs from dimension-variant data. This controllable video diffusion approach enables precise

*Equal Contribution. †Corresponding author.

### 1. Introduction

In the context of computer graphics and vision, understanding and generating 3D and 4D content are pivotal to create realistic visual experiences [5, 54]. By representing spatial (3D) and temporal (4D) dimensions, videos serve as a powerful medium for capturing dynamic real-world scenes. Despite substantial advancements in 3D and 4D reconstruction technologies [19, 44, 47, 52], there remains a critical shortage of large-scale 3D and 4D video datasets, limiting the potential for high-quality 3D and 4D scene generation from a single image. This scarcity poses a fundamental challenge in constructing photorealistic and interactive environments.

Fortunately, recent advancements in video diffusion models have shown considerable promise in understanding and simulating real-world environments [4, 58]. Driven by advanced video diffusion models, recent works [24, 43, 55, 60] have made attempts to leverage the spatial and temporal priors embedded in video diffusion to generate 3D and 4D content from a single image. Despite these rapid developments, existing methods either concentrate on the object-level generation with video diffusion trained on static or dynamic mesh renderings [24, 43, 55] or employ timeintensive per-scene optimization for coarse scene-level generation [60] (e.g., Score Distillation Sampling [36]). This leaves the generation of coherent and realistic 3D/4D scenes an open challenge.

In this paper, we present DimensionX, a novel approach to create high-fidelity 3D and 4D scenes from a single image with controllable video diffusion. While recent video diffusion models are capable of producing realistic results, it remains difficult to reconstruct 3D and 4D scenes directly from these generated videos, primarily due to their poor spatial and temporal controllability during the generation process. Our key insight is to decouple the temporal and spatial factors in video diffusion, allowing for precise control over each individually and in combination. To achieve the dimension-aware control, we establish a comprehensive framework to collect datasets that vary in spatial and temporal dimensions. With these datasets, we present ST-Director, which separates spatial and temporal priors in video diffusion through dimension-aware LoRAs. Additionally, by analyzing the denoising mechanics in video diffusion, we develop a training-free composition method that achieves hybrid-dimension control. With this control, DimensionX generates sequences of spatially and temporally variant frames, enabling the reconstruction of 3D appearances and 4D dynamic motions. To handle complex real-world scenes with our ST-Director, we design a trajectory-aware approach for 3D generation and an identity-preserving denoising mechanism for 4D generation. Extensive experiments demonstrate that our DimensionX outperforms previous methods in terms of visual quality and generalization for 3D and 4D scene generation,

indicating that video diffusion models offer a promising direction for creating realistic, dynamic environments.

In summary, our main contributions are:

- • We present DimensionX, a novel framework for generating photorealistic 3D and 4D scenes from only a single image using controllable video diffusion.
- • We propose ST-Director, which decouples the spatial and temporal priors in video diffusion models by learning (spatial and temporal) dimension-aware modules with our curated datasets. We further enhance the hybriddimension control with a training-free composition approach according to the essence of video diffusion denoising process.
- • To bridge the gap between video diffusion and real-world scenes, we design a trajectory-aware mechanism for 3D generation and an identity-preserving denoising approach for 4D generation, enabling more realistic and controllable scene synthesis.
- • Extensive experiments manifest that our DimensionX delivers superior performance in video, 3D, and 4D generation compared with baseline methods.

### 2. Related work

Controllable Video Diffusion. The integration of additional conditions for conditional image and video generation [62] has seen notable success. Much of the prior research in video generation has focused on injecting various control signals to guide diffusion models. For instance, some approaches control video diffusion using camera pose trajectories, often utilizing ControlNet [15, 49], Pl¨ucker coordinates embeddings [2, 21, 56], or other coordinate embeddings [50, 57]. Other methods leverage reference videos [27] or object motion trajectories [49, 64] to drive generation. Additionally, several studies have explored other control signals, such as action-based controls [12], sketches and depth maps [14]. However, these methods often rely on specific data annotations, which not only limit their scalability but also lead to performance degradation when external control signals are injected into video models. Rather than injecting additional control signals, some methods customize video diffusion models by fine-tuning on a set of reference videos with similar patterns [13, 51, 65], primarily focusing on Unet-based video diffusion models with spatial and temporal layers. However, this form of customization often lacks generalization. In this work, our method is designed for controlling the DiTbased [35] video diffusion model, CogVideoX [58], which leverages the 3D VAE and full attention to effectively blend spatial and temporal information. Our method eliminates the need for additional signal injection, offering scalability and generalization without sacrificing performance.

3D Generation with Diffusion Priors. Leveraging 2D diffusion priors for generating 3D content has revolution-

[Figure 51]

Generated Videos

[Figure 52]

S-Director

[Figure 53]

DiT Block

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Single Image

[Figure 61]

- S-Director
- T-Director

[Figure 62]

[Figure 63]

[Figure 64]

A = 𝑁(0, 𝜎 ) B = 0

[Figure 65]

[Figure 66]

FullAttention

Feedforward

𝑁×

Gate

Gate

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

T-Director

[Figure 73]

[Figure 74]

[Figure 75]

A = 𝑁(0, 𝜎 ) B = 0

[Figure 76]

[Figure 77]

[Figure 78]

(a) ST-Director

LoRA Injection

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]|
|---|

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

|[Figure 87]<br><br>[Figure 88]|
|---|

[Figure 89]

S-Director

[Figure 90]

S-Director

[Figure 91]

T-Director Single Image

[Figure 92]

Keyframe Selection

×𝐾

Single Image

|[Figure 93]|
|---|

×𝑁

Reference Video 𝑉ref

Generated Frames

Refine

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

𝑉ref

[Figure 105]

3D Scene Optim.

×𝑁

×𝐾 ×𝑁

T-Director

[Figure 106]

| |
|---|

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

4D Scene Optim.

×𝑁 S-Director

[Figure 114]

×𝐾

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

𝑉ref

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

|[Figure 127]|
|---|

[Figure 128]

×𝑁

×𝐾

Temporal-variant Frames

(b) 3D Scene Generation

Spatial-variant Videos Multi-view Videos

###### (c) 4D Scene Generation

- Figure 2. Pipeline of DimensionX. Our framework is mainly divided into three parts. (a) Controllable Video Generation with STDirector. We introduce ST-Director to decompose the spatial and temporal parameters in video diffusion models by learning dimensionaware LoRA on our collected dimension-variant datasets. (b) 3D Scene Generation with S-Director. Given one view, a high-quality 3D scene is recovered from the video frames generated by S-Director. (c) 4D Scene Generation with ST-Director. Given a single image, a temporal-variant video is produced by T-Director, from which a key frame is selected to generate a spatial-variant reference video. Guided by the reference video, per-frame spatial-variant videos are generated by S-Director, which are then combined into multi-view videos. Through the multi-loop refinement of T-Director, consistent multi-view videos are then passed to optimize the 4D scene.

ized the field of 3D generation. Score Distillation Sampling (SDS) [25, 36, 48] distills 2D diffusion priors to produce high-fidelity 3D meshes from text inputs. To further enhance the 3D consistency, several works have explored the object-level generation by conditioning 2D diffusion models on multi-view camera poses [30, 31, 39, 53, 59]. Similar techniques have been further applied in the scene-level generation [38]. More recent approaches leverage video diffusion models to generate novel views from a single image input, achieving impressive results at both the objectlevel [6, 11, 43] and scene-level [6, 11, 29]. Additionally, ReconX [28] addresses the challenge of sparse-view inputs by employing video interpolation techniques, showcasing the potential of video diffusion models for 3D scene generation. In this work, we unleash the power of video diffusion models in a novel way to generate 3D objects and scenes from a single image input, achieving the competitive performance with fewer training costs and data required.

hours to generate a 4D asset with obvious inconsistency. To enhance the consistency and generation efficiency, subsequent works [24, 55] filter out high-quality dynamic mesh data from the large-scale Objaverse dataset [8, 9]. Based on these dynamic meshes, a large number of multi-view videos are rendered to train a multi-view video diffusion model. Although these models can generate high-quality 4D multi-view videos, they mainly focus on the objectcentric setting rather than the complex scene. For the generation of 4D scenes, the lack of sufficient data poses significant challenges in producing multi-view videos that are utilized to reconstruct the whole scene. More recently, 4Real [60] firstly proposes distilling the pre-trained video diffusion prior with the SDS loss to produce a photorealistic dynamic scene. Unlike the aforementioned works, our approach emphasizes generating temporally and spatially decomposed videos, which are subsequently merged to create multi-view videos for high-quality 4D scene reconstruction.

4D Generation with Diffusion Priors. Similar to 3D generation, 4D generation has seen significant advancements with the pre-trained diffusion models, including image and video diffusion. Early works [1, 37, 46, 66] adopt the SDS technique to per-scene optimize the 4D representation from a text or image input. However, these methods tend to cost

### 3. Methodology

Given a single image, our goal is to generate high-quality 3D and 4D scenes with controllable video diffusion. To achieve the effective control in respect of spatial and tem-

poral dimensions, we first develop a systemic framework to build the dimension-variant datasets (Sec. 3.1). With the curated datasets, we introduce ST-Director to decouple the spatial and temporal basis through the dimensionaware lora, allowing the precise dimension-aware control. Furthermore, we explore the denoising mechanism during the video generation process and introduce a trainingfree dimension-aware composition for effective hybriddimension control (Sec. 3.2). To better leverage controllable video diffusion for generating high-quality scenes, we design a trajectory-aware mechanism for 3D generation and an identity-preserving denoising approach for 4D generation (Sec. 3.3 and Sec. 3.4).

#### 3.1. Building Dimension-variant Dataset

To decouple spatial and temporal parameters in video diffusion, we introduce a framework to collect spatial- and temporal-variant videos from open-source datasets. Notably, we employ a trajectory planning strategy for spatialvariant data and flow guidance for temporal-variant data.

Trajectory planning for spatial-variant data. To acquire the spatial-variant dataset, we propose reconstructing photorealistic 3D scenes and rendering videos consistent with our spatial variations. To facilitate the selection and planning of rendering paths, we need to compute the coverage range of the cameras throughout the entire scene. Given N cameras in a scene, we first compute the center C and principal axes A along the direction x, y, and z using the Principal Component Analysis (PCA) technique:

N i=1 pi

, A = SVD(P − C), (1)

C =

N

where pi denotes the position of camera i, P = {pi,1 ≤ i ≤ N} ∈ RN×3 represents the position set of N cameras, and SVD is the Singular Value Decomposition operation. Next, we need to calculate the lengths L of each axis from the projection distance D:

D = (P − C) · A (2) L = max(D) − min(D). (3)

Built on the above calculation, we have already figured out the distribution of the camera throughout the entire scene. To cope with various scenes, we establish the following rules to filter out the qualifying data: (1) Camera Distribution: We calculate the center of the scene and judge how cameras capture around the scene. (2) Bounding Box Aspect Ratio: The aspect ratio of the bounding box should meet the requirement for various S-Directors. For instance, the aspect ratio of x and y axis should not vary too greatly, which helps in selecting appropriate 360-degree surrounding videos. (3) Camera-to-Bounding Box Distance: We calculate the distance from each camera position

to the closest plane of the bounding box and prioritize data with smaller total distances to ensure better camera placement.

With the filtered dataset, it is necessary to compute the occupancy field within the scene to help us plan the feasible region for the rendering cameras. After reconstructing the entire scene’s 3DGS from multi-view images, we render multi-view images and their corresponding depth maps, and then use TSDF [7] to extract the scene’s mesh from the RGB-D data. More details can be found in appendix.

Flow guidance for temporal-variant data. To achieve the temporal control, we aim to filter the temporal-variant data to fine-tune the video diffusion model. Specifically, we leverage the optical flow [42] to filter out the temporalvariant videos. For temporal-variant videos, optical flow maps frequently exhibit extensive white regions, which could serve as an effective selection criterion.

#### 3.2. ST-Director for Controllable Video Generation

Inspired by the concept of orthogonal decomposition in linear algebra, we propose a method to decouple spatial and temporal dimensions in video generation for more precise control. We conceptualize each video frame It(u,v) as a projection from a 4D space R3 × R1, where u and v are the image coordinates in the frame, and the 4D space consists of three spatial dimensions x,y,z and a temporal dimension t. In this framework, the 4D space consists of a static background B ⊂ R3 and dynamic objects Oi(t) ⊂ R3, where i indexes the moving objects at time t. The entire scene at time t is then represented as:

S(t) = B ∪

i

Oi(t) . (4)

Each video frame at time t is, therefore, a 2D projection of this 3D scene structure onto the image plane, governed by the camera’s parameters at that moment. To formalize this, we define the projection function PC(t), which projects the 3D scene S(t) onto a 2D image:

It(u,v) = PC(t)(S(t)), (5) where C(t) represents the camera parameters at time t.

However, this reduction from 4D to 2D is typically treated as an inseparable blend of spatial and temporal dimensions in most video generation methods, complicating independent control over each dimension, which makes its difficult to recover the 4D space. To address this, we introduce two orthogonal basis directors: the S-Director (Spatial Director) and the T-Director (Temporal Director), which allow us to separate spatial and temporal variations within the video generation process. With these orthogonal basis directors, we can more flexibly control video generation, generating frames along a single axis or combining

Base

- S-Director
- T-Director

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

Step0Step3Video

Step0

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

Denoising

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

VideoStep0Step3

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

Step10

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

Video

- Figure 3. Visualization of Attention Map. The left row shows the attention maps during the denoising process of the original video diffusion model. The right row, from top to bottom, illustrates the attention map variation of S-Director and T-Director, respectively. Starting from step 0, the early denoising steps (before step 10 of total denoising step 50) have determined the outline and layouts of output videos. Specifically, the spatial component is recovered earlier than the temporal information during the denoising process.

both directors to achieve projections that represent any desired perspective within the 4D space.

##### 3.2.1 Dimension-aware Decomposition

To decompose spatial and temporal variations, we define the spatial and temporal equivalence relations that capture the behavior of points in the 4D space under different conditions. More details can be found in our appendix. From these two equivalence relations, we derive two types of quotient spaces, R4/ ∼S and R4/ ∼T. The S-Quotient Space, R4/ ∼S, captures the spatial trajectory of camera viewpoints in a video by collapsing the temporal dimension, as each frame represents different spatial perspectives at a single point in time. Conversely, the T-Quotient Space, R4/ ∼T, describes temporal object motion trajectories from a fixed camera position, collapsing the spatial dimension in terms of viewpoint, with the primary variation being temporal as objects move within the scene. Together, these two quotient spaces, representing spatial and temporal decompositions of the 4D space, enable us to interpret video as a structured decomposition of 4D space into distinct spatial and temporal components.

Building on these quotient spaces, we associate videos from our spatial-variant dataset with the S-Quotient Space, while videos from our temporal-variant dataset correspond to the T-Quotient Space. In order to train the S-Director and T-Director to generate videos specifically within these spatial and temporal structures, we employ LoRA [17],

a fine-tuning method that is both parameter-efficient and computationally light, training each director separately on one of the two datasets to guide the video diffusion model. Specifically, the S-Director is trained on the spatial-variant dataset, learning patterns in which time is held constant (S(t) = S0), thereby generating videos within the SQuotient Space, as illustrated in the top right of Fig. 3, where It(u,v) = PC(t)(S0). Similarly, the T-Director is trained on the temporal-variant dataset, learning patterns where the camera remains stationary (C(t) = C0), resulting in videos within the T-Quotient Space, as shown in the bottom right of Fig. 3, with It(u,v) = PC0

(S(t)).

##### 3.2.2 Tuning-free Dimension-aware Composition

With this orthogonal basis of directors, we achieve flexible control over video generation, where each director independently captures frame sequences along its designated axis, producing either It(u,v) = PCt

(S(0)) or It(u,v) = PC0

(S(t)), respectively. However, most videos naturally involve a blend of spatial and temporal elements, making it essential to combine both directors to capture richer, multifaceted perspectives within the 4D space, represented as It(u,v) = PCt

(S(t)). To achieve this enhanced level of control, we aim to merge the S-Director and T-Director, allowing for dynamic adjustment that aligns video generation with specific spatial, temporal, or combined spatiotemporal intents. In pursuit of this goal, we examine the underlying mechanics both of the base model’s and each director’s de-

noising process by visualizing the attention maps generated by the base model alongside both directors (as shown in Fig.

- 3). These visualizations reveal two key observations:

- Observation 1: The initial steps of the denoising process are critical for defining the generated video.

From the attention maps, we observe that during the initial steps of the denoising process, both the base model and the two directors establish foundational sketches that closely align with the final generated results. These preliminary outlines capture essential spatial and temporal structures early on, effectively setting the direction for the remaining denoising steps. Additionally, we notice a distinct difference in how these changes unfold: in the base model, both temporal and spatial alterations occur simultaneously, creating a unified evolution across both dimensions. In contrast, when utilizing the S-Director and T-Director, only one dimension changes at a time, either temporal or spatial, depending on the director.

- Observation 2: Spatial information is constructed earlier than temporal information.

Similar to findings from Motionclone [27], we observe that the synthesis of object motion remains underdeveloped in the early stages of the denoising process. Specifically, with S-Director, the attention maps reveal that the structural outlines of the final video appear much earlier than with temporal control. As evidence, Fig. 3 shows that at the same step 0 and 3 of the denoising loop, the object remains stationary with the T-Director, while the S-Director already guides the camera to move through the scene.

Based on the two observations above, we propose a training-free method, Switch-Once, a novel approach to compose diverse LoRAs. This approach combines the SDirector and T-Director to generate videos that seamlessly blend spatial and temporal information, achieving a balanced synthesis represented by It(u,v) = PCt

(S(t)). Following Observation 2, we initiate the denoising process with the S-Director to establish comprehensive camera motion across the scene. Then, as indicated by Observation 1, we switch to the T-Director after the initial steps of the denoising process (our experiments show that transitioning at the 4th or 5th step yields optimal results), thereby enhancing the quality of object motion in the final video. The resulting video, as shown in Fig. 4, demonstrates the effectiveness of this balanced approach.

#### 3.3. 3D Scene Generation with S-Director

Built upon the S-Director, our video diffusion model is able to generate long-range consistent frames from a single image, allowing for the reconstruction of photorealistic scenes. To better generalize to real-world scenarios, where spatial variations are diverse and camera trajectories are highly flexible, we introduce a trajectory-aware mechanism to handle different camera movements. Specifically, to cover a

wide range of camera trajectory patterns C(t), we train multiple types of S-Directors, each tailored to a specific camera motion. In the 3D world, camera movements are defined by 6 degrees of freedom (DoF), with each DoF allowing movement in both positive and negative directions for translation and rotation, resulting in 12 distinct motion patterns. Additionally, we also train orbital motion category S-Director, where the camera follows a smooth, circular path around the subject, capturing a unique perspective beyond the standard DoF-based movements. With diverse and controllable S-Directors, we adopt the trajectory-aware mechanism in both single-view and sparse-view settings, enabling generalizable generation of real-world 3D scenes.

Single-view Scene Generation. Given a single image I, our goal is to reconstruct the 3D scene with generated video

frames Ii Ni=1, where N represents the frame length. Although current video diffusion models have shown potential

for long video generation, the total duration still falls far short of the frame count required for real-world scene reconstruction. Specifically, the powerful open-source video diffusion model (e.g. CogVideoX [58]) currently generates a maximum of only 49 frames, whereas reconstructing a large scene (e.g. 360 degree scene) typically requires hundreds of multi-view images. To address this, we extend the video diffusion model to generate 145 frames (three times the original frame count).

Sparse-view Scene Generation. In applications where the accuracy and detail in 3D scene generation are paramount, using sparse view inputs can significantly enhance the fidelity of the generated content. In this setting, we propose incorporating a video interpolation model and an adaptive S-Director to achieve a smooth and consistent transition between the sparse views. First, we develop a video diffusion model to generate the high-quality interpolated video, which takes two images as the start and end frames. Specifically, given two-view images I1,I2 , we concatenate the noisy latent z2 = E(I2) to the ending of sequential latents, mirroring the common practice of concatenating the first frame’s latent z1 = E(I1) with its noisy counterpart. The objective function for the video diffusion process is formulated as

t∼p,ϵ∼N(0,I),t ∥ϵ − ϵθ(zt,t,z1,z2,c)∥22 ,

Ldiffusion = Ez

(6) where zt is the noisy latent sequence from training videos, and ϵθ represents the model’s prediction of the noise at timestep t, conditioned on the first and last frame latent: z1 and z2. With the interpolated video diffusion model, we then train various S-Directors to provide refined camera motion guidance, ensuring smooth and consistent transition between the sparse-view images. In particular, we tailor two key strategies to fully leverage the guidance prior carried in S-Directors: early-stopping training and adaptive

Input Image DimensionX (Ours) Dream Machine 1.6 w. Camera Motion

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

CameraZoomOutCameraStatic

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Prompt: A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. Camera static.

[Figure 202]

[Figure 203]

[Figure 204]

w. T-Director

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

Prompt: A majestic Siberian Husky stands atop a snowy mound, its sharp blue eyes filled with intelligence. The dog’s thick black-and-white fur contrasts against the soft twilight sky, where wisps of clouds drift peacefully. Bare trees surround the scene, standing as quiet guardians in the fading light. The dog stays still and the camera pull out.

w. S-Director

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

CameraOrbitRight

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Prompt: Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. Camera orbit right.

w. ST-Director

- Figure 4. Qualitative comparison in dimension-aware video generation. Given the same image and text prompt, the first row is the temporal-variant video generation (camera static), the second row is the spatial-variant video generation (camera zoom out), and the third row is the spatial- and temporal-variant video generation (camera orbit right).

trajectory-planning. Our findings reveal that training the SDirector to an early phase sufficiently achieves robust trajectory guidance, where the camera movement can be flexibly modulated with changes in input viewpoints. Furthermore, to deal with various viewpoint interpolations of spare views, we propose adaptively selecting the appropriate SDirector according to the coordinate relation between the input images. More details can be found in Appendix.

Powered by our proposed long-video diffusion model and diverse S-Directors, extensive scenes can be directly reconstructed from the generated videos. In particular, given sparse-view (i.e. as few as one) images and a chosen camera motion type, which can be a basic trajectory or a combination of these motion primitives, our video diffusion can generate a consistent long video along the specified path. To alleviate the inconsistency behind generated videos, we adopt an confidence-aware gaussian splatting procedure to reconstruct the 3D scene. Initialized with the point cloud and estimated camera poses from DUSt3R [45], 3D gaussian splatting is optimized with additional LPIPS loss [63] and confidence maps acquired from DUSt3R. We adopt the 3DGS loss as follow:

Lconf = C (λ1L1 + λssimLssim + λlpipsLlpips), (7) where C is the confidence maps, and λ1,λssim,λlpips repre-

sent the coefficients. More details can be found in appendix.

#### 3.4. 4D Scene Generation with ST-Director

Equipped with spatial and temporal controlled video diffusion, we can recover a high-quality 4D dynamic scene from a single image. A direct way is to stitch together the spatial-variant videos generated for each frame in temporalvariant videos into multi-view videos, which are then used to reconstruct the 4D scene. However, this method has a key challenges: 3D consistency. Maintaining consistency in the background and object appearance across spatialvariant videos is challenging, causing severe jitter and discontinuity in the 4D scene. To address the above issue, we propose an identity-preserving denosing strategy, including the reference video latent sharing and appearance refinement process, to enhance the consistency of all spatialvariant videos.

Given an input image I, our goal is to generate a photorealistic 4D scene with dynamic objects and high-quality backgrounds. First, we employ T-Director to generate a

temporal-variant video frames Ii Ni=1 for the input image, from which a reference frame Iref is selected to produce

corresponding spatial-variant video frames vref = Ii Ki=1, where K represents the number of cameras. Subsequently,

vref is used to guide the generation of spatial-variant videos

Input Views Ground Truth DimensionX (Ours) ViewCrafter InstantSplat

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

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

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

- Figure 5. Qualitative Comparison in sparse-view 3D generation. Given two large-angle views, our approach obviously outperforms other baselines.

across all temporal-variant video frames Ii Ni=1, which are then combined into multi-view videos Iji Ni=1

K j=1

.

Iji Ni=1 represents the temporal-variant video captured from the camera j. Despite leveraging the reference video to guide the generation process, minor shape inconsistencies still exist, causing temporal jitter or inter-view misalignment. To mitigate these issues, we introduce an extra appearance refinement process to further enforce consistency across the multi-view video. With consistent multiview videos, we choose the deformable 3D Gaussian Splatting [52] to model the dynamic scene.

Reference Video Latent Sharing. Through our empirical study, we propose choosing the reference frame based on the dynamic object’s mask size and the magnitude of optical flow values, allowing us to acquire a frame that best encompasses the dynamic object. With the chosen reference frame Iref, S-Director is applied to produce the corresponding spatial-variant video vref. Applying the forward diffusion process on vref, we derive the noisy latent code zref as following:

###### zref = √αtz0 + √1 − αtϵ, ϵ ∼ N(0,1), (8)

where z0 = E(vref), representing the compressed latent by the encoder E, and √αt determines the strength of using the reference video. Starting from the same initialization latent zref, all frames are subsequently denoised to produce spatial-variant videos with strong coherence. Moreover, to further enhance consistency among these frames, we pro-

pose blending the denoised zt of each frame with the corresponding reference video latent zref,t at the early denoising steps:

zt = λzt + (1 − λ)zref,t, (9) where λ is an adjustable hyperparameter.

Appearance Refinement. To further enhance the stability and consistency between the combined multi-view videos, the appearance refinement process is applied to the dynamic video from each viewpoint. Inspired by the image-to-image translation SDEdit [32], we apply random noise to each

multiview video vj = Iji Ni=1, and perform multi-step denoising, acquiring smooth and high-quality videos with the

video diffusion prior:

###### vjrefine = fθ (vj + ϵ(t0);t0,c), (10)

where t0 represents the forward diffusion timestep, and vjrefine is the refined video with the denoise function fθ of T-Director. In addition, we repeat the refine process during the middle timestep to enhance the smoothness.

Having acquired consistent multiview videos Iji Ni=1

K j=1

, we use the deformable 3dgs [52] to reconstruct the 4d scene. Following the previous reconstruction methods, we apply the L1, total-variational, and ssim loss to optimize the scene:

###### L = L1 + Ltv + Lssim. (11)

Tank and Temples MipNeRF360 LLFF DL3DV PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Methods

ZeroNVS [38] 12.31 0.301 0.567 15.84 0.327 0.536 15.62 0.497 0.354 12.39 0.251 0.559 ViewCrafter [61] 15.18 0.499 0.319 15.65 0.404 0.378 17.56 0.620 0.337 14.78 0.422 0.417

Single-View

Ours 17.11 0.613 0.199 18.91 0.527 0.333 20.38 0.744 0.200 18.28 0.642 0.215

DNGaussian [22] 12.13 0.292 0.511 15.21 0.127 0.632 17.51 0.586 0.409 14.99 0.286 0.432 InstantSplat [10] 18.70 0.634 0.258 16.80 0.574 0.296 22.33 0.818 0.149 18.30 0.691 0.222 ViewCrafter [61] 18.76 0.637 0.216 18.49 0.691 0.212 21.60 0.823 0.155 19.19 0.686 0.196

Sparse-View

Ours 20.42 0.668 0.185 20.21 0.713 0.184 25.11 0.913 0.067 21.69 0.780 0.124

Table 1. Quantitative comparison of single-view and sparse-view scenarios. Our approach outperforms other baselines in all metrics both in terms of single-view and sparse-view settings.

### 4. Experiment

In this section, we conduct extensive experiments on realworld and synthetic datasets to assess the controllability of our DimensionX, as well as its 3D and 4D scene generation capabilities using ST-Director. We begin with a comprehensive illustration about our experimental details (Sec. 4.1). Then, in Sec. 3.2, we provide both quantitative and qualitative evaluation in controllable video generation. Following that, we report the quantitative and qualitative results of our approach in comparison to other baselines under various scenarios, including single-view and sparse-view 3D generation (Sec. 4.3). Subsequently, we present the 4D scene generation results of our approach (Sec. 4.4). Finally, we conduct various ablation studies to evaluate the effectiveness of our design (Sec. 4.5).

#### 4.1. Experimental Setup

Implementation Details. We choose the open-source I2V model CogVideoX [58], which adopts the diffusion transformer architecture, as our video diffusion model. For the ST-Director training, we set the LoRA rank to 256, and finetune the LoRA layers with 3000 steps at the learning rate 1e-3. To enlarge the video frames, we first modify the RoPE [41] positional embedding to extend the video length to 145 frames and decrease the resolution to 480×320. For the training of video interpolation models, we first full finetune the base model for 2,000 steps at the learning rate 5e-5, then we train the S-Director using the same LoRA configuration but for only 1,000 steps. During the inference stage, we adopt the DDIM sampler [40] with classifier-free guidance [16]. In 3DGS optimization stage, we select the first and end frames of generated videos to produce the initialization point cloud and 49 frames from videos to optimize the scene. We follow the original 3DGS pipeline [19], and set the optimization step to 7000 steps. The hyperparameters λ1, λssim, and λlpips are 0.8, 0.2, and 0.3, respectively. During the deformable 3DGS phrase, we use the generated 32-view videos as the training data, and apply the original deformable 3DGS to reconstruct the 4D scene.

Datasets. In our whole framework, our video diffusion

Consistency↑ Dynamic ↑ Aesthetic ↑ CogVideoX [58] 93.56 11.76 57.81

Dream Machine 1.6 93.69 38.24 68.96 Ours 97.69 47.06 70.82

Table 2. Qualitative comparison for controllable video generation. Our DimensionX outperforms baseline models in all metrics, including the Consistency, Dynamic, and Aesthetic scores.

model is mainly trained on three datasets: DL3DV-10K [26], OpenVid [34], and RealEstate-10K [67]. OpenVid1M [34] is a curated high-quality open-sourced video dataset, including 1 million video clips with diverse motion dynamics and camera controls. DL3DV-10K [26] is a widely-collected 3D scene dataset with high-resolution multi-view images, including diverse indoor and outdoor scenes. RealEstate-10K is a dataset from youtube, mianly including the captures of indoor scenes. Applying our designed data collection framework, we build the dimensionvariant dataset from DL3DV-10K and OpenVid. We select 100 high-quality temporal-variant videos from OpenVid to train T-Director. For each S-Director type, 100 videos are rendered according to the specific camera trajectory to train the corresponding LoRA. To enlarge the video frame, we filter high-quality videos exceeding 145 frames from the RealEstate-10K [67] and OpenVid [34] to full fine-tune the video diffusion model. With the same dataset, we fine-tune a video interpolation model with the first and end frame guidance. To further verify the 3D generation ability of our DimensionX, we compare our approach with other baselines on Tank-and-Temples [20], MipNeRF360 [3], NeRFLLFF [33], and DL3DV-10K [26].

#### 4.2. Controllable Video Generation.

Baselines and Evaluation Metrics. We compare our DimensionX with the original CogVideoX [58](open-source) and Dream Machine 1.6 (closed-source product). We collect hundreds of images as our evaluation dataset. Following the previous benchmark VBench [18], we evaluate the Subject Consistency, Dynamic Degree, and Aesthetic Score of generated videos as our metrics.

Input View Video Novel View Video

Input Image

|[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]|
|---|

|[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|
|---|

[Figure 258]

|[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]|
|---|

|[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]|
|---|

|[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]|
|---|

|[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]|
|---|

[Figure 271]

|[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]|
|---|

|[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]|
|---|

|[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]|
|---|

|[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]|
|---|

[Figure 284]

|[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]|
|---|

|[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]|
|---|

- Figure 6. Qualitative results of 4D scene generation. Given a real-world or synthetic single image, our DimensionX produces coherent and intricate 4D scenes with rich features.

Quantitative and Qualitative Comparison. The qualitative result in Table 2 demonstrates the impressive performance of our approach, including the better visual quality and 3D consistency. As shown in Fig. 4, we can observe that our DimensionX achieves effective decomposition of spatial and temporal parameters of video diffusion model, while Dream Machine cannot decouple the dimension-aware control, even though we utilize the camera motion and prompt constraint. Moreover, for the hybriddimension control, including spatial and temporal motion, in comparison to Dream Machine, our DimensionX generates more impressive and dynamic videos. Both quantitative and qualitative results indicate that our approach can create controllable videos while maintaining the dynamic motions and subject consistency.

#### 4.3. 3D Scene Generation.

Baselines and Evaluation Metrics. In the single-view setting, we compare our approach with two generative methods: ZeroNVS [38] and ViewCrafter [61]. For the sparseview scenario, we select two sparse-view reconstruction methods and one sparse-view generation baseline, including: DNGaussian [22], InstantSplat [10], and ViewCrafter [61]. We adopt PSNR, SSIM, and LPIPS as the metric for

our quantitative results. Specifically, In both single-view and sparse-view settings, we begin by reconstructing the 3D scene from the given images, followed by calculating the metrics using renderings from novel views.

Quantitative and Qualitative Comparison. The quantitative comparison results are presented in Table 1. We can observe that DimensionX outperforms the baselines in all metrics, demonstrating the impressive performance of our approach. As presented in Fig. 5, in both single-view (More details can be found in our appendix.) and sparse-view settings, our approach can reconstruct high-quality 3D scenes, while other baselines fail to handle the challenging cases.

#### 4.4. 4D Scene Generation

We evaluate our DimensionX on both real-world and synthetic datasets. Specifically, we adopt the Neu3D [23], which contains high-resolution multi-view video of different scenes, to verify the performance of our approach in real-world video to 4D generation. As shown in Fig. 6, given a single image, our DimensionX generates highly consistent dynamic videos from large-angle novel views.

w. S-Director w/o. S-Director Input Images Ground Truth Generated video Rendering Generated video Rendering

PSNR: 17.41 SSIM: 0.6024 LPIPS: 0.1890

PSNR: 15.25

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

SSIM: 0.4316 LPIPS: 0.3882

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

Figure 7. Ablation study on the sparse-view 3D generation: The absence of S-Director results in lower reconstruction quality.

#### 4.5. Ablation Study

Trajectory-aware mechanism for 3D generation. In sparse view 3D generation, we leverage S-Director to guide video interpolation model. As illustrated in Fig. 7, when handling the large-angle sparse view, the absence of SDirector often results in the ”Janus problem”, where multiple heads are generated, significantly degrading reconstruction quality.

Identity-preserving denoising strategy for 4D generation. In 4D scene generation, we conduct experiments on real-world images to analyze our identity-preserving denoising for 4D scene generation. As shown in Fig. 8, we ablate the design of reference video latent sharing and appear refinement in terms of the consistency among different frames of a novel view. Specifically, we can observe that directly combing per-frame videos causes severe inconsistency, including the background and subject shape. Through the reference video latent sharing, the global background and appearance exhibit a high consistency across different frames. Building on reference video latent sharing, appearance refinement enhances the coherence of appearance details.

Input View Video Base + Reference Video Full

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

Figure 8. Ablation study on 4D generation We ablate the design of reference video latent sharing and appearance refinement.

### 5. Conclusion

In this paper, we introduce DimensionX, a novel framework to create photorealistic 3D and 4D scenes from only

a single image with controllable video diffusion. Our key insight is to introduce ST-Director to decouple the spatial and temporal priors in video diffusion models by learning the dimension-aware LoRA on dimension-variant datasets. Furthermore, we investigate the denoising process of video diffusion and introduce a tuning-free dimensionaware composition to achieve the hybrid-dimension control. Powered by the controllable video diffusion, we can recover accurate 3D structures and 4D dynamics from the sequential generated video frames. To further enhance the generalization of our DimensionX in real-world scenes, we tailor a trajectory-aware strategy for 3D scene generation and an identity-aware mechanism for 4D scene generation. Extensive experiments on various real-world and synthetic datasets demonstrate that our approach achieve the stateof-the-art performance in controllable video generation, as well as 3D and 4D scene generation.

Limitations and future work. Despite the remarkable achievements, our DimensionX is limited by the diffusion backbone. Although current video diffusion models are capable of synthesizing vivid results, they still struggle with understanding and generating subtle details, which restricts the quality of the synthetic 3D and 4D scenes. Additionally, the prolonged inference procedure of video diffusion models hampers the efficiency of our generation process. In the future, it is worthy to explore how diffusion models can be integrated for more efficient end-to-end 3D and 4D generation. We believe that our research provides a promising direction to create a dynamic and interactive environment with video diffusion models.

### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006, 2024. 3
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffu-

- sion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 2
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 9
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [5] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890, 2024. 2
- [6] Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3d: Video diffusion models are effective 3d generators. arXiv preprint arXiv:2403.06738, 2024. 3
- [7] Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pages 303–312, 1996. 4
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 3
- [9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36, 2024. 3
- [10] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2024. 9, 10
- [11] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 3
- [12] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024. 2
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2025. 2
- [15] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling

- camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 9
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 9
- [19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2, 9

- [20] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: benchmarking large-scale scene reconstruction. 36(4), 2017. 9
- [21] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. arXiv preprint arXiv:2405.17414,

2024. 2

- [22] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20775–20785,

2024. 9, 10

- [23] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521–5531, 2022. 10
- [24] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024. 2, 3
- [25] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 3
- [26] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 9
- [27] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin.

- Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 2, 6
- [28] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767, 2024. 3
- [29] Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024. 3
- [30] Fangfu Liu, Diankun Wu, Yi Wei, Yongming Rao, and Yueqi Duan. Sherpa3d: Boosting high-fidelity text-to-3d generation via coarse 3d prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20763–20774, 2024. 3
- [31] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970–9980, 2024. 3
- [32] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 8
- [33] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1–14, 2019. 9
- [34] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 9
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2

- [36] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 3
- [37] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142,

2023. 3

- [38] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994, 2023. 3, 9, 10
- [39] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 3
- [40] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 9

- [41] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 9

- [42] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 4

- [43] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2025. 2, 3
- [44] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764,

2024. 2

- [45] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 7
- [46] Xinzhou Wang, Yikai Wang, Junliang Ye, Zhengyi Wang, Fuchun Sun, Pengkun Liu, Ling Wang, Kai Sun, Xintong Wang, and Bin He. Animatabledreamer: Text-guided nonrigid 3d model generation and reconstruction with canonical score distillation. arXiv preprint arXiv:2312.03795, 2023. 3
- [47] Yikai Wang, Xinzhou Wang, Zilong Chen, Zhengyi Wang, Fuchun Sun, and Jun Zhu. Vidu4d: Single generated video to high-fidelity 4d reconstruction with dynamic gaussian surfels. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2
- [48] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [49] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [50] Daniel Watson, Saurabh Saxena, Lala Li, Andrea Tagliasacchi, and David J Fleet. Controlling space and time with diffusion models. arXiv preprint arXiv:2407.07860, 2024. 2
- [51] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 2
- [52] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Com-

- puter Vision and Pattern Recognition (CVPR), pages 20310– 20320, 2024. 2, 8
- [53] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024. 3
- [54] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, YanPei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d gaussian splatting. Computational Visual Media, 10(4):613– 642, 2024. 2
- [55] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 2, 3
- [56] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 2
- [57] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2
- [58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 6, 9
- [59] Junliang Ye, Fangfu Liu, Qixiu Li, Zhengyi Wang, Yikai Wang, Xinzhou Wang, Yueqi Duan, and Jun Zhu. Dreamreward: Text-to-3d generation with human preference. arXiv preprint arXiv:2403.14613, 2024. 3
- [60] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Laszlo A Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. arXiv preprint arXiv:2406.07472, 2024. 2, 3
- [61] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 9, 10
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023. 2
- [63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7
- [64] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. arXiv preprint arXiv:2407.21705, 2024. 2
- [65] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng

Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465,

- 2023. 2

- [66] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603,

2023. 3

- [67] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018. 9

