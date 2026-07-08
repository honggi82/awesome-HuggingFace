## DreamGaussian4D: Generative 4D Gaussian Splatting

Jiawei Ren∗1 Liang Pan∗2 Jiaxiang Tang1,3 Chi Zhang1 Ang Cao4 Gang Zeng3 Ziwei Liu1

1 S-Lab, Nanyang Technological University 2 Shanghai AI Laboratory 3 Peking University 4 University of Michigan

# arXiv:2312.17142v3[cs.CV]10Jun2024

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

Input Generated 4D Model

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

0s 1min 4min30s

Optimization Progress

|[Figure 26]|[Figure 27]|[Figure 28]| |[Figure 29]|[Figure 30]|
|---|---|---|---|---|---|

Composited Scene

Figure 1: DreamGaussian4D generates 4D contents in minutes by leveraging 4D Gaussian Splatting. Exported meshes can be efficiently composited and rendered in 3D engines (e.g., Blender or Unreal Engine).

### Abstract

4D content generation has achieved remarkable progress recently. However, existing methods suffer from long optimization times, a lack of motion controllability, and a low quality of details. In this paper, we introduce DreamGaussian4D (DG4D), an efficient 4D generation framework that builds on Gaussian Splatting (GS). Our key insight is that combining explicit modeling of spatial transformations with static GS makes an efficient and powerful representation for 4D generation. Moreover, video generation methods have the potential to offer valuable spatialtemporal priors, enhancing the high-quality 4D generation. Specifically, we propose an integral framework with two major modules: 1) Image-to-4D GS - we initially generate static GS with DreamGaussianHD, followed by HexPlane-based dynamic generation with Gaussian deformation; and 2) Video-to-Video Texture Refinement - we refine the generated UV-space texture maps and meanwhile enhance their temporal consistency by utilizing a pre-trained image-to-video diffusion model. Notably, DG4D reduces the optimization time from several hours to just a few minutes, allows the generated 3D motion to be visually controlled, and produces animated meshes that can be realistically rendered in 3D engines.

∗Equal contribution

Preprint. Under review.

### 1 Introduction

Remarkable progress has been witnessed in generative models, demonstrating significant recent advancements and innovations in generating diverse digital content, such as 2D images [40, 42], videos [52, 3], and 3D scenes [21, 19, 47]. While a few recent research works [44, 20, 59, 2] have been devoted to dynamic 3D (i.e., 4D) generation, achieving consistency and high quality in the generation of 4D scenes is far from being fully resolved.

4D scenes are often represented by dynamic Neural Radiance Fields (NeRF), which are expected to show consistent appearance, geometry, and motions from arbitrary viewpoints. By combining the benefits of video and 3D generative models, MAV3D [44] achieves text-to-4D generation by distilling text-to-video diffusion models on a HexPlane [4]. Consistent4D [20] introduces a videoto-4D framework to optimize a Cascaded DyNeRF for 4D generation from a statically captured static video. With multiple diffusion priors, Animate124 [59] could animate a single in-the-wild image into 3D videos through textual motion descriptions. Using a hybrid SDS, 4D-fy [2] achieves compelling text-to-4D generation based on multiple pre-trained diffusion models. However, all the aforementioned methods [44, 20, 59, 2] need several hours to generate a single 4D NeRF, which limits their application potential. Furthermore, it is usually challenging to effectively control their generated motions. This dissatisfaction comes from several factors. First, the underlying implicit 4D representations of the aforementioned methods are not efficient enough, suffering from slow rendering speed and less regularized motions. Second, the stochastic nature of video SDS adds difficulties to convergence and introduces instability and artifacts to the final results.

In this work, we introduce DreamGaussian4D (DG4D), which can efficiently generate dynamic scenes from a single image (or a video sequence) in just a few minutes. Our core idea is to leverage explicit 4D representations and video-driven optimization to accelerate the optimization process for 4D generation. For 4D representations, we employ 3D Gaussian Splatting (GS) [22] to explicitly represent the static

Table 1: 4D Generation in Several Minutes. DG4D generates promising 4D assets with significant improvement in convergence speed (iteration numbers) and rendering speed (seconds needed for each iteration).

Method Time Iterations

MAV3D [44] 6.5 hr 12k Animate124 [59] - 20k Consistent4D [20] 2.5 hr 10k 4D-fy [2] 23 hr 120k Dream-in-4D [60] 10.5 hr 20k AYG [26] - 20k

- 3D scene and HexPlane to describe dynamic displacement maps. First, we introduce a new training recipe and propose the image-to-3D framework DreamGaussianHD to initialize a static 3D GS, effectively alleviating the underoptimization problem in DreamGaussian [47]. We then optimize a HexPlane to introduce realistic motions into the static 3D GS by predicting Gaussian deformations at each timestamp.

Ours (Image-to-4D GS Generation) 6.5 mins 0.7k

+ Video-to-Video Texture Refinement 10 mins 0.75k

Unlike the commonly used score distillation from video diffusion models, our model learns the motion from a driving video. In practice, users can select a desired driving video, which can be efficiently generated (typically in about one minute) using an image-to-video generation model (e.g., SVD [3] or Sora [34]). This approach allows flexible selection and control of the expected motions in 4D generation while maintaining a high level of diversity in motion generation. Subsequently, the generated 4D GS could be exported into an animated mesh sequence. To further improve the temporal coherence of the generated sequence, we can optionally refine the per-frame texture map by using a Video-to-Video optimization pipeline with an image-to-video diffusion model. In total, generating a 14-frame 4D animation takes approximately 6.5 minutes or 10 minutes with the optional texture refinement (see Table 1). The generated 4D content by DG4D can be visualized in Fig. 1.

In summary, our contributions are as follows: (1) Principled Image-to-4D Generation Framework: We propose a principled image-to-4D generation framework that employs a synergy of imageconditioned 3D-generation and video-generation models. This allows direct control and selection of the expected 3D content and its motions, enabling high-quality and diverse 4D generation. (2) Explicit 4D Representation: By utilizing Gaussian Splatting and HexPlane, the proposed DG4D explicitly represents 4D scenes with 3D GS and their deformations at different timestamps. The explicit modeling of spatial transformation significantly reduces the 4D generation time from several hours to just a few minutes. (3) Video-to-Video Texture Refinement: We introduce a Video-toVideo texture refinement strategy that further enhances the quality of exported animated meshes by

maintaining temporal consistency, making the framework more friendly to deploy in a real-world setting. (4) Superior Performance: Experimental results show that DG4D can effectively generate diverse 4D content with higher quality and shorter optimization time than existing methods.

### 2 Related works

- Image-to-3D Generation. Image-to-3D generation could be regarded as a conditional generation task that creates 3D assets from a single reference image, often employing advanced techniques such

- as diffusion models [18]. Point-E [33] and Shap-E [21] are trained to generate 3D point clouds or Neural Radiance Fields (NeRF) [32] based on image features, but their quality is limited by spatial resolution and the availability of high-quality 3D datasets. Some methods [31, 48] leverage powerful

- 2D diffusion models [40, 29, 9] and adapt them to 3D using score distillation sampling (SDS) [37]. For instance, Magic123 [39] integrates both image and text inputs to distill high-quality 3D models via NeRF, while DreamGaussian [47] reduces optimization times with Gaussian splatting [22]. Alternatively, the challenge can be approached as a single-view 3D reconstruction task, where various works [55, 8, 7, 50, 12, 45] employ auto-encoder structures to learn 3D priors for this complex problem, typically constrained to specific categories of synthetic objects [5]. Recent developments include One-2-3-45 [28, 27] which uses 2D diffusion models [29, 43] to generate multi-view images for training an efficient reconstruction model, and LRM [19] which utilizes a transformer-based architecture to enhance scale on large datasets [10, 57] by directly regressing a triplane-based NeRF.

4D Representations. Significant advancements have been made in the representation of dynamic 3D scenes (4D scenes). Research has approached 4D scene representation either as a function of spatial coordinates x,y,z with an additional time dimension t or latent codes [54, 15, 23, 24], or by modeling 4D scenes through deformation fields combined with static canonical 3D scenes [38, 35, 36, 11, 49, 58, 25]. A primary challenge in 4D representations is the extensive computational time, often requiring many hours for a single scene. To address this, many approaches achieve impressive 4D reconstruction results by utilizing explicit or hybrid representations, including planar decomposition for 4D spacetime grids [4, 14, 41], hash representations [51], and other structures [13, 1, 17]. Recently, Gaussian Splatting [22] has gained attention for its balance of speed and quality in reconstructions. The extension of static Gaussian Splatting into dynamic contexts, such as Dynamic 3D Gaussians [30] and 4D Gaussian Splatting [53, 56], utilizes deformation networks to predict time-dependent adjustments, offering a promising direction in 4D scene representation.

4D Generation. The objective of 4D generation is to create dynamic 3D scenes, applicable across various graphics fields such as animation, gaming, and virtual reality. Current methodologies employ text-to-video diffusion models to distill 4D content [44], notably using advanced 4D representations like Hexplane [4] or K-plane [14]. These models synthesize camera trajectories and calculate SDS on the rendered videos. Recent advancements have aimed to enhance photorealism by integrating multiple diffusion priors, providing a more robust supervision signal [2, 60]. However, the excessive optimization time and computational demands limit practical deployment. Additionally, the 3D content often lacks motion diversity and control. Recent initiatives attempt to generate 4D models from single images, yet they still face challenges related to lengthy optimization and insufficient motion control. Notably, Consistent4D [20] offers a novel approach by deriving 4D models from static input videos, closely aligning with our work. Our study differs by focusing on image-conditioned video generation, enabling varied motions using the same static model. Concurrently, research like AYG [26] utilizes Gaussian Splatting for high-fidelity 4D generation, but our method achieves comparable results with fewer than 5% of the optimization iterations required by these approaches.

- 3 Our Approach

As illustrated in Fig. 2, DG4D mainly consists of two stages, 1) Image-to-4D GS Generation, and 2) Video-to-Video Texture Refinement. In the first stage, we represent the dynamic 4D scene using static 3D GS and its deformations. From an input image, a static 3D GS is generated using the enhanced method, DreamGaussianHD. Subsequently, Gaussian deformations at various timestamps are estimated by optimizing a time-dependent deformation field on the static 3D Gaussians, ensuring that the shape and texture of each deformed frame correspond to every frame of the video generated based on the input image. This process results in an animated mesh sequence. In the second stage, our objective is to refine the texture maps of this mesh sequence to enhance temporal consistency. This

I) Image-to-4D GS Generation Input Image

Driving Video

[Figure 31]

|[Figure 32]|Image-to-Video Diffusion|
|---|---|
| | |

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

[Figure 37]

MSE Reference View

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

+Multiview Opt.

Random View

Deform

[Figure 45]

+Background Fix.

[Figure 46]

[Figure 47]

DreamGaussianHD

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

SDS

3D-aware

3D GS: (𝑝 , 𝑟 , 𝑠 ) Image Diffusion Prior

4D GS: (∆𝑝 , ∆𝑟 , ∆𝑠 )

###### II) Video-to-video Texture Refinement

Rendered Video

Synthetic Camera Trajectory

Video Diffusion Prior

[Figure 52]

[Figure 53]

[Figure 54]

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Render

[Figure 64]

Add Noise

|[Figure 65]|
|---|

mesh

T=0.7

MSE

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

UV Input Image map

Denoise

Multi-step

Refined Video

Figure 2: DreamGaussian4D Framework. We first obtain a static 3D GS model using DreamGaussianHD and a driving video with an image-to-video diffusion model. We then optimize a deformation network that learns to deform the static 3D GS at different time stamps, supervised by the MSE loss to the driving video and SDS losses. Finally, per-frame meshes can be exported and the texture maps can be refined with a video-to-video pipeline.

is achieved through a Video-to-Video refinement pipeline that utilizes a pre-trained image-to-video diffusion model to improve texture quality. The entire framework can be completed in approximately 10 minutes, with the first stage requiring about 6.5 minutes and the second stage about 3.5 minutes.

- 3.1 Image-to-4D GS Generation

- 3.1.1 DreamGaussianHD for Static Generation

Despite its rapid optimization speed, the original DreamGaussian [47] introduces significant blurriness to the unseen areas of static models, as illustrated in Fig. 3. This blurriness adversely affects the subsequent dynamic optimization process. Therefore, we first design better implementation practices to reliably enhance the image-to-3D generation quality of DreamGaussian at the cost of a reasonable increase in optimization time. We summarize these improved practices as DreamGaussianHD.

[Figure 75]

[Figure 76]

[Figure 77]

- (1) Multi-view Optimization. Apart from the reference view, DreamGaussian typically samples one random view at each optimization iteration for SDS. This approach covers only part of the Gaussians and leads to unbalanced optimization and convergence. As observed in previous works, increasing the number of sampled views (batch size) at each optimization step can significantly mitigate this issue [37, 6]. Sampling 16 views, for instance, yields high-quality geometry in the unseen regions of the 3D Gaussians. As a trade-off, this approach incurs an increase in memory usage during SDS computation and lengthens the optimization duration.
- (2) Fixing Background Color. DreamGaussian uniformly samples the background color from black and white. However, most 3D-aware image diffusion models render the training objects with a white background. We have observed that renderings with a black background introduce additional noise into the optimization process, ultimately resulting

- a) DreamGaussian
- b) + Multiview optim.
- c) + Fixed background

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Input Image

View angle

Figure 3: DreamGaussianHD. Multi-view Optimization significantly improves the texture and geometries. Fixing Background Color further enhances the level of detail.

in blurriness. By consistently setting the background color to white, we achieve more detailed and refined results in the optimized 3D GS.

#### 3.1.2 Gaussian Deformation for Dynamic Generation

Unlike other methods [2] that perform SDS supervision using a video diffusion model, we propose to use explicit supervision from any video depicting the input image. This video can be created by artists like those in videoto-4D [20], or generated automatically from an image-to-video model. In this study, we mainly utilize Stable Video Diffusion [3], the current open-source SoTA video generation model, to generate videos from input images:

Space

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

Grid

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Feature

[Figure 101]

[Figure 102]

Query

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

| |
|---|
| |
| |
| |
| |
| |

[Figure 109]

XZ

[Figure 110]

|∆ Position<br><br>[Figure 111]<br><br>|
|---|

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

XY

[Figure 118]

[Figure 119]

[Figure 120]

YZ

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

|∆ Rotation<br><br>|
|---|

[Figure 127]

[Figure 128]

[Figure 129]

MLP

[Figure 130]

Space-Time

[Figure 131]

Static 3D GS XYZ

[Figure 132]

a

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

|∆ Scaling|
|---|

|3D GS at 𝒕|
|---|

Zero

Time Initialization

Query

Zt

Xt Yt

𝒕 ∈ {𝟏,…𝑻}

HexPlane

Figure 4: Deformation Field with HexPlane. HexPlane provides an explicit mapping for deformation. The resolution provides a regularization to the generated motion.

{IRef}Tτ=1 = fψ(ϵ;IInput), (1) where IInput represents the input im-

age, {IRef}Tτ=1 is the driving video, ϵ denotes random noise, and fψ is the image-to-video diffusion model. Since our method does not rely on the video diffusion model later, users can choose highquality videos with better temporal consistency and motion generated by different random seeds, which enables better visual controllability and diversity for image-to-4D generation.

HexPlane for 4D Representation. To further augment the static 3D Gaussians into dynamic ones, we train a HexPlane [4] as the 4D representation to predict the position displacement, rotation, and scale of each Gaussian given its location (x,y,z) and timestamp t. As shown in Fig. 4, HexPlane [4] decomposes a 4D field into six feature planes, spanning each pair of coordinate axes. Besides the fast speed, this decomposition represents 4D fields as weighted summations of a set of learnable 4D basis functions, which inherently regularizes features of 4D fields and ensures their smoothness (see original paper [4] for this discussion). In our case, controlling the spatial and temporal axis resolution of HexPlane allows us to regularize the local spatial rigidness and temporal abruptness of motions, leading to better results. Specifically, we extract features from HexPlane representation and regress position displacement, rotation change, and scale changes from an MLP decoder.

Static-to-Dynamic Initialization. However, randomly initializing the deformation network can cause a divergence between the dynamic and static models, leading to convergence at a suboptimal mode. As exemplified in Fig. 5, the back of the panda was black and white in the static stage. When initialized differently, the back could turn fully black after dynamic optimization. To mitigate this, the deformation model should be initialized to predict zero deformation

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

- a) static
- b) w/o zero-init
- c) w/ zero-init

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- at the start of training. Note that if we initialize all the network weights with zero, the weights cannot be updated due to the zero gradient issue. Therefore, we utilize zero-initialization for the final few linear layers and employ several residual connections to ensure that the weights and inputs of each layer are not simultaneously zero.

time

Figure 5: Zero Initialization. Non-zero initialization can induce undesirable random changes, such as turning the back of the panda completely black.

Deformation Field Optimization. We optimize the deformation field given the driving video from the reference view. We fix the camera to the reference view, and minimize the Mean Squared Error (MSE) between the rendered image and the driving video frame at each timestamp:

T

1 T

||f(ϕ(S,τ),oRef) − IRefτ ||22, (2)

LRef =

τ=1

where IRefτ is the τ-th frame in the video, oRef is the reference view point and f is the rendering function. To propagate the motion from the reference view to the whole 3D model, we leverage

Zero-1-to-3-XL [9] to predict the deformation of the unseen part. Although image diffusion models only perform per-frame prediction, the temporal consistency can be mostly preserved thanks to the regularization provided by HexPlane. Similar to the training practice in DreamGaussianHD, multiple views are sampled for each time step.

∇ϕLSDS = Et,τ,ϵ,o[(ϵθ(Iˆ;t,IRefτ ,o) − ϵ)

∂I ∂ϕ

], and Iˆ = f(ϕ(S,τ),o), (3)

where ϵ is a random noise, ϵθ is the noise predictor of a 3D-aware image diffusion model, and o is a random viewpoint. Thanks to the static model initialization, we can start the SDS at a lower noise

level. Specifically, we start SDS with a Tmax = 0.5, which is lower than the common practice where Tmax = 0.98. Optionally, the static 3D GS can be fine-tuned in the optimization, and we freeze the static 3D GS by default

#### 3.2 Video-to-Video Texture Refinement

Following the initial 4D GS generation stage, we obtain a continuous sequence of 3D scenes. Mesh extraction for each frame could be conducted similarly to DreamGaussian [47], by performing local density queries and color back-projection. The resulting 3D mesh sequences exhibit impressive generative quality, especially in 3D geometries and 4D movements, closely aligning with the 2D animations chosen by the user. However, the textures in the meshes derived from 3D Gaussians tend to be blurry due to SDS ambiguity.

To address this, similar to DreamGaussian, a texture refinement module could be introduced, allowing users to finetune UV-space textures. Unlike the per-frame UV-space refinement in DreamGaussian, which denoises each frame separately and may cause flickering in adjacent frames, we employ a Video-to-Video pipeline to enhance the UV-space texture map while maintaining temporal consistency. This process begins with synthesizing a camera trajectory, where the camera moves at a constant speed along 0 elevations from a randomly chosen horizontal angle. We then render the video and introduce noise at level 0.7 to it. Finally, an image-to-video diffusion model is utilized to transform this noisy video into a clean, denoised version:

[Figure 151]

[Figure 152]

[Figure 153]

- a) I2I
- b) V2V

[Figure 154]

[Figure 155]

[Figure 156]

time

Figure 6: Video-to-Video Texture Refinement. DreamGaussian-like image-to-image (I2I) optimization leads to poor temporal consistency and flickering in adjacent frames. Video-to-Video (V2V) optimization alleviates the issue.

{IRefined}Tτ=1 = fψ({Iˆ}Tτ=1 + ϵ;IInput), (4)

where ϵ is a random noise at the specified level and {Iˆ}Tτ=1 is the rendered video. The MSE loss is computed between the two videos:

LRefine = ||{Iˆ}Tτ=1 − {IRefined}Tτ=1||22. (5)

The loss is then back-propagated to improve the texture maps at all time steps. As shown in Fig. 6, the image-to-image optimization has no temporal consistency restriction since the per-frame refined meshes have individual texture maps. In contrast, the utilized video-to-video texture refinement provides temporal consistency, which results in smoother temporal changes.

### 4 Experiments

Implementation Details. We run all experiments on a single 80 GB A100 GPU. We implement the DG4D framework on the open-source repositories DreamGaussian [47] and 4D Gaussian Splatting [53]. For driving video generation, we use Stable Video Diffusion to generate 14 frames. For static optimization, we run 500 iterations with a batch size of 16 for 2 minutes. We linearly decay Tmax from 0.98 to 0.02. For dynamic representation, we run 200 iterations with batch size 4 for 4.5 minutes, with Tmax linearly decaying from 0.5 to 0.02. For the optional mesh refinement, we run 50 iterations with a constant T = 0.7 for 3.5 minutes.

|[Figure 157]<br><br>Input Image|[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>Animate124|
|---|---|
| |[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>Ours|
|[Figure 172]<br><br>Input Image|[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>Animate124|
| |[Figure 186]<br><br>Ours|

Figure 7: Qualitative Results for Image-to-4D. Comparing against Animate124 [59], our model achieves better faithfulness to the input image, larger motions, and greater details.

- Table 2: Quantitative Results on Image-to-4D Image Alignment. †: computed on 8 examples available at [59].

Method CLIP-I↑

Zero-1-to-3-V 0.7925 RealFusion-V 0.8026 Animate124 0.8544

Ours† 0.9227

t1 t2 t1 t2 t1 t2

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

b) Novel View 1 c) Novel View 2

a) Input Video

Figure 8: Qualitative Results for Video-to-4D. Renders are shown in two novel views at two timesteps.

#### 4.1 Comparisons with State-of-the-Art Methods

Image-to-4D. A qualitative image-to-4D comparison is shown in Fig. 7. We collect results from DG4D and Animated124 conditioned on 12 images and conduct a user study on 38 participants. Participants are asked to select the best one in three evaluation axes: image alignment, 3D appearance, and motion quality. The results are presented in Table 4, where our approach achieves a high win rate across all axes. We also quantify the image alignment using the metric CLIP-I provided by [59]. CLIP-I measures the cosine similarity of CLIP image embedding between reference-view renders and the reference image. The results are provided in Table 2, where DG4D has a clear advantage. Moreover, DG4D has a significantly shorter optimization time as shown in Table 1. All results are without refinement for DG4D.

Video-to-4D. Our approach is also compatible with the video-to-4D setting [20]. A qualitative video-to-4D result is provided in Fig. 8. We quantitatively evaluate our approach on a benchmark provided by [20]. Seven 32-frame static-view videos are evaluated, and metrics are computed with their ground-truth novel views. We use the first frame of each video for the static 3D generation. Two variants of our approach are used: Ours-Fast downsamples the video to 16 frames and trains 500 iters with one random view, and Ours uses all 32 frames and trains 500 iterations with four random views. The results are shown in Table 3, where both variants of our approach significantly outperform the baseline using much less time.

#### 4.2 Ablation Studies and Analysis

Motion Representation. Motion representation is essential for generating 4D content. We evaluated various motion representations, including framewise 3DGS with and without static initialization, MLP, and HexPlane. Employing framewise 3DGS without static initialization effectively conducts

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

- a) Ours
- b) S/4

- c) Sx4

HexPlane (Ours)

- a)

Framewise Fitting w/o init.

- b)

Framewise

Fitting w/ init.

- c)
- d) MLP

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

- d) T/4
- e) Tx4

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Adjacent frames 1 Adjacent frames 2

Adjacent frames 1

Adjacent frames 2

Figure 10: HexPlane Resolution Analysis. S is spatial resolution and T is temporal resolution.

Figure 9: Motion Representation Analysis. HexPlane with static initialization achieves the best generation performance.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

- a) 4D GS

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Video-to-Video

Refinement

c)

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Mesh

Sequence

- b)

- a) T = 0.6

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

c) T = 0.8

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

- b) T = 0.7

Figure 11: Mesh Texture Refinement. After V2V texture refinement, the texture quality and consistency could be highly improved.

Figure 12: Noise Level Analysis. Different noise levels T used in video SDS refinement.

- 3D generation independently for each frame, often leading to motion blur, resulting in noisy and inaccurate 3D models. Conversely, using static initialization enhances the clarity, though some blurriness persists. Framewise fitting is time-intensive, requiring several minutes per frame, significantly extending the duration for generating sequences (e.g., 14 frames). Overall, motion representations generally yield higher quality generation compared to framewise fitting. However, using MLP for learning motions frequently produces misalignments and minor local jitter. Among the evaluated methods, HexPlane consistently delivers superior performance. Qualitative comparisons are illustrated in Fig. 9.

HexPlane Resolution. We analyze the effect of HexPlane resolution in Fig. 10. Empirically, the spatial resolution affects the rigidity of the motion, larger resolution gives more flexibility to the structure change but also challenges the optimization. The temporal resolution affects the smoothness of motion, a too-low temporal resolution causes some Gaussian to lag behind the reference motion, and a too-high resolution adds difficulty to optimization. We use the HexPlane with resolution 32 × 32 × 32 (i.e., H = 32, and T = 32), which yields the best generation results.

Mesh Texture Refinement. DG4D enables users to extract mesh sequences following the acquisition of 4D GS in the initial stage. While the rendered images from GS typically display impressive outcomes, the textures of the extracted meshes may exhibit noise and artifacts. By implementing Video-to-Video (V2V) texture refinement, users can enhance the UV texture quality and consistency across the 3D mesh sequence. The qualitative improvements are demonstrated in Fig. 11.

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

#### Table 3: Quantitative Results for Video-to-

- 4D. Best is bolded and second best is underlined.

- a) Motion 1
- b) Motion 2
- c) Motion 3

Method LPIPS↓ CLIP↑ FVD↓ Time↓

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

D-NeRF 0.51 0.68 2327.83 K-planes 0.38 0.72 2295.68 Zero123 0.15 0.90 1571.60 Consistent4D 0.16 0.87 1133.44 2 hrs

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Ours-Fast 0.13 0.91 775.90 5 mins Ours 0.12 0.92 729.74 15 mins

#### Table 4: User Study.

time

Figure 14: Controllable Motions. DreamGaussian4D allows easy control of the generated motions. Different 3D motions can be generated from different driving videos for the same input image.

Win% Draw% Lose%

Image Alignment 87.28 3.28 9.42 3D Appearance 77.63 6.14 16.22 Motion Quality 70.17 6.57 23.24

Noise Level. We also evaluate different noise levels for the video SDS optimization. The qualitative generation results are visualized in Fig. 12. The refined texture maps could be over-smoothed if using a small noise level (e.g., T = 0.6). Conversely, the refined texture maps could be too noisy if using a large noise level (e.g., T = 0.8). We also evaluate the setting, annealing T from 0.7 to 0.95, which often leads to noisy results. Hence, we choose to fix T = 0.7 in our experiments.

#### 4.3 Advantages and Applications

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

Diverse Motions. Different from most existing 4D generation approaches [2, 20] using SDS, our method allows better controllability and more diversity in the motions. Different 4D motions can be generated from different driving videos. In Fig. 14, we generate three different driving videos for an input image, which results in three distinct 3D motions.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Adjacent frames 1 Adjacent frames 2

Figure 13: 4D Generation based on LGM [46].

#### Dynamic Scene Composition. By exporting

4D GS to textured meshes, we could directly generate composited dynamic scenes with engines (e.g., Blender). The rendered example images from different view angles could be visualized in Fig. 1. More qualitative results can be found in the appendix and our project page.

- 4D Generation based on LGM. LGM [46] is a recently proposed feedforward method for image-to-

3D generation using 3DGS, which offers faster generation with higher quality compared to SDS-based methods. By optimizing virtual camera settings, we facilitate 4D GS generation from the static 3D output of LGM. Qualitative 4D generation results are illustrated in Fig. 13.

- 5 Conclusion

We propose DreamGaussian4D (DG4D), an efficient image-to-4D generation framework with 4D Gaussian Splatting. DG4D significantly reduces the optimization time from several hours to minutes. Moreover, we show that driving motion generation using generated videos allows explicit control of the 3D motion. Lastly, DG4D allows mesh extraction and temporally coherent texture optimization, which facilitates real-world applications.

Limitations. Despite high efficiency, the generation quality is not as high as many photo-realistic

- 4D generation methods. In addition, we assume a good-quality driving video, which would largely influence the 4D generation quality.

Broader Impacts. The approach enables fast generation of dynamic 3D content and can be widely applied to videos, gaming, and mixed reality applications. However, it should be used with caution to prevent malicious impersonation.

### References

- [1] Jad Abou-Chakra, Feras Dayoub, and Niko Sünderhauf. Particlenerf: A particle-based encoding for online neural radiance fields. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5975–5984, 2024.
- [2] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. arXiv preprint arXiv:2311.17984, 2023.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141, 2023.
- [5] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.
- [6] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023.
- [7] Zhiqin Chen, Andrea Tagliasacchi, and Hao Zhang. Bsp-net: Generating compact meshes via binary space partitioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 45–54, 2020.
- [8] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5939–5948, 2019.
- [9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663, 2023.
- [10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, pages 13142–13153, 2023.
- [11] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14304–14314. IEEE Computer Society, 2021.
- [12] Shivam Duggal and Deepak Pathak. Topologically-aware deformation fields for single-view 3d reconstruction. In CVPR, pages 1536–1546, 2022.
- [13] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022.
- [14] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12479–12488, 2023.
- [15] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5712–5721, 2021.
- [16] Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, and Ulrich Neumann. Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365, 2024.
- [17] Shanyan Guan, Huayu Deng, Yunbo Wang, and Xiaokang Yang. Neurofluid: Fluid dynamics grounding with particle-driven neural radiance fields. In International Conference on Machine Learning, pages 7919–7929. PMLR, 2022.

- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020.
- [19] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023.
- [20] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848, 2023.
- [21] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023.
- [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ToG, 42(4):1–14, 2023.
- [23] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521–5531, 2022.
- [24] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6498–6508, 2021.
- [25] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4273–4284, 2023.
- [26] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. arXiv preprint arXiv:2312.13763, 2023.
- [27] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023.
- [28] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023.
- [29] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328, 2023.
- [30] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023.
- [31] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In CVPR, pages 8446–8455, 2023.
- [32] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
- [33] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022.
- [34] Sora Team OpenAI. Creating video from text, 2024.
- [35] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5865–5874, 2021.
- [36] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021.

- [37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [38] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021.
- [39] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, HsinYing Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.
- [41] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16632–16642, 2023.
- [42] Shelly Sheynin, Oron Ashual, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, and Yaniv Taigman. Knn-diffusion: Image generation via large-scale retrieval. arXiv preprint arXiv:2204.02849, 2022.
- [43] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model, 2023.
- [44] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023.
- [45] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. arXiv preprint arXiv:2312.13150, 2023.
- [46] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024.
- [47] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.
- [48] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023.
- [49] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, and Christian Theobalt. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12959–12970, 2021.
- [50] Alex Trevithick and Bo Yang. Grf: Learning a general radiance field for 3d representation and rendering. In ICCV, pages 15182–15192, 2021.
- [51] Haithem Turki, Jason Y Zhang, Francesco Ferroni, and Deva Ramanan. Suds: Scalable urban dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12375–12385, 2023.
- [52] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [53] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. arXiv preprint arXiv:2310.08528, 2023.
- [54] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9421–9431, 2021.

- [55] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir Mech, and Ulrich Neumann. Disn: Deep implicit surface network for high-quality single-view 3d reconstruction. Advances in neural information processing systems, 32, 2019.
- [56] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101, 2023.
- [57] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In CVPR, 2023.
- [58] Wentao Yuan, Zhaoyang Lv, Tanner Schmidt, and Steven Lovegrove. Star: Self-supervised tracking and reconstruction of rigid objects in motion with neural rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13144–13152, 2021.
- [59] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023.
- [60] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for text-and image-guided 4d scene generation. arXiv preprint arXiv:2311.16854, 2023.

##### Table 5: Quantitative comparisons between ours and others on Consistent4D dataset. Baseline results are from [16].

Pistol Guppie Crocodile Monster Skull Trump Aurorus Mean LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑ LPIPS↓ CLIP↑

Method

D-NeRF [38] 0.52 0.66 0.32 0.76 0.54 0.61 0.52 0.79 0.53 0.72 0.55 0.60 0.56 0.66 0.51 0.68 K-planes [14] 0.40 0.74 0.29 0.75 0.19 0.75 0.47 0.73 0.41 0.72 0.51 0.66 0.37 0.67 0.38 0.72 C4D [20] 0.10 0.90 0.12 0.90 0.12 0.82 0.18 0.90 0.17 0.88 0.23 0.85 0.17 0.85 0.16 0.87 GauFlow [16] 0.10 0.94 0.10 0.93 0.10 0.90 0.17 0.92 0.17 0.92 0.20 0.85 0.15 0.89 0.14 0.91

Ours 0.08 0.94 0.10 0.94 0.10 0.89 0.15 0.95 0.13 0.95 0.16 0.90 0.13 0.88 0.12 0.92

- A More implementation details

- A.1 3D Gaussian Splatting settings.

We initialized the static 3D GS with 5,000 Gaussians. We did not use the Spherical Harmonics. For DreamGaussianHD, we set the dense percentage to 0.1, the densification interval to 100, densification gradient threshold to 0.05. All other settings follow DreamGaussian [47].

- A.2 HexPlane settings

We use 32 for both spatial and temporal resolutions. No multi-resolution is applied. The grid feature dimension is 32. We set the learning rate for the MLP to 0.00064 and the learning rate for the HexPlane grid to 0.0064. Other settings follow 4D Gaussian Splatting [53].

- A.3 Evaluation settings.

Image-to-4D evaluation. For the user study, we describe the evaluation axis to participants as follows:

- • Image Alignment. Which one looks the most like the input image?
- • 3D Appearance. Which one has a better-defined 3D geometry and a nicer look?
- • Motion Quality. Which one has a more natural and realistic motion?

For the computation of the CLIP-I metric, we use the 8 images provided on their official website.

Video-to-4D evaluation. We use the code and data provided by [20] in their officially released code. All videos are rendered from animated character assets. A horizontal angle of 0 degrees is used for input and other novel views are evaluated at -75 degrees, 15 degrees, 105 degrees, and 195 degrees. Similarity metric can be thus computed between the generated 4D content to the original animated character.

- A.4 4D Generation based on LGM

We can use LGM [46] in replace of DreamGaussianHD to generate static 3D from the input image or a frame in the input video. Since LGM often generates 3D GS that misaligns with the input image, we render the LGM model from [-180◦, 180 ◦] azimuths at 0◦ elevation, and select the azimuth that has the lowest L2 distance to the input image on the RGB space. Then, we render the 3D GS from the selected azimuth and the other three orthogonal viewpoints to get four multiview images. Finally, we feed the multiview images into LGM to get the static initialization that aligns well with the input image. Following LGM’s camera setting, we use a camera radius of 1.5 in the dynamic optimization.

- B More quantitative results We show detailed evaluation metrics for the video-to-4D benchmark in Table 5.
- C More qualitative results More image-to-4D results are provided in Figure 15.

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

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Input Image Generated 4D Model

- Figure 15: Qualitative results. Renders are shown with gradually changing time stamps and view angles.

#### C.1 More qualitative result on video to 4D

More video-to-4D results are provided in Figure 16.

### D More Ablations and Analysis

#### D.1 Deformation field representations.

We analyze the effect of different deformation field representations, including a pure MLP network and frame-wise Gaussian. The pure MLP network takes the position embedding and time embedding as the input, and outputs deformation variables. The frame-wise Gaussians use the static 3D GS as an initialization and fit a separate 3D GS for each video frame independently. We train them under the same setting and show the result in Figure 17. The MLP network is under-optimized and the frame-wise Gaussian lacks temporal consistency.

t1 t2 t1 t2 t1 t2

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

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

a) Input Video b) Novel View 1 c) Novel View 2

- Figure 16: Qualitative results for video-to-4D. Renders are shown in two novel views at two timesteps.

#### D.2 Temporal loss.

We add the temporal loss described by [30] to the optimization but can not observe significant improvement. A brief comparison is shown in Figure 18.

t1 t2 t1 t2 t1 t2

t1 t2

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

a) HexPlane

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

b) MLP

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

c) Framewise

b) Input View c) Novel View 1

d) Novel View 2

a) Input Video

#### Figure 17: Ablation on motion representation.

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

a) Ours

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

b) + Temporal regularization

Adjacent frames 1 Adjacent frames 2

#### Figure 18: Ablation on temporal loss.

