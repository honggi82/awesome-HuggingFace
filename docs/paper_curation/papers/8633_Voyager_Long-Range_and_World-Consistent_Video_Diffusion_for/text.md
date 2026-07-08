Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation

TIANYU HUANG∗, Harbin Institute of Technology, China WANGGUANDONG ZHENG∗, Southeast University, China TENGFEI WANG†, Tencent Hunyuan, China YUHAO LIU, City University of Hong Kong, China ZHENWEI WANG, City University of Hong Kong, China JUNTA WU, Tencent Hunyuan, China JIE JIANG, Tencent Hunyuan, China HUI LI, Harbin Institute of Technology, China RYNSON W.H. LAU, City University of Hong Kong, China WANGMENG ZUO†, Harbin Institute of Technology, China CHUNCHAO GUO, Tencent Hunyuan, China

arXiv:2506.04225v1[cs.CV]4Jun2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

15 30 45 60

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>|
|---|

WorldExplorationWorldReconstruction

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

15 30 45 60

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

15 30 45 60

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

15 30 45 60

Input Camera Control Generated Video

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

0 15 30 45 60

Generated RGB

[Figure 67]

|[Figure 68]|
|---|

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

0 15 30 45 60

Generated Depth Generated Point Cloud

Fig. 1. Voyager is a world-consistent video generation and reconstruction framework. Up: Voyager can generate 3D-consistent scene videos for world exploration following custom camera trajectories. Bottom: Voyager jointly generates aligned depth and RGB video for effective and direct 3D reconstruction.

Real-world applications like video gaming and virtual reality often demand the ability to model 3D scenes that users can explore along custom camera trajectories. While significant progress has been made in generating 3D objects from text or images, creating long-range, 3D-consistent, explorable 3D

scenes remains a complex and challenging problem. In this work, we present Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Unlike existing approaches, Voyager achieves end-to-end scene generation and reconstruction with inherent consistency across frames, eliminating the need for 3D reconstruction pipelines (e.g., structure-from-motion or

∗Both authors contributed equally to this research. †∗Corresponding author.

multi-view stereo). Our method integrates three key components: 1) WorldConsistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence 2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency, and 3) Scalable Data Engine: A video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Collectively, these designs result in a clear improvement over existing methods in visual quality and geometric accuracy, with versatile applications. See more at https://voyager-world.github.io.

- 1 Introduction

The creation of high-fidelity, explorable 3D scenes that users can navigate seamlessly, powers broad applications ranging from video gaming and film production to robotic simulation. Yet, traditional workflows for constructing such 3D worlds remain bottlenecked by manual effort, requiring painstaking layout design, asset curation, and scene composition. While recent data-driven methods [Liu et al. 2024; Meng et al. 2024; Xiang et al. 2024; Xie et al. 2024; Zhao et al. 2025] have shown promise in generating objects or simple scenes, their ability to scale to complex scenes is limited by the scarcity of high-quality 3D scene data. This gap highlights the need for frameworks that enable scalable generation of user-navigable virtual worlds with 3D consistency.

Recently, a growing number of works [Chen et al. 2025; Gao* et al. 2024; He et al. 2024; Ma et al. 2025; Ren et al. 2025; Wang et al. 2024b; Yu et al. 2024b; Zhou et al. 2025] have explored the use of novel view synthesis (NVS) and video generation as alternative paradigms for world modeling. These methods, while demonstrating impressive capabilities in generating visually appealing and semantically rich content, still face several challenges. 1) Long-Range Spatial Inconsistency. Due to the absence of explicit 3D structural grounding, they often struggle to maintain spatial consistency and coherent viewpoint transitions during the generation process, especially when generating videos with long-range camera trajectories.

- 2) Visual Hallucination. While several works [Chen et al. 2025; Ren et al. 2025] have attempted to leverage 3D conditions to enhance geometric consistency, they typically rely on partial RGB images as guidance, i.e., novel-view images rendered from point clouds reconstructed with input views. However, such representation may introduce significant visual hallucinations in complex scenes, such as the incorrect occlusions in Figure. 2, which may introduce inaccurate supervision during training. 3) Post-hoc 3D Reconstruction. While these approaches can synthesize visually satisfying content, post-hoc 3D reconstructions are still required to obtain usable 3D content. This process is time-consuming and inevitably introduces geometric artifacts [Weber et al. 2024], making it inadequate for real-world applications.

To address these challenges, we introduce Voyager, a framework designed to synthesize long-range, world-consistent RGB-D(epth) videos from a single image and user-specified camera trajectories. At the core of Voyager is a novel world-consistent video diffusion that utilizes an expandable world caching mechanism to ensure spatial consistency and avoids visual hallucination. Starting from an

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

### RGB-OnlyRGB-D

### DepthRGB

| |
|---|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Input View Novel View 1 Novel View 2 Output

Fig. 2. Partial RGB images and partial depth maps rendered from point clouds at different frames. In scenarios involving complex occlusion relationships, partial RGB images often exhibit significant visual artifacts. In contrast, partial depth maps can accurately represent occlusions.

image, we construct an initial world cache by unprojecting it into 3D space with a depth map. This 3D cache is then projected into target camera views to obtain partial RGB-D observations, which guides the diffusion model to maintain coherence with the accumulated world state. Crucially, the generated frames are fed back to update and expand the world cache, creating a closed-loop system that supports arbitrary camera trajectories while maintaining geometric coherence.

Unlike methods [Chen et al. 2025; Ma et al. 2025; Ren et al. 2025; Yu et al. 2024b] relying only on RGB conditioning, Voyager explicitly leverages depth information as a spatial prior, enabling more accurate 3D consistency during video generation. By simultaneously generating aligned RGB and depth sequences, our framework supports direct 3D scene reconstruction without requiring additional 3D reconstruction steps like structure-from-motion.

Despite promising performance, diffusion models struggle to generate long videos in a single pass. To enable long-range world exploration, we propose world caching scheme and smooth video sampling for auto-regressive scene extension. Our world cache accumulates and maintains point clouds from all previously generated frames, expanding as video sequences grow. To optimize computational efficiency, we design a point culling method to detect and remove redundant points with real-time rendering, minimizing memory overhead. Leveraging cached point clouds as a proxy, we develop a smooth sampling strategy that auto-regressively extends video length while ensuring smooth transitions between clips.

Training such a model requires large-scale videos with accurate camera poses and depth, but existing datasets often lack these annotations. To address this, we introduce a data engine for scalable video reconstruction that automatically estimates camera poses and metric depth for arbitrary scene videos. With metric depth estimation, our data engine ensures consistent depth scales across diverse sources, enabling high-quality training data generation. Using this pipeline, we compile a dataset of over 100,000 video clips, combining real-world captures and synthetic Unreal Engine renders.

Extensive experiments demonstrate the effectiveness of Voyager in scene video generation and 3D world reconstruction. Benefiting from joint depth modeling, our results in Figure 1 exhibit more coherent geometry, which not only enable direct 3D reconstruction but also support infinite world expansion while preserving the original spatial layout. Additionally, we explore applications such as 3D generation, video transfer, and depth estimation, further showcasing the potential of Voyager in advancing spatial intelligence.

Our contributions can be summarized as:

- • We introduce Voyager, a world-consistent video diffusion model for scene generation. To the best of our knowledge, Voyager is the first video model that jointly generates RGB and depth sequences with given camera trajectories.
- • We propose an efficient world caching scheme and autoregressive video sampling approach, extending Voyager to world reconstruction and infinite world exploration.
- • We propose a scalable video data engine for camera and metric depth estimation, with over 100,000 training pairs prepared for the video diffusion model.

2 Related Work

- 2.1 Camera-Controllable View Generation

Existing camera-controllable generation models can be categorized into three types: novel view synthesis [Hong et al. 2023; Kerbl et al. 2023; Mildenhall et al. 2021; Wu et al. 2024] produces new viewpoints through multi-view reconstruction. These methods rely on dense viewpoints and struggle to handle single-view inputs. The second method [Guo et al. 2023; He et al. 2024; Liu et al. 2023; Wang et al. 2024b; Zhou et al. 2025] implicitly incorporates camera parameters into the model, training it to generate images from the corresponding viewpoints, but often suffers from viewpoint inconsistency. The third method [Chen et al. 2025; Ma et al. 2025; Ren et al. 2025; Seo et al. 2024] leverages point clouds obtained by warping the input view as conditions for novel view generation, significantly improving spatial consistency. However, the warped images still contain artifacts that negatively affect model training. In this work, we introduce warping depth as an additional conditioning input and generate both RGB and depth content.

- 2.2 Long-Range Video Generation

Current video models are limited in their ability to generate long videos in a single pass. To extend video length, existing research explores training-free methods [Lu et al. 2024; Wang et al. 2023], hierarchical strategies [He et al. 2022; Yin et al. 2023], and autoregressive frameworks [Henschel et al. 2024; Yin et al. 2024]. However, the first two approaches cannot scale to infinitely long videos, while the auto-regressive strategy relies on memory caches that struggle to retain information from distant past frames. To address this limitation, we propose world cache with point culling in this work that efficiently preserves spatial information and enables the generation of arbitrarily long videos with smooth video sampling in an auto-regressive inference.

- 3 Preliminaries of Video Diffusion Models

Diffusion models learn to denoise a data distribution 𝑝(𝑥) through an iterative process, in which a forward diffusion process gradually adds noise to the data x0 ∼ 𝑝(x0), and a reverse process learns to recover x0 from the noisy data x𝑡.

In the context of video generation, diffusion models are extended to learn temporal dynamics by incorporating 3D convolutional architectures [Tran et al. 2015; Yu et al. 2023b] and attention mechanisms [Brooks et al. 2024; Zhang et al. 2023]. To reduce the computation cost, latent diffusion [Blattmann et al. 2023; Rombach et al.

2022] is widely used to compress the video to a low-dimensional latent space.

In this work, our video model is based on Hunyuan-Video [Kong et al. 2024]. Formally, given an input text prompt 𝑦 and a groundtruth video sequence [𝐼0, ...,𝐼𝑇−1] ∈ R𝑇×3×𝐻×𝑊 , the model first extracts the video latent z0 with shape (𝑐𝑇𝑡 +1)×𝐶 × 𝑐𝐻𝑠 × 𝑊𝑐𝑠 by a 3DVAE, where 𝑐𝑡 and 𝑐𝑠 denote the compression rate for the temporal and spatial axis. To train a denoising model 𝜃, noisy latent z𝑡 is then fed to a full-attention DiT [Li et al. 2024; Peebles and Xie 2022] model, which follows the strategy of "Dual-stream to Single-stream" hybrid model [Labs 2024]. Patched video and text latents are processed independently in dual-stream Transformer blocks 𝑓𝐷𝑖 , while in the second phase, these latents are concatenated in single-stream blocks 𝑓𝑆𝑖. To further support image-conditioned video generation, the latent feature of the input image is concatenated to z𝑡 channel-wise. The training objective is to predict the velocity u𝑡 = 𝑑z𝑡/𝑑𝑡 by minimizing the mean squared error between the estimated velocity uˆ𝑡 and the ground-truth u𝑡. Finally, the latent z0 is recovered by the first-order Euler ordinary differential equation (ODE) solver, and the video 𝑣 is reconstructed by the 3D-VAE decoder.

4 Methodology: Voyager

Given an image 𝐼0 ∈ R3×𝐻×𝑊 , our goal is to create an explorable world based on a user-defined camera trajectory. However, there is a gap between video generation and 3D world modeling, which mainly stems from three aspects: (1) the inconsistency of long-range video extension, (2) the hallucination of visual conditions for video generation, and (3) the incapability of reconstructing the world from video outputs. To address these issues, we propose Voyager, a world-consistent video generation framework that can directly produce rgb-depth frames with corresponding camera parameters for long-range world exploration. In this section, we first introduce a geometry-injected frame condition to compensate for perceptual hallucination under visual conditions. (Sec. 4.1). With this input condition, we propose a depth-fused video diffusion model to ensure spatial consistency and context-based blocks to enhance its viewpoint control (Sec. 4.2). For 3D world reconstruction and longrange exploration, we propose world caching with point culling and smooth video sampling in the auto-regressive inference (Sec. 4.3). We further propose a scalable video data engine to prepare camera and metric depth for the training of the above model (Sec. 4.4).

4.1 Geometry-Injected Frame Condition

For the control of video viewpoint, camera parameter [Bai et al. 2025; Zhou et al. 2025] is a straightforward condition, but this implicit condition is nontrivial to the training of video models. Recent works [Chen et al. 2025; Ma et al. 2025; Ren et al. 2025] attempt to reconstruct the point cloud 𝑝 ∈ R𝑁×6 from videos as an explicit control, where 𝑁 is the number of points and each point is represented by 6D coordinates (𝑥,𝑦,𝑧,𝑟,𝑔,𝑏). The warped RGB condition 𝐼ˆ𝑣 for a novel view 𝑣 can then be rendered according to the camera, which is a partial image with blank regions.

Nonetheless, such a partial RGB image is insufficient to ensure spatial consistency, e.g., complex occlusion relationships in a scene may lead to visual hallucinations. To enforce spatially consistent

Geometry-Injected World Cache

User Input

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

depthestimation

|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Camera

[Figure 104]

|[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]|
|---|

[Figure 111]

[Figure 112]

Image

[Figure 113]

Encoder

[Figure 114]

DiT🔥

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Double-Stream 𝑓

Control Block 𝑓

×𝑁

[Figure 121]

Text

Decoder

“A mountain lake reflecting the peaks and a rustic cabin on stilts.”

Single-Stream 𝑓

Control Block 𝑓

×𝑁

- Fig. 3. Overview of Voyager: Given the input image and camera trajectories, we first render partial RGB images and depth maps for each viewpoint as the condition for video generation. Our world-consistent video diffusion model is trained to generate RGB-D frames simultaneously, thus supporting the direct reconstruction of the 3D world. The projected points are store in our world cache efficiently, which can be rendered as condition for the next round generation.

control during training, we introduce an additional geometric condition partial depth map, which is aligned with the partial RGB image. Specifically, we first estimate the depth map 𝐷𝑘 and corresponding camera parameters 𝑐𝑘 for each frame 𝐼𝑘 of the video. Since only the first frame is visible in video inference, we create a point cloud 𝑝0 by projecting 𝐷0 with 𝑐0. For the 𝑘-th frame, its partial image 𝐼ˆ𝑘 and partial depth 𝐷ˆ𝑘 are acquired by masking the invisible region with the rendering mask 𝑀𝑘 = render(𝑝0,𝑐𝑘).

advantage of DiT’s full-attention structure, allowing for the interaction of visual and geometric information at the pixel level. To this end, we concatenate the rgb and depth images along the height axis as I𝑘 = [𝐼𝑘, Φ,𝐷𝑘]ℎ, as well as condition maps Iˆ𝑘 = [𝐼ˆ𝑘, Φ,𝐷ˆ𝑘]ℎ and masks 𝑀𝑘 = [𝑀𝑘, Φ,𝑀𝑘]ℎ. Here, we add a placeholder row Φ between the rgb and depth images to help the model separate these two types of content. The new video latents are presented as z𝑡,′ 0 = 𝑓emb(concat(z𝑡, zˆ0,𝑚)), where zˆ0 is the latent of [Iˆ𝑘]𝑇𝑘=−01

and 𝑚 is the down-sampled map of [𝑀𝑘]𝑇𝑘=−01 via max-pooling. Accordingly, z𝑡,′ 0 is fed to the diffusion model similar to Eq. 1-2. The diffusion model is thus trained to generate rgb-depth video frames. Context-Based Control Enhancement. The above concatenation mechanism incorporates conditional information only at the input of the DiT model, leading to weak enforcement of the geometric conditions and resulting in misalignment between generated frames and input conditions.

4.2 World-Consistent Video Diffusion

Conditioned with partial RGB and depth maps, our intention is to generate plausible content for the invisible regions, ensuring consistency with the spatial information provided by the partial conditions.

For this purpose, the common practice [Labs 2024; Ren et al. 2025] is to concatenate the condition latents zrgb and zdepth with original noisy latents z𝑡 along the channel axis and project the concatenated latents back to the Transformer dimension via the

To enhance the geometric-following capabilities, following [Bian et al. 2025], we further inject the diffusion model with lightweight modules. Concretely, we replicate the first block from the doublestream and single-stream modules as the Control blocks 𝑓ˆ𝐷 and 𝑓ˆ𝑆. Given the input video latent z𝑡,′ 0, we have the following operations for each Transformer block 𝑖:

patch-embedding layer 𝑓emb: z𝑡,′ 0 = 𝑓emb(concat(z𝑡, zrgb, zdepth)). Then, the projected latents z𝑡,′ 0 are fed to double-stream and singlestream blocks sequentially, which is formulated as,

z𝑡,𝑖′ , z′𝑦,𝑖 = 𝑓𝐷𝑖 (z𝑡,𝑖′ −1, z′𝑦,𝑖−1,𝑡),𝑖 = 1, ..., 𝑁𝐷, (1)

z𝐷 = 𝑓ˆ𝐷(z𝑡,′ 0), z𝑆 = 𝑓ˆ𝑆 (z𝐷), (3) z𝑡,𝑖′ = z𝑡,𝑖′ + 𝑙𝐷(z𝐷), z𝑡,𝑖′′ = z𝑡,𝑖′′ + 𝑙𝑆 (z𝑆), (4)

z𝑡,𝑖′′ = 𝑓𝑆𝑖(z𝑡,𝑖′′ −1,𝑡),𝑖 = 1, ..., 𝑁𝑆, (2) where z′𝑦 is the text latents. 𝑁𝐷 and 𝑁𝑆 denote the block number of each stream. z𝑡,′′0 is initialized as the concatenation of z𝑡,𝑁′ and z′𝑦,𝑁 .

where 𝑙𝐷 and 𝑙𝑆 are zero-initialized linear layers. Early-stage latent features preserve more contextual information, so that the integration into each block can strengthen pixel-level controllability.

Although the video model can best preserve the pre-trained parameters in this way, the spatial conditions is only used in channelwise. The missing parts in our partial maps can range from small cracks to large blank areas, depending on the extent of the viewpoint change. This trivial solution struggles to handle variable situations. Depth-Fused Video Generation. Instead of relying solely on partial depth as the input condition for completing the missing regions in the RGB frames, we propose to simultaneously generate both complete RGB and depth frames. As a result, the video model can take

4.3 Long-Range World Exploration

For long-range or even infinite video generation, auto-regressive is a natural choice. This paradigm recursively generates future frames or clips based on previously generated content, maintaining temporal continuity over time. However, due to the limited memory capacity of video diffusion models, auto-regressive methods are often restricted to conditioning on only a few preceding frames or clips.

Table 1. Quantitative comparison of novel view synthesis on RealEstate10K.

# Method PSNR ↑ SSIM ↑ LPIPS ↓

SEVA 16.648 0.613 0.349 ViewCrafter 16.512 0.636 0.332 See3D 18.189 0.694 0.290 FlexWorld 18.278 0.693 0.281 Voyager 18.751 0.715 0.277

This limited context leads to inevitable information loss, making it fundamentally infeasible to retain and propagate the full scene history. In contrast to previous auto-regressive methods, Voyager exploits point-cloud conditions for generation, which is a scalable representation to store the whole history information. To enable infinite generation, we propose world caching with point culling to efficiently store spatial information and adopt smooth video sampling to ensure the consistency of consecutive clips.

World Caching with Point Culling. With input camera parameters and corresponding RGB-D video frames, point clouds can be projected to 3D space as 𝑝ˆ ∈ R(𝑇×𝐻×𝑊 )×3, where 𝑇 is the number of total frames. As the video continues to extend, the number of points can easily grow to millions, posing significant challenges in terms of memory and computational efficiency. To address that, we propose to maintain a world cache, which eliminates redundant points while preserving essential geometric information. Specifically, we incrementally add new points to the cache on a per-frame basis: given the accumulated point cloud 𝑝ˆ from previous frames, we render a visibility mask 𝑀 = render(𝑝,𝑐ˆ 𝑖) from the current camera view 𝑐𝑖. Points in the invisible regions are added to 𝑝ˆ first. For the visible regions, if the angle between the surface normal of existing points and the current view direction exceeds 90 degrees, the new point is also updated into the cache, because these existing points cannot be seen at the current viewpoint. This strategy reduces the number of stored points by approximately 40% and avoids noise accumulation caused by multi-frame aggregation.

Smooth Video Sampling. Conditioned on the above world cache, our video model can access the complete spatial information from previous frames. However, although each independently generated video clip is spatially consistent, there can still be color discrepancies, making them unsuitable for direct concatenation. We adopt two strategies to ensure smoother transitions between adjacent clips. (1) We first divide the input video into overlapping segments, where the length of the overlapping region is half of one segment. For each segment, the overlapping region is initialized with the generated results from the previous segment, serving as the noise initialization for the current segment’s overlap region. (2) After completing inference for the consecutive two segments, we apply averaging across the overlapping regions and introduce a light-level noise injection to the merged segments. A final round of denoising is then performed to refine transitions. In this way, we ensure the efficient generation of multiple clips while maintaining visual consistency across consecutive video frames.

- 4.4 Scalable Video Data Engine

Training such a video model demands large-scale video frames with corresponding camera parameters and depth maps. We carefully

Table 2. Quantitative comparison of Gaussian Splattig reconstruction on RealEstate10K. Baselines require additional reconstruction step [Wang et al. 2025], while Voyager performs better with our generated depth.

# Method Post Rec. PSNR ↑ SSIM ↑ LPIPS ↓

SEVA VGGT 15.581 0.602 0.452 ViewCrafter VGGT 16.161 0.628 0.440 See3D VGGT 16.764 0.633 0.440 FlexWorld VGGT 17.623 0.659 0.425 Voyager VGGT 17.742 0.712 0.404 Voyager - 18.035 0.714 0.381

curate over 100,000 video clips from both real-captured videos and 3D renderings, and propose a scalable video data engine to automatically annotate required 3D information for arbitrary scene videos. Data Curation. We selected two open-source real-world datasets, i.e., RealEstate [Zhou et al. 2018] and DL3DV [Ling et al. 2024] for the training. RealEstate contains 74,766 video clips related to real estate scenes, primarily featuring indoor home scenes, along with some outdoor environments. DL3DV provides 10K real-scene videos, but most of them suffer from rapid or shaky camera movements. We curate 3,000 high-quality videos from this dataset and segment them into approximately 18,000 video clips. Additionally, to increase the diversity of generation content, we collected 1,500 Unreal Engine scene models and rendered over 10,000 video samples to augment the dataset. In the end, we collected over 100,000 video clips from these datasets.

Data Annotation. Accurate camera parameters and depth are crucial for model training, but RealEstate and DL3DV do not provide such ground-truth data. Existing methods [Chen et al. 2025; Ren et al. 2025; Schwarz et al. 2025] adopt dense stereo models [Teed and Deng 2021] to prepare training pairs, struggling to produce geometrically consistent depth. We propose a more robust data processing engine. Specifically, we first use VGGT [Wang et al. 2025] to estimate camera parameters and depth for all video frames. The depth estimated by VGGT is not accurate enough, but it is aligned with camera poses. To further improve the depth estimation, we then employ MoGE [Wang et al. 2024a] as a robust depth estimator and align the two depth maps with least squares optimization.

Finally, since our UE data provides metric depth values, we need to align all the estimated depth to a standard scale. We estimate the metric depth range of the scene using Metric3D [Hu et al. 2024] and map the previous depths into this range. This way, we can automatically annotate camera and depth for videos from any source.

5 Experiments 5.1 Video Generation

We evaluate the video generation quality of Voyager by comparing four open-source camera-controllable video generation methods on image-to-video generation, including SEVA [Zhou et al. 2025], ViewCrafter [Yu et al. 2024b], See3D [Ma et al. 2025], and FlexWorld [Chen et al. 2025]. Among these methods, ViewCrafter, See3D, and FlexWorld control the viewpoints with point cloud conditions, which are similar to our method. SEVA directly takes camera parameters as input conditions.

Dataset and Metrics. We randomly select 150 video clips from the test set of RealEstate [Zhou et al. 2018] as our test dataset. Since the

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

| |
|---|

[Figure 129]

[Figure 130]

[Figure 131]

| |
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

| |
|---|

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

|[Figure 149]|
|---|

Input Ours FlexWorld SEVA See3D ViewCrafter GT

- Fig. 4. Qualitative results on video generation. Compared to the baselines, our model can generate a more reasonable unseen region and meanwhile preserve the content in the input view.

video clips do not provide ground-truth cameras, we estimate the camera parameters and depth maps with the same pipeline in our data engine. To evaluate the visual quality of generated videos, we adopt PSNR, SSIM, and LPIPS to measure the similarity between the generated frames and the ground truth.

Results. We report the quantitative results on Table 1. Our method outperforms all the baselines, demonstrating the high generation quality of our video model. The qualitative comparison in Figure 4 also showcases our capability of generating photorealistic videos. Especially in the last case of Figure 4, only our method can preserve the details of products in the input image. However, other methods are prone to generating artifacts, e.g., in the first example of Figure 4, these methods fail to provide reasonable predictions when the camera movement is too large.

- 5.2 Scene Generation

diverse worlds, e.g., indoor and outdoor, photorealistic and stylized. In each example, an input image and a camera trajectory are provided. The metrics evaluate the controllability and quality of generation, and an average score is presented to show the overall performance. We compare six top methods in the existing benchmark, including two 3D methods WonderJourney [Yu et al. 2023a] and WonderWorld [Yu et al. 2024a], and four video methods EasyAnimate [Xu et al. 2024], Allegro [Zhou et al. 2024], Gen-3 [Runway 2024], and CogVideoX [Yang et al. 2024].

The scores are reported in Table 3. Voyager achieves the highest score on this benchmark. The score shows that our method has a competitive performance on camera control and spatial consistency, compared with 3D-based methods. Our subjective quality is the highest among all methods, further demonstrating the visual quality of our generated videos. Notably, since our video condition is constructed with metric depth, the camera movement in our results are larger than other methods, which is much harder to generate.

To evaluate the quality of scene generation, we further compare the quality of scene reconstruction with generated videos based on Sec. 5.1. Since the compared baselines only produce RGB frames, we first exploit VGGT [Wang et al. 2025] to estimate camera parameters and initialize the point clouds for the generated videos of these methods. Thanks to the capability of generating RGB-D content, our results can be directly used in 3DGS reconstruction.

5.4 Ablation Studies

To verify the effectiveness of our proposed designs, we conduct ablation studies on our world-consistent video diffusion and longrange world exploration.

World-Consistent Video Diffusion We evaluate our video models trained in the three stages separately on Worldscore benchmark, i.e., (a) model trained only on RGB conditions, (b) model trained on RGB-D conditions, and (c) model attached with additional control blocks. As shown in Table 4, fusing depth conditions in training can significantly enhance the capability of camera control. The control blocks can further improve the spatial consistency of generated results. We also provide qualitative results in Figure 7. The RGB-only model may generate inconsistent content when the camera moves to an unseen region. The results of RGB-D model is more consistent with the input image, but it could still produce some minor artifacts. Our final model generates the most reasonable results.

In Table 2, our reconstruction results with VGGT post-hoc outperform the compared baselines, indicating that our generated videos are more consistent in aspect of geometry. The results are even better when initializing point clouds with our own depth output, which demonstrates the effectiveness of our depth generation for scene reconstruction. The qualitative results in Figure 3 illustrate the same conclusion. Particularly in the last case, our method retains most details of the chandelier, while baseline methods even fail to reconstruct a basic shape.

- 5.3 World Generation

Long-range video generation. We evaluate the quality of point culling and smooth sampling in Figure 8. For point culling, storing all points introduces noise, while storing points in the invisible region is insufficient. Results with additional normal check have comparable

Besides the in-domain comparison on RealEstate, we test Voyager on WorldScore [Duan et al. 2025] static benchmark on world generation. WorldScore consists of 2,000 static test examples that span

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

| |
|---|

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

Input Ours FlexWorld SEVA See3D ViewCrafter GT

Fig. 5. Qualitative results on Gaussian Splatting reconstruction. Our results present much more details than the compared baselines.

Table 3. Quantitative comparison on WorldScore Benchmark. Bold and underline indicates the 1st, Bold indicates the 2nd, underline indicates the 3rd.

|Method<br><br>|WorldScore Average<br><br>|Camera Control<br><br>Object Control<br><br>Content Alignment|3D Consistency<br><br>Photometric Consistency<br><br>Style Consistency<br><br>Subjective Quality<br><br>|
|---|---|---|---|
|WonderJourney WonderWorld|63.75 72.69<br><br>|84.6 37.1 35.54 92.98 51.76 71.25<br><br>|80.6 79.03 62.82 66.56 86.87 85.56 70.57 49.81<br><br>|
|EasyAnimate Allegro Gen-3 CogVideoX-I2V<br><br>|52.85 55.31 60.71 62.15|26.72 54.5 50.76 24.84 57.47 51.48 29.47 62.92 50.49 38.27 40.07 36.73<br><br>|67.29 47.35 73.05 50.31 70.5 69.89 65.6 47.41<br><br>68.31 87.09 62.82 63.85 86.21 88.12 83.22 62.44<br><br><br>|
|Voyager|77.62|85.95 66.92 68.92<br><br>|81.56 85.99 84.89 71.09<br><br>|

Table 4. Ablation study on Worldscore.

Camera Control

Content Alignment

3D Consistency

Metric

Ours (RGB-only) 74.98 48.92 68.86 Ours (RGB-D) 85.04 65.72 78.58 Ours (full) 85.95 68.92 81.56

visual performance with storing all points, but save almost 40% storage. For smooth sampling, the video clip without sampling may exhibit inconsistencies compared to the first clip. Smooth sampling ensures a seamless transition between two consecutive segments.

6 Application

Benefiting from our depth-fused video generation, Voyager supports various 3D-related applications.

Long Video Generation. As explained in Sec. 4.3, our method allows long-range video generation with efficient world caching and smooth video sampling. In Figure 6(a), we provide an example consisting of three video clips, with totally different camera trajectories among clips. The results present camera controllability and spatial consistency of the generated video, demonstrating that our method is capable of long-range world exploration.

Image-to-3D Generation. Native 3D generative models can hardly handle the generation of multiple objects. In Figure 6(b), we use three state-of-the-art 3D generation methods Trellis [Xiang et al. 2024], Rodin v1.5 [RodinAI 2025], and Hunyuan-3D v2.5 [Hunyuan3D

2025] to generate a simple combination where a car leans against a tent. Rodin failed to generate the tent, while Trellis produced a tent with missing parts. Hunyuan successfully generated two complete objects, but the spatial relationship was inaccurate, with the tent being too far from the car. Our method not only generates the correct content, but also produces more realistic visual effects. The tent is even visible through the car window in the side view.

Depth-Consistent Video Transfer. Generating a spatially consistent video with a different style typically requires training a stylized video model. However, to achieve the desired effect with our model, we only need to replace the reference image while retaining the original depth condition. As shown in Figure 6(c), we can change the original video to American-style or to the night.

Video Depth Estimation. Our video model is naturally capable of estimating video depth. In Figure 6(d), our predicted depth can preserve the details on the architectures.

7 Conclusion

In this paper, we present Voyager, a world-consistent video generation framework for long-range world exploration. The proposed RGB-D video diffusion model can produce spatially consistent video sequences that align with the input camera trajectories, allowing direct 3D scene reconstruction. This supports auto-regressive and consistent world expansion. Experiments demonstrate high visual fidelity and strong spatial coherence in both generated videos and point clouds.

|[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]|
|---|

|[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>|
|---|

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

|[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>|
|---|

(a)

Input

Clip-1 Clip-2 Clip-3

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

“A Tesla Model 3 next to a yellow tent”

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- (b)
- (c)
- (d)

[Figure 215]

[Figure 216]

Trellis Rodin v1.5 Hunyuan-3D v2.5 Ours

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Change to an Americanstyle village

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

15 30 45 60

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Change to nighttime

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

15 30 45 60

Input Output

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

RGB Input

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

15 30 45 60

5

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Depth Output

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

5 15 30 45 60

- Fig. 6. Applications: (a) Long-range video generation. (b) Image-to-3D generation. (c) World-consistent video style transfer. (d) Monocular video depth estimation.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

GTRGB-onlyRGB-DOursInput

(a) (b) (c) (d)

[Figure 276]

- Fig. 7. Qualitative results on ablation study. We compare the video models in our three training stages. Our final model achieves the highest quality.

[Figure 277]

[Figure 278]

[Figure 279]

num.: 302,992 num.: 132,715 num.: 194,253

PointCullingSmoothSampling

(a) all points (b) invisible points (c) + normal check

[Figure 280]

[Figure 281]

[Figure 282]

(a) video clip 1 (b) w/o sampling (c) w/ sampling

Fig. 8. Qualitative results on ablation study. We compare the video models in our three training stages. Our final model achieves the highest quality.

References

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. 2025. ReCamMaster: CameraControlled Generative Rendering from A Single Video. arXiv:2503.11647 [cs.CV] https://arxiv.org/abs/2503.11647

Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. 2025. VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. arXiv preprint arXiv:2503.05639 (2025).

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22563–22575.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. 2024. Video generation models as world simulators. OpenAI Blog 1 (2024), 8.

Luxi Chen, Zihan Zhou, Min Zhao, Yikai Wang, Ge Zhang, Wenhao Huang, Hao Sun, Ji-Rong Wen, and Chongxuan Li. 2025. FlexWorld: Progressively Expanding 3D Scenes for Flexiable-View Synthesis. arXiv:2503.13265 [cs.CV] https://arxiv.org/ abs/2503.13265

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. 2025. WorldScore: A Unified Evaluation Benchmark for World Generation. arXiv preprint arXiv:2504.00983

(2025).

Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur Brussee, Ricardo MartinBrualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole*. 2024. CAT3D: Create Anything in 3D with Multi-View Diffusion Models. Advances in Neural Information Processing Systems (2024).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. arXiv preprint arXiv:2404.02101 (2024).

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221 (2022).

Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2024. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773 (2024).

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. 2024. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

Hunyuan3D. 2025. Hunyuan-3D. (2025). https://3d-models.hunyuan.tencent.com Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.

- 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
- 4 (2023), 139–1.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux. Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. 2024. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748 (2024).

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22160–22169.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot One Image to 3D Object. arXiv:2303.11328 [cs.CV]

Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang Zhang.

2024. Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. In European Conference on Computer Vision. Springer, 265–282.

Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. 2024. Freelong: Training-free long video generation with spectralblend temporal attention. arXiv preprint arXiv:2407.19918 (2024).

Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. 2025. You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale. In IEEE/CVF conference on computer vision and pattern recognition.

Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. 2024. Lt3sd: Latent trees for 3d scene diffusion. arXiv preprint arXiv:2409.08215 (2024).

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

William Peebles and Saining Xie. 2022. Scalable Diffusion Models with Transformers. arXiv preprint arXiv:2212.09748 (2022).

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin NimierDavid, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. 2025. GEN3C:

3D-Informed World-Consistent Video Generation with Precise Camera Control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

RodinAI. 2025. Rodin. (2025). https://hyper3d.ai Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Runway. 2024. Introducing gen-3 alpha: A new frontier for video gneration. (2024). https://runwayml.com/research/introducing-gen-3-alpha

Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulò, Lorenzo Porzi, and Peter Kontschieder. 2025. A Recipe for Generating 3D Worlds From a Single Image. arXiv preprint arXiv:2503.16611 (2025).

Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. 2024. GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping. arXiv preprint arXiv:2405.17251 (2024).

Zachary Teed and Jia Deng. 2021. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems 34 (2021), 16558–16569.

Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. 2015. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision. 4489–4497.

Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li.

2023. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264 (2023).

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651 (2025).

Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. 2024a. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. arXiv preprint arXiv:2410.19115 (2024).

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024b. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Ethan Weber, Riley Peterlinz, Rohan Mathur, Frederik Warburg, Alexei A Efros, and Angjoo Kanazawa. 2024. Toon3D: Seeing Cartoons from a New Perspective. arXiv preprint arXiv:2405.10320 (2024).

Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. 2024. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21551–21561.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024).

Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. 2024. Citydreamer: Compositional generative model of unbounded 3d cities. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9666–9675.

Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. 2024. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991 (2024).

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024).

Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. 2023. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346 (2023).

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. 2024. From slow bidirectional to fast causal video generators. arXiv preprint arXiv:2412.07772 (2024).

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. 2024a. WonderWorld: Interactive 3D Scene Generation from a Single Image. arXiv:2406.09394 (2024).

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, and Charles Herrmann. 2023a. WonderJourney: Going from Anywhere to Everywhere. arXiv preprint arXiv:2312.03884 (2023).

Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. 2023b. Language Model Beats Diffusion–Tokenizer is Key to Visual Generation. arXiv preprint arXiv:2310.05737 (2023).

Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024b. ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis. arXiv preprint arXiv:2409.02048 (2024).

Zhimeng Zhang, Zhipeng Hu, Wenjin Deng, Changjie Fan, Tangjie Lv, and Yu Ding. 2023. Dinet: Deformation inpainting network for realistic face visually dubbing on high resolution video. In Proceedings of the AAAI conference on artificial intelligence, Vol. 37. 3543–3551.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. 2025. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 (2025).

Jensen (Jinghao) Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. 2025. Stable Virtual Camera: Generative View Synthesis with Diffusion Models. arXiv preprint arXiv:2503.14489 (2025).

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018).

Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. 2024. Allegro: Open the Black Box of Commercial-Level Video Generation Model. arXiv preprint arXiv:2410.15458

(2024).

In this supplement, we will introduce more details of training implementation (Sec. A), our video diffusion model (Sec. B), and our video data engine (Sec. C). Finally, we provide more generation results in Sec. D.

VGGT Camera params

Metric3D Metric depth

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

- A Implementation Details

Our trainingbasicallyfollowstheimage-to-video model of HunYuanVideo [Kong et al. 2024]. We divide the training into three stages: the first stage only trains the RGB video model; in the second stage, depth is introduced into the training; and in the third stage, the DiT parameters are frozen and ControlNet blocks are incorporated for training. We use all three datasets in the first training stage. However, DL3DV is removed in the second stage due to its fast camera motion, which makes it unsuitable for depth training. In the third stage, we train solely on the UE dataset with its ground-truth depth. During training, we randomly select a width-height ratio from [1, 1.25, 1.5, 1.75] to support the generation of videos with multiple aspect ratios. The number of generation frames for a single pass is 49.

“A mountain lake reflecting the peaks and a rustic cabin on stilts.”

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

|[Figure 315]|
|---|

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

|[Figure 321]<br><br>[Figure 322]<br><br>[Figure 323]|
|---|

Noisy latent Image latent Cond latent Mask

×𝑁

×𝑁

Control Block 𝑓

Control Block 𝑓

Double-Stream 𝑓

Single-Stream 𝑓

- Stage-1
- Stage-2
- Stage-3

Fig. 9. Details of world-consistent diffusion model.

- B World-Consistent Video Diffusion We provide the details of our video diffusion model in Figure 9. The input of the diffusion model includes noisy latents z𝑡, input image latents z𝑟0, condition latents zˆ0, and down-sampled mask𝑚. To align

the temporal dimension, we pad z𝑟0 with zero latents. In the first stage of training, only the RGB-related latents are concatenated

in the channel dimension and are then fed to the diffusion model. In the second stage, we inject depth-related latents into the input. We fine-tune the parameters of the original diffusion structure in the first two stages, two additional Transformer blocks are trained in the final stage. The aggregated features in these two blocks are added back on a pixel-wise basis.

- C Scalable Video Data Engine

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

quantile scale

MoGE

[Figure 340]

[Figure 341]

[Figure 342]

Relative depth

[Figure 343]

Depth estimate

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

least squares

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

Fig. 10. Overview of our scalable video data engine.

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Input Ours 3DGS

Fig. 11. Training warp images compare.

poses. To further improve the depth estimation, we then employ MoGE [Wang et al. 2024a] as a robust depth estimator. Specifically, we first convert the depth into disparity, and then use a least squaresbased optimization strategy to minimize the disparity difference between the depth frames generated by VGGT and those generated by MoGE. The optimization is represented as:

2

1 𝑑𝑉𝐺𝐺𝑇

𝑠𝑐𝑎𝑙𝑒 𝑑𝑀𝑜𝐺𝐸 + 𝑏𝑖𝑎𝑠 −

, (5)

min

M ·

𝑠𝑐𝑎𝑙𝑒,𝑏𝑖𝑎𝑠

where 𝑠𝑐𝑎𝑙𝑒 and 𝑏𝑖𝑎𝑠 represent the scale and shift factors respectively. The mask M represents the valid non-sky regions.

Finally, to ensure scale uniformity across datasets, we estimate the metric depth range using Metric3D [Hu et al. 2024] and map the estimated depths into this range.

𝑞(0.8, dMetric3D) − 𝑞(0.2, dMetric3D) 𝑞(0.8, dMoGE) − 𝑞(0.2, dMoGE)

Accuratecamera parameters anddepth are crucial for model training. As shown in Fig. 10, we propose a more robust data processing pipeline. Compared to Flexworld [Chen et al. 2025], since our depth estimation method is more consistent and accurate than the depth rendered by 3DGS, our warped images are more precise, as shown in Fig. 11.

(6)

𝑠metric =

𝑑metric = 𝑠metric · 𝑑𝑀𝑜𝐺𝐸 (7)

𝑅 𝑠metric ·𝑇 0 1

(8)

𝐶𝑐𝑎𝑚𝑚𝑒𝑡𝑟𝑖𝑐 =

Specifically, we first use VGGT [Wang et al. 2025] to estimate camera parameters and depth for all video frames. The depth estimated by VGGT is not accurate enough, but it is aligned with camera

where 𝑞(𝑝, x) represents the 𝑝-th quantile of vector x, while 𝑑𝑚𝑒𝑡𝑟𝑖𝑐 and 𝐶𝑐𝑎𝑚𝑚𝑒𝑡𝑟𝑖𝑐 denote the final metric depth and camera extrinsics, respectively.

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

Input Ours VGGT

Fig. 12. Comparison of initialization point clouds of ours and VGGT.

D More Results

We provide visualization results for the initialization of 3D reconstruction in Figure 12. Our point cloud results are much better than VGGT, demonstrating that our depth estimation is more accurate than VGGT.

We also provide more generation results in Figure 13 and Figure 14.

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

|[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]|
|---|

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

|[Figure 383]<br><br>[Figure 384]<br><br>[Figure 385]<br><br>[Figure 386]<br><br>|
|---|

[Figure 387]

5

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

|[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]|
|---|

[Figure 398]

5

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

|[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>|
|---|

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

|[Figure 416]<br><br>[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>|
|---|

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

|[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]<br><br>[Figure 429]<br><br>[Figure 430]<br><br>|
|---|

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

|[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]<br><br>|
|---|

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

|[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]|
|---|

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

|[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]<br><br>[Figure 460]|
|---|

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

|[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]<br><br>|
|---|

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

|[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>|
|---|

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

|[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]|
|---|

Input Camera Control Generated Video

Fig. 13. More Results.

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

|[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]<br><br>[Figure 501]<br><br>[Figure 502]<br><br>|
|---|

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

|[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]<br><br>[Figure 513]<br><br>|
|---|

[Figure 514]

5

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

|[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]|
|---|

[Figure 525]

5

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

|[Figure 532]<br><br>[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>|
|---|

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

|[Figure 543]<br><br>[Figure 544]<br><br>[Figure 545]<br><br>[Figure 546]<br><br>|
|---|

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

|[Figure 553]<br><br>[Figure 554]<br><br>[Figure 555]<br><br>[Figure 556]|
|---|

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

|[Figure 563]<br><br>[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]<br><br>[Figure 567]<br><br>|
|---|

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

|[Figure 574]<br><br>[Figure 575]<br><br>[Figure 576]<br><br>[Figure 577]<br><br>[Figure 578]<br><br>|
|---|

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

|[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]<br><br>[Figure 589]<br><br>|
|---|

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

|[Figure 596]<br><br>[Figure 597]<br><br>[Figure 598]<br><br>[Figure 599]<br><br>[Figure 600]<br><br>|
|---|

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

|[Figure 607]<br><br>[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]<br><br>[Figure 611]<br><br>|
|---|

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

|[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]<br><br>[Figure 621]|
|---|

Input Camera Control Generated Video

Fig. 14. More Visualization Results.

