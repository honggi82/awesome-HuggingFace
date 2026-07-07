##### SPATIALGEN: Layout-guided 3D Indoor Scene Generation

# arXiv:2509.14981v4[cs.CV]15Jan2026

Chuan Fang1∗, Heng Li1, Yixun Liang1, Jia Zheng2, Yongsen Mao2, Yuan Liu1, Rui Tang2, Zihan Zhou2, Ping Tan1 1Hong Kong University of Science and Technology, 2Manycore Tech Inc.

[Figure 1]

[Figure 2]

https://manycore-research.github.io/SpatialGen

[Figure 3]

Layout Estimator

|[Figure 4]|
|---|

[Figure 5]

A modern minimalist living room.

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

Text to image to 3D scene Single image to 3D scene Create new 3D Scene from video

Figure 1. Given a 3D semantic layout, SPATIALGEN can generate a 3D indoor scene conditioned on either a textual description (left) or a reference image (middle). Furthermore, it can transform a real-world scene, where its 3D layout is estimated from a video by a layout estimator [27], into some brand new scenes.

###### Abstract

Creating high-fidelity 3D models of indoor environments is essential for applications in design, virtual reality, and robotics. However, manual 3D modeling remains timeconsuming and labor-intensive. While recent advances in generative AI have enabled automated scene synthesis, existing methods often face challenges in balancing visual quality, diversity, semantic consistency, and user control. A major bottleneck is the lack of a large-scale, high-quality

*Work done during an internship at Manycore Tech Inc.

dataset tailored to this task. To address this gap, we introduce a comprehensive synthetic dataset, featuring 12,328 structured annotated scenes with 57,431 rooms, and 4.7M photorealistic 2D renderings. Leveraging this dataset, we present SpatialGen, a novel multi-view multi-modal diffusion model that generates realistic and semantically consistent 3D indoor scenes. Given a 3D layout and a reference image (derived from a text prompt), our model synthesizes appearance (color image), geometry (scene coordinate map), and semantic (semantic segmentation map) from arbitrary viewpoints, while preserving spatial consistency across modalities. SpatialGen consistently generates supe-

rior results to previous methods in our experiments. We are open-sourcing our data and models to empower the community and advance the field of indoor scene understanding and generation.

###### 1. Introduction

Indoor scene generation aims to produce spatially coherent and photorealistic 3D indoor environments. As a fundamental challenge in computer vision, this task underpins diverse applications, including immersive films and games, interior design, and augmented/virtual reality (AR/VR). Moreover, it also provides diverse and physically realistic environments in robotic simulation for training and evaluating robot navigation and interaction capabilities.

A major consideration in developing 3D scene generation methods is the trade-off between realism and scene diversity. Procedural modeling methods [9, 33, 55] leverage hand-crafted heuristic rules and geometric constraints in the graphics engines, which produce highly realistic and physically plausible indoor environments. However, these scenes lack diversity. Recent 3D generative methods automatically generate scene layouts [28, 46] or other 3D representations like NeRFs [1] and 3D Gaussians [21]. But these methods exhibit limited layout and appearance realism, primarily due to the scarcity of annotated 3D data. In comparison, image-based methods utilize diffusion models to generate panoramas [39, 47] or multi-view images [11, 45] followed by 3D reconstruction. By leveraging powerful 2D priors, these methods show promise in striking a better balance between realism and scene diversity. Image-based methods, however, face additional challenges in multi-view semantic consistency. While recent video generation methods [26, 34, 56] have improved temporal coherence, synthesizing semantically consistent content when exploring beyond input views remain highly challenging.

To this end, 3D semantic layout prior (Figure 1) has been employed in the literature to guide the generation process. But due to the lack of a large-scale dataset with paired 3D layout and images (or videos), existing layoutconditioned methods resort to one of the following two strategies: score distillation [4, 6, 52, 67] and panoramaas-proxy [8, 39]. The former directly distills powerful 2D pre-trained models for 3D content creation, avoiding the need for large-scale training data. But due to the inherent limitation of the SDS method [30], results produced by these methods suffer from severe visual artifacts (e.g., over-saturation, lack of details). In contrast, the latter makes use of a special type of data, namely panoramas, for which large datasets with diverse scenes and annotations are available (e.g., Structured3D dataset [63]). However, since panorama images are captured at fixed camera locations, models trained on such data have limited ability to

extrapolate to novel viewpoints, restricting their application in real-world tasks.

To overcome these limitations, we collect a new indoor scene dataset on a much larger scale. Our dataset features 4.7M panoramic images with precise 2D and 3D layout annotations, spanning 57,431 rooms and 12,328 scenes. With this dataset, we take a new approach to 3D scene synthesis by building a scalable multi-view diffusion (MVD) model conditioned on 3D layout priors, which achieves high semantic consistency while maintaining the realism and scene diversity in the results.

We introduce SPATIALGEN, a novel framework for highfidelity 3D indoor scene generation from a 3D room layout. First, we convert the 3D semantic layout into view-specific representations comprising coarse semantic maps and scene coordinate maps [40]. Second, we design a layout-guided attention mechanism that alternatively operates through: (i) cross-view attention for consistent information propagation across different viewpoints; (ii) cross-modal attention for fine-grained feature alignment between appearance, semantic, and geometric representations. This mechanism enables the joint synthesis of photorealistic RGB images, precise object semantic maps, and accurate scene coordinates for both input and novel viewpoints. Finally, we employ an iterative multi-view generation strategy to ensure complete scene coverage, followed by 3D Gaussian splatting optimization that reconstructs an explicit radiance field to enable free-viewpoint rendering.

Our main contributions are summarized as follows:

- • We introduce a new large-scale dataset featuring over 4.7M panoramic images of 57,431 rooms and precise 2D and 3D layout annotations. This dataset fills a critical gap in 3D scene modeling by providing comprehensive multiview data with structural annotations.
- • We present SPATIALGEN, a new framework for layoutguided indoor scene generation. At the core of this framework is a novel multi-view multi-modal image diffusion method conditioned on a given layout prior, which generates semantically and geometrically consistent images from arbitrary viewpoints.
- • Extensive evaluations conducted on text or image to 3D scene generations demonstrate that our method generates substantially more realistic and plausible 3D scenes.

###### 2. Related Work

Procedural & 3D-based Scene Generation. Procedural generation (PCG) methods [32, 33] create 3D scenes with hand-crafted rules or constraints. Recent approaches integrate large language models (LLMs), either to generate scene layouts for subsequent object retrieval or shape synthesis [9, 44, 53], or to act as agents that produce Python scripts controlling procedural frameworks [17, 43].

3D-based methods generate 3D scene representations us-

Table 1. Statistics of the datasets for indoor scene generation. †: object annotations are provided by Ctrl-Room [8]. Dataset (year) source #scenes #images #objects image type

annotations layouts objects

SUN R-GBD (2015) real - 10.3K 59K perspective image • • ScanNet (2017) real 1,513 2.5M 36K regular video • Matterport3D (2017) real 90 10.8K 41K sparse panoramas • ScanNet++ v2 (2024) real 1,006 11.1M 111K regular video •

Structured3D (2020) syn. 3,500 196.5K 150K† panorama image • •† Hypersim (2021) syn. 461 77.4K 58K regular video • SPATIALGEN dataset (ours) syn. 12,328 4.7M 1M panoramic video • •

ing generative models trained on datasets with 3D annotations. ATISS [28] and DiffuScene [46] predict compact layout parameters for scene objects. DiffInDScene [19], PDD [23], and SceneFactor [2] introduce a semantic layout as an intermediate guide to generate the indoor scene with an explicit geometric representation. But the lack of annotated 3D scene datasets results in subpar performance and limited generalization of such methods.

Image-based Scene Generation. In contrast, image-based methods exploit strong 2D priors in pretrained diffusion models to obtain photorealistic and diverse results. MVDiffusion [47] and PanoFusion [58] finetune a latent diffusion model [36] to generate a 360-degree panorama of a scene. Text2Room [16] and LucidDreamer [5] start with an initial RGB image and iteratively build the 3D scene by progressively warping and inpainting. CAT3D [11] and Bolt3D [45] trained a multi-view LDM to generate novel views from input images, followed by a 3D reconstruction. Despite these advances, existing methods struggle to synthesize large viewpoint changes [50, 56] and semantically coherent scenes [11, 38, 45] beyond observed areas.

The line of work that most closely relates to ours employs 3D layout prior to guide the generation process. Setthe-Scene [6], SceneCraft [52], and Layout2Scene [4] generate 3D scenes by distilling the pretrained image diffusion models conditioned on a given semantic layout. These methods achieve better view and semantic consistency, but the realism and controllability remain limited. While CtrlRoom [8] and ControlRoom3D [39] generate panoramas with high visual fidelity, they struggle to extrapolate the scene beyond a fixed camera location without resorting to a dedicated room completion procedure. We argue that these limitations stem from the scarce scale and diversity of available 3D scene datasets, hindering the learning of robust 3D priors.

Indoor Scene Dataset. Existing indoor datasets are either captured from real-world scenes using RGB [22, 66] or RGB-D [3, 7, 41, 54] sensors or professionally designed with curated 3D CAD furniture models [10, 35, 42, 63]. The real-world dataset provides a physically realistic appearance observation of 3D scenes; however, collecting and an-

[Figure 20]

Figure 2. Illustration of our dataset. For each scene, we provide comprehensive panoramic renderings and 3D layout annotation.

notating these data typically requires significant resources in terms of cost and labor. On the other hand, indoor synthetic datasets address the constraints of real-world data by supplying extensive, varied, and richly annotated scenes. In addition, Structured3D [63] and Hypersim [35] utilize the advanced render engine for photorealistic image rendering with accurate 2D labels. However, the camera view is limited, which restricts the downstream application.

###### 3. SPATIALGEN Dataset

We summarize the commonly used indoor scene dataset for layout-conditioned scene synthesis in Table 1. As one can see, the real-world datasets suffer from a limited number of scenes, incomplete 3D annotations, and inconsistent annotation quality. Synthetic datasets are easier to annotate with ground-truth 3D labels, but they still have limitations in scene diversity (for example, Hypersim only has 461 scenes) or camera viewpoints (for example, Structured3D provides a single panorama for each room).

In this paper, we build a new dataset to train generative

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Layout-Guided Multi-View Multi-Model Diffusion

[Figure 27]

[Figure 28]

Multi-view

[Figure 29]

Model-view Multi-modal Alternating Attention

[Figure 30]

[Figure 31]

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|[Figure 42]|
|---|

[Figure 43]

[Figure 44]

###### …

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Input Image

[Figure 50]

[Figure 51]

Multi-modal

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

3D Reconstruction & Understanding

[Figure 56]

[Figure 57]

Noisy image, coordinate, semantic maps

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

…

[Figure 65]

…

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Render

[Figure 70]

Generated Views

Multi-view Multi-modal Diffusion

[Figure 71]

Condition coordinate & semantic maps

[Figure 72]

3D Layout

Figure 3. Overall pipeline. SPATIALGEN takes as input a 3D semantic layout and one or more posed images, to create a 3D scene. First, we generate per-view RGB images, scene coordinate maps, and semantic segmentation maps from a Layout-Guided Multi-view Multi-modal diffusion model. Then, we adopt an iterative dense view generation strategy to generate images at more sampled viewpoints. Finally, these images are fed into a 3D reconstruction method to produce the final result.

on a 3D layout with reference images. Specifically, given a semantic layout and one or more source images, SPATIALGEN first utilizes a layout-guided multi-view multimodal diffusion model to generate dense views of the target scene via Iterative Dense View Generation (detailed in Section 4.3). Then, we recover those dense views to a unified semantic Gaussian Splatting via an off-the-shelf reconstruction method [57].

models for 3D indoor scenes. Our dataset is based on a large repository of house designs sourced from an online platform in the interior design industry. Most of these designs are designed by professional designers and are intended for real-world production.

As shown in Figure 2, we create physically plausible camera trajectories that navigate smoothly through each scene while avoiding obstacles. These trajectories are sampled at 0.5m intervals to ensure comprehensive spatial coverage. For each viewpoint, we generate photorealistic panoramic renderings using an industry-leading rendering engine, capturing color, depth, normal, semantic, and instance segmentation data. We further convert the panoramic image into multiple perspective images using equilib [14]. To ensure both quality and diversity, we apply rigorous filtering criteria during dataset curation, resulting in 12,328 distinct scenes encompassing 57,431 individual rooms with diverse room types. Each scene is annotated with precise 3D layouts and divided into 57,381/50 scenes for training/testing, respectively.

We start by providing a brief overview of multi-view diffusion models in Section 4.1. Then, we introduce our layout-guided latent diffusion model in Section 4.2, followed by the iterative generation scheme in Section 4.3 and the 3D reconstruction process in Section 4.4.

###### 4.1. Preliminaries

Multi-view Diffusion Model. A multi-view latent diffusion model takes a single or multiple posed source views as input and generates multiple novel images in some target camera views. To incorporate multi-view conditioning, it typically involves two designs: (1) 2D attention layers are improved to a 3D-aware or multi-view aware attention mechanism, such as epipolar constraint [13], to capture multi-view features across different source views. (2) Camera poses are encoded by Plucker coordinate maps [29, 59] and then processed by a Transformer to compute viewconditioned embeddings.

Figure 2 also shows some panoramas and 3D layout annotation from our dataset. Our dataset offers comprehensive structural layout annotations, including architecture elements (i.e., walls, doors, and windows). We further simulate diverse camera motion patterns from panoramic video data to train and evaluate the generation capabilities of existing approaches – a crucial advantage over limited rulebased trajectories from existing dataset. The empirical benefits of this dataset are demonstrated in the experiments.

Given M input views IM = {I1,...,IM} with camera poses CM = {C1,...,CM}. Multi-view diffusion aims to predict N new view images IN = {IM+1,...,IM+N} in camera poses CN = {CM+1,...,CM+N}. In other words, the multi-view latent diffusion model aims to learn the following joint distribution,

###### 4. Method

We introduce SPATIALGEN, a novel method that generates Gaussian Splatting [20] scenes with semantics conditioned

###### p(IN | IM,CM+N), (1)

where CM+N = {C1,...,CM+N} includes camera poses for both input and output views.

Attention Across Views

(RGB as example)

|[Figure 73]| | |
|---|---|---|
| | | |
| | | |

|[Figure 74]| | |[Figure 75]| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

Layout Condition in MVD. A 3D semantic layout provides an informative description of the scene. Following previous works [8, 39, 52], we represent the layout as a set of semantic bounding boxes of objects {bk}Kk=1, where each box bk includes the center location lk ∈ R3, the size sk ∈ R3, the orientation rk ∈ R around the vertical axis, and the category label zk. For each viewpoint with camera parameter Cn = (Kn,Tn), we render a semantic map Snlayout and a depth map Dnlayout of the bounding box of the 3D layout. The Dnlayout is then converted to the scene coordinate map Pnlayout = Tn · (Kn−1 · Dnlayout). In this way, we obtain the input layout conditions SlayoutM+N,PlayoutM+N = {Snlayout,Pnlayout}Mn=1+N for the latent diffusion model. We use scene coordinate maps instead of depth maps to represent 3D scene geometry because they encode the scene in a globally consistent manner. As discussed in previous work [45, 61], they facilitate learning multi-view geometric consistency in the latent diffusion model. Therefore, extending Eq. (1), the joint distribution to be learned for layout-conditioned multi-view image generation can be formulated as follows,

[Figure 76]

| | | |
|---|---|---|
| | | |
| | | |

|[Figure 77]| | |
|---|---|---|
| | | |
| | | |

|[Figure 78]| | |
|---|---|---|
| | | |
| | | |

Attention Across Modalities

- Figure 4. Multi-view and multi-modal alternating attention. It alternates between enforcing multi-view consistency and multimodal fidelity within a unified attention mechanism.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

a) Image-VAE b)Ours W/O 𝓛𝒈𝒓𝒂𝒅 c) Ours W/ 𝓛𝒈𝒓𝒂𝒅 d)Ground Truth

- Figure 5. Comparison of reconstruction results for scene coordinate map. The image VAE (a) generates noisy results, and the SCM-VAE without gradient loss (b) produces distorted results. Our SCM-VAE (c) accurately reconstructs the scene geometry.

p(IN | IM,SlayoutM+N,PlayoutM+N,CM+N). (2)

Note that the layout conditions Sn and Pn only provide a coarse description of the bounding box without pixel-level details, as shown in Figure 3.

We use a v-parametrization and a v-prediction loss for the diffusion model [37]. Following CAT3D [11], the model is trained on a total of 8 views, with randomly sampled {1,3,7} views as source views. We use the ground-truth scene coordinate map for warping in training, and use the predicted scene coordinates during inference.

###### 4.2. Layout-guided Multi-view Diffusion Model

Since the input layout maps do not contain pixel-level details, we further predict a pixel-wise semantic map Sn, scene coordinate map Pn for each viewpoint. This joint learning scheme improves 3D consistency in two ways:

Multi-view Multi-modal Alternating Attention. With our formulation Eq. (3), our objective is to generate output that is consistent with multiple viewpoints and modalities. We observe that a simple modification to the standard architecture, namely an alternating attention mechanism, is effective in preserving the desired consistency. As illustrated in Figure 4, the new architecture operates through complementary attention pathways: cross-view attention and cross-modal attention. Inspired by previous works [11, 24], our cross-view attention processes reshaped tokens along the view dimension (e.g., {tI1,tI2,...,tIM+N} for all RGB images), allowing feature aggregation across multiple views in each modality. While cross-modal attention operates within each view, observing modality-specific tokens (e.g., {tIn,tSn,tPn } for image, semantics, and geometry) to achieve fine-grained feature alignment. This design achieves a balance between integrating information across different views and different modalities.

- • Explicit 3D supervision. By explicitly integrating both geometric and semantic maps into the latent diffusion model, SPATIALGEN leverages direct 3D supervision to achieve high-fidelity novel view synthesis results while maintaining cross-view consistency.
- • Cross-view guidance. With the additional pixel-wise scene coordinate maps, we provide fine-grained guidance for diffusion by computing a warped image at any target view from the input images. Specifically, we adopt the point cloud based render [18] to obtain the warped image

Inwarp,∀n ∈ {M +1,...,M +N}. The warped image is encoded and concatenated with the original noise map In to form a conditioning signal for the target view, which

we denote as augmented target views Iˆn = [In;Inwarp]. The joint distribution to be learned now becomes,

p(ˆIN,SM+N,PM+N | IM,SlayoutM+N,PlayoutM+N,CM+N).

Scene Coordinate Map VAE (SCM-VAE). A standard image VAE pretrained on RGB images generalizes well to se-

(3)

mantic maps, but fails to accurately reconstruct scene coordinate maps, leading to poor geometric fidelity. See Figure 5(a) for an example. To address this, we introduce SCM-VAE, which encodes a scene coordinate map P into a latent representation z as z = ξ(P) and reconstructs z into a scene coordinate map with an uncertainty map as {P,ˆ c} = D(z), where ξ denotes the encoder and D is the decoder. The SCM-VAE is trained by fine-tuning the decoder D with an additional output dimension c from an image diffusion VAE, while keeping the encoder ξ frozen. c is activated by c = 1+exp(c) to ensure a strictly positive confidence [48]. The training objective combines standard VAE reconstruction with geometry-specific loss:

L = Lrec + λ1Lgrad, (4) Lrec = c ⊙ ∥Pˆ − P∥ − α log c, (5)

4

∥(∇Pˆis − ∇Pis)∥, (6)

Lgrad =

s=1

where α = 0.2 and ⊙ denotes element-wise multiplication. Here, we follow previous monocular depth estimation works [12] to use a multiscale gradient loss Lgrad to improve boundary sharpness in the decoded scene coordinate map. As we can see in Figure 5, our SCM-VAE with Lgrad outperforms the one without the term, especially around complex object boundaries and flat areas.

###### 4.3. Iterative Dense View Generation

Our goal is to generate a complete 3D scene aligned to the given layout and text or image prompt. Although our layout-guided MVD model can generate an arbitrary number of views in principle, it is limited by GPU memory constraints. Thus, instead of generating all views at once, we adopt an iterative view synthesis strategy, a similar approach is also used in previous work [26, 56]. The main idea is to incrementally maintain a colored global point cloud of the scene to enforce appearance consistency between iterations. During each iteration, the point cloud P is projected onto the target views Iwarp to provide pixel-aligned guidance for consistent generation. The SCMs of the target view generated by our diffusion model will be inserted into P. In this way, we can effectively reduce error accumulation. Furthermore, by incorporating the uncertainty map c, we filter out 3D points with uncertainty below a predefined threshold, resulting in cleaner warped images.

The iterative process, detailed in Algorithm 1, proceeds as follows: First, an initial point cloud is built by obtaining the scene coordinate maps PM for the input views IM. Second, at the beginning of each iteration, we render warped images for the target views from the point cloud. Then, we perform inference with not only the input images Im, but also the warped images to ensure global consistency. We

Algorithm 1: Iterative Dense View Generation

Input: input views IM, camera poses for all iterations {CM, CN1, . . . , CNK}, global point cloud P initialized as empty, layout-guided multi-view diffusion model U(·).

Initialization: Obtain the SM and PM: {SM, PM} ← U(IM, CM); Update global point cloud: P ← PM; for k ← 1 to K do

Render warped images: IwarpN

← Render(P, CNk); Augment the target views ˆINk = [INk; IwarpN

k

];

k

Run inference: {ˆINk, SNk, PNk} ← U(IM, CM+Nk); Update global point cloud: P ← P PNk;

end Output: all generated views: {IN1, IN2, . . . , INK}.

then update the point cloud accordingly. Finally, we collect the images generated from all iterations as output.

###### 4.4. Scene Reconstruction and Understanding

Building upon RaDe-GS [57], we reconstruct a 3D scene representation from the densely generated color, geometric, and semantic images. Following Feature-3DGS [65], We augment the standard 3D Gaussians with a semantic feature in each point. The scene is initialized from the predicted point cloud P. During differentiable rendering optimization, we employ a depth supervision loss that utilizes the predicted scene coordinate maps, enabling rapid convergence in just 7,000 steps. As shown in Figure 6, the pipeline produces high-fidelity RGB renderings and geometrically accurate depth reconstructions.

###### 5. Experiments 5.1. Experiment Setup

Benchmark Datasets. We use both existing datasets (i.e., Hypersim [35] and Structured3D [63]) and our new dataset. As discussed in Section 3, one key difference of these datasets is in the diversity of camera viewpoints. Since our dataset provides abundant panorama images at dense locations in each room, we can design various camera movement patterns to conduct an extensive evaluation.

Specifically, we introduce four distinct camera trajectories with different amounts of view overlap and distance between input and target views: (i) Forward: the trajectory follows a linear path with minimal directional variation, simulating steady camera movement. (ii) Inward Orbit: both the input and output views are directed toward the center of the room, ensuring substantial view overlap; (iii) Outward Orbit: the input and output views are at the same location, but oriented differently, with less than 45◦ overlap at adjacent views. (iv) Random Walk: input and output views are sampled from a continuous random-walk path,

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

Layout SpatialGen SpatialGen† SceneCraft Set-the-Scene

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

Layout SpatialGen SceneCraft Set-the-Scene

- Figure 6. Qualitative comparison to score distillation methods on Hypersim [35] (top row) and our dataset (bottom row), In each case, we show the generated color images and depth maps.

Table 2. Comparison to score distillation methods.

###### 5.2.1 Comparison to Score Distillation Methods

Dataset Method CLIP Sim. (%) ↑ Image Reward ↑

For this experiment, we compare to two open-source methods: Set-the-Scene [6] and SceneCraft [52]. The other methods such as Layout2Scene [4] and GALA3D [67] do not release their codes. We conduct experiments on both Hypersim [35] and our new datasets. Evaluation is performed on the test scenes of each target dataset following standard protocols.

Set-the-Scene [6] 25.18 -2.005 SceneCraft [52] 26.94 -1.096 SPATIALGEN† 25.93 -1.168

Hypersim [35]

SPATIALGEN 27.59 -0.285

Set-the-Scene [6] 25.23 -2.100 SceneCraft [52] 18.93 -2.267 SPATIALGEN 27.89 -0.123

Our Dataset

For Hypersim dataset, we directly use the official checkpoints of Set-the-Scene and SceneCraft. To illustrate the benefit of our proposed large-scale dataset, we include two training configurations for our model: SPATIALGEN† which is trained solely on Hypersim, and SPATIALGEN which is trained on a combination of Hypersim and our datasets.

resulting in minimal view overlap. Please refer to the appendix for the visualization of these settings.

Implementation Details. We implement SPATIALGEN in PyTorch. Both the SCM VAE and the latent diffusion model are fine-tuned from stable diffusion 2.1 (SD-2.1) [36]. We use AdamW optimizer [25]. For SCM VAE, we freeze the encoder and only fine-tune the decoder for 10K steps with a batch size of 64. For the latent diffusion model, we finetune it for 35K steps with a batch size of 128. The first 16K steps use resolution 256×256, whereas the remaining steps use resolution 512 × 512. All models are fine-tuned on 64 NVIDIA RTX 4090 GPUs. The learning rate starts at 10−4 and decays by a factor of 0.01 at 90% of the total training process. We render warped images using PyTorch3D [18].

For our dataset, we fine-tune the layout ControlNet of SceneCraft and adapt the layouts to match the input of Setthe-Scene. All methods on our dataset use the Inward Orbit camera trajectory for consistency.

Quantitative Results. Table 2 reports the performance of all methods with established 2D rendering metrics: CLIP similarity score [31] to measure text-image alignment and Image Reward [51] to assess human aesthetic preference.

As one can see, our method performs slightly worse than SceneCraft when trained solely on Hypersim. We hypothesize that Hypersim is too small for powerful latent multi-view diffusion models like the one employed by our method. When trained on both Hypersim and our datasets, our method outperforms both SDS methods on all metrics. SPATIALGEN achieves a significantly higher image-reward score than models trained solely on Hypersim, validating the benefit of our large-scale dataset for high-quality 3D scene generation. Furthermore, when tested on our dataset, SPATIALGEN consistently outperforms the baselines, with a significantly higher image-reward score.

For text-to-3D scene generation, we further train a layout ControlNet [60] to generate the reference image for our latent diffusion model.

###### 5.2. Text-to-3D Scene Generation

As discussed before, existing layout-conditioned text-to-3D methods can be grouped into two categories: score distillation methods [4, 6, 52, 67] and panorama-as-proxy methods [8, 39]. In the following, we compare to these two groups separately.

Qualitative Results. The advantage of our method can be

Table 3. Comparison to panorama-as-proxy method.

Dataset Method CLIP Sim. (%) ↑ Image Reward ↑ Structured3D [63]

Ctrl-Room [8] 25.03 -1.016 SPATIALGEN 23.90 -1.405

Ctrl-Room [8] 22.63 -1.546 SPATIALGEN 27.89 -0.123

Our Dataset

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Ctrl-RoomSpatialGenCtrl-RoomSpatialGen

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

| | |
|---|---|

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Layout

- Figure 7. Qualitative comparison between our method and the panorama-as-proxy baseline [8] on Structured3D [63] (top row) and our dataset (bottom row).

better observed in Figure 6. On both Hypersim and our datasets, SPATIALGEN generates photorealistic scenes with superior details that are well-aligned with the specified layout. In contrast, SceneCraft struggles to balance layout adherence with text-prompt fidelity, and Set-the-Scene takes nearly two hours to synthesize a radiance field that still lacks fine-grained details. While SPATIALGEN† produces blurry and artifact images, further underscores the benefit of our proposed dataset to the final output quality.

###### 5.2.2 Comparison to Panorama-as-Proxy Methods

Next, we compare our method to panorama-as-proxy methods. We choose Ctrl-Room [8] as the baseline because the source code and checkpoints are available. We conduct experiments on both Structured3D and our datasets. For Structured3D dataset, we use ground-truth layouts to ensure consistent inputs for both methods. For our dataset, we use the Inward Orbit camera trajectory for consistency.

Quantitative Results. Table 3 reports the performance of both methods, we take the renderings of the generated mesh from Ctrl-Room for comparison. On Structured3D dataset (which provides only a single panorama per scene), our method still achieves competitive performance. The relatively lower scores are expected as Ctrl-Room is specifically trained to synthesize a single panorama at a fixed camera location. In contrast, our method generates multiple perspective images without explicitly exploiting the fact that all images are taken from a single location.

Nevertheless, the critical advantage of our method is revealed on our dataset: when rendering from novel viewpoints, the performance of Ctrl-Room degrades significantly, whereas our method consistently produces highquality results from arbitrary views.

Qualitative Results. Figure 7 further highlights the advantage our method over panorama-as-proxy method. On our dataset, Ctrl-Room fails to synthesize coherent novel views, exhibiting severe distortions and artifacts. In contrast, our method is not limited to a single camera position (i.e., panorama generation). It achieves high-quality panorama generation on Structured3D while also enabling photorealistic novel view synthesis on our dataset.

###### 5.3. More Experiments and Ablations

Additionally, we evaluate SPATIALGEN on image-to-3D scene generation to validate its generative capability and 3D consistency. Recognizing that well-defined 3D semantic layouts are often unavailable in practice, we further extend the evaluation to video inputs. In this setting, we leverage the state-of-the-art layout estimation model [27] to parse a 3D layout directly from the video. Additional results, implementation details, and ablations are provided in Appendix.B and Appendix.C.

###### 6. Conclusion & Limitations

We present SPATIALGEN, a novel framework for layoutguided 3D indoor scene synthesis. At the core of our pipeline is a multi-view multi-modal diffusion model, which generates images with high visual quality and geometric consistency. To train this model, we collect a new synthetic indoor scene dataset with 4.7M panoramic renderings of 57,431 rooms and the 3D layout annotations. These advancements open new possibilities for downstream applications such as interior design, embodied AI, and AR/VR.

Limitations. First, the cross-view and cross-modal attention introduces additional computational cost to the diffusion model, which limits SPATIALGEN to generate more images at a time. Second, the camera sampling strategy might affect the generation quality. Moreover, since our method relies on an initial RGB image, it is sensitive to any misalignment between the image and the provided 3D layout. We plan to address these challenges in the future.

###### Acknowledgments

This work was supported in part by the Key R&D Program of Zhejiang Province (2025C01001) and HKUST Project 24251090T019. We thank Yingqi Shen, Liangbin Hu, and Fuchun Dong (Manycore Tech) for dataset support; Chenfeng Hou for SDS-based experiments; Zhiwei Wang for ControlNet testing; and Kunming Luo for figure suggestions.

###### References

- [1] Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, Afshin Dehghan, and Joshua Susskind. GAUDI: A neural architect for immersive 3d scene generation. In Adv. Neural Inform. Process. Syst., pages 25102–25116, 2022. 2
- [2] Aleksey Bokhovkin, Quan Meng, Shubham Tulsiani, and Angela Dai. SceneFactor: Factored latent 3D diffusion for controllable 3D scene generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 628–639, 2025. 3
- [3] Angel X. Chang, Angela Dai, Thomas A. Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In IEEE Int. Conf. 3D Vis., pages 667–676, 2017. 3
- [4] Minglin Chen, Longguang Wang, Sheng Ao, Ye Zhang, Kai Xu, and Yulan Guo. Layout2Scene: 3d semantic layout guided scene generation via geometry and appearance diffusion priors. arXiv preprint arXiv:2501.02519, 2025. 2, 3, 7
- [5] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. LucidDreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 3
- [6] Dana Cohen-Bar, Elad Richardson, Gal Metzer, Raja Giryes, and Daniel Cohen-Or. Set-the-Scene: Global-local training for generating controllable nerf scenes. In IEEE Int. Conf. Comput. Vis. Worksh., pages 2920–2929, 2023. 2, 3, 7, 14
- [7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5828–5839,

2017. 3

- [8] Chuan Fang, Yuan Dong, Kunming Luo, Xiaotao Hu, Rakesh Shrestha, and Ping Tan. Ctrl-Room: controllable text-to-3d room meshes generation with layout constraints. In IEEE Int. Conf. 3D Vis., 2025. 2, 3, 5, 7, 8, 14
- [9] Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun R. Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. LayoutGPT: Compositional visual planning and generation with large language models. In Adv. Neural Inform. Process. Syst., pages 18225–18250, 2023. 2
- [10] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, and Hao Zhang. 3D-FRONT: 3d furnished rooms with layouts and semantics. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10933–10942, 2021. 3
- [11] Ruiqi Gao, Aleksander Hoł y´nski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, and Ben Poole. CAT3D: Create anything in 3d with multi-view diffusion models. In Adv. Neural Inform. Process. Syst., pages 75468–75494, 2024. 2, 3, 5, 16
- [12] Cl´ement Godard, Oisin Mac Aodha, and Gabriel J Brostow. Unsupervised monocular depth estimation with left-right consistency. In IEEE Conf. Comput. Vis. Pattern Recog.,

pages 270–279, 2017. 6

- [13] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press,

2003. 4

- [14] haruishi43. Equilib, 2020. 4, 12
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Adv. Neural Inform. Process. Syst., 30, 2017. 15
- [16] Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3D meshes from 2D text-to-image models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7909–7920, 2023. 3
- [17] Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid, and Alireza Fathi. SceneCraft: An llm agent for synthesizing 3d scenes as blender code. In Int. Conf. Mach. Learn., 2024. 2
- [18] Justin Johnson, Nikhila Ravi, Jeremy Reizenstein, David Novotny, Shubham Tulsiani, Christoph Lassner, and Steve Branson. Accelerating 3d deep learning with pytorch3d. In SIGGRAPH Asia 2020 Courses, 2020. 5, 7
- [19] Xiaoliang Ju, Zhaoyang Huang, Yijin Li, Guofeng Zhang, Yu Qiao, and Hongsheng Li. DiffInDScene: Diffusion-based high-quality 3D indoor scene generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4526–4535, 2024. 3
- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 4

- [21] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3D: Real-world camera trajectory and 3D scene generation from text. In Adv. Neural Inform. Process. Syst., pages 75125–75151, 2024. 2
- [22] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. DL3DV-10K: A large-scale scene dataset for deep learning-based 3d vision. In IEEE Conf. Comput. Vis. Pattern Recog., pages 22160– 22169, 2024. 3
- [23] Yuheng Liu, Xinke Li, Xueting Li, Lu Qi, Chongshou Li, and Ming-Hsuan Yang. Pyramid diffusion for fine 3d large scene generation. In Eur. Conf. Comput. Vis., pages 71–87,

2024. 3

- [24] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single image to 3d using cross-domain diffusion. In IEEE Conf. Comput. Vis. Pattern Recog., pages 9970– 9980, 2024. 5
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Int. Conf. Learn. Represent., 2017. 7
- [26] Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. You see it, you got it: Learning 3d creation on pose-free videos at scale. In IEEE Conf. Comput. Vis. Pattern Recog., pages 2016–2029,

2025. 2, 6

- [27] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. SpatialLM: Training large language models for structured indoor modeling. In Adv. Neural Inform. Process. Syst., 2025. 1, 8, 16, 20
- [28] Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, and Sanja Fidler. ATISS: Autoregressive transformers for indoor scene synthesis. In Adv. Neural Inform. Process. Syst., pages 12013–12026, 2021. 2, 3
- [29] Julius Plucker. On a new geometry of space. Phil. Trans. R. Soc, 155:725–791, 1865. 4
- [30] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. DreamFusion: Text-to-3d using 2d diffusion. In Int. Conf. Learn. Represent., 2023. 2
- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Int. Conf. Mach. Learn., pages 8748–8763, 2021. 7
- [32] Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, and Jia Deng. Infinite photorealistic worlds using procedural generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 12630–12641, 2023. 2
- [33] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21783–21794, 2024. 2
- [34] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132, 2025. 2
- [35] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In IEEE Int. Conf. Comput. Vis., pages 10912–10922, 2021. 3, 6, 7, 12, 13, 14
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10684–10695, 2022. 3, 7
- [37] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In Int. Conf. Learn. Represent., 2022. 5
- [38] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, and Jiajun Wu. Zeronvs: Zeroshot 360-degree view synthesis from a single image. In IEEE Conf. Comput. Vis. Pattern Recog., pages 9420–9429, 2024. 3

- [39] Jonas Schult, Sam Tsai, Lukas H¨ollein, Bichen Wu, Jialiang Wang, Chih-Yao Ma, Kunpeng Li, Xiaofang Wang, Felix Wimbauer, Zijian He, Peizhao Zhang, Bastian Leibe, Peter Vajda, and Ji Hou. ControlRoom3D: Room generation using semantic proxy rooms. In IEEE Conf. Comput. Vis. Pattern Recog., pages 6201–6210, 2024. 2, 3, 5, 7
- [40] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. SceneCoordRegression: Scene coordinate regression forests for camera relocalization in RGB-D images. In IEEE Conf. Comput. Vis. Pattern Recog., pages 2930–2937, 2013. 2
- [41] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. SUN RGB-D: A RGB-D scene understanding benchmark suite. In IEEE Int. Conf. Comput. Vis., pages 567–576, 2015. 3
- [42] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica Dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2024. 3
- [43] Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 3D-GPT: Procedural 3D modeling with large language models. arXiv preprint arXiv:2310.12945, 2023. 2
- [44] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. LayoutVLM: Differentiable optimization of 3D layout via vision-language models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 29469–29478, 2025. 2
- [45] Stanislaw Szymanowicz, Jason Y Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T Barron, and Philipp Henzler. Bolt3D: Generating 3D scenes in seconds. In IEEE Int. Conf. Comput. Vis., 2025. 2, 3, 5
- [46] Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. DiffuScene: Scene graph denoising diffusion probabilistic model for generative indoor scene synthesis. In IEEE Conf. Comput. Vis. Pattern Recog., pages 20507–20518, 2024. 2, 3
- [47] Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. MVDiffusion: Enabling holistic multiview image generation with correspondence-aware diffusion. In Adv. Neural Inform. Process. Syst., pages 51202–51233,

2023. 2, 3

- [48] Sheng Wan, Tung-Yu Wu, Wing H. Wong, and Chen-Yi Lee. Confnet: Predict with confidence. In Int. Conf. Acoust. Speech Signal Process., pages 2921–2925, 2018. 6
- [49] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Trans. Image Process., 13(4): 600–612, 2004. 14
- [50] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan.

- MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, 2024. 3
- [51] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for textto-image generation. Adv. Neural Inform. Process. Syst., 36: 15903–15935, 2023. 7
- [52] Xiuyu Yang, Yunze Man, Junkun Chen, and Yu-Xiong Wang. SceneCraft: Layout-guided 3d scene generation. Adv. Neural Inform. Process. Syst., 37:82060–82084, 2024. 2, 3, 5, 7, 14
- [53] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language guided generation of 3D embodied AI environments. In IEEE Conf. Comput. Vis. Pattern Recog., pages 16227–16237, 2024. 2
- [54] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3D indoor scenes. In IEEE Int. Conf. Comput. Vis., pages 12–22,

2023. 3

- [55] Lap Fai Yu, Sai Kit Yeung, Chi Keung Tang, Demetri Terzopoulos, Tony F Chan, and Stanley J Osher. Make it home: automatic optimization of furniture arrangement. ACM Trans. Graph., 30(4), 2011. 2
- [56] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2, 3, 6
- [57] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. RaDe-GS: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467,

2024. 4, 6

- [58] Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. Taming stable diffusion for text to 360 panorama image generation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 6347–6357, 2024. 3
- [59] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In Int. Conf. Learn. Represent., 2024. 4
- [60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE Int. Conf. Comput. Vis., pages 3836–3847, 2023. 7
- [61] Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista Martin, Kevin Miao, Alexander Toshev, Joshua Susskind, and Jiatao Gu. World-consistent video diffusion with explicit 3D modeling. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21685–21695, 2025. 5
- [62] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conf. Comput. Vis. Pattern Recog., pages 586–595, 2018. 14
- [63] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic

- dataset for structured 3d modeling. In Eur. Conf. Comput. Vis., pages 519–535, 2020. 2, 3, 6, 8, 14
- [64] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. ADE20K: Semantic understanding of scenes through the ADE20K dataset. Int. J. Comput. Vis., 127(3):302–321, 2019. 12
- [65] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3DGS: Supercharging 3D gaussian splatting to enable distilled feature fields. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21676– 21685, 2024. 6
- [66] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Trans. Graph., 37(4),

2018. 3

- [67] Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. GALA3D: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting. In Int. Conf. Mach. Learn., 2024. 2, 7

##### SPATIALGEN: Layout-guided 3D Indoor Scene Generation Supplementary Material

In the appendix, we provide more details of SPATIALGEN dataset in Section A, additional experimental results in Section B, and ablation studies in Section C.

###### A. SPATIALGEN Dataset

###### A.1. Dataset Construction

Data Curation. Our dataset is sourced from an online platform in the interior design industry, providing a largescale collection of professional designs intended for realworld applications. We employ a rigorous multi-stage filtering pipeline to ensure both the quality and diversity of the dataset.

The curation process begins by selecting scenes based on four key criteria: (i) professional designer ratings, (ii) the number of renderings generated by the design, (iii) a total floor area exceeding 20m2, and (iv) the presence of more than 35 unique objects.

Then, we extract individual rooms from each selected scene and apply additional filters to retain only those rooms that (i) have a floor area greater than 8m2 and (ii) contain more than 3 unique objects.

For rendering, we use an industry-leading rendering engine to generate images. We simulate physically plausible camera trajectories that navigate smoothly within each room while avoiding obstacles. After rendering, we implement strict quality control measures by discarding lowquality images—specifically those with camera-object intersections, overexposure, or inadequate lighting, as illustrated in Figure 9.

The final dataset consists of 12,328 distinct scenes, 57,431 individual rooms covering a variety of room types, and 4.7M photo-realistic panoramic renderings. The total floor area across all scenes is approximately 914,687m2.

Camera configuration. We capture panoramic renderings at intervals of 0.5m to ensure comprehensive scene coverage, as shown in the top-left of Figure 8. Each panoramic rendering is generated at a resolution of 1024×2048 and includes color, albedo, depth, normal, semantic, and instance maps. The entire rendering process requires approximately 54K GPU hours.

Following an obstacle-avoiding camera trajectory within each room, we obtain dense sequences of panoramic images. Thanks to the 360◦ field-of-view (FoV) of panoramas, we can simulate an unlimited number of perspective images with varying camera configurations. For each panoramic viewpoint, we generate perspective views with different fields-of-view and rotation angles using equirectangular-toperspective projection [14], as illustrated in Figure 10.

Furthermore, we introduce four distinct camera trajectories with varying amounts of view overlap and distances between input and target views: (i) Forward: a linear path with minimal directional variation, simulating steady camera movement; (ii) Inward Orbit: both input and output views are oriented toward the center of the room, ensuring significant view overlap; (iii) Outward Orbit: the input and output views share the same location but have different orientations, resulting in less than 45◦ overlap between adjacent views; and (iv) Random Walk: input and output views are sampled along a continuous random-walk path, with minimal view overlap.

###### A.2. Dataset Statistics

Room type statistics. The resulting dataset contains 12,592 living and dining rooms, 2,179 living rooms, 2,524 study rooms, 8,540 kitchens, 8,460 bathrooms, 1,464 balconies, 9,049 master bedrooms, 8,603 secondary bedrooms, 2,793 children’s rooms, and 4,418 other room types, as illustrated in Figure 11 representing a diverse and substantial collection of indoor environments.

Object category statistics. The raw online designs initially contained approximately 65,000 object categories. We filtered out niche object classes specific to interior design and mapped the remaining objects to 62 common categories from ADE20K [64]. We then curated the object bounding boxes according to the following criteria: (i) objects outside the room layout were discarded; (ii) objects with any edge shorter than 0.1m or longer than 1.8m were excluded. This process yielded a total of 1,046,637 object bounding boxes. Figure 12 shows the distribution of object categories throughout our dataset, excluding the spotlight and other categories (containing 250K and 240K instances, respectively) to improve visualization of the remaining categories.

###### A.3. Dataset Visualization

As shown in Figure 8, our dataset provides high-quality panoramic renderings accompanied by precise 2D annotations and comprehensive 3D structural layouts, including architecture elements (e.g., walls, windows, and doors), which distinguishes it from existing datasets like Hypersim [35], offering extensive evaluation opportunities for scene generation and spatial understanding tasks.

###### B. Additional Experiments and Results

In this section, we show more results of text-to-3D scene generation in Section B.1, and conduct comprehensive experiments of image-to-3D scene generation in Section B.2.

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

Forward

Forward

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

Inward orbit

Inward orbit

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Outward orbit

Outward orbit

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

Randomwalk

Randomwalk

Figure 8. Examples of SPATIALGEN dataset.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

camera–furniture collision over-exposure insufficient illumination

Figure 9. Example of low-quality renderings.

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

- Figure 11. Room type distribution.

[Figure 237]

- Figure 12. Object category distribution.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Perspective image

FoV variation

Rotation variation

Panorama

Figure 10. Camera configuration.

Furthermore, we show generation results from common videos (unposed) captured in indoor scenes in Section B.3.

ing photorealistic and layout-faithful scenes with superior detail. This advantage is evident even against our model trained only on Hypersim (SPATIALGEN†), which produces blurring and ambiguous results, highlighting the benefits of

###### B.1. Text to 3D Scene Generation

As demonstrated in Figure 13, our method outperforms SDS-based baselines on the Hypersim dataset [35], produc-

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

Layout SpatialGen SpatialGen† SceneCraft Set-the-Scene

Figure 13. Qualitative comparison of text-to-3D scene on Hypersim [35] dataset. In each case, we show the generated color images and depth map.

[Figure 273]

[Figure 274]

[Figure 275]

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

###### Layout SpatialGen SceneCraft Set-the-Scene

Figure 14. Qualitative comparison of text-to-3D scene on SPATIALGEN dataset. In each case, we show the generated color images.

our dataset.

In Figure 14, we compare against SceneCraft [52] and Set-The-Scene [6] under diverse 3D layouts on SPATIALGEN dataset. As the increase of layout complexity, while competing methods fail to generate meaningful radiance fields or capture scene details, our approach consistently delivers realistic and coherent results for complex scenes like living and dining rooms.

We further compare our method against the panoramaas-proxy based method, Ctrl-Room [8], on both the Structured3D [63] and SPATIALGEN dataset. We split the panoramic image into 8 perspective images for a direct comparison, as shown in Figure 15.

For the SPATIALGEN dataset, we render a layoutsemantic panorama from a random viewpoint to use as input for Ctrl-Room. We then spatially align its resulting mesh with our generated scene for a fair comparison. The results,

presented in Figure 16, demonstrate that Ctrl-Room exhibits severe stretching artifacts and scale misalignment at novel viewpoints. In contrast, our method consistently produces photorealistic and fully 3D-consistent renderings from all views.

###### B.2. Image to 3D Scene Generation

In this section, we conduct additional image-to-3D scene generation experiments with a focus on two key aspects: (i) generation capability – the ability to synthesize missing regions for large viewpoint changes; (ii) semantic consistency – the ability to produce semantically consistent views aligned with the 3D scene layout. Given the lack of accessible literature on the layout-conditioned image-to-3D scene generation task, we compare our multi-view generation model against the version without utilizing layout priors. We employ PSNR, SSIM [49], LPIPS [62], and

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

SpatialGenCtrl-Room

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

SpatialGenCtrl-Room

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

Layout

- Figure 15. Qualitative comparison with Ctrl-Room on Structured3D for panorama generation. We split the panorama into eight perspective images for a direct comparison. Our method achieves competitive RGB synthesis compared with Ctrl-Room, resulting in photo-realistic scenes that are well-aligned with the provided layout.

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

SpatialGenCtrl-RoomSpatialGenCtrl-Room

Layout

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

- Figure 16. Qualitative comparison with Ctrl-Room on SPATIALGEN dataset. Ctrl-Room exhibits severe stretching artifacts and scale misalignments at novel viewpoints. In contrast, our method consistently produces photorealistic and fully 3D-consistent renderings from all views.

FID [15] to evaluate the quality of image generation.

Quantitative Results. Table 4 reports quantitative results of our method under different camera trajectories. Under all trajectories, the semantic layout improves the results across all metrics. Furthermore, the improved FID shows that our method with layout guidance can capture the underlying data distribution more effectively. These results collectively underscore the critical role of incorporating 3D layout information in novel view synthesis.

Qualitative Results. Figure 17 shows example outputs of our method, including RGB images, scene coordinate maps, and semantic maps. Removing the layout input leads to severe artifacts in occluded regions, revealing the limitations of image diffusion models in capturing 3D scene structures. In addition, the semantic map contains unknown content, suggesting degraded semantic prediction without layout input. In contrast, our method with layout guidance generates better novel view images and achieves more reasonable se-

[Figure 379]

[Figure 380]

[Figure 381]

Layout

[Figure 382]

Input image

SpatialGen w/ layout guidance SpatialGen w/o layout guidance

- Figure 17. Qualitative comparison of image-to-3D scene generation on our dataset. Given a single input image, our method with layout guidance consistently generates better color images, scene coordinate maps, and semantic maps.

Table 4. Experimental results on image-to-3D scene generation under four distinct camera trajectories: Forward, Inward Orbit, Outward Orbit, and Random Walk, with gradually reduced view overlaps.

serve the structural layout of the original video while altering its stylistic and semantic content based on the text description. We validate this video-to-new-scenes application on the SpatialLM test set. Figure 23 shows qualitative results.

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Forward

###### B.4. Failure Cases

- SPATIALGEN (w/o layout) 11.47 0.49 0.59 67.96 SPATIALGEN (w/ layout) 17.59 0.69 0.32 34.98 Inward orbit

- SPATIALGEN (w/o layout) 12.57 0.48 0.54 64.14 SPATIALGEN (w/ layout) 17.30 0.66 0.33 35.57

Because SPATIALGEN relies on an initial RGB image, if the images happen to be inconsistent with the given 3D layout, then it fails to generate consistent results. In this example of text-to-3D scene generation in Figure 19, we first convert the provided layout and textual description into an initial RGB image. However, the initialization process yields an RGB output (highlighted in red) that does not align with the given layout—specifically, the orientation of the bed is entirely reversed compared to the ground truth. Consequently, during the subsequent generation stage, the resulting outputs exhibit inconsistencies and fail to align properly with both the layout and the initial RGB image.

Outward orbit SPATIALGEN (w/o layout) 11.14 0.60 0.47 76.73

- SPATIALGEN (w/ layout) 13.32 0.59 0.46 57.76

Random walk SPATIALGEN (w/o layout) 11.26 0.45 0.59 98.42

- SPATIALGEN (w/ layout) 14.07 0.62 0.45 52.10

mantic and geometric predictions.

In Figure 18, given a reference image (highlighted in orange box) across four diverse camera trajectories , our method successfully generates 3D-consistent novel views and synthesizes semantically plausible content for areas beyond the original input view.

###### C. Ablation Studies

In this section, we conduct additional ablation studies to verify the layout guidance, design choice of network architecture, and the number of input views.

Ablation on layout guidance. We first study the effect of layout guidance to validate our design. We compare our full model (denoted as W/ Layout) against a variant that removes layout priors (denoted as W/O Layout). The W/O Layout variant is implemented similarly to CAT3D [11] but incorporates our multi-view multi-modal alternating attention module to enable multi-modal output. Both models are trained identically for single-image 3D scene generation.

###### B.3. Video-to-3D Scene Generation

Given the fact that a well-defined 3D layout is not easy to obtain, we apply SPATIALGEN to the challenging task of generating novel 3D scenes from videos. By leveraging a state-of-the-art layout estimation model, SpatialLM [27], we get the reconstructed 3D layout from the video. Then, we perform text-to-3D scene generation conditioned on this layout and additional user-provided text prompts. This approach allows us to generate entirely new scenes that pre-

Figure 21 presents a faithful comparison, showing generated RGB outputs, scene coordinate maps, and seman-

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

Forward

[Figure 387]

[Figure 388]

Inward 360

[Figure 389]

Outward 360

[Figure 390]

Randomwalk

- Figure 18. Qualitative results on SPATIALGEN dataset under various camera trajectories. From left to right: input view and target views. First Row (forward): sampled views follow a progressive forward-moving path. Second row (inward orbit): views are directed toward the center of the room, ensuring substantial overlap between them. Third row (outward orbit): views are positioned at the center of the room, looking outward, with an angle of less than 45◦ between two adjacent views. Bottom (random walk): views are selected from a continuous random-walk camera trajectory, producing aggressive viewpoint changes.

[Figure 391]

+

A modern minimalist bedroom

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

|[Figure 396]|
|---|

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

|[Figure 401]|
|---|

FLUXControlNet

[Figure 402]

LayoutSemantic

RGBs Generations

Layout-semantic Initial rgb GT

Initializing process

Generationprocess

- Figure 19. Failure case analysis. The initial RGB image, generated from a layout and text, misorients the bed (red, vs. ground truth). This initial error causes subsequent generations to conflict with the layout.

###### W/O Layout W/ Layout Warped Images

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

###### Ground Truth Scene-Coordinate Map Input camera W/O Layout W/ Layout GT

Figure 20. Comparing geometric prediction quality between our (W/ layout) and (W/O layout). The first two columns show the predicted scene coordinate maps, where our method (W/ layout) achieves better alignment with ground-truth geometry (brown color point cloud) compared to the counterpart without layout guidance (W/O layout). Correspondingly, the warped images projected by the predicted scene coordinates demonstrate improved spatial consistency and reduced artifacts.

tic maps (top to bottom) from a given input (left-most column). As the red circles highlight, the W/O Layout variant produces artifacts in occluded regions, exhibits imperfect image-pose alignment, and generates degraded dense predictions. These failures indicate the inherent limitations of relying solely on image diffusion priors for 3D scene generation. In contrast, our full model W/ Layout leverages explicit layout guidance to achieve superior novel-view synthesis and more accurate geometry and semantic predictions.

nates for the input view, demonstrating that the W/ layout predictions achieve better alignment with the ground truth. This superior alignment provides more accurate warped images for all target viewpoints, explaining its clear superiority over the W/O layout baseline.

Ablation on Alternating Attention mechanism. We further evaluate the design choice of Multi-view Multi-modal Alternating Attention (AA). We compare our full model (denoted as W/ AA) against a variant that disables the multi-

Furthermore, Figure 20 provides an in-depth analysis of 3D consistency. We visualize the predicted scene coordi-

[Figure 409]

[Figure 410]

W/ Layout W/O Layout

[Figure 411]

[Figure 412]

[Figure 413]

## +

[Figure 414]

[Figure 415]

[Figure 416]

## +

[Figure 417]

Figure 21. Ablation on the effectiveness of layout as guidance.

Table 5. Effect of the number of input views in the inward orbit setting.

modal attention (denoted as W/O AA). Both models share the same training protocol as the aforementioned.

Qualitative results in Figure 22 reveal a critical weakness in the ablated model. As indicated by the red circles, the W/O AA model fails to produce semantically meaningful segmentations. This deficiency corrupts the geometry in the scene coordinate maps and resulting in less coherent multiview RGB generation. This demonstrates that without an explicit mechanism to align and refine information across modalities (RGB, geometry, semantics), the model cannot effectively leverage their synergies.

#input views PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓

1 17.30 0.66 0.33 35.57 3 17.83 0.67 0.31 28.72 6 18.33 0.67 0.31 21.93

Ablation on number of input views. In Table 5, we evaluated SPATIALGEN using different numbers of input views in the inward orbit camera configuration. Increasing the number of input views enhances all metrics, particularly the FID score; this implies that a greater input views enhances semantic consistency.

Our complete model W/ AA directly addresses this limitation. By facilitating cross-modal interaction, the AA mechanism enables more precise semantic labels and geometrically consistent scene coordinates. This improvement subsequently elevates the fidelity and view-consistency of the final RGB outputs, confirming that the alternating attention is pivotal for high-quality, multi-modal scene generation.

[Figure 418]

[Figure 419]

#### W/ AA W/O AA

[Figure 420]

[Figure 421]

[Figure 422]

### +

[Figure 423]

[Figure 424]

### +

[Figure 425]

Figure 22. Ablation on the effectiveness of the Multi-view Multi-modal Alternating Attention mechanism.

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

Layout Estimator

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

+

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

This modern bedroom exudes a sense of tranquility and sophistication with its minimalist design.

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Layout Estimator

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

+

This modern living room exudes a sleek and sophisticated ambiance, designed for relaxation and entertainment.

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Figure 23. Video-to-New-3D Scene Generation on the SpatialLM Test set [27]. By leveraging the state-of-the-art scene layout estimation method, SpatialLM [27], we get the reconstructed 3D layout from the video. Then, we perform text-to-3D scene generation conditioned on this layout and additional user-provided text prompts. For clearer visualization of 3D consistency and multi-modal prediction capabilities, we put depth maps here instead of displaying the coordinate maps directly.

