# arXiv:2408.13252v2[cs.CV]21Feb2025

## LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation

SHUAI YANG∗, Shanghai Jiao Tong University, China and Shanghai AI Laboratory, China JING TAN∗, The Chinese University of Hong Kong, China and Shanghai AI Laboratory, China MENGCHEN ZHANG, Zhejiang University, China and Shanghai AI Laboratory, China TONG WU†, The Chinese University of Hong Kong, China and Shanghai AI Laboratory, China YIXUAN LI, The Chinese University of Hong Kong, China and Shanghai AI Laboratory, China GORDON WETZSTEIN, Stanford University, USA ZIWEI LIU, Nanyang Technological University, Singapore DAHUA LIN†, The Chinese University of Hong Kong, China and Shanghai AI Laboratory, China

[Figure 1]

Layer 3 (Reference Panorama) Autumn park scene with people sitting on benches surrounded by colorful trees, storybook illustration style.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Layer 2 Layer 1 Layer 0

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Fig. 1. Overview of LayerPano3D. Guided by simple text prompts, LayerPano3D leverages multi-layered 3D panorama to create hyper-immersive panoramic scene with 360◦ × 180◦ coverage, enabling free 3D exploration among complex scene hierarchies.

3D immersive scene generation is a challenging yet critical task in computer vision and graphics. A desired virtual 3D scene should 1) exhibit omnidirectional view consistency, and 2) allow for large-range exploration in complex scene hierarchies. Existing methods either rely on successive scene expansion via inpainting or employ panorama representation to represent large FOV scene environments. However, the generated scene suffers from semantic drift during expansion and is unable to handle occlusion among scene hierarchies. To tackle these challenges, we introduce LayerPano3D, a novel framework for full-view, explorable panoramic 3D scene generation from a single text prompt. Our key insight is to decompose a reference 2D panorama into multiple layers at different depth levels, where each layer reveals the unseen space from the reference views via diffusion prior. LayerPano3D comprises multiple dedicated designs: 1) We introduce a new panorama dataset Upright360 , comprising 9k high-quality and upright panorama images, and finetune the advanced Flux model on Upright360 for high-quality, upright and consistent panorama generation related tasks. 2) We pioneer the Layered 3D Panorama as underlying representation to manage complex scene hierarchies and lift it into 3D Gaussians to splat detailed 360-degree omnidirectional scenes with unconstrained viewing paths. Extensive experiments demonstrate that our framework generates state-of-the-art 3D panoramic scene in both full view consistency and immersive exploratory experience. We believe that LayerPano3D holds promise for advancing 3D

panoramic scene creation with numerous applications. More examples please visit our project page: ys-imtech.github.io/projects/LayerPano3D/

1 INTRODUCTION

The development of spatial computing, including virtual and mixed reality systems, greatly enhances user engagement across various applications, and drives demand for explorable, high-quality 3D environments. We contend that a desired virtual 3D scene should 1) exhibit high-quality and consistency in appearance and geometry across the full 360◦ × 180◦ view; 2) allow for exploration among complex scene hierarchies with clear parallax. In recent years, many approaches in 3D scene generation [Gao et al. 2024; Li et al. 2024; Zhang et al. 2023b] were proposed to address these needs.

One branch of works [Chung et al. 2023; Fridman et al. 2023; Höllein et al. 2023; Ouyang et al. 2023; Yu et al. 2023] seeks to create extensive scenes by leveraging a “navigate-and-imagine” strategy, which successively applies novel-view rendering and outpaints unseen areas to expand the scene. However, this type of approaches suffer from the semantic drift issue: long sequential scene expansion easily produces incoherent results as the out-paint artifacts

accumulate through iterations, hampering the global consistency and harmony of the generated scene.

Another branch of methods [Chen et al. 2022; Tang et al. 2023b; Wang et al. 2022, 2023b; Zhang et al. 2024] employs Equirectangular Panorama to represent 360◦, large field of view (FOV) environments

- in 2D. However, the absence of large-scale panoramic datasets hinders the capability of panorama generation systems, resulting in low-resolution images with simple structures and sparse assets. Moreover, 2D panorama [Tang et al. 2023b; Wang et al. 2022; Zhang et al. 2024] does not allow for flexible scene exploration. Even when lifted to a panoramic scene [Zhou et al. 2024b], the simple spherical structure fails to provide complex scene hierarchies with clear parallax, leading to occluded spaces that cause blurry renderings, ambiguity, and gaps in the generated 3D panorama. Some methods [Zhou et al. 2024a] typically use inpainting-based disocclusion strategy to fill in the unseen spaces, but they require specific, predefined rendering paths tailored for each scene, limiting the potential for flexible exploration.

To this end, we present LayerPano3D, a novel framework that leverages Multi-Layered 3D Panorama for explorable, full-view consistent scene generation from text prompts. The main idea is to create a Layered 3D Panorama by first generating a reference panorama and treating it as a multi-layered composition, where each layer depicts scene content at a specific depth level. In this regard, it allows us to create complex scene hierarchies by placing occluded assets in different depth layers at full appearance.

Our contributions are two-fold. First, to generate high-quality and coherent 360◦ × 180◦ panoramas, we curate a new dataset, namelyUpright360,consistingof9khigh-quality,upright panorama images, and finetune the advanced Flux [Labs 2023] model with panorama LoRA on it for panorama generation and inpainting. This feed-forward pipeline prevents semantic drifts during panorama generation, while ensuring a consistent horizon level across all views.

Second, we introduce the Layered 3D Panorama representation as a general solution to handle occlusion for different types of scenes with complex scene hierarchies, and lift it to 3D Gaussians [Kerbl et al. 2023] to enable large-range 3D exploration. By leveraging pretrained panoptic segmentation prior and K-Means clustering, we streamline an automatic layer construction pipeline to decompose the reference panorama into different depth layers. The unseen space at each layer is synthesized with the Flux-based inpainting pipeline.

Extensive experiments demonstrate the effectiveness of LayerPano3D in generating hyper-immersive layered panoramic scene from a single text prompt. LayerPano3D surpasses state-of-the-art methods in creating coherent, plausible, text-aligned 2D panorama and full-view consistent, explorable 3D panoramic environments. Furthermore, our framework does not require any scene-specific navigation paths, providing more user-friendly interface for nonexperts. We believe that LayerPano3D effectively enhances the accessibility of full-view, explorable AIGC 3D environments for real-world applications.

2 RELATED WORKS

- 2.1 3D Scene Generation

Due to the recent success of diffusion models [Poole et al. 2022; Tang et al. 2023a], 3D scene generation has also achieved some development. Scenescape [Fridman et al. 2023] and DiffDreamer [Cai et al. 2023], for example, explore perpetual view generation through the incremental construction of 3D scenes. One major branch of work employ step-by-step inpainting from pre-defined trajectories. Text2Room [Höllein et al. 2023] creates room-scale 3D scenes based on text prompt, utilizing textured 3D meshes for scene representation. Similarly, LucidDreamer [Chung et al. 2023] and WonderJourney [Yu et al. 2023] can generate domain-free 3D Gaussian splatting scenes from iterative inpainting. However, this line of work often suffer from the semantic drift issue, resulting in unrealistic scene from artifact accumulation and inconsistent semantics. While some other approaches [Cohen-Bar et al. 2023; Vilesov et al. 2023; Zhang et al. 2023b] endeavor to integrate objects with environments, they yield relatively low quality of comprehensive scene generation. Recently, our concurrent works, DreamScene360 [Zhou et al. 2024b] and HoloDreamer [Zhou et al. 2024a] also employ panorama as prior to construct panoramic scenes. However, they only achieve the 360◦ × 180◦ field of view at a fixed viewpoint based on a single panorama of low-quality and simple structure, and do not support free roaming within the scene. In contrast, our framework leverages Multi-Layered 3D Panorama representation to construct highquality, fully enclosed scenes that enable larger-range exploration paths in 3D scene.

- 2.2 Panorama Generation

Panorama generation methods are often based on GANs or diffusion models. Early in this field, with the different forms of deep generative neural networks, GAN-based panorama generation methods explore many paths to improve quality and diversity. Among them, Text2Light [Chen et al. 2022] focuses on HDR panoramic images by employing a text-conditioned global sampler alongside a structureaware local sampler. However, training GANs is challenging and they encounter the issue of mode collapse. Recently, some studies have utilized diffusion models to generate panoramas. MVDiffusion [Tang et al. 2023b] generates eight perspective views with multi-branch UNet but the resulting closed-loop panorama only captures the 360◦ × 90◦ FOV. The image generated from MultiDiffusion [Bar-Tal et al. 2023] and Syncdiffusion [Lee et al. 2023] is more like a long-range image with wide horizontal angle as they do not integrate camera projection models. PanoDiff [Wang et al. 2023a] can generate 360◦ panorama from one or more unregistered Narrow Field-of-View (NFoV) images with pose estimation and controlling partial FOV LDM, while the quality and diversity of results are limited by the scarcity of panoramic image training data like most other methods [Li and Bansal 2023; Wang et al. 2023b; Wu et al. 2024]. In contrast, our model can generate Multi-Layered 3D Panorama for immersive, high-quality, and coherent scene generation from text prompts.

- 3 METHOD

The goal of our work is to create a panoramic scene guided by text prompts. This generated scene encompasses a complete 360◦ × 180◦

###### Multi-layer Panorama Construction Panoramic 3D Scene Optimization

##### Created 3D Scene

[Figure 15]

Layer 0

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Gaussian Selector

Layer 1

PanoInpaint

[Figure 20]

[Figure 21]

[Figure 22]

3D Gaussians

[Figure 23]

[Figure 24]

[Figure 25]

SRModule

[Figure 26]

[Figure 27]

[Figure 28]

Layer 2

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Layer Decompose

.

[Figure 35]

Layer N ( Reference Panorama )

[Figure 36]

[Figure 37]

[Figure 38]

Input Text

Flux+panoLoRA

a boat on lake, trees and rocks near the lake. a house and port in front of a house, anime style

Panorama Depth Estimation Layer Alignment

Fig. 2. Pipeline Overview of LayerPano3D. Our framework consists of two stages, namely multi-layer panorama construction and panoramic 3D scene optimization. LayerPano3D streamlines an automatic generation pipeline without any manual efforts to design scene-specific navigation paths for expansion or completion.

field of view from various viewpoints within an extensive range in the scene, while allowing for immersive exploration along complex trajectories. LayerPano3D consists two stages. In Stage I (Sec. 3.1), we first generate the reference panorama from text prompt by finetuning flux [Labs 2023] with panorama LoRA in our Upright360 dataset. With the reference panorama, we construct our Layered 3D Panorama representation by iterative layer decomposition, completion and alignment process. In Stage II (Sec. 3.2), the Layered 3D Panorama is lifted to 3D Gaussians in a cascaded manner to enable large-range 3D exploration.

- 3.1 Multi-Layer Panorama Generation

We introduce the Layered 3D Panorama representation based on the following assumption: “an enclosed 3D scene contains a background and various assets positioned in front of it”. In this regard, using Layered 3D Panorama for 3D scene generation is a general approach to handle occlusion for various types of scene. To create a complete scene, we first generate a high-quality reference panorama from a single text prompt and decompose it into 𝑁 + 1 layers along the depth dimension. As shown in Fig. 2, these layers, arranged from the farthest to the nearest, represent both the scene background (layer 0) and the layouts behind the observation point.

- 3.1.1 Reference Panorama Generation and Upright360 . A good panorama captures rich scene details, creating an immersive and comprehensive view. This depth of detail fosters a stronger connection and deeper comprehension of the scene. Unlike the advancement in standard image generation, panorama generation still faces quality gaps. To ensure upright and geometrically consistent scenes, we curate a new dataset, namely Upright360 , comprising high

quality, upright panorama images, and finetune the Flux [Labs 2023] with LoRA on the Upright360 dataset. This lightweight training approach achieves optimal performance even with limited high-quality panorama data and can be directly extended to subsequent tasks, for example panorama inpainting for layer completion. For data curation, we first collect around 15k raw panorama images: 9684 panorama image samples are collected from Matterport3D [Chang et al. 2017], 1824 images from the web, and 3592 synthetic panoramas generated from Blockadelabs. Based on this, we leverage GeoCalib [Veicht et al. 2024], a state-of-the-art single-perspective-image calibration method, to filter upright panoramas. Specifically, we employ the Equirectangular Projection (ERP), a mapping technique that projects a 3D sphere onto a 2D plane, to generate four perspective views from each panorama. These views are extracted at a fixed field of view (FOV) of 90°, an elevation of 0°, and four distinct azimuths (0°, 90°, 180°, 270°). We then use GeoCalib to calibrate the four views, computing their pitch and roll variances for filtering. Panoramas with variances exceeding 1.0 are classified as non-upright and excluded: 𝑉𝑎𝑟(𝑝𝑖𝑡𝑐ℎ1:4) > 1.0,𝑉𝑎𝑟(𝑟𝑜𝑙𝑙1:4) > 1.0, resulting in the creation of the final Upright360 dataset. This dataset comprises 9423 high-quality panoramas, rigorously filtered from the original collection. Building upon the Upright360 dataset, we fine-tune Flux to develop a panorama LoRA [Hu et al. 2021b] for reference panorama generation.

3.1.2 LayerDecomposition. As showninFig.2,thereferencepanorama

is decomposed by first identifying the scene assets and then cluster these assets in different layers according to depth. First, we employ an off-the-shelf panoptic segmentation model [Jain et al. 2023] pretrained on ADE20K [Zhou et al. 2017] to automatically find all scene

assets visible in the reference panorama. A good layer decomposition requires that the layer assets share a similar depth level within layers and are distant from assets in other layers. In this sense, we assign each asset a depth value and apply K-Means to cluster these masks into different groups. Given the reference panorama depth map, the depth value for each asset mask is determined by calculating the 75th percentile of the depth values within the masked region. According to the depth values, the assets are clustered into 𝑁 groups from layer 0 to 𝑁 − 1 and are merged into layer masks to guide the subsequent layer completion.

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

Layer 0

active gaussian

𝑑𝑟𝑎𝑦,2

[Figure 49]

[Figure 50]

frozen gaussian

𝑑𝑟𝑎𝑦,1

Layer N

active point cloud

(a) (b) (c)

Fig. 3. Illustration of the Gaussian Selector. Given the new asset point cloud, the Gaussian Selector identifies the active Gaussians for next layer’s optimization.

- 3.1.3 Layer Completion. With the layer mask, we focus on completing the unseen content caused by asset occlusion. In order to synthesize background pixels instead of creating new elements, we directly utilize the above trained panorama lora and integrate it into the Flux-Fill model to accomplish domain transfer, thereby employing it as a panoramic canvas inpainter. Specifically, at each layer,

our model takes the layer mask 𝑀𝑙, the reference panorama, and the “empty scene, nothing” [Zhang and Agrawala 2024] prompt as input, and output coherent content at the masked area. The inpainted panorama at layer 𝑙 is denoted 𝑃𝑙 and is used as supervision to the subsequent panoramic 3D Gaussian scene optimization. Note that, we additionally apply SAM [Kirillov et al. 2023] to extend the layer mask, based on the inpainted panorama from the previous layer, to eliminate unwanted new generations from inpainting.

Moreover, to enable large-range rendering in 3D, where observers can examine scenes from varying distances, the unprocessed texturesofdistantassetsmayappear blurred as the observer approaches. Therefore, distant layers require higher resolution to preserve texture details at different viewpoints. To address this, Super Resolution (SR) module [Yang et al. 2023] is employed to enhance the resolution of the layered panorama from layer 0 (background layer) to layer 𝑁 (reference panorama), achieving a 2× upscale in resolution. SR processing significantly improves the texture quality of distant objects, maintaining their visual clarity and texture details even when observed from a closer perspective.

- 3.1.4 LayerAlignment. GiventheLayered3DRGBPanorama [𝑃𝑙]𝑙𝑁=0, we perform the depth prediction and alignment to ensure consistency in a shared space. To begin with, we apply the 360MonoDepth [Rey-Area et al. 2022], to first estimate the layer 𝑁 (reference

3.2 Panoramic 3D Gaussian Scene Optimization

- 3.2.1 3D Scene Initialization. To enable large-range 3D exploration, we lift the Layered 3D Panorama to 3D Gaussians [Kerbl et al. 2023], where the Gaussians are initialized from the layered 3D panoramic point clouds. Considering the intrinsic spherical structure of panorama, we can easily transform an equirectangular image

𝑃 ∈ R𝐻×𝑊 ×3 into 3D point cloud 𝑆(𝜃,𝜙,𝑃𝑑𝑒𝑝𝑡ℎ). Each pixel (𝑢,𝑣) is represented as a 3D point and the angles 𝜃,𝜙 are computed as 𝜃 = (2𝑢/𝑊 − 1)𝜋, 𝜙 = (2𝑣/𝐻 − 1)𝜋/2.

Then, the corresponding 3D coordinates (𝑋,𝑌,𝑍) from the depth value 𝑃𝑑𝑒𝑝𝑡ℎ(𝜃𝑢,𝜙𝑣) are derived as follows:

𝑋 = 𝑃𝑑𝑒𝑝𝑡ℎ(𝜃𝑢,𝜙𝑣) cos𝜙𝑣 cos𝜃𝑢, 𝑌 = 𝑃𝑑𝑒𝑝𝑡ℎ(𝜃𝑢,𝜙𝑣) sin𝜙𝑣, 𝑍 = 𝑃𝑑𝑒𝑝𝑡ℎ(𝜃𝑢,𝜙𝑣) cos𝜙𝑣 sin𝜃𝑢.

(2)

Based on this transformation, we can extract the point cloud for each layer panorama to initialize 3D Gaussians.

Drastic depth changes at layout edges introduce noisy stretched outliers that would turn into artifacts during scene refinement. Therefore, we propose an outlier removal module that specifically targets stretched point removal using heuristic point cloud filtering strategies. As stretched points are usually sparsely distributed in space, we design the point filtering strategy based on its distance from the neighbors. First, we filter out all points with the minimum distance to neighbors over threshold 𝛽1. Then, we eliminate points with very few neighbors. The idea is to calculate the number of neighbors of each point within a given radius and drop the points where their number of neighbors is below threshold 𝛽2. To speed up the calculation, we map the points into 3D grids, then remove all points within grids that have less than 𝛽2 number of neighbors.

- 3.2.2 3D Scene Refinement. During scene refinement, we devise two types of Gaussian training schemes for varying scene content: the base Gaussian for reconstructing the scene background and the layer Gaussian for optimizing scene layouts. Additionally, a Gaussian selector module is introduced between layer Gaussians to facilitate scene composition.

panorama) as the reference depth 𝑃𝑑𝑒𝑝𝑡ℎ𝑁 . Then, to align the layer depth in 3D space, we find it infeasible to simply compute a global shift and scale as in [Chung et al. 2023; Höllein et al. 2023] due to the nonlinear nature of ERP. Therefore, we leverage depth inpainting model F𝑑𝑒𝑝𝑡ℎ from [Liu et al. 2024] to directly restore depth values based on the inpainted RGB pixels. F𝑑𝑒𝑝𝑡ℎ harnesses strong generalizability from large-scale diffusion prior and synthesizes inpainted depth values at an aligned scale with the base depth. We start from reference panoramic depth 𝑃𝑑𝑒𝑝𝑡ℎ𝑁 to implement step-by-step restoration from layer 𝑁 − 1 to layer 0:

In scene refinement, the base Gaussian model is initialized on a whole of the background point cloud, and the layer Gaussian model initiates on and optimizes the foreground assets. In practice, we project the layer mask M𝑙 onto point clouds and use the masked points to initiate Gaussians. The optimized Gaussians from previous layers are frozen to avoid unwanted modification. In this way, the scene background is optimized once in the base Gaussian to reduce

𝑃𝑑𝑒𝑝𝑡ℎ𝑙 = F𝑑𝑒𝑝𝑡ℎ(𝑃𝑙,𝑀𝑙 ⊙ 𝑃𝑑𝑒𝑝𝑡ℎ𝑙+1 ), (1)

where, in layer 𝑙, the inpainted panorama 𝑃𝑙 and masked depth map 𝑀𝑙 ⊙ 𝑃𝑑𝑒𝑝𝑡ℎ𝑙+1 are provided as inputs to F𝑑𝑒𝑝𝑡ℎ for restoration.

unnecessary computation and conflicts of Gaussians in subsequent layers.

We observe that the quality of the optimized scene is easily hampered by unaligned layers, and sometimes F𝑑𝑒𝑝𝑡ℎ does fail to produce perfectly aligned layer depths. Gaussians at layer 𝑙 could span into unwanted depth levels and block assets in the subsequent layer,

- as illustrated in Fig. 3(a). To handle this issue, we introduce the Gaussian selector module to detect these conflicted Gaussians, reactivate them from frozen, and optimize them away from the blockage. First, the selector computes the distance vector from the camera center o = (0, 0, 0) to each new point p, as in Fig. 3(b). The absolute distance from asset points p and scene Gaussians g to the camera is

denoted as 𝑑p and 𝑑g respectively:𝑑p = ||p − o||2, 𝑑g = ||g − o||2. By examining all Gaussians that on the same ray with asset points but at a closer distance: p /𝑑p = g /𝑑g, 𝑑g < 𝑑p, we mark them as active ( Fig. 3(c)). For efficient memory storage and fast look-up, we hash the distance vectors into a 3D grid. The mapping function from vector coordinates to grid indices writes: 𝑓 (p) = ceil(𝛽3 log(p + 1)).

4 EXPERIMENTS

- 4.1 Implementation Details

In the layered panorama construction stage, we train the panorama LoRA starting from Flux [Labs 2023] with a batch size of 1 and learning rateof10−5 for 100KiterationsonourcuratedUpright360dataset. The trainingisdone using6NVIDIA A100 GPUs for 3 days. For Layer Decomposition, we employ OneFormer [Jain et al. 2023] to obtain the panoptic segmentation map for the reference panorama. Background categories are manually determined (i.e. sky, floor, ceiling, etc.) to filter out background components in asset masks. Generally, we cluster all asset masks into 𝑁 = 3 layers via KNN and merge all masks within each layer to form a unified layer mask. With the obtained layer mask, we integrate the above trained panorama LoRA into FLux-Fill model and combine with LaMa [Suvorov et al. 2021] to achieve multi-layer completion and apply 360MonoDepth [ReyArea et al. 2022] to predict the reference panorama depth. In the 3D panoramic scene optimization stage, we lift the panorama RGBD into 3D point clouds. For scene initialization, we set 𝛽1 to 0.0001 and 𝛽2 to 4 based on empirical practice. These point clouds are used to initialize the base Gaussian model and the layer Gaussian model. During the scene refinement stage, we optimize the base Gaussian model for 3,000 iterations, then the layer Gaussians each for 2,000 iterations. The training objective for base and layer Gaussian is the L1 loss and D-SSIM term between the ground-truth views and the rendered views. We use a single 80G A100 GPU for reconstruction and the reconstruction time for each layer is 1.5 minutes on average for 1024 × 1024 resolution inputs.

- 4.2 Comparison Methods.

To evaluate the performance of our approach in text-driven 3D panoramic scene generation domain, we compare with existing methods intwophases:2D PanoramaGenerationand3D Panoramic

Scene Reconstruction. For 2D Panorama Generation, we compare the quality and creativity of 2D panorama with Text2light [Chen et al. 2022] (GAN-based HDR panorama generation), Diffusion360 [Feng

Table 1. Quantitative comparison with SoTA methods on 2D Panorama Generation. Bold indicates the best result.

Method FID ↓ Aesthetic ↑ CLIP ↑ User Study (AUR) ↑

Text2light 286.90 4.57 18.69 1.34 Panfusion 283.80 4.78 21.22 2.38 Diffusion360 274.03 5.07 21.65 2.52 Ours 223.51 5.86 22.25 3.76

et al. 2023] (diffusion-based text-to-panorama generation) and Panfusion [Zhang et al. 2024] (dual-branch diffusion-based generation). For 3D Panoramic Scene Reconstruction, we compare with Text2Room [Höllein et al. 2023] (iterative indoor scene expansion with textured mesh), LucidDreamer [Chung et al. 2023] (single-view scene generation with 3DGS), and Dreamscene360 [Zhou et al. 2024b] (text-guided panoramic 3DGS scene generation).

- 4.3 Qualitative Comparison

- 4.3.1 2D Panorama Generation. We show some qualitative comparisons with several state-of-the-art panorama generation works in Fig. 6. Text2Light [Chen et al. 2022] struggles to effectively interpret text prompt due to being trained on a realistic HDRI dataset based on the VQGAN structure, and the components in the generated panorama are relatively simple. The results by PanFusion [Zhang et al. 2024] are ambiguous and low in quality. While the instances generated by Diffusion360 [Feng et al. 2023] exhibit superior quality in comparison to the aforementioned methods, they lack intricate scene details and are prone to the generation of artifacts. In contrast, our method achieves the highest quality, presenting creative and reasonable generations.
- 4.3.2 3D Panoramic Scene Reconstruction. We present qualitative comparisonswithText2Room[Höllein et al.2023],LucidDreamer[Chung et al. 2023], and DreamScene360 [Zhou et al. 2024b] across two dimensions. First, for full 360◦ × 180◦ view consistency, we render multiple views from scene center point with single input image and text prompts. As shown in Fig. 4, LucidDreamer [Chung et al. 2023] and Text2room [Höllein et al. 2023] fail to cover the full 360◦ × 180◦ view, resulting in semantic incoherence and artifacts due to their successive inpainting-based strategy. DreamScene360 [Zhou et al. 2024b] supports a 360◦ × 180◦ view at a single fixed viewpoint, but the quality of the generated results is relatively low. In contrast, our model excels in maintaining full 360◦ × 180◦ view consistency while demonstrating superior content creativity. Second, to evaluate novel path rendering, we design a zigzag trajectory to guide the camera’s movement through the scene, with novel view renderings sampled along the trajectory for comparison. Fig. 5 shows 6 random samples from this fixed flythrough trajectory. Compared with all three methods, our model achieves a more complete 3D scene with consistent textures and a reasonable geometric structure.

- 4.4 Quantitative Comparison

4.4.1 2D Panorama Generation. We adopt three metrics for quantitative comparisons: 1) FID [Heusel et al. 2017] evaluates both fidelity and diversity; 2) Aesthetic [Schuhmann et al. 2022] evaluates the aesthetics of panorama; 3) CLIP [Hessel et al. 2021] measures the compatibility of results with input prompts. Moreover, a user study is also conducted to further evaluate the quality of panoramas,

Rendered Panorama View 1 View 2 View 3 View 4 Rendered Panorama View 1 View 2 View 3 View 4

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

Dreamscene360

Ours

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

LucidDreamer

Text2room

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Fig. 4. Qualitative comparisons in full 360°×180° Scene. We compare the panorama and multiple views of the scene generated by four methods. LayerPano3D exhibits consistent and rich details across full 360◦ × 180◦ coverage, while other methods show obvious inconsistencies and disorganized patterns in regions that deviate from the input view.

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

OursLucidDreamerText2RoomDreamScene360

Input View &Render Trajectory

View 1 View 2 View 3 View 4 View 5 View 6

- Fig. 5. Qualitative comparisons in Large-range Scene Exploration. We show the novel view renderings along a zigzag trajectory to compare the capability of large-range scene exploration. Our method is able to maintain high-quality content rendering and does not show distortion or gaps in unseen space, which shows the ability of LayerPano3D to create hyper-immersive panoramic scenes.

- Table 2. Qualitative comparison with SoTA methods on 3D Panoramic Scene. LayerPano3D achieves high-quality reconstruction and novel view synthesis while maintaining upright panoramic scene compared to other methods.

Appearance Geometry User Study (AUR) NIQE ↓ BRISQUE ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Pitch-Mean ↓ Pitch-Var ↓ 360◦ × 180◦ ↑ Free-path ↑

Method

Text2room [Höllein et al. 2023] 5.231 46.127 30.126 0.882 0.038 2.029 1.724 1.69 2.31 LucidDreamer [Chung et al. 2023] 5.822 52.102 36.108 0.954 0.026 2.813 2.189 1.81 1.31 DreamScene360 [Zhou et al. 2024b] 5.051 39.891 30.056 0.958 0.062 1.328 2.018 2.86 2.77 Ours 4.023 38.287 42.057 0.986 0.015 0.732 0.032 3.64 3.61

where we project 4 views at a fixed FOV (90◦) to the user for sorting. Here, we report the average Average User Ranking (AUR) [Zhang et al. 2023a], which is computed based on an integrated assessment of coherence, plausibility, aesthetics, and compatibility dimensions. As shown in Tab. 1, our method achieves the best scores among

all quantitative metrics and human evaluation, demonstrating its fidelity, alignment with text and overall consistency.

4.4.2 3D Panoramic Scene Reconstruction. Following [Zhou et al. 2024b], we adopt non-reference image quality assessment metrics, NIQE [Mittal et al. 2012b] and BRISQUE [Mittal et al. 2012a], to

###### Ours PanFusion Text2Light Diffusion360

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Bustling city street at sunset, skyscrapers, streets, cars, realistic

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

A Japanese village by a serene river with cherry blossoms and Mount Fuji

- Fig. 6. Qualitative comparisons in Panorama Generation. LayerPano3Ddemonstrates superior capability in generating high-quality outputs with precise alignment to text prompt, outperforming other methods in fidelity and input adherence. evaluate novel view quality along scene navigation paths. We also follow [Zhou et al. 2024b] to measure the rendering quality with PSNR, SSIM and LPIPS [Zhang et al. 2018]. In terms of geometry evaluation, we render 4 orthogonal views (90◦ FOV, 0◦ elevation and { 0◦, 90◦, 180◦, 270◦ } azimuths) and predict the Pitch-Mean and PitchVar (mean and variance of the elevation angles) with [Veicht et al. 2024] to evaluate whether the scenes are upright. As shown in Tab. 2, our method surpasses the existing methods in both novel view quality metrics (NIQE and BRISQUE), 3D reconstruction metrics (PSNR, SSIM, LPIPS) and while ensuring the upright panoramic scene. Furthermore, we conduct another user study for 3D panoramic scene evaluation from two aspects: 1) 360◦ × 180◦ view consistency and

2) novel path rendering quality. For the first aspect, we render 60 frames to cover 360-degree view at the 0-degree and 45-degree elevation respectively for evaluation. For the second aspect, we use the same trajectory as in Fig. 5 to render navigation videos for evaluation. We invite 52 users including graduate students that expertise

- in 3D and average users to rank the 40 results from 4 methods. The average ranking is shown in Tab. 2. Our LayerPano3D achieves the best performance in both 360◦ × 180◦ view consistency and novel path rendering quality among all four approaches.

4.5 Analysis and Ablative Study

In this section and supp., we show the analysis and ablation on the Gaussian Selector (Sec. 4.5.1), Multi-layer design (single vs. multi; Sec. 4.5.2), Layer Gaussians representation (supp.), layer inpainting (supp.) and 3DGS optimization efficiency (supp.).

- 4.5.1 Ablation on Gaussian Selector. Our Gaussian selector is proposed to select the part of Gaussians that appears in the front of newly added scene assets. By selecting these Gaussians and reactivating them in the optimization, the model achieves accurate appearance and geometry at the current layer. As shown in Fig. 7, the leftmost column is the scene Gaussians at layer 0. When adding the building assets at the first layer, the sky Gaussians from the previous layer partially block the building assets (right column). After using the Gaussian selector to select and optimize the sky Gaussians, these Gaussians learn to either be translucent and pruned for low opacity or move to be a part of the building assets. Therefore in the middle column, we observe a consistent scene with no obvious blockage of the new building assets thanks to the Gaussian Selector.

- 4.5.2 Analysis on Panorama Renderings at Off-center Viewpoints. In Fig. 8, we demonstrate that LayerPano3D is robust to render consistent panorama images at various locations besides the original camera location in the center. We sample four camera locations

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

View0 View1

View0 View1

View0 View1

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

View2 View3

View2 View3

View2 View3

Layer 0 Layer 1, w/ Gaussian Selector

Layer 1, w/o Gaussian Selector

- Fig. 7. Ablation on the Gaussian Selector. With the Gaussian Selector, the merged Gaussians are optimized to faithfully reconstruct the groundtruth panorama views.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Reference Panorama

- A

- B

- C

- D

Ours (multi-layered panorama) Single panorama

[Figure 141]

A

C B

D

- Fig. 8. Analysis on Panorama Rendering at Off-center Viewpoints. Compared with the single-layer variant, LayerPano3D render 360◦ × 180◦ consistent panorama at various off-center viewpoints without any holes or gaps from occlusion.

on circular trajectories on the hemisphere centered at the origin and render 24 views at (−45◦, 0◦, 45◦) elevation to compose new panorama images. By evaluating panorama renderings at new viewpoints, we show that our generated panoramic scene is 360◦ × 180◦ consistent and enclosed, robust to various viewpoints at any angle. Compared to the single-layered 3D panorama, our multi-layered 3D panorama exhibits no gaps or holes from the scene occlusion, demonstrating our capability for larger-range, complex 3D exploration in the generated scenes.

5 CONCLUSION

In this paper, we propose LayerPano3D, a novel framework that generates hyper-immersive panoramic scene from a single text prompt. Our key contributions are two-fold. First, we propose the text-guided anchor view synthesis pipeline to generate detailed and consistent reference panorama. Second, we pioneer the Layered 3D Panorama representation to show complex scene hierarchies at multiple depth layers, and lift it to Gaussians to enable largerange 3D exploration. Extensive experiments show the effectiveness of LayerPano3D in generating 360◦ × 180◦ consistent panorama at various viewpoints and enabling immersive roaming in 3D space. We believe that LayerPano3D holds promise to advance high-quality, explorable 3D scene creation in both academia and industry.

Limitations and Future Works. LayerPano3D leverages good pre-trained prior to construct panoramic 3D scene, i.e., panoramic depth prior for 3D lifting. Therefore, the created scene might contain artifacts from inaccurate depth estimation. With advancements in more robust panorama depth estimation, we hope to create highquality panoramic 3D scenes with finer asset geometry.

REFERENCES

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. 2023. MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation. arXiv:2302.08113 [cs.CV]

Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein. 2023. DiffDreamer: Towards Consistent Unsupervised Single-view Scene Extrapolation with Conditional Diffusion Models. In ICCV. IEEE, 2139–2150.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. 2017. Matterport3D: Learning from RGB-D Data in Indoor Environments. International Conference on 3D Vision (3DV) (2017).

Zhaoxi Chen, Guangcong Wang, and Ziwei Liu. 2022. Text2light: Zero-shot text-driven hdr panorama generation. ACM Transactions on Graphics (TOG) 41, 6 (2022), 1–16.

Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. 2023. LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes. CoRR abs/2311.13384 (2023).

Dana Cohen-Bar, Elad Richardson, Gal Metzer, Raja Giryes, and Daniel Cohen-Or. 2023. Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes. arXiv:2303.13450 [cs.CV]

Mengyang Feng, Jinlin Liu, Miaomiao Cui, and Xuansong Xie. 2023. Diffusion360: Seamless 360 Degree Panoramic Image Generation based on Diffusion Models. arXiv:2311.13141 [cs.CV]

Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. 2023. SceneScape: TextDriven Consistent Scene Generation. arXiv:2302.01133 [cs.CV]

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo MartinBrualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. 2024. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314 (2024).

Leon A Gatys, Alexander S Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2414–2423.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In EMNLP (1). Association for Computational Linguistics, 7514–7528.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In NIPS. 6626–6637.

Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. 2023. Text2room: Extracting textured 3d meshes from 2d text-to-image models. arXiv preprint arXiv:2303.11989 (2023).

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021b. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 [cs.CL] https://arxiv.org/abs/2106.09685

Ronghang Hu, Nikhila Ravi, Alexander C. Berg, and Deepak Pathak. 2021a. Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image. In ICCV. IEEE, 12508–12517.

Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. 2023. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2989–2998.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139:1–139:14.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.

2023. Segment anything. arXiv preprint arXiv:2304.02643 (2023). Black Forest Labs. 2023. FLUX. https://github.com/black-forest-labs/flux. Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. 2023. SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions. arXiv:2306.05178 [cs.CV] Haoran Li, Haolin Shi, Wenli Zhang, Wenjun Wu, Yong Liao, Lin Wang, Lik hang Lee, and Pengyuan Zhou. 2024. DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling. arXiv:2404.03575 [cs.CV]

Jialu Li and Mohit Bansal. 2023. PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation. arXiv:2305.19195 [cs.CV]

Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. 2022. Infinitenaturezero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision. Springer, 515–534.

Zhiheng Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jie Xiao, Kai Zhu, Nan Xue, Yu Liu, Yujun Shen, and Yang Cao. 2024. InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior. arXiv preprint arXiv:2404.11613 (2024).

Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 2012a. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing 21, 12 (2012), 4695–4708.

Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. 2012b. Making a “completely blind” image quality analyzer. IEEE Signal processing letters 20, 3 (2012), 209–212.

Hao Ouyang, Kathryn Heal, Stephen Lombardi, and Tiancheng Sun. 2023. Text2Immersion: Generative Immersive Scene with 3D Gaussians. arXiv preprint arXiv:2312.09242 (2023).

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).

Manuel Rey-Area, Mingze Yuan, and Christian Richardt. 2022. 360monodepth: Highresolution 360deg monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3762–3772.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. 2022. LAION-5B: An open large-scale dataset for training next generation image-text models. In NeurIPS.

Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 2020. 3d photography using context-aware layered depth inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8028–8038.

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. 2021. Resolution-robust Large Mask Inpainting with Fourier Convolutions. arXiv:2109.07161 [cs.CV]

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023a. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023).

Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. 2023b. MVDiffusion: Enabling Holistic Multi-view Image Generation with CorrespondenceAware Diffusion. arXiv:2307.01097 [cs.CV]

Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys.

2024. GeoCalib: Learning Single-image Calibration with Geometric Optimization. arXiv:2409.06704 [cs.CV] https://arxiv.org/abs/2409.06704

Alexander Vilesov, Pradyumna Chari, and Achuta Kadambi. 2023. CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting. arXiv:2311.17907 [cs.CV]

Guangcong Wang, Yinuo Yang, Chen Change Loy, and Ziwei Liu. 2022. StyleLight: HDR Panorama Generation for Lighting Estimation and Editing. arXiv:2207.14811 [cs.CV]

Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue. 2023b. Customizing 360-Degree Panoramas through Text-to-Image Diffusion Models. arXiv:2310.18840 [cs.CV]

Jionghao Wang, Ziyu Chen, Jun Ling, Rong Xie, and Li Song. 2023a. 360-Degree Panorama Generation from Few Unregistered NFoV Images. In Proceedings of the 31st ACM International Conference on Multimedia. ACM. https://doi.org/10.1145/ 3581783.3612508

Tianhao Wu, Chuanxia Zheng, and Tat-Jen Cham. 2024. PanoDiffusion: 360-degree Panorama Outpainting via Diffusion. arXiv:2307.03177 [cs.CV]

Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. 2023. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. arXiv preprint arXiv:2308.14469 (2023).

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T. Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, and Charles Herrmann. 2023. WonderJourney: Going from Anywhere to Everywhere. CoRR abs/2312.03884 (2023).

Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. 2024. Taming Stable Diffusion for Text to 360 Panorama Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6347–6357.

Lvmin Zhang and Maneesh Agrawala. 2024. Transparent Image Layer Diffusion using Latent Transparency. arXiv:2402.17113 [cs.CV] https://arxiv.org/abs/2402.17113

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023a. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Qihang Zhang, Chaoyang Wang, Aliaksandr Siarohin, Peiye Zhuang, Yinghao Xu, Ceyuan Yang, Dahua Lin, Bolei Zhou, Sergey Tulyakov, and Hsin-Ying Lee. 2023b. SceneWiz3D: Towards Text-guided 3D Scene Composition. arXiv:2312.08885 [cs.CV]

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba.

2017. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition. 633–641.

Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, and Li Yuan. 2024a. HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions. arXiv preprint arXiv:2407.15187 (2024).

Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi. 2024b. DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting. arXiv preprint arXiv:2404.06903 (2024).

Created Scene (Extracted) Layer 2 (Reference Panorama) Layer 1 Layer 0

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Magician's cabin in a tranquil forest, beside a flowing river.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Cherry blossoms in full bloom, petals falling by a tranquil river, vibrant colors.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Sandy beach, large driftwood in the foreground, calm sea beyond, realism style

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Ancient stone monoliths standing in a grassy field under a cloudy sky.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

### Magical academy courtyard with floating orbs of light, ancient stone buildings, and a large tree in the center, mystical and enchanting.

- Fig. 9. Additional results of LayerPano3D on Diverse Generation. LayerPano3D generates various hyper-immersive scene with consistent and rich details across full 360◦ × 180◦ coverage.

Input View & 3DP Worldsheet InfiniteNature-Zero Render Trajectory

LucidDreamer

#### 3D GS Ours

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

- View 1
- View 2
- View 3

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

- 2
- 3

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

- 1

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

- 2

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

- View 1
- View 2
- View 3

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

3

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

1

- Fig. 10. Analysis on Layer Gaussians Representation with 3DP [Shih et al. 2020], Worldsheet [Hu et al. 2021a], LucidDreamer [Chung et al. 2023], and Single-view 3D GS [Kerbl et al. 2023] on novel view renderings along a zigzag trajectory. InfiniteNature-Zero [Li et al. 2022] are shown with three random views from its fixed trajectory.

[Figure 215]

[Figure 216]

Input Masked Panorama Ours LaMa-Inpainting Stable-Diffusion-Inpainting

[Figure 217]

[Figure 218]

- Fig. 11. Analysis on the Layer Completion Inpainting. We present the panorama inpainting results for three methods guided by the same text prompt: “empty scene, nothing” [Zhang and Agrawala 2024]. Our model effectively handles complex scenarios, delivering clear results with consistent and coherent structures.

by resizing the panorama to 512 × 1024, then crop it into 4 windows with a stride of 256. The final image is formed by seamlessly connecting the panorama’s tail and head. Each window is 512 × 512, and we compute the average Style Loss across the 6 combinations of these cropped views.

- 6 ADDITIONAL EXPERIMENT DETAILS

- 6.1 More Evaluation Details on 2D Panorama Generation.

We use various metrics to evaluate the coherence, fidelity, diversity, aesthetic and compatibility of generated panoramas with the input prompt.

Table 3 presents a quantitative comparison among the methods. Bold indicates the best result, and underline indicates the secondbest result. Our method achieve the optimal scores in FID, Aesthetic and CLIP metrics, which indicates the high quality in creativity, fidelity, and compatibility with input prompts of our method. The results of Intra-Style demonstrate that our method achieves global coherence across the image, maintaining a consistent overall style. Although Text2light [Chen et al.2022] has a smaller Intra-Style score, this is due to its tendency to generate monotonous panoramas with extensive uniform color block backgrounds. Moreover, the generated contents are largely unrelated to the guidance provided by the input prompts. Consequently, the metric of Intra-Style for Text2light has no comparison significance.

- • (Fidelity & Diversity) FID [Heusel et al. 2017]: Fréchet Inception Distance(FID) is employed to assess both fidelity and diversity. We calculate FID between panoramas from Matterport3D [Chang et al. 2017] and the generated panoramas.
- •(Aesthetic)Aesthetic[Schuhmannetal.2022]:Foreach panorama, we randomly project 20 views at a fixed FOV (90◦) with resolution 512 × 512 and calculate their average aesthetic scores.
- • (Compatibility) CLIP [Hessel et al. 2021]: Compatibility with the input prompt is evaluated using the mean of CLIP scores, mirroring the approach for aesthetic evaluation.
- • (Coherence) Intra-Style [Gatys et al. 2016]: To assess coherence, we introduce Intra-Style, computed as the average Style Loss between pairs of window images from the same panorama. We begin

- Table 3. Quantitative comparison with SoTA methods. Bold indicates the best result, and underline indicates the second-best result.

Method FID ↓ Aesthetic ↑ CLIP ↑ Intra-Style ↓

Text2light 286.90 4.57 18.69 0.31 Panfusion 283.80 4.78 21.22 18.66 Diffusion360 274.03 5.07 21.65 3.70 Ours 223.51 5.86 22.25 1.63

- Table 4. Memory usage at each layer in Layer Gaussians Optimization. Our optimization strategy ensures that memory consumption remains at a low level.

Layer ID Layer 0 Layer 1 Layer 2 Layer 3 Memory (MB) 1997.04 2079.75 2280.99 2507.61 Newly Added GS 1702242 129215 157183 102450

- 7 ADDITIONAL ANALYSIS AND ABLATIVE STUDY

- 7.1 Analysis on 3DGS optimization efficiency.

For time efficiency, as we mentioned in Sec. 4.1, we use a single 80G A100 GPU for 3DGS optimization and the optimization time for each layer is 1.5 minutes on average for 1024 × 1024 resolution inputs. For memory efficiency, if we use a pixel-aligned 3DGS for optimization, we would easily encounter OOM as the layers increase. Here we have two steps to reduce the memory cost. First, as we described in methods section, we select new assets at each layer and only optimize the new assets combined with active Gaussians

- at each layer. In this way, our model does not introduce additional Gaussians to represent the same asset. Second, although we use layer mask to select point clouds in a pixel-aligned manner, but

we downsample the point cloud to be under 𝑁𝑚𝑎𝑥 points at each layer before 3DGS initialization to not exceed the GPU memory, and remain a small size for visualization and rendering. Empirically, we set 𝑁𝑚𝑎𝑥 to 2,000,000. We also show a breakdown of GPU memory usage at each layer for a random case in Table 4. The maximum memory usually does not exceed 3000 MB for all cases.

- 7.2 Analysis on Layer Gaussians Representation.

In the main paper, we validate the effectiveness of the Layer Gaussians representation in addressing occlusion for hyper-immersive panoramic scene generation, through experiments on full 360◦×180◦ view consistency and large-scale exploratory trajectory rendering capability. Building on this, we extend our discussion to explore the application of this representation in single-image to scene task. We show the qualitative comparisons with 3DP [Shih et al. 2020], Worldsheet [Hu et al. 2021a], InfiniteNature-Zero [Li et al. 2022], LucidDreamer [Chung et al. 2023] and 3D GS [Kerbl et al. 2023] in Fig. 10. The camera moves along a zigzag trajectory into the scene, and the novel view renderings are sampled along the trajectory for comparison among all methods. For InfiniteNature-Zero [Li et al. 2022], we showed 3 random samples from its fixed fly-through trajectory. Compared to all five methods, our model achieves more complete 3D scene with consistent texture and accurate geometry in both occluded and non-occluded space, demonstrating our ability of high-quality image-conditioned 3D scene creation.

7.3 Analysis on Layer Completion Inpainting.

We discuss the effectiveness of our panorama inpainter in layer completion. We compare the inpainting results among three approaches: LaMa [Suvorov et al. 2021], stable diffusion inpainting model, and our proposed inpainter. As illustrated in Fig. 11, LaMa produces inconsistent texture and blurry artifacts at large-scale inpainting. Pure stable diffusion tends to produce distorted new elements due to the domain gap between perspective and panoramic images. In contrast, thanks to the panoramic lora and the introduced controllable generation strength, our module delivers clean inpainting results with coherent and plausible structures in the masked regions.

