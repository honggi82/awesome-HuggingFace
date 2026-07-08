# arXiv:2504.17670v2[cs.CV]26May2025

## DiMeR: Disentangled Mesh Reconstruction Model

LUTAO JIANG∗, HKUST(GZ), China JIANTAO LIN∗, HKUST(GZ), China KANGHAO CHEN∗, HKUST(GZ), China WENHANG GE∗, HKUST(GZ), China XIN YANG, HKUST(GZ), China and HKUST, China YIFAN JIANG, HKUST(GZ), China YUANHUIYI LYU, HKUST(GZ), China XU ZHENG, HKUST(GZ), China YINCHUAN LI, Noah’s Ark Lab, China YING-CONG CHEN†, HKUST(GZ), China and HKUST, China

[Figure 1]

Fig. 1. DiMeR takes the image or text inputs and generates detailed 3D meshes.

We propose DiMeR, a novel geometry-texture disentangled feed-forward model with 3D supervision for sparse-view mesh reconstruction. Existing methods confront two persistent obstacles: (i) textures can conceal geometric errors, i.e. , visually plausible images can be rendered even with wrong geometry, producing multiple ambiguous optimization objectives in geometry-texture mixed solution space for similar objects; and (ii) prevailing mesh extraction methods are redundant, unstable, and lack 3D supervision. To solve these challenges, we rethink the inductive bias for mesh reconstruction. First, we disentangle the unified geometry-texture solution space, where a single input admits multiple feasible solutions, into geometry and texture spaces individually. Specifically, given that normal maps are strictly consistent with geometry and accurately capture surface variations, the normal maps serve as the sole input for geometry prediction in DiMeR, while the

∗ Equal contribution. † Corresponding author.

texture is estimated from RGB images. Second, we streamline the algorithm of mesh extraction by eliminating modules with low performance/cost ratios and redesigning regularization losses with 3D supervision. Notably, DiMeR still accepts raw RGB images as input by leveraging foundation models for normal prediction. Extensive experiments demonstrate that DiMeR generalises across sparse-view-, single-image-, and text-to-3D tasks, consistently outperforming baselines. On the GSO and OmniObject3D datasets, DiMeR significantly reduces Chamfer Distance by more than 30%. Project Page: https://lutao2021.github.io/DiMeR_page/

1 INTRODUCTION

The tasks of 3D reconstruction and generation have garnered significant attention, largely due to the advancements made by NeRF [Mildenhall et al. 2021] and 3DGS [Kerbl et al. 2023]. However, transforming them into the mesh poses a challenge. In this paper, we

|the 3D model, offering a more reconstruction. Building on this Occam’s Razor [Blumer et al. texture unified solution space,<br><br>one input, into two individual exclusively utilize normal maps RGB images for obtaining the mal, depth, and silhouette-mask<br><br>guided by appearance-based geometry should be capable of under arbitrary environmental Therefore, we add the statistical placing the predicted untextured ments with randomly assigned challenge, limitations remaining inal regularization losses with incorporate 3D supervision via<br><br>simplify the modules with<br><br>improved mesh extraction resolution, compared with|
|---|

Fig. 2 (a) and (c), the normal maps consistently align with the surface of

Texture Geometry Normal Texture Geometry Normal

[Figure 2]

[Figure 3]

reliable input format for geometry re inductive bias and the Principle of O 1987], we disentangle the geometryte where multiple solutions correspond to al spaces. To get the geometry, we ex s as the sole input, while utilizing RG texture. Each of them is trained with task-specific supervision: the geometry branch is constrained by norm losses, whereas the texture branch is objectives. Moreover, an accurate ge rendering the light map correctly un conditions and various materials. Th l expectation supervision signal by pl d mesh model in multiple environm materials. To address the second ch in FlexiCubes, we replace the origin eikonal loss [Gropp et al. 2020] and in ground-truth mesh. Furthermore, we low performance/cost ratios. Our im algorithm allows us a higher extraction re other reconstruction models.

(a) Texture Hides Geometry

Input

Prediction GT

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Generator

OK, right

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

OK, but fail

(b) Training Objective Conflict Geo-Tex Solution Space

Disentangled Solution Space (Ours)

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

+

[Figure 22]

[Figure 23]

…

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

+

[Figure 29]

[Figure 30]

Geo Tex One-to-Many One-to-One One-to-One

Geo Tex

(c) Solution Space Disentangle

Leveraging recent foundation models for normal map prediction [He et al. 2024a; Ye et al. 2024], we can generate highly accurate normal maps from RGB images with minimal error and latency (only 200ms). To validate this point and choose the optimal model for DiMeR, we conduct a benchmark evaluation for object-level normal prediction. To further improve the robustness of DiMeR in practical applications, we introduce noise to the input, normal maps, during training. Equipped with these models, DiMeR also accepts RGB images as raw input, the same as other methods.

- Fig. 2. (a) exhibits difficulty in distinguishing geometry from RGB images. (b) shows the conflict input-GT pairs in datasets due to problem (a), hindering the training. (c) illustrates our idea: disentangle the mixed solution space containing multiple feasible solutions into two separate spaces with unbiased input. Samples are from the Objavers dataset [Deitke et al. 2023].

focus on mesh representation, which is easy to adapt to downstream applications, such as the gaming industry, VR, robotics, etc .

Enhanced by the introduction of the extensive 3D dataset, Objaverse [Deitke et al. 2024, 2023], numerous 3D reconstruction and generative models emerge. One notable advancement is LRM [Hong et al. 2023], which pioneers the feed-forward generation of a NeRF model from RGB images. Subsequent works [Ge et al. 2024; Liu et al. 2024; Wang et al. 2025; Wei et al. 2024; Xu et al. 2024a; Yang et al. 2024] extend LRM’s NeRF representation to mesh. However, two key issues persist in these methods. First, the reliance on RGB images as input leads to significant ambiguity in training. As shown in Fig. 2(a), the texture often hides the underlying geometry, thereby leading to the conflict of training objectives exhibited in Fig. 2(b). Furthermore, as demonstrated in Fig. 2(c), RGB images can be rendered from compositions of multiple wrong geometries and textures, driving the network toward an undesirable averaged solution. Second, most of the existing mesh reconstruction methods employ FlexiCubes [Shen et al. 2023] to extract the mesh and utilize differential rasterization for optimization. However, the Signed Distance Field (SDF) grid defined in FlexiCubes only promises the meaning of positive and negative signs for surface extraction, which makes it difficult to apply 3D supervision. Moreover, some of its components are redundant for this task, and its regularization losses lead to serious instability in training.

Our DiMeR model is capable of effectively handling various tasks, including sparse-view reconstruction, single-image-to-3D, and textto-3D. Extensive experiments demonstrate that DiMeR significantly outperforms previous methods. Specifically, on the GSO dataset, DiMeR reduces Chamfer Distance by 22%, with an upper bound improvement of 32% when using real normal map inputs.

In general, our contributions can be summarized as follows:

- • Rethinking the inductive bias for mesh reconstruction, we propose DiMeR, a disentangled framework to train and predict geometry from normal maps and texture from RGB images separately, with decoupled supervision signals.
- • We enhance the mesh extraction algorithm for this task and introduce the 3D ground truth supervision.
- • We conduct a benchmark for the foundation models of normal map prediction in object-level tasks.
- • Numerous experiments demonstrate the superiority and robustness of our DiMeR on reconstruction, single-imageto-3D, and text-to-3D tasks.

2 RELATED WORKS 2.1 3D Generative Models

To solve these two challenges, we propose DiMeR, a geometrytexture disentangled feed-forward sparse-view mesh reconstruction model with 3D supervision. To address the first challenge, training ambiguity, we exploit the inductive bias derived from the consistency between normal maps and 3D geometry. As shown in

Building upon advancements in 2D diffusion models, DreamFusion [Poole et al. 2022] introduced score distillation sampling (SDS) to train 3D representation models like NeRF [Mildenhall et al. 2021]

and 3DGS [Kerbl et al. 2023] based on text input. Subsequently, numerous methods have been developed to enhance this approach [Bai

- et al. 2023; Chen et al. 2023; Cheng et al. [n.d.]; Jiang et al. 2024; Li et al. 2025; Liang et al. 2023; Lin et al. 2023; Lukoianov et al. 2024; Metzer et al. 2023; Raj et al. 2023; Shi et al. 2023b; Tang et al. 2023; Wang et al. 2023a,b; Yi et al. 2023; Zhou et al. [n.d.]]. However, a significant limitation of these methods is the need to train a separate 3D model for each text input, which can take tens of minutes or even hours per text. Some approaches attempt to address this by employing SDS to train a feed-forward network [Jiang and Wang 2024; Li et al. 2023b; Lorraine et al. 2023; Qian et al. 2024], but these are limited to a few specific text subjects, reducing the diversity of the outputs. Recently, the introduction of large-scale 3D datasets, such as Objaverse [Deitke et al. 2024, 2023], has enabled models like LRMs [Hong et al. 2023; Tochilkin et al. 2024] to explore feedforward reconstruction from a single image. Following this, several methods have been developed to create sparse-view reconstruction models [Li et al. 2023a; Tang et al. 2025; Xu et al. 2024b; Zhang et al. 2024a] based on NeRF or 3DGS. To support real-world applications, leveraging differential marching cube algorithms [Shen et al. 2023; Wei et al. 2023], some methods focus on direct mesh generation [Ge
- et al. 2024; Liu et al. 2024; Wang et al. 2025; Wei et al. 2024; Xu et al. 2024a]. Additionally, several 3D diffusion models [Gupta et al. 2023; He et al. 2024b; Li et al. 2024a; Lin et al. 2025a; Ren et al. 2024a,b; Szymanowicz et al. 2024; Xiang et al. 2024; Zhang et al. 2023; Zhang and Wonka 2024; Zhang et al. 2024b] emerge, but they are limited to the generation task and lack strict correspondence with the input image. Moreover, their inference times range from tens of seconds to several minutes. Inspired by auto-regressive (AR) models [Tian et al. 2024; Xie et al. 2024; Zhou et al. 2024], some researchers have shifted focus to mesh AR generation [Chen et al. 2024a,b,c; Siddiqui et al. 2024; Tang et al. 2024; Wang et al. 2024; Weng et al. 2024]. However, these methods typically require the number of mesh faces to be fewer than 6,000, and they exhibit low robustness. Concurrently, similar to us, Hi3DGen [Ye et al. 2025] also found that exclusive utilization of normal maps can enhance the quality of geometry and implemented a diffusion model based on this.

In this paper, we focus on feed-forward sparse-view mesh reconstruction. Differently, we disentangle the framework into dual branches that predict geometry solely from normal and predict texture from RGB. To ensure that each branch performs its intended role, we assign branch-specific, unambiguous supervision signals.

- 2.2 Multi-view Diffusion Model

Multi-view diffusion models are designed to generate multi-view images or normal maps from a single image or text prompts, instead of directly producing corresponding 3D models. This approach is gaining popularity due to the relative simplicity of its task definition, where multi-view images are synthesized first, followed by the use of sparse-view reconstruction models to complete the 3D model generation process. Zero123 [Liu et al. 2023] introduces explicit control by embedding camera parameters into the conditions of 2D diffusion models. Following, many methods have achieved significant success to synthesis multi-view images and normal maps [Li et al. 2023a, 2024b; Lin et al. 2025b; Long et al. 2024; Lu et al. 2024;

Melas-Kyriazi et al. 2024; Shi et al. 2023a,b; Voleti et al. 2024; Wang and Shi 2023; Wu et al. 2024]. We employ the image-input 2.5D model, such as zero123++ [Shi et al. 2023a] and Era3D [Li et al. 2024b], to perform the single-image-to-3D task, while we use the text-input 2.5D diffusion model, such as Kiss3DGen, to accomplish the text-to-3D task. With the continuing progress of such models, DiMeR has the potential to further enhance generation quality.

- 2.3 Normal Prediction Foundation Models

Surface normals precisely describe local surface variation and orientation, making them crucial for 3D reconstruction. The robustness and accuracy of recent normal prediction foundation models have reached practical levels. Marigold [Ke et al. 2024] first integrates diffusion models into depth and normal estimation, by preserving the prior that the visual generative models learned. Subsequent works [Bae and Davison 2024; Fu et al. 2024; He et al. 2024a; Ye et al. 2024] have further boosted performance while markedly reducing inference latency. Collectively, these advances provide the possibility for high-quality 3D reconstruction exclusively from predicted normal maps, enabling DiMeR with the RGB image as input.

3 METHOD

As shown in Fig. 3, the objective of our DiMeR is to reconstruct the 3D mesh geometry from normal maps and derive texture from RGB images. We introduce Geometry Branch in Sec. 3.1, Texture Branch in Sec. 3.2, and applications in Sec. 3.3.

- 3.1 Geometry Branch

As illustrated in Fig. 2, a single RGB image admits many equally plausible solutions in the geometry–texture joint solution space, encouraging the network to learn over-smoothed averages. Normal maps, in contrast, are uniquely determined by the underlying surface and faithfully encode fine geometric variation. Guided by Occam’s razor [Blumer et al. 1987], we therefore feed only normal maps to the geometry branch, eliminating appearance-induced ambiguities and simplifying the correspondence between input and output. This design establishes a clearer relationship between the network’s inputs and outputs, ultimately reducing the training complexity. Supervision is likewise restricted to geometry-specific losses, discarding ambiguous RGB rendering terms. We further regularize geometry by rendering the untextured mesh with physically based rendering (PBR) under diverse illuminations and materials, matching the resulting lighting maps to statistical expectations. Finally, we improve the mesh extraction algorithm for greater efficiency and robustness and incorporate direct 3-D supervision.

Network Structure. As shown in Fig. 3, the geometry branch of our DiMeR model initiates with normal maps N ∈ R𝐾×𝐻×𝑊 ×3 of 𝐾 randomly selected views, alongside their associated camera embeddings 𝜻 ∈ R𝐾×16. We opt for a random sampling of input views to improve the model’s capability to interpret camera embeddings from arbitrary directions and add slight noise to them, thereby enhancing robustness and reducing dependency on specific input configurations. Furthermore, this also reduces the requirements for the user input, allowing users to provide inputs from unfixed view directions. The normal maps N and their associated camera embeddings 𝜻 are

Geometry Branch (Sec. 3.1)

[Figure 31]

[Figure 32]

[Figure 33]

GT SDF Eikonal

[Figure 34]

SDFDecoder

Normal Mask Depth

PBRRender

[Figure 35]

FlexiCubes

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

… Triplane Decoder

Normal Encoder

Specular Light

[Figure 41]

[Figure 42]

[Figure 43]

Patch Embedding

Tri-plane Feature

SDF Grid Predicted Mesh

GT Mesh Normal Maps

Diffuse Light

With Different

Coordinates Rasterize

Texture Branch (Sec. 3.2)

Forward Step

RGBDecoder

[Figure 44]

[Figure 45]

[Figure 46]

Sampling

| | | |
|---|---|---|
| | | |
| | | |

… Triplane Decoder

Image Encoder

Supervision Signal

3D Coordinates GT Image Pixel Feature

Result Image

Patch Embedding

Tri-plane Feature

2D Per-pixel Feature

RGB Images

- Fig. 3. The framework of our DiMeR. The upper part is the geometry branch, and exclusively uses normal maps as input. The lower part is the texture branch.

encoded into patch-wise representations P𝑔 ∈ R𝐾×𝐷×𝐶 using a ViT-based Normal Encoder, where 𝐷 is the number of patches of each view and 𝐶 is the dimension of the feature. Similar to the approach taken by LRM [Wei et al. 2024], we utilize a Triplane Decoder to gather information from the Patch Embedding P𝑔 using several transformer layers [Vaswani et al. 2017] to synthesize triplane [Chan et al. 2022] features F𝑔 ∈ R3×𝐻′×𝑊′×𝐶𝑔. Subsequently, we extract an SDF grid from the triplane features F𝑔 to apply the differential isosurface construction algorithm, FlexiCubes [Shen et al. 2023], to obtain the vertices and faces for the mesh. Finally, we can rasterize the mesh to get the normal maps, masks, and depth maps for arbitrary views. By providing the environment map and assigning different materials (metallic and roughness) to the mesh, we can render the light map (including specular and diffuse) using PBR, for enhancing the supervision from different lighting conditions, which will be introduced in the following part.

Furthermore, the design does not produce true SDF representations. To address these issues, we employ the eikonal loss [Gropp et al. 2020] to regularize the whole space as the SDF field, specifically by ensuring the norm of the gradient with respect to the coordinates is normalized to 1. Nevertheless, computing the derivative for a 𝑁3 grid poses significant challenges in terms of computational and GPU memory costs and potential overfitting at specific grid positions. To mitigate this, we propose randomly sampling positions within the space to compute the eikonal expectation loss, effectively reducing computational demands while maintaining the integrity of the regularization, i.e. ,

L𝑒𝑖𝑘 = E𝒙(∥∇𝒙SDF(𝒙)∥2 − 1)2,𝒙 ∈ R3 ∼ 𝑈𝑛𝑖𝑓𝑜𝑟𝑚(−1, 1), (1)

where we sample 200𝐾 𝒙 in each iteration to calculate the expectation. Moreover, we use the GT SDF value to supervise the SDF value of grid vertices 𝒗 ∈ R𝑁3×3 in FlexiCubes,

Mesh Extraction Algorithm. Original FlexiCubes algorithm requires two MLP networks to allow the different weights (each edge and vertex in the grid) and the deformation of the grid. However, this incurs excessive computational and GPU memory overhead. Specifically, for a 𝑁3 grid, it needs to compute the deformation of 𝑁3 vertices and the weight of 12 × 𝑁3 edges and 8 × 𝑁3 vertices. Though it is powerful for the tasks of Flexicubes itself, extensive experiments prove that these components contribute disproportionately high computational overhead with minimal performance gains. As shown in Tab. 7, we found that removing these networks from the pre-trained model does not adversely affect performance. Therefore, to enable higher efficient training and higher spatial resolutions, we prune these components to improve computational efficiency and improve the spatial resolution.

L𝑠𝑑𝑓 = ∥SDF(𝒗) − SDFGT(𝒗))∥22. (2)

To reduce computational overhead, we cache the grid of these SDF values for each object in the training set.

Drawing inspiration from Photometric Stereo [Woodham 1980], we introduce the PBR [Kajiya 1986] losses. The premise is that if the specular and diffuse light maps of a 3D mesh under different environmental lighting conditions and various materials can be accurately rendered in PBR, then the geometry of the predicted mesh model can be deemed correct. Therefore, we introduce the statistical expectation loss of PBR to supervise the geometry branch,

2

L𝑠𝑝𝑒𝑐 = E𝑒,𝑚,𝑟 Spec(Oˆ,𝑒,𝑚,𝑟) − Spec(O,𝑒,𝑚,𝑟)

Optimization. Given the inherent ambiguity introduced by the RGB texture shown in Fig. 2, we exclude RGB loss to enhance training stability. Consequently, we now exclusively employ geometryrelated losses to supervise the geometry branch of our model.

+ LPIPS Spec(Oˆ,𝑒,𝑚,𝑟), Spec(O,𝑒,𝑚,𝑟) , (3)

2

L𝑑𝑖𝑓 𝑓 = E𝑒,𝑚,𝑟 Diff(Oˆ,𝑒,𝑚,𝑟) − Diff(O,𝑒,𝑚,𝑟)

In its original implementation, FlexiCubes incorporates three regularization losses to regularize the SDF grid values generated by the network. However, extensive experimentation reveals that this approach yielded low stability [Ge et al. 2024; Xu et al. 2024a].

+ LPIPS Diff(Oˆ,𝑒,𝑚,𝑟), Diff(O,𝑒,𝑚,𝑟) , (4) where Oˆ is the predicted mesh model, O is the ground truth mesh

model, 𝑒, 𝑚, 𝑟 are the randomly sampled environment, metallic, and

roughness, Spec(·) and Diff(·) are the rendering functions of specular and diffuse light map, LPIPS(·) is the perception loss [Zhang et al. 2018]. Notably, during the training, we sample different environment, metallic, and roughness to render the light maps for a single object.

We also employ the commonly used normal, depth, and mask losses to supervise the geometry branch. Specifically,

L𝑛𝑜𝑟 = MGT ⊗ (1 − Nˆ · NGT), (5) L𝑑𝑒𝑝 = MGT ⊗ |Dˆ − DGT|, (6) L𝑚𝑎𝑠𝑘 = (Mˆ − MGT)2, (7)

where ⊗ denotes element-wise production, MGT and Mˆ are the rendered mask from ground truth mesh model and predicted mesh model, similarly, N and D represent normal map and depth map.

In general, the overall loss function is

L𝑔 = L𝑒𝑖𝑘 + L𝑠𝑑𝑓 + L𝑠𝑝𝑒𝑐 + L𝑑𝑖𝑓 𝑓 + L𝑛𝑜𝑟 + L𝑑𝑒𝑝 + L𝑚𝑎𝑠𝑘. (8)

- 3.2 Texture Branch

Network Structure. As demonstrated in Fig. 3, the texture branch starts from RGB images I ∈ R𝐾×𝐻×𝑊 ×3 with the camera embeddings 𝜁. Similar as the geometry branch, we use a ViT-based Image Encoder to get the Patch Embedding P𝑐 ∈ R𝐾×𝐷×𝐶 and utilize a Triplane Decoder to assemble the information from P𝑐 to get the triplane features F𝑐 ∈ R3×𝐻′×𝑊′×𝐶𝑐 for the texture field representation [Oechsle et al. 2019]. Given the predicted shape from the geometry branch, we rasterize the vertex coordinates 𝒗 into image space,

𝐶𝑜𝑜𝑟𝑑I = Rast(𝒗, Camera), (9) where the pixel value of𝐶𝑜𝑜𝑟𝑑I ∈ R𝐻×𝑊 ×3 is the global coordinate. Next, we query the texture feature FI ∈ R𝐻×𝑊 ×𝐶 on triplane F𝑐 for each pixel,

FI = Sample(𝐶𝑜𝑜𝑟𝑑I, F𝑐). (10) Finally, we decode the color feature to predict the image

Iˆ = RGB_Decoder(FI). (11) Optimization. In this branch, we only use RGB loss to supervise the network. Specifically,

L𝑡 = (Iˆ − IGT)2 + LPIPS(Iˆ, IGT). (12)

- 3.3 Applications

Besides the sparse-view reconstruction task, our DiMeR is also capable of performing image/text-to-3D tasks.

Single-image-to-3D. Given the input image, we first employ Zero1-2-3++ [Shi et al. 2023a] or Era3D [Li et al. 2024b] to generate six images from different viewpoints. Specifically, the output from zero123++ consists of six views, including the combinations of azimuth and elevation, (30, 20), (90, -10), (150, 20), (210, -10), (270, 20), and (330, -10). Next, we apply the SoTA normal prediction model Lotus [He et al. 2024a] or StableNormal [Ye et al. 2024] to predict the normal maps for these six views. Since the predicted normal maps are initially in the local camera coordinate system, we subsequently transform them into the global coordinate system using the transformation matrices corresponding to the six view directions. Finally,

- (a) Sparse-views-to-3D Task

[Figure 47]

[Figure 48]

MVDiffusion

NormalPred

[Figure 49]

DiMeR

- (b) Single-Image-to-3D Task

“a shiny silver robot cat”

Text-to-MV

DiMeR

- (c) Text-to-3D Task

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

NormalPred

DiMeR

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Fig. 4. Pipelines for sparse-views, single-image-, and text-to-3D.

we feed the six transformed normal maps and the RGB images into our DiMeR model to generate the textured mesh. Text-to-3D. This task is approached through two distinct pipelines:

- (I) The first pipeline involves using a text-to-image model to generate an RGB image from the input text. Subsequently, we apply the single-image-to-3D pipeline to complete the reconstruction.
- (II) With the advancement of diffusion models, Kiss3DGen [Lin et al. 2025b] fine-tunes the SoTA text-to-image generative model, FLUX [BlackForestLabs 2024], to simultaneously output RGB images along with corresponding normal maps, ensuring high multi-view consistency. Since our DiMeR supports a dynamic number of input views, we can directly feed the four views from Kiss3DGen into DiMeR for 3D model reconstruction. The generated high-quality models are presented in Fig. 1 and the supplementary materials.

4 EXPERIMENT 4.1 Implementation Details

Datasets. We train DiMeR with the filtered Objaverse [Deitke et al. 2023] according to the mesh quality, in a total of 98, 526 objects. For test datasets, we choose the widely used GSO [Downs et al. 2022] and OmniObject3D [Wu et al. 2023]. We use all 1, 029 objects in GSO and randomly select 5 objects for each class in OmniObject3D. Evaluation Protocol. For 3D metrics, we sample 32, 000 points on the surface to compute commonly used Chamfer Distance (CD) and F1-Score@0.1 to evaluate the quality of geometry. For 2D metrics, we compute the PSNR, SSIM, and LPIPS to evaluate the rendering quality over 8 rendered views. We rescale and align the generated meshes and ground truth meshes for fair comparison.

Training. We set the total batch size to 64, with learning rate of 4 × 10−6 for geometry branch and 4 × 10−5 for texture branch. The resolution of the triplane is 3 × 64 × 64, and the SDF grid is 1923, which is higher than baselines benefiting from our enhancement of mesh extraction methods. The resolutions of input and supervision are 512 × 512. For PBR statistical expectation loss, we place the predicted meshes in 10 different lighting environments and apply 10 different materials, rendering from different 10 views during

Dataset GSO OmniObject3D Metric CD (↓) F1 (↑) PSNR (↑) SSIM (↑) LPIPS (↓) CD (↓) F1 (↑) PSNR (↑) SSIM (↑) LPIPS (↓)

InstantMesh 0.045 0.964 18.51 0.846 0.150 0.039 0.983 18.44 0.842 0.153 PRM 0.041 0.977 21.68 0.869 0.126 0.034 0.991 21.65 0.865 0.135 DiMeR (GT) 0.028 0.992 23.40 0.883 0.095 0.024 0.996 23.04 0.871 0.112 Δ 31.7% ↓ +0.015 +1.72 +0.014 24.6% ↓ 29.4% ↓ +0.005 +1.39 +0.006 17.0% ↓

DiMeR (Lotus) 0.033 0.988 22.57 0.874 0.103 0.034 0.989 21.88 0.866 0.126 DiMeR (SN) 0.032 0.988 22.89 0.875 0.103 0.030 0.993 22.15 0.865 0.121

- Table 1. Quantitative results for reconstruction task. CD means Chamfer Distance. DiMeR (Lotus) and DiMeR (SN) are the reconstruction results from the normal map predicted by Lotus [He et al. 2024a] and StableNormal [Ye et al. 2024] separately. DiMeR (GT) is from the ground truth normal. value means first-best, value means second-best, value means third-best.

Dataset GSO OmniObject3D Metric CD (↓) F1 (↑) CD (↓) F1 (↑) CRM 0.144 0.781 0.114 0.854 InstantMesh 0.066 0.950 0.074 0.937 PRM 0.059 0.961 0.064 0.957 Trellis 0.119 0.859 0.090 0.902 DiMeR 0.052 0.981 0.060 0.964

- Table 2. Single-image-to-3D task. All the methods use the same single image input. Our DiMeR is equipped with Stable-Zero123++ [Shi et al. 2023a] and StableNormal [Ye et al. 2024].

meaningful and valuable evaluation. Notably, while Trellis produces high-quality mesh generation, issues with consistency in the input image persist. This inconsistency can be attributed to Trellis’ generative nature rather than being a deterministic model. This point is also highlighted in the qualitative comparisons in Fig. 7. In contrast, the reconstruction models, such as our DiMeR, PRM, and InstantMesh, have the advantages for the accurate alignment with input image based on the prediction of zero123++.

- 4.3 Qualitative Comparison

Reconstruction Task. As demonstrated in Fig. 5, we present a visual qualitative comparison of various methods. A comparison between the rows labeled "Ours" and "Ours (Lotus)" shows similar performance, highlighting that normal prediction models effectively support DiMeR. This suggests that DiMeR, when combined with such models, is capable of surpassing previous methods in realistic applications. Furthermore, DiMeR outperforms previous mesh reconstruction models, such as InstantMesh and PRM, in terms of reconstructing finer details.

Single-image-to-3D. As shown in Fig. 7, we compare our method with SoTA methods including Trellis [Xiang et al. 2024], PRM [Ge et al. 2024], MeshFormer [Liu et al. 2024], InstantMesh [Xu et al. 2024a] and CRM [Wang et al. 2025]. Notably, since the 3D results for MeshFormer are only available from their project page and the corresponding input images are not provided, we are unable to conduct a direct comparison using the same input. In contrast, the other methods use the same input images for comparison. Due to the inherent characteristics of the generative diffusion model, Trellis often generates 3D mesh models that exhibit inconsistencies with the input images, although it maintains high quality. Specifically, the cup’s holes in the second column and the number of pillars in the third column are mismatched. Moreover, the other methods encounter difficulties in generating holes and rings accurately. In summary, our DiMeR achieves the best consistency and quality.

- 4.4 Benchmark for Normal Prediction Foundation Models

training. We train the geometry branch for two days and the texture branch for one day on 8 H100 GPUs.

- 4.2 Quantitative Comparison

Reconstruction Task. As shown in Tab. 1, we compare our DiMeR on the sparse-view reconstruction task using the same 6 randomly sampled input views. Since some sparse-view reconstruction methods, like CRM [Wang et al. 2025], are limited to only support specific views (six orthogonal views), we compare them in single-imageto-3D tasks. Additionally, because MeshFormer [Liu et al. 2024] is not open-source work, we are unable to perform an accurate quantitative comparison. Therefore, we only provide qualitative visual comparisons. For the comparison, we select state-of-the-art (SoTA) methods that are accessible, including InstantMesh [Xu et al. 2024a] and PRM [Ge et al. 2024]. Experiments show that our method can surpass the SoTA methods by a large margin, whatever using GT (31.7% gain) or predicted normal maps (22.0% gain) from StableNormalTurbo [Ye et al. 2024]. Notably, when equipped with normal map prediction models, the input to DiMeR remains the same as the baselines, relying solely on RGB images. Furthermore, following the improvement of normal prediction models, there is still room for continued improvement in the performance of DiMeR.

Sinle-Image-to-3D Task. As demonstrated in Tab. 2, we compare our DiMeR with CRM [Wang et al. 2025], InstantMesh [Xu et al. 2024a], PRM [Ge et al. 2024], and Trellis [Xiang et al. 2024] using same single image input. Our pipeline for this task is shown in Fig. 4(b), where we use Lotus [He et al. 2024a] to predict normal maps from the output of zero123++ [Shi et al. 2023a]. Since the single-image-to-3D problem is inherently ill-posed, the unseen portions of the data cannot be accurately inferred from a single image alone. Consequently, we select 500 relatively clear data points for

To determine whether recent normal-prediction foundation models meet the quality requirements of our pipeline, we evaluate Lotus [He et al. 2024a], StableNormal [Ye et al. 2024], DSINE [Bae and Davison 2024], Marigold [Ke et al. 2024], and GeoWizard [Fu et al. 2024] on the GSO [Downs et al. 2022] and OmniObject3D [Wu et al. 2023] benchmarks. For each object, six randomly sampled views are

Dataset GSO OmniObject3D Metric mean (↓) median (↓) 11.25◦ (↑) 22.5◦ (↑) 30◦ (↑) mean (↓) median (↓) 11.25◦ (↑) 22.5◦ (↑) 30◦ (↑) Latency (↓) GeoWizard 17.673 14.307 48.309 74.908 83.097 23.129 20.156 28.272 61.215 74.122 2102 ms Marigold 17.400 14.305 47.303 76.058 84.197 22.934 20.243 28.867 61.201 73.824 260 ms DSINE 17.953 14.857 45.543 72.951 82.535 23.010 20.116 29.182 62.140 75.082 59 ms Lotus-G 17.151 13.920 45.343 76.831 85.277 21.523 19.048 30.836 64.828 77.359 130 ms SN V1.8.1 16.818 14.743 39.860 74.524 86.424 21.205 19.468 25.677 61.917 77.916 236 ms Lotus-D 16.606 13.377 47.218 78.076 86.166 21.065 18.622 32.118 66.216 77.968 130 ms

Table 3. Benchmark for normal map prediction of foundation models on object scenario. The latency is evaluated on a single A800 GPU.

Input RGB RGB+Normal Normal

CD 0.041 0.041 0.028 F1 0.971 0.981 0.992

Table 4. The ablation studies of different input formats.

Method FlexiCubes Ours CD 0.037 0.028 F1 0.975 0.992

Table 5. The ablation studies of regularization losses.

w/o w/

CD 0.039 0.028 F1 0.973 0.992

Table 6. The ablation studies of PBR expectation losses.

Method CD F1 GPU Mem Infer

w/ 0.045 0.964 73GB 0.5s w/o 0.045 0.963 48GB 0.2s

Table 7. The ablation studies of the effectiveness of Deformation and Weight MLP. GPU Mem is training occupancy.

rendered, producing paired RGB images, masks, and ground-truth normal maps. Using the RGB inputs, we measure mean and median angular error, the proportion of pixels with error below 11.25◦, 22.5◦, and 30◦, and inference time. When computing these metrics, we only use the foreground pixels. As summarised in Tab. 3, StableNormal and Lotus offer the best balance of accuracy and speed, adding only negligible latency. Correspondingly, as demonstrated in Tab. 1, even with errors, our DiMeR still outperforms previous methods by a large margin, accepting the same RGB input. Among the reported metrics, the mean error and the fraction of pixels with error below the threshold 30◦ are most indicative of prediction stability. Large errors markedly impact reconstruction quality. Ongoing advances in normal-prediction models are therefore expected to further improve DiMeR’s performance. We also provide the qualitative comparison in Fig. 6.

- 4.5 Ablation Studies

In this section, we validate our key designs. All experiments are conducted based on the GSO dataset.

Input Disentanglement. As shown in Tab. 4, we compare the performance of DiMeR using different input formats. The comparison between the first two columns, “RGB” and “RGB+Normal”, illustrates that incorporating geometry information results in a slight improvement in effectiveness. Furthermore, the third column, labeled “Normal”, demonstrates a significant performance gain over “RGB+Normal”. This improvement underscores the strong inductive bias between normal maps and 3D geometry. Additionally, since the input is encoded as patch embeddings, using mixed input produces more patches, resulting in increased GPU memory usage and computational overhead. In contrast, the exclusive use of normal maps maintains the same resource consumption as RGB inputs.

Regularization Loss. As demonstrated in Tab. 5, we show that Eq. 1 and Eq. 2 can replace the original loss functions used in FlexiCubes, performing improved performance. With the regularization loss employed in FlexiCubes, the training process becomes unstable and struggles to proceed beyond 10,000 iterations, resulting in unsatisfactory network convergence. By introducing the eikonal loss and incorporating 3D ground truth, we stabilize the training process, achieving significantly better performance.

PBR Loss. As shown in Tab. 6, we validate the effectiveness of the PBR expectation loss (Eq. 3 and Eq. 4). If the lighting map can be accurately computed under varying environmental lighting conditions and across different materials, we can conclude that the predicted mesh aligns well with the ground truth mesh. To achieve this, we assign different materials to the single predicted mesh and place it in various environments. The introduction of PBR losses leads to significant improvements.

Deformation and Weight MLP in FlexiCubes. As shown in Tab. 7, we demonstrate that the improvements gained from the deformation and weight MLP are not worthy enough compared with their computational cost. The experiments are conducted using the official pretrained weights of InstantMesh, and similar experiments based on PRM are provided in the supplementary material. Upon removing the deformation network and weight network from FlexiCubes, we observe minimal impact on inference performance, almost no decrease. However, these two networks significantly increase computational workload (about 2.5× computation overhead) and GPU memory consumption (about 1.5× GPU memory occupancy in training). Consequently, we opt to exclude them from DiMeR in order to improve the spatial resolution.

5 CONCLUSION

In this paper, we propose DiMeR, a disentangled dual-stream framework with 3D supervision for feed-forward sparse-view mesh reconstruction. By driving the geometry branch exclusively with normal maps and leaving RGB information to a separate texture branch, DiMeR clearly separates conflicting objectives and grounds training on unambiguous supervision signals. To enhance the training effectiveness and spatial resolution, DiMeR improves the mesh extraction algorithm by redesigning the regularization losses, introducing 3D ground-truth supervision, and removing redundant modules. Extensive experiments confirm that DiMeR surpasses state-of-the-art baselines across multiple tasks, such as sparse-view-to-3D, imageto-3D, and text-to-3D, highlighting both its effectiveness and robustness. As normal-prediction models continue to improve, DiMeR’s performance is likely to advance further.

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

InputOursOurs(Lotus)PRMInstantMeshGT

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

Fig. 5. The qualitative comparison for sparse view reconstruction.

Input GT GeoWizard Marigold DSINE Lotus-G SN V1.8.1 Lotus-D

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

Fig. 6. The qualitative comparison for normal prediction foundation models.

Input Ours Trellis PRM MeshFormer* InstantMesh CRM

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

Fig. 7. The qualitative comparison for single-image-to-3D. Please note that the results of MeshFormer are obtained from their project page and do not use the same input as other methods.

[Figure 199]

“A battle mech in a mix of red, blue, and black color, with a cannon on the head”

“Detailed facial sculpt, horned head, tapered horns, deep set eyes, prominent cheekbones, furrowed brow”

“Pink teapot model symmetrical, curved spout, rounded body, flat base, circular lid, elongated handle, tapered top”

“A person wearing a virtual reality headset, sitting position, bent legs, clasped hands”

[Figure 200]

[Figure 201]

“Charlie Brown, a cartoon character in a yellow and black outfit, upright posture”

“a blue and yellow dragon-like creature or toy”

Fig. 8. The generation results for text-to-3D.

REFERENCES

Gwangbin Bae and Andrew J Davison. 2024. Rethinking inductive biases for surface normal estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9535–9545.

Haotian Bai, Yuanhuiyi Lyu, Lutao Jiang, Sijia Li, Haonan Lu, Xiaodong Lin, and Lin Wang. 2023. CompoNeRF: Text-guided multi-object compositional NeRF with editable 3D scene layout. arXiv preprint arXiv:2303.13843 (2023).

BlackForestLabs. 2024. Flux.1 Model Family. (2024). https://blackforestlabs.ai/ announcing-black-forest-labs Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K Warmuth. 1987. Occam’s razor. Information processing letters 24, 6 (1987), 377–380.

Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. 2022. Efficient geometry-aware 3D generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16123–16133. Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873 (2023).

Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Yanru Wang, Zhibin Wang, Chi Zhang, et al. 2024a. MeshXL: Neural Coordinate Field for Generative 3D Foundation Models. arXiv preprint arXiv:2405.20853 (2024).

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. 2024b. MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers. arXiv preprint arXiv:2406.10163 (2024).

Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. 2024c. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. arXiv preprint arXiv:2408.02555 (2024).

Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, and Li Yuan. [n.d.]. Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts. In The Twelfth International Conference on Learning Representations.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. 2024. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems 36 (2024).

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13142–13153.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. 2022. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2553–2560.

Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. 2024. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision. Springer, 241–258.

Wenhang Ge, Jiantao Lin, Guibao Shen, Jiawei Feng, Tao Hu, Xinli Xu, and Ying-Cong Chen. 2024. PRM: Photometric Stereo based Large Reconstruction Model. arXiv preprint arXiv:2412.07371 (2024).

Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. 2020. Implicit geometric regularization for learning shapes. In Proceedings of the 37th International Conference on Machine Learning. 3789–3799.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Oğuz. 2023. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371

(2023).

Hao He, Yixun Liang, Luozhou Wang, Yuanhao Cai, Xinli Xu, Hao-Xiang Guo, Xiang Wen, and Yingcong Chen. 2024b. LucidFusion: Generating 3D Gaussians with Arbitrary Unposed Images. arXiv preprint arXiv:2410.15636 (2024).

Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. 2024a. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124 (2024). Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Lutao Jiang, Hangyu Li, and Lin Wang. 2024. A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness. In Proceedings of the 32nd ACM International Conference on Multimedia. 6803–6812.

Lutao Jiang and Lin Wang. 2024. BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis. arXiv preprint arXiv:2403.11273 (2024). James T Kajiya. 1986. The rendering equation. In Proceedings of the 13th annual conference on Computer graphics and interactive techniques. 143–150.

Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. 2024. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition. 9492–9502.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (2023).

Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. 2023a. Instant3d: Fast textto-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214 (2023).

Ming Li, Pan Zhou, Jia-Wei Liu, Jussi Keppo, Min Lin, Shuicheng Yan, and Xiangyu Xu. 2023b. Instant3D: Instant Text-to-3D Generation. arXiv:2311.08403 [cs.CV]

Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wei Xue, Wenhan Luo, et al. 2024b. Era3d: high-resolution multiview diffusion using efficient row-wise attention. Advances in Neural Information Processing Systems 37 (2024), 55975–56000.

Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. 2024a. CraftsMan: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner. arXiv preprint arXiv:2405.14979 (2024).

Zongrui Li, Minghui Hu, Qian Zheng, and Xudong Jiang. 2025. Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation. In European Conference on Computer Vision. Springer, 274–291.

Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen.

2023. LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching. arXiv preprint arXiv:2311.11284 (2023).

Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. 2025a. DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation. arXiv preprint arXiv:2501.16764 (2025).

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3d: Highresolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 300–309.

Jiantao Lin, Xin Yang, Meixi Chen, Yingjie Xu, Dongyu Yan, Leyi Wu, Xinli Xu, Lie Xu, Shunsi Zhang, and Ying-Cong Chen. 2025b. Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation. arXiv preprint arXiv:2503.01370 (2025).

Minghua Liu, Chong Zeng, Xinyue Wei, Ruoxi Shi, Linghao Chen, Chao Xu, Mengqi Zhang, Zhaoning Wang, Xiaoshuai Zhang, Isabella Liu, et al. 2024. Meshformer: High-quality mesh generation with 3d-guided reconstruction model. arXiv preprint arXiv:2408.10198 (2024).

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision. 9298–9309.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9970–9980.

Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, and James Lucas. 2023. ATT3D: Amortized Text-to-3D Object Synthesis. arXiv preprint arXiv:2306.07349 (2023).

Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, and Yao Yao. 2024. Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8744–8753.

Artem Lukoianov, Haitz Sáez de Ocáriz Borde, Kristjan Greenewald, Vitor Campagnolo Guizilini, Timur Bagautdinov, Vincent Sitzmann, and Justin Solomon. 2024. Score Distillation via Reparametrized DDIM. arXiv preprint arXiv:2405.15891 (2024).

Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. 2024. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. arXiv preprint arXiv:2402.08682 (2024).

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. 2023. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12663–12673.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas Geiger.

2019. Texture fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF international conference on computer vision. 4531–4540.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations.

Guocheng Qian, Junli Cao, Aliaksandr Siarohin, Yash Kant, Chaoyang Wang, Michael Vasilkovsky, Hsin-Ying Lee, Yuwei Fang, Ivan Skorokhodov, Peiye Zhuang, et al. 2024. Atom: Amortized text-to-mesh using 2d diffusion. arXiv preprint arXiv:2402.00867 (2024).

Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. 2023. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508 (2023).

Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. 2024a. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4209–4219.

Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, and Jiahui Huang. 2024b. Scube: Instant large-scale scene reconstruction using voxsplats. arXiv preprint arXiv:2410.20030 (2024).

Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2023. Flexible Isosurface Extraction for Gradient-Based Mesh Optimization. ACM Trans. Graph. 42, 4 (2023), 37–1.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023a. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023).

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023b. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512

(2023).

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. 2024. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19615–19625.

Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. 2024. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10208–10217.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. 2025. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision. Springer, 1–18.

Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. 2024. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. arXiv preprint arXiv:2409.18114 (2024).

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2023. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023).

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. 2024. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37 (2024), 84839–84865.

Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. 2024. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151 (2024).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. 2024. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision. Springer, 439–457. Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. 2023a. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12619–12629.

Peng Wang and Yichun Shi. 2023. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201 (2023).

Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. 2024. LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models. arXiv preprint arXiv:2411.09595 (2024).

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023b. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. arXiv preprint arXiv:2305.16213 (2023).

Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. 2025. Crm: Single image to 3d textured mesh with convolutional reconstruction model. In European Conference on Computer Vision. Springer, 57–74.

Xinyue Wei, Fanbo Xiang, Sai Bi, Anpei Chen, Kalyan Sunkavalli, Zexiang Xu, and Hao Su. 2023. Neumanifold: Neural watertight manifold reconstruction with efficient and high-quality rendering support. arXiv preprint arXiv:2305.17134 (2023).

Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. 2024. Meshlrm: Large reconstruction model for high-quality mesh. arXiv preprint arXiv:2404.12385 (2024).

Haohan Weng, Yikai Wang, Tong Zhang, CL Chen, and Jun Zhu. 2024. PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance. arXiv preprint arXiv:2405.16890 (2024).

Robert J Woodham. 1980. Photometric method for determining surface orientation from multiple images. Optical engineering 19, 1 (1980), 139–144.

Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. 2024. Unique3d: High-quality and efficient 3d mesh generation from a single image. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. 2023. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 803–814.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024).

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2024. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 (2024).

Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. 2024a. Instantmesh: Efficient 3d mesh generation from a single image with sparseview large reconstruction models. arXiv preprint arXiv:2404.07191 (2024).

Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. 2024b. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv preprint arXiv:2403.14621 (2024).

Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. 2024. Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation. arXiv preprint arXiv:2411.02293 (2024).

Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. 2024. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–18.

Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. 2025. Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging. arXiv preprint arXiv:2503.22236 (2025).

Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. 2023. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. arXiv preprint arXiv:2310.08529 (2023).

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 2023. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–16.

Biao Zhang and Peter Wonka. 2024. Lagem: A large geometry model for 3d representation learning and diffusion. arXiv preprint arXiv:2410.01295 (2024).

Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. 2024a. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision. Springer, 1–19.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024b. CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. 2024. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 (2024).

Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. [n.d.]. GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting. In Forty-first International Conference on Machine Learning.

