## Urban Architect: Steerable 3D Urban Scene Generation with Layout Prior

Fan Lu1 Kwan-Yee Lin2† Yan Xu3 Hongsheng Li2,4,5 Guang Chen1† Changjun Jiang1 1Tongji University 2Shanghai AI Laboratory 3University of Michigan

4The Chinese University of Hong Kong 5CPII

{lufan,guangchen,cjjiang}@tongji.edu.cn

linjunyi9335@gmail.com, yxumich@umich.com, hsli@ee.cuhk.edu.hk

# arXiv:2404.06780v1[cs.CV]10Apr2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]<br><br>4|
|---|

|[Figure 7]<br><br>[Figure 8]<br><br>1|[Figure 9]<br><br>2|
|---|---|

|[Figure 10]<br><br>3|
|---|

3 2

|[Figure 11]|[Figure 12]|
|---|---|
| | |

|[Figure 13]|[Figure 14]|
|---|---|
| | |

|[Figure 15]|[Figure 16]|
|---|---|
| | |

|[Figure 17]<br><br>[Figure 18]| |
|---|---|
| | |

1

4

“night time” “rainy” “in winter, with snow”

3D Layout

“minecraft style”

Figure 1. Urban Architect for steerable 3D urban scene generation. We present Urban Architect, a method to generate steerable 3D urban scenes by introducing 3D layout as an additional prior to complement textual descriptions. The framework enjoys three key properties: (1) Large-scale urban scene creation. The scale goes beyond 1000m driving distance in our experiments. (2) High quality. The generated scene enables photo-realistic rendering (upper row) and obeys geometric consistency (the left side of the first camera’s upper image). (3) Steerable creation process. It supports various scene editing effects by fine-tuning the generated scene in a breeze (e.g., style editing (lower row)). Project page: https://urbanarchitect.github.io.

### Abstract

tual descriptions, and meanwhile enables steerable generation. Based on the 3D layout representation, we propose two modifications to the current text-to-3D paradigm – (1) We introduce Layout-Guided Variational Score Distillation (LG-VSD) to address model optimization inadequacies. It incorporates the geometric and semantic constraints of the 3D layout into the the fabric of score distillation sampling process, effectuated through an elegant formula extension into a conditional manner. (2) To handle the unbounded nature of the urban scenes, we represent the 3D scene with a Scalable Hash Grid structure, which incrementally adapts to the growing scale of urban scenes. Extensive experiments substantiate the robustness of our framework, showcasing its capability to scale text-to-3D generation to large-scale urban scenes that cover over 1000m driving distance for the first time. We also present various scene editing demonstrations (e.g., style editing, object manipulation, etc.), showing the complementary powers of both 3D layout prior and text-

Text-to-3D generation has achieved remarkable success in digital object creation, attributed to the utilization of large-scale text-to-image diffusion models. Nevertheless, there is no paradigm for scaling up the methodology to urban scale. The complexity and vast scale of urban scenes, characterized by numerous elements and intricate arrangement relationships, present a formidable barrier to the interpretability of ambiguous textual descriptions for effective model optimization. In this work, we surmount the limitations via a paradigm shift in the current text-to-3D methodology, accomplished through a compositional 3D layout representation serving as an additional prior. The 3D layout comprises a set of semantic primitives with simple geometric structures (e.g., cuboids, ellipsoids, and planes), and explicit arrangement relationships. It complements tex-

†corresponding authors.

to-image diffusion models in our framework for steerable urban scene generation.

### 1. Introduction

A steerable paradigm to create 3D urban scenes with realism and flexibility hosts utmost significance to various applications (e.g., autonomous driving simulation, virtual reality, games, and etc.). In previous research, tasks related to urban-scale scene creation, such as 3D-aware image synthesis [2, 38, 68, 70] and 3D scene reconstruction [40, 43], have made strides with different methodologies and specializations. However, they are all stuck in the dilemma of trading-off among scene scale, quality, flexibility, and geometric consistency. This prompts a pivotal inquiry: What methodology could facilitate the realization of steerable 3D urban scene creation?

Text-to-3D generation offers a promising direction. Rooted in the flexible and versatile textual condition, it enjoys the excellence of both remarkable quality from textto-image diffusion models, and the geometric consistency from 3D representations. Notable examples include, among many others, DreamFusion [51], Magic3D [37] and ProlificDreamer [65], etc. At the heart of these advances is the idea of Score Distillation Sampling (SDS) [51]. It brings the best of two worlds together by optimizing a 3D model via aligning the distribution of rendered images with the target distribution derived from the diffusion model. Nevertheless, the ascendancy faces a formidable challenge when directly extending the paradigm to urban-scale scenes.

Urban scenes, characterized by high complexity and unboundedness, pose two key challenges for existing text-to3D paradigms to create high-quality 3D content: (1) How to optimize a scene with dense concepts and intricate arrangement relationships? Text prompts are inherently ambiguous and struggle to provide fine-grained conditional signals. This granularity gap between text and real-world urban scene content triggers the insufficient optimization predicament of SDS-based methodology, where SDS (and its variant VSD [65]) would fail to capture the complex multimodal distribution of real-world urban scene, given only the ambiguous textual condition. (2) How to represent a 3D scene that is unbounded and vast in spatial scale? Previous text-to-3D methods mainly utilize bounded models such as bounded NeRF [46, 51] and tetrahedron mesh [37, 59] as 3D representation, which is inapplicable to unbounded urban scenes with arbitrary scales.

In this work, we present Urban Architect, a lightweight solution that utilizes the power of a compositional 3D layout representation to overcome the above dilemmas and extend the current text-to-3D paradigm to urban scale. The 3D layout representation is an in-hand specific prior, which could complement the ambiguities of text prompts in a breeze.

As shown in Fig. 1, the 3D layout comprises a set of semantic primitives, where instances of an urban scene are represented using these simple geometric structures (e.g., cuboids for buildings and cars, ellipsoids for vegetation, and planes for roads), making it effective for guiding model optimization and easy for users to assemble. Centered on the representation, we provide two modifications to the current text-to-3D paradigm.

To address model optimization inadequacies, we propose Layout-Guided Variational Score Distillation (LG-VSD). It is a conditional extension of SDS and VSD, which integrates additional conditions from the 3D layout into the optimization process. Concretely, we first map the 3D layout constraints to 2D space by rendering 2D semantic and depth maps at arbitrary camera views, which results in a sequence of 2D conditions with 3D consistency. To preserve the original power of the pre-trained 2D diffusion model, and meanwhile inject our explicit constraints, we leverage ControlNet [72] to integrate conditions from the rendered semantic and depth maps. Consequently, we could control the score distilling process with the 2D conditional signals derived from the 3D layout. As these signals provide explicit prior to semantic and geometric distribution of the scene, they could enforce the optimization process to match the 3D layout consistency and therefore lead to high-fidelity results. Moreover, as LG-VSD only changes the sampling process, the modification is seamlessly grafted to the original diffusion model. Thus, we could harness the inherent refinement ability of the diffusion model to further improve the rendering quality. To ensure the semantic and geometric consistency between coarse and refined stages, we introduce a layout-aware refinement strategy to fine-tune the generated scene in practice.

We further present a Scalable Hash Grid (SHG) structure as a pragmatical and effective means to represent 3D unbounded urban scenes. Uniquely tailored for urban scenarios, SHG embodies two key design principles. Firstly, instead of encapsulating the entire scene within a single 3D model, SHG is a hash structure that is adaptively updated with the dynamic change of camera trajectories. This incremental functionality liberates it from a fixed spatial range that current 3D representations suffer. Second, we exploit the geometric information of the 3D layout to constrain the sampling space of the hash grid, thereby enhancing the overall rendering quality.

Extensive experiments substantiate the capacity of the proposed method to generate complex 3D urban scenes with high quality, diversity, and layout consistency based on the provided 3D layouts. The robust constraints from the 3D layout prior, combined with the scalable hash grid, facilitate the generation of large-scale urban scenes covering over 1000m driving distance. By leveraging the flexible and finegrained constraints of our layout representation, along with

the text-based style control, our pipeline supports various scene editing effects, including instance-level editing (e.g., object manipulation) and style editing (e.g., transferring city style and weather).

### 2. Related Work

3D Representations for Urban Scene. Various 3D representations have been proposed to model 3D scenes. Explicit representations, such as point clouds and meshes, pose challenges for optimization through purely 2D supervisions given their discrete nature, especially in large-scale scenes due to the high complexity and ill-posed optimization process. NeRF [46] adopts implicit neural representations to encode densities and colors of the scene and exploits volumetric rendering for view synthesis, which can be effectively optimized from 2D multi-view images. Given its superior performance in view synthesis, NeRF has been used to model diverse 3D scenes. Numerous works have enhanced NeRF in terms of rendering quality [4–6, 27], efficiency [11, 19, 32, 48, 60], etc. Some works have also extended NeRF to large-scale urban scenes [43, 54, 61, 67, 69]. For example, URF [54] adopts LiDAR point clouds to facilitate geometric learning and uses a separate network to model the sky. However, prior works often assume a fixed scene range and contract distant structures, encountering challenges when extending to urban scenes with arbitrary scales.

Text-to-3D. Early text-to-3D methods adopted CLIP [52] to guide the optimization but encountered difficulties in creating high-quality 3D content [25, 29, 47, 57]. Recently, large-scale diffusion models [3, 55, 56] have demonstrated significant success in text-to-image generation, capable of generating 2D images with high fidelity and diversity. They further enable fine-grained control and editing by employing additional conditioning models [44, 72]. For example, ControlNet [72] incorporates 2D pixel-aligned conditional signals to control the generation. SDEdit [44] achieves conditional image synthesis and editing using reverse SDE. The achievements in 2D content creation have propelled text-to3D generation [12, 28, 37, 45, 51, 62, 65]. As the core of most current text-to-3D methods, Score Distillation Sampling (SDS) [51] optimizes a 3D model by aligning 2D images rendered at arbitrary viewpoints with the distribution derived from a text-conditioned diffusion model. However, SDS suffers from issues such as over-smoothness and oversaturation. VSD [65] proposes a particle-based variational framework to enhance the generation quality of SDS. Some method also attempts to incorporate 3D information into the generation process to enhance 3D consistency [13, 31, 45]. For example, AvatarCraft [31] utilize SMPL [41] mesh for 3D human avatar generation and animation. Nonetheless, most prior works are tailored for single object-level generation. Some methods have attempted to achieve scene-level

generation, while the scene scales are quite limited [15, 39]. When applied to large-scale urban scenes, current text-to3D methods face challenges in modeling the intricate distribution from the purely text-conditioned diffusion model. Alternative methods have explored incrementally reconstructing and inpainting the scene to create room-scale [24] or zoom-out [18] scenes. However, this paradigm is susceptible to cumulative errors and poses challenges in scalability for larger scenes.

3D Generative Models. Many 3D generative models leverage GAN [21], VAE [34], and diffusion model [23] to model the distribution of 3D representations [1, 64, 66, 71]. However, this paradigm requires a large amount of 3D data and is often limited to single-object generation. NeuralFieldLDM [33] trains a scene-level 3D diffusion model by preparing abundant voxel-based neural fields, while the scales are limited to the pre-defined 3D voxel grid. Another line of work employs 3D-aware GANs to optimize a 3D model from 2D image collections. Early 3D-aware GANs are mainly designed for object-level scene generation [10, 16, 20, 58]. Recently, some methods have extended the pipeline to larger scenes including indoor scenes [7, 17], landscapes [14], and urban scenes [2, 38, 70]). For example, GSN [17] proposes a local latent grid representation for indoor scene generation. SceneDreamer [14] aims at generating landscapes, which contain repeated textures and structures. CityDreamer [68] further enhances SceneDreamer’s foreground generation quality for city scene generation. The key modification is separating building instances from other background objects. The main insight of these two methods to achieve unbounded scene generation is splitting scenes into local windows, and encoding local scene features implicitly in hash grids. However, their procedural generation process constrains the diversity and the handling of structural details, which are critical in urban scene generation. Besides, the implicit scene encoding-based hash representation limits the ability to handle complex scenes. Some methods have also attempted to incorporate user inputs to control urban scene generation. CC3D [2] integrates 2D layout conditions in the 3D generative model for urban scene generation. UrbanGIRAFFE [70] and InfiniCity [38] incorporate 3D voxel grids to enhance quality and scalability. Nevertheless, owing to inherent constraints such as limited training data and model capacity, they still suffer from structural distortions and the lack of details, and often overfit to the training scenarios.

### 3. Method

We propose Urban Architect for large-scale 3D urban scene generation. The overall pipeline is illustrated in Fig. 2. Given the 3D layout of the desired scene, we first optimize the scene by distilling a pre-trained text-to-image diffusion model via the proposed Layout-Guided Varia-

∇ ℒ ≜ 𝔼 , , [𝜔(𝑡)(𝜖 (𝑥 ,𝑡,𝑦,ℱ 𝐿 𝑇 − 𝜖 (𝑥 ,𝑡,𝑇,𝑦,ℱ 𝐿 𝑇 )))

𝜕𝑔(𝜃,𝑇) 𝜕𝜃

]

condition

###### ℒ ℒ

𝒙

generate

update

3D Layout Prior (𝑳)

###### (a) Layout-Guided Variational Score Distillation (LG-VSD)

control

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Diffusion Model (𝝐 )

generate

ControlNet (ℱ(⋅))

LoRA (𝝐 )

render

[Figure 28]

[Figure 29]

[Figure 30]

𝝐

[Figure 31]

𝒙

update

ℒ

ℒ

⨁

Semantic & Depth (𝐿(𝐓))

(c) Scalable Hash Grid

###### (b) Layout-Aware Refinement

update

[Figure 32]

constrain

[Figure 33]

[Figure 34]

[Figure 35]

𝐻

update

render

ℒ

[Figure 36]

add noise

𝐻 𝐻 𝐻 𝐻

𝒙

denoise

(d) Scene Editing

Instance Editing Style Editing

Edited 3D Layout Style Prompts “Rainy”

[Figure 37]

[Figure 38]

[Figure 39]

fine-tune

Scalable Hash Grid

LG-VSD

“Cartoon style” “Night time”

Editing Results

Editing Signals

Figure 2. Overview of Urban Architect. We introduce Urban Architect, a method that generates urban-scale 3D scenes with 3D layout instruction and textural descriptions. The scene is represented by a neural field that is optimized by distilling a pre-trained diffusion model in a conditional manner. (a) Rather than relying solely on the text-based guidance, we propose to control the distilling process of Variational Score Distillation (VSD) via the 3D layout of the desired scene, introducing Layout-Guided Variational Score Distillation (LG-VSD). (b) We refine the local details via a layout-aware refinement strategy. (c) To model unbounded urban scenes, we discretize the 3D representation into a scalable hash grid. (d) We support various scene editing effects by fine-tuning the generated scene.

tional Score Distillation (LG-VSD), which complements textual descriptions with additional 3D layout constraints (Sec. 3.2). Then, the generated scene is refined by a layoutaware refinement strategy to enhance quality (Sec. 3.3). The 3D scene is represented using a scalable hash grid representation to tackle the unbounded nature of urban scenarios (Sec. 3.4). Finally, we demonstrate the ability of our pipeline to achieve diverse scene editing effects by finetuning the generated scene (Sec. 3.5).

where t ∼ U(0.02,0.98), ω(t) weights the loss given the time step t, and xt is the randomly perturbed rendered image. Although SDS can produce reasonable 3D content aligned with the text prompt, it exhibits issues such as oversaturation, over-smoothing, and low diversity.

Variational Score Distillation (VSD). To address the limitations of SDS, instead of optimizing a single sample point, VSD proposes to optimize a distribution q0µ(x0|y) of possible 3D representations with µ(θ|y) corresponding to the text prompt y. VSD adopts a set of 3D parameters {θ}ni=1 as particles to represent the scene distribution and optimizes the 3D model by matching the score of noisy real images and that of noisy rendered images at each time step t. In practice, the score function of noisy rendered images is estimated by optimizing a low-rank adaptation (LoRA) model ϵϕ [26] of the pre-trained diffusion model ϵp. The gradient of VSD loss can be formulated as:

#### 3.1. Preliminaries

Score Distillation Sampling (SDS). SDS achieves text-to3D generation by distilling a pre-trained text-to-image diffusion model ϵp to optimize a differentiable 3D representation parameterized by θ. SDS encourages the image x0 = g(θ,T) rendered at an arbitrary camera view T to match the distribution p0(x0|y) derived from the diffusion model conditioned on the text prompt y. In practice, SDS optimizes the 3D model by adding random noise ϵ ∼ N(0,I) to x0 and calculating the difference between the predicted noise from the diffusion model ϵp and the added noise ϵ. The gradient of SDS loss can be formulated as:

∇θLVSD(θ) ≜ Et,ϵ,T ω(t)(ϵp(xt, t, y)−

(2)

∂g(θ, T) ∂θ

.

ϵϕ(xt, t, T, y))

VSD proves to be effective in alleviating issues of SDS such as over-saturation and over-smoothing, and successfully generates high-fidelity and diverse 3D content.

∂g(θ, T) ∂θ

∇θLSDS(θ) ≜ Et,ϵ,T ω(t)(ϵp(xt, t, y) − ϵ)

, (1)

#### 3.2. Layout-Guided Variational Score Distillation

ℋ

VSD relies on text prompts to guide model optimization. However, given the inherent ambiguity of text prompts y, it is challenging for VSD to estimate the distribution of a complex urban scene from a diverse target distribution p0(x0|y). Therefore, introducing additional constraints to obtain a more compact distribution is crucial for high-quality 3D urban scene generation. A straightforward approach is to simply fine-tune the diffusion model to constrain the distribution to the desired urban scene. However, this operation can only provide style constraints and VSD still struggles to capture the intricate geometric and semantic distribution of urban scenes, leading to unsatisfactory results.

ℋ ℋ

|ℋℋ|
|---|

ℋ ℋ

|ℋ|
|---|

|ℋ|
|---|

|ℋ ℋ|
|---|

Single Hash Grid Scalable Hash Grid (Ours)

Figure 3. Illustration of the scalable hash grid representation. We decomposed the scene into a set of stuff and object hash grids (i.e., {Hks, Hko}). The grids grow with the camera trajectory in a dynamic manner.

Given the above observations, we introduce 3D layout L as a prior to provide additional constraints to the target distribution with a formula shift (i.e., p0(x0|y) → p0(x0|y,L)). Nonetheless, the direct integration of the 3D information into the 2D framework poses a challenge. We observe that the projected semantic and depth maps provide a comprehensive description of the 3D layout in 2D space. They offer both semantic and geometric constraints of the scene, and inherently exhibit multi-view consistency. Thus, we propose to condition the distilling process via the 2D semantic and depth maps rendered at the given camera pose T. Specifically, we first train a ControlNet [72] F by utilizing the 2D semantic and depth maps rendered from 3D layouts as conditions, along with the corresponding 2D images as ground truths. At each training step of VSD, we render the 2D signals from the conditional 3D layout L at a randomly sampled camera view T. Subsequently, the features produced by F are integrated into both the diffusion model ϵp and the LoRA model ϵϕ, leading to a compact, scene-specific target distribution p0(x0|y,F(L(T))). Formally, the gradient of our layout-guided VSD (LG-VSD) loss can be written as:

#### 3.3. Layout-Aware Refinement

Inspired by SDEdit [44], we observe that the resampling process of diffusion models can effectively refine the rendered images. Specifically, given a random camera pose, the rendered image Ir is perturbed to Ip with random noise given the time step t. Then Ip will be denoised to the refined image If using the diffusion model. When t → 0, the resampling process tends to refine local details of the image Ir while maintaining its overall structure. Still and all, the geometric and semantic consistencies in long-horizon trajectory cannot be guaranteed, as such a resampling process is agnostic to contextual 3D structures among frames. To this end, we further enable layout-aware refinement by conditioning the denoising steps using the rendered semantic and depth maps, leading to better consistency. The hash grids are then updated with the MSE loss between Ir and If, i.e., LMSE = ||Ir − If||22.

#### 3.4. Scalable Hash Grid

∇θLLG-VSD(θ) ≜ Et,ϵ,T ω(t)(ϵp(xt, t, y, F(L(T))−

We introduce a Scalable Hash Grid representation to support unbounded urban scene generation with arbitrary scales. Our representation is based on Zip-NeRF [6], which combines the fast hash grid-based representation of Instant-NGP [48] with the anti-aliasing ability of MipNeRF 360 [5], achieving high-quality and efficient neural rendering. As shown in Fig. 3, instead of modeling the entire scene using a single 3D model, we propose to decompose the scene into a set of stuff grids {H1s,··· ,HNs } and object grids {H1o,··· ,HMo } to enable flexible spatial expansion and further object manipulation.

(3)

∂g(θ, T) ∂θ

ϵϕ(xt, t, T, y, F(L(T))))

.

By maximizing the likelihood of the rendered 2D images in the layout-conditioned diffusion model at arbitrary camera views, LG-VSD gradually optimizes the 3D representation θ to align with the semantic and geometric distribution of the desired scene layout.

Except for the LG-VSD loss, we also use an additional CLIP [52] loss (i.e., LCLIP) to guide the generation. Concretely, we encourage the rendered image Ir to include consistent content information with the generated image Ig from the layout-controlled diffusion model. We use the pretrained image encoder of CLIP to extract features from Ir and Ig and calculate the squared L2 distance as the loss.

Stuff Grid. Each stuff grid Hks models a static space within an axis-aligned bounding box (AABB) centered at tk ∈ R3 and of size sk ∈ R3. All structures, excluding objects (i.e., cars in our implementation) are modeled within stuff grids. Given an incoming camera pose T, we first sample points within the camera frustum. Subsequently, the correspond-

ing Hks can be easily determined by transforming the sampled points from world space to the canonical space of each

stuff grid. A new stuff grid (e.g., HNs +1) will be added to the stuff sets when the points fall out of all existing stuff grids. In this way, the stuff representation can be freely updated according to the camera trajectories, making it fully scalable to arbitrary scales.

Object Grid. Similarly, each object grid Hko also models a space within a 3D bounding box while is parameterized by a

rotation matrix Rk ∈ SO(3), a translation vector tk ∈ R3, and a size sk ∈ R3, as it is not axis-aligned.

Layout-Constrained Rendering. For each pixel of the image, we sample a set of points on the camera ray r and assign each point xi to the corresponding stuff or object grid. Thanks to the 3D layout prior, we can simply constrain the sampling space to the interiors of layout instances, which accelerates convergence and leads to better rendering quality. Thereafter, we transform xi from world space to the canonical space of the corresponding hash grid (stuff or object) to predict the density σi and color ci for subsequent volumetric rendering. Besides, to model the sky region of urban scenes, we follow URF [54] to predict the color in sky region using the ray direction d with a separate MLP Rsky(·). Formally, the rendered pixel color C(r) can be calculated as:

C(r) =

N

i

Tiαici + (1 −

N

i

αi = 1 − e(−σ

iδi), Ti =

Tiαi)Rsky(d),

- i−1
- j=1

(1 − αj).

(4)

With the scalable hash grid representation, our pipeline can achieve large-scale scene generation in a breeze. In contrast to the generative hash grid representation employed in SceneDreamer [14], which encodes scene features implicitly, we explicitly split scenes into stuff and object grids for better scalability.

#### 3.5. Scene Editing

Instance-Level Editing. The compositional 3D layout representation naturally supports instance-level scene editing. For stuff elements (e.g., buildings, trees, etc.), we can delete or insert instances in the scene layout and then fine-tune the generated scene using our LG-VSD for possible missing region completion. For objects (i.e., cars), the manipulation (e.g., rotation, translation, etc.) can be simply applied simultaneously on the instance layout and also the object grid parameters (i.e., Rk,tk,sk).

Style Editing. Benefiting from the inherent capacity of the large-scale diffusion model, we can easily transfer the generated scene to various styles. Specifically, given the generated scene, we simply fine-tune it via our LG-VSD,

[Figure 40]

Road and Sidewalk Car Building Vegetation and Terrain

Figure 4. Basic primitives. We provide several basic primitives of common objects in urban scenes (e.g., road and sidewalk, car, building, etc.)

while adding a style text prompt in the distilling process, e.g., “Cartoon style”, “Night time”, etc.

#### 3.6. Layout Construction

3D layout construction is not the main focus of this work. In practice, we use in-hand layout data provided by the KITTI-360 dataset to train the ControlNet. Nonetheless, there are also several alternatives. For example, users can easily create the desired layout in 3D modeling software (e.g., Blender) given the simple geometric structures of basic primitives. As shown in Fig. 4, we provide several basic primitives in urban scenes to ease the process.

We further provide an alternative method to generate 3D scene layout automatically based on SinDDM [35]. SinDDM learns the intrinsic distribution of the training image via a multi-scale diffusion process, enabling the generation of new images with arbitrary scales. Specifically, we first compress the 3D layout to the 2D ground plane and obtain a 2D representation of the 3D scene. Then, we train the SinDDM model based on the single 2D example. As shown in Fig. 5, the trained model can produce diverse new samples with arbitrary scales and reasonable arrangements. We also generate scenes based on the generated 3D layouts and the rendered results are provided in the bottom row of Fig. 5.

### 4. Experiments 4.1. Experimental Setup

Dataset. We mainly perform experiments on the KITTI360 dataset [36] for quantitative comparisons with baseline methods. KITTI-360 dataset is captured in urban scenarios covering a driving distance of around 73.7km and offers 3D bounding box annotations for various classes (e.g., building, car, road, vegetation, etc), forming extensive 3D scene layouts. Each layout instance is modeled using a triangle mesh, enabling fast rendering of semantic and depth maps.

Implementation Details. We use Stable Diffusion 2.1 [55] as the text-to-image diffusion model. The overall framework is implemented using PyTorch [50]. We use diffusers [63] to implement the diffusion model. AdamW [42] is used as the optimizer with an learning rate of 1 × 10−3. For the training of ControlNet [72], we concatenate the ren-

We use the officially released code and pre-trained weights provided by the authors of CC3D for the evaluation of CC3D and EG3D. Due to the different settings of UrbanGIRAFFE, we re-train the model using its official released code. Text2Room is initially designed for roomscale textured mesh generation. We adapt it to our scenes by integrating our pre-trained ControlNet into the inpainting model of Text2Room, enabling the generation of urban scenes based on our 3D layout information. SceneDreamer aims to generate 3D landscapes and we apply it to urban scenes using the projected height and semantic field from our layout representation as the input. We then re-train both SceneDreamer and the required SPADE [49] model for urban scene generation. For quantitative evaluation, we use Fr´echet Inception Distance (FID) [22] and Kernel Inception Distance (KID) [8] as evaluation metrics, which are computed with 5000 randomly sampled images. Quantitative comparisons with InfiniCity are excluded due to the different settings and the incomplete released code.

Samples with same scale

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

generate

Single example

generate

Samples with different scale

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

4

2

3

1

|[Figure 51]<br><br>1|
|---|

|[Figure 52]<br><br>2|
|---|

|[Figure 53]<br><br>3|
|---|

|[Figure 54]<br><br>4|
|---|

Results. According to the quantitative results in Table 1, the proposed method outperforms previous methods in terms of both FID and KID by a significant margin. For example, the FID of our method is about 24% lower than that of the most competitive baseline method, CC3D. We further provide visualization results of generated samples in Fig. 6 for qualitative comparison. According to the results, Text2Room tends to degrade rapidly under the influence of cumulative errors, resulting in a failure to produce valid results. Besides, 3Daware GAN-based methods (i.e., SceneDreamer, UrbanGIRAFFE, EG3D, CC3D and Infinicity) suffer from structural distortions and lack fine-grained details, due to the limited model capacity and the complexity of urban scenes. In contrast, our method is capable of generating high-quality 3D urban scenes with rich details, demonstrating a notable advancement over previous methods. To further investigate the effect of our 3D layout prior, we add height fields of 3D layouts as an extra condition to CC3D, resulting in an enhanced version termed CC3D+H. As shown in Table 2 and Fig. 7, CC3D+H marginally outperforms CC3D, indicating that our 3D layout prior is beneficial for urban scene generation. However, it remains significantly inferior to our method, which demonstrates the effectiveness and necessity of our carefully designed pipeline to fully exploit the potential of 3D layout priors and large-scale text-to-image diffusion models. Please refer to our project page for video results.

- Figure 5. Automatic Layout Generation. We present an alternative method for automatic 3D layout generation given a single example of the layout. In the top two rows, We display generated layouts with different scales and provide the corresponding 3D layout given the generated 2D sample in the third row. The rendering results of the generated scene are displayed in the bottom row.

dered semantic and depth maps from 3D layouts as the input conditional signal. We crop and resize the original image (with a resolution of 1408 × 376) in the KITTI360 dataset [36] to a resolution of 512 × 512. During the refinement, we further adopt monocular depth estimation method [53] to predict the depth for the rendered image and align the scale and shift with the rendered depth from the generated hash grid. We then refine the geometry by adding an L1 error between the rendered and the aligned monocular depth. Additionally, we employ a semantic segmentation network [30] to predict sky masks and encourage the accumulated density αi to converge to 0 in the sky region. The training of a single scene is conducted on a single NVIDIA V100 GPU. We initially optimize the scene at 2562 resolution and then refine it at 5122 resolution in the layout-aware refinement stage. Convergence for a scene covering a driving distance of ∼ 100m takes about 12 hours.

User Study. We further conduct a user study, wherein 20 volunteers are solicited to rate the rendered videos of each method across 3 dimensions, i.e., perceptual quality, realism, and 3D consistency. The score ranges from 1 to 5, with 5 denoting the best quality. The results in Fig. 8 indicate users’ preference for our results across all aspects.

Baselines and Settings. We compare the proposed method with several 3D generative methods applied to urban scenes, including EG3D [10], CC3D [2], UrbanGIRAFFE [70], Text2Room [24], SceneDreamer [14] and InfiniCity [38].

##### Large-Scale Generation Capability. Notably, the scales

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

Text2RoomSceneDreamer

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

EG3DUrbanGIRAFFECC3DOursInfiniCity

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

Scene 1 Scene 2

- Figure 6. Qualitative comparison. We display two scenes generated by different methods for comparison. Most of the results are directly borrowed from the original paper, except for Text2Room and SceneDreamer, which were adapted and re-trained within our settings. The proposed pipeline significantly outperforms previous baseline methods and achieves high-quality rendering. Best zoom in for more details.

Table 1. Quantitative comparison on the KITTI-360 dataset.

Method Text2Room [24] UrbanGIRAFFE [70] SceneDreamer [14] EG3D [10] CC3D [2] Ours FID↓ 134.3 118.6 122.39 109.3 79.1 59.8 KID↓ 0.116 0.143 0.113 0.121 0.082 0.059

Table 2. Quantitative comparison with CC3D+H.

with longer driving distances and higher quality by large a margin. Fig. 9 shows our generation results in two regions with 32GB GPU memory, each covering an area of ∼ 600 × 400m2 and spanning a driving distance of over 1000m. Even in such large-scale scenes, our method still achieves high-quality rendering results, showcasing the superior scalability of our pipeline in the arbitrary-scale urban generation.

Method CC3D [2] CC3D+H [2] Ours FID↓ 79.1 77.8 59.8 KID↓ 0.082 0.077 0.059

CC3D CC3D+H Ours

[Figure 103]

[Figure 104]

[Figure 105]

#### 4.2. Scene Editing

Instance-Level Editing. As shown in Fig. 10 (a) (b), we showcase two kinds of instance-level editing: (a) Object manipulation. We achieve object rotation, translation, repeat, and scaling by simply manipulating the object in the layout. (b) Stuff editing. We can further edit stuff elements in the scene (e.g., building removing, tree adding) by editing the layouts while keeping the other components unchanged. Style Editing. We present several style editing samples corresponding to the same generated scene in Fig. 10 (c). The results show that we can transfer the style of the generated scene via style text prompts (e.g., “Foggy”, “Vangogh paint”, etc), while preserving the overall structure. Notably, we can also generate urban scene of various city styles by simply adding text prompt-based conditions (e.g., “in Kyoto”, “in Hawaii”, etc).

[Figure 106]

[Figure 107]

[Figure 108]

- Figure 7. Qualitative comparison with CC3D+H. Our method significantly outperforms CC3D+H, which is a enhanced version of CC3D.

| |Perceptual Quality<br><br>Realism<br><br>3D Consistency| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

EG3D CC3D UrbanGIRAFFE InfiniCity Ours

- 1
- 2
- 3
- 4
- 5

- Figure 8. User study. The score ranges from 1 to 5, with 5 denoting the best quality.

Transfer to Other Scenes. Although the conditioning model is initially trained on the KITTI-360 dataset, we can transfer our method to generate scenes of other urban datasets by fine-tuning the ControlNet. As shown in Fig. 12, we generate scenes on the NuScenes [9] dataset and further transfer them into a different style (i.e., “Snowflakes drifting”).

#### 4.3. Ablation Studies

Effectiveness of LG-VSD. To verify the effectiveness of LG-VSD, we first conduct experiments by directly using the vanilla VSD, which does not yield reasonable results. By further applying our layout-constrained sampling strategy, we achieved more plausible results (shown in the left column of Fig. 13). Nonetheless, due to the inherent ambiguity of text prompts, VSD tends to converge to a trivial solution, resulting in unrealistic results that lack fine-grained details. To obtain realistic urban-like generation results, we further fine-tune the noise prediction network of Stable Diffusion on the KITTI-360 dataset to constrain the style distribution. However, in the absence of semantic and geo-

of scenes generated by most prior 3D-aware GAN-based methods are limited (typically less than 100m). SceneDreamer can generate scenes with larger scales, while the resolution of a local scene window is limited to 2048×2048 with 40GB GPU memory, which corresponds to a spatial coverage of ∼ 400 × 400m2 at a voxel resolution of 0.2m. Besides, the generated results also lack fine-grained details. In contrast, our method can achieve urban scene generation

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

2 3

|[Figure 115]<br><br>[Figure 116]|1|[Figure 117]|[Figure 118]<br><br>2|[Figure 119]|[Figure 120]<br><br>3|[Figure 121]|[Figure 122]<br><br>4|[Figure 123]<br><br>[Figure 124]|5|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

4

1

5

[Figure 125]

2 3

1

|[Figure 126]<br><br>[Figure 127]|1|[Figure 128]<br><br>[Figure 129]|2|[Figure 130]<br><br>[Figure 131]|3|[Figure 132]<br><br>[Figure 133]|4|[Figure 134]|[Figure 135]<br><br>5|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

- 4

- 5

1

- Figure 9. Large-scale generation capability. We generate two large-scale 3D urban scenes with high quality, each covering an area of ∼ 600 × 400m2 and spanning a driving distance of over 1000m.

Remove a building

Add a tree

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Original

Rotation

Translation + Repeat

Scaling “Foggy”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

“Night time” “Vangogh paint”

“Cartoon style” (a) Object Editing (b) Stuff Editing (c) Style Editing

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

- Figure 10. Samples of scene editing results. The proposed pipeline achieves instance-level editing (including object editing and stuff editing) and global style editing.

Table 3. Ablation studies on LG-VSD and layout-constrained sampling strategy on the KITTI-360 dataset.

ing the sampling space of our scalable hash grid using the 3D layout prior), we perform experiments by dropping this constraint (i.e., sampling in the full space). As shown in the left column of Fig. 14, the proposed pipeline can still produce plausible results without the constraint, while yielding much blurrier rendering results. Results in Table 3 further demonstrate the effectiveness of our layout-constrained sampling strategy.

Method FID↓ KID↓ w/o LG-VSD 167.1 0.203 w/o layout-constrained sampling 143.5 0.148 Ours 59.8 0.059

metric constraints from the 3D layout, VSD fails to capture the complex distribution of urban scenes, yielding unsatisfactory results (shown in the middle column of Fig. 13). Quantitative comparison in Table 3 also indicates that our LG-VSD exhibits much lower FID and KID than VSD.

Layout-Aware Refinement. We perform experiments to analyze the effectiveness of the layout-aware refinement strategy. As shown in the middle column of Fig. 14, aliasing artifacts are evident without the refinement, while the refined results are smoother and more realistic.

Layout-Constrained Sampling. To investigate the impact of the layout-constrained sampling strategy (i.e., constrain-

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

“in Kyoto” “in Hawaii” “in Venice” “in Dubai” “in San Francisco city”

- Figure 11. Transfer to other cities. The generated urban scene can be transferred to different city styles given the text prompt by finetuning the generated hash grid.

[Figure 166]

[Figure 167]

NuScenes style “Snowflakes drifting”

[Figure 168]

- Figure 12. Transfer to other urban styles. We generate scenes in NuScenes style and transfer them to a different weather.

Vanilla VSD VSD + Finetuned SD LG-VSD (Ours)

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

#### 4.4. Further Analysis

Large Camera View Shifting. As shown in Fig. 15, we display the rendering results by rotating the camera from −45deg to 45deg from the training trajectory. Owing to the high 3D consistency, the generated scene demonstrates strong robustness against large camera view shifting.

Figure 13. Effectiveness of LG-VSD. Without our 3D layout prior, VSD fails to generate high-quality urban scenes due to the absence of effective guidance, even with the fine-tuned diffusion model.

Diversity. To explore the diversity of generated results, we conduct experiments by employing different random seeds while maintaining the same layout. As shown in Fig. 16, given the same scene layout, the proposed pipeline can generate diverse scenes with elements (e.g., cars, buildings) that have different appearances and illumination.

in Fig. 17. The 3D triangle mesh reveals the consistent 3D structures of the generated scene.

### 5. Conclusion and Limitations

3D Visualization. To further explore the 3D consistency of the generated scene, we extract a triangle mesh from the generated hash grid representation and display the results

We have presented Urban Architect, a method for steerable 3D urban scene generation. The core of our method lies in

w/o layout constraint w/o refinement Ours

|[Figure 175]<br><br>1|
|---|

|[Figure 176]<br><br>2|
|---|

|[Figure 177]<br><br>3|
|---|

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

###### 3

2

- Figure 14. Ablations on layout-constrained sampling and layout-aware refinement strategies. The rendering results are blurry without the layout-constrained sampling strategy. The layout-aware refinement strategy further enhances the generation quality, leading to more realistic results.

[Figure 186]

[Figure 187]

[Figure 188]

- Figure 15. Large camera view shifting. The camera rotates from −45 deg to 45 deg from left to right.

Seed2Seed1

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

- Figure 16. Generated results with different random seeds. We display the generated scenes with two different random seeds given the same scene layout.

1

3D Mesh 3D Layout

Figure 17. 3D mesh visualization. We extract a 3D triangle mesh from the generated scene and provide rendered 2D images from corresponding camera poses.

is achieved by conditioning the score distillation sampling process with 3D layout information. Moreover, to address the unbounded nature of urban scenes, we design a scalable hash grid representation to adapt to arbitrary scene scales. Our framework facilitates high-quality, steerable 3D urban scene generation, capable of scaling up to generate largescale scenes covering a driving distance of over 1000m. Our flexible representation, coupled with the inherent capability of text-to-image diffusion models, empowers our pipeline to support diverse scene editing effects, including instance-level editing and style editing. Despite achieving high-quality and steerable generation, the current optimization process cannot satisfy pixel-level scene control. We hope our method can be a starting point for the urban-scale scene generation task, and leave addressing the above issues as future work.

### References

- [1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In International conference on machine learning, pages 40–49. PMLR, 2018. 3
- [2] Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Xingguang Yan, Gordon Wetzstein, Leonidas Guibas, and Andrea Tagliasacchi. Cc3d: Layout-conditioned generation of compositional 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7171–7181, 2023. 2, 3, 7, 9

incorporating a 3D layout representation as a robust prior to complement textual descriptions. Based on the 3D layout, we introduce Layout-Guided Variational Score Distillation (LG-VSD) to integrate the 3D geometric and semantic constraints into the current text-to-3D paradigm, which

- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 3
- [4] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864,

2021. 3

- [5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5470–5479, 2022. 5
- [6] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19697–19705, 2023. 3, 5
- [7] Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, et al. Gaudi: A neural architect for immersive 3d scene generation. Advances in Neural Information Processing Systems, 35:25102–25116, 2022. 3
- [8] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 7
- [9] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 9
- [10] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123–16133, 2022. 3, 7, 9
- [11] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, pages 333–350. Springer,

2022. 3

- [12] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22246–22256, 2023. 3
- [13] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 3
- [14] Z. Chen, G. Wang, and Z. Liu. Scenedreamer: Unbounded 3d scene generation from 2d image collections. IEEE trans-

- actions on pattern analysis and machine intelligence, 45(12): 15562–15576, 2023. 3, 6, 7, 9
- [15] Dana Cohen-Bar, Elad Richardson, Gal Metzer, Raja Giryes, and Daniel Cohen-Or. Set-the-scene: Global-local training for generating controllable nerf scenes. arXiv preprint arXiv:2303.13450, 2023. 3
- [16] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. Gram: Generative radiance manifolds for 3d-aware image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10673– 10683, 2022. 3
- [17] Terrance DeVries, Miguel Angel Bautista, Nitish Srivastava, Graham W Taylor, and Joshua M Susskind. Unconstrained scene generation with locally conditioned radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14304–14313, 2021. 3
- [18] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. arXiv preprint arXiv:2302.01133, 2023. 3
- [19] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5501–5510, 2022. 3
- [20] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances In Neural Information Processing Systems, 35:31841–31854, 2022. 3
- [21] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 3
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [24] Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 7909–7920, 2023. 3, 7, 9
- [25] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: zero-shot textdriven generation and animation of 3d avatars. ACM Transactions on Graphics (TOG), 41(4):1–19, 2022. 3
- [26] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 4
- [27] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19774–19783, 2023. 3
- [28] Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, ZhengJun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422, 2023. 3
- [29] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 867–876, 2022. 3
- [30] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2989–2998, 2023. 7
- [31] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14371–14382, 2023. 3
- [32] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG), 42(4):1–14, 2023. 3
- [33] Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, and Sanja Fidler. Neuralfield-ldm: Scene generation with hierarchical latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8496–8506, 2023. 3
- [34] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [35] Vladimir Kulikov, Shahar Yadin, Matan Kleiner, and Tomer Michaeli. Sinddm: A single image denoising diffusion model. In International Conference on Machine Learning, pages 17920–17930. PMLR, 2023. 6
- [36] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3292–3310, 2022. 6, 7
- [37] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 2, 3
- [38] Chieh Hubert Lin, Hsin-Ying Lee, Willi Menapace, Menglei Chai, Aliaksandr Siarohin, Ming-Hsuan Yang, and Sergey Tulyakov. Infinicity: Infinite-scale city synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22808–22818, 2023. 2, 3, 7
- [39] Yiqi Lin, Haotian Bai, Sijia Li, Haonan Lu, Xiaodong Lin, Hui Xiong, and Lin Wang. Componerf: Text-guided multiobject compositional nerf with editable 3d scene layout. arXiv preprint arXiv:2303.13843, 2023. 3

- [40] Jeffrey Yunfan Liu, Yun Chen, Ze Yang, Jingkang Wang, Sivabalan Manivasagam, and Raquel Urtasun. Real-time neural rasterization for large scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8416–8427, 2023. 2
- [41] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 3
- [42] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [43] Fan Lu, Yan Xu, Guang Chen, Hongsheng Li, Kwan-Yee Lin, and Changjun Jiang. Urban radiance field representation with deformable neural mesh primitives. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 465–476, 2023. 2, 3
- [44] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2021. 3, 5
- [45] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12663–12673, 2023. 3
- [46] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3
- [47] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa. Clip-mesh: Generating textured meshes from text using pretrained image-text models. In SIGGRAPH Asia 2022 conference papers, pages 1–8, 2022. 3
- [48] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15, 2022. 3, 5
- [49] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2337–2346,

2019. 7

- [50] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 6
- [51] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations, 2023. 2, 3
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning

- transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 5
- [53] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 7
- [54] Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. Urban radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12932–12942, 2022. 3, 6
- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 6
- [56] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 3
- [57] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18603–18613,

2022. 3

- [58] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems, 33:20154–20166, 2020. 3
- [59] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087–6101,

2021. 2

- [60] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5459– 5469, 2022. 3
- [61] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8248–8258, 2022. 3
- [62] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 3

- [63] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj,

- and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 6
- [64] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4563–4573, 2023. 3
- [65] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 3
- [66] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in neural information processing systems, 29, 2016. 3
- [67] Yuanbo Xiangli, Linning Xu, Xingang Pan, Nanxuan Zhao, Anyi Rao, Christian Theobalt, Bo Dai, and Dahua Lin. Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In European conference on computer vision, pages 106–122. Springer, 2022. 3
- [68] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Citydreamer: Compositional generative model of unbounded 3d cities. arXiv preprint arXiv:2309.00610, 2023. 2, 3
- [69] Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, and Dahua Lin. Grid-guided neural radiance fields for large urban scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8296–8306, 2023. 3
- [70] Yuanbo Yang, Yifei Yang, Hanlei Guo, Rong Xiong, Yue Wang, and Yiyi Liao. Urbangiraffe: Representing urban scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9199–9210, 2023. 2, 3, 7, 9
- [71] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. arXiv preprint arXiv:2210.06978, 2022. 3
- [72] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3836–3847, 2023. 2, 3, 5, 6

