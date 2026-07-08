## Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video

Hongchi Xia1,2 Zhi-Hao Lin1 Wei-Chiu Ma3 Shenlong Wang1 1University of Illinois Urbana-Champaign 2Shanghai Jiao Tong University 3Cornell University https://video2game.github.io/

[Figure 1]

[Figure 2]

# arXiv:2404.09833v1[cs.CV]15Apr2024

[Figure 3]

[Figure 4]

collect coins

View 1 View 2

run

[Figure 5]

[Figure 6]

stand

break objects

[Figure 7]

[Figure 8]

Vase drops

[Figure 9]

[Figure 10]

drive

Throw ball at vase

bump into a car

Figure 1. Video2Game takes an input video of an arbitrary scene and automatically transforms it into a real-time, interactive, realistic and browser-compatible environment. The users can freely explore the environment and interact with the objects in the scene.

#### Abstract

such as video games, virtual reality applications, and selfdriving simulators. This process, however, is complex and expensive. It demands the skills of experts in the field and the use of professional software development tools [27, 30]. For instance, Grand Theft Auto V [29], known for its intricately detailed environment, was one of the most expensive video games ever developed, with a budget over $265 million primarily for asset creation. Similarly, the development of the CARLA autonomous driving simulator [23] involves a multidisciplinary team of 3D artists, programmers, and engineers to meticulously craft and texture the virtual cityscapes, creating its lifelike environments.

Creating high-quality and interactive virtual environments, such as games and simulators, often involves complex and costly manual modeling processes. In this paper, we present Video2Game, a novel approach that automatically converts videos of real-world scenes into realistic and interactive game environments. At the heart of our system are three core components: (i) a neural radiance fields (NeRF) module that effectively captures the geometry and visual appearance of the scene; (ii) a mesh module that distills the knowledge from NeRF for faster rendering; and (iii) a physics module that models the interactions and physical dynamics among the objects. By following the carefully designed pipeline, one can construct an interactable and actionable digital replica of the real world. We benchmark our system on both indoor and large-scale outdoor scenes. We show that we can not only produce highly-realistic renderings in real-time, but also build interactive games on top.

An appealing alternative to extensive manual modelling is creating environments directly from the real world. For instance, photogrammetry, a technique for constructing digital replicas of objects or scenes from overlapping realworld photographs, has been utilized for environment creation [59, 60]. Success stories also span various games and simulators. However, most use cases are limited to creating object assets and necessitate significant postprocessing, such as material creation, texturing, and geometry fixes [75]. People thus turns to neural radiance fields (NeRFs) [52], as it offers a more promising approach to

#### 1. Introduction

Crafting a visually compelling and interactive environment is crucial for immersive experiences in various domains,

[Figure 11]

| | |
|---|---|
| | |
| | |

[Figure 12]

View

| | |
|---|---|
| | |
| | |

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

|[Figure 17]<br><br>[Figure 18]|
|---|
| |
|[Figure 19]<br><br>[Figure 20]|
|[Figure 21]<br><br>[Figure 22]|
| |
|[Figure 23]<br><br>[Figure 24]|
| |
| |
|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|
|[Figure 29]|
| |
|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]|
| |
| |
|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|

|[Figure 36]<br><br>[Figure 37]<br><br>|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

| |
|---|

MLP Shader

Feature texture

MLP Shader

[Figure 44]

Feature texture

[Figure 45]

[Figure 46]

| | |
|---|---|
| | |

[Figure 47]

[Figure 48]

| | |
|---|---|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Single video Textured and collisionenabled Mesh

Large-scale NeRF

Interactive environment

Game Interactive Environment

Input Video Base NeRF Rigid-body Dynamics Neural Textured Mesh

Figure 2. Overview of Video2Game: Given multiple posed images from a single video as input, we first construct a large-scale NeRF model that is realistic and possesses high-quality surface geometry. We then transform this NeRF model into a mesh representation with corresponding rigid-body dynamics to enable interactions. We utilize UV-mapped neural texture, which is both expressive and compatible with game engines. Finally, we obtain an interactive virtual environment that virtual actors can interact with, can respond to user control, and deliver high-resolution rendering from novel camera perspectives – all in real-time.

modeling large scenes. With careful design [17, 28, 45, 54, 69], NeRF is able to render free-viewpoint, photo-realistic images efficiently. However, crafting an interactive environment entails more than just creating a visually high-fidelity digital twin; it also involves building a physically plausible, immersive, real-time and importantly, interactive world tailored to user experiences. Furthermore, we expect such a virtual world to be compatible with real-time interaction interfaces such as common game engines. Despite its promise, the use of NeRF to create interactive environments from real-world videos remains largely unexplored.

In this paper, we introduce Video2Game, a novel approach to automatically converting a video of a scene into a realistic and interactive virtual environment. Given a video as input, we first construct a NeRF that can effectively capture the geometric and visual information of a (large-scale, unbounded) scene. Then we distill the NeRF into a game engine-compatible, neural textured mesh. This significantly improves the rendering efficiency while maintains the overall quality. To model the interactions among the objects, we further decompose the scene into individual actionable entities and equip them with respective physics model. Finally, we import our automatically generated assets into a WebGL-based game engine and create a playable game. The resulting virtual environment is photo-realistic, interactive, and runs in real-time. See Fig. 1 for demonstration. In summary, our key contributions are:

- • A novel 3D modeling algorithm for real-time, freeviewpoint rendering and physical simulation, surpassing state-of-the-art NeRF baking methods with added rigidbody physics for enhanced simulation.
- • An automated game-creation framework to transform a scene video into an interactive, realistic environment, compatible with current game engines.

#### 2. Related Works

Given a single video, we aim to create a real-time, interactive game where the agents (e.g., the character, the car) can navigate and explore the reconstructed digital world, interact with objects in the scene (e.g., collision and manipulate objects), and achieve their respective tasks (e.g., collecting coins, shooting targets). We draw inspirations from multiple areas and combine the best of all. In this section, we will briefly review those closely related areas which forms the foundation of our work.

Novel view synthesis (NVS): Our work builds upon the success of novel view synthesis [18, 31, 41, 71], which is crucial for our game since it enables the agents to move freely and view the reconstructed world seamlessly from various perspectives. Among all these approaches [32, 68, 77, 95, 96], we exploit neural radiance field (NeRF) [52] as our underlying representation. NeRF has emerged as one of the most promising tools in NVS since its introduction [56–58], and has great performance across a wide range of scenarios [42, 63, 84, 91]. For instance, it can be easily extended to handle various challenging real-world scenarios such as learning from noisy camera poses [44, 79], reflectance modeling for photo-realistic relighting [78, 93], and real-time rendering [20, 45, 62, 74, 85]. In this work, we combine recent advances in NeRF with physics modeling to build an immersive digital world from one single video, moving from passive NVS to our complete solution for embodied world modeling where agents can actively explore and interact with the scene.

Controllable video generation: Using different control signals to manipulate the output of a visual model has garnered great interest in the community. This has had a profound impact on content creation [65, 66], digital editing [15, 40], and simulation [36, 37, 46]. One could also lever-

age large foundation models to control video content using text [65, 66]. However, they lack fine-grained and real-time control over the generated content. Alternatively, training (conditional) generative models for each scene enables better disentanglement of dynamics (e.g., foreground vs. background) and supports better control. For instance, one can represent a self-driving scene [37] or a Pacman game [36] as latent codes and generate video frames based on control inputs with a neural network. One can also learn to control the players within tennis games [50, 51, 89, 90]. Our work falls under the second line of research, where the model takes user control signals (e.g., a keystroke from the keyboard) as input and responds by rendering a new scene. However, instead of focusing on a specific scene (e.g., tennis games), we have developed a pipeline that allows the creation of a playable environment from a single video of a generic scene. Additionally, we model everything in 3D, which enables us to effectively capture not only view-dependent appearance but also physical interactions among rigid-body equipped objects. Importantly, we adopt a neural representation compatible with graphics engines, enabling users to play the entire game in their browser at an interactive rate.

Data-driven simulation: Building a realistic simulation environment has been a longstanding challenge. [23, 35, 76, 80]. While it’s promising, we come close to mirror the real world only in recent years [14, 19, 48, 49, 67, 83, 84]. The key insight of these work is to build models by leveraging real-world data. Our work closely aligns with this line of research on building high-fidelity simulators from real-world data, with a few key differences. First, existing works mainly focus on offline training and evaluation [14, 49, 83, 84], whereas our system runs at an interactive rate and allows for online, real-time control. Second, some existing works[47, 49, 81, 97] need additional data modality like LiDAR point clouds for geometry reconstruction, but RGB video is all we need. Third, most photo-realistic simulators don’t model physical interactions. However, we supports various physics modeling and allows agents to interact with the environment. Last, existing simulators are typically resource-intensive , while our system is lightweight and can be easily accessible in common engines.

#### 3. Video2Game

Given a sequence of images or a video of a scene, our goal is to construct an interactable and actionable digital twin, upon which we can build real-time, interactive games or realistic (sensor) simulators. Based on the observations that prevalent approaches to constructing digital replica mainly focus on visual appearance and ignore the underlying physical interactions, we carefully design our system such that it can not only produce high-quality rendering across viewpoints, but also support the modeling of physical actions (e.g., navigation, collision, manipulation, etc). At the heart

[Figure 54]

[Figure 55]

[Figure 56]

Figure 3. Visualization of automatically computed collision geometry: Sphere collider (green), box collider (yellow), convex polygon collider (purple) and trimesh collider (red).

of our systems is a compositional implicit-explicit 3D representation that is effective and efficient for both sensor and physics simulation. By decomposing the world into individual entities, we can better model and manipulate their physical properties (e.g., specularity, mass, friction), and simulate the outcomes of interactions more effectively.

We start by introducing a NeRF model that can effectively capture the geometric and visual information of a large-scale, unbounded scene (Sec. 3.1). Next, we present an approach to convert the NeRF into a game-engine compatible mesh with neural texture maps, significantly improving the rendering efficiency while maintaining the quality (Sec. 3.2). To enable physical interactions, we further decompose the scene into individual actionable entities and equip them with respective physics models (Sec. 3.3). Finally, we describe how we integrate our interactive environment into a WebGL-based game engine, allowing users to play and interact with the virtual world in real time on their personal browser. Fig. 2 provides an overview of our proposed framework.

##### 3.1. Large-scale NeRF

Preliminaries: Instant-NGP [54] is a notable variant of NeRF, which represents the radiance field with a combination of spatial hash-based voxels and neural networks: c,σ = Fθ(x,d;Φ) = MLPθ(It(x,Φ),d). Given a 3D point x ∈ R3 and a camera direction d ∈ R2 as input, Instant-NGP first interpolate the point feature It(x,Φ) from the adjacent voxel features Φ. Then the point feature and the camera direction are fed into a light-weight multilayer perception (MLP) to predict the color c ∈ R3 and density σ ∈ R+. To render the scene appearance, we first cast a ray r(t) = o+td from the camera center o through the pixel center in direction d, and sample a set of 3D points {xi} along the ray. We then query their respective color {ci} and density {σi} and obtain the color of the pixel through alpha-composition: CNeRF(r) = i wici. Similarly, the

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

RGBDepthNormal

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Ground Truth Instant-NGP Nerfacto Ours

Figure 4. Qualitative comparisons among NeRF models. The rendering quality of our base NeRF is superior to baselines, and with leveraging monocular cues, we substantially improve rendered geometry compared to other baselines. This significantly facilitates NeRF baking in subsequent stages. Here we consider depths measured by LiDAR point cloud in KITTI-360 and compute normals based on it.

expected depth can be computed by: DNeRF(r) = i witi. Here, wi indicates the blending weight that is derived from the densities {σi}. We refer the readers to [52] for more details. To learn the voxel features Φ and the MLP weights θ, we compute the difference between the ground truth color and the rendered color: Lrgb = r∥CGT(r) − CNeRF(r)∥22. Large-scale NeRF: While Instant-NGP [54] has shown promising results on densely observed and bounded scenes, its performance starts to degrade when extending to sparsely-captured, large-scale, unbounded environments. To mitigate these issues, we propose several enhancements:

###### c,σ,s,n = Fθ(x,d;Φ) = MLPθ(It(Ct(x),Φ),d). (1)

First of all, we exploit the contraction function Ct(x) [16] to map the unbounded coordinates into a bounded region. In addition to radiance and density, we predict the semantics s and the surface normal n of the 3D points, guided with

- 2D priors to better regularize the scene geometry. Furthermore, we divide large-scale scenes into several blocks [72] to capture the fine-grained details. We now describe these enhancements in more details. Depth: High-quality geometry is critical for modeling physical interactions. Inspired by MonoSDF [88], we leverage off-the-shelf monocular depth estimators [26, 33] to guide and improve the underlying geometry. We first predict the depth of the scene from rendered RGB images. Then we minimize the discrepancy between the rendered depth and the predicted depth via Ldepth =

r ∥(aDNeRF(r) + b) − Dmono(r)∥22, where a and b are the scale and shift that aligns the two distribution [61].

Surface normals: Similar to depth, we encourage the normal estimated from NeRF to be consistent with the normal predicted by the off-the-shelf estimator [26, 33]. The normal of a 3D point xi can be either analytically derived from the estimated density ni = (1 − ∇

xσi

∥∇σi∥), or predicted by the MLP header as in Eq. 1. We could aggregate them via volume render: N(r) = i wini. Em-

pirically we find that adopting both normals and promoting their mutual consistency works the best, since the MLP header offers more flexibility. We thus employ Lnormal = ∥Nmlp(r) − Nmono(r)∥22 + ∥Nmlp(r) − Ndensity(r)∥22.

Semantics: We also predict semantic logits for each sampled 3D points with our MLP. This helps us capture the correlation between semantics and geometry [42, 94]. We render the semantic map with volume rendering SNeRF(r) =

i wisi and compute the cross-entropy with that of a 2D segmentation model Lsemantics = CE(Smono,SNeRF).

Regularization: We additionally adopt two regularization terms. To reduce floaters in the scene, for each randomly sampled 3D point x, we penalize its density by Lsp = 1 − exp(−ασ(x)), where α > 0 [86]. For each sky pixel (which we derived from the semantic MLP), we encourage its depth DNeRF(rsky) to be as far as possible. The loss is defined as: Lsky = rsky exp(−DNeRF(rsky)).

Blocking: Capitalizing on a single Instant-NGP to cover an extraordinarily large scene such as KITTI-360 [43] would often lead to inferior results. We thus adopt a strategy akin to BlockNeRF [72] where we divided the whole scene into numerous blocks and model each region with a separate Instant-NGP. Adjacent regions maintain substantial overlaps to ensure smooth transition.

Learning: We jointly optimize the voxel feature Φ and the MLP weights θ by minimizing the following loss:

LNeRFtotal = Lrgb+Lnormal+Lsemantics+Ldepth+Lsky+Lsp (2)

##### 3.2. NeRF Baking

We aim to create a digital replica that users (or agents) can freely explore and act upon in real time. Although our large-scale NeRF effectively renders high-quality images and geometry, its efficiency is limited by the computational costs associated with sampling 3D points. The underlying volume density representation further complicates

KITTI-360 Gardenvase Interactive Compatibility

Method Representation

PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ Real time Rigid-body physics Scene decomposition InstantNGP [54]

27.46 0.853 0.165 25.90 0.757 0.191 ✗ ✗ ✗ Nerfacto [73] 23.20 0.763 0.238 22.16 0.517 0.283 ✗ ✗ ✗ Video2Game 27.62 0.871 0.131 26.57 0.815 0.143 ✗ ✗ ✗

Volume

Gauss. Spl. [34] Points 17.85 0.615 0.428 27.50 0.858 0.099 ✓ ✗ ✗ MobileNeRF [20]

19.67 0.627 0.452 22.80 0.505 0.365 ✓ ✗ ✗ BakedSDF* [85] 22.37 0.757 0.302 22.68 0.514 0.369 ✓ ✓ ✗ Video2Game 23.35 0.765 0.246 22.81 0.508 0.363 ✓ ✓ ✓

Mesh

- Table 1. Quantitative results on novel view synthesis and interactive compatibility analysis. Video2Game produces better or comparable results across scenes, suggesting the effectiveness of our NeRF and mesh model. The performance improves the most when tackling unbounded, large-scale scenes in KITTI-360. We note that existing NeRFs cannot reach the interactive rate required for real-time games. While point-based rendering significantly improves the speed, it does not support rigid body physics. BakedSDF [85] represents the whole scene with one single mesh, thus does not support object-level interactions.

the problem. For instance, it’s unclear how to define physical interaction with such a representation (e.g., defining collision). The representation is also not compatible with common graphics engines. While recent software, such as the NeRFStudio Blender plugin and LumaAI Unreal addon, has made some strides, their interaction capabilities and scene geometry quality are still not optimal for real-time user engagement, especially when the scene is large and the observations are relatively sparse. To overcome these challenges, we draw inspiration from recent NeRF meshing advancements and present a novel NeRF baking framework that efficiently transforms our NeRF representation into a game-engine compatible mesh. As we will show in Sec. 4, this conversion greatly enhances rendering efficiency while preserving quality and facilitates physical interactions.

Mesh representation: Our mesh M = (V,F,T) comprises vertices V ∈ R|V |×3, faces F ∈ N|F|×3 and a UV neural texture map T ∈ RH×W×6. Following [74], we store the base color in the first three dimension of T, and encode the specular feature in the rest. The initial mesh topology are obtained by marching cubes in the NeRF density field. We further prune the invisible faces. conduct mesh decimation and edge length regularization. The UV coordinate of each vertex is calculated via xatlas [11].

Rendering: We leverage differentiable renderers [39] to render our mesh into RGB images CR and depth maps DR. Specifically, we first rasterize the mesh into screen space and obtain the UV coordinate for each pixel i. Then we sample the corresponding texture feature Ti = [Bi;Si] and feed it into our customized shader. Finally, the shader computes the sum of the view-independent base color Bi and the view-dependent MLP MLPshaderθ (Si,di):

CR = Bi + MLPshaderθ (Si,di). (3) The MLP is lightweight and can be baked in GLSL.

Learning: We train the shader MLP MLPshaderθ and the neural texture map T by minimizing the color difference

between the mesh and the ground truth, and the geometry difference between the mesh and the NeRF model:

LmeshT,θ =

r

∥CR(r) − CGT(r)∥ + ∥DR(r) − DNeRF(r)∥. (4)

Anti-aliasing: Common differentiable rasterizers only take the center of each pixel into account. This may lead to aliasing in the learned texture map. To resolve this issue, we randomly perturb the optical center of the camera by 0.5 pixels along each axis at every training step. This ensure all the regions within a pixel get rasterized.

Relationship to existing work: Our approach is closely related to recent work on NeRF meshing [20, 62, 74, 85], but there exist key differences. While MobileNeRF [20] also adopts an explicit mesh with neural textures, they mainly capitalize on planar primitives. The quality of the reconstructed mesh is thus inferior. BakedSDF [85] offers excellent runtime and rendering quality, but their vertex coloring approach has limited resolution for large scenes. NeRF2Mesh [74] lacks depth distillation and doesn’t adopt contraction space for unbounded scenes. They also have a sophisticated multi-stage training and multi-resolution refinement process. Finally, MeRF [62], though efficient, still relies on volume-rendering.

##### 3.3. Representation for Physical Interaction

Our mesh model facilitates efficient novel-view rendering in real time and allows for basic rigid-body physical interactions. For example, the explicit mesh structure permits an agent to “stand” on the ground. Nevertheless, beyond navigation, an agent should be capable of performing various actions including collision and manipulation. Furthermore, a scene comprises not only the background but also interactable foreground objects, each possessing unique physical properties. For instance, a street-bound car is much heavier than a flower vase. When struck by another object, a car may barely move but the vase may fall and shatter.

To enhance physical interaction realism, we decompose the scene into discrete, actionable entities, each endowed with specific physical characteristics (e.g., mass, friction). This approach, in conjunction with rigid-body physics, allows for the effective simulation that adheres to physical laws.

Scene decomposition: Directly editing and decomposing a mesh is extremely difficult due to topology change. Fortunately, neural fields are inherently compositional in

- 3D. By identifying the objects each spatial region belongs to, we can use neural fields to guide the decomposition of

the mesh. Specifically, we sample a 3D point xi within each voxel i and determine its semantic category either through the predicted semantic logits si or by verifying whether the point is within a specified bounding box. The process is repeated for all voxels to segment the entire scene. Then, for each object, we perform NeRF meshing individually, setting the density of the remaining areas to zero. The intersections between objects are automatically resolved by marching cube. Finally, we initialize the neural texture of these new, individual meshes from the original mesh model. For newly created faces, we employ nearest neighbor inpainting on the neural texture map, which empirically yields satisfactory results. Fig. 1 shows an example where a vase is separated from a table. The middle of the table is original occluded yet we are able to maintain high-quality rendering.

Physical parameters reasoning: The next step is to equip decomposed individual meshes with various physicsrelated attributes so that we can effectively model and simulate their physical dynamics. In this work, we focus on rigid body physics, where each entity i is represented by a collision geometry coli, mass mi, and friction parameters fi. We support fours types of collision geometry with different levels of complexity and efficiency: box, sphere, convex polygon, and triangle mesh. Depending on the object and the task of interest, one can select the most suitable collision check for them. For other physical parameters (e.g. mass, friction), one can either set them manually or query large language models (LLMs) for an estimation.

Physical interactions: Rigid body dynamics, while simple, can support a variety of interactions. With the collision check, an user/agent can easily navigate through the environment while respecting the geometry of the scene. The agents will no longer be stuck in a road or cut through a wall. It also allows the agent to interact with the objects in the scene. For instance, one can push the objects towards the location of interest. The object movement will be determined by its mass and other physical properties such as the friction. We can also manipulate the objects by adopting a magnet grasper, following AI2-Thor [38]. This opens the avenue towards automatic creation of realistic, interactive virtual environment for robot learning.

##### 3.4. Interactive Environment

We deploy our interactive environment within a real-time, browser-based game engine. We manage the underlying logic and assets using Sketchbook [3], a Game Engine based on Three.js that leverages WebGL [4] for rendering. This combination ensures high efficiency while offering the flexibility and sophistication required for intricate rendering tasks. It also allows us to easily integrate content from different scenes together. We have further extended Sketchbook’s capabilities by implementing a GLSL-based shader [2]. This enables real-time computation of our MLP-based specular shader during deployment. For physics simulation, we use Cannon.js [1], which assures realism and efficiency in the motion within our interactive environment. It supports not only rigid body dynamics but also more sophisticated modeling techniques. For example, we can precompute the fracturing effect for dynamic objects. Upon experiencing a significant force, these objects are realistically simulated by the real-time physics engine, which handles the interactions between the fractured pieces and the rest of the scene, such as their falling and settling on the ground. Besides browser-based engine, the virtual environments from Video2Game pipeline could be also integrated into both Blender [21] and Unreal engines [27] (see Fig. 6).

#### 4. Experiments

We begin by presenting our experimental setup, followed by a comparison of our model with state-of-the-art approaches. Next, we conduct an extensive analysis of our model’s distinctive features and design choices. Then we demonstrate how we constructed a web browser-compatible game capable of delivering a smooth interactive experience exceeding 100 frames per second (FPS), all derived from a single video source. Finally, we showcase the capabilities of our model in robot simulation through two demonstrations.

##### 4.1. Setup

Dataset: We evaluate the effectiveness of Video2Game across three distinct scenes in various scenarios, including “Gardenvase” [16], an outdoor object-centric scene; the KITTI-360 dataset [43], a large-scale self-driving scene with a sequence that forms a closed loop, suitable for carracing and Temple Run-like games; and finally, an indoor scene from the VR-NeRF [82] dataset, showcasing the potential for robot simulations.

Metrics: To evaluate the quality of the rendered images, we adopt PSNR, SSIM, and LPIPS [92]. For geometry reconstruction, we evaluate with LiDAR point cloud in KITTI-360 dataset. Root mean square deviation (RMSE), mean absolute error (MAE), and outlier rate are applied to measure the disparity existing in estimated geometry.

Our model: For NeRF, we adopt hashgrid encoding [53] and two-layer MLP for each header. For textured mesh,

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

G.T. MobileNeRF [20] BakedSDF [85] Ours

- Figure 5. Qualitative comparisons among mesh models. We compare our mesh rendering method with others in Garden scene [16].

[Figure 77]

[Figure 78]

Blender Unreal

- Figure 6. Video2Game in Blender and Unreal Engine.

hashing encoding. Nerfacto [73] extends the classic NeRF with learnable volumetric sampling and appearance embedding. 3D Gaussian Splatting [34] leverages 3D Gaussians and achieves fast training and rendering. MobileNeRF [20] adopts a hybrid NeRF-mesh representation. It can be baked into a texture map and enable real-time rendering. BakedSDF [85] adopts a volume-surface scene representation. It models view-dependent appearance efficiently by baking spherical Gaussians into the mesh.

Method Outlier-%↓ RMSE↓ MAE↓

Instant-NGP [54] 22.89 4.300 1.577 Nerfacto [73] 50.95 8.007 2.863 Gauss. Spl. [34] 91.08 11.768 8.797 BakedSDF* (offline) [85] 43.78 5.936 2.509 Video2Game (Our NeRF) 13.23 3.028 1.041

- Table 2. Quantitative evaluation on NeRF geometry. Our NeRF renders significantly more accurate depth compared with the baselines. The unit is meter and the outlier threshold is 1.5 meters.

Volume Rendering Mesh Rastization PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Method

Vanilla NGP 27.46 0.853 0.165 22.54 0.716 0.350 + Regularization terms 27.52 0.861 0.157 22.97 0.732 0.303 + Monocular cues 27.62 0.871 0.131 23.35 0.765 0.246

Table 3. Ablation studies on KITTI-360.

we conduct marching cubes on the NeRF and post-process it to a fixed precision. We set the texture image size to 4096x4096. For GLSL shader, we design a light-weight two-layer MLP, which enables efficient real-time rendering. For KITTI-360 (see Sec. 3.1), we divide the whole scene into 16 blocks and create a skydome mesh for the sky.

Baselines: To evaluate the visual and geometry quality of our model, we compare against SOTA approaches in neural rendering and neural reconstruction. Instant-NGP [54] is a NeRF-based method that exploits multi-resolution

##### 4.2. Experimental results

Novel view synthesis: Tab. 1 shows the rendering performance and interactive compatibility of our model against the baselines on KITTI-360 [43] and Gardenvase [16]. Our NeRF achieves superior performance when compared to state-of-the-art neural volume render approaches across different scenes. Though [34] performs best in Gardenvase, it fails to handle the sparse camera settings in KITTI-360, where it learns bad 3D orientations of Gaussians. Our baked mesh outperforms other mesh rendering baselines significantly in KITTI-360 and performs similarly in Gardenvase as shown in Fig. 5. Additionally, our pipeline has the highest interactive compatibility among all baselines.

Geometry reconstruction: Our model performs significantly better than the baseline regarding geometry accuracy (see Tab. 2). We provide some qualitative results in Fig. 4, demonstrating that our model can generate high-quality depth maps and surface normals, whereas those produced by the baselines contain more noise.

Ablation study: To understand the contribution of each component in our model, we begin with the basic InstantNGP [54] and sequentially reintroduce other components. The results in Tab. 3 show that our regularization and monocular cues improve the quality of both volume rendering in NeRF and mesh rasterization. Additionally, we do observe a decline in rendering performance when convert-

ing NeRF into game engine-compatible meshes.

##### 4.3. Video2Game

We have shown our approach’s effectiveness in rendering quality and reconstruction accuracy across various setups. Next, we demonstrate the construction of a web browsercompatible game enabling player control and interaction with the environment at over 100 FPS.

Data preparation: We build our environments based on videos in Gardenvase [16], KITTI-360 [43] and VR-NeRF [82] mentioned in Sec. 4.1, using our proposed approach. The outcomes include executable environments with mesh geometry, materials, and rigid-body physics, all encoded in GLB and texture files.

Game engine: We build our game based on several key components in our game engine mentioned in Sec. 3.4. By leveraging them, our game generates a highly realistic visual rendering as well as physical interactions (see Fig. 1) and runs smoothly at an interactive rate across various platforms, browsers, and hardware setups (see Tab. 4). As for other game engines (see Fig. 6), in Blender [21] we showcase the compatibility of our exported assets with other game engines. For Unreal [27], we further demonstrate a real-time game demo where a humanoid robot can freely interact within the Gardenvase scene, such as standing on the table and kicking off the central vase. These prove the compatibility of our proposed pipeline.

Interactive game features: Movement in games: Agents can navigate the area freely within the virtual environment where their actions follow real-world physics and are constrained by collision models. Shooting game: For realistic shooting physics, we calculated the rigid-body collision dynamics for both the central vase and the surrounding scene (see Fig. 3), separated using mesh semantic filtering. We used a box collider for the vase and convex polygon colliders for the background. The player shoots footballs with a sphere collider at the vase on the table, causing it to fly off and fall to the ground (see Fig. 1). Temple-Run like game: The agent collects coins while running in the KITTI Loop composed of four streets in KITTI-360. Obstructive chairs on the road can be smashed thanks to pre-computed fracture animations. The agent can also drive and push roadside vehicles existing in the scene forward by crashing into them. This interactivity is achieved through rigid-body dynamics simulation and collision modeling.

Robot simulation: We demonstrate the potential of leveraging Video2Game for robot simulation using the VRNeRF dataset. We reconstruct the scene and segment simulatable rigid-body objects (e.g., the fruit bowl on the table). We show two demos in Fig. 7: a Stretch Robot pushing the bowl off the table and a Fetch Robot performing pick-and-place actions. We employ PyBullet [22] to simulate the underlying physics with the help of corresponding collision mod-

Platform FPS (hz) CPU-Usage (%) GPU-Usage (%)

Mac M1 Pro Mac OS, Chrome 102 34 70 Intel Core i9 + NV 4060 Windows, Edge 240 6 74 AMD 5950 + NV 3090 Linux, Chrome 144 20 40

Table 4. Runtime Analysis. Our interactive environment can run in real-time across various hardware setup and various platforms. User actions may slightly vary, which could lead to minor variations in runtime.

[Figure 79]

[Figure 80]

push fall

[Figure 81]

[Figure 82]

[Figure 83]

pick place

[Figure 84]

Figure 7. Robot simulation in VRNeRF [82] dataset. We demonstrate the possibility of conducting robot learning in our virtual environments using Stretch Robot [8] and Fetch Robot [5].

els. Since real-time grasping simulation is challenging, following existing robot simulation frameworks [12, 13, 38], objects near the Fetch gripper are automatically picked up. This demonstrates our model’s ability to convert a real-time video stream into a virtual environment, allowing robots to rehearse before acting in the real environment.

#### 5. Conclusion

We present a novel approach to converting real-world video footage into playable, real-time, and interactive game environments. Specifically, we combine the potential of NeRF modeling with physics modeling and integrate them into modern game engines. Our approach enables any individual to transform their surroundings into an interactive digital environment, unlocking exciting possibilities for 3D content creation, with promising implications for future advancements in digital game design and robot simulation.

Acknowledgements This project is supported by the Intel AI SRS gift, the IBM IIDAI Grant, the InsperIllinois Innovation Grant, the NCSA Faculty Fellowship, and NSF Awards #2331878, #2340254, and #2312102. We greatly appreciate the NCSA for providing computing resources. We thank Derek Hoiem and Albert Zhai for helpful discussions. We thank Jingkang Wang, Joao Marques and Hanxiao Jiang for proofreading.

#### References

- [1] Cannon.js. https : / / schteppe . github . io / cannon.js/. 6
- [2] GLSL. https://www.khronos.org/opengl/ wiki/OpenGL_Shading_Language. 6
- [3] Sketchbook. https://github.com/swift502/ Sketchbook. 6
- [4] WebGL. https://www.khronos.org/webgl/. 6
- [5] Fetch Mobile Manipulator. https://fetchrobotics. borealtech.com/robotics-platforms/fetchmobile-manipulator/?lang=en. 8, 3
- [6] ngp-pl. https://github.com/kwea123/ngp_pl. 3
- [7] PyMesh. https://github.com/PyMesh/PyMesh. 1
- [8] Stretch® Research Edition. https://hello-robot. com/product. 8, 3
- [9] urdf-loaders. https://github.com/gkjohnson/ urdf-loaders. 3
- [10] V-HACD. https://github.com/kmammou/vhacd. 2
- [11] Xatlas. https://github.com/mworchel/xatlaspython. 5
- [12] RoboTHOR: An Open Simulation-to-Real Embodied AI Platform. CVPR, 2020. 8
- [13] ManipulaTHOR: A Framework for Visual Object Manipulation. In CVPR, 2021. 8
- [14] Alexander Amini, Tsun-Hsuan Wang, Igor Gilitschenski, Wilko Schwarting, Zhijian Liu, Song Han, Sertac Karaman, and Daniela Rus. Vista 2.0: An open, data-driven simulator for multimodal sensing and policy learning for autonomous vehicles. In ICRA, 2022. 3
- [15] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In ECCV, 2022. 2
- [16] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 4, 6, 7, 8, 1, 5
- [17] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. ECCV, 2022. 2
- [18] Shenchang Eric Chen and Lance Williams. View interpolation for image synthesis. In Proceedings of the 20th annual conference on Computer graphics and interactive techniques, 1993. 2
- [19] Yun Chen, Frieda Rong, Shivam Duggal, Shenlong Wang, Xinchen Yan, Sivabalan Manivasagam, Shangjie Xue, Ersin Yumer, and Raquel Urtasun. Geosim: Realistic video simulation via geometry-aware composition for self-driving. In CVPR, 2021. 3
- [20] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. CVPR, 2023. 2, 5, 7, 1
- [21] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 6, 8

- [22] Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016–2021. 8, 2
- [23] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator, 2017. 1, 3
- [24] Shivam Duggal, Shenlong Wang, Wei-Chiu Ma, Rui Hu, and Raquel Urtasun. Deeppruner: Learning efficient stereo matching via differentiable patchmatch. In ICCV, 2019. 1
- [25] D. Ponsa E. Rublee E. Riba, D. Mishkin and G. Bradski. Kornia: an open source differentiable computer vision library for pytorch. In Winter Conference on Applications of Computer Vision, 2020. 4
- [26] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786–10796, 2021. 4
- [27] Epic Games. Unreal engine. 1, 6, 8
- [28] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2
- [29] Rockstar Games. Grand theft auto v. 2014. 1
- [30] John K Haas. A history of the unity game engine. 2014. 1
- [31] Benno Heigl, Reinhard Koch, Marc Pollefeys, Joachim Denzler, and Luc Van Gool. Plenoptic modeling and rendering from image sequences taken by a hand-held camera. In DAGM-Symposium, 1999. 2
- [32] Ronghang Hu, Nikhila Ravi, Alexander C Berg, and Deepak Pathak. Worldsheet: Wrapping the world in a 3d sheet for view synthesis from a single image. In ICCV, 2021. 2
- [33] O˘guzhan Fatih Kar, Teresa Yeo, Andrei Atanov, and Amir Zamir. 3d common corruptions and data augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18963–18974, 2022. 4
- [34] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG), 42(4):1–14, 2023. 5, 7, 1, 3, 4
- [35] Pradeep K Khosla and Takeo Kanade. Parameter identification of robot dynamics. In IEEE conference on decision and control, 1985. 3
- [36] Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with gamegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1231–1240, 2020. 2, 3
- [37] Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable high-quality neural simulation. In CVPR, 2021. 2, 3
- [38] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, Aniruddha Kembhavi, Abhinav Gupta, and Ali Farhadi. Ai2-thor: An interactive 3d environment for visual ai, 2022. 6, 8
- [39] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for

- high-performance differentiable rendering. ACM Transactions on Graphics, 39(6), 2020. 5, 2
- [40] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-aware text-driven layered video editing. arXiv, 2023. 2
- [41] Marc Levoy and Pat Hanrahan. Light field rendering. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, 1996. 2
- [42] Yuan Li, Zhi-Hao Lin, David Forsyth, Jia-Bin Huang, and Shenlong Wang. Climatenerf: Physically-based neural rendering for extreme climate synthesis. arXiv, 2022. 2, 4
- [43] Yiyi Liao, Jun Xie, and Andreas Geiger. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE TPAMI, 2022. 4, 6, 7, 8, 1, 3, 5
- [44] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In ICCV, 2021. 2
- [45] Zhi-Hao Lin, Wei-Chiu Ma, Hao-Yu Hsu, Yu-Chiang Frank Wang, and Shenlong Wang. Neurmips: Neural mixture of planar experts for view synthesis. In CVPR, 2022. 2
- [46] Zhi-Hao Lin, Bohan Liu, Yi-Ting Chen, David Forsyth, Jia-Bin Huang, Anand Bhattad, and Shenlong Wang. Urbanir: Large-scale urban scene inverse rendering from a single video, 2023. 2
- [47] Jeffrey Yunfan Liu, Yun Chen, Ze Yang, Jingkang Wang, Sivabalan Manivasagam, and Raquel Urtasun. Real-time neural rasterization for large scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8416–8427, 2023. 3
- [48] Fan Lu, Yan Xu, Guang Chen, Hongsheng Li, Kwan-Yee Lin, and Changjun Jiang. Urban radiance field representation with deformable neural mesh primitives, 2023. 3
- [49] Sivabalan Manivasagam, Shenlong Wang, Kelvin Wong, Wenyuan Zeng, Mikita Sazanovich, Shuhan Tan, Bin Yang, Wei-Chiu Ma, and Raquel Urtasun. Lidarsim: Realistic lidar simulation by leveraging the real world. In CVPR, 2020. 3
- [50] Willi Menapace, Stephane Lathuiliere, Sergey Tulyakov, Aliaksandr Siarohin, and Elisa Ricci. Playable video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10061–10070,

2021. 3

- [51] Willi Menapace, St´ephane Lathuiliere, Aliaksandr Siarohin, Christian Theobalt, Sergey Tulyakov, Vladislav Golyanik, and Elisa Ricci. Playable environments: Video manipulation in space and time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3584–3593, 2022. 3
- [52] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2, 4
- [53] Thomas M¨uller. tiny-cuda-nn, 2021. 6, 1
- [54] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 2, 3, 4, 5, 7, 1

- [55] Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML10), pages 807–814, 2010. 1, 2
- [56] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In CVPR, 2021. 2
- [57] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021.
- [58] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv, 2021. 2
- [59] Cristina Portal´es, Jos´e Luis Lerma, and Santiago Navarro. Augmented reality and photogrammetry: A synergy to visualize physical and virtual city environments. ISPRS Journal of Photogrammetry and Remote Sensing, 65(1):134–142,

2010. 1

- [60] Charalambos Poullis, Andrew Gardner, and Paul Debevec. Photogrammetric modeling and image-based rendering for rapid virtual environment creation. In Proceedings of ASC,

2004. 1

- [61] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE TPAMI, 2020. 4
- [62] Christian Reiser, Richard Szeliski, Dor Verbin, Pratul P Srinivasan, Ben Mildenhall, Andreas Geiger, Jonathan T Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for real-time view synthesis in unbounded scenes. arXiv preprint arXiv:2302.12249, 2023. 2, 5
- [63] Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas Funkhouser, and Vittorio Ferrari. Urban radiance fields. In CVPR, 2022. 2
- [64] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 4
- [65] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv, 2022. 2, 3
- [66] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv, 2023. 2, 3
- [67] Sanghyun Son, Yi-Ling Qiao, Jason Sewall, and Ming C Lin. Differentiable hybrid traffic simulation. TOG, 2022. 3
- [68] Pratul P Srinivasan, Richard Tucker, Jonathan T Barron, Ravi Ramamoorthi, Ren Ng, and Noah Snavely. Pushing the boundaries of view extrapolation with multiplane images. In CVPR, 2019. 2
- [69] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 2

- [70] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers, 2021. 4
- [71] Richard Szeliski and Polina Golland. Stereo matching with transparency and matting. In Sixth International Conference on Computer Vision, 1998. 2
- [72] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In CVPR, 2022. 4
- [73] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa. Nerfstudio: A modular framework for neural radiance field development. arXiv,

2023. 5, 7, 1, 3, 4

- [74] Jiaxiang Tang, Hang Zhou, Xiaokang Chen, Tianshu Hu, Errui Ding, Jingdong Wang, and Gang Zeng. Delicate textured mesh recovery from nerf via adaptive surface refinement. arXiv preprint arXiv:2303.02091, 2023. 2, 5
- [75] ADRIAN SAVARI Thomas, MOHD FAHRUL Hassan, MUSTAFFA Ibrahim, M Nasrull, A Rahman, SZ Sapuan, and F Ahmad. A study on close-range photogrammetry in image based modelling and rendering (imbr) approaches and post-processing analysis. Journal of Engineering Science and Technology, 14(4):1912–1923, 2019. 1
- [76] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IROS, 2012. 3
- [77] Shubham Tulsiani, Richard Tucker, and Noah Snavely. Layer-structured 3d scene inference via view synthesis. In ECCV, 2018. 2
- [78] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In CVPR, 2022. 2
- [79] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. Nerf–: Neural radiance fields without known camera parameters. arXiv, 2021. 2
- [80] Bernhard Wymann, Eric Espi´e, Christophe Guionneau, Christos Dimitrakakis, R´emi Coulom, and Andrew Sumner. Torcs, the open racing car simulator. Software available at http://torcs.sourceforge.net, 2000. 3
- [81] Yuwen Xiong, Jingkang Ma, Wei-Chiu Wang, and Raquel Urtasun. Ultralidar: Learning compact representations for lidar completion and generation. CVPR, 2023. 3
- [82] Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel Rota Bul`o, Lorenzo Porzi, Peter Kontschieder, Aljaˇz Boˇziˇc, Dahua Lin, Michael Zollh¨ofer, and Christian Richardt. VR-NeRF: High-fidelity virtualized walkable spaces. In SIGGRAPH Asia Conference Proceedings, 2023. 6, 8, 1, 2
- [83] Zhenpei Yang, Yuning Chai, Dragomir Anguelov, Yin Zhou, Pei Sun, Dumitru Erhan, Sean Rafferty, and Henrik Kretzschmar. Surfelgan: Synthesizing realistic sensor data for autonomous driving. In CVPR, 2020. 3
- [84] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urta-

sun. Unisim: A neural closed-loop sensor simulator. CVPR,

2023. 2, 3

- [85] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin, Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron, and Ben Mildenhall. Bakedsdf: Meshing neural sdfs for realtime view synthesis. Siggraph, 2023. 2, 5, 7, 1
- [86] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields, 2021. 4
- [87] Zehao Yu, Anpei Chen, Bozidar Antic, Songyou Peng, Apratim Bhattacharyya, Michael Niemeyer, Siyu Tang, Torsten Sattler, and Andreas Geiger. Sdfstudio: A unified framework for surface reconstruction, 2022. 5
- [88] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. in arXiv, 2022. 4
- [89] Haotian Zhang, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, and Kayvon Fatahalian. Learning physically simulated tennis skills from broadcast videos. ACM Trans. Graph. 3
- [90] Haotian Zhang, Cristobal Sciutto, Maneesh Agrawala, and Kayvon Fatahalian. Vid2player: Controllable video sprites that behave and appear like professional tennis players. ACM Transactions on Graphics (TOG), 40(3):1–16, 2021. 3
- [91] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv, 2020. 2
- [92] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [93] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM TOG, 2021. 2
- [94] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J. Davison. In-place scene labelling and understanding with implicit scene representation, 2021. 4
- [95] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv, 2018. 2
- [96] Yiming Zuo and Jia Deng. View synthesis with sculpted neural points. in arXiv, 2022. 2
- [97] Vlas Zyrianov, Xiyue Zhu, and Shenlong Wang. Learning to generate realistic lidar point clouds. In ECCV, 2022. 3

## Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video

### Supplementary Material

#### A. Additional Results and Analysis

More qualitative results. We provide more qualitative comparison results among baselines [20, 34, 54, 73, 85] and our proposed method. For comparisons between InstantNGP [54], Nerfacto [73], 3D Gaussian Splatting [34] and our base NeRF in KITTI-360 dataset [43] and Garden scene in Mipnerf-360 Dataset [16], see Fig. 8 and Fig. 9. We observe that our method renders less noisy geometries while maintaining a superior or comparable visual quality. Especially, 3D Gaussian Splatting [34] fails to learn correct 3D orientations of Gaussians in sparse settings like KITTI360 [43], leading to weird color renderings in novel views and noisy geometry rendering. As for mesh rendering qualitative comparison between [20, 85] and ours, see Fig. 10. Our mesh rendering has similar and comparable rendering results in Garden scene [16]. However, in KITTI-360 dataset [43] which is extremely large-scale and open, the performance of MobileNeRF [20] drops dramatically and BakedSDF [85] generates slightly blurry in road-aside car rendering, while our mesh rendering is not only superior in KITTI-360 dataset [43], but it also maintains stable performance across different datasets.

#### B. Dataset Details

##### B.1. KITTI-360 Dataset

We build “KITTI-Loop game” based on KITTI-360 Dataset [43]. We use frames from sequence 0. The loop we build in our game utilizes frames 4240-4364, 63546577, 7606-7800, and 10919-11050. We compose those four snippets into a closed loop in our game. For baseline comparison and ablation study, we perform experiments on two blocks containing frames 7606-7665 and 10919-11000. We split the validation set every 10 frames (frames 7610, 7620, 7630, 7640, 7650, and 7660 for the first block; frames 10930, 10940, 10950, 10960, 10970, 10980, 10990 for the second block). We report the average metrics of two blocks.

##### B.2. Mipnerf-360 Dataset

We build the “Gardenvase game” based on the Garden scene of Mipnerf-360 Dataset [16]. We split the validation set every 20 frames.

##### B.3. VRNeRF Dataset

We build our robot simulation environment based on the “table” scene of VRNeRF Dataset [82].

#### C. Video2Game Implementation Details

##### C.1. Base NeRF Training Details

Network architecture and hyper-parameters Our network consists of two hash grid encoding [54] components Itd and Itc and MLP headers MLPdθ

, and MLPnθ

, MLPcθ

, MLPsθ

c

s

d

, each with two 128 neurons layers inside. Taking 3D position input x, density σ is calculated following σ = MLPdθ

n

(Itd(Ct(x),Φd)). Color feature f = Itc(Ct(x),Φc). Then we calculate c,s,n from feature f and direction d through c = MLPcθ

d

(f,d), s = MLPsθ

(f) and n = MLPnθ

c

s

(f) respectively. All parameters involved in training our base NeRF can be represented as NGP voxel features Φ = {Φd,Φc} and MLP parameters θ = {θd,θc,θs,θn}. To sum up, we get c,σ,s,n = Fθ(x,d;Φ) = MLPθ(It(Ct(x),Φ),d). The detailed diagram of our NeRF can be found in Fig. 11.

n

Our hash grid encoding [54] is implemented by tinycuda-nn [53], and we set the number of levels to 16, the dimensionality of the feature vector to 8 and Base-2 logarithm of the number of elements in each backing hashtable is 19 for Itd and 21 for Itc. As for activation functions, we use ReLU [55] inside all MLPs, Softplus for density σ output, Sigmoid for color c output, Softmax for semantic s output and no activation function for normal n output (We directly normalize it instead).

KITTI-Loop additional training details In KITTI-Loop which uses KITTI-360 Dataset [43], we also leverage stereo depth generated from DeepPruner [24]. Here we calculate the actual depth from disparity and the distance between binocular cameras and adopt L1 loss to regress. We haven’t used any LiDAR information to train our base NeRF in KITTI-360 Dataset [43].

##### C.2. Mesh Extraction and Post-processing Details

Mesh Post-processing details In mesh post-processing, we first utilize all training camera views to prune the vertices and faces that can’t be seen. Next, we delete those unconnected mesh components that have a small number of faces below a threshold so as to delete those floaters in the mesh. Finally, we merge close vertices in the mesh, then perform re-meshing using PyMesh [7] package, which iteratively splits long edges over a threshold, merges short edges below a threshold and removes obtuse triangles. Remeshing helps us get better UV mapping results since it makes the mesh “slimmer” (less number of vertices and

faces) and has similar lengths of edges. After the postprocessing, we get meshes with a relatively small number of vertices and faces while still effectively representing the scene.

Special settings in KITTI-Loop In KITTI-Loop, we partition the whole loop into 14 overlapping blocks. Since we adopt pose normalization and contract space in each block when training, it needs alignments when we compose them together. For each block, we first generate its own mesh. We partition the whole contract space ([−1,1]3) into 3*3*3 regions, and perform marching cubes with the resolution of 256*256*256 in each region. We then transform those vertices back from contract space to the coordinates before contraction. We then perform mesh post-processing here. To compose each part of the mesh in KITTI-Loop together, we then transform the mesh to KITTI-Loop world coordinates. For those overlapping regions, we manually define the block boundary and split the mesh accordingly. Finally, we add a global sky dome over the KITTI-Loop.

##### C.3. NeRF Baking Details

For each extracted mesh, we bake the NeRF’s color and specular components to it with nvdiffrast [39].

GLSL MLP settings We adopt a two-layer tiny MLP with 32 hidden neurons. We use ReLU [55] activation for the first layer and sigmoid for the second. We re-implement that MLP with GLSL code in Three.js renderer’s shader.

Initialization of texture maps and MLP shader Training the textures T = [B;S] and MLP shader MLPshaderθ all from scratch is slow. Instead, we adopt an initialization procedure. Inspired by [74, 85], we encode the 3D space by hash encoding [54] ItM and an additional MLP MLPMθ

. Specifically, we first rasterize the mesh into screen space, obtain the corresponding 3D position xi on the surface of the mesh within each pixel, transform it into contract space Ct(xi), and then feed it into ItM and MLPMθ

0

to get the base color Bi and specular feature Si, represented as Bi, Si = MLPMθ

0

(ItM(Ct(xi),Φ0)). Finally we computes the sum of the view-independent base color Bi and the view-dependent specular color following CR = Bi + MLPshaderθ (Si,di). The parameters Φ0,θ0,θ are optimized by minimizing the color difference between the mesh model and the ground truth: LrenderinitializeΦ0,θ0,θ =

0

r ∥CR(r) − CGT(r)∥22. Anti-aliasing is also adopted in the initialization step by perturbing the optical center of the camera. With learned parameters, every corresponding 3D positions xi in each pixel of 2D unwrapped texture maps T = [B;S] is initialized following Bi, Si =

𝝈

Density

|𝑀𝐿𝑃𝑑|
|---|

x: (x, y, z) d: (θ, φ)

Contraction

𝒄

Color

[Figure 85]

|𝑀𝐿𝑃𝑐|
|---|

[Figure 86]

Density Field 𝐼𝑡𝑑

𝑠

Semantic

|𝑀𝐿𝑃𝑠|
|---|

𝒏

Unbounded Scene

Normal

Color Field 𝐼𝑡𝑐

|𝑀𝐿𝑃𝑛|
|---|

Figure 11. Video2Game NeRF Module: The diagram of our designed NeRF.

(ItM(Ct(xi),Φ0)) and the parameters of MLPshaderθ is directly copied from initialization stage.

MLPMθ

0

##### C.4. Physical Module Details

Physical dynamics It is important to note that our approach to generating collision geometries is characterized by meticulous design. In the case of box collider generation, we seamlessly repurpose the collider used in scene decomposition. When it comes to triangle mesh colliders, we prioritize collision detection efficiency by simplifying the original mesh. Additionally, for convex polygon colliders, we leverage V-HACD [10] to execute a precise convex decomposition of the meshes.

Physical parameters assignments. Physical parameters for static objects, such as the ground, were set to default values. For interactive instances like cars and vases, we could query GPT-4 with box highlights and the prompts as shown on the left. Note that we reason about mass and friction using the same prompt. The output is a range, and we find that selecting a value within this range provides reasonable results. See Fig. 12 for an example. Unit conversion from the metric system to each engine’s specific system is needed.

##### C.5. Robot Simulation Details

Data preparation We demonstrate the potential of leveraging Video2Game for robot simulation using the VRNeRF [82] dataset. We reconstruct the scene and segment simulatable rigid-body objects (e.g., the fruit bowl on the table). Then collision models are generated for those physical entities for subsequent physical simulations.

Physical simulation To simulate the interactions between robots and physical entities in a dynamic environment, we employ PyBullet [22], a Python module designed for

[Figure 87]

[Figure 88]

[Figure 89]

G.T.InstantNGPNerfactoGuass.Spl.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

N/A

[Figure 98]

[Figure 99]

[Figure 100]

Ours

- Figure 8. Qualitative comparisons among NeRF models [54, 73] and 3D Gaussian Splatting [34] in KITTI-360 Dataset [43]. We provide NeRF rendering depths and normals for comparison as well. For 3D Gaussian Splatting, only rendering depth is provided. Here we consider depths measured by LiDAR point cloud in KITTI-360 and compute normals based on it as our ground truth.

physics simulations in the realms of games, robotics, and machine learning. Given the intricate dynamics of articulated robots, PyBullet serves as a powerful tool for conducting physics calculations within the context of robot simulation. Our approach involves loading all generated collision models and URDF 1 files for both the Stretch Robot [8] and Fetch Robot [5]. Utilizing PyBullet’s integrated robotic inverse kinematics, we can effectively control the mechanical arms of the robots to interact with surrounding objects. Specifically, for the Stretch Robot, we define a predefined path for its arm, enabling it to exert a direct force to displace the central bowl off the table. On the other hand, for the Fetch Robot, we leverage the collision boxes specified in its URDF file. Our manipulation involves grasping the corresponding collision model of the central bowl on the table, eschewing the use of the magnetic gripper for object control. Subsequently, the robot lifts the bowl and relocates it to a different position. Following the simulations in PyBullet, we extract physics calculation results, including joint values and the position of the robots’ base link. These results are then exported and integrated into the rendering engine of Three.js for further visualization and analysis.

Rendering in robot simulation We import the URDF files of our robots into our engine using the urdf-loader [9] in Three.js, a library that facilitates the rendering and con-

1http://wiki.ros.org/urdf

figuration of joint values for the robots. Leveraging precomputed physics simulations in PyBullet, which are based on our collision models, we seamlessly integrate these simulations into the Three.js environment. This integration allows us to generate and render realistic robot simulation videos corresponding to the simulated physics interactions.

##### C.6. Training time

For base NeRF training, it takes 8 hours for training 150k iterations on an A6000. For the NeRF baking procedure, the initialization and training take 4 hours on an A5000.

#### D. Baseline Details

##### D.1. Instant-NGP

We adopt the re-implementation of Instant-NGP [54] in [6]. We choose the best hyper-parameters for comparison. For normal rendering, we calculate by the derivative of density value.

##### D.2. Nerfacto

Nerfacto is proposed in Nerfstudio [73], an integrated system of simplified end-to-end process of creating, training, and testing NeRFs. We choose their recommended and default settings for the Nerfacto method.

[Figure 101]

G.T.

N/A N/A

[Figure 102]

[Figure 103]

[Figure 104]

NerfactoGauss.Spl.InstantNGP

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

N/A

[Figure 110]

[Figure 111]

[Figure 112]

Ours

- Figure 9. Qualitative comparisons among NeRF models [54, 73] and 3D Gaussian Splatting [34] in Garden scene [16]. We provide NeRF rendering depths and normals for comparison as well. For 3D Gaussian Splatting, only rendering depth is provided.

##### D.3. 3D Gaussian Splatting

For 3D Gaussian Splatting [34] training in Garden scene [16], we follow all their default settings. In the

KITTI-360 Dataset, there are no existing SfM [64] points. We choose to attain those 3D points by LoFTR [70] 2D image matching and triangulation in Kornia [25] using existing

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

G.T. MobileNeRF BakedSDF Ours

- Figure 10. Qualitative comparisons in mesh rendering. We compare our proposed mesh rendering method to MobileNeRF [20] and BakedSDF [85] in KITTI-360 Dataset [43].

[Figure 129]

##### D.5. BakedSDF

[Figure 130]

[Figure 131]

We adopt the training codes of BakedSDF [85] in SDFStudio [87], from which we can attain the exported meshes with the resolution of 1024x1024x1024 by marching cubes. For the baking stage, we adopt three Spherical Gaussians for every vertices and the same hyper-parameters of NGP [54] mentioned in [85]. We follow the notation BakedSDF [85] used in its paper, where “offline” means volume rendering results.

#### E. Limitation

Although Video2Game framework could learn viewdependent visual appearance through its NeRF module and mesh module, it doesn’t learn necessary material properties for physics-informed relighting, such as the metallic property of textures. Creating an unbounded, relightable scene from a single video, while extremely challenging, can further enhance realism. We leave this for future work.

Figure 12. Example of physical property reasoning by GPT-4.

camera projection matrixs and matching results. We choose their best validation result throughout the training stage by testing every 1000 training iterations.

##### D.4. MobileNeRF

In Garden scene [16], we directly follow the default settings of MobileNeRF [20]. For training in KITTI-360 Dataset [43], we adopt their “unbounded 360 scenes” setting for the configurations of polygonal meshes, which is aligned with KITTI-360 Dataset.

