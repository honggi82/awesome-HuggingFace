## City-on-Web: Real-time Neural Rendering of Large-scale Scenes on the Web

# arXiv:2312.16457v2[cs.CV]1Apr2024

##### Kaiwen Song1, Xiaoyi Zeng1, Chenqu Ren2, and Juyong Zhang1

1 University of Science and Technology of China 2 East China Normal University

https://ustc3dv.github.io/City-on-Web/

Abstract. Existing neural radiance field-based methods can achieve real-time rendering of small scenes on the web platform. However, extending these methods to large-scale scenes still poses significant challenges due to limited resources in computation, memory, and bandwidth. In this paper, we propose City-on-Web, the first method for real-time rendering of large-scale scenes on the web. We propose a block-based volume rendering method to guarantee 3D consistency and correct occlusion between blocks, and introduce a Level-of-Detail strategy combined with dynamic loading/unloading of resources to significantly reduce memory demands. Our system achieves real-time rendering of large-scale scenes at approximately 32FPS with RTX 3060 GPU on the web and maintains rendering quality comparable to the current state-of-the-art novel view synthesis methods.

Keywords: real-time rendering · neural rendering · large-scale reconstruction

### 1 Introduction

Neural radiance field (NeRF) has significantly advanced the field of scene reconstruction, showing an unparalleled ability to capture complex details across diverse environments. Existing works have demonstrated its ability to render small scenes with exceptional quality and performance in real-time [2, 5, 9, 20, 24,29,37,40,43,46–48]. NeRF has also been successfully applied to the rendering of large scenes in offline settings, achieving exceptional visual fidelity and generating intricately detailed results [11,35,39,43,45].

Despite these successes, real-time neural rendering of large scenes on the web remains profoundly challenging due to inherent computational power, memory, and bandwidth limitations on commodity devices. MERF [30] has recently achieved significant progress by employing a baking technique to reduce query network calls in the rendering pipeline, thereby enabling real-time rendering of small-scale scenes on the web. However, MERF struggles to capture intricate details in large scenes due to its limited resolution. A naive solution would be to simply increase the volumetric representation’s resolution, but this approach

would lead to unacceptable increases in memory usage, scaling with O(N3), and a significant decrease in rendering speed.

To overcome these limitations, we integrate MERF with a block-based strategy [39] for reconstructing large scenes, a method supported by numerous studies [35, 39, 50]. This approach not only improves the model’s representational ability but also controls memory growth at an O(N2) rate because we divide the scene based on ground coordinates without dividing the height. However, there are certain challenges associated with a resource-independent block-based rendering approach on the web. Specifically, rendering on the web faces limitations on the number and resolution of texture units that can be loaded into a shader (typically no more than 16 texture units), which prevents loading all block resources into a single shader. Consequently, we load the rendering resources of different blocks into their respective shaders. Nevertheless, rendering with different shaders causes issues with 3D consistency. Specifically, when a ray traverses multiple blocks, sampling points might belong to different blocks loaded by different shaders, preventing standard volume rendering. We are thus compelled to render each block sequentially and subsequently combine the rendering results of the different blocks. To this end, we propose a block-based volume rendering strategy and demonstrate that this method of sequential block rendering is equivalent to volume rendering, thereby ensuring correct occlusion and 3D consistency of the rendering results.

Moreover, when viewing from a higher altitude viewpoint, the rendering resources of all scene blocks are needed. Nonetheless, loading all blocks for rendering is impractical due to the excessive memory usage that would surpass the capacity of standard consumer devices. To address this issue, we draw inspiration from traditional graphics techniques [6–8,12,14,18,23] to create Level-of-Detail (LOD) for each block’s resources, dynamically selecting resources for rendering based on the camera’s position and field of view. This approach significantly reduces the resource demands during rendering, paving the way for smoother user experiences even on less capable devices.

In summary, the contributions of this paper include the following aspects:

- • We propose a block-based multi-shader volume rendering method, ensuring real-time high-fidelity rendering of large-scale scenes.
- • We employ an LOD strategy and dynamic loading/unloading strategies to adaptively manage rendering resources and significantly reducing the quantity of resources loaded and ensuring efficient resource utilization for largescale scene rendering.
- • Our experiments demonstrate that our system achieves real-time rendering of large-scale scenes at approximately 32FPS with 1080P resolution on an RTX 3060 GPU, while maintaining rendering quality comparable to the current state-of-the-art (SOTA) methods for large-scale scenes.

[Figure 1]

Training

Depth-sorting among blocks

Block-based volume rendering

[Figure 2]

𝒄𝒄𝒊𝒊

[Figure 3]

[Figure 4]

Shader Block 𝒊𝒊

| | |
|---|---|
| | |

𝛼𝛼𝒊𝒊

[Figure 5]

[Figure 6]

Block 𝒊𝒊

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Shader Block 𝒋𝒋

[Figure 16]

𝒄𝒄𝒋𝒋

[Figure 17]

Generate LOD

[Figure 18]

𝛼𝛼𝒋𝒋

Block 𝒋𝒋

𝑀𝑀

𝑘𝑘−1

Dynamic Loading

(1 − 𝛼𝛼𝑗𝑗) 𝑐𝑐𝑘𝑘

𝐶𝐶 𝑟𝑟 =

|[Figure 19]|
|---|

|[Figure 20]|
|---|

𝒄𝒄𝒌𝒌

Shader Block 𝒌𝒌

[Figure 21]

[Figure 22]

𝑘𝑘=1

𝑗𝑗=1

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

𝛼𝛼𝒌𝒌

𝛼𝛼𝑘𝑘: opacity of shader k 𝑐𝑐𝑘𝑘: color of shader k

[Figure 27]

[Figure 28]

result

gt

[Figure 29]

color & opacity

[Figure 30]

Block 𝒌𝒌

Rendering

###### LOD Generation

virtual grids

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

[Figure 44]

[Figure 45]

[Figure 46]

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

Downsample

Retrain a shared deferred MLP

Optimize together

- Fig. 1: Overview of City-on-Web pipeline. During the training phase, we uniformly partition the scene and reconstruct it at the finest LOD. To ensure 3D consistency, we use a resource-independent block-based volume rendering strategy (Sec. 4.2). For LOD generation, we downsample virtual grid points and retrain a coarser model (Sec. 4.4). This approach supports subsequent real-time rendering by facilitating the dynamic loading of rendering resources.

### 2 Related Work

Large-scale Scene Reconstruction. For radiance field reconstruction of largescale scenes, a key issue lies in enhancing the model’s ability to adequately capture and render extensive scenes. Some works [10,35,39] address this by adopting a divide-and-conquer strategy, segmenting expansive scenes into smaller blocks, and applying localized NeRF processing to each. This approach significantly improves both the reconstruction quality and the model’s scalability to larger scenes. Switch-NeRF [50] employs a gating network to dispatch 3D points to different NeRF sub-networks. Grid-NeRF [45] utilizes a compact multiresolution feature plane and combines the strengths of smoothness from vanilla NeRF with the local detail capturing ability of feature grid-based methods [4, 26, 32], efficiently reconstructing large scenes with fine details. NeRF++ [48] enhances the reconstruction of unbounded scenes through its innovative multi-spherical representation. On the other hand, Mip-NeRF 360 [1] introduces a scene contraction function to effectively represent scenes that extend to infinity, addressing the challenge of vast spatial extents. F2-NeRF [41] takes this a step further by implementing a warping function for local spaces, ensuring a balance of computational resources and training data across different parts of the scene.

Real-time Rendering. Early works mainly focus on the real-time rendering of a simple single object. NSVF [20] improves NeRF by introducing a more efficient sparse voxel field, significantly accelerating rendering speed while maintaining high-quality output. KiloNeRF [29] utilizes thousands of small MLPs, each responsible for a tiny scene region, significantly reducing network evalua-

tion time. In contrast, SNeRG [13] leverages pre-computed sparse grids, allowing for direct retrieval of radiance field information without needing network evaluation. Termi-NeRF [28] terminates ray marching in less impactful scene regions, slashing computation time. DONeRF [27] focuses on one sample using a depth oracle network, speeding up rendering while preserving scene quality. Recently, there have been developments that enable real-time rendering of neural radiance fields in small scenes. MERF [30] improves upon SNeRG by utilizing a voxel and triplane hybrid representation to reduce memory usage. MobileNeRF [5] introduces the polygon rasterization rendering pipeline, running NeRF-based novel view synthesis in real-time on mobile devices. BakedSDF [46] bakes volumetric representation into meshes and utilizes spherical harmonics for representing view-dependent color, while NeRF2Mesh [37] iteratively refine both the geometry and appearance of the mesh. Furthermore, several methods [38,42] exploit the real-time rendering attributes of mesh representations alongside the robust representational potential of volume representations, particularly for rendering hair, translucent materials, and similar entities. These hybrid methods facilitate the achievement of high-fidelity real-time rendering. Recently, 3D Gaussian splatting [16] achieves real-time rendering by utilizing a novel 3D Gaussian scene representation and a rasterization-based rendering pipeline. However, extending this representation to large scenes is challenging due to its substantial memory consumption.

Level of Detail. Substantial works are devoted to integrating LOD methods into the fabric of traditional computer graphics [6–8,12,14,18,22,23], aiming to streamline rendering processes, reduce memory footprint, bolster interactive responsiveness. Recently, some works have begun to apply LOD to neural implicit reconstruction. NGLoD [34] represents LOD through a sparse voxel octree, where each level of the octree corresponds to a different LOD, allowing for a finer discretization of the surface and more detailed reconstruction as the tree depth increases. Takikawa et al. [33] efficiently encode 3D signals into a compact, hierarchical representation using vector-quantized auto decoder method. BungeeNeRF [43] employs a hierarchical network structure, where the base network focuses on learning a coarse representation of the scene, and subsequent residual blocks are tasked with progressively refining this representation. TrimipRF [15] and LOD-Neus [51] leverage multi-scale triplane and voxel representations to capture scene details at different scales, effectively implementing anti-aliasing to enhance the rendering and reconstruction quality.

### 3 Background

Our exploration begins with an in-depth analysis of two influential works, SNeRG [13] and MERF [30], which have both set benchmarks for real-time rendering of the radiance field. SNeRG precomputes and stores a Neural Radiance Fields model in a sparse 3D voxel grid. Each active voxel in SNeRG contains several attributes: density, diffuse color, and specular feature vector that captures view-dependent effects. Additionally, an indirection grid is used to enhance rendering by either

indicating empty macroblocks or pointing to detailed texels in a 3D texture atlas. This representation allows real-time rendering on standard laptop GPUs.

The indirection grid assists in raymarching through the sparse 3D grid by passing empty regions and selectively accessing non-zero densities σi, diffuse colors ci, and feature vectors fi from baked textures. Integrating along each ray r(t) = o + td, they compute the sum of the weights, which can be considered as the pixel’s opacity:

α(r) =

i

wi, wi =

- i−1
- j=1

(1 − αj)αi,αi = 1 − e−σ

iδi. (1)

The step size δi during ray marching is equal to the voxel width for an occupied voxel. The color Cd(r) and specular feature Fs(r) along the ray are accumulated using the same weights to compute the final diffuse color and specular feature of ray:

wifi. (2)

Cd(r) =

wici, Fs(r) =

i

i

Subsequently, the accumulated diffuse color and specular feature vector, along with the positional encoding PE(·) of the ray’s view direction, are concatenated to pass through a lightweight deferred MLP Φ to produce a view-dependent residual color:

C(r) = Cd + Φ(Cd,Fs,PE(d)). (3)

While SNeRG achieves impressive real-time rendering results, its voxel representation demands substantial memory, which poses limitations for further applications. MERF presents a significant reduction in memory requirements in comparison to extant radiance field methods like SNeRG. By leveraging hybrid low-resolution sparse voxel and 2D high-resolution triplanes, MERF optimizes the balance between performance and memory efficiency. Moreover, it incorporates two pivotal strategies to bridge the gap between training and rendering performance. MERF simulates finite grid approach during training, querying MLPs at virtual grid corners and simulates quantization during training to mimic the rendering pipeline closely.

### 4 Method

In this section, we present a method for representing and rendering large scenes on the web. Our approach utilizes a block and LOD strategy for rendering largescale scenes, as described in (Sec. 4.1). A block-based volume rendering approach is introduced for the seamless blending of different blocks, utilized both during the training and rendering stages to ensure consistency (Sec. 4.2). Sec. 4.3 details the optimization strategies. The generation and refinement of LODs Sec. 4.4 are also explained. Sec. 4.5 elucidates the baking strategy suitable for the representation.

#### 4.1 Large-scale Radiance Field

Although real-time rendering methods like MERF can achieve high-quality realtime rendering for small-scale scenes, they face representational capacity challenges when applied to larger scenes. As mentioned in Sec. 1, utilizing a single MERF model to represent vast scenes is problematic due to its limited resolution, especially in terms of detailed and accurate reconstruction. Therefore, we represent scenes using multiple blocks. However, this approach necessitates employing an LOD strategy to reduce the number of resources that need to be loaded during the rendering phase. Thus, we adopt a block-based and LOD strategy for representing the whole scene in the rendering stage. We will elaborate on the representation used in the rendering stage and provide the representation used to reconstruct the scene in the finest LOD in the training stage.

Training Stage. In the training stage, we only represent and optimize the finest LOD. We uniformly partition the entire scene into K blocks {Bk}Kk=1, each centered at ck = (xk,yk), on the xy plane (i.e., the ground plane). This approach stems from the observation that large scenes typically exhibit smaller scales in the z-direction compared to the xy-plane, prompting us to partition based on the ground plane and avoid subdividing along the z-axis. For a point p = (px,py,pz) ∈ R3, we determine its corresponding block Bk based on its xy coordinates, denoted as pproj = (px,py):

∥pproj − ck∥∞ (4)

p ∈ Bk,k = arg min

k

Within block k, the following trainable components are introduced: (1) fk is an attribute query function that adopts a hash encoding and an MLP decoder that outputs attributes of points such as densities, diffuse color and specular feature (2) Φk is a tiny deferred MLP account for view-dependent effects. (3) ψk is a proposal MLP for sampling.

Rendering Stage. In the rendering stage, our scene representation includes hierarchical L LODs representation for the scene. Specifically, as shown in the right figure, we merge 2×2 blocks into one block between two consecutive LODs. As a result, for LOD l, where l ∈ {1,2,...,L}, there are K/4l−1 blocks. In each block, the following baked textures are used for rendering: (1) fk,l is an attribute query function that takes the coordinates of a sample point as input and directly accesses the opacity, diffuse color and specular features of the sample point from the baked sparse voxel and triplane textures. (2) Φk,l represents a tiny deferred MLP that accounts for view-dependent effects. (3) ψk,l is used as a multi-level occupancy grid for sampling.

#### 4.2 Block-based Volume Rendering

In the rendering stage, we create multiple shaders to render distinct blocks. Specifically, one shader is allocated for storing the texture of an individual block. Each block subsequently renders an image respective to the current camera view. However, a simplistic averaging of these resultant rendering outputs can lead

[Figure 116]

|[Figure 117]|
|---|

|[Figure 118]|
|---|

Incorrect blendingIncorrect depth sorting

[Figure 119]

| |
|---|

[Figure 120]

[Figure 121]

[Figure 122]

𝒘

[Figure 123]

[Figure 124]

|[Figure 125]|
|---|

|[Figure 126]|
|---|

[Figure 127]

|[Figure 128]|
|---|

[Figure 129]

𝒘

[Figure 130]

| |
|---|

[Figure 131]

[Figure 132]

𝒕

(a) Incorrect blending

(b) Ours alpha blending

- Fig. 2: Visualization comparison between the alpha blending method and others. (a) Top image: incorrect occlusion without depth sorting. Bottom image: in-

correct rendering results when simply using αi/( j αj) as blending weights. (b) Left: rendering results of four separate blocks and the final blending result. Right: visualiza-

tion of sample points’ rendering weights before and after alpha blending.

to discernible seams and does not ensure correct occlusion at the inter-block boundaries as shown in Fig. 2a. Therefore, we employ a block-based volume rendering strategy and combine it with depth sorting followed by alpha blending to ensure seamless boundaries and correct occlusion at the edges.

Specifically, in the rendering stage, for a ray r(t) passing through M blocks with a total of N samples, where each block k has nk samples, we perform volume rendering within each block to obtain its individual rendering diffuse

color Ckd, specular feature Fk and opacity αk of the ray in block Bk according to Eqs. (1) and (2). Then we get final rendering color Ck of block Bk according to Eq. (3). Subsequently, to correctly handle occlusion in rendering, we depthsort the blocks and apply volume rendering across multiple blocks in sequence, using opacity to generate the blending weights:

M

C(r) =

k=1

k−1

(1 − αj)Ck. (5)

j=1

[Figure 133]

Under the Lambertian surface setting where the specular color is zero, the color obtained from volume rendering on the total of N ray samples from Eq. (2) is equal to the results produced by our approach of conducting volume rendering within each block followed by inter-block volume rendering Eq. (5).

Deferred Rendering

| | |
|---|---|
| | |

Fig. 3: Block-based volume rendering. "DR" denotes deferred rendering. Φ represents the deferred MLP.

The proof is given in the supplementary. Thus, our rendering approach maintains correct occlusion and keeps 3D consistency when using multiple shaders rendering on the web as shown in Fig. 2b.

The volume rendering process in MERF involves integrating all sample points together, followed by deferred rendering. In contrast, as shown in Fig. 3, our block-based volume rendering is fundamentally based on segmented integration. Without the deferred rendering process, it is entirely equivalent to traditional volume rendering. However, if we adhere to the MERF rendering pipeline during the training process, it will lead to discrepancies between the rendering results during the training and rendering phases, ultimately affecting the rendering quality. To minimize this gap, we adopt the same rendering pipeline during the training stage as we do in the rendering stage.

Specifically, in the training stage, for ray r(t), we uniformly sample between the near and far boundaries based on the scene’s bounding box. Then, we determine that this point is inside Bk according to Eq. (4) and query the corresponding proposal MLP ψk of Bk to sample probability distributions along the ray. Similarly, we also query the corresponding fk to obtain the attributes of the rendering sample points. Lastly, like in the rendering stage, we render each block sequentially to obtain the color and opacity for the ray in Bk and use Eq. (5) to derive the final rendering result.

#### 4.3 Optimization

In the training stage, we reconstruct the finest LOD model by optimizing it with various losses:

Ltrain = Lcb + Lglobal + λ1Ls3im + λ2Lprop + λ3Ldist + λ4Ls + λ5Lopacity. (6)

Here, we use Charbonnier loss [3] Lcb for reconstruction and S3IM loss [44] Ls3im to assist model in capturing high-frequency details. Additionally, we use the interlevel loss Lprop to provide a supervision signal for proposal MLP and distortion loss Ldist to reduce floaters like Mip-nerf 360 [1].

Sparsity Loss. We random uniform sample points set P within the bounding box of the scene and apply L1 regularization on the opacity of sample points αi to encourage model to predict sparse occupied space:

1 |P| p

|αi| (7)

Lsparse =

i∈P

Opacity Loss. We introduce a regularization term for the opacity of the block. This regularization encourages the opacity of the block to be as close to

- 0 or 1 as possible, implying either full transparency or full opaqueness:

(αklog(αk) + (1 − αk)log(1 − αk)). (8)

Lopacity = −

k

Regularization of Deferred MLPs. In the deferred rendering context, various combinations of specular and diffuse colors can satisfy multi-view consistency constraints. This situation often leads to incorrect disentanglement of

these color components. Our training process involves using a deferred MLP within each block, but this approach does not guarantee the smoothness of specular color across block boundaries and the multitude of possible combinations of specular and diffuse colors.

Inspired by Grid-NeRF [45], which utilizes the smoothness of MLP to regularize explicit grid representations. We also utilize a global deferred MLP to regularize the rendering outputs from smaller, block-specific deferred MLPs, ensuring the global smoothness of specular color. In particular, we combine the specular color generated by this global deferred MLP with the diffuse color to obtain the final rendering result. We then supervise the rendering result using ground truth images in the form of a Charbonnier loss, denoted as Lglobal, to regularize the smaller deferred MLPs. Notably, this global deferred MLP is significantly larger and thus possesses sufficient representational capacity compared to the smaller deferred MLPs designated for each block. Therefore, the global MLP does not limit the model’s representational capacity.

#### 4.4 LOD Generation

To ensure high-quality rendering from elevated viewpoints and reduce resource usage for distant scene blocks, our method generates multiple LODs for the scene. One conventional approach to generate LODs would be to retrain the entire scene using fewer blocks, that is, at a lower representation resolution, but this method extends the training time a lot. Additionally, considering the specialized photography techniques employed for capturing large scenes, usually from aerial or top-down perspectives, it is challenging to ensure appearance consistency across models trained separately for extrapolated views if we retrain the entire scene from scratch.

Therefore, we generate LODs based on the scene’s finest LOD acquired during the training stage. Specifically, we simulate the virtual grid to store rendering attributes like MERF in the training stage. As merging M × M blocks into

2 × M2 blocks to generate LODs, we initially downsample the resolution of the virtual grid in each block by a factor of 2. Subsequently, we freeze the training of the query function fk within these submodels and retrain a new shared deferred MLP Φk,l across merged blocks. Finally, we continue to jointly optimize these submodels and the deferred MLP to adapt to lower-resolution voxels and triplanes.

M

#### 4.5 Baking

blocks. Every 2l × 2l blocks can be baked into a single texture asset fk,l in the baking stage thanks to the dowmsampling when generating LODs. Thus, in the rendering stage, a single shader is responsible for rendering these 2l × 2l blocks.

For LOD l, we merge M × M blocks into M2

l × M2l

Specifically, we render all training rays to collect ray samples initially. Samples with opacity and weight values above a certain threshold are retained, and

###### Table 1: Quantitative comparison on the Matrix City, Campus, and Rubble datasets. We report PSNR, LPIPS, and SSIM on the test views. The best and second best results are highlighted.

Matrix City Campus Rubble

PSNR↑ LPIPS↓ SSIM↑ PSNR↑ LPIPS↓ SSIM↑ PSNR↑ LPIPS↓ SSIM↑

NeRFacto 24.95 0.456 0.688 23.47 0.255 0.689 19.02 0.538 0.512 Instant-NGP 23.55 0.597 0.629 21.91 0.478 0.549 20.37 0.629 0.478 Mega-NeRF 25.43 0.517 0.674 22.28 0.472 0.565 23.68 0.558 0.525 Grid-NeRF 24.90 0.480 0.698 - - - - - Ours 25.87 0.332 0.734 24.73 0.192 0.736 21.32 0.482 0.539

samples below the threshold are discarded. The preserved samples are used to mark the adjacent eight grid points as occupied in the binary occupancy grids ϕk,l. After generating binary grids to identify occupied voxels, we follow MERF by baking high-resolution 2D planes and a low-resolution 3D voxel grid in each block to get the attribute function fk,l used in the rendering stage. Only the non-empty 3D voxels are stored using a block-sparse format. We downsample the occupancy grid with max-pooling for efficient rendering. To further save storage, we compress textures into the PNG format.

### 5 Experiments

#### 5.1 Experiments setup

Dataset and Metric. Our experiments span across various scales and environments. We have incorporated a real-world urban scene dataset (Campus) and public datasets consisting of real-world rural rubble scenes (Rublle) [39] and synthetic city-scale data (MatrixCity) [17]. Our datasets were recorded under uniform, cloudy lighting conditions to minimize variation. To obtain precise pose information, we employed an annular capturing approach, which has a higher overlap rate compared to grid-based capturing methods. The dataset covers an area of 1200×800 square meters. It includes a total of 6515 images. We use 99% data for training, and the rest is used as the test dataset. To assess the quality and fidelity of our reconstructions, we employ various evaluation metrics, including PSNR, SSIM and LPIPS [49].

Implementations and Baselines. Our method takes posed multi-view images captured using a fly-through camera as input. The training code is built on the nerfstudio framework [36] with tiny-cuda-nn [25] extension. Our real-time viewer is a JavaScript web application whose rendering is implemented through GLSL. We set the 5123 resolution for the voxel and 20482 resolution for the triplane within each block. We use a 4-layer MLP with 64 hidden dimensions

[Figure 134]

Fig. 4: Qualitative comparisons with existing SOTA methods. By testing different methods across diverse scales and environments, it clearly reveals that our approach excels in recovering finer details and achieves a higher quality of reconstruction.

as an encoder after multi-resolution hash encoding to output density, color, and specular feature. Moreover, a 3-layer MLP with 16 hidden dimensions tiny deferred MLP is developed to predict residual view-dependent color. We sample 16384 rays per batch and use Adam optimizer with an initial learning rate of

- 1 × 10−2 decaying exponentially to 1 × 10−3. The global deferred MLP is a 6-layer MLP with 128 hidden dimensions. Our model is trained with 50k iterations on one NVIDIA A100 GPU. We split the scene into 24 blocks for Campus scene, and split other scenes into four blocks to reconstruct these scenes. We perform qualitative comparisons between our method and existing SOTA methods for large-scale reconstruction. The Campus dataset is partitioned into six parts. NeRFacto, Instant-NGP, and Grid-NeRF were applied to reconstruct one of these parts. NeRFacto and Instant-NGP are utilized with the highest hash encoding resolution of 81923. Mega-NeRF divides the Campus scene into 24 blocks and splits another dataset into four blocks to evaluate its performance.

Our experiments focus on a single part in Campus dataset for comparative analysis with existing real-time rendering methods. This part contains over 1600 images, covering an area of approximately 600 × 400 square meters. We divide this part of the data into four parts for a fair comparison with the MERF method. Other methods did not divide the dataset when conducting experiments. Moreover, we benchmark current real-time rendering methods using three critical parameters: Peak GPU memory usage (VRAM), frames per second (FPS), and

on-disk storage (DISK). We report these metrics based on tests conducted on an NVIDIA RTX 3060 GPU with 1920 × 1080 resolution.

#### 5.2 Results Analysis

[Figure 135]

We systematically evaluate the performance of both baseline models and our method through qualitative and quantitative comparisons in Tab. 1 and Fig. 4. Notably, our method demonstrates a remarkable enhancement in visual fidelity as reflected by the SSIM and LPIPS metrics, which indicate the extent of detail restoration. Despite a reduction in PSNR compared to the SOTA methods, this is attributable to the fact that LPIPS and SSIM are more sensitive to the recovery of fine details, whereas PSNR mainly measures pixel-wise color accuracy. Our approach achieves higher fidelity reconstructions, revealing finer details due to our partitioned reconstruction strategy.

Fig. 5: Visualization of our LOD result.

- Table 2: Comparison with existing real-time methods. We divide the scene into four blocks and 16 blocks. We split the data into four parts, with each part being reconstructed by an individual MERF for a fair comparison. We present the results of Gauss splatting [16] after training 30,000 and 100,000 iterations for demonstration. VRAM and DISK are denoted in megabytes (MB).

PSNR↑ SSIM↑ LPIPS↓ VRAM↓ DISK↓ FPS↑

MobileNeRF 19.99 0.516 0.712 712.3 242.1 68 BakedSDF 22.24 0.627 0.413 544.8 515.3 223 MERF(4 blocks) 24.02 0.713 0.254 592.4 121.2 58 GS(3w iters) 23.78 0.745 0.263 1469.3 1469.1 77 GS(10w iters) 24.94 0.783 0.227 1467.1 1467.1 58

Ours(4 Blocks) 24.82 0.741 0.190 526.6 114.4 46 Ours(16 Blocks) 25.13 0.779 0.167 2040.7 464.5 34

In our evaluation, detailed in Tab. 2, we compare our method with current real-time rendering methods, using one part of the Campus dataset for testing. These tests, are performed on an NVIDIA RTX 3060 Laptop GPU at a 1920 × 1080 resolution. The results demonstrate that our method excels in reconstruction quality. We represent each scene block using voxels and triplanes,

and store the baked grid attributes as images. This strategy significantly reduces the memory. This reduction notably accelerates resource transmission for webbased rendering applications. However, it is observed that our frame rate during rendering is lower compared to other methods. This is attributed to their rendering pipeline based on mesh rasterization, which is in contrast to our method, which utilizes volume rendering.

#### 5.3 LOD Result

Tab. 3 presents the quantitative rendering results at various LODs, along with the corresponding DISK and VRAM usage. With increasing LOD, the resources required for rendering significantly decrease. Notably, our method’s lowest LOD still maintains high-fidelity rendering results, as demonstrated in Fig. 5. Our LOD strategy significantly streamlines the management of resource loading on web platforms, which is particularly advantageous in rendering distant blocks, as it requires less VRAM. It is worth noting that the VRAM usage presented in Tab. 3 represents the cumulative memory consumption of all blocks. Our dynamic loading strategy adaptively selects resources to load based on the camera’s field of view and the distance to each block, effectively keeping the peak VRAM usage around 1100MB.

Table 3: The LOD results on the whole Campus dataset. VRAM and DISK are denoted in megabytes (MB).

PSNR↑ SSIM↑ LPIPS↓ VRAM↓ DISK↓

LOD3 23.72 0.660 0.306 132.1 40.2 LOD2 24.23 0.682 0.297 841.6 201.7 LOD1 24.73 0.736 0.192 3970.2 1259.6

#### 5.4 Ablation Study

We conduct ablation studies to demonstrate the impact of the contributions introduced to our method.

Ablation on Our Method. In Tab. 4, we conduct an ablation study of our method on one part of Campus dataset “ours(4 blocks)” means we use four blocks with 5123 voxel resolution and 20483 triplane resolution for scene reconstruction. “ours(16 blocks)” means we use 16 blocks with 5123 voxel resolution and 20483 triplane resolution for scene reconstruction. In “model with 10243 Res.”, we train a one block MERF model with 10243 voxel resolution and 40962 triplane resolution. In “model with 20483 Res.”, we train a one block MERF model with 20483 voxel resolution and 81922 triplane resolution. Our method has higher rendering quality and requires less storage space. In “no alpha blending”, we

instead our alpha blending with simply using αi/( j αj) as blending weights. This non-occlusion-aware blending strategy significantly reduces the rendering

quality. In “no consistent training”, we use MERF’s volume rendering pipeline in the training stage. In “no global deferred MLP”, we remove global deferred MLP. Without the regularization of deferred MLPs, the quality of the reconstruction has decreased.

- Table 4: Ablation study on our method. The result is tested on one section of the Campus dataset. VRAM and DISK are denoted in megabytes (MB).

PSNR↑ SSIM↑ LPIPS↓ VRAM↓ DISK↓

model with 10243 Res. 24.05 0.710 0.201 540.9 147 model with 20483 Res. 24.42 0.751 0.184 3073.2 457 no alpha blending 24.03 0.684 0.345 565.4 153 no consistent training 24.21 0.702 0.281 514.4 110 no global deferred mlp 24.61 0.712 0.198 536.6 126 ours(4 blocks) 24.82 0.741 0.190 526.6 114 ours(16 Blocks) 25.13 0.779 0.167 2040.7 464

Ablation on LOD Generation. Tab. 5 shows ablation study on LOD generation. We use our LOD generation method as basline. In “downsample”, we simply downsample the model without re-optimization. In “retrain from scratch”, we do not use the finest LOD model to generate LOD. Instead, we trained the same resolution model from scratch.

Table 5: Ablation study on LOD generation.

PSNR↑ SSIM↑ LPIPS↓ Training Time ↓

downsample 22.43 0.614 0.362 0hours retrain from scratch 24.17 0.722 0.224 12hours ours 24.20 0.724 0.204 4hours

### 6 Conclusion

In this work, we introduced City-on-Web, which to our knowledge is the first system that enables real-time neural rendering of large-scale scenes on the web using laptop GPUs. Our choice of a block-based volume rendering strategy, tailored for resource-independent web environments, achieves seamless integration between blocks. Our carefully designed LOD generation and refinement strategy support dynamic loading, minimizing necessary resources on the web while en-

suring the best visual experience. Extensive experiments have also fully proved the effectiveness of City-on-Web.

### References

- 1. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470– 5479 (2022) 3, 8
- 2. Cao, J., Wang, H., Chemerys, P., Shakhrai, V., Hu, J., Fu, Y., Makoviichuk, D., Tulyakov, S., Ren, J.: Real-time neural light field on mobile devices. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8328–8337 (2023) 1
- 3. Charbonnier, P., Blanc-Féraud, L., Aubert, G., Barlaud, M.: Deterministic edgepreserving regularization in computed imaging. IEEE Transactions on image processing 6(2), 298–311 (1997) 8
- 4. Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields. In: European Conference on Computer Vision. pp. 333–350. Springer (2022) 3
- 5. Chen, Z., Funkhouser, T., Hedman, P., Tagliasacchi, A.: Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16569–16578 (2023) 1, 4, 23
- 6. Clark, J.H.: Hierarchical geometric models for visible surface algorithms. Communications of the ACM 19(10), 547–554 (1976) 2, 4
- 7. Crassin, C., Neyret, F., Lefebvre, S., Eisemann, E.: Gigavoxels: Ray-guided streaming for efficient and detailed voxel rendering. In: Proceedings of the 2009 symposium on Interactive 3D graphics and games. pp. 15–22 (2009) 2, 4
- 8. Duchaineau, M., Wolinsky, M., Sigeti, D.E., Miller, M.C., Aldrich, C., MineevWeinstein, M.B.: Roaming terrain: Real-time optimally adapting meshes. In: Proceedings. Visualization’97 (Cat. No. 97CB36155). pp. 81–88. IEEE (1997) 2, 4
- 9. Garbin, S.J., Kowalski, M., Johnson, M., Shotton, J., Valentin, J.: Fastnerf: Highfidelity neural rendering at 200fps. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14346–14355 (2021) 1
- 10. Gu, J., Jiang, M., Li, H., Lu, X., Zhu, G., Shah, S.A.A., Zhang, L., Bennamoun, M.: Ue4-nerf: Neural radiance field for real-time rendering of large-scale scene. Advances in Neural Information Processing Systems 36 (2024) 3, 25
- 11. Guo, J., Deng, N., Li, X., Bai, Y., Shi, B., Wang, C., Ding, C., Wang, D., Li, Y.: Streetsurf: Extending multi-view implicit surface reconstruction to street views. arXiv preprint arXiv:2306.04988 (2023) 1
- 12. Guthe, S., Wand, M., Gonser, J., Straßer, W.: Interactive rendering of large volume data sets. In: IEEE Visualization, 2002. VIS 2002. pp. 53–60. IEEE (2002) 2, 4
- 13. Hedman, P., Srinivasan, P.P., Mildenhall, B., Barron, J.T., Debevec, P.: Baking neural radiance fields for real-time view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5875–5884 (2021) 4
- 14. Hoppe, H.: Progressive meshes. In: Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques. p. 99–108. SIGGRAPH ’96, Association for Computing Machinery, New York, NY, USA (1996). https://doi. org/10.1145/237170.237216, https://doi.org/10.1145/237170.237216 2, 4

- 15. Hu, W., Wang, Y., Ma, L., Yang, B., Gao, L., Liu, X., Ma, Y.: Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19774–19783

(2023) 4

- 16. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 4, 12, 23, 25
- 17. Li, Y., Jiang, L., Xu, L., Xiangli, Y., Wang, Z., Lin, D., Dai, B.: Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3205–3215

(2023) 10

- 18. Lindstrom, P., Pascucci, V.: Visualization of large terrains made easy. In: Proceedings Visualization, 2001. VIS’01. pp. 363–574. IEEE (2001) 2, 4
- 19. Liu, J.Y., Chen, Y., Yang, Z., Wang, J., Manivasagam, S., Urtasun, R.: Neural scene rasterization for large scene rendering in real time. In: The IEEE International Conference on Computer Vision (ICCV) (2023) 25
- 20. Liu, L., Gu, J., Zaw Lin, K., Chua, T.S., Theobalt, C.: Neural sparse voxel fields. Advances in Neural Information Processing Systems 33, 15651–15663 (2020) 1, 3
- 21. Lorensen, W.E., Cline, H.E.: Marching cubes: A high resolution 3d surface construction algorithm. In: Seminal graphics: pioneering efforts that shaped the field, pp. 347–353 (1998) 23
- 22. Losasso, F., Hoppe, H.: Geometry clipmaps: terrain rendering using nested regular grids. In: ACM Siggraph 2004 Papers, pp. 769–776 (2004) 4
- 23. Luebke, D.: Level of detail for 3D graphics. Morgan Kaufmann (2003) 2, 4
- 24. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duckworth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo collections. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7210–7219 (2021) 1
- 25. Müller, T.: tiny-cuda-nn (4 2021), https://github.com/NVlabs/tiny-cuda-nn 10
- 26. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022) 3, 23
- 27. Neff, T., Stadlbauer, P., Parger, M., Kurz, A., Mueller, J.H., Chaitanya, C.R.A., Kaplanyan, A., Steinberger, M.: Donerf: Towards real-time rendering of compact neural radiance fields using depth oracle networks. In: Computer Graphics Forum. vol. 40, pp. 45–59. Wiley Online Library (2021) 4
- 28. Piala, M., Clark, R.: Terminerf: Ray termination prediction for efficient neural rendering. In: 2021 International Conference on 3D Vision (3DV). pp. 1106–1114. IEEE (2021) 4
- 29. Reiser, C., Peng, S., Liao, Y., Geiger, A.: Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14335–14345 (2021) 1, 3
- 30. Reiser, C., Szeliski, R., Verbin, D., Srinivasan, P., Mildenhall, B., Geiger, A., Barron, J., Hedman, P.: Merf: Memory-efficient radiance fields for real-time view synthesis in unbounded scenes. ACM Transactions on Graphics (TOG) 42(4), 1–12

(2023) 1, 4, 22, 23, 24

- 31. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016) 22

- 32. Sun, C., Sun, M., Chen, H.T.: Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5459–5469 (2022) 3
- 33. Takikawa, T., Evans, A., Tremblay, J., Müller, T., McGuire, M., Jacobson, A., Fidler, S.: Variable bitrate neural fields. In: ACM SIGGRAPH 2022 Conference Proceedings. SIGGRAPH ’22, Association for Computing Machinery (2022) 4
- 34. Takikawa, T., Litalien, J., Yin, K., Kreis, K., Loop, C., Nowrouzezahrai, D., Jacobson, A., McGuire, M., Fidler, S.: Neural geometric level of detail: Real-time rendering with implicit 3d shapes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11358–11367 (2021) 4
- 35. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Barron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8248–8258 (2022) 1, 2, 3, 19
- 36. Tancik, M., Weber, E., Ng, E., Li, R., Yi, B., Wang, T., Kristoffersen, A., Austin, J., Salahi, K., Ahuja, A., et al.: Nerfstudio: A modular framework for neural radiance field development. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–12

(2023) 10, 23

- 37. Tang, J., Zhou, H., Chen, X., Hu, T., Ding, E., Wang, J., Zeng, G.: Delicate textured mesh recovery from nerf via adaptive surface refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023) 1, 4
- 38. Turki, H., Agrawal, V., Bulò, S.R., Porzi, L., Kontschieder, P., Ramanan, D., Zollhöfer, M., Richardt, C.: Hybridnerf: Efficient neural rendering via adaptive volumetric surfaces. In: Computer Vision and Pattern Recognition (CVPR) (2024) 4
- 39. Turki, H., Ramanan, D., Satyanarayanan, M.: Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12922–12931 (2022) 1, 2, 3, 10, 19, 23, 25
- 40. Wan, Z., Richardt, C., Božič, A., Li, C., Rengarajan, V., Nam, S., Xiang, X., Li, T., Zhu, B., Ranjan, R., et al.: Learning neural duplex radiance fields for real-time view synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8307–8316 (2023) 1
- 41. Wang, P., Liu, Y., Chen, Z., Liu, L., Liu, Z., Komura, T., Theobalt, C., Wang, W.: F2-nerf: Fast neural radiance field training with free camera trajectories. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4150–4159 (2023) 3
- 42. Wang, Z., Shen, T., Nimier-David, M., Sharp, N., Gao, J., Keller, A., Fidler, S., Müller, T., Gojcic, Z.: Adaptive shells for efficient neural radiance field rendering. ACM Trans. Graph. 42(6) (2023). https://doi.org/10.1145/3618390, https: //doi.org/10.1145/3618390 4
- 43. Xiangli, Y., Xu, L., Pan, X., Zhao, N., Rao, A., Theobalt, C., Dai, B., Lin, D.: Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In: European conference on computer vision. pp. 106–122. Springer (2022) 1, 4
- 44. Xie, Z., Yang, X., Yang, Y., Sun, Q., Jiang, Y., Wang, H., Cai, Y., Sun, M.: S3im: Stochastic structural similarity and its unreasonable effectiveness for neural fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18024–18034 (2023) 8

- 45. Xu, L., Xiangli, Y., Peng, S., Pan, X., Zhao, N., Theobalt, C., Dai, B., Lin, D.: Grid-guided neural radiance fields for large urban scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8296– 8306 (2023) 1, 3, 9, 23
- 46. Yariv, L., Hedman, P., Reiser, C., Verbin, D., Srinivasan, P.P., Szeliski, R., Barron, J.T., Mildenhall, B.: Bakedsdf: Meshing neural sdfs for real-time view synthesis. In: ACM SIGGRAPH 2023 Conference Proceedings, SIGGRAPH 2023, Los Angeles, CA, USA, August 6-10, 2023. pp. 46:1–46:9. ACM (2023) 1, 4, 23
- 47. Yu, A., Li, R., Tancik, M., Li, H., Ng, R., Kanazawa, A.: Plenoctrees for real-time rendering of neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5752–5761 (2021) 1
- 48. Zhang, K., Riegler, G., Snavely, N., Koltun, V.: Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492 (2020) 1, 3
- 49. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018) 10
- 50. Zhenxing, M., Xu, D.: Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In: The Eleventh International Conference on Learning Representations (2022) 2, 3
- 51. Zhuang, Y., Zhang, Q., Feng, Y., Zhu, H., Yao, Y., Li, X., Cao, Y.P., Shan, Y., Cao, X.: Anti-aliased neural implicit surfaces with encoding level of detail. arXiv preprint arXiv:2309.10336 (2023) 4

### A Derivation of Block-based Volume Rendering

Traditional volume rendering methods, like those used in Block-NeRF [35] and Mega-NeRF [39], necessitate accessing radiance and opacity for all sampling points, even if the points belong to different blocks. This requirement means that when the rendering resources of multiple shaders cannot communicate with each other, their volume rendering algorithm cannot work. On the contrary, our block-based volume rendering algorithm is tailored for such resource-independent settings, allowing for the independent rendering of each block before blending their outputs. In this section, we will prove the equivalence between this resourceindependent volume rendering and the traditional volume rendering under Lambertian conditions.

In traditional volume rendering, consider a ray r with M sampling points. For each sampling point pi, the diffuse color, feature, and opacity are represented as ci, fi, and αi respectively. The final diffuse color Cd of the ray r can be obtained by

- i−1
- j=1

M

Cˆ d(r) =

(1 − αj)αici (9)

i=1

In block-based volume rendering, let’s assume the ray, with M sampling points, traverses through K blocks. Within block Bk, there are Nk sampling points. For sampling point pki in block Bk, the diffuse color, feature, and opacity output by the shader or submodel of block Bk are represented as cki , fki , and αik, respectively.

cki = cN

1+···+Nk−1+i fki = fN

(10)

1+···+Nk−1+i αik = αN

1+···+Nk−1+i

Then, for this ray, diffuse color Ckd, specular feature Fk and opacity αk of block Bk are caculated as follows:

- i−1
- j=1

Nk

Ckd =

(1 − αjk) · αikcki

i=1

- i−1
- j=1

Nk

(1 − αjk) · αikfki

Fk =

i=1

- i−1
- j=1

Nk

αk =

(1 − αjk) · αik

i=1

(11)

By blending the rendering results of each block using Eq. (8), the final diffuse color Cd and specular feature F of the ray r can be obtained.

k−1

K

(1 − αj) · Ckd

Cd(r) =

j=1

k=1

k−1

K

(1 − αj) · Fk

F(r) =

j=1

k=1

From Eq. (11), we get the following equation:

(12)

- i−1
- j=1

Nk

1 − αk = 1 −

(1 − αjk) · αik

i=1

= 1 − α1k − (1 − α1k)α2k − (1 − α1k)(1 − α2k)α3k − ···

= (1 − α1k)(1 − α2k − (1 − α2k)α3k − ···)

= (1 − α1k)(1 − α2k)(1 − α3k − ···)

= ···

Nk

(1 − αik)

=

i=1

(13)

Consequently, we can obtain the following by substituting Eq. (13) into Eq. (12).

k−1

K

(1 − αj) · Ckd

Cd(r) =

j=1

k=1

Nj

k−1

K

(1 − αij) · Ckd

(14)

=

j=1

i=1

k=1

N1+···+Nk−1

K

(1 − αi) · Ckd

=

i=1

k=1

By substituting Ckd from Eq. (11) into Eq. (14), we demonstrate the equivalence between Eq. (9) and Eq. (12).

N1+···+Nk−1

- i−1
- j=1

Nk

K

(1 − αjk) · αikcki

(1 − αl)

Cd(r) =

i=1

k=1

l=1

N1−1

c1N

= α11c11 + ··· +

(1 − αi1)αN1

1

1

i=1

block1

N2−1

N1

(1 − αj1) α12c21 + ··· +

(1 − αi2)αN2

c2N

+

2

2

j=1

i=1

block2

##### +···

NK−1

NK−1

N1

(1 − αjK−1) α1KcK1 + ··· +

(1 − αj1)···

cKN

(1 − αiK)αNK

+

K

K

j=1

j=1

i=1

blockK

N1−1

= α11c11 + ··· +

(1 − αi1)αN1

c1N

(15)

1

1

i=1

block1

N2−1

N1

N1

(1 − αj1)α12c21 + ··· +

(1 − αj1)

(1 − αi2)αN2

c2N

+···

+

2

2

j=1

j=1

i=1

block2

NK−1

N1

(1 − αiK−1)α1KcK1 + ···

(1 − αi1)···

+

i=1

i=1

blockK

NK−1

NK−1

N1

(1 − αiK−1)

(1 − αi1)···

(1 − αiK)αNK

cKN

+

)

K

K

i=1

i=1

i=1

blockK

N1−1

(1 − αi)αN

= α1c1 + ··· +

cN

1

1

i=1

block1

N1+N2−1

N1

(1 − αi)αN

(1 − αj)αN

1+1 + ··· +

+···

1+1cN

1+N2cN

+

1+N2

i=1

j=1

block2

N1+···+NK−1

(1 − αj)αN

1+···+NK−1+1 + ···

+

1+···+NK−1+1cN

j=1

blockK

N1+···+NK−1

(1 − αi)αN

+

1+···+NKcN

1+···+NK

i=1

blockK

N1+···+NK

- i−1
- j=1

(1 − αj)αici

=

i=1

- i−1
- j=1

M

(1 − αj)αici = Cˆ d(r)

=

i=1

(16)

It can be observed that block-based volume rendering approach in the resourceindependent environment, such as in our case where multiple shaders are used to render the entire scene, is equivalent to the MERF’s volume rendering method [30] without deferred rendering.

### B More Details on Experiments

In this section, we provide more details on our custom Campus dataset, implemention details and settings of comparison methods.

#### B.1 Dataset

The Campus dataset is captured at an altitude of about 180 meters, covering an area of approximately 960,000 m2. We adopt a circular data capture method for areial photography, as shown in Fig. 6. We find that this method often results in a higher overlap rate, allowing for a more accurate estimation of camera poses. Our dataset was captured over 8 hours on a cloudy day, with a fixed exposure setting to ensure almost identical appearance of photos taken at different time. We used Colmap [31] to estimate camera poses. Feature matching was done using a vocabulary tree, followed by a hierarchical mapper followed by a few iterations of triangulation and bundle adjustment to estimate camera poses.

#### B.2 Implementation Details

For blocks at the boundaries of the entire scene, an unbounded scene representation is required to represent areas outside the block boundaries. We follow the same approach as MERF [30] to compute ray-AABB intersections trivially. To be specific, we employ the scene contraction function to project the scene external to the unit sphere into a cube, which has a radius of 2. The definition of the j − th coordinate for a contracted point is as follows:

 

xj if ∥x∥∞ ≤ 1

xj

∥x∥∞ if xj ̸= ∥x∥∞ > 1 2 − |x1

, (17)

contract(x)j =



xj

|xj| if xj = ∥x∥∞ > 1

j|

We use an A100 GPU for training. In Sections 5.1 and 5.2, we perform training for 50,000 iterations with a batch size of 16384 pixels for the Campus dataset, taking approximately 48 hours, while the training for other datasets takes around 24 hours. The experiments described in Sections 5.2 and 5.4 are subjected to training for 30,000 iterations with the same batch size, which is completed in approximately 12 hours. Training losses are initially balanced with λ1 = 1.0, λ2 = 1.0, λ3 = 0.01, λ4 = 0.05, λ5 = 0.001 and sample 214 samples for computing sparsity loss. During LOD genreation phase, we utilize the same loss function and hyperparameters as those employed during the training stage. We freeze the training of submodels for 5,000 iterations and then refine them and shared global deferred mlp jointly for an additional 10,000 iterations.

[Figure 136]

Fig. 6: Visulization of Campus Dataset.

#### B.3 Comparative Method Settings

In Sec. 5.1, we adopt the official implementations of Mega-NeRF [39], NeRFacto [36], and Grid-NeRF [45]. Additionally, we use an unofficial implementation of Instant-NGP3. Specifically, NeRFacto is trained with a batch size of 65,536 for 30,000 iterations. Instant-NGP [26] is trained for 500,000 iterations with a batch size of 4096. Grid-NeRF [45] is trained for 50,000 iterations with a batch size of 16384. Mega-NeRF [39] is trained for 500,000 iterations for each block, using a batch size of 1024. In Sec. 5.2, For MobileNeRF [5], We initialize a 1923 grid to generate polygonal meshes while adhering to default parameters for other setups. We use the open-source version4 for BakedSDF [46], conducting training in two phases: 20,000 and 50,000 iterations, respectively with a batch size of 16,384 and use 10243 to extract meshes by marching cubes [21]. We employ the official implementation of MERF [30] and 3D Gaussian Splatting [16]. During the MERF [30] experiments, the data is divided into four parts for a fair comparison with our method, and each block is trained with 32,768 batch size for 30,000 iterations, using default parameter settings. For 3D Gaussian splatting [16], we demonstrate the results after training for 30,000 and 100,000 iterations using the default settings.

### C More Results

We present more qualitative comparisons among our method, MERF [30] (4 blocks), and 3D Gaussian Splatting [16] (100k iters) on the Campus Dataset in Fig. 7. Additionally, we showcase rendering results from different LODs to demonstrate the appearance consistency across various LODs in Fig. 8.

- 3 https://github.com/ashawkey/torch-ngp
- 4 https://github.com/hugoycj/torch-bakedsdf

[Figure 137]

Fig. 7: Qualitative comparison of real-time rendering methods.

### D Real-Time Viewer

Our real-time viewer platform is based on the MERF volume renderer [30], leveraging an OpenGL fragment shader to execute ray marching and deferred rendering. It accesses feature and density information through texture look-ups. However, we have implemented several improvements tailored for large scenes to enhance performance.

Dynamic Loading. We employ dynamic loading strategy to determine the level-of-detail and blocks to be loaded, thereby reducing VRAM usage. We use the center points obtained in Section 4.1 for each block on the xy plane as the xy-components, and assign the height of the highest part within the block

- as the z-component. This defines the 3D center point for each block during the rendering stage. Additionally, we use the rectangle formed by the block’s four xy corner points and the previously determined z value to define the block’s region. By projecting this region onto the camera plane, we can determine whether the block is visible to the camera.

During the rendering phase, we first eliminate blocks that are not visible for cameras. Starting with the coarsest LOD, we check if the distance from the camera to all visible blocks within the finer LOD exceeds a certain threshold. For blocks beyond this threshold, rendering is performed using the resources loaded

LOD 3 LOD 2 LOD 1 GT

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

| |
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Fig. 8: Qualitative comparison of different levels-of-detail.

- at the current LOD. Otherwise, the evaluation progresses to the subsequent, finer LOD. This stepwise process continues until we reach the finest LOD or complete the rendering of the entire image.

Depth sorting. We compute the distance from each block’s center point to the camera using only their xy-components, leveraging this calculated distance for depth sorting. We adopt this approach as our scene segmentation occurs solely in the xy plane, without division along the z-axis.

### E Discussion

Discussion. UE4-NeRF [10] and NeuRas [19], both meshbased rendering approaches, offer fast rendering speed but face challenges when representing large, detail-rich scenes like those including dense foliage. These mesh representations can consume extensive memory, often amounting to several gigabytes. Additionally, 3D Gaussian Splatting [16], typically requires millions to tens of millions of points. The inherent properties of Gaussian splatting, which involve numerous Gaussrelated attributes, further increase the VRAM needed for rendering. The substantial VRAM consumption characteristic of these methods

[Figure 154]

Fig. 9: Comparison to Mega-NeRF in Rubble dataset. The Rubble dataset exhibits significant lighting variations. The two images displayed above capture the same location but under vastly different lighting conditions. While our method reconstruct more details than Mega-NeRF [39], the deferred shading model has a limited capacity to represent the lighting and exposure variations as strong view-dependent effects. This limitations leads to marginally lower PSNR values.

poses significant challenges for implementing rendering on web platforms and resource-constrained devices, as they struggle to accommodate the high memory demands.

Limitation & Future Work. Our method still has some limitations. Since we derive alpha blending across shaders based on the Lambertian surface assumption, visible seams may occur at the boundaries between blocks on nonLambertian surfaces, such as water surfaces. Combining physically-based rendering with multiple shaders blending may alleviate this problem. Additionally, while our approach achieves real-time rendering of large scenes on consumergrade laptops, the inherently resource-intensive nature of large scenes makes that real-time rendering on mobile devices remains a challenge. Moreover, while our method recovers more intricate geometrical detail, it frequently results in color discrepancies with the ground truth image due to unstable lighting conditions and variable exposure, as shown in Fig. 9.

