# arXiv:2603.29387v1[cs.CV]31Mar2026

## Extend3D: Town-Scale 3D Generation

Seungwoo Yoon Jinmo Kim Jaesik Park* Seoul National University

{dotori000, jmkim1012, jaesik.park}@snu.ac.kr

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Landmark Occluded Area

[Figure 6]

[Figure 7]

Single 2D Image

3D Vatican City

Figure 1. The result of Extend3D. We generated a large-scale 3D scene from an image of Vatican City captured from Google Earth [9].

### Abstract

In this paper, we propose Extend3D, a training-free pipeline for 3D scene generation from a single image, built upon an object-centric 3D generative model. To overcome the limitations of fixed-size latent spaces in object-centric models for representing wide scenes, we extend the latent space in the x and y directions. Then, by dividing the extended latent space into overlapping patches, we apply the object-centric 3D generative model to each patch and couple them at each time step. Since patch-wise 3D generation with image conditioning requires strict spatial alignment between image and latent patches, we initialize the scene using a point cloud prior from a monocular depth estimator and iteratively refine occluded regions through SDEdit. We

*Corresponding author.

discovered that treating the incompleteness of 3D structure as noise during 3D refinement enables 3D completion via a concept, which we term under-noising. Furthermore, to address the sub-optimality of object-centric models for subscene generation, we optimize the extended latent during denoising, ensuring that the denoising trajectories remain consistent with the sub-scene dynamics. To this end, we introduce 3D-aware optimization objectives for improved geometric structure and texture fidelity. We demonstrate that our method yields better results than prior methods, as evidenced by human preference and quantitative experiments. project page

### 1. Introduction

In the modern era, 3D scene assets are essential across fields such as game development, filmmaking, animation, simulation, and other areas of content production. Creating detailed 3D scenes requires substantial human effort and resources, even with the provided 3D assets. Therefore, a tailored generative model for 3D scenes would help reduce such costs and enhance productivity in industries.

Despite recent advances in 3D generative models, which have enabled the creation of production-ready high-quality 3D objects, generating large-scale 3D scenes remains challenging. One of the main challenges is that most current 3D datasets [3, 4, 8] consist of object-centric data, and lack cases with complex arrangements of multiple objects and a background. Consequently, previous data-centric approaches were unable to generate large general scenes. Moreover, existing latent generative models [44, 47] represent 3D data with a fixed latent size, thereby limiting the level of detail of generated results. As the 3D scene grows in size, the output becomes blurry due to the limited latent dimensionality, resembling a low-resolution image. To adequately represent the scene’s details, the latent size should be adapted to the scale of the result.

Therefore, research has been conducted to develop training-free pipelines for generating 3D scenes using object-centric models. Previous work has explored generating 3D scene blocks through an outpainting process [7, 49]. However, results from these approaches indicate that outpainting can degrade block consistency, particularly in large-scale scenes, making seams visible. Moreover, they rely entirely on the sub-scene generation capabilities of object-centric models, which are not sufficient.

In this paper, we introduce Extend3D, a novel trainingfree pipeline for generating 3D scenes from a single image. To achieve greater detail and scalability in large-scale 3D scene generation, we have expanded the latent space of a pre-trained 3D object generation model. Inspired by training-free, high-resolution image generation methods, such as those presented in recent works [1, 6, 10, 17, 20, 21, 42], we divide the extended latent space into overlapping patches and generate them simultaneously. Unlike previous outpainting methods, our approach automatically refines fine object details within the scene. This is possible because neighboring overlapping patches can influence each other, increasing the likelihood of accurately reconstructing their 3D representations.

However, there are challenges in 2D-3D spatial alignment and the object-centrality of pretrained models. To overcome this, we used the input image and the point cloud extracted from the monocular depth estimator [39] as priors to initialize and optimize the extended latents. We initialize the structure from the point cloud and refine the occluded regions using SDEdit [27] with under-noising. We optimize

the latents at each time step using 3D-aware optimization objectives to align the image and point cloud, ensuring that the denoising paths remain consistent with the sub-scene dynamics.

The qualitative results show that our method is scalable and generalizable. Through human preference and quantitative experiments, we demonstrate that our method outperforms state-of-the-art models in terms of geometry, appearance, and completeness, and is more faithful to the given image. Through an ablation study, we also demonstrate that overlapping patch-wise flow, initialization, and optimization are crucial for training-free 3D scene generation.

The main contributions of this paper are:

- • We extend the latent space to integrate object-centric models into 3D scene generation, enabling a more generalizable and scalable generation pipeline.
- • We introduce an overlapping patch-wise flow with image conditioning that captures local information and mitigates errors arising from object-centric models.
- • We incorporate an iterative under-noised SDEdit process and 3D-aware optimization to complete occluded regions in the monocular depth point cloud and to overcome the deviation of object-centric models from scene dynamics.

- 2. Related Work
- 3D generative models. There have been numerous recent studies on generative models that can generate 3D objects conditioned on text or images. Currently, their main approach is the latent flow model [22, 33] applied to voxelbased or set-based latents.

Trellis [44] generates 3D Gaussians [13], radiance field [29], and mesh, using two steps of latent flow models where each generates a voxelized sparse structure and structured latents. Hunyuan3D [47] utilizes the latent flow model to generate shapes with set-based latents, as proposed in [45]. TripoSG [18] also uses the set-based latent representation of [45] to generate a mesh. These models have the limitation that they are trained with object-centric datasets. Moreover, structurally, current flow-based approaches suffer from the limitation that their latent size is predefined, so the output 3D can only have a confined range of details. We solve these problems by extending the latents to represent a large-scale scene.

To overcome the issues of object-centric models, some attempts have been made to train models using 3D scene datasets. BlockFusion [43] trains a diffusion model to generate cropped sub-scenes and generate the scene by extrapolation. PDD [23] trains a multi-scale diffusion model for coarse-to-fine scene generation. LT3SD [28] generates a 3D scene hierarchically with a latent tree representation. NuiScene [16] trains an autoregressive model with chunk VAE and vector sets. Nevertheless, since all of these methods are trained on limited datasets, they can generate 3D

scenes with fewer categories than object-centric models. They also do not consider detailed model conditioning, such as image conditions, when designing hierarchical frameworks. Unlike them, our method can generate general 3D scenes with detailed image conditioning.

Training-free 3D scene generation. Recent advances in object-centric 3D generative models and the shortage of 3D scene datasets have led researchers to develop training-free 3D scene generation pipelines using these object-centric models.

SynCity [7] generates tiles of 3D sub-scenes sequentially with Trellis from a text using Flux inpainting [15]. Because SynCity attaches separate 3D sub-scenes, there are inconsistencies between tiles, and seams are visible. An imageto-3D scene generation pipeline, 3DTown [49], initializes a scene with the point cloud from VGGT [37] and then completes it patch by patch using RePaint [24] and Trellis. Although 3DTown can generate 3D towns from images with high fidelity, it can only be used with restricted input due to the limitations of object-centric models (e.g., vanishing floors). Also, regardless of initialization, some objects in the scene ignore certain input information, such as rotation. EvoScene [48] further leverages a video diffusion model [36] on 3DTown, but suffers from similar problems.

To address the problems of separate and sequential 3D sub-scene generation, we simultaneously generate 3D subscenes with interacting denoising paths. With small transitions between overlapping patches, the generation process can effectively capture local information and prevent geometrical errors through simultaneous generation. Also, unlike previous works that rely solely on sub-scene generation using an object-centric model, we optimize the latent representation at each step to prevent paths from transitioning from sub-scene to object dynamics.

Training-free high-resolution image generation. In the field of image generation, training-free high-resolution image generation has been widely researched and has led to massive discoveries on the dynamics of the scaled-up latent denoising process. The primary purpose of this area is to generate high-resolution images from pre-trained models trained on relatively low-resolution data.

MultiDiffusion [1] generates a high-resolution image from text with an extended 2D latent with overlapping patches. DemoFusion [6] solves the object repetition problem of Multidiffusion with two ideas: progressive upsampling and dilated sampling. Later research [20, 21], additionally refines dilated sampling.

When these methods are naively applied to extended 3D latent generation, however, we found that they fail to generate 3D scenes with high fidelity due to the unique dynamics of the model’s image-condition, 3D, and object centrality. For instance, the floor vanishes, or poorly correlated patches

lead to repeated objects. We therefore provide structure priors to generate a high-fidelity 3D scene.

Generation with priors. Several studies are trying to provide priors for pre-trained generative models for various purposes. SDEdit [27] is a representative method of image editing that can be applied to [11, 22, 33, 34]. SDEdit partially noise the original image, producing an edited image whose perturbed distributions retain the original image’s style, meeting the intended image style. Readout Guidance [25] trains a small neural network to extract properties (e.g., pose, depth, or edges) from the intermediate latent representation. Then, it computes the loss with respect to the property and provides a loss gradient as guidance, similar to the classifier guidance [5].

We apply SDEdit in our Extend3D to refine the initialized structure. Unlike image editing, we propose an undernoising technique designed for the 3D completion task. Also, instead of guidance, we optimize the intermediate latent with a loss explicitly designed for 3D scene generation, assuming that the priors have ground-truth knowledge of 3D structure and texture.

### 3. Preliminaries

#### 3.1. Latent Flow Model for 3D Generation

A modern approach for high-quality 3D generative models is the latent flow model. They use voxelized latents of fixed size or set-based latents (e.g., point clouds) within a confined region to represent 3D space. While our approach is not restricted to a specific generative model, it can be applied to general voxel-based latents or set-based latent flow models. We illustrate our idea using Trellis [44], which is one of the leading 3D generative models.

Trellis generates 3D representations with two steps of latent flow models given a condition CI encoded from an image I by DINOv2 [31], and both steps are generalizable to flow models for voxelized or set-based latents. The first step of the model generates a sparse structure (SS) {pi} ⊂ [M]3 (where [M] := {0,1,...,M − 1}), which represents a set of occupied coordinates in a voxel grid. In sparse structure generation, low-resolution voxelized noise ZSS1 ∈ RN×N×N is denoised to ZSS0 with vector field vSS, decoded with decoder D, and activated voxel coordinates are collected as:

d dt

ZSS1 ∼ N(0,I),

ZSSt = vSS(ZSSt ,CI,t), (1) {pi} = {p : D(ZSS0 )p > 0}. (2)

As the decoder is trained as a VAE, there is a trained encoder E that encodes the occupancy grid O ∈ RM×M×M into a low-resolution latent representation. The second step of the model conducts denoising on a structured latent

[Figure 8]

Optimize with (Sec. 4.3)

[Figure 9]

× step

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Overlapping Patch-wise Flow

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

|[Figure 20]|
|---|

Monocular Depth Estimator

(Fig. 3) Extended Vector Extended Latent

Extended Latent

Point Cloud Image

[Figure 21]

[Figure 22]

[Figure 23]

Output

Initialization

Priors

Sparse Structure Denoising (Sec. 4.1)

Optimize with (Sec 4.3)

[Figure 24]

×

× step

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Overlapping Patch-wise Flow

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

denoise decode

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

| |
|---|

noise to 𝑡 encode

[Figure 54]

(Fig. 3) Extended Vector Extended Latent

Extended Latent

Occupancy Voxel

Extended Latent

Extended Latent

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

2. Structured Latent Generation (Sec. 4.1)

1. Sparse Structure Generation (Iterative SDEdit) (Sec. 4.2)

- Figure 2. An overall pipeline of our Extend3D. Extend3D consists of two parts: sparse structure generation and structured latent generation. In the denoising part of both steps, an overlapping patch-wise flow was used (Sec. 4.1 and Fig. 3). In sparse structure generation, iterative SDEdit is used to initialize the structure (Sec. 4.2). Vector fields in both steps are optimized with priors (Sec. 4.3).

(SLAT), where a set-based latent feature is matched to a coordinate of sparse structure as:

generation, we extended the 3D latents of a pre-trained object-centric 3D generative model [44] to represent more detailed, larger 3D scenes. We extend the latents in the x and y coordinates, and a portion of the extended latent serves as

ZtSLAT = {(pi,zi,t)} ⊂ [M]3 × Rl, (3) zi,1 iid∼ N(0,I),

d dt

- a conventional latent for the pre-trained object-centric 3D generative model.

To handle extended latents, we divide them into overlapping patches, generated simultaneously via separate but coupled denoising paths conditioned on image patches (Sec. 4.1). Additionally, to address the underlying issues of the object-centric model (e.g., vanishing floor, inability to generate sub-scenes, and randomly rotated objects) and to mitigate the problems associated with patch-wise generation (e.g., repeated objects and seams between patches), we incorporate priors into the generation process. We first initialize the scene with a point cloud from a depth estimator and perform iterative under-noised SDEdit. This completes the occluded area and refines the scene while generating the structure (Sec. 4.2). We then optimize the scene at every time step using the point cloud and an image of the entire scene. We also propose a loss function that treats the point cloud as a prior for the voxel-based latent (Sec. 4.3). The overall pipeline is illustrated in Fig. 2 and Sec. A.1.

4.1. Overlapping Patch-wise Flow

In order to generate a detailed 3D structure and texture, we introduce an extended latent for sparse structure ZSSt ∈ RaN×bN×N and an extended SLAT ZtSLAT ⊂ [aM] × [bM] × [M] × Rl where Zt can refer to both. Here, a and

- b are extension factors. (From here, we will use ↓ to notate non-extended latents or vectors.)

ZtSLAT = vSLAT(ZtSLAT,CI,t), (4)

with invariant pi and vector field vSLAT. SLAT is then decoded to 3D representations such as 3D Gaussians, radiance field, or mesh by sparse decoders (DGS, DNeRF, and Dmesh), and usually. In this paper, we will use the notations Zt that can refer to both ZSSt and ZtSLAT, and v for vSS and vSLAT for simplicity.

#### 3.2. SDEdit

We introduce SDEdit to refine the initialized structure, treating scene generation as a 3D sub-scene editing task. SDEdit

noises latent of a “guide” (e.g., image to be edited) Z(0g) to Zt

and denoises it to Z0 to get the edited result. With the added noise, the perturbed distribution meets the intended distribution while preserving information in the guidance. Although SDEdit was designed for diffusion models [35], we can integrate it into flow models, with the following equations:

start

= (1 − tstart) · Z(0g) + tstart · ϵ; ϵ ∼ N(0,I), (5)

Zt

start

d dt

Zt = v(Zt,C,t), (6)

where C refers to the editing condition. When tstart increases, the denoising path gets longer, causing the effect of conditioning and generative models to be enlarged.

### 4. Method

We divide these latents into overlapping patches with a division factor d. We refer to the (i,j)-th latent patches as ϕSSi,j(ZSSt ) and ϕi,jSLAT(ZtSLAT). This process can be described

Extend3D is a training-free pipeline that generates a 3D scene from a single scene image. To implement 3D scene

###### Overlapping Patch Sampling (𝝓)

###### Overlapping Patch-wise Flow

Sliding Window 𝕎

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

| | |
|---|---|
| | |
|𝐾⁄𝑑| |

Patchify

𝐾

Image Condition Averaged

[Figure 64]

[Figure 65]

[Figure 66]

Extended Latent

Patch (𝑖,𝑗)

Flow (𝒗𝒕↓) Flow (𝒗𝒕↓) Flow (𝒗𝒕↓)

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

𝜙

Σ𝜙

[Figure 71]

[Figure 72]

[Figure 73]

###### Overlapping Patch Coupling (𝚺𝝓 𝟏)

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

𝜙 , 

Zero Padding

| | | |
|---|---|---|
| | | |

Extended Latent 𝒁

Extended Vector 𝒗𝒕

[Figure 79]

[Figure 80]

[Figure 81]

Flow (𝒗𝒕↓)

[Figure 82]

Overlapping Patches (Latent)

Overlapping Patches (Vector)

Extended Latent

Patch (𝑖,𝑗)

- Figure 3. Overlapping patch-wise flow. The extended latent is divided into latent patches with the sliding window. We then obtain the patch vector for each latent patch and merge them into a single extended latent vector, thereby coupling the patches.

as a N3 or M3-sized sliding window W moving with stride N/d or M/d to sample patches, illustrated as sampling in Fig. 3. The patches can be mapped back to their original positions by setting the values at the other positions to zero (zero padding), thereby coupling them, as illustrated in Fig. 3. We represent these inverse mappings as (ϕSSi,j)−1 and (ϕi,jSLAT)−1. We leave the rigorous definitions of the mappings in Sec. A.5.

consistent global structure. We apply dilated sampling during the sparse structure generation phase and leave the details to Sec. A.3.

#### 4.2. Initialize with Prior

When directly denoising sparse structure from pure Gaussian noise using Eq. (9), all patches fail to initialize each sub-scene due to the inherent limitation of the object-centric models. Moreover, the coarse structure is determined during the early denoising stage [42], before the patches are sufficiently coupled, so that the image condition and the 3D latent are not well spatially aligned. Consequently, the output becomes noisy, fragmented, and unstable as in Fig. 7 (B). This motivates the need for a robust structural prior at initialization.

We also patchify the image condition with ψi,j, which crops the image region to exactly match the (i,j)-th 3D patch (see details in Sec. A.2). Similar to MultiDiffusion, we get the vector field of the extended latents by merging the vector fields for each patch, where the overlapping regions are averaged across the patches, as illustrated in the left side of Fig. 3. The entire overlapping patch sampling, merging, and denoising process can be formulated as:

Inspired by 3DTown [49], we initialize the scene structure with a point cloud P extracted from a monocular depth estimator. Specifically, we adopt MoGe-2 [38, 39] for our Extend3D. The predicted point cloud is voxelized into an occupancy grid O0 ∈ RaM×bM×M. Because the monocular depth estimator cannot infer the occluded regions, the resulting occupancy voxel grid contains empty areas that should be rectified using the pre-trained generative model. To address this, with an encoded voxel grid Z(0g) = E(O0), our Extend3D performs SDEdit. Unlike standard SDEdit,

vi,j(Zt, I, t) = v↓ ϕi,j(Zt), Cψ

i,j(I), t , (7) v(Zt, I, t) =

ϕ−i,j1 vi,j ⊘

, (8)

1W

i,j

i,j

i,j

d dt

Zt = v(Zt,I,t), (9)

where ⊘ is an element-wise division. Equation (7) can be calculated independently from the other patches and in parallel, but the dynamics of different patches, even far away, can be coupled by overlaps.

① ② ③

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

The advantage of divided but coupled dynamics is the ability to refine errors in other patches. By detecting slight movement of the sliding window, our method can identify local information from changes in the image and in latent features between patches. Additionally, because some objects are at the centers of patches, we can leverage the object-centric model more effectively. The beneficial effect of overlapping patch-wise flow can be found in Fig. 7 (A).

[Figure 90]

[Figure 91]

[Figure 92]

|𝒁 𝒁<br><br>𝑡 𝑡<br><br>① noise ②<br><br>③ denoise|
|---|

Figure 4. Motivation of under-noising. The blue arrows represent actual noising or denoising, while the purple arrow illustrates how the model is presumed to perceive.

Noted in DemoFusion [6], AccDiffusion [21], and CutDiffusion [20], dilated sampling is crucial for generating a

which applied Eq. (5), we introduce under-noising:

= (1−tnoise)·Z(0g)+tnoise·ϵ, ϵ ∼ N(0,I), (10)

##### Zt

start

where tstart > tnoise, ensuring that the latent is denoised more aggressively than it was originally noised. By undernoising the guide structure, the pre-trained model may treat missing or occluded parts as additional noise, illustrated as the arrow 2 in Fig. 4. Finally, the denoising process, represented as arrow 3 , allows such areas to be filled. This is similar to adding high-frequency noise to enhance image detail in image super-resolution [12]. We empirically validate this choice in Sec. 5.4.

SDEdit can fill the unwanted empty areas. However, it often fails to fully complete the scene, leaving some holes. To mitigate this, as a single SDEdit process partially refines the structure, we apply SDEdit iteratively as: On = SDEdit(On−1), represented in Fig. 2. This process iteratively fills the occluded region of O0 and eventually completes the scene.

#### 4.3. Optimize with Prior

During denoising, sub-scenes deviate from a scene-like structure toward an object-like structure due to the objectcentric model’s properties, leading to distortion or a vanishing floor, even with proper initialization. To prevent deviation and to align the denoising paths with the conditioning, we optimize the extended latents over time steps using the point cloud and the image. When solving Eq. (9) with the discrete ODE solver, instead of moving directly along v(Zt,I,t), we use vˆt, an optimized vector starting from v(Zt,I,t). By optimization, we can leverage the pretrained model for the occluded region while optimizing on the seen region, as in Readout Guidance [25]. In addition, optimizing the vector field can improve consistency across patches by simultaneously optimizing the entire scene, as in [17]. We introduce two optimization losses, one for sparse structure generation and one for structured latent generation, as explained in Sec. 3.1 and illustrated in Fig. 2.

##### In the sparse structure generation step, we define:

1 |P| p∈P

log σ((D(ZSSt − t · vˆt))p), (11)

LSS = −

where σ is a sigmoid function. The loss function is designed to enforce that the initialized voxels do not disappear during the denoising process, motivated by binary cross-entropy loss. It gives a positive signal on predicted voxels where points exist. Voxels with dense point clouds will have more weight in the loss. While this loss can be minimized by increasing the number of voxels, combined with the pretrained model every time step, it merely prevents the desired voxels from disappearing, rather than creating undesired voxels. Moreover, for the same reason, it can smoothly

connect the point cloud priors and generated voxels, not just by attaching two distinct voxel grids. With the LSS, we optimize vˆt with Adam optimizer [14].

In the structured latent generation step, we apply the extended rendering loss [40, 46] as follows:

Iˆ = Render(DGS(ZtSLAT − t · vˆ),P), (12) LSLAT = LPIPS(Iˆ,I) − SSIM(Iˆ,I), (13)

where Render is a differentiable renderer (such as Gaussian splatting) and P is a camera parameter of an image viewpoint provided by the depth estimator. This optimizes the entire scene with an image in the original camera view. Because an object-centric model often loses details in scene textures, this optimization helps refine them. Also, it makes the seams invisible because the boundary is optimized at every time step, ensuring paths are more consistent with each other. With LSLAT, we also optimize vˆt with Adam.

Please refer to Sec. A.1 for the algorithm details.

### 5. Experiments

- 5.1. Human Preference Table 1. Human preference win rate (%) of our method.

versus. Geometry Faithfulness Appearance Completeness Trellis [44] 50.0 66.4 67.1 62.1

Hunyuan3D [47] 73.6 75.7 75.0 75.0 EvoScene [48] 87.1 87.9 87.1 87.1

To score the visual aestheticity of 3D scenes, we conducted a human preference study. We compared our method with Trellis [44] and Hunyuan3D-2.1 [47], the current best open-sourced 3D generation models, and EvoScene, which is specifically designed for large-scene generation. Human annotators (10 participants) ranked the methods on four criteria: geometry, faithfulness, appearance, and completeness, given 14 images and 3D scenes. As a result (Tab. 1), our Extend3D outperformed previous methods in four criteria.

- 5.2. Quantitative Results

Table 2. Quantitative results.

LPIPS ↓ SSIM ↑ PSNR ↑ CD ↓ F-score (0.05) ↑ Trellis [44] 0.650 0.239 10.0 0.0315 0.442

Hunyuan3D [47] 0.683 0.255 10.4 0.0192 0.567

EvoScene [48] 0.482 0.310 13.2 0.0188 0.498 Ours w/o LSLAT 0.400 0.333 13.8 0.0078 0.708

Ours 0.240 0.611 20.4 0.0086 0.694

We render the 3D scene into the camera view of the input image using the camera parameter estimator [39], and obtain LPIPS, SSIM, and PSNR scores on 100 input images [9, 15, 19, 30, 41] spanning diverse wide scenes. As shown in Tab. 2, our method achieved the best scores across

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

Ours EvoScene Trellis Hunyuan3D-2.1

Figure 5. Qualitative result of our Extend3D. Our 3D scene generation result (with a = b = 2) is compared to the results of state-ofthe-art 3D generative models. While previous methods may not accurately represent the image or lose scene details, our method effectively expresses the image condition in 3D. The input image is generated using Flux.1 [dev] [15]. We provide additional results in Sec. A.7.

[Figure 103]

[Figure 104]

Ours SynCity

Figure 6. Qualitative comparison with SynCity. The results are generated from the text prompt, medieval market.

three metrics, indicating that it is most faithful to the input image in terms of structure and texture. Using 45 images and ground-truth mesh pairs from the UrbanScene3D dataset [19], we evaluated the geometric results using the Chamfer Distance (CD) and F-score with a threshold of 0.05. Table 2 shows that our method surpasses the results of the previous methods.

Table 3. Comparison between 3D scene generation methods.

CLIP ↑ HPSv3 ↑ Intra-LPIPS ↓ SynCity [7] 0.251 3.254 0.631

Ours 0.276 3.519 0.571

We also compared our method with the state-of-theart training-free 3D scene generation pipeline, SynCity. Since SynCity is text-conditioned, whereas ours is imageconditioned, we first generated scene images from keywords using ChatGPT [30] and then used Extend3D. We render the results of SynCity and Extend3D to get CLIP score [32], HPSv3 [26], and Intra-LPIPS [17]. Here, IntraLPIPS refers to LPIPS between patches within a single scene, measuring the patch consistency. Table 3 shows that our Extend3D is superior to SynCity in text compatibility, quality, and patch-wise consistency.

#### 5.3. Qualitative Results

Figure 1 shows the scalability of our method. Given a townscale scene image, the extended latent can fully capture details, including landmarks and small buildings, and produce

a 36× larger result than the original latent space. We present more examples of wide scenes in Fig. 17 and Fig. 18. Also, our Extend3D can represent general scenes. It can generate a town, a table of foods, a study scene, and an indoor room, illustrated in Fig. 5 and Sec. A.7. In diverse cases, our method outperformed previous 3D generative models. Also, compared with SynCity, our method can generate scenes without patch boundaries, as shown in Fig. 6 and Fig. 15. Moreover, because SynCity generates a scene in an outpainting approach, it cannot refine the unnatural edge of the water. Compared to EvoScene in Fig. 16, our results had less distorted geometry and more detailed textures.

#### 5.4. Ablation Study

We conducted an ablation study on three proposed methods in our Extend3D and presented the results in Fig. 7. When we obtained the results for Fig. 7 (A) and (B), we did not optimize the latent to emphasize the structural difference.

Overlapping Patch-wise Flow. We claim that the coupled paths of patches can mutually rectify and effectively capture local information. To validate this argument, we compare results for varying division factors. As illustrated in Fig. 7 (A), d = 2 distorted local structure, while d = 4 did not. These results demonstrate that patch interactions correct each other, and the sliding window’s stride enables the extended latent to capture finer details.

Initialize with Prior. In the first part of Fig. 7 (B), the re-

[Figure 105]

###### (A) Overlapping Patch-wise Flow

(B) Initialization with Priors

[Figure 106]

|[Figure 107]|
|---|

[Figure 108]

|[Figure 109]|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

𝑑 = 2 𝑑 = 4

w/o Initialization w/ Initialization

[Figure 113]

(C) Optimization with Priors

𝒕𝐧𝐨𝐢𝐬𝐞 = 𝟎.𝟔 𝒕𝐬𝐭𝐚𝐫𝐭 = 𝟎.𝟖

|[Figure 114]| |
|---|---|
| | |

|[Figure 115]|
|---|

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

|[Figure 120]| |
|---|---|
| | |

|[Figure 121]|
|---|

𝑡 = 0.6 𝑡 = 0.6

[Figure 122]

𝑡 = 0.8 𝑡 = 0.8

+SS Optimization +SLat Optimization

Figure 7. Ablation study. All the images, except for the ablation of under-noising, are taken from the input image camera viewpoint. We set a = b = 2 to generate the 3D scenes in this figure.

Table 5. Ablation study on prior initialization and optimization

###### Table 4. Ablation study for varying division factor d.

LPIPS ↓ SSIM ↑ PSNR ↑ CD ↓ F-score (0.05) ↑

LPIPS ↓ SSIM ↑ PSNR ↑ CD ↓ F-score (0.05) ↑

p.w. flow only 0.606 0.209 9.63 0.0348 0.261 + initialize 0.425 0.312 13.0 0.0083 0.693 + SS optim. 0.400 0.333 13.8 0.0078 0.708 + SLAT optim. 0.240 0.611 20.4 0.0086 0.694

d = 2 0.251 0.598 19.8 0.0088 0.692 d = 4 0.240 0.611 20.4 0.0086 0.694 d = 8 0.237 0.615 20.5 0.0079 0.699

###### Table 6. Ablation study on under-noising.

tnoise / tstart LPIPS ↓ SSIM ↑ PSNR ↑ CD ↓ F-score (0.05) ↑

sults with and without initialization are compared. Without initialization (i.e., tstart = 1), the structure is totally broken with important objects disappeared, buildings not in proper position, etc. We therefore conclude that proper initialization is essential for extended latents. In the second part, we compared three results with different tnoise and tstart. When tnoise = tstart (usual SDEdit), the structure maintained the holes in the initial point cloud or was destroyed due to the tstart trade-off in SDEdit. However, with tnoise < tstart (under-noising), the occluded region of the initial structure is completed naturally.

0.6 / 0.6 0.388 0.324 13.4 0.0081 0.657 0.8 / 0.8 0.550 0.216 9.91 0.0292 0.378

0.8 / 0.6 (over-noise)

0.622 0.219 9.79 0.0518 0.249 0.6 / 0.8 (under-noise)

###### 0.387 0.327 13.5 0.0078 0.680

hances the quality of the scenes. From the geometric results in Tab. 5, we find that initialization and SS optimization refine the geometry, whereas SLAT optimization sometimes degrades it. Since SLAT optimization usually enhances a 3D scene’s texture, there is a trade-off between geometry and texture when optimizing the SLAT. Table 6, conducted without optimization and with niter = 1 as in the qualitative experiment, demonstrates that under-noising is the best choice of tstart and tnoise in 3D completion.

Optimize with Prior. Figure 7 (C) shows the ablation study for optimization with priors. Starting from the base model, we sequentially added sparse structure and SLAT optimization. Without sparse structure optimization, the floor and parts of the objects vanished, as in Sec. A.7. Structured latent optimization could refine seams and distortion between patches compared to those without optimization. Furthermore, the overall quality of the scene’s structure and texture improved (e.g., the fork and chips in the figure).

### 6. Conclusion

We propose a training-free 3D scene generation pipeline, Extend3D. By the extended latent space of the pre-trained object-centric model, we enabled scalable 3D scene generation. We demonstrate that our method (overlapping patchwise flow, initialization, and optimization) and its schemes (iterative SDEdit, under-noising, and 3D-aware optimization objectives) achieve notable improvements in image-

In addition, as in Sec. 5.2, we quantitatively evaluated the generated scenes. Table 4 and Tab. 5 show the effectiveness of our proposed methods, consistent with the qualitative observation that increasing d and providing priors en-

guided 3D scene generation.

Limitations. We found three limitations in our method. Firstly, occluded region completion is sometimes incomplete, for example, the one representing a room in Sec. A.7. Secondly, SLAT optimization requires considerable memory, especially for large scenes (computational cost analysis is provided in Sec. A.4). Lastly, our framework shows limited performance on street-level images. The problem is due to a significant mismatch between the scales of the x and y coordinates, arising from the vanishing points. It would be a direction for future work to implement 3D generation from a wider range of image types.

### Acknowledgements

This work was supported by the IITP grant funded by MSIT [NO.RS-2021-II211343: AI Graduate School (Seoul National University) (5%), NO.RS-2025-25442338: AI Star Fellowship (45%), and NO.RS-2025-02303703: Realworld multi-space fusion and 6DoF free-viewpoint immersive visualization for extended reality (50%)].

### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. In ICML, 2023. 2, 3
- [2] Hanke Chen, Yuan Liu, and Minchen Li. Trellisworld: Training-free world generation from object generators. arXiv preprint arXiv:2510.23880, 2025. 13
- [3] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In CVPR, 2022. 2
- [4] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2
- [5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3
- [6] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no $$$. In CVPR, 2024. 2, 3, 5
- [7] Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. In ICCV, 2025. 2, 3, 7, 11, 20
- [8] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. IJCV, 129(12):3313–3337, 2021. 2
- [9] Google. Google earth. https://earth.google.com/ web, 2025. 1, 6, 22, 23, 24
- [10] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng

- Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In ICLR, 2024. 2
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [12] Jinho Jeong, Sangmin Han, Jinwoo Kim, and Seon Joo Kim. Latent space super-resolution for higher-resolution image generation with diffusion models. In CVPR, 2025. 6
- [13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 42(4), 2023. 2
- [14] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 6
- [15] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 6, 7, 15, 24
- [16] Han-Hung Lee, Qinghong Han, and Angel X. Chang. Nuiscene: Exploring efficient generation of unbounded outdoor scenes. In ICCV, 2025. 2
- [17] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. In NeurIPS, 2023. 2, 6, 7
- [18] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025. 2
- [19] Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and Hui Huang. Capturing, reconstructing, and simulating: the urbanscene3d dataset. In ECCV, 2022. 6, 7, 21, 24, 25
- [20] Mingbao Lin, Zhihang Lin, Wengyi Zhan, Liujuan Cao, and Rongrong Ji. Cutdiffusion: A simple, fast, cheap, and strong diffusion extrapolation method. arXiv preprint arXiv:2404.15141, 2024. 2, 3, 5
- [21] Zhihang Lin, Mingbao Lin, Meng Zhao, and Rongrong Ji. Accdiffusion : An accurate method for higher-resolution image generation. In ECCV, 2024. 2, 3, 5, 11
- [22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 2, 3
- [23] Yuheng Liu, Xinke Li, Xueting Li, Lu Qi, Chongshou Li, and Ming-Hsuan Yang. Pyramid diffusion for fine 3d large scene generation. In ECCV, 2024. 2
- [24] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In CVPR,

2022. 3

- [25] Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski. Readout guidance: Learning control from diffusion features. In CVPR, 2024. 3, 6
- [26] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In ICCV, 2025. 7
- [27] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2, 3

- [28] Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. Lt3sd: Latent trees for 3d scene diffusion. In CVPR, 2025. 2
- [29] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2
- [30] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 24
- [31] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024. 3
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 7
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [34] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [35] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 4
- [36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3
- [37] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 3
- [38] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In CVPR, 2025. 5
- [39] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong

- Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. In NeurIPS, 2025. 2, 5, 6
- [40] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 6
- [41] Joey Wilson, Jingyu Song, Yuewei Fu, Arthur Zhang, Andrew Capodieci, Paramsothy Jayakumar, Kira Barton, and Maani Ghaffari. Motionsc: Data set and network for realtime semantic mapping in dynamic environments. RA-L, 7

(3):8439–8446, 2022. 6, 14, 24

- [42] Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. In WACV, 2025. 2, 5
- [43] Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, and Pan Ji. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. TOG, 43(4),

2024. 2

- [44] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In CVPR, 2025. 2, 3, 4, 6, 11
- [45] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. TOG, 42(4):1–16,

2023. 2

- [46] Richard Zhang, Phillip Isola, Alexei Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [47] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 2, 6, 11
- [48] Kaizhi Zheng, Yue Fan, Jing Gu, Zishuo Xu, Xuehai He, and Xin Eric Wang. Self-evolving 3d scene generation from a single image. arXiv preprint arXiv:2512.08905, 2025. 3, 6, 11, 21
- [49] Kaizhi Zheng, Ruijian Zhang, Jing Gu, Jie Yang, and Xin Eric Wang. Constructing a 3d town from a single image. arXiv preprint arXiv:2505.15765, 2025. 2, 3, 5

## Extend3D: Town-Scale 3D Generation Supplementary Material

### A. Appendix

#### A.1. Algorithms

- Algorithm 1 Sparse Structure Generation

- 1: Input: I, P
- 2:
- 3: O0 ← 1P
- 4: Define schedule
- 5: [tstart = t1 > ... > tk = 0]
- 6: for 0 ≤ n < niter do
- 7: Z(0g) ← E(On)
- 8: Sample ϵ ∼ N(0,I)
- 9: Zt

1 ← (1 − tnoise) · Z(0g) + tnoise · ϵ

- 10: for 1 ≤ m < k do
- 11: Initialize vˆ ← v(Zt

m

,I,tm)

- 12: Optimize vˆ with Adam
- 13: Zt

m+1 ← Zt

m

+ (tm+1 − tm) · vˆ

- 14: end for
- 15: On+1 ← (D(Z0) > 0)
- 16: end for
- 17: return {p : (On

iter

)p > 0}

- Algorithm 2 Structured Latent Generation

- 1: Input: {pi},I,P
- 2:
- 3: Initialize Z1 with {pi}
- 4: Define schedule [1 = t1 > ... > tk = 0]
- 5: for 1 ≤ m < k do
- 6: Initialize vˆ ← v(Zt

m

,I,tm)

- 7: Optimize vˆ with Adam
- 8: Zt

m+1 ← Zt

m

+ (tm+1 − tm) · vˆ

- 9: end for
- 10: return Z0

- Algorithm 3 Extend3D

- 1: Input: I
- 2:
- 3: P,P ← MoGe2(I)
- 4: {pi} ← SS(I,P)
- 5: Z ← SLAT({pi},I,P)
- 6: return (DGS(Z),Dmesh(Z))

#### A.2. Image Patchification

We precisely patchify an input image using point cloud coordinates extracted from the monocular depth estimator. We can map a pixel Ix,y to a coordinate of the point cloud qx,y in the extended latent space. When qx,y ∈ Wi,j, the 3D

area corresponding to the given pixel is in the patch (i,j) since the structure was initialized with the depth estimator. We therefore define the image patch ψi,j(I) by collecting all Ix,y where corresponding qx,y ∈ Wi,j, setting the other pixels to be black, and cropping out black regions so that the image becomes square.

#### A.3. Dilated Sampling

We follow the recipe for dilated sampling in [21]. In this section, we assume that the unextended latent shape is K × K × K for the sake of generalization. We divide the extended latent into K ×K non-overlapping patches so that the size of each patch is a×b×K. We then randomly sample a pillar of 3D latent in each patch. The sampled pillars are attached by maintaining their relative positions to be a K×K×K shaped latent. We sample a×b samples without replacement, and we call them dilated samples. The dilated samples are passed through the pre-trained model with the image condition CI without image patchification. When dilated sampling is applied, Eq. (8) is altered to be:

v(Zt,I,t) = (1 − γt)PatchWise(Zt,CI,t)

+ γtDilated(Zt,CI,t),

(14)

γt = 0.5cosα(π − πt) + 0.5, (15)

where PatchWise is equal to Eq. (8) and α is a hyperparameter. We set α = 5 in our experiments. We use dilated sampling only for sparse structure generation because we empirically observed that it worsens texture when applied to structured latent generation.

#### A.4. Computational Cost

###### Table 7. Computational costs of 3D generation methods. (a = b = 2, d = 4, niter = 5)

VRAM (GB) Time (min) Trellis [44] 17 0.06

Hunyuan3D [47] 56 1.78

SynCity [7] 49 52.0 EvoScene [48] 68 35.3

Ours 28 14.1

We compared the computational cost of our method with previous methods in Tab. 7, using an NVIDIA RTX Pro 6000. Although our pipeline is heavier than object-centric methods (Trellis or Hunyuan3D), it requires substantially less memory and time than other scene-level pipelines.

Memory cost. For a = b = 2 and d = 4 case, the peak GPU memory used for our method is 28GB. Most of the memory is utilized in SLAT optimization process. Without SLAT optimization, it took 14GB VRAM. Also, for a = b = 6 and d = 4, without the SLAT optimization, the peak memory was 61 GB. Table 2 shows that our Extend3D achieves promising results even without SLAT optimization. Therefore, we can ensure reasonable quality with limited GPU resources.

Inference time. The computational complexity of our method is O(abd2niter). A larger (larger a and b), better detailed (larger d), and more complete (larger niter) scene requires more inference time, so there is a trade-off between output quality and inference time.

#### A.5. Rigorous Definition

Here, we provide a detailed and rigorous definition of ϕSSi,j, ϕi,jSLAT, (ϕSSi,j)−1, and (ϕi,jSLAT)−1 for clarity. Firstly, we define patchification ϕi,j through sliding windows as:

WKi,j =

iK d

,

iK d

+ K ×

jK d

,

jK d

+ K ×[K], (16)

ϕSSi,j(ZSSt ) = (ZSSt )WN

i,j

∈ RN×N×N, (17)

ϕi,jSLAT ZtSLAT = p − iMd , jMd , 0 ,zt :

(p,zt) ∈ ZtSLAT, p ∈ WMi,j .

(18)

The subtraction in Eq. (18) is for coordinate normalization into [M]3. Secondly, the inversions of these processes are defined as:

(ϕSSi,j)−1(X)x,y,z = 1(x,y,z)∈WN

i,j

d ,y−jNd ,z, (19)

· Xx−iN

(ϕi,jSLAT)−1(X) = X + iMd , jMd , 0 ,0

∪ (p,0) : p ∈ {pi}, ∀z (p,z) ∈/ X ,

(20) setting the value of other positions to zero.

#### A.6. Ablation on Iterative SDEdit

We evaluated geometric results of different values of niter. The result with niter = 0 represents the decoded SLAT with geometry from the monocular depth estimator. Compared to the niter = 0 case, with a single under-noised SDEdit step, the geometry improved noticeably, indicating

Table 8. Ablation study on varying number of iterations niter.

CD ↓ F-score (0.05) ↑

- niter = 0 0.0079 0.647

- niter = 1 0.0075 0.681

- niter = 2 0.0077 0.688

- niter = 3 0.0082 0.677

[Figure 123]

[Figure 124]

[Figure 125]

|[Figure 126]|
|---|

|[Figure 127]|
|---|

[Figure 128]

𝑛 = 1 𝑛 = 5

Figure 8. Qualitative effect of iterative SDEdit

that our method can address incompleteness in depth estimation. This also implies that our method is robust to the minor errors of the depth estimator. While a single SDEdit step improved results, additional SDEdit steps sometimes degraded output quality, as shown in Tab. 8. Since the qualitative results, such as Figure 8, require iterations to complete the geometry, there is a trade-off between scene completion and geometric detail.

#### A.7. More Results

We provide additional comparisons and large scene reconstruction results in Fig. 9, Fig. 10, Fig. 11, Fig. 12, Fig. 13, Fig. 14, Fig. 15, Fig. 16, Fig. 17, and Fig. 18.

#### A.8. Quantitative Evaluation Set

The evaluation sets used for appearance evaluation (LPIPS, SSIM, PSNR) and for geometry evaluation (CD, F-score) are illustrated in Fig. 19 and Fig. 20, respectively. We used the keywords from the original SynCity research for a textprompted generation comparison. The keyword list is as follows: autumn, campsite, campus, cyberpunk, mars, medieval market, medieval, oasis, paris, SF, solarpunk, and suburban. Given a keyword, we used a prompt below for image generation before Extend3D:

Create a realistic, high-resolution aerial (bird’seye) view of {keyword}. The image should be perfectly framed so that no buildings, objects, or landscape features are cropped or cut off. Show the entire scene clearly from directly above, with natural lighting, realistic depth, and detailed textures. The result should look like a professional drone photograph taken at high altitude, with balanced composition and clean edges.

#### A.9. SLAT Decoding

While our main contribution is on the latent generation and most of our results in the paper are represented in 3D Gaussian Splatting, we also introduce an overlapping-patch SDF representation for mesh decoding. Unlike 3D Gaussians, if mesh patches are decoded disjointly and simply attached, the seams will be discontinuous, because there will be no faces. Therefore, similar to the overlapping patch-wise flow in our paper, we first decode SLAT into SDFs in an overlapping patch-by-patch manner. These SDF patches are then combined into a single SDF by averaging the values in the overlapping regions. Inspired by TRELLISWorld [2], we apply a cosine-based weighting to avoid near-zero SDF values at the seams. Finally, we extract the mesh from the combined SDF in a single step.

Unlike meshes, we decode SLAT into 3D Gaussians with disjoint SLAT patches, since it is not necessary to consider connectivity in 3D Gaussians.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Ours

EvoScene

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Trellis Hunyuan3D-2.1

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

EvoScene

Ours

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Trellis Hunyuan3D-2.1

- Figure 9. Example results for diverse images. In this figure, we set a = b = 2. We compared our results with state-of-the-art open-source 3D generative models. The second image is from CarlaSC dataset [41]. The other image is generated by ChatGPT [30].

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Ours

EvoScene

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Trellis

Trellis Hunyuan3D-2.1

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Ours

EvoScene

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Trellis

Trellis Hunyuan3D-2.1

- Figure 10. Example results for diverse images. In this figure, we set a = b = 2. We compared our results with state-of-the-art opensource 3D generative models. The images are generated by ChatGPT [30] and Flux.1 [dev] [15].

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Ours

EvoScene

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Trellis Hunyuan3D-2.1

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Ours

EvoScene

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Trellis

Hunyuan3D-2.1

- Figure 11. Example results for diverse images. In this figure, we set a = b = 2. We compared our results with state-of-the-art opensource 3D generative models. The images are generated by ChatGPT [30].

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Ours

EvoScene

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Trellis Hunyuan3D-2.1

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Ours

EvoScene

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Trellis Hunyuan3D-2.1

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Ours

EvoScene

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Trellis Hunyuan3D-2.1

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Ours

EvoScene

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Trellis Hunyuan3D-2.1

Figure 13. Example results for diverse images. In this figure, we set a = b = 2. We compared our results with state-of-the-art opensource 3D generative models. The images are generated by ChatGPT [30].

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Ours

EvoScene

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Trellis Hunyuan3D-2.1

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Ours

EvoScene

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Trellis Hunyuan3D-2.1

[Figure 249]

[Figure 250]

Autumn

[Figure 251]

[Figure 252]

SF

[Figure 253]

[Figure 254]

Oasis

[Figure 255]

[Figure 256]

Solarpunk

Ours SynCity

- Figure 15. Example results of comparison to SynCity. We compared our results with SynCity [7]. Given a prompt, we first generated an image with ChatGPT [30] and used Extend3D for our results. The input prompts are on the left.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Ours EvoScene

- Figure 16. Example results of comparison to EvoScene [48]. In this figure, we set a = b = 2. The images are from ChatGPT [30] and UrbanScene3D [19].

[Figure 265]

[Figure 266]

[Figure 267]

###### Figure 17. The large scale result of Extend3D. We generated large scale (a = b = 6) 3D scene from the image of K¨oln captured from Google Earth [9]. We didn’t use SLAT optimization in this result due to the memory shortage.

[Figure 268]

[Figure 269]

[Figure 270]

###### Figure 18. The large scale result of Extend3D. We generated large scale (a = b = 6) 3D scene from the image of Athens captured from Google Earth [9]. We didn’t use SLAT optimization in this result due to the memory shortage.

[Figure 271]

###### Figure 19. 100 images for appearance evaluation. The images are from ChatGPT [30], Flux.1 [dev] [15], CarlaSC [41], Google Earth [9], and UrbanScene3D [19].

[Figure 272]

###### Figure 20. 45 images for geometry evaluation. The images are from UrbanScene3D [19] synthetic scenes, together with ground truth meshes.

