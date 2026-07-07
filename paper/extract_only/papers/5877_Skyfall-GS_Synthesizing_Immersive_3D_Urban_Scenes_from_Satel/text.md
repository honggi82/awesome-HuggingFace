## Skyfall-GS: Synthesizing Immersive 3D Urban Scenes from Satellite Imagery

Jie-Ying Lee1, Yi-Ruei Liu2, Shr-Ruei Tsai1, Wei-Cheng Chang1, Chung-Ho Wu1, Jiewen Chan1, Zhenjun Zhao3, Chieh Hubert Lin4, and Yu-Lun Liu1

1 National Yang Ming Chiao Tung University, 2 UIUC 3 University of Zaragoza, 4 UC Merced jayinnn.cs14@nycu.edu.tw, yulunliu@cs.nycu.edu.tw

# arXiv:2510.15869v3[cs.CV]18Mar2026

[Figure 1]

Input Satellite View

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Generated 3D Facade

Input Multi-view Satellite Imagery Generated 3D City

Fig. 1: Our method synthesizes high-quality, immersive 3D urban scenes solely from multi-view satellite imagery, enabling realistic drone-view navigation without relying on additional 3D or street-level training data. Given multiple satellite images from diverse viewpoints and dates (left), our method leverages 3D Gaussian Splatting combined with pre-trained text-to-image diffusion models in an iterative refinement framework to generate realistic 3D block-scale city from limited satellite-view input (right). Our method significantly enhances visual fidelity, geometric sharpness, and semantic consistency, enabling real-time immersive exploration.

Abstract. Synthesizing large-scale, explorable, and geometrically accurate 3D urban scenes is a challenging yet valuable task for immersive and embodied applications. The challenge lies in the lack of large-scale and high-quality real-world 3D scans for training generalizable generative models. In this paper, we take an alternative route to create largescale 3D scenes by leveraging readily available satellite imagery for realistic coarse geometry and open-domain diffusion models for high-quality close-up appearance synthesis. We propose Skyfall-GS, a novel hybrid

framework that synthesizes immersive city-block scale 3D urban scenes by combining satellite reconstruction with diffusion refinement, eliminating the need for costly 3D annotations, and also featuring real-time, immersive 3D exploration. We tailor a curriculum-driven iterative refinement strategy to progressively enhance geometric completeness and photorealistic texture. Extensive experiments demonstrate that SkyfallGS provides improved cross-view consistent geometry and more realistic textures compared to state-of-the-art approaches. Project page: https://skyfall-gs.jayinnn.dev/

Keywords: 3D Gaussian Splatting · Satellite Imagery · Scene Synthesis

### 1 Introduction

High-quality, immersive, and semantically plausible 3D urban scenes are essential for a wide range of applications, including gaming, filmmaking, navigation planning, and robotics. The ability to create a large-scale and 3D-grounded environment supports realistic rendering and immersive experience for storytelling, demonstration, and embodied physics simulation. However, due to limited 3Dinformed data, building a generative model for realistic and navigable 3D cities remains challenging. It is expensive and labor-intensive to acquire large-scale 3D and textured reconstructions of cities with detailed geometry, while using Internet image collections faces challenges in camera pose registration and excessive data noise (e.g., transient objects and different times of the day). These constraints prevent existing 3D city generation frameworks from creating realistic and diverse appearances. With this observation, we propose an alternative route for virtual city creation with a two-stage pipeline: partial and coarse geometry reconstruction from multi-view satellite imagery, then close-up appearance completion and synthesis using an open-domain diffusion model.

Satellite imagery offers a compelling alternative due to its extensive geographic coverage, automated collection, and high-resolution capabilities. For instance, Maxar’s WorldView-3 satellite captures approximately 680,000 km2 of imagery daily at resolutions up to 31 cm per pixel. Such data encodes a large volume of real-world environment semantics, enabling scalable 3D urban scene creation. However, in Figure 2(a), we show that directly applying 3D reconstruction methods to satellite imagery is insufficient for creating navigable and immersive 3D cities. The substantial invisible regions (e.g., building facades) and limited satellite-view parallax create incorrect geometry and artifacts.

Completing and enhancing the geometry and texture in the ground view requires a significant influx of extra information. In Figure 2(b), we study a few state-of-the-art methods in city generation [101, 103]. These methods produce oversimplified building geometries and unrealistic appearances due to strong assumptions, particularly the reliance on semantic maps and height fields as the sole inputs, and overfitting to small-scale, domain-specific datasets. Such an

Skyfall-GS 3

observation motivates us to leverage an open-domain foundation image generation model as an external information source, providing better zero-shot generalization and diversity. Ground-level novel-view renderings from the GSreconstructed scene suffer from severe degradation, including floater artifacts and texture smearing, caused by insufficiently constrained Gaussians in regions with limited satellite-view parallax. To address this, we employ an open-domain foundation image generation model to directly refine these degenerate renderings into photorealistic and geometrically consistent outputs, exploiting the model’s rich visual priors to recover plausible appearances. The refined outputs serve

- as pseudo ground-truth to supervise iterative GS scene optimizations, progressively improving the visual fidelity and geometric consistency across the scene. To stabilize the convergence, we carefully design a curriculum-based view selection and iterative refinement process, where the sampled view angles gradually fall from the sky to the ground over time. Accordingly, we name our framework Skyfall-GS. In Figure 1 and Figure 2, we show that Skyfall-GS substantially enhances texture with 3D-justified geometry compared to the relevant baselines.

Skyfall-GS is a novel hybrid framework that synthesizes immersive 3D urban scenes by combining satellite reconstruction with diffusion refinement, eliminating the need for fixed-domain training on 3D data. Skyfall-GS operates on readily available satellite imagery as the only input, then synthesizes realistic aerial-view appearances and maintains a strong satellite-to-ground 3D consistency. Moreover, Skyfall-GS supports real-time and interactive rendering. Through experiments on diverse environments, we show that Skyfall-GS has better generalization and robustness compared to state-of-the-art methods. Our ablation study shows that each component improves perceptual plausibility and semantic consistency. Skyfall-GS paves the way for scalable 3D urban virtual scene creation, enabling applications in virtual entertainment, simulation, and robotics.

In summary, our contributions include:

- – We introduce Skyfall-GS, the first method to synthesize immersive, real-time free-flight navigable 3D urban scenes solely from multi-view satellite imagery using generative refinement without domain-specific training.
- – An open-domain generative refinement approach that exploits rich visual priors from pre-trained text-to-image diffusion models to recover photorealistic appearances from degenerate satellite reconstructions.
- – A curriculum-learning-based iterative refinement strategy that progressively enhances reconstruction quality from higher to lower viewpoints, significantly improving visual fidelity in occluded regions.

### 2 Related Work

Gaussian Splatting. 3D Gaussian Splatting (3DGS) [33] offers real-time view synthesis rivaling NeRFs [1–3, 58, 61, 66], with Mip-Splatting [111] addressing scale-change aliasing. In-the-wild variants handle appearance variation and transient objects [10, 14, 26, 35, 76, 92, 104, 113], while scaling to large scenes is addressed via partitioning and LOD [8,21,34,49,86,89]. For sparse-view settings, depth and co-regularization priors guide reconstruction [42,48,67,114,116,121].

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Naïve 3DGS Sat-NeRF Skyfall-GS (Ours) CityDreamer GaussianCity Skyfall-GS (Ours)

###### (a) 3D Reconstruction (b) City Generation

- Fig. 2: Limitations of existing novel-view synthesis methods from satellite imagery. (a) Sat-NeRF [56] and naive 3DGS [33] yield blurred or distorted building facades due to insufficient geometric detail and limited parallax from satellite viewpoints. (b) City generation methods [101,103] produce oversimplified building geometries and unrealistic appearances, primarily due to strong assumptions about the input data, and overfitting to small-scale, domain-specific datasets. In comparison, our method synthesizes more realistic appearances and geometries from aerial views.

Satellite and aerial 3D reconstruction. Classical SfM-MVS pipelines extract DSMs from satellite imagery [5,17,78,115], with neural variants improving geometric fidelity [12,19,41,52,57,72,120]. Sat-NeRF [56] and SatMVS [16,18] apply NeRF and RPC-based warping to satellite imagery respectively, yet neither recovers occluded facades, while FusionRF [84] improves depth via multispectral acquisitions. EOGS [77] adapts 3DGS with affine cameras and shadow mapping for sparse multi-date satellite imagery, and SkySplat [28] outputs Gaussians from satellite imagery in a feed-forward manner. For aerial imagery, AGS [100] introduces Ray-Gaussian Intersection for large-scale surface reconstruction, CityGaussian [54,55] achieves city-scale rendering via scene partitioning and LOD, and Horizon-GS [31] unifies aerial-to-ground reconstruction. However, no existing method jointly handles multi-date appearance variations in satellite imagery while synthesizing ground-level perspectives.

Urban scene synthesis. Cross-view synthesis methods [11, 74, 82, 88, 105, 106, 112] generate ground-level images or videos from satellite imagery without explicit 3D representations, while others employ intermediate 3D structures such as point clouds [45], density fields [69, 71], or voxels [43], and recent methods [27, 32, 44, 70, 107] advance toward explicit 3D generation via diffusion- and lifting-based approaches. Layout-conditioned methods synthesize cities from BEV maps: InfiniCity [47] lifts infinite-pixel BEV maps to 3D via octree voxel completion and neural rendering, CityDreamer [101] introduces compositional generative models separating buildings from backgrounds, and GaussianCity [103] extends this paradigm to 3D Gaussian Splatting [102]; LLM-driven procedural approaches [79, 118, 119] further generate layouts from text, OSM, or semantic maps. Although CityX [118] can accept satellite images as layout guidance, none of these methods faithfully synthesize photorealistic ground-level views directly conditioned on the observed satellite texture.

Diffusion models for 3D reconstruction and editing. Diffusion models [38,75] have emerged as powerful generative priors for image synthesis. Score

[Figure 18]

###### Skyfall-GS 5

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Episode

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

RefinedRender

[Figure 38]

1

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Update

[Figure 43]

[Figure 44]

[Figure 45]

###### Pseudo Camera Depth Supervision

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Text-to-Image Diffusion Model

…

[Figure 54]

…

[Figure 55]

[Figure 56]

[Figure 57]

Current3DGSRender

𝑁

[Figure 58]

Render

Input: Multi-View Satellite Images

(c) Render Refinement

(a) Reconstruction Stage: Reconstruction from Satellite Image

(b) Synthesis Stage: Curriculum-Based Iterative Dataset Update

- Fig. 3: Overview of the proposed Skyfall-GS pipeline. Our method synthesizes immersive and free-flight navigable city-block scale 3D scenes solely from multi-view satellite imagery in two stages. (a) In the Reconstruction Stage, we first reconstruct the initial 3D scene using 3DGS, enhanced by pseudo-camera depth supervision to address limited parallax in satellite images. We integrate an appearance modeling to handle varying illumination conditions across multi-date satellite images. (b) In the Synthesis Stage, we introduce a curriculum-based Iterative Dataset Update (IDU) refinement technique leveraging (c) a pre-trained T2I diffusion model [38] with prompt-to-prompt editing [36]. By iteratively updating training datasets with progressively refined renders, our approach significantly reduces visual artifacts, improving geometric accuracy and texture realism, particularly in previously occluded areas such as building facades.

distillation pipelines [46,68,93] lift 2D priors into 3D; DreamGaussian [87] and GaussianDreamer [109] extend this to Gaussian Splatting, while MVDream [81] and sparse-view methods [7, 9, 20, 51, 53, 59, 98, 99] address multi-view consistency. Inversion-based methods [36, 60, 64, 65] and occlusion-aware inpainting works [13,15,50,62,63,80,91,95–97,108] enable scene editing and 3D Gaussian inpainting. Instruct-NeRF2NeRF [23] introduced the Iterative Dataset Update (IDU) paradigm, iteratively refining a NeRF via InstructPix2Pix [6] edits; LucidDreamer [9] and WonderWorld [110] adopt IDU for progressive scene expansion via extrapolation, and RealmDreamer [83] generates forward-facing scenes via iterative inpainting and depth diffusion. However, none of these can be naively applied to our setting: inpainting-based approaches treat facade regions as unknown and synthesize them freely, whereas satellite imagery does capture building facades at an oblique angle, providing appearance constraints that must be respected. Ignoring this leads to synthesized facades inconsistent with the satellite observations.

### 3 Method

Our two-stage pipeline (Figure 3) transforms satellite images into immersive 3D cities. In the Reconstruction Stage (Section 3.1), we fit a 3D Gaussian Splatting model, adding illumination-adaptive appearance modeling and regularizers for sparse, multi-date views. In the Synthesis Stage (Section 3.2), we recover occluded regions, e.g., facades, through curriculum-based Iterative Dataset Up-

date, repeatedly refining renders with text-guided diffusion edits. The loop keeps textures faithful to the satellite input while preserving geometry, yielding complete, navigable urban scenes from satellite data alone.

Preliminary. 3D Gaussian Splatting (3DGS) [33] encodes a scene as Gaussians with center µi, covariance Σi, opacity αi, and view-dependent color. Each Gaussian projects to the image plane with covariance: Σi′ = JWΣiWTJT, where W is the viewing transformation and J is the affine-projection Jacobian. Pixels are alpha-composited front-to-back. Parameters are trained with:

Lcolor = λD-SSIM DSSIM(C, Cˆ ) + (1 − λD-SSIM)∥Cˆ − C∥1 . (1)

#### 3.1 Initial 3DGS Reconstruction from Satellite Imagery

The initial 3DGS reconstruction must faithfully preserve the texture and geometry of satellite imagery to provide a robust foundation for synthesis. We employ appearance modeling to handle variations in multi-date imagery. Since limited satellite parallax creates floating artifacts, we apply regularization techniques to constrain both texture and geometry.

Approximated camera parameters. Satellite imagery typically uses the rational polynomial camera (RPC) model, directly mapping image coordinates to geographic coordinates. To integrate with the 3DGS pipeline, we employ SatelliteSfM [115] to approximate perspective camera parameters (extrinsic and intrinsic) from RPC and generate sparse SfM points as initial 3DGS points.

Appearance modeling. Multi-date satellite imagery exhibits significant appearance variations due to global illumination changes, seasonal factors, and transient objects (Figure 3(a)). Following WildGaussians [35], we use trainable per-image embeddings {ej}Nj=1 (with N input images) to handle varying illumination. We also employ trainable per-Gaussian embeddings gi to capture localized appearance changes, e.g. shadow variations. A lightweight MLP f computes affine color transformation parameters (β,γ) as (β,γ) = f(ej,gi,c¯i), where ej is the per-image embedding, gi is the per-Gaussian embedding, and c¯i denotes the zeroth-order spherical harmonics (SH). Let cˆi(r) be the i-th Gaussian’s viewdependent color conditioned on the ray direction r. The transformed color c˜i is computed as c˜i(r) = γ · cˆi(r) + β. To avoid modeling the appearance changes as view-dependent effects, we limit SH coefficients to zero- and first-order terms. At inference, we fuse the learned appearance into a standard 3DGS by selecting a fixed image embedding e∗ and evaluating f(e∗,gi,c¯i) for every Gaussian to compute a static color. The embeddings and MLP are then discarded, yielding a portable representation compatible with standard 3DGS renderers and enabling real-time rendering at 60 FPS (1920×1080) on a MacBook Pro M4 Pro.

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

RefinedRenderRender(b)(a)

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

78° 50° 27° 17°

- Fig. 4: The motivation of curriculum strategy. Renderings of the initial 3D reconstruction from varied elevation angles reveal progressive degradation as the viewing angle decreases.

Fig. 5: Render refinement. (a) Original 3DGS render with artifacts and blurry textures; (b) Refined result showing enhanced geometry and texture quality.

Opacity regularization. We observe that numerous floaters in reconstructed scenes exhibit low opacity. To encourage geometry to adhere closely to actual surfaces, we propose entropy-based opacity regularization:

[αi log(αi) + (1 − αi) log(1 − αi)] . (2)

Lop = −

i

This regularization promotes binary opacity distributions, allowing low-opacity Gaussians to be more aggressively pruned during densification. Incorporating this term significantly sharpens geometric reconstruction, providing a better foundation for subsequent synthesis.

Pseudo camera depth supervision. To further reduce floating artifacts, we sample pseudo-cameras positioned closer to the ground during optimization. From these pseudo-cameras, we render RGB images IRGB and corresponding alpha-blended depth maps DˆGS. We then use an off-the-shelf monocular depth estimator, MoGe [90], to predict scale-invariant depths Dˆest from these renders. We use the absolute value of Pearson correlation (PCorr) to supervise the depth:

Cov(DˆGS, Dˆest) Var(DˆGS)Var(Dˆest)

Ldepth = 1 − ∥PCorr(DˆGS, Dˆest)∥1 ; PCorr(DˆGS, Dˆest) =

.

(3)

Optimization. Combining all components, the overall loss for the reconstruction stage is defined as:

Lsat(G, C) = Lcolor + λopLop + λdepthLdepth , (4)

where G is the 3DGS representation, C is the set of ground-truth satellite images, λop and λdepth weight opacity regularization and depth supervision relative to the color reconstruction loss.

#### 3.2 Synthesize via Curriculum-Based Iterative Datasets Update

The iterative dataset update (IDU) technique [23,59] repeatedly executes renderedit-update cycles across multiple episodes to progressively synthesize 3D scenes. Unlike previous methods that sample camera poses from original training views [23] or simple orbits [59], we introduce a curriculum-based refinement schedule over Ne episodes that specifically addresses the geometric and visual limitations inherent to satellite imagery, producing structurally accurate and photorealistic reconstructions of occluded areas.

Curriculum learning strategy. As illustrated in Figure 4, we observe that 3DGS trained from satellite imagery produces higher-quality renders at higher elevation angles but degenerates at lower elevation angles. Leveraging this insight, we introduce a curriculum-based synthesis strategy, which progressively lowers viewpoints across optimization episodes. Specifically, we define Np look-

- at points {Pj}Nj=1p uniformly placed throughout the scene and uniformly sample Nv camera positions along orbital trajectories with controlled elevation angles and radii. Our iterative dataset update (IDU) process starts at higher elevations and progressively descends toward lower perspectives. This approach gradually reveals previously occluded regions, improving geometric detail and texture realism, as validated in our ablation studies (Section 4.2).

Render refinement by text-to-image diffusion model. As illustrated in Figure 5(a), renderings from initial 3DGS contain blurry texture and artifacts. To address this, we leverage prompt-to-prompt editing with a pre-trained textto-image diffusion model to synthesize disocclusion areas, remove artifacts, and enhance geometry. Prompt-to-prompt editing [24] modifies input images, which are described by the source prompt, to align with the target prompt while preserving structural content. Although originally designed for real or synthetically generated images, we demonstrate its effectiveness for refining degraded 3DGS renders produced from satellite-view training, enabling high-quality appearance enhancement without disrupting the scene’s satellite-consistent geometry. Specifically, we use FlowEdit [36] with the FLUX.1 [dev] diffusion model [39]. Our source prompts describe the original degraded features. The target prompts specify the desired high-quality attributes (details in Supplementary). This approach significantly improves rendering quality. It yields sharper geometry, richer textures, and physically coherent shadows (see Figure 5). This strengthens the 3DGS training dataset for more accurate reconstructions.

Multiple diffusion samples. While diffusion models effectively refine individual 3DGS renders, independently applying the diffusion model across viewpoints introduces inconsistencies. Furthermore, 3DGS is well known to suffer from overfitting on single views, as pointed out by CoR-GS [114], causing artifacts when rendering from novel viewpoints.

Ideally, the optimal denoising diffusion process should produce a distribution where all views maintain consistent 3D appearance. However, independent 2D denoising on each view does not preserve 3D consistency, resulting in a denoising trajectory distribution that is a superset of the optimal trajectories. Selecting a single denoising trajectory from this expanded distribution is unlikely to yield the optimal 3D-consistent result, leading to the artifacts observed in Figure 9(c).

To mitigate this, we synthesize Ns independently refined samples per view, effectively sampling multiple trajectories from the denoising distribution. During optimization, the photometric loss Lcolor implicitly averages over these Ns samples. Rather than committing to a single potentially suboptimal denoising path, this approach allows the 3DGS optimization to find a consensus representation that balances fidelity to individual samples while promoting geometric coherence across views. Ablation studies (Section 4.2) and Figure 9(c) confirm that this strategy successfully balances detail preservation with structural coherence.

Iterative dataset update. Our curriculum-based Iterative Dataset Update (IDU), detailed in Algorithm 1, optimizes the 3DGS over Ne episodes. In each episode, we render curriculum-guided views and refine them using FlowEdit [36] with specified prompts and strengths to generate a new training set. As the curriculum descends to lower altitudes, rendering quality steadily improves, particularly in previously occluded regions, as illustrated in Figure 6. We provide detailed parameters in the Supplementary.

Algorithm 1 3DGS Refinement via Iterative Dataset Updates

Input: Ne, Nv, Ns, Np: Number of episodes, views per point, samples per view, and look-at points. Input: {Ri}Ni=1e ,{Ei}Ni=1e : Radius and elevation sequences; {Pj}Nj=1p : Target look-at points. Input: Φ = (Tsrc, Ttgt, nmin, nmax): FlowEdit parameters. Input: G: Initial 3DGS from satellite-view training. Output: G′: Refined 3DGS.

- 1: G′ ← G
- 2: for i = 1 to Ne do
- 3: cam_views ← OrbitViews({P}, Ri, Ei, Nv) ▷ Generate Np × Nv views
- 4: render_views ← Render(G′, cam_views) ▷ Render RGB images
- 5: refine_views ← Refine(render_views, Φ, Ns) ▷ Refine renders using FlowEdit
- 6: G′ ← Train(G′, refine_views) ▷ Update 3DGS using refined views
- 7: end for
- 8: return G′

Optimization. For each episode i, we optimize the 3DGS using:

LIDU(Gi−1, C˜i) = Lcolor + λdepthLdepth , (5)

where Gi−1 denotes the previous episode’s 3DGS model, and C˜i are the current refined images. We provide more implementation details in Supplementary.

[Figure 117]

[Figure 118]

[Figure 119]

(a) After Reconstruction Stage (b) After Episode 1 (c) After Episode 2

[Figure 120]

[Figure 121]

[Figure 122]

(d) After Episode 3 (e) After Episode 4 (f) After Episode 5 (Final)

- Fig. 6: Visualization of progressive refinement. This figure illustrates the stepby-step evolution of the synthesized 3D scene. Starting from the initial reconstruction state (a), the geometry and texture are progressively refined through successive stages of the iterative process (b-e), culminating in the final high-fidelity result (f).

### 4 Experiments

Implementation details. The Reconstruction Stage runs for 30,000 iterations with λD-SSIM = 0.2, λop = 10, and λdepth = 0.5. The Synthesis Stage comprises Ne = 5 episodes of 10,000 iterations each, with Nv = 6 cameras and Ns = 2 samples per look-at point. Training images are sampled 75% from IDU-refined views and 25% from original satellite views, preserving consistency with the input satellite imagery. All experiments run on a single RTX A6000 (48GB) GPU. Full details are in Supplementary.

Datasets. We evaluate our method on high-resolution RGB satellite imagery from two sources. First, the 2019 IEEE GRSS Data Fusion Contest (DFC2019) [40] featuring WorldView-3 captures of Jacksonville, Florida (2048×2048 pixels, 35 cm/pixel resolution). Camera parameters and sparse points are generated using SatelliteSfM [115]. We evaluate on four standard AOIs: JAX_004, JAX_068, JAX_214, and JAX_260, following Sat-NeRF [56] and EOGS [77]. Second, for geographic diversity, we use the GoogleEarth dataset [101] (training data for CityDreamer [101] and GaussianCity [103]) containing NYC scenes. We use four scenes (004, 010, 219, 336) with training views rendered at 80° elevation to approximate satellite conditions. Google Earth Studio (GES) [22] renders serve as ground truth for both datasets. See Supplementary for more detail.

Baselines. As our method bridges satellite-based 3D reconstruction and city generation, we select baselines from both fields. For satellite reconstruction, we compare with Sat-NeRF [56] and EOGS [77] on DFC2019 (they require RPC

Table 1: Quantitative comparison of different methods on DFC2019 [40]. The results show that our method consistently achieves the best performance, indicating superior perceptual fidelity compared to all baselines. Metrics are computed between renders from each method and reference frames from GES. Red ,

orange , and yellow indicate the best, second best, and third best results, respectively.

Distribution Metrics Pixel-level Metrics Methods FIDCLIP ↓ CMMD ↓ PSNR ↑ SSIM ↑ LPIPS ↓

- 3D Reconstruction Sat-NeRF [56] 86.52 4.788 10.08 0.268 0.862 EOGS [77] 87.67 5.291 7.26 0.168 0.958 CoR-GS [114] 84.95 5.692 11.55 0.351 0.947 Mip-Splatting† [111] 86.72 5.404 11.91 0.319 0.819

Our Approach Ours 27.03 2.110 12.41 0.322 0.790 † Enhanced with our appearance modeling.

Table 2: Quantitative comparison of different methods on GoogleEarth dataset [101]. The results show that our approach consistently achieves the best performance, indicating superior perceptual fidelity compared to all baselines. Metrics are computed between renders from each method and reference frames from GES.

Red , orange , and yellow indicate the best, second best, and third best results, respectively.

Distribution Metrics Pixel-level Metrics Methods FIDCLIP ↓ CMMD ↓ PSNR ↑ SSIM ↑ LPIPS ↓ City Generation

CityDreamer [101] 36.66 4.200 12.58 0.267 0.558 GaussianCity [103] 28.76 2.915 13.41 0.291 0.540

3D Reconstruction

CoR-GS [114] 26.35 3.758 13.35 0.299 0.412 Mip-Splatting [111] 16.09 2.086 14.13 0.302 0.379

Our Approach Ours 10.29 1.959 14.42 0.302 0.393

input unavailable in GoogleEarth), plus Mip-Splatting [111] (with our appearance modeling enabled on DFC2019) and CoR-GS [114] on both datasets. For city generation, we compare with CityDreamer [101] and GaussianCity [103] on GoogleEarth dataset (their training dataset).1 We use official implementations with default settings.

Evaluation metrics. Skyfall-GS primarily aims to synthesize and enhance the invisible or low-coverage regions from the satellite view; therefore, our evaluation focuses on quality and diversity assessment metrics. We report FIDCLIP [37] and CMMD [30] which use the CLIP [73] backbone, as InceptionV3 [85] used in classic FID [25] and KID [4] is unsuitable for modern generative tasks. We complement these with user studies for perceptual quality assessment. We also report pixel-aligned metrics (PSNR [29], SSIM [94], LPIPS [117]) as secondary references. However, these metrics are unsuitable for generative tasks, as the synthetic elements in the obscured regions cannot match the invisible groundtruth. Moreover, on DFC2019, systematic illumination and color gaps between WorldView-3 imagery and GES references make pixel-level scores unreliable.

#### 4.1 Comparisons with Baselines

Quantitative comparison. We evaluate against satellite reconstruction and city generation baselines by dividing rendered frames into 144 patches (512×512

1 Many methods lack available code or models: Sat2Scene [44] (inference only), Sat2Vid [45], EONeRF [57], Sat-DN [52], SatelliteRF [120], Sat-Mesh [72], CrossViewDiff [43], SkySplat [28], MagicCity [107], Sat3DGen [70], Sat2City [27], and Sat2RealCity [32].

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

068214

###### (a)DFC2019(b)GoogleEarth

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Input Satellite Image GES Reference Ours w/o IDU CoR-GS Mip-Splatting

Ours

Sat-NeRF EOGS

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

010336

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Input Satellite Image GES Reference Ours w/o IDU CityDreamer GaussianCity CoR-GS

Ours

Mip-Splatting

- Fig. 7: Qualitative comparison on (a) DFC2019 and (b) GoogleEarth datasets. The leftmost column shows a representative input satellite image. Our method consistently surpasses baselines in geometric accuracy and texture quality at low-altitude novel views. On (a), our synthesis exhibit sharp building contours and detailed rooftops, whereas Sat-NeRF [56] produces severe fog, EOGS [77] yields excessive darkness, CoR-GS [114] suffers from heavy blurring, and Mip-Splatting [111] exhibits prominent floaters and artifacts. On (b), CityDreamer [101] and GaussianCity [103] over-simplify building geometry and miss scene-specific details, CoR-GS [114] and MipSplatting [111] produce blurry facades, while our method delivers sharper geometry, richer facade details, and correctly recovers distinctive features such as the red pavement in scene 010.

pixels). Reference frames are extracted from GES at 17◦ elevation for DFC2019 (30 frames/AOI, 4,320 total images) and 45◦ for GoogleEarth (24 frames/scene, 3,456 total images); all methods are then evaluated on matching videos generated with identical camera parameters. As shown in Tables 1 and 2, our method achieves highly competitive performance on both datasets. While Mip-Splatting attains a slightly better average LPIPS on GoogleEarth, our approach consistently outperforms all baselines across the vast majority of distribution and pixellevel metrics, a result further corroborated by the qualitative and user studies below, demonstrating robust synthesis across diverse urban environments. Notably, CoR-GS achieves a higher SSIM on DFC2019. We attribute this to its tendency to produce overly smooth and blurry reconstructions, which artificially inflate SSIM scores due to SSIM’s known insensitivity to blurring.

Qualitative comparison. Figure 7(a) compares our method on DFC2019 against Sat-NeRF [56], EOGS [77], CoR-GS [114], and Mip-Splatting [111], all of which exhibit significant distortions and blurry textures at lower viewpoints. Figure 7(b) further includes CityDreamer [101] and GaussianCity [103], which over-simplify geometry and miss scene-specific details (e.g., the red pavement in

100

90.3 92.0 93.8 Sat-NeRF

UserPreference(%)

EOGS CoR-GS Mip-Splatting Ours

80

| |
|---|

| |
|---|

60

| |
|---|

40

20

5.1 1.1 1.1 2.3 4.5 1.1 1.1 1.1 4.0 0.6 1.1 0.6

0

Geometric Accuracy

Spatial Alignment

Overall Perceptual Quality

(a) Compare on DFC2019 dataset.

100

CityDreamer 79.0 79.0 81.8

UserPreference(%)

GaussianCity CoR-GS Mip-Splatting Ours

80

| |
|---|

| |
|---|

60

| |
|---|

40

20

8.5 10.2 8.0

6.8 3.4 2.3 6.8 2.8 1.1 5.7 3.4 1.1

0

Geometric Accuracy

Spatial Alignment

Overall Perceptual Quality

(b) Compare on GoogleEarth dataset.

- Fig. 8: User study results. Our method consistently outperforms Sat-NeRF [56], EOGS [77], CoR-GS [114], Mip-Splatting [111], CityDreamer [101] and GaussianCity [103], achieving particularly high scores in geometric accuracy and overall perceptual quality. (a) details the comparison on the DFC2019 dataset [40], while (b) details the comparison on the GoogleEarth dataset [101].

scene 010). Furthermore, our approach yields sharper building contours, higher texture fidelity, and fewer artifacts across both datasets. Notice that SkyfallGS also recovers the challenging facade details in occluded regions and complex structures (e.g., vegetation and bridges) that require high precision. Additional qualitative results are provided in Supplementary.

User studies. We conducted two user studies with 44 participants each, evaluating geometric accuracy, spatial alignment, and overall perceptual quality across

- 4 scenes per study. In each study, participants were presented with side-by-side renderings and asked to select the result that best matched each criterion. In the first study, participants compared Sat-NeRF [56], EOGS [77], CoR-GS [114], Mip-Splatting [111], and our approach; in the second, CityDreamer [101] and GaussianCity [103] replaced Sat-NeRF and EOGS (see Supplementary for full survey details). As shown in Figure 8, our method achieves dominant win rates of ≈90–94% on DFC2019 and ≈79–82% on GoogleEarth across all three criteria, confirming strong and consistent human preference for our approach over all baselines in both satellite reconstruction and city generation settings.

- 4.2 Ablation Studies We conduct ablation studies on the JAX_068 AOI.

Ablation on the reconstruction stage. We ablate appearance modeling, opacity regularization, and pseudo-camera depth supervision (see Table 3 and Figure 9). For this ablation, we evaluate at higher elevation angles to assess render quality during the IDU process, rather than at the final low-altitude viewpoints. Appearance modeling is crucial for multi-date convergence, opacity regularization removes floating artifacts (Figure 9(a)), and depth supervision sharpens geometry in planar regions (Figure 9(b)). Together, they yield the lowest FIDCLIP/CMMD scores. Additionally, we validate geometric accuracy using DFC2019 LiDAR data [40] by unprojecting 3DGS depth renders into DSMs for

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

###### (a) (b) (c)

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

w/o op. reg. w/ op. reg. w/o depth super. w/ depth super.

𝑁 = 1

𝑁 = 2

𝑁 = 3

𝑁 = 5

###### (d) (e)

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

|[Figure 176]|
|---|

|[Figure 177]|
|---|

Randomly sample Curriculum learning

Ours Context-free Prompt Replaced w/ SDEdit

- Fig. 9: Satellite-view training and IDU refinement ablation. (a) Opacity regularization reduces floating artifacts and yields denser reconstructions. (b) Pseudocamera depth supervision improves geometry in planar, texture-less regions such as rooftops and roads. (c) Multiple diffusion samples per view reduce high-frequency geo-

metric noise and enhance texture consistency, Ns = 2 achieves the optimal results. (d) Curriculum learning progressively introduces challenging views, significantly improving geometric coherence in occluded regions over random sampling. (e) A context-free prompt causes only minor degradation in facade details, demonstrating robustness to prompts. Replacing our refinement with SDEdit leads to severe quality degradation, as noising-denoising fails to synthesize details while preserving satellite-defined geometry.

comparison. Both opacity regularization and pseudo-depth supervision improve geometry, with their combination achieving the lowest MAE/RMSE.

Ablation on the synthesis stage. We isolate multi-sample diffusion and curriculum view progression. Ns = 2 achieves optimal visual results (Figure 9(c)), while Ns = 5 yields the lowest CMMD but requires 1.5× longer training with marginal gains, so we adopt Ns = 2. Curriculum scheduling outperforms random sampling in restoring occluded geometry (Figure 9(d), Table 4). Replacing our refinement with SDEdit [60] under the same schedule causes significant quality degradation, as noising-denoising cannot hallucinate plausible details while preserving satellite-defined geometry (Figure 9(e), Table 4). A context-free prompt yields negligible difference, confirming robustness to prompts, with full prompts provided in Supplementary.

#### 4.3 Performance and Scalability

Training efficiency. We evaluate runtime on the JAX_214 AOI using a single NVIDIA RTX A6000 (48GB). The full pipeline completes in ∼6h 45min, split across 1h 35min for reconstruction and 5h 10min for synthesis. Within each IDU episode of ∼1h, compute roughly split between render refinement and 3DGS update at ∼30min and ∼32min respectively, with initial rendering negligible at

[Figure 178]

Table 4: Ablation on the synthesis stage. We evaluate sample counts (Ns), core components, and compare against baselines.

Table 3: Ablation on the reconstruction stage. Appearance modeling secures convergence. Opacity regularization and depth supervision enhance visual fidelity and geometric accuracy.

Method Variation FIDCLIP ↓ CMMD ↓ Time (h) Multiple Samples (Ns)

Components Perceptual Metrics Geometric Metrics

App. Mod.

Op. Reg.

Depth Sup.

Ns = 1 34.11 3.189 3.44 Ours (Ns = 2) 28.35 2.875 6.37 Ns = 3 28.64 2.769 7.19 Ns = 5 29.17 2.677 9.80

FIDCLIP ↓ CMMD ↓ MAE (m)↓ RMSE (m)↓

✗ ✗ ✗ Failed Failed Failed Failed ✓ ✗ ✗ 41.90 2.450 3.542 5.218 ✓ ✓ ✗ 39.95 2.395 2.980 4.527 ✓ ✓ ✓ 38.01 2.307 2.250 3.483

Component Ablation

w/o Curriculum 33.79 3.361 w/ Context-free Pmt. 30.78 2.981 Replaced w/ SDEdit 64.74 4.138 -

[Figure 179]

[Figure 180]

Generated 3D Scene

[Figure 181]

JAX_214JAX_260

[Figure 182]

[Figure 183]

[Figure 184]

- Fig. 10: Multi-block scalability on combined JAX_214 and JAX_260 AOIs. Skyfall-GS jointly optimizes a seamless 3D scene spanning ∼1km × 512m from two adjacent satellite tiles (left). The zoomed inset highlights a shared building, showing no stitching artifacts and confirming scalability to multi-block environments.

∼4s. We consider this a reasonable trade-off, as our method entirely bypasses time- and labor-intensive physical data collection.

Multi-block scalability via combined imagery. To evaluate multi-block scalability, we combined two adjacent AOIs (JAX_214, JAX_260) that overlap along a highway into a single ∼1km × 512m dataset, scaling the IDU look-at grid to 6 × 3 with 5 episodes. Training took ∼9 hours on a single RTX A6000, yielding ∼3.5M Gaussians and a ∼46GB memory footprint. As shown in Figure 21, highway and buildings shared between both AOIs show no stitching artifacts and remain geometrically consistent across the boundary, confirming robust scaling to multi-block environments.

### 5 Conclusion

We present Skyfall-GS, the first method to synthesize real-time, immersive, and freely navigable 3D urban scenes solely from multi-view satellite imagery, requiring no human intervention or domain-specific 3D training data. By combining

- 3D Gaussian Splatting with open-domain diffusion priors in a curriculum-based iterative refinement strategy, our method effectively addresses long-standing challenges, including limited parallax, illumination variations, and large-scale

occlusions. Extensive experiments demonstrate consistent outperformance over reconstruction and generation baselines, including Sat-NeRF, EOGS, CoR-GS, Mip-Splatting, CityDreamer, and GaussianCity. We hope this work paves the way for scalable, automated 3D urban scene creation, with promising future directions including city-wide scaling and dynamic scene modeling.

Limitations. The fixed heuristic camera trajectory works well for most scenarios, but may leave blind spots in complex urban geometries, occasionally causing minor artifacts in heavily occluded regions at extreme street-level perspectives. Additionally, our framework requires off-nadir satellite views to synthesize building facades, leaving synthesis from purely nadir imagery as an open challenge.

### Acknowledgements

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-EA49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- 1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields (2021)
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. CVPR (2022)
- 3. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-NeRF: Anti-aliased grid-based neural radiance fields. In: ICCV (2023)
- 4. Binkowski, M., Sutherland, D.J., Arbel, M., Gretton, A.: Demystifying MMD GANs. In: International Conference on Learning Representations (2018)
- 5. Bosch, M., Kurtz, Z., Hagstrom, S., Brown, M.: A multiple view stereo benchmark for satellite imagery. In: 2016 IEEE Applied Imagery Pattern Recognition Workshop (AIPR). pp. 1–9. IEEE (2016)
- 6. Brooks, T., Holynski, A., Efros, A.A.: InstructPix2Pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 7. Chen, H., Wu, J., Jin, Y., Peng, J., Mao, X., Chi, M., Yao, M., Peng, B., Li, J., Cao, Y.: VI3DRM: Towards meticulous 3D reconstruction from sparse views via photo-realistic novel view synthesis (2024), https://arxiv.org/abs/2409.08207
- 8. Chen, Y., Lee, G.H.: DOGS: Distributed-oriented gaussian splatting for largescale 3D reconstruction via gaussian consensus. Advances in Neural Information Processing Systems 37, 34487–34512 (2024)
- 9. Chung, J., Lee, S., Nam, H., Lee, J., Lee, K.M.: LucidDreamer: Domain-free generation of 3D gaussian splatting scenes. arXiv preprint arXiv:2311.13384 (2023)
- 10. Dahmani, H., Bennehar, M., Piasco, N., Roldao, L., Tsishkou, D.: SWAG: Splatting in the wild images with appearance-conditioned gaussians (2024), https: //arxiv.org/abs/2403.10427
- 11. Deng, B., Tucker, R., Li, Z., Guibas, L., Snavely, N., Wetzstein, G.: Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion. In: SIGGRAPH 2024 Conference Papers (2024)
- 12. Derksen, D., Izzo, D.: Shadow neural radiance fields for multi-view satellite photogrammetry. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 1152–1161 (June 2021)
- 13. Dihlmann, J.N., Engelhardt, A., Lensch, H.P.: SIGNeRF: Scene integrated generation for neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)
- 14. Fan, C.D., Chang, C.W., Liu, Y.R., Lee, J.Y., Huang, J.L., Tseng, Y.C., Liu, Y.L.: SpectroMotion: Dynamic 3d reconstruction of specular scenes. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21328–21338

(2025)

- 15. Fang, J., Wang, J., Zhang, X., Xie, L., Tian, Q.: GaussianEditor: Editing 3D gaussians delicately with text instructions. In: CVPR (2024)
- 16. Gao, J., Liu, J., Ji, S.: Rational polynomial camera model warping for deep learning based satellite multi-view stereo matching. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6148–6157 (2021)
- 17. Gao, J., Liu, J., Ji, S.: A general deep learning based framework for 3D reconstruction from multi-view stereo satellite images. ISPRS Journal of Photogrammetry and Remote Sensing 195, 446–461 (2023). https://doi.org/https: //doi.org/10.1016/j.isprsjprs.2022.12.012, https://www.sciencedirect. com/science/article/pii/S0924271622003276

- 18. Gao, J., Liu, J., Ji, S.: A general deep learning based framework for 3d reconstruction from multi-view stereo satellite images. ISPRS Journal of Photogrammetry and Remote Sensing 195, 446–461 (2023)
- 19. Gao, K., Lu, D., He, H., Xu, L., Li, J.: Enhanced 3D urban scene reconstruction and point cloud densification using gaussian splatting and google earth imagery

(2024), https://arxiv.org/abs/2405.11021

- 20. Gao, R., Holynski, A., Henzler, P., Brussee, A., Martin-Brualla, R., Srinivasan, P.P., Barron, J.T., Poole, B.: CAT3D: Create anything in 3D with multi-view diffusion models. Advances in Neural Information Processing Systems (2024)
- 21. Gao, Y., Li, H., Chen, J., Zou, Z., Zhong, Z., Zhang, D., Sun, X., Han, J.: CityGSX: A scalable architecture for efficient and geometrically accurate large-scale scene reconstruction (2025), https://arxiv.org/abs/2503.23044
- 22. Google: Google earth studio. https://earth.google.com/studio (2024), accessed: 2025-05-14
- 23. Haque, A., Tancik, M., Efros, A., Holynski, A., Kanazawa, A.: InstructNeRF2NeRF: Editing 3D scenes with instructions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023)
- 24. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 (2022)
- 25. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: GANs trained by a two time-scale update rule converge to a local nash equilibrium. In: Advances in Neural Information Processing Systems. pp. 6626–6637 (2017)
- 26. Hou, H.Y., Hsu, C.C., Huang, Y.C., Shen, M.Y., Sun, W.F., Sun, C., Chang, C.C., Liu, Y.L., Lee, C.Y.: 3d gaussian splatting with grouped uncertainty for unconstrained images. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)
- 27. Hua, T., Jiang, L., Chen, Y.C., Zhao, W.: Sat2City: 3D city generation from a single satellite image with cascaded latent diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27978–27988

(2025)

- 28. Huang, X., Liu, X., Wan, Y., Zheng, Z., Zhang, B., Xiong, M., Pei, Y., Zhang, Y.: SkySplat: Generalizable 3d gaussian splatting from multi-temporal sparse satellite images. arXiv preprint arXiv:2508.09479 (2025)
- 29. Huynh-Thu, Q., Ghanbari, M.: Scope of validity of PSNR in image/video quality assessment. Electronics Letters 44(13), 800–801 (2008)
- 30. Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., Kumar, S.: Rethinking FID: Towards a better evaluation metric for image generation. In: CVPR (2024)
- 31. Jiang, L., Ren, K., Yu, M., Xu, L., Dong, J., Lu, T., Zhao, F., Lin, D., Dai, B.: Horizon-GS: Unified 3D gaussian splatting for large-scale aerial-to-ground scenes. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26789–26799 (2025)
- 32. Kang, Y., Wang, X., Wu, Z., Shi, Y., Zhu, H.: Sat2RealCity: Geometry-aware and appearance-controllable 3D urban generation from satellite imagery (2025), https://arxiv.org/abs/2511.11470
- 33. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3D gaussian splatting for real-time radiance field rendering. ACM TOG (2023)
- 34. Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., Drettakis, G.: A hierarchical 3D gaussian representation for real-time rendering of very large datasets. ACM TOG (2024)

- 35. Kulhanek, J., Peng, S., Kukelova, Z., Pollefeys, M., Sattler, T.: WildGaussians: 3D gaussian splatting in the wild. NeurIPS (2024)
- 36. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: FlowEdit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629 (2024)
- 37. Kynkäänniemi, T., Karras, T., Aittala, M., Aila, T., Lehtinen, J.: The role of ImageNet classes in fréchet inception distance. In: Proc. ICLR (2023)
- 38. Labs, B.F.: FLUX. https://github.com/black-forest-labs/flux (2024), accessed: 2025-02-28
- 39. Labs, B.F.: Official weights of FLUX.1 dev. https://huggingface.co/blackforest-labs/FLUX.1-dev (2024), accessed: 2025-02-28
- 40. Le Saux, B., Yokoya, N., Hänsch, R., Brown, M.: Data fusion contest 2019 (DFC2019) (2019). https://doi.org/10.21227/c6tm-vw12, https://dx.doi. org/10.21227/c6tm-vw12
- 41. Leotta, M.J., Long, C., Jacquet, B., Zins, M., Lipsa, D., Shan, J., Xu, B., Li, Z., Zhang, X., Chang, S.F., Purri, M., Xue, J., Dana, K.: Urban semantic 3D reconstruction from multiview satellite imagery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (June 2019)
- 42. Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., Gu, L.: DNGaussian: Optimizing sparse-view 3D gaussian radiance fields with global-local depth normalization. arXiv preprint arXiv:2403.06912 (2024)
- 43. Li, W., He, J., Ye, J., Zhong, H., Zheng, Z., Huang, Z., Lin, D., He, C.: CrossViewDiff: A cross-view diffusion model for satellite-to-street view synthesis (2024), https://arxiv.org/abs/2408.14765
- 44. Li, Z., Li, Z., Cui, Z., Pollefeys, M., Oswald, M.R.: Sat2Scene: 3D urban scene generation from satellite images with diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7141–7150 (June 2024)
- 45. Li, Z., Li, Z., Cui, Z., Qin, R., Pollefeys, M., Oswald, M.R.: Sat2Vid: Street-view panoramic video synthesis from a single satellite image (2021), https://arxiv. org/abs/2012.06628
- 46. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3D: High-resolution text-to-3D content creation. In: CVPR (2023)
- 47. Lin, C.H., Lee, H.Y., Menapace, W., Chai, M., Siarohin, A., Yang, M.H., Tulyakov, S.: InfiniCity: Infinite-scale city synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023)
- 48. Lin, C.Y., Wu, C.H., Yeh, C.H., Yen, S.H., Sun, C., Liu, Y.L.: FrugalNeRF: Fast convergence for extreme few-shot novel view synthesis without learned priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 11227–11238 (2025)
- 49. Lin, J., Li, Z., Tang, X., Liu, J., Liu, S., Liu, J., Lu, Y., Wu, X., Xu, S., Yan, Y., Yang, W.: VastGaussian: Vast 3D gaussians for large scene reconstruction (2024), https://arxiv.org/abs/2402.17427
- 50. Liu, K.H., Yang, C.K., Chen, M.H., Liu, Y.L., Lin, Y.Y.: Corrfill: Enhancing faithfulness in reference-based inpainting with correspondence guidance in diffusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1618–1627. IEEE (2025)
- 51. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1-to-3: Zero-shot one image to 3D object. In: CVPR (2023)

- 52. Liu, T., Zhao, S., Jiang, W., Guo, B.: Sat-DN: Implicit surface reconstruction from multi-view satellite images with depth and normal supervision (2025), https: //arxiv.org/abs/2502.08352
- 53. Liu, X., Chen, J., Kao, S.h., Tai, Y.W., Tang, C.K.: Deceptive-NeRF/3DGS: Diffusion-generated pseudo-observations for high-quality sparse-view reconstruction. arXiv preprint arXiv:2305.15171 (2023)
- 54. Liu, Y., Luo, C., Fan, L., Wang, N., Peng, J., Zhang, Z.: CityGaussian: Real-time high-quality large-scale scene rendering with gaussians. In: European Conference on Computer Vision. pp. 265–282. Springer (2025)
- 55. Liu, Y., Luo, C., Mao, Z., Peng, J., Zhang, Z.: CityGaussianV2: Efficient and geometrically accurate reconstruction for large-scale scenes (2024), https://arxiv. org/abs/2411.00771
- 56. Marí, R., Facciolo, G., Ehret, T.: Sat-NeRF: Learning multi-view satellite photogrammetry with transient objects and shadow modeling using RPC cameras. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 1310–1320 (2022)
- 57. Marí, R., Facciolo, G., Ehret, T.: Multi-date earth observation NeRF: The detail is in the shadows. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 2034–2044 (June 2023)
- 58. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duckworth, D.: NeRF in the wild: Neural radiance fields for unconstrained photo collections. In: CVPR (2021)
- 59. Melas-Kyriazi, L., Laina, I., Rupprecht, C., Neverova, N., Vedaldi, A., Gafni, O., Kokkinos, F.: IM-3D: Iterative multiview diffusion and reconstruction for highquality 3D generation. International Conference on Machine Learning (2024)
- 60. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations (2022)
- 61. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM (2021)
- 62. Mirzaei, A., Aumentado-Armstrong, T., Brubaker, M.A., Kelly, J., Levinshtein, A., Derpanis, K.G., Gilitschenski, I.: Watch your steps: Local image and scene editing by text instructions. In: ECCV (2024)
- 63. Mirzaei, A., Aumentado-Armstrong, T., Derpanis, K.G., Kelly, J., Brubaker, M.A., Gilitschenski, I., Levinshtein, A.: Spin-NeRF: Multiview segmentation and perceptual inpainting with neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20669– 20679 (2023)
- 64. Miyake, D., Iohara, A., Saito, Y., Tanaka, T.: Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models (2024), https:// arxiv.org/abs/2305.16807
- 65. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794 (2022)
- 66. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41(4), 102:1–102:15 (Jul 2022). https://doi.org/10.1145/3528223.3530127, https://doi.org/10. 1145/3528223.3530127

- 67. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: RegNeRF: Regularizing neural radiance fields for view synthesis from sparse inputs. In: CVPR (2022)
- 68. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: DreamFusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988 (2022)
- 69. Qian, M., Tan, B., Wang, Q., Zheng, X., Xiong, H., Xia, G.S., Shen, Y., Xue, N.: Seeing through satellite images at street views. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–18 (2026). https://doi.org/10.1109/ TPAMI.2026.3652860
- 70. Qian, M., Xia, Z., Liu, C., Ma, S., Wang, W., Ke, Z., Tan, B., Zhang, H., Xia, G.S.: Sat3DGen: Comprehensive street-level 3D scene generation from single satellite image. In: International Conference on Learning Representations (ICLR) (2026)
- 71. Qian, M., Xiong, J., Xia, G.S., Xue, N.: Sat2density: Faithful density learning from satellite-ground image pairs. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023)
- 72. Qu, Y., Deng, F.: Sat-Mesh: Learning neural implicit surfaces for multi-view satellite reconstruction. Remote Sensing 15(17) (2023). https://doi.org/10.3390/ rs15174297, https://www.mdpi.com/2072-4292/15/17/4297
- 73. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning. pp. 8748–8763. PMLR (2021)
- 74. Regmi, K., Borji, A.: Cross-view image synthesis using conditional gans (2018), https://arxiv.org/abs/1803.03396
- 75. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022)
- 76. Sabour, S., Goli, L., Kopanas, G., Matthews, M., Lagun, D., Guibas, L., Jacobson, A., Fleet, D.J., Tagliasacchi, A.: SpotLessSplats: Ignoring distractors in 3D gaussian splatting. arXiv preprint arXiv:2406.20055 (2024)
- 77. Savant Aira, L., Facciolo, G., Ehret, T.: Gaussian splatting for efficient satellite image photogrammetry. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)
- 78. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)
- 79. Shang, Y., Lin, Y., Zheng, Y., Fan, H., Ding, J., Feng, J., Chen, J., Tian, L., Li, Y.: UrbanWorld: An urban world model for 3D city generation. arXiv preprint arXiv:2407.11965 (2024)
- 80. Shen, J., Agudo, A., Moreno-Noguer, F., Ruiz, A.: Conditional-flow NeRF: Accurate 3d modelling with reliable uncertainty quantification. In: European Conference on Computer Vision. pp. 540–557. Springer (2022)
- 81. Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: MVDream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023)
- 82. Shi, Y., Campbell, D., Yu, X., Li, H.: Geometry-guided street-view panorama synthesis from satellite imagery. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(12), 10009–10022 (2022)
- 83. Shriram, J., Trevithick, A., Liu, L., Ramamoorthi, R.: RealmDreamer: Textdriven 3D scene generation with inpainting and depth diffusion. In: International Conference on 3D Vision (3DV) (2025)

- 84. Sprintson, M., Chellappa, R., Peng, C.: FusionRF: High-fidelity satellite neural radiance fields from multispectral and panchromatic acquisitions. arXiv preprint arXiv:2409.15132 (2024)
- 85. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: CVPR (2016)
- 86. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Barron, J.T., Kretzschmar, H.: Block-NeRF: Scalable large scene neural view synthesis. In: CVPR (2022)
- 87. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: DreamGaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653

(2023)

- 88. Toker, A., Zhou, Q., Maximov, M., Leal-Taixé, L.: Coming down to earth: Satellite-to-street view synthesis for geo-localization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6488– 6497 (2021)
- 89. Turki, H., Ramanan, D., Satyanarayanan, M.: Mega-NeRF: Scalable construction of large-scale NeRFs for virtual fly-throughs. In: CVPR (2022)
- 90. Wang, R., Xu, S., Dai, C., Xiang, J., Deng, Y., Tong, X., Yang, J.: MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision (2024), https://arxiv.org/abs/2410.19115
- 91. Wang, Y., Yi, X., Wu, Z., Zhao, N., Chen, L., Zhang, H.: View-consistent 3D editing with gaussian splatting (2025), https://arxiv.org/abs/2403.11868
- 92. Wang, Y., Wang, J., Qi, Y.: WE-GS: An in-the-wild efficient 3D gaussian representation for unconstrained photo collections (2024), https://arxiv.org/abs/ 2406.02407
- 93. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: ProlificDreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems 36, 8406–8441 (2023)
- 94. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004)
- 95. Weber, E., Holynski, A., Jampani, V., Saxena, S., Snavely, N., Kar, A., Kanazawa, A.: NeRFiller: Completing scenes via generative 3D inpainting. In: CVPR (2024)
- 96. Wu, C.H., Chen, Y.J., Chen, Y.H., Lee, J.Y., Ke, B.H., Mu, C.W.T., Huang, Y.C., Lin, C.Y., Chen, M.H., Lin, Y.Y., Liu, Y.L.: AuraFusion360: Augmented unseen region alignment for reference-based 360◦ unbounded scene inpainting (2025), https://arxiv.org/abs/2502.05176
- 97. Wu, J., Bian, J.W., Li, X., Wang, G., Reid, I., Torr, P., Prisacariu, V.: GaussCtrl: Multi-view consistent text-driven 3D gaussian splatting editing. ECCV (2024)
- 98. Wu, R., Gao, R., Poole, B., Trevithick, A., Zheng, C., Barron, J.T., Holynski, A.: CAT4D: Create anything in 4D with multi-view video diffusion models (2024), https://arxiv.org/abs/2411.18613
- 99. Wu, R., Mildenhall, B., Henzler, P., Park, K., Gao, R., Watson, D., Srinivasan, P.P., Verbin, D., Barron, J.T., Poole, B., Holynski, A.: ReconFusion: 3D reconstruction with diffusion priors. arXiv preprint arXiv:2312.02981 (2023)
- 100. Wu, Y., Liu, J., Ji, S.: 3D gaussian splatting for large-scale surface reconstruction from aerial images (2024), https://arxiv.org/abs/2409.00381
- 101. Xie, H., Chen, Z., Hong, F., Liu, Z.: CityDreamer: Compositional generative

- model of unbounded 3D cities. In: CVPR (2024)

102. Xie, H., Chen, Z., Hong, F., Liu, Z.: CityDreamer4D: Compositional generative

- model of unbounded 4D cities. arXiv preprint arXiv:2501.08983 (2025)

- 103. Xie, H., Chen, Z., Hong, F., Liu, Z.: Generative gaussian splatting for unbounded 3D city generation. In: CVPR (2025)
- 104. Xu, J., Mei, Y., Patel, V.M.: Wild-GS: Real-time novel view synthesis from unconstrained photo collections (2024), https://arxiv.org/abs/2406.10373
- 105. Xu, N., Qin, R.: Geospecific view generation geometry-context aware highresolution ground view inference from satellite views. In: European Conference on Computer Vision. pp. 349–366. Springer (2024)
- 106. Xu, N., Qin, R.: Satellite to groundscape-large-scale consistent ground view generation from satellite views. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6068–6077 (2025)
- 107. Yao, X., Wang, X., Wu, H., Ping, C., Zhang, D., Xiong, H.: MagicCity: Geometryaware 3d city generation from satellite imagery with multi-view consistency. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25325–25334 (2025)
- 108. Ye, M., Danelljan, M., Yu, F., Ke, L.: Gaussian grouping: Segment and edit anything in 3D scenes. In: ECCV (2024)
- 109. Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X.: GaussianDreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6796–6807 (2024)
- 110. Yu, H.X., Duan, H., Herrmann, C., Freeman, W.T., Wu, J.: WonderWorld: Interactive 3D scene generation from a single image. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5916–5926 (2025)
- 111. Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-Splatting: Alias-free 3D gaussian splatting. In: CVPR (2024)
- 112. Ze, X., Song, Z., Wang, Q., Lu, J., Shi, Y.: Controllable satellite-to-street-view synthesis with precise pose alignment and zero-shot environmental control (2025), https://arxiv.org/abs/2502.03498
- 113. Zhang, D., Wang, C., Wang, W., Li, P., Qin, M., Wang, H.: Gaussian in the wild: 3D gaussian splatting for unconstrained image collections. arXiv preprint arXiv:2403.15704 (2024)
- 114. Zhang, J., Li, J., Yu, X., Huang, L., Gu, L., Zheng, J., Bai, X.: CoR-GS: Sparseview 3d gaussian splatting via co-regularization. In: ECCV (2024)
- 115. Zhang, K., Sun, J., Snavely, N.: Leveraging vision reconstruction pipelines for satellite imagery. In: IEEE International Conference on Computer Vision Workshops (2019)
- 116. Zhang, L., Rupnik, E.: Sparsesat-NeRF: Dense depth supervised neural radiance fields for sparse satellite images. arXiv preprint arXiv:2309.00277 (2023)
- 117. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018)
- 118. Zhang, S., Zhou, M., Wang, Y., Luo, C., Wang, R., Li, Y., Zhang, Z., Peng, J.: Cityx: Controllable procedural content generation for unbounded 3d cities (2024), https://arxiv.org/abs/2407.17572
- 119. Zhou, M., Wang, Y., Hou, J., Zhang, S., Li, Y., Luo, C., Peng, J., Zhang, Z.: SceneX: Procedural controllable large-scale scene generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 10806–10814 (2025)
- 120. Zhou, X., Wang, Y., Lin, D., Cao, Z., Li, B., Liu, J.: SatelliteRF: Accelerating 3D reconstruction in multi-view satellite images with efficient neural radiance fields. Applied Sciences 14(7) (2024). https://doi.org/10.3390/app14072729, https://www.mdpi.com/2076-3417/14/7/2729

###### 121. Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: FSGS: Real-time few-shot view synthesis using gaussian splatting (2023)

### A Supplementary Material

This supplementary material provides additional details complementing the main paper, organized into the following sections:

- 1. Implementation Details (Sec. B): Expanded descriptions of method components including pseudo-camera depth supervision, 3DGS reconstruction parameters, and the FlowEdit-based refinement process. This section also covers system profiling (rendering efficiency, memory consumption) and mathematical validation of the RPC-to-perspective approximation.
- 2. Main Paper Experiments Detail & Results (Sec. C): Comprehensive dataset statistics, the user study methodology, and the evaluation protocol. Additionally, this section provides the complete per-scene quantitative benchmarks and extended qualitative comparisons across both the DFC2019 and GoogleEarth datasets.
- 3. Additional Experiments (Sec. D): Further evaluations demonstrating the robustness of our approach, including: (i) synthesis on complex irregular geometries and bridges; (ii) visualizing transient object handling; (iii) refinement prompt sensitivity analysis; (iv) multi-block scalability via combined imagery; (v) episode-vs-coverage analysis quantifying curriculum effectiveness; and (vi) structural consistency under different random seeds.

We provide an interactive HTML visualization (main.html) for exploring video results across scenes and viewing conditions. To verify that Skyfall-GS learns a coherent, globally explorable 3D structure rather than overfitting to IDU-sampled viewpoints, we include free-flight trajectory renders along novel, smooth camera paths withheld during training. These renders confirm that our method produces a fully immersive and structurally consistent exploration experience, and enable direct perceptual comparison against baseline approaches and Google Earth Studio reference videos.

### B Implementation Details

#### B.1 Method Components

Codebase. Our method extends the Mip-Splatting [111] codebase with custom modules for satellite imagery processing and our curriculum-based IDU refinement pipeline.

Pseudo camera depth supervision. We sample cameras with varied azimuths and decreasing elevations, using random per-image embeddings. MoGe [90] provides scale-invariant depth estimation. We sample 24 views every 10 iterations, with look-at points (x,y,z), where x,y ∼ N(0,128) and z = 0. Camera azimuths are uniformly sampled between 0 and 2π, while elevation angles and radii linearly decrease from 80◦ to 45◦ and 300 to 250 units, respectively. Rendered RGB images (IRGB) are 1024 × 1024 pixels. We illustrate the 3DGS rendered RGB image IRGB, scale-invariant depth Dest estimated by MoGe [90] and depth from 3DGS DGS in Figure 11.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Render RGB IRGB Estimated Depth Dest 3DGS Depth DGS

- Fig. 11: Pseudo-cam Depth Supervision. We use MoGe [90] to estimate the scaleinvariant depth Dest from the rendered RGB image IRGB. The rightmost figures show the rasterized depth DGS from 3DGS.

3DGS reconstruction from satellite imagery. Our satellite-view optimization process runs for 30,000 iterations, with densification enabled between iterations 1,000 and 21,000. We modify several key parameters in the standard 3DGS implementation to address satellite imagery’s unique challenges. First, to prevent undesirable Gaussian elongation artifacts common with overhead views, we reduce the scaling learning rate from 0.005 to 0.001. Second, we implement pruning of Gaussians with maximum covariance exceeding 20 to eliminate floating artifacts. The loss function weights are set to λD-SSIM = 0.2, λop = 10, and λdepth = 0.5 for optimal reconstruction quality. For the GoogleEarth dataset, we disable the opacity regularization, since the dense training views are sufficient to eliminate floaters. For appearance modeling, we adopt the architecture from WildGaussians [35], implementing an appearance MLP with 2 hidden layers (128 neurons each) and ReLU activation functions. The per-image and per-Gaussian embedding dimensions are set to 32 and 24, respectively, with learning rates of 0.001, 0.005, and 0.0005 for per-image embeddings ej, per-Gaussian embeddings gi, and the appearance MLP f, respectively. The complete satellite-view training requires approximately 1 hour on a single NVIDIA RTX A6000 GPU.

FlowEdit-based refinement. We set FlowEdit noise parameters nmin = 4 and nmax = 10 to balance artifact removal with detail preservation. Our source prompt (“Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts.”) characterizes initial renders, while the target prompt (“Clear satellite

image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures.”) guides refinement. These parameters were determined through experimentation, with lower noise values preserving more original structure but removing fewer artifacts, and higher values creating more significant changes but potentially altering underlying geometry. All other FlowEdit parameters use default values.

Curriculum-based refinement. Our IDU process comprises Ne = 5 episodes of 10,000 iterations each, with densification through iteration 9,000. At the start of IDU, we randomly select and fix a single per-image appearance embedding ej. Opacity regularization is disabled during IDU, as our curriculum naturally mitigates floating artifacts through multi-view consistency, enabling Gaussians to retain variable opacities beneficial for semi-transparent structures [33]. For DFC2019 [40] dataset, we utilize Np = 9 look-at points in a 3×3 grid (512 units wide, centered at origin), with Nv = 6 cameras per point and Ns = 2 samples per view. Camera elevations decrease from 85◦ to 45◦ and radii from 300 to 250 units across episodes. For GoogleEarth [101] dataset, we utilize Np = 16 look-at point at origin, with Nv = 6 cameras per point and Ns = 2 samples per view. Camera elevations decrease from 85◦ to 45◦ and radius is fixed 600-unit across episodes. All training images are rendered at 2048×2048 resolution. Our training strategy samples 75% from refined images and 25% from original satellite images, this sampling strategy makes sure that the final 3DGS scene faithfully follows the semantic and layout in the input satellite imagery. The complete synthesizing stage requires approximately 6 hours on a single NVIDIA RTX A6000 GPU.

#### B.2 Resource and Approximation Analysis

Rendering efficiency. Our method achieves 11 FPS on the modest NVIDIA T4, significantly outperforming CityDreamer’s 0.18 FPS, which runs on the far more powerful NVIDIA A100 (5× the CUDA cores, 10× the memory bandwidth). GaussianCity reaches comparable speeds (10.72 FPS) but requires the high-end A100. Furthermore, our fused representation enables real-time rendering at 40 FPS on consumer hardware (MacBook Air M2), demonstrating that our method enables high-quality 3D urban navigation without specialized computing resources.

Memory consumption. We distinguish between peak memory and final memory. The peak memory usage reaches 46 GB during the synthesis stage, driven by the overhead of loading the diffusion model (FLUX.1) and temporary densification of Gaussians. However, the final training memory footprint is significantly lower (28.04 GB) as our method actively prunes redundant and low-opacity points. In terms of scene complexity, the refinement process densifies the scene by approximately 27%, increasing the Gaussian count from ∼1.65 million (reconstruction stage) to ∼2.1 million, specifically targeting the vertical facade geometry missing in the initial satellite reconstruction.

Validity of RPC to perspective approximation. We adopt the methodology proposed in SatelliteSfM [115] to approximate the satellite linear pushbroom sensor as a perspective camera. This approximation relies on the “weak perspective” assumption, which holds valid when the satellite altitude (Z) is significantly larger than the depth variation within the scene (∆Z), i.e., Z ≫ ∆Z. Given that satellites orbit at distances of hundreds of kilometers while terrestrial depth variations are limited to a few hundred meters, the ratio ∆Z/Z remains negligible, allowing the geometry to converge to a perspective model. The approximation is achieved by generating a dense grid of 3D-2D correspondences using the rigorous RPC model and solving for a projection matrix P via the Direct Linear Transformation (DLT) method, which is subsequently decomposed (P = K[R|t]) to recover camera parameters. Quantitative evaluations demonstrate that this process introduces negligible error: the average maximum forward projection error against the rigorous RPC model is only 0.126 pixels, and the difference in triangulated 3D points is typically less than 5 cm. Furthermore, this initialization allows Bundle Adjustment to achieve sub-pixel accuracy, with median reprojection errors recorded at 0.864 pixels, confirming the suitability of this approximation for high-fidelity 3D reconstruction.

### C Main Paper Experiments Detail & Results

#### C.1 Dataset Details

DFC2019 dataset. The number of training images and geographical coordinates for each AOI is provided in Table 5. We also include four additional AOIs from Jacksonville to demonstrate our method’s robustness across varying scene characteristics. The number of training images and geographical coordinates for these additional AOIs is provided in Table 6. These additional AOIs feature distinct characteristics: one contains a city hall building (JAX_164), another includes an American football stadium (JAX_175), while the remaining two exhibit other notable urban features (JAX_168 and JAX_264).

GoogleEarth dataset. The GoogleEarth dataset, introduced by CityDreamer [101],

contains semantic maps, height fields and renders from Google Earth Studio [22] of New York City. This dataset is used to train the generative model in CityDreamer [101] and GaussianCity [103]. We pick four AOIs which contain diverse city elements, including complex architectures (004), squares (010), resident area (219) and riverside (336). However, original GES renders provided in GoogleEarth dataset are rendered from a lower elevation angle, which is not similar to satellite imagery. Therefore, for each AOI, we render 60 images from GES using an orbit trajectory with 80◦ of elevation angle and 2219 meter of radius. These new renders serve as the input of our methods. The AOI ID, geographical coordinates, and the number of input images are detailed in Table 7.

- Table 5: Number of training images and geographical coordinate per Area of Interest (AOI). These AOIs correspond to standard evaluation scenarios established by previous works, ensuring consistent and fair comparisons with existing baselines (e.g., Sat-NeRF [56]).

AOI JAX_004 JAX_068 JAX_214 JAX_260 # of training image 9 17 21 15

Geographical coordinate 81.70643◦W, 30.35782◦N 81.66375◦W, 30.34880◦N 81.66353◦W, 30.31646◦N 81.66350◦W, 30.31184◦N

- Table 6: Number of training images and geographical coordinates for additional AOIs. We selected 4 additional AOIs with distinct characteristics: JAX_164 features a city hall building, JAX_175 contains an American football stadium, while the remaining two AOIs present other notable urban structures.

AOI JAX_164 JAX_168 JAX_175 JAX_264 # of training image 20 21 21 21

Geographical coordinate 81.66362◦W, 30.33032◦N 81.65297◦W, 30.33037◦N 81.63696◦W, 30.32583◦N 81.65285◦W, 30.31189◦N

#### C.2 Evaluation Protocol

User study. We asked participants three specific questions and instructed them to select one video that best addressed each question:

- 1. Geometric Accuracy: “Which video’s 3D structures (buildings, terrain, objects) more accurately represent the real-world geometry when compared to the ground truth video?”
- 2. Spatial Alignment: “Which video’s layout and positioning of elements better matches the satellite imagery reference?”
- 3. Overall Perceptual Quality: “Considering all aspects (geometry, textures, lighting, consistency), which video presents a more convincing and highquality 3D representation of the scene?”

For the user study on DFC2019 dataset, each participant viewed videos from Sat-NeRF [56], EOGS [77], CoR-GS [114], Mip-Splatting [111], and our complete method, alongside Google Earth Studio reference footage and the original satellite imagery. For the user study on the GoogleEarth dataset, each participant viewed videos from CityDreamer [101], GaussianCity [103], CoR-GS [114], Mip-Splatting [111] and our complete method, alongside Google Earth Studio reference footage and the reference satellite imagery.

Comparison details. For quantitative comparisons with Sat-NeRF [56], MipSplatting [111] and our method without IDU refinement, we used consistent camera parameters across all methods: 17◦ elevation angle, 328-unit radius, and 20◦ field of view, with cameras targeting the AOI’s origin. For comparisons with CityDreamer [101] and GaussianCity [103], we use 45◦ elevation angle, 1067unit radius, and 20◦ field of view, with cameras also targeting the AOI’s origin. These parameters were selected to ensure equitable comparison with similar scene coverage across methods.

- Table 7: Number of training images and geographical coordinate per Area of Interest (AOI). We pick 4 AOIs from the GoogleEarth [101] dataset, ensuring fair comparisons with existing baselines (e.g., CityDreamer [101] and GaussianCity [103])

AOI 4WorldFinancialCtr (004) 10UnionSquareE#5P (010) 219E12thSt (219) 336AlbanySt (336) # of training image 60 60 60 60

Geographical coordinate 74.01587◦W, 40.71473◦N 73.98975◦W, 40.73482◦N 73.98690◦W, 40.73187◦N 74.01753◦W, 40.71020◦N

#### C.3 Results

Per-scene quantitative comparison. We provide per-scene quantitative comparisons in Tables 8 and 9. On the DFC2019 dataset (Table 9), our method achieves the best FIDCLIP and CMMD scores across all four AOIs by a substantial margin, reducing FIDCLIP from the next-best ∼84 to ∼27 on average. Pixel-level metrics are more mixed: our method leads on PSNR in three out of four scenes, while CoR-GS occasionally achieves higher SSIM scores due to its tendency to produce overly smooth reconstructions, which artificially inflate SSIM as discussed in the main paper. LPIPS improvements are consistent, with our method achieving the best or second-best score on all four scenes. On the GoogleEarth dataset (Table 8), our method achieves the best FIDCLIP and CMMD in three out of four. The one exception is scene 219, a low-rise residential area with limited vertical facades, where Mip-Splatting achieves lower FIDCLIP (7.06 vs. 7.80) and CMMD (1.589 vs. 2.640). We attribute this to this scene’s relatively flat geometry, which reduces the benefit of our curriculumbased facade synthesis and allows reconstruction-only baselines to perform competitively. This is consistent with our method’s design intent: the gains from diffusion-based refinement are most pronounced in scenes with significant occluded vertical structure. Despite this single exception, our method achieves the best average performance across both datasets, confirming its robustness across diverse urban typologies.

Additional qualitative comparisons. Due to space constraints in the main paper, we present additional qualitative comparison results in this supplementary material for scenes JAX_004 and JAX_260 from the DFC2019 dataset, and scenes 004 and 219 from the GoogleEarth dataset. Figure 12 shows orbital view comparisons with Sat-NeRF [56], Mip-Splatting [111], CoR-GS [114], and EOGS [77], while Figure 13 presents city-scale view comparisons with CityDreamer [101], GaussianCity [103], and CoR-GS [114]. These additional results further demonstrate the consistent superiority of our method across diverse urban environments.

Additional visual results. We also provide qualitative results on four AOIs presents on the main paper (Table 5), as well as four additional AOIs (Table 6) from Jacksonville to demonstrate our method’s robustness across diverse urban environments. As shown in Figure 14 and Figure 15, these AOIs contain

- Table 8: Quantitative comparison on each AOI of DFC2019 [40]. Our method consistently outperforms baseline methods on distribution metrics and most pixel-level metrics, indicating superior image synthesis quality. Metrics are computed between renders from each method and reference frames from GES. Red , orange , and yellow indicate the best, second best, and third best results, respectively.

Distribution Metrics Pixel-level Metrics* Scene Methods FIDCLIP ↓ CMMD ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Sat-NeRF 77.71 3.655 12.00 0.2282 0.8649 EOGS 106.69 5.889 8.21 0.1266 1.0165 CoR-GS 84.43 5.432 11.24 0.2550 0.9763 Mip-Splatting 85.28 5.010 13.08 0.2415 0.8134 Ours 24.07 1.481 12.93 0.2449 0.8450

JAX_004

Sat-NeRF 92.59 5.371 9.86 0.2590 0.8383 EOGS 86.08 5.536 6.39 0.1593 0.9944 CoR-GS 88.05 6.402 11.78 0.3231 1.0065 Mip-Splatting 93.29 6.201 11.66 0.2908 0.8435 Ours 28.38 2.893 11.82 0.2939 0.8193

JAX_068

Sat-NeRF 89.52 5.308 8.92 0.2659 0.8384 EOGS 71.03 4.362 7.39 0.2296 0.8890 CoR-GS 83.17 5.405 11.67 0.4075 0.9075 Mip-Splatting 80.63 5.073 11.26 0.3845 0.8049 Ours 26.10 2.000 12.31 0.3886 0.7410

JAX_214

Sat-NeRF 86.28 4.819 9.52 0.3178 0.9050 EOGS 86.87 5.378 7.04 0.1568 0.9319 CoR-GS 84.13 5.530 11.50 0.4164 0.8976 Mip-Splatting 87.68 5.333 11.64 0.3584 0.8136 Ours 29.58 2.067 12.59 0.3589 0.7532

JAX_260

distinctive architectural features: JAX_004 showcases a residential area with mixed housing types and green spaces; JAX_164 features a prominent city hall building with its characteristic dome and symmetrical facade; JAX_175 encompasses an American football stadium with its distinctive oval structure and surrounding parking facilities; JAX_168 contains a commercial district with varied building heights and dense urban layout. Despite these varied urban typologies, our method successfully generates coherent three-dimensional renderings that preserve the spatial relationships and architectural features present in the satellite imagery. These additional results further validate the generalizability of our approach across diverse urban landscapes without requiring scene-specific parameter adjustments.

Multi-date appearance variation. The use of multi-date satellite imagery introduces a significant challenge, as images of the same location, when captured on different days, exhibit drastic variations in appearance. As shown in Figure 16,

- Table 9: Quantitative comparison on each scenes of the GoogleEarth dataset [101]. The results show that our approach consistently achieves highly competitive performance, indicating superior geometric and perceptual fidelity across most metrics compared to the baselines. Metrics are computed between renders from each method and reference frames from GES. Red , orange , and yellow indicate the best, second best, and third best results, respectively.

Distribution Metrics Pixel-level Metrics Scene Methods FIDCLIP ↓ CMMD ↓ PSNR ↑ SSIM ↑ LPIPS ↓

CityDreamer 34.54 4.297 13.06 0.3519 0.5643 GaussianCity 29.76 2.833 14.00 0.3785 0.5654 CoR-GS 29.17 4.092 13.57 0.3760 0.4426 Mip-Splatting 24.57 2.611 14.59 0.3823 0.4278 Ours 14.81 1.549 15.22 0.3859 0.4117

004

CityDreamer 39.95 3.948 12.24 0.1387 0.5541 GaussianCity 28.65 2.715 12.90 0.1661 0.5330 CoR-GS 30.10 3.741 12.90 0.1807 0.4209 Mip-Splatting 11.55 1.914 13.63 0.1823 0.3514 Ours 9.43 2.484 13.71 0.1800 0.3959

010

CityDreamer 42.54 4.444 11.63 0.1344 0.5465 GaussianCity 32.78 2.898 12.37 0.1676 0.5248 CoR-GS 16.39 3.227 12.64 0.1791 0.3971 Mip-Splatting 7.06 1.589 13.26 0.1795 0.3321 Ours 7.80 2.640 13.32 0.1764 0.3840

219

CityDreamer 29.59 4.110 13.39 0.4430 0.5654 GaussianCity 23.87 3.215 14.36 0.4532 0.5376 CoR-GS 29.74 3.970 14.29 0.4591 0.3875 Mip-Splatting 21.20 2.232 15.05 0.4631 0.4039 Ours 9.11 1.164 15.45 0.4664 0.3805

336

these differences can fundamentally alter the scene’s geometry and texture. Effectively synthesizing novel views requires a model capable of intelligently disentangling the static 3D scene structure from these challenging, temporally-varying appearance factors.

### D Additional Experiments

Qualitative results on complex geometries. To demonstrate the robustness of our framework beyond standard city-block layouts, we evaluate our method on scenes featuring irregular and historically significant architectures. As shown in Figure 15, we present synthesis results for Neuschwanstein Castle and Wells Cathedral. These scenes pose significant challenges due to their intricate non-Manhattan geometries, including sharp spires, varying elevations, and gothic architectural details. Despite these complexities, our method successfully

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- 004

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

260

Input Satellite Image GES Reference Sat-NeRF Mip-Splatting

CoR-GS EOGS Ours w/o IDU Ours

- Fig. 12: Additional qualitative comparison on the DFC2019 dataset with Sat-NeRF [56], Mip-Splatting [111], CoR-GS [114], and EOGS [77]. Our method significantly outperforms baseline approaches in both geometric accuracy and texture quality when rendering low-altitude novel views. Note the superior building geometry, facade details, and reduced floating artifacts in our final result.

[Figure 207]

[Figure 208]

004219

Input Satellite Image GES Reference Ours Ours w/o IDU CityDreamer GaussianCity CoR-GS Mip-Splatting

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

- Fig. 13: Additional qualitative comparison on the GoogleEarth dataset with CityDreamer [101], GaussianCity [103], CoR-GS [114], and MipSplatting [111]. Our method is able to synthesize texture and geometry that is closer to the reference GES render.

disentangles the underlying geometry from the satellite input and hallucinates plausible high-frequency details for facades that are heavily occluded in the nadir views. This confirms that our hybrid reconstruction-generation approach is not limited to simple urban prisms but extends effectively to complex, free-form structures.

Synthesis of bridges. In addition to dense building clusters, we evaluate our method’s performance on scenes with complex topological structures, such as bridges. Figure 18 illustrates renders of bridges in JAX_068, JAX_214 and JAX_175, a typically difficult case for standard photogrammetry due to the thin structural components. Our method successfully recovers the connectivity of the bridge span while synthesizing realistic water textures. The diffusionbased refinement effectively regularizes the geometry, preventing the characteristic "melting" artifacts often observed in thin structures when using satellite-only reconstruction.

Input Satellite Image

3DGS Render

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

214260068004

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

- Fig. 14: Qualitative results across primary scenes. Visualization of satellite image inputs and corresponding rendered frames for our four main AOIs.

Visualizing transient object handling via per-image embeddings. A key challenge in multi-date satellite reconstruction is the handling of dynamic elements, such as moving vehicles and pedestrians, which can introduce ghosting artifacts. Our approach addresses this by learning per-image appearance embeddings ej that capture photometric variations specific to each capture date. As visualized in Figure 19, rendering the same viewpoint across 20 distinct appearance embeddings reveals that transient objects exhibit significant variability, appearing clearly in some embeddings while fading or vanishing in others. This qualitative evidence suggests that our appearance modeling effectively acts as a “sink” for transient data that does not align with the static 3D geometry. By absorbing these inconsistencies into the appearance code rather than the geometric parameters, the optimization naturally disentangles transient elements

from the underlying static structure, ensuring a clean and consistent geometric reconstruction.

Sensitivity analysis of refinement text prompts. A practical concern for deployment is whether our method is sensitive to the specific wording of the FlowEdit text prompts. To investigate this, we evaluate six prompting strategies listed in Table 10, ranging from highly descriptive source/target pairs (Baseline) to vague descriptions (Vague Source, Vague Target), domain-specific variants that emphasize geometry or texture (Focus Geometry, Focus Texture), and a minimal context-free pair (Context Free). As shown in Figure 20, the visual quality of the refined renders remains largely consistent across all strategies, with only minor differences in fine-grained texture details. This robustness suggests that our method does not require careful prompt engineering and is tolerant to moderate variations in prompt specificity.

- Table 10: List of text prompts used in sensitivity analysis. We evaluate six different prompting strategies to test the robustness of our method.

Strategy Source Prompt (Psrc) Target Prompt (Ptar)

Baseline Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts.

Clear satellite image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures.

Vague Source A blurry satellite image of an urban area. Clear satellite image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures.

Vague Target Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts.

A clear satellite image of an urban area.

Focus Geometry

Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts.

Clear satellite image of an urban area with geometrically precise buildings, flat rooftops, straight edges, and well-defined roads.

Focus Texture Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts.

Clear satellite image of an urban area with realistic, high-resolution textures, detailed facades, clear vegetation, and natural lighting.

No Prompt - Context Free distorted, blurring, warping artifacts clear, sharp, smooth edges, natural light-

ing, well-defined textures

Multi-block scalability via combined imagery. To validate the scalability of our framework across larger, continuous urban regions, we combined satellite imagery from two adjacent areas (JAX_214 and JAX_260) into a single,

unified dataset covering approximately 1km × 512m. To accommodate the expanded spatial extent, we scaled the Iterative Dataset Update (IDU) look-at grid to 6 × 3, while maintaining our standard 5-episode refinement schedule (10,000 iterations per episode). The total training time for this expanded region took approximately 9 hours (1.5 hours for the initial reconstruction stage and 7.5 hours for the synthesis stage) on a single NVIDIA RTX A6000 GPU. The final converged scene yielded ∼3.5 million Gaussians with a peak final training memory footprint of ∼46GB. As shown in Figure 21, the expanded area demonstrate seamless boundary consistency, particularly evident along the connecting highway structures, confirming that our method scales robustly to multi-block environments without introducing stitching artifacts at the region boundaries.

Episode-vs-coverage analysis of curriculum strategy. To quantify the effectiveness of the IDU module in revealing occluded regions, we present an Episode-vs-Coverage analysis (Figure 22). Since ground truth 3D geometry is unavailable for these satellite scenes, we use the final converged 3DGS model as a proxy for the total scene surface. We compute the cumulative coverage by optimizing a visibility attribute for every Gaussian point against the camera poses utilized in each episode. As shown in the figure, the coverage ratio steadily increases from ∼0.50 in Episode 1 to ∼0.75 in Episode 5. This consistent gain confirms that our curriculum strategy, which progressively lowers camera elevation from 85◦ to 45◦, successfully reveals and reconstructs vertical facade geometry that was initially occluded in the top-down satellite views. However, we acknowledge a limitation in this metric: because it calculates coverage based on reconstructed points, it cannot account for “true holes” (surface areas that were never generated at all because they were completely occluded from all sampled views). Future work could address this by dynamically sampling IDU cameras to target specific geometric uncertainties or detected holes.

Stochastic appearance diversity. To demonstrate the generative capacity of our hybrid framework, we evaluate the stochastic diversity of the synthesized textures in Figure 23. By varying the random seed during the diffusion refinement stage while maintaining the same geometric initialization, our method produces diverse yet plausible surface details for identical underlying structures. As illustrated in the figure, detailed features such as the text on the red building signage vary distinctively (e.g., “Outeil” vs. “CUTAN”). Crucially, the macroscopic building footprint remains geometrically fixed, confirming that our framework successfully disentangles the reconstruction of physical geometry (grounded in satellite constraints) from the generative synthesis of high frequency appearance.

Input Satellite Image

3DGS Render

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

264168175164

[Figure 244]

[Figure 245]

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

- Fig. 15: Qualitative results across additional scenes. Visualization of satellite image inputs and corresponding rendered frames for four additional AOIs with distinctive characteristics: JAX_164 features a city hall building, JAX_175 contains an American football stadium, while JAX_168 and JAX_264 present other notable urban structures.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

004068

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

214260

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Sample 1 Sample 2 Sample 3 Sample 4

- Fig. 16: Visualization of multi-date satellite imagery of the DFC2019 dataset. Note the substantial shifts in appearance, including changes in illumination, transient objects, and surface characteristics, which introduce challenges for consistent

- 3D reconstruction.

NeuschwansteinCastleWellsCathedral

Input Satellite Image

3DGS Rendered Views

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

- Fig. 17: Qualitative results on complex geometries. Given multi-view satellite imagery, our method synthesizes novel views of irregular historical structures, shown here for Neuschwanstein Castle and Wells Cathedral.

[Figure 280]

[Figure 281]

[Figure 282]

(a) JAX_068 (b) JAX_175 (c) JAX_214

- Fig. 18: Qualitative results for bridges. We present the render results for bridges appears in JAX_068, JAX_214 and JAX_175, demonstrating the method’s ability to handle complex mulit-level structures that are typically challenging for standard reconstruction pipelines.

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Emb. 00 Emb. 01 Emb. 02 Emb. 03 Emb. 04

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Emb. 05 Emb. 06 Emb. 07 Emb. 08 Emb. 09

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Emb. 10 Emb. 11 Emb. 12 Emb. 13 Emb. 14

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Emb. 15 Emb. 16 Emb. 17 Emb. 18 Emb. 19

- Fig. 19: Visualizing transient object handling via per-image embeddings. We render the same viewpoint using 20 different learned appearance embeddings (Emb. 00–19). Observe that transient objects, such as the vehicles on the road, exhibit varying degrees of visibility across different embeddings (e.g., clearly visible in some, faded or absent in others), while the static building geometry remains consistent. This qualitatively demonstrates that our per-image appearance modeling effectively disentangles transient elements from the underlying static 3D structure, preventing dynamic artifacts from corrupting the geometric reconstruction.

[Figure 303]

##### Fig. 20: Refine renders with different prompt strategies.

[Figure 304]

###### Fig. 21: Multi-block reconstruction from combined JAX_214 and JAX_260 AOIs. Skyfall-GS produces a seamless, artifact-free 3D scene across a ∼1km × 512m urban region, demonstrating scalability to multi-block environments without boundary stitching artifacts.

Episode vs Coverage

0.75

0.70

0.65

Coverage

0.60

0.55

0.50

1 2 3 4 5 Episode

JAX avg NYC avg Total avg

###### Fig. 22: Episode-vs-Coverage analysis. The plot illustrates the cumulative surface coverage ratio increasing across refinement episodes. The curriculum-based strategy effectively exposes occluded regions, particularly vertical facades, as the camera elevation descends.

[Figure 305]

- Fig. 23: Demonstration of stochastic appearance diversity while preserving geometric consistency. Our method generates diverse plausible textures for identical underlying geometry across different random seeds. Notice how the red signage text on the building facade varies distinctively (e.g., “Outeil” vs. “CUTAN”) while the building’s structural footprint remains fixed, confirming that our framework successfully disentangles geometric reconstruction (grounded in satellite data) from generative appearance synthesis (variable via diffusion).

