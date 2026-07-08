# arXiv:2505.15765v2[cs.CV]4Oct2025

## CONSTRUCTING A 3D SCENE FROM A SINGLE IMAGE

Kaizhi Zheng1 Ruijian Zha3,4 Zishuo Xu1 Jing Gu1 Jie Yang4 Xin Eric Wang1,2 1UC Santa Cruz 2UC Santa Barbara 3Columbia University 4Utopai Studios {kzheng31,xwang366}@ucsc.edu

[Figure 1]

[Figure 2]

"Mediterranean Village"

[Figure 3]

[Figure 4]

SceneFuse-3D (Ours) Hunyuan3D-2

"Snow Town"

[Figure 5]

"Netherlands-style Town"

[Figure 6]

[Figure 7]

[Figure 8]

Reference Image

[Figure 9]

Trellis TripoSG

Coherent Generation from SceneFuse-3D(Ours)

Diverse Generation from SceneFuse-3D(Ours)

Figure 1: 3D Scene Generation from a Single Image. Given a top-down reference image (center), SceneFuse-3D generates coherent and realistic 3D scenes that preserve geometry, texture, and layout compared to other state-of-the-art end-to-end image-to-3D generation models. Our method also generalizes across diverse styles (right), producing high-quality outputs without any 3D training.

ABSTRACT

Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in realworld applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce SceneFuse-3D, a training-free framework designed to synthesize coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning. Extensive experiments across diverse scenes show that SceneFuse-3D outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, TripoSG, and LGM, in terms of geometry quality, spatial

coherence, and texture fidelity. Our results demonstrate that high-quality coherent 3D scene-level asset generation is achievable from a single top-down image using a principled, training-free pipeline. Project website: https://eric-ai-lab.

github.io/3dtown.github.io/

- 1 INTRODUCTION

Constructing 3D scene environments serves as an essential component in simulation, robotics, digital content creation, and virtual world building. They enable scalable training for autonomous agents, immersive game environments, and rapid digital twin construction. However, constructing detailed and coherent 3D scenes typically requires either expensive 3D scanning equipment, multi-view data collection, or labor-intensive modeling. In contrast, generating 3D scenes from a single top-down image offers a lightweight and accessible alternative, making it possible to bootstrap comprehensive environments from minimal input.

Despite its appeal, generating a coherent and complex 3D scene from a single image presents several fundamental challenges. First, the synthesized scene must exhibit consistent geometry across novel views, which is difficult for pure volumetric rendering methods such as Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) or 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023). While these techniques excel at photorealistic appearance modeling, they often suffer from geometry artifacts, especially in occluded or sparsely visible regions, leading to multiview inconsistencies and structural implausibility (Li et al., 2022; Chung et al., 2023; Zhang et al., 2024a; Yu et al., 2024b). Second, the global layout of the generated scene must remain faithful to the input image. This is particularly challenging when considering the entire scene as a single asset and utilizing image-to-3D asset generators (Xiang et al., 2024; Team, 2025; Li et al., 2025), which often fail to preserve the spatial relationships between elements, resulting in distorted or semantically misaligned arrangements. Third, the local fidelity of individual objects should align closely with the visual evidence in the input. Due to the resolution constraints of 3D representations and the domain shift from object-level training to scene-level inference, previous image-to-3D generators are prone to producing low-quality meshes and misaligning textures.

To address these challenges, we propose SceneFuse-3D, a training-free framework for generating complex 3D scenes from a single top-down image, by enhancing the capability of image-to-3D object generators. Our method combines two core components: region-based generation and spatial-aware 3D inpainting. Each targets specific challenges in existing pipelines. We divide the scene into overlapping regions and synthesize each independently. This modular approach enables spatial upscaling and improves local alignment by grounding generation on localized image crops. To maintain global coherence and object continuity across regions, we estimate a coarse 3D structure from monocular depth and landmark detection, forming a spatial prior. A masked rectified flow mechanism then completes missing parts while preserving known content, enhancing structural consistency and object-level fidelity throughout the scene.

Through extensive experiments, we demonstrate that SceneFuse-3D generates realistic, diverse, and geometrically consistent 3D scenes from a single top-down image. Our method significantly outperforms strong baselines, including Trellis (Xiang et al., 2024), Hunyuan3D-2 (Team, 2025), TripoSG (Li et al., 2025), and LGM (Tang et al., 2024b), across both human preference and GPTbased evaluations. Quantitative results show notable gains in geometry quality, layout coherence, and texture fidelity, while qualitative comparisons highlight SceneFuse-3D’s ability to preserve spatial structure and fine-grained detail. These results underscore the effectiveness of our modular, training-free approach to 3D scene synthesis.

Our main contributions are summarized as follows:

- • We propose SceneFuse-3D, a training-free framework for generating structured 3D scenes from a single top-down image, leveraging pretrained object-centric generators for zero-shot scene asset synthesis.
- • We develop a modular generation strategy that combines region-wise latent synthesis with spatial-aware 3D inpainting, effectively addressing resolution bottlenecks, image-geometry misalignment, and inter-region inconsistency.

- • We conduct comprehensive evaluations on diverse scenes and show that SceneFuse-3D outperforms state-of-the-art baselines in geometry quality, layout coherence, and texture realism under both human and GPT-4o-based assessments.

- 2 RELATED WORK
- 3D Scene Generation with 2D Generative Models Progress in 2D generative models (Rombach et al., 2022; Ho et al., 2020; Chang et al., 2024; Ramesh et al., 2021) has enabled pipelines that outpaint views and then reconstruct 3D via depth fusion or volumetric representations such as NeRF (Mildenhall et al., 2021) and 3DGS (Kerbl et al., 2023). Early work focused on indoor scenes (Wiles et al., 2020; Koh et al., 2021; H¨ollein et al., 2023; Koh et al., 2023), with later methods tackling natural scenes (Li et al., 2022; Cai et al., 2023; Fridman et al., 2023; Chung et al., 2023) and improving reconstruction through stronger depth pipelines (Yu et al., 2024b; Zhang et al., 2024c;a; Yang et al., 2024; Shriram et al., 2024; Engstler et al., 2024; Yu et al., 2024a). Despite compelling renders, these approaches often exhibit geometric inconsistency due to hallucinations, especially in occluded regions. Panoramic (Stan et al., 2023; Li et al., 2024b; Wu et al., 2023; Schult et al., 2024; Liang et al., 2024) and multiview generation (Liu et al., 2024; Tang et al., 2023c; Gao et al.,

2024) improve coverage but typically restrict camera motion and still struggle with accurate geometry. Concurrently, Syncity (Engstler et al., 2025) assembles block-wise 3D generations yet produces compact layouts and does not take full scene images as input. In contrast, we directly generate geometry-consistent scene assets with arbitrary layout from a single top-down image.

3D Scene Generative Model Beyond 2D-to-3D pipelines, recent work directly generates scenes in native 3D. One line uses LLMs (Feng et al., 2023; Zhou et al., 2024; C¸elen et al., 2024; Hu et al., 2024; Sun et al., 2023) or diffusion models (Tang et al., 2024a; Lin & Mu, 2024; Zhai et al., 2024; Vilesov et al., 2023; Maillard et al., 2024) to predict scene layouts that are then populated with assets. These methods yield semantically plausible arrangements but are often limited to predefined categories, struggle with coherent background geometry, and can suffer inter-object collisions. A separate line models scenes directly in latent 3D spaces (Wu et al., 2024; Meng et al., 2024; Ren et al., 2023; Liu et al., 2023b; Lee et al., 2024; Chai et al., 2023), e.g., BlockFusion’s triplane blocks (Wu et al., 2024) and LT3SD/XCube’s TUDF/voxel representations (Meng et al., 2024; Ren et al., 2023). While architecturally strong, these approaches require large, domain-specific 3D datasets (e.g., indoor rooms or urban layouts), limiting generalization to unseen scene types. In contrast, our method is training-free and modular, generating diverse 3D scene assets directly from a single top-down image.

Image-to-3D Asset Generation A parallel line of work generates single 3D assets from one image. Early methods combine 2D diffusion with NeRF/3DGS optimization (Poole et al., 2022; Lin et al., 2022; Tang et al., 2023b; Liu et al., 2023a; Tang et al., 2023a; 2024b) but often trade speed for geometry fidelity. Newer approaches directly generate 3D latents via diffusion (Gupta & Gupta, 2023; Xiong et al., 2024; Li et al., 2024a; Vahdat et al., 2022; Zhang et al., 2024b), and rectified-flow models further improve quality/efficiency (Xiang et al., 2024; Team, 2025; Li et al., 2025). However, these methods are trained on large object-centric datasets (e.g., Objaverse-XL (Deitke et al., 2023)), so applying them to full scenes faces limited 3D resolution and domain shift between objects and scenes, leading to spatial inconsistencies and layout hallucinations. We build on pretrained rectified-flow generators (Xiang et al., 2024) and mitigate these issues with a region-based strategy plus spatialaware 3D inpainting, yielding scene assets with high geometric fidelity and global layout coherence from a single top-down image.

3 METHOD

Given a single top-down image of an unknown scene, our objective is to synthesize a high-quality 3D scene that is geometrically and visually consistent with the input view. Figure 2 illustrates an overview of our pipeline.

From the scene image, we first construct the scene latents with structured latent representations (Sec. 3.1) from spatial priors (Sec. 3.2). Then, we divide the scene-level latent into region-level latents for sequential processing. For each region, we extract the latents from the latest scene-level structured latents and take those as priors for spatial-aware 3D completion (Sec. 3.3) by using the

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

...

[Figure 16]

Foreground Landmarks

Region1 Region2 Region N

Point Cloud

[Figure 17]

[Figure 18]

Reference Image

3D Scene Asset

Region Fusion

Inital Scene Structured Latent

Final Scene Structured Latent

Region Extraction

###### Spatial Prior Initialization Region-based Generation & Fusion

[Figure 19]

|Sparse Structure Generator| |
|---|---|
| | |

|Structured Latent Generator<br><br>|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

Region Reference Image

[Figure 23]

3D Region Asset

Region Structured Latent (W/O Local Features) Region Structured Latent

Initial Region Structured Latent

Spatial-aware 3D Completion for Single Region

- Figure 2: Overview of the SceneFuse-3D Pipeline. Given a single top-down image, we first estimate a coarse scene structure via monocular depth and landmark extraction to initialize the scene latent (Spatial Prior Initialization). The scene is divided into overlapping regions for localized synthesis and progressively fused into a coherent global latent (Region-based Generation & Fusion). Each region is completed using a two-stage masked rectified flow pipeline with a sparse structure generator Gs and a structured latent generator GL (Spatial-aware 3D Completion). The final 3D scene is decoded from the completed structured latent.

base 3D generator, Trellis (Xiang et al., 2024). Later, we fuse the updated region latents to the scene latents for cross-region consistency (Sec. 3.4). After finishing all regions, we leverage pretrained object decoders on the complete scene structured latents to obtain a 3D scene asset. Details are explained below.

- 3.1 STRUCTURED LATENT REPRESENTATION

To leverage the pretrained knowledge of the base 3D generator, we construct the scene with a structured latent representation z (Xiang et al., 2024):

z = {(pi,fi)}Li=1, pi ∈ {0,1,...,K − 1}3, fi ∈ RC, (1)

where pi denotes the positional index of an active voxel in the 3D grid, and fi represents the associated latent feature vector of dimension C. Here, K is the resolution of the voxel grid, and L is the total

number of active voxels. In general, pi captures the coarse structural layout of the object, while fi encodes fine-grained local appearance and shape information. For the pretrained models, the resolution K should be equal to N = 64. To upscale the resolution for the scene, we construct scene structured latents with resolution M, where M > N (M = 2N by default).

For the general image-to-3D asset generation, there are two pretrained rectified flow transformers: a sparse structure generator Gs and a structured latent generator GL. At inference time, Gs first generates the active voxel positions {pi}L from the noisy grids VT with Gaussian noise. These positions are then used by GL to generate the corresponding latent features {fi}L from noisy features FT. Both generators are conditioned on an input image condition CI, encoded by DINOv2 (Oquab

- et al., 2023). Finally, the structured latent representation z is decoded into a 3D object O using object decoders, which include different sparse 3D VAE decoders for 3D Gaussians, Radiance Fields, and

mesh generation. The overall process is described as follows:

{pi}L = Gs(VT |CI), VT ∼ N(0,I)[N,N,N] (2) {fi}L = GL(FT |CI,{pi}L), FT ∼ N(0,I)[L,C] (3)

O = ObjectDecoder(z), z = {(pi,fi)}Li=1 (4)

- 3.2 SPATIAL PRIOR INITIALIZATION

Given a top-down image, the image-to-3D generator may produce outputs in arbitrary orientations. To resolve this ambiguity and provide a consistent structural prior, we initialize the scene using point clouds. Specifically, we employ a monocular depth estimator (Wang et al., 2025) to predict a depth image and infer camera parameters, from which we construct pixel-wise point clouds. Due to occlusions, these point clouds contain missing regions, which will later be filled by the image-to-3D generator.

However, occluded areas of complex objects (e.g., buildings) may have multiple plausible completions. To enforce cross-region consistency in such cases, we propose first generating landmark objects independently and then conditioning subsequent generations on their geometry. To achieve this, we use Florence2 (Xiao et al., 2024) to propose landmark bounding boxes and SAM2 (Ravi et al., 2024) to extract instance masks. Each detected landmark is then processed individually using the 3D generator to obtain instance-level meshes. These meshes are aligned with the raw point clouds using ICP (Rusinkiewicz & Levoy, 2001), replacing the original landmark regions with mesh-derived point clouds.

After normalization, we voxelize the aggregated scene point clouds at resolution M to obtain the initial voxelized scene, including foreground voxels V0f, background voxels V0b, and the full scene voxel set V0:

V0 = {V0b,V0f} = {pi}L, pi ∈ {0,1,...,M − 1}3 (5)

Finally, we initialize the corresponding voxel features F0 as zeros and construct the initial structured latent representation for the scene as z0 = {(V0,F0)}, shown in the top-left of Figure 2.

- 3.3 REGION-BASED GENERATION We adopt a region-based strategy to overcome the limitations of applying pretrained object-centric

- 3D generators directly to full scenes. These models are trained on single-object data, where each object occupies the full latent space at a fixed resolution N. When extended to an entire scene, this limited capacity results in low-resolution geometry and missing details. Moreover, direct scene-level generation from a single top-down image often leads to layout distortions and semantic hallucinations,

- as the model struggles to maintain spatial relationships or align 3D content with image cues. To address this, we divide the scene into overlapping regions and condition each on its corresponding image crop, enabling locally grounded and high-fidelity generation.

To implement this, we divide the initial scene voxel grid V0 (of resolution M) into a set of overlapping region-level subgrids {V0(r)}Rr=1, each with shape N3, where N is the resolution used by the pretrained 3D generator. For each region r, we also extract the corresponding image crop to obtain a localized image conditioning input CI(r). This ensures that the generation in each region is locally grounded in the image evidence.

Spatial-aware 3D Completion While region-based generation improves local fidelity, it introduces a new challenge: how to maintain global consistency, especially across overlapping regions. Furthermore, since the 3D generator may create assets from any orientation, we need to alleviate the misalignment between image conditions and region latents to enable further region fusion. To address these challenges, we draw inspiration from training-free inpainting methods in 2D diffusion models, such as RePaint (Lugmayr et al., 2022), and adapt a similar approach for 3D generation. We propose to use a masked rectified flow pipeline for 3D completion, which treats the partially completed global scene latent as a constraint and performs conditional generation over the current region by completing only the unknown parts.

Given a region-level subgrids V0(r), we designate known active voxels as positions {p(i,r0)}L

r,0 with corresponding latent features {fi,(r0)}L

r,0. For the coarse structure generation, we define a binary mask m(sr) where inactive voxels are marked for regeneration. We use the sparse structure generator Gs to complete the region structure and obtain active voxel positions {p(ir)}L

r. Next, for the fine-grained local features generation, we retain original features for positions overlapping with known voxels; otherwise, features are initialized with Gaussian noise. A second binary mask m(Lr) identifies unknown features for regeneration. We then use the structured latent generator GL to obtain the inpainted local features {fi(r)}L

r. Finally, we can construct the region structured latent z(r) = {(p(ir),fi(r))}L

r. These two completions both leverage the masked rectified flow pipeline (Details in the next paragraph). The whole process is shown at the bottom of Figure 2 and can be formally written as follows:

{p(ir)}L

#### = Gs(VT |CI(r),{p(i,r0)}L

#### ,m(sr)), VT ∼ N(0,I)[N,N,N] (6) {fi(r)}L

r

r,0

,m(Lr)), FT ∼ N(0,I)[L

,{fi,(r0)}L

#### = GL(FT |CI(r),{p(ir)}L

r,C] (7) z(r) = {(p(ir),fi(r))}L

r,0

r

r

r (8)

Masked Rectified Flow for Completion We adopt a masked generation strategy based on rectified flow to complete the unknown regions of a structured 3D latent. Let xknown denote the known latent values to preserve, and let m ∈ {0,1} be a binary mask that indicates which parts of the latent should be regenerated (m = 1) and which should remain fixed (m = 0).

We initialize the latent variable xT ∼ N(0,I) with Gaussian noise, representing the unknown region

- at the final time step. For each timestep t = T,T−1,...,1, we perform U resampling steps to improve stability and smoothness (Lugmayr et al., 2022). At each resampling iteration u, we first

compute the flow field vθ(xt,t) using the rectified flow model and apply an Euler update to obtain the intermediate latent:

= xt − ∆t · vθ(xt,t), where ∆t = 1. (9) We then re-noise the known region using a forward noise operator:

#### xt

prev

forward step(x,t) = (1 − t) · x + [σmin + (1 − σmin) · t] · ϵ, ϵ ∼ N(0,I), (10) and merge it back into the latent using the mask m:

+ (1 − m) ⊙ forward step(xknown,tprev). (11)

prev ← m ⊙ xt

xt

prev

σmin denotes the minimum noise scale used by the pretrained rectified model. If t > 1 and the latent needs to be resampled, we apply additional forward noise to the merged latent: xt ← forward step(xt

,∆t). Otherwise, we simply continue with xt ← xt

. This masked

prev

prev

rectified flow process iterates until t = 0, at which point the completed latent x0 is returned. The full procedure is outlined in Algorithm 1 in Appendix B.

- 3.4 REGION FUSION

For each generated region, we update the scene-level structured latent z0 by replacing the corresponding part with the region-level latent z(r). Because regions are extracted using a patchification strategy, some may contain only partial observations of foreground landmarks. To preserve landmark integrity, we discard those structured latents corresponding to partial foregrounds during fusion.

Each region is extracted from the latest version of the scene-level latent, ensuring consistency across regions. If a region overlaps with previously generated ones, its overlapping voxels are constrained to match the existing content during generation. This enforces continuity and avoids inconsistencies in overlapping areas, leading to smooth transitions between adjacent regions while preserving already synthesized content.

Once all regions have been processed, the final scene-level latent is decoded using object decoders to produce scene-level meshes and 3D Gaussians. The complete textured scene is then rendered using a combination of physically based rendering (PBR) baking and Gaussian Splatting. Additional implementation details, including the patchification strategy, are provided in Appendix B.

- Table 1: Quantitative comparisons of SceneFuse-3D and baselines. We report human preference win rates and GPT-4o-based weighted win rates (%) across geometry, layout, and texture quality. SceneFuse-3D consistently outperforms all baselines by large margins in both evaluations.

Human Preference Win Rate (Percentage) GPT-4o-based Weighted Win Rate (Percentage) Models Geometry Quality Layout Coherence Texture Coherence Geometry Quality Layout Coherence Texture Coherence

Trellis (Xiang et al., 2024) 31.50% 30.00% 33.00% 17.58% 19.04% 14.75% SceneFuse-3D (Ours) 68.50% 70.00% 67.00% 82.42% 80.96% 85.25%

Hunyuan3D-2 (Team, 2025) 33.50% 33.50% 31.50% 11.66% 12.13% 7.67% SceneFuse-3D (Ours) 66.50% 66.50% 88.34% 88.34% 87.87% 92.33%

TripoSG (Li et al., 2025) 22.50% 23.00% 26.00% 24.60% 22.34% 22.79% SceneFuse-3D (Ours) 77.50% 77.00% 74.00% 75.40% 77.66% 77.21%

LGM (Tang et al., 2024b) 0.00% 0.00% 0.00% 0.60% 0.00% 0.00% SceneFuse-3D (Ours) 100.00% 100.00% 100.00% 100.00% 100.00% 100.00%

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

Benchmark To the best of our knowledge, there is no established benchmark for 3D outdoor scene mesh generation from single images. Therefore, we construct a custom test set by prompting GPT-4o (Hurst et al., 2024) to generate 100 diverse top-down scene images in a variety of styles, such as “snow village,” “desert town,” and more. The generation details can be found in Appendix C.

Metrics Without ground-truth meshes, we evaluate pairwise model comparisons per reference image on three criteria, geometry quality, layout coherence, and texture coherence, assessing detail fidelity, spatial arrangement, and texture–image alignment, respectively. Each pair is annotated by two AMT workers to yield a human-preference win rate. We also report a GPT-4o weighted win rate: given the same pair, GPT-4o outputs token probabilities for “A/B”; we use the probability of the chosen option as a soft vote and average over pairs. All scenes are rendered to RGB images with identical Blender settings; further GPT prompt details appear in Appendix C. Additionally, we report rendered-view image metrics between each method’s render and its reference image: CLIP similarity (openai/clip-vit-base-patch32, higher is better) and FID (Inception-V3, lower is better), averaged over the test set; these complement the human/GPT criteria by measuring image-level fidelity.

Baselines We compare SceneFuse-3D with four image-to-3D generation methods: Trellis (Xiang et al., 2024), Hunyuan3D-2 (Team, 2025), TripoSG (Li et al., 2025), and LGM (Tang et al., 2024b). While Trellis, Hunyuan3D-2 and TripoSG represent the state-of-the-art end-to-end 3D transformer models, LGM represents the multi-view generation-based 3D generator. All models are evaluated in a zero-shot setting using official pretrained checkpoints. To ensure fair comparison, background removal is disabled, and the full input image is encoded for all methods. We also experimented with the progressive novel-view method WonderWorld (Yu et al., 2025). As it does not yield a consistent editable mesh, it is not included in our mesh-based quantitative tables; we provide qualitative comparisons and discussion in Appendix A.2.

- 4.2 MAIN RESULTS Table 1, 2 and Figure 3 present the main quantitative and qualitative comparisons between SceneFuse-
- 3D and baseline methods. The results clearly demonstrate that SceneFuse-3D consistently outperforms Trellis (Xiang et al., 2024), Hunyuan3D-2 (Team, 2025), TripoSG (Li et al., 2025), and LGM (Tang et al., 2024b) across geometry, layout, and texture quality, as evaluated by both human annotators and GPT-4o. These improvements can be attributed to our region-based design and spatially guided generation strategy, which together promote better alignment between image features and 3D content, while preserving scene-wide consistency.

Quantitative Analysis SceneFuse-3D ’s region-wise decomposition aligns each latent block with a localized image crop, reducing the domain gap between object-centric training and scene-level inference. This design boosts texture fidelity, with GPT-4o assigning a 92.3% win rate versus only 7.7% for Hunyuan3D-2. The upscaled resolution further enhances structural detail, reflected in geometry improvements of +37 points over Trellis (68.5% vs. 31.5%) and +55 points over TripoSG (77.5% vs. 22.5%). Spatial priors and masked 3D inpainting stabilize layouts and smooth inter-region

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

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

Reference Image SceneFuse-3D Trellis Hunyuan3D-2 TripoSG LGM

- Figure 3: Qualitative comparisons between SceneFuse-3D and baselines. Given a single top-down image (left column), we compare 3D scene outputs generated by SceneFuse-3D, Trellis (Xiang et al., 2024), Hunyuan3D-2 (Team, 2025), TripoSG (Li et al., 2025), and LGM (Tang et al., 2024b). SceneFuse-3D consistently produces globally coherent scenes with fine-grained geometry, accurate object layouts, and realistic textures across a variety of styles and environments. In contrast, Trellis often produces oversimplified geometry; Hunyuan3D-2 suffers from structural inconsistencies and domain mismatch; TripoSG exhibits repetition artifacts and layout drift; LGM cannot generate consistent multi-view scene images for 3D construction.

transitions, yielding higher layout coherence (70.0% vs. 30.0% for Trellis in human study; 87.9% vs. 12.1% for Hunyuan3D-2 in GPT-4o evaluation). LGM, leveraging multi-view image generation, fails to provide consistent geometry in this setting and collapses to oversimplified outputs.

Rendered-view metrics in Table 2 support these findings. SceneFuse-3D achieves the highest CLIP similarity (0.8030), indicating stronger semantic and visual alignment with the input than all baselines. It also records the best FID (258.74), showing the closest distributional match in image structure, while Trellis follows at 288.23, and the others exceed 300. Together, these results confirm that SceneFuse-3D not only excels in human and GPT-4o preference studies but also in reference-based image metrics, underscoring its advantages in both semantic fidelity and distributional realism.

Qualitative Analysis Qualitatively, SceneFuse-3D produces scene assets with clear structure, consistent layout, and fine-grained surface details that closely match the reference top-down image. In contrast, Trellis often generates overly centralized, low-resolution structures and lacks peripheral detail. Hunyuan3D-2 exhibits notable issues with layout distortion and geometry hallucinations despite acceptable textures in isolated parts. TripoSG maintains some compositional structure but frequently introduces repeated objects and ignores the layout evidence within the reference image. LGM can only produce a hollow cube with the scene texture mapped onto its faces. SceneFuse-3D ’s region-wise generation and spatial inpainting pipeline helps it avoid these artifacts while maintaining both global coherence and local fidelity.

These findings confirm that spatial decomposition and prior-guided inpainting are effective principles for lifting single-view image inputs into coherent, high-quality 3D scenes. Additional qualitative comparisons are available in Figure 5 in Appendix A.1.

4.3 ABLATION STUDY

We conduct ablation studies to evaluate the contributions of key components in SceneFuse-3D: the region-based generation strategy and the use of pre-generated landmarks. Both ablation studies still

##### Table 2: Rendered-view metrics on the reference images. CLIP similarity and FID are computed between each method’s render and the corresponding reference image.

Models CLIP ↑ FID ↓ SceneFuse-3D (Ours) 0.8030 258.74 Trellis (Xiang et al., 2024) 0.7760 288.23 Hunyuan3D-2 (Team, 2025) 0.7716 302.26 TripoSG (Li et al., 2025) 0.7203 353.78 LGM (Tang et al., 2024b) 0.7510 353.86

##### Table 3: Ablation Study Results. Win rates for geometry, layout, and texture show that removing region-based generation or landmark conditioning degrades performance, highlighting the importance of both components in SceneFuse-3D.

Human Preference Win Rate (Percentage) GPT-4o-based Weighted Win Rate (Percentage) Models Geometry Quality Layout Coherence Texture Coherence Geometry Quality Layout Coherence Texture Coherence SceneFuse-3D (w/o regions) 20.00% 17.00% 13.50% 6.11% 8.48% 6.08% SceneFuse-3D 80.00% 83.00% 86.50% 92.89% 91.52% 92.92% SceneFuse-3D (w/o landmarks) 41.50% 46.00% 42.00% 36.90% 44.21% 38.26% SceneFuse-3D 59.50% 54.00% 58.00% 64.10% 56.79% 61.74%

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

SceneFuse-3D (w/o Landmarks)

SceneFuse-3D (w/o Regions)

Reference Image SceneFuse-3D

- Figure 4: Qualitative Ablation Results. Left: Reference image. Middle: SceneFuse-3D without landmark conditioning. Right: SceneFuse-3D without region-based generation. Landmark conditioning ensures consistency for foreground objects, especially for objects across regions, while region-based generation preserves overall detail and coherence.

apply the spatial-aware 3D completion. Quantitative results are shown in Table 3, and qualitative comparisons are illustrated in Figure 4.

Without Region-Based Generation In this setting, the entire scene latent is directly passed into the pretrained 3D generator, without being split into localized regions. This leads to severe performance drops across all metrics. The results suggest that holistic generation fails to make full use of the pretrained model’s capacity, which was originally trained on single-object inputs. Without localized conditioning, the model struggles to resolve spatial context and image-to-3D correspondence, producing low-resolution and spatially incoherent outputs. As illustrated in Figure 4 (right), buildings lose structural sharpness and alignment, and the overall layout becomes underspecified.

Without Landmark Conditioning Removing landmark-aware initialization (depth-only prior) degrades geometry and layout, especially around large foreground structures (e.g., gates/towers). Landmarks act as semantic and geometric anchors across patches: they fix orientation/scale and provide reliable context for masked completion. Without them, region completions drift at boundaries, yielding duplicated fa¸cades or misaligned parts across neighboring patches (Fig. 4, middle).

Overall, these ablations show complementary roles: region-wise decomposition keeps generation within the base model’s effective receptive field and preserves local detail, while landmark anchors enforce cross-patch continuity and stabilize global structure—both are necessary for coherent, highfidelity scenes.

- 5 CONCLUSION

To address the challenge of generating high-quality, coherent 3D scenes from a single image, we proposed SceneFuse-3D, a training-free framework that decomposes scenes into overlapping regions and guides generation with spatial priors. A spatial-aware 3D completion with masked rectified

flow preserves local object fidelity while enforcing global coherence. Empirically, SceneFuse-3D outperforms existing methods across geometry, texture, and layout, underscoring the promise of modular, spatially grounded generation for stereotype 3D scene synthesis from minimal input.

REFERENCES

Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein. Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2139–2150, 2023.

Ata C¸elen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. I-design: Personalized llm interior designer. arXiv preprint arXiv:2404.02838, 2024.

Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, and Noah Snavely. Persistent nature: A generative model of unbounded 3d worlds. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20863–20874, 2023. URL https://api.semanticscholar. org/CorpusID:257687856.

Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, et al. Flux: fast software-based communication overlap on gpus through kernel fusion. arXiv preprint arXiv:2406.06858, 2024.

Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram S. Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. ArXiv, abs/2307.05663, 2023. URL https://api.semanticscholar.org/CorpusID:259836993.

Paul Engstler, Andrea Vedaldi, Iro Laina, and Christian Rupprecht. Invisible stitch: Generating smooth 3d scenes with depth inpainting. arXiv preprint arXiv:2404.19758, 2024.

Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. arXiv preprint arXiv:2503.16420, 2025.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36:18225–18250, 2023.

Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. Advances in Neural Information Processing Systems, 36:39897–39914, 2023.

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024.

Anchit Gupta and Anchit Gupta. 3dgen: Triplane latent diffusion for textured mesh generation. ArXiv, abs/2303.05371, 2023. URL https://api.semanticscholar.org/CorpusID: 257427345.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7909–7920, 2023.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid, and Alireza Fathi. Scenecraft: An llm agent for synthesizing 3d scenes as blender code. In Forty-first International Conference on Machine Learning, 2024.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson. Pathdreamer: A world model for indoor navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14738–14748, 2021.

Jing Yu Koh, Harsh Agrawal, Dhruv Batra, Richard Tucker, Austin Waters, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson. Simple and effective synthesis of indoor 3d scenes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 1169–1178, 2023.

Jumin Lee, Sebin Lee, Changho Jo, Woobin Im, Juhyeong Seon, and Sung-Eui Yoon. Semcity: Semantic scene generation with triplane diffusion. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28337–28347, 2024. URL https: //api.semanticscholar.org/CorpusID:268363839.

Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. ArXiv, abs/2405.14979, 2024a. URL https://api.semanticscholar.org/CorpusID: 270045086.

Wenrui Li, Fucheng Cai, Yapeng Mi, Zhe Yang, Wangmeng Zuo, Xingtao Wang, and Xiaopeng Fan. Scenedreamer360: Text-driven 3d-consistent scene generation with panoramic gaussian splatting. arXiv preprint arXiv:2408.13711, 2024b.

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025.

Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision, pp. 515–534. Springer, 2022.

Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6517–6526, 2024.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 300–309, 2022. URL https://api.semanticscholar.org/CorpusID:253708074.

Chenguo Lin and Yadong Mu. Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior. arXiv preprint arXiv:2402.04717, 2024.

Aoming Liu, Zhong Li, Zhang Chen, Nannan Li, Yi Xu, and Bryan A Plummer. Panofree: Tuningfree holistic multi-view image generation with cross-view self-guidance. In European Conference on Computer Vision, pp. 146–164. Springer, 2024.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9264–9275, 2023a. URL https://api.semanticscholar. org/CorpusID:257631738.

Yuheng Liu, Xinke Li, Xueting Li, Lu Qi, Chongshou Li, and Ming-Hsuan Yang. Pyramid diffusion for fine 3d large scene generation. ArXiv, abs/2311.12085, 2023b. URL https: //api.semanticscholar.org/CorpusID:265308971.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471, 2022.

L´eopold Maillard, Nicolas Sereyjol-Garros, Tom Durand, and Maks Ovsjanikov. Debara: Denoisingbased 3d room arrangement generation. Advances in Neural Information Processing Systems, 37: 109202–109232, 2024.

Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. Lt3sd: Latent trees for 3d scene diffusion. ArXiv, abs/2409.08215, 2024. URL https://api.semanticscholar.org/CorpusID: 272600456.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/, 2025.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. ArXiv, abs/2209.14988, 2022. URL https://api.semanticscholar.org/ CorpusID:252596091.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4209–4219, 2023. URL https: //api.semanticscholar.org/CorpusID:273025441.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Szymon Rusinkiewicz and Marc Levoy. Efficient variants of the icp algorithm. In Proceedings third international conference on 3-D digital imaging and modeling, pp. 145–152. IEEE, 2001.

Jonas Schult, Sam Tsai, Lukas H¨ollein, Bichen Wu, Jialiang Wang, Chih-Yao Ma, Kunpeng Li, Xiaofang Wang, Felix Wimbauer, Zijian He, et al. Controlroom3d: Room generation using semantic proxy rooms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6201–6210, 2024.

Jaidev Shriram, Alex Trevithick, Lingjie Liu, and Ravi Ramamoorthi. Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion. arXiv preprint arXiv:2404.07199, 2024.

Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, et al. Ldm3d: Latent diffusion model for 3d. arXiv preprint arXiv:2305.10853, 2023.

Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 3d-gpt: Procedural 3d modeling with large language models. arXiv preprint arXiv:2310.12945, 2023.

Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. Diffuscene: Denoising diffusion models for generative indoor scene synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20507–20518, 2024a.

Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. ArXiv, abs/2309.16653, 2023a. URL https: //api.semanticscholar.org/CorpusID:263131552.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pp. 1–18. Springer, 2024b.

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22762–22772, 2023b. URL https: //api.semanticscholar.org/CorpusID:257757320.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36: 1363–1389, 2023c.

Tencent Hunyuan3D Team. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation, 2025.

Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems, 35:10021–10039, 2022.

Alexander Vilesov, Pradyumna Chari, and Achuta Kadambi. Cg3d: Compositional generation for text-to-3d via gaussian splatting. arXiv preprint arXiv:2311.17907, 2023.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. Synsin: End-to-end view synthesis from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7467–7477, 2020.

Tianhao Wu, Chuanxia Zheng, and Tat-Jen Cham. Panodiffusion: 360-degree panorama outpainting via diffusion. arXiv preprint arXiv:2307.03177, 2023.

Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. ACM Transactions on Graphics (TOG), 43(4):1–17, 2024.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.

Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4818–4829, 2024.

Bojun Xiong, Si-Tong Wei, Xin-Yang Zheng, Yan-Pei Cao, Zhouhui Lian, and Peng-Shuai Wang. Octfusion: Octree-based diffusion models for 3d shape generation. ArXiv, abs/2408.14732, 2024. URL https://api.semanticscholar.org/CorpusID:271962988.

Yiying Yang, Fukun Yin, Jiayuan Fan, Xin Chen, Wanzhang Li, and Gang Yu. Scene123: One prompt to 3d scene generation via video-assisted and consistency-enhanced mae. arXiv preprint arXiv:2408.05477, 2024.

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. arXiv preprint arXiv:2406.09394, 2024a.

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6658–6667, 2024b.

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5916–5926, 2025.

Guangyao Zhai, Evin Pınar Ornek,¨ Dave Zhenyu Chen, Ruotong Liao, Yan Di, Nassir Navab, Federico Tombari, and Benjamin Busam. Echoscene: Indoor scene generation via information echo over scene graph diffusion. In European Conference on Computer Vision, pp. 167–184. Springer, 2024.

Jingbo Zhang, Xiaoyu Li, Ziyu Wan, Can Wang, and Jing Liao. Text2nerf: Text-driven 3d scene generation with neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 30(12):7749–7762, 2024a.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating highquality 3d assets. ACM Transactions on Graphics (TOG), 43:1 – 20, 2024b. URL https: //api.semanticscholar.org/CorpusID:270619933.

Songchun Zhang, Yibo Zhang, Quan Zheng, Rui Ma, Wei Hua, Hujun Bao, Weiwei Xu, and Changqing Zou. 3d-scenedreamer: Text-driven 3d-consistent scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10170–10180, 2024c.

Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Gala3d: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting. arXiv preprint arXiv:2402.07207, 2024.

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

Reference Image SceneFuse-3D Trellis Hunyuan3D-2 TripoSG

Figure 5: More qualitative comparisons between SceneFuse-3D and baselines. From the image, we can find that SceneFuse-3D can generate more coherent scenes from diverse scene images. LGM is skipped since it fails to generate structured scenes for all inputs.

- A MORE RESULTS

- A.1 ADDITIONAL QUALITATIVE RESULTS

- Figure 5 presents additional qualitative results comparing SceneFuse-3D with Trellis (Xiang et al., 2024), Hunyuan3D-2 (Team, 2025), and TripoSG (Li et al., 2025) across a diverse set of visual scenes. These examples further demonstrate the robustness and generality of our approach across different architectural styles, spatial layouts, and artistic domains.

SceneFuse-3D consistently produces scene assets that are geometrically detailed, visually coherent, and well-aligned with the input reference images. In contrast, baseline models frequently suffer from artifacts such as repeated structures, layout collapse, or low-resolution textures. These comparisons

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

Reference Image View 1 View 2 View 3 View 4

- Figure 6: Results of the WonderWorld. As long as the viewpoint changes, we can find that WonderWorld cannot generate consistent geometry information for 3D scene generation.

further highlight the benefits of our region-based generation and spatial-aware inpainting pipeline in generating diverse 3D scenes from a single image without training.

- A.2 ADDITIONAL BASELINE

WonderWorld (Yu et al., 2025) enables novel view synthesis of 3D scenes from a single image through text-guided inpainting, which is designed to be effective when a coarse single-view Gaussian representation is initialized and can be refined and enriched across different views. In our task of constructing a consistent 3D scene, we set up the camera trajectory to rotate along the vertical normal to the horizontal plane of the reference images for a comprehensive scene synthesis. However, WonderWorld struggles to generate view-consistent structures and often fails to update the Gaussians effectively due to suboptimal inpainting results based on the existing observations, shown in Fig. 6.

- B METHOD DETAILS

Region Extraction Strategy. To generate overlapping regions with balanced seams, we first compute the tight bounding box of all occupied voxels and tile it with a base grid of non-overlapping patches of size (px,py,pz). For the vertical (z) axis, start positions are evenly interpolated so that the entire height is covered with the minimum number of full patches. Next, we insert seam patches to equalise overlap: for every pair of neighbouring base-grid origins we create an additional patch whose origin is the midpoint between them—first along the x-axis (keeping y,z fixed), then along the y-axis (keeping x,z fixed). The union of base and seam origins is deduplicated and sorted, after which the corresponding voxel sub-volumes are extracted. The procedure returns the list of patch origins and their binary masks, and is invoked once per scene to define the region schedule used in our region-wise generation and fusion pipeline.

Algorithm 1: Masked Rectified Flow for Completion Pipeline Input : vθ(x,t) — learned flow field; xknown — known latent to preserve;

m ∈ {0,1} — mask for regeneration (1=regenerate, 0=preserve); T — total steps; U — Resample times per step; σmin — minimum noise scale

Output:x0 — regenerated latent /* Forward-noise operator */ ϵ ∼ N(0,I) forward step(x,t) = (1 − t)x + σmin + (1 − σmin)t ϵ /* Initialization */ xT ∼ N(0,I); for t = T,T − 1,...,1 do

tprev ← t − 1; ∆t ← t − tprev; for u = 1,...,U do

v ← vθ(xt,t) ; /* predict flow field */ xt

prev ← xt − ∆tv ; /* Euler update on unknown */ xˆt

prev ← forward step(xknown, tprev) ; /* re-noise known */ xt

; if u < U and t > 1 then

prev ← m ⊙ xt

#### + (1 − m) ⊙ xˆt

prev

prev

, ∆t); else

xt ← forward step(xt

prev

### ; return x0 = x0;

xt ← xt

prev

Masked Rectified Flow Algorithm For completeness, we provide the full algorithmic details of the masked rectified flow completion process used in our spatial-aware 3D inpainting pipeline. While the core formulation is introduced in Section 3.3, this pseudocode (Algorithm 1) clarifies the iterative update, re-noising, and resampling procedures that enable conditional generation of unknown regions while preserving the known latent structure.

Implementation Details For all rectified flow generators, we step the sampling time step T = 50. The classifier-free guidance scales are 7.5 and 5 for the spare structure generator and structured latent generator. Resampling time U during the masked rectified flow is set to 2. The whole pipeline can be loaded with an NVIDIA RTX A5000 GPU with 24G VRAM.

- C EXPERIMENT SETTINGS

Test Set Generation To evaluate models’ performance under stylistically diverse conditions, we curated a human-verified synthesized image test set with 100 top-down views. Given an example image, we ask ChatGPT-o3 (OpenAI, 2025) to generate image prompts for top-down scene views with these requirements: 1280 × 720 resolution, quasi-orthographic three-quarter (“isometric”) camera, one hero landmark at the image centre, 10–20 surrounding buildings, daylight illumination, and “no far-away object”. Then, we used GPT-4o (Hurst et al., 2024) to generate scene images according to the corresponding prompts.

Human Evaluation Given a reference image and observations of two generated scenes, we ask the human annotator to answer these three questions:

- • Which scene, A or B, has geometry that is more detailed, precise, and closer to the reference image?
- • Which scene, A or B, demonstrates a spatial layout and arrangement of objects that is more coherent and closely aligned with the layout in the reference image?

- • Which scene, A or B, exhibits textures that are significantly more coherent and consistent with the reference image?

GPT-4o-based Evaluation For GPT-based automatic evaluation, we ask the same questions as the human evaluation and prompt the model to directly return the answer with top-5 token log probability. Then, we extract the token probability of ‘A’ and ‘B’ (0 if not included) as the answer weights. We treat P(A) as a soft vote when our method is option A (and analogously for B). Weighted win rate is computed as P(win) if our model wins, and 1 − P(lose) otherwise, then averaged across all pairs.

- D LIMITATION Although SceneFuse-3D delivers strong scene-level results, several challenges remain. The pretrained

- 3D generator we adopt is trained on single-object imagery; even after region decomposition, the underlying distribution mismatch can lead to patch-level hallucinations—for example, duplicated fac¸ades or unrealistic roof shapes. Future work could mitigate this via scene-level fine-tuning or domain adaptation.

Our coarse spatial prior contains many holes where occlusions obscure geometry. Regions dominated by such voids sometimes inherit empty or oversmoothed surfaces from the generator. Integrating uncertainty-aware depth completion, multi-view cues, or semantic priors may yield denser scaffolds and more reliable inpainting in future work.

