# arXiv:2501.09433v1[cs.CV]16Jan2025

## CaPa: Carve-n-Paint Synthesis for Efficient 4K Textured Mesh Generation

Hwan Heo Jangyeong Kim Seongyeong Lee

Jeong A Wi Junyoung Choi Sangjun Ahn * Graphics AI Lab, NC Research

{hwanheo, jangyeongk, seongyeong2, jaywi, jychoi13, sjahn21}@ncsoft.com

[Figure 1]

Figure 1. Comparison of mesh quality with state-of-the-art image-to-3D methods. CaPa can generate a hyper-quality textured mesh in under 30 seconds, providing 3D assets ready for commercial applications such as games, movies, and VR/AR.

### Abstract

The synthesis of high-quality 3D assets from textual or visual inputs has become a central objective in modern generative modeling. Despite the proliferation of 3D generation algorithms, they frequently grapple with challenges such as multi-view inconsistency, slow generation times, low fidelity, and surface reconstruction problems. While some studies have addressed some of these issues, a comprehensive solution remains elusive. In this paper, we introduce CaPa, a carve-and-paint framework that generates high-fidelity 3D assets efficiently. CaPa employs a two-stage process, decoupling geometry generation from texture synthesis. Initially, a 3D latent diffusion model generates geometry guided by multi-view inputs, ensuring structural consistency across perspectives. Subsequently, leveraging a novel, model-agnostic Spatially Decoupled Attention, the framework synthesizes high-resolution textures (up to 4K) for a given geometry. Furthermore, we propose a 3D-aware occlusion inpainting algorithm that fills untextured regions, resulting in cohesive results across the entire model. This pipeline generates high-quality 3D assets in less than 30 seconds, providing ready-to-use outputs for commercial applications. Experimental results demonstrate that CaPa excels in both texture fidelity and geometric stability, establishing a new standard for practical, scalable 3D asset generation.

*corresponding author.

### 1. Introduction

The demand for scalable, high-quality 3D assets is rapidly growing across industries such as gaming, film, and VR/AR. While recent advancements in machine learning have led to remarkable achievements in text and image generation, extending these successes to 3D content creation has been challenging. The high-dimensional nature of 3D data, coupled with a scarcity of diverse and high-quality datasets, has constrained the progress of 3D generative models and created unique challenges in achieving efficiency and fidelity.

Despite these inherent obstacles, researchers have made considerable efforts to adapt generative techniques for 3D, resulting in unprecedented growth and exploration. A prominent approach involves lifting 2D diffusion models to 3D through Score Distillation Sampling (SDS) [33, 48]. However, SDS often encounters limitations such as the Janus problem (texture discrepancies across views), difficulties in achieving high-quality mesh, and prolonged generation times. These shortcomings arise largely because of the overreliance on indirect 3D information from 2D models, which hampers both the efficiency and fidelity of generated outputs.

In response, recent advances in Large Reconstruction Models (LRM) [13, 60] have resolved the slow speed issue by leveraging autoregressive generative models [47]. While promising, these methods continue to face trade-offs, particularly in output quality. Moreover, most 3D generation methods rely on NeRF [29] or Gaussian Splatting [16] as their

base representation, which has challenges for accurate surface reconstruction. When converting their representations into usable forms (i.e., mesh), quality degradation occurs, remaining a key bottleneck for downstream tasks.

To address these challenges, we introduce CaPa, a twostage Carve-n-Paint framework designed to generate highquality, clean 3D meshes efficiently. By decoupling geometry generation and texture synthesis, CaPa enhances both flexibility and performance at each stage, allowing for precise reconstruction of the mesh and detailed texture output.

In the geometry generation stage, CaPa leverages a 3D latent diffusion model. It focuses on producing an occupancy field that is not only structurally consistent across multiple views but also mesh-compatible. This process utilizes multiview inputs to guide the generation, ensuring precise geometry that seamlessly integrates with the subsequent texture synthesis stage. The resulting high-quality mesh provides a stable foundation, crucial for achieving cohesive textures.

In the texture synthesis stage, we employ a novel Spatially Decoupled Cross Attention, which generates highresolution textures (up to 4K). By isolating and processing view-specific features within a unified diffusion framework, this method effectively resolves multi-view inconsistencies, such as the Janus problem. Notably, its model-agnostic approach does not require architectural modifications or extensive retraining. This design integrates smoothly with large pre-trained models like SDXL [32], outperforming previous approaches. As a result, it produces geometry-aligned, cohesive, and high-fidelity textures as shown in Figure 1.

Additionally, to further enhance texture completeness, CaPa introduces a 3D-aware occlusion inpainting algorithm. This algorithm efficiently fills untextured regions by generating a UV map that preserves surface locality, enabling seamless inpainting while minimizing visible seams.

Comprehensive experiments validate the effectiveness of CaPa, demonstrating significant improvements in both texture fidelity and geometric stability over existing methods. CaPa consistently delivers high-fidelity polygonal meshes in a fraction of the time, setting a new standard for practical and scalable 3D asset generation.

The core contributions of this work are summarized as:

- • Mesh-Optimized 3D Generation Pipeline: Separating geometry and texture generation, enabling effective meshing and enhancing the practicality of 3D asset generation.
- • Spatially Decoupled Cross Attention: A model-agnostic, training-free solution to generate 4K textures without additional training, super-resolution, and janus problem.
- • 3D-aware Fast Occlusion Inpainting: A robust inpainting solution for handling occluded regions in 3D textures, reducing visible seams, and preserving texture fidelity.
- • State-of-the-art Results: CaPa achieves significantly higher texture and geometry fidelity, marking a major step forward in practical 3D asset generation.

### 2. Related Work

#### 2.1. 3D Asset Generation

Given the recent advancements in neural rendering [29] and diffusion-based generative models, researchers have explored this success to the 3D generation area. A notable approach is Score-Distillation Sampling (SDS) [33, 48], which utilizes 2D diffusion priors to generate 3D assets. Several subsequent studies [19, 20, 43, 46, 50, 58] have provided a more robust theoretical foundation, post-processing, or better visual fidelity. Despite these advances, these methods’ reliance on Neural Radiation Fields (NeRF) as the underlying 3D representation results in slow generation times and poor quality after mesh conversion.

Another prominent limitation of SDS is the “Janus problem,” where artifacts such as multiple faces appear. To mitigate this issue, researchers have explored enhancing diffusion models with a multi-view generation [23, 40, 49] by adjusting U-Net to interact across these views. Although these methods largely resolved the Janus problem, the SDS’s progressive updating still results in slower generation times. To accelerate the slow speed, certain studies [27, 54] have proposed generating normal maps from multi-view images and employing differentiable mesh updates or sparse view neural reconstruction. Nevertheless, this approach often leads to unstable geometry, or low-fidelity textures (refer to the second, third column of Figure 6).

#### 2.2. 3D-Native Reconstruction Models

Recently, Large Reconstruction Models (LRMs) [13, 41, 45, 51, 56, 60], a transformer models for directly generating 3D assets have emerged. LRMs predict the NeRF or Gaussian Splatting parameters directly through feed-forward inferences. These approaches significantly reduce generation times and produce 3D assets in near real-time. However, scaling these methods to higher resolutions remains challenging, with most state-of-the-art LRMs currently limited to low-fidelity outputs (see the fourth sample in Fiure 1).

Recently, the research community has explored alternative approaches for 3D shape generation through 3D latent diffusion, diverging from the LRM paradigm. Michelangelo [63] introduced VAE for 3D latent representation, establishing an aligned latent space for images, text, and 3D data. Several following works [52, 55, 62] enhance its geometry understanding and address its limitations in diversity. Concurrently, CLAY [62] demonstrated that 3D geometry generation, particularly using DiT [31] (Diffusion Transformer), could scale similar to other large foundational models in deep learning. However, these models focus primarily on shape generation. To synthesize textures, they rely on external texturing methods [4, 5, 24] or multi-view diffusion. Additionally, all the aforementioned methods are time-consuming, and aligning textures with generated shapes remains difficult.

#### 2.3. Texture Generation

Numerous texture generation algorithms have recently been developed with the success of 2D generative diffusion models. Common approaches [5, 35] utilize iterative processes, first rendering depth maps and then generating corresponding images for multiple viewpoints. However, these methods mainly encounter multi-view consistency issues. Recognizing this challenge, recent approaches [9, 17, 41, 59] often focus on synchronized output within a limited number of views, typically 4 or 6 orthogonal viewpoints, to improve alignment throughout the entire model.

Despite these impressive results, the limited number of views inevitably yields occlusions, leading to incomplete textures. Aware of this problem, various strategies have been explored, especially for inpainting using ControlNet in UV space [41, 59]. However, these solutions frequently introduce locality biases in UV space, which do not accurately reflect the 3D once mapped onto a 2D Cartesian plane. As a result, such methods often produce degraded details in occluded areas, compromising texture coherence.

Concurrent work, MVPaint [6], has similarly identified these limitations, suggesting surface-based extrapolation as a potential solution. In contrast, we address this issue using 3Daware occlusion inpainting, which preserves texture fidelity and enhances overall consistency.

### 3. Methodology

This section introduces CaPa, a novel 3D asset generation framework. CaPa operates in two stages: (1) 3D geometry synthesis using multi-view guided 3D latent diffusion and (2)

- 2D texture generation via 2D latent diffusion. First, we generate a 3D geometry by leveraging multi-view guided 3D latent diffusion to capture structural details from diverse perspectives. Second, to create an initial texture, we synthesize four orthogonal view images along the generated mesh. The untextured regions, caused by occlusions, are then completed using a 3D-aware inpainting algorithm to ensure a seamless appearance across all surfaces. Both stages are aligned through shared multi-view images, ensuring consistent geometry and texture. Using a fully feed-forward approach, the entire 3D asset generation process is completed in less than 30 seconds. Figure 2 illustrates the CaPa pipeline.

#### 3.1. Geometry Generation via 3D Latent Diffusion

For 3D geometry generation, we leverage the recently proposed ShapeVAE to model the distribution of 3D geometry, while ensuring alignment with both image and text latent spaces. Using ShapeVAE, we train a 3D latent diffusion model [36] guided by multi-view (MV) images to achieve better alignment with texture generation. This MVconditioned 3D latent diffusion model can efficiently generate high-quality 3D shapes, even under complex scenarios.

##### 3.1.1 Latent Space for Geometry Representation

We adopt a perceiver-based encoder that captures essential 3D geometric features, in line with Michelangelo [63]. The encoder is trained to reconstruct neural fields from input point clouds, ensuring efficient representation learning.

Shape Encoder. Given a ground-truth 3D shape, we aim to encode it into a latent representation, Es. First, we sample points from the surface P ∈ RN×3, along with their corresponding normal vectors n. These points and normals are then processed using cross-attention mechanisms that incorporate Fourier encodings and spherical harmonics to capture fine geometric details.

Shape Decoder. The perceiver-based decoder D reconstructs the neural field from latent embeddings Es. Given a query point p ∈ R3 and Es, the decoder predicts the occupancy value for each point. The reconstruction process is optimized using binary cross-entropy (BCE) loss,

Lvae = Ep∈R3 BCE O ˆ(p),D (p|Es) + λklLkl (1)

where Oˆ(p) represents the occupancy prediction for the query point p, and Lkl for KL divergence regularization.

##### 3.1.2 Multi-View Guided 3D Latent Diffusion

Leveraging the learned latent space, we employ a 3D latent diffusion model to generate geometry. However, due to its reliance on the latent space learned by ShapeVAE, this model often demonstrates limited geometric understanding. Moreover, establishing a unified generation pipeline requires consistency between the generated shape and texture.

To address these challenges, we train a multi-view guided

###### 3D latent diffusion model that incorporates richer geometric priors and maintains alignment between shape and texture. Specifically, we use an off-the-shelf MV diffusion model to generate a set of multi-view images ˆI from the input condition (e.g., an image or text). This guidance provides stronger priors for 3D shape generation, ensuring improved alignment with texture synthesis in subsequent steps.

Finally, we train the multi-view guided 3D diffusion model ϵθ. The training objective follows the standard reverse denoising process of latent diffusion [12, 36, 42]:

L3D-LDM := Eϵ∼N(0,1),t ∥ϵ − ϵθ(Es,t,τθ(ˆI,π))∥22 (2)

where τθ extracts camera embeddings from the multi-view images ˆI to incorporate camera-specific parameters πi.

After applying the Marching Cubes [26] to extract a 3D mesh from the occupancy field, we employ a robust remeshing process to ensure smooth and high-quality manifold geometry. We carefully design the post-processing

[Figure 2]

[Figure 3]

Stage1:GeometryGenerationStage2:TextureGeneration

###### 3D Latent Diffusion

|𝐸𝑠|
|---|

[Figure 4]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Shape latent

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

MVDiffusion

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 10]

Input

3D Geometry

MV images Conditioning

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

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

[Figure 29]

Back-Projecting

3D-Aware

[Figure 30]

2D Latent Diffusion w/ Spatially Decoupled Attention

Occlusion Inpainting

High-Quality MV images w/o Janus Rendered Input

Final Textured Mesh

- Figure 2. CaPa pipeline. We first generate 3D geometry using a 3D latent diffusion model. Using the learned 3D latent space with ShapeVAE, we train a 3D Latent Diffusion Model that generates 3D geometries, guided by multi-view images to ensure alignment between the generated shape and texture. After the 3D geometry is created, we render four orthogonal views of the mesh, which serve as inputs for texture generation. To produce a high-quality texture while preventing the Janus problem, we utilize a novel, model-agnostic spatially decoupled attention. Finally, we obtain a hyper-quality textured mesh through back projection and a 3D-aware occlusion inpainting algorithm.

pipeline to address common meshing challenges, such as aliasing and non-manifold structures. This remeshing strategy also supports artist-preferred quadrilateral meshes and high-resolution UV-mapped textures. For a comprehensive discussion, please refer to Appendix C.1.

#### 3.2. Texture Generation for Input Geometry

For texture generation, we first render four orthogonal views of the generated mesh, which serve as the input for texture synthesis. Ensuring texture consistency across these multiple views is critical but inherently challenging, as traditional approaches often struggle with the Janus problem.

A conventional solution for multi-view texture generation involves using MVDiffusions with geometry-guidance ControlNet [61], as suggested in CLAY [62] and MVPaint [6]. However, these approaches encounter several limitations: the output resolution for each view in MVDiffusion models typically remains constrained to a low 256-pixel resolution, resulting in suboptimal visual fidelity. Additionally, existing MVDiffusions rely on modified U-Net [37] architectures that differ from the original 2D latent diffusion model [36], thus precluding the reuse of pre-trained ControlNet and necessitating resource-intensive retraining.

##### 3.2.1 Spatially Decoupled Cross Attention

In response to these challenges, we introduce the Spatially Decoupled Cross Attention, a model-agnostic approach that amplifies multi-view information. To ensure texture

consistency, we first concatenate the rendered images into a single batch and synthesize textures for all views simultaneously. Feature channels are then replicated and assigned to view-specific guidance, allowing each channel to focus on its respective spatial region.

View-Specific Feature Enhancement. For a hidden feature ht ∈ RC×W×H at time step t, we split the feature channels into positive hf and negative components hb, both of size hf,hb ∈ RC/2×W×H. These features are then replicated across the N views:

h˜t = hf,...,hf

| hb,...,hb

N

N

∈ RCN×W×H. (3)

Then, we independently encode each low-quality multi-view guide image and combine them into a unified feature vector:

g˜ = gf1,...,gfN|gb1,...,gbN ∈ RCN×F. (4) The combined feature g˜, together with replicated hidden states h˜t, undergoes cross-attention,

Z = XAttn(˜g,h˜t) (5)

As a result of the attention, each channel group focuses on its designated view, preserving view-specific information. Finally, we aggregate the features corresponding only to the spatial region of the i-th view, ensuring that the guidance is attended to only in its respective region. Figure 3 illustrates the spatially decoupled attention mechanism.

positive

[Figure 31]

[Figure 32]

| |𝑔1| | | | |
|---|---|---|---|---|---|
|𝑔2| | | | | |
|𝑔3| | | | | |
|𝑔4| | | | | |

|𝐼1|
|---|

|𝐼2|
|---|

Image Encoder

[Figure 33]

[Figure 34]

|𝐼4|
|---|

|𝐼3|
|---|

negative

Low-Quality MV images

[Figure 35]

[Figure 36]

###### Spatially Decoupled Attention

[Figure 37]

| | | |
|---|---|---|
| | | |

𝑔̃

[Figure 38]

[Figure 39]

[Figure 40]

Decoupled

Attention map

|ℎ𝑡|
|---|

ℎ̃ 𝑡 = [ℎ𝑓,…,ℎ𝑓|ℎ𝑏,…, ℎ𝑏] 𝑔̃

[ℎ𝑓|ℎ𝑏]

Input Rendered view In Denoising U-Net

High-Quality Mesh-Aligned Images

[Figure 41]

- Figure 3. Spatially Decoupled Cross Attention. To produce highquality multi-view images for a given geometry, we design a model-agnostic Spatially Decoupled Cross Attention. During crossattention in denoising U-Net, we replicate hidden feature channels so that each duplicated channels focuses solely on the designated view. Since the design is model-agnostic, we can utilize an external ControlNet to guide the textures aligned with the input mesh.

Advantages of Spatially Decoupled Attention. The proposed mechanism allows each view’s spatial region to be processed independently within the same U-Net architecture, preserving view-specific details and ensuring consistency across perspectives. Therefore, it can utilize large pre-trained generative models like ControlNet or SDXL [32] without any architectural modifications or finetuning.

Intuitively, this approach amplifies low-quality, geometryunaligned multi-view guide images to high-quality, geometry-aligned multi-view images in a single diffusion process. Unlike conventional MVDiffusions, which require modified U-Net architecture and extensive retraining, the proposed method generates consistent, high-fidelity textures in a model-agnostic manner, bypassing traditional limitations.

##### 3.2.2 Occlusion Inpainting

In the initial texture generation, we create four orthogonal images for 3D geometry, which are then back-projected to produce a coarse-textured mesh. However, due to the limited number of views, occlusions inevitably arise in regions not visible from these views. To address this issue, we propose a novel 3D-aware occlusion inpainting algorithm.

- 3D-Aware Occlusion Mapping and Inpainting. As discussed in Section 2.3, existing occlusion inpainting methods have challenges for speed or locality bias. To overcome these issues, we propose a more adaptive, geometry-aware occlusion inpainting approach.

This method identifies and isolates the untextured regions on the mesh surface, targeting only the areas affected by occlusion. To capture the occluded areas more accurately, we employ k-means clustering using the face normal orientations and spatial coordinates of untextured faces on the 3D mesh surface. This clustering process yields several repre-

Occlusion-Aware Viewpoint Clustering

Specialized UV

[Figure 42]

|[Figure 43]<br><br>|
|---|

|[Figure 44]|
|---|

2D Inpainting

[Figure 45]

[Figure 46]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 47]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Final Mesh

Occlusion Normal

Occlusion Point

Figure 4. 3D-Aware Occlusion Inpainting. First, we cluster the normal and spatial coordinates of the occluded face. Using clustered centers as viewpoints, we create specialized UV maps through projection mapping. This approach captures surface locality, allowing 2D diffusion-based inpainting to effectively fill occluded regions. Note that this UV map is utilized solely for occlusion.

sentative cluster centroids, each corresponding to a distinct occluded region on the mesh. For each centroid, we define a viewpoint on the unit sphere that faces the untextured area, projecting the coarse-textured mesh from this viewpoint to capture a region-specific view of the occlusion.

We then aggregate these projections onto a single 2D plane, creating a specialized UV map dedicated solely to inpainting occluded areas. Unlike traditional UV maps, this occlusion-specific UV map maintains a more accurate representation of locality on the mesh surface, thus preserving the spatial coherence required for 2D inpainting methods. By leveraging this UV map, we use the same 2D diffusion model for texture generation and inpainting. This approach yields a high-fidelity, 3D-aware texture that effectively fills occluded regions. Empirically, we found that k = 6 covers most occluded areas, and further apply the extrapolation technique for other remaining regions. Figure 4 illustrates the 3D-aware occlusion inpainting scheme.

### 4. Experiments

#### 4.1. Implementation Details

Dataset. We utilize the Objaverse [8] dataset for ShapeVAE and 3D latent diffusion training. We preprocess this dataset by filtering out low-quality, non-manifold meshes, retaining 150K high-quality objects with high CLIP [34] scores.

Geometry Generation. We employ a perceiver-based [15] transformer architecture for ShapeVAE, as used in Michelangelo [63], modifying it with Fourier featuring and spherical harmonics encoding. For the multi-view guided 3D latent diffusion stage, we adopt a U-Net-based transformer. The entire training takes around 8 days on 32 NVIDIA A100 GPUs. Finally, we integrate a custom-trained MVDiffusion model,

following the setup in ImageDream [49], but modify it to treat the input view as the frontal view rather than estimating.

Texture Generation. We implement the entire texture generation mechanism for texture synthesis upon the SDXL [32]. We utilize pre-trained IP Adapter [57] layer and CLIP [34] for the image feature extractor, Depth-ControlNet [61] for a geometric guide. Appendix C.2 provides the additional implementation specifications.

#### 4.2. Qualitative Comparison

##### 4.2.1 Comparison with State-of-the-Art Methods

We evaluate the proposed CaPa by comparing it with various state-of-the-art image-to-3D approaches. Specifically, we include SDS-based [33] technique: DreamCraft3D [43]; MVDiffusion with its correponding normals for sparse neural reconstruction: Era3D [18] (an enhanced version of Wonder3D [25]), or Unique3D [54] for mesh optimization; and finally, the LRM-based method: SF3D [3]. Note that all assets were converted into a polygonal mesh using official codes to ensure consistency and fair evaluation.

High-Fidelity Mesh with Consistent Multi-View Textures. As shown in Figure 6, CaPa significantly outperforms existing state-of-the-art methods, especially in not just frontal view, but also in back and side views, where competitors typically exhibit substantial degradation. Compared to sparse view NeRF reconstruction methods like Era3D [18], CaPa demonstrates markedly superior texture quality. In relation to Unique3D [54], which generates assets through mesh optimization based on multi-view images and corresponding normals, CaPa exhibits significantly higher geometric stability (See the second asset). Finally, compared to the SDSbased method [43], CaPa consistently produces high-fidelity polygonal meshes while accelerating the generation process and improving texture quality across multiple views. These results highlight CaPa’s effectiveness and robustness in producing high-quality 3D assets, showcasing its advantages over current methodologies in the field.

To further validate the effectiveness of CaPa’s texture generation process, we compared the textured outputs of our mesh with results from SyncMVD [24] and FlashTex [9], a leading texturing method. As shown in Figure 5, CaPa successfully addresses the Janus problem. In contrast, previous methods demonstrate limited id-consistency and still exhibit the Janus problem, relying primarily on textual input alone.

#### 4.3. Quantitative Comparison

We present the quantitative results in Table 1, using 50 randomly selected test images generated via GPT [30]. We report the CLIP [34] score and FID [11], evaluated by measuring the similarity between each input image and the corresponding rendered outputs across 30 random views, thereby

[Figure 48]

Figure 5. Comparison of Texturing Method. Unlike prior works, CaPa effectively resolved the Janus problem with consistent ID.

indirectly assessing multi-view consistency. As shown in the table, CaPa achieves significantly higher scores than SOTA techniques, demonstrating its capability to produce consistent textures while preserving alignment with the input.

Method CLIP (I-I) ↑ FID ↓ Time ↓

Ours 86.34 47.56 ∼30 seconds DreamCraft3D [43] 77.61 75.66 ∼3 hours

Unique3D [54] 81.92 67.17 ∼2 minutes

Era3D [18] 66.81 89.18 ∼10 minutes SF3D [3] 70.18 84.52 ∼10 seconds

Table 1. Quantitative results. CaPa outperforms all the competitors by a significant margin in both CLIP score and FID score, with a reasonable generation time.

#### 4.4. Ablation Study and Discussion

We conduct comprehensive ablation studies to substantiate the effectiveness of each design element within CaPa, showing the importance of each component when the generation of high-quality 3D assets. Figure 7 shows the ablation results of each component.

Multi-View Guidance in 3D Latent Diffusion. As shown in Figure 7 (a), the multi-view guidance for 3D latent diffusion provides more comprehensive information on 3D geometry, which is crucial for the stable generation of 3D assets. It enables smoother, higher-quality geometry generation and minimizes quality loss during the meshing.

Janus Prevention of Spatially Decoupled Attention. The proposed Spatially Decoupled Attention effectively addresses the Janus problem by independently guiding each view during texture generation. Figure 7 (b) shows the texture generation results of CaPa, both with and without Spatially Decoupled Attention, revealing the adapter’s substantial capability to prevent multi-face generation.

Occlusion Inpainting. In this section, we demonstrate that our novel 3D-aware occlusion inpainting significantly outperforms previous occlusion inpainting solutions. In comparison to UV-ControlNet [59] and automatic view-selection methods [5], our approach achieves more comprehensive

[Figure 49]

Input

Ours ~30 seconds

Unique3D ~2 minutes

Era3D ~10 minutes

DreamCraft3D ~3 hours

SF3D ~10 seconds

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

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

- Figure 6. Qualitative Comparison of Image-to-3D Generation. We compare CaPa with state-of-the-art Image-to-3D methods. Here, all the assets are converted to polygonal mesh, using its official code. The proposed CaPa significantly outperforms both geometry stability and texture quality, especially for the back and side view’s visual fidelity and texture coherence.

and seamless occlusion inpainting within a single diffusion inference step. This results in fewer visible seams and a more cohesive texture, as shown in Figure 7 (c).

To further validate our inpainting method, we report the FID [11] and KID [2] scores between the generated textures and the occlusion-filled textures, as shown in Table 2. Lower

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

| |
|---|

| |
|---|

| |
|---|

[Figure 131]

w/o Spatially Decoupled Attention

w/o Multi-view Guidance

[Figure 132]

[Figure 133]

[Figure 134]

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

[Figure 138]

Input

Input Render

UV-ControlNet Ours

Automatic View Selection

w/ Spatially Decoupled Attention

w/ Multi-view Guidance

(a) Multi-View Guidance (b) Spatially Decoupled Attention (c) 3D-Aware Occlusion Inpainting

- Figure 7. Ablation Study. (a) demonstrates that using multi-view guidance significantly increases the geometry quality. (b) shows our Spatially Decoupled Attention effectively resolves the Janus problem, achieving high-fidelity texture coherence, (c) reveals our occlusion inpainting outperforms previous inpainting methods like UV-ControlNet, presented in Paint3D [59].

FID and KID scores indicate better alignment and visual coherence. Our method achieves an FID of 55.23 and KID of 13.46, significantly outperforming UV-ControlNet, demonstrating its ability to preserve texture fidelity and semantic accuracy. While the automatic view-selection method [5] achieves slightly lower scores, it requires 20 seconds for iterative smoothing, resulting in blurred output. In contrast, our approach maintains sharpness and semantic fidelity while achieving faster, visually coherent textures.

Method FID ↓ KID ↓ Time ↓ Ours 55.23 13.46 ∼5 sec.

Automatic View-Selection [5] 62.31 15.83 ∼20 sec. UV-ControlNet [59] 128.71 37.38 ∼5 sec.

Table 2. Quantitative Comparison of Occlusion Inpainting. Our 3D-aware inpainting restores occlusions with minimal semantic drift and improves contextual alignment efficiently.

#### 4.5. Scalability of CaPa

A major advantage of the Spatially Decoupled Attention in texture generation is its model-agnostic nature. Unlike previous MVDiffusion [40, 49], our texturing scheme can be directly applied to pre-existing large generative models and its variants, including LoRA [14] and ControlNet. Additionally, the pipeline supports flexible extensions, such as 3D inpainting via text prompts. We demonstrate CaPa’s scalability through 3D inpainting and LoRA integration examples. As shown in Figure 8, CaPa requires no additional training for 3D stylization, underscoring its scalability with a wide range of 2D generative communities.

#### 4.6. Supplementary Material

To further validate CaPa, we present additional results in Appendix B, including diverse image/text-to-3D outputs, com-

[Figure 139]

Figure 8. Scalability of CaPa. (a) Original result of CaPa. (b) 3D inpainting result using text-prompt (“orange sofa, orange pulp”). CaPa’s texture generation extends smoothly to 3D inpainting, stylizing the generated asset. (c) CaPa w/ LoRA [14] adaptation. The model-agnostic approach allows CaPa to leverage pre-trained LoRA (balloon style) without additional 3D-specific retraining.

parative experiments addressing Janus artifacts, and analysis of occlusion edge cases. Additionally, we discuss the limitations of our work in Appendix A. We kindly encourage readers to refer to these additional experiments.

### 5. Conclusion

In this study, we propose CaPa, an efficient framework for high-quality 3D asset generation which separates 3D geometry from 2D texture synthesis. Using multi-view guided 3D latent diffusion of the occupancy field minimizes the quality loss during mesh extraction. For texture synthesis, a spatially decoupled cross-attention addresses the Janus problem without additional training. This model-agnostic solution integrates with large generative models like SDXL [32] and achieves superior fidelity of the texture output. Finally, we present a novel 3D-aware occlusion inpainting algorithm, capturing 3D locality in UV space such that 2D diffusionbased inpainting effectively fills the occlusion. All these stages are fully feed-forward, making the entire generation process complete in less than 30 seconds. In summary, CaPa offers high-fidelity 3D synthesis at practical speeds, enabling immediate use in downstream applications.

### References

- [1] Raphael Bensadoun, Yanir Kleiman, Idan Azuri, Omri Harosh, Andrea Vedaldi, Natalia Neverova, and Oran Gafni. Meta 3d texturegen: Fast and consistent texture generation for 3d objects, 2024. 18
- [2] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In ICLR, 2018. 7
- [3] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun Jampani. SF3D: Stable fast 3D mesh reconstruction with uvunwrapping and illumination disentanglement. arXiv, 2024. 6, 14
- [4] Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and Kangxue Yin. Texfusion: Synthesizing 3d textures with text-guided image diffusion models. In ICCV, 2023. 2
- [5] Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. Text2tex: Text-driven texture synthesis via diffusion models. In ICCV, 2023. 2, 3, 6, 8, 15, 18
- [6] Wei Cheng, Juncheng Mu, Xianfang Zeng, Xin Chen, Anqi Pang, Chi Zhang, Zhibin Wang, Bin Fu, Gang Yu, Ziwei Liu, and Liang Pan. Mvpaint: Synchronized multi-view diffusion for painting anything 3d, 2024. 3, 4
- [7] Blender Online Community. Blender - a 3d modelling and rendering package, 2024. 11, 19
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsanit, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 5
- [9] Kangle Deng, Timothy Omernick, Alexander Weiss, Deva Ramanan, Jun-Yan Zhu, Tinghui Zhou, and Maneesh Agrawala. Flashtex: Fast relightable mesh texturing with lightcontrolnet. In ECCV, 2024. 3, 6, 11
- [10] Antoine Gu´edon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In CVPR, 2024. 16
- [11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS. Curran Associates, Inc., 2017. 6, 7
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [13] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3d. In ICLR, 2024. 1, 2
- [14] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR,

2022. 8

- [15] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In ICML, 2021. 5
- [16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 2023. 1

- [17] Jangyeong Kim, Donggoo Kang, Junyoung Choi, Jeonga Wi, Junho Gwon, Jiun Bae, Dumim Yoon, and Junghyun Han. Rocotex: A robust method for consistent texture synthesis with diffusion models, 2024. 3, 18
- [18] Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wenhan Luo, Ping Tan, Wenping Wang, Qifeng Liu, and Yike Guo. Era3d: High-resolution multiview diffusion using efficient row-wise attention. In NeurIPS, 2024. 6, 14
- [19] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. In ICLR, 2024. 2
- [20] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In CVPR,

2024. 2

- [21] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation, 2024. 18
- [22] Yehonathan Litman, Or Patashnik, Kangle Deng, Aviral Agrawal, Rushikesh Zawar, Fernando De la Torre, and Shubham Tulsiani. Materialfusion: Enhancing inverse rendering with material diffusion priors. arXiv, 2024. 11
- [23] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In ICCV, 2023. 2
- [24] Yuxin Liu, Minshan Xie, Hanyuan Liu, and Tien-Tsin Wong. Text-guided texturing by synchronized multi-view diffusion. arXiv, 2023. 2, 6
- [25] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single image to 3d using cross-domain diffusion. In CVPR, 2024. 6
- [26] William E. Lorensen and Harvey E. Cline. Marching cubes: A high resolution 3d surface construction algorithm. SIGGRAPH, 1987. 3, 16
- [27] Yuanxun LU, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quarn, Xun Cao, and Yao Yao. Direct2.5: Diverse text-to-3d generation via multi-view 2.5d diffusion. In CVPR, 2024. 2
- [28] Botsch Mario and Kobbelt Leif. A remeshing approach to multiresolution modeling. In SIGGRAPH, 2004. 16
- [29] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 2
- [30] OpenAI and et al. Gpt-4 technical report, 2024. 6
- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2
- [32] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 2, 5, 6, 8, 18
- [33] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR, 2023. 1, 2, 6

- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 5, 6, 18
- [35] Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. Texture: Text-guided texturing of 3d shapes. In SIGGRAPH, 2023. 3
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 4
- [37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 4
- [38] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. In NeurIPS, 2021. 16
- [39] Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. TOG, 2023. 16
- [40] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3d generation. In ICLR, 2024. 2, 8
- [41] Yawar Siddiqui, Tom Monnier, Filippos Kokkinos, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. Meta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials. arXiv, 2024. 2, 3
- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [43] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. In ICLR,

2024. 2, 6, 14

- [44] Jiaxiang Tang, Hang Zhou, Xiaokang Chen, Tianshu Hu, Errui Ding, Jingdong Wang, and Gang Zeng. Delicate textured mesh recovery from nerf via adaptive surface refinement. In ICCV, 2023. 16
- [45] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. LGM: large multi-view gaussian model for high-resolution 3d content creation. In ECCV, 2024. 2
- [46] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In ICLR, 2024. 2
- [47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeruIPS, 2023. 1
- [48] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR,

2023. 1, 2

- [49] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv, 2023. 2, 6, 8

- [50] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023. 2
- [51] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. In ECCV, 2024. 2
- [52] Li Weiyu, Liu Jiarui, Chen Rui, Liang Yixun, Chen Xuelin, Tan Ping, and Long Xiaoxiao. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv, 2024. 2
- [53] Jakob Wenzel, Tarini Marco, Panozzo Daniele, SorkineHornung Olga, et al. Instant field-aligned meshes. TOG,

2015. 18

- [54] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv, 2024. 2, 6, 14
- [55] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer,

2024. 2

- [56] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv, 2024. 2
- [57] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv, 2023. 6, 18
- [58] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In CVPR,

2024. 2

- [59] Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. Paint3d: Paint anything 3d with lighting-less texture diffusion models. In CVPR, 2024. 3, 6, 8, 15
- [60] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: large reconstruction model for 3d gaussian splatting. In ECCV, 2024. 1, 2
- [61] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 4, 6, 18
- [62] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A controllable large-scale generative model for creating highquality 3d assets. TOG, 2024. 2, 4
- [63] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. In NeurIPS,

2023. 2, 3, 5

Appendix

- A. Discussions and Limitations

In the main manuscript, we introduced CaPa, a carve-and-paint framework designed for fast, scalable, and high-fidelity 3D asset generation. While CaPa demonstrates significant practical advancements, some limitations remain.

Understanding of Physically Based Rendering Material. A notable limitation of the current CaPa pipeline is the lack of intrinsic support for Physically Based Rendering (PBR) materials. While CaPa is designed to generate high-quality textures, it does not inherently address PBR material understanding. Nonetheless, CaPa’s model-agnostic architecture allows for integration with external frameworks, making it compatible with recent material-aware diffusion models such as LightControlNet (as proposed in FlashTex [9]) and MaterialFusion [22]. By leveraging these material-oriented models, CaPa can be extended to facilitate PBR-aware asset generation, as illustrated in Figure 9.

Original Lighting 1 Lighting 2 Lighting 3

|[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]|
|---|

[Figure 146]

[Figure 147]

PBR-aware CaPa

- Figure 9. Result of the CaPa with PBR Understanding. We demonstrate CaPa’s capability for disentangling physically based rendering (PBR) materials. The figure shows PBR-aware generation results under various lighting conditions: ‘city,’ ‘studio,’ and ‘night,’ using Blender’s default environment settings [7]. As shown, CaPa effectively adapts to different light environments, highlighting its potential for PBR-aware asset generation.

In line with FlashTex and MaterialFusion, we utilized SDS optimization to generate PBR-aware 3D assets with material diffusion models. For additional information on SDS application for PBR separation, we refer readers to FlashTex [9] and MaterialFusion [22]. As demonstrated in Figure 9, CaPa exhibits adaptability in achieving PBR material understanding. However, we observed that SDS optimization often led to a decrease in the original texture fidelity and increased overall generation time, typically extending beyond 10 minutes per asset. Additionally, we found that the process sometimes produced a diffuse appearance that differed slightly from the originally generated texture.

Future work will focus on balancing material disentanglement with CaPa’s original objective of rapid generation, aiming to achieve efficient, high-quality, and PBR-compatible textured mesh generation. Building on CaPa’s base mesh and existing normal maps, one potential approach could involve training a diffusion model that takes normal map inputs to disentangle and separately generate diffuse, specular, and roughness maps. This would enable finer control over material properties, allowing CaPa to produce even more physically accurate textures while maintaining efficiency.

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

###### Figure 10. Additional Image-to-3D Results of CaPa. CaPa can generate diverse objects from textual, and visual input. The result demonstrates our diversity across the various categories, marking a significant advancement in practical 3D asset generation methodologies.

[Figure 180]

[Figure 181]

Ours ~30 seconds

Unique3D ~ 2 minutes

Era3D ~ 10 minutes

DreamCraft3D ~3 hours

SF3D ~10 seconds

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

[Figure 225]

[Figure 226]

[Figure 227]

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

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

- Figure 11. Additional Comparison of Image-to-3D Generation. CaPa significantly outperforms both geometry stability and texture quality, especially for the back and side view’s visual fidelity and texture coherence.

### B. Additional Experiments

#### B.1. Additional Image-to-3D Results of CaPa

In Figure 10, we present additional image-to-3D asset generation results, with close-up renderings highlighting CaPa’s superior texture and geometric stability fidelity. Each row illustrates a detailed comparison between the base geometry and the final textured model, highlighting CaPa’s ability to produce cohesive and visually rich textures. Close-up views underscore the fidelity of fine details, such as the texture of the tiger’s skin, dragon’s scales, and intricate patterns on the bag. As shown, CaPa achieves state-of-the-art quality in texture coherence and structural consistency.

We present additional comparative experiments of state-of-the-art Image-to-3D asset generation methods in Figure 11. Consistent with Section 4 in the main manuscript, we validate CaPa against Unique3D [54], Era3D [18], DreamCraft3D [43], and SF3D [3]. Each row represents a different 3D asset (e.g., tiger, dragon, toy robot, and character model) with full-color renderings and corresponding grayscale meshes for each method. As shown in Figure 11, CaPa consistently delivers highquality textures and detailed geometry within a significantly reduced runtime (around 30 seconds) compared to other methods. This comparison highlights the efficiency and fidelity balance achieved by our approach.

#### B.2. Impact of Spatially Decoupled Cross Attention on Janus Artifacts

##### B.2.1 Qualitative Analysis

In this section, we present a qualitative comparison to demonstrate the effectiveness of the proposed spatially decoupled cross-attention mechanism in mitigating the Janus problem, a common issue in 3D asset generation when handling multi-view inputs. As seen in Figure 12, spatially decoupled cross attention markedly improves multi-view consistency and eliminates Janus artifacts, demonstrating the necessity of spatial decoupling when incorporating multi-view guidance. This mechanism thus enhances CaPa’s ability to generate high-quality 3D assets with accurate textures and structure across all views, supporting its robustness for diverse applications.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- (a)
- (b)
- (c)

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

- Figure 12. Impact of Spatially Decoupled Cross Attention on Janus Artifacts. In this additional figure, We demonstrate the capability of Janus prevention in the proposed spatially decoupled cross-attention mechanism. Each row depicts, (a) with spatially decoupled cross attention, (b) without spatially decoupled cross attention, and (c) a mesh rendering of the current view, respectively.

##### B.2.2 Quantitative Analysis

Additionally, we present the quantitative analysis of the Janus prevention capability of the CaPa. Before the discussion, we acknowledge that directly evaluating multi-view inconsistency is challenging, as traditional metrics often do not capture the specific artifacts introduced by the Janus problem. To circumvent this problem, we assessed the impact of the Janus effect by measuring CLIP similarity between rendered normal maps and texture. We hypothesize that the presence of Janus artifacts leads to a misalignment between the geometry and the texture’s appearance, resulting in a decrease in the CLIP similarity score. Using this assumption, CLIP similarity serves as a proxy metric to evaluate the Janus artifacts assessment.

As shown in Table 3, with spatially decoupled attention, the average similarity score was 85.37, compared to 81.28 without it. Although the difference is modest, likely due to initial geometry guidance aligning texture and mesh. This result supports that spatially decoupled attention resolves the Janus effect and improves multi-view consistency in generated 3D assets.

Method CLIP (N-I) ↑ w/ Spatially Decoupled Attention 85.37

w/o Spatially Decoupled Attention 81.28

Table 3. Quantitative analysis of Janus Artifacts, measuring a CLIP score between rendered normals and textures across random views.

#### B.3. Analysis of the Occlusion

In this section, we provide an in-depth analysis of occlusion inpainting to address limitations in our main method and clarify its comparative strengths and weaknesses. While our paper demonstrates the effectiveness of our 3D-aware occlusion inpainting approach across diverse scenarios, there are complex cases that challenge even the most advanced techniques. This supplementary section highlights these edge cases and evaluates the relative performance of three methods: our 3D-aware inpainting, automatic view selection [5], and UV ControlNet [59]. By examining these approaches, we aim to shed light on the specific strengths and limitations of our method in comparison to alternatives, ultimately guiding future improvements. To explore occlusion handling, we compare (a) our proposed 3D-aware occlusion inpainting, (b) automatic view selection, and (c) UV ControlNet, with an emphasis on texture fidelity, seam visibility, and processing efficiency.

As seen in Figure 13, both our method and automatic view selection exhibit high fidelity and maintain strong texture coherence, effectively blending textures with the surrounding context in most instances. This results in fewer visible seams, a

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

- (a)
- (b)
- (c)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 13. Qualitative results for different occlusion inpainting methods. (a) shows results from our 3D-aware occlusion inpainting method, (b) uses automatic view selection, and (c) employs UV ControlNet.

crucial factor in generating visually coherent 3D assets. However, when faced with certain edge cases, such as complex or highly occluded areas (e.g., the panda example), both methods (a) and (b) encounter challenges in filling occluded regions.

On the other hand, as demonstrated in Figure 13 (c), UV ControlNet shows its capability to fill all occlusions comprehensively in UV space, albeit at a reduced fidelity level. This trade-off reflects the inherent limitations of 3D-aware methods and automatic view selection when attempting to achieve high coverage in extreme cases. UV ControlNet achieves full occlusion fill by directly addressing gaps in the UV space, albeit with a reduction in texture detail, which may not be ideal for high-fidelity applications. In contrast, our 3D-aware inpainting offers superior texture fidelity and coherence across most scenarios. This trade-off between coverage and detail emphasizes the potential for future enhancements that balance high-quality inpainting with comprehensive occlusion handling.

#### B.4. Text-to-3D Results of CaPa

Lastly, Figure 14 demonstrates CaPa’s capabilities in generating high-quality 3D assets from textual prompts. Results across different object categories, including characters, common objects, and cultural artifacts, affirm CaPa’s generalization ability to maintain geometric stability and texture consistency in various shapes and surface types.

### C. Additional Details of CaPa

#### C.1. Mesh Extraction and Remeshing Algorithm

In 3D asset generation, obtaining a high-quality mesh is essential for downstream applications, as it allows the asset to be integrated effectively into various environments, such as rendering engines and simulation pipelines. While recent advancements in Neural Rendering, such as NeRF and Gaussian Splatting, these representations suffer from accurate surface reconstruction [10, 44]. As a result, extracting meshes from Gaussian Splatting or NeRF degrades mesh quality after a typical mesh conversion algorithm like Marching Cube [26].

In response, DMTet [38] and FlexiCubes [39] have addressed this problem by optimizing meshes through external learnable parameters for meshing. While these approaches significantly improve output mesh quality, the computational cost is intensive and still generates incomplete or suboptimal geometries.

##### C.1.1 Initial Mesh Extraction

Unlike previous works with NeRF or Gaussian Splatting representation, our 3D diffusion model operates on occupancy fields that inherently align better with mesh extraction methods. This compatibility enables us to achieve cleaner, higher-fidelity geometry, avoiding the substantial quality loss common in meshing processes for NeRF and GS-based assets. We first extract the initial mesh of the generated occupancy field by leveraging the Marching Cube algorithm [26].

##### C.1.2 Automated Remeshing Post-Processing

After the mesh conversion, we design an automated post-processing pipeline to obtain a cleaner topology. This post-processing enhances the geometric quality and accelerates the extraction process. Our pipeline begins by ensuring manifold properties of the generated mesh through a series of cleaning and remeshing steps. The mesh cleaning process enhances quality by removing unreferenced or isolated elements and repairing non-manifold components. Floater removal processes unreferenced vertices by merging duplicates and removing zero-area faces as well as small, isolated components. For non-manifold repair, faces connected to non-manifold edges are iteratively removed, starting with the smallest faces, to maintain 2-manifold properties. Our remeshing step offers flexibility by supporting both triangular and quadrilateral mesh options.

Triangular Remeshing. For triangular remeshing, following Botsch et al. [28], we iteratively refine the base mesh through isotropic remeshing to form equilateral triangles. This process involves alternating edge lengths and vertex valence equalization to generate a uniformly remeshed surface. Specifically, edge lengths are adjusted to a predefined target constant through edge splitting and collapsing, achieving uniform face sizes across the mesh. Then, vertex valence deviation is minimized toward 6 (or 4 at boundaries) by applying edge flips, thereby enhancing structural uniformity. Following these steps, an area-based tangential smoothing process is applied to preserve the surface shape. Each vertex is adjusted toward a centroid, calculated as

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

“A human head skull, realistic, 4K”

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

“A zombie werewolf, wearing a suit”

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

“A koala samurai”

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

“A cute cat with white, golden fur, 3D assets”

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

“Wooden chair with three legs”

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

“Master Yoda from StarWars, with wand”

- Figure 14. Text-to-3D Results of CaPa. CaPa can generate diverse objects from textual, and visual input. The result underscores CaPa’s strengths in generating high-resolution textures that align with well-defined geometries.

a weighted average of neighboring vertices based on area. The gravity-weighted centroid gi for each vertex pi is given by:

1 pj∈N(pi) A(pj)

A(pj)pj, (6)

gi :=

pj∈N(pi)

where A(pj) denotes the area associated with vertex pj, and N(pi) is the set of neighboring vertices of pi. This weighted average ensures that vertices with larger areas have a stronger influence, thereby smoothing the mesh more uniformly.

To ensure tangential smoothing on the surface, the update vector is projected back onto the tangent plane of pi as follows: pi ← pi + λ I − ninTi (gi − pi), (7)

where ni is the normal vector of pi, and λ is a damping factor used to avoid oscillations. This projection moves each vertex in the tangent direction toward its centroid, smoothing the mesh while preserving its overall shape.

Quadrilateral Remeshing. Following Jakob et al. [53], quadrilateral remeshing is designed to produce structured meshes composed of quadrilateral faces, offering a uniform and easily manageable layout ideal for various applications requiring high-quality surface control. This process is accomplished through two primary stages: 1) Orientation Field Optimization and 2) Position Field Optimization.

- 1) Orientation Field Optimization: This initial step establishes directional alignment across the mesh surface by defining an

orientation field. This field prescribes specific directional vectors at each vertex, effectively creating a structured “grid” on the surface. By aligning each edge along these pre-defined directions, the mesh gains a cohesive pattern that ensures smooth transitions across faces, an essential property for generating a clean quadrilateral structure.

- 2) Position Field Optimization: Once edge orientations are set, the positions of vertices are optimized to achieve consistent

spacing between edges, regulated by a global scale factor. This scale factor, provided by the user, enforces uniform edge lengths across the entire mesh, ensuring that each quadrilateral face remains similarly sized. Vertices are locally adjusted on a 2D tangent plane to fit within this structured layout, resulting in a smooth, evenly distributed quadrilateral grid.

By combining these two stages, the quadrilateral remeshing process yields a quadrilateral mesh that maintains both geometric coherence and structural regularity, making it suitable for applications demanding high fidelity and uniform surface representation. This approach allows for a stable, grid-like arrangement that balances surface alignment with positional consistency across the mesh. For further details on quadrilateral remeshing, we refer readers to the Instant Field-Alignment [53].

Experimental Results. Figure 15 illustrates the results before and after remeshing. In Figure 15 (b), the triangle remeshing outcome yields a highly regular mesh compared to Figure 15 (a), with uniformly distributed Voronoi regions around each vertex. Similarly, as shown in Figure 15 (c), each quadrilateral face is uniform, and the overall mesh contains fewer irregular vertices. These results demonstrate how post-processing forms a smooth and stable mesh structure.

In conclusion, our automated post-processing pipeline improves the uniformity and quality of generated meshes while significantly reducing processing time. Our approach yields visually superior, computationally efficient meshes through manifold enforcement, flexible remeshing techniques, and advanced texture unwrapping, optimizing the overall 3D asset generation workflow.

#### C.2. Texture Generation

As mentioned in the main manuscript, we first render four orthogonal views of the generated mesh for texture synthesis. Then, we generate corresponding images for rendered inputs, by leveraging our novel spatially decoupled attention mechanism. In this stage, to achieve a state-of-the-art quality of texture generation, we have implemented our texture generative scheme on the SDXL [32]. We have utilized pre-trained IP Adapter [57] layer and CLIP [34] for the image feature extractor, DepthControlNet [61] for a geometric guide to align with the geometry. Notably, to generate texture efficiently, we have adopted the SDXL lightning model [21] baseline checkpoint of our texture generation and occlusion inpainting. With this diffusion pipeline, we have operated only 8 steps per inference with DPM-solver.

However, at this stage, directly applying textures from four views often results in visible seams where the views meet. In response, we leveraged a smoothing mechanism specifically for the side views, similar to the previous works [1, 5, 17]. When generating the side textures, we mask out the regions already visible in the front and back views, using a confidence

[Figure 313]

- Figure 15. Results of Our Remeshing Algorithm. We employ a carefully designed remeshing scheme after geometry generation for better practical usage for broader applications. (a) shows the original polygonal mesh, (b) shows remeshed output of quadrilateral faces, and (c) shows remeshed output of triangular faces.

map that softly transitions the mask. Here, the confidence map is calculated by measuring the cosine similarity between the viewing direction and the normal vector of the mesh faces. We apply differential denoising during texture synthesis, which only partially influences the masked regions. The confidence map controls the strength of the effect. This allows the side textures to blend smoothly with the front and back views.

Remarks for Occlusion Inpainting. It is important to note that the occlusion-specific UV map is only used for inpainting purposes. As this map captures a limited set of projections, it may introduce artifacts such as visible seams if used for final texturing. To generate the actual texture map, we utilize Blender’s smart UV projection [7], to ensure a consistent and high-quality texture across the entire mesh surface.

