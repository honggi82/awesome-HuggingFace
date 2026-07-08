[Figure 1]

## WonderZoom: Multi-Scale 3D World Generation

Jin Cao* Hong-Xing Yu* Jiajun Wu

Stanford University https://wonderzoom.github.io/

# arXiv:2512.09164v1[cs.CV]9Dec2025

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Prompt: “A yellow bird is standing on the windowsill”

Input Image

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

User interaction

[Figure 22]

Prompt: “A lizard is clinging to the wall.”

[Figure 23]

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

Generated 3D scene

Prompt: “The SUV has a shiny chrome bumper.” : camera movement : camera zoom-in

Figure 1. Multi-scale 3D world generation from a single image. WonderZoom enables interactive exploration across spatial scales. Users can zoom into any region and specify prompts to generate new fine-scale content while maintaining cross-scale consistency. Here we show three zoom-in sequences. We attach an interactive viewer in the supplmentary material.

#### Abstract

to “zoom into” a 3D region and auto-regressively synthesize previously non-existent fine details from landscapes to microscopic features. Experiments demonstrate that WonderZoom significantly outperforms state-of-the-art video and 3D models in both quality and alignment, enabling multiscale 3D world creation from a single image. We show video results and an interactive viewer of generated multi-scale 3D worlds in https://wonderzoom.github.io/.

We present WonderZoom, a novel approach to generating 3D scenes with contents across multiple spatial scales from a single image. Existing 3D world generation models remain limited to single-scale synthesis and cannot produce coherent scene contents at varying granularities. The fundamental challenge is the lack of a scale-aware 3D representation capable of generating and rendering content with largely different spatial sizes. WonderZoom addresses this through two key innovations: (1) scale-adaptive Gaussian surfels for generating and real-time rendering of multi-scale 3D scenes, and (2) a progressive detail synthesizer that iteratively generates finer-scale 3D contents. Our approach enables users

#### 1. Introduction

3D world generation has emerged as a transformative capability in computer vision, enabling the synthesis of immersive environments from minimal input [7, 9, 13, 24, 50, 51]. However, despite the inherently multi-scale nature of realworld scenes, existing approaches remain fundamentally

*Equal contribution.

constrained to single-scale generation. They can produce landscape-level environments and room-scale scenes, but fail to synthesize coherent content across multiple spatial scales, e.g., a tiny ladybug lying on a sunflower in a vast field. This limitation prevents existing approaches from generating rich, detailed worlds that span from panoramic vistas down to microscopic surface details, restricting their applicability for interactive exploration and content creation.

The fundamental challenge underlying this limitation is the absence of a scale-adaptive 3D representation suitable for scene generation. Traditional Level-of-Detail (LoD) representations [26] were designed for efficiently rendering pre-existing graphics content, where all geometric details are known in advance. Recent hierarchical representations like Hierarchical 3D Gaussian Splatting [17] and Mip-NeRF [1] extend these principles to neural reconstruction, efficiently encoding scenes at multiple scales. But critically, they still assume access to complete multi-scale image data upfront for one-pass optimization. Both paradigms, rendering and reconstruction, fundamentally conflict with generation, where images do not exist a priori and must be synthesized progressively. In generation, we must create coarse-scale content first, then iteratively synthesize finer details conditioned on both the coarser structure and user-specified prompt and regions of interest. This requires representations that can grow dynamically as new fine-scale content is generated, not static hierarchies optimized with complete supervision. Current generation methods [50, 51] sidestep this challenge entirely by restricting themselves to single scales, while naive application of existing hierarchical representations would demand generating all scales simultaneously, which is a computationally intractable approach that violates the inherent coarse-to-fine nature of multi-scale synthesis.

To address this challenge, we propose WonderZoom, a novel framework for multi-scale 3D world generation from a single image. Our approach introduces two key technical innovations: (1) scale-adaptive Gaussian surfels, a dynamically updatable hierarchical representation that, unlike existing multi-scale methods, supports incremental refinement as new content is generated. It allows adding arbitrary levels of detail without re-optimization, and (2) a progressive detail synthesizer that iteratively generates fine-grained 3D structures conditioned on both coarser scales and userspecified regions and viewpoints. These components work synergistically: the scale-adaptive representation provides a persistent 3D canvas that grows in detail over time, while the synthesizer produces coherent multi-scale content through a controlled coarse-to-fine generation process. By enabling dynamic updates to the 3D representation as new scales are synthesized, WonderZoom fundamentally shifts from the reconstruction paradigm to multi-scale generation, overcoming the computational and architectural barriers that constrain existing methods to single scales.

Our approach enables users to interactively “zoom into” any region of the generated 3D scene, triggering autoregressive synthesis of previously non-existent details, e.g., from an entire landscape down to microscopic surface features. Unlike traditional multi-resolution rendering that simply reveals pre-existing details, WonderZoom generates new content on-demand, creating coherent structures that were never part of the original input or coarse generation. This capability allows infinite exploration of generated worlds at arbitrary levels of detail. In summary, our contributions are threefold:

- • We propose WonderZoom, the first approach to enable multi-scale 3D world generation from a single image, supporting seamless transitions from macro to micro scales.
- • We introduce scale-adaptive Gaussian surfels, a dynamically updatable representation that grows incrementally with newly generated finer-scale content, while maintaining real-time rendering performance.
- • We demonstrate and evaluate multi-scale 3D generation across diverse scenarios including natural environments, villages, and urban scenes, achieving consistent quality across scale transitions while significantly outperforming state-of-the-art video and 3D generation models in both perceptual quality and prompt alignment.

- 2. Related Work
- 3D World Generation. Early 3D scene generation methods focused on novel view synthesis from a single image, constructing renderable representations like layered depth images [33, 38], radiance fields [34, 36, 49], multi-plane images [37, 56], and point features [28, 43], though these only supported small viewpoint changes from the input. Later works explored generating more significant viewpoint changes and multiple connected scenes. Infinite Nature [24] and its follow-ups [4, 5, 21] pioneered perpetual view generation for natural scenes with a neural renderer. Recent methods [20, 22, 35, 48, 55] expanded this capability to explicit 3D, e.g., SceneScape [9] and Text2Room [13] generate meshes from text prompts, WonderJourney [50] and WonderWorld [51] creates diverse connected 3D scenes using LLMs and point-based representations, LucidDreamer [7] and CAT3D [10] focus on room-scale environments with 3D Gaussian splatting. Another line of work specializes in city-scale generation [8, 23, 45, 46], producing largescale 3D Gaussian splatting representations of urban environments. However, these methods operate at a single spatial scale aligned with their input—–generating either landscapes, rooms, or cities, but not content across scales. In contrast, we enable multi-scale 3D generation where users can progressively zoom into any region to synthesize entirely new content at finer scales, creating details that were never visible or implied in the original input image.

Multi-scale 3D Representations. Classical computer graphics has long addressed multi-scale rendering through Levelof-Detail (LoD) techniques [26], which adaptively select geometric complexity based on viewing distance, and mipmapping, which precomputes texture pyramids for efficient antialiased rendering. These traditional methods assume all geometric and texture details exist upfront, making them suitable only for rendering pre-authored content, not for progressive generation. Recent neural 3D reconstruction methods have incorporated similar multi-scale principles, e.g., MipNeRF [1] introduces integrated positional encoding to handle scale ambiguity, with extensions like Mip-NeRF 360 [2] and Zip-NeRF [3] improving unbounded scene representation. In the Gaussian splatting [16] domain, Mip-Splatting [52] addresses aliasing through 3D smoothing filters, while Hierarchical 3D Gaussian Splatting [17] builds explicit LoD hierarchies for efficient rendering. Octree-GS [30] and ScaffoldGS [25] use spatial hierarchies to manage primitives across scales. However, both traditional LoD and these neural hierarchical representations share a critical limitation: they are fundamentally designed for scenarios where content at all scales is known: either pre-authored (traditional LoD) or reconstructed from complete multi-scale image supervision (neural methods). This paradigm is incompatible with generation, where fine-scale content must be synthesized progressively without pre-existing data. Our approach addresses this gap by organically integrating a scale-adaptive representation that can be dynamically refined with a progressive generation pipeline.

Controllable Content Synthesis. Controllable video generation methods have made significant strides in conditional synthesis, accepting camera trajectories [11, 32], depth maps [53], or semantic masks as inputs to guide generation. However, these approaches cannot perform multi-scale generation due to the absence of training data that captures coherent content across vastly different spatial scales. Superresolution techniques have evolved from 2D image enhancement to 3D domains, including mesh refinement, point cloud upsampling [54], and neural field super-resolution [39]. Yet these methods focus on sharpening and refining pre-existing content rather than generating entirely new cross-scale structures. A recent work, Generative Powers of Ten [42], demonstrates infinite zoom generation by jointly sampling multiple scales through coordinated diffusion processes, though this remains limited to 2D images. Hierarchical generation approaches like Progressive GANs [15] and cascaded diffusion models [12] synthesize content at increasing resolutions through staged refinement. Our approach uniquely extends these capabilities to 3D, combining controllable generation with true multi-scale synthesis—enabling users to interactively zoom into any region and generate coherent new content across vastly different spatial scales, from environmental to microscopic levels that never existed in the

original input.

#### 3. Approach

Formulation. We target multi-scale 3D world generation from a single image. Given an input image I0 and a sequence of user-specified prompts {U1,...,Un} with corresponding camera viewpoints {C0,...,Cn},Ci ∈ R4×4 that progressively zoom into regions of interest, our goal is to generate a sequence of 3D scenes {E0,E1,...,En} at increasing spatial granularities. Here, E0 represents the initial 3D scene reconstructed from the input image I0, while each subsequent scene Ei (i > 0) represents finer-scale content that is spatially contained within the previous scene Ei−1, creating a nested hierarchy where zooming reveals newly synthesized details rather than pre-existing geometry. This process can be repeated multiple times from the same initial image I0 with different camera sequences and prompt sequences. Figure 1 illustrates this capability, where we demonstrate three distinct zoom sequences from a single input.

Challenges. The major technical bottleneck preventing multi-scale generation is the lack of scale-adaptive 3D representations suitable for generation. Existing multi-scale representations, from classical Level-of-Detail techniques to recent hierarchical methods like Hierarchical 3D Gaussian Splatting [17], are designed for either rendering pre-existing graphics content or reconstruction with complete multi-scale image supervision available upfront. However, generation imposes fundamentally different requirements: we need to create coarse-scale content Ei−1 first, then iteratively synthesize finer details Ei conditioned on both the coarser structure Ei−1 and user-specified prompts Ui and regions of interest defined by Ci. This demands representations that can grow dynamically as new scales are generated while maintaining real-time rendering capability, a capability absent in existing methods that assume static, pre-optimized hierarchies. Another challenge lies in synthesizing semantically meaningful content that follows user prompts Ui while maintaining geometric and appearance consistency with previous scales Ei−1. Unlike simple super-resolution that merely enhances existing details, we may need to generate entirely new structures (e.g., a bird or a lizard as in Figure 1) that were not implied in the coarser representation.

Overview. We propose WonderZoom to enable multi-scale 3D world generation through two key technical innovations. To address the first challenge, we introduce scale-adaptive Gaussian surfels (Sec. 3.1) that allow dynamic updates without re-optimization. This representation enables adding arbitrarily many scales Ei while maintaining real-time rendering capability at any scale, as new finer-scale surfels can be seamlessly integrated into the existing hierarchy without modifying coarser levels. To address the second challenge, we design a progressive detail synthesizer (Sec. 3.2) that gen-

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Scaleconsistent depth registration

New-scale image synthesis

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Optimize Input image Initialized 3D scene

User specifies Prompt & Camera

New scale image

Coarse-scale depth

New scale depth

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

Auxiliary view synthesis

[Figure 62]

[Figure 63]

[Figure 64]

Scale-aware opacity modulation

[Figure 65]

[Figure 66]

[Figure 67]

Dynamic update

Sample cameras

Dynamic update

New scale image & depth Auxiliary views

New scale 3D scene New scale 3D scene

Figure 2. WonderZoom overview. From an input image, we first reconstruct an initialized 3D scene. Users specify prompts and camera viewpoints to generate finer-scale content. Our progressive detail synthesizer creates new-scale images, registers depth to maintain geometric consistency, and synthesizes auxiliary views for complete 3D scene creation. Our scale-adaptive Gaussian surfels enable dynamic updates without re-optimization, seamlessly integrating new content while preserving real-time rendering.

erates new fine-grained 3D structures Ei from user prompts Ui while ensuring consistency with the previous scale Ei−1. The synthesizer leverages the coarse geometry as spatial conditioning to guide the generation of coherent fine-scale content, going beyond simple super-resolution to create semantically meaningful details. We show an illustration of our framework in Figure 2. We summarize the complete multi-scale generation control loop in Algorithm 1 in supplementary material.

isting surfels, and (2) supporting real-time rendering at any observation scale.

Dynamic updating. The core idea of our dynamic representation is that we incrementally add new surfels to represent each new scale without modifying existing ones. When we create the initial scene E0 from the input image I0, we generate N0 surfels to represent the coarse-scale geometry and appearance. When we subsequently generate the finer-scale scene E1 from a zoomed-in view I1 at camera C1, we add N1 new surfels to the representation, resulting in a total of N = N0 + N1 surfels. This process continues: when generating Ei, we add Ni new surfels, bringing the total to N = ik=0 Nk. Crucially, the surfels from previous scales remain unchanged: we only append new surfels that capture the finer details visible at the current scale. This additive mechanism naturally enables dynamic updates: each new scale simply extends the existing representation rather than requiring global re-optimization, allowing the multi-scale world to grow organically as users explore different regions at increasing levels of detail.

##### 3.1. Scale-adaptive Gaussian Surfels

Definition. We introduce scale-adaptive Gaussian surfels to represent our multi-scale scenes {E0,...,En}. Formally, we model the scenes as a radiance field represented by a set of Gaussian surfels {gj}Nj=1. Each surfel is parameterized as g = {p,q,s,o,c,snative}, where p denotes the 3D spatial position, q denotes the orientation quaternion, s = [sx,sy] denotes the scales of the x-axis and y-axis, o denotes the opacity, and c denotes the view-independent RGB color. The Gaussian kernel follows the same formulation as in prior work [51], with covariance matrix Σ = Qdiag(s2x,s2y,ϵ2)QT where Q is the rotation matrix obtained from q and ϵ is a small thickness parameter. The key addition is snative, the native scale at which the surfel was created, which enables scale-aware rendering as we describe later. In WonderZoom, we sequentially generate each scene, starting from E0 and progressively adding finer-scale content through En. This demands our representation to satisfy two requirements: (1) capable of dynamic updates given new scale images Ii at viewpoints Ci without re-optimizing ex-

Scale-aware opacity modulation for real-time rendering of multi-scale scenes. Since we represent multi-scale content with surfels across different scales, the same surface may be covered by multiple layers of surfels from E0 through Ei. Directly rendering all surfels would cause aliasing and reduce rendering speed. To address this, we introduce scaleaware opacity modulation based on each surfel’s native scale:

dnative fxnativefynative

snative =

(1)

where dnative is the surfel’s depth relative to Ci (the cam-

era view where the surfel was created) and fxnative,fynative are the focal lengths of Ci. During rendering at camera Crender, we compute the current rendering scale srender =

drender/ fxrenderfyrender for each surfel. For surfels at intermediate scales (0 < i < n), we define parent and

child scale bounds: sparent = dparent/ fxparentfyparent where dparent and fparent are defined relative to Ci−1, and schild = dchild/ fxchildfychild where dchild and fchild are defined relative to Ci+1. The rendered opacity is then modulated as:

o˜ = o · α, (2) where



1 if no parent and srender ≥ snative log(sparent)−log(srender) log(sparent)−log(snative) if sparent ≥ srender ≥ snative log(srender)−log(schild) log(snative)−log(schild) if snative ≥ srender ≥ schild



α =

1 if no child and srender ≤ snative 0 otherwise.



(3) This design ensures surfels are most visible at their native scale (α = 1 when srender = snative) and fade smoothly when viewed at different scales. Notably, surfels at the coarsest scale (i = 0) remain fully visible when zoomed out, while surfels at the finest scale (i = n) remain fully visible when zoomed in, ensuring complete scene coverage at all observation scales.

|Proposition 1 (Seamless Scale Transition). Our scaleaware opacity modulation ensures smooth visual transitions between adjacent scales without discontinuities. Specifically, consider two surfels gj and gk located at the same 3D position p but created at adjacent scales Ei−1 and Ei respectively. When the rendering scale srender transitions between their native scales, i.e., when srender ∈ [snativek ,snativej ], the sum of their modulated opacity weights satisfies:<br><br>αk(srender) + αj(srender) = 1. (4)<br><br>This property holds because the linear interpolation in log space for gk decreasing from its native scale matches exactly with the complementary interpolation for gj increasing toward its child scale bound. As a result, the total contribution from overlapping surfels at different scales remains constant during zoom operations, eliminating popping artifacts and ensuring visually continuous scale transitions. This partition of unity is fundamental to maintaining coherent appearance as users navigate across the multi-scale hierarchy.|
|---|

Optimization. Our scale-aware opacity modulation preserves the differentiability of the rendering pipeline, thereby we use gradient-based optimization for surfel parameters. When creating surfels for a new scale Ei from image Ii, we generate pixel-aligned surfels following the same approach as prior work [51], where each surfel corresponds to a pixel in Ii. We also follow the same geometry-based initialization: each surfel’s position p is initialized using the estimated depth map via back-projection, orientation q from the estimated surface normal, and scales s according to the Nyquist sampling theorem to ensure appropriate coverage without excessive overlap. The color c is initialized from the corresponding pixel RGB values, the native scale snative is computed based on the creation viewpoint Ci, and opacity is initialized to o = 0.1 for stable optimization. We then optimize the opacity, orientation, and scales (while keeping positions, colors, and native scales fixed) using Adam [19] with a photometric loss L = 0.8L1+0.2LD-SSIM [16] against the input image Ii. This lightweight optimization refines the surfel geometry while preserving the multi-scale structure.

##### 3.2. Progressive Detail Synthesizer

Goal. Given the coarse-scale scene Ei−1, a target camera viewpoint Ci, and a user prompt Ui, our goal is to generate an image Ii and its corresponding depth map Di that are geometrically consistent with Ei−1 while incorporating the content specified in Ui. Note that Ui may describe entirely new structures not visible or implied in Ei−1 (e.g., a ladybug on a sunflower), requiring our approach to go beyond simple super-resolution to synthesize semantically meaningful content. Since we aim to generate a complete 3D scene Ei that can be rendered from varying viewpoints, we additionally generate a set of auxiliary images {Iki }Kk=1 from neighboring viewpoints to augment Ii, enabling optimization of a more complete 3D structure that extends beyond the single input view. This subsection describes our three-stage pipeline: new scale image generation from the coarse scene and prompt, scale-consistent depth registration to maintain geometric coherence, and auxiliary view synthesis for complete 3D reconstruction.

New scale image synthesis. To generate the finer-scale image Ii, we begin by rendering a coarse observation from the previous scale: Oi = render(Ei−1,Ci), where Ci has a larger focal length than Ci−1 to zoom into the region of interest. Since Oi is obtained through direct zoom-in rendering and thus lacks fine details, we apply extreme superresolution to synthesize high-frequency content. However, extreme zoom ratios require additional semantic guidance beyond what is visible in Oi. We therefore extract semantic context from the previous scale using a vision-language model (VLM): S = VLM(Oi−1), where Oi−1 is the rendered image at the previous scale. The super-resolved image is then generated as I′i = SR(Oi,S), conditioned on both

the coarse observation and semantic context. To incorporate user-specified content Ui that may include entirely new structures absent in Ei−1, we apply a controllable image editing model: Ii = Edit(I′i,Ui). This two-stage approachsuper-resolution followed by semantic editing—enables both faithful detail enhancement of existing structures and insertion of novel content specified by the user.

Scale-consistent depth registration. To estimate a depth map Di that maintains geometric consistency with Ei−1, we employ a multi-stage registration approach. First, we render a target depth map from the existing geometry: Dtargeti = render depth(Ei−1,Ci), which provides sparse depth values for regions visible in the previous scale. We then fine-tune a monocular depth estimator Dθ to align its predictions with this target depth by minimizing:

Ldepth = u,v ∥Dtargeti (u,v) − Dθ(Ii)(u,v)∥ · m(u,v) u,v m(u,v)

, (5)

where m(u,v) = 1 if Dtargeti (u,v) is defined, and m(u,v) = 0 for undefined regions due to zoom-in effect. This fine-

tuning ensures that the estimated depth Di = Dθ(Ii) aligns with the coarse geometry while still predicting reasonable depths for newly visible regions. To further refine the registration, we apply segment-wise depth alignment using SAMgenerated masks to correct for local depth inconsistencies as in prior work [50, 51], and for any newly added structures from the editing stage (e.g., the ladybug in Figure 2), we use Grounded SAM [31] to isolate these regions and estimate their depth while maintaining consistency with surrounding geometry.

Auxiliary view synthesis. While Ii provides detailed content at the target viewpoint Ci, a single image is insufficient to reconstruct a complete 3D scene that can be rendered from arbitrary viewpoints. To address this, we synthesize auxiliary views {Iki }Kk=1 from neighboring camera positions using a camera-controlled video diffusion model. We first render conditioning frames from the current partial scene: {Oki } = {render(Eipartial,Cki )}Kk=1, where Eipartial is the initial scene constructed from Ii alone, and {Cki } are camera viewpoints sampled around Ci. Along with these frames, we generate corresponding masks {Mki } indicating regions requiring synthesis (e.g., occluded areas not visible in Ii). The video diffusion model then generates temporally consistent frames: {Iki } = VideoDiff({Oki },{Mki }), conditioned on the partial observations and masks. We then leverage a video depth model to estimated depth {Dki } for these generated frames, and the resulting image-depth pairs are used to optimize a more complete 3D scene following the same optimization procedural as described in Sec. 3.1. This auxiliary view synthesis enables us to construct complete 3D scenes Ei that extend beyond the single input view while maintaining coherence with the generated content. In practice, we also

apply it to help generate the coarsest-scale scene E0.

#### 4. Experiments

In our experiments, we evaluate WonderZoom on multi-scale world generation and compare it to existing methods. We also perform ablation studies to analyze WonderZoom.

Baselines. We are not aware of any prior method that allows multi-scale 3D scene generation. Therefore, we consider state-of-the-art methods in general-purpose 3D scene generation including WonderWorld [51] and HunyuanWorld [35]. Besides 3D-based approaches, we further include state-ofthe-art camera-controlled video generation models, including Gen3C [32] and Voyager [14]. We use these baselines’ official codes for comparison.

Test examples. For comparison with the baselines, we collect publicly available real images and generate synthetic images as our testing examples, and we also use examples from Wang et al. [42]. We evaluate on 32 generated scenes from 8 test input images, spanning diverse scene types such as a field, a city, a forest, and underwater. Among them, a sunflower image and a coral image are synthetic, and all others are real images. For each test example, we generate 4 new-scale scenes in additional to the input scale, i.e., we generate {E0,··· ,E4}. For a fair comparison, we use fixed camera paths and the same text prompts for all methods.

Metrics. For quantitative comparison, we adopt the following evaluation metrics: (1) We collect 200 human study two-alternative force choice (2AFC) results on the rendering of new scale scenes, i.e., {E1,··· ,E4}. (3) To evaluate the alignment of generated scenes w.r.t. text prompts, we render 9 sudoku-like novel views around each generated scene Ei,1 ≤ i ≤ 4, and compute the CLIP [29] scores of the prompt versus the rendered images. (4) We evaluate rendered novel view image quality with CLIP-IQA+ [40], Q-align IQA [44], and NIQE [27]. (4) We also measure the aesthetics of novel views by the Q-align IAA [44]. We leave more details in the supplementary material.

Implementation details. In our implementation, we use Chain-of-Zoom [18] as our super-resolution model. We use Gen3C [32] as the camera-controlled video diffusion model in auxiliary view synthesis. We estimate image depth by MoGe [41] and video depth by GeometryCrafter [47]. We leave more details in the supplementary material. We will release the full code and software for reproducibility.

##### 4.1. Comparison

Qualitative showcase. We show qualitative comparison in Figure 3 and in the appendix. Meanwhile we show more examples generated by our method in Figures 4. We also strongly encourage the reader to see video results and to interactively view generated worlds in the HTML in our supplementary materials. From the qualitative comparison,

[Figure 68]

[Figure 69]

[Figure 70]

: Camera movement : Camera zoom-in

WonderWorldOurs

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

Gen3CVoyagerHunyuan

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

Prompt: A yellow bird is standing on the windowsill

Input image

###### Figure 3. Comparison of WonderZoom with baselines on multi-scale 3D world generation.

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

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Input image

Generated multi-scale 3D worlds by WonderZoom

Figure 4. Qualitative results of WonderZoom on multi-scale 3D world generation.

OursOursOurs

we find that the state-of-the-art 3D scene generation methods and the controllable video generation methods are not able to create multi-scale scenes. In particular, 3D methods always

generate blurry zoom-in views as their 3D scene representations (i.e., Gaussian surfels in WonderWorld [51] and meshes in HunyuanWorld [35]) do not support dynamic updating

views are not aligned with the prompts. In contrast, WonderZoom allows creating new scale structures that are closely aligned with the prompts, and generates high-quality novel views at any new scale.

Method CS↑ CIQA↑ QIQA↑ NIQE↓ QIAA↑ Time/s WonderWorld [51] 0.2687 0.5064 1.081 21.74 1.339 9.3 HunyuanWorld [35] 0.2510 0.2827 1.058 15.21 1.302 704.2 Gen3C [32] 0.3004 0.5489 2.992 4.924 2.018 306.7 Voyager [14] 0.2609 0.5746 3.148 4.913 2.929 596.6 WonderZoom (Ours) 0.3432 0.7035 3.926 3.695 2.986 62.1

Quantitative comparison. We show the quantitative metrics in Table 1 and 2. WonderZoom outperforms all baseline methods in terms of alignment, novel view quality, aesthetics metrics, as well as human’s preferences. This further validates our observations through visual comparison.

- Table 1. Quantitative comparison. “CS” denotes CLIP score, “CIQA” denotes CLIP-IQA+, “QIQA” denotes Q-align IQA, “QIAA” denotes Q-align IAA, and “Time” measures the time used in generating a new-scale scene.

Zoom-in Accuracy Visual Quality Prompt Match

Over WonderWorld [51] 80.7% 98.3% 98.2% Over HunyuanWorld [35] 83.2% 98.7% 98.9% Over Gen3C [32] 77.8% 83.8% 96.1% Over Voyager [14] 76.1% 81.7% 90.9%

- Table 2. Human study 2AFC results of favor rate of WonderZoom (Ours) over baseline methods.

Ours w/o mod Ours

[Figure 201]

[Figure 202]

Figure 5. Ablation on the opacity modulation.

Methods

Metrics

GPU memory FPS

Ours w/o mod. 7.96G 1.4 Ours 3.40G 97.2

- Table 3. Comparison of computational cost for variants about scaleadaptive opacity modulation.

##### 4.2. Ablation study

We evaluate how the key technical components affect the multi-scale generation performances. We focus on the scaleaware opacity modulation, depth registration, and auxiliary view synthesis.

Scale-aware opacity modulation. We consider a variant “Ours w/o mod.” which removes our scale-aware opacity modulation. We show a visual comparison in Figure 5 and a quantitative comparison on computational cost in Table 3. From the table, we can see that without our scale-aware opacity modulation, the computational burden makes it intractable for multi-scale real-time rendering. Furthermore, we observe from the visual result that it creates blurry renderings due to the lack of an appropriate mechanism for rendering multi-scale surfels. In contrast, ours maintains a high-quality rendering while requiring lower GPU memory and providing much faster rendering speed.

Depth registration. We consider a variant “Ours w/o depth registration” that removes the scale-consistent depth registration from WonderZoom. We show a visual comparison in Figure 6. As we can see in the comparison, removing our depth registration creates significant shape distortion on the new detail depth estimation, i.e., the newly generated beetle is distorted when observed from novel views. Our depth registration significantly alleviates this artifact.

[Figure 203]

[Figure 204]

Auxiliary view synthesis. We compare our model with “Ours w/o auxiliary view”. As shown in Figure 7, our auxiliary view synthesis is critical in generating a complete 3D scene, while removing it leads to missing regions as revealed by the grey areas.

Ours w/o depth registration Ours

- Figure 6. Ablation study on our depth registration.

[Figure 205]

[Figure 206]

Ours w/o auxiliary view Ours

- Figure 7. Ablation study on auxiliary view synthesis.

#### 5. Conclusion

WonderZoom allows multi-scale 3D world generation from a single image. Through the scale-adaptive Gaussian surfels and a progressive detail synthesizer, we enable users to interactively zoom into any region and synthesize entirely new details while maintaining cross-scale consistency and real-time rendering. Our experiments demonstrate significant improvements over existing 3D-based and video-based methods in both visual quality and prompt alignment. WonderZoom opens new possibilities for interactive content creation and virtual world exploration across multiple scales.

when new scale images are generated. Camera-controlled video models are able to zoom in, yet their control is imprecise compared to explicit 3D methods, and their generated

Limitations. WonderZoom can struggle in extreme zooming into pure texture regions (we show failure cases in the Appendix in the supplementary materials) because it relies on semantic cues to inform what to generate in the next scale. Future work may explore texture-specific priors or procedural generation that can hallucinate plausible micro-structures when semantic cues are insufficient.

#### References

- [1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864, 2021. 2, 3
- [2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 3
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697– 19705, 2023. 3
- [4] Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein. DiffDreamer: Towards consistent unsupervised singleview scene extrapolation with conditional diffusion models. In ICCV, 2023. 2
- [5] Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, and Noah Snavely. Persistent nature: A generative model of unbounded 3d worlds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20863–20874, 2023. 2
- [6] Jianqi Chen, Yilan Zhang, Zhengxia Zou, Keyan Chen, and Zhenwei Shi. Dense pixel-to-pixel harmonization via continuous image representation. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2023. 1
- [7] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 1, 2
- [8] Paul Engstler, Aleksandar Shtedritski, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Syncity: Training-free generation of 3d worlds. arXiv preprint arXiv:2503.16420, 2025. 2
- [9] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. arXiv preprint arXiv:2302.01133, 2023. 1, 2
- [10] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 2
- [11] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling

camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3

- [12] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 3
- [13] Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. arXiv preprint arXiv:2303.11989, 2023. 1, 2
- [14] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson WH Lau, Wangmeng Zuo, and Chunchao Guo. Voyager: Longrange and world-consistent video diffusion for explorable 3d scene generation. arXiv preprint arXiv:2506.04225, 2025. 6, 8
- [15] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021. 3
- [16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14,

2023. 3, 5

- [17] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4), 2024. 2, 3
- [18] Bryan Sangwoo Kim, Jeongsol Kim, and Jong Chul Ye. Chain-of-zoom: Extreme super-resolution via scale autoregression and preference alignment, 2025. 6
- [19] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 5

- [20] Haoran Li, Haolin Shi, Wenli Zhang, Wenjun Wu, Yong Liao, Lin Wang, Lik-hang Lee, and Pengyuan Zhou. Dreamscene: 3d gaussian-based text-to-3d scene generation via formation pattern sampling. arXiv:2404.03575, 2024. 2
- [21] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision, pages 515–534. Springer,

2022. 2

- [22] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N Plataniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Navigating 3d scenes from a single image. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 798–810,

2025. 2

- [23] Chieh Hubert Lin, Hsin-Ying Lee, Willi Menapace, Menglei Chai, Aliaksandr Siarohin, Ming-Hsuan Yang, and Sergey Tulyakov. Infinicity: Infinite-scale city synthesis. In ICCV,

2023. 2

- [24] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Con-

- ference on Computer Vision, pages 14458–14467, 2021. 1, 2
- [25] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 3
- [26] David Luebke, Martin Reddy, Jonathan D Cohen, Amitabh Varshney, Benjamin Watson, and Robert Huebner. Level of detail for 3D graphics. Elsevier, 2002. 2, 3
- [27] Anish Mittal, Rajiv Soundararajan, and Alan Conrad Bovik. Making a “completely blind” image quality analyzer. IEEE Signal Processing Letters, 20:209–212, 2013. 6
- [28] Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. ACM Transactions on Graphics (ToG), 38(6):1–15, 2019. 2
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [30] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 3
- [31] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks, 2024. 6
- [32] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed worldconsistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3, 6, 8
- [33] Meng-Li Shih, Shih-Yang Su, Johannes Kopf, and Jia-Bin Huang. 3d photography using context-aware layered depth inpainting. In CVPR, 2020. 2
- [34] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, Jo˜ao F Henriques, Christian Rupprecht, and Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. arXiv:2406.04343,

2024. 2

- [35] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025. 2, 6, 7, 8
- [36] Alex Trevithick and Bo Yang. Grf: Learning a general radiance field for 3d scene representation and rendering. In arXiv:2010.04595, 2020. 2
- [37] Richard Tucker and Noah Snavely. Single-view view synthesis with multiplane images. In CVPR, 2020. 2
- [38] Shubham Tulsiani, Richard Tucker, and Noah Snavely. Layerstructured 3d scene inference via view synthesis. In ECCV,

2018. 2

- [39] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, and Shi-Min Hu. Nerf-sr: High quality neural radiance fields using supersampling. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6445–6454, 2022. 3
- [40] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 6
- [41] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision, 2024. 6
- [42] Xiaojuan Wang, Janne Kontkanen, Brian Curless, Steven M Seitz, Ira Kemelmacher-Shlizerman, Ben Mildenhall, Pratul Srinivasan, Dor Verbin, and Aleksander Holynski. Generative powers of ten. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7173– 7182, 2024. 3, 6
- [43] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. SynSin: End-to-end view synthesis from a single image. In CVPR, 2020. 2
- [44] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In ICML, 2024. 6
- [45] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. Citydreamer: Compositional generative model of unbounded 3d cities. In CVPR, 2024. 2
- [46] Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, and Ziwei Liu. GaussianCity: Generative gaussian splatting for unbounded 3D city generation. arXiv 2406.06526, 2024. 2
- [47] Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, SongHai Zhang, and Ying Shan. Geometrycrafter: Consistent geometry estimation for open-world videos with diffusion priors. arXiv preprint arXiv:2504.01016, 2025. 6
- [48] Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. Layerpano3d: Layered 3d panorama for hyper-immersive scene generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 2
- [49] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. Pixelnerf: Neural radiance fields from one or few images. arXiv:2012.02190, 2020. 2
- [50] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney: Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 6
- [51] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. In CVPR, 2025. 1, 2, 4, 5, 6, 7, 8
- [52] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. arXiv preprint arXiv:2311.16493, 2023. 3

- [53] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 3
- [54] Yan Zhang, Wenhan Zhao, Bo Sun, Ying Zhang, and Wen Wen. Point cloud upsampling algorithm: A systematic review. Algorithms, 15(4):124, 2022. 3
- [55] Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Suya Bharadwaj, Tejas You, Zhangyang Wang, and Achuta Kadambi. Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting. arXiv preprint arXiv:2404.06903, 2024. 2
- [56] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv:1805.09817, 2018. 2

[Figure 207]

## WonderZoom: Multi-Scale 3D World Generation

### Supplementary Material

- A. Algorithm We provide an algorithm of WonderZoom in Alg. 1
- B. Additional Results

We provide additional visual results in Figures 9,10, 11 and 12 to show that WonderZoom significantly outperforms other baselines in terms of visual quality.

- C. Failure Cases

As shown in Fig. 13, when zooming repeatedly into the cluster of branches, the scene eventually collapses into pure texture patterns with no remaining semantic cues (e.g., individual branches or leaves). Since WonderZoom relies on the semantics of the current-scale image to infer what should appear at the next scale, such texture-only regions become under-constrained, making further refinement in more new scales unreliable, and finally fail to generate a multi-scale 3D world.

This failure does not occur when recognizable structure is still present, but represents an inherent limitation when the input region no longer contains semantic information.

- D. Additional Details

Additional implementation details. All images are processed at a resolution of 720 × 1088. We use GPT-4V as our VLM for semantic context extraction and editing prompt generation. The initial camera focal length is set to fx = fy = 1024, with progressive zoom-in operations increasing the focal length for finer scales, typically we multiply the current focal length by 8 for a new scale. We use INR-Harmonization [6] after image editing for improved shading consistency.

Human study details. We use Prolific to recruit participants for our human preference evaluation. For each comparison, we collect responses from around 200 participants worldwide. The survey is implemented using Google Forms, and all responses are fully anonymized for both the participants and the authors. Each question presents two zoom-in sequences of the same scene generated by two different methods. Participants are shown the images in a left–right layout: each side contains (1) a global view of the scene and (2) a zoomed-in view of the same region, indicated by a red bounding box and connecting lines. The left–right order of methods is randomized for every participant and every question. Participants are instructed to carefully compare the two sides and make a two-alternative forced choice (2AFC).

[Figure 208]

Figure 8. An example of our user study.

For each comparison, we ask three questions: (i) “Which one looks like the camera is moving closer?” (ii) “Which one looks better to your eyes?” and (iii) “Which one fits the prompt better?” We compare our method with four baselines across six scenes, this yields 24 comparison pairs and 72 questions in total. Each participant answers all 72 questions. A screenshot of the survey interface is provided in Figure 8.

Algorithm 1 Multi-Scale 3D World Generation Control Loop

Input: Initial image I0, initial camera C0 ∈ R4×4 Output: Multi-scale scene hierarchy {E0,E1,...,En} Runtime output: Real-time rendered observation Orender Runtime user control: Camera viewpoint Crender, zoom region Ci+1, (optional) edit prompt Ui+1

- 1: Initialize: E0 ← ReconstructScene(I0,C0) ▷ Initial 3D scene from input image
- 2: Crender ← C0 ▷ Initialize rendering camera
- 3: i ← 0 ▷ Current scale index
- 4: Thread 1: Real-time Scale-Adaptive Rendering ▷ Continuous rendering loop
- 5: while true do
- 6: srender ← drender/ fxrenderfyrender ▷ Compute rendering scale

- 7: Orender ← RenderWithOpacityModulation( ik=0 Ek,Crender) ▷ Sec. 3.1
- 8: Crender ← UserCameraControl() ▷ Interactive camera update
- 9: end while
- 10: Thread 2: Progressive Detail Synthesis ▷ Triggered by user zooming into region of interest with prompt Ui+1 at camera Ci+1
- 11: // Stage 1: New Scale Image Synthesis
- 12: Oi+1 ← Render(Ei,Ci+1) ▷ Coarse observation at zoomed view
- 13: S ← VLM(Render(Ei,Ci)) ▷ Extract semantic context
- 14: I′i+1 ← SuperResolution(Oi+1,S) ▷ Extreme super-resolution
- 15: if Ui+1 ̸= ∅ then
- 16: Ii+1 ← ControlledEdit(I′i+1,Ui+1) ▷ Insert user-specified content
- 17: else
- 18: Ii+1 ← I′i+1
- 19: end if
- 20: // Stage 2: Scale-Consistent Depth Registration
- 21: Dtargeti+1 ← RenderDepth(Ei,Ci+1) ▷ Target depth from coarse scale
- 22: Di+1 ← DepthRegistration(Ii+1,Dtargeti+1 ) ▷ Fine-tune depth estimator
- 23: // Stage 3: Scale-Adaptive Surfel Generation
- 24: Eipartial+1 ← InitializeSurfels(Ii+1,Di+1,Ci+1)
- 25: ▷ Create surfels with snative = dnative/ fxnativefynative

- 26: // Stage 4: Auxiliary View Synthesis
- 27: {Cki+1}Kk=1 ← SampleNeighboringViews(Ci+1)
- 28: {Iki+1,Dki+1} ← AuxiliaryViewSynthesis(Eipartial+1 ,{Cki+1})
- 29: // Stage 5: Optimization
- 30: Ei+1 ← OptimizeSurfels(Eipartial+1 ,{Ii+1,I1i+1,...,IKi+1})
- 31: ▷ Optimize {q,s,o} with L = 0.8L1 + 0.2LD-SSIM
- 32: i ← i + 1 ▷ Increment scale index

[Figure 209]

[Figure 210]

[Figure 211]

: Camera movement : Camera zoom-in

WonderWorldOurs

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

Gen3CVoyagerHunyuan

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

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Input image

Prompt: A yellow bird is standing on the windowsill

###### Figure 9. Visual comparison of multi-scale 3D world generation results.

[Figure 277]

[Figure 278]

[Figure 279]

: Camera movement : Camera zoom-in

WonderWorldOurs

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

Gen3CVoyagerHunyuan

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

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

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

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

Input image

Prompt: A lizard is resting on a wooden windowsill

WonderWorldOurs

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

[Figure 363]

Gen3CVoyagerHunyuan

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

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

Prompt: A Lego miniature toy is on the lamp

Input image

###### Figure 10. Visual comparison of multi-scale 3D world generation results.

[Figure 410]

[Figure 411]

[Figure 412]

: camera movement : camera zoom-in

Gen3CVoyagerHunyuanWonderWorldGen3CVoyagerHunyuanOurs

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

Prompt: A conch is lying on the beach

Input image

WonderWorldOurs

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

Input image

Prompt: A beetle is on the tree bark

Figure 11. Visual comparison of multi-scale 3D world generation results.

[Figure 543]

[Figure 544]

[Figure 545]

: camera movement : camera zoom-in

Gen3CVoyagerHunyuanWonderWorldGen3CVoyagerHunyuanOurs

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

Prompt: A butterfly is on the leaf

Input image

WonderWorldOurs

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

Input image

Prompt: A clownfish is swimming among the coral

Figure 12. Visual comparison of multi-scale 3D world generation results.

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

###### Figure 13. A failure case of WonderZoom. When zooming too deeply into the tree region, the view collapses into texture-like patterns instead of meaningful branch structures.

