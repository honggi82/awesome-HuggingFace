## AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model

Yutian Chen1,2, Shi Guo1,†, Renbiao Jin1,3, Tianshuo Yang1,4, Xin Cai1,2, Yawen Luo2, Mingxin Yang1,3, Mulin Yu1, Linning Xu1,2, and Tianfan Xue2,1,5

1Shanghai AI Lab 2CUHK MMLab 3Shanghai Jiao Tong University 4The University of Hong Kong 5CPII under InnoHK

# arXiv:2604.19747v1[cs.CV]21Apr2026

https://yutian10.github.io/AnyRecon/

Abstract. Sparse-view 3D reconstruction is essential for modeling scenes

from casual captures, but remain challenging for non-generative reconstruction. Existing diffusion-based approaches mitigates this issues by synthesizing novel views, but they often condition on only one or two capture frames, which restricts geometric consistency and limits scalability to large or diverse scenes. We propose AnyRecon, a scalable framework for reconstruction from arbitrary and unordered sparse inputs that preserves explicit geometric control while supporting flexible conditioning cardinality. To support long-range conditioning, our method constructs a persistent global scene memory via a prepended capture view cache, and removes temporal compression to maintain frame-level correspondence under large viewpoint changes. Beyond better generative model, we also find that the interplay between generation and reconstruction is crucial for large-scale 3D scenes. Thus, we introduce a geometry-aware conditioning strategy that couples generation and reconstruction through an explicit 3D geometric memory and geometry-driven capture-view retrieval. To ensure efficiency, we combine 4-step diffusion distillation with context-window sparse attention to reduce quadratic complexity. Extensive experiments demonstrate robust and scalable reconstruction across irregular inputs, large viewpoint gaps, and long trajectories.

Keywords: 3D Reconstruction · Video Diffusion Model

### 1 Introduction

Novel view synthesis and 3D reconstruction are fundamental problems in computer vision and graphics, enabling applications ranging from immersive virtual environments to augmented reality and visual effects. Recent advances in neural reconstruction methods, including implicit representations such as NeRF [14] and explicit point-based approaches like 3D Gaussian Splatting [8], have demonstrated remarkable visual fidelity. However, these methods rely on densely sampled multi-view images captured under controlled acquisition setups. While realworld visual data—such as handheld captures or Internet videos—are typically

† indicates corresponding author.

Cameras Input Generated Frames Interpolation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Frame 10 Frame 20 Frame 30 Frame 40

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|[Figure 22]|
|---|

Frame 10 Frame 20 Frame 30 Frame 40

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Extrapolation

[Figure 27]

[Figure 28]

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

Frame 1

Frame 40 Frame 80

Large Scene

[Figure 43]

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

|[Figure 51]|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Frame 120 Frame 160 Frame 180

Frame 220

|[Figure 56]|
|---|

- Fig. 1: AnyRecon demonstrates robust performance across multiple reconstruction settings: (Top) Interpolation, filling in gaps between distant captured views; (Middle) Extrapolation, synthesizing novel content beyond the observed range; and (Bottom) Large Scene Reconstruction, maintaining consistency across long-trajectory sequences. The camera visualization (left) illustrates the sparse input poses (red) and the dense generated path (blue).

sparse and irregular. Enabling reconstruction from such arbitrary view would allow scalable conversion of everyday captures into explorable 3D scenes.

Recent work try to mitigate this sparse view challenge by using more views created by diffusion-based novel view synthesis. While early efforts employ image generative models to infer 3D structure [23,24], more recent works leverage video generation models to synthesize novel views, as they can better capture crossview coherence through temporal modeling. One line of work [2,16,29] conditions diffusion models primarily on projected point cloud renderings, providing only one or two captured RGB frames (e.g., the first and last views). Although renderings offer coarse geometric guidance, limited real-image conditioning weakens appearance fidelity and global context, making generation sensitive to incomplete or low-quality geometry. Another line of work [1,6] relies solely on RGB images and camera poses without explicitly incorporating reconstructed geometry into the generation process. By learning geometric consistency implicitly, these methods struggle to maintain precise pose alignment and spatial consistency, limiting their deployment for real-world 3D reconstruction under irregular observations.

In this work, we aim to enable high-quality and large scale 3D reconstruction from sparse inputs. Unlike prior sparse-view diffusion models restricted to only one or two reference views, our diffusion model flexibly conditions on an arbitrary number of captured RGB images alongside point cloud renderings. However, supporting flexible input cardinality introduces many challenges. First, input images may be arbitrary captured with large viewpoint gaps, while existing video diffusion frameworks are suboptimal for non-sequential inputs, as they rely on temporally causal latent compression [12]. Moreover, to reconstruct a large complex scene, it is impossible to fit all input into diffusion model at once.

AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model 3

Therefore, a robust iterative reconstruction strategy is required to reconstruct a scene by small segment. These challenges demand a reconstruction framework that preserves fine-grained control while remaining computationally efficient.

To address these challenges, we propose AnyRecon, a scalable framework for sparse-view 3D reconstruction. First, we develop a diffusion-based novel view synthesis model that supports arbitrary and unordered sparse inputs while maintaining explicit geometric control. Specifically, we construct a 3D point cloud from the sparse captures and render it into target viewpoints to serve as geometric conditions. To handle flexible inputs, we maintain a global memory cache by prepending these original captures to the rendered priors in the sequence, thereby enabling long-range conditioning across arbitrary viewpoints. Furthermore, to ensure frame-level correspondence under large viewpoint changes, we remove temporal compression in the latent encoder. At last, to reduce computation for large-scale reconstruction, we adopt a 4-step diffusion distillation strategy and introduce a context-window sparse attention mechanism that restricts attention to local temporal windows and geometry-aligned retrieved views.

Furthermore, to support segment-by-segment reconstruction of a large-scale scene, we introduce a geometry-aware conditioning strategy that couples generation and reconstruction. Specifically, this strategy creates an iterative loop where generated outputs continuously update a shared 3D geometry, which in turn guides the conditioning of subsequent segments. First, we build an explicit 3D Geometry Memory by back-projecting newly generated images into the initial point cloud, enabling incrementally updated geometric memory across trajectory segments. Second, when conditioning a new trajectory segment, we perform geometry-driven view selection from a captured view bank based on geometric contribution and spatial overlap with the current reconstruction, rather than relying on image-level similarity or field-of-view (FOV) heuristics [3,10,28]. Together, these components form a closed geometric loop between reconstruction and generation, ensuring that diffusion is guided by spatially informative observations and improving robustness under occlusion and complex scene layouts.

Extensive benchmarks demonstrate that AnyRecon delivers superior results compared to state-of-the-art baselines. As shown in Fig. 1, our method facilitates seamless view interpolation and long-range extrapolation across diverse, large-scale scenes (over 200 frames), maintaining high fidelity despite sparse and irregular input captures.

Our key contributions are summarized as follows:

- – A flexible sparse-view reconstruction framework. We propose a videodiffusion-based approach that supports arbitrary and unordered conditioning views, enabling robust 3D reconstruction.
- – A geometry-aware conditioning design. We couple generation and reconstruction via a 3D Geometry Memory with back-projection and geometrydriven capture-view retrieval, ensuring spatially grounded diffusion guidance.
- – An efficient diffusion architecture. By removing temporal compression and adopting diffusion distillation with block sparse attention, our method

generalizes across varying numbers of input views while maintaining computational efficiency.

### 2 Related Work

#### 2.1 Traditional Sparse-View Reconstruction.

Sparse-view reconstruction is inherently ill-posed due to the vast unobserved regions and geometric ambiguities arising from limited inputs. Early efforts addressed this by incorporating various geometric priors and regularization techniques. For instance, FreeNeRF [25] and RegNeRF [15] employ frequency-domain regularization and depth smoothness constraints to stabilize optimization when only sparse views are available. Other approaches focus on leveraging auxiliary supervision from pre-trained models to provide additional scene constraints. Specifically, SPARF [17] utilizes correspondence field and optical flow to enforce multi-view consistency, while MonoSDF [30] and DS-NeRF [5] integrate monocular depth and normal maps as supplementary signals to refine surface geometry.

While these methods improve reconstruction to some extent, they rely on the limited information present in the sparse inputs and often struggle to synthesize plausible details in large disoccluded areas. This motivates the shift toward the diffusion-based approaches discussed above, which leverage large-scale generative priors to hallucinate consistent geometry and appearance beyond the captured observations.

#### 2.2 3D Reconstruction with Diffusion Model

Recent advancements in diffusion models have significantly propelled 3D and 4D generation. Pioneer works such as ReconFusion [24] and FreeNeRF [25] supervise novel views by sampling from diffusion priors during the reconstruction process. Specifically, ReconFusion [24] utilizes a diffusion model conditioned on sparse inputs to predict pseudo-ground-truth images for novel views, which are then used to optimize NeRF or 3DGS. To improve optimization efficiency, DeceptiveNeRF [13] and 3D-GS Enhancer [6] first render coarse pseudo-images from a sparse-view-reconstructed representation and refine these views using diffusion models, avoiding the need to query the diffusion prior at every step. However, these image-based approaches often lack cross-view geometric coherence and necessitate computationally expensive iterative refinements.

To address the temporal and spatial consistency issues of image-based models, recent research has shifted towards video-based generators. One category of methods [1, 6] relies primarily on RGB images and camera poses to implicitly learn spatial consistency. While flexible, these methods struggle with precise pose alignment without explicit geometric guidance. Alternatively, geometry-aware video generators [2,4,16,29] incorporate projected point cloud renderings to provide coarse structural priors. These point clouds are typically estimated using pretrained geometry reconstruction models [19–22], which infer sparse or dense

[Figure 57]

###### AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model 5

[Figure 58]

[Figure 59]

[Figure 60]

(C) Context-Window Sparse Attention (§3.3)

(A) Geometry-Aware Retrieval (§3.4)

(B) Unordered Contextual Video Diffusion (§3.2)

Captured View Bank 𝐼

Selected Views 𝐼 Novel Views 𝐼

[Figure 61]

Context Seq. Generation Seq.

|[Figure 62]<br><br>[Figure 63]|
|---|

|[Figure 64]<br><br>[Figure 65]|
|---|

|[Figure 66]<br><br>[Figure 67]|
|---|

|[Figure 68]<br><br>[Figure 69]|
|---|

|[Figure 70]<br><br>[Figure 71]|
|---|

[Figure 72]

|[Figure 73]<br><br>[Figure 74]<br><br>Memory|
|---|

|[Figure 75]<br><br>[Figure 76]|
|---|

|[Figure 77]<br><br>[Figure 78]<br><br>Global|
|---|

|[Figure 79]<br><br>[Figure 80]|
|---|

|[Figure 81]<br><br>[Figure 82]|
|---|

M

Low Related Views High Related Views

|[Figure 83]|
|---|

|Unordered<br><br>[Figure 84]|
|---|

Vis k

|[Figure 85]<br><br>[Figure 86]<br><br>Visibility Mask|
|---|

|[Figure 87]<br><br>[Figure 88]<br><br>𝑀|
|---|

|[Figure 89]<br><br>[Figure 90]|
|---|

[Figure 91]

[Figure 92]

Uno Contextual Video Diffusion

[Figure 93]

Video Diffusion Transformer with Context-Window Sparse Attention

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Novel Views 𝒱

Full-Attention Region

Update

|[Figure 101]<br><br>[Figure 102]<br><br>Generated|
|---|

|[Figure 103]<br><br>[Figure 104]<br><br>Frames|
|---|

|[Figure 105]<br><br>[Figure 106]|
|---|

|[Figure 107]<br><br>[Figure 108]|
|---|

|[Figure 109]<br><br>[Figure 110]<br><br>𝐼|
|---|

ed s

Context-Window Sparse Attention Region

| |
|---|

3D Geometry Memory ℳ

- Fig. 2: Pipeline of AnyRecon. Given arbitrary sparse input views organized in a capture view bank Icap, we perform geometry-aware retrieval to select spatially informative views for each novel trajectory segment. The selected views Isel, together with geometry renderings under target viewpoints Irender, are fed into a video diffusion transformer equipped with context-window sparse attention for scalable long-range conditioning. The are then used to update the 3D geometry memory Mgeo, fo between generation and reconstruction.

- 3D structures from input images before projection into the diffusion pipeline. Nevertheless, these diffusion model typically condition on a very limited number of captured frames (e.g., only the first and last), which limits their ability to capture global scene context and fine-grained appearance fidelity. In contrast, our AnyRecon utilizes a global scene memory to incorporate an arbitrary number of reference views Isel and enforces strict spatial alignment through channel-wise concatenation of visibility masks Mt and rendered observations Irender, closing the gap between generative priors and explicit 3D geometry.

- 3 Method

re Retrieval

[Figure 111]

|[Figure 112]<br><br>generated(a)novelGeometry-Awareviews forming a closed loop|
|---|

#### 3.1 Overview

We present AnyRecon, a framework for sparse-view 3D reconstruction that supports arbitrary and unordered inputs while preserving geometric consistency across long viewpoint changes. Our method alternates between (1) diffusionbased trajectory generation to create more views and (2) geometry refinement using the generated views. This forms a closed loop that progressively reconstructs the scene, enabling scalable processing of long trajectories and large-scale inputs.

As illustrated in Fig. 2, our framework operates in three key stages to form the generation-reconstruction loop. First, Initial Geometry Construction: All input views are organized into a captured view bank Icap, from which an initial

- 3D geometry memory Mgeo is established via a feed-forward point map estimation method (e.g., VGGT [19] or π3 [22]). Second, Novel View Generation: To

synthesize novel views on user-specified trajectory Vnovel, we chop the entire trajectory into small for efficiency. For each segment, we perform geometry-aware

retrieval (§3.4) to select views Isel important for reconstructing this segment from all capture views Icap. These selected views, along with point-cloud renderings Irender and visibility masks Mt derived from the current Mgeo, are fed into our proposed unordered contextual video diffusion (§3.2) to synthesize novel views Iˆnovel on the trajectory. To mitigate computational complexity, we incorporate context-window sparse attention and an efficient 4-step sampling strategy (§3.3). Third, Geometry Updating: The geometry reconstructed from the newly synthesized views Iˆnovel is extracted to update the global memory Mgeo (§3.4). This updated geometry is subsequently fed back into the next segment’s retrieval and generation steps, completing the iterative reconstruction loop.

#### 3.2 Unordered Contextual Video Diffusion

To achieve robust reconstruction from diverse inputs, AnyRecon transitions from a standard sequential video generator to a geometry-conditioned diffusion model. Specifically, this module takes the retrieved reference views Isel and the rendered geometric guidance Irender under target viewpoints Vnovel as joint contextual inputs to synthesize a sequence of high-fidelity novel views Iˆnovel. To ensure both precise spatial alignment and awareness of occlusions, the target noisy latents are concatenated along the channel dimension with the rendered point-cloud observations Irender and their corresponding visibility masks Mt, both of which are derived from the 3D geometry memory Mgeo. Beyond this spatial geometry conditioning, effectively utilizing sparse and unordered inputs requires breaking the strict temporal continuity assumptions inherent in standard video diffusion models. To fully decouple the generation process from temporal dependencies and handle arbitrary viewpoint gaps, we introduce two key architectural innovations: a global scene memory for flexible context injection, and a non-compressive latent encoding to prevent spatial-temporal feature entanglement.

Global Scene Memory. To support an arbitrary number of conditioning views without being constrained by fixed-length input buffers, we introduce a Global Scene Memory mechanism as shown in Fig. 2(B). Specifically, the retrieved reference views Isel are set in the beginning of each chunk and serve as a persistent global key–value (KV) memory cache within the video diffusion transformer. (See Sec. 4.4 for further comparative analysis.)

Instead of modeling captured and target views as temporally adjacent frames in a single sequence [2, 29], this design treats conditioning capture views as a flexible and queryable 3D memory, enabling generation along arbitrary spatial trajectories independent of the capture sequence. The model then generates novel views based on this memory, enabling spatially consistent reconstruction.

Non-Compressive Latent Encoding. Traditional video diffusion models [18] often use temporal compression (e.g., 3D-VAEs) to reduce dimensionality, which relies on an assumption of temporal smoothness. However, this prior fails in sparse-view scenarios where large viewpoint gaps break the continuity between adjacent frames. Compressing across time in such irregular sequences causes feature entanglement between disparate views, obscuring the precise spatialtemporal alignment necessary for reconstruction as shown in Fig. 3 (c) or (d).

[Figure 113]

[Figure 114]

[Figure 115]

###### AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model 7

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

50 steps 50 steps 50 steps

(a) CaptureView1

(b) CaptureView2 (c) FullTC (d) PartialTC (e) w/oTC

[Figure 121]

[Figure 122]

[Figure 123]

###### 4 steps

4 steps

[Figure 124]

(f) Input Render

(g) GT

(h) GT (i) Full Attention (j) Sparse Attention

- Fig. 3: Ablation on temporal compression (TC), 4-step distillation and sparse attention. Full temporal compression follows Wan by keeping only the first frame uncompressed while compressing subsequent frames (e.g., ×4), whereas partial

|geometric|
|---|

|the|
|---|

|compresses to preserve|
|---|

temporal compression

th captured input views uncompressed t rve accurate

com only the rendered maps and keeps

ge cues.

To overcome this limitation, AnyRecon employs Non-Compressive Latent Encoding. By using a frame-wise 2D VAE, we bypass temporal pooling and preserve the one-to-one mapping between latent tokens and pixel coordinates, enabling robust geometry-aware synthesis even with wide-baseline inputs.

#### 3.3 Efficient Sparse Attention and 4-Step Diffusion Sampling

To maintain high-fidelity synthesis across extended sequences while ensuring computational efficiency, we introduce two key optimizations: a context-window sparse Attention mechanism to handle the expanded token space and a 4-step diffusion sampling strategy to accelerate the generation process.

Context-Window Sparse Attention. Although the non-compressive encoding and global scene memory enhance rendering quality, they significantly expand the sequence length L, resulting in prohibitive O(L2) complexity. To mitigate this, we introduce a context-window sparse attention mechanism (Fig. 2(C)) where each frame in the target trajectory Inovel restricts its receptive field to a local temporal window and a selectively retrieved subset of geometry-aligned reference views Isel. This mechanism focuses the model’s capacity on visually relevant regions, ensuring scalability for large-scale scenes.

4-Step Diffusion Sampling. To accelerate the inference of the Wan video diffusion model, we employ Distribution Matching Distillation [26,27] to distill the pre-trained model into a student network capable of high-quality generation in just 4 steps. We discretize the continuous noise schedule into a fixed trajectory Tsteps = {1000,750,500,250,0}. The optimization objective minimizes the Kullback-Leibler (KL) divergence between the student’s generated distribution and the real distribution, approximated via the score difference between a frozen teacher and a trainable critic. To implement this, the generator loss Lgen is for-

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 132]

[Figure 133]

[Figure 134]

###### 8 Y. Chen et al.

[Figure 135]

[Figure 136]

[Figure 137]

|[Figure 138]<br><br>Inconsistent ❌<br><br>[Figure 139]|
|---|

|[Figure 140]|
|---|

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

(e) 2nd Chunk Output (w/o Update)

- (c) 1st Chunk Input
- (d) 1st Chunk Output

- (a) Point Cloud ℳ w/o Update

- (b) Point Cloud ℳ w/ Update

[Figure 148]

|[Figure 149]<br><br>Consistent ✅<br><br>[Figure 150]|
|---|

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

|[Figure 156]|
|---|

[Figure 157]

Capture view

[Figure 158]

2nd Chunk view (f) 2nd Chunk Output (w/ Update)

1st Chunk view

[Figure 159]

- Fig. 4: Explicit 3D Geometry Memory Update. Without memory update, newly generated trajectory segments are not integrated into the reconstructed point cloud, leading to incomplete geometry and inconsistent rendering in subsequent chunks. Our explicit memory incrementally integrates generated views into the point cloud, maintaining coherent scene structure across trajectory segments.

[Figure 160]

mulated as a pseudo-regression objective with a stop-gradient (sg) operator:

2

xˆψ(zt) − xˆϕ(zt) σnorm

- 1

- 2

x ˆθ(zt) − sg x ˆθ(zt) + η

, (1)

Lgen = Ez

t,t

2

where xˆθ,xˆψ, and xˆϕ denote the denoised predictions (xˆ0) derived from the student, teacher, and critic respectively; η is the step size, and σnorm acts as a time-dependent normalization factor. Concurrently, the critic is optimized via a standard denoising score matching objective on the student’s generated samples: Lcritic = Ez

t,t[∥xˆϕ(zt) − xclean∥22], where xclean is the original noise-free output produced by the student. To stabilize the training dynamics, we employ an alternating update schedule between the student generator and the critic.

Together, these optimizations achieve up to a 20× speedup in generation over the vanilla diffusion implementation, without obvious rendering degradation.

#### 3.4 Geometry-Aware Conditioning Strategy

To support long-trajectory generation and maintain scene-level consistency, we couples the diffusion process with an explicit 3D representation. This coupling forms a recursive loop: newly generated views provide the visual data to expand the 3D reconstruction, while the updated geometry offers precise spatial anchors to guide subsequent generation.

3D Geometry Memory Update. Fig. 4 illustrates the critical role of our explicit 3D geometry memory Mgeo. Without updating Mgeo with newly reconstructed points, newly generated trajectory segments are not integrated into the global scene representation. Consequently, conditioning subsequent generation stages on an incomplete Mgeo (Fig. 4(a)) results in a significant visual and geometric mismatch between previously synthesized views (Fig. 4(d)) and those generated in later stages (Fig. 4(e)).

[Figure 161]

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]<br><br>[Figure 165]<br><br>Render RGB|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]<br><br>[Figure 169]<br><br>Render Index|
|---|

NovelView

Capture View Novel View

- Fig. 5: Geometry-aware Memory Retrieval. While FOV- or similarity-based methods would select all four views, our geometry-aware retrieval accounts for 3D spatial overlap and visibility. In the render index map (right), each color corresponds to a source view, representing its geometric contribution to the target perspective. This mechanism effectively excludes occluded views (e.g., the yellow view) that provide no valid support, leading to more reliable conditioning for generation.

To address this, we maintain Mgeo as an incrementally updated point cloud that evolves alongside the generation process. After synthesizing a segment of the novel trajectory, we employ the feed-forward point map estimation model π3 [22] to extract 3D geometry from the generated views together with the original ones. This newly reconstructed geometry then replaces the existing memory Mgeo. As shown in Fig. 4(b), this fusion successfully recovers missing scene details, such as the chair’s structure.

By iteratively integrating points from generated frames, Mgeo evolves into a spatially consistent backbone that anchors each new segment to the global structure. This explicit update mechanism prevents error accumulation, effectively mitigating geometric drift across extended trajectories—as evidenced by the alignment between the early and late stages shown in Fig. 4(d) and (f).

Geometry-Driven View Selection. When reconstructing scene-level environments, the captured view bank Icap often contains a massive number of images, making it infeasible to input all reference views into the video diffusion model simultaneously. Therefore, selecting an informative subset is critical for synthesis fidelity and computational efficiency. Specifically, incorporating spatially irrelevant reference views introduces redundant conditioning that distracts the model’s spatial reasoning and increases inference latency. Conversely, omitting highly relevant reference views leads to under-constrained generation, causing the synthesized sequence to deviate from the ground-truth observations.

Fig. 5 illustrates our retrieval strategy. Given a target novel viewpoint (left), conventional FOV- or similarity-based retrieval would select all capture views due to apparent angular or appearance proximity. However, such heuristics ignore occlusion and true geometric support. Instead, we perform geometry-driven retrieval based on the current 3D geometry memory Mgeo. Specifically, we render Mgeo from the target viewpoint to generate a visibility index map (Fig. 5, right), which identifies the source-view attribution for every visible 3D point.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

InterpolationExtrapolation

Rendered Input ViewCrafter

3DGS Difix3D+

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Capture Image

Uni3c Ours GT

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Rendered Input 3DGS Difix3D+ ViewCrafter

[Figure 185]

[Figure 186]

[Figure 187]

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]<br><br>Image|
|---|

|[Figure 191]<br><br>Capture|
|---|

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Im Uni3c Ours GT

Fig. 6: Quality Results on DL3DV Dataset [11].

This visibility map allows us to quantify the geometric contribution of each capture view to the current target perspective. Formally, let C = {(Ii,Pi)} denote the set of capture views and their poses. For each candidate view i, we compute how many of the visible points under the target viewpoint originate from view i during reconstruction as:

si = |Vnovel ∩ Si| |Vnovel|

, (2)

where Vnovel denotes the set of geometry points visible from the target viewpoint, and Si represents the subset of points in Mgeo reconstructed from capture view i. Views that contribute few or no visible points (e.g., the occluded yellow view in Fig. 5) receive low scores and are filtered out. We select the top-k views according to {si} as conditioning inputs for the diffusion model. By conditioning retrieval on target-view visibility rather than appearance similarity, our method ensures that selected views provide direct geometric support for the current generation. This visibility-aware mechanism improves robustness under occlusion and complex spatial layouts, leading to more reliable novel view synthesis.

### 4 Experiments

#### 4.1 Datasets

We train AnyRecon on the DL3DV-10K [11] dataset, a large-scale collection of high-quality 3D indoor and outdoor scenes. The original video sequences are partitioned into clips of 40 frames each at the resolution of 512×896. To emulate diverse and irregular input scenarios while strengthening the model’s generative

priors, we employ a randomized conditioning sampling strategy. Specifically, for each clip, we fix the first frame as a base reference and randomly select N ∈ [2,4] additional conditioning views. To balance the model’s ability to handle both narrow-baseline interpolation and wide-baseline synthesis, we sample these additional indices from either the first 20 frames (50% probability) or the entire 40-frame window (50% probability). The selected conditioning views Isel are then processed by our feed-forward reconstruction module π3 [22] to establish the initial 3D geometry memory Mgeo. Finally, we project the point-cloud observations from Mgeo onto the target novel viewpoints Vnovel to generate the corresponding Irender and visibility masks Mt, forming the complete training pairs for our geometry-controlled generative model.

#### 4.2 Implementation Details

We implement AnyRecon by fine-tuning the Wan2.1-I2V-14B [18] model using LoRA [7] with a rank of 32. The training procedure is executed in three distinct stages to ensure stable convergence and efficient high-resolution synthesis. First, we perform full self-attention fine-tuning for 100k iterations, allowing the model to adapt its internal generative priors to our geometry-controlled input space. Second, we transition to the sparse attention mechanism (§3.3) and conduct a 10k-iteration warm-up phase. Specifically, we configure the block sparse attention with a 2 × 8 × 8 block size, restricting each frame to attend only to the selectively retrieved subset of geometry-aligned reference views Isel, alongside its 8 preceding and 8 succeeding adjacent views. This formulation enables the model to maintain long-range spatial consistency within the truncated receptive fields. Finally, we apply DMD2 distillation [26] for an additional 30k iterations, effectively compressing the denoising process into a 4-step sampling trajectory while preserving high-fidelity structural details. All experiments are conducted on 64 NVIDIA A800 GPUs using the AdamW optimizer with a constant learning rate of 1 × 10−4 for the initial stages and 1 × 10−5 during distillation.

#### 4.3 Comparison Results

Metrics and Baselines. We employ three widely-recognized metrics to quantitatively evaluate the synthesized results: Peak Signal-to-Noise Ratio (PSNR) for pixel-level accuracy, Structural Similarity Index (SSIM) for structural integrity, and Learned Perceptual Image Patch Similarity (LPIPS) for high-level perceptual quality. For a comprehensive comparison, we benchmark AnyRecon against three state-of-the-art diffusion-based 3D reconstruction and novel view synthesis methods: Difix3D+ [23], which focuses on geometry-refined image synthesis; ViewCrafter [29], which utilizes video diffusion priors for view interpolation; and Uni3C [2], a unified framework for cross-domain 3D consistency. These baselines represent the current frontier in leveraging generative models for sparse-view scenarios, providing a rigorous reference for assessing our model’s advancements in spatial reasoning and efficiency.

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

Rendered Input 3DGS Difix3D+ ViewCrafter

InterpolationExtrapolation

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

| |[Figure 212]| | |
|---|---|---|---|
|[Figure 213]<br><br>Capture Im| |[Figure 214]<br><br>[Figure 215]<br><br>age| |

Uni3C Ours GT

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Rendered Input 3DGS Difix3D+ ViewCrafter

[Figure 224]

[Figure 225]

[Figure 226]

|[Figure 227]<br><br>[Figure 228]<br><br>Capture Im|[Figure 229]<br><br>age|
|---|---|
|[Figure 230]|[Figure 231]|

Uni3C Ours GT

[Figure 232]

[Figure 233]

[Figure 234]

Fig. 7: Quality Results on Tanks and Temples Dataset [9].

Evaluation Benchmarks. To evaluate the generalization and robustness of AnyRecon, we conduct extensive testing on 10 scenes from DL3DV-Evaluation set [11], and 5 scenes from Tanks and Temples Dataset [9]. For each test sequence, we we sample 40 frames at a resolution of 512 × 896 frames; specifically for the high-density Tanks and Temples sequences, we perform a 1/5 temporal sub-sampling to ensure a challenging baseline for sparse-view reconstruction. Our evaluation is categorized into two distinct configurations: Interpolation and Extrapolation. In the Interpolation setting, we provide the 1st, 21st, and 40st frames as captured views V to assess the model’s sparse-view completion capability across large baseline gaps. In the Extrapolation setting, we provide the 1st, 11th, 21st, and 31st frames as conditioning inputs to specifically test the model’s generative synthesis ability in hallucinating visually and structurally coherent content for the unobserved tail of the trajectory.

Quantitative comparisons are summarized in Table 1, and qualitative visualizations across different datasets are presented in Fig. 6 and Fig. 7. As illustrated, Difix3D+ fails to handle scenarios with large viewpoint gaps, often leaving significant artifacts inherited from sparse-view 3DGS reconstructions. While ViewCrafter and Uni3C leverage video diffusion priors, their inability to incorporate multiple conditioning frames during the diffusion process leads to generated views that do not strictly align with the captured observations. This results in cross-view geometric inconsistencies, loss of fine-grained details, and noticeable color shifts. In contrast, AnyRecon effectively leverages its global scene memory to complete missing regions in novel views based on the captured views, while simultaneously hallucinating plausible new content that maintains both structural integrity and appearance consistency. Moreover, AnyRecon achieves the best efficiency among all compared methods, requiring only 105 seconds per sequence. These results demonstrate that AnyRecon not only improves recon-

###### Table 1: Quantitative comparison under interpolation and extrapolation settings.

Interpolation Extrapolation Method PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Time (s)∗ ↓ DL3DV Difix3D+ [23] 17.88 0.551 0.290 18.74 0.576 0.261 1200 ViewCrafter [29] 15.86 0.463 0.394 15.51 0.459 0.406 170 Uni3C [2] 16.33 0.471 0.319 15.69 0.457 0.344 340 Ours 20.95 0.656 0.151 21.16 0.660 0.158 105 Tanks and Temples Difix3D+ [23] 19.43 0.629 0.163 18.67 0.594 0.190 1200 ViewCrafter [29] 15.85 0.474 0.364 15.83 0.481 0.361 170 Uni3C [2] 16.77 0.514 0.263 16.54 0.502 0.274 340 Ours 20.37 0.639 0.158 20.30 0.629 0.181 105

∗ All reported times represent the average inference duration for generating a 40-frame scene at a resolution of 512 × 896.The inference time for Difix3D+ is calculated based on its default 30-iteration refinement process (∼40s per iteration), excluding the initial 3DGS optimization time. The latency for all other methods, including ours, represents the duration of a single-pass diffusion model inference.

struction fidelity but also significantly reduces inference latency, making it more practical for real-world applications.

#### 4.4 Ablation Study

Temporal Compression. To verify the impact of different temporal processing strategies on the reconstruction quality, we conduct an ablation study with three configurations: full temporal compression, partial temporal compression (where only rendered maps are compressed while captured views remain uncompressed), and our proposed dense attention without temporal compression. The quantitative results and visual comparisons are presented in Table 2 and Fig. 3(c)(d)(e), respectively.

As illustrated in the results, both full and partial temporal compression lead to a noticeable degradation in visual fidelity. Specifically, these compressionbased models struggle to preserve fine-grained structural details, such as the intricate metal grids shown in Fig. 3(c)(d), where the structures appear fractured or blurred. This is primarily because temporal down-sampling discards highfrequency spatial information that is crucial for thin-structure reconstruction. In contrast, our configuration without temporal compression effectively maintains the complete geometric details and sharp textures by attending to the original resolution of both rendered and captured views. This validates the necessity of preserving full temporal resolution to ensure high-fidelity scene synthesis in complex environments.

Distillation and Sparse Attention. We further evaluate the impact of our acceleration strategies, including model distillation and sparse Attention, as detailed in Table 2 and Fig. 3(e)(i)(j) . While the dense attention baseline

###### Table 2: Ablation study on temporal compression (TC) and inference efficiency. We evaluate our model across various diffusion steps and attention strategies on the DL3DV Dataset [11] interpolation configuration.

50 Diffusion Steps 4 Distilled Steps (w/o TC) Metric Full TC Paritail TC w/o TC Full Atten. Sparse Atten.

PSNR ↑ 20.16 21.10 21.57 21.32 20.95 SSIM ↑ 0.616 0.661 0.687 0.673 0.656 LPIPS ↓ 0.179 0.153 0.140 0.148 0.151

Time (s)∗ ↓ 210+(15) 270+(15) 1820+(15) 140+(15) 90+(15)

∗The reported Time (s) is formatted as “DiT inference time + (15)”, where the 15s accounts for the encoder and decoder overhead. All values represent the average duration for generating a 40-frame video at a resolution of 512 × 896. Since the Full TC requires input sequences of length 4n + 1, its runtime is measured on a 41-frame sequence and scaled by a factor of 40/41 to align with the 40-frame baseline.

(50 steps) achieves the highest reconstruction quality, its inference time (1820s) is prohibitively expensive for practical applications. By applying 4-step distillation, we significantly reduce the latency to 140s with only a marginal drop in PSNR (0.24 dB). Furthermore, the integration of sparse Attention provides a substantial boost in efficiency, further compressing the inference time to 90s—a 20× speedup compared to the original dense baseline. Although the sparse constraints lead to a slight decrease in metrics (e.g., PSNR of 20.95), the visual quality remains highly competitive, and the significant reduction in computational overhead makes AnyRecon much more viable for real-time 3D reconstruction tasks. This trade-off demonstrates that the combination of geometry-guided sparse attention and step distillation effectively balances high-fidelity synthesis with rapid deployment.

Global Scene Memory. To validate the necessity of the global scene memory, we conduct an ablation study on the DL3DV Dataset under the interpolation configuration. We compare our full model, which prepends three retrieved reference views into the global memory cache, against a baseline that only conditions the video diffusion model on a single initial frame. Note that to ensure a fair comparison, the explicit point-cloud guidance (Irender) in the baseline is still rendered using the geometry accumulated from all three views. Quantitative and qualitative comparisons are provided in Table 3 and Fig. 8, respectively. Visually, relying solely on rendered point-cloud maps proves insufficient, as these intermediate renderings naturally suffer from projection artifacts—specifically, floating points, blurry boundaries, and inconsistent colors. Consequently, the baseline model struggles to recover high-fidelity textures, resulting in missing details on the tableware and noticeable color shifts on the background wall. In contrast, by maintaining the raw captured views in the global scene memory, our full model allows the diffusion network to flexibly query uncorrupted, high-frequency textural details. This mechanism effectively suppresses geometric artifacts and

##### successfully restores complex structures like the tableware, demonstrating the critical role of the global memory in preserving visual fidelity.

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Rendered Input w/o Global Memory w/ Global Memory GT

Fig. 8: Quality comparison on global scene memory.

Table 3: Quantitative comparison on global scene memory.

###### Method PSNR↑ SSIM↑ LPIPS↓

w/o Global Scene Memory 20.18 0.634 0.205 w/ Global Scene Memory 20.95 0.656 0.151

### 5 Limitation

AnyRecon’s performance depends on the quality of its 3D geometric memory. While resilient to minor inaccuracies—such as pose misalignments, noise, or artifacts—the framework requires basic structural coherence. In extreme cases with minimal view overlap, the initial reconstruction may fail, providing insufficient guidance for diffusion and resulting in suboptimal frame synthesis.

### 6 Conclusion

We presented AnyRecon, a scalable and flexible framework designed for highquality 3D reconstruction from sparse and irregular inputs. Addressing the limitations of existing diffusion-based methods in handling arbitrary views and large-scale scenes, we developed a novel video diffusion architecture that integrates explicit geometric control via point cloud renderings. By removing temporal compression and introducing a global memory cache, our model effectively maintains frame-level correspondence and supports unordered input conditioning. Furthermore, we proposed a geometry-aware conditioning strategy that establishes a closed loop between generation and reconstruction. Through the implementation of a 3D Geometry Memory and a geometry-driven view selection mechanism, AnyRecon enables robust, segment-by-segment reconstruction of complex, large-scale environments. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art baselines in view interpolation, extrapolation, and large-scene consistency, offering a practical solution for converting casual, sparse real-world captures into explorable 3D assets.

### References

- 1. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14834–14844 (2025) 2, 4
- 2. Cao, C., Zhou, J., Li, S., Liang, J., Yu, C., Wang, F., Xue, X., Fu, Y.: Uni3c: Unifying precisely 3d-enhanced camera and human motion controls for video generation. In: Proceedings of the SIGGRAPH Asia 2025 Conference Papers. pp. 1–12

(2025) 2, 4, 6, 11, 13

- 3. Chen, S., Wei, C., Sun, S., Nie, P., Zhou, K., Zhang, G., Yang, M.H., Chen, W.: Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028 (2026) 3
- 4. Chen, W., Bi, J., Huang, Y., Zheng, W., Duan, Y.: Scenecompleter: Dense 3d scene completion for generative novel view synthesis. arXiv preprint arXiv:2506.10981

(2025) 4

- 5. Deng, K., Liu, A., Zhu, J.Y., Ramanan, D.: Depth-supervised NeRF: Fewer views and faster training for free. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2022) 4
- 6. Gao*, R., Holynski*, A., Henzler, P., Brussee, A., Martin-Brualla, R., Srinivasan, P.P., Barron, J.T., Poole*, B.: Cat3d: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems (2024) 2, 4
- 7. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id= nZeVKeeFYf9 11
- 8. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., et al.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023) 1
- 9. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics 36(4) (2017) 12
- 10. Li, R., Torr, P., Vedaldi, A., Jakab, T.: Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25690–25699 (2025) 3
- 11. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al.: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22160–22169 (2024) 10, 12, 14
- 12. Liu, J., Li, J., Deng, J., Li, G., Zhou, S., Fang, Z., Lao, S., Deng, Z., Zhu, J., Ma, T., et al.: Dreamontage: Arbitrary frame-guided one-shot video generation. arXiv preprint arXiv:2512.21252 (2025) 2
- 13. Liu, X., Chen, J., Kao, S.h., Tai, Y.W., Tang, C.K.: Deceptive-nerf: Enhancing nerf reconstruction using pseudo-observations from diffusion models (2023) 4
- 14. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) 1
- 15. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5480–5490 (2022) 4

- 16. Ren, X., Shen, T., Huang, J., Ling, H., Lu, Y., Nimier-David, M., Müller, T., Keller, A., Fidler, S., Gao, J.: Gen3c: 3d-informed world-consistent video generation with precise camera control. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6121–6132 (2025) 2, 4
- 17. Truong, P., Rakotosaona, M.J., Manhardt, F., Tombari, F.: Sparf: Neural radiance fields from sparse and noisy poses. IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2023) 4
- 18. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,

- X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
- Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025) 6, 11

- 19. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025) 4, 5
- 20. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025) 4
- 21. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20697–20709 (2024) 4
- 22. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: π3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 (2025) 4, 5, 9, 11
- 23. Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z., Ling, H.: Difix3d+: Improving 3d reconstructions with single-step diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26024–26035 (2025) 2, 11, 13
- 24. Wu, R., Mildenhall, B., Henzler, P., Park, K., Gao, R., Watson, D., Srinivasan, P.P., Verbin, D., Barron, J.T., Poole, B., et al.: Reconfusion: 3d reconstruction with diffusion priors. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21551–21561 (2024) 2, 4
- 25. Yang, J., Pavone, M., Wang, Y.: Freenerf: Improving few-shot neural rendering with free frequency regularization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8254–8263 (2023) 4
- 26. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T.: Improved distribution matching distillation for fast image synthesis. In: NeurIPS (2024) 7, 11
- 27. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: CVPR (2024) 7
- 28. Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Context as memory: Scene-consistent interactive long video generation with memory retrieval. In: Proceedings of the SIGGRAPH Asia 2025 Conference Papers. pp. 1–11 (2025) 3
- 29. Yu, W., Xing, J., Yuan, L., Hu, W., Li, X., Huang, Z., Gao, X., Wong, T.T., Shan, Y., Tian, Y.: Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024) 2, 4, 6, 11, 13

###### 30. Yu, Z., Peng, S., Niemeyer, M., Sattler, T., Geiger, A.: Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems 35, 25018–25032 (2022) 4

