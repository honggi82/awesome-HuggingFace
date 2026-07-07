## Voxify3D: Pixel Art Meets Volumetric Rendering

[Figure 1]

Yi-Chuan Huang Jiewen Chan Hao-Jen Chien Yu-Lun Liu National Yang Ming Chiao Tung University

# arXiv:2512.07834v2[cs.CV]26Apr2026

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

(a)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

K-means Max-Min

Input

(c)

20x 30x 50x

[Figure 18]

Input (b) Input Ours

Mediun-Cut Input Ours

2 colors (d) (e)

S-A

Input 4 colors 12 colors

Figure 1. Stylized voxel art with controllable abstraction. Voxify3D converts 3D meshes into stylized voxel art using discrete color palettes, pixel art supervision, and voxel-based radiance fields. This teaser showcases the flexibility and quality of our method. (a) Diverse voxel art outputs across object types and use cases. (b) Comparison of different palette selection methods. (c) Control over the resolution of the voxel grid (20×, 30×, 50×) allows a balance of detail and abstraction. (d) The variation in color count (2, 4, 8) shows the impact of palette size on expressiveness. (e) Input-output comparisons on multiple objects demonstrate faithful voxel structure, semantic clarity, and voxel art aesthetics.

#### Abstract

palette-constrained Gumbel-Softmax quantization enabling differentiable optimization over discrete color spaces with controllable palette strategies. This integration addresses fundamental challenges: semantic preservation under extreme discretization, pixel-art aesthetics through volumetric rendering, and end-to-end discrete optimization. Experiments show superior performance (37.12 CLIP-IQA, 77.90% user preference) across diverse characters and controllable abstraction (2-8 colors, 20×-50× resolutions). Project page: https://yichuanh.github.io/Voxify-3D/

Voxel art is a distinctive stylization widely used in games and digital media, yet automated generation from 3D meshes remains challenging due to conflicting requirements of geometric abstraction, semantic preservation, and discrete color coherence. Existing methods either over-simplify geometry or fail to achieve the pixel-precise, palette-constrained aesthetics of voxel art. We introduce Voxify3D, a differentiable two-stage framework bridging 3D mesh optimization with 2D pixel art supervision. Our core innovation lies in the synergistic integration of three components: (1) orthographic pixel art supervision that eliminates perspective distortion for precise voxel-pixel alignment; (2) patch-based CLIP alignment that preserves semantics across discretization levels; (3)

#### 1. Introduction

Voxel art is a distinctive form of 3D digital artwork, characterized by its minimalist aesthetic and discrete volumet-

[Figure 19]

[Figure 20]

[Figure 21]

Input Instruct-N2N [30] Ours

[Figure 22]

[Figure 23]

[Figure 24]

Input Vox-E [82] Ours

[Figure 25]

[Figure 26]

[Figure 27]

Input Blender Ours

- Figure 2. Existing methods often miss key features in voxelization. While IN2N [30], Vox-E [82], and Blender (Geometry Nodes) generate outputs that are coarse, blurry, or semantically inconsistent, they frequently lose critical elements such as facial features. In contrast, our method preserves structural details and produces visually appealing voxel art with sharp abstraction.

ric structure. Despite its growing popularity in games and digital media, creating high-quality voxel art remains challenging, requiring significant artistic expertise and manual effort. While recent works have achieved promising results in 2D pixel art stylization [4, 18, 29, 101], these techniques do not trivially extend to 3D voxel art. Directly using 2D pixel art for 3D reconstruction faces fundamental obstacles: projection-induced misalignment, multi-view inconsistencies, and ambiguous color representations.

Current voxel art generation from 3D meshes is limited. Simple downsampling loses semantic features, yielding overly coarse outputs. Voxel-based neural radiance fields [7, 22, 86] target photorealistic rendering, not stylistic abstraction. Neural editing methods [9, 30, 92] struggle with clean, discrete representations. Procedural tools like Blender’s Geometry Nodes require extensive manual tuning and lack unified optimization for discrete color control and semantic preservation, both critical for voxel art aesthetics. As Fig. 2 shows, existing methods miss key features.

Voxel art generation poses three interrelated challenges that cannot be addressed by naively combining existing tech-

niques: (1) Alignment: Perspective projection causes pixelvoxel misalignment, producing blurry gradients during optimization. Prior neural stylization [55, 94] uses perspective rendering, unsuited for discrete art styles. (2) Semantic Preservation: As resolution decreases, critical features (facial details, limb articulation) collapse. Standard perceptual losses on full images fail to capture local semantic importance. (3) Discrete Optimization: Voxel art requires small palettes (2-8 colors), but gradient-based methods produce continuous values. Existing quantization [20] lacks differentiability or user-controllable palette extraction.

We present Voxify3D, a principled framework addressing these challenges through synergistic technical design. We bridge 3D optimization with 2D pixel art supervision via: (1) six-view orthographic rendering that eliminates perspective distortion for precise alignment; (2) patch-based CLIP loss adapted to preserve semantics across discretization levels; (3) palette-constrained Gumbel-Softmax enabling differentiable discrete optimization with flexible extraction strategies. This integration requires careful synchronization of rendering strategy, loss formulation, and quantization timing, not a simple combination.

Our two-stage pipeline first initializes coarse voxel geometry and color via neural volume rendering, then refines using orthographic pixel art supervision with semantic and discrete color constraints. Our technical contributions include:

- • Orthographic pixel art supervision. First framework to bridge 2D pixel art with 3D voxel optimization by eliminating perspective misalignment, enabling precise gradient flow for discrete stylization across six canonical views.
- • Resolution-adaptive semantic preservation. Patch-based CLIP formulation maintaining object identity under extreme discretization (20×-50×), addressing semantic collapse that standard perceptual losses fail to prevent.
- • Palette-constrained differentiable quantization. Endto-end optimization pipeline integrating Gumbel-Softmax with flexible palette extraction (4 strategies), temperature scheduling, and logit-based representation for controllable discrete color spaces (2-8 colors).

- 2. Related Work
- 3D Representations: From Pixels to Voxels. Pixel art generation evolved from interpolation [26] and contentaware downscaling [40, 43] to deep learning: paired [37] and unsupervised translation [29, 101], GANs [18, 83], diffusion [4], and vector methods [36, 38, 104]. For 3D, voxel-based methods accelerate neural fields [59, 65, 67] through explicit grids [7, 24, 70, 78, 81, 86, 107], differentiable voxelization [62], unified frameworks [100], hierarchical structures [79], sparse architectures [14], and compression [47, 111]. Multi-scale voxel representations [53], geometry-aware voxel features [89], tensorial decomposition [8], and MVS-based methods [85] further enhance

reconstruction quality. Voxels support geometry processing [17], storage [71], and simulation [61]. Recent feedforward generation achieves scale through structured latents [102], hierarchical diffusion reaching 1024³ [79], cascaded point clouds [110], transformers on voxelized shapes [68], and voxelized SDFs [48]. Unlike 2D stylization or 3D photorealism, we address discrete, palette-constrained voxel art by bridging pixel art supervision with volumetric optimization via orthographic alignment, extending voxel radiance fields [86] with palette quantization.

Stylization and Discrete Color Control. Neural 3D stylization progressed from score distillation [74] and CLIP guidance [66] to zero-shot transfer [55], painterly rendering [16, 23, 94, 113], high-resolution generation [12, 52], and local control [15, 19, 27, 57]. Gumbel-Softmax [39, 63, 84] enables discrete optimization in NAS [5, 54], VQ-VAE [87, 90], and neural fields [11, 58]. Score-based generative models [6] provide conditional generation through likelihood matching. Palette methods include 2D quantization [4], 3D color decomposition [44], material extraction [60], vector quantization [33], and interactive editing [45], with alternatives like VQGAN [20] and latent upsampling [64]. In contrast to continuous stylization and fixed codebooks, we integrate Gumbel-Softmax with user-controllable palette extraction (K-means, Max-Min, Median Cut, Simulated Annealing), synchronized scheduling, and logit-based representation for pixel-precise voxel art.

Multi-view Supervision and Semantic Preservation. Multi-view consistency uses RL refinement [103], view aggregation [85, 106], and latent diffusion [91]. Orthographic projection serves specialized domains: aerial orthophotos [13, 109], CAD reconstruction [114], and furniture assembly [32]. CLIP [77] enables semantic guidance [1, 10, 21, 42, 69, 72, 88, 93], with text supervision extending to semantic segmentation [98]. Semantic preservation under discretization uses masked autoencoders [50], context-aware transformers [108], semantic structures [51], geometry-aware downsampling [73], and hierarchical upsampling [80]. Unlike perspective stylization or orthographic reconstruction, we combine orthographic rendering with pixel art supervision, designing resolution-adaptive patchbased CLIP loss preventing semantic collapse at 20×-50× discretization where image-level losses fail.

Applications and Datasets. Mesh generation exploits diffusion and sparse views [3, 31, 35, 49, 52, 56, 95, 97, 102, 105], with character datasets [96, 99]. Game assets require structural decomposition [34], PBR materials [112], and procedural libraries [41, 76]. Fabrication includes LEGO generation [25, 75], Earth voxelization [46], and 3D printing [2]. These inform our evaluation but don’t address mesh-to-voxelart conversion with semantic fidelity, palette constraints, and controllable abstraction.

#### 3. Method

We propose a two-stage framework for converting 3D meshes into stylized voxel art with high fidelity and semantic consistency (Fig. 3). Stage 1 (Sec. 3.1) builds a coarse voxel radiance field using DVGO [86] to establish geometric and color foundations. Stage 2 (Sec. 3.2) refines the grid under orthographic pixel-art supervision, with CLIP-based loss (Sec. 3.3) for semantic alignment and depth loss for geometric preservation. To achieve clean abstraction and a coherent palette, we replace the RGB grid with a learned color-logit grid and apply Gumbel-Softmax for differentiable palette quantization (Sec. 3.4). This pipeline retains abstract details, enforces a dominant palette, and conveys the distinctive style of voxel art across resolutions.

##### 3.1. Coarse Voxel Grid Training

The first stage adapts DVGO [86] to build a coarse voxel representation. Unlike NeRFs using MLPs, DVGO directly optimizes two explicit voxel grids: a density grid d for spatial occupancy and a color grid c = (r,g,b) for appearance. This explicit structure enables faster training and efficient rendering.

We partition the object’s bounding box into a grid of resolution (W/cell size)3, where W is the canonical orthographic image width (pixels) and cell size is the number of pixels per voxel edge. Each voxel stores density d and RGB color c. The rendered color C(r) along a camera ray r is computed as:

 −

 ,

k−1

N

C(r) =

Tkαkck, Tk = exp

djδj

j=1

k=1

αk = 1 − exp(−dkδk), (1)

where N is the number of samples along the ray, dk the density, δk the distance between consecutive samples, Tk the accumulated transmittance, and αk the opacity at sample k.

The coarse voxel grid is optimized with:

Ltotal = Lrender + λdLdensity + λbLbg, (2)

where Lrender minimizes the MSE between rendered and target colors to ensure visual fidelity, Ldensity regularizes the density to suppress noise, prevent near-clip artifacts, and employs total variation (TV) regularization to enforce spatial smoothness, and Lbg uses entropy loss to maintain clear geometry and reduce background artifacts. This stage provides a good initialization for color and density.

##### 3.2. Orthographic Pixel Art Fine-tuning

To utilize the abstract features and clean edges of pixel art for 3D grid supervision, we fine-tune the voxel space by rendering orthographic projections from six axis-aligned views and

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

(a) Coarse voxel grid training (b) Orthographic pixel art fine-tuning

[Figure 32]

𝑵𝟑: voxel resolution

###### (a) Coarse voxel grid training (b) Orthographic pixel art fine-tuning

[Figure 33]

[Figure 34]

6 canonical views

⋯

𝑵𝟑

[Figure 35]

[Figure 36]

⋯

6 canonical views

### 𝑵𝟑

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

Pixel art Generator

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Pixel art Generator

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Persp. proj.

[Figure 64]

Ortho.

⋮ proj.

RGB to Logit

[Figure 65]

[Figure 66]

###### Ortho. proj.

[Figure 67]

[Figure 68]

⋮

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

⋯

⋯

[Figure 89]

🎨

[Figure 90]

###### MSE loss Depth loss CLIP loss Alpha loss

!

5

MSE Loss Depth loss CLIP Loss Alpha Loss

𝐶

Input Mesh

MSE loss

MSE loss

Voxel Art

Voxel Art

logits Color palette

logits

[Figure 91]

Color palette

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

#!,#,$ ~ #&'()*(0,1)

𝐺 , ,  ~ 𝐺𝑢𝑚𝑏𝑒𝑙(0,1)

Patch

[Figure 98]

5

Patch

𝐶

max

[Figure 99]

max

Input Mesh

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

Gumbel Noise Color Logit Noisy Logit

[Figure 100]

Random Logit Color Logit Noisy Logit

[Figure 101]

4&

[Figure 102]

[Figure 103]

[Figure 104]

𝑐

[Figure 105]

Softmax

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

Softmax

Clip Loss

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Clip Loss

[Figure 125]

[Figure 126]

[Figure 127]

!

𝜏

4&

3!,#,$(!)

color

𝑐

color

𝑠 , , (𝜏)

"!,#,$

0 = 2!,#,$

𝑌 , , 

𝜃 = 𝜆 , , 

Color weights

Render

5

Color weight

Render

𝐶

(d) Discrete Color Assignment via Gumbel Softmax

(c) CLIP-Guided Optimization

(c) CLIP-Guided Optimization (d) Discrete Color Assignment via Gumbel Softmax

- Figure 3. Our two-stage voxel art generation pipeline. (a) Coarse voxel grid training: Given a 3D mesh, we render multi-view images and optimize a voxel-based radiance field (DVGO [86]) using MSE loss to learn coarse RGB and density. (b) Orthographic pixel art fine-tuning: We refine the voxel grid using six orthographic pixel art views, which also serve to extract a discrete color palette (e.g., via k-means). Optimization includes appearance, depth, and alpha losses. (c) CLIP-guided optimization: A CLIP loss computed over rendered patches and mesh images encourages semantic alignment while being memory-efficient. (d) Differentiable discrete color selection via Gumbel-Softmax: Each voxel stores palette logits. Gumbel-Softmax enables differentiable sampling for end-to-end color optimization, yielding coherent, stylized voxel art.

[Figure 128]

| |
|---|

[Figure 129]

[Figure 130]

| |
|---|

Perspective projection Orthographic projection

left view

[Figure 131]

| |
|---|

left view misalignw/pixel art alignw/pixel art

[Figure 132]

[Figure 133]

zoom in

zoom in

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

Input mesh

|[Figure 137]|
|---|

6 views pixel art

[Figure 138]

[Figure 139]

- Figure 4. Perspective vs. Orthographic. (Left) Six-view pixel art pipeline. (Right) Perspective views (red) misalign pixels, while six orthographic views (green) enable precise pixel–voxel alignment.

Ldepth = ∥D(r) − Dgt∥1 , (4) where C(r) and D(r) are the rendered color and depth along ray r, Cpixel is the RGB color from the pixel-art supervision, and Dgt is the mesh-projected depth.

We also use an alpha loss to suppress density in background regions, enforcing background transparency to avoid floating density artifacts:

Lα = ∥Mα ⊙ α¯∥2 , (5)

where Mα ∈ {0,1}H×W is a binary mask from the pixel art alpha channel (1 for background), and α¯ denotes the accumulated ray opacity from volume rendering, which is encouraged to be 0 for background rays to allow full transparency. This encourages transparent regions in the pixel art to remain fully transmissive, preventing the formation of undesired voxels in areas without valid supervision.

comparing them against pixel art supervision generated by the pixel art generator [101]. This six-view setup compactly covers the major surfaces of the object, while orthographic rendering formulates parallel ray casting ri(t) = oi + td, where oi is the ray origin of pixel pi and d is the fixed ray direction. All rays are parallel, ensuring pixel-to-voxel alignment without perspective distortions (Fig. 4).

By leveraging pixel art as the supervision signal, each voxel grid more effectively captures and expresses the most important structural and appearance information.

##### 3.3. CLIP-based Semantic Loss

We apply two foundational losses to supervise geometry and structure:

To incorporate semantic supervision, we sample half of the total rays to form patches for computing a CLIP-based perceptual loss. During training, we randomly sample patch rays

Lpixel = ∥C(r) − Cpixel∥22 (3)

(opatch,dpatch) from rendered images Imesh of input mesh. Given the rendered patch Iˆpatch and the corresponding meshbased patch Ipatchmesh, we extract CLIP features [21, 77] and compute a perceptual loss via cosine similarity:

Lclip = 1 − cos CLIP(Iˆpatch), CLIP(Ipatchmesh) , (6)

where cosine similarity is defined as cos(a,b) = ∥a⟨a,b∥ ∥⟩b∥, and CLIP(·) denotes the CLIP image encoder output. This loss encourages voxel-rendered outputs to remain semantically aligned with the input mesh while supporting stylized abstraction, as illustrated in stage (c) of Fig. 3.

##### 3.4. Discrete Color Selection via Gumbel-Softmax

To generate clean and stylized voxel appearances while allowing flexible color selection strategies, we adopt a palettebased quantization scheme where each voxel selects a color from a predefined palette. This palette is extracted from the six-view pixel art images using a chosen clustering method before Gumbel-Softmax quantization.

Instead of regressing RGB values, each voxel (i,j,k) stores a color logit vector λi,j,k ∈ RC, where C is the number of discrete colors in the predefined palette.

During training, Gumbel noise Gi,j,k ∼ Gumbel(0,1) ∈ RC is added to produce noisy logits:

###### Yi,j,k = λi,j,k + Gi,j,k, (7)

where Yi,j,k,n denotes the noisy logit for the n-th palette color at voxel (i,j,k), with n ∈ {1,...,C}. A temperaturecontrolled softmax [39, 63] is then applied:

exp(Yi,j,k,n/τ)

, (8)

si,j,k,n(τ) =

C n′=1 exp(Yi,j,k,n′/τ)

where si,j,k,n(τ) is the probability of selecting the n-th color in the palette for voxel (i,j,k), and τ is the temperature parameter controlling distribution sharpness.

In early training, we use the soft distribution si,j,k directly. Later, we switch to the straight-through variant, where the forward pass uses a one-hot selection at arg maxn si,j,k, while gradients are backpropagated through the soft weights. We anneal the temperature τ during training to encourage smooth exploration in the early stages and sharper, more discrete selections later. The sampled RGB value is computed as:

C

RGBi,j,k =

si,j,k,n · cn, (9)

n=1

where cn ∈ R3 is the n-th color in the palette.

After training, we directly select the color with the highest logit:

RGBvoxeli,j,k = cargmax

λi,j,k,n, (10)

n

producing fully discrete voxel outputs. This process is illustrated in stage (d) of Fig. 3.

To enhance flexibility in stylization, this strategy allows users to choose the color selection method and number of colors, enabling explicit control over both color richness and overall style of the voxel art, making the design process more aligned with practical usage scenarios.

##### 3.5. Loss Summary and Training Procedure

The overall loss optimized during fine-tuning is a weighted sum of multiple components that jointly supervise pixel-art faithfulness, geometry consistency, semantic alignment, and spatial regularity:

Ltotal = λpixel·Lpixel+λdepth·Ldepth+λalpha·Lalpha+λclip·Lclip,

(11) where Lpixel, Ldepth, and Lclip encourage pixel-level accuracy, depth consistency, and semantic alignment, respectively, while Lalpha suppresses background opacity to yield clean silhouettes. In Stage 2, rays are split into two groups: (1) Lpixel, Ldepth, and Lalpha, and (2) Lclip on rendered patches, all computed via volumetric rendering (Eq. (1)). Thus, geometric supervision of the density grid is provided by Lpixel, Ldepth, and Lalpha, while semantic supervision comes from Lclip, which guides voxel appearance toward the intended pixel-art style.

#### 4. Experiments 4.1. Experimental Setup

Dataset. We evaluate our method on three mesh datasets: Rodin [96], Unique3D [99], and TRELLIS [102]. Rodin and Unique3D primarily feature character 3D assets with rich semantic details, making them ideal for evaluating voxel abstraction and stylized representation. We also evaluate on diverse categories including architecture and vehicles; see supplementary material for details.

Implementation details. Training follows a two-stage schedule: (a) Coarse Voxelization: optimize the voxel grid for 8000 iterations to capture global structure; (b) Pixel Art Supervision: fine-tune for 6500 iterations with MSE, Depth, and CLIP losses on six orthographic views, rendered at a resolution of 1200 × 1200, using fixed 80 × 80 patches randomly sampled each iteration for CLIP loss. In the final 2000 iterations, supervision is applied only to the front view to enhance key abstract features. Gumbel-Softmax sampling is performed over a fixed palette, with temperature τ annealed from 1.0 to 0.1.

Baseline methods. We compare against:

1. Pixel art to 3D extension: Render the input mesh into images, stylize them into pixel art, then train the original DVGO with these pixel-art images, using the coarse voxel grid as the final output.

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

| |
|---|

|25x|
|---|

| |
|---|

| |
|---|

| |
|---|

|40x|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 163]|
|---|

| |
|---|

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

| |
|---|

|40x|
|---|

|[Figure 185]|
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

|50x|
|---|

| |
|---|

| |
|---|

| |
|---|

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

[Figure 203]

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

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

|50x|
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 208]|
|---|

| |
|---|

[Figure 209]

|50x|
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

[Figure 223]

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

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

|50x|
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 230]|
|---|

| |
|---|

|50x|
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

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Input IN2N [Haque et al.

Pixel art to 3D extension

VoX-E [Sella et al. 2023] Blender Ours Input IN2N [Haque et al.

Pixel art to 3D extension

VoX-E [Sella et al. 2023] Blender Ours

2023]

2023]

- Figure 5. Qualitative comparisons on character models from the Rodin [96] dataset. We compare our voxel art results with Pixel art to 3D extension, IN2N [30], Vox-E [82], and Blender’s voxelization. Our method produces stylized yet consistent voxel representations with pixel art aesthetics.

Table 1. Average CLIP-IQA scores over all 35 examples. Best scores are highlighted.

- 2. IN2N [30]: Language-guided mesh editing with viewconsistent 3D stylization.
- 3. Vox-E [82]: Language-to-voxel generation prioritizing semantics over fine geometry.
- 4. Blender Geometry Nodes: Procedural mesh-to-voxel conversion, fast but without semantic or stylization control.

Method Pixel IN2N Vox-E Blender Ours CLIP-IQA 35.53 23.93 35.02 36.31 37.12

##### 4.3. Quantitative Comparisons

##### 4.2. Qualitative Comparisons

To assess stylization fidelity and semantic preservation, we adopt the CLIP-IQA framework. For each character, we use GPT-4 to generate a detailed textual description based on the original mesh images, prepended with “A voxel art of...” (e.g., “A voxel art of a pink teddy bear with a red bow and heart-shaped feet”). We use OpenAI’s ViT-B/32 CLIP model and compute the average cosine similarity between each prompt and the rendered images from different methods.

We qualitatively compare our method with Pixel art to 3D extension, IN2N, Vox-E, and Blender on eight character meshes from the evaluation datasets (Fig. 5), with an additional eight groups of comparisons provided in the supplementary material.

IN2N preserves coarse structure but suffers from large variations across different guidance images, often failing to produce consistent voxelized results; Vox-E yields smoother volumes yet misses the discrete, blocky style of voxel art; Blender produces clean abstraction through procedural voxelization, which is akin to simple downsampling, but requires manual tuning and lacks semantic alignment.

As shown in Table 1, the reported CLIP-IQA scores are averaged over all 35 cases. Our method consistently achieves the highest score, demonstrating superior semantic alignment and stylized abstraction across a diverse set of character meshes.

Our method preserves key cues (e.g., ears, eyes) with sharp edges across 25×–50× resolutions, achieving both expressive stylization and semantic fidelity. Additional results are provided in the supplementary material.

##### 4.4. Ablation Study

We analyze the contribution of each component by removing one module at a time. Qualitative results are shown in Fig. 6,

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

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

|| |
|---|
|
|---|

|| |
|---|
|
|---|

| |
|---|

| |
|---|

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

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

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

|ours|
|---|

|ours|
|---|

|ours|
|---|

|ours|
|---|

|ours|
|---|

|ours|
|---|

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

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

|| |
|---|
|
|---|

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

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

|ours|
|---|

|[Figure 302]<br><br>ours|
|---|

|[Figure 303]<br><br>ours|
|---|

|ours|
|---|

|ours|
|---|

|ours|
|---|

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

(1) w/o Stage 1 (4) w/o depth loss (5) w/o Clip loss (6) w/o Gumbel

Input

(2) w/o Stage 2 (7) Ours

(3) w/o ortho. project

- Figure 6. Ablation study on key components. We evaluate the impact of each module by removing it individually: (1) w/o Stage 1, (2) w/o Stage 2, (3) w/o orthographic projection (replaced with perspective projection), (4) w/o depth loss, (5) w/o CLIP loss, and (6) w/o Gumbel Softmax (resulting in continuous and unconstrained colors). Each row shows a different input, and zoom-in regions highlight local differences. Removing each component leads to noticeable degradation in geometry, color consistency, or semantic fidelity, while the full model produces coherent and stable voxel stylization.

3 colors

[Figure 310]

[Figure 311]

Max-Min

Median Cut

Simulated Annealing

K-means

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

| |
|---|

Input 2 colors 4 colors 8 colors

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

- Figure 7. Effect of Palette Selection and Color Count. Each row corresponds to a different palette extraction method: K-means, Max-Min, Median Cut, and Simulated Annealing. Each column shows increasing color counts (2, 3, 4, 8). Each method produces unique color clustering effects.

Table 2. Ablation study of model components (CLIP-IQA). We evaluate the impact of each component by removing it individually. Our full model achieves the best performance. Removing stage design or orthographic projection leads to significant degradation, while depth loss, CLIP loss, and Gumbel Softmax contribute to consistent improvements in semantic alignment.

Method Stage 1 Stage 2 Ortho. Proj. Depth loss CLIP loss Gumbel CLIP-IQA

- (1) w/o Stage 1 ✓ ✓ ✓ ✓ ✓ 28.42

- (2) w/o Stage 2 ✓ 34.32

- (3) w/o ortho. proj. ✓ ✓ ✓ ✓ ✓ 27.38

- (4) w/o depth loss ✓ ✓ ✓ ✓ ✓ 39.75

- (5) w/o CLIP loss ✓ ✓ ✓ ✓ ✓ 39.23

- (6) w/o Gumbel ✓ ✓ ✓ ✓ ✓ 39.31

- (7) Ours ✓ ✓ ✓ ✓ ✓ ✓ 40.06

that Stage 1 provides geometric stability, while Stage 2 is the key component that enables semantic abstraction and stylized voxel representation.

Orthographic projection. Replacing orthographic projection with perspective projection causes severe color misalignment, as pixel colors no longer correspond to voxel locations, resulting in inconsistent appearance (Fig. 6).

Depth loss. Without depth loss, the model produces plausible views individually but fails to preserve global 3D structure, leading to geometric distortions.

CLIP loss. Removing CLIP loss reduces semantic clarity and leads to less coherent color regions, highlighting the importance of semantic alignment.

and quantitative results are reported in Table 2.

Stage-wise optimization. Without Stage 1, the model lacks proper geometric initialization, leading to distorted shapes. Without Stage 2, the result degenerates to a coarse DVGO grid without abstraction or semantic refinement. This shows

Gumbel-Softmax. Without Gumbel-Softmax, colors become mixed and lack clear boundaries, failing to produce clean and discrete voxel-style color patterns.

Summary. Fig. 6 and Table 2 show that all components contribute to geometry, semantic alignment, and discrete color

[Figure 327]

- Figure 8. Fabrication: LEGO render. Rendered using KeyShot 2023. Our method extends to LEGO applications, where achieving rich visual results within the limited color palette is crucial for practical fabrication.

Table 3. User studies. (a) 35 examples (72 participants). (b) Color quantization (10 art-trained).

peal. In the second part (geometry evaluation), participants compared grayscale renderings to assess shape preservation.

As shown in Table 3 (a), our method is consistently preferred, achieving 77.90%, 80.36%, and 96.55% in abstractness, appeal, and geometry, respectively.

(b) Color quantization preference (%)

(a) Image quality (user votes, %)

Metric Abstract Appeal Geometry Ours 77.90 80.36 96.55 Others 22.10 19.64 3.45

w/o Gumbel w/ Gumbel Preferred 11.11 88.89

Expert Study on Color Preference. We further conduct a focused evaluation on color quantization with 10 art-trained participants. As shown in Table 3 (b), 88.89% of participants prefer the results with Gumbel-Softmax, demonstrating its effectiveness in producing clearer structures and more distinct color regions.

representation. Our full model achieves the best performance (40.06 CLIP-IQA). Removing any component consistently degrades quality. The quantitative results in Table 2 are averaged over five objects.

More details on study design, questions, and additional results are provided in the supplementary material.

Palette controllability. We analyze palette design by varying the number of colors (2, 3, 4, 8) and extraction methods (K-means, K-means with rare color boosting, Median Cut, Max-Min, and Simulated Annealing).

#### 5. Conclusion

We introduce Voxify3D, a novel framework for transforming 3D meshes into stylized voxel art with strong semantic abstraction and structural consistency. By combining coarse voxel optimization, orthographic pixel art supervision, and palette-based color quantization, our method achieves expressive and visually appealing results across a variety of character assets. Extensive experiments and user studies confirm its advantages over existing baselines in both geometric faithfulness and artistic stylization.

The palette is extracted from all pixels across the six input pixel-art views to form a shared discrete color set for voxel optimization.

As shown in Fig. 7, increasing palette size improves color richness, while different strategies affect color distribution. K-means favors dominant tones, while the rare-color variant preserves infrequent details, and other methods (Median Cut, Max-Min, Simulated Annealing) promote balanced or diverse palettes.

In addition to digital results, we demonstrate the fabrication potential of our voxel outputs via LEGO-style assemblies (Fig. 8), enabled by their discrete structure and limited color palette.

Smaller palettes lead to stronger abstraction, whereas larger palettes improve detail. Our method maintains consistent voxel structure while enabling flexible control over color abstraction. Additional results and detailed palette selection strategies are provided in the supplementary material.

Limitations and Future Work. Voxify3D struggles with highly intricate shapes, where thin structures or fine facial details may be lost at low voxel resolutions. Future work may explore integrating geometric priors or training strategies to enhance detail preservation and scalability, as well as adopting assembly-aware fabrication strategies inspired by LEGO brick design and connection principles to improve the physical realizability of voxel-based models.

##### 4.5. User Study

We conducted a user study with 72 participants to evaluate our method against four baselines: Pixel Art to 3D extension, IN2N [30], Vox-E [82], and Blender Geometry Nodes.

The study consists of two parts. In the first part (35 examples), participants evaluated abstract detail and visual ap-

Acknowledgements. This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628E-A49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

#### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [2] Christoph Bader, Dominik Kolb, James C Weaver, Sunanda Sharma, Ahmed Hosny, Jo˜ao Costa, and Neri Oxman. Making data matter: Voxel printing for the digital fabrication of data across scales and domains. Science advances, 4(5): eaas8652, 2018. 3
- [3] Maciej Bala, Yin Cui, Yifan Ding, Yunhao Ge, Zekun Hao, Jon Hasselgren, Jacob Huffman, Jingyi Jin, J.P. Lewis, Zhaoshuo Li, Chen-Hsuan Lin, Yen-Chen Lin, Tsung-Yi Lin, Ming-Yu Liu, Alice Luo, Qianli Ma, Jacob Munkberg, Stella Shi, Fangyin Wei, Donglai Xiang, Jiashu Xu, Xiaohui Zeng, and Qinsheng Zhang. Edify 3d: Scalable high-quality 3d asset generation. arXiv preprint arXiv:2411.07135, 2024. 3
- [4] Alexandre Binninger and Olga Sorkine-Hornung. Sd-πxl: Generating low-resolution quantized imagery via score distillation. In SIGGRAPH Asia Conference Papers, pages 1–12, 2024. 2, 3
- [5] Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations (ICLR), 2019. 3
- [6] Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Yi-Chen Lo, Chia-Che Chang, Yu-Lun Liu, Yu-Lin Chang, ChiaPing Chen, and Chun-Yi Lee. Denoising likelihood score matching for conditional score-based data generation. arXiv preprint arXiv:2203.14206, 2022. 3
- [7] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pages 333–350. Springer,

2022. 2

- [8] Bo-Yu Chen, Wei-Chen Chiu, and Yu-Lun Liu. Improving robustness for joint optimization of camera pose and decomposed low-rank tensorial radiance fields. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 990–1000, 2024. 2
- [9] Jun-Kun Chen, Jipeng Lyu, and Yu-Xiong Wang. Neuraleditor: Editing neural radiance fields via manipulating point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12300–12310, 2023. 2
- [10] Lianggangxu Chen, Xuejiao Wang, Jiale Lu, Shaohui Lin, Changbo Wang, and Gaoqi He. Clip-driven open-vocabulary 3d scene graph generation via cross-modality contrastive

- learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27863– 27873, 2024. 3
- [11] Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, and Wenwu Zhu. Q-dit: Accurate post-training quantization for diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28306–28315, 2025. 3
- [12] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22246–22256, 2023. 3
- [13] Shihan Chen, Qingsong Yan, Yingjie Qu, Wang Gao, Junxing Yang, and Fei Deng. Ortho-nerf: generating a true digital orthophoto map using the neural radiance field from unmanned aerial vehicle images. Geomatics, Natural Hazards and Risk, 15(1):741–760, 2024. 3
- [14] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21674–21683, 2023. 2
- [15] Yingshu Chen, Huajian Huang, Tuan-Anh Vu, Ka Chun Shum, and Sai-Kit Yeung. Stylecity: Large-scale 3d urban scenes stylization. In European conference on computer vision, pages 395–413. Springer, 2024. 3
- [16] SeungJeh Chung, JooHyun Park, and HyeongYeop Kang. 3dstyleglip: Part-tailored text-guided 3d neural stylization. arXiv preprint arXiv:2404.02634, 2024. 3
- [17] David Coeurjolly, Pierre Gueth, and Jacques-Olivier Lachaud. Regularization of voxel art. In ACM SIGGRAPH 2018 Talks, pages 1–2. 2018. 3
- [18] Fl´avio Coutinho and Luiz Chaimowicz. Generating pixel art character sprites using gans. arXiv preprint arXiv:2208.06413, 2022. 2
- [19] Dale Decatur, Itai Lang, Kfir Aberman, and Rana Hanocka. 3d paintbrush: Local stylization of 3d shapes with cascaded score distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4473–4483, 2024. 3
- [20] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021. 2, 3
- [21] Kevin Frans, Lisa Soros, and Olaf Witkowski. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 3, 5
- [22] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501–5510, 2022. 2
- [23] Haruo Fujiwara, Yusuke Mukuta, and Tatsuya Harada. Stylenerf2nerf: 3d style transfer from style-aligned multi-view

images. In SIGGRAPH Asia 2024 Conference Papers, pages

- 1–10, 2024. 3

- [24] Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

2

- [25] Jiahao Ge, Mingjun Zhou, and Chi-wing Fu. Learn to create simple lego micro buildings. ACM Transactions on Graphics (TOG), 43(6):1–13, 2024. 3
- [26] Timothy Gerstner, Doug DeCarlo, Marc Alexa, Adam Finkelstein, Yotam Gingold, and Andrew Nealen. Pixelated image abstraction with integrated user constraints. Computers & Graphics, 37(5):333–347, 2013. 2
- [27] Guilherme Gomes Haetinger, Jingwei Tang, Raphael Ortiz, Paul Kanyuk, and Vinicius Azevedo. Controllable neural style transfer for dynamic meshes. In Acm siggraph 2024 conference papers, pages 1–11, 2024. 3
- [28] Google DeepMind. Gemini models: Product overview. https : / / deepmind . google / technologies / gemini/, 2025. Accessed: November 21, 2025. 14, 17, 19
- [29] Chu Han, Qiang Wen, Shengfeng He, Qianshu Zhu, Yinjie Tan, Guoqiang Han, and Tien-Tsin Wong. Deep unsupervised pixelization. ACM Transactions on Graphics (TOG), 37(6):243:1–243:11, 2018. 2
- [30] Ayaan Haque, Matthew Tancik, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8874–8884, 2023. 2, 6, 8, 16
- [31] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 3
- [32] Wentao Hu, Jia Zheng, Zixin Zhang, Xiaojun Yuan, Jian Yin, and Zihan Zhou. Plankassembly: Robust 3d reconstruction from three orthographic views with learnt shape programs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18495–18505, 2023. 3
- [33] Siyu Huang, Jie An, Donglai Wei, Jiebo Luo, and Hanspeter Pfister. Quantart: Quantizing image style transfer towards high visual fidelity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5947–5956, 2023. 3
- [34] Yuhang Huang, Shilong Zou, Xinwang Liu, and Kai Xu. Part-aware shape generation with latent 3d diffusion of neural voxel fields. IEEE Transactions on Visualization and Computer Graphics, 2025. 3
- [35] Zixuan Huang, Mark Boss, Aaryaman Vasishta, James M Rehg, and Varun Jampani. Spar3d: Stable point-aware reconstruction of 3d objects from single images. arXiv preprint arXiv:2501.04689, 2025. 3
- [36] Yuki Igarashi and Takeo Igarashi. Pixel art adaptation for handicraft fabrication. Computer Graphics Forum (Pacific Graphics 2022), 41(7):489–494, 2022. 2
- [37] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pages 1125–1134, 2017. 2

- [38] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [39] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In Proceedings of the 5th International Conference on Learning Representations (ICLR), 2017. 3, 5
- [40] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. arXiv preprint arXiv:1603.08155, 2016. 2
- [41] Bumsoo Kim, Sanghyun Byun, Yonghoon Jung, Wonseop Shin, Sareer UI Amin, and Sanghyun Seo. Minecraftify: Minecraft style image generation with text-guided image editing for in-game application. arXiv preprint arXiv:2402.05448, 2024. 3
- [42] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [43] Johannes Kopf, Ariel Shamir, and Pieter Peers. Contentadaptive image downscaling. ACM Transactions on Graphics (TOG), 32(6):1–8, 2013. 2
- [44] Zhengfei Kuang, Fujun Luan, Sai Bi, Zhixin Shu, Gordon Wetzstein, and Kalyan Sunkavalli. Palettenerf: Palette-based appearance editing of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20691–20700, 2023. 3
- [45] Jae-Hyeok Lee and Dae-Shik Kim. Ice-nerf: Interactive color editing of nerfs via decomposition-aware weight optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3491–3501, 2023. 3
- [46] Ryan Hardesty Lewis. Voxelizing google earth: A pipeline for new virtual worlds. In ACM SIGGRAPH 2024 Labs, pages 1–2. 2024. 3
- [47] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Liefeng Bo. Compressing volumetric radiance fields to 1 mb. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4222–4231, 2023. 2
- [48] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusionsdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12642–12651, 2023. 3
- [49] Ming-Feng Li, Yueh-Feng Ku, Hong-Xuan Yen, Chi Liu, Yu-Lun Liu, Albert YC Chen, Cheng-Hao Kuo, and Min Sun. Genrc: Generative 3d room completion from sparse image collections. In European Conference on Computer Vision, pages 146–163. Springer, 2024. 3
- [50] Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M Alvarez, Sanja Fidler, Chen Feng, and Anima Anandkumar. Voxformer: Sparse voxel transformer for camerabased 3d semantic scene completion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9087–9098, 2023. 3
- [51] Yuan Li, Zhihao Liu, Bedrich Benes, Xiaopeng Zhang, and Jianwei Guo. Svdtree: Semantic voxel diffusion for single

- image tree reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4692–4702, 2024. 3
- [52] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 3
- [53] Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, and Yu-Lun Liu. Frugalnerf: Fast convergence for extreme few-shot novel view synthesis without learned priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11227–11238, 2025. 2
- [54] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. In International Conference on Learning Representations (ICLR), 2019. 3
- [55] Kunhao Liu, Fangneng Zhan, Yiwen Chen, Jiahui Zhang, Yingchen Yu, Abdulmotaleb El Saddik, Shijian Lu, and Eric P Xing. Stylerf: Zero-shot 3d style transfer of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8338– 8348, 2023. 2, 3
- [56] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36, 2024. 3
- [57] Richard Liu, Daniel Fu, Noah Tan, Itai Lang, and Rana Hanocka. Wir3d: Visually-informed and geometry-aware 3d shape abstraction. arXiv preprint arXiv:2505.04813, 2025. 3
- [58] Weihang Liu, Xue Xian Zheng, Jingyi Yu, and Xin Lou. Content-aware radiance fields: Aligning model complexity with scene intricacy through learned bitwidth quantization. In European Conference on Computer Vision (ECCV), pages 239–256. Springer, 2024. 3
- [59] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13–23, 2023. 2
- [60] Ivan Lopes, Fabio Pizzati, and Raoul de Charette. Material palette: Extraction of materials from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4379–4388, 2024. 3
- [61] Frank Losasso, Fr´ed´eric Gibou, and Ronald Fedkiw. Simulating water and smoke with an octree data structure. ACM Transactions on Graphics (TOG), 23(3):457–462, 2004. 3
- [62] Yihao Luo, Yikai Wang, Zhengrui Xiang, Yuliang Xiu, Guang Yang, and ChoonHwai Yap. Differentiable voxelization and mesh morphing. arXiv preprint arXiv:2407.11272,

2024. 2

- [63] Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete

- random variables. In Proceedings of the 5th International Conference on Learning Representations (ICLR), 2017. 3, 5
- [64] Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin. Pulse: Self-supervised photo upsampling via latent space exploration of generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3
- [65] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16539–16548, 2023. 2
- [66] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13492– 13502, 2022. 3
- [67] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [68] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d: Exploring plain diffusion transformers for 3d shape generation. Advances in neural information processing systems, 36:67960–67971,

2023. 3

- [69] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. In European Conference on Computer Vision (ECCV), pages 531–547. Springer, 2022. 3
- [70] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 2
- [71] Ken Museth. Vdb: High-resolution sparse volumes with dynamic topology. ACM Transactions on Graphics (TOG), 32(3):27:1–27:22, 2013. 3
- [72] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. arXiv preprint arXiv:2103.17249, 2021. 3
- [73] Sai Karthikey Pentapati, Anshul Rai, Arkady Ten, Chaitanya Atluru, and Alan Bovik. Geoscaler: Geometry and renderingaware downsampling of 3d mesh textures. In 2025 IEEE International Conference on Image Processing (ICIP), pages 1007–1012. IEEE, 2025. 3
- [74] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3
- [75] Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, and Jun-Yan Zhu. Generating physically stable and buildable lego designs from text. arXiv preprint arXiv:2505.05469, 2025. 3
- [76] Adarsh Pyarelal, Aditya Banerjee, and Kobus Barnard. Modular procedural generation for voxel maps. In AAAI Fall Symposium, pages 85–101. Springer, 2021. 3

- [77] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 5
- [78] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

- 2

[79] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale

- 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3

- [80] Xuanchi Ren, Yifan Lu, Hanxue Liang, Zhangjie Wu, Huan Ling, Mike Chen, Sanja Fidler, Francis Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. Advances in Neural Information Processing Systems, 37:97670–97698, 2024. 3
- [81] Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao, and Andreas Geiger. Voxgraf: Fast 3d-aware image synthesis with sparse voxel grids. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2
- [82] Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. Vox-e: Text-guided voxel editing of 3d objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2, 6, 8, 16
- [83] Ygor Rebouc¸as Serpa and Maria Andr´eia Formico Rodrigues. Towards machine-learning assisted asset generation for games: A study on pixel art sprite sheets. In 2019 18th Brazilian Symposium on Computer Games and Digital Entertainment (SBGames), pages 219–228. IEEE, 2019. 2
- [84] Rushi Shah, Mingyuan Yan, Michael Curtis Mozer, and Dianbo Liu. Improving discrete optimisation via decoupled straight-through estimator. arXiv preprint arXiv:2410.13331,

2024. 3

- [85] Chih-Hai Su, Chih-Yao Hu, Shr-Ruei Tsai, Jie-Ying Lee, Chin-Yang Lin, and Yu-Lun Liu. Boostmvsnerfs: Boosting mvs-based nerfs to generalizable view synthesis in largescale scenes. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3
- [86] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5459– 5469, 2022. 2, 3, 4, 14
- [87] Yuhta Takida, Yukara Ikemiya, Takashi Shibuya, Kazuki Shimada, Woosung Choi, Chieh-Hsin Lai, Naoki Murata, Toshimitsu Uesaka, Kengo Uchida, Wei-Hsiang Liao, et al. Hq-vae: Hierarchical discrete representation learning with variational bayes. arXiv preprint arXiv:2401.00365, 2023. 3
- [88] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In International Conference on Learning Representations (ICLR), 2024. 3

- [89] Tao Tu, Shun-Po Chuang, Yu-Lun Liu, Cheng Sun, Ke Zhang, Donna Roy, Cheng-Hao Kuo, and Min Sun. Imgeonet: Image-induced geometry-aware voxel representation for multi-view 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6996–7007, 2023. 2
- [90] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 3
- [91] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2024. 3
- [92] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern

- Recognition (CVPR), 2022. 2

[93] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern

- Recognition (CVPR), 2022. 3

- [94] Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Nerf-art: Text-driven neural radiance fields stylization. IEEE Transactions on Visualization and Computer Graphics, 30(8):4983–4996, 2023. 2, 3
- [95] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), pages 52–67,

2018. 3

- [96] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, and Baining Guo. Rodin: A generative model for sculpting 3d digital avatars using diffusion. arXiv preprint arXiv:2212.06135, 2022. 3, 5, 6, 14, 17, 19
- [97] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [98] Ji-Jia Wu, Andy Chia-Hao Chang, Chieh-Yu Chuang, Chun-Pei Chen, Yu-Lun Liu, Min-Hung Chen, Hou-Ning Hu, Yung-Yu Chuang, and Yen-Yu Lin. Image-text codecomposition for text-supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26794–26803, 2024. 3
- [99] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024. 3, 5
- [100] Shuang Wu, Songlin Tang, Guangming Lu, Jianzhuang Liu, and Wenjie Pei. Univoxel: Fast inverse rendering by unified

- voxelization of scene representation. In European Conference on Computer Vision, pages 360–376. Springer, 2024. 2
- [101] Zongwei Wu, Liangyu Chai, Nanxuan Zhao, Bailin Deng, Yongtuo Liu, Qiang Wen, Junle Wang, and Shengfeng He. Make your own sprites: Aliasing-aware and cell-controllable pixelization. ACM Transactions on Graphics (TOG), 41(6): 1–16, 2022. 2, 4, 14
- [102] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 3, 5
- [103] Desai Xie, Jiahao Li, Hao Tan, Xin Sun, Zhixin Shu, Yi Zhou, Sai Bi, S¨oren Pirk, and Arie E Kaufman. Carve3d: Improving multi-view reconstruction consistency for diffusion models with rl finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6369–6379, 2024. 3
- [104] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. Accepted by CVPR 2024; arXiv preprint arXiv:2312.16476. 2
- [105] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 3

- [106] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. Consistnet: Enforcing 3d consistency for multiview images diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7079–7088, 2024. 3
- [107] Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [108] Zhu Yu, Runmin Zhang, Jiacheng Ying, Junchen Yu, Xiaohai Hu, Lun Luo, Si-Yuan Cao, and Hui-Liang Shen. Context and geometry aware voxel transformer for semantic scene completion. Advances in Neural Information Processing Systems, 37:1531–1555, 2024. 3
- [109] Dongdong Yue, Xinyi Liu, Yi Wan, Yongjun Zhang, Maoteng Zheng, Weiwei Fan, and Jiachen Zhong. Nerfortho: Orthographic projection images generation based on neural radiance fields. International Journal of Applied Earth Observation and Geoinformation, 136:104378, 2025. 3
- [110] LAN Yushi, Shangchen Zhou, Zhaoyang Lyu, Fangzhou Hong, Shuai Yang, Bo Dai, Xingang Pan, and Chen Change Loy. Gaussiananything: Interactive point cloud flow matching for 3d generation. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [111] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng.

- Cat-3dgs: A context-adaptive triplane approach to ratedistortion-optimized 3dgs compression. arXiv preprint arXiv:2503.00357, 2025. 2
- [112] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 3
- [113] Yuechen Zhang, Zexin He, Jinbo Xing, Xufeng Yao, and Jiaya Jia. Ref-npr: Reference-based non-photorealistic radiance fields for controllable scene stylization. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 4242–4251, 2023. 3
- [114] Zheng Zhou, Zhe Li, Bo Yu, Lina Hu, Liang Dong, Zijian Yang, Xiaoli Liu, Ning Xu, Ziwei Wang, Yonghao Dang, et al. Gaussiancad: Robust self-supervised cad reconstruction from three orthographic views using 3d gaussian splatting. arXiv preprint arXiv:2503.05161, 2025. 3

#### A. Overview

This supplementary material provides additional details that complement our main paper. We include implementation details (Sec. B) covering our codebase and training architecture, pixel art generator, logit grid initialization, parameter settings, loss design, temperature annealing schedule, crossview inconsistency handling, and palette selection strategies. We also provide experimental information (Sec. C) including our CLIP-IQA evaluation protocol, user study details, expert study on color preference, and run time analysis. Additionally, we present additional qualitative results (Sec. D) with more comparisons against baselines, results with varying palette settings, and results under different voxel sizes. We also provide comparisons with recent voxel art generation methods, including Gemini 3 [28] and Rodin [96]. Finally, we show failure cases and analyze potential future directions (Sec. E).

#### B. Implementation Details

Codebase and training architecture. Our implementation builds on DVGO [86]. We adopt a two-stage training pipeline. In Stage 1, we follow DVGO to train a coarse voxel grid, which initializes both color and density representations. In Stage 2, the input consists of six orthographic views stylized into pixel art. Using orthographic projection, each pixel from the pixel art is directly aligned with the voxel grid, ensuring per-pixel to voxel correspondence. After 4500 iterations, training is restricted to the front view, which typically contains the most salient semantic features (e.g., facial structures), allowing the model to refine key abstract details while maintaining consistency from the earlier multi-view supervision.

###### Pixel art generator.

Our pipeline requires stylized pixel art inputs rather than simple low-resolution downsampling.

We adopt the MYOS [101] generator to transform mesh renderings into high-quality pixel art, which preserves sharp boundaries and stylized abstractions. As illustrated in Fig. 9, na¨ıve downsampling produces blurry textures, while MYOS yields pixelated structures with clear edges, better aligned with voxel abstraction.

Logit grid initialization. In Stage 2, we initialize each voxel’s logit vector by the negative distance between its Stage 1 RGB color and the palette entries. This provides a stable bias toward closer colors and converges better than random initialization.

Parameter settings. We summarize the key training parameters for Stage 1 (voxel grid initialization) and Stage 2 (logit grid optimization).

Loss design. We adopt different objectives across the two training stages.

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

25x

|[Figure 331]|
|---|

|[Figure 332]|
|---|

60x

Input

Down sampling Pixel Art

Figure 9. Downsample vs. Pixel Art

Stage 1 (Coarse voxelization). The voxel grid is optimized with MSE reconstruction loss, regularized by density and background terms:

Ltotal = Lrender + λdLdensity + λbLbg,

where Lrender is MSE between rendered and target colors, Ldensity applies density regularization and total variation smoothing, and Lbg uses entropy to suppress background noise. This stage provides a stable initialization for both color and density.

Stage 2 (Pixel-art supervision). The fine-tuning objective combines pixel accuracy, geometry regularization, semantic alignment, and silhouette clarity:

Ltotal = λpixelLpixel + λdepthLdepth + λalphaLalpha + λclipLclip. Implementation details. Lpixel (MSE) is up-weighted to ensure faithful color abstraction. Ldepth is scaled by voxel resolution: 20 normally, and increased to 30 after step 4500. Lalpha enforces clean silhouettes via transparency regularization. Lclip is applied until step 6000, using 80×80 patches per iteration for semantic alignment. After step 6000, optimization focuses mainly on background transparency (Lalpha), while CLIP loss is disabled. This scheduling ensures early semantic guidance, followed by refinement of geometry and silhouettes. The detailed training parameters are included in Tab. 4, and the specific loss weights are detailed in Tab. 5.

Temperature annealing schedule. We apply a step-wise annealing schedule for the Gumbel-Softmax temperature τ, gradually lowering it to encourage sharper palette selection as training progresses. The temperature starts high to allow exploration of multiple colors, and progressively decreases to enforce deterministic palette assignments toward convergence. The complete annealing schedule is shown in Tab. 6.

Cross-view inconsistency. Supervision from six orthographic views keeps inconsistencies minimal, mostly near

Table 4. Training parameters for Stage 1 (left) and Stage 2 (right).

Parameter Value Iterations (Niters) 8000 Batch size (Nrand) 8192 Learning rate (density grid) 1 × 10−1 Learning rate (color grid k0) 1 × 10−1 LR decay step 20

Parameter Value Iterations (Niters) 6500 Batch size (Nrand) 8192 Learning rate (density grid) 5 × 10−3 Learning rate (logit grid) 1 × 10−1 LR decay step 20

Table 5. Loss weights used in our implementation.

λpixel λdepth λalpha λclip λb λd ×10

[Figure 333]

[Figure 334]

10 / 20 (30 after 4500 iter) ×20 ×1 (until 6000 iter)

×0 default (Stage 1)

×0.5 (Stage 1)

[Figure 335]

boundaries. To further refine salient cues, the last 2000 iterations are trained only on the front view (rich in facial details), reinforcing key features while preserving global consistency from earlier multi-view supervision.

30x

[Figure 336]

[Figure 337]

Palette selection strategies. We explored multiple strategies for extracting compact color palettes from input images:

[Figure 338]

40x

- • K-means clustering: baseline method that partitions pixels into C clusters and uses centroids as representative colors.
- • K-means with rare color boosting: explicitly incorporates infrequent colors to prevent palette collapse into dominant tones.
- • Median cut: recursively splits the RGB space by channel ranges to ensure balanced coverage of color distributions.
- • Max–min picking: iteratively selects farthest colors in feature space to maximize palette diversity.
- • Simulated annealing: formulates palette extraction as a discrete optimization problem, refining palettes via stochastic search.

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

50x

[Figure 343]

[Figure 344]

50x

#### C. Experimental Information

CLIP-IQA evaluation protocol. We evaluate stylization fidelity and semantic preservation using CLIP-IQA. Text prompts (“A voxel art of...”) are generated by GPT-4 from input mesh images, and cosine similarity is computed between the prompts and rendered results using CLIP (ViT-B/32), averaged over 35 cases. While training employs CLIP loss in an image–image setting, evaluation is conducted with text prompts, ensuring that CLIP-IQA reflects semantic fidelity rather than overfitting to the training objective.

Input Blender Ours

Figure 10. Greyscale examples.

We conducted a user study with 72 participants, who were presented with 35 colored voxel art examples and 4 grayscale voxel renderings Fig. 10. The interface is illustrated in Fig. 11.

Furthermore, we report CLIP-IQA scores across multiple voxel resolutions (Tab. 7), demonstrating that our method consistently maintains semantic alignment under different discretization levels.

Each colored example was accompanied by the following two questions:

User study details.

• Abstract detail: “Which voxel art version most clearly

Table 6. Step-wise annealing schedule of the Gumbel-Softmax temperature τ during Stage 2.

< 1000 1000–2999 3000–3999 4000–4999 5000–6000 > 6001 1.0 0.8 0.3 0.6 0.3 0.1

[Figure 345]

[Figure 346]

[Figure 347]

Figure 11. User study UI.

Table 7. CLIP-IQA ablation across voxel sizes. CLIP loss improves semantic alignment consistently.

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Voxel Size 25× 30× 40× 50×

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

w/o CLIP Loss 40.89 40.55 38.92 38.64 w/ CLIP (ours) 41.35 41.03 40.07 40.14

|[Figure 360]|
|---|

|[Figure 361]|
|---|

|[Figure 362]|
|---|

|[Figure 363]|
|---|

|[Figure 364]|
|---|

|[Figure 365]|
|---|

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

and prominently represents abstract details, such as facial features, clothing, and textures?”

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

Input w/o Gumbel (w/ Gumbel) Ours Input w/o Gumbel (w/ Gumbel) Ours

- • Voxel art appeal: “Which version looks most visually appealing as a voxel art character, like something you might see in Minecraft or a stylized game?” For the grayscale examples, participants answered:
- • Geometry preservation: “Which grayscale voxel rendering more closely resembles the original 3D mesh in terms of overall geometry?”

Figure 12. Ablation user study of Gumbel. Four representative examples comparing results with and without Gumbel-Softmax. Without Gumbel-Softmax, voxel colors become blurred and features less distinct.

88.89% favored the with Gumbel-Softmax results for voxelart appeal (Table 3 (b)), confirming its importance in producing dominant tones and clear edges.

Expert study on color preference. We further conducted a focused evaluation on color quantization with 10 art-trained participants, all of whom had formal undergraduate education in art or design. Participants were asked to compare voxel art results with and without Gumbel-Softmax across 10 example pairs, and answered the following two questions:

Runtime analysis. On a single RTX 4090, the total runtime of our two-stage pipeline is approximately ∼20 minutes, depending on the voxel resolution. Overall, our method is substantially faster than manual voxel art creation.

- • Abstract detail: “Which voxel art version most clearly and prominently represents abstract details, such as facial features, clothing, and textures?”
- • Voxel art appeal: “Which version looks most visually appealing as a voxel art character, like something you might see in Minecraft or a stylized game?” As illustrated in Fig. 12, we present four representative ex-

#### D. Additional Qualitative Results

More comparisons with baselines. In total, we evaluated 35 character models for CLIP-IQA. Here, we additionally present 8 representative examples for qualitative comparison against the baselines: Pixel art to 3D extension, IN2N [30], Vox-E [82], and Blender Geometry Nodes, as illustrated in Fig. 13. While IN2N [30] is effective in certain

amples comparing results with and without Gumbel-Softmax. Across responses from 10 participants on 10 question pairs,

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

|[Figure 376]|
|---|

|[Figure 377]|
|---|

|[Figure 378]|
|---|

|[Figure 379]|
|---|

| |
|---|

|[Figure 380]|
|---|

| |
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 399]

| |
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

| |
|---|

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 422]

[Figure 423]

| |
|---|

|[Figure 424]|
|---|

|[Figure 425]|
|---|

|Fail|
|---|

|[Figure 426]|
|---|

|[Figure 427]|
|---|

|[Figure 428]|
|---|

|[Figure 429]|
|---|

| |
|---|

|Fail|
|---|

|[Figure 430]|
|---|

|[Figure 431]|
|---|

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

[Figure 443]

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

|[Figure 447]<br><br>Fail|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

| |
|---|

|[Figure 450]|
|---|

|[Figure 451]|
|---|

| |
|---|

|[Figure 452]|
|---|

|Fail|
|---|

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

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

| |
|---|

| |
|---|

| |
|---|

Input IN2N [Haque et al. 2023] VoX-E [Sella et al. 2023] Blender Ours Input IN2N [Haque et al. 2023] VoX-E [Sella et al. 2023] Blender Ours

Pixel art to 3D extension

Pixel art to 3D extension

- Figure 13. Additional qualitative comparisons with baselines. Eight representative examples compared with Pixel, IN2N, Vox-E, and Blender Geometry Nodes.

[Figure 462]

| |
|---|

[Figure 463]

| |
|---|

[Figure 464]

| |
|---|

Input mesh Blender Ours

|[Figure 465]|[Figure 466]|[Figure 467]|
|---|---|---|

Input pixel art

- Figure 14. Representative failure cases. Complex shapes with finegrained geometric details are difficult to represent under limited voxel resolution, resulting in loss of intricate structures.

details, resolution, and color palette selection. In contrast, our method enables fine-grained control over voxel resolution and palette constraints while faithfully preserving the visual characteristics through multi-view optimization. This demonstrates the advantage of Voxify3D for controllable and appearance-faithful voxel art generation.

Comparison with single-image 3D reconstruction. We also compare with Rodin [96], which performs well for image-to-mesh generation but is not designed for voxel art. As shown in Fig. 18, Rodin sometimes produces non-voxel outputs (right), and due to the single-image input, it often fails to capture reliable depth, resulting in flat structures (left). This further underscores the benefit of our multi-view voxel optimization pipeline.

cases, we found it often fails in our setting. This is mainly because each guidance image used during training can differ significantly, leading to large inconsistencies across views.

Results with varying palette settings. As shown in Fig. 15, we present comparisons under different color selection strategies and palette sizes, with K-means adopted as our default palette extraction method.

#### E. Failure cases and analysis

Finally, representative failure cases are shown in Fig. 14, mainly arising from complex shapes that exceed the capacity of the limited voxel resolution. These examples suggest that voxel art is better suited for capturing abstract details conveyed through color patterns and tonal contrasts, whereas fine-grained geometric intricacies are more likely to be lost under coarse discretization. A promising future direction is to adopt adaptive voxel resolutions, where regions requiring fine details use smaller voxels while simpler areas maintain coarser ones, enabling better preservation of geometric complexity without sacrificing the aesthetic appeal of voxel art.

Results under different voxel sizes. Fig. 16 illustrates voxel art renderings generated with varying voxel resolutions, demonstrating how grid granularity influences the level of abstraction, sharpness of edges, and overall visual fidelity of the outputs.

Comparison with LLM-based voxel generation. We also compare with Gemini 3 [28], the latest state-of-the-art large language model, which can generate 3D voxel art through code generation in AI Studio. As shown in Fig. 17, while Gemini 3 excels at creating interactive voxel-based applications and can produce detailed voxel art through its advanced coding capabilities, it lacks precise control over abstraction

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

2 colors 3 colors 4 colors 8 colors

2 colors

3 colors 4 colors 8 colors

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

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

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

K-means

K-means

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

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

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

| |
|---|

| |
|---|

[Figure 504]

[Figure 505]

Max-Min

Max-Min

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

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

input

[Figure 514]

input

[Figure 515]

[Figure 516]

Median Cut

Median Cut

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

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

Simulated Annealing

Simulated Annealing

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

Figure 15. Results with varying palette settings. Examples using different palette extraction strategies and palette sizes.

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

input 20x 25x 30x 40x 50x 60x

###### Figure 16. Results under different voxel sizes.

|[Figure 548]|
|---|

|[Figure 549]|
|---|

|[Figure 550]|
|---|

|[Figure 551]|
|---|

|[Figure 552]|
|---|

|[Figure 553]|
|---|

[Figure 554]

[Figure 555]

OursGemini3Input

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

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

|[Figure 560]|
|---|

|[Figure 561]|
|---|

|[Figure 562]|
|---|

|[Figure 563]|
|---|

|[Figure 564]|
|---|

|[Figure 565]|
|---|

- Figure 17. Comparison with Gemini 3 [28]. While Gemini 3 can generate voxel art through code, it lacks precise control over resolution, palette, and visual fidelity to input references.

[Figure 566]

[Figure 567]

[Figure 568]

| |
|---|

[Figure 569]

| |
|---|

[Figure 570]

| |
|---|

[Figure 571]

| |
|---|

|[Figure 572]|
|---|

[Figure 573]

| |
|---|

[Figure 574]

| |
|---|

|[Figure 575]|
|---|

[Figure 576]

| |
|---|

[Figure 577]

[Figure 578]

[Figure 579]

| |
|---|

Input pixel art

Input mesh Rodin [Wang et al. (2022b)]

Ours

Input pixel art

Input mesh Rodin [Wang et al. (2022b)]

Ours

- Figure 18. Comparison with Rodin [96]. Rodin excels at image-to-mesh but is not tailored for voxel art, often yielding non-voxel outputs (right) or flat geometry (left).

