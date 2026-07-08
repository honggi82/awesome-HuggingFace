# arXiv:2409.12957v2[cs.CV]17Mar2025

### 3DTopia-XL: Scaling High-quality 3D Asset Generation via Primitive Diffusion

Zhaoxi Chen1 Jiaxiang Tang2 Yuhao Dong1,3 Ziang Cao1 Fangzhou Hong1 Yushi Lan1 Tengfei Wang3 Haozhe Xie1

Tong Wu3,4 Shunsuke Saito Liang Pan3 Dahua Lin3,4 Ziwei Liu1

1 S-Lab, Nanyang Technological University 2 Peking University 3 Shanghai AI Laboratory 4 The Chinese University of Hong Kong

https://3dtopia.github.io/3DTopia-XL/

[Figure 1]

Figure 1. 3DTopia-XL generates high-quality 3D assets with smooth geometry and spatially varied textures and materials. The output asset (GLB mesh) can be seamlessly ported into graphics engines for physically-based rendering.

#### Abstract

The increasing demand for high-quality 3D assets across various industries necessitates efficient and automated 3D content creation. Despite recent advancements in 3D generative models, existing methods still face challenges with optimization speed, geometric fidelity, and the lack of assets for physically based rendering (PBR). In this paper, we introduce 3DTopia-XL, a scalable native 3D generative model designed to overcome these limitations. 3DTopia-XL leverages a novel primitive-based 3D representation, PrimX, which encodes detailed shape, albedo, and material field into a compact tensorial format, facilitating the modeling of high-resolution geometry with PBR assets. On top of the

novel representation, we propose a generative framework based on Diffusion Transformer (DiT), which comprises 1) Primitive Patch Compression, 2) and Latent Primitive Diffusion. 3DTopia-XL learns to generate high-quality 3D assets from textual or visual inputs. Extensive qualitative and quantitative evaluations are conducted to demonstrate that 3DTopia-XL significantly outperforms existing methods in generating high-quality 3D assets with fine-grained textures and materials, efficiently bridging the quality gap between generative models and real-world applications.

#### 1. Introduction

High-quality 3D assets are essential for various real-world applications like films, gaming, and virtual reality. How-

ever, creating high-quality 3D assets involves extensive manual labor and expertise. Therefore, it further fuels the demand for automatic 3D content creation techniques, which automatically generate 3D assets from visual or textual inputs by using 3D generative models.

Fortunately, rapid progress has been witnessed in the field of 3D generative models recently. Existing stateof-the-art techniques can be sorted into three categories. 1) Methods based on Score Distillation Sampling (SDS) [46, 55] lift 2D diffusion priors into 3D representation by per-scene optimization. However, these methods suffer from time-consuming optimization, poor geometry, and multifaceted inconsistency. 2) Methods based on sparseview reconstruction [19, 68] that leverage large models to regress 3D assets from single- or multi-view images. Most of these methods are built upon triplane-NeRF [5] representation. However, due to the triplane’s parameter inefficiency, the valid parameter space is limited to low resolutions in those models, leading to relatively low-quality 3D assets. Plus, reconstruction-based models also suffer from a low-diversity problem as deterministic methods. 3) Methods as native 3D generative models [30, 72] aim to model the probabilistic distribution of 3D assets, generating 3D objects given input conditions. Yet, few of them are capable of generating high-quality 3D objects with Physically Based Rendering (PBR) assets, which are geometry, texture, and material packed into a GLB file.

To address the limitations above, we propose 3DTopiaXL, a high-quality native 3D generative model for 3D assets at scale. Our key idea is scaling the powerful diffusion transformer [45] on top of a novel primitive-based 3D representation. At the core of 3DTopia-XL is an efficient 3D representation, PrimX, which encodes the shape, albedo, and material of a textured mesh in a compact N ×D tensor, enabling the modeling of high-resolution geometry with PBR assets. In specific, we anchor N primitives to the positions sampled on the mesh surface. Each primitive is a tiny voxel, parameterized by its 3D position, a global scale factor, and corresponding spatially varied payload for SDF, RGB, and material. Note that the proposed representation differentiates itself from the shape-only representation M-SDF [72] that PrimX encodes shape, color, and material in a unified way. It also supports efficient differentiable rendering, leading to the great potential to learn from not only 3D data but also image collections. Moreover, we carefully design initialization and fine-tuning strategy that enables PrimX to be rapidly tensorized from a textured mesh (GLB file) which is ten times faster than the triplane under the same setting.

Thanks to the tensorial and compact PrimX, we scale the 3D generative modeling using latent primitive diffusion with Transformers, where we treat each 3D object as a set of primitives. In specific, the proposed 3D generation framework consists of two modules. 1) Primitive Patch Compres-

sion uses a 3D VAE for spatial compression of each individual primitive to get latent primitive tokens; and 2) Latent Primitive Diffusion leverages the Diffusion Transformers (DiT) [45] to model global correlation of latent primitive tokens for generative modeling. The significant efficiency of the proposed representation allows us to achieve high-resolution generative training using a clean and unified framework without super-resolution to upscale the underlying 3D representation or post-hoc optimization-based mesh refinement.

In addition, we also carefully design algorithms for highquality 3D PBR asset extraction from PrimX, to ensure reversible transformations between PrimX and textured mesh. An issue for most 3D generation models [62, 68] is that they use vertex coloring to represent the object’s texture, leading to a significant quality drop when exporting their generation results into mesh format. Thanks to the high-quality surface modeled by Signed Distance Field (SDF) in PrimX, we propose to extract the 3D shape with zero-level contouring and sample texture and material values in a high-resolution UV space. This leads to high-quality asset extraction with considerably fewer vertices, which is also ready to be packed into GLB format for downstream tasks.

Extensive experiments are conducted both qualitatively and quantitatively to evaluate the effectiveness of our method in text-to-3D and image-to-3D tasks. Moreover, we do extensive ablation studies to motivate our design choices for a better efficiency-quality tradeoff in the context of generative modeling with PrimX. In conclusion, we summarize our contributions as follows: 1) We propose a novel 3D representation, PrimX, for high-quality 3D content creation, which is efficient, tensorial, and renderable. 2) We introduce a scalable generative framework, 3DTopia-XL, tailored for generating high-quality 3D assets with highresolution geometry, texture, and materials. 3) Practical techniques for assets extraction from 3D representation to avoid quality gap. 4) We demonstrate the superior quality and impressive applications of 3DTopia-XL for image-to3D and text-to-3D tasks.

#### 2. Related Work

Sparse-view Reconstruction Models. Recent advancements have been focusing on deterministic reconstruction methods that regress 3D assets from single- or multi-view images. Large Reconstruction Model (LRM) [14, 19] has shown that end-to-end training of a triplane-NeRF [5] regression model scales well to large datasets and can be highly generalizable. Although it can significantly accelerate generation speed, the generated 3D assets still exhibit relatively lower quality due to representation inefficiency and suffer from a low-diversity problem as a deterministic method. Subsequent works have extended this method to improve generation quality. For example, us-

ing multi-view images [2, 20, 27, 53, 60, 62, 66, 68] generated by 2D diffusion models as the input can effectively enhance the visual quality. However, the generative capability is actually enabled by the frontend multi-view diffusion models [29, 32, 36, 52] which cannot produce multiview images with accurate 3D consistency. Another direction is to use more efficient 3D representations such as Gaussian Splatting [6, 23, 56, 70, 73, 76] and triangular mesh [28, 34, 63, 75, 80]. However, few of them can generate high-quality PBR assets with sampling diversity.

Native 3D Diffusion Models. Similar to 2D diffusion models for image generation, efforts have been made to train a 3D native diffusion model [9, 31, 49, 79] on conditional 3D generation. However, unlike the universal image representation in 2D, there are many different choices for 3D representations. Voxel-based methods [40] can be directly extended from 2D methods, but they are constrained by the demanding memory usage, and suffer from scaling up to high-resolution data. Point cloud based methods [41, 42] are memory-efficient and can adapt to large-scale datasets, but they hardly represent the watertight and solid surface of the 3D assets. Implicit representations such as triplaneNeRF offer a better balance between memory and quality [4, 7, 9, 13, 21, 33, 43, 61]. There are also methods based on other representations such as meshes and primitives [8, 35, 69, 71, 72]. However, these methods still struggle with generalization or producing high-quality assets. Recent methods attempt to adapt latent diffusion models to 3D [18, 26, 30, 57, 65, 74, 77, 78]. These methods first train a 3D compression model such as a VAE to encode 3D assets into a more compact form, which allows the diffusion model to train more effectively and show strong generalization. However, they either suffer from low-resolution results or are incapable of modeling PBR materials. In this paper, we propose a new 3D latent diffusion model on a novel representation, PrimX, which can be efficiently computed from a textured mesh and unpacked into high-resolution geometry with PBR materials.

#### 3. Methodology

###### 3.1. PrimX: An Efficient Representation for Shape, Texture, and Material

Before diving into details, we outline the following design principles for 3D representation in the context of highquality large-scale 3D generative models: 1) Parameterefficient: provides a good trade-off between approximation error and parameter count; 2) Rapidly tensorizable: can be efficiently transformed into a tensor, which facilitates generative modeling with modern neural architectures; 3) Differentiably renderable: compatible with differentiable renderer, enabling learning from both 3D and 2D data.

Given the aforementioned principles, we propose a novel

primitive-based 3D representation, namely PrimX, which represents the 3D shape, texture, and material of a textured mesh as a compact N ×D tensor. It can be efficiently computed from a textured mesh (in GLB format) and directly rendered into 2D images via a differentiable rasterizer.

###### 3.1.1. Definition

Preliminaries. Given a textured 3D mesh, we denote its 3D shape as S ∈ R3, where x ∈ S are spatial points inside the occupancy of the shape, and x ∈ ∂S are the points on the shape’s boundary, i.e. the shape’s surface. We model the 3D shape as SDF as follows:

FSSDF(x) = sign(x,S) · d(x,∂S), (1)

where d(x,∂S) = miny∈S ||x − y||2 is the distance function, sign(x,S) = 1 when x ∈ S, and sign(x,S) = −1 when x ∈/ S. Moreover, given the neighborhood of the surface, U(∂S,δ) = {d(x,∂S) < δ}, the space-varied color and material functions of the target mesh are defined:

FSRGB(x) = sign(x,U)·C(x),FSMat(x) = sign(x,U)·ρ(x),

(2) where sign(x,U) = 1 when x ∈ U, and sign(x,U) = 0 when x ∈/ U. The C(x) : R3 −→ R3 and ρ(x) : R3 −→ R2 are corresponding texture sampling functions to get albedo and material (metallic and roughness) from UV-aligned texture maps given the 3D point x. Eventually, all shape, texture, and material information of a 3D mesh can be parameterized by the volumetric function FS = (FSSDF ⊕FSRGB ⊕ FSMat) : R3 −→ R6, where ⊕ denotes concatenation.

PrimX Representation. We aim to approximate FS with a neural volumetric function FV : R3 −→ R6 parameterized by a N × D tensor V. For efficiency, our key insight is to define FV as a set of N volumetric primitives distributed on the surface of the mesh:

V = {Vk}Nk=1,where Vk = {tk,sk,Xk}. (3) Each primitive Vk is a tiny voxel with a resolution of a3, parameterized by its 3D position tk ∈ R3, a global scale factor sk ∈ R+, and corresponding spatially varied feature payload Xk ∈ Ra×a×a×6 within the voxel. Note that, the payload in PrimX could be spatially varied features with any dimensions. Our instantiation here is to use a six-channel local grid Xk = {XkSDF,XkRGB,XkMat} to parameterize SDF, RGB color, and material respectively.

Inspired by Yariv et al. [72] where mosaic voxels are globally weighted to get a smooth surface, the approximation of a textured mesh is then defined as a weighted combination of primitives:

N k=1

[wk(x) · I(Xk,(x − tk)/sk)], (4)

FV(x) =

where I(Vk,x) denotes the trilinear interpolant over the voxel grid Xk at position x. The weighting function wk(x)

[Figure 2]

[Figure 3]

[Figure 4]

|𝑁 Primitives|
|---|

|Surface 𝜕𝓢|
|---|

Textured Mesh Shape (SDF)

[Figure 5]

Features in Each Primitive

[Figure 6]

[Figure 7]

|Position ∈ ℝ3|
|---|

||Scale ∈ ℝ|
|---|
| |
|---|---|
|SDF ∈ ℝ𝑎3| |
| | |
|RGB ∈ ℝ𝑎3×3| |
| | |
|Material ∈ ℝ𝑎3×2| |

[Figure 8]

[Figure 9]

a

[Figure 10]

Albedo (RGB) Material

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

a a

[Figure 15]

(Sec. 3.1.2) (Sec. 3.1.1)

Rapid Tensorization

𝓥k 𝑘=1𝑁 ∈ ℝ𝑁×(3+1+𝑎3×6)

𝑵 × 𝑫 Tensor

PrimX

- Figure 2. Illustration of PrimX. We propose to represent the 3D shape, texture, and material of a textured mesh as a compact N ×D tensor (Sec. 3.1.1). We anchor N primitives to the positions sampled on the mesh surface. Each primitive Vk is a tiny voxel with a resolution of a3, parameterized by its 3D position tk ∈ R3, a global scale factor sk ∈ R+, and corresponding spatially varied payload Xk ∈ Ra×a×a×6 for SDF, RGB, and material. This tensorial representation can be rapidly computed from a textured mesh within 1.5 minutes (Sec. 3.1.2). of each primitive is defined as:

UV space (1024 × 1024). Then, we get sampling points in 3D and query {FVRGB,FVMat} to get corresponding albedo and material values. Note that, we mask the UV space to get the index of valid vertices for efficient queries. Moreover, we dilate the UV texture maps and inpaint the dilated region with the nearest neighbors of existing textures, ensuring albedo and material maps smoothly blend outwards for anti-aliasing. Finally, we pack geometry, UV mapping, albedo, and material maps into a GLB file, which is ready for the graphics engine and various downstream tasks.

wˆk(x) N j=1 wˆj(x)

, (5)

wk(x) =

x − tk

sk ||∞). (6) Once the payload of primitives is determined, we can leverage a highly efficient differentiable renderer to turn PrimX into 2D images. In specific, given a camera ray r(t) = o + td with camera origin o and ray direction d, the corresponding pixel value I is solved by the following integral:

s.t. wˆk(x) = max(0,1 − ||

###### 3.1.2. Computing PrimX from Textured Mesh

We introduce our efficient fitting algorithm in this section that computes PrimX from the input textured mesh in a short period of time so that it is scalable on large-scale datasets for generative modeling. Given a textured 3D mesh FS, our goal is to compute PrimX such that FV(x) ≈ FS(x), s.t. x ∈ U(∂S,δ). Our key insight is that the fitting process can be efficiently achieved via a good initialization followed by lightweight finetuning.

tmax

dT(t) dt

FVRGB(r(t))

dt, (7)

I =

tmin

where T(t) is an exponential function of the SDF field to represent the opacity field:

t

FVSDF(r(t)) α

)2]dt. (8)

exp[−(

T(t) =

tmin

Initialization. We assume all textured meshes are provided in GLB format which contains triangular meshes, texture and material maps, and corresponding UV mappings. The vertices of the target mesh are first normalized within the unit cube. To initialize the position of primitives, we first apply uniform random sampling on the mesh surface to get Nˆ candidate initial points. Then, we perform farthest point sampling on this candidate point set to get N valid initial positions for all primitives. This two-step initialization of position ensures good coverage of FV over the boundary neighborhood U while also keeping the high-frequency shape details as much as possible. Then, we compute the L2 distance of each primitive to its nearest neighbors, taking the value as the initial scale factor for each primitive.

And α = 0.005 is the hyperparameter that controls the variance of the opacity field during this conversion.

To wrap up, the learnable parameters of a textured 3D mesh modeled by PrimX are primitive position t ∈ RN×3, primitive scale s ∈ RN×1, and voxel payload X ∈ RN×a

3×6 for SDF, albedo, and material. Therefore, each textured mesh can be represented as a compact N × D tensor, where D = 3 + 1 + a3 × 6 by concatenation.

PBR Asset Extraction. Once PrimX is constructed, it encodes all geometry and appearance information of the target mesh within the N × D tensor. Now, we introduce our efficient algorithm to convert PrimX back into a textured mesh in GLB file format. For geometry, we can easily extract the corresponding 3D shape with Marching Cubes algorithm [37] on zero level set of FVSDF. For PBR texture maps, we first perform UV unwrapping in a high-resolution

To initialize the payload of primitives, we first compute candidate points in global coordinates using initialized positions tk and scales sk as tk + skI for each primitive,

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

PrimPatch

[Figure 23]

Encoder PrimPatch

[Figure 24]

[Figure 25]

###### Latent Primitive Diffusion (Sec 3.3)

Conditioner

|𝑁 × 𝐷|
|---|

𝑁 × d

PrimX (Sec 3.1)

[Figure 26]

[Figure 27]

[Figure 28]

𝑲𝑽

TransformerLayer

TransformerLayer

“A cute unicorn”

CrossAttention

SelfAttention

Feedforward

###### or

[Figure 29]

[Figure 30]

[Figure 31]

Decoder

𝑸

…

[Figure 32]

[Figure 33]

Denoise

[Figure 34]

[Figure 35]

Ada LN

Timestep 𝑡

𝓥𝑇 𝓥0

𝓥𝑇−1

Input Condition

Prim Patch VAE (Sec 3.2) PBR Asset

- Figure 3. Overview of 3DTopia-XL. As a native 3D diffusion model, 3DTopia-XL is built upon a novel 3D representation PrimX (Sec. 3.1). This compact and expressive representation encodes the shape, texture, and material of a textured mesh efficiently, which allows modeling high-resolution geometry with PBR assets. Furthermore, this tensorial representation facilitates our patchbased compression using primitive patch VAE (Sec. 3.2). We then use our novel latent primitive diffusion (Sec. 3.3) for 3D generative modeling, which operates the diffusion and denoising process on the set of latent PrimX, naturally compatible with Transformerbased neural architectures.

where I is the unit local voxel grid with a resolution of a3. To initialize the SDF value, we query the SDF function converted from the 3D shape at each candidate point, i.e. XkSDF = FSSDF(tk + skI). Notably, it is non-trivial to get a robust conversion from arbitrary 3D shape to volumetric SDF function. Our implementation is based on an efficient ray marching with bounding volume hierarchy that works well with non-watertight topology. To initialize the color and material values, we sample the corresponding albedo colors and material values from UV space using geometric functions FSRGB and FSMat. Specifically, we compute the closest face and corresponding barycentric coordinates for each candidate point on the mesh, then interpolate UV coordinates and sample from the texture maps to get the value. Finetuning. Even if the initialization above offers a fairly good estimate of FS, a rapid finetuning process can further decrease the approximation error via gradient descent. Specifically, we optimize the well-initialized PrimX with a regression-based loss on SDF, albedo, and material values:

L(x;V) = λSDF||FSSDF(x) − FVSDF(x)||1

(9)

+ λ(||FSRGB(x) − FVRGB(x)||1

+ ||FSMat(x) − FVMat(x)||1),

where ∀x ∈ U, and λSDF,λ are loss weights. We employ a two-stage finetuning strategy where we optimize with λSDF = 10 and λ = 0 for the first 1k iterations and λSDF = 0 and λ = 1 for the second 1k iterations. More details are provided in the supplementary document.

###### 3.2. Primitive Patch Compression

In this section, we introduce our lightweight patch-based compression on individual primitives that turns 3D primitives into latent tokens for efficient generative modeling.

We opt for using a variational autoencoder [24] (VAE) operating on local voxel patches which compresses the payload of each primitive into latent tokens, i.e. Fae : RD −→ Rd. Specifically, the autoencoder Fae consists of an encoder E and a decoder D building with 3D convolutional layers. The encoder Fae has a downsampling rate of 48 that compresses the voxel payload Xk ∈ Ra

3×6 into the voxel latent Xˆk ∈ R(a/2)

3×1. We train Fae with reconstruction loss: Lae(X;E,D) = E[||Xk−D(E(Xk))||2+λklLkl(Xk,E)],

(10) where λkl is the weight for KL regularization over the latent space. Note that, unlike other works on 2D/3D latent diffusion models [50, 77] that perform global compression over all patches, our VAE only compresses each local primitive patch independently and defers the modeling of global semantics and inter-patch correlation to the diffusion model. Once the VAE is trained, we can compress the raw PrimX

- as Vk = {tk,sk,E(Xk)}. It leads to a low-dimensional parameter space for the diffusion model as V ∈ RN×d, where d = 3 + 1 + (a/2)3. In practice, this compact parameter space significantly allows more model parameters given a fixed computational budget, which is the key to scaling up 3D generative models in high resolution.

3.3. Latent Primitive Diffusion

On top of PrimX (Sec. 3.1) and the corresponding VAE (Sec. 3.2), the problem of 3D object generation is then converted to learning the distribution p(V) over large-scale datasets. Our goal is to train a diffusion model [17] that takes as input random noise VT and conditions c, and predicts PrimX samples. Note that, the target space for denoising is VT ∈ RN×d, where d = 3 + 1 + (a/2)3.

In specific, the diffusion model learns to denoise VT ∼ N(0,I) through denoising steps {VT−1,...,V0} given conditional signal c. As a set of primitives, PrimX is naturally compatible with Transformer-based architectures, where we treat each primitive as a token. Moreover, the permutation equivariance of PrimX removes the need for any positional encoding in Transformers.

Our largest latent primitive diffusion model gΦ is a 28layer transformer, with cross-attention layers to incorporate conditional signals, self-attention layers for modeling interprimitive correlations, and adaptive layer normalization to inject timestep conditions. The model gΦ learns to predict

- at timestep t given input condition signal:

gΦ(Vt,t,c) = {AdaLN[SelfAttn(CrossAttn(Vt,c,c)),t]}28,

(11) where CrossAttn(q,k,v) denotes the cross-attention layer with query, key, and value as input. SelfAttn(·) denotes the self-attention layer. AdaLN(·,t) denotes adaptive layer normalization layers to inject timestep conditioned

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

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

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

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

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

MLP MLP w/ PE Triplane Dense Voxels PrimX Ground Truth

Sparse Latent Points

- Figure 4. Evaluations of different 3D representations. We evaluate the effectiveness of different representations in fitting the ground truth’s shape, texture, and material (right). All representations are constrained to a budget of 1.05M parameters. PrimX achieves the highest fidelity in terms of geometry and appearance with significant strength in runtime efficiency (Table 1) at the same time.

Table 1. Quantitative evaluations of different 3D representations. We evaluate the approximation error of different representations for shape, texture, and material. All representations adhere to a parameter budget of 1.05M. PrimX shows the best fitting quality, especially for the geometry (also shown in Figure 4), while having the most speedy fitting runtime. The top three techniques are highlighted in red, orange, and yellow, respectively.

Representation Runtime CD ×10−4 ↓ PSNR-FSSDF ↑ PSNR-FSRGB ↑ PSNR-FSMat ↑ MLP 14 min 4.502 40.73 21.19 13.99 MLP w/ PE 14 min 4.638 40.82 21.78 12.75

Triplane 16 min 9.678 39.88 18.28 16.46 Dense Voxels 10 min 7.012 41.70 20.01 15.98

PrimX 1.5 min 1.310 41.74 21.86 16.50

modulation to cross-attention, self-attention, and feedforward layers. Moreover, we employ the pre-normalization scheme [67] for training stability. For noise scheduling, we use discrete 1,000 noise steps with a cosine scheduler during training. We opt for “v-prediction” [51] with ClassifierFree Guidance (CFG) [16] as the training objective for better conditional generation quality and faster convergence:

Ldiff(Φ) = Et∼[1,T],V0,Vt[||( √α¯tϵ −

(12)

√1 − α¯tV0) − gΦ(Vt,t,c¯(b))||22],

where ϵ is the noise sampled from Gaussian distribution, α¯t = ti=0(1 − βi) and βt comes from our cosine beta scheduler. And b ∼ B(p0) is a random variable sampled from Bernoulli distribution taking 0, 1 with probability p0 and 1 − p0 respectively. And the condition signal under CFG is defined as c¯(b) = b · c + (1 − b) · ∅, where ∅ is the learnable embedding for unconditional generation.

#### 4. Experiments 4.1. Representation Evaluation

Evaluation Protocol. We first evaluate different designs of 3D representations in the context of 3D generative modeling. Our evaluation principles focus on two aspects: 1) runtime from GLB mesh to the representation, and 2) approximation error for shape, texture, and material given a fixed computational budget. Given 30 GLB meshes ran-

domly sampled from our training dataset, we take the average fitting time till convergence as runtime measured on an A100 GPU. For geometry quality, we evaluate the Chamfer Distance (CD) and the Peak Signal-to-Noise Ratio (PSNR) of SDF values between ground truth meshes and fittings. For appearance quality, we evaluate the PSNR of albedo and materials values. All metrics are computed on a set of 500k points sampled near the shape surface.

Baselines. Given our final hyperparameters of PrimX, where N = 2048,a = 8, we fix the number of parameters of all representations to 2048 × 83 ≈ 1.05M for comparisons. We compare four alternative representations: 1) MLP: a pure Multi-Layer Perceptron with 3 layers and 1024 hidden dimensions; 2) MLP w/ PE: the MLP baseline with Positional Encoding (PE) [39] to the input coordinates; 3) Triplane [5]: three orthogonal 2D planes with a resolution of 128 × 128 and 16 channels, followed by a two-layer MLP decoder with 512 hidden dimensions. 4) Dense Voxels: a dense 3D voxel with a resolution of 100×100×100. 5) Sparse Points: 2048 latent points followed by a threelayer MLP as the decoder. All methods are trained with the same objective (Eq. 9) and points sampling strategy as ours. Results. Quantitative results are presented in Table 1, which shows that PrimX achieves the least approximation error among all methods, especially for geometry (indicated by CD). Besides the best quality, the proposed representation demonstrates significant efficiency in terms of runtime with nearly 7 times faster convergence speed compared with the second best, making it scalable on large-scale datasets. Figure 4 shows qualitative comparisons. MLP-based methods have periodic artifacts, especially for the geometry. Triplane and dense voxels yield bumpy surfaces and grid artifacts. Instead, PrimX produces the best quality with smooth geometry and fine-grained details like the tapering beard.

###### 4.2. Image-to-3D Generation

Comparison Methods. We run evaluations against two types of methods: 1) sparse-view reconstruction models, and 2) image-conditioned diffusion models.

| |
|---|

[Figure 65]

[Figure 66]

HDRI for Rendering

[Figure 67]

| |
|---|

[Figure 68]

[Figure 69]

HDRI for Rendering

[Figure 70]

| |
|---|

[Figure 71]

[Figure 72]

HDRI for Rendering

[Figure 73]

LGM InstantMesh Real3D CRM CraftsMan ShapE LN3Diff Ours Metallic / Roughness

Input Image

- Figure 5. Image-to-3D comparisons. For each method, we take the textured mesh predicted from the input image into Blender and render it with the target environment map. We compare our single-view conditioned model with sparse-view reconstruction models and imageconditioned diffusion models. 3DTopia-XL achieves the best visual and geometry quality. Thanks to our capability to generate spatially varied PBR assets shown on the rightmost, our generated mesh can also produce vivid reflectance with specular highlights and glossiness.

The reconstruction-based models, like LGM [56], InstantMesh [68], Real3D [20], CRM [62], are deterministic methods that learn to reconstruct 3D objects given four or six input views. They enable single-view to 3D synthesis by leveraging pretrained diffusion models [29, 52] to generate multiple views from the input single image. However, those methods may suffer from multi-view inconsistency caused by the frontend 2D diffusion models. The feed-forward diffusion models, like CraftsMan [30], ShapE [21], LN3Diff [26], are probabilistic methods that learn to generate 3D objects given input image conditions. All methods above only model the shape and color without considering roughness and metallic while our method is suitable to produce those assets with sampling diversity. To fairly compare the quality in real-world use cases, we place the output mesh of each method into Blender [10] and render it with an environment map. For methods that cannot produce PBR materials, we assign the default diffuse material.

Results. Figure 5 demonstrates qualitative results. Existing reconstruction-based models fail to produce good results which may suffer from multiview inconsistency and incapability to support spatially varied materials. Moreover,

Table 2. Analysis of number (N) and resolution (a) of PrimX.

# Primitives Resolution # Parameters PSNR-FSSDF ↑ PSNR-FSRGB ↑ PSNR-FSMat ↑

N = 64 a3 = 323 2.10M 61.05 22.18 18.10 N = 256 a3 = 163 1.05M 59.05 23.50 18.61 N = 512 a3 = 83 0.26M 59.57 22.58 18.50 N = 512 a3 = 163 2.10M 62.89 23.92 18.21 N = 2048 a3 = 83 1.05M 62.52 24.23 18.53

most of reconstruction models are built upon triplane representation which is not parameter-efficient. This downside limits the spatial resolution of the underlying 3D representation, leading to the bumpy surface indicated by the rendered normal. On the other hand, existing 3D diffusion models fail to generate objects that are visually aligned with the input condition. While CraftsMan is the only method with a comparable surface quality as ours, they are only capable of generating 3D shapes without textures and materials. In contrast, 3DTopia-XL achieves the best visual and geometry quality among all methods. Thanks to our capability to generate spatially varied PBR assets, our generated mesh can produce vivid reflectance with specular highlights even under harsh environmental illuminations.

###### 4.3. Text-to-3D Generation

We conduct quantitative evaluations against native text-to3D generative models, which achieves text-to-3D genera-

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

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

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

N/A

|𝑁 = 64, 𝑎 = 32|𝑁 = 256,𝑎 = 16|𝑁 = 512,𝑎 = 8|𝑁 = 512,𝑎 = 16|𝑵 = 𝟐𝟎𝟒𝟖,𝒂 = 𝟖|
|---|---|---|---|---|
| | | | | |

Ground Truth

#Param = 2.10M #Param = 1.05M #Param = 0.26M #Param = 2.10M #Param = 1.05M

- Figure 6. Ablation studies of the number and resolution of primitives. Our final setting (N = 2048, a = 8) has the optimal approximation quality of ground truth, especially for fine-grained details like thin rotor blades.

tion without text-to-multiview diffusion models. Given a set of unseen text prompts, we take the CLIP Score [48] as the evaluation metric following previous work [18, 21]. We mainly compare two methods with open-source implementations: Shap-E [21] and 3DTopia [18]. As shown in Table 4, our method achieves better alignment between input text and rendering of the generated asset. We defer the qualitative results in the supplementary.

###### 4.4. Further Analysis

Number and Resolution of Primitives. As a structured and serialized 3D representation, the number of primitives and the resolution of each primitive are two critical factors for the efficiency-quality tradeoff in PrimX. More and larger primitives often lead to better approximation quality. However, it results in a longer set length and deeper feature dimensions, causing inefficient long-context attention computation and training difficulty of the diffusion model. Therefore, we explore the impact of the number and resolution of primitives on different parameter budgets. We evaluate the PSNR of SDF, albedo, and material values given 500k points sampled near the surface. As shown in Table 2 and Figure 6, given a fixed parameter count, a larger set with compact primitives lead to better approximations. Furthermore, a longer sequence will increase the GFlops of DiT which also leads to better generation quality shown in the supplementary. Therefore, we use a large set of primitives with a relatively small local resolution.

Patch Compression Rate. The spatial compression rate of our primitive patch-based VAE (Sec. 3.2) is also an important design choice. Empirically, a higher compression rate leads to a more efficient latent diffusion model but risks information loss. Therefore, we analyze different compression rates given two different set lengths N = 256 and N = 2048 with the same parameter count of PrimX. We measure the PSNR between the VAE’s output and input on

- 1k random samples. Table 3 shows the results where the final choice of N = 2048 with compression rate f = 48 achieves the optimal VAE reconstruction. The setting with N = 256,f = 48 has the same compression rate but lower reconstruction quality and a latent space with higher resolution, which we find difficulty in the convergence of the

Table 3. Analysis of different compression rates for VAE. The scalar f stands for the compression rate between the input to the VAE and the latent code (i.e. the ratio between the 2nd and 3rd column of the table).

# Primitives VAE input Latent f PSNR ↑

N = 256 6 × 163 6 × 43 64 22.92 N = 256 6 × 163 1 × 43 384 19.80 N = 256 6 × 163 1 × 83 48 23.33 N = 2048 6 × 83 1 × 23 384 18.48 N = 2048 6 × 83 1 × 43 48 24.51

Table 4. Text-to-3D Evaluations. We evaluate the CLIP Score between input prompts and front-view renderings of output 3D assets.

Methods CLIP Score ↑

ShapE 21.98 3DTopia 22.54

Ours 24.33

[Figure 91]

[Figure 92]

[Figure 93]

|[Figure 94]|
|---|

| |
|---|

[Figure 95]

[Figure 96]

Denoising

[Figure 97]

a) 3D Inpaintingb) Interpolation

Mask Guidance

Conditioner

[Figure 98]

[Figure 99]

[Figure 100]

|[Figure 101]|
|---|

| |
|---|

[Figure 102]

[Figure 103]

Denoising

[Figure 104]

Mask Guidance

Conditioner

[Figure 105]

|𝜶 = 𝟎|
|---|

“A banana”

|𝜶 = 𝟏|
|---|

“An airplane that looks like banana”

Figure 7. 3D Generative Applications. a) 3D Inpainting for image-to-3D and b) Interpolation between text-to-3D samples.

latent primitive diffusion model gΦ.

Inpainting and Interpolation. We show 3D generative applications like 3D inpainting and interpolation in Figure 7, showing the unique property of 3D native diffusion model compared to reconstruction methods.

Besides the ablation studies above, we also analyze 1) the model scaling, 2) the sampling diversity, 3) PrimX initialization, 4) user study, 5) more comparisons and 6) more visual results, which are deferred to the supplementary due to the space limit.

#### 5. Conclusion

We present 3DTopia-XL, a native 3D diffusion model for PBR asset generation given textual or visual inputs. Central to our approach is PrimX, an innovative primitive-based 3D representation that is parameter-efficient, tensorial, and renderable. It encodes shape, albedo, and material into a compact N×D tensor, enabling the modeling of high-resolution geometry with PBR assets. On top of PrimX, we propose Latent Primitive Diffusion for scalable 3D generative models, together with practical techniques to export PBR assets ready for graphics pipelines.

Acknowledgement. This study is supported by the National Key R&D Program of China No.2022ZD0160102. This research/project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-PhD-2021-08-019), the Ministry of Education, Singapore, under its MOE AcRF Tier

- 2 (MOET2EP20221- 0012, MOE-T2EP20223-0002), NTU NAP, and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). References

- [1] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018. 13
- [2] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun Jampani. Sf3d: Stable fast 3d mesh reconstruction with uvunwrapping and illumination disentanglement, 2024. 3
- [3] Zoya Bylinskii, Laura Herman, Aaron Hertzmann, Stefanie Hutka, and Yile Zhang. Towards better user studies in computer graphics and vision. arXiv preprint arXiv:2206.11461,

2022. 16

- [4] Ziang Cao, Fangzhou Hong, Tong Wu, Liang Pan, and Ziwei Liu. Large-vocabulary 3d diffusion model with transformer. arXiv preprint arXiv:2309.07920, 2023. 3
- [5] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123–16133, 2022. 2, 6
- [6] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields. arXiv preprint arXiv:2407.04699, 2024. 3
- [7] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. arXiv preprint arXiv:2304.06714, 2023. 3
- [8] Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, and Ziwei Liu. Primdiffusion: Volumetric primitives diffusion for 3d human generation. In Thirtyseventh Conference on Neural Information Processing Systems, 2023. 3, 12
- [9] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In CVPR, pages 4456–4465, 2023. 3
- [10] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 7, 14, 15
- [11] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, pages 13142– 13153, 2023. 12, 13, 18, 20

- [12] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items, 2022. 12, 13
- [13] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023. 3
- [14] Zexin He and Tengfei Wang. Openlrm: Open-source large reconstruction models. https://github.com/ 3DTopia/OpenLRM, 2023. 2
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, G¨unter Klambauer, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a nash equilibrium. CoRR, abs/1706.08500, 2017. 13
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6, 23
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 5
- [18] Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Tengfei Wang, Liang Pan, Dahua Lin, and Ziwei Liu. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. arXiv preprint arXiv:2403.02234,

2024. 3, 8

- [19] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2, 15, 18, 23
- [20] Hanwen Jiang, Qixing Huang, and Georgios Pavlakos. Real3d: Scaling up large reconstruction models with realworld images, 2024. 3, 7
- [21] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 3, 7, 8, 13
- [22] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proc. CVPR, 2020. 12
- [23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ToG, 42(4):1–14, 2023. 3, 12
- [24] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 5
- [25] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 21, 23

- [26] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. arXiv preprint arXiv:2403.12019, 2024. 3, 7, 13
- [27] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023. 3
- [28] Mengfei Li, Xiaoxiao Long, Yixun Liang, Weiyu Li, Yuan Liu, Peng Li, Xiaowei Chi, Xingqun Qi, Wei Xue, Wenhan Luo, et al. M-lrm: Multi-view large reconstruction model. arXiv preprint arXiv:2406.07648, 2024. 3

- [29] Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wenhan Luo, Ping Tan, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. arXiv preprint arXiv:2405.11616, 2024. 3, 7
- [30] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 2, 3, 7
- [31] Yuhan Li, Yishun Dou, Xuanhong Chen, Bingbing Ni, Yilin Sun, Yutian Liu, and Fuzhen Wang. 3dqd: Generalized deep 3d shape prior via part-discretized diffusion process, 2023. 3
- [32] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023. 3
- [33] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023. 3
- [34] Minghua Liu, Chong Zeng, Xinyue Wei, Ruoxi Shi, Linghao Chen, Chao Xu, Mengqi Zhang, Zhaoning Wang, Xiaoshuai Zhang, Isabella Liu, Hongzhi Wu, and Hao Su. Meshformer : High-quality mesh generation with 3d-guided reconstruction model. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [35] Zhen Liu, Yao Feng, Michael J Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Score-based generative 3d mesh modeling. arXiv preprint arXiv:2303.08133, 2023. 3
- [36] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023. 3
- [37] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353, 1998. 4, 24
- [38] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 23
- [39] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 6
- [40] Norman M¨uller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In CVPR, pages 4328–4338, 2023. 3
- [41] Charlie Nash and Christopher KI Williams. The shape variational autoencoder: A deep generative model of partsegmented 3d objects. In Computer Graphics Forum, pages 1–12. Wiley Online Library, 2017. 3
- [42] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generat-

- ing 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 3, 23
- [43] Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, and Sergey Tulyakov. Autodecoding latent 3d diffusion models. arXiv preprint arXiv:2307.05445, 2023. 3
- [44] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 16, 18, 22
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 2, 12, 23
- [46] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2
- [47] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, 2017. 13
- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 8, 22
- [49] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 5, 23
- [51] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 6, 23
- [52] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 3, 7, 13
- [53] Yawar Siddiqui, Tom Monnier, Filippos Kokkinos, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, et al. Meta 3d assetgen: Text-to-mesh generation with highquality geometry, texture, and pbr materials. arXiv preprint arXiv:2407.02445, 2024. 3
- [54] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 23
- [55] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 2

- [56] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 3, 7, 13, 18

- [57] Zhicong Tang, Shuyang Gu, Chunyu Wang, Ting Zhang, Jianmin Bao, Dong Chen, and Baining Guo. Volumediffusion: Flexible text-to-3d generation with efficient volumetric encoder. arXiv preprint arXiv:2312.11459, 2023. 3
- [58] Gregory K. Wallace. The jpeg still picture compression standard. Commun. ACM, 34(4):30–44, 1991. 17
- [59] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023. 13
- [60] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024, 2023. 3
- [61] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4563–4573, 2023. 3
- [62] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024. 2, 3, 7, 13
- [63] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality mesh. arXiv preprint arXiv:2404.12385, 2024. 3
- [64] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image, 2024. 16
- [65] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024. 3
- [66] Desai Xie, Sai Bi, Zhixin Shu, Kai Zhang, Zexiang Xu, Yi Zhou, S¨oren Pirk, Arie Kaufman, Xin Sun, and Hao Tan. Lrm-zero: Training large reconstruction models with synthesized data. arXiv preprint arXiv:2406.09371, 2024. 3
- [67] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pages 10524–10533. PMLR, 2020. 6, 23
- [68] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 2, 3, 7, 13

- [69] Xiang Xu, Joseph Lambourne, Pradeep Jayaraman, Zhengqing Wang, Karl Willis, and Yasutaka Furukawa. Brepgen: A b-rep generative diffusion model with structured latent geometry. ACM Transactions on Graphics (TOG), 43

(4):1–14, 2024. 3

- [70] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv preprint arXiv:2403.14621, 2024. 3
- [71] Xingguang Yan, Han-Hung Lee, Ziyu Wan, and Angel X Chang. An object is worth 64x64 pixels: Generating 3d object via image diffusion. arXiv preprint arXiv:2408.03178,

2024. 3

- [72] Lior Yariv, Omri Puny, Natalia Neverova, Oran Gafni, and Yaron Lipman. Mosaic-sdf for 3d generative models. arXiv preprint arXiv:2312.09222, 2023. 2, 3, 12, 13, 16
- [73] Xuanyu Yi, Zike Wu, Qiuhong Shen, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, Shuicheng Yan, Xinchao Wang, and Hanwang Zhang. Mvgamba: Unify 3d content generation as state space sequence modeling. arXiv preprint arXiv:2406.06367, 2024. 3
- [74] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. arXiv preprint arXiv:2301.11445, 2023. 3, 12, 13
- [75] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu, and Yansong Tang. Geolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation. arXiv preprint arXiv:2406.15333, 2024. 3
- [76] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. arXiv preprint arXiv:2404.19702, 2024. 3
- [77] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. arXiv preprint arXiv:2406.13897,

2024. 3, 5, 12, 13

- [78] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. arXiv preprint arXiv:2306.17115, 2023. 3
- [79] Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, and Heung-Yeung Shum. Locally attentional sdf diffusion for controllable 3d shape generation. ACM Transactions on Graphics (SIGGRAPH), 42(4), 2023. 3
- [80] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable singleview 3d reconstruction with transformers. arXiv preprint arXiv:2312.09147, 2023. 3

#### A. Appendix

This supplementary material is organized as follows:

- • Sec. A.1 provides further discussions, including the main difference between PrimX and existing 3D representations (Sec. A.1.1) and limitations (Sec. A.1.2).
- • Sec. A.2 introduces further experiments and evaluations, including quantitative results on Objaverse [11] and GSO [12] datasets (Sec. A.2.1), user study (Sec. A.2.3), model scaling (Sec. A.2.4), sampling diversity (Sec. A.2.5), additional ablation studies on PrimX initialization (Sec. A.2.7), VAE designs (Sec. A.2.8), PBR extraction (Sec. A.2.9), differentiability (Sec. A.2.10) and more qualitative results (Sec. A.2.11).
- • Sec. A.3 documents the implementations details of 3DTopia-XL, including dataset and PrimX hyperparameters (Sec. A.3.1), conditioner and captions (Sec. A.3.2), model details and hyperparameters (Sec. A.3.3), and algorithms of reversible conversion between PrimX and mesh (Sec. A.3.4).
- • Besides, we also attach a demo video to demonstrate the key idea and qualitative results.

###### A.1. Discussion

###### A.1.1. Difference with Related Work

The core of our work is the proposed novel 3D representation, PrimX, that can model high-quality 3D shape, texture, and material in a unified and tensorial representation. It is worth highlighting the advantages of PrimX compared with other 3D representations in the generative context.

PrimX v.s. Implicit Vector Set. Previous works [74, 77] introduce the implicit vector set to encode a 3D shape globally. PrimX differentiates itself from the implicit vector set in three aspects:

- • PrimX encodes not only the shape but also texture and material in a unified way, which removes the necessity for a two-stage framework [77] that generates shape and texture separately. Although the implicit vector set has the potential to model more information other than the occupancy field, there is no existing work to demonstrate and justify this design. We suspect that texture and material information are surface-aligned which is far more expensive and difficult to model for the implicit vector set that uses dense modeling of the entire 3D space by using 3D points as queries.
- • PrimX is differentiable renderable while implicit vector set can be only exported to meshes.
- • PrimX is explicit and explainable for each token feature which facilitates 1) data augmentation by applying color transformation similar to [22]; and 2) downstream tasks like inpainting by explicitly masking certain tokens.

PrimX v.s. PrimDiffusion [8]. Chen et al. [8] proposes using volumetric primitives for 3D human generation, learning primitive-based representation from multiview posed images. PrimX has unique differences:

- • PrimX requires no 3D template for generative modeling by directly denoising the position of primitives. In contrast, PrimDiffusion requires a template mesh as the anchor for all primitives which works for 3D human generation where the target subject has a shared 3D canonical space. However, this is not the case for general objects as there is no mesh template.
- • PrimX simplifies the parameter space of volumetric primitives by using only position, a single scale, and voxelized payload, while the prior work models per-axis rotation and scale factors additionally. This simplification significantly saves the computational cost while achieving a comparable quality in our preliminary study.
- • PrimX models the target’s geometry as SDF field and is capable of learning from both 3D data and 2D data. The work above models the target’s geometry as volumetric opacity field and can only learn from 2D images.

PrimX v.s. M-SDF [72]. M-SDF introduces a shape-only representation to encode SDF of 3D mesh into mosaic voxels. PrimX has two distinct differences compared to it:

- • M-SDF only represents 3D shape, while our method finds a unified way to encode shape, texture, and material with high quality. It is non-trivial to represent shape, texture, and material within our tensorial and sparse representation. The shape is typically a 3D volumetric function while textural information is only surface-aligned. It is important to note that 1) proper instantiation of texture sampling function (Sec. A.3.1) and 2) carefully designed initialization strategy (Alg. 1) for PrimX are critical for representing shape, texture and material in high quality.
- • M-SDF is specialized to 3D domain while our representation can be differentiably rendered into 2D images.

PrimX v.s. 3DGS [23]. As a trending representation for 3D reconstruction, 3DGS is known for its efficiency as a primitive-based volumetric representation. However, the number of Gaussians required to represent a high-quality 3D object is considerably high (hundreds of thousands) compared with PrimX (N=2048). This long context property will lead to training difficulty and inefficient attention computation in the generative context where the set of Gaussians is operated by DiT [45]. Instead, PrimX can be treated as an “interpolation” between fully point-based representation (3DGS) and fully voxel-based representation (dense voxel) that groups primitives into explicitly structured local voxels. This hybrid operation significantly reduces the number of primitives, leading to a shorter context that boosts the training of the Transformer.

- Table 5. Quantitative evaluations of Image-to-3D on the Objaverse dataset [11]. We evaluate the fidelity, diversity and distributional similarity of 3DTopia-XL for single image to 3D generation compared with existing baselines. The evaluation is performed on a subset of Objaverse consisting of 600 random samples. KIDmat denotes the KID measured in the material space by rendering materials as colored

- 2D images. FPD and KPD denote the FID and KID metrics measured on 3D points in the space of PointNet++ [47]. COV and MMD denote Coverage Score and Minimum Matching Distance measured in the space of Chamfer Distance.

Method Paradigm KID (×10−2) ↓ KIDmat(×10−2) ↓ FPD ↓ KPD (×10−2) ↓ COV ↑ MMD (×10−3) ↓

LGM [56] Sparse-view Reconstruction 0.55 - 39.86 18.54 35.83 19.88 CRM [62] Sparse-view Reconstruction 1.93 - 37.09 14.37 31.83 25.12

ShapE [21] Native 3D Diffusion 11.95 - 20.27 7.55 59.67 14.98 LN3Diff [26] Native 3D Diffusion 0.89 - 29.36 11.27 33.50 22.64 Ours Native 3D Diffusion 1.02 0.69 15.74 3.59 50.31 14.63

- Table 6. Quantitative evaluations of Image-to-3D on the GSO dataset [12]. We evaluate the fidelity, diversity and distributional similarity of 3DTopia-XL for single image to 3D generation compared with existing baselines. The evaluation is performed on a subset of GSO dataset consisting of 300 random samples. FPD and KPD denote the FID and KID metrics measured on 3D points in the space of PointNet++ [47]. COV and MMD denote Coverage Score and Minimum Matching Distance measured in the space of Chamfer Distance.

Method Paradigm KID (×10−2) ↓ FPD ↓ KPD (×10−2) ↓ COV ↑ MMD (×10−3) ↓

LGM [56] Sparse-view Reconstruction 0.94 34.44 11.12 33.11 23.32 CRM [62] Sparse-view Reconstruction 1.67 26.61 4.82 38.57 17.37

InstantMesh [68] Sparse-view Reconstruction 0.91 21.54 4.15 37.01 16.31 ShapE [21] Native 3D Diffusion 13.75 22.80 4.03 40.43 18.58 Ours Native 3D Diffusion 1.38 16.16 3.86 55.58 14.35

###### A.1.2. Limitations and Future Work

It is important to note that 3DTopia-XL has been trained on a considerably large-scale dataset. However, there is still room for improvement in terms of quality. Different from existing high-quality 3D diffusion models [72, 77] which operate on 3D representations that are not differentiably renderable, 3DTopia-XL maintains the ability to directly learn from 2D image collections thanks to PrimX’s capability of differentiable rendering (Eq. 7 in the main paper). This opens up new opportunities to learn 3D generative models from a mixture of 3D and 2D data, which can be a solution to the lack of high-quality 3D data. Moreover, as an explicit representation, PrimX is interpretable and easy to drive. By manipulating primitives or groups of primitives, it is also fruitful to explore dynamic object generation and generative editing.

###### A.2. Additional Experiments

###### A.2.1. Quantitative Comparisons on Objaverse and GSO

We conduct extensive quantitative evaluations for imageto-3D on Objaverse [11] and GSO [12] datasets against existing methods including: 1) sparse-view reconstruction model: LGM [56], CRM [62], InstantMesh [68] and 2) native 3D diffusion models: ShapE [21] and LN3Diff [26].

Evaluation Metrics. For photometric quality, we benchmark KID [1] over the 2D renderings under random environmental lighting against ground truth. KIDmat is also cal-

culated to measure the quality of the generated PBR materials. We render the metallic and roughness into colored 2D images according to gltf 2.0 specifics1, and measure KID against ground truth. For geometric quality, we measure Point Cloud FID [15] (FPD) and Point Cloud KID [1] using the pretrained PointNet++ [47] provided by following previous work [21, 72]. Moreover, we also evaluate Coverage Score (COV) and Minimum Matching Distance (MMD) in the space of Chamfer Distance (CD) following previous work [72, 74]. We perform the farthest point sampling over the output mesh for each method to obtain 4096 points for evaluation. We randomly sample 300 objects on the GSO dataset and 600 objects on the Objaverse dataset for evaluation, respectively.

Results. As shown in Figure 5 and Figure 6, our method achieves the best 3D geometric quality, indicating the superiority of the proposed representation and generative modeling. Note that the methods based on sparse-view reconstruction rely on pretrained 2D multiview diffusion models [52, 59] to reconstruct 3D objects from sparse input views. Therefore, their models have more input visual information and thus achieve slightly better visual quality. However, this cascaded pipeline prone to yielding 3D distortions due to 3D inconsistency of 2D diffusion models, indicated by the worse 3D metrics of them. Most importantly,

1https://registry.khronos.org/glTF/specs/2.0/ glTF-2.0.html#metallic-roughness-material

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

Input Renderings Normal Metallic Roughness

- Figure 8. 3DTopia-XL can generate 3D assets directly from single-view images without relying on 2D image-to-multiview diffusion models. We visualize the input single-view image and corresponding HDRIs for environmental lighting on the left. Please note the highquality results and spatially varied materials generated by our method. All scenes are rendered using Blender [10].

[Figure 136]

[Figure 137]

“a teapot”

[Figure 138]

[Figure 139]

“a teapot”

[Figure 140]

[Figure 141]

“a donut with pink icing”

[Figure 142]

[Figure 143]

“a wristband”

[Figure 144]

[Figure 145]

“a heavy hammer”

[Figure 146]

[Figure 147]

“a chess piece (queen)”

[Figure 148]

[Figure 149]

“a steel wine glass”

[Figure 150]

[Figure 151]

“a watering can”

[Figure 152]

[Figure 153]

“a coffee cup”

[Figure 154]

[Figure 155]

“a light bulb”

Input Renderings Normal Metallic Roughness

- Figure 9. 3DTopia-XL can generate 3D assets directly from texts without relying on 2D text-to-image diffusion models, which is uniquely different from sparse-view reconstruction models [19]. We visualize the input text prompts and corresponding HDRIs for environmental lighting on the left. Please note the sampling diversity and spatially varied materials generated by our method. All scenes are rendered using Blender [10].

- 3DTopia-XL is the only method capable of producing PBR materials from images or texts among all methods.

###### A.2.2. Additional Comparisons

We show image-to-3D comparisons with Unique3D [64] in Fig. 10(c). As a per-sample optimization-based method using multiview images, Unique3D excels in texture quality but suffers from geometry artifacts (weird geometry from novel views) due to the inconsistency of 2D diffusion models. We believe that training a feedforward reconstruction method with PrimX would be a good geometry initialization and mesh constraint for Unique3D, which shortens its optimization time.

Moreover, we demonstrate our image-to-3D generation using a casual phone capture in Fig. 10(b), indicating the generalizability of 3DTopia-XL to the real-world domain.

###### A.2.3. User Study

We conduct an extensive user study to evaluate image-to3D performance quantitatively. We opt for an output evaluation [3] for user study, where each volunteer is shown with a pair of results comparing a random method against ours, and asked to choose the better one in four aspects: 1) Overall Quality, 2) Image Alignment, 3) Surface Smoothness, and 4) Physical Correctness. One of the samples presented to the attendees is shown in Figure 12. A total number of 48 paired samples are provided to 27 volunteers for the flip test. We summarize the average preference percentage across all four dimensions in Figure 11. 3DTopia-XL is the best one among all methods. Although the image alignment of our method is only a slight improvement against reconstructionbased methods like CRM, the superior quality of geometry and the ability to model physically based materials are the keys to producing the best overall quality in the final rendering.

###### A.2.4. Scaling

We further investigate the scaling law of 3DTopia-XL against model sizes and iterations. For metrics, we use Fr´echet Inception Distance (FID) computed over 5k random samples without CFG guidance. Specifically, we consider Latent-FID which is computed in the latent space of our VAE and Rendering-FID which is computed on the DINO [44] embeddings extracted from images rendered with Eq. 7 in the main paper. Figure 13 shows how LatentFID and Rendering-FID change as the model size increases. We observe consistent improvements as the model becomes deeper and wider. Table 7 also demonstrates that longer sequence (smaller patches) leads to better performance, which may come from the findings in the vanilla DiT that increasing GFlops leads to better performance.

###### A.2.5. Sampling Diversity

At last, we demonstrate the impressive sampling diversity of 3DTopia-XL as a generative model, as shown in Figure 14.

Table 7. Longer sequence leads to better convergence. Given a fixed PrimX parameter budget of 1.05M, we compare the models trained with {N = 256, a = 16} and {N = 2048, a = 8}.

Setting Rendering-FID ↓ Latent-FID ↓ N = 256 76.31 104.8

N = 2048 16.16 24.43

Given the same input image and varying random seeds, our model can generate diverse high-quality 3D assets with different geometry and spatially varied PBR materials.

###### A.2.6. Generation of Inner Geometric Structures

We present results with plausible and diverse inner structures in Fig. 15. Thanks to the native 3D representation PrimX, 3DTopia-XL can generate well-defined and diverse inner structures from images / texts, which facilitates downstream tasks such as physical simulations and compositional object generation. Additional results on generating thin structure is shown in Fig. 10(a).

###### A.2.7. Ablation study on PrimX Initialization

In this section, we conduct ablation studies on the impact of different initialization strategies for mesh to PrimX conversion (Algorithm 1). We compare three alternatives here:

- • Uniform + Farthest (Ours): 1) we first perform uniform sampling to get Nˆ candidate points; and 2) we run farthest point sampling on the candidate point set to get N primitives and initialize their scales to ensure coverage.
- • Farthest: directly perform farthest point sampling to get N primitives with a unique global scale factor as in MSDF [72].
- • Coverage: 1) we first perform farthest point sampling to

get 43N primitives; 2) a uniformly sampled point set is used to test the coverage by existing primitives, and points

not covered are held out; and 3) we perform the second farthest point sampling on the held-out set to get the rest

1 4N primitive.

As shown in Figure 16, the “Farthest” solution is sensitive to the topology, which may lead to the insufficient number of primitive allocated to the flattened surface with a few mesh faces, causing the gap in the drill. Our final solution achieves comparable quality with the complicated “Coverage” solution and is capable of modeling finegrained geometric details and consistent texture and material with ground truth. However, due to unnecessary computation overhead introduced by the latter solution, we choose the “Uniform + Farthest” initialization strategy as the final solution which is simple but effective. Quantitative results in Table 8 also confirm the above observation.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

##### (a)

##### (b) (c)

[Figure 163]

Input

Output GLB Input Output GLB Input Ours Unique3D (Lengthy Optim. & Inconsistency)

Figure 10. (a) Thin structures. (b) Real-world inputs. (c) Comparison with Unique3D, which shows geometry inconsistency.

[Figure 164]

[Figure 165]

ShapE

ShapE

Real3D

Real3D

Ours

Ours

LGM

LGM

InstantMesh CRM Craftsman

InstantMesh

CRM Craftsman

[Figure 166]

[Figure 167]

ShapE

ShapE

Real3D

Real3D

Ours

Ours

LGM

LGM

InstantMesh

InstantMesh CRM Craftsman

CRM Craftsman

- Figure 11. User study. We quantitatively evaluate comparison methods by conducting preference tests against our method on four dimensions. The results show that 3DTopia-XL has the highest preference rate compared with other methods.

[Figure 168]

- Figure 12. User study sample. For each sample in the user study, we present to the attendee with the input image (upper left) and target environment illuminations (bottom left) for rendering the mesh. Each volunteer is asked to choose the better one from A/B across four dimensions: 1) Overall quality, 2) Image alignment, 3) Surface smoothness, and 4) Physical correctness of renderings. The order and notation of methods are randomized and anonymized.

ber of principal components and projects PrimX onto a low-dimensional space. Take the projection matrix consisting of principal components as P, the encoding process denotes as XP while the decoding process denotes as XPPT. This compression mechanism relies on a set of known principal components at inference time.

• DCT: Discrete Cosine Transformation similar to the JPEG [58] compression algorithm. As our PrimX is naturally divided by 8 for each primitive, we can traverse through each voxel’s value in zig-zag order and perform DCT for spatial compression.

As shown in Figure 17, PCA fails to reproduce smooth geometry at all compression rates due to the loss of spatial structure during the encoding process. DCT, as a widely used spatial compression algorithm for 2D images, can also achieve reasonably good compression quality at a relatively lower compression rate (3072 to 384). However, it fails at a higher compression rate and cannot achieve comparable quality as patch-wise VAE.

###### A.2.8. Ablation study on VAE Designs

Given our goal to do spatial compression of VAE in Primitive Patch Compression, we additionally compare two alternatives suitable for spatial compression:

• PCA: Principal Component Analysis uses a certain num-

###### A.2.9. Ablation study on PrimX2Mesh Algorithm

As we mentioned in the main paper existing work on 3D generation did not carefully deal with mesh extraction when converting the underlying 3D representations to GLB formatted meshes. In this paper, we carefully design the PrimX2Mesh algorithm as outlined in Alg. 2. Furthermore,

[Figure 169]

[Figure 170]

- Figure 13. Scaling up 3DTopia-XL improves FID. As the computation and model size scale up, the model performance improves consistently. For metrics, we consider Latent-FID which is computed in the latent space of our VAE and Rendering-FID which is computed on the DINO [44] embeddings extracted from images.

Table 8. Quantitative evaluations of different initialization strategies for mesh to PrimX.

Solution PSNR-FSSDF ↑ PSNR-FSRGB ↑ PSNR-FSMat ↑ Uniform + Farthest 72.12 26.26 21.65

Farthest 56.86 14.30 10.16 Coverage 71.38 26.06 21.41

we compare our PBR mesh extraction with different design alternatives including:

- • Vert Coloring: a baseline that uses vertex coloring as many prior works;
- • No Material: a baseline that does not pack PBR materials in the GLB mesh during the conversion;
- • Low-res UV: a baseline that unwraps textures in a lowresolution UV space (256 × 256);
- • No Inpainting: a baseline that does not perform texture inpainting based on UV adjacency.

The results are presented in Figure 18. As clearly shown in the top two rows, neither vertex coloring nor ignoring material can produce vivid reflectance under natural illumination. This is due to the fact that both two modes cannot support PBR materials modeling. Low-resolution UV space introduces texture noises as well as material noises. In contrast, our high-resolution UV space (1024 × 1024) leads to high-quality texture representation. The baseline without UV inpainting shows severe zig-zag artifacts for the texture, which result from the empty region of UV space not properly inpainted. The above comparisons further justify the ingredients in Alg. 2 including 1) using UV texturing, 2) using high-resolution UV space, and 3) doing UV inpainting based on the vertices and faces adjacencies.

###### A.2.10. PrimX can Learn from Both 2D and 3D

Recall the three design principles for 3D representation in the context of high-quality large-scale 3D generative models: 1) Parameter-efficient: provides a good trade-off between approximation error and parameter count; 2) Rapidly tensorizable: can be efficiently transformed into a tensor, which facilitates generative modeling with modern neural architectures; 3) Differentiably renderable: compatible with differentiable renderer, enabling learning from both 3D and 2D data.

PrimX is the representation that satisfies all the above criteria. We have extensively introduced how to learn from 3D dataset [11] for PrimXin Alg. 1 and Alg. 2. Here, we further demonstrate the great potential of PrimX to leverage knowledge from 2D images. Thanks to our explicit design that maintains the differentiability of the rendering process of PrimX, this tensorial 3D representation can be efficiently rasterized into 2D images given a perspective camera view. By back-propagating the gradient of image reconstruction loss to the payload in PrimX, we can improve the texture quality represented in PrimX, as shown in Figure 19. This further demonstrates that PrimX also has its potential in sparse-view reconstruction models similar to LGM [56] and LRM [19]. By replacing the underlying 3D representation

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

###### Input Renderings Normal Metallic Roughness

- Figure 14. Sampling diversity. Given the same input image, 3DTopia-XL can generate diverse 3D assets by varying random seeds only. Zoom in for diverse shapes and spatially varied PBR materials.

|[Figure 189]<br><br>Bottom-up view from inside<br><br>|
|---|

[Figure 190]

[Figure 191]

## “a watering can”

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Variant 1 Variant 2

Figure 15. Due to native 3D representation, 3DTopia-XL can generate well-defined and diverse inner structures from images / texts.

as PrimX, we can also unlock a reconstruction-based model that can learn from both 2D and 3D data.

###### A.2.11. More Results

We present more image-conditioned and text-conditioned generation results in Figure 8 and Figure 9 respectively.

###### A.3. Implementation Details

###### A.3.1. Data Standardization

Datasets. The scale and quality of 3D data determine the quality and effectiveness of 3D generative models at scales. We filter out low-quality meshes, such as fragmented shapes and large-scale scenes, resulting in a refined collection of

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

###### Uniform + Farthest Farthest Coverage Ground Truth

Figure 16. The impact of different initialization strategies for mesh to PrimX algorithm (Alg. 1).

[Figure 200]

[Figure 201]

[Figure 202]

###### Ours vs. PCAOurs vs. DCT

Ours: 3072 (8×8×8×6) → 64 PCA: 3072 (8×8×8×6) → 128 PCA: 3072 (8×8×8×6) → 32

[Figure 203]

[Figure 204]

[Figure 205]

Ours: 3072 (8×8×8×6) → 64 DCT: 3072 (8×8×8×6) → 6×64 DCT: 3072 (8×8×8×6) → 6×1

Figure 17. Ablation studies on different VAE designs for Primitive Patch Compression. For patch-wise spatial compression, we additionally compare our method with two alternatives: 1) PCA and 2) DCT given various compression rates.

256k objects from Objaverse [11].

Standardization Pipeline. Computing PrimX on largescale datasets involves two critical steps: 1) Instantiation of sampling functions {FSSDF,FSRGB,FSMat} from a GLB file and 2) Execution of the fitting algorithm in Sec. 3.1.2 of the main paper, i.e. Alg. 1. Given the massive amount of meshes from diverse sources in Objaverse, there are challenges for properly instantiating the sampling functions in a universal way such as fragmented meshes, non-watertight

shapes, and inconsistent UVs. We develop a unified data loading pipeline that standardizes the objects across different textured mesh definitions (vertex color, UV map, part-wise material, etc.). Our standardized procedure starts with loading the GLB file as a connected graph. We filter out subcomponents that have less than 3 face adjacency which typically represent isolated planes or grounds. After that, all mesh subcomponents are globally normalized to the unit cube [−1,1] given one unique global bounding box. Then, we instantiate geometric sampling functions

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

No MaterialVert ColoringLow-res UVNo Inpainting

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

###### Ours

Renderings Normal Metallic Roughness

- Figure 18. Ablation study on PrimX to mesh algorithm (Alg. 2). Please check Sec. A.2.9 for more details of our experimental setup.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Input Image w/ Refinement by Differentiable Rendering w/o Refinement by Differentiable Rendering

- Figure 19. The capability of PrimX to learn from 2D data. Thanks to being differentiable renderable, PrimX can also learn from 2D images to further refine the results. This also implies its great potential to train on heterogeneous data from both 2D and 3D collections.

for each mesh subcomponent for SDF, texture, and material values. For the model that lack PBR material, we will assign a default diffuse material according to Blender’s principle BSDF node.

PrimX Hyperparameters. To get a tradeoff between computational complexity and approximation error, we choose our PrimX to have N = 2048 primitives where each primitive’s payload has a resolution of a = 8. It indicates that the sequence length of our primitive diffusion Trans-

former is also 2048 where each token has a dimension of d = 3 + 1 + (a/2)3 = 68. For the rapid finetuning stage for computing PrimX, we sample 500k points from the target mesh, where 300k points are sampled on the surface and 200k points are sampled with a standard deviation of 0.01 near the surface. The finetuning stage is run for 2k iterations with a batch size of 16k points using an Adam [25] optimizer at a learning rate of 1 × 10−4.

- Algorithm 1: Computing PrimX from a Textured Mesh (GLB format)

Input : GLB mesh FS, number of primitives N, voxel resolution a, number of candidates Nˆ

▷ Initialization FS ← (FSSDF ⊕ FSRGB ⊕ FSMat) ▷ parse volumetric sampling functions

ˆtk k∈[Nˆ] ← uniform random sampling of ∂S

{tk}k∈[N] ← farthest point sampling of ˆtk k∈[Nˆ] for i ← 1 to N do

si ← L2 distance to its nearest neighbors in {tk}k∈[N] XiSDF ← FSSDF(ti + siI) ▷ I is the local voxel grid tuvi ← UV and barycentric coordinates of the nearest face for (ti + siI) XiRGB ← FSRGB(tuvi ) XiMat ← FSMat(tuvi ) Xi ← (XiSDF ⊕ XiRGB ⊕ XiMat) ▷ ⊕ denotes concatenation Vi ← {ti,si,Xi}

V ← {Vk}k∈[N]

▷ Rapid Finetuning while not converged do

{xi}i∈[B] ← random sampling of U(∂S,δ) with a batch size of B Take a gradient descent step with ∇VL(x;V) ▷ Eq. 9 in the main paper

Output: V

- Algorithm 2: Extracting a Textured Mesh (GLB format) from PrimX

Input : PrimX V = {tk,sk,Xk}k∈[N], Marching Cubes resolution A, chunk size B

FVSDF,FVRGB,FVMat ← FV

▷ Shape Extraction {xi}i∈[A3] ← Initialize a unit cube with a resolution of A × A × A for i ← 1 to A3 do

if mink ||xi − {tk}k∈[N] ||2 > sk then

FSSDF(xi) ← mink ||xi − {tk}k∈[N] ||2 · sign(XkSDF) ▷ No query of PrimX else

FSSDF(xi) ← FVSDF(xi) ▷ Run in parallel with a chunk size B in practice {V,F} ← Marching Cubes on the zero level set of FSSDF(xi) i∈[A

3]

▷ Texture and Material Extraction Empty texture maps (FSRGB,FSMat) and UV Mapping ← UV unwrapping on {V,F} {xuvi } ← Get validate sampling points in 3D with a rasterizer FSRGB(xuvi ) ← FVRGB(xuvi ) FSMat(xuvi ) ← FVMat(xuvi ) (FSRGB,FSMat) ← inpainting with nearest neighbors based on UV mapping adjacency S ← V,F,FSRGB,FSMat,UV Mapping ▷ Packed in GLB format Output: S

###### A.3.2. Condition Signals

Conditioners. The conditional generation formulation in Sec. 3.3 of the main paper is compatible with most modalities. In this paper, we mainly explored conditional generation on two modalities, images and texts. For image-conditioned models, we leverage pretrained DI-

NOv2 model [44], specifically “DINOv2-ViT-B/14”2, to extract visual tokens from input images (at a resolution of 518 × 518) and take it as the input condition c. For textconditioned models, we leverage the text encoder of the pretrained image-language model [48], namely “CLIP-ViT-

2https://github.com/facebookresearch/dinov2

L/14”3, to extract language tokens from input texts.

Images. Thanks to our high-quality representation PrimX and its capability for efficient rendering, we do not need to undergo the complex and expensive rendering process like other works [19], which renders all raw meshes into 2D images for training. Instead, we opt to use the front-view image rendered by Eq. 7 which is 1) efficient enough to compute on-the-fly, and 2) consistent with the underlying representation compared with the rendering from the raw mesh.

Text Captions. We use 200,000 samples from Objaverse to generate text captions. For each object, six different views are rendered against a white background. We then use GPT-4V to generate keywords based on these images, focusing on aspects such as geometry, texture, and style. While we pre-define certain keywords for each aspect, the model is also encouraged to generate more context-specific keywords. Once the keywords are obtained, GPT-4 is employed to summarize them into a single sentence, beginning with ’A 3D model of...’. These text captions are subsequently prepared as input conditions.

- A.3.3. Model Details Architecture. We train the latent primitive diffusion

model gΦ using a Transformer-based architecture [45] for scalability. Our final model (Eq. 11 in the main paper) is built with 28 layers with 16-head attentions and 1152 hidden dimensions, leading to a total number of ∼1B parameters. Moreover, we employ the pre-normalization scheme [67] for training stability. For noise scheduling, we use discrete 1,000 noise steps with a cosine scheduler during training. We opt for “v-prediction” [51] with ClassifierFree Guidance (CFG) [16] as the training objective for better conditional generation quality and faster convergence.

Channel-wise Normalization. Most importantly, given the distribution gap between the 3D coordinate t and the latent E(X), one may carefully deal with the normalization of the input data to the diffusion model. Recall our diffusion target is a hybrid tensor V = {t,s,E(X)}, where E(X) is the 3D latent in the KL-regularized VAE that is close to a Gaussian distribution. However, the 3D coordinate t is not normally distributed in the 3D space. This inter-channel distribution gap within the diffusion target will lead to suboptimal convergence if the data is globally normalized by a scalar (which is the common practice in 2D diffusion models4). Intuitively, our latent primitive diffusion model aims

- 3https://github.com/mlfoundations/open_clip
- 4https : / / github . com / huggingface / diffusers /

issues/437

to solve a hybrid problem of point diffusion [42] and latent diffusion [50] simultaneously. To bridge this gap, we propose to normalize the input data in a channel-wise manner. Specifically, we trace channel-wise statistics (mean and standard deviation) over 50k random samples from the dataset. During the training phase, we keep them as constant normalizing factors and apply them to the input of the latent primitive diffusion model.

Training. We train gΦ with a batch size of 1024 using an AdamW [38] optimizer. The learning rate is set to 1×10−4 with a cosine learning rate warmup for 3k iterations. The probability of condition dropout for CFG is set to p0 = 0.1. During training, we apply EMA (Exponential Moving Average) on the model’s weight with a decay of 0.9999 for better training stability. The image-conditioned model is trained on 16 nodes of 8 A100 GPUs for 350k iterations, which takes around 14 days to converge. The text-conditioned model is trained on 16 nodes of 8 A100 GPUs for 200k iterations, which takes around 5 days to converge.

VAE. The 3D VAE for patch-wise primitive compression is built with 3D convolutional layers. We train the VAE on a subset of the entire dataset with 98k samples, finding it generalizes well on unseen data. The training takes 60k iterations with a batch size of 256 using an Adam [25] optimizer with a learning rate of 1×10−4. Note that, this batch size indicates the total number of PrimX samples per iteration. As our VAE operates on each primitive independently, the actual batch size would be N × 256. We set the weight for KL regularization to λkl = 5 × 10−4. The training is distributed on 8 nodes of 8 A100 GPUs, which takes about 18 hours.

Inference. By default, we evaluate our model with a 25step DDIM [54] sampler and CFG scale at 6. We find the optimal range of the DDIM sampling steps is 25 ∼ 100 while the CFG scale is 4 ∼ 10. The inference can be efficiently done on a single A100 GPU within 5 seconds.

A.3.4. Reversible Conversion between PrimX and Mesh Mesh to PrimX. As introduced in the main paper (Sec. 3.1.2), we leverage a two-stage strategy to compute PrimX from a textured mesh. Given a textured mesh FS that contains the shape, albedo, and material information, we convert it into PrimX with N primitives via a good initialization followed by a rapid finetuning. Here, we introduce more details of this procedure in Algorithm 1. Our implementation to instantiate the volumetric sampling function of SDF that works for non-watertight mesh is derived from cuBVH5.

5https://github.com/ashawkey/cubvh

PrimX to Mesh. As introduced in the main paper (Sec. 3.1.1), PrimX can be inversely converted back to a textured mesh in GLB format with minimal loss of information. The key is to utilize a high-resolution UV space for texturing instead of vertex coloring. We specify the details of this procedure in Algorithm 2, where we use xatlas6 for UV unwrapping, nvdiffrast7 for mesh-based rasterizer, and mcubes8 for Marching Cubes [37].

- 6https://github.com/jpcy/xatlas
- 7https://github.com/NVlabs/nvdiffrast
- 8https://github.com/pmneila/PyMCubes

