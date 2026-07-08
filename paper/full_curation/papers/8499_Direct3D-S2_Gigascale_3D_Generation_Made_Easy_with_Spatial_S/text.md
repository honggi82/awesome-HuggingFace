arXiv:2505.17412v2[cs.CV]26May2025

# Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention

Shuang Wu1,2* Youtian Lin1,2∗ Feihu Zhang2 Yifei Zeng1,2 Yikang Yang1 Yajie Bao2

Jiachen Qian2 Siyu Zhu3 Xun Cao1 Philip Torr4 † Yao Yao1‡ 1Nanjing University 2DreamTech 3Fudan University 4University of Oxford

[Figure 1]

- Figure 1. Mesh generation results from our method on different input images. Our method can generate detailed and complex 3D shapes. The meshes show fine geometry and high visual quality, demonstrating the strength of our approach for high-resolution 3D generation.

## Abstract

Generating high-resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges. We introduce Direct3D-S2, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the Spatial Sparse Attention (SSA) mechanism, which greatly enhances the efficiency of Diffusion Transformer (DiT) computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, substantially reducing computational overhead and achieving a 3.9× speedup in the

*Equal contribution. Work done during internship at DreamTech. †Chief scientific advisor of DreamTech. ‡Corresponding author.

forward pass and a 9.6× speedup in the backward pass. Our framework also includes a variational autoencoder (VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared to previous methods with heterogeneous representations in 3D VAE, this unified design significantly improves training efficiency and stability. Our model is trained on public available datasets, and experiments demonstrate that Direct3D-S2 not only surpasses state-of-the-art methods in generation quality and efficiency, but also enables training at 1024³ resolution using only 8 GPUs, a task typically requiring at least 32 GPUs for volumetric representations at 2563 resolution, thus making gigascale 3D generation both practical and accessible. Project page: https://www.neural4d.com/research/direct3d-s2.

## 1. Introduction

Generating high-quality 3D models directly from text or images offers significant creative potential, enabling rapid 3D content creation for virtual worlds, product prototyping, and various real-world applications. This capability has garnered increasing attention across domains such as gaming, virtual reality, robotics, and computer-aided design.

Recently, large-scale 3D generative models based on implicit latent representations have made notable progress. These methods leverage neural fields for shape representation, benefiting from compact latent codes and scalable generation capabilities. For instance, 3DShape2Vecset [47] pioneered diffusion-based shape synthesis by using a Variational Autoencoder (VAE) [14] to encode 3D shapes into a latent vecset, which can be decoded into neural SDFs or occupancy fields and rendered via Marching Cubes [24]. The latent vecset is then modeled with a diffusion process to generate diverse 3D shapes. CLAY [49] extended this pipeline with Diffusion Transformers (DiT) [30], while TripoSG [18] further improved fidelity through rectified flow transformers and hybrid supervision. However, implicit latent-based methods often rely on VAEs with asymmetric 3D representations, resulting in lower training efficiency that typically requires hundreds of GPUs.

Explicit latent methods have emerged as a compelling alternative to implicit ones, offering better interpretability, simpler training, and direct editing capabilities, while also adopting scalable architectures such as DiT [30]. For instance, Direct3D [39] proposes to use tri-plane latent representations to accelerate training and convergence. XCube [32] introduces hierarchical sparse voxel latent diffusion for 10243 sparse volume generation, but only restricted to millions of valid voxels, limiting the final output quality. Trellis [40] integrates sparse voxel representations of 2563 resolution, with the rendering supervision for the VAE training. In general, due to high memory demands, existing explicit latent methods are limited in output resolution. Scaling to 10243 with sufficient latent tokens and valid voxels remains challenging, as the quadratic cost of full attention in DiT renders high-resolution training computationally prohibitive.

To address the challenge of high-resolution 3D shape generation, we propose Direct3D-S2, a unified generative framework that utilizes sparse volumetric representations. At the core of our approach is a novel Spatial Sparse Attention (SSA) mechanism, which substantially improves the scalability of diffusion transformers in high-resolution 3D shape generation by selectively attending to spatially important tokens via learnable compression and selection modules. Specifically, we draw inspiration from the key principles of Native Sparse Attention (NSA) [46], which integrates compression, selection, and windowing to identify relevant tokens based on global-local interactions. While

NSA is designed for structurally organized 1D sequences, it is not directly applicable to unstructured, sparse 3D data. To adapt it, we redesign the block partitioning to preserve 3D spatial coherence and revise the core modules to accommodate the irregular nature of sparse volumetric tokens. This enables efficient processing of large token sets within sparse volumes. We implement a custom Triton [37] GPU kernel for SSA, achieving a 3.9× speedup in the forward pass and a 9.6× speedup in the backward pass compared to FlashAttention-2 at 10243 resolution.

Our framework also includes a VAE that maintains a consistent sparse volumetric format across input, latent, and output stages. This unified design eliminates the need for cross-modality translation, commonly seen in previous methods using mismatched representations such as point cloud input, 1D vector latent, and dense volume output, thereby improving training efficiency, stability, and geometric fidelity. After the VAE training, the DiT with the proposed SSA will be trained on the converted latents, enabling scalable and efficient high-resolution 3D shape generation.

Extensive experiments demonstrate that our approach successfully achieves high-quality and efficient gigascale 3D generation, a milestone previously unattainable by explicit 3D latent diffusion methods. Compared to prior native 3D diffusion techniques, our model consistently generates highly detailed 3D shapes while considerably reducing computational costs. Notably, Direct3D-S2 requires only 8 GPUs to train on public datasets [8, 9, 20] at a resolution of 10243, in stark contrast to prior state-of-the-art methods, which typically require 32 or more GPUs even for training at 2563 resolution.

## 2. Related work

#### 2.1. Multi-view Generation and 3D Reconstruction

Large-scale 3D generation has been advanced by methods such as [16, 22, 23, 42], which employ multi-view diffusion models [38] trained on 2D image prior models like Stable Diffusion [33] to generate multi-view images of 3D shapes. These multi-view images are then used to reconstruct 3D shapes via generalized sparse-view reconstruction models. Follow-up works [21, 27, 36, 43, 48] further improve the quality and efficiency of reconstruction by incorporating different 3D representations. Despite these advances, these methods still face challenges in maintaining multi-view consistency and shape quality. The synthesized images may fail to faithfully represent the underlying 3D structure, which could result in artifacts and reconstruction errors. Another limitation is the reliance on rendering-based supervision, such as Neural Radiance Fields (NeRF) [28] or DMTet [34]. While this avoids the need for direct 3D supervision (e.g., meshes), it adds significant complexity and computational overhead to the training process. Rendering-

based supervision can be slow and costly, especially when scaled to large datasets.

#### 2.2. Large Scale 3D Latent Diffusion Model

Motivated by recent advances in Latent Diffusion Models (LDMs) [33] in 2D image generation, several methods have extended LDMs to 3D shape generation. These approaches broadly fall into two categories: vecset-based methods and voxel-based methods. Implicit vecset-based methods, such as 3DShape2Vecset [47], Michelangelo [50], CLAY [49], and CraftsMan3D [17], represent 3D shapes using latent vecset and reconstruct meshes through neural SDFs or occupancy fields. However, implicit methods are typically constrained by the size of vecset: larger vecset leads to more complex mappings to the 3D shape and requires longer training times. In contrast, voxel-based methods, such as XCube [32], Trellis [40], and more recent works [11, 45], employ voxel grids as latent representations, providing more interpretability and easier training. Nevertheless, voxel-based methods face limitations in latent resolution due to cubic growth in GPU memory requirements and high computational costs associated with attention mechanisms. To address this issue, our work specifically targets reducing the computational overhead of attention mechanisms, thereby enabling the generation of highresolution voxel-based latent representations that were previously infeasible.

#### 2.3. Efficient Large Tokens Generation

Generating large tokens efficiently is a challenging problem, especially for high-resolution data. Native Sparse Attention (NSA) [46] addresses this by introducing adaptive token compression that reduce the number of tokens involved in attention computation, while maintaining performance comparable to full attention. NSA has been successfully applied to large language models [31, 46] and video generation [35], showing significant reductions in attention cost. In this paper, we extend token compression to 3D data and propose a new Spatial Sparse Attention (SSA) mechanism. SSA adapts the core ideas of NSA but modifies the block partitioning strategy to respect 3D spatial coherence. We also redesign the compression, selection, and window modules to better fit the properties of sparse 3D token sets. Another line of work, such as linear attention [13], reduces attention complexity by approximating attention weights with linear functions. Variants of this technique have been applied in image [41, 53] and video generation [26] to improve efficiency. However, the absence of non-linear similarity can lead to a significant decline in the performance of the model.

## 3. Sparse SDF VAE

While variational autoencoders (VAEs) have become the cornerstone of 2D image generation by compressing pixel representations into compact latent spaces for efficient diffusion training, their extension to 3D geometry faces fundamental challenges. Unlike images with standardized pixel grids, 3D representations lack a unified structure, such as meshes, point clouds, and implicit fields, each requires specialized processing. This fragmentation forces existing 3D VAEs into asymmetric architectures with compromised efficiency. For instance, prominent approaches [6, 49, 51] based on vecset [47] encode the input point cloud into a vector set latent space before decoding it into SDF field, while Trellis [40] and XCube [32] rely on differentiable rendering or neural kernel surface reconstruction [12] to bridge their latent spaces to usable meshes. These hybrid pipelines introduce computational bottlenecks and geometric approximations that limit their scalability to highresolution 3D generation. In this paper, we propose a fully end-to-end sparse SDF VAE that employs a symmetric encoding-decoding network to encode high-resolution sparse SDF volumes into a sparse latent representation, substantially improving training efficiency while maintaining geometric precision.

Given a mesh represented as a signed distance field (SDF) volume V with resolution R3 (e.g., 10243), the SSVAE first encodes it into a latent representation z = E(V ), then reconstructs the SDF through the decoder V˜ = D(z). Direct processing of dense R3 SDF volumes proves computationally prohibitive. To address this, we strategically focus on valid sparse voxels where absolute SDF values fall below threshold τ:

V = {(xi,s(xi)) |s(xi)| < τ}|iV=1| , (1) where s(xi) denotes the SDF value at position xi.

#### 3.1. Symmetric Network Architecture

Our fully end-to-end SDF VAE framework adopts a symmetric encoder-decoder network architecture, as illustrated in the upper half of Figure 2. Specifically, the encoder employs a hybrid framework combining sparse 3D convolution networks and transformer networks. We first extract local geometric features through a series of residual sparse 3D CNN blocks interleaved with 3D mean pooling operations, progressively downsampling the spatial resolution. We then process the sparse voxels as variable-length tokens and utilize shifted window attention to capture local contextual information between the valid voxels. Inspired by Trellis [40], the feature of each valid voxel is augmented with positional encoding based on its 3D coordinates before being fed into 3D shift window attention layers. This hybrid design outputs sparse latent representations at reduced

###### Reconstruction Loss

###### SS-VAE

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

SS-VAE Encoder

###### SS-VAE Decoder

3D Assets SDF Multi-Resolution Latents

Multi-Resolution Sparse SDF

Multi-Resolution Sparse SDF

###### SS-DiT

Marching Cube

[Figure 9]

Decoding

Image Condition

DINOv2

[Figure 10]

[Figure 11]

[Figure 12]

MeshExtraction

SS-DiT

MarchingCube

Block

CrossAttention

Spatial Sparse Attention

Norm+Linear

Norm+Linear

Norm+MLP

Noised Latents Latents

Sparse SDF

3D Mesh

- Figure 2. The framework of our Direct3D-S2. We propose a fully end-to-end sparse SDF VAE (SS-VAE), which employs a symmetric encoder-decoder network to efficiently encode high-resolution sparse SDF volumes into sparse latent representations z. Then we train an image-conditioned diffusion transformer (SS-DiT) based on z, and design a novel Spatial Sparse Attention (SSA) mechanism that substantially improves the training and inference efficiency of the DiT.

resolution (Rf )3, where f denotes the downsampling factor. The decoder of our SS-VAE adopts a symmetric structure

with respect to the encoder, leveraging attention layers and sparse 3D CNN blocks to progressively upsample the latent representation and reconstruct the SDF volume V˜.

- 3.2. Training Losses

#### 3.3. Multi-resolution Training

To enhance training efficiency and enable our SS-VAE to encode meshes across varying resolutions, we utilize the multi-resolution training paradigm. Specifically, during each training iteration, we randomly sample a target resolution from the candidate set {2563,3843,5123,10243}, then trilinearly interpolate the input SDF volume to the selected resolution before feeding it into the SS-VAE.

The decoded sparse voxels V˜ contain both the input voxels V˜in and additional valid voxels V˜extra. We enforce supervision on the SDF values across all these spatial positions. To enhance geometric fidelity, we impose additional supervision on the active voxels situated near the sharp edges of the mesh, specifically in regions exhibiting high-curvature variations on the mesh surface. Moreover, the term of KLdivergence regularization is imposed on the latent representation z to constrain excessive variations in the latent space. The overall training objective of our SS-VAE is formulated as:

## 4. Spatial Sparse Attention and DiT

Through our SS-VAE framework, 3D shapes can be encoded into latent representations z. Following a methodology analogous to Trellis [40], we serialize the latent tokens z and train a rectified flow transformer-based 3D shape generator conditioned on input images. To ensure efficient generation of high-resolution meshes, we propose spatial sparse attention that substantially accelerates both training and inference processes. Furthermore, we introduce a sparse conditioning mechanism to extract the foreground region of the input images, thereby reducing the number of conditioning tokens. The architecture of the DiT is illustrated in the lower half of Figure 2.

1 |V˜c|

s(x)−s˜(x) 22, c ∈ {in,ext,sharp},

Lc =

(x, s˜(x))∈V˜c

(2) Ltotal =

λc Lc + λKL LKL, (3)

c

where λin, λext, λsharp and λKL denote the weight of each term.

Sparse 3D Compression

Compressed

3DPooling

Attention Selected

Selection Compress

3DConv

Query

Linear

Spatial Blockwise Selection

GatedOutput

Attention

Top-n

Key & Value Linear

Sparse 3D Window

Attention

Window

Window

Blocks Inactive Blocks Attention Score

Input Tokens

Figure 3. The framework of our Spatial Sparse Attention (SSA). We partition the input tokens into blocks based on their 3D coordinates, and then construct key-value pairs through three distinct modules. For each query token, we utilize sparse 3D compression module to capture global information, while the spatial blockwise selection module selects important blocks based on compression attention scores to extract fine-grained features, and the sparse 3D window module injects local features. Ultimately, we aggregate the final output of SSA from the three modules using predicted gate scores.

#### 4.1. Spatial Sparse Attention

Given input tokens q, k, v ∈ RN×d, where N denotes the token length, and d represents the head dimension, the standard full attention is formulated as:

###### ot = Attn qt,k,v

N

pt,i vi N

, t ∈ [0,N), (4)

=

i=1

pt,j

j=1

pt,j = exp

q⊤t kj

√

d

. (5)

As the resolution of SS-VAE escalates, the length of input tokens grows substantially, reaching over 100k at a resolution of 10243, leading to prohibitively low computational efficiency in attention operations. Inspired by NSA (Native Sparse Attention) [46], we proposes Spatial Sparse Attention mechanism, which partitions key and value tokens into spatially coherent blocks based on their geometric relationships and performs blockwise token selection to achieve significant acceleration.

A naive implementation involves treating latent tokens z as a 1D sequence and partitioning it into fixed-length blocks based on token indices, analogous to NSA. However, this approach suffers from two critical limitations: On the one hand, tokens within the same block may not be spatially adjacent in 3D space, despite sharing contiguous indices. On the other hand, due to the sparse voxel structure, blocks with identical indices across different samples occupy divergent spatial regions. These issues collectively lead to unstable training convergence. To resolve these challenges, we propose partitioning blocks based on 3D coordinates. As illustrated in Figure 3, we divide the 3D space into subgrids of size m3, where active tokens from sparse voxels residing in the same subgrid are grouped into one block. Our Spatial Sparse Attention comprises three core modules: sparse 3D compression, spatial blockwise selection, and sparse 3D window. The attention computation proceeds as follows:

ot = ωtcmpAttn(qt,kcmpt ,vtcmp)

+ ωtslcAttn(qt,kslct ,vtslc)

+ ωtwinAttn(qt,kwint ,vtwin),

(6)

where kt and vt represent the selected key and value tokens

in each module for query qt, respectively. ωt is the gating score for each module, obtained by applying a linear layer followed by a sigmoid activation to the input features.

Sparse 3D Compression. After partitioning input tokens into spatially coherent blocks based on their 3D coordinates, we leverage a compression module to extract block-level representations of the input tokens. Specifically, we first incorporate intra-block positional encoding for each token within a block of size m3cmp, then employ sparse 3D convolution followed by sparse 3D mean pooling to compress the entire block:

kcmpt = δ(kt + PE(kt)), (7)

where kcmpt denotes the block-level key token, PE(·) is absolute position encoding, and δ(·) represents operations of sparse 3D convolution and sparse 3D mean pooling. The sparse 3D compression module effectively captures blocklevel global information while reducing the number of tokens, thereby enhancing computational efficiency.

Spatial Blockwise Selection. The block-level representations only contain coarse-grained information, necessitating the retention of token-level features to enhance the fine details in the generated 3D shapes. However, the excessive number of input tokens leads to computationally inefficient operations if all tokens are utilized. By leveraging the sparse 3D compression module, we compute the attention scores scmp between the query q and each compression block, subsequently selecting all tokens within the top-k blocks exhibiting the highest scores. The resolution mslc of the selection blocks must be both greater than and divisible by the resolution mcmp of the compression blocks. The relevance score sslct for a selection block is aggregated from its constituent compression blocks. GQA (Grouped-Query Attention) [4] is employed to further improve computational efficiency, the attention scores of the shared query heads within each group are accumulated as follows:

hs

scmpt,h ,i, (8)

sslct =

i∈Bcmp

h=1

where Bcmp denotes the set of compression blocks within the selection block, and hs represents the number of shared heads within a group. The top-k selection blocks with the highest sslct scores are selected, and all tokens contained within them are concatenated to form kslct and vtslc, which are used to compute the spatial blockwise selection attention.

We implement the spatial blockwise selection attention kernel using Triton [37], with two key challenges arising from the sparse 3D voxel structures: 1) the number of tokens varies across different blocks, and 2) tokens within

the same block may not be contiguous in HBM. To address these, we first sort the input tokens based on their block indices, then compute the starting index C of each block as kernel input. In the inner loop, C dynamically governs the loading of corresponding block tokens. The complete procedure of forward pass is formalized in Algorithm 1.

Sparse 3D Window. In addition to sparse 3D compression and spatial blockwise selection modules, we further employ an auxiliary sparse 3D window module to explicitly incorporate localized feature interactions. Drawing inspiration from Trellis [40], we partition the input token-containing voxels into non-overlapping windows of size m3win. For each token, we formulate its contextual computation by dynamically aggregating active tokens within the corresponding window to form kwint and vtwin, followed by localized self-attention calculation exclusively over this constructed token subset.

Through the modules of sparse 3D compression, spatial blockwise selection, and sparse 3D window, corresponding key-value pairs are constructed. Subsequently, attention calculations are performed for each module, and the results are aggregated and weighted according to gate scores to produce the final output of the spatial sparse attention mechanism.

#### 4.2. Sparse Conditioning Mechanism

Existing image-to-3D models [17, 39, 49] typically employ DINO-v2 [29] to extract pixel-level features from conditional images, followed by cross-attention operation with noisy tokens to achieve conditional generation. However, for a majority of input images, more than half of the regions consist of background, which not only introduces additional computational overhead but may also adversely affect the alignment between the generated meshes and the conditional images. To mitigate this issue, we propose a sparse conditioning mechanism that selectively extracts and processes sparse foreground tokens from input images for cross-attention computation. Formally, given an input image I, the sparse conditioning tokens c are computed as follows:

c = Linear(f(EDINO(I))) + PE(f(EDINO(I))), (9)

where EDINO is the DINO-v2 encoder, f(·) denotes the operation of extracting the foreground tokens based on the mask, PE(·) is the absolute position encoding, and Linear(·) represents a linear layer. Then we perform cross attention using the finalized sparse conditioning tokens c and the noisy tokens.

#### 4.3. Rectified Flow

We employ rectified flow objective [10, 19] to train our generative model. Rectified flow defines forward process as lin-

Algorithm 1 Spatial Blockwise Selection Attention Forward Pass Require: q ∈ RN×(h

kv×d, number of key/value heads hkv, number of the shared heads hs, number of the selected blocks T, indices of the selected blocks I ∈ RN×h

kv×hs)×d, k ∈ RN×h

kv×d and v ∈ RN×h

kv×T, the number of divided key/value blocks Nb, index of the first token in each block C ∈ RN

b+1, block size Bk.

- 1: Divide the output o ∈ RN×(h

kv×hs)×d into (N,hkv) blocks, each of size hs×d. Divide the logsumexp l ∈ RN×(h

kv×hs) into (N,hkv) blocks, each of size hs.

- 2: Sort all tokens within q, k and v according to their respective block indices.
- 3: for t = 1 to N do
- 4: for h = 1 to hkv do
- 5: Initialize ot,h = (0)h

s×d ∈ Rh

s×d, logsumexp lt,h = (0)h

s ∈ Rh

s, and mt,h = (−inf)h

s ∈ Rh

s.

- 6: Load qt,h ∈ Rh

s×d, It,h ∈ RT from HBM to on-chip SRAM.

- 7: for j = 1 to T do
- 8: Load the starting token index bs = C(I

(j)

t,h) and ending token index be = C(I

(j)

t,h)+1 − 1 of the I(t,hj)th block from HBM to on-chip SRAM.

- 9: for i = bs to be by Bk do
- 10: Load ki, vi ∈ RB

k×d from HBM to on-chip SRAM.

- 11: Compute s(t,hi) = qt,hkTi ∈ Rh

s×Bk.

- 12: Compute m(t,hi) = max(mt,h,rowmax(st,h(i))) ∈ Rh

s.

- 13: Compute p(t,hi) = es

(i)

t,h−m(t,hi) ∈ Rh

s×Bk.

- 14: Compute ot,h = em

t,h−m(t,hi) ot,h + pt,h(i)vi.

- 15: Compute lt,h = m(t,hi) + log(el

t,h−m(t,hi) + rowsum(p(t,hi))), mt,h = mt,h(i).

- 16: end for
- 17: end for
- 18: Compute ot,h = em

t,h−lt,hot,h.

- 19: Write ot,h and lt,h to HBM as the (t,h)-th block of o and l, respectively.
- 20: end for
- 21: end for
- 22: Return the output o and the logsumexp l.

ear trajectory between data distribution and standard normal distribution:

x(t) = (1 − t)x0 + tϵ, (10)

where ϵ is the noise, and t denotes the timestep. Our generative model is trained to predict the velocity field from noisy samples to the data distribution. The training loss is formulated using conditional flow matching, formulated as follows:

0,ϵ∥vθ(xt,c,t) − (ϵ − x0)∥22, (11) where vθ is the neural networks.

LCFM = Et,x

## 5. Experiments

#### 5.1. Datasets

Our Direct3D-S2 is trained on publicly available 3D datasets including Objaverse [9], Objaverse-XL [8], and ShapeNet [5]. Due to the prevalence of low-quality meshes

in these collections, we curated approximately 452k 3D assets through rigorous filtering for training. Following prior approach [49] in geometry processing, we first convert the original non-watertight meshes into watertight ones, then compute ground-truth SDF volumes that serve as both input to and supervision for our SS-VAE. For training our image-conditioned DiT, we render 45 RGB images per mesh at 1024 × 1024 resolution with random camera parameters. The camera configuration space is defined as follows: elevation angles ranging from 10◦ to 40◦, azimuth angles spanning [0◦,180◦], and focal lengths varying between 30mm and 100mm. To rigorously evaluate the geometric fidelity of meshes generated by Direct3D-S2, we established a challenging benchmark comprising highly detailed images sourced from professional communities including Neural4D [3], Meshy [2], and CivitAI [1]. The quantitative assessment employs ULIP-2 [44], Uni3D [52] and OpenShape [20] metrics to measure shape-image alignment between generated meshes and conditional input images, enabling systematic comparison with state-of-the-art 3D generation methods.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

##### Input Trellis Hunyuan3D 2.0 TripoSG Hi3DGen Ours

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

Figure 4. Qualitative comparisons between other image-to-3D methods and our approach.

#### 5.2. Implementation Details

VAE. We utilize active voxels from volumes with SDF values less than τ = 1281 as inputs to the SS-VAE. The downsampling factor f for the encoder is set to 8, and the channel

dimension of the latent representation z is configured to 16. The weights for the various losses are set as: λin = 1.0, λext = 1e − 1, λsharp = 1.0, and λKL = 1e − 3. We employ the AdamW [25] optimizer with an initial learning rate

Trellis Huanyuan3D 2.0 TripoSG Hi3DGen Ours

- Table 1. Training configurations for DiT at four voxel resolutions. Res., NT, LR, BS, and TT denote resolution, number of tokens, learning rate, batch size, and total training time, respectively.

Res. NT LR BS TT

2563 ≈2058 1e-4 8 × 8 2 days 3843 ≈5510 1e-4 8 × 8 2 days 5123 ≈10655 5e-5 8 × 8 2 days

10243 ≈45904 2e-5 2 × 8 1 day

- Table 2. Quantitative comparisons of meshes generated by different methods in the image-to-3D task.

4.21 4.17

3.52

3.33

3.25

3.07

3.04

2.93

2.81

2.78

- 2

- 2.5
- 3

3.5

4

- 4.5

Image Consistency Overall Quality

Figure 5. User Study for Image-to-3D Generation.

Methods ULIP-2 ↑ Uni3D ↑ OpenShape ↑

state-of-the-art image-to-3D approaches. Our systematic evaluation employs three multimodal models: ULIP-2 [44], Uni3D [52], and OpenShape [20], to assess the similarity between the generated meshes and input images. The quantitative results are reported in Table 2, where it is evident that our Direct3D-S2 outperforms the other approaches across three metrics, indicating that the meshes produced by our Direct3D-S2 achieve better alignment with the input images. Moreover, we present qualitative comparisons in Figure 4. Although the other methods generate overall satisfactory results, they struggle to capture finer structures due to resolution limitations, as illustrated by the railings of the house and surrounding branches of trees in the first row. In contrast, thanks to our proposed SSA mechanism, our Direct3D-S2 is capable of generating high-resolution meshes, achieving superior quality even for these intricate details. We provide more qualitative comparisons with both open-source and closed-source approaches in Figure 12.

Trellis [40] 0.2825 0.3755 0.1732 Hunyuan3D 2.0 [51] 0.2535 0.3738 0.1699 TripoSG [18] 0.2626 0.3870 0.1728 Hi3DGen [45] 0.2725 0.3723 0.1689 Ours 0.3111 0.3931 0.1752

of 1e − 4. To enhance training efficiency, we first conduct multi-resolution training using SDF volumes at three resolutions of {2563,3843,5123} over a period of one day on 8 A100 GPUs, with a batch size of 4 per GPU. Subsequently, we fine-tune the SS-VAE for one additional day at a resolution of 10243 with a learning rate of 1e − 5 with a batch size of 1 per GPU.

DiT. Our SS-DiT comprises 24 layers of DiT blocks with a hidden dimension of 1024. We employ Grouped-Query Attention (GQA) [4] with a group number set to 2, where each group contains 16 attention heads. The hidden dimension of each head is configured as 32. For the spatial sparse attention (SSA) mechanism, we configure the resolution of the compression blocks to mcmp = 4 , the resolution of the selection blocks to mslc = 8, and the size of the sparse 3D windows mwin = 8. We utilize DINO-v2 Large [29] to extract features from conditional images, with input images having a resolution of 518×518. For the DiT, we implement a progressive training strategy that gradually increases the resolution from 2563 to 10243 to accelerate convergence. Table 1 presents the average number of latent tokens, learning rate, batch size, and training duration settings at different resolutions. We employ the AdamW optimizer and trained the model for a total of 7 days on 8 A100 GPUs. For the 10243 resolution, we further filtered 68k high-fidelity 3D assets for training. Additionally, similar to Trellis [40], we trained an extra DiT to predict the indices of the sparse latent tokens z, which took 7 days on 8 A100 GPUs.

In addition, we conducted a user study with 40 participants evaluating 75 unfiltered meshes generated by our Direct3D-S2 and other image-to-3D methods. Participants scored each output using two criteria: image consistency and overall geometric quality, with scores ranging from 1 (poorest) to 5 (excellent). As shown in Figure 5, Our Direct3D-S2 demonstrates statistically superiority over other approaches across both evaluation metrics.

## 6. Comparison of VAE

To validate the reconstruction quality of our SS-VAE, we curated a challenging validation set from the Objaverse [9] dataset, comprising meshes with complex geometric structures. Qualitative comparisons with competing methods are shown in Figure 6. It can be observed our SS-VAE achieves superior reconstruction accuracy at 5123 resolution, and demonstrates markedly improved performance on complex geometries at 10243 resolution. Notably, thanks to our fully end-to-end SDF reconstruction framework, SSVAE requires only 2 days of training on 8 A100 GPUs, significantly fewer than competing methods that typically de-

#### 5.3. Quantitative and Qualitative Comparisons

To empirically validate the effectiveness of our Direct3D-S2 framework, we conduct comprehensive experiments against

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

###### Trellis XCube Dora Ours (ퟓퟏ  ) Ours (ퟏퟎ   ) GT

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

Figure 6. Qualitative comparisons of VAE reconstruction results. Note that we used a latent token length of 4096 during the inference of Dora [6].

mand at least 32 GPUs for equivalent training durations.

important regions globally, resulting in a notable improvement in mesh quality. We also observed that not utilizing the window (cmp+slc) did not result in a significant drop in model performance, but slowed convergence, demonstrating that local feature interaction contributes to more stable training and enhances convergence speed.

#### 6.1. Ablation Studies

Image-to-3D Generation in Different Resolution. We present the generation results of our Direct3D-S2 across four resolutions {2563,3843,5123,10243} in Figure 8. The results demonstrate that increasing resolution progressively improves mesh quality. At lower resolutions 2563 and 3843, the generated meshes exhibit limited geometric details and misalignment with input images. At 5123 resolution, the meshes display enhanced high-frequency geometric details. Further increasing the resolution to 10243 yields meshes with sharper edges and improved alignment with input image details.

Runtime of Different Attention Mechanisms. We implemented a custom Triton [37] GPU kernel for SSA. And we compare the forward and backward execution times of our SSA with those of FlashAttention-2 [7] across various number of tokens, using the implementation from Xformers [15] for FlashAttention-2. The comparison results are shown in Figure 7, which indicate that our SSA achieves comparable speeds to FlashAttention-2 when the number of tokens is low; however, as the number of tokens increases, the speed advantage of our SSA becomes more pronounced. Specifically, when the number of tokens reaches 128k, the forward and backward speeds of our SSA are 3.9× and 9.6× faster than those of FlashAttention-2, respectively, demonstrating the efficiency of our proposed SSA.

Effect of Each Module in SSA. We validated the effect of the three modules in SSA at resolution 5123, with the results presented in Figure 9. When using only the sparse 3D window module (win), the generated meshes exhibited detailed structures but suffered from surface irregularities due to the lack of global context modeling. Introducing the sparse 3D compression module (win+cmp) showed minimal performance changes, which is reasonable as this module primarily serves to obtain the attention scores for the blocks. After incorporating the spatial blockwise selection module (win+cmp+slc), the model can focus on the most

Effectiveness of SSA. We conduct ablation studies to validate the robustness of SSA. Given the insufficient geometric fidelity at 2563/ 3843 resolutions, which do not adequately reflect the model’s precision, and prohibitive computational

###### Forward Time

| |0.7 1.4 2.3 2.5 3.9<br><br>Flash Attention 2 Spatial Sparse Attention<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

800 600 400 200

Times(ms)

0

8K 16K 32K 64K 128K

Number of Tokens

###### Backward Time

| |1.4 2.5 3.6 6.3 9.6<br><br>Flash Attention 2 Spatial Sparse Attention<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

2000

Times(ms)

1600

1200

800

400

0

8K 16K 32K 64K 128K

Number of Tokens

Figure 7. Comparison of the forward and backward time of our SSA and FlashAttention-2.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Input ퟐ   ퟑ     ퟐ  ퟎퟐ 

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

- Figure 8. The visualization results of our Direct3D-S2 for image-to-3D generation across four resolutions: {2563, 3843, 5123, 10243}.

costs at 10243 resolution, we perform experiments at 5123 resolution. We establish three comparative configurations: 1) Full attention: directly training the DiT with full attention proves to be inefficient. Therefore, following Trellis’ latent packing strategy [40], we group latent tokens within

- 23 local regions to reduce the number of tokens before feeding them into the DiT blocks. 2) NSA: process latent tokens as 1D sequences with fixed-length block partitioning, disregarding spatial coherence. 3) Our proposed SSA. The qualitative results are illustrated in Figure 10. It is evident that the full attention variant produces meshes with highfrequency surface artifacts, attributed to its forced packing operation that disrupts local geometric continuity. The

NSA implementation exhibits training instability due to positional ambiguity in block partitioning, resulting in less smooth meshes. In contrast, our SSA not only preserves the details of the meshes, but also yields a smoother and more organized surface, thereby demonstrating the effectiveness of our proposed SSA mechanism.

Effect of Sparse Conditioning Mechanism. We perform ablation experiments to validate the effect of the sparse conditioning mechanism at 5123 resolution. As demonstrated in Figure 11, the exclusion of non-foreground conditioning tokens through sparse conditioning enables the generated meshes to achieve notably better alignment with the input images.

### Input win win+cmp cmp+slc win+cmp+slc

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

- Figure 9. Ablation studies for the three modules of SSA at resolution 5123, where win, cmp, and slc denote the sparse 3D window, sparse 3D compression, and spatial blockwise selection modules, respectively.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Input Full Attention NSA SSA(Ours)

[Figure 127]

[Figure 128]

- Figure 10. Ablation studies of our proposed SSA mechanism.

[Figure 129]

[Figure 130]

###### Input w/o sparse conditioning w/ sparse conditioning

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Figure 11. Ablation studies for sparse conditioning mechanism.

- 7. Conclusion In this work, we presented a novel framework for highresolution 3D shape generation, dubbed Direct3D-S2. The key contribution of our approach is the design of Spatial Sparse Attention (SSA) mechanism, which significantly accelerates the training and inference speed of DiT. The integration of fully end-to-end symmetric sparse SDF VAE further enhances training stability and efficiency. Extensive experiments demonstrate that our Direct3D-S2 outperforms existing state-of-the-art image-to-3D methods in generation

quality, while requiring only 8 GPUs for training.

## 8. Limitations

Our proposed spatial sparse attention achieves significant speed improvements over FlashAttention-2. However, the forward pass exhibits a notably smaller acceleration ratio compared to the backward pass. This discrepancy primarily stems from the computational overhead introduced by topk sorting operations during the forward pass. We acknowledge this limitation and will prioritize optimizing these operations in future work.

## References

- [1] Civitai. https://civitai.com/. 7
- [2] Meshy. https://www.meshy.ai/. 7
- [3] Neural4d. https://www.neural4d.com/. 7
- [4] Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245,

2023. 6, 9

- [5] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. 7
- [6] Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, and Ping Tan. Dora: Sampling and benchmarking for 3d shape variational auto-encoders. arXiv preprint arXiv:2412.17808,

2024. 3, 10

- [7] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 10
- [8] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. In NeurIPS, 2023. 2, 7
- [9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 7, 9
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 6
- [11] Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology 3d shape modeling. arXiv preprint arXiv:2503.21732, 2025. 3
- [12] Jiahui Huang, Zan Gojcic, Matan Atzmon, Or Litany, Sanja Fidler, and Francis Williams. Neural kernel surface reconstruction. In CVPR, 2023. 3
- [13] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, 2020. 3
- [14] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 2
- [15] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022. 10

- [16] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In ICLR, 2024. 2
- [17] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 3, 6
- [18] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025. 2, 9
- [19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 6
- [20] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. In NeurIPS, 2023. 2, 7, 9
- [21] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In CVPR, 2024. 2
- [22] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. In NeurIPS, 2024. 2
- [23] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In CVPR,

2024. 2

- [24] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353. 1998. 2
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 8
- [26] Kaiyue Lu, Zexiang Liu, Jianyuan Wang, Weixuan Sun, Zhen Qin, Dong Li, Xuyang Shen, Hui Deng, Xiaodong Han, Yuchao Dai, et al. Linear video transformer with feature fixation. arXiv preprint arXiv:2210.08164, 2022. 3
- [27] Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, and Yao Yao. Direct2. 5: Diverse text-to-3d generation via multiview 2.5 d diffusion. In CVPR, 2024. 2
- [28] B Mildenhall, PP Srinivasan, M Tancik, JT Barron, R Ramamoorthi, and R Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2
- [29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6, 9

- [30] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 2
- [31] Piotr Pie˛kos, Róbert Csordás, and Jürgen Schmidhuber. Mixture of sparse attention: Content-based learnable sparse attention via expert-choice routing. arXiv preprint arXiv:2505.00315, 2025. 3
- [32] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In CVPR, 2024. 2, 3
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [34] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. In NeurIPS,

2021. 2

- [35] Xin Tan, Yuetao Chen, Yimin Jiang, Xing Chen, Kun Yan, Nan Duan, Yibo Zhu, Daxin Jiang, and Hong Xu. Dsv: Exploiting dynamic sparsity to accelerate large-scale video dit training. arXiv preprint arXiv:2502.07590, 2025. 3
- [36] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In ECCV,

2024. 2

- [37] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019. 2, 6, 10
- [38] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023. 2
- [39] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. In NeurIPS, 2024. 2, 6
- [40] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 2, 3, 4, 6, 9, 11
- [41] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 3
- [42] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 2

- [43] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. In ECCV, 2024. 2
- [44] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong,

Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In CVPR,

- 2024. 7, 9

- [45] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. arXiv preprint arXiv:2503.22236,

2025. 3, 9

- [46] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardwarealigned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025. 2, 3, 5
- [47] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023. 2, 3
- [48] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In ECCV, 2024. 2
- [49] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 2, 3, 6, 7
- [50] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. In NeurIPS,

2023. 3

- [51] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 3, 9
- [52] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773,

2023. 7, 9

- [53] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. arXiv preprint arXiv:2405.18428, 2024. 3

[Figure 139]

###### Figure 12. More qualitative comparisons between other open-source image-to-3D methods and our approach. Best viewed with zoom-in.

[Figure 140]

###### Figure 13. Qualitative comparisons between closed-source commercial image-to-3D models and our approach. Note that for each closedsource model we use the default setting of their web app. Best viewed with zoom-in.

