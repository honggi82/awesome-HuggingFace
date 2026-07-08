arXiv:2506.07932v1[cs.GR]9Jun2025

# Squeeze3D: Your 3D Generation Model is Secretly an Extreme Neural Compressor

Rishit Dagli Yushi Guan Sankeerth Durvasula Mohammadreza Mofayezi Nandita Vijaykumar University of Toronto {rishit, guanyushi, sankeerth, mofayezi, nandita}@cs.toronto.edu

squeeze3d.github.io

## Abstract

We propose Squeeze3D, a novel framework that leverages implicit prior knowledge learnt by existing pre-trained 3D generative models to compress 3D data at extremely high compression ratios. Our approach bridges the latent spaces between a pre-trained encoder and a pre-trained generation model through trainable mapping networks. Any 3D model represented as a mesh, point cloud, or a radiance field is first encoded by the pre-trained encoder and then transformed (i.e. compressed) into a highly compact latent code. This latent code can effectively be used as an extremely compressed representation of the mesh or point cloud. A mapping network transforms the compressed latent code into the latent space of a powerful generative model, which is then conditioned to recreate the original 3D model (i.e. decompression). Squeeze3D is trained entirely on generated synthetic data and does not require any 3D datasets. The Squeeze3D architecture can be flexibly used with existing pre-trained 3D encoders and existing generative models. It can flexibly support different formats, including meshes, point clouds, and radiance fields. Our experiments demonstrate that Squeeze3D achieves compression ratios of up to 2187× for textured meshes, 55× for point clouds, and 619× for radiance fields while maintaining visual quality comparable to many existing methods. Squeeze3D only incurs a small compression and decompression latency since it does not involve training object-specific networks to compress an object.

## 1 Introduction

The rapid advancement of 3D data acquisition and representation technologies over the past decade has significantly expanded the availability and generation of high-resolution 3D content across various domains in different formats, including, meshes, point clouds, and radiance fields (which could be extracted from a NeRF [72] or a 3DGS [46]). The widespread use of 3D data necessitates the development of techniques that enable efficient transmission, storage, and processing of large-scale 3D representations. To this end, compression and the use of compressed representations for 3D data are of utmost importance, e.g., in streaming, autonomous navigation, digital twins, remote sensing.

A large body of research proposes techniques to compress meshes, point clouds, neural radiance fields (NeRFs) [72], and 3D Gaussian Splats (3DGS) [46]. These approaches aim to maximize the compression ratio while retaining reconstruction quality. For example, traditional mesh decimation techniques [87, 28, 58, 52] remain foundational for polygon reduction, but their reliance on handcrafted simplification rules limits their ability to preserve fine geometric details at extreme compression ratios. MPEG’s G-PCC and V-PCC standards [88, 61] use projection-based methods for point cloud compression; however, these approaches incur overheads for representing fine details

Preprint.

[Figure 1]

[Figure 2]

###### Mesh Point Cloud Radiance Field

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Original: 1016.18 MB Compressed: 0.4645 MB (2187.7x)

Original: 18.02 MB Compressed: 0.316 MB (57.0x)

Original: 2787.36 MB Compressed: 4.32 MB (645.2x)

[Figure 8]

Squeeze3D

Compressed (0.003 MB)

Ground Truth (6.11 MB)

Reconstructed

- Figure 1: We showcase extreme compression of 3D models while preserving perceptual quality. Top: Our method compresses a diverse collection of 3D models in various formats: meshes, point clouds, and radiance fields. Bottom: Detailed comparison between the original “Pikachu” model (6.11 MB) and the reconstruction after compression. The object was compressed to merely 0.003 MB.

and packing. Prior works also propose a range of compression methods for NeRFs [56, 22, 100, 74] or 3DGS models [23, 51]. Several works propose autoencoder-style networks that compress 3D models into small latent vectors [121, 30, 129, 10]. Usually, the compression ratios achieved by these methods are of the order of 100x for meshes and the order of 10x for point clouds, but typically much lower.

Our goal is to develop a framework for extreme compression of 3D data stored in any format while retaining high visual quality. Recent years have seen significant and continued advances and development of powerful generative models. In this work, we aim to leverage the implicit prior knowledge learnt by the powerful 3D generative models [115, 44] to enable extreme compression ratios. Recent works also propose techniques to leverage generative models for 3D compression [122, 17, 121, 107, 123, 8, 84] and in one case achieves extreme compression for meshes [122]. However, these approaches require training specialized encoders and generative models for a single 3D format. In contrast, our goal is to flexibly use existing encoders and generator models that provide adaptability as encoders/generative models evolve and flexibility across 3D formats.

We propose Squeeze3D, a compression framework, that generates a highly compressed latent vector that can be used to recreate the original 3D data using an existing pre-trained generative model. Squeeze3D comprises three key components: (1) the input 3D data is encoded with a pre-trained encoder. This allows us to extend Squeeze3D to other 3D encoders. (2) We train two small neural networks that we call forward mapping network and reverse mapping network. The forward mapping network maps the encoded representations into an extremely compressed latent space. The reverse mapping network converts the code from the compressed latent space to the latent space of the generative model. (3) We use a pre-trained generation model to generate the original 3D data using the code generated by the reverse mapping network. Squeeze3D can be flexibly implemented with any pre-trained encoder and generative model.

The forward and reverse mapping networks are trained for any given encoder-generator pair. We first artificially generate a 3D dataset via random prompts to the generator model. This 3D dataset is encoded using the pre-trained encoder. The set of latents produced from the pre-trained encoders (training inputs) and their corresponding latents from the pre-trained generator (ground truth) are used to train the forward and reverse mapping networks. We propose a loss function that minimizes redundant information in the compressed latent space.

Squeeze3D can be flexibly applied to 3D data in different formats. We implement and evaluate our method for mesh, point cloud, and radiance field compression using 3 existing encoders and 5 existing generative models. We demonstrate that our method achieves significantly higher or on-par compression ratios than any existing compression technique for meshes, point clouds, and radiance fields with reconstruction quality that is on par with many prior approaches. We demonstrate a compression ratio of 2187× for a subset of the Objaverse [20] dataset, 55× on a subset of ShapeNet [7], 614.9× on a collection of radiance fields [39]. While Squeeze3D expectedly cannot achieve state-of-the-art reconstruction quality, we qualitatively show that it is able to retain high visual quality.

Contributions. (1) To the best of our knowledge, this is the first framework that leverages preexisting pre-trained generative models to enable extreme compression of 3D data.

- (2) We demonstrate the feasibility of establishing correspondences between disparate latent manifolds originating from neural architectures with fundamentally different structures, optimization objectives, and training distributions.
- (3) We evaluate Squeeze3D for mesh, point cloud, and radiance field compression and demonstrate that generative models are a promising approach for extreme compression of 3D models. Squeeze3D can be flexibly extended to different encoders, generative models, and 3D formats.

## 2 Related Works

### 2.1 Compressing Explicit 3D Representations

Classical compression techniques for meshes include approaches that reorder the structure of triangles and faces in the mesh to enable compressed encodings of elements based on their local structure and perform quantization [19, 105, 86, 106, 24]. Most of these techniques are lossless and thus achieve high fidelity, however, they are inherently limited in their ability to significantly reduce file sizes since they preserve all details of the mesh representation. To overcome the limitations of lossless compression, lossy techniques have emerged as popular alternatives. Geometry simplification methods aim to reduce the number of polygons in a mesh while retaining as much of the original structure as possible [52, 28, 98, 99]. Extensions of these methods have incorporated surface intrinsic properties, such as the mesh Laplacian, as a basis for simplification [52] or error metrics [28, 15] and couple this with entropy coding [49]. Recently, there have also been approaches that couple these with learned models [79].

Surface Upsampling. Traditional subdivision algorithms [130, 6, 64] refine coarse meshes by splitting polygonal faces into finer elements, often paired with displacement mapping [37] to enhance detail. However, these methods rely on hard-coded priors and fixed polynomial interpolations, which can overly smooth the reconstructed geometry and fail to capture intricate details. Neural approaches [34, 62] address these limitations by embedding geometric information into learnable parameters.

### 2.2 Neural Graphic Primitives

Neural networks are increasingly being used to represent 2D images [95, 102, 27], 3D objects and scenes [96, 43, 77, 102, 114, 18, 72, 68], surface representations [101, 104, 81, 22], occupancy networks [69, 14], and signed distance fields [71, 76, 3]. These methods, out of the box, can also be used to compress 3D models in some format since the learned neural network weights are often already significantly smaller. Many NGP compression methods often employ standard neural network techniques to compress MLP by knowledge distillation [92], pruning [40, 113, 45], quantization [91, 128, 117, 29, 126], factorizong tensor grids [26, 75], low-rank approximation [92, 109, 103, 41], and using codebooks [55, 53, 54] for quantization. Another approach is to compress feature grids or learnable embeddings [93, 11, 78] or by compressing extracted voxels [108, 9] in contrast to compressing the MLP often. Another set of approaches combines many of these orthogonal improvements to compressing NeRFs [57]. However, these methods often require training networks per scene or object, incurring significant compression latencies.

[Figure 9]

[Figure 10]

Textured Mesh

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Radiance Fields

3D Encoder

3D Generator

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

Point Cloud

Compressed Representation

Latents for Generator

3D geometry in any format

Encoded Representations

Compression Decompression

- Figure 2: Overview of our Method. Squeeze3D bridges arbitrary latent spaces between 3D encoders and generators through trainable mapping networks. During compression, a 3D geometry is encoded and then transformed into a compact representation via the forward mapping network. During decompression, the reverse mapping network converts this representation into the generator’s latent space, which is then used to reconstruct the original geometry.

- 2.3 3D Autoencoders and Generators

Autoencoders encode inputs into a latent space and then back into the original input using an encoderdecoder pair; thus, they could be used as compression algorithms. Early approaches relied on volumetric representations, discretizing 3D shapes into voxel grids to leverage ConvNets for encoding and decoding [63, 35, 59, 5, 73]. While these approaches are effective for regular grid structures, these methods faced scalability challenges due to cubic memory growth with resolution increases. Subsequent advancements focused on spectral methods for encoding 3D shapes in frequency domains, offering compact latent representations but requiring precomputed basis functions that limited generalizability across shape categories [118, 76].

Several works [127, 13, 111, 50] use a VAE [47] to compress 3D data into a compact latent which could be used as a compressed representation. There also exist many approaches that train generators to reconstruct radiance fields [48, 39, 112], Gaussians [110, 66, 112], or voxels [83, 112]. Some recent works also pose the problem as learning in a token-space [8, 12, 94, 122, 124, 116, 65, 60, 38, 67, 16], or with diffusion models [121, 2, 84, 25, 82, 89]. One such generative model [122] is able to compress meshes with high compression ratios. Compared to these methods that use autoencoders or generative models for compression, Squeeze3D does not require training specialized encoder-generators for each representation. Instead, Squeeze3D aims to use existing encoders and generative models. This enables flexibly adapting the approach as encoders and generative models evolve and supporting different 3D formats.

The work closest to our method is Generative Latent Coding (GLC) [42], which trains an autoencoderstyle generative model to compress images, particularly compressing the latent representations for an image obtained through VQ-VAE [107]. However, this method is not designed for 3D data.

- 3 Method

In this work, we introduce Squeeze3D, a technique to generate highly compressed representations of 3D models by leveraging the implicit prior knowledge learnt by 3D generative models. We also leverage the availability of many 3D encoders to support an extensible set of 3D formats.

The Squeeze3D architecture is depicted in Figure 2. The key ideas of Squeeze3D are as follows. (1) We leverage existing 3D encoders to generate encoded representations for a given 3D format. Thus, the Squeeze3D architecture can be extended to support different 3D formats by using new or existing encoders. This approach also enables smaller mapping networks, as we now introduce. (2) We use small neural networks to convert from this encoded representation to a highly compressed latent representation (during compression) and then back into the latent space of a 3D generative model (for decompression). We refer to these neural networks as the forward and reverse mapping networks, and they effectively map between the latent space of a 3D encoder to that of a 3D generative model. Thus Squeeze3D can leverage a new 3D generative model by retraining the mapping networks. (3) We propose an additional loss term that enables robust training of the mapping networks to generate

Table 1: Notation. The notation we use to describe our method.

Symb. Description Symb. Description G A 3D geometry in some format E 3D encoder model: E(G)  → zE ∈ RdE G 3D generator model: G(zG, c)  → G′ zE Latent code from the encoder: zE ∈ RdE zG Latent code for the generator: zG ∈ RdG zcomp Compressed representation zcomp ∈ RdC c Conditioning information (e.g. text prompt,

FθE Forward mapping network: FθE(zE)  →

image) for G

zcomp FθD Reverse mapping network: FθD(zcomp)  →

dC Dimensionality of compressed representa-

tion

zG

dG Dimensionality of generator latent space dE Dimensionality of encoder latent space

a highly compact latent representation that can be used to store/transmit the 3D model. We share an overview of the notation we use to describe our method in Table 1.

Mapping networks offers two major benefits over a encoder-generator pair such as MeshAnything [12]: (1) The mapping networks typically provide significantly higher compression ratios than existing VAE approaches; (2) Mapping networks provide more flexibility in choice of 3D format, for example, InstantMesh or LRM require multi-view images as input rather than a mesh.

### 3.1 Bridging Latent Spaces

Squeeze3D comprises two pre-trained models: (1) a 3D encoder E that maps 3D representations to a latent space, and (2) a 3D generator G that synthesizes 3D models of the same initial representation. For a given 3D representation (mesh, point cloud, radiance field) G, the pre-trained encoder E produces a latent representation

zE = E(G) ∈ Rd

E (1)

This latent zE encapsulates G, but is not in a highly-compressed format. Additionally, we cannot directly use this representation with the generator G, as G operates in a different latent space. The

generator G synthesizes a 3D model G′ given a latent code zG and in some cases conditioning information c, G′ = G(zG,c). To bridge these disparate latent spaces, we train two mapping networks:

Forward Mapping network FθE: Maps from the encoder’s latent space to the compressed space, zcomp = FθE(zE). Reverse Mapping networks FθD: Maps from the compressed space to the generator’s latent space, zG = FθD(zcomp).

We train FθE, and FθD together and keep the encoder E and generator G networks frozen.

Compression. To compress any 3D model G, the model is first encoded using the pre-trained encoder E and then mapped into a highly compressed latent zcomp using the forward mapping network FθE

zcomp = FθE (E (G)) (2)

Decompression. To decompress the model from its highly compressed latent representation zcomp, we use the reverse mapping network FθD to obtain the latent in the 3D generator space. The 3D generator then reconstructs the original 3D model G (G′)

G′ = G FθD (zG) (3)

### 3.2 Training Squeeze3D

In order to train the Squeeze3D architecture, we need to train the mapping networks for any given pair of pre-trained 3D encoders and pre-trained 3D generator models. To train these mapping networks, we need training samples from the encoder’s latent space and the corresponding latents in the generator’s latent space. These latents in the generator’s latent space serve as “ground truth” samples during the training process. We summarize our training process in Figure 3. We now describe how to generate a training dataset with samples from both of these latent spaces.

[Figure 29]

[Figure 30]

Synthetic Ground Truths

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

3D Generator

3D Encoder

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

(paired data) Synthetic Ground Truths

Compressed Representation

Latents for Generator

Encoded Representations

Encoder Latents

(a) Creating the Training Data (b) Training the Mapping Networks

- Figure 3: Training Squeeze3D. We show an overview of (a) our process of creating synthetic data to train the mapping networks and (b) our process of training the mapping networks.

Given a pre-trained 3D generator G and encoder E, we first sample a diverse collection of conditioning inputs C = {ci}Ni=1 appropriate for the generator model (e.g. text prompts for text-to-3D generators, images for image-to-3D generators, or random noise for unconditional generators). For each conditioning input ci, we sample latent vectors zG

from the generator G and then synthesize a

i

- 3D model using the generator: G′i = G(zG

,ci). Then we encode this synthetic or generated 3D model using the pre-trained encoder E, zE

i

= E(Gi′). This methodology gives us paired synthetic data, i.e., latents, for any given pair of encoder and generator. {(zE

i

)}Ni=1, which provides the

#### ,zG

i

i

necessary supervision for training our mapping networks. The mapping networks FθE and FθD together with the generated dataset using the loss shown in Equation (4).

orthogonality of compressed representation λgram∥FθE(zE)FθE(zE)⊤ − I∥2F + λgen∥FθD FθE (zE) − GT∥22

L =

, (4)

reconstruction loss

where GT represents the synthetic ground-truth latents and ∥ · ∥F denotes the Frobenius norm.

The loss function includes an reconstruction loss term that allows us to minimize the difference between the generated latents and the corresponding ground-truth latents. We also add another term, which we refer to as gram loss. When training Squeeze3D with only the reconstruction loss term, we found that they concentrate information along a small subset of dimensions, effectively rendering many dimensions redundant.

To understand this, we empirically analyzed the latent vectors zcomp = FθE(zE) ∈ Rd

C produced by our forward mapping network. For any batch of size B, of encoded 3D models {zE

i}Bi=1, we can compute the matrix Z ∈ RB×d

). First, we observe that if we do a singular value decomposition Z = UΣV⊤, the singular values in Σ = diag(σ1,σ2,...,σd

C where each row is FθE(zE

i

) exhibits an extremely skewed distribution: σ1 ≫ σ2 ≫ ... ≫ σd

C

where σi represents the i-th singular value of Z, arranged in descending order. The condition number κ = σ

C

σdC is typically very

1

large, indicating that the effective rank of Z is much lower than dC. Second, we observe that the correlation matrix C = B1 Z⊤Z ∈ Rd

C×dC has many off-diagonal elements with large magnitudes in comparison with diagonal elements. This indicates that the dimensions of the compressed representation encode redundant information. Particularly,

( di=1C λi)2 dC i=1 λ2i

≪ dC, (5)

deff =

where λi are the eigenvalues. These observations indicate that most of the information in the compressed representation was concentrated along a few dominant directions, with most dimensions contributing negligibly. For compression, this represents a severe inefficiency in utilizing the available parameter budget.

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

| |
|---|

Draco NGF Corto Squeeze3D Ground Truth

- Figure 4: Qualitative mesh compression results. We compare Squeeze3D to state-of-the-art methods. Our approach maintains visually important geometric details. Additional results in§ C.

To address this, we propose the Gram loss term that is computed on the outputs of the first mapping network FθE to force the outputs of FθE to be orthonormal. A semi-orthogonal matrix A ∈ Rm×n is defined as a matrix that satisfies either AA⊤ = Im (if m ≤ n) or A⊤A = In (if n ≤ m). Our gram loss term when minimized forces FθE(zE)FθE(zE)⊤ ≈ I, which is precisely the condition for FθE(zE) to be a semi-orthogonal matrix (when dC ≤ dE).

## 4 Experiments

### 4.1 Experimental Setup

We train Squeeze3D to compress three 3D formats: textured 3D meshes, point clouds, and radiance fields i.e. grids of (rgbσ). For 3D meshes, we train our approach with MeshAnything [12] as the encoder and train mapping networks for three 3D generators: Shap-E [44], OpenLRM [36, 32], and InstantMesh [115]. For point clouds, we train mapping networks for PointNet++ [80] as the encoder and LION [119] as the decoder. For radiance fields, we train mapping networks for NeRFMAE [39] as the encoder and the generator. We compress radiance fields for evaluation to use existing generation models such as [39] that only generate radiance fields in this format. We present additional implementation details in Section 4.1. We also perform experiments on a separatelysourced collection of meshes and radiance fields to evaluate the effectiveness of Squeeze3D for out-of-distribution data in Appendix C and ablations in Appendix C.

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

Training Dataset Creation. We use the method described in Section 3.2 to train Squeeze3D for each of our evaluated encoder-generator pair. We now list the datasets that were used to create these latent training datasets.

[Figure 84]

[Figure 85]

[Figure 86]

Lorem ipsum

Shap-E [44] as the generator. We build a list of 2500 prompts using LLaMA3, each of which is used four times to build a dataset of 10,000 objects (details in the supplementary).

Draco† Squeeze3D Ground Truth

LRM [36, 32] or InstantMesh [115] as the generator. We rendered 10,000 random objects from Objaverse [20] which serve as image conditions. We also make sure that our rendering follows any conventions the 3D generator expects the input images to follow, for instance, white backgrounds or no backgrounds.

Figure 5: Qualitative point cloud compression results. We show qualitative results comparing Squeeze3D to state-of-the-art methods. Our approach achieves significantly higher compression ratios while maintaining perceptually important geometric details.

LION [119] as the generator. We were able to generate plausible point clouds of 3D objects without any conditioning data from random noise.

NeRF-MAE [39] as the generator. We generate radiance fields from the NeRF-MAE [39] dataset.

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

SparsePCGC VQRF Squeeze3D Ground Truth

Figure 6: Qualitative radiance field compression results. We show qualitative results comparing Squeeze3D to state-of-the-art methods. Our approach achieves a significantly higher compression ratio while maintaining visually important geometric details.

While many other compression methods require the 3D objects to have certain properties like watertightness, we do not impose any such constraints. We split the datasets into training (80%), and validation (10%) sets generated from our approach. We also built a test (10%) set of objects from the datasets: Objaverse, ShapeNet, and NeRF-MAE, on which we report our metrics.

Evaluation metrics. We report the standard widely-used metrics, PSNR ↑, MS-SSIM ↑, and LPIPS ↓ [125] for reconstruction quality of meshes and radiance fields. We report standard metrics, PCQM ↑ [70], and PointSSIM ↑ [1] for reconstruction quality of point clouds. For all the baselines we compare against, we report average compression ratios, as well as compression and decompression times.

### 4.2 3D Mesh Compression

We compare Squeeze3D applied to mesh compression with existing approaches in Table 2. While [120] achieves very high compression ratios on 3D meshes, we are unable to compare Squeeze3D with it due to the absence of code and models, and thus qualitatively contrast in Section 2.3. We make the following observations.

First, Squeeze3D achieves a mean compression ratio of 2187× (6.43MB → 3kB), compared to state-of-the-art compression methods: DeepSDF [76], by more than an order of magnitude (131×, 6.43MB → 49kB). Despite this extreme compactness, Squeeze3D preserves perceptual quality, achieving an LPIPS of 0.0274 versus 0.3704 for DeepSDF which completely fails to reconstruct complex large meshes.

Second, Squeeze3D achieves similar reconstruction quality (LPIPS of 0.0274) as that of approaches such as Draco∗ (LPIPS of 0.1039), Draco† (LPIPS of 0.0397), and Corto [49] (LPIPS of 0.1374). Squeeze3D also achieves better quality than Neural Subdivision [62] and DeepSDF [76]. Though compared to the state-of-the-art non-learned mesh compression, Squeeze3D cannot achieve as high reconstruction quality, we note that Squeeze3D achieves a significantly higher compression ratio than these approaches. Neural Geometry Fields [22] performs better in terms of quality and compression ratios than non-learned methods but does significantly worse in terms of compression size when compared with our approach. We conclude that in achieving very high compression rates, Squeeze3D offers the highest reconstruction quality. Thus, we demonstrate that leveraging 3D generative models is a promising approach for 3D compression.

While using Squeeze3D may not be as fast as some non-learned approaches like Draco [24] or Corto [49], Squeeze3D is often faster than other learned methods. Squeeze3D is particularly much faster than training a network per object, like in NGF [22]. NGF takes on average 152638 ms to compress objects and 507 ms to decompress objects, compared to 270 ms to compress an object and 1476 ms to decompress objects for Squeeze3D. We qualitatively compare against Draco, Corto, and NGF for compressing meshes in Figure 4. We note that the results from our approach retain high visual quality due to the use of priors from a generative model.

### 4.3 3D Point Cloud Compression

We compare our method applied to 3D point cloud compression with previous approaches in Table 3. We notice that our approach achieves a significantly higher compression ratio of 117 (117/1.00) opposed

- Table 2: Mesh Compression Results. Quantitative comparison of Squeeze3D with state-of-the-art

3D mesh compression methods. We report compression ratio (CR), compression and decompression times, and quality metrics (PSNR, MS-SSIM, and LPIPS). (±) represents standard deviations.

Method CR (×) (MB) ↑ Compress (ms) ↓ Decompress (ms) ↓ PSNR ↑ MS-SSIM ↑ LPIPS ↓ Draco∗ [24] + JPEG 6.9215 (6.43 / 0.93) 70.30 29.97 15.45 (±1.27) 0.7468 (±0.09) 0.2437 (±0.09) Draco† [24] + JPEG 6.7001 (6.43 / 0.96) 69.05 28.33 23.33 (±1.32) 0.9576 (±0.03) 0.1039 (±0.06) Draco‡ [24] + JPEG 6.6087 (6.43 / 0.97) 68.15 27.46 38.91 (±1.30) 0.9992 (±0.00) 0.0045 (±0.00) Draco§ [24] + JPEG 6.1968 (6.43 / 1.04) 66.30 25.81 48.55 (±1.54) 1.0000 (±0.00) 0.0004 (±0.00) Corto [49] + JPEG 45.93 (6.43 / 0.14) 50.52 8.20 20.92 (± 2.79) 0.8619 (± 0.08) 0.1374 (± 0.08) Neural Subd. [62] + JPEG 11.28 (6.43 / 0.57) 61104.12 0.00 15.95 (±2.18) 0.8525 (±0.04) 0.1513 (±0.05) DeepSDF [76] + JPEG 131.22 (6.43 / 0.05) 887.78 578.53 8.47 (± 0.23) 0.7039 (± 0.07) 0.3704 (± 0.08) NGF [22] + JPEG 42.87 (6.43 / 0.15) 152637.87 507.21 35.45 (± 3.02) 0.9987 (± 0.08) 0.0054 (± 0.03) Squeeze3D (InstantMesh) 2187.0748 (6.43 / 0.003) 270.24 1476.00 27.50 (±3.13) 0.9796 (±0.02) 0.0274 (±0.02)

- Table 3: Point Cloud Compression. Quantitative comparision of Squeeze3D with state-of-the-art point cloud compression methods. We report compression ratio (CR), compression and decompression times, and quality metrics (PCQM and PointSSIM). (±) represents standard deviations.

Method CR (×) (KB) ↑ Compress (ms) ↓ Decompress (ms) ↓ PCQM ↑ PointSSIM ↑ Draco‡ [24] 15.6417 (117 / 7.48) 0.35 0.34 3.2875 (± 0.13) 0.9722 (± 0.11) Draco§ [24] 22.4138 (117 / 5.22) 0.25 0.23 2.1039 (± 0.08) 0.9535 (± 0.12) Squeeze3D (LION) 58.5000 (117 / 2.00) 3.85 12.74 1.8437 (±1.13) 0.4484 (±0.11)

- Table 4: Radiance Field Results. Quantitative comparison of Squeeze3D with state-of-the-art 3D radiance field compression methods. We report compression ratio (CR), compression and decompression times, and quality metrics (PSNR, MS-SSIM, and LPIPS). (±) represents standard deviations.

Method CR (×) (MB) ↑ Compress (ms) ↓ Decompress (ms) ↓ PSNR ↑ MS-SSIM ↑ LPIPS ↓ SparsePCGC [108] 78.9218 (58.07 / 0.74) 301.39 680.82 22.2588 ± 0.92 0.8947 ± 0.02 0.1400 ± 0.02 VQRF [54] 40.2493 (58.07 / 1.45) 120.14 20.58 29.5537 ± 0.01 0.9749 ± 0.00 0.0618 ± 0.00 Squeeze3D (NeRF-MAE) 619.4133 (58.07 / 0.09) 45.63 75.80 26.62 ± 2.57 0.9533 ± 0.04 0.0743 ± 0.02

to 22.41 (117 / 5.22) by previous approaches. Our approach, while achieving significantly higher compression ratios, leads to only a 0.6898 lower PCQM. We show qualitative results in Figure 5.

### 4.4 Radiance Field Compression

We compare Squeeze3D applied to radiance field compression with SparsePCGC [108] and VQRF [54] in Table 4. We choose these approaches to compare against since these works (or a setting of these works), akin to our setup, only require (rgbσ) grids. We notice that our approach achieves a significantly higher compression ratio of 619× opposed to 40× by previous approaches. Squeeze3D achieves these significantly higher compression rates with only a 0.0125 drop in LPIPS. We show qualitative results in Figure 6.

## 5 Discussion and Limitations

The most significant limitation of our approach is its inherent dependency on the quality and expressiveness of the underlying 3D generative model. The decompressed outputs from Squeeze3D fundamentally cannot exceed the quality of what the generator can produce, as the generator serves as both a prior knowledge base and a quality ceiling. In our evaluation, we observed that reconstruction fidelity is directly correlated with the generative capabilities of the chosen generator model. For instance, when using the Shap-E [44] generator, we found it challenging to faithfully reproduce highly complex objects due to limitations in the model’s semantic capacity (Appendix C). This limitation, however, positions Squeeze3D to benefit automatically from future advancements in 3D generative modeling.

Since Squeeze3D is designed as a compression-decompression framework that should not require per-scene retraining, the generalization capability of our mapping networks to unseen 3D models

is crucial. We evaluated this aspect by testing on a dataset of 158 3D meshes and 227 radiance fields, which served as out-of-distribution samples, and found that our approach maintains consistent performance in Appendix A. Nevertheless, for certain outlier cases where the input 3D model contains features far from the distribution seen during training, compression quality may degrade. Thus, for Squeeze3D to be effectively deployed in practical applications, a fallback mechanism would be beneficial to handle such outlier cases. This could involve either a hybrid approach that combines our method with traditional compression techniques or an adaptive system that detects when the mapping networks are likely to produce low-quality results and switches to alternative compression methods.

## 6 Conclusion

In this work, we introduce Squeeze3D, a novel framework for 3D compression that leverages the rich priors contained within existing 3D generation models. Our approach bridges arbitrary latent spaces between different models, enabling unprecedented compression ratios while maintaining high visual fidelity. The key benefit of our approach lies in its ability to use the semantic information already encoded in pre-trained 3D generation models, effectively serving as a neural compressor. By training networks that map between latent spaces, we eliminate the need for per-scene optimization that plagues many neural compression methods. This results in faster compression and decompression times compared to many other learned approaches while maintaining competitive visual quality metrics. We hope this work inspires more research on using generative models for compressing 3D data.

## References

- [1] Evangelos Alexiou and Touradj Ebrahimi. Towards a point cloud structural similarity metric. In 2020 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), pages 1–6, 2020.
- [2] Anonymous. Diff-PCC: Diffusion-based neural compression for 3d point clouds, 2024.
- [3] Matan Atzmon and Yaron Lipman. Sal: Sign agnostic learning of shapes from raw data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2565–2574, 2020.
- [4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016.
- [5] Andrew Brock, Theodore Lim, J. M. Ritchie, and Nick Weston. Generative and discriminative voxel modeling with convolutional neural networks, 2016.
- [6] Edwin Catmull and James Clark. Recursively generated b-spline surfaces on arbitrary topological meshes. In Seminal graphics: pioneering efforts that shaped the field, pages 183–188. 1998.
- [7] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository, 2015.
- [8] Jen-Hao Rick Chang, Yuyang Wang, Miguel Angel Bautista Martin, Jiatao Gu, Xiaoming Zhao, Josh Susskind, and Oncel Tuzel. 3d shape tokenization via latent flow matching, 2025.
- [9] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pages 333–350. Springer, 2022.
- [10] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models, 2025.
- [11] Yihang Chen, Qianyi Wu, Mehrtash Harandi, and Jianfei Cai. How far can we compress instant-ngp-based nerf? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20321–20330, June 2024.
- [12] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. Meshanything: Artist-created mesh generation with autoregressive transformers, 2024.
- [13] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. arXiv preprint arXiv:2409.12957, 2024.
- [14] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5939–5948, 2019.
- [15] Jonathan Cohen, Marc Olano, and Dinesh Manocha. Appearance-preserving simplification. In Proceedings of the 25th annual conference on Computer graphics and interactive techniques, pages 115–122, 1998.
- [16] Mingyue Cui, Junhua Long, Mingjian Feng, Boyang Li, and Huang Kai. Octformer: Efficient octree-based transformer for point cloud compression with local enhancement. Proceedings of the AAAI Conference on Artificial Intelligence, 37(1):470–478, Jun. 2023.
- [17] Ruikai Cui, Weizhe Liu, Weixuan Sun, Senbo Wang, Taizhang Shang, Yang Li, Xibin Song, Han Yan, Zhennan Wu, Shenzhou Chen, Hongdong Li, and Pan Ji. Neusdfusion: A spatial-aware generative model for 3d shape completion, reconstruction, and generation, 2024.
- [18] Thomas Davies, Derek Nowrouzezahrai, and Alec Jacobson. On the effectiveness of weight-encoded neural implicit 3d shapes, 2021.
- [19] Michael Deering. Geometry compression. In Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’95, page 13–20, New York, NY, USA, 1995. Association for Computing Machinery.
- [20] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects, 2023.

- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.
- [22] Venkataram Edavamadathil Sivaram, Tzu-Mao Li, and Ravi Ramamoorthi. Neural geometry fields for meshes. In ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH ’24, New York, NY, USA, 2024. Association for Computing Machinery.
- [23] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps, 2024.
- [24] Frank Galligan, Michael Hemmer, Ondrej Stava, Fan Zhang, and Jamieson Brettle. Google/draco: a library for compressing and decompressing 3d geometric meshes and point clouds, 2018.
- [25] Juan D. Galvis, Xingxing Zuo, Simon Schaefer, and Stefan Leutengger. Sc-diff: 3d shape completion with latent diffusion models, 2024.
- [26] Quankai Gao, Qiangeng Xu, Hao Su, Ulrich Neumann, and Zexiang Xu. Strivec: Sparse tri-vector radiance fields, 2023.
- [27] Rui Gao and Rajeev K. Jaiman. H-siren: Improving implicit neural representations with hyperbolic periodic functions, 2024.
- [28] Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’97, page 209–216, USA, 1997. ACM Press/Addison-Wesley Publishing Co.
- [29] Cameron Gordon, Shin-Fang Chng, Lachlan MacDonald, and Simon Lucey. On quantizing implicit neural representations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 341–350, 2023.
- [30] Sara Hahner and Jochen Garcke. Mesh convolutional autoencoder for semi-regular meshes of different sizes. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 885–894, January 2022.
- [31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, pages 630–645, Cham, 2016. Springer International Publishing.
- [32] Zexin He and Tengfei Wang. Openlrm: Open-source large reconstruction models. https://github. com/3DTopia/OpenLRM, 2023.
- [33] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023.
- [34] Amir Hertz, Rana Hanocka, Raja Giryes, and Daniel Cohen-Or. Deep geometric texture synthesis. arXiv preprint arXiv:2007.00074, 2020.
- [35] Georg Hess, Johan Jaxing, Elias Svensson, David Hagerman, Christoffer Petersson, and Lennart Svensson. Masked autoencoder for self-supervised pre-training on lidar point clouds. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW), page 350–359. IEEE, January 2023.
- [36] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d, 2024.
- [37] Hugues Hoppe, Tony DeRose, Tom Duchamp, Mark Halstead, Hubert Jin, John McDonald, Jean Schweitzer, and Werner Stuetzle. Piecewise smooth surface reconstruction. In Proceedings of the 21st annual conference on Computer graphics and interactive techniques, pages 295–302, 1994.
- [38] Xiao Huo, Junhui Hou, Shuai Wan, and Fuzheng Yang. Rendering-oriented 3d point cloud attribute compression using sparse tensor-based transformer, 2025.
- [39] Muhammad Zubair Irshad, Sergey Zakharov, Vitor Guizilini, Adrien Gaidon, Zsolt Kira, and Rares Ambrus. Nerf-mae: Masked autoencoders for self-supervised 3d representation learning for neural radiance fields. In European Conference on Computer Vision, pages 434–453. Springer, 2024.
- [40] Berivan Isik. Neural 3d scene compression via model compression. arXiv preprint arXiv:2105.03120, 2021.

- [41] Yiping Ji, Hemanth Saratchandran, Cameron Gordon, Zeyu Zhang, and Simon Lucey. Efficient learning with sine-activated low-rank matrices, 2025.
- [42] Zhaoyang Jia, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. Generative latent coding for ultra-low bitrate image compression. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26088–26098, 2024.
- [43] Chiyu Jiang, Avneesh Sud, Ameesh Makadia, Jingwei Huang, Matthias Nießner, Thomas Funkhouser, et al. Local implicit grid representations for 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6001–6010, 2020.
- [44] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions, 2023.
- [45] Yeonsung Jung, Heecheol Yun, Joonhyung Park, Jin-Hwa Kim, and Eunho Yang. Prunerf: Segmentcentric dataset pruning via 3d spatial consistency. arXiv preprint arXiv:2406.00798, 2024.
- [46] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.
- [47] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [48] Adam R Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Sona Mokrá, and Danilo Jimenez Rezende. Nerf-vae: A geometry aware 3d scene generative model. In International Conference on Machine Learning, pages 5742–5752. PMLR, 2021.
- [49] Visual Computing Lab. Corto: Mesh compression library. https://github.com/cnr-isti-vclab/ corto, 2025. Accessed: March 03, 2025.
- [50] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In European Conference on Computer Vision, pages 112–130. Springer, 2024.
- [51] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21719–21728, June 2024.
- [52] Thibault Lescoat, Hsueh-Ti Derek Liu, Jean-Marc Thiery, Alec Jacobson, Tamy Boubekeur, and Maks Ovsjanikov. Spectral mesh simplification. Computer Graphics Forum, 39(2):315–324, 2020.
- [53] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Liefeng Bo. Compressing volumetric radiance fields to 1 mb. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4222–4231, 2023.
- [54] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Liefeng Bo. Compressing volumetric radiance fields to 1 mb. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4222–4231, 2023.
- [55] Lingzhi Li, Zhongshu Wang, Zhen Shen, Li Shen, and Ping Tan. Compact real-time radiance fields with neural codebook. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 2189–2194. IEEE, 2023.
- [56] Sicheng Li, Hao Li, Yiyi Liao, and Lu Yu. NeRFCodec: Neural Feature Compression Meets Neural Radiance Fields for Memory-Efficient Scene Representation . In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21274–21283, Los Alamitos, CA, USA, June 2024. IEEE Computer Society.
- [57] Sicheng Li, Hao Li, Yiyi Liao, and Lu Yu. Nerfcodec: Neural feature compression meets neural radiance fields for memory-efficient scene representation, 2024.
- [58] Wei Li, Yufei Chen, Zhicheng Wang, Weidong Zhao, and Lin Chen. An improved decimation of triangle meshes based on curvature. In Duoqian Miao, Witold Pedrycz, Dominik Slezak,´ Georg Peters, Qinghua Hu, and Ruizhi Wang, editors, Rough Sets and Knowledge Technology, pages 260–271, Cham, 2014. Springer International Publishing.
- [59] Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M. Alvarez, Sanja Fidler, Chen Feng, and Anima Anandkumar. Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion, 2023.

- [60] Zujie Liang and Fan Liang. Transpcc: Towards deep point cloud compression via transformers. In Proceedings of the 2022 International Conference on Multimedia Retrieval, ICMR ’22, page 1–5, New York, NY, USA, 2022. Association for Computing Machinery.
- [61] Hao Liu, Hui Yuan, Qi Liu, Junhui Hou, and Ju Liu. A comprehensive study and comparison of core technologies for mpeg 3-d point cloud compression. IEEE Transactions on Broadcasting, 66(3):701–717, 2019.
- [62] Hsueh-Ti Derek Liu, Vladimir G Kim, Siddhartha Chaudhuri, Noam Aigerman, and Alec Jacobson. Neural subdivision. arXiv preprint arXiv:2005.01819, 2020.
- [63] Juncheng Liu, Steven Mills, and Brendan McCane. Variational autoencoder for 3d voxel compression. In 2020 35th International Conference on Image and Vision Computing New Zealand (IVCNZ), pages 1–6, 2020.
- [64] Charles Loop. Smooth subdivision surfaces based on triangles. 1987.
- [65] Zhe Luo, Wenjing Jia, and Stuart W. Perry. Transformer-based geometric point cloud compression with local neighbor aggregation. In DICTA, pages 223–228, 2023.
- [66] Qi Ma, Yue Li, Bin Ren, Nicu Sebe, Ender Konukoglu, Theo Gevers, Luc Van Gool, and Danda Pani Paudel. Shapesplat: A large-scale dataset of gaussian splats and their self-supervised pretraining. arXiv preprint arXiv:2408.10906, 2024.
- [67] Miguel Marques and Luís A. Da Silva Cruz. Explorations on 3d point clouds coding using transformers and patches. In 2022 10th European Workshop on Visual Information Processing (EUVIP), pages 1–6, 2022.
- [68] Julien N. P. Martel, David B. Lindell, Connor Z. Lin, Eric R. Chan, Marco Monteiro, and Gordon Wetzstein. Acorn: Adaptive coordinate networks for neural scene representation, 2021.
- [69] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4460–4470, 2019.
- [70] Gabriel Meynet, Yana Nehmé, Julie Digne, and Guillaume Lavoué. Pcqm: A full-reference quality metric for colored 3d point clouds. In 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), pages 1–6, 2020.
- [71] Mateusz Michalkiewicz, Jhony K Pontes, Dominic Jack, Mahsa Baktashmotlagh, and Anders Eriksson. Implicit surface representations as layers in neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4743–4752, 2019.
- [72] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99–106, December 2021.
- [73] Szilárd Molnár and Levente Tamás. Variational autoencoders for 3d data processing. Artificial Intelligence Review, 57(2):42, 2024.
- [74] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4):1–15, July 2022.
- [75] Anton Obukhov, Mikhail Usvyatsov, Christos Sakaridis, Konrad Schindler, and Luc Van Gool. TT-NF: Tensor train neural fields, 2023.
- [76] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 165–174, 2019.
- [77] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 523–540. Springer, 2020.
- [78] Tuan Pham and Stephan Mandt. Neural nerf compression. arXiv preprint arXiv:2406.08943, 2024.
- [79] Rolandos Alexandros Potamias, Stylianos Ploumpis, and Stefanos Zafeiriou. Neural mesh simplification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18583– 18592, 2022.

- [80] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- [81] Marie-Julie Rakotosaona, Fabian Manhardt, Diego Martin Arroyo, Michael Niemeyer, Abhijit Kundu, and Federico Tombari. Nerfmeshing: Distilling neural radiance fields into geometrically-accurate 3d meshes, 2023.
- [82] Andres Ramirez-Jaime, Gonzalo R. Arce, Nestor Porras-Diaz, Oleg Ieremeiev, Andrii Rubel, Vladimir Lukin, Mateusz Kopytek, Piotr Lech, Jarosław Fastowicz, and Krzysztof Okarma. Generative diffusion models for compressed sensing of satellite lidar data: Evaluating image quality metrics in forest landscape reconstruction. Remote Sensing, 17(7), 2025.
- [83] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [84] Barbara Roessle, Norman Müller, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, Angela Dai, and Matthias Nießner. L3dg: Latent 3d gaussian diffusion. In SIGGRAPH Asia 2024 Conference Papers, December 2024.
- [85] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241, Cham, 2015. Springer International Publishing.
- [86] J. Rossignac. Edgebreaker: connectivity compression for triangle meshes. IEEE Transactions on Visualization and Computer Graphics, 5(1):47–61, 1999.
- [87] William J. Schroeder, Jonathan A. Zarge, and William E. Lorensen. Decimation of triangle meshes. SIGGRAPH Comput. Graph., 26(2):65–70, July 1992.
- [88] Sebastian Schwarz, Marius Preda, Vittorio Baroncini, Madhukar Budagavi, Pablo Cesar, Philip A Chou, Robert A Cohen, Maja Krivoku´ca, Sébastien Lasserre, Zhu Li, et al. Emerging mpeg standards for point cloud compression. IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 9(1):133–148, 2018.
- [89] Yiting Shao, Xiaodong Yang, Wei Gao, Shan Liu, and Ge Li. 3d point cloud attribute compression using diffusion-based texture-aware intra prediction. IEEE Transactions on Circuits and Systems for Video Technology, 34(10):9633–9646, 2024.
- [90] Nicholas Sharp et al. Polyscope, 2019. www.polyscope.run.
- [91] William Shen and Willie McClinton. Accelerating nerfs: Optimizing neural radiance fields with specialized hardware architectures.
- [92] Jinglei Shi and Christine Guillemot. Distilled low rank neural radiance field with quantization for light field compression. arXiv preprint arXiv:2208.00164, 2022.
- [93] Seungjoo Shin and Jaesik Park. Binary radiance fields. Advances in neural information processing systems, 36, 2024.
- [94] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19615–19625, June 2024.
- [95] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7462–7473. Curran Associates, Inc., 2020.
- [96] Vincent Sitzmann, Michael Zollhöfer, and Gordon Wetzstein. Scene representation networks: Continuous 3d-structure-aware neural scene representations. Advances in Neural Information Processing Systems, 32, 2019.

- [97] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014.
- [98] Vitaly Surazhsky and Craig Gotsman. Explicit surface remeshing. In Proceedings of the 2003 Eurographics/ACM SIGGRAPH symposium on Geometry processing, pages 20–30, 2003.
- [99] Andrzej Szymczak, Jarek Rossignac, and Davis King. Piecewise regular meshes: Construction and compression. Graphical Models, 64(3-4):183–198, 2002.
- [100] Towaki Takikawa, Alex Evans, Jonathan Tremblay, Thomas Müller, Morgan McGuire, Alec Jacobson, and Sanja Fidler. Variable bitrate neural fields. In ACM SIGGRAPH 2022 Conference Proceedings, SIGGRAPH ’22, New York, NY, USA, 2022. Association for Computing Machinery.
- [101] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler. Neural geometric level of detail: Real-time rendering with implicit 3D shapes. 2021.
- [102] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7537–7547. Curran Associates, Inc., 2020.
- [103] Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, and Gang Zeng. Compressible-composable nerf via rank-residual decomposition. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 14798–14809. Curran Associates, Inc., 2022.
- [104] Jiaxiang Tang, Hang Zhou, Xiaokang Chen, Tianshu Hu, Errui Ding, Jingdong Wang, and Gang Zeng. Delicate textured mesh recovery from nerf via adaptive surface refinement. arXiv preprint arXiv:2303.02091, 2022.
- [105] Gabriel Taubin and Jarek Rossignac. Geometric compression through topological surgery. ACM Trans. Graph., 17(2):84–115, April 1998.
- [106] Costa Touma and Craig Gotsman. Triangle mesh compression. Proceedings - Graphics Interface, pages 26–34, 1998. Graphics Interface ’98 ; Conference date: 18-06-1998 Through 20-06-1998.
- [107] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In

I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

- [108] Jianqiang Wang, Dandan Ding, Zhu Li, Xiaoxing Feng, Chuntong Cao, and Zhan Ma. Sparse tensor-based multiscale representation for point cloud geometry compression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):9055–9071, 2023.
- [109] Jiaxin Wang, Weichen Dai, Kangcheng Ma, and Wanzeng Kong. Neural radiance fields with hash-low-rank decomposition. Applied Sciences, 14(23), 2024.
- [110] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. In European Conference on Computer Vision, pages 456–473. Springer, 2024.
- [111] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024.
- [112] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.
- [113] Xiufeng Xie, Riccardo Gherardi, Zhihong Pan, and Stephen Huang. Hollownerf: Pruning hashgrid-based nerfs with trainable collision mitigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3480–3490, 2023.
- [114] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. Computer Graphics Forum, 41(2):641–676, 2022.

- [115] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.
- [116] Xiaoxuan Yang, Guo Lu, Donghui Feng, Zhengxue Cheng, Guosheng Yu, and Li Song. Coarse-to-fine transformer for lossless 3d medical image compression. In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP), pages 1–5, 2024.
- [117] Yiying Yang, Wen Liu, Fukun Yin, Xin Chen, Gang Yu, Jiayuan Fan, and Tao Chen. Vq-nerf: Vector quantization enhances implicit neural representations. arXiv preprint arXiv:2310.14487, 2023.
- [118] Maciej Zamorski, Maciej Zie˛ba, Piotr Klukowski, Rafał Nowak, Karol Kurach, Wojciech Stokowiec, and Tomasz Trzci´nski. Adversarial autoencoders for compact representations of 3d point clouds. Computer Vision and Image Understanding, 193:102921, 2020.
- [119] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [120] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023.
- [121] Bowen Zhang, Tianyu Yang, Yu Li, Lei Zhang, and Xi Zhao. Compress3d: A compressed latent space for 3d generation from a single image. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 276–292, Cham, 2025. Springer Nature Switzerland.
- [122] Jinzhi Zhang, Feng Xiong, and Mu Xu. 3d representation in 512-byte:variational tokenizer is the key for autoregressive 3d generation, 2024.
- [123] Jinzhi Zhang, Feng Xiong, and Mu Xu. G3pt: Unleash the power of autoregressive modeling in 3d generation via cross-scale querying transformer. arXiv preprint arXiv:2409.06322, 2024.
- [124] Junteng Zhang, Gexin Liu, Dandan Ding, and Zhan Ma. Transformer and upsampling-based point cloud compression. In Proceedings of the 1st International Workshop on Advances in Point Cloud Compression, Processing and Analysis, APCCPA ’22, page 33–39, New York, NY, USA, 2022. Association for Computing Machinery.
- [125] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.
- [126] Zhiyu Zhang, Guo Lu, Huanxiong Liang, Zhengxue Cheng, Anni Tang, and Li Song. Rate-aware compression for nerf-based volumetric video. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 3974–3983, 2024.
- [127] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in neural information processing systems, 36:73969–73982, 2023.
- [128] Hongliang Zhong, Jingbo Zhang, and Jing Liao. Vq-nerf: Neural reflectance decomposition and editing with vector quantization. IEEE Transactions on Visualization and Computer Graphics, 30(9):6247–6260, 2023.
- [129] Yi Zhou, Chenglei Wu, Zimo Li, Chen Cao, Yuting Ye, Jason Saragih, Hao Li, and Yaser Sheikh. Fully convolutional mesh autoencoder using efficient spatially varying kernels. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9251–9262. Curran Associates, Inc., 2020.
- [130] Denis Zorin, Peter Schröder, and Wim Sweldens. Interpolating subdivision for meshes with arbitrary topology. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pages 189–192, 1996.

## Appendix Contents

- A Additional Implementation Details 18

- A.1 Network Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- B Evaluation Details 20

- B.1 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.2 Collection of Additional Meshes for Evaluation . . . . . . . . . . . . . . . . . . . 21
- B.3 Collection of Additional Radiance Fields for Evaluation . . . . . . . . . . . . . . . 21

- C Additional Results 21

- C.1 Results on our Mesh Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.2 Results on our Radiance Field Collection . . . . . . . . . . . . . . . . . . . . . . 21
- C.3 Qualitative Results on Meshes. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- C.4 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Societal Impact 23
- E Results Library 24

- A Additional Implementation Details We share additional implementation details to reproduce Squeeze3D.

### A.1 Network Design

Input

Input

Input

For each of the encoder-decoder pairs, we train the mapping networks, which are feed-forward neural networks. We summarize network architectures in Figure 7.

Feature

Feature

Feature

Flatten

Flatten

Flatten

RadianceFieldMappingNetworks

MeshMappingNetworks

PointCloudMappingNetworks

Linear

Linear

Linear

Conv3D LayerNorm3D GELU

LayerNorm

LayerNorm GELU Dropout

Meshes. The mapping networks together first flatten the input and apply a linear layer to project it into a hidden dimension. This is followed by LayerNorm [4], a GELU nonlinearity [33], and dropout [97]. A second linear layer then produces another hidden representation, which is added residually to the output of the first linear layer and passed through a second LayerNorm. After another GELU and dropout, a final linear transformation projects into the latent space.

GELU Dropout x2

ResidualBlock3D Dropout3D

Linear GELU

Linear

x4

Dropout x12

Conv3D LayerNorm3D

GELU x2

Linear

ResidualBlock3D Dropout3D ConvT3D LayerNorm3D

GELU x2

Point Clouds. The mapping networks together consist of flattening the input, a linear projection feeds into a LayerNorm [4]; this normalised vector is stored as a global residual. The sequence then passes through GELU-activated [33] hidden layers of uniform

ResidualBlock3D ResidualBlock3D

[Figure 99]

Feature Output

Feature

Feature Output

Output

(a) (b) (c)

[Figure 100]

[Figure 101]

[Figure 102]

Figure 7: Network Architectures.

width, each followed by dropout regularisation on the activations [97]. We use two skip-connection schemes: (i) the global residual is re-added immediately after the first hidden layer, and (ii) every fourth hidden layer receives a local residual that adds its own input to its output, creating short four-layer paths. A final linear projection transforms the resulting representation into the target latent space.

Radiance Fields. The mapping networks together consist of a custom LayerNorm that first permutes tensors so that LayerNorm [4] can operate across channels before restoring the usual NCDHW layout. We then build ResidualBlocks, which mirrors a bottleneck ResNet-V2 design [31]: a 3 × 3 × 3 conv–LN–GELU [33] stem, a 1 × 1 × 1 → 3 × 3 × 3 bottleneck branch that doubles the channel count, and two skip paths: (i) projection shortcut for stride/width mismatches and (ii) “within-block” residual that re-adds the pre-bottleneck activation just before the final GELU. The encoder begins with a 96-channel input, expands to 192 channels, and applies two strided ResidualBlock units to down-sample from 403 latents to 203 and then 103. Thus we have a narrow bottleneck of 24 channels that serves as the latent code. The symmetric decoder inverts this pathway with a 3×3×3 convolution, residual processing, and two transposed-conv up-sampling stages that restore the original resolution, all regularised by spatial Dropout in 3D. U-Net–style [85] skip connections add encoder feature maps to decoder activations whenever spatial dimensions match.

### A.2 Training

Our results are collected on an Intel(R) Core(TM) i7-13700K CPU machine with one NVIDIA RTX4090 GPU and 128GB memory. We list the choices we make during training for each encoder-generator pair.

For all the models we train with MeshAnything [12] as the encoder, we use the 350 million parameter version of MeshAnything and use the features created by MeshAnything as the encoded inputs. These encoded inputs (257, 1024) are relatively larger compared to our desired compression size.

Table 5: Model Sizes. We show the combined size of the forward mapping network and the reverse mapping network we train for each setting. We denote the compressed size in parentheses (·) for the point cloud methods.

Encoder-Decoder Pair zcomp Parameters (M) MeshAnything [12]-InstantMesh [115] 770 96.12 MeshAnything [12]-Open LRM [36, 32] 1024 87.51 MeshAnything [12]-Shap-E [44] 1024 134.53 PointNet++ [80]-LION [119] 1024 2.11 PointNet++ [80]-LION [119] 2048 6.53 PointNet++ [80]-LION [119] 4096 22.29 PointNet++ [80]-LION [119] 8192 81.48 NeRF-MAE [39] 24000 86.46

MeshAnything [12]-InstantMesh [115]. While generating the dataset, we ensure that the input condition images are in the RGBA format and have no background. We experiment to have the mapping networks generate: multiview ViT embeddings [21] that InstantMesh uses as well as the triplane representation InstantMesh uses. We experimentally observed better performance with generating triplane representations, thus our results report this setting.

MeshAnything [12]-Open LRM [36, 32]. While generating the dataset, we ensure that the input condition images are in the RGBA format and have no background. Due to the lack of public code for the LRM, we use the OpenLRM implementation. We experiment with both the openlrm-mix-base-1.1 and openlrm-obj-base-1.1, and we experimentally observed the openlrm-obj-base-1.1 to have better performance. We train the mapping networks to generate the triplane latents for LRM.

MeshAnything [12]-Shap-E [44]. We experiment with Shap-E in the text-to-image mode. The mapping networks generate the implicit MLP representations.

PointNet++ [80]-LION [119]. We experiment with the SSG and MSG versions of PointNet++ and experimentally observed SSG without normals to work the best. We use the PointNet++ checkpoints pre-trained for classification. We consider the outputs from the point set feature learning module as the encoded representations. We use the all categories model for LION. The mapping networks are trained to generate both the global and local latents for LION.

- Table 6: Training Hyperparameters. We show the training hyperparameters for the mapping networks we train.

Hyperparameter LRM Shap-E InstantMesh NeRF-MAE Training Precision FP-32 FP-32 FP-32 FP-32 Compressed Size 1024 1024 770 24000 Dropout 0.35 0.35 0.35 0.2 Epochs 700 700 700 2000 Batch Size 16 8 16 4 Optimizer Muon Adam Muon Muon

λ = 10−2

Optimizer Parameters

NS = 6 β1 = 0.9 NS = 6 NS = 6 β = 0.95 β2 = 0.999 β = 0.95 β = 0.95

- Initial Learning Rate 10−2 10−4 10−3 10−2 Final Learning Rate 10−7 10−4 10−7 10−5 Scheduler Linear Decay Constant Linear Decay Linear Decay Epochs Decay 700 N/A 600 1000 Gradient Accum. Steps 1 2 1 1 Gradient Clipping None None None None

Hyperparameter LION (1024) LION (2048) LION (4096) LION (8192) Training Precision FP-32 FP-32 FP-32 FP-32 Hidden Size 1024 2048 4096 8192 Dropout 0.3 0.3 0.3 0.3 Epochs 4000 4000 1000 400 Batch Size 16 16 16 16 Optimizer Muon Muon Muon Muon Optimizer Parameters β = 0.95 β = 0.95 β = 0.95 β = 0.95

NS = 6 NS = 6 NS = 6 NS = 6

- Initial Learning Rate 10−3 10−3 10−3 10−3 Final Learning Rate 10−7 10−7 10−7 10−7 Scheduler Linear Decay Linear Decay Linear Decay Linear Decay Epochs Decay 1000 1000 1000 1000 Gradient Accum. Steps 1 1 1 1 Gradient Clipping None None None None

We provide the model sizes in Table 5. We provide the hyperparameters we use to train the mapping networks in Table 6.

## B Evaluation Details

### B.1 Baselines

Meshes. We compare Squeeze3D against non-learned baselines: Draco [24] (with multiple settings) and Corto [49], the best performing mesh compression techniques. For all settings of Draco, we use a compression level of 10. The setting § represents 14 bits for the position attribute, 14 bits for texture coordinates, 14 bits for normal vector attributes, and 14 bits for any generic attribute. The setting ‡ represents 11 bits for the position attribute, 10 bits for texture coordinates, 8 bits for normal vector attributes, and 8 bits for any generic attribute. The setting † represents 7 bits for the position attribute, 7 bits for texture coordinates, 7 bits for normal vector attributes, and 7 bits for any generic attribute. The setting ∗ represents 4 bits for the position attribute, 4 bits for texture coordinates, 4 bits for normal vector attributes, and 4 bits for any generic attribute. We also compare our approach against learned approaches: Neural Subdivision [62], DeepSDF [76], and Neural Geometry Fields [22]. All the baseline models we compare against do not support texture images, thus we pair these methods with JPEG to compress the accompanying texture images.

- Table 7: Mesh Compression Results. Quantitative comparison of Squeeze3D for compressing meshes from our collection of meshes (Appendix B.2).

Method CR (×) (MB) ↑ Compress (ms) ↓ Decompress (ms) ↓ PSNR ↑ MS-SSIM ↑ LPIPS ↓ Ours 1933.33 (5.80 / 0.003) 270.24 1476.00 26.6353 0.9905 0.0124

Point Clouds. We compare Squeeze3D against Draco [24], the best performing point cloud com-

pression technique. For all settings of Draco, we use a compression level of 10. The setting § represents 14 bits for the position attribute, 14 bits for texture coordinates, 14 bits for normal vector attributes, and 14 bits for any generic attribute. The setting ‡ represents 11 bits for the position attribute, 10 bits for texture coordinates, 8 bits for normal vector attributes, and 8 bits for any generic attribute.

Radiance Fields. We compare Squeeze3D against SparsePCGC [108], and VQRF [54], the best performing applicable techniques to radiance field compression. Particularly, note that our method compresses arbitrary radiance fields and thus most existing NeRF and 3D Gaussian Splat compression echniques are not applicable to our problem. To evaluate VQRF [54], we use the voxel pruning and vector quantization steps which are relevant for compressing radiance field grids.

### B.2 Collection of Additional Meshes for Evaluation

Since Squeeze3D is designed as a compression-decompression framework that should not require per-scene retraining, the generalization capability of our mapping networks to unseen 3D models is very important. Thus, we also collected a dataset of 158 high-quality 3D meshes from the following sources,

- • OpenLRM Demo: https://huggingface.co/spaces/zxhezexin/OpenLRM
- • InstantMesh Demo: https://huggingface.co/spaces/TencentARC/InstantMesh
- • Hunyuan3D-2 Demo: https://huggingface.co/spaces/tencent/Hunyuan3D-2
- • TRELLIS Demo: https://huggingface.co/spaces/JeffreyXiang/TRELLIS
- • SPAR3D Demo: https://huggingface.co/spaces/stabilityai/ stable-point-aware-3d
- • SORA-3D Demo: https://huggingface.co/spaces/ginipick/SORA-3D

### B.3 Collection of Additional Radiance Fields for Evaluation

Since Squeeze3D is designed as a compression-decompression framework that should not require per-scene retraining, the generalization capability of our mapping networks to unseen 3D models is very important. Thus, we also report results on the unseen “ai” subset of the NeRF-MAE dataset [39].

## C Additional Results

### C.1 Results on our Mesh Collection

To demonstrate that our approach can compress meshes which are not in the same distribution that our training dataset was derived from: Objaverse [20], we report the results of our method with MeshAnything [12] as the encoder and InstantMesh [115] as the generator in Table 7. Our mapping networks are trained on the dataset we derived from Objaverse [20], but we test Squeeze3D on our collection of meshes. On our collection of meshes, Squeeze3D achieves on average only a 0.86 dB reduction in the PSNR and a 0.015 reduction in LPIPS.

### C.2 Results on our Radiance Field Collection

Our mapping networks are trained on the dataset we derived from NeRF-MAE [39], but we test Squeeze3D on only the radiance fields in the ai subset of the dataset which was collected by seprately

- Table 8: Radiance Field Results. Quantitative comparison of Squeeze3D for compressing radiance fields from the ai subset of the NeRF-MAE dataset (Appendix B.3).

Method CR (×) (MB) ↑ Compress (ms) ↓ Decompress (ms) ↓ PSNR ↑ MS-SSIM ↑ LPIPS ↓ Ours 657.8888 (59.21 / 0.09) 45.63 75.80 22.40 0.8958 0.1400

[Figure 103]

[Figure 104]

Linear Interpolation

- Figure 8: Interpolation. The compressed representations we obtain can also be interpolated. In these examples we obtain the compressed representation for the leftmost and rightmost meshes and linearly interpolate between them.

Ground Truth Reconstruction

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

- Figure 9: Multi-view visualization of compressed and reconstructed meshes. The consistent appearance across different viewing angles demonstrates that Squeeze3D learns correct transformations between latent spaces and produces coherent 3D reconstructions. This confirms that our compressed representation encodes complete 3D information rather than view-dependent features.

than other datsets. On our collection of meshes, Squeeze3D achieves on average only a 4.22 dB reduction in the PSNR as we show in Table 8.

### C.3 Qualitative Results on Meshes.

We also notice that our approach works well with high-resolution complex meshes as we show in Figure 12. In Figure 9, we show a mesh we reconstruct with our method through multiple camera angles to demonstrate that our approach learns a correct transformation between the latent spaces and the reconstruction is consistent. We also observe that the compression representation our method learns can also be interpolated as we show in Figure 8. In Figure 11 we show examples using two different generative models indicating that Squeeze3D can be extended to use other generative models for compression. In Figure 10, we observe that our method learns to effectively represent intricate geometrical details.

### C.4 Ablations

We conduct several ablation studies to better understand the behavior of Squeeze3D and analyze the trade-offs between compression ratio, reconstruction quality, and computational efficiency. First, we investigate how varying the size of our compressed representation affects decompression time. In Table 9, we show that the compressed size affects the compression and decompression times

[Figure 112]

[Figure 113]

[Figure 114]

| |
|---|

| |
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Ground Truth Reconstructed

- Figure 10: Squeeze3D preserves geometry details. We show some meshes compressed with Squeeze3D as wireframes. Notice that Squeeze3D preserves many finegrained geometric details.

- Table 9: Ablations. Ablation study on compressed representation size. We analyze how varying the dimensionality of the compressed latent space affects compression time, decompression time, and reconstruction quality (measured by PCQM and PointSSIM).

Size (zcomp) Compression (s) Decompression (s) PCQM ↑ PointSSIM ↑

1024 3.44 12.33 1.4047 0.3640 2048 3.51 12.39 1.3311 0.4249 4096 3.85 12.74 1.8437 0.4318 8192 5.3 14.19 1.5665 0.4473

proportionally since changes in the compressed size affect the size of the neural network. Next, we examine how different compression sizes affect reconstruction quality. Table 9 shows a correlation between latent dimension and reconstruction quality. We observe substantial improvements in PointSSIM when increasing the compressed size from 1024 to 2048, however, this improvement in quality starts diminishing beyond the compressed size of 2048.

## D Societal Impact

Our method has the potential to advance social good by lowering the bandwidth, storage, and energy requirements associated with the ever-growing volumes of 3D data. By shrinking large meshes, point clouds, and radiance fields to kilobytes, Squeeze3D can make high-fidelity 3D assets practical for mobile devices, distance learning, cultural-heritage archiving, and tele-presence, thereby broadening access to immersive content while also reducing the carbon footprint of cloud infrastructure. However, extreme compression built on powerful generative priors also carries risks: it may facilitate unauthorized replication and distribution of copyrighted 3D models; compressed latents could be

[Figure 124]

[Figure 125]

OpenLRM

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Shap-E

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Reconstructed (comp. 0.003 MB) V: 72595, F: 11516

Ground Truth (8.2 MB) V: 77851, F: 120812

Ground Truth Reconstructed

- Figure 11: Compression results using different 3D generators. Squeeze3D is agnostic to the choice of a 3D generation model. Thus, we show compression results with the 3D generators: OpenLRM, and Shap-E. We choose 3D meshes that lie in the representation capacity of the chosen 3D generators.

Figure 12: Compressing Complex Meshes. Squeeze3D can be used to compress highly complex textured 3D meshes (in this case 77851 vertices and 120812 faces).

used to conceal contraband or embed hidden payloads; and the ease of disseminating photorealistic

- 3D scenes may accelerate the creation of deceptive “deep-fake” environments.

- E Results Library We present additional results in Figures 13 to 25.

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

##### Figure 13: Results Library.

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

##### Figure 14: Results Library.

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

[Figure 277]

[Figure 278]

##### Figure 15: Results Library.

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

##### Figure 16: Results Library.

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

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

##### Figure 17: Results Library.

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

[Figure 422]

##### Figure 18: Results Library.

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

##### Figure 19: Results Library.

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

##### Figure 20: Results Library.

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

[Figure 543]

[Figure 544]

[Figure 545]

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

##### Figure 21: Results Library.

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

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

##### Figure 22: Results Library.

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

##### Figure 23: Results Library.

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

##### Figure 24: Results Library.

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

##### Figure 25: Results Library.

