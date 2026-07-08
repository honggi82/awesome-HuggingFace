## Channel-wise Vector Quantization

Wei Song1,2,3∗ Tianhang Wang1,3∗ Yitong Chen1,4 Tong Zhang6 Zuxuan Wu1,4 Min Li3 Jiaqi Wang1,5† ‡ Kaicheng Yu2†

1Shanghai Innovation Institute 2Westlake University 3Zhejiang University 4Fudan University 5JD.COM 6University of Chinese Academy of Sciences

# arXiv:2605.26089v2[cs.CV]1Jun2026

Image feature 𝑧̂

Traditional VQ-based AR: generate patch by patch

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

“a photo of an apple”

h

c

w

(b) CVQ

###### (a) VQ

CAR (ours): next channel prediction

…

[Figure 8]

𝑧

𝑧 …

8 tokens 16 tokens 32 tokens 64 tokens 128 tokens 256 tokens

| | | |
|---|---|---|
| | |1|
|9|3| |
|5|4|9|
|2|6|2|
| | | |
|3|8|5|
| | | |

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

5

2

|3|1|
|---|---|
|4|9|

|2|
|---|

|9|
|---|

2

5 …

3

- 7 4
- 8 1

|7|
|---|

|5|
|---|

“a photo of an apple”

1

|4|
|---|

|2|
|---|

|6|
|---|

|2|
|---|

2

3

|8|
|---|

|3|
|---|

|8|
|---|

|5|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|ID|Embedding|
|---|---|
|idx=1<br>idx=2<br>idx=3<br><br><br>idx=N<br><br>…|…<br><br>w<br><br>h|

|ID<br><br>|Embedding|
|---|---|
|idx=1<br>idx=2<br>idx=3<br><br><br>idx=N<br><br>…|…<br><br>c|

“an image of a cute cat”

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

“Fattoush Salad with Roasted Potatoes”

CVQ Codebook

VQ Codebook

Figure 1: (Left) VQ vs. CVQ. Conventional VQ assigns an index to each 1×1×c patch feature vector, while CVQ assigns an index to each channel of the feature map. (Right) AR vs. CAR. Traditional autoregressive (AR) models generate images patch by patch in raster scan order (here, Emu3 [41]), whereas our channel-wise autoregressive (CAR) model generates images channel by channel. For example, given the prompt “a photo of an apple”, the model first sketches a red circular shape corresponding to the apple’s outline and dominant color, then progressively depicts its appearance, and finally adds fine-grained visual details such as specular highlights and yellow speckles.

### Abstract

We present Channel-wise Vector Quantization (CVQ), a novel image tokenization paradigm that replaces patch-wise tokens with channel-wise tokens. Unlike conventional vector quantization, which assigns a discrete token to each patch feature vector, CVQ quantizes each channel of the feature map. This formulation represents an image as discrete levels of visual details, rather than as a grid of spatial patches. Based on CVQ, we introduce a new visual autoregressive framework with “next-channel prediction”. Instead of rendering images patch by patch in raster order, our Channel-wise Auto-Regressive (CAR) model predicts image channels sequentially, producing progressively enriched visual details. Specifically, it first sketches global structure and then refines fine-grained attributes, akin to a human artist’s workflow. Empirically, we show that: (1) CVQ achieves 100% codebook utilization with a 16K+ codebook size without any bells and whistles, and substantially improves reconstruction quality over conventional VQ; and (2)

∗Co-first authors. †Corresponding authors. ‡Project Leader.

Preprint.

CAR attains a DPG score of 86.7 and a GenEval score of 0.79, demonstrating strong effectiveness for text-to-image generation. The code is available at this link.

### 1 Introduction

Vector Quantization (VQ) [39, 10] is a fundamental technique for discretizing continuous image representations and serves as a cornerstone of discrete image generation [4, 33, 35]. However, since its introduction, the community has largely adhered to a patch-wise paradigm by default, where each index is assigned to represent a local 1×1×c feature vector, as illustrated in Fig. 1 (left).

We argue that this long-standing convention imposes two major limitations. (1) Insufficient codebook usage, which leads to severe information loss and, consequently, poor reconstruction quality. While prior efforts have attempted to mitigate this issue [58, 59, 12], these works typically rely on complex tricks or extra parameters that increase structural complexity, or require token factorization [51, 33], which projects image features into a low-dimensional space for code index lookup. Such dimensionality reduction limits representational capacity [31] and substantially compromises token expressiveness [24]. (2) Not naturally suited to sequence-to-sequence modeling. Next-token prediction has achieved remarkable success in language models [1, 50], and the vision community seeks to mirror this success. However, language is inherently a 1D sequential signal (left-to-right), whereas images are spatial. Patch-wise tokenization discretizes images into 2D grids of tokens, which are then mechanically flattened into a 1D sequence to accommodate autoregressive (AR) learning (e.g., via raster scan or z-curve). This structural mismatch results in a suboptimal token ordering for unidirectional AR modeling [18, 27, 54], as it disrupts local spatial dependencies among neighboring tokens [37, 17]. Furthermore, as discussed in [56], the strong local spatial bias of patch tokens makes it difficult to impose an AR-friendly ordering on them (e.g., via nested dropout) [30].

In this paper, we show that a simple change in the quantization axis naturally resolves both limitations. Specifically, we shift VQ from a patch-wise to a channel-wise formulation. Unlike standard VQ, which learns a discrete spatial codebook of patch-wise indices, our CVQ learns a discrete sequential codebook of channel-wise indices. Our motivation is intuitive: people typically draw by layering different levels of visual information to form a complete image. For example, when drawing an apple, an artist may first outline its overall shape and color tone before depicting finer details like the stem and speckles. Interestingly, as shown in Fig. 2, an autoencoder reflects a similar behavior by distributing different levels of visual information across channels, which jointly determine the final complete image.

Original

Channel 2

Channel 42 Channel 60

Original Channel 2 Channel 34 Channel 78

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Recon.

Color Leaf stalk Textures

Recon.

Color Apple stem Textures

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Knocking out the corresponding channel

Knocking out the corresponding channel

Figure 2: We visualize several channel activation maps from our VQVAE encoder and ablate individual channels by zeroing them before reconstruction. Interestingly, removing a channel selectively alters the corresponding content in the reconstructed image, with effects ranging from global appearance (e.g., leaf color) to fine-grained structural (e.g., removing the apple stem).

Motivated by this, unlike conventional VQ, which discretizes spatial vectors at each x,y location, we propose to discretize each feature channel. Since each channel tends to capture different aspects of visual information, CVQ represents an image as a 1D token sequence with progressively enriched visual content. In contrast to conventional VQ, which imposes an artificial spatial ordering on the token sequence, CVQ internalizes spatial information within each token, as each channel encodes global spatial structure. As a result, the resulting token sequence forms a clean 1D representation, free from the limitations induced by patch-based tokenization.

On the other hand, as discussed in [58, 59], low codebook utilization in VQ arises from biased optimization dynamics, where only a small fraction of codebook vectors receive updates while the rest remain stagnant. We argue that this issue stems from the redundancy among image patches [15]. As analyzed in Fig. 4 (a), for two images with similar textures, patch-wise partitioning produces highly overlapping embeddings, causing a large number of patch-wise embeddings within each training batch to be clustered to the same codebook indices (Fig. 4 (b)). Consequently, codebook updates concentrate on only a small subset of entries, eventually leading to dead codes and codebook

collapse (Fig. 4 (c)). In contrast, partitioning features along the channel dimension yields more separable embeddings across images, resulting in broader codebook coverage and improved codebook utilization.

In summary, our contributions are as follows:

- • We propose a novel image tokenization paradigm that represents an image as a 1D sequence of channels with progressively enriched visual content, rather than a 2D grid of spatial patches.
- • We provide a new perspective on the design space of vector quantization, showing that a simple change in the quantization axis effectively mitigates codebook collapse without introducing additional modules or constraints.
- • Based on CVQ, we reformulate autoregressive image generation as a progressive next-channel prediction framework, and show that, with simple nested dropout [30], the 1D channel sequence can form a more AR-friendly ordering compared to the 2D raster-scan ordering.

### 2 Related Works

Vector Quantization Vector quantization is essential for discretizing visual signals into tokens. VQ-VAE [39] first introduced a learnable codebook to obtain discrete latent representations. Building upon this, VQGAN [10] incorporates adversarial and perceptual losses to enhance image fidelity. RQ-VAE [22] and MoVQ [57] further reduce quantization error through multi-stage quantization and vector modulation. Despite these advances, low codebook utilization remains a central challenge in VQ. To mitigate this issue, ViT-VQGAN [51] proposes token factorization by projecting image features into a low-dimensional space for code index lookup. FSQ [26] and LFQ [52] extend this idea by quantizing representations into a small set of fixed values to prevent codebook collapse. However, such approaches substantially limit representational capacity [24, 31]. Recent works, including VQGAN-LC [58] and SimVQ [59], achieve higher codebook utilization through CLIPfeature initialization and learnable bases. However, these methods rely on more complex training pipelines and additional parameters compared to the original VQ architecture. IBQ [31] focuses on improving gradient propagation through index backpropagation and is orthogonal to our work. To the best of our knowledge, [49] is the only work that performs quantization along channels. However, it solely serves as an auxiliary component to support anytime sampling under computational constraints, and has not been systematically studied or evaluated as an independent vector quantization method. So far, dominant VQ methods remain patch-wise and thus inherit the limitations of 2D grid tokenization.

Autoregressive Visual Generation Inspired by the success of Large Language Models (LLMs) [3, 1, 38, 50], autoregressive (AR) image generation has become a current research hotspot. Existing methods typically tokenize images into 2D grids using VQGAN-like models and flatten them into 1D raster-scan sequences, creating a structurally misaligned ordering for next-token prediction and limiting AR performance [33, 35, 41]. On the other hand, VAR [37] proposes next-scale prediction, which tokenizes images into multi-scale 2D tokens with bidirectional modeling within each scale, achieving promising results. Infinity [13] further scales this approach to a much larger vocabulary size. However, VAR deviates from the standard next-token prediction paradigm of LLMs, and its multi-scale hierarchy relies on heuristic partitioning. Another line of works explores compact 1D visual tokenization. TiTok [53], SpectralAR [17] and Hita [56] aggregate image representations into 1D sequences via learnable queries, while FlexTok [2] and Semanticist [42] extend the detokenizer to diffusion models. However, these methods relies on additional modules, such as token aggregation and diffusion decoders [2, 42, 8], leading to more complex architectures, potential information bottlenecks, and increased learning difficulty. In contrast, CVQ operates at a different conceptual level: its 1D structure arises directly from the quantization process, enabling a 1D token sequence without specialized architectural design while maintaining the standard next-token prediction paradigm.

### 3 Method

#### 3.1 Channel-wise Vector Quantization

Traditional VQ frameworks map continuous latent variables to a discrete spatial codebook. As shown in Fig. 3(a), given an input image I ∈ RH×W×3, it is encoded into a latent representation

[Figure 43]

[Figure 44]

[Figure 45]

Codebook 𝓏

AR

[Figure 46]

[Figure 47]

[Figure 48]

Patch by patch

- (a) conventional approach
- (b) channel-wise approach

VQ

|ID|Embedding|
|---|---|
|idx=1<br>idx=2<br>idx=3<br><br><br>idx=N<br><br>…|…|

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Pixel Space Token Space

Poor codebook usage Poor reconstruction

[Figure 53]

𝑧̂ 𝑧

Input recon.

[Figure 54]

###### LLM

[Figure 55]

2 9 3 1

E D

7 5 4 9

h

argmin ∈𝒵 𝑧̂ − 𝑧

4 2 6 2

c

c

8 3 8 5

Look up Encoder

’

w

Decoder

[Figure 56]

[Figure 57]

A photo of a salad

Patch-wise Vector Quantization

h * w tokens

From global attributes to detailed attributes

CAR

CVQ Codebook 𝓏 (ours)

|ID|Embedding|
|---|---|
|idx=1<br>idx=2<br>idx=3<br><br><br>idx=N<br><br>…|…..|

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

100% codebook usage Improved image quality

Pixel Space

[Figure 68]

𝑧̂ 𝑧

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Token Space

Input recon.

…

…

[Figure 78]

[Figure 79]

###### LLM

5

2

###### E D

3

1

h

argmin ∈𝒵 𝑧̂ − 𝑧

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

c

h

Look up

w

’

Decoder

Encoder

[Figure 88]

[Figure 89]

w

A photo of a salad

Channel-wise Vector Quantization

c tokens

- Figure 3: (a) Conventional VQ and standard autoregressive image generation. Conventional VQ assigns an index to each 1×1×c patch feature vector, which causes poor codebook usage and reconstruction ability. Furthermore, standard AR models predict tokens patch by patch, which imposes inherent sequential constraints on 2D spatial modeling. (b) Proposed CVQ and CAR. Our CVQ assigns indices to each h×w×1 channel feature vector, ensuring near-optimal 100% codebook usage and improved image quality. Building on this, the proposed CAR model predicts visual content hierarchically, progressing from global structures to fine details.

Z ∈ Rh×w×c, where h = H/f, w = W/f and f is the downsample ratio. In conventional VQ schemes, Z is treated as a collection of M = h × w spatial patch vectors, formulated as





- z(1,1,:) z(1,2,:) ··· z(1,w,:)
- z(2,1,:) z(2,2,:) ··· z(2,w,:)

, (1)

Z =

 

 

... .

. .

z(h,1,:) z(h,2,:) ··· z(h,w,:)

where z(i,j,:) ∈ R1×1×c denotes a spatial vector at the position (i,j). The quantization process involves a nearest-neighbor lookup within the codebook Cspatial = {en}Nn=1, where N denotes the codebook size:

2 2

z(qi,j,:) = argmin en∈Cspatial

z(i,j,:) − en

. (2)

However, this conventional VQ paradigm typically suffers from codebook collapse, where only a small portion of the codebook receives gradient updates during training. [58, 59, 31].

We argue that this issue stems from the similarity and repetition between image patches [15]. As illustrated in Fig. 4(a), the patch-wise partitioned embeddings from different images are heavily entangled, resulting in substantial overlap of codebook indices both within and across images (Fig. 4(b)). Such redundancy and recurrence cause patches to cluster around the same vectors during early training. This, in turn, leads to the “death” of other cluster centers, ultimately resulting in codebook collapse (Fig. 4(c)).

In contrast to conventional VQ, CVQ introduces a distinct quantization mechanism, As shown in Fig. 3(b). The latent vector is seen as

Z = z(:,:,1),z(:,:,2),...,z(:,:,c) , (3) where z(:,:,k) ∈ Rh×w×1 denotes the kth channel of Z. For notational convenience, we denote z(:,:,k)

- as z(k) in the remainder of the paper. The quantization process is

2 2

z(k) − en

z(qk) = argmin en∈Cchannel

, (4)

where Cchannel represents the channel-wise codebook comprising codewords en ∈ Rh×w×1. For the lookup of two-dimensional representations, we adopt a simple formulation based on the Frobenius

(a) Continuous patch-/channel-wise embedding

(c) Codebook entries used per training batch

by channel by patch

Image feature

init. 500k steps

1k steps 3k steps

…

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

h

- Img 0
- Img 1

c

w

|[Figure 94]|
|---|

t-SNE

t-SNE

|[Figure 95]|
|---|

|[Figure 96]|
|---|

Full Codebook

Full Codebook

Full Codebook

Full Codebook

Used by Batch

Used by Batch

Used by Batch

Used by Batch

[Figure 97]

- Img 0
- Img 1

Training Process of VQ

init.

1k steps

3k steps 500k steps

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

(b) Look up in codebook

[Figure 103]

|[Figure 104]|
|---|

Full Codebook Used by Batch

Full Codebook Used by Batch

Full Codebook Used by Batch

Full Codebook Used by Batch

Full Codebook

Full Codebook

- Code used by Img0 (35)

- Code used by Img1 (28) Overlap (52)

- Code used by Img0 (220)

- Code used by Img1 (224) Overlap (12)

Training Process of CVQ

- Figure 4: (a) t-SNE visualization of embedding distributions. For images with similar appearances but distinct semantics. Patch-wise partitioned embeddings of the two images are highly intermingled within a shared distribution, whereas channel-wise partitioning produces clearly separable clusters. (b) t-SNE of codebook vectors used by each image. Patch similarity leads to significant intra- and inter-image overlap across codebook entries. In contrast, channel-wise embeddings are distributed over a broader range of codebook points, resulting in higher utilization. More results are provided in Appendix. A. (c) t-SNE of codebook entries used per training batch. Conventional VQ rapidly collapses to a small subset of outlier entries, while CVQ maintains high codebook utilization throughout the training process.

norm, which is equivalent to flattening the matrix into a vector and performing a nearest-neighbor lookup. The forward pass is defined as

z(qk) = z(k) + sg[en − z(k)], (5)

where sg[·] denotes the stop-gradient operator. For the backward pass, CVQ utilizes the standard Straight-Through Estimator (STE) [39] approach, where gradients are copied directly from the quantized representation zq to the continuous latent z.

Within the CVQ framework, each codebook index represents a global channel feature rather than a local spatial patch. As illustrated in Fig. 4(a), partitioning representations along the channel dimension yields features that are highly distinguishable across different images. This property promotes high codebook utilization and reduces overlap, as evidenced in Fig. 4(b). Consequently, the t-SNE visualization in Fig. 4(c) shows that CVQ activates a significantly larger portion of the codebook within each training batch (batch size = 64) compared to standard VQ.

Tokenizer Training Following the standard VQGAN approach [10], the CVQ tokenizer is trained with a reconstruction objective comprising pixel-wise ℓ2 loss, commitment loss, LPIPS loss [55], and adversarial loss with PatchGAN discriminator [19].

#### 3.2 Channel-wise Autoregressive Generation

As illustrated in Fig. 3(b), we reformulate next-token prediction in autoregressive visual modeling by shifting from the traditional next-patch prediction paradigm to a Next-Channel Prediction (NCP) strategy, where the autoregressive unit is a channel rather than a patch token.

Through the CVQ tokenizer, an image is represented as a 1D sequence of c discrete channel tokens, X = {xi}ci=1. Subsequently, the decoder-only transformer is trained autoregressively to predict the next channel token conditioned on the textual context. The autoregressive likelihood of the entire image is thus given by

c

p x(k) | x(<k) . (6)

p(X) =

k=1

To feed the channel tokens with dimension h × w into the transformer backbone, a two-layer MLP projector is applied to align their dimensionality with the LLM backbone.

Since channels do not possess an inherent order, we apply nested channel dropout during the tokenizer training phase to establish an ordered coarse-to-fine sequence for AR training. Given a token sequence

- Table 1: Reconstruction performance of different VQ methods on ImageNet-1K. The vanilla form of CVQ consistently improves reconstruction fidelity over existing VQ-based methods across both 256- and 1024-token budgets, outperforming established baselines such as MoVQ and VQ-LC

- at comparable vocabulary scales. † denotes models we reproduced using the same training setup as our CVQ tokenizer.

Codebook Size rFID ↓ SSIM↑ PSNR↑ Codebook Usage↑

VQ METHODS # Tokens Ratio

256 tokens Vanilla VQ† [10] 16×16 16 16,384 4.84 0.542 19.93 4.5% Dynamic VQ [16] 16×16 {8, 16, 32} 1,024 4.08 - - Dynamic VQ† [16] 16×16 {8, 16, 32} 16,384 3.62 0.558 20.07 5.2% VQ-LC [58] 16×16 16 16,384 3.01 0.564 - 99.0% VQ-LC [58] 16×16 16 100,000 2.62 0.589 - 99.0% SimVQ† [59] 16×16 16 16,384 2.63 0.556 21.10 100.0% Vanilla CVQ (ours) 256 16 16,384 2.60 0.565 20.94 100.0%

1024 tokens Vanilla VQ [10] 32×32 8 8,192 1.49 - - Vanilla VQ† [10] 32×32 8 16,384 1.32 0.692 23.35 2.8% VQ w/ rotation trick [12] 32×32 8 16,384 1.10 - - 27.0% VQ-FC [51] 32×32 8 8,192 1.28 - - 95.0% VQ-LC [58] 32×32 8 100,000 1.29 0.716 - 99.5% RVQ [22] 8×8×16 32 16,384 1.83 - - MoVQ [57] 16×16×4 16 1,024 1.12 0.673 22.42 MoVQ [57] 16×16×4 16 16,384 1.05 - - Vanilla CVQ (ours) 1024 16 16,384 0.88 0.723 25.02 100.0%

of length c, only the first ckeep channels are retained, where ckeep ∈ {1,...,c} is chosen randomly, while the remaining channels are masked to zero. This allows the model to learn a coarse-to-fine hierarchy [30], in which early channels capture global structure and later channels progressively encode finer details. We discuss the effect of the channel dropout strategy in Sec. 4.3. Additional details are provided in Appendix B.

### 4 Experiments

This section provides a comprehensive experimental analysis of CVQ’s performance in downstream image reconstruction and generation tasks. To ensure a strictly fair comparison, we maintain dimensionality parity between CVQ and VQ baselines throughout our experiments. By setting

- c = h × w = 256, both methods operate with identical lookup complexity, memory usage, and training overhead.

#### 4.1 Visual Reconstruction

Training Setup We train two versions of our vision tokenizer with 256 and 1024 dimensions, yielding 256 / 1024 tokens, respectively. Unless otherwise specified, we use a codebook size of 16,384. All models are trained on ImageNet-1K [9] at 256 × 256 resolution for 100 epochs. We use Adam (β1 = 0.5, β2 = 0.9) with a learning rate of 1 × 10−4, weight decay of 10−4, and a global batch size of 256. We further extend CVQ to variable resolution in Appendix D.

Main Results We measured reconstruction FID (rFID), PSNR, and SSIM on the ImageNet-1K (val). As shown in Table 1, compared to conventional VQ, whose codebook utilization collapses to only 4.5% when scaling the codebook size to 16,384, CVQ natively supports 100% codebook utilization without requiring any additional modifications. This outperforms prior VQ improvement methods such as VQGAN-FC, which require additional token factorization, and VQGAN-LC, which relies on initializing the codebook using features extracted from a pretrained CLIP model and introduces extra projector parameters. Consequently, CVQ demonstrates a remarkable and consistent improvement over traditional VQ-based methods across different token budgets. With 256 tokens, CVQ achieves an rFID of 2.60, significantly improving over vanilla VQGAN (4.99) and outperforming SimVQ (2.63) in reconstruction fidelity. Under the 1024-token setting, the improvement becomes more pronounced:

- Table 2: Quantitative comparison on text-to-image generation benchmarks. We categorize methods by attention mask into Bidirectional, Unidirectional, and Hybrid approaches. Bidirectional methods include diffusion and MaskGIT; unidirectional methods follow the standard AR paradigm of next-token prediction; while hybrid methods include the VAR family, which uses block-wise attention masks with full attention within each block. NPP, NSP, and NCP denote next-patch, next-scale, and next-channel prediction, respectively. Among them, only NPP and NCP follow the standard next-token prediction paradigm. † indicates results obtained with prompt rewriting or self-CoT.

GenEval↑ DPG↑ Two Obj. Position Color Attri. Overall Global Relation Overall Bidirectional

Type Methods # Para.

SDXL [28] 2.6B 0.74 0.15 0.23 0.55 83.27 86.76 74.65 SD3 (d=38) [11] 8B 0.89 0.34 0.47 0.71 - - Lumina-Next [61] 1.7B 0.49 - 0.15 0.55 86.89 86.59 80.50 FLUX-dev [21] 12B - - - 0.67 - - 84.00 FLUX-schnell [21] 12B - - - 0.71 - - 84.80 SANA [46] 1.6B 0.77 - 0.47 0.66 - 91.90 84.80

Diff.

TiTok [53] 0.6B - - - 0.49 - - TA-TiTok [20] 1.1B 0.58 0.13 0.34 0.55 - - -

Mask.

Hybrid

VARGPT-1.1 [60] 9B 0.53 0.13 0.21 0.53 84.83 88.13 78.59 STAR [25] 1.7B - - - 0.55 - - TokenFLow [29] 7B 0.72 0.45 0.42 0.63 78.72 85.22 73.38 HART [34] 0.7B - - - 0.56 - - 80.89 SWITTI [40] 3B - - - 0.62 - - -

NSP∗

Infinity [13] 2B 0.85† 0.49† 0.57† 0.73† 93.11 90.76 83.46 InfinityStar [23] 8B 0.90† 0.62† 0.67† 0.79† 91.68 91.87 86.55

Unidirectional

LlamaGen [33] 0.8B 0.34 0.07 0.04 0.32 - - 64.84 Chameleon [35] 7B - - - 0.39 - - -

Emu3 [41] 8B 0.81† 0.49† 0.45† 0.66† - - 80.60 Lumina-mGPT [61] 7B 0.77 - 0.32 0.56 - 91.29 79.68 Janus [43] 1.3B 0.68 0.46 0.42 0.61 - - Liquid [44] 7B 0.73 0.17 0.37 0.55 - - UniTok [24] 7B 0.71 0.26 0.45 0.59 - - MUSE-VL [48] 7B - - - 0.57† - - NextStep-1 [36] 14B - - - 0.73† - - 85.28

NPP∗

NCP∗ CAR (4B) 4B 0.88 0.63 0.58 0.75 86.98 93.62 83.82 CAR (8B) 8B 0.92 0.66 0.66 0.79 89.40 94.16 86.72

CVQ attains a lower rFID of 0.88 and a higher PSNR of 25.02 dB, surpassing strong baselines such as MoVQGAN (1.05 rFID) and VQGAN-LC (1.29 rFID).

#### 4.2 Visual Generation

Training Recipe Our CAR models are initialized from the pre-trained Qwen3-4B/8B backbones [50]. The training process is conducted on 80M text-image pairs and is divided into two stages:

- • Stage I: To align the CVQ features with the LLM latent space, we employ a 2-layer MLP projector. This projector maps the 256-dimensional channel embeddings to the hidden dimension of the LLM backbones (2560 for the 4B model and 4096 for the 8B model). During this stage, only the MLP projector and the LLM head are optimized, while the LLM backbone remains frozen.
- • Stage II: In the second stage, we perform end-to-end optimization across all parameters, including the MLP projector, the LLM backbone, and the LLM head. We list the data sources and training hyperparameters in the Appendix.

Main Results As illustrated in Fig. 5, CAR produces progressively detailed image content as more channels are generated. As shown in Table 2, among unidirectional methods, CAR (4B) achieves competitive or superior performance compared with strong AR baselines such as NextStep-1 (14B) and Emu3 (8B). Scaling CAR to 8B further improves performance, reaching a GenEval score of 0.79 and a DPG overall score of 86.72, competitive with strong VAR methods such as Infinity and

CAR: next channel prediction

…

[Figure 105]

1 tokens 4 tokens 8 tokens 16 tokens 32 tokens 64 tokens

128 tokens 256 tokens

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

A photo of a pizza

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Green trees and a river with mountains and clouds in the background

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

A snowy mountain.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

A lone figure, cloaked in a heavy mist, stands on an ancient cobblestone street...

#### Figure 5: The generation process of CAR.

InfinityStar. These results suggest that CVQ provides an effective token ordering for AR learning while preserving a simple next-token prediction formulation. Qualitative results are shown in Fig. 6.

In addition to semantic-alignment benchmarks, we report MJHQ-30K FID in Table 6. CAR achieves an FID of 6.42, outperforming both 1D masked-token baselines and the standard 2D-token baseline.

#### 4.3 Discussions

Codebook Size and Usage As shown in Table 3, scaling up the codebook size leads to a severe decline in utilization for patch-wise VQ, dropping from 20.3% to 1.1%, with only marginal improvements in rFID. In contrast, CVQ maintains nearly 100% utilization, even at massive scales (up to 65,536). This enables CVQ to effectively leverage larger vocabularies for improved performance, reducing rFID from 3.64 to 2.32. Notably, such advantage becomes increasingly evident at scale: at a codebook size of 65K, CVQ achieves a 52% improvement in reconstruction fidelity over the VQ baseline, demonstrating promising scaling properties.

Nested Dropout As shown in Table 4, CVQ without nested dropout exhibits weaker AR generation performance, as channel tokens do not possess an inherent order, whereas patch tokens still retain a natural spatial ordering despite suboptimal raster scanning. Yet, compared with 2D patch tokens, the 1D nature of channel-wise tokens makes it easy to impose a meaningful AR-friendly ordering. With a simple nested dropout strategy [30], CVQ learns a coarse-to-fine channel order, effectively improving AR performance while maintaining reconstruction quality. In contrast, such an ordering strategy is difficult to apply to 2D patch tokenization, as each patch token is strongly tied to local content and therefore cannot naturally represent a global coarse-to-fine progression. The implantation details of VQ w/ dropout is in Appendix. B.

- Table 3: Ablation on codebook usage across different codebook scales. Conventional VQ suffers from codebook collapse as the size increases, while CVQ maintains high utilization across all scales. This allows CVQ to effectively leverage larger vocabularies, achieving a rFID reduction of up to 52% compared to the VQ baseline.

VQ CVQ Util.↑ rFID ↓ Util.↑ rFID ↓

Codebook Size

1,024 20.3% 5.25 100% (+79.7) 3.64(↓31%) 4,096 12.6% 4.99 100% (+87.4) 3.07(↓38%)

16,384 4.5% 4.84 100% (+95.5) 2.60(↓46%) 65,536 1.1% 4.86 96.1% (+95.0) 2.32(↓52%)

Table 4: Ablation on nested dropout. (1) Standard patch-wise VQ is incompatible with nested dropout; (2) nested channel dropout substantially improves CVQ’s generation performance (GenEval +0.12, DPG +9.38) while maintaining reconstruction quality, effectively inducing an ARfriendly channel ordering without sacrificing reconstruction fidelity. (Trained on a 60M subset.)

Recon. Generation

Method

rFID ↓ SSIM↑ PSNR↑ GenEval↑ DPG↑ VQ w/o dropout 4.84 0.542 19.93 0.69 79.86

VQ w/ dropout 15.32 0.322 3.43 - CVQ w/o dropout 2.60 0.565 20.94 0.62 72.76

###### CVQ w/ dropout 2.62 0.568 21.01 0.74 (+0.12) 82.14 (+9.38)

[Figure 138]

Text Artistic Surrealistic Landscape Indoor Count Color Others

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

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

#### Figure 6: Qualitative results on text-to-image generation.

Versus 1D Tokenizers While related in target, CVQ and existing 1D tokenizer works operate at different conceptual levels: the former is a quantization method, whereas the latter primarily focus on additional modifications at the model architecture level (e.g., via learnable queries [53, 20, 17] or diffusion decoders [2, 42, 14]). CVQ’s 1D structure arises directly from the quantization process itself, enabling a 1D token sequence without specialized model architectural design.

Here, we provide a direct comparison with recent 1D tokenizers. Since these methods typically employ stronger training recipes, such as two-stage training, external proxy codes, or enhanced networks for LPIPS and GAN losses, we report results under both the standard VQGAN-style training protocol and a stronger TA-TiTok-style recipe in Table 5.

Table 5: Comparison with representative 1D tokenizers. †: standard VQGAN-style recipe, ‡: two-stage recipe with decoder reinforcement [53], ∗: models reproduced with the official codebase.

Method Diff. Dec. rFID ↓ SSIM↑ PSNR↑ FlexTok [2] ✓ 4.20 - FlexTok-1.4B [2] ✓ 1.45 0.465 18.53 SpectralAR‡ [17] ✗ 4.03 - TiTok-64† [53] ✗ 5.15 - TiTok-256†∗ [53] ✗ 3.84 0.542 19.92 CVQ† ✗ 2.60 0.565 20.94 Same enhanced training recipe [20] TA-TiTok-64 [20] ✗ 2.43 - TA-TiTok-128 [20] ✗ 1.53 - TiTok-64 [53] ✗ 4.25 - TiTok-128 [53] ✗ 2.63 - TiTok-256∗ [53] ✗ 1.51 0.558 20.05 CVQ ✗ 1.29 0.569 20.97

Table 6: FID performance on MJHQ-30K. We further evaluate text-to-image generation fidelity across different methods on MJHQ-30K.

Method Type 1D Token Res. FID ↓ SD-XL [28] Diff. ✗ 1024 9.55 PixArt [7] Diff. ✗ 512 6.14 FlowTok-H [14] Diff. ✓ 256 7.15 Show-o [47] Discrete Diff. ✗ 256 14.99 TiTok [53] Mask. ✓ 256 8.50 TA-TiTok [20] Mask. ✓ 256 7.51 STAR [25] NSP ✗ 256 5.67 SWITTI [40] NSP ✗ 512 9.50 SWITTI [40] NSP ✗ 1024 8.10 LlamaGen [33] NTP ✗ 512 25.59 VILA-U [45] NTP ✗ 256 12.81 Janus [43] NTP ✗ 256 10.10 UniTok [24] NTP ✗ 256 7.46 DualToken [32] NTP ✗ 256 7.88 CAR NTP ✓ 256 6.42

### 5 Conclusion and Future Works

Conclusion In this work, we introduce CVQ, a simple yet effective quantization paradigm that discretizes images along the channel dimension. CVQ achieves high codebook utilization and high reconstruction fidelity without architectural modifications or auxiliary loss terms. Building upon CVQ, we proposed CAR, a generative framework that shifts the autoregressive paradigm from traditional spatial patch prediction to next-channel prediction. Our results highlight channel-wise tokens as a promising direction for autoregressive image generation and offer new insight into rethinking the fundamental unit of visual tokenization.

Future Works Despite the encouraging results, several important directions remain for future work. First, CVQ can be naturally combined with recent advances in VQ, such as SimVQ [59] and IBQ [31], to further improve representational capacity. Second, the autoregressive formulation of CAR makes it a natural fit for unified vision models that jointly perform visual understanding and generation within a single architecture. Finally, extending the channel-wise quantization paradigm to the temporal dimension represents a promising direction for learning efficient and compact video representations.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3
- [2] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=DgdOkUUBzf. 3, 9, 16
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 2020. 3
- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022. 2
- [5] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021. 18
- [6] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 18
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 9
- [8] Yitong Chen, Zuxuan Wu, Xipeng Qiu, and Yu-Gang Jiang. Catok: Taming mean flows for one-dimensional causal image tokenization. In CVPR, 2026. 3
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009. 6
- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021. 2, 3, 5, 6
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 7

- [12] Christopher Fifty, Ronald G Junkins, Dennis Duan, Aniketh Iyengar, Jerry W Liu, Ehsan Amid, Sebastian Thrun, and Christopher Ré. Restructuring vector quantization with the rotation trick. The Thirteenth International Conference on Learning Representations, 2025. 2, 6
- [13] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15733–15744,

2025. 3, 7

- [14] Ju He, Qihang Yu, Qihao Liu, and Liang-Chieh Chen. Flowtok: Flowing seamlessly across text and image tokens. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16629–16640, 2025. 9
- [15] Kaiming He and Jian Sun. Statistics of patch offsets for image completion. In European conference on computer vision, pp. 16–29. Springer, 2012. 2, 4
- [16] Mengqi Huang, Zhendong Mao, Zhuowei Chen, and Yongdong Zhang. Towards accurate image coding: Improved autoregressive image generation with dynamic vector quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22596–22605, June 2023. 6
- [17] Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Yueqi Duan, Jie Zhou, and Jiwen Lu. Spectralar: Spectral autoregressive visual generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 3, 9
- [18] Zhihao Huang, Xi Qiu, Yukuo Ma, Yifu Zhou, Junjie Chen, Hongyuan Zhang, Chi Zhang, and Xuelong Li. Nfig: multi-scale autoregressive image generation via frequency ordering. arXiv preprint arXiv:2503.07076, 2025. 2
- [19] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017. 5
- [20] Dongwon Kim, Ju He, Qihang Yu, Chenglin Yang, Xiaohui Shen, Suha Kwak, and LiangChieh Chen. Democratizing text-to-image masked generative models with compact text-aware one-dimensional tokens. arXiv preprint arXiv:2501.07730, 2025. 7, 9
- [21] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. 7
- [22] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11523–11532, 2022. 3, 6
- [23] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. arXiv preprint arXiv:2511.04675, 2025. 7
- [24] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025. 2, 3, 7, 9
- [25] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-to-image generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 7, 9
- [26] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=8ishA3LxN8. 3
- [27] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 45–55, 2025. 2

- [28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. The Twelfth International Conference on Learning Representations, 2024. 7, 9
- [29] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 7
- [30] Oren Rippel, Michael Gelbart, and Ryan Adams. Learning ordered representations with nested dropout. In International Conference on Machine Learning, pp. 1746–1754. PMLR, 2014. 2, 3, 6, 8, 16
- [31] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2, 3, 4, 10
- [32] Wei Song, Yuran Wang, Zijia Song, Yadong Li, Zenan Zhou, Long Chen, Jianhua Xu, Jiaqi Wang, and Kaicheng Yu. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025. 9
- [33] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2, 3, 7, 9
- [34] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. The Thirteenth International Conference on Learning Representations,

2025. 7

- [35] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2, 3, 7
- [36] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, Hongyu Zhou, Kenkun Liu, Ailin Huang, Bin Wang, Changxin Miao, Deshan Sun, En Yu, Fukun Yin, Gang Yu, Hao Nie, Haoran Lv, Hanpeng Hu, Jia Wang, Jian Zhou, Jianjian Sun, Kaijun Tan, Kang An, Kangheng Lin, Liang Zhao, Mei Chen, Peng Xing, Rui Wang, Shiyu Liu, Shutao Xia, Tianhao You, Wei Ji, Xianfang Zeng, Xin Han, Xuelin Zhang, Yana Wei, Yanming Xu, Yimin Jiang, Yingming Wang, Yu Zhou, Yucheng Han, Ziyang Meng, Binxing Jiao, Daxin Jiang, Xiangyu Zhang, and Yibo Zhu. Nextstep1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025. 7
- [37] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 2024. 2, 3
- [38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [39] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In Advances in Neural Information Processing Systems, 2017. 2, 3, 5
- [40] Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, and Dmitry Baranchuk. Switti: Designing scale-wise transformers for text-to-image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025. 7, 9
- [41] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 3, 7

- [42] Xin Wen, Bingchen Zhao, Ismail Elezi, Jiankang Deng, and Xiaojuan Qi. “Principal components” enable a new language of images. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 3, 9, 16
- [43] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 7, 9
- [44] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators. International Journal of Computer Vision, 2026. 7
- [45] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. The Thirteenth International Conference on Learning Representations, 2025. 9
- [46] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. The Thirteenth International Conference on Learning Representations, 2025. 7
- [47] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 9

- [48] Rongchang Xie, Chen Du, Ping Song, and Chang Liu. Muse-vl: Modeling unified vlm through semantic discrete encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24135–24146, 2025. 7
- [49] Yilun Xu, Yang Song, Sahaj Garg, Linyuan Gong, Rui Shu, Aditya Grover, and Stefano Ermon. Anytime sampling for autoregressive models via ordered autoencoding. arXiv preprint arXiv:2102.11495, 2021. 3
- [50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2, 3, 7
- [51] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In The Tenth International Conference on Learning Representations, 2022. 2, 3, 6
- [52] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion

- tokenizer is key to visual generation. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=gzqrANCF4g. 3

- [53] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 2024. 3, 7, 9
- [54] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18431–18441, 2025. 2
- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018. 5

- [56] Anlin Zheng, Haochen Wang, Yucheng Zhao, Weipeng Deng, Tiancai Wang, Xiangyu Zhang, and Xiaojuan Qi. Holistic tokenizer for autoregressive image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16916–16926, October

2025. 2, 3

- [57] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022. 3, 6
- [58] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%. In Advances in Neural Information Processing Systems,

2024. 2, 3, 4, 6

- [59] Yongxin Zhu, Bocheng Li, Yifei Xin, Zhihua Xia, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2, 3, 4, 6, 10
- [60] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, and Yuexian Zou. Vargpt-v1.1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning. arXiv preprint arXiv:2504.02949, 2025. 7
- [61] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 2024. 7

### A More t-SNE Visualizations of Embedding Distributions.

Patch-wise embedding

Channel-wise embedding

Patch-wise embedding

Channel-wise embedding

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

[Figure 214]

[Figure 215]

Patch-wise embedding

Channel-wise embedding

Patch-wise embedding

Channel-wise embedding

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

[Figure 222]

[Figure 223]

- Img 0
- Img 1

Patch-wise embedding

Channel-wise embedding

Patch-wise embedding

Channel-wise embedding

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

- Img 0
- Img 1

[Figure 230]

[Figure 231]

- Img 0
- Img 1

Figure 7: More visualizations of feature map channels

### B Details and Analysis of the Nested Channel Dropout Strategy

#### B.1 Implementation Details of Nested Channel Dropout

During the training phase, we introduce a structured dropout mechanism to enforce a coarse-to-fine hierarchy within the latent space. For a given latent representation Z ∈ Rh×w×c, we define a truncated configuration where only the first ckeep channels are retained, while the remaining c − ckeep channels are masked (set to zero). This truncated latent, denoted as Z⊥c

, effectively forces the model to compress the most critical visual information into the lower-indexed channels. The optimization objective for a specific channel configuration ckeep is formulated as follows:

keep

), (7)

Lnested(ckeep) = Lrecon(Z⊥c

##### )+Lquant(Z⊥c

##### )+Llpips(Z⊥c

##### )+λGAN(ckeep)LGAN(Z⊥c

keep

keep

keep

keep

where Lrecon and Llpips denote the standard reconstruction and perceptual losses, respectively. The quantization loss Lquant is computed exclusively over the active ckeep channels to ensure valid codebook mapping at reduced dimensions.

A key challenge in training nested representations is that the discriminator often suffers from instability when operating on extremely sparse latent features, which can lead to unnatural artifacts in reconstructions using few channels. To mitigate this, we introduce an adaptive GAN weight λGAN governed by a sigmoid function:

##### λ0 1 + e−η(c

keep−2c) , (8)

λGAN(ckeep) =

where η = 0.05 controls the transition smoothness, and λ0 = 1 is the base GAN weight. This formulation ensures that the adversarial influence is gradually introduced as the channel capacity increases, preserving stable convergence for sparse configurations.

To balance hierarchical representation learning with overall reconstruction fidelity, we adopt a stochastic training strategy. In each iteration, nested channel dropout is performed with probability α (the dropout ratio), where the number of active channels ckeep is sampled from a discrete uniform

- Table 7: Ablation on nested channel dropout ratio. We evaluate the impact of varying the dropout ratio α on reconstruction fidelity and downstream text-to-image generation. While reconstruction performance remain stable across different ratios, generation performance improves significantly with the introduction of channel dropout, with comparable gains observed at α = 25% and α = 50%.

α

Recon. Generation

rFID ↓ SSIM↑ PSNR↑ Geneval↑ DPG↑ 0% 2.60 0.565 20.94 0.62 72.76

25% 2.62 0.568 21.01 0.74 (+0.12) 82.14 (+9.38) 50% 2.61 0.562 20.88 0.75 (+0.13) 81.96 (+9.20)

distribution U(1,c). Conversely, with probability 1 − α, the model undergoes standard training using the full channel configuration. The resulting hybrid optimization objective is formulated as:

Lsum = α · Ec

keep∼U(1,c)[Lnested(ckeep)] + (1 − α)Ltotal, (9) where Ltotal represents the loss under the complete channel configuration (ckeep = c). B.2 Analysis of Nested Channel Dropout

[30] demonstrates that a nested dropout strategy, which stochastically removes nested sets of hidden units, enforces an ordered representation where importance decreases with the dimension index. For semi-linear autoencoders (comprising a linear or sigmoidal encoder with a linear decoder) under L2 loss, they establish that: (i) its optimal solutions are a subset of standard autoencoder optima, preserving global performance (Theorem 1); (ii) the weight matrices must be commutative in their truncation and inversion (Theorem 2); and (iii) with an orthonormality constraint (ΓTΓ = IK), the model uniquely recovers the PCA solution (Theorem 3).

However, these theoretical guarantees necessitate stringent conditions: the architecture must be semi-linear and optimized via L2 reconstruction loss. While the proposed unit sweeping technique fixes converged units to mitigate gradient decay at higher indices, its iterative nature remains computationally expensive for modern deep learning.

Nevertheless, empirical evidence [2, 42] suggests that even with deeper architectures, complex loss functions, and advanced frameworks like diffusion models, randomly dropping tokens in a nested manner produces ordered representations that benefit AR generation without significant loss in expressivity. As shown in Table 7, reconstruction quality is preserved, indicating no degradation in representational capacity. Additionally, generative performance at 25% and 50% dropout exceeds the 0% baseline, suggesting that structured ordering facilitates AR generation, with similar results at 25% and 50% indicating that a moderate degree of nested order is sufficient.

Dropout Ratio Table 7 illustrates the influence of the channel dropout ratio on both reconstruction and generation performance. We observe that reconstruction metrics remain stable across the tested range, with no significant degradation in rFID, SSIM, or PSNR as the dropout ratio increases from 0% to 50%. However, the generation capability is substantially enhanced by the introduction of dropout; specifically, the GenEval and DPG scores show a marked improvement when moving from 0% to 25%, increasing from 0.61 to 0.72 and 72.76 to 82.13, respectively. Further increasing the dropout ratio to 50% results in comparable generation performance to the 25% setting, suggesting that a moderate dropout ratio is sufficient to optimize the model’s generative robustness without sacrificing its reconstruction fidelity.

C Progressive Channel Analysis

Finally, we analyze how reconstruction changes as more channels are revealed to the decoder. We progressively keep the first n channels and mask the remaining channels before decoding.

- Table 8 shows a clear coarse-to-fine trend. From 32 to 128 channels, rFID, SSIM, and PSNR all improve rapidly, indicating that early channels capture global appearance, object identity, and coarse structure. After 128 channels, rFID improvement slows down, while SSIM and PSNR continue to increase steadily. This indicates that later channels mainly refine local structure and high-frequency texture, providing quantitative evidence that channel-wise decomposition naturally organizes visual information from global semantics to fine details.

- Table 8: Reconstruction with progressively added channels. We keep only the first n channels before decoding. Early channels rapidly recover global appearance and semantic structure, while later channels mainly refine local structure and texture.

# Channels rFID ↓ SSIM↑ PSNR↑

32 30.24 0.410 16.96 64 11.84 0.472 18.38 96 6.91 0.506 19.21

128 4.05 0.524 19.82 160 3.46 0.533 20.17 192 3.03 0.539 20.39 224 2.88 0.542 20.55 256 2.63 0.568 21.01

D Variable-Resolution Extension

In the main experiments, we keep a fixed 256 × 256 resolution setting to enable strictly controlled comparison with VQ, yielding codewords with a spatial size of 16 × 16.

Here, we show a simple practice that extends CVQ to variable resolutions with lightweight resampling modules before and after quantization: before lookup, a fixed set of learnable queries cross-attend features of arbitrary spatial size (h × w) to a fixed size (h0 × w0); after lookup, we dynamically generate h × w target queries to project the quantized channels back to the decoder’s desired spatial size. This plays a role analogous to the pre- and post-quantization projections commonly used in patch-wise VQ tokenizers, while preserving CVQ’s channel-token formulation.

For variable-resolution training, we progressively scale input resolution from 256 to 512 and 1024. For patch-wise VQ, the embedding dim is fixed at 256, yielding token counts of 256, 1024, and 4096, respectively. For CVQ, we also fix the embedding dim to 256 (16 × 16) and control the token budget to the intermediate value of the VQ setting (i.e., 1024) for a fair comparison.

- Table 9: Variable-resolution reconstruction. CVQ is trained with a fixed codeword size and uses resolution-dependent resampling before and after quantization.

Method Train Res. Inference Res. rFID ↓ PSNR↑ VQ 256 256 4.84 19.93 CVQ 256 256 0.88 25.02 VQ variable 256 4.97 19.56 CVQ variable 256 0.92 24.96 VQ variable 512 2.05 22.01 CVQ variable 512 0.96 24.55

- Table 10: Resolution scaling under matched token budgets. When CVQ is allocated the same number of tokens as VQ at each resolution, its reconstruction quality also improves with resolution.

Method Token Budget Train Res. Inference Res. rFID ↓/ PSNR↑ VQ 256 variable 256 4.97 / 19.56 CVQ 256 variable 256 2.63 / 21.06 VQ 1024 variable 512 2.05 / 22.01 CVQ 1024 variable 512 0.96 / 24.55

As shown in Table 9, CVQ generalizes across resolutions and consistently outperforms the corresponding VQ baseline. When the token budget is held fixed, CVQ may slightly trade reconstruction fidelity for efficiency at higher resolutions because each token must cover a larger spatial support. This is a controlled quality-efficiency trade-off rather than an inherent inability to model high-resolution images. As shown in Table 10, when CVQ is allocated the same token budget as VQ at each resolution, reconstruction quality improves with resolution and remains stronger than VQ.

Discussion CVQ provides a practical advantage for autoregressive generation by decoupling token count from image resolution. While patch-wise methods exhibit quadratic token growth as the latent

grid scales, CAR maintains nearly constant inference cost by keeping the channel sequence length fixed. Instead of increasing token count, CVQ encodes spatial resolution within each channel token.

### E Hyperparameters and Data Source

Our CAR training process consists of two stages. During Stage I, only the MLP projector and the LLM head are optimized with a learning rate of 1 × 10−4, while the LLM backbone remains frozen. During Stage II, all components are trained with a learning rate of 2 × 10−5. For both stages, we use the AdamW optimizer with β1 = 0.9, β2 = 0.96, and a weight decay of 1 × 10−3.

The training data includes a filtered subset of ImageNet-21K, LAION-Aesthetics-12M, CC12M [5], Megalith-10M 4, BLIP-3o-short [6], BLIP-3o-long [6], and a 6M in-house aesthetics dataset.

4Re-captioned by Qwen3-VL.

