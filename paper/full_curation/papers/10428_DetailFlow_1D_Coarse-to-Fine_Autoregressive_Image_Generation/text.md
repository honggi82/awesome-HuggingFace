# arXiv:2505.21473v2[cs.CV]11Nov2025

## DetailFlow: 1D Coarse-to-Fine Autoregressive Image Generation via Next-Detail Prediction

#### Yiheng Liu∗, Liao Qu∗, Huichao Zhang, Xu Wang†, Yi Jiang, Yiming Gao, Hu Ye, Xian Li, Shuai Wang, Daniel K. Du, Fangmin Chen, Zehuan Yuan, Xinglong Wu ByteDance Inc. https://github.com/ByteFlow-AI/DetailFlow

16×16 48×48 64×64 96×96 128×128 160×160 192×192 224×224 256×256

[Figure 1]

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

gFID(lowerisbetter)

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

8 tokens 16 tokens 24 tokens 32 tokens 48 tokens 64 tokens 80 tokens 96 tokens 128 tokens

Coarse-to-fine 1D AR generation

(a)

| | | | | | | |
|---|---|---|---|---|---|---|
| |Sm|aller circl|e indicates|faster inf|erence spe|ed|
| |L|lamaGen| | | | |
| | | |PAR| |ViT-V|QGAN|
| | | | | | | |
| | | | |VAR| | |
| | | | |FlexV|AR| |
| |Det|ailFlow-16 DetailF|low-32| | | |
| | | |Det|ailFlow-64| | |
| | | | | | | |

4.50

4.25

4.00

3.75

3.50

3.25

3.00

2.75

2.50

200 400 600 800 1000

Token Sequence Length (less is better) (b)

Figure 1: (a) Progressive generation results from DetailFlow. Our proposed 1D tokenizer encodes tokens with an inherent semantic ordering, where each subsequent token contributes additional high-resolution information. The sequences illustrate how image resolution and inferred 1D tokens incrementally increase from left to right. (b) Comparison of our DetailFlow approach with existing methods, showing that DetailFlow achieves better image quality with fewer tokens and times.

### Abstract

This paper presents DetailFlow, a coarse-to-fine 1D autoregressive (AR) image generation method that models images through a novel next-detail prediction strategy. By learning a resolution-aware token sequence supervised with progressively degraded images, DetailFlow enables the generation process to start from the global structure and incrementally refine details. This coarse-to-fine 1D token sequence aligns well with the autoregressive inference mechanism, providing a more natural and efficient way for the AR model to generate complex visual content. Our compact 1D AR model achieves high-quality image synthesis with significantly fewer tokens than previous approaches, i.e. VAR/VQGAN. We further propose a parallel inference mechanism with self-correction that accelerates generation speed by approximately 8× while reducing accumulation sampling error inherent in teacher-forcing supervision. On the ImageNet 256×256 benchmark, our method achieves 2.96 gFID with 128 tokens, outperforming VAR (3.3 FID) and FlexVAR (3.05 FID), which both require 680 tokens in their AR models. Moreover, due to the significantly reduced token count and parallel inference mechanism, our method runs nearly 2× faster inference speed compared to VAR and FlexVAR. Extensive experimental results demonstrate DetailFlow’s superior generation quality and efficiency compared to existing state-of-the-art methods.

1⋆ Equal contribution. 2† Project leader.

Preprint. Under review.

### 1 Introduction

Autoregressive (AR) models like [30, 31, 4, 9, 1, 17, 40, 22] have demonstrated exceptional success in natural language processing through their scalability, flexibility, and ability to model complex sequential dependencies. Building on these strengths, researchers have extended AR modeling to image generation, creating unified frameworks [38, 43, 46, 29] for visual generation tasks. AR image generation enables structured, step-by-step synthesis, offering advantages in controllability and multimodal integration.

Conventional AR image generation methods adopts raster scan approaches [13, 48, 43], which flattens 2D image tokens into 1D sequences, forcing models to predict patches in a counter-intuitive order that disrupts spatial continuity. Recent work Visual Autoregressive Modeling (VAR) [39] adopts a next-scale prediction framework that emulates human sketching via coarse-to-fine 2D parallel generation. However, it requires extensive volumes of multi-scale tokens, particularly at high resolutions. For instance, Infinity [15] requires 10,521 tokens to synthesize a 1024×1024 image, leading to substantial computational and memory overhead during training. This bottleneck highlights a critical challenge in balancing generation quality with efficiency for high-resolution AR image synthesis.

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

|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|
|---|---|---|---|---|---|---|---|---|

(a) Next-token/next-patch prediction

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

|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|𝒛|
|---|---|---|---|---|---|---|---|---|---|

[Figure 63]

⋯

[Figure 64]

###### Modeling feature residual

[Figure 65]

(b) VAR: next-scale prediction

Figure 2: Comparison of different prediction strategies in image generation. (a) Traditional 2D raster-scan nexttoken/next-patch prediction. (b) Next-scale prediction in VAR [39]. (c) Our proposed next-detail prediction, which predicts 1D tokens encoding fine-grained details for high-resolution image generation.

Recent line of research [49, 2] attempt to address this challenge through querybased attention mechanisms that compress 2D images into adaptive 1D token sequences. These approaches remove constraints associated with fixed spatial positions, enabling adaptive compression of spatial redundancy. This significantly reduces token counts, alleviating computational overhead during generation. However, current 1D tokenization methods remain inherently constrained by resolution-specific tokenizers, limiting flexibility in generating images at arbitrary resolutions without retraining.

To address these limitations, we propose DetailFlow, a novel coarse-to-fine 1D tokenizer explicitly establishing a semantic and resolution-dependent mapping between token sequences and image resolutions. Specifically, DetailFlow employs progressively degraded images to supervise token sequences of increasing lengths, inherently embedding a coarse-to-fine semantic ordering within the tokens. Consequently, during inference, DetailFlow autoregressively generates tokens in a coarseto-fine manner, progressively outputing higher-resolution images with enriched visual details, as illustrated in Fig. 1(a). This approach allows 256×256 images to be represented with just 128 semantically ordered tokens, significantly fewer than VAR-based methods. With similar AR model sizes, DetailFlow achieves a gFID score below 3, demonstrating superior quality and efficiency.

Furthemore, traditional VQGAN-based raster-scan methods inherently disable parallel prediction of successive tokens due to spatial dependencies. DetailFlow’s 1D learnable latent space offers greater flexibility. This enables substantial acceleration of inference speed through parallel successive token prediction (speedup proportional to the parallel token numbers).

During experimentation, we observe that sampling errors significantly degrade subsequent generation quality due to autoregressive teacher forcing training regimes. To mitigate this accumulation errors, we introduce a self-correction training strategy during tokenizer learning. Specifically, controlled quantization errors are introduced during token quantization, and subsequent tokens are trained to correct these inaccuracies. This approach fosters token sequences capable of self-correction, significantly enhancing overall generation quality and providing an effective solution to mitigate error accumulation during autoregressive inference.

To sum up, our contributions include:

- • Next-detail prediction paradigm. As shown in 2, we introduce a novel coarse-to-fine 1D autoregressive image generation framework that progressively refines images from global structures to fine details. This 1D coarse-to-fine token sequence is more aligned with the inference paradigm of autoregressive models.
- • Improved token efficiency. DetailFlow significantly reduces token requirements, achieving 2.96 gFID using a 326M parameter AR model with only 128 tokens and nearly 2× faster inference speed on the ImageNet 256×256 benchmark, compared to 680 tokens required by recent state-of-the-art methods such as VAR [39] and FlexVAR [19].
- • Accelerated parallel inference. Our parallel decoding mechanism combined with a selfcorrection training strategy boosts inference speed by ∼8×, simultaneously mitigating the error accumulation typically observed in autoregressive models.
- • Dynamic-resolution 1D tokenization. DetailFlow uniquely supports dynamic resolution in 1D tokenizers, employing a single 1D tokenizer capable of generating variable-length token sequences, thereby enabling flexible image decoding at multiple resolutions without additional retraining.

### 2 Related Work

#### 2.1 Image tokenizer

The image tokenizer is crucial for generative tasks, as it affects both the quality of generation and the model architecture. An image tokenizer typically encodes high-dimensional visual data into a compressed latent space, representing the image as a sequence of discrete tokens. This process is then reversed by a decoder to reconstruct the original image. Numerous studies have explored and validated this general framework, demonstrating its effectiveness in various model designs.

2D image tokenizer. Early work such as VQ-VAE [42, 32] introduces a discrete latent representation using vector quantization, enabling token-based image modeling by structuring images as grids of

- 2D tokens in the latent space. VQGAN [13] improves this approach by incorporating adversarial training to improve perceptual quality. Efficient-VQGAN [5] focuses on reducing computational overhead while maintaining reconstruction quality. RQ-VAE [23] adopts residual quantization to enrich the representation capacity. MoVQ [50] employs a multi-codebook design for more flexible token utilization.

1D image tokenizer. Unlike 2D tokenizers, which retain spatial structure information in the generated token sequence, the 1D tokenizer TiTok[49] integrates 2D image information into a 1D token sequence through a self-attention mechanism. This process removes spatial redundancy, thereby enhancing information compression capabilities. However, these tokens lack an inherent order, which poses challenges for their application in next-token prediction within autoregressive models. FlexTok [2] addresses this problem by using a tail-drop tokenizer training strategy, which forces information to focus on earlier tokens, thus generating a coarse-to-fine ordered token sequence. However, under a 1.33B AR model, the generation performance, as measured by gFID, deteriorates from approximately

- 1.9 at 32 tokens to 2.5 at 256 tokens. This limitation, where high-quality generation is only achievable with a small number of tokens, restricts its ability to scale to higher-resolution images that require more tokens for reconstruction.
- 2.2 Visual generation

Autoregressive models. The definition of an appropriate token sequence is crucial for autoregressive models in image generation tasks. Inspired by NLP, the next-token prediction paradigm for sequentially generating discrete image tokens has been naturally extended to decoder-only Transformer architectures. VQGAN [13], Parti [48] and EMU3 [43] adopt a raster-scan strategy to arrange 2D image tokens into a 1D ordered token sequence by scanning row by row. While this method facilitates token-level prediction in a manner analogous to text generation, alternative approaches have explored more structured generation schemes. Notably, VAR [39] introduces a next-scale prediction framework, which departs from the conventional sequential token prediction by generating coarser-to-finer representations across multiple scales. This hierarchical strategy enables the model to capture global

(a) Coarse-to-fine tokenizer training (b) Self-correction training

Image tokens Query tokens

|𝒄 𝒄|
|---|

|𝒄 𝒄|
|---|

[Figure 66]

|𝒒 𝒒|
|---|

|⋯|
|---|

|𝒒 𝒒|
|---|

|𝒒 𝒒|
|---|

|𝒒 𝒒|
|---|

𝒒

⋯

Standard Quantization

Top-50 Noisy Quantization

ViT Encoder

Clean tokens Noisy tokens

|𝒄 𝒄| |
|---|---|
| | |

|𝒄 𝒄|
|---|

|𝒄 𝒄|
|---|

|𝒄 𝒄|
|---|

⋯

𝒄

||𝒛 𝒛|
|---|
<br><br>|𝒛 𝒛|
|---|
<br><br>𝒁 𝒁|
|---|

High-resolution encoding

###### Quantization

𝒁

Mask tokens

𝒁 𝒁

𝒁

Projection

|⋯|
|---|

|𝒛 𝒛|
|---|

|𝒛 𝒛|
|---|

|𝒛 𝒛|
|---|

||×|
|---|
<br><br>|×|
|---|
|
|---|

⋯

|×|
|---|

Image tokens

New Query tokens

|⋯|
|---|

⋯

ViT Decoder

[Figure 67]

Low-resolution reconstruction

ViT Encoder

|⋯|
|---|

| | |
|---|---|
| | |

⋯

(c) AR model training and decoding

Quantization

[Figure 68]

𝒁

𝒁 𝒁 𝒁

Correction tokens Sequence with noisy tokens

[Figure 69]

|𝒛 𝒛|
|---|

|𝒛 𝒛|
|---|

|𝒛 𝒛|
|---|

𝒛 𝒛 ⋯

Target tokens

||𝒛 𝒛|
|---|
<br><br>𝒁<br><br>|𝒛 𝒛|
|---|
<br><br>𝒁|
|---|

||𝒛 𝒛|
|---|
<br><br>𝒁<br><br>|𝒛 𝒛|
|---|
<br><br>⋯<br><br>|
|---|

| | |
|---|---|
|𝒛 𝒛| |

⋯

[Figure 70]

Autoregressive Transformer [C] 𝒛 ⋯

Self-correction token sequence

Decoder Decoding

|𝒛 𝒛|
|---|

|𝒛 [M]<br><br>|
|---|

Input tokens

𝒁 𝒁 :  𝒁 : 

|Dropped tokens Class token Mask token<br><br>|×|
|---|
<br><br>𝒛 Noisy token [C] [M]<br><br>|𝒛 𝒛|
|---|
<br><br>Token group|
|---|

- Figure 3: (a) Coarse-to-fine tokenizer training. The encoder maps high-resolution images to 1D latent token sequences. Decoding with more tokens yields higher-resolution outputs, with earlier tokens capturing global structure and later ones refining details. (b) Self-correction training. Randomly perturbed tokens are re-encoded, and encourages subsequent tokens to correct errors from earlier noisy tokens. (c) Autoregressive (AR) model training and decoding. AR model predicts the first group of tokens in a next-token prediction manner, followed by parallel prediction of subsequent groups. At inference, more predicted tokens lead to higher-resolution outputs.

image structure before refining local details. Similarly, CART [33] proposes a "next-detail" prediction strategy that compositionally generates an image by first predicting a "base" factor, which captures the global structure, and then iteratively adding "detail" factors to refine local features.

Masked-prediction model. MaskGIT [7] is a transformer-based image synthesis framework that leverages masked visual token modeling to generate images through parallel decoding. Unlike autoregressive models, it predicts all tokens simultaneously and iteratively refines them. TiTok [49] uses it for image generation with 1D token sequence. MUSE [6] extends MaskGIT to text-to-image generation by integrating a pretrained language model with MaskGIT’s parallel decoding.

Diffusion models synthesize data through a forward-backward stochastic process: gradually corrupting data with Gaussian noise and learning to reverse this degradation via iterative denoising. Latent diffusion models [34] first proposes modeling in the latent space using a U-Net architecture. DiT [28] replaces conventional U-Net with a transformer architecture for latent image processing, demonstrating superior scalability and achieving better image generation quality.

### 3 Method

Encoding images into a 2D grid of tokens is a natural and widely adopted strategy, leveraging the inherent spatial structure of images. However, using a 1D tokenizer allows for more efficient compression, representing images with fewer tokens and offering greater flexibility in controlling the information content of each token. This flexibility is critical for balancing image quality and computational efficiency.

#### 3.1 Preliminary Background on 1D Tokenizer

The general structure of tokenizers typically consist of an encoder, a quantizer, and a decoder. The encoder embeds an input image into continuous image tokens, the quantizer discretizes them into discrete tokens, and the decoder reverses this process to reconstruct the original image. Similarly, the 1D tokenizer adheres to this paradigm but introduces slight modifications to the encoder and decoder.

H f ×Wf ×D, where f is the patch size and D is the patch feature dimension. These patches are concatenated with a set of learnable 1D query tokens Q ∈ RN×D and fed into the encoder. N is the number of query tokens. The encoder outputs continuous latent tokens C ∈ RN×D, which are then discretized by a quantizer into discrete tokens Z ∈ RN×d using a codebook of dimension d.

Given an input image X ∈ RH×W×3, it is first patchified to non-overlapping patches P ∈ R

For reconstruction, the discrete tokens Z are first projected and then concatenated with learnable mask tokens M ∈ R

H f ×Wf ×D [3], created by duplicating a single mask embedding m ∈ R1×D. The decoder output corresponding to the mask tokens is regressed to the pixel values through a linear projection. 2D position embeddings are applied for image tokens and mask tokens. 1D position embeddings are applied for query tokens and latent tokens.

#### 3.2 Coarse-to-Fine 1D Latent Representation

Human perception and image creation are inherently hierarchical processes, starting with a rough global structure and progressively refining local details. This hierarchical approach reduces complexity and improves quality by breaking down the task into simpler steps. Inspired by this, we design a coarse-to-fine information ordering for 1D latent tokens, enabling the model to progressively generate images from global to fine-grained details.

As illustrated in Fig. 3(a), we enforce an information ordering in the 1D latent space by leveraging the correlation between image resolution and semantic granularity: lower-resolution images primarily preserve global structure, while higher-resolution images capture increasingly detailed content. We define a resolution mapping function R(n) linking the number of used tokens n to a target resolution

√

hw = R(n). Early tokens are trained to capture coarse structures at low resolutions, while later tokens refine high-frequency details. To enforce this, we use causal (unidirectional) attention among query tokens in the encoder, while maintaining bidirectional attention for image tokens in the encoder and all the tokens in decoder.

rn =

During training, we randomly sample n ∈ [1,N] and reconstruct a downsampled version of X at resolution R(n) using only the first n latent tokens Z1:n = {z1,z2,...,zn}. Mask tokens and positional embeddings of the decoder are adjusted according to the target image size. The model is supervised to reconstruct downsampled image from this partially observed latent sequence, ensuring that the earlier tokens specialize in capturing global structure while later tokens incrementally contribute finer details.

Formally, the conditional entropy of the i-th token, H(zi | Z1:i−1), quantifies the incremental information it contributes. The total entropy up to token n is:

n

H(zi | Z1:i−1). (1)

H(rn) =

i=1

By leveraging this hierarchical decomposition, our method ensures that each token contributes meaningfully to the reconstruction of image details, enabling a progressive and efficient representation of high-dimensional image data.

Assuming the entropy per pixel is H(r = 1), the total image entropy at resolution r × r scales as:

H(r) ∝ r2H(r = 1), (2)

indicating a nonlinear relationship between the number of tokens and reconstructable resolution. To model this, we define R(n) (shown in Fig. 4(b)) as:

R − 1 (N − 1)α b

(N − n)α, where R(1) = 1, R(N) = R. (3)

R(n) = R − b(N − n)α = R −

R is the maximum resolution supported by the tokenizer, and coefficient b is calculated using the condition R(1) = 1, ensuring the first token corresponds to a resolution of 1 × 1. The condition R(N) = R ensures the total N tokens corresponds to the maximum resolution R. The hyperparameter α controls the degree of nonlinearity. The continuous resolution value R(n) is rounded to the nearest multiple of f to obtain the final target resolution for compatibility with the decoder’s patch size f. For simplicity, this step is omitted in formulas and visualizations.

#### 3.3 Parallel Inference Acceleration

Generating high-resolution images often requires thousands of tokens, making purely sequential next-token prediction inefficient. To address this, we partition the 1D token sequence into M groups of g tokens each. During tokenization, we apply bidirectional attention within groups and causal attention across groups. For coarse-to-fine training, we randomly sample an integer k ∈ [1,M] and reconstruct the downsampled image using the first k groups Z1:k = {Z1,Z2,...,Zk}.

For the autoregressive (AR) model, considering that the first group controls the global structure, which is extremely important, we keep the tokens in the first group as causal attention [44] and use next-token prediction as shown in Fig. 3(c). Subsequent groups use bidirectional attention internally and causal attention between groups, enabling parallel prediction of g tokens per step.

However, the independent sampling of tokens within a group during parallel inference disrupts intra-group dependencies, introducing sampling errors. The teacher-forcing training paradigm does not equip the AR model with the ability to self-correct such errors. To mitigate this, prior works [23, 45] introduce a depth transformer head to sequentially predict tokens within a group, but at the cost of model complexity. Alternatively, Infinity [15] simulates self-correction by randomly flipping a quantized value for one scale and recalculating the residuals at subsequent scales, enabling the AR model to learn self-correction capabilities from self-correction token sequences.

Inspired by these ideas, we need to obtain a 1D self-correction token sequence where a sampling error occurs at one position, and the subsequent token sequence can correct the sampling error. Therefore, we inject stochastic perturbations into the quantization process. Specifically, as shown in Fig. 3(b) we randomly select a token group Cm,m ∈ [1,k − 1] and, during quantization, sample each token from the top-50 nearest codebook entries, yielding a noisy group Z˜m. To obtain the subsequent correlation tokens, we introduce a correction mechanism: feeding the clean tokens Z1:m−1, noisy tokens Z˜m (with gradient truncation), and new query tokens back into the encoder. A projection layer ensures dimensional compatibility. The encoder then generates corrected tokens {Zˆm+1,...,ZˆM}, forming a self-correction sequence {Z1:m−1,Z˜m,Zˆm+1:k}.

To jointly train standard sequence modeling and self-correction, we concatenate the original sequence Z1:k and the self-correction sequence {Z1:m−1,Z˜m,Zˆm+1:k} of the same input image along the batch dimension, independently decoding both to reconstruct the same target image. This trains the encoder to learn to output subsequent token sequences that can correct sampling errors in the preceding tokens. This design enables the tokenizer to provide both standard and self-correction sequences simultaneously, which enables the AR model to learn self-correction capabilities alongside standard token modeling, improving image quality during parallel inference.

#### 3.4 Training Objective

Since early tokens encode global structure, we explicitly align the first latent token z1 with the globally pooled features extracted by the pre-trained Siglip2 model [41]. This allows the first token

to better embed global information. Specifically, the first token z1 is projected through a three-layer MLP and aligned via cosine similarity as the alignment loss:

Lalign = −cos(MLP(z1),Siglip2(X)). (4)

In addition to the alignment loss, the final training objective of the tokenizer also includes the reconstruction loss, perceptual loss [11, 20], adversarial loss [14, 18] and VQ codebook loss [12], following the implementations and weighting schemes used in SoftVQ-VAE [8].

### 4 Experiments

#### 4.1 Implementation

Tokenizer Setup. The encoder including 12 layers is initialized with the weigths of Siglip2-NaFlex [41], yielding a parameter count of 184M. In contrast, the decoder is trained from scratch, comprising 86M parameters. The discrete latent space is defined by a codebook with 8,192 entries and a dimension of 8. Tokenizer training is conducted on the ImageNet-1K [10], using 256×256 resolution inputs to the encoder and dynamically varying output resolutions (up to 256 × 256) from the decoder.

Table 1: Comparison of class-conditional image generation on ImageNet-1k at 256 × 256 resolution. Models marked with † leverage additional training data beyond ImageNet. The ‡ symbol indicates methods that do not employ classifier-free guidance. The ⋄ symbol denotes models that generate images at 384 × 384 resolution, which are subsequently downsampled to 256 × 256 for evaluation. #Tokens represents the total number of tokens predicted during autoregressive (AR) model training. Flex indicates whether the AR model supports dynamic resolution, meaning it can decode images of different resolutions using varying numbers of tokens. For DetailFlow, the notation 16*8 indicates that the 128 tokens are partitioned into M = 16 groups, each containing g = 8 tokens. Reported evaluation metrics include rFID and gFID. The inference steps of FlexTok [2] include the steps of the AR model and the Diffusion model.

Type Tokenizer rFID↓ Generator Type Param. #Tokens gFID↓ Step Time(s) Flex Continuous modeling

2D GigaGAN[21] - GigaGAN[21] GAN 569M - 3.45 1 - ✓ 2D VAE†[34] 0.27 LDM-4[34] Diff. 400M 4096 3.60 250 - ✓ 2D SD-VAE[36] 0.62 SiT-XL/2[27] Diff. 675M 1024 2.06 250 - ✓ 2D VAE[24] 0.53 MAR-H[24] AR+Diff. 943M 256 1.55 64 28.24 ✓

Discrete modeling

|1D TiTok-S[49] 1.71<br><br>1D FlexTok[2] 1.45<br>2D ImageFolder[25] 0.80<br>|MaskGIT[7] Mask. 287M 128 1.97 64 0.13 ✗<br><br>LlamaGen[37] AR+Diff. 1.33B<br><br>32 1.86‡ 57 - ✗ 256 ∼2.5‡ 281 - ✗ VAR-GPT[39] VAR 362M 286*2 2.60 10 0.13 ✗<br><br>|
|---|---|
|2D VQGAN†[7] 2.28 2D ViT-VQGAN†[47] 1.28 2D RQ-VAE[23] 3.20 2D LlamaGen[37] 2.19 2D O.-MAGVIT2[26] 1.17 2D PAR[44] 0.94 2D VAR†[39] 0.90 2D FlexVAR†[19] -|MaskGIT[7] Mask. 227M 256 6.18‡ 8 0.13 ✓ VIM-Large[47] AR 1.7B 1024 4.17‡ 1024 >6.38 ✓ RQTran.-re[23] AR 3.8B 256 3.8‡ 64 5.58 ✓ GPT-L[37] AR 343M 256 3.8 256 12.58 ✓ AR-B[26] AR 343M 256 3.08 256 - ✓ PAR-L-4[44] AR 343M 576 3.76⋄ 147 3.38 ✓ VAR-d16[39] VAR 310M 680 3.30 10 0.15 ✓ VAR-d16[19] VAR 310M 680 3.05 10 0.15 ✓<br><br>|
|1D DetailFlow-16 1.22 1D DetailFlow-32 0.80 1D DetailFlow-64 0.55<br><br>|GPT-L [37] AR 326M 16*8 2.96 23 0.08 ✓ GPT-L [37] AR 326M 32*8 2.75 39 0.16 ✓ GPT-L [37] AR 326M 64*8 2.62 71 0.38 ✓<br><br>|

We train the tokenizer for 250 epochs with a batch size of 256 using a cosine learning rate decay strategy, starting from an initial learning rate of 1e-4. To ensure robust modeling of the entire latent token sequence, we reconstruct full-resolution images (n = N) with an 80% probability, and with a 20% probability, we randomly reconstruct lower-resolution images by sampling the first n = kg tokens, where k ∈ [1,M −1],g = 8,M = N/g. We achieve full codebook utilization (100%) across all tokenizers.

AR Model Setup. For downstream generation, we adopt an autoregressive (AR) model based on the LlamaGen architecture [37]. The AR model is trained on ImageNet-1K using a cosine-decayed learning rate schedule for 300 epochs. 30% of the training data consists of curated self-correction token sequences. During inference, we apply sampling with Top-K=8192 and Top-P=1. ClassifierFree Guidance (CFG) is tuned to its optimal value to balance generation diversity and fidelity. The default CFG value is 1.5. The inference time is measured on a single A100 using the batch size 1.

Metrics. We evaluate image reconstruction quality on the ImageNet-1K validation set using rFID [16], Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM). For assessing image generation quality, we report Frechet Inception Distance (FID) [16], Inception Score (IS) [35], as well as Precision and Recall metrics.

#### 4.2 State-of-the-art image generation

In Table 1, we evaluate our proposed method, DetailFlow, on the ImageNet 256 × 256 benchmark, comparing it against a range of state-of-the-art generative models—including GANs [21], diffusion models [34, 36, 24], masked prediction models [7, 2], and autoregressive (AR) models [37, 26, 39, 19].

Reconstruction w/ Error Correction

1.375

Reconstruction w/o Error Correction

Baseline (No Noise)

1.350

1.325

rFID(lowerisbetter)

1.300

1.275

1.250

1.225

1.200

2 4 6 8 10 12 14 16

Noised Group Index (a)

250

200

200

150

FID(lowerisbetter)

Resolution

150

100

gFID rFID

100

(n)

50

50

0

20 40 60 80 100 120

Token number (b)

N = 64

6.5

N = 128

6.0

gFID(lowerisbetter)

5.5

5.0

4.5

4.0

3.5

3.0

0.5 1.0 1.5 2.0

(c)

- Figure 4: (a) Reconstruction metrics before and after self-correction when adding noise to latent tokens of a group (tokenizer with 128 tokens, group size 8, trained for 200 epochs). (b) Impact of token count on image resolution, reconstruction quality (rFID), and generation quality (gFID), with all evaluations conducted on images resized to 256 × 256. The tokenizer is identical to (a). (c) Influence of the hyperparameter α in the mapping function R(n) on generation metrics, using tokenizers trained for 50 epochs.

Compared to existing 2D tokenizers [37, 26, 39, 19] that support dynamic resolution, our method delivers higher quality with shorter sequence length among ar models. DetailFlow-16 achieves a lower gFID of 2.96 using only 128 tokens, surpassing VAR (3.3 FID) [39] and FlexVAR (3.05 FID) [19], which both require 680 tokens. Additionally, its reduced token count and parallel inference make it nearly twice as fast as VAR and FlexVAR during inference. This is attributed to the fact that the 1D tokenizer effectively eliminates spatial information redundancy, allowing more information to be carried with fewer tokens. Although ImageFolder [25] reports a gFID score comparable to ours, its lack of support for dynamic resolution constrains its practical applicability. While ImageFolder employs a 1D tokenizer architecture, the resulting latent tokens are still arranged in a 2D structure and preserve explicit spatial information, limiting it to decoding images at a fixed resolution.

Compared to existing 1D tokenizers [49, 2], our approach addresses several key limitations. Notably, prior 1D tokenizers do not support multi-resolution image generation and parallel inference with self-correction mechanisms. Additionally, TiTok [49] lacks an explicit and structured token ordering necessary for effective AR modeling, while FlexTok [2] demonstrates limited scalability, with performance degrading as token count increases. For instance, using a 1.33B AR model, its gFID score rises from roughly 1.9 at 32 tokens to about 2.5 at 256 tokens. In contrast, DetailFlow supports coarse-to-fine image generation (Fig. 1(a)), allowing for the prediction of more tokens to decode higher-resolution images. It also supports parallel inference with self-correction to accelerate the image generation process, making it both scalable and efficient.

#### 4.3 Ablation Study

In Fig. 4(a), we analyzes the reconstruction performance before and after self-correction when noise is injected into different groups of latent tokens. When noise is applied to earlier token groups (i.e., smaller group indices), reconstruction quality deteriorates sharply. This is expected, as early tokens encode the global structure, and inaccuracies in this region propagate large-scale distortions throughout the image. Moreover, in these cases, the self-correction mechanism exhibits limited effectiveness, since later tokens—responsible mainly for fine details—lack the capacity to rectify global errors. As the group index increases, the model demonstrates strong ability to correct errors introduced by noisy tokens. However, when noise is injected into the final few groups, the correction performance again deteriorates, as fewer subsequent tokens are available to facilitate error correction.

Table 2: Ablation study of DetailFlow. These tokenizers with N = 128 latent tokens are trained with only 50 epochs. Reported evaluation metrics include rFID, PSNR, gFID and Recall.

|Setting|rFID↓ PSNR↑<br><br>|gFID↓ Rec↑ Step|
|---|---|---|
|Baseline<br><br>+causal encoder<br><br>+coarse-to-fine<br><br>+parallel(g = 4)<br><br>+self-correction<br><br>+first group causal<br><br>+alignment loss<br><br>|1.73 19.47 1.87 19.49 1.92 19.31 1.87 19.27<br><br>1.81 19.16 1.68 19.05<br><br>|3.97 0.50 128 3.66 0.53 128<br><br>3.33 0.54 128<br>4.11 0.50 32 3.68 0.51 32 3.59 0.51 35 3.35 0.55 35<br>|

These results underscore the importance of early tokens for capturing global structure. To enhance their reliability, our model incorporates self-correction training, causal modeling for the first token group, and an alignment loss on the initial token. We further perform an ablation study to evaluate each component‘s contribution. Starting from a baseline that encodes images into an unordered token sequence, we progressively add modules to measure their effects.

First, introducing a causal encoder establishes a simple sequential order among the tokens, which substantially improves the model’s capacity for autoregressive generation. Building on this, we implement a coarse-to-fine tokenizer training strategy by supervising reconstructions at multiple resolutions. The observed improvements in the gFID metric from 3.66 to 3.33 validate that enforcing such coarse-to-fine semantic ordering is both effective and advantageous.

Next, we explore parallel prediction of token groups. Although this design reduces the inference steps from 128 to 32, it introduces degradation in generation quality, primarily due to the accumulation of sampling errors across groups. To mitigate this, we incorporate a self-correction mechanism, which substantially restores synthesis quality by allowing the model to iteratively refine predictions. This reduces the gFID score from 4.11 to 3.68. Fig. 5 shows that selfcorrection training improves global structure and detail quality, suggesting that self-correction effectively mitigates the impact of sampling errors.

Self-correction training

[Figure 71]

[Figure 72]

[Figure 73]

w/

[Figure 74]

[Figure 75]

[Figure 76]

w/o

Figure 5: Qualitative comparison of AR model outputs with (w/) and without (w/o) self-correction training.

Further enhancement is achieved by applying causal next-token prediction specifically to the first group of tokens, yielding an additional improvement of 0.09 in gFID. This stabilizes the generation process and improves the fidelity of the final output. Finally, to strengthen global semantic consistency, we align the first token’s representation with Siglip2 global image features via an alignment loss. This further lowers the gFID from 3.59 to 3.35, indicating that anchoring the initial token to global structural information provides stronger guidance for the entire generation process.

- Fig. 4(b) shows how increasing the number of latent tokens impacts output resolution, reconstruction metrics, and generation metrics. As the number of tokens grows, the latent sequence encodes finer-grained information, facilitating higher-resolution image decoding. Both reconstruction and generation metrics improve correspondingly, with the autoregressive model benefiting from richer detail prediction, leading to enhanced image quality and resolution.
- Fig. 4(c) examines the effect of hyperparameter α in the mapping function R(n) on generation metrics. The parameter α governs the relationship between token quantity and decoding resolution. Smaller values of α are preferable, as they indicate that fewer additional tokens are required to support higher-resolution images. Across different total token counts, the optimal value of α is consistently 1.5. This value, greater than 1, reflects that higher-resolution regions demand more tokens compared to lower-resolution regions, aligning with Eq. 2. Meanwhile, α being less than 2 highlights the tokenizer’s ability to compress spatial redundancy inherent in the image data.

### 5 Conclusion

In this paper, we propose a novel autoregressive image generation method DetailFlow, introducing a new image generation paradigm called Next-Detail Prediction. By leveraging a 1D tokenizer trained on progressively degraded images, DetailFlow establishes a direct correspondence between token sequences and image resolution levels, enabling a coarse-to-fine generation strategy that enhances visual fidelity. This method effectively compresses image information into a smaller token sequence while maintaining high-quality image generation. Furthermore, a parallel decoding mechanism with self-correction improves inference speed without compromising image quality. Overall, DetailFlow achieves an effective balance among training cost, inference efficiency, and image quality, offering a scalable solution for high-resolution, autoregressive image synthesis.

### References

- [1] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [2] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025.
- [3] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In Proceedings of the International Conference on Learning Representations (ICLR).
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 33:1877– 1901, 2020.
- [5] Shiyue Cao, Yueqin Yin, Lianghua Huang, Yu Liu, Xin Zhao, Deli Zhao, and Kaigi Huang. Efficient-vqgan: Towards high-resolution image generation with efficient vision transformers. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 7368–7377, 2023.
- [6] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.
- [7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 11315–11325, 2022.
- [8] Hao Chen, Ze Wang, Xiang Li, Ximeng Sun, Fangyi Chen, Jiang Liu, Jindong Wang, Bhiksha Raj, Zicheng Liu, and Emad Barsoum. Softvq-vae: Efficient 1-dimensional continuous tokenizer. arXiv preprint arXiv:2412.10958, 2024.
- [9] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1– 113, 2023.
- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255. Ieee, 2009.
- [11] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. Advances in Neural Information Processing Systems (NeurIPS), 29, 2016.
- [12] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021.
- [13] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2020.
- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.
- [15] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024.

- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.
- [17] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [18] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.
- [19] Siyu Jiao, Gengwei Zhang, Yinlong Qian, Jiancheng Huang, Yao Zhao, Humphrey Shi, Lin Ma, Yunchao Wei, and Zequn Jie. Flexvar: Flexible visual autoregressive modeling without residual prediction. arXiv preprint arXiv:2502.20313, 2025.
- [20] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Proceedings of the European Conference on Computer Vision (ECCV), pages 694–711. Springer, 2016.
- [21] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 10124–10134, 2023.
- [22] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. 2023.
- [23] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 11523–11532, 2022.
- [24] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems (NeurIPS), 37:56424–56445, 2024.
- [25] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. In Proceedings of the International Conference on Learning Representations (ICLR).
- [26] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Openmagvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.
- [27] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In Proceedings of the European Conference on Computer Vision (ECCV), pages 23–40. Springer, 2024.
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.
- [29] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024.
- [30] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. San Francisco, CA, USA, 2018.
- [31] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [32] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

- [33] Siddharth Roheda, Rohit Chowdhury, Aniruddha Bala, and Rohan Jaiswal. Cart: Compositional auto-regressive transformer for image generation. arXiv preprint arXiv:2411.10180, 2024.
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.
- [35] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems (NeurIPS), 29, 2016.
- [36] StabilityAI. Sd-vae.
- [37] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [38] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [39] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in Neural Information Processing Systems (NeurIPS), 37:84839–84865, 2024.
- [40] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [41] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [42] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017.
- [43] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [44] Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, and Xihui Liu. Parallelized autoregressive visual generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [45] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.
- [46] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [47] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.
- [48] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

- [49] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems (NeurIPS), 37:128940–128966, 2024.
- [50] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems (NeurIPS), 35:23412–23425, 2022.

### A Technical Appendices and Supplementary Material

- A.1 Implementation Details

- A.1.1 Tokenizer Setup

We initialize the encoder using the SigLIP2-NaFlex weights and retain only the first 12 layers to reduce memory consumption, resulting in an encoder of 184M parameters. As shown in Fig. 6(a), the attention design of the encoder includes bidirectional attention within image tokens and within each latent token group (except the first), causal attention within the first latent group, and causal attention across groups.

- A.1.2 AR model Setup

𝒑 𝒑 𝒑 𝒑 𝒑 𝒒 𝒒 𝒒 𝒒 𝒒 𝒒

Image tokens Query tokens

[C]

[M]

𝒛 𝒛

𝒛 𝒛 𝒛 𝒛 𝒛 𝒛

Next-token prediction

Next-group parallel prediction

(a) Attention mask of encoder (b) Attention mask of AR model

Figure 6: The attention mask in the proposed method. We use a group size g = 2 for latent tokens as an example.

Building on the autoregressive LlamaGen architecture, we incorporate learnable mask tokens inspired by PAR [44] to enable parallel decoding. We also use the 2D positional encoding to enhance the model’s ability to distinguish between different token groups and intra-group tokens. As illustrated in Fig. 6(b), the attention pattern of the AR model is structured such that the first group uses causal attention for next-token prediction, while subsequent groups apply bidirectional attention internally and causal attention across groups. This configuration allows for parallel prediction of later groups while preserving autoregressive dependencies. Training is conducted primarily on 16 A800 GPUs.

- A.2 Additional Results

- A.2.1 Additional Ablation Study

Impact of Coarse-to-fine Training Probability. During the training process of DetailFlow, we introduce a probabilistic strategy to balance between full-resolution and degraded-resolution image reconstruction. Specifically, with a certain probability, the model is trained with complete token sequences to reconstruct full-resolution images, while with the remaining probability, it is trained on downsampled images with partial tokens to encourage learning the coarse-to-fine ordering of token representations.

Table 3 reports an ablation study on the impact of different probabilities of coarse-to-fine training on image generation quality. The results show that even a 20% probability of degraded-resolution training enables the tokenizer to learn a hierarchical token structure effectively. However, increasing this probability further shifts training focus toward reconstructing downsampled images with partial tokens, which hinders the model’s capacity to learn full-sequence representations. These findings highlight the importance of balancing coarse and fine training to optimize efficiency under limited computational resources.

- Table 3: The impact of coarse-to-fine training probability on reconstruction and generation metrics. These tokenizers with N = 128 latent tokens are trained with only 50 epochs and the GPT-L AR model are trained with 100 epochs. Reported evaluation metrics include rFID, PSNR, SSIM, gFID, sFID, Inception Score(IS), Precision (Pre) and Recall (Rec).

|Probability|rFID↓ PSNR↑ SSIM↑|gFID↓ sFID↓ IS↑ Pre↑ Rec↑|
|---|---|---|
|10% 20% 30%<br><br>|2.0 19.3 0.61<br><br>1.8 19.2 0.60<br><br>1.9 19.3 0.61<br><br><br>|4.18 5.60 202.6 0.85 0.51<br><br>3.81 5.46 230.1 0.86 0.51<br>4.18 5.90 215.3 0.86 0.49<br>|

- Table 4: The impact of group size g on reconstruction and generation metrics. These tokenizers with N = 256 latent tokens are trained with 250 epochs and the GPT-L AR model are trained with 300 epochs. Reported evaluation metrics include rFID, PSNR, SSIM, gFID, sFID, Inception Score(IS), Precision (Pre) and Recall (Rec).

|Group|rFID↓ PSNR↑ SSIM↑<br><br>|gFID↓ sFID↓ IS↑ Pre↑ Rec↑ Step|
|---|---|---|
|32*8 (g = 8) 16*16 (g = 16)<br><br>|0.80 20.8 0.67<br>0.81 20.8 0.68<br>|2.75 5.77 250.8 0.81 0.58 39 2.88 6.03 238.2 0.80 0.59 31<br><br>|

- Table 5: The impact of Classifier-Free Guidance (CFG) on generation metrics. The tokenizer is DetailFlow-32 with N = 256 latent tokens. Reported evaluation metrics include gFID, sFID, Inception Score(IS), Precision (Pre) and Recall (Rec).

|CFG|gFID↓ sFID↓ IS↑ Pre↑ Rec↑<br><br>|
|---|---|
|1.4 1.5 1.6 1.7|2.79 5.87 229.1 0.80 0.60 2.75 5.77 250.8 0.81 0.58<br><br>2.81 5.66 268.1 0.82 0.57<br><br>3.02 5.61 283.2 0.83 0.56<br>|

Impact of Group Size on Parallel Token Prediction. DetailFlow employs a parallel decoding strategy that predicts g tokens simultaneously to accelerate inference by reducing the number of decoding steps. Table 4 analyzes the impact of varying group sizes g on image reconstruction and generation, with the total token length fixed at N = 256. Results indicate that reconstruction quality remains largely unaffected by changes in g, likely because the total number of tokens is constant. However, image generation quality shows a slight decline of 0.13 gFID as g increases, potentially due to higher sampling noise from parallel prediction. Despite this, the degradation is minimal, suggesting that the self-correction training effectively counteracts sampling errors. This allows for increased parallelism and faster inference with only a marginal loss in image quality.

Impact of Classifier-Free Guidance. Table 5 presents the impact of varying the Classifier-Free Guidance (CFG) scale on generation quality using DetailFlow-32 as the tokenizer. It can be observed that an appropriate CFG value 1.5 can effectively balance generation quality and diversity.

A.2.2 State-of-the-art image generation

- Table 6 presents a detailed comparison between our proposed method DetailFlow and state-of-the-art approaches under similar AR model sizes. The results demonstrate that our model, DetailFlow-16, achieves a higher gFID score and significantly outperforms existing methods in terms of Recall metric, while attaining comparable performance in Precision metric. These findings indicate that DetailFlow-16 is capable of generating higher-quality images using fewer tokens and at a faster inference speed.

Furthermore, as the token count increases to 256 and 512, the performance of DetailFlow continues to improve. In particular, DetailFlow-32 strikes a strong balance across image quality, diversity, and generation speed. This suggests that, with a comparable or lower training cost (in terms of token number) and inference cost, DetailFlow consistently outperforms existing models, highlighting its efficiency and effectiveness in autoregressive image generation.

Table 6: Comparison of AR methods for class-conditional image generation on ImageNet-1k at 256 × 256 resolution under similar AR model size. Models marked with † leverage additional training data beyond ImageNet. The ⋄ symbol denotes models that generate images at 384 × 384 resolution, which are subsequently downsampled to 256 × 256 for evaluation. O.-MAGVIT2 means Open-MAGVIT2. Reported evaluation metrics include rFID, PSNR, gFID, Inception Score(IS), Precision (Pre) and Recall (Rec).

|Method|rFID↓PSNR↑<br><br>|GeneratorgFID↓ IS↑ Pre↑ Rec↑ Param.#TokensStepTime(s)|
|---|---|---|
|LlamaGen[37] O.-MAGVIT2[26] PAR[44] VAR†[39]<br><br>FlexVAR†[19] DetailFlow-16 DetailFlow-32 DetailFlow-64<br><br>|2.19 20.79 1.17 22.64 0.94 -<br><br>0.9 -<br><br>- -<br><br>1.22 19.39 0.80 20.80 0.55 22.49<br><br><br>|GPT-L 3.81 248.3 0.83 0.52 343M 256 256 12.58<br><br>AR-B 3.08 258.3 0.85 0.51 343M 256 256 PAR-L-4 3.76⋄ 218.9⋄0.84⋄0.50⋄ 343M 576 147 3.38 VAR-d16 3.30 274.4 0.84 0.51 310M 680 10 0.15 VAR-d16 3.05 291.3 0.83 0.52 310M 680 10 0.15<br><br>GPT-L 2.96 221.4 0.82 0.57 326M 16*8 23 0.08 GPT-L 2.75 250.8 0.81 0.58 326M 32*8 39 0.16 GPT-L 2.62 245.3 0.80 0.60 326M 64*8 71 0.38<br><br>|

#### A.3 Future Work

To ensure fair comparison with existing methods, the tokenizer in our experiments is trained on square images with equal height and width. However, our proposed framework, DetailFlow, is not inherently restricted to this setting. Both its encoder and decoder adopt the SigLIP2-NaFlex architecture, which natively supports inputs of arbitrary resolution and aspect ratio. By resizing positional encodings to match input dimensions, the model remains compatible with non-square images. Under this design, the implicit, learnable 1D latent tokens effectively represent images of any resolution or aspect ratio.

To enable autoregressive (AR) models to generate images with a specified aspect ratio, it is essential to condition the model during both training and inference. This can be achieved by incorporating aspect ratio information via natural language prompts or special tokens that encode the target ratio. Given the relationship between image resolution and token count rn =

√

hw = R(n), the model is guided to predict a specific number of latent tokens corresponding to the desired resolution and aspect ratio.

#### A.4 Limitations

DetailFlow achieves efficient token compression by embedding 2D image information into a 1D coarse-to-fine token sequence using a query-token-driven tokenizer. However, this design introduces limitations, particularly in high-resolution image reconstruction. Capturing fine-grained visual details often requires several thousand latent tokens, substantially increasing the tokenizer’s computational cost during training.

In contrast, conventional 2D tokenizers, with spatially consistent strategies, are trainable on lowresolution images and generalize effectively to higher resolutions. 1D tokenizer lacks this scalability, making it less efficient in high-resolution settings.

To mitigate the high training cost, a progressive training strategy proves effective. Since both the encoder and decoder support variable input resolutions, training can begin with low-resolution images and less latent tokens to establish robust encoding and decoding. The model is then fine-tuned on high-resolution data, enabling adaptation to finer spatial details without retraining from scratch. This approach enhances training efficiency while preserving the model’s flexibility across resolutions.

#### A.5 More Visual Results

Fig. 7 and Fig. 8 present reconstruction and generation examples produced by DetaiFlow across various categories. As the number of tokens increases, both the detail and resolution of the images improve accordingly, illustrating a clear coarse-to-fine progression. This observation highlights the effectiveness of our coarse-to-fine training strategy, which imposes a structured semantic order on the token sequence that aligns well with the autoregressive prediction paradigm.

16x16 48x48 64x64 96x96 128x128 160x160 192x192 224x224 256x256

###### GT

|[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]|
|---|

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

8 tokens 16 tokens 24 tokens 32 tokens 48 tokens 64 tokens 80 tokens 96 tokens 128 tokens

Coarse-to-fine Reconstruction

- Figure 7: Progressive reconstruction results from DetailFlow-16. Our method encodes 2D image content into a coarse-to-fine 1D token sequence, with early tokens capturing global structure and later tokens introducing details necessary for both finer texture and higher resolution. As the number of tokens increases, the reconstructed images exhibit progressive improvements in both detail and resolution. The last column is the original Ground Truth (GT) image.

16x16 48x48 64x64 96x96 112x112 128x128 160x160 192x192 224x224 256x256

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

[Figure 279]

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

8 tokens 16 tokens 24 tokens 32 tokens 40 tokens 48 tokens 64 tokens 80 tokens 96 tokens 128 tokens

Coarse-to-fine AR generation

- Figure 8: Progressive generation results from DetailFlow-16. Our proposed 1D tokenizer encodes tokens with an inherent semantic ordering, where each subsequent token contributes additional high-resolution information. The sequences illustrate how image resolution and inferred 1D tokens incrementally increase from left to right.

