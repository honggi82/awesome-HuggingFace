# arXiv:2504.00999v1[cs.CV]1Apr2025

## MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization

Siyuan Li1,3∗ Luyuan Zhang2∗ Zedong Wang4 Juanxi Tian3 Cheng Tan1,3 Zicheng Liu1,3 Chang Yu3 Qingsong Xie5† Haonan Lu5 Haoqian Wang2 Zhen Lei6,7,8† 1Zhejiang University 2Tsinghua University 3Westlake University 4The Hong Kong University of Science and Technology 5OPPO AI Center 6CAIR, HKISI-CAS 7MAIS CASIA 8University of Chinese Academy of Sciences

##### Abstract

Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great success in both self-supervised pre-training and image generation. However, most existing methods struggle to address the trade-off in shared latent space for generation quality vs. representation learning and efficiency. To push the limits of this paradigm, we propose MergeVQ, which incorporates token merging techniques into VQ-based autoregressive generative models to bridge the gap between visual generation and representation learning in a unified architecture. During pretraining, MergeVQ decouples top-k semantics from latent space with a token merge module after self-attention blocks in the encoder for subsequent Look-up Free Quantization (LFQ) and global alignment and recovers their fine-grained details through cross-attention in the decoder for reconstruction. As for the second-stage generation, we introduce MergeAR, which performs KV Cache compression for efficient raster-order prediction. Experiments on ImageNet verify that MergeVQ as an AR generative model achieves competitive performance in both representation learning and image generation tasks while maintaining favorable token efficiency and inference speed. The source code will be available at https://apexgen-x.github.io/ MergeVQ.

##### 1. Introduction

Vector Quantization (VQ) [60] has garnered increasing attention for its ability to encode continuous visual signals into discrete tokens, enabling autoregressive (AR) models to process visual modalities. Since VQGAN [21], most visual AR generative models have adopted a two-stage design: first encode signals into discrete latent space for pretraining, then generate them with an autoregressive Transformer. Besides generation, BEiT [3] proposed Masked Image Modeling (MIM) based on the VQ framework, achiev-

*Equal contribution. †Corresonding authors.

|1|1|0|0|0|0| |
|---|---|---|---|---|---|---|
| | | | | | |𝐿 × 𝐾|
|0|0|1|1|0|0| |
| | | | | | | |
|0|0|0|0|1|0| |
|0|0|0|0|0|1| |

Ground-truth

Source Recovery

Source Matrix

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

| | |
|---|---|
| |𝑋෠|
| | |

| |
|---|

| |
|---|

| |
|---|

ToMe Encoder

𝑋

Decoder

| |
|---|

| |
|---|

| |
|---|

LFQ

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

(a) MergeVQ Tokenizer Pre-training

|1|1|0|0|0|0| |
|---|---|---|---|---|---|---|
| | | | | | |𝐿 × 𝐾|
|0|0|1|1|0|0| |
| | | | | | | |
|0|0|0|0|1|0| |
|0|0|0|0|0|1| |

Predicted

| |
|---|

Random-order NTP

Source Matrix

| |
|---|

| |
|---|

| | |
|---|---|
| |𝑋෠|
| | |

| |
|---|

| |
|---|

|𝑃1|
|---|

|𝑃3|
|---|

|𝑃5|
|---|

|𝑃6|
|---|

Source Recovery

|[S]|
|---|

Decoder

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

Start

| |
|---|

| |
|---|

Randomized AR Generator

| |
|---|

(b) Randomized AR Generation

|𝑃1|
|---|

|𝑃1|
|---|

|𝑃1|
|---|

|𝑃1|
|---|

|𝑃3|
|---|

|𝑃3|
|---|

|𝑃3|
|---|

|𝑃5|
|---|

|𝑃6|
|---|

…

| |
|---|

Position Cache

… … …

| |
|---|

| |
|---|

Is duplicate?

|[S]|
|---|

| |
|---|

|[S]|
|---|

| |
|---|

| |
|---|

| |
|---|

Is duplicate?

| | |
|---|---|
| |𝑋෠|
| | |

Start

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

Decoder

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

…

KVCache

| |
|---|

| |
|---|

Raster-order NTP

(c) MergeAR Generation

Figure 1. MergeVQ learning paradigms. (a) The MergeVQ Tokenizer extracts K semantic tokens with decoupled positional information (retained in the source matrix) by ToMe [7] while quantizing spatial details by LFQ [49, 73], which will be recovered and reconstructed correspondingly. (b) MergeVQ with randomorder Generator [51] generates K discrete tokens with associated position instructions while trained Source Prediction and decoder restore position details. (c) MergeAR Generator predicts L tokens efficiently in a raster-order with tailored KV Cache compression to remove the redundancy within Next-token Prediction (NTP) [57].

ing successful latent-based pretraining [36, 38] and thus attracting growing interest in unifying visual representation learning and generation tasks in a shared latent space [82].

However, recent studies [45, 83] have shown that visual generation and representation capabilities often lack consistency [72] under a VQ-based learning framework, i.e., improvements in one task may not necessarily benefit the others. This inconsistency is conjectured to arise from the

competing objectives for identical embedding space: representation learning tasks emphasize inter-class discrimination to maximize high-level semantics, while generative tasks prioritize the reconstruction of details. In addition, training obstacles brought by VQ itself further limit the optimization process. For example, the gradient approximation in canonical VQ (e.g., VQGAN) sets an optimization bottleneck for the first-stage training. Moreover, the quantization of embedding space inevitably strips away fine-grained spatial information, which requires the models to reconstruct images with the loss of details and thus affects both the representation learning and generation.

As such, efforts have been made to extract rich semantic features from visual signals for quantization to improve the representation capacity of generative models [62, 85]. However, these coarse-grained semantics often sacrifice detailed information, making it difficult to support high-quality image reconstruction and generation, resulting in significant performance degradation. In this paper, we argue that representation learning and generation are not completely conflicting but with intrinsic complementarity. The crux lies in exploiting such complementarity while minimizing the information loss, which requires specific designs. To achieve this, we propose to decouple coarse-grained semantics from latent space during training and recover them for reconstruction to meet the different needs while minimizing the information loss and overhead. By leveraging token merging techniques [7], the encoder compresses latent space into K semantic tokens while preserving the fine-grained spatial information as positions within a source matrix, as illustrated in Figure 1. During reconstruction, the latent fine-grained details can be restored with this source matrix, while the K compressed tokens serve as high-level semantics for global alignment [10, 82]. Based on this intuition, we propose MergeVQ, which employs token merging and Look-up Free Quantization (LFQ) for spatial and channel compression. Extensive experiments show that MergeVQ as an AR generative model achieves competitive performance in both image generation and representation learning with favorable efficiency. Our contributions can be summarized as:

- • We present a fresh learning paradigm that integrates token merging into a VQ-based AR generation framework, where high-level semantics are decoupled from patients in the first-stage training and can be restored with source matrix for details reconstruction, thus effectively reducing information loss while bridging the gap between representation learning and generation in a unified model.
- • We offer two schemes for MergeVQ’s second-stage generation. (i) We propose MergeAR, which performs KVCache compression for efficient raster-order prediction. (ii) With the source recovery module, existing randomorder generators can also be directly used for generation.
- • Experiments show MergeVQ’s competitive performance in both visual representation learning and image generation, with favorable token efficiency and inference speed.

##### 2. Related Work

###### 2.1. Auto-regressive Image Generation

Vector Quantization Tokenizer. Vector quantization, pioneered by VQ-VAE [60] and enhanced by VQ-GAN [21] through adversarial training and Transformer integration, faces three key challenges in traditional cluster-based approaches: (i) Gradient approximation: The straight-through estimator introduces imprecise encoder gradients, an issue mitigated through extended training in MAGVIT-v2 [72] and OpenMAGVIT2 [46]. (ii) Inefficient codebook usage: The commitment loss often leads to uneven gradient distributions and codebook collapse. Solutions include priors regularization in RegVQ [78] and Kepler Codebook [41], and EMA normalization in BEiT.v2 [53] and ViT-VQGAN [69]. (iii) Discrete representation bottleneck. VQ discards fine-grained details, hindering reconstruction fidelity. RQ [32] addresses this through hierarchical quantization to preserve information. Look-up Free Quantization performs channel-wise quantization, improving codebook usage while reducing overhead. Attempts such as FSQ [49], MAGVIT-v2 [72], OpenMAGVIT2 [46], and advanced variants [31, 65, 81] demonstrate results that exceed vanilla VQ. Another line reduces inference latency with AdaptiveLength Quantization to reduce the number of vision tokens by queries with cross-attention [20, 75], attention-based token pruning [29], or token grouping strategies [19].

Autoregressive Generation. VQGAN introduced AR visual generation by adopting the raster-order Next Token Prediction (NTP) in GPT [54, 55]. Subsequent works, including LlamaGen [57] and OpenMAGVIT2 [46], have extended this paradigm. In parallel, studies have focused on accelerating generation through non-autoregressive decoding, e.g., MaskGiT variants [4, 11] and MAR [37], which leverages masked prediction for parallel inference. Recent advancements explore random-order AR generation, where positions are predicted prior to token embeddings (RandAR [51]) or learnable positional encodings are utilized for prediction (RAR [74]).

###### 2.2. Unifying Representation and Generation

Since BEiT [3] first combined Masked Modeling with VQ for pre-training, research unifying representation and generation within a latent space has gained increasing interest [34]. These studies, typically conducted within clusterbased VQ frameworks, fall into two categories: (i) Using Pre-training Techniques in Quantized Space. MQ-VAE [29] quantizes semantic tokens by masking important ones for reconstruction. MAGE performs Masked Modeling directly in latent space during second-stage generation training, while BEiT abandons second-stage generation, using Masked Modeling as the second stage itself. (ii) Using representative tasks to enhance generation quality. DiGIT [85] extracts semantic tokens from pre-trained models for representation learning while using a finely crafted decoder for generation. VQ-KD [62] employs a pre-trained teacher

[Figure 1]

[Figure 2]

Class Token

###### Encoder

###### Global Alignment

|[C]|
|---|

|0|
|---|

|[C]| |
|---|---|
| | |

MLP

Teacher Label

[Figure 3]

| |
|---|

|1|
|---|

[Figure 4]

| |
|---|

|2|
|---|

Patchify

|[S]|
|---|

| |
|---|

| |
|---|

| |
|---|

… …

|1|
|---|

Attention

CNN

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 5]

+Token Merge Vector Quantization

|3|
|---|

Causal Decoder

| |
|---|

|7|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|6|
|---|

|8|
|---|

[Figure 6]

| |
|---|

[Figure 7]

[Figure 8]

[Figure 9]

| | |9| |
|---|---|---|---|
|𝐿 × 𝐷Po| | |s|

| |
|---|

| |
|---|

|9|
|---|

[Figure 10]

###### Token Merge

Token Quantization Image Generation

[Figure 11]

[Figure 12]

[Figure 13]

Pos Embed 𝑘 × 𝑑

Gen

[Figure 14]

||[C]|
|---|
<br><br>|i|
|---|
<br><br>|j|
|---|
<br><br>Class /Start<br><br>Token<br><br>Positional Encoding<br><br>Recovery Tokens<br><br>| |
|---|
<br><br>Continuous Tokens<br><br>| |
|---|
<br><br>Quantized<br><br>Tokens|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

###### Token Dequantization

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

| | | |
|---|---|---|
| |1| |

|1|
|---|

| |
|---|

[Figure 18]

| |
|---|

|1|
|---|

|2|
|---|

| |
|---|

|Attention|
|---|

[Figure 19]

| |
|---|

|3|
|---|

| |
|---|

|3|
|---|

Token … Unmerge

…

Cross- … Attention

…

…

CNN

| |
|---|

|7|
|---|

|6|
|---|

| |
|---|

[Figure 20]

Reconstruction

|8|
|---|

| |
|---|

| |
|---|

|6|
|---|

[Figure 21]

|9|
|---|

| | | |
|---|---|---|
|𝐿 × 𝐷| | |

| | | |
|---|---|---|
|𝐿 × 𝑑| | |

|9|
|---|

[Figure 22]

Decoder

Source Recovery

Rec Query

Figure 2. Overview of MergeVQ framework, which contains two stages and three groups of subtasks (Sec. 3.1). (a) As for representation learning (Sec. 3.2), K semantic tokens are extracted by the encoder with self-attention and token merging [7], which can be aligned globally with a pre-trained teacher while learning contextual information by predicting the source matrix. (b) As for reconstruction (Sec. 3.3), taking K merged and quantized tokens as the input, the positional information can be retained by the Source Recovery module, and then highquality details will be reconstructed. (c) As for generation (Sec. 4), we utilize the source matrix to construct a causal mask for training and leverage the KV cache to prune repeated tokens during inference for efficient generation.

model to guide token reconstruction. REPA [76] proposes that representation alignment can significantly improve the training efficiency and generation quality of diffusion models. Some approaches align visual and text codebooks via CLIP-inspired methods [79]. SPAE [71] utilizes hierarchical codebooks to align visual representations with frozen LLMs, while V2L Tokenizer [83] employs both global and local tokenizers for multi-modal alignment.

###### 2.3. Token Compression in Transformer

Token compression techniques have emerged as crucial components for improving efficiency in Transformer-based architectures, particularly in ViTs and LLMs. As for the Transformer encoder, ToMe variants [6, 8, 9, 12] employ lightweight bipartite soft matching (BSM) to achieve pruning-like efficiency gains, enhancing ViT throughput with minimal performance degradation. However, BSMbased methods often incur information loss among tokens due to their heuristic merging rules. Clustering-based token merging strategies, including k-means [48] and spectral clustering [5], have been explored to address this issue through more controllable operations. Yet, these techniques introduce computationally intensive iterative protocols in ViT layers. As for decoder architectures, recent advancements in KV cache compression (e.g., StreamLLM [66], FastGen [24], SnapKV [40], and H2O [80]) propose to optimize memory usage and inference speed via selective token retention and key-value pair compression. While these methods significantly enhance LLM inference efficiency, they are not directly applicable in the training phase.

##### 3. MergeVQ Learning Paradigm

###### 3.1. MergeVQ Framework

This section introduces MergeVQ, a VQ-based visual representation learning and auto-regressive image generation framework, and formalizes its core components.

Token Merge Encoding: Given an input image X ∈ RH×W×3, we employ a two-stage encoder Eϕ,θ(·) for feature extraction. First, a CNN encoder Eϕ(·) extracts feature map Z ∈ R

f ×Wf ×D, where f is the downsampling factor and d denotes the channel dimension. This feature is then flattened into a L-length token sequence ZL ∈ RL×D as:

H

ZL = Eϕ(X). (1)

In the second stage, we employ an attention-based encoder with token merging modules [7], denoted as Eθ(·), to further compress ZL into condensed K-length tokens ZK ∈ RK×D alongside a source matrix S ∈ RK×L that encodes spatial relationships between merged and original tokens:

S,ZK = Eθ(ZL). (2) The whole encoding process of MergeVQ is thus as:

S,ZK = Eϕ,θ(X). (3)

To ensure that ZK retains rich high-level semantics, we also impose global alignment constraints discussed in Sec. 3.2.

Quantization: We adopt LFQ to discretize the merged latent ZK. Concretely, the codebook C comprises binary vectors defined as: C = ×di=1{−1,1}, |C| = 2d, where

d is the quantized dimension. As such, each token zKi ∈ ZK is quantized element-wise: zKi = sign(zKi) = −1 · I(zKi < 0) + I(zKi > 0). Then, the index of quantized feature zmi is computed as a binary integer: Index(zKi) =

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

56

LinearProbeAccuracy(%)

ReconstructionFID(rFID)

54

MergeVQ (G) MergeVQ (G+R) MergeVQ (R)

52

50

N j=1 2k−1·I(zKij > 0), yielding quantized tokens Z˜K as:

MergeVQ (G) MergeVQ (G+R) MergeVQ (R)

48

46

###### Z˜K = Q(ZK,C), (4)

44

Token Recovery and Reconstruction: The key design lies in exploiting the spatial priors in source matrix S, which inherently encodes fine-grained positional dependencies between original L-length tokens and compressed ones during merging. We thus propose the recovery module Rω(·,·) to map quantized Z˜K back to Z˜L with the original length:

42

40

36 64 100 144 196 256

36 64 100 144 196 256

# Tokens

# Tokens

(a) IN-1K Reconstruction (b) IN-1K Linear Probe

- Figure 3. Analysis of kept tokens in reconstruction and representation learning. Three MergeVQ tokenizers are trained with 128 resolution for 30 epochs on ImageNet-1K. They keep 256, 144, and 36 tokens with ToMe [7] in the encoder during training. In inference, we evaluate rFID and linear probing top-1 accuracy with diverse merge ratios to show the trade-off between generation and representation. Please view Sec. 5 and Appendix B for details.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

K=64 K=144 K=256 Original

[Figure 27]

[Figure 28]

- Figure 4. Visualization of MergeVQ (G+R) reconstruction. With the kept tokens varying from 64 to 256, clustering maps of ToMe Attention indicate that MergeVQ can extract discriminative semantic tokens while recovering contextual positions and details.

Z˜L = Rω(Z˜K,S). (5)

This enables MergeVQ to retain both the coarse-grained semantics and fine-grained details, effectively balancing compression and reconstruction. The recovered Z˜L is then decoded into pixel space by Dψ(·) for reconstruction:

Xˆ = Dψ(Z˜L). (6)

By unifying the efficiency of ToMe with the spatial priors in S, MergeVQ aims to achieve loss-aware encoding: merged tokens are not merely reduced computational overhead but retained positional information for recoverable details.

###### 3.2. Harmonize Representation and Generation

to recover S from the quantized tokens Z˜K since S is unavailable during generation. In particular, the decoder

As aforementioned, we suppose that the overlooked explicit modeling of latent token-level context might serve as a critical gap for autoregressive generation, where next-token prediction relies on coherent spatial and semantic relationships that existing VQ techniques fail to capture. To address this, we introduce an additional Source Recovery task to the firststage learning, which trains the model to recover the token context encoded in source matrix S (illustrated in Figure 2).

with L learnable recovery queries Qr ∈ RL×d attends to Z˜K through cross-attention for semantics interaction. Subsequently, two self-attention layers further refine Qr into Q˜ ∈ RL×d, capturing latent token relationships. Since the source matrix records the positional relationships between K-centered Z˜K and original ZL, we use Z˜K as clustering centers to classify Q˜, which can be formulated as:

Attention with Token Merging: Building on ToMe [7], we iteratively merge tokens across N attention layers while maintaining a binary source S ∈ {0,1}K×L that records the ancestry of each merged token. Given the initial sequence

###### Sˆ⊤ = arg max softmax Q ˜Z˜K⊤ . (8)

We employ cross-entropy as our learning objective Lsrc to measure the difference between Sˆ and S, as:

ZL(0) = Eϕ(X), the l-th ToMeAttention merges tokens as: S(l),ZK(l) = ToMeAttention(l) ZL(l),S(l−1),r , (7)

Si,j log(Sˆi,j) + (1− Si,j)log(1− Sˆi,j). (9)

Lsrc = −

i,j

where S(0) = IL and l ∈ [1,N]. Note that ZL(l+1) = ZK(l) with l ≤ N −1. At each layer, the top 2r tokens by similar-

As such, this enforces the model to internalize how tokens were merged—a form of token-level context absent in existing VQ. During second-stage AR generation, when S is inaccessible, the trained decoder infers context directly from Z˜K, enabling accurate recovery for high-quality generation.

ity score are merged into r tokens, reducing sequence length to K = L−rN after N layers as Eq. (2). As such, S inherently preserves the positional information of merged tokens ZK during encoding, which enables subsequent recovery. Please view Appendix A.1 for implementation details.

Global Alignment: To further enhance token representations for discriminative tasks, we align the merged tokens ZK with global image semantics through the selfdistillation proposed by DINO [10]. We uniformly sample an image X from the training set, apply random augmenta-

Source Recovery Model: As mentioned above, canonical VQ methods discard the contextual interactions in latent space. MergeVQ addresses this via a lightweight transformer decoder as the Source Recovery Model that learns

tions to generate views u and v, and feed them into the DINOv2 encoder Eθ′(·) [50] and MergeVQ. The predicted category distributions from the [CLS] tokens, vt = Pθ[CLS′ ](v) and ut = Pθ[CLS](u), are aligned by minimizing the crossentropy between them, which can be formulated as:

L[CLS] = −Pθ[CLS′ ](v)⊤ log Pθ[CLS](u). (10)

This ensures ZK encodes semantically rich visual concepts while retaining compatibility with subsequent recovery.

###### 3.3. Token Recovery and Reconstruction

This section details how MergeVQ bridges token compression with high-fidelity reconstruction in first-stage training.

Token Recovery for Reconstruction: As stated in Sec. 3.1, we perform token recovery to restore finegrained positional information before reconstruction. This is achieved through the source matrix S as denoted in Eq. (5) Specifically, we utilize the positional information in S to expand ZK back into a sequence of length L. For example, if the i-th row of S satisfies S(i,j1) = 1 and S(i,j2) = 1, we recover the L-length Z˜L such that Z˜Lj

= Z˜Lj

= Z˜Ki, which can thus be implemented as:

1

2

L

K

. (11)

Z˜L = [˜zl]Ll=1 = S⊤Z˜K =

z˜Ki × sil

i=1

i=1

Subsequently, we apply the decoder Dψ to reconstruct the recovered Z˜L as Eq. (6). Note that we obtain the groundtruth source matrix during first-stage encoding, allowing straightforward token recovery. In the second phase, the predicted source matrix Sˆ could also be obtained from the pre-trained Source Recovery Model discussed in Sec. 2.2, which enables context-aware image token expansion.

Hybrid Model with Weight Initialization: Mainstream generative models typically rely on CNNs for feature extraction, while pure Transformer-based architectures are comparatively rare. However, in visual representation learning, Transformers are prevalent. MergeVQ combines these paradigms into a hybrid one: the CNN encoder Eϕ(·) extracts low-level features, providing inductive bias for pixelaligned reconstruction. Subsequent layers Eθ(·) employ Transformer with ToMeAttention for dynamic downsampling, balancing attention efficiency with representation capabilities. To further exploit these benefits, we integrate a pre-initialized Transformer into our architecture. The network details are illustrated in Figure 2 and Appendix A.1.

Adaptive Merge Ratios for Diverse Tasks: Unlike existing adaptive-length quantization strategies [39, 75], our MergeVQ utilizes variable merge ratios r during training instead of fixed sequence lengths. The ToMe module provides flexibility for different tasks through adjustable merge ratios. Experiments show that representation learning and reconstruction tasks benefit from diverse merge ratio settings. For instance, as shown in Figure 3, representation learning (Sec. 3.2) favors larger merge ratios [27, 30],

| |G+R (Stage-1)<br><br>G+R (Stage-2)<br><br><br>R (Stage-1)| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| |G (Stage-1)<br><br>G (Stage-2)<br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

0.6

0.7

0.5

0.6

0.5

0.4

Probability

0.4

0.3

0.3

0.2

0.2

0.1

0.1

0.0

0.0

36 49 64 81 100 121 144 169 196 225 256

2252562893243614004414845295766256767297848419009611024

(a)

(b)

Figure 5. Distribution of merge ratios sampling in training. (a) With 256 tokens in total, MergeVQ (R) and (G+R) sample the square number as kept token numbers in [36, 100] and [121, 225] with exponential and Gaussian distributions for stage-1 training, while the G+R version sampling from [144, 256] for stage-2 training. (b) With 1024 tokens in total, MergeVQ (G) samples the square kept number in [225, 400] and [256, 1024] with Gaussian and exponential distributions in both stage-1 and stage-2 training.

which might help capture the discriminative global patterns. Therefore, we present three variants: the Representation (R) version for enhanced generalization, the Generation and Representation (R+G) version bridging both objectives, and the Generation (G) one preserving spatial fidelity for highquality synthesis. More importantly, we propose a merge ratio sampling strategy in Figure 5 to expose the model to varying token counts, thus further enhancing the robustness and generalization capability of MergeVQ through the twostage training. In practice, we retained three versions of merged token counts: 256 for (G), 144 for (R+G), and 36 for (R), respectively. During training, we determine the corresponding ratio r by sampling the number of tokens retained, focusing on a range around the target token count for each version. We employ exponential distribution sampling for the (G) and (R) and discrete Gaussian distribution sampling for (G+R). Please refer to Appendix A for sampling details.

##### 4. MergeVQ for Efficient Generation

MergeVQ supports two different AR generation paradigms: (i) raster-order generation with our tailored MergeAR for KV cache compression and (ii) the random-order one that employs randomized AR generators like RandAR [51] enhanced by our Source Recovery Model (in Sec. 3.2).

###### 4.1. MergeAR with KV Cache Compression

MergeAR exploits the intrinsic redundancy with autoregressive token sequences by dynamically pruning duplicates to accelerate raster-order generation while preserving the spatial coherence with a position-recording system.

During training, we first sample a merge ratio r as in Appendix. A, which determines the number of merged visual tokens and results in K discretized tokens along with their ground-truth source matrix S. To regulate the level of sparsity, we introduce a Merge Instruction Token M, which serves as an indicator of merging extent. Using the source matrix S and target Z˜K, we construct a causal mask to guide the training process. Concretely, we derive a sparsity-

|𝑃1|𝑃1|𝑃3|𝑃3|𝑃5|𝑃5|𝑃5|𝑃8|𝑃8|
|---|---|---|---|---|---|---|---|---|

|𝑃1|𝑃1|𝑃3|𝑃3|𝑃5|𝑃5|𝑃5|𝑃8|?|
|---|---|---|---|---|---|---|---|---|

Position

Source Matrix

|[C]|
|---|

Class Token

Cache

|[C]|𝑀|
|---|---|

|[C]|𝑀|
|---|---|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | |C|heck|Dup|lica|tion|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | |?|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

|𝑀|
|---|

Merge Instruction

Start

Start

|𝑃|
|---|

Position Cache

| |
|---|

Incoming Token

Causal Mask

KVCache

| |
|---|

Recent Token

| |
|---|

Cached Token

Generating Token 9th

| |
|---|

Evicted Token

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|

Target

| |
|---|

Content Token

(a) MergeAR Training (b) MergeAR Inference

Figure 6. Illustration of MergeAR pipeline. (a) MergeAR training with the source matrix and K-sparse target sequences from the MergeVQ tokenizer to build a causal mask with duplicated tokens masked out, taking a class token and a merge instruction token as the starting conditions. (b) MergeAR inference that generates L tokens in the raster order with duplicated tokens detected and removed in the position and KV Caches.

inducing causal mask M ∈ {0,1}L×L denoted as:

i−1

M(i,j) = 1,when S(i,j) = 1 and 1 ∈/

S(k,j). (12)

k=1

This ensures each original token is represented by at most one merged token in the context. In the inference phase, we construct the KV cache similarly to the causal mask. As shown in Figure 6, when generating the t-th token, MergeAR compares it against existing tokens in the KV cache. If it is a duplicate, its position will be marked as a redundant one in the Position Cache and excluded from it when the slide window moves away. Otherwise, its content token and position will be added and kept forever.

###### 4.2. Randomized AR with Source Recovery

The concurrent Randomized AR techniques (e.g., RandAR [51]) generate tokens in arbitrary orders to improve parallelism and zero-shot generalization. Concretely, they introduce positional encoding prediction, whose objective pθ(x|P) could be formulated as:

N

pθ xπn(n) | P1π(1),xπ1(1),...,xπn(−n1−1),Pnπ(n) , (13)

n=1

where xπi (i) is the i-th token in this randomly shuffled Nlength sequence, and π(i) denotes its original position in

raster order. We then insert a positional instruction token Piπ(i) before each image token xπi (i). MergeVQ can also smoothly employ randomized generation with its Source Recovery Model (Sec. 3.2), where the K quantized tokens ZKq obtained in the first stage serve as target image tokens, and the predicted source Sˆ is used as the contextual information. After generating K generated tokens, we invoke the source recovery model Rω(·,·) and decoder, as in Eq.(6) and Eq.(5), to recover L-length tokens. Thus, when S is inaccessible in inference, MergeVQ is able to conduct context-aware token expansion for visual generation.

##### 5. Experiments

###### 5.1. Implementation Details

Visual Tokenizer Setup. We offer three MergeVQ versions for visual representation learning and generation: MergeVQ (G) for pure generation, MergeVQ (G+R) for both generation and representation, and MergeVQ (R) for representation learning only. As detailed in Appendix A.1, we present three architectures of these versions with the latent embedding dimension of 512, whose encoders have 63M, 62M, and 86M parameters. As discussed in Sec. 3.2, we apply the hybrid model that contains 4 and 5 hierarchical stages of ResNet blocks [26] with 12-layer of ToMe Attention blocks [7] at the last stage for the encoder networks in MergeVQ (G) and MergeVQ (R+G), as well as LFQ layer [73] with the dimension of 18. The corresponding decoder shares a similar architecture as encoders without ToMe modules. For fair comparisons, MergeVQ (R) adopts ViT-B [18] with random initialization as encoder but still adopts an identical decoder and LFQ as MergeVQ (G+R). As for the token number after quantization, the raw output numbers of the three versions are 1024, 256, and 256, and we merge them to 256, 144, and 36 tokens during training and inference. All versions are trained by AdamW optimizer [44] with (β1,β2) of (0.5,0.9), a default learning rate of 1e − 4, and a total batch size of 256 for 270∼300 epochs on ImageNet-1K without annotations. As for reconstruction, models are trained in 256 × 256 resolutions with a combination of ℓi reconstruction loss, GAN loss, perceptual loss, entropy penalty, commitment loss, and LeCAM regularization as MAGVITv2, combined with our proposed source recovery loss Lsrc and alignment loss L[CLS].

Visual Generator Setup. Following LlamaGen [57] and the concurrent work RandAR [51]1, we conduct three versions of AR generators with MergeVQ tokenizers: MergeVQ with vanilla LlamaGen for classical raster-order generation, MergeVQ with MergeAR (built upon LlamaGen) for efficient generation, and MergeVQ with RandAR for random-order generation. As for the third version, it requires the pre-trained Source Recovery module to predict the source matrix with the generated sequences as mentioned in Sec. 4.2, which can be a 2-layer standard Transformer decoder with 512 embedding dimensions at 7M parameters. We adopt LlamaGen-L as the generator architecture, which is a 24-layer Transformer decoder [55] in LLaMA-based architecture [59] and trained by AdamW optimizer [44] with a weight decay of 0.05, a basic learning rate of 4 × 104, and a batch size of 1024 for 300 epochs. View Appendix A.2 for more details.

###### 5.2. Self-supervised Pre-training

We evaluated self-supervised pre-trained models by linear probing (Lin.) [27] and end-to-end fine-tuning (FT) [3] protocols on ImageNet-1K. Table 1 shows that MergeVQ vari-

1More studies of MergeAR and the combination of MergeVQ with concurrent AR works [51, 74] will be updated in the arXiv preprint.

Table 1. Comparison of self-supervised pre-training on ImageNet-1K. The top-1 accuracy of linear probing (Lin.) and fully fine-tuning (FT) results are reported. ‡ denotes using the multi-crop augmentation or additional data. We summarize the target for alignment (Align.) and reconstruction (Rec.), the pre-training epochs, the encoder architecture type, and the number of learnable parameters (#Param) of the encoder and latent tokens (#Tokens), where MIM and TMM denote Masked Image Modeling and Token-merge Modeling.

|Support Tasks|Method Date<br><br>|Align. Rec. Epochs Target Target<br><br>|Encoder #Param #Tokens Type<br><br>|Accuracy↑ Lin. FT|
|---|---|---|---|---|
|Contrastive Pre-training|BYOL [25] NeurIPS’2020<br><br>MoCov3 [13] ICCV’2021 DINO‡ [10] ICCV’2021 DINOv2‡ [50] TMLR’2024<br><br>|MSE ✗ 800<br><br>InfoNCE ✗ 300 CE ✗ 300 CE ✗ 1000|R50-W2 94M 7×7 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196<br><br>|75.6 −<br>76.7 83.2 78.2 83.6 84.5 85.7<br>|
|MIM Pre-training|BEiT [3] ICLR’2022 iBOT‡ [82] ICLR’2022 MAE [27] CVPR’2022 SimMIM [67] CVPR’2022<br><br>CAE [14] IJCV’2023 PeCo [17] AAAI’2023 A2MIM [33] ICML’2023 I-JEPA [1] CVPR’2023 EVA-02 [22] CVPR’2024<br><br>|✗ DALLE 800 CE EMA 800<br><br>✗ RGB 1600<br>✗ RGB 800<br>✗ DALLE 1600<br>✗ VQVAE 800<br>✗ RGB 800<br>✗ RGB 600<br>✗ EVA-CLIP 300<br>|ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196 ViT-B 86M 196<br><br>|56.7 83.2 76.0 84.0 68.0 83.6<br><br>67.9 83.8 70.4 83.6<br><br>− 84.5<br><br>68.8 84.2 72.9 −<br><br><br>− 84.0|
|Generative<br><br>|ViT-VQGAN [69] ICLR’2022 MaskGIT [11] CVPR’2022 LlamaGen [57] NeurIPS’2024 Titok-B [75] NeurIPS’2024 REPA [76] ICLR’2025|✗ RGB 100<br><br>✗ RGB 200<br><br>✗ RGB 40<br><br>✗ VQGAN 200<br><br><br>DINOv2 Velocity 100<br><br>|VIM-Base 650M 1024 BERT 227M 256 CNN 72M 1024<br><br>Titok-B 86M 64 SiT-L/2 458M 1024|65.1 − 57.4 − 47.6 − 53.9 − 71.1 −<br><br>|
|Generative & Pre-training<br><br>|MAGE-C [36] CVPR’2023<br><br>DiGIT [85] NeurIPS’2024<br><br>MergeVQ (G+R) Ours MergeVQ (R) Ours<br><br>|InfoNCE VQGAN 1600<br><br>DINOv2 RGB 200 DINOv2 RGB+TMM 270 DINOv2 RGB+TMM 300<br><br>|ViT-B 24+86M 196<br><br>ViT 219M 256 Hybrid 63M 144 ViT-B 86M 36<br><br>|78.2 82.9<br><br>71.7 − 77.9 82.0<br><br>79.8 84.2<br><br><br>|

ants substantially outperform prior models like BYOL, MoCov3, and DINOv2 in performance and efficiency, notably with fewer tokens achieving superior accuracy. MergeVQ (R), which focuses on representation learning, achieves impressive results with only 36 tokens. With fewer tokens than DINOv2 (196), MergeVQ (R) achieves 79.8% Lin. accuracy and 84.2% FT accuracy, leveraging a flexible and discriminative latent space for both efficiency and performance. MergeVQ (G+R) performs slightly lower than MergeVQ (R) due to its inclusion of generation alongside representation learning, highlighting the trade-off between tasks, which require more tokens, and pretraining, which benefits from coarse-grained latent. Despite this, MergeVQ (G+R) remains competitive, reaching 77.9% of Lin. and 82.3% of FT, demonstrating competitive results while handling both generative and representation objectives.

###### 5.3. Image Generation

Reconstruction. Table 2 compares the reconstruction performance of VQ-based tokenizers on 256 × 256 ImageNet1K. MergeVQ (G+R) achieves an effective balance between reconstruction and token efficiency (nearly a 100%-utilized LFQ codebook with dynamic token lengths), leading to an rFID of 1.48. This outperforms methods that use larger codebooks and more tokens, such as RQ-VAE and LlamaGen. MergeVQ (G), applying the same codebook but with 256 tokens, hits an even lower rFID of 0.54, excelling in reconstruction quality. Overall, MergeVQ variants show high performance by optimizing codebook and token usage. While MergeVQ (G+R) slightly sacrifices rFID for han-

Table 2. Comparison of reconstruction on 256×256 ImageNet1K with reconstruction FID (rFID) of VQ tokenizers. We sum up the types, sizes, and dims of the codebook with its usage ratio. Ratio and #Tokens denote the downsampling rate and token number.

|Method|VQ Codebook<br><br>Type Size Dim Usage↑|Ratio #Tokens rFID<br><br>↓ ↓<br><br>|
|---|---|---|
|Taming-VQGAN [21] SD-VQGAN [56] RQ-VAE [32] MaskGIT [11] LlamaGen [57] TiTok-L-32 [75] TiTok-B-64 [75] VQGAN-LC [84] VQ-KD [62] MAGVIT-v2 [72] OpenMAGVIT2 [46] MaskBiT [65] MergeVQ (R) MergeVQ (G+R) MergeVQ (G+R)<br><br>|Cluster 210 256 49% Cluster 210 4 − Cluster 214 256 − Cluster 210 256 − Cluster 214 8 97% Cluster 212 16 − Cluster 212 12 −<br><br>CLIP 105 8 99% DINO 213 32 100% LFQ 218 1 100% LFQ 218 1 100% LFQ 214 1 100% LFQ 218 1 86% LFQ 218 1 99% LFQ 218 1 99%<br><br>|16 162 7.94 16 162 5.15 16 162 3.20 16 162 2.28 16 162 2.19 − 32 2.21 − 64 1.70<br><br>16 162 2.62<br><br>16 162 3.41<br><br><br>16 162 1.16<br><br>16 162 1.17<br><br><br>16 162 1.37 16 144 4.67 16 144 1.48 16 256 1.12<br><br>|
|ViT-VQGAN [69] OmiTokenizer [61] LlamaGen [57] TiTok-S-128 [75] VQGAN-LC [84]<br><br>MergeVQ (G) MergeVQ (G)<br><br>|Cluster 213 8 96%<br><br>Cluster 213 8 −<br><br>Cluster 214 8 97%<br><br><br>Cluster 212 16 −<br><br>CLIP 105 8 99% LFQ 218 1 100% LFQ 218 1 100%<br><br>|8 162 1.28 8 162 1.11<br><br>8 162 0.59 − 128 1.71<br>8 162 1.29 8 256 1.06 8 1024 0.54<br><br><br>|

dling both generation and representation, it remains competitive, highlighting the trade-off between these objectives.

Class Conditional Generation. As shown in Table 3, MergeVQ (G+R) and MergeVQ (G) stand out as compet-

Table 3. System comparsion of class-conditional generation on 256×256 ImageNet-1K. Generation Fr´echet inception distance (gFID) and inception score (IS) are reported with ADM [16]. “# P” means the parameter number, step means sampling steps, and ‡ denotes training tokenizers on OpenImages. Note that “-cfg” or “-re” denotes using classifier-free guidance or rejection sampling, and “-384” denotes for generating images at 384×384 resolutions and then resize back to 256 × 256 for evaluation.

|Type<br><br>|Tokenizer Generator # P. Step gFID↓ IS↑|
|---|---|
|Diff.|LDM-4 [56] 400M 250 3.60 247.7 UViT-L/2 [2] 287M 250 3.40 219.9 UViT-H/2 [2] 501M 250 2.29 263.9<br><br>VAE‡ DiT-XL/2 [52] 675M 250 2.27 278.2 MDTv2-XL/2 [23] 676M 250 1.58 314.7 SiT-XL [47] 675M 250 2.06 270.3 DiMR-XL/2R [42] 505M 250 1.70 289.0<br><br>|
|Mask.<br><br>|VQGAN MaskGIT [11] 177M 8 6.18 182.1 TiTok-B-64‡ MaskGIT-ViT [11] 177M 8 2.48 262.5 TiTok-S-128‡ MaskGIT-UViT-L [2] 287M 64 1.97 281.8<br><br>MAR MAR-B-cfg [37] 208M 100 2.31 281.7 MAR MAR-L-cfg [37] 479M 100 1.78 296.0|
|VAR<br><br>|VAR-d16 [58] 310M 10 3.30 274.4 VAR‡ VAR-d20 [58] 600M 10 2.57 302.6<br><br>VAR-d24 [58] 1.0B 10 2.09 312.9|
|AR<br><br>(raster)|VQGAN GPT2 [55] 1.4B 256 15.78 74.3<br><br>VQGAN GPT2-re [55] 1.4B 256 5.20 280.3 VIT-VQGAN VIM-L [69] 1.7B 1024 4.17 175.1 ViT-VQGAN VIM-L-re [69] 1.7B 1024 3.04 227.4<br><br>RQ-VAE RQ-Trans.-re [32] 3.8B 64 3.80 323.7<br><br>MAGVIT-v2 MAGVIT-cfg [70] 307M 256 1.78 319.4 LlamaGen LlamaGen-L [57] 343M 256 3.80 248.3 LlamaGen LlamaGen-L-384 [57] 343M 576 3.07 256.1 LlamaGen LlamaGen-XL [57] 775M 256 3.39 227.1 LlamaGen LlamaGen-XL-384 [57] 775M 576 2.62 244.1<br><br>OpenMAGVIT2 OpenMAGVIT2-B[46] 343M 256 3.08 258.3 OpenMAGVIT2 Open-MAGVIT2-L[46] 804M 256 2.51 271.7<br><br>MaskBit LlamaGen-cfg [57] 305M 256 1.52 328.6|
|AR & PT<br><br>|VQGAN MAGE-L [36] 230M 20 6.93 195.8 VQGAN DiGIT [85] 732M 256 3.39 206.0<br><br>MergeVQ (G+R) LlamaGen-L [57] 343M 256 3.28 251.6 MergeVQ (G+R) MergeAR (Ours) 343M 256 3.25 253.8<br><br>MergeVQ (G) MergeAR (Ours) 343M 1024 3.05 260.9<br><br>|
|AR<br><br>(random)|LlamaGen RandAR-L-cfg [51] 343M 88 2.55 288.8 LlamaGen RandAR-L-cfg [51] 775M 88 2.25 317.8<br><br>MergeVQ (G+R) RandAR-L-cfg [51] 343M 64 2.63 279.5 MergeVQ (G) RandAR-L-cfg [51] 343M 88 2.24 320.4<br><br>|

itive models. MergeVQ (G+R) uses 144 latent tokens and our MergeAR and achieves a gFID of 3.27 and an IS of 253.8 without CFG. When CFG and the concurrent RandAR generator are applied, it improves to a gFID of 2.63 and an IS of 279.5, surpassing most AR models. On the other hand, MergeVQ (G) with MergeAR, which uses 256 tokens and 1024 steps, demonstrates even better performance, with a gFID of 3.05 and an IS of 260.9 without CFG, and achieving a gFID of 2.24 and IS of 320.4 with CFG and RandAR. By leveraging fewer tokens than several resourceintensive models (e.g., VQGAN and ViT-VQGAN with large scales), MergeVQ variants excel in class-conditional image generation by balancing generation quality and efficiency, setting a new benchmark for models in this domain. This makes MergeVQ particularly promising for realworld applications where efficiency and generation quality

- Table 4. Ablation of three versions of MergeVQ tokenizers with the number of kept tokens during training for pre-training (linear probing Acc.) and reconstruction (rFID) tasks on ImageNet-1K.

|#Tokens|G rFID (↓)<br><br>|G+R rFID (↓) # Step (↓) Acc. (↑) FLOPs (↓)<br><br>|R Acc. (↑)|
|---|---|---|---|
|256 196 144 100 64 36|1.41 1.89 2.03 2.96 4.74 −<br><br>|2.15 64 48.6 76.2G<br><br>2.53 49 49.5 74.8G<br><br>3.07 36 51.0 73.4G<br><br>4.62 25 51.2 72.4G 6.51 16 51.8 71.5G 8.94 9 52.1 71.7G<br><br><br>|− 51.2 52.5 53.9 54.1 54.3<br><br>|

- Table 5. Ablation of main modules for MergeVQ generation with reconstruction (rFID) and generation (gFID) evaluation.

|Version|R G<br><br>|rFID gFID # Token|
|---|---|---|
|(G+R) (G+R) (G+R)+RandAR<br><br>|Ground-truth S ✗<br><br>2-layer Cross-Attention ✗ 2-layer Cross-Attention LlamaGen-L<br><br>|1.48 − 144 1.71 − 144 1.71 2.63 144<br><br>|
|(G+R)+LlamaGen (G)+LlamaGen (G)+MergeAR<br><br>|✗ LlamaGen-L<br>✗ LlamaGen-L<br><br>✗ LlamaGen-L<br><br><br>|− 3.28 256<br><br>− 3.14 1024 − 3.05 1024<br><br>|

are both crucial. Using fewer tokens while maintaining high image quality, MergeVQ variants achieve competitive results with a more streamlined and efficient approach compared to advanced diffusion and GAN-based models.

###### 5.4. Ablation Study

We conduct ablation studies on ImageNet-1K. As for tokenizers, Table 4 shows that MergeVQ (G) and MergeVQ (R) could achieve the best reconstruction and pre-training performance with 256 tokens (i.e., adaptive downsampling instead of convolution projection) and 36 tokens (i.e., a small number of semantic tokens for better global alignment). MergeVQ (G+R) could well balance the reconstruction performance with the pre-training and efficiency (fewer steps and FLOPs) by 144 tokens. As for generation, we validate these variants in Sec. 4. As shown in Table 5, Source Recovery is essential to restore positional information for MergeVQ (G+R) with RandAR, which could approximate the ground-truth S recover positions for AR generator. Table 3 and Table 5 show that KV Cache compression in MergeAR could be useful when the generated sequence is redundant, improving vanilla LlamaGen by 0.09 vs. 0.03 gFID with our MergeVQ (G) and MergeVQ (G+R).

##### 6. Conclusion

This paper presents MergeVQ, a unified framework that bridges competing objectives of visual representation learning and image generation. It incorporates flexible token merging-based designs to balance compact latent space and fine-grained generation. In addition, we propose MergeAR, a KVCache compressive technique that yields considerable speed gains while retaining superior second-stage image generation ability. Experiments show that MergeVQ achieves competitive performance in both pre-training and image generation, which highlights MergeVQ’s versatility to adapt to both generative and discriminative demands.

##### Acknowledgement

This work was supported in part by Chinese National Natural Science Foundation Projects U23B2054, 62276254, 62306313, the Beijing Science and Technology Plan Project Z231100005923033, Beijing Natural Science Foundation L221013, and the InnoHK program. This work was done when Juanxi Tian interned at Westlake University. We also thank OPPO AI Center and AI Station of Westlake University for the support of GPUs.

##### References

- [1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael G. Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629, 2023. 7
- [2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22669– 22679, 2023. 8
- [3] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pretraining of image transformers. In International Conference on Learning Representations (ICLR), 2022. 1, 2, 6, 7
- [4] Victor Besnier, Mickael Chen, David Hurych, Eduardo Valle, and Matthieu Cord. Halton scheduler for masked generative image transformer. In International Conference on Learning Representations (ICLR), 2025. 2
- [5] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral clustering with graph neural networks for graph pooling. In International Conference on Machine Learning (ICML), pages 874–883. PMLR, 2020. 3
- [6] Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW), pages 4599– 4603, 2023. 3
- [7] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In International Conference on Learning Representations (ICLR), 2023. 1, 2, 3, 4, 6
- [8] Maxim Bonnaerens and Joni Dambre. Learned thresholds token merging and pruning for vision transformers, 2023. 3
- [9] Qingqing Cao, Bhargavi Paranjape, and Hannaneh Hajishirzi. PuMer: Pruning and merging tokens for efficient vision language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12890–12903, Toronto, Canada, 2023. Association for Computational Linguistics. 3
- [10] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In International Conference on Computer Vision (ICCV), 2021. 2, 4, 7
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 7, 8

- [12] Mengzhao Chen, Wenqi Shao, Peng Xu, Mingbao Lin, Kaipeng Zhang, Fei Chao, Rongrong Ji, Yu Qiao, and Ping Luo. Diffrate : Differentiable compression rate for efficient vision transformers. In International Conference on Computer Vision (ICCV), 2023. 3
- [13] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In International Conference on Computer Vision (ICCV), pages 9640–9649, 2021. 7
- [14] Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for self-supervised representation learning. arXiv preprint arXiv:2202.03026, 2022. 7, 2
- [15] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In Conference on Neural Information Processing Systems (NeurIPS), 2021. 1
- [16] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. ArXiv, abs/2105.05233, 2021. 8
- [17] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. Peco: Perceptual codebook for bert pre-training of vision transformers. arXiv preprint arXiv:2111.12710,

2021. 7

- [18] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. 6, 1
- [19] Zhihao Duan, Ming Lu, Jack Ma, Yuning Huang, Zhan Ma, and Fengqing Zhu. Qarv: Quantization-aware resnet vae for lossy image compression. IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2023. 2
- [20] Shivam Duggal, Phillip Isola, Antonio Torralba, and William T Freeman. Adaptive length image tokenization via recurrent allocation. arXiv preprint arXiv:2411.02393, 2024. 2
- [21] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021. 1, 2, 7
- [22] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 7
- [23] Shanghua Gao, Pan Zhou, Mingg-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In International Conference on Computer Vision (ICCV), pages 23107–23116, 2023. 8
- [24] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms, 2024. 3
- [25] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. In Conference on Neural Information Processing Systems (NeurIPS), 2020. 7

- [26] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. 6, 1
- [27] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 5, 6, 7, 2
- [28] Jonathan Ho. Classifier-free diffusion guidance. ArXiv, abs/2207.12598, 2022. 2
- [29] Mengqi Huang, Zhendong Mao, Quan Wang, and Yongdong Zhang. Not all image regions matter: Masked vector quantization for autoregressive image generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2002–2011, 2023. 2
- [30] Ziyu Jiang, Yinpeng Chen, Mengchen Liu, Dongdong Chen, Xiyang Dai, Lu Yuan, Zicheng Liu, and Zhangyang Wang. Layer grafted pre-training: Bridging contrastive learning and masked image modeling for label-efficient representations. In International Conference on Learning Representations (ICLR), 2023. 5
- [31] Ahmed Khalil, Robert Piechocki, and Raul SantosRodriguez. Ll-vq-vae: Learnable lattice vectorquantization for efficient representations. arXiv preprint arXiv:2310.09382, 2023. 2
- [32] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11523– 11532, 2022. 2, 7, 8
- [33] Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Kai Wang, Lei Shang, Baigui Sun, Haoyang Li, and Stan.Z.Li. Architecture-agnostic masked image modeling - from vit back to cnn. In International Conference on Machine Learning (ICML), 2023. 7, 2
- [34] Siyuan Li, Luyuan Zhang, Zedong Wang, Di Wu, Lirong Wu, Zicheng Liu, Jun Xia, Cheng Tan, Yang Liu, Baigui Sun, et al. Masked modeling for self-supervised representation learning on vision and beyond. arXiv preprint arXiv:2401.00897, 2023. 2
- [35] Siyuan Li, Zedong Wang, Zicheng Liu, Cheng Tan, Haitao Lin, Di Wu, Zhiyuan Chen, Jiangbin Zheng, and Stan Z. Li. Moganet: Multi-order gated aggregation network. In International Conference on Learning Representations (ICLR),

2024. 1

- [36] Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 7, 8
- [37] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 8
- [38] Xiaotong Li, Yixiao Ge, Kun Yi, Zixuan Hu, Ying Shan, and Ling yu Duan. mc-beit: Multi-choice discretization for image bert pre-training. In European Conference on Computer Vision (ECCV), 2022. 1
- [39] Xiang Li, Hao Chen, Kai Qiu, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregres-

- sive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 5
- [40] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation, 2024. 3
- [41] Junrong Lian, Ziyue Dong, Pengxu Wei, Wei Ke, Chang Liu, Qixiang Ye, Xiangyang Ji, and Liang Lin. Kepler codebook. In International Conference on Machine Learning (ICML),

2024. 2

- [42] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. In Conference on Neural Information Processing Systems (NeurIPS),

2024. 8

- [43] Zicheng Liu, Siyuan Li, Di Wu, Zhiyuan Chen, Lirong Wu, Jianzhu Guo, and Stan Z. Li. Automix: Unveiling the power of mixup for stronger classifiers. In European Conference on Computer Vision (ECCV), 2022. 2
- [44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. 6
- [45] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26429–26445, 2024. 1
- [46] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 2, 7, 8, 1
- [47] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision (ECCV), 2024. 8
- [48] Dmitrii Marin, Jen-Hao Rick Chang, Anurag Ranjan, Anish Prabhu, Mohammad Rastegari, and Oncel Tuzel. Token pooling in vision transformers for image classification. In IEEE Winter Conference on Applications of Computer Vision (WACV), pages 12–21, 2023. 3
- [49] Fabian Mentzer, David C. Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. In International Conference on Learning Representations (ICLR), 2024. 1, 2
- [50] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Q. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Huijiao Xu, Herv´e J´egou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024. 5, 7, 1
- [51] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T. Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 5, 6, 8

- [52] William S. Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. 8
- [53] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. ArXiv, abs/2208.06366, 2022. 2
- [54] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training, 2018. 2
- [55] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. 2, 6, 8
- [56] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2022. 7, 8
- [57] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 1, 2, 6, 7, 8
- [58] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 8, 1
- [59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. ArXiv, 2023. 6
- [60] A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In ArXiv, 2017. 1, 2
- [61] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024. 7
- [62] Luting Wang, Yang Zhao, Zijian Zhang, Jiashi Feng, Si Liu, and Bingyi Kang. Image understanding makes for a good tokenizer for image generation. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 7
- [63] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvtv2: Improved baselines with pyramid vision transformer. Computational Visual Media (CVMJ), 2022. 1
- [64] Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, and Xihui Liu. Parallelized autoregressive visual generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [65] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. Transactions on Machine Learning Research (TMLR), 2024. 2, 7
- [66] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations (ICLR), 2024. 3

- [67] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 7, 2
- [68] Zhiqiu Xu, Yanjie Chen, Kirill Vishniakov, Yida Yin, Zhiqiang Shen, Trevor Darrell, Lingjie Liu, and Zhuang Liu. Initializing models with larger ones. ArXiv, 2023. 1
- [69] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 2, 7, 8
- [70] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10459–10469, 2023. 8
- [71] Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming Yang, Kevin P. Murphy, Alexander G. Hauptmann, and Lu Jiang. Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms. In ArXiv,

2023. 3

- [72] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. In International Conference on Learning Representations (ICLR),

2023. 1, 2, 7

- [73] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In International Conference on Learning Representations (ICLR), 2024. 1, 6, 2
- [74] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. ArXiv, abs/2411.00776, 2024. 2, 6
- [75] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 5, 7
- [76] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations (ICLR), 2025. 3, 7
- [77] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In International Conference on Computer Vision (ICCV), pages 6023–6032, 2019. 2
- [78] Jiahui Zhang, Fangneng Zhan, Christian Theobalt, and Shijian Lu. Regularized vector quantization for tokenized image synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18467–18476, 2023. 2

- [79] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling. arXiv preprint arXiv:2408.01181, 2024. 3
- [80] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, Zhangyang Wang, and Beidi

Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models, 2023. 3

- [81] Yue Zhao, Yuanjun Xiong, and Philipp Krahenbuhl. Image and video tokenization with binary spherical quantization. ArXiv, 2024. 2
- [82] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832,

2021. 1, 2, 7

- [83] Lei Zhu, Fangyun Wei, and Yanye Lu. Beyond text: Frozen large language models in visual signal comprehension. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27037–27047, 2024. 1, 3
- [84] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 7
- [85] Yongxin Zhu, Bocheng Li, Hang Zhang, Xin Li, Linli Xu, and Lidong Bing. Stabilize the latent space for image autoregressive modeling: A unified perspective. In Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 7, 8

## MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization

### Supplementary Material

##### A. Implementation Details

###### A.1. Stage 1: MergeVQ Tokenizer

Tokenizer Network. MergeVQ introduces hybrid encoders with self-attention blocks [18] using ToMe modules [7], built after the bottom of the pure CNN blocks (Residual modules with 3×3 convolutions [26]) proposed in MAGVITv2 [73]. We provide three versions of MergeVQ tokenizers, where the G and G+R versions use the hybrid encoders, while the R version uses the vanilla ViT-B [18]. The specific network configurations, experimental settings, and training details are thoroughly described in Table A1. The corresponding decoder shares a similar architecture as encoders except for using ToMe modules and replacing FFN with MixFFN [35, 63]. MergeVQ (G+R) and (G) versions initialize the parameters in the Transformer encoder with the DINOv2 [50] pre-trained model (i.e., DINOv2Base) by weight selection [68], while MergeVQ (R) adopts ViT-B [18] without pre-training as the encoder. Following the setup of the OpenMAGVIT2 [46] codebase, we also remove the gradient penalty loss and replace StyleGAN with PatchGAN as the discriminator (not employing DINO discriminator as VAR [58] in the current version). During training, we apply the reconstruction loss, the GAN loss, the perceptual loss, and the commitment loss, combined with the proposed source recovery loss Lsrc as Eq. (9) and the alignment loss L[CLS] as Eq. (10).

Source Recovery Model. The network details of the Source Recovery module in MergeVQ are shown in Table A2, where we utilize two Transformer decoder blocks to predict the source matrix Sˆ with quantized tokens Z˜K. As for implementation, we utilize the standard Transformer decoder to compute from the K quantized tokens (as KV embeddings) and L learnable recovery queries (as query position embeddings) similar to Maskformer [15]. As for MergeVQ with Randomized AR generators, we further fine-tuned this module with the learned generator after stage-2 training. Although the Source Recovery model was optimized in the stage-1 training (regarded as the contextual representation learning task), the additional fine-tuning could further enhance its robustness and generalization abilities for the generation task. As for MergeAR, it does not require the assistance of the Source Recovery module, which achieves speed-up by the proposed KV Cache compression.

Token Merge Module. Following the design principle of ToMe [7], The Token Merge Module reduces the number of tokens to improve efficiency while maintaining accuracy. Unlike token pruning, which drops tokens, ToMe combines

Table A1. Configuration of the network, weights of loss functions, and training settings for the three versions of MergeVQ tokenizers on ImageNet-1K. Note that the network designs are specified for the encoder, and the reported FLOPs are calculated for the encoder and decoder with ToMe [7] on 256 × 256 resolutions.

|Settings<br><br>|G G+R R|
|---|---|
|Base channels CNN Stage number Channel multiplier Residual Blocks Attention Blocks Downsampling ratio Vocabulary size Keep token number|64 64 768 4 5 − [1, 2, 4, 8] [1, 1, 2, 4, 8] [1] [4, 4, 4, 4] [4, 4, 4, 4, 4] −<br><br>[0, 0, 0, 12] [0, 0, 0, 0, 12] [12]<br><br>[1, 1/2, 1/4, 1/8] [1, 1/2, 1/4, 1/8, 1/16] [1/16]<br><br><br>218<br><br>256 144 36<br><br>|
|Discriminator loss Perceptual loss LeCam regularization L2 reconstruction Commitment loss LFQ Entropy loss Source recovery loss Alignement loss|0.8 0.7 0.01 1.0 0.25 0.1<br><br>0.5 0.5 1.0 0.1 1.0 1.0<br><br>|
|Optimizer (β1, β2) Weight decay Training epochs Base learning rate Batch size LR scheduler Gradient clipping EMA decay|AdamW (0.5, 0.9) 0.0<br><br>270 270 300<br><br>1e-4 256<br><br>Step Step Cosine − − 5.0<br><br>0.999<br><br>|
|#Param. of Encoder FLOPs of Encoder #Param. of Decoder FLOPs of Decoder|62.3M 62.7M 86.6M<br><br>97.5G 46.4G 9.5G<br><br>82.8M 83.4M 83.4M 169.2G 65.6G 65.6G<br><br>|

similar tokens into one representation, preserving more information and reducing accuracy loss, making it a practical, lightweight approach for both inference and training. Specifically, the token merging process consists of the following four steps:

- • Tokens are evenly divided into two groups, A and B, based on their odd or even positions.
- • Each token in A is paired with most similar token in B.
- • The r most similar pairs are selected for merging.
- • The features of tokens in these pairs are averaged to create a single representation.

Token similarity is determined using the keys (K) from the self-attention mechanism, with metrics like cosine similarity or dot product to measure similarity between tokens in A and B. Since merged tokens represent multiple originals, attention computation is affected. To address this, the softmax attention scores are adjusted by adding log s, where s

Table A2. Configuration of generators and Source Recovery model in MergeVQ or MergeAR for image generation on ImageNet-1K.

|Settings|LlamaGen-L RandAR-L Source Recovery<br><br>|
|---|---|
|Base channels Depth Attention heads FFN dimension Dropout Mask schedule Label smoothing # Parameter<br><br>|1024 1024 384 24 24 2 16 16 8 4096 4096 1536 0.1 0.1 0<br><br>Arccos Arccos −<br><br>0.1 0.1 − 343M 343M 7M|
|Optimizer (β1, β2) Weight decay Training epochs Base learning rate Batch size LR scheduler Gradient clipping|AdamW AdamW AdamW (0.9, 0.99) (0.9, 0.95) (0.9, 0.95) 5e-2 5e-2 1e-2<br><br>300 300 5 (optional)<br><br>4 × 10−4 4 × 10−4 1 × 10−4 1024 1024 256 Step Step Step 1.0 1.0 −<br><br>|

is the token size, ensuring merged tokens have the correct influence and maintain consistency in representation.

A = softmax

QK⊤ √

+ log s , (14)

d

where A denotes the attention weight matrix, Q denotes the query matrix, derived from the input tokens, K denotes the key matrix, also derived from the input tokens, log s denotes the size adjustment term, where s represents the length of the sequence, indicating the number of original patches it represents after merging. In practice, two types of merging schedules are provided: (1) Linearly Decreasing Schedule. The number of merged tokens linearly decreases as the layer depth increases. (2) Square Decreasing Schedule. The number of merged tokens decreases as the layer depth increases in the squared schedule. These strategies allow flexibility in balancing computational efficiency and model performance. We choose the square decreasing schedule.

###### A.2. Stage 2: MergeVQ Generation

We conducted raster-order and random-order autoregressive (AR) generation experiments based on LlamaGen [57] (modified by OpenMAGVIT2 [46]) and RandAR [51]. Using the LlaMA-based architecture, we adopted 2D RoPE, SwiGLU, and RMSNorm, which have been shown to be effective in previous works and thoroughly described in Table A2. The class embedding, indexed from a set of learnable embeddings, serves as the starting token. As for MergeAR, we also insert a Merge Instruction token, which is a learnable embedding token with a given merge number. For MergeVQ with RandAR [51], the classifier-free guidance (CFG) [28] with a linear sampling schedule is adopted as randomized AR variants [64, 74], where the optimal CFG weight is determined through a sweep with a step size of 0.1 across all methods.

###### A.3. Merge Ratio Sampling Strategy

Although our proposed MergeVQ framework can target certain tasks (representation learning or generation) by choosing a certain merge ratio, it can also benefit from a wide range of merge ratios, a kind of data augmentation that enhances the generation abilities with dynamic merge ratios. During training, we determine the corresponding merge ratio r by sampling the number of tokens retained, focusing on a range around the target token count for each version. For the versions with 256 and 36 semantic tokens, we use a discrete exponential distribution to sample the varying token counts as follows:

P(T = k) = (1 − exp(−λ))exp(−λk), (15)

where T represents the variation in the number of tokens with the index k ≥ 0. As for the G and R versions, the number of retained tokens is K = (16 − T)2 and (6 + T)2. As for the (R+G)-version in Figure 5, we use a discrete Gaussian distribution for sampling.

2

exp(−(k−µ)

2σ2 ) Z

, k ∈ Z, (16) where retained semantic tokens in the training are (12+T)2.

P(T = k) =

###### A.4. Evaluation of Representation Learning

As for the linear probing protocol, we follow MAE variants [14, 27] to evaluate the linear classification performance in the latent token space of trained models. Specifically, we train a parameter-free BN layer and a linear layer for 90 epochs using AdamW optimizer with a batch size of 1024, the Cosine annealing learning rate scheduler, where the initial learning rate is set to 1 × 10−3. As for the fine-tuning protocol, we follow SimMIM variants [33, 67] to fully finetune the pre-trained encoder for 100 epochs with AdamW optimizer and a batch size of 1024, which requires advanced augmentations and training strategies for modern architectures [43, 77]. The MergeVQ tokenizers use all tokens (i.e., not applying ToMe) for both the linear probing and full finetuning evaluations in Table 1, which could yield better performance with all vision tokens in the encoder. Meanwhile, the MergeVQ (R) tokenizer utilizes 144 tokens for reconstruction evaluation in Table 2. We found that it will degenerate rFID and cause more computational overhead when using all tokens because of the distribution gaps between 36-token pre-training and 256-token evaluation.

##### B. More Experiment Results

We evaluate the reconstruction of MergeVQ (G) and MergeVQ (G+R) tokenizers at different merging ratios A1. The specific results can be seen in the figure, where we compare our experimental results with those of MAGVIT2 [73]. We also visualize the generation results of MergeVQ variants in Figure A2, where the reconstruction quality progressively improves as the merge ratio decreases. The G+R version also achieves competitive results with 144 tokens.

#### MergeVQ

Merge ratio

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

###### InputMAGVIT2

G+RG

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

###### MAGVIT2MAGVIT2MAGVIT2InputInputInput

G+RGG+RGGG+R

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

- Figure A1. Visualization of tokenizer reconstruction on ImageNet-1K. We conducted reconstruction experiments with our G version using 1024, 576, 400, 256, and 144 tokens and with our G+R version using 256, 196, 144, 100, 64, and 36 tokens. The reconstruction results are shown in the figure. As the number of retained tokens increases, the reconstruction becomes more realistic.

###### MergeVQ(G) MergeVQ(G+R)

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

- Figure A2. Visualization of class conditional generation with MergeVQ variants on ImageNet-1K. The G version performs generation on 256 tokens, and the G+R version performs generation on 144 tokens.

