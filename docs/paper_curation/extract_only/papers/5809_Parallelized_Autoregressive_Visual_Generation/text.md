## Parallelized Autoregressive Visual Generation

Yuqing Wang1 Shuhuai Ren3 Zhijie Lin2† Yujin Han1 Haoyuan Guo2 Zhenheng Yang2 Difan Zou1 Jiashi Feng2 Xihui Liu1*

1University of Hong Kong 2ByteDance 3Peking University

# arXiv:2412.15119v3[cs.CV]3Apr2025

### Abstract

Autoregressive models have emerged as a powerful approach for visual generation but suffer from slow inference speed due to their sequential token-by-token prediction process. In this paper, we propose a simple yet effective approach for parallelized autoregressive visual generation that improves generation efficiency while preserving the advantages of autoregressive modeling. Our key insight is that parallel generation depends on visual token dependencies—tokens with weak dependencies can be generated in parallel, while strongly dependent adjacent tokens are difficult to generate together, as their independent sampling may lead to inconsistencies. Based on this observation, we develop a parallel generation strategy that generates distant tokens with weak dependencies in parallel while maintaining sequential generation for strongly dependent local tokens. Our approach can be seamlessly integrated into standard autoregressive models without modifying the architecture or tokenizer. Experiments on ImageNet and UCF-101 demonstrate that our method achieves a 3.6× speedup with comparable quality and up to 9.5× speedup with minimal quality degradation across both image and video generation tasks. We hope this work will inspire future research in efficient visual generation and unified autoregressive modeling. Project page: https: //yuqingwang1029.github.io/PAR-project.

### 1. Introduction

Autoregressive modeling has achieved remarkable success in language modeling [4, 38, 39, 55, 56], inspiring its application to visual generation [7, 8, 12, 25, 35, 36, 40, 41, 43, 51, 53, 58, 59, 69]. These models show great potential for visual tasks due to their strong scalability and unified modeling capabilities [14, 52, 65]. Current autoregressive visual generation approaches typically rely on a sequential tokenby-token generation paradigm: visual data is first encoded into token sequences using an autoencoder [12, 60, 71],

†Project lead. ∗Corresponding author.

(a) Non-local parallel generation of weakly dependent tokens（Ours)

|1|5a|6a|7a|2|5b|6b|7b|
|---|---|---|---|---|---|---|---|
|8a| | | |8b| | | |
| | | | | | | | |
| | | | | | | | |
|3|5c|6c|7b|4|5d|6d|7b|
|8c| | | |8d| | | |
| | | | | | | | |
| | | | | | | | |

[Figure 1]

[Figure 2]

(b) Local parallel generation of strongly dependent tokens

|1|2|3|4|5a|5b|5c|5d|
|---|---|---|---|---|---|---|---|
|6a|6b|6c|6d|7a|7b|7c|7d|
|8a|8b|8c|8d| | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 3]

[Figure 4]

Figure 1. Comparison of different parallel generation strategies. Both strategies generate initial tokens [1,2,3,4] sequentially then generate multiple tokens in parallel per step, following the order [5a-5d] to [6a-6d] to [7a-7d], etc. (a) Our approach generates weakly dependent tokens across non-local regions in parallel, preserving coherent patterns and local details. (b) The naive method generates strongly dependent tokens within local regions simultaneously, while independent sampling for strongly correlated tokens can cause inconsistent generation and disrupted patterns, such as distorted tiger faces and fragmented zebra stripes.

then an autoregressive transformer [61] is trained to predict these tokens following a raster scan order [8]. However, this strictly sequential generation process leads to a slow generation speed, severely limiting its practical applications [29, 65]. In this work, we aim to develop an efficient autoregressive visual generation approach that improves generation speed while maintaining the generation quality.

An intuitive way to improve generation efficiency is to predict multiple tokens in parallel at each step. In language modeling, methods like speculative decoding [6, 26, 28] and Jacobi decoding [22, 47] achieve parallel generation through auxiliary draft models or iterative refinement. In the visual domain, approaches like MaskGIT [5] employ non-autoregressive paradigms with masked modeling strategies, while VAR [53] achieves faster speed through next-scale prediction that requires specially designed multi-

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

LlamaGen: 576 steps, 12.41s per image

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

PAR-4x: 147 steps, 3.46s per image

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

PAR-16x: 51 steps, 1.31s per image

- Figure 2. Visualization comparison of our parallel generation and traditional autoregressive generation (LlamaGen [51]). Our approach (PAR) achieves 3.6-9.5× speedup over LlamaGen with comparable quality, reducing the generation time from 12.41s to 3.46s (PAR-4×) and 1.31s (PAR-16×) per image. Time measurements are conducted with a batch size of 1 on a single A100 GPU.

scale tokenizers and longer token sequences. However, the introduction of additional models and specialized architectures increases model complexity, and may limit the flexibility of autoregressive models as a unified solution across different modalities.

In this work, we ask: can we achieve parallel visual generation while maintaining the simplicity and flexibility of standard autoregressive models? We find that parallel generation is closely tied to token dependencies—tokens with strong dependencies need sequential generation, while weakly dependent tokens can be generated in parallel. In autoregressive models, each token is generated through sampling (e.g., top-k) to maintain diversity. Parallel generation requires independent sampling of multiple tokens simultaneously, but the joint distribution of highly dependent tokens cannot be factorized for independent sampling, leading to inconsistent predictions, as demonstrated by the distorted local patterns in Fig. 1 (b). For visual data, such dependencies are naturally correlated with spatial distances—while locally adjacent tokens exhibit strong dependencies, spatially distant tokens often have weak correlations. This motivates us to reconsider how to organize tokens for generation: by identifying spatially distant tokens with weak correlations, we can group them for simultaneous prediction. Such non-local grouping allows us to maintain sequential generation for strongly dependent local tokens while enabling parallel generation across differ-

ent spatial regions. Moreover, we observe that initial tokens in each local region play a crucial role in establishing the global structure - generating them in parallel could lead to conflicting structures across regions, such as repeated parts in different regions without global coordination(see middle row of Fig. 5). Therefore, the initial tokens in each local region should be generated sequentially to establish the global visual structure.

Based on these insights, we propose a simple yet effective approach for parallel generation in autoregressive visual models. Our key idea is to identify and group weakly dependent visual tokens for simultaneous prediction while maintaining sequential generation for strongly dependent ones. To achieve this, we first divide the image into local regions and generate their initial tokens sequentially to establish global context, then perform parallel generation by identifying and grouping tokens at corresponding positions across spatially distant regions. The process is illustrated in Fig. 1 (a). Our approach can be seamlessly implemented within standard autoregressive transformers through a reordering mechanism, with a few learnable token embeddings to facilitate the transition between sequential and parallel generation modes. By ensuring each prediction step has access to all previously generated tokens across regions, we maintain the autoregressive property and preserve global context modeling capabilities. With non-local parallel generation, our approach significantly reduces the number of

inference steps and thereby accelerates generation, while maintaining comparable visual quality through careful token dependency handling.

We verify the effectiveness of our approach on both image and video generation tasks using ImageNet [10] and UCF-101 [48] datasets. For image generation, our method achieves around 3.9× fewer generation steps and 3.6× actual inference-time speedup with comparable generation quality. With more aggressive parallelization, we achieve around 11.3× reduction in steps and 9.5× actual speedup with minimal quality drop (within 0.7 FID for image and 10 FVD for video). The qualitative comparison of generation results between our method and the baseline is shown in Fig. 2. The experiments demonstrate the effectiveness of

- our approach across different visual domains and its compatibility with various tokenizers like VQGAN [12] and MAGVIT-v2 [71].

In summary, we propose a simple yet effective parallelized autoregressive visual generation approach that carefully handles token dependencies. Our key idea is to identify and group weakly dependent tokens for simultane-

- ous prediction while maintaining sequential generation for strongly dependent ones. Our approach can be seamlessly integrated into standard autoregressive models without architectural modifications. Through extensive experiments with different visual domains and tokenization methods, we demonstrate considerable speedup while preserving generation quality, making autoregressive visual generation more practically usable for real-world applications.

### 2. Related Work

Autoregressive Visual Generation. Autoregressive modeling has been explored in visual generation for years, from early pixel-based approaches [43, 58, 59] to current tokenbased methods. Modern approaches typically follow a twostage paradigm: first compressing visual data into compact token sequences through discrete tokenizers [12, 60, 71], then training a transformer [61] to predict these tokens autoregressively in raster scan order [25, 40, 69]. This paradigm has been successfully extended to video generation [21, 66, 67], where tokens from different frames are predicted sequentially. However, the strictly sequential generation process leads to slow inference speed that scales with sequence length.

Parallel Prediction in Sequential Generation. Various approaches have been proposed to accelerate sequential generation. In language modeling, speculative decoding [6, 26, 28] employs a draft model to generate candidate tokens for main model verification, while Jacobi decoding [22, 47] enables parallel generation through iterative refinement. In visual generation, MaskGIT [5] adopts a non-autoregressive approach with BERT-like masked modeling strategies, taking a different modeling paradigm from

traditional autoregressive generation. VAR [53] proposes next-scale prediction that progressively generates tokens at increasing resolutions, though requiring specialized multilevel tokenizers and longer token sequences. In contrast, our approach enables efficient parallel generation while preserving the autoregressive property and model simplicity, readily applicable to various visual tasks without specialized architectures or additional models.

### 3. Method

In this section, we present our approach for parallelized visual autoregressive generation. We first discuss the relationship between token dependencies and parallel generation in Sec. 3.1. Based on these insights, we propose our parallel generation approach in Sec. 3.2. Finally, we present the model architecture and implementation details that realize this process within autoregressive transformers in Sec. 3.3.

#### 3.1. Token Dependencies and Parallel Generation

Standard autoregressive models adopt token-by-token sequential generation, which significantly limits generation efficiency. To improve efficiency, we explore the possibility of generating multiple tokens in parallel. However, a critical question arises: which tokens can be generated in parallel without compromising generation quality? In this section, we analyze the relationship between token dependencies and parallel generation through pilot studies, providing guidance for designing parallelized autoregressive visual generation models.

Pilot Study. In language modeling, researchers have attempted to group adjacent tokens for multi-token prediction [6, 22, 26, 28, 47, 49, 63]. However, our pilot study reveals that directly predicting adjacent tokens leads to significant quality degradation in visual generation (see Fig. 1(b) and Tab. 4 (d)). In autoregressive generation, each token is generated through sampling strategies (e.g., top-k) to maintain diversity. When generating multiple tokens in parallel, these tokens need to be sampled independently. However, for adjacent visual tokens with strong dependencies, their joint distribution cannot be factorized into independent distributions, as each token is heavily influenced by its neighbors. The impact of such independent sampling is clearly demonstrated in the figure, where generating adjacent tokens in parallel leads to inconsistent local structures like distorted tiger faces and fragmented zebra stripes, as tokens are sampled without considering their neighbors’ decisions. Design Principles. These observations suggest that parallel generation should focus on weakly correlated tokens to minimize the impact of independent sampling. For visual tokens, dependencies naturally decrease with spatial distance - tokens from distant regions typically have weaker correlations than adjacent ones. This motivates us to per-

|1| | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | |[Figure 23]| | |
| | | | | | |
| | | | | | |
| | | | | | |

|1| | |2| | |
|---|---|---|---|---|---|
| | | | | | |
| | | | |[Figure 24]| |
| | | | | | |
| | | | | | |
| | | | | | |

|1| | |2| | |
|---|---|---|---|---|---|
| | | | | | |
|[Figure 25]| | | | | |
|3| | | | | |
| | | | | | |
| | | | | | |

|1| | |2| | |
|---|---|---|---|---|---|
| | | | | | |
| | |[Figure 26]| | | |
|3| | |4| | |
| | | | | | |
| | | | | | |

|1|5a| |2|5b| |
|---|---|---|---|---|---|
| | | | | | |
|[Figure 27]| | | | | |
|3|5c| |4|5d| |
| | | | | | |
| | | | | | |

|1|5a|6a|2|5b|6b|
|---|---|---|---|---|---|
|7a|8a|9a|7b|8b|9b|
|10a|11a|12a|10b|11b|12b|
|3|5c|6c|[Figure 28]<br><br>4|5d|6d|
|7c|8c|9c|7d|8d|9d|
|10c|11c|12c|10d|11d|12d|

[Figure 29]

1 5a 2

6a 6b

5b

3 4

5c 5d

6c 6d

(1) Next token prediction for the initial tokens of different regions (2) Multi token prediction at aligned positions across regions

- Figure 3. Illustration of our non-local parallel generation process. Stage 1: sequential generation of initial tokens (1-4) for each region (separated by dotted lines) to establish global structure. Stage 2: parallel generation at aligned positions across different regions (e.g., 5a-5d), then moving to next aligned positions (6a-6d, 7a-7d, etc.) for parallel generation. Same numbers indicate tokens generated in the same step, and letter suffix (a,b,c,d) denotes different regions .

form parallel generation across distant regions rather than within local neighborhoods. However, we find that not all distant tokens can be generated in parallel. The initial tokens of each regions are particularly crucial as they jointly determine the global image structure. Parallel generation of these initial tokens, despite their spatial distances, could lead to conflicting global decisions, resulting in issues like repeated patterns or incoherent patches across regions (see the middle row in Fig. 5).

Based on these insights, we propose three key design principles for parallelized autoregressive generation: 1) generate initial tokens for each region sequentially to establish proper global structure; 2) maintain sequential generation within local regions where dependencies are strong; and 3) enable parallel generation across regions where dependencies are weak through proper token organization.

#### 3.2. Non-Local Parallel Generation

Based on the above principles, we propose our approach that enables parallel token prediction while maintaining autoregressive properties. The process is illustrated in Fig. 3.

Cross-region Token Grouping. Let {vi}Hi=1×W denote a sequence of visual tokens arranged in a H × W grid. We

first partition the token grid into M × M regions. Each region contains k := M H × WM tokens. We then group tokens at corresponding positions across different regions. Let vj(r) denote the token at position j in region r, where r ∈ {1,...,M2} and j ∈ {1,...,k}. We then organize these tokens into groups based on their corresponding positions across regions:

2)

2)

2) k ] .

[v1(1), · · · , v(M

1 ], [v2(1), · · · , v(M

2 ], · · · , [vk(1), · · · ., v(M

(1)

This organization groups together tokens at the same relative position across different regions, facilitating our parallel generation process.

Stage 1: Sequential Generation of Initial Tokens of Each Region. We first generate one initial token for each region sequentially (marked as “1-4” in Fig. 3) to establish the global context. As shown in Fig. 3 (1), we start with the top-left region and generate the initial token for each region

by sampling from the conditional probability distribution:

v1(i) ∼ P(v1(i)|v1(<i)), i ∈ {1,...,M2}, (2)

where v1(i) denotes the initial token of the i-th region. Since the number of regions (M2) is small and fixed, this sequential generation introduces minimal overhead while providing crucial global context for subsequent parallel generation.

Stage 2: Parallel Generation of Cross-region Tokens. After initializing tokens for all regions, we proceed with parallel generation of the remaining tokens. As illustrated in Fig.3 (2), at each step, we identify the next position j within each region following a raster scan order and simultaneously predict tokens at this position across all regions (e.g., tokens 5a-5d are generated in parallel). The parallel generation at each step can be formulated as:

2

2

{vj(r)}M

r=1 ∼ P({vj(r)}M

r=1|v<j), (3)

where {vj(r)}M

2

r=1 represents the set of tokens at position j across all regions to be generated in parallel, and v<j includes both initial tokens and tokens from previous parallel steps. For example, with M = 2 on a 24 × 24 token grid, after generating 4 initial tokens sequentially, we predict M2 = 4 tokens in parallel at each subsequent step, reducing the total number of generation steps from 576 to 147 (i.e., 4+5764−4). While enabling parallel prediction, our approach maintains the autoregressive property as each prediction is still conditioned on all previous tokens. The key difference is that tokens at corresponding positions across regions, which exhibit weak dependencies, are now generated simultaneously instead of sequentially.

#### 3.3. Model Architecture Details

We illustrate our parallel generation framework using a standard autoregressive transformer for class-conditioned image generation.

Framework Implementation. As shown in Fig. 4 (a), our model architecture consists of an autoregressive transformer that processes the input sequence and generates visual tokens. The input sequence begins with a class token

(a) Model Implementation

(b) Visible Tokens during Generation

Input Input

Parallel Predicted Tokens

C 1 2 3 4 M1 M2 M3 5a 5b 5c 5d

C 1 2 3 4 5a 5b 5c 5d 6a 6b 6c

Target Seq

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- 1
- 2
- 3
- 4
- 5a

- 1
- 2
- 3
- 4
- 5a

- 4 M1 M2 M3 5a 5b 5c 5d

|1|2|3|4|
|---|---|---|---|

- 5a 5b 5c 5d 6a 6b 6c 6d

- 6a 6b 6c 6d

- 7a 7b 7c 7d ...

Output

Output

Autoregressive Transformer

- 5b
- 5c
- 5d

- 5b
- 5c
- 5d

|C|1|2|3|
|---|---|---|---|

Input Seq ...

- 6a
- 6b
- 6c
- 6d

- 6a
- 6b
- 6c
- 6d

CLS Token Learnable Tokens for parallel decoding

Ours Traditional Single token prediction

- Figure 4. Overview of our parallel autoregressive generation framework. (a) Model implementation. The model first generates initial tokens sequentially [1,2,3,4], then uses learnable tokens [M1,M2,M3] to help transition into parallel prediction mode. (b) Comparison of visible context between our parallel prediction approach (left) and traditional single-token prediction (right). The colored cells indicate available context during generation. In traditional AR, when predicting token 6d, the model can access all previous tokens including 6a−6c. Without full attention, our parallel approach would limit each token (e.g., 6b) to only see tokens up to the same position in the previous group (e.g., up to 5b). We enable group-wise full attention to allow access to the entire previous group.

(C) followed by visual tokens to be generated. To achieve n-token parallel prediction, we design a special sequence structure with three distinct parts: 1) initial sequential tokens [1,2,...,n] that are generated one at a time, 2) a transition part with n − 1 learnable tokens [M1,M2,M3] that helps the model enter parallel prediction mode, and 3) subsequent token groups that are predicted n tokens at a time (e.g., [5a,5b,5c,5d], [6a,6b,6c,6d]). For predicting each group, the model takes all previous tokens as input while maintaining a fixed offset of n tokens between input and target sequences. The learnable tokens share the same dimension as regular tokens for seamless integration. To maintain spatial relationships under our reordered sequence, we employ 2D Rotary Position Embedding (RoPE) [50], which preserves each token’s original spatial position information regardless of its sequence position. The above designs enable parallel prediction while preserving the standard autoregressive transformer architecture.

Group-wise Bi-directional Attention with Global Autoregression. Our framework combines sequential generation of initial tokens with parallel generation of subsequent token groups. As illustrated in Fig.4 (b), in traditional autoregressive models, when predicting token 6d, the model can access all previous tokens including 6a − 6c. However, naive parallel generation with causal masking would restrict each token (e.g., 6b) to only see tokens up to the same position in the previous group (e.g., up to 5b), limiting the available context. To address this limitation while maintaining parallelism, we enable bi-directional attention within each prediction group while preserving causal attention between groups. This allows each token in the current group to access the entire previous group as context (e.g., all tokens [5a − 5d] are visible when predicting any token in [6a − 6d]). This design enriches the local context for parallel prediction while maintaining the global autoregressive property, ensuring compatibility with standard optimizations like KV-cache.

Model Params Layers Hidden Heads

PAR-L 343M 24 1024 16 PAR-XL 775M 36 1280 20 PAR-XXL 1.4B 48 1536 24 PAR-3B 3.1B 24 3200 32

Table 1. Model sizes and architecture configurations of PAR. The configurations are following previous works [34, 39, 51, 55].

Extension to Video Generation. Our parallel generation framework can be naturally extended to video generation. The tokenization process reduces both spatial and temporal dimensions, resulting in tokens arranged in a T × H × W grid, where each latent frame aggregates information from multiple input frames. We treat these temporally compressed tokens similarly to image tokens and apply our parallel generation strategy along the spatial dimensions, with the only modification being the use of 3D position embeddings. While we also explored parallel generation along the temporal dimension, we found it less effective than spatial parallelization. This is because temporal dependencies exhibit stronger sequential characteristics that are fundamental to video coherence, making them less suitable for parallel prediction compared to spatial relationships. The exploration of effective temporal parallel strategies remains as future work.

### 4. Experiments

#### 4.1. Experimental Setup

Image Generation. For fair comparison with existing token-by-token autoregressive visual generation methods, we adopt similar settings as [51], using a VQGAN tokenizer [12] with a 16,384 codebook size and 16× downsampling ratio. Models are trained on ImageNet-1K [10] for 300 epochs, with 384×384 images tokenized into 24×24 sequences. We evaluate on the ImageNet validation set at 256×256 resolution using FID [15] as the primary met-

Ours: sequential initial tokens + parallel distant tokens

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Naive parallel: all distant tokens in parallel without sequential initialization

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Local parallel: parallel prediction of adjacent tokens

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 5. Qualitative comparison of parallel generation strategies. Top: Our method with sequential initial tokens followed by parallel distant token prediction produces high-quality and coherent images. Middle: Direct parallel prediction without sequential initial tokens leads to inconsistent global structures. Bottom: Parallel prediction of adjacent tokens results in distorted local patterns and broken details.

Type Model #Para. FID↓ IS↑ Precision↑ Recall↑ Steps Time(s)↓

BigGAN [3] 112M 6.95 224.5 0.89 0.38 1 − GigaGAN [20] 569M 3.45 225.5 0.84 0.61 1 − StyleGan-XL [44] 166M 2.30 265.1 0.78 0.53 1 0.08

GAN

ADM [11] 554M 10.94 101.0 0.69 0.63 250 44.68

CDM [17] − 4.88 158.7 − − 8100 − LDM-4 [42] 400M 3.60 247.7 − − 250 − DiT-XL/2 [37] 675M 2.27 278.2 0.83 0.57 250 11.97

Diffusion

Mask MaskGIT [5] 227M 6.18 182.1 0.80 0.51 8 0.13 VAR VAR-d30 [53] 2B 1.97 334.7 0.81 0.61 10 0.27 MAR MAR [27] 943M 1.55 303.7 0.81 0.62 64 28.24

VQGAN [12] 227M 18.65 80.4 0.78 0.26 256 5.05 VQGAN [12] 1.4B 15.78 74.3 − − 256 5.05 VQGAN-re [12] 1.4B 5.20 280.3 − − 256 6.38

AR

ViT-VQGAN [68] 1.7B 4.17 175.1 − − 1024 >6.38 ViT-VQGAN-re [68] 1.7B 3.04 227.4 − − 1024 >6.38

RQTran. [25] 3.8B 7.55 134.0 − − 256 5.58 RQTran.-re [25] 3.8B 3.80 323.7 − − 256 5.58

LlamaGen-L [51] 343M 3.07 256.1 0.83 0.52 576 12.58 LlamaGen-XL [51] 775M 2.62 244.1 0.80 0.57 576 18.66 LlamaGen-XXL [51] 1.4B 2.34 253.9 0.80 0.59 576 24.91 LlamaGen-3B [51] 3.1B 2.18 263.3 0.81 0.58 576 12.41

AR

PAR-L-4× 343M 3.76 218.9 0.84 0.50 147 3.38 PAR-XL-4× 775M 2.61 259.2 0.82 0.56 147 4.94 PAR-XXL-4× 1.4B 2.35 263.2 0.82 0.57 147 6.84 PAR-3B-4× 3.1B 2.29 255.5 0.82 0.58 147 3.46

AR

PAR-XXL-16× 1.4B 3.02 270.6 0.81 0.56 51 2.28 PAR-3B-16× 3.1B 2.88 262.5 0.82 0.56 51 1.31

Table 2. Class-conditional image generation on ImageNet 256×256 benchmark. “↓” or “↑” indicate lower or higher values are better. “-re” means using rejection sampling. PAR-4× and PAR-16× means generating 4 and 16 tokens per step in parallel, respectively.

ric, complemented by IS and Precision/Recall [24]. We experiment with model sizes from 343M to 3.1B parameters (Tab.1), reporting both generation steps and latency time.

Video Generation. We evaluate on the UCF-101 [48] dataset using MAGVIT-v2 tokenizer [71] reproduced by [41]. Each 17-frame video (128×128 resolution) is compressed by 8× spatially and 4× temporally into a 5×16×16 token sequence (1280 tokens per video). For fair comparison, we implement both next-token prediction and our parallel generation approach using the same architecture. The position of video codes is encoded via 3D positional embeddings. Our reproduced MAGVIT-v2 tokenizer uses a 64K visual vocabulary instead of the original 262K to facilitate model training. We use Fr´echet Video Distance (FVD) [57] to evaluate generation quality.

Detailed training configurations for both video and image generation are provided in the supplementary material.

#### 4.2. Main Results

##### 4.2.1. Image Generation

Tab. 2 presents comprehensive comparisons of classconditional image generation with various state-of-the-art methods, including GAN [3, 20, 44](one-shot generation), Diffusion [11, 17, 37, 42] (iterative denoising), Mask [5] (mask token prediction), VAR [53] (next-scale prediction), MAR [27] (continuous mask token prediction), and AR [12, 51, 68] (autoregressive generation). Our PAR achieves competitive performance while maintaining faster inference speed than most state-of-the-art models. Specifically, when comparing with representative models from different categories, our method shows advantages. Compared with the mask-based method MaskGIT [5], our method achieves substantially better generation quality (FID 2.29 vs. 6.18) despite requiring more steps. For VAR [53], while it achieves slightly better FID (1.97 vs. 2.29), our method maintains a simpler framework with fewer tokens per image and preserves the pure autoregressive nature, making it more flexible for multi-modal integration.

Compared to our baseline model LlamaGen [51], PAR achieves 3.9× reduction in generation steps (147 vs. 576) and 3.58× speedup in wall-clock time (3.46s vs. 12.41s) while maintaining comparable quality (FID 2.29 vs. 2.18). With more aggressive parallelization, PAR-3B-16× further accelerates generation to 1.31s (9.5× speedup) with only 0.7 FID degradation compared to the baseline, demonstrating the effectiveness of our parallel generation strategy in balancing efficiency and quality.

##### 4.2.2. Video Generation

We evaluate our approach on the UCF-101 [48] dataset for class-conditional video generation. Table 3 shows comparisons with various state-of-the-art methods across different categories. Among recent works, MAGVIT-v2 [71]

Type Method #Param FVD↓ Steps Time(s)

VideoFusion [31] N/A 173 - Make-A-Video [45] N/A 81.3 - HPDM-L [46] 725M 66.3 - -

Diffusion

MAGVIT [70] 306M 76 - MAGVIT-v2 [71] 840M 58 - -

Mask.

CogVideo [18] 9.4B 626 - TATS [13] 321M 332 - OmniTokenizer [64] 650M 191 5120 336.70

AR

MAGVIT-v2-AR [71] 840M 109 1280 -

PAR-1× 792M 94.1 1280 43.30 PAR-4× 792M 99.5 323 11.27 PAR-16× 792M 103.4 95 3.44

AR

Table 3. Comparison of class-conditional video generation methods on UCF-101 benchmark. FVD measures generation quality, where lower values (↓) indicate better performance. PAR1× represents our token-by-token baseline, while PAR-4× and PAR-16× indicate our parallel generation variants with different speedup ratios, achieving competitive FVD scores with significantly reduced generation steps and wall-clock time.

achieves strong performance with an FVD of 58 using masked token prediction, while its autoregressive variant MAGVIT-v2-AR obtains an FVD of 109 with 1280 generation steps. Our next-token-prediction baseline (PAR-1×) achieves a competitive FVD of 94.1, demonstrating the effectiveness of our implementation. More importantly, our parallel generation variants significantly reduce both generation steps and wall-clock time while maintaining comparable quality. Specifically, PAR-4× reduces the generation steps from 1280 to 323 with minimal FVD increase (99.5 vs. 94.1), achieving 3.8× speedup (11.27s vs. 43.30s). Further parallelization with PAR-16× achieves 12.6× speedup (3.44s vs. 43.30s) with 103.4 FVD, while reducing generation steps to 95. Due to space limit, we provide visualization results of video generation in supplementary materials.

#### 4.3. Ablation Study

In this section, we conduct comprehensive ablation studies to investigate the effectiveness of our key design choices on the ImageNet 256×256 validation set (Tab. 4). Unless specified, we use the PAR-XL model with parallel group size n=4 as default setting.

Initial sequential token generation. We first evaluate the importance of initial sequential token generation by comparing models with and without this phase in Tab. 4 (a). Results show that initial sequential generation reduces FID from 3.67 to 2.61, with only 3 additional steps (147 vs. 144). We also visualize the comparison in Fig. 5. Without initial sequential generation (middle row), the generated images exhibit inconsistent global structures, such as misaligned dogs with duplicated body parts, as initial tokens are generated without awareness of each other. In contrast, our approach with initial sequential generation (top row) produces more coherent and natural-looking images. The

FID↓ IS↑ steps↓ w/o 3.67 221.36 144

w 2.61 259.17 147

- (a) Importance of initial sequential token generation. Sequential generation of initial tokens improves FID by 1.06 with negligible step increase.

n FID↓ IS↑ steps↓

1 2.34 253.90 576 4 2.35 263.24 147

16 3.02 270.57 51

- (b) Number of parallel predicted tokens (PAR-XXL). n=1 is the tokenby-token baseline. n=4 reduces steps by 4× with similar FID (2.35 vs. 2.34), while n=16 reduces steps by 11.3× at the cost of 0.67 FID.

attn FID↓ IS↑ steps↓ causal 3.64 228.08 147

full 2.61 259.17 147

- (c) Attention pattern between parallel tokens. Full attention allows complete context access from previous parallel groups (vs. causal attention’s limited access), bringing 1.03 FID improvement.

order pattern FID↓ IS↑ steps↓

raster one 2.62 244.08 576 distant one 2.64 262.72 576 raster multi 5.64 265.46 147 distant multi 2.61 259.17 147

- (d) Comparison of different scan orders under single-token and multitoken prediction. Our region-based distant ordering shows similar performance with raster scan in single-token setting, but significantly outperforms in multi-token prediction (2.61 vs. 5.64 FID).

Params FID↓ IS↑ steps

343M 3.76 218.92 147 775M 2.61 259.17 147

1.4B 2.35 263.24 147 3.1B 2.29 255.46 147

- (e) Scaling of model size (4× parallel). Generation quality steadily improves with more parameters, from 343M (FID 3.76) to 3.1B (FID 2.29).

Table 4. Ablation studies on image generation model designs.

results illustrate the importance of initial sequential token generation for establishing proper global structure.

Number of parallel predicted tokens. The number of tokens predicted in parallel (n) controls the trade-off between efficiency and quality. As shown in Tab. 4(b), with n = 4 (M = 2), our approach reduces generation steps from 576 to 147 while maintaining comparable quality (FID 2.35 vs. 2.34). Further increasing to n = 16 (M = 4) achieves more aggressive parallelization with only 51 steps, at the cost of slight quality degradation (FID increase of 0.67). This is consistent with our analysis that tokens from distant regions have weaker dependencies and can be generated in parallel. As shown in Fig. 2, both PAR-4× and PAR-16× preserve visual fidelity while achieving significant speedup (3.46s and

- 1.31s vs. 12.41s). Impact of attention pattern. To enable effective parallel prediction while preserving rich context modeling, we study different attention patterns between parallel predicted tokens. With n = 4 parallel tokens, enabling full attention within groups reduces FID from 3.64 to 2.61 compared to causal attention, as it allows each token to access complete context from previous groups. This supports our design of combining bi-directional attention within groups with autoregressive attention between groups. Impact of token ordering and prediction pattern. We compare raster scan and our distant ordering under different prediction settings. As shown in Tab. 4(d), while both achieve comparable quality in single-token prediction (FID
- 2.62 vs. 2.64), their performance differs significantly with multi-token prediction - raster scan degrades severely (FID 5.64) while our distant ordering maintains quality (FID

- 2.61). This indicates that the choice of parallel predicted tokens is critical. When using raster scan, adjacent tokens with strong dependencies are forced to generate simultaneously, leading to distorted local patterns as shown in Fig. 5 (bottom row). In contrast, our region-based distant ordering groups weakly correlated tokens for parallel prediction, preserving both local details and global coherence (top row). Model scaling analysis. In Tab. 4(e), we study how our parallel prediction approach scales with model size. With n = 4 parallel tokens, increasing model size from 343M to
- 3.1B parameters steadily improves generation quality (FID decreases from 3.76 to 2.29). Comparing with sequential generation baseline (LlamaGen) in Tab. 2, while smaller models show a noticeable quality gap (343M: FID 3.76 vs. 3.07), larger models achieve comparable performance (775M: 2.61 vs. 2.62; 1.4B: 2.35 vs. 2.34) while reducing generation steps from 576 to 147. This demonstrates that increased model capacity helps mitigate the quality tradeoff from parallel prediction, suggesting stronger capability in modeling joint distribution of parallel tokens.

### 5. Conclusion

We propose PAR, a noval autoregressive visual generation approach that enables efficient parallel generation while preserving the advantages of autoregressive modeling. Our key finding is that the feasibility of parallel generation depends on token dependencies - tokens with weak dependencies can be generated in parallel while strongly dependent tokens lead to inconsistent results. Based on this insight, our PAR organizes tokens based on their dependency strengths rather than spatial proximity. The effectiveness of our approach across different visual domains validates this strategy for efficient autoregressive visual generation. We hope our work can inspire future research on visual generation and other sequence prediction tasks.

### Acknowledgements

This work is partially supported by the National Nature Science Foundation of China (No. 62402406). The authors would like to thank Peize Sun, Yi Jiang, Xian Liu and Yao Teng for helpful discussions on autoregressive models.

### References

- [1] Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410, 2016. 5
- [2] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and R Devon Hjelm. Mine: mutual information neural estimation. arXiv preprint arXiv:1801.04062, 2018. 5
- [3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 6, 7
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Sastry, et al. Language models are fewshot learners. In NeurIPS, 2020. 1
- [5] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 1, 3, 6, 7
- [6] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, JeanBaptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023. 1, 3
- [7] Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, et al. Next token prediction towards multimodal intelligence: A comprehensive survey. arXiv preprint arXiv:2412.18619, 2024. 1
- [8] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, pages 1691–1703, 2020. 1
- [9] Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999. 5
- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 3, 5, 1, 2
- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, pages 8780–8794,

2021. 6, 7

- [12] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021. 1, 3, 5, 6, 7
- [13] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In ECCV, pages 102–118, 2022. 7
- [14] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 1

- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 30, 2017. 5
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 1
- [17] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research, 23(1):2249–2281, 2022. 6, 7
- [18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2022. 7
- [19] Edwin T Jaynes. Probability theory: The logic of science. Cambridge university press, 2003. 5
- [20] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, pages 10124– 10134, 2023. 6, 7
- [21] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [22] Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. arXiv preprint arXiv:2403.00835, 2024. 1, 3
- [23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 1
- [24] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. NeurIPS, 32, 2019. 7
- [25] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022. 1, 3, 6
- [26] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In ICML, 2023. 1, 3
- [27] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024. 6, 7
- [28] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024. 1, 3
- [29] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 1
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2018. 1

- [31] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023. 7
- [32] XuanLong Nguyen, Martin J Wainwright, and Michael I Jordan. Estimating divergence functionals and the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11):5847–5861, 2010. 5
- [33] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 5
- [34] OpenLM-Research. Openllama 3b. https : //huggingface.co/openlm- research/open_ llama_3b, 2023. 5
- [35] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T. Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. arXiv preprint arXiv:2412.01827, 2024. 1
- [36] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, pages 4055–4064, 2018. 1
- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 6, 7
- [38] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 1
- [39] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 1, 5
- [40] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821– 8831, 2021. 1, 3
- [41] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-autoregressive modeling. arXiv preprint arXiv:2502.07737, 2025. 1, 7
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 6, 7
- [43] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017. 1, 3
- [44] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022. 6, 7
- [45] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2022. 7
- [46] Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, and Sergey Tulyakov. Hierarchical patch diffusion models for high-resolution video generation. In CVPR, 2024. 7

- [47] Yang Song, Chenlin Meng, Renjie Liao, and Stefano Ermon. Accelerating feedforward computation via parallel nonlinear equation solving. In ICML, 2021. 1, 3
- [48] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 3, 7, 1,

- 2, 4

[49] Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

- 3

- [50] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5, 2

- [51] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 2, 5, 6, 7
- [52] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1
- [53] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 1, 3, 6, 7
- [54] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pages 1–5. IEEE, 2015. 5
- [55] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 5
- [56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [57] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [58] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. In NeurIPS, 2016. 1, 3
- [59] A¨aron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In ICML,

2016. 1, 3

- [60] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017. 1, 3
- [61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1, 3

- [62] Sergio Verd´u. α-mutual information. In 2015 Information Theory and Applications Workshop (ITA), pages 1–6. IEEE,

2015. 5

- [63] Chunqi Wang, Ji Zhang, and Haiqing Chen. Semiautoregressive neural machine translation. arXiv preprint arXiv:1808.08583, 2018. 3
- [64] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024. 7
- [65] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1
- [66] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 3
- [67] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [68] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 6, 7
- [69] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 1, 3
- [70] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In CVPR, 2023. 7
- [71] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR, 2024. 1, 3, 7

## Parallelized Autoregressive Visual Generation Supplementary Material

### Appendix

The supplementary material includes the following additional information:

- • Sec. A provides more implementation details for PAR.
- • Sec. B demonstrates the compatibility of our approach with typical LLM engineering optimizations.
- • Sec. C provides more visualization results, including zero-shot high-resolution generation and long-range dependency examples.
- • Sec. D provides the analysis of visual token dependencies.

### A. Implementation details for PAR

Image Generation. For image generation, we train our models on the ImageNet-1K [10] training set, consisting of 1,281,167 images across 1,000 object classes. Following the setting in LlamaGen [51], we pre-tokenize the entire training set using their VQGAN [12] tokenizer and enhance data diversity through ten-crop transformation. For inference, we adopt classifier-free guidance [16] to improve generation quality. The detailed training and sampling hyperparameters are listed in Tab. 5.

config value

training hyper-params optimizer AdamW [30] learning rate 1e-4(L,XL)/2e-

4(XXL,3B) weight decay 5e-2 optimizer momentum (0.9, 0.95) batch size 256(L,XL)/ 512(XXL,3B) learning rate schedule cosine decay ending learning rate 0 total epochs 300 warmup epochs 15 precision bfloat16 max grad norm 1.0 dropout rate 0.1 attn dropout rate 0.1 class label dropout rate 0.1

sampling hyper-params temperature 1.0 guidance scale 1.60 (L) / 1.50 (XL) /

1.435 (XXL) / 1.345 (3B)

Table 5. Detailed Hyper-parameters for Image Generation.

Video Generation. For video generation, we train our mod-

els on the UCF-101 [48] training set, which contains 9.5K training videos spanning 101 action categories. Videos are processed as 8fps random clips and tokenized by our reimplementation of MAGVIT-v2 [71] (as their code is not publicly available), achieving a reconstruction FVD score of 32 on UCF-101. For inference, we use classifier-free guidance [16] with top-k sampling to improve generation quality. The detailed training and sampling hyper-parameters are listed in Tab. 6.

config value

training hyper-params optimizer AdamW [30] learning rate 1e-4 weight decay 5e-2 optimizer momentum (0.9, 0.95) batch size 256 learning rate schedule cosine decay ending learning rate 0 total epochs 3000 warmup epochs 150 precision bfloat16 max grad norm 1.0 dropout rate 0.1 attn dropout rate 0.1 class label dropout rate 0.1

sampling hyper-params temperature 1.0 guidance scale 1.15 top-k 8000

Table 6. Detailed Hyper-parameters for Video Generation.

### B. Compatibility with Typical LLM Engineering Optimizations

We investigate whether our algorithmic parallel generation approach can complement typical engineering optimizations used in LLM inference. All experiments were conducted on a single NVIDIA A100 GPU with batch size 1, generating 384×384 resolution images. For simplicity, we only implemented PyTorch’s compile feature (leveraging CUDA graph optimization) in our PAR model. As a comparison point, we tested LlamaGen [51] with vLLM [23] optimizations, which includes both page attention and CUDA graph optimizations.

As shown in Tab. 7, our algorithmic improvements and engineering optimizations are orthogonal and provide com-

Model Resolution Optimization Latency LlamaGen-3B 384 none 12.41s LlamaGen-3B 384 vLLM 4.12s PAR-3B-4x 384 none 3.46s PAR-3B-4x 384 compile 1.15s PAR-3B-16x 384 compile 0.43s

Table 7. Compatibility with LLM engineering optimizations. Even with just PyTorch compile optimization, our PAR approach achieves substantial speedups compared to LlamaGen with more comprehensive vLLM optimizations.

[Figure 48]

[Figure 49]

Figure 6. Zero-shot generation at 512×512 resolution. Our model successfully generates coherent high-resolution images despite being trained at 384×384 resolution.

plementary benefits. Even without any engineering optimization, PAR-3B-4x (3.46s) outperforms LlamaGen-3B with comprehensive vLLM optimizations (4.12s). When implementing just the simple CUDA graph optimization through PyTorch compile, PAR-3B-4x achieves 1.15s latency, a 3.6× improvement over optimized LlamaGen. With more aggressive parallelization, PAR-3B-16x with compile further reduces latency to 0.43s, demonstrating our approach’s flexibility in speed-quality trade-offs. These results confirm that algorithm-level optimizations (reducing sequential steps) and engineering-level optimizations (improving computational efficiency) are orthogonal approaches that, when combined, maximize generation efficiency beyond what either can achieve alone.

### C. More Visualization Results

Zero-shot Generation on Higher Resolutions. Fig. 6 demonstrates our model’s capability for zero-shot generation at higher resolutions (512×512) using Rotary Position Embedding [50]. Despite being trained on 384×384 images, our approach effectively maintains coherent global structures and detailed local patterns in higher resolution generation. This shows the flexibility of our parallel generation framework and its compatibility with positional encoding methods that support resolution extrapolation.

Long-range Dependency Handling While our approach leverages the observation that spatially distant tokens typ-

[Figure 50]

[Figure 51]

Figure 7. Long-range dependency handling. Our method successfully maintains consistency between distant but strongly related elements (deer antlers, vehicle wheels), even when generating tokens from different spatial regions in parallel.

ically have weaker dependencies, certain visual elements exhibit strong long-range dependencies. Fig. 7 showcases our model’s ability to maintain consistency between distant but strongly dependent visual elements, such as symmetric features (deer antlers, vehicle wheels) and coherent structures across the image.

Addtional Image Generation Visualization. In Fig.8 and Fig.9, we provide additional visualization results of PAR-4× and PAR-16× image generation on ImageNet [10] dataset, respectively.

Addtional Video Generation Visualization. In Fig.10, we provide the visualization results of video generation using our model on the UCF-101[48] dataset. The results are sampled from 128×128 resolution videos with 17 frames. As shown in the figure, even with 16× parallelization (PAR16×), our method shows no obvious quality degradation compared to single-token prediction (PAR-1×), producing smooth motion and stable backgrounds across frames.

### D. Analysis of Visual Token Dependencies

In Sec.3.1, we demonstrated through pilot studies that parallel generation of adjacent tokens leads to quality degradation due to strong dependencies, while tokens from distant regions can be generated simultaneously. In this section, we provide a theoretical perspective of conditional entropy to explain this observation and our design. We use conditional entropy to measure the token dependencies quantitatively - lower conditional entropy between tokens indicates stronger dependency, while higher conditional entropy suggests weaker dependency and thus potential for parallel generation. We further validate our PAR design from the perspective of conditional entropy - In AR-based generation, each step predicts a conditional distribution of the next tokens given all previous tokens. Higher conditional entropy indicates higher difficulty for the model to predict the next tokens. In this section, we first introduce the estimation of conditional entropy in Sec.D.1, and then validate our proposed approach by analyzing the relationship between to-

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

###### Figure 8. Additional image generation results of PAR-4× across different ImageNet [10] categories.

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

###### Figure 9. Additional image generation results of PAR-16× across different ImageNet [10] categories.

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

PAR-1x

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

PAR-4x

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

PAR-16x

Figure 10. Video generation results on UCF-101 [48]. Each row shows sampled frames from a 17-frame sequence at 128×128 resolution, generated by PAR-1×, PAR-4×, and PAR-16× respectively across different action categories.

ken dependencies and spatial distances in Sec. D.2.

#### D.1. Conditional Entropy Estimation

Given a visual token sequence {v1,v2,...,vn}, our goal is to estimate the conditional entropy H(vk|{vj}j<k) where the token feature vi ∈ Rd and {vj}j<k is the set of (all) visual tokens that precede vk in the sequence. This conditional entropy measures the uncertainty of the current token vk given the previously occurring visual tokens, thereby characterizing the dependency between vk and the set {vj}j<k. It is important to emphasize that we do not require the exact value of H(vk|{vj}j<k). Instead, we aim to reflect the trends in H(vk|{vj}j<k) under different scenarios, such as given different sets of {vj}j<k and considering different positions of vk given the same set of {vj}j<k.

In particular, we characterize the relationship between the token vk and the previous ones as the following model

vk = f({vj}j<k) + ϵk (4)

where vk is the next token we focus on and {vj}j<k is the conditioning token(s), f(·) is a deterministic function, and ϵk is the random additive error term. Then the conditional entropy H(vk|{vj}j<k) satisfies

H(vk|{vj}j<k) = H(f({vj}j<k) + ϵk|{vj}j<k)

= H(ϵk|{vj}j<k), (5) where the second equation holds since f(·) is a deterministic function. However, exactly calculating H(ϵk|{vj}j<k) is intractable as we cannot access the entire data distribution. To this end, inspired by prior research on bounding techniques for entropy and mutual information estimation [1, 2, 32, 33, 54, 62], we seek their upper bound as a proxy for showing the trends of the conditional entropy for different tokens. In particular, we have

- 1

- 2

log((2πe)d|Σ|), (6)

H(ϵk|{vj}j<k) ≤ H(ϵk) ≤

where Σ denotes the covariance matrix of ϵk. Notably, the first inequality naturally holds and the second inequality follows from the maximum entropy theory [9, 19], which is achievable when ϵk follows a Gaussian distribution.

Based on Eq. 6, we can estimate the trend of conditional entropy changes by calculating the determinant of the residual covariance matrix, i.e., |Σ|. In order to obtain the additive errors ϵ, we consider training a parameterized model fθ(·) to get the function f and characterize ϵ as the residual errors. The detailed algorithm is provided in Algorithm 1.

#### D.2. Entropy Analysis on ImageNet Data and PAR

Based on the conditional entropy estimation method introduced above, we conduct experiments on ImageNet to analyze token dependencies and validate our parallel generation strategy. We randomly sample 10,000 images from ImageNet [10] and extract their features using VQGAN [12]

Figure 11. Visualization of token conditional entropy maps. Each map shows the conditional entropy of all tokens when conditioned on a reference token (blue square). Darker red indicates lower conditional entropy and thus stronger dependency with the reference token. The visualization shows that tokens exhibit strong dependencies with their spatial neighbors and weak dependencies with distant regions.

Algorithm 1 Conditional Entropy Estimation Input:

- 1: m: number of data points
- 2: {vi,1,vi,2,...,vi,n}mi=1: visual token sequences, where each vi,j ∈ Rd
- 3: k: index of the target token
- 4: fθ: parameterized model

Output: Estimated conditional entropy Hˆ(vk|{vj}j<k)

- 5: Initialize empty lists X and Y
- 6: for i = 1 to m do
- 7: Xi ← {vi,j}j<k
- 8: Yi ← vi,k
- 9: Append (Xi,Yi) to (X,Y)
- 10: end for
- 11: Train a model fθ to estimate Y given X using (X,Y)
- 12: Initialize empty list E for residuals
- 13: for (X,Y ) in (X,Y) do
- 14: Ypred ← fθ(X)
- 15: ϵk ← Y − Ypred
- 16: Append ϵk to Ek
- 17: end for
- 18: Compute residual covariance matrix Σˆ of Ek
- 19: Hˆ(vk|{vj}j<k) ← 12 log((2πe)d|Σˆ|)

- 20: return Hˆ(vk|{vj}j<k)

encoder, followed by vector quantization to obtain continuous features from the codebook.

We first analyze the dependencies between tokens at different positions. For each position j in the feature map, we calculate the conditional entropy H(vi|vj) where i ̸= j, given the token vj at the j-th position and considering all tokens vi at other positions. It should be noted that Algorithm 1 is not limited to H(vk|{vj}j<k) where the given visual tokens {vj} must satisfy j < k. This is because any given tokens vj and vi can be considered to satisfy Eq. 4, making the proposed method applicable in calculating H(vi|vj). Fig. 11 presents the experimental results. We observe that given different token positions vi, the adjacent tokens typically exhibit lower conditional entropy (shown in redder

multi-token order single-token order

difference of entropy

|1|5a|6a|7a|2|5b|6b|7b|
|---|---|---|---|---|---|---|---|
|8a| | | |8b| | | |
| | | | | | | | |
| | | | | | | | |
|3|5c|6c|7b|4|5d|6d|7b|
|8c| | | |8d| | | |
| | | | | | | | |
| | | | | | | | |

|1|5|9|13|2|6|10|14|
|---|---|---|---|---|---|---|---|
|17| | | |18| | | |
| | | | | | | | |
| | | | | | | | |
|3|7|11|15|4|8|12|16|
|19| | | |20| | | |
| | | | | | | | |
| | | | | | | | |

[Figure 166]

our order

（a) （b) （c)

|1|2|3|4|5a|5b|5c|5d|
|---|---|---|---|---|---|---|---|
|6a|6b|6c|6d|7a|7b|7c|7d|
|8a|8b|8c|8d| | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|1|2|3|4|5|6|7|8|
|---|---|---|---|---|---|---|---|
|9|10|11|12|13|14|15|16|
|17|18|19|20| | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 167]

raster scan

（d) （e) （f)

Figure 12. Conditional entropy differences between parallel and sequential generation in different orders. (a)(d) show parallel (4 tokens) generation strategies and (b)(e) show sequential generation strategies for our proposed order and raster scan order respectively. Numbers indicate generation step in each order. (c)(f) visualize the conditional entropy increase when switching from sequential to parallel generation for each order, where darker red indicates larger entropy increase and thus higher prediction difficulty. Both orders generate the first four tokens sequentially (shown as white regions in entropy maps). Our proposed order that generates tokens from different spatial blocks in parallel shows smaller entropy increases compared to raster scan order that generates consecutive tokens simultaneously, indicating parallel generation across spatial blocks introduces less prediction difficulty than generating adjacent tokens simultaneously.

colors). This indicates that the dependencies between adjacent tokens are stronger compared to the dependencies between tokens that are farther apart in position. This observation aligns with the spatial locality in visual data, where nearby regions have stronger correlations than distant ones.

Next, we analyze how different token ordering strategies affect the difficulty of parallel generation in Fig. 12. To simulate the prediction difficulty during generation, we compute each token's conditional entropy given all its previous tokens - higher conditional entropy indicates more uncertainty and thus higher prediction difficulty at that position. By comparing the conditional entropy difference between sequential (one token at a time) and parallel generation (predicting multiple tokens simultaneously), we can quantify the increased difficulty introduced by parallel generation at each position. We conduct experiments with 4-token parallel prediction under two ordering strategies: our proposed generation order that first generates the initial four tokens sequentially to establish global structure, then generates tokens from different spatial blocks in parallel, and the raster scan ordering that directly predicts consecutive tokens simultaneously after the initial four tokens.

For our proposed order, we aim to characterize the entropy increase caused by the parallel generation, when compared to the entirely sequential generation methods. In par-

ticular, let vk(r) be the token at position k in region r, we define Vk,rseq and Vk,rpar by the sets of the previous tokens of vk(r) for sequential and parallel generations (see Fig. 12(a)(b)). Then the conditional entropy of the sequential generation (single-token) and parallel generation (multi-token) are defined as H(vk(r)|Vk,rseq) and H(vk(r)|Vk,rpar). We characterize the entropy increase caused by the parallel generation, i.e.,

###### H(vk(r)|Vk,rpar) − H(vk(r)|Vk,rseq). (7)

As a comparison, we also consider the raster scan order, where the tokens are exactly arranged based on their positions, denoted as v1,v2,.... In this setting, given the current token vk, we define Vkseq and Vkpar by the sets of the previous tokens of vk for sequential and parallel generations (see Fig. 12(d)(e)). Then, we will also characterize the entropy increase caused by the parallel generation in the raster scan order, i.e.,

###### H(vk|Vkpar) − H(vk|Vkseq). (8)

The numerical results of (7) and (8) are presented in Fig. 12(c) and (f). It can be seen that both orderings maintain identical conditional entropy for the first four tokens due to the sequential generation. For subsequent tokens, our proposed order leads to significantly smaller conditional entropy increases compared to the raster scan order. This indicates that when switching from sequential to parallel generation, generating tokens from different spatial blocks introduces less prediction difficulty than generating consecutive tokens in raster scan order. The result quantitatively validates our design.

