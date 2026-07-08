arXiv:2506.03065v1[cs.CV]3Jun2025

# Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers

Pengtao Chen1 Xianfang Zeng2‡ Maosen Zhao1 Peng Ye3 Mingzhu Shen4 Wei Cheng2 Gang Yu2 Tao Chen1∗

1 Fudan University 2 StepFun 3 The Chinese University of Hong Kong 4 Imperial College London Code: https://github.com/Peyton-Chen/Sparse-vDiT

### Abstract

While Diffusion Transformers (DiTs) have achieved breakthroughs in video generation, this long sequence generation task remains constrained by the quadratic complexity of attention mechanisms, resulting in significant inference latency. Through detailed analysis of attention maps in Video Diffusion Transformer (vDiT), we identify three recurring sparsity patterns: diagonal, multi-diagonal, and vertical-stripe structures. And even 3-6% attention heads can be skipped. Crucially, these patterns exhibit strong layer-depth and head-position correlations but show limited dependence on the input content. Leveraging these findings, we propose Sparse-vDiT, a sparsity acceleration framework for vDiT comprising: 1) Pattern-optimized sparse kernels that replace dense attention with computationally efficient implementations for each identified sparsity pattern. 2) An offline sparse diffusion search algorithm that selects the optimal sparse computation strategy per layer and head via hardware-aware cost modeling. After determining the optimal configuration, we fuse heads within the same layer that share the same attention strategy, enhancing inference efficiency. Integrated into state-of-the-art vDiT models (CogVideoX1.5, HunyuanVideo, and Wan2.1), Sparse-vDiT achieves 2.09×, 2.38×, and 1.67× theoretical FLOP reduction, and actual inference speedups of 1.76×, 1.85×, and 1.58×, respectively, while maintaining high visual fidelity, with PSNR values reaching 24.13, 27.09, and 22.59. Our work demonstrates that latent structural sparsity in vDiTs can be systematically exploited for long video synthesis.

### 1 Introduction

In recent years, diffusion models have achieved significant advances in image generation [27], prompting growing interest in extending them to video synthesis. Early approaches, such as SVD [2] and Dynamicrafter [42], employed a 2D+1D framework that provided computational efficiency but lacked real-time interaction between spatial and temporal features, resulting in limited spatiotemporal consistency. Recent progress in 3D full-attention Video Diffusion Transformers (vDiT) [26] has effectively addressed these limitations. Built on this foundation, models such as OpenSora [18], CogVideoX [43], HunyuanVideo [16], and Wan2.1 [34] demonstrate strong spatiotemporal coherence and high video quality. These methods have been widely applied in fields including animation generation [11, 12], video editing [46, 35], and world modeling [25, 10].

Although 3D full-attention vDiT models demonstrate strong video generation performance and are widely adopted, they suffer from high computational costs and large inference latency. For instance, generating a 5-second 720p video at 24 fps using the HunyuanVideo model on a single NVIDIA A800 GPU takes approximately fifty minutes. This inefficiency primarily results from the joint spatiotemporal tokenization process, which generates up to 120k tokens in this setting. Given

∗Corresponding author. ‡Project leader. Work was done when interned at StepFun. Preprint. Under review.

[Figure 1]

Preproc

3D Full Attn

FFN

×𝑁

Postproc

- Figure 1: The architecture of vDiT and inference latency analysis of its two variants, CogVideoX1.5 and HunyuanVideo, across different components. The latency of the attention module dominates under long sequence settings, and its proportion increases as the sequence length grows.

that attention complexity scales quadratically with sequence length [33], this leads to a substantial computational burden. As shown in Figure 1, in the classical model CogVideoX1.5, which is based on the vDiT architecture, attention accounts for 77% of the total latency at 86k tokens. Specifically, for HunyuanVideo with 120k tokens, attention accounts for 81% of the total inference latency, and this proportion increases with longer sequence length. Thus, 3D full attention is the primary bottleneck in inference efficiency for vDiT-based video generation.

Fortunately, the 3D full attention mechanism exhibits significant redundancy despite its considerable computational cost. First, we observe that some attention heads in vDiT are redundant, as skipping their computations has minimal effect on the final output. Second, redundancy is also present in the computation of the attention map, namely in the QKT product. We find that vDiT attention maps commonly follow four distinct patterns: full attention, diagonal sparsity, multi-diagonal sparsity, and vertical-strip sparsity. The latter three patterns suggest that computing the full attention map is often unnecessary. Further experiments reveal that these sparse patterns remain stable across different input texts and are primarily determined by the position of attention within the vDiT architecture. This fixed redundancy provides a strong basis for optimization.

Building on these findings, we propose Sparse-vDiT, a sparse method designed to accelerate vDiT for video generation. To reduce redundancy among attention heads, we introduce a head skipping strategy. We observe that vDiT’s attention maps commonly follow three sparse patterns: diagonal, multi-diagonal, and vertical-stripes. To enable actual speedup, we design predefined kernels tailored to each pattern. Since these sparsity patterns are relatively fixed and input-invariant, we develop an offline sparse diffusion search algorithm that identifies the optimal attention pattern for each head using only a small number of samples. After the search, the computation pattern of each head is fixed. We then group heads with the same sparsity pattern within each layer and fuse them to further accelerate inference by leveraging their fixed structure. We conducted experiments on three widely used vDiT-based models: CogVideoX1.5, HunyuanVideo, and Wan2.1. On CogVideoX1.5, Sparse-vDiT achieved a 2.09× reduction in theoretical FLOPs and a 1.76× end-to-end speedup in real, while keeping the LPIPS score low at 0.14, and even outperforming the baseline in the ImageQual metric. On HunyuanVideo, Sparse-vDiT achieved a 2.38× reduction in theoretical FLOPs and a 1.85× speedup, with generation quality reaching SSIM = 0.87 and PSNR = 27.03. On Wan2.1, Sparse-vDiT achieved a 1.67× reduction in theoretical FLOPs and a 1.58× speedup, with generation quality reaching SSIM = 0.80 and PSNR = 22.59. These results indicate that Sparse-vDiT effectively balances computational efficiency and generation quality.

The contributions of our paper are as follows:

- • We find that attention heads in vDiT are partly redundant. Meanwhile, many heads often exhibit recurring sparse attention patterns, including diagonal sparsity, multi-diagonal sparsity, and verticalstripe sparsity. These patterns are consistent across different inputs but are closely related to the attention position within the vDiT architecture.
- • Building on these insights, we propose Sparse-vDiT, which accelerates vDiT by skipping redundant heads and applying pattern-aligned sparse attention kernels. It introduces an offline sparse diffusion search that selects the optimal sparse mode for each head using a small number of samples, followed by intra-layer fusion of heads with identical attention patterns to enhance inference efficiency.

- • Sparse-vDiT achieves 2.09×, 2.38× and 1.67× theoretical FLOPs reduction on CogVideoX1.5, HunyuanVideo and Wan2.1, respectively. It also delivers 1.76×, 1.85×, and 1.58× end-to-end video generation speedups while maintaining comparable generation quality, with PSNR scores of 24.13, 27.09, and 22.59. Sparse-vDiT consistently outperforms existing state-of-the-art (SOTA) methods, such as SVG and MInference.

### 2 Related Work

Efficient Diffusion Model. Diffusion models are inherently slow because of their iterative denoising process, leading to growing interest in accelerating inference. Existing approaches include pruning methods [8, 3] that reduce model parameters, quantization techniques [37, 28, 49] that decrease parameter bit-width and computational overhead, and caching strategies [24, 4, 29] that trade memory for computation speed. However, most of these methods are primarily designed for image generation, with relatively few acceleration methods specifically tailored for video diffusion models. For video diffusion, techniques like PAB [50], TeaCache [20], FasterCache [23], and AdaCache [15] reuse features by exploiting the similarity between adjacent denoising steps. Other methods reduce the number of timesteps using distillation [19, 45] or compress latent spaces using high-ratio VAEs [30]. In contrast, our approach accelerates inference by exploiting the sparsity in vDiT’s attention.

Efficient Attention Mechanism. The attention [33] is central to transformers but suffers from quadratic complexity in sequence length, limiting efficiency in long sequences. To address this, various sparse attention methods have been proposed. In traditional vision, Swin Transformer [22], NAT [9], and Sparse Transformers [5] restrict attention to local windows. Similarly, Longformer [1] applies windowed attention in NLP. Large language models [32] have identified attention sink phenomena [40, 39], introducing streaming attention that combines sink masking with windowing. Later works, such as MInference [14] and FlexPrefill [17], explore diverse static and dynamic sparse patterns. In diffusion models, DiTFastAttn [44, 47] noted strong local neighbor attention in DiTs, enabling acceleration via windowed attention and cached contexts. CLEAR [21], DiG [51], and SANA [41] further exploit the sparsity of the attention mechanism to achieve linearized computation. For video diffusion, Efficient-vDiT [6] observed that each frame in the attention primarily attends to a fixed set of other frames. This observation introduces tile-based attention to linearized computation. SVG [38] identified spatiotemporal sparsity in video attention and optimized attention computation through data reordering and an online scheme. However, this paper thoroughly reveals multiple patterns and invariances of redundancy in vDiT attention. Based on these findings, we propose an offline sparse acceleration framework that integrates head skipping with three attention sparsity patterns. Considering the fixed nature of offline optimization, fusion optimization is performed on a fixed attention pattern at each attention layer.

### 3 Preliminary

Full Attention. The multi-head attention mechanism [33] constitutes a fundamental building block in vDiT. Let the input hidden features be denoted as I ∈ RB×N×D, where B is the batch size, N the number of tokens, and D the original feature dimension. Through learnable linear projections, I is transformed into three tensors: query (Q), key (K) and value (V ). Each of these tensors has dimensions RB×H×N×d, where H denotes the number of attention heads, and d = D/H represents the reduced feature dimension per head. The attention outputs refined features O ∈ RB×N×D preserving the original dimension of I. The attention transformation process is defined as follows: for each head h ∈ {1,...,H},

√

d)Vh ∈ RB×N×d, (1)

Attention(Qh,Kh,Vh) = softmax(QhKhT/

where Qh,Kh,Vh are slice operations on the head dimension. Then, merging along the head dimension yields the final output O of the attention. For the full attention mechanism, the entire process described above is executed.

√

Sparse Attention. In Eq 1, softmax(QhKhT/

d) is known as the attention map, where each value represents how much one token attends to another at the corresponding position. Since its computational complexity is O(N2), generating the attention map takes up most of the computation in the attention mechanism. However, in practice, a token usually attends to only a small number of

Frame 1 Tokens

Frame 1

Frame 1 Frame 2 Frame N

###### T: V: I: …

| | | | |
|---|---|---|---|
| |Cro|ss| |
| |Fra|me| |
| | | | |

| |
|---|
| |
| |
| |
| |
| |

[Figure 2]

|T-T|T-V|
|---|---|
|V-T|V-V >99%|

Frame1Frame2FrameN

|I-I|I-I|I-I|I-I|
|---|---|---|---|
|I-I|I-I|I-I|I-I|
|I-I|I-I|I-I|I-I|
|I-I|I-I|I-I|I-I|

VAE&Patchify

Frame N Tokens same

| | | | | | |
|---|---|---|---|---|---|

Frame N

| | | | |
|---|---|---|---|
| |Se|lf| |
| |Fra|me| |
| | | | |

| |
|---|
| |
| |
| |
| |
| |

[Figure 3]

…

Attention Map (All Part) Attention Map (Video Region) Attention Map (Image Region)

- Figure 2: Visualization of the vDiT attention map showing four interaction regions. The dominant V-V region has diagonal blocks for self-frame and off-diagonal blocks for cross-frame interactions.

other tokens, rather than maintaining global attention. This results in most values in the attention map being close to zero, showing strong sparsity. In most cases, it is sufficient to compute only the dense regions of the attention map to obtain a sufficiently accurate result. If the sparsity pattern of the attention map is structured, computations involving sparse regions can be omitted at the hardware level using Triton [31] or CUDA, enabling practical acceleration.

- 4 Method Table 1: Quantitative impact of skipping different ratios of attention heads on the final generation.

#### 4.1 Attention Mechanism in vDiT

###### CogVideoX1.5 PSNR ↑ SSIM ↑ LPIPS ↓

In the following, we present the attention mechanism employed in vDiT. We first describe the distinctive layout of attention maps tailored for video generation. Next, we demonstrate that the attention mechanism exhibits substantial redundancy. Finally, we show that this redundancy is largely intrinsic to the model architecture and remains relatively insensitive to variations in the input.

skipping 1% 36.62 0.96 0.01 skipping 3% 33.31 0.95 0.02 skipping 6% 30.02 0.92 0.04

skipping 10% 26.87 0.85 0.09

###### HunyuanVideo PSNR ↑ SSIM ↑ LPIPS ↓

skipping 1% 31.84 0.95 0.02 skipping 3% 28.94 0.91 0.06 skipping 6% 24.21 0.81 0.12

skipping 10% 17.98 0.72 0.22

#### 4.1.1 Attention Map in vDiT.

Current mainstream vDiT models, such as CogVideoX and HunyuanVideo, mainly adopt the MMDiT paradigm [7]. In this design, the token sequence is formed by concatenating text tokens and video tokens, and the corresponding attention map is shown on the left side of Figure 2. The attention map is divided into four parts based on token type and position: T–T, T–V, V–T, and V–V, where T denotes text tokens and V denotes video tokens. Text tokens make up only a small portion of the sequence, while video tokens account for over 99%. In the V–V region (the middle part of Figure 2), video tokens are arranged in the temporal order of frames. As a result, the diagonal blocks correspond to self-frame interactions among image (frame) tokens. In contrast, the off-diagonal blocks correspond to cross-frame interactions, as illustrated on the right part of Figure 2.

#### 4.1.2 Analyzing Attention Redundancy in vDiT

We find that attention in vDiT contains considerable redundancy. Some attention heads are nonessential, and skipping them results in minimal performance loss. Moreover, the attention maps exhibit patterns of structured sparsity, which can be exploited to enable efficient sparse computation.

Head Skipping. Not all attention heads in vDiT contribute equally to performance. Based on a minimum mean squared error (MSE) criterion, we evaluate head skipping on CogVideoX1.5 and HunyuanVideo. As shown in Table 1, in CogVideoX1.5, skipping 6% of the attention heads preserves generation quality comparable to the original model. In HunyuanVideo, skipping 3% of the heads similarly causes little degradation in video quality. These results indicate that certain attention heads in vDiT are redundant, suggesting that head skipping may be a practical means to improve efficiency.

Layer 2, Head 6 Layer 5, Head 8 Layer 14, Head 21 Layer 28, Head 15

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Full Attention Pattern Diagonal Pattern Multi-Diagonal Pattern Vertical-Stripe Pattern

Figure 3: Visualization of the four recurring attention patterns in vDiT.

However, relying solely on skipping is insufficient to achieve high efficiency. As a coarse-grained method, it results in performance degradation beyond a certain threshold. As shown in Table 1, both models exhibit noticeable degradation when the skip ratio reaches 10%. Therefore, a more fine-grained strategy is required to achieve a greater speedup.

Given that the sparsity of attention maps can improve the efficiency of transformer models, we conduct an in-depth analysis of the attention map in vDiT. Taking CogVideoX as an example, we visualize its attention maps in Figure 3 and identify four recurring patterns:

Full Attention Pattern. The attention values are evenly distributed, indicating global interactions among tokens. Applying sparse computation to such dense patterns often degrades performance, making efficiency optimization difficult.

Diagonal Pattern. Large values appear along the main diagonal, representing interactions among neighboring tokens within the same frame (as shown in Figure 2). This pattern reflects the model’s ability to capture self-frame structure. Since most off-diagonal values are close to zero, the full attention can be well approximated by computing only the diagonal elements of the attention map. This structured sparsity allows for efficient acceleration using window attention [1].

Multi-Diagonal Pattern. Large values are distributed along multiple evenly spaced diagonals. These diagonals align with the diagonal blocks in the I-I region of Figure 2, indicating strong attention between tokens at nearby spatial positions across different frames. Therefore, this pattern is associated with vDiT’s ability to model cross-frame consistency. By rearranging tokens [38], this pattern can be transformed into a diagonal structure suitable for optimization with window attention.

Vertical-Stripe Pattern. In the attention map, large values form a vertical stripe pattern, suggesting the presence of global tokens that strongly attend to all others in vDiT. This structured sparsity also enables efficient computation by a sparse kernel.

- Figure 4: t-SNE visualization of attention patterns along the head dimension on a VBench subset, with different layers indicated by distinct colors. Patterns from different prompts exhibit clustering.

#### 4.1.3 Invariant Property of Attention Patterns

We revealed the presence of diverse attention patterns in vDiT above. We further observe that these patterns are strongly correlated with the depth of the attention layers, while being largely independent of the input text. To verify this, we randomly sampled 50 diverse prompts from VBench as a subset

Head Skipping

[Figure 12]

𝑴𝟎

[Figure 13]

[Figure 14]

[Figure 15]

Full Attention Sparsity: 𝑆 = 0

Q K V

0 =

O

𝑀 𝑀 𝑀 𝑀 𝑀

[Figure 16]

𝑴𝟏

Skip Head Sparsity: 𝑆 = 1

𝑂 𝑂 𝑂 𝑂

𝑂

[Figure 17]

Sparse Attention Kernel

SparseLayerN

Calculate MSE with Full Attention

[Figure 18]

LayerN

[Figure 19]

𝑴𝟐

[Figure 20]

[Figure 21]

Q K V

Kernel

- Sparse Attention 1 Sparsity: 𝑆

𝑴𝟑

- Sparse Attention 2 Sparsity: 𝑆

𝐽𝑢𝑑𝑔𝑒𝑟 = (𝑀𝑆𝐸 𝑶 − 𝑶 + 𝜆×(1 − 𝑆 ) > 𝜖)

O

   ,…, 

False

True

[Figure 22]

𝑀 {    𝑶  𝑶   ×(    )}

Head Fusion after Search

2 1 4

𝑀 𝑀 𝑀 𝑀 𝑀

Full Attention

5 9 𝐻

𝑴𝟒

[Figure 23]

…

Skip Head

Offline Sparse Diffusion Search

Sparse Attention 3 Sparsity: 𝑆

Diagonal Attention

- Figure 5: Overview of the Sparse-vDiT. We first predefine five types of attention mode M0:4. Then, using an offline sparse diffusion search algorithm, we select the best attention mode for each layer and head in vDiT. After the search, for heads set to skip attention, we set their outputs to zero. For the three sparse attention patterns, we create specialized sparse attention kernels to speed up computation. Finally, heads within the same layer that use the same attention mode are fused to improve efficiency.

and used them to generate videos. For each layer and each attention head in vDiT, we saved the corresponding attention maps. Since we only needed to determine the pattern types, we stored the maps as memory-efficient image files. We then used a ResNet50 to extract high-dimensional features from the images and applied t-SNE to project them into a 2D space along the head dimension. The results are shown in Figure 4, where different colors represent different layers. We observed that, regardless of the head, the attention patterns from different layers form distinct clusters, while those from different prompts tend to cluster together. This confirms that the attention patterns exhibit strong correlations with attention position in vDiT but are minimally affected by the input content.

#### 4.2 Sparse-vDiT: A Sparse Acceleration Framework for vDiT

In the previous part, we identified two types of redundancy in the attention mechanism of vDiT: redundancy within the attention heads and redundancy in the attention map computation. We also found that this redundancy is intrinsic to vDiT and only weakly dependent on the input text. Based on these findings, we introduce Sparse-vDiT, a sparse acceleration method designed for vDiT. This method determines the most effective sparse strategy for each head in each layer through offline search, resulting in acceleration. The overall structure of Sparse-vDiT is illustrated in Figure 5.

Sparse Computation Pre-definition. To reduce redundancy in the attention head, we apply the skip strategy M1, which bypasses the entire process in Eq 1. To maintain consistent output dimensions, the attention output is set to zero. The sparsity of the skip strategy is defined as S1 = 1, while the sparsity of full attention M0 is S0 = 0. Regarding the three sparse forms Mi(i = 2,3,4) shown in Figure 3, we have designed specific sparse kernels to reduce the computation of softmax(QKT/

√

d). The sparsity of these kernels is determined by the ratio of actual computation blocks to the total number of blocks, denoted as Si(i = 2,3,4), as shown in Figure 5. In the Sparse-vDiT framework, the sparsity of these kernels is predefined and treated as a fixed constant.

Offline Sparse Diffusion Search. In vDiT, different heads at various layers exhibit distinct attention patterns. Given the set M = {Mi(i = 0,...,4)} of attention computation modes, the challenge lies in selecting the most appropriate mode for each head. In Sparse-vDiT, we propose an offline sparse diffusion search method to address this. As shown in Figure 5, for each layer in every step of vDiT, we pass the inputs through M0 to M4, obtaining the corresponding hidden state results O0 to O4. We then compute the MSE distances between O1 to O4 and O0, that is, MSE(Oi − O0),i = 1,...,4, which represent the loss introduced by the sparse attention computation. Our final loss is

##### Li = MSE(Oi − O0) + λ × (1 − Si), (2)

where the sparsity penalty is added and λ balances quality and computational cost. If all losses in L = {Li(i = 1,...,4)} exceed the desired threshold ϵ, the head retains full attention. Otherwise, the sparse mode with the smallest loss replaces full attention. The specific formulation is as follows:

 

M0(Q,K,V ) , if

(Li > ϵ) Margmin

(3)

Attention(Q,K,V ,M) =

i=1,...,4



i{Li}(Q,K,V ) , otherwise

where ϵ controls the overall sparsity ratio during inference. As discussed in the previous part, vDiT’s sparse attention pattern is inherent after pretraining and largely independent of input types. Thus, the search in Sparse-vDiT is offline and requires only a small number of input samples. Once the search is completed, the sparse modes for the entire inference process are fixed. This fixity allows heads with the same sparse mode within a layer to be fused, further accelerating the inference.

### 5 Experiment

#### 5.1 Experimental Settings

Pretrained Model. To evaluate the effectiveness of Sparse-vDiT, we conducted text-to-video generation experiments using three leading open-source pretrained vDiT models: CogVideoX1.5 [43], HunyuanVideo [16], and Wan2.1 [34]. CogVideoX1.5 generates 81 frames at a resolution of 1360×768, while HunyuanVideo generates 129 frames at 1280×720. In the latent space encoded by the 3D-VAE, the vDiT in CogVideoX1.5 processes 45,106 tokens, including 226 text tokens and 11 video frames with 4,080 tokens each. The vDiT in HunyuanVideo processes 119,056 tokens, including 256 text tokens and 33 frames with 3,600 tokens each. And the vDiT in Wan2.1 processes 75,600 tokens, including 21 frames with 3,600 tokens each.

Dataset & Evaluation Metrics. We adopted a comprehensive evaluation framework covering both video generation quality and efficiency. For quality evaluation, we used three types of metrics. The first category measures reconstruction fidelity after inference acceleration, including Peak Signal-toNoise Ratio (PSNR) [50], Structural Similarity Index Measure (SSIM) [36], and Learned Perceptual Image Patch Similarity (LPIPS) [48]. The remaining two categories assess frame-level visual quality and temporal consistency, using the Imaging Quality (ImageQual) and Subject Consistency (SubConsist) metrics from the VBench [13]. For efficiency evaluation, we considered theoretical FLOPS, actual inference latency, and the speedup relative to the pretrain model. Regarding evaluation datasets, we followed the original protocol of CogVideoX [43], using prompts from the GPT-enhanced version of VBench. For HunyuanVideo, we used prompts from the Penguin Video Benchmark [16].

Baseline. We compared several existing acceleration methods for vDiT, including both classical approaches and state-of-the-art techniques. These methods include MInference [14], a classical sparse acceleration technique migrated from large language models. WinAttn [1], which applies sparse acceleration along both temporal and spatial dimensions of video. SVG [38], the current stateof-the-art method for sparse accelerating vDiTs, and PAB [50], a caching-based method designed specifically for video diffusion models.

Implementation Details. The baselines MInference, PAB, and SVG are implemented using their official code and configurations. Since PAB only provides code for CogVideo, we do not include it in the evaluation on HunyuanVideo. The window sizes for WinAttn-Spatial and WinAttn-Temporal follow the settings used in SVG. In SVG, full attention is applied during the first 10 steps, and we follow the same setup for all baselines. However, this constraint is not required for Sparse-vDiT on CogVideoX1.5. SVG also applies full attention to the first two layers of vDiT, and we adopt the same configuration for our baselines, although it is unnecessary for Sparse-vDiT. Both CogVideoX1.5 and HunyuanVideo inference results were obtained on a single NVIDIA A800 GPU, while Wan2.1 was obtained on a single NVIDIA H800 with a batch size of 1.

#### 5.2 Experimental Results Analysis

The qualitative and quantitative results are shown in Figure 6 and Table 2, respectively. Both consistently demonstrate that Sparse-vDiT effectively accelerates video diffusion models without compromising the quality of generation. This can be explained as follows.

Table 2: Comparison of video generation quality and efficiency between Sparse-vDiT and the baseline. XBench refers to VBench for CogVideoX1.5 and Wan2.1 evaluation and Penguin Video Bench for HunyuanVideo. CogVideoX1.5 & HunyuanVideo on single A800, Wan2.1 on H800, batch size 1.

Against Original XBench Score

Method

SSIM (↑) PSNR (↑) LPIPS (↓) ImageQual (↑) SubConsist (↑) PFLOPS (↓) Latency (↓) Speedup (↑) CogVideoX1.5 [43] - - - 63.28% 92.96% 147.87 901s 1.00×

- MInference [14] 0.61 14.63 0.37 56.04% 87.12% 84.89 634s 1.42× WinAttn (Spatial) [1] 0.64 19.07 0.32 64.84% 90.92% 72.34 531s 1.69× WinAttn (Temporal) [1] 0.69 19.64 0.28 63.69% 92.66% 72.34 537s 1.67× PAB [50] 0.72 20.93 0.23 59.03% 92.38% 105.88 630s 1.43× SVG [38] 0.75 21.92 0.22 63.11% 92.49% 74.57 550s 1.64× Sparse-vDiT (Ours) 0.82 24.13 0.14 63.45% 92.66% 70.69 511s 1.76× HunyuanVideo [16] - - - 67.28% 96.79% 612.37 3166s 1.00× MInference [14] 0.64 19.23 0.43 60.53% 88.96% 293.87 2042s 1.55× WinAttn (Spatial) [1] 0.56 17.81 0.56 63.55% 90.26% 258.84 1755s 1.80× WinAttn (Temporal) [1] 0.80 23.76 0.22 67.32% 96.38% 258.84 1764s 1.79× SVG [38] 0.86 26.83 0.14 67.06% 96.54% 259.79 1802s 1.75× Sparse-vDiT (Ours) 0.87 27.09 0.12 67.13% 96.69% 257.09 1715s 1.85× Wan2.1 [34] - - - 67.61% 91.95% 660.49 1935s 1.00×

- MInference [14] 0.62 15.49 0.36 63.29% 89.32% 469.79 1453s 1.33× WinAttn (Spatial) [1] 0.68 19.14 0.25 67.27% 91.34% 401.21 1265s 1.53× WinAttn (Temporal) [1] 0.73 20.29 0.21 67.40% 91.47% 401.21 1280s 1.51× SVG [38] 0.78 21.96 0.18 67.18% 91.27% 403.50 1298s 1.49× Sparse-vDiT (Ours) 0.80 22.59 0.16 67.35% 91.39% 397.39 1228s 1.58×

Reconstruction Fidelity. On both CogVideoX1.5 and HunyuanVideo, Sparse-vDiT achieves the best performance across all fidelity metrics. For CogVideoX1.5, Sparse-vDiT yields an SSIM of

- 0.82, significantly higher than the closest baseline, SVG (0.75), and substantially higher than earlier sparse methods, such as MInference (0.61) and PAB (0.72). Similarly, the PSNR for Sparse-vDiT is 24.13 dB, surpassing all baselines, with the suboptimal result from SVG at 21.92 dB. Most notably, Sparse-vDiT achieves a substantially lower LPIPS score (0.14), indicating greater perceptual similarity to the original outputs. The trends hold consistently on HunyuanVideo, where Sparse-vDiT again records the highest SSIM (0.87) and PSNR (27.09), along with the lowest LPIPS (0.12). The margins are particularly significant compared to early techniques such as WinAttn (Temporal), which, while effective (SSIM: 0.76, LPIPS: 0.22), still underperforms relative to Sparse-vDiT. These results confirm the strong preservation of spatial and perceptual detail after applying our acceleration scheme.

Visual Quality. The ImageQual score from the VBench benchmark quantifies the frame-level visual quality as judged by pretrained evaluation models. Sparse-vDiT performs on par with or better than most baselines, achieving 63.45% on CogVideoX1.5 and 67.13% on HunyuanVideo. Although WinAttn (Spatial) slightly surpasses Sparse-vDiT in ImageQual on CogVideoX1.5 (64.84%), it comes with lower fidelity scores and higher LPIPS, suggesting a potential overfitting to local texture patterns at the cost of content preservation. On HunyuanVideo, Sparse-vDiT delivers ImageQual scores highly comparable to the best-performing methods, including SVG (67.06%) and WinAttn (Temporal) (67.32%). These results indicate that Sparse-vDiT maintains competitive frame-level realism while significantly outperforming others in reconstructive metrics, highlighting its balanced and robust generation performance.

Temporal Consistency. Temporal coherence is critical in video generation, and the SubConsist metric evaluates the consistency of subjects and motion across frames. Sparse-vDiT delivers stateof-the-art temporal stability in both benchmarks. On CogVideoX1.5, its SubConsist score reaches 92.66%, on par with the strongest existing methods, including WinAttn (Temporal) and PAB. On HunyuanVideo, Sparse-vDiT attains 96.69%, closely matching the best score of 96.79% from the original unaccelerated model. This observation is particularly important because many acceleration methods compromise temporal stability in favor of spatial quality. The ability of Sparse-vDiT to achieve high consistency while also delivering best-in-class fidelity underscores the effectiveness of its sparse acceleration strategy. By preserving computation in more temporally sensitive heads, Sparse-vDiT minimizes temporal artifacts common in other sparsity approaches.

Visualization. Figure 6 shows a visual comparison between the video results generated by SparsevDiT and those from the top three baseline methods. We observe that MInference produces blurry results, while PAB shows over-smoothing, as indicated by the yellow box in the first row. Both SVG and PAB lose some fine details, as shown in the white box in the second row. For object contours, SVG exhibits a slight misalignment, as indicated by the red box in the third row. In contrast, our method remains closely aligned with the pretrained model in all these aspects.

CogVideoX1.5 (147.87 PFLOPS) MInference (84.89 PFLOPS) SVG (74.57 PFLOPS) PAB (105.88 PFLOPS) Sparse-vDiT (70.69PFLOPS)

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

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

Ground Truth Blurring Occurs Contours Bias & Loss Detail Loss Detail & Smooth Similar to Ground Truth

- Figure 6: Visual comparison between the proposed Sparse-vDiT and the baseline method. The green box indicates the ground truth. Yellow boxes highlight differences in blurriness and smoothness. White boxes highlight differences in fine details, while red boxes emphasize contour comparisons.

Table 3: Ablation study on the effects of hyperparameters λ and ϵ in Sparse-vDiT.

|Hyperparameter|SSIM ↑ PSNR ↑ LPIPS ↓ ImageQual ↑ SubConsist ↑ Speedup ↑<br><br>|
|---|---|
|λ<br><br>0 0.1<br><br>0.5<br><br>1<br><br><br>|0.8182 24.0864 0.1501 63.37% 92.61% 1.74× 0.8180 24.0558 0.1503 63.35% 92.62% 1.73× 0.8212 24.1311 0.1477 63.45% 92.66% 1.76× 0.8203 24.0946 0.1479 63.37% 92.58% 1.73×|
|ϵ<br><br>0.5<br><br>1 3 5<br><br><br>10<br><br>|0.8512 25.4929 0.1219 63.26% 92.60% 1.68× 0.8212 24.1311 0.1477 63.45% 92.66% 1.76× 0.7883 22.7048 0.1785 63.34% 92.66% 1.81× 0.7716 22.0171 0.1947 63.27% 92.45% 1.87× 0.7399 20.8411 0.2231 63.30% 92.49% 1.91×|

Computational Efficiency. One of the primary objectives of Sparse-vDiT is to achieve significant inference acceleration without compromising output quality. On CogVideoX1.5, it reduces computational cost from 147.87 to 70.69 PFLOPS (52.2% reduction), and on HunyuanVideo, from 612.37 to 257.09 PFLOPS (57.9%). These are the lowest among all compared methods, demonstrating the effectiveness of our sparsity strategy. In real-world latency, Sparse-vDiT consistently outperforms all baselines, reducing inference time from 901 seconds to 511 seconds on CogVideoX1.5 and from 3166 seconds to 1715 seconds on HunyuanVideo. These improvements are critical for time-sensitive applications. In terms of speedup, Sparse-vDiT achieves the highest ratios: 1.76× on CogVideoX1.5 and 1.85× on HunyuanVideo, surpassing all baseline methods. These results highlight the practical advantages of our sparsity.

Overall, Sparse-vDiT achieves an optimal trade-off between generation quality and efficiency, setting a new state-of-the-art for accelerated vDiT. These results confirm that Sparse-vDiT is not only a theoretically elegant solution but also a highly practical one, enabling scalable deployment of vDiT in latency-sensitive applications.

#### 5.3 Ablation

There are two hyperparameters in Sparse-vDiT, λ and ϵ. The parameter λ controls the trade-off between efficiency loss and quality loss, while ϵ regulates the overall sparsity of the vDiT. This section analyzes their impact through experiments on CogVideoX1.5.

Quality-Efficiency trade-off. With ϵ fixed at its optimal value of 1, we vary λ across 0, 0.1, 0.5, and

- 1. Results are reported in Table 3. Comparisons across metrics show that both λ = 0.5 and λ = 1 yield strong generation quality. However, λ=1 is less efficient. Thus, λ = 0.5 offers a better trade-off between generation quality and efficiency, and is used as the default configuration in Table 2.

Performance under different levels of sparsity. Fixing λ at 0.5, we evaluate ϵ values of 0.5, 1, 3, 5, and 10. Table 3 illustrates that increasing ϵ leads to greater sparsity, resulting in higher acceleration. For instance, ϵ = 10 achieves a speedup of 1.91×. However, higher sparsity can impair the quality of generation, as reflected in performance metrics. Notably, at ϵ = 5, Sparse-vDiT achieves a 1.87× speedup while still outperforming the SVG baseline (1.64× speedup). In practice, ϵ can be adjusted to achieve the desired balance between quality and efficiency.

### 6 Conclusion and Limitation

We propose Sparse-vDiT, an efficient inference method for vDiT based on structured sparsity. It combines predefined sparsity patterns with an offline diffusion-guided search to assign the most suitable configuration to each attention head. Experiments on CogVideo and HunyuanVideo demonstrate theoretical speedups of 2.09× and 2.38×, and actual speedups of 1.76× and 1.85×, respectively. Despite the acceleration, video quality remains comparable to that of the original models, with PSNR values of 24.13 and 27.09. These results highlight Sparse-vDiT’s ability to balance efficiency and generation quality, establishing a new state-of-the-art for sparsity-based vDiT acceleration.

Limitation: In our framework, the sparse kernel for attention is predefined. However, in practice, the predefined sparsity level may not fully align with the actual sparsity of the attention maps, potentially leading to under- or over-sparsification. We believe that enabling adaptive sparsity adjustment based on the characteristics of the attention maps, or establishing a more principled approach to sparsity design, could further enhance both sparsification effectiveness and generative performance.

### References

- [1] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [3] Thibault Castells, Hyoung-Kyu Song, Bo-Kyeong Kim, and Shinkook Choi. Ld-pruner: Efficient pruning of latent diffusion models using task-agnostic insights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 821–830, 2024.
- [4] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. Delta-dit: A training-free acceleration method tailored for diffusion transformers. arXiv preprint arXiv:2406.01125, 2024.
- [5] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
- [6] Hangliang Ding, Dacheng Li, Runlong Su, Peiyuan Zhang, Zhijie Deng, Ion Stoica, and Hao Zhang. Efficient-vdit: Efficient video diffusion transformers with attention tile. arXiv preprint arXiv:2502.06155, 2025.
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https://arxiv. org/abs/2403.03206, 2.
- [8] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.
- [9] Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6185–6194, 2023.
- [10] Haoran He, Yang Zhang, Liang Lin, Zhongwen Xu, and Ling Pan. Pre-trained video generative models as world simulators. arXiv preprint arXiv:2502.07825, 2025.
- [11] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023.
- [12] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024.

- [13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [14] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.
- [15] Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Chenyang Zhang, Michael S Ryoo, and Tian Xie. Adaptive caching for faster video generation with diffusion transformers. arXiv preprint arXiv:2411.02397, 2024.
- [16] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [17] Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A contextaware sparse attention mechanism for efficient long-sequence inference. arXiv preprint arXiv:2502.20766, 2025.
- [18] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [19] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025.
- [20] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.
- [21] Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Clear: Conv-like linearization revs pre-trained diffusion transformers up. arXiv preprint arXiv:2412.16112, 2024.
- [22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.
- [23] Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. arXiv preprint arXiv:2410.19355, 2024.
- [24] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15762–15772, 2024.
- [25] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [28] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1972–1981, 2023.

- [29] Mingzhu Shen, Pengtao Chen, Peng Ye, Guoxuan Xia, Tao Chen, Christos-Savvas Bouganis, and Yiren Zhao. MD-dit: Step-aware mixture-of-depths for efficient diffusion transformers. In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.
- [30] Rui Tian, Qi Dai, Jianmin Bao, Kai Qiu, Yifan Yang, Chong Luo, Zuxuan Wu, and YuGang Jiang. Reducio! generating 1024times1024 video within 16 seconds using extremely compressed motion latents. arXiv preprint arXiv:2411.13552, 2024.
- [31] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019.
- [32] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [34] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [35] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024.
- [36] Zhou Wang and Alan C Bovik. A universal image quality index. IEEE signal processing letters, 9(3):81–84, 2002.
- [37] Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Ptq4dit: Post-training quantization for diffusion transformers. arXiv preprint arXiv:2405.16005, 2024.
- [38] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.
- [39] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. arXiv preprint arXiv:2410.10819, 2024.
- [40] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.
- [41] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [42] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2024.
- [43] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [44] Zhihang Yuan, Hanling Zhang, Lu Pu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattn: Attention compression for diffusion transformer models. Advances in Neural Information Processing Systems, 37:1196–1219, 2024.

- [45] Yuanhao Zhai, Kevin Lin, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Chung-Ching Lin, David Doermann, Junsong Yuan, and Lijuan Wang. Motion consistency model: Accelerating video diffusion with disentangled motion-appearance distillation. Advances in Neural Information Processing Systems, 37:111000–111021, 2024.
- [46] Chi Zhang, Chengjian Feng, Feng Yan, Qiming Zhang, Mingjin Zhang, Yujie Zhong, Jing Zhang, and Lin Ma. Instructvedit: A holistic approach for instructional video editing. arXiv preprint arXiv:2503.17641, 2025.
- [47] Hanling Zhang, Rundong Su, Zhihang Yuan, Pengtao Chen, Mingzhu Shen Yibo Fan, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattnv2: Head-wise attention compression for multimodality diffusion transformers. arXiv preprint arXiv:2503.22796, 2025.
- [48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [49] Maosen Zhao, Pengtao Chen, Chong Yu, Yan Wen, Xudong Tan, and Tao Chen. Pioneering 4-bit fp quantization for diffusion models: Mixup-sign quantization and timestep-aware fine-tuning, 2025.
- [50] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024.
- [51] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. arXiv preprint arXiv:2405.18428, 2024.

## Appendix for SPARSE-VDIT

### A Algorithm Implementation

Figure 5 presents the overall process of the offline sparse diffusion search algorithm, with implementation details provided in the accompanying pseudocode. By optimizing across layers and heads, the algorithm selects attention patterns for each head in vDiT. These optimized patterns are subsequently used to accelerate inference.

Algorithm 1 Offline Sparse Diffusion Search Input: Pretraine vDiT model P (N layers and H heads), hyperparameter λ and ϵ, predefined attention

pattern Mi and sparsity Si, timestep T Output: Attention pattern config f for i in0,...,4 do

▷ Predefined Attention Kernel. Compile sparse attention Mi accoding to Si

#### end

▷ Offline Sparse Search. f=[] xT ∼ N(0,I) for t inT,...,1 do

xpt = Preprocess P(xt)

▷ vDiT Layers. for n in1,...,N do

Q,K,V = Linear, RoPE and Norm P(xpt)

▷ Attention Part (Our Optimization Object). loss = [] xgtt = M0(Q,K,V ) xot = zeros_like(xgtt ) for i in1,...,4 do

xit = Mi(Q,K,V ) loss.append(MSE(xit,xgtt ) + λSi)

#### end

▷ Per-Head Optimization. for h in1,...,H do

if (lossh > ϵ).sum ≥ 4 then

xot,h = xgtt,h f.append(0)

end else

i = argmin(lossh) xot,h = xit,h f.append(i)

end end

▷ FFN Part. xpt = FFN P(xot)

#### end

▷ Denoising. xt−1 = Solver(xt,xpt)

end returnf

Table 4: Comparison of video generation quality and efficiency between Sparse-vDiT and the baseline. All reported efficiency metrics are measured on a single NVIDIA H800 GPU with a batch size of 1.

Against Original VBench Score

Method

Latency (↓) Speedup (↑) SSIM (↑) PSNR (↑) LPIPS (↓) ImageQual (↑) SubConsist (↑)

Wan2.1 - - - 67.61% 91.95% 1935s 1.00× SVG 0.78 21.96 0.18 67.18% 91.27% 1298s 1.49× Sparse-vDiT (Ours) 0.80 22.59 0.16 67.35% 91.39% 1228s 1.58× Sparse-vDiT + FP8 (Ours) 0.79 22.39 0.16 67.22% 91.29% 1089s 1.78×

### B Performance on more pretrain models

Recent models with a Self-Attn and Cross-Attn structure, such as wan2.1, have also demonstrated strong performance. To further assess Sparse-vDiT, we evaluate it under this architecture as well. As shown in Table 4, our Sparse-vDiT framework introduces a sparse video Diffusion Transformer that achieves a 1.78× speedup (1089ms vs. 1298ms) over SVG while maintaining superior perceptual quality (SSIM: 0.80, LPIPS: 0.16), leveraging structured sparsity and FP8 quantization to reduce latency by 17% with negligible quality degradation (<0.5% drop on VBench), outperforming Wan2.1 and SVG across all metrics (+0.0204 SSIM) and demonstrating hardware-efficient scalability on H800 GPUs with near-linear acceleration, bridging the gap between theoretical sparsity and practical deployment in diffusion-based video generation.

### C More visual results

Due to space constraints, the main manuscript only compares visualization results for a limited set of baseline methods. Here, we present additional visualizations. Figures 8, Figure 7, and Figure 9 compare our method against all baselines. The WinAttn method exhibits significant contour shifts, while SVG shows smaller deviations. PAB and MInference suffer from frame smoothing and blurring. In contrast, our method preserves the overall contour consistent with the pretrained model and achieves the highest acceleration ratio to date, effectively balancing generation speed and quality. Beyond individual frame quality, frame-to-frame consistency is visualized in Figure 10 and Figure 11. Sparse-vDiT closely matches the pretrained model’s temporal consistency, indicating strong frame coherence. The visualization results start on the next page.

Prompt: A sleek Mars rover, equipped with advanced scientific instruments and cameras, traverses the rugged, reddish terrain of the Martian surface. The scene opens with a panoramic view of the barren landscape, featuring rocky outcrops and distant mountains under a dusty, pinkish sky. The rover's wheels leave distinct tracks in the fine Martian dust as it methodically navigates around boulders and craters. Close-up shots reveal its robotic arm extending to collect soil samples, while its high-resolution cameras scan the horizon for geological features. The video captures the quiet, otherworldly beauty of Mars, emphasizing the rover's relentless exploration and the vast, untouched expanse of the alien planet.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Pretrain Model (147.87 PFLOPS, Speedup 1×, SSIM 1.00)

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

MInference (84.89 PFLOPS , Speedup 1.42×, SSIM 0.61)

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

WinAttn (Spatial) (72.34 PFLOPS , Speedup 1.69×, SSIM 0.64)

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

WinAttn (Temperal) (72.34 PFLOPS , Speedup 1.67×, SSIM 0.69)

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

PAB (105.88 PFLOPS , Speedup 1.42×, SSIM 0.72)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

SVG (74.57 PFLOPS , Speedup 1.64×, SSIM 0.75)

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Sparse-vDiT (Ours, 70.69 PFLOPS , Speedup 1.76×, SSIM 0.82)

- Figure 7: More visual comparison between the proposed Sparse-vDiT and the baseline method. Our method maximizes computational speedup while maintaining high fidelity to the pretrain model.

Prompt: A drone captures a breathtaking aerial view of a festive celebration in a snow-covered town square, centered around a towering, brilliantly lit Christmas tree adorned with twinkling lights and ornaments. The scene is alive with vibrant fireworks bursting in the sky, casting colorful reflections on the snow below. The starry night sky serves as a magical backdrop, enhancing the festive atmosphere. Below, people in warm winter attire gather, their faces illuminated by the glow of the tree and fireworks, creating a heartwarming sense of community and joy. The drone's perspective showcases the entire scene, from the sparkling tree to the dazzling fireworks and the serene, star-filled sky above.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Pretrain Model (147.87 PFLOPS, Speedup 1×, SSIM 1.00)

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

MInference (84.89 PFLOPS , Speedup 1.42×, SSIM 0.61)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

WinAttn (Spatial) (72.34 PFLOPS , Speedup 1.69×, SSIM 0.64)

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

WinAttn (Temperal) (72.34 PFLOPS , Speedup 1.67×, SSIM 0.69)

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

PAB (105.88 PFLOPS , Speedup 1.42×, SSIM 0.72)

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

SVG (74.57 PFLOPS , Speedup 1.64×, SSIM 0.75)

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Sparse-vDiT (Ours, 70.69 PFLOPS , Speedup 1.76×, SSIM 0.82)

- Figure 8: More visual comparison between the proposed Sparse-vDiT and the baseline method. Our method maximizes computational speedup while maintaining high fidelity to the pretrain model.

Prompt: A sleek motorcycle, its chrome glistening, glides effortlessly through a vast, snow-covered field under a clear, azure sky. The rider, clad in a black leather jacket, helmet, and goggles, leans forward, expertly navigating the pristine, untouched snow. The motorcycle's tires leave a trail of crisp, white powder in their wake, creating a mesmerizing contrast against the dark rubber. As the bike accelerates, the engine's roar echoes through the serene, wintry landscape, sending flurries of snow into the air. The sun casts long shadows, highlighting the rider's skill and the motorcycle's powerful, streamlined design.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Pretrain Model (147.87 PFLOPS, Speedup 1×, SSIM 1.00)

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

MInference (84.89 PFLOPS , Speedup 1.42×, SSIM 0.61)

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

WinAttn (Spatial) (72.34 PFLOPS , Speedup 1.69×, SSIM 0.64)

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

WinAttn (Temperal) (72.34 PFLOPS , Speedup 1.67×, SSIM 0.69)

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

PAB (105.88 PFLOPS , Speedup 1.42×, SSIM 0.72)

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

SVG (74.57 PFLOPS , Speedup 1.64×, SSIM 0.75)

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Sparse-vDiT (Ours, 70.69 PFLOPS , Speedup 1.76×, SSIM 0.82)

- Figure 9: More visual comparison between the proposed Sparse-vDiT and the baseline method. Our method maximizes computational speedup while maintaining high fidelity to the pretrain model.

Prompt: A vibrant, multi-colored ice cream cone sits on a rustic wooden table, its creamy swirls beginning to soften under the warm sunlight streaming through a nearby window. The camera zooms in to capture the intricate details of the melting ice cream, with droplets slowly forming and trickling down the cone. The rich, velvety texture of the ice cream contrasts with the rough, weathered surface of the table. As the melting continues, the colors blend together, creating a mesmerizing, almost artistic pattern of swirls and drips. The scene evokes a sense of fleeting summer moments, with the gentle sound of a distant breeze and the soft hum of nature in the background.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

2.5s 5.0s

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

7.5s 10.0s

Pretrain Model

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

2.5s 5.0s

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

7.5s 10.0s

Sparse-vDiT (Ours)

Prompt: A close-up shot reveals an artist's hand, steady and skilled, holding a fine-tipped brush as it glides across a canvas. The brush, dipped in vibrant hues of blue and green, leaves delicate, intricate strokes that blend seamlessly into a mesmerizing landscape. The artist's fingers, speckled with paint, move with precision and grace, capturing the essence of a serene meadow under a twilight sky. The canvas, illuminated by soft, natural light, showcases the evolving masterpiece, with each brushstroke adding depth and emotion. The scene is intimate, focusing on the tactile connection between the artist and their creation, highlighting the passion and dedication poured into every detail.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

2.5s 5.0s

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

7.5s 10.0s

Pretrain Model

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

2.5s 5.0s

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

7.5s 10.0s

Sparse-vDiT (Ours)

- Figure 10: More visual comparison between the proposed Sparse-vDiT and the pretrain model. Beyond demonstrating superior performance in frame generation quality, our method exhibits robust capabilities in maintaining inter-frame consistency.

Prompt: A charming boat with a red and white exterior sails leisurely along the serene Seine River, its gentle wake creating ripples in the water. The iconic Eiffel Tower stands majestically in the background, bathed in the golden hues of a setting sun. Passengers on the boat, dressed in casual summer attire, lean against the railings, capturing the picturesque moment with their cameras. The boat glides past historic bridges adorned with ornate lampposts, while the lush greenery of riverside parks adds a touch of tranquility. The scene is framed by the soft glow of twilight, casting a magical ambiance over the entire landscape.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

2.5s 5.0s

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Sparse-vDiT (Ours)

7.5s 10.0s

Pretrain Model

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

2.5s 5.0s

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

10.0s

7.5s

###### Sparse-vDiT (Ours)

Prompt: In a breathtaking fantasy landscape, towering crystal mountains shimmer under a sky painted with swirling auroras of green and purple. A serene, emerald lake reflects the vibrant colors, while bioluminescent plants and flowers glow softly along its shores. Majestic, winged creatures soar gracefully above, their feathers glinting in the ethereal light. Ancient, twisted trees with golden leaves line a cobblestone path that winds through the scene, leading to a grand, floating castle in the distance, its spires reaching towards the heavens. The air is filled with the gentle hum of magic, creating an atmosphere of wonder and enchantment.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

2.5s 5.0s

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

7.5s 10.0s

Pretrain Model

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

2.5s 5.0s

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

7.5s 10.0s

Sparse-vDiT (Ours)

- Figure 11: More visual comparison between the proposed Sparse-vDiT and the pretrain model. Beyond demonstrating superior performance in frame generation quality, our method exhibits robust capabilities in maintaining inter-frame consistency.

