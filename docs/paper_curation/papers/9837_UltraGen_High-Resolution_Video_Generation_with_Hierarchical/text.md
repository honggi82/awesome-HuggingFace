## UltraGen: High-Resolution Video Generation with Hierarchical Attention

Teng Hu1∗ Jiangning Zhang2* Zihan Su1 Ran Yi1†

1Shanghai Jiao Tong University 2Zhejiang University {hu-teng, ranyi}@sjtu.edu.cn 186368@zju.edu.cn

Project page: https://sjtuplayer.github.io/projects/UltraGen

# arXiv:2510.18775v1[cs.CV]21Oct2025

[Figure 1]

Figure 1. Typical video generation models exhibit significant quality degradation and increased processing time with higher resolutions, whereas our UltraGen delivers superior video quality at resolutions beyond 2K while achieving 4.78× speedup compared to the popular Wan-T2V-1.3B baseline [32] (81 frames, 4×H20 GPUs). Enlarge for better visual effects.

### Abstract

Recent advances in video generation have made it possible to produce visually compelling videos, with wide-ranging applications in content creation, entertainment, and virtual reality. However, most existing diffusion transformer based video generation models are limited to low-resolution outputs (≤720P) due to the quadratic computational complexity of the attention mechanism with respect to the output width and height. This computational bottleneck makes native high-resolution video generation (1080P/2K/4K) impractical for both training and inference. To address this challenge, we present UltraGen, a novel video generation framework that enables i) efficient and ii) end-to-end native high-resolution video synthesis. Specifically, UltraGen features a hierarchical dual-branch attention ar-

chitecture based on global-local attention decomposition, which decouples full attention into a local attention branch for high-fidelity regional content and a global attention branch for overall semantic consistency. We further propose a spatially compressed global modeling strategy to efficiently learn global dependencies, and a hierarchical cross-window local attention mechanism to reduce computational costs while enhancing information flow across different local windows. Extensive experiments demonstrate that UltraGen can effectively scale pre-trained lowresolution video models to 1080P and even 4K resolution for the first time, outperforming existing state-of-the-art methods and super-resolution based two-stage pipelines in both qualitative and quantitative evaluations.

### 1. Introduction

The field of video generation [19, 29, 31, 36] has undergone rapid development in recent years, unlocking a diverse array of downstream applications, including video customization [15, 16, 24], video editing [4, 20, 23], and video motion control [13, 14, 17]. With the emergence of powerful diffusion-based generative models [10], the quality, coherence, and diversity of generated videos have significantly improved, narrowing the gap between synthetic and realworld content. Based on diffusion transformers [26], stateof-the-art models such as Wan [32] and HunyuanVideo [21] have demonstrated impressive capabilities in synthesizing temporally consistent and semantically rich videos, making remarkable progress in high-quality video generation.

Despite these advancements, current video generation models still suffer from a critical limitation: restricted resolution. Since the advanced video generation models [21, 32] are based on diffusion transformers [26], they inherently suffer from the quadratic computational complexity of the full-attention mechanism with respect to the spatiotemporal size of the input, i.e., O((T · H · W)2), where T, H, and W denote the temporal length, height, and width of the video, respectively. For instance, doubling the width and height will result in a 16-fold increase in computational cost, making high-resolution video generation prohibitively expensive for both training and inference. To mitigate this, existing approaches [2, 9, 29] often resort to a two-stage pipeline that first generates low-resolution videos and subsequently applies video super-resolution models. However, this paradigm merely enhances visual clarity and fails to introduce enough visual details, leading to the synthesis of pseudo high-resolution content with limited authenticity and richness.

To address these challenges, we propose UltraGen, a hierarchical attention-based framework for native highresolution video generation. UltraGen offers an efficient and scalable solution that transforms pre-trained lowresolution video diffusion models into end-to-end highresolution generators with significantly reduced computational overhead. Concretely, we propose a dual-branch video generation architecture that decouples the full attention mechanism into local and global attention branches. The local attention branch focuses on generating finegrained content within individual local spatial windows, while the global attention branch captures holistic video semantics and ensures coherence across different local windows. To efficiently model global dependencies without incurring prohibitive costs, we design a spatially compressed global modeling module that compresses spatial information via frame-wise convolutions before applying attention, so that the self-attention is conducted at a smaller spatial size, followed by 3D convolutions to restore spatial fidelity and enhance temporal continuity. Furthermore, to ensure

effective information flow across different local windows, we propose a hierarchical cross-window local attention mechanism. By partitioning the local windows of adjacent layers differently and creating intersections between them, our model enables seamless interaction and consistency across spatial local windows, further improving the video generation quality.

We conduct extensive experiments by extending the Wan-1.3B model to support native 1080P and 4K video generation, which is the first to achieve native high-quality 4K synthesis in the field. Comparisons against state-of-the-art models, including Wan and Hunyuan Video, as well as twostage pipelines (low-resolution generation + super resolution), demonstrate that UltraGen significantly outperforms existing methods both qualitatively and quantitatively, validating the effectiveness and scalability of our approach.

- • We propose UltraGen, a novel high-resolution video generation framework based on global-local attention decomposition, which enables scalable extension of lowresolution pre-trained video diffusion models to support 1080P and 4K resolution in an end-to-end manner.
- • We design a Spatially Compressed Global Attention Mechanism that significantly reduces computation cost of global context modeling. By compressing spatial information via frame-wise convolution, conducting selfattention at a smaller spatial size, and decoding through 3D convolution, our method efficiently captures holistic semantics while keeping temporal coherence.
- • We introduce a Hierarchical Cross-window Local Attention Mechanism that facilitates efficient interaction among local regions. By allowing intersecting regions between attention windows of adjacent layers, it ensures smooth content transitions and enhances local detail.
- • UltraGen is the first model to achieve native high-quality 4K video generation. Extensive experiments demonstrate its superior ability in HD video generation.

### 2. Related Work

#### 2.1. Video Generation Foundation Models

The advent of diffusion models [10] has greatly advanced video generation. Early methods [1, 8] typically extend text-to-image diffusion models [28] by adding temporal modules to capture frame dynamics. While somewhat effective, these approaches often separate spatial and temporal modeling, limiting their ability to capture holistic spatiotemporal dependencies and resulting in less coherent videos. With DiT [22], transformer-based architectures have become the leading paradigm in video generation [37, 41]. These models treat videos as spatiotemporal volumes, flattening them into 1D token sequences across time, height, and width. Full self-attention is then used to jointly model spatial and temporal relationships, leading to

notable improvements in temporal consistency and spatial detail. Recent work has further advanced video generation by leveraging large transformer backbones and massive video datasets. Notably, models like Wan [32] and HunyuanVideo [21] show that scaling up model size and data significantly enhances video quality and diversity. These models achieve impressive text-to-video synthesis, producing videos with rich content and improved temporal consistency. However, due to the quadratic complexity of selfattention, they remain limited to relatively low resolutions (e.g., 720P), and scaling to higher resolutions is still a major challenge.

#### 2.2. High-resolution Video Generation

To enable high-resolution generation, some existing methods such as Wan [32] and HunyuanVideo [21] train their models on videos of various resolutions, allowing them to scale to arbitrary output sizes. However, when generating videos at resolutions beyond 2K, these approaches often produce blurry results, as illustrated in Fig. 1. In contrast to directly modeling high-resolution generation, other methods [2, 9, 11, 29, 34], such as Align-Your-Latents [2], adopt a two-stage process: they first generate low-resolution videos and then apply super-resolution [6, 40, 42] to upscale the output. However, super-resolution primarily improves visual sharpness without introducing sufficient new details, resulting in pseudo high-resolution content that lacks authenticity and richness. Some recent works [5, 33] have made progress in long video generation by leveraging linear attention mechanisms [7] or test-time training [39]; however, they have paid limited attention to scaling up the spatial resolution of videos. To address these challenges, we investigate native high-definition (HD) video generation, aiming to overcome the high computational costs while producing high-quality HD videos.

### 3. Preliminaries

DiT-based Video Generation. Most state-of-the-art Diffusion Transformer (DiT) based video generation models (e.g., Wan [32] and HunyuanVideo [21]) adopt a fullattention-based framework, which builds upon the Transformer architecture to model spatiotemporal dependencies in video sequences. Typically, a 3D variational autoencoder (3D-VAE) is first used to encode an input video into a latent representation of shape D ×T ×H ×W, where D denotes the hidden dimension, T, H, and W represent the temporal frames, height, and width, respectively. This downsampling strategy effectively reduces the sequence length and makes training tractable for medium-sized videos. Then, the video latents are reshaped into a 1D token sequence with sequence length N = T × H × W via a patchify module.

Once the token sequence is obtained, video generation models apply full self-attention mechanisms across the en-

tire sequence. For a sequence of N tokens, the self-attention module computes an N × N attention map, which scales quadratically with the sequence length.

The computational complexity of self-attention is O(N2 · D), which becomes prohibitively expensive as the video resolution increases. For instance, doubling the height and width of the video leads to a four-fold increase in the number of tokens and a sixteen-fold increase in the size of the attention map. This quadratic scaling severely limits the feasibility of generating high-resolution videos (e.g., 1080P and even 4K) using existing full-attention architectures in terms of training and inference costs.

### 4. UltraGen: Born for HD Video Generation

#### 4.1. Time-Aware Global-Local Attention

As discussed in Sec. 3, in DiT-based video generation, the computational complexity of full attention is O((TWH)2× D), which grows quadratically with the spatial size (W × H) of the generated video. To address this, we restrict attention to a fixed local region by introducing an attention window of size (W0,H0). This ensures that, regardless of the overall spatial dimensions, attention is computed only within each (W0,H0) window. By applying this local attention mechanism to cover the entire frame, the total computational cost increases only linearly with the number of windows, rather than quadratically with frame size. Thus, the overall complexity is reduced to O((TW0H0)2×D) up to a constant factor, effectively avoiding quadratic scaling. However, relying solely on local attention ignores dependencies across windows, potentially leading to isolated or inconsistent content. To address this, we introduce a global attention mechanism that connects all local windows, enabling the model to capture long-range dependencies and maintain semantic consistency across the frame, thereby supporting high-resolution video generation with coherent semantics.

Therefore, we propose a novel global-local attention mechanism that decomposes the original full attention module into two complementary components: global attention and local attention. Specifically, the local attention module partitions the video sequence into multiple independent sub-regions and applies attention within each region separately, significantly reducing the overall computational cost. In parallel, the global attention module models the interactions across different local regions, injecting holistic spatiotemporal information into each local branch. This hierarchical design enables efficient and scalable attention modeling while preserving both local detail and global coherence.

Local Attention Mechanism. For a video latent representation zv ∈ RB×(T·W·H)×D, we aim to reduce the computational burden of self-attention by introducing a local

[Figure 2]

Figure 2. Overview of our UltraGen that decomposes the full-attention into a global attention branch (Sec. 4.2) for overall semantic consistency and a local attention branch (Sec. 4.3) for high-fidelity regional content, boosting high-efficiency and high-resolution video generation.

attention mechanism that approximates full self-attention with lower computational cost.

We partition the video latent zv along the spatial dimensions (H and W) into m non-overlapping, equally sized local windows, each with dimensions B × (T · W0 · H0) × D. For each local window, selfattention is applied independently, and the results are aggregated along the spatial dimensions to produce the final local attention output with the original resolution:

{zl(i)}mi=1 = Partition(zv), zl′(i) = Self-Attention(zl(i)), zl = Aggregate({zl′(i)}mi=1),

(1)

where Partition(·) divides zv into m local windows, Self-Attention(·) is applied within each window, and Aggregate(·) concatenates the outputs along the spatial dimensions to reconstruct the local attention result zl ∈ RB×T×H×W×D (detailed designs are in Sec. 4.3).

Global Attention Mechanism. Local attention reduces computational cost but limits focus to individual windows, potentially causing semantic inconsistencies. For example, a prompt describing ”a dog” might lead to multiple independent versions across windows.

To address this, we introduce a global attention module to capture long-range dependencies and ensure semantic consistency. We compress the spatial information of the video latent zv into a lower-resolution zg ∈ RB×(T·H

g·Wg)×D using a convolution module, apply global self-attention at this reduced size, and decompress the result

to the original resolution:

zg′ = Eg(zv), zg′′ = Self-Attention(WQgzg′ ,WKg zg′ ,WVgzg′ ), zg = Dg(zg′′).

(2)

where Eg is the compression encoder, and Dg is the decompression function, ensuring zg matches the original video latent size (detailed designs are in Sec. 4.2).

Time-aware Global-Local Composition. The local and global attention mechanisms yield two latent representations: the local latent zl, capturing fine-grained details, and the global latent zg , providing semantically coherent global context. To produce videos that are both globally consistent and locally detailed, we introduce a global-local fusion module that combines these representations using a learnable fusion factor α.

During the diffusion process, different denoising timesteps t focus on various video aspects: early timesteps emphasize global structure, while later ones refine details. Thus, the fusion factor α should dynamically adjust with the timestep, shifting focus from global to local information. To achieve this, we predict α based on timestep t. We embed t into a 256-dimensional time feature vector using Sinusoidal Encoding, then project it into a D-dimensional fusion factor via an MLP to fuse zl and zg:

α(t) = MLP(SinEncode(t)), R1 → RD zfused = α(t) · zg + (1 − α(t)) · zl.

(3)

#### 4.2. Spatially-Compressed Global Attention

In this section, we detailedly introduce our spatiallycompressed global attention module, which is designed to

capture global video context while maintaining computational efficiency. The key idea is to compress the spatial dimensions of video latents before performing attention, so that the self-attention is conducted at a smaller spatial size, and then decompress them back to the original resolution using spatiotemporal convolution. This reduces the attention cost without sacrificing global modeling capability.

Spatial Compression. A video can be considered as a sequence of consecutive images, and it is well-known that images can be spatially downsampled to lower resolutions while preserving global semantics at the cost of some local details. Leveraging this property, we propose to spatially compress the video latent by downsampling its width and height by a factor of k. This aligns the computational cost of global attention with that of our local attention module.

Specifically, given a video latent z ∈ RB×T×H×W×D, we apply a k × k 2D convolution with stride k along the spatial dimensions (H,W) to obtain a compressed latent zc ∈ RB×T×H

′×W′×D, where H′ = Hk and W′ = Wk . To reduce the number of parameters and computational cost in the compression layer, we adopt a channel-wise (i.e., depthwise) convolution mechanism, where each hidden dimension is processed by a separate convolution kernel with a single input and output channel. Moreover, to ensure training stability at the early stage, we initialize the convolutional kernel weights to be 1/(k × k), which initially behaves as average pooling.

Global Attention with Domain-aware LoRA. Once we obtain the compressed video latent zc ∈ RB×T×H

′×W′×D, we proceed to apply global self-attention over it. However, employing both local and global attention mechanisms requires maintaining two attention weights for each, which significantly increases computational overhead. To address this, we propose a domain-aware LoRA mechanism, which adapts the local attention parameters for global modeling. Specifically, for each projection weight W ∈ {WQ,WK,WV } and the FFN parameters WFFN, we introduce a lightweight, trainable low-rank residual [12] that specializes in global attention. The adapted weight is defined as:

##### Wglobal = W + ∆WLoRA = W + AWBW, (4)

where AW ∈ Rd×r and BW ∈ Rr×d are low-rank matrices with rank r ≪ d, and d is the input/output dimension. The same formulation is applied to WFFN.

Spatiotemporal Decompression. After obtaining the globally modeled compressed latent zcglobal ∈ RB×T×H

′×W′×D, we need to decompress it back to the original video resolution T × H × W.

Specifically, we first apply bilinear interpolation to upsample the spatial resolution from H′ × W′ to H × W. Then, to mitigate the over-smoothing effect caused by interpolation, we apply a convolutional refinement module.

Since video frames exhibit not only spatial but also temporal continuity, spatial-only operations may lead to temporal discontinuities. Therefore, we utilize a 3D convolution to perform joint spatio-temporal processing to ensure temporally consistent decompression. The overall process is formulated as:

zg = Conv3D(BilinearUpsample(zcglobal)), (5)

where zg denotes the decompressed global latent, and Conv3D denotes a 3D convolution operation over the temporal and spatial dimensions. This enables effective restoration of spatial details while preserving temporal coherence.

#### 4.3. Cross-window Hierarchical Local Attention

In order to avoid the quadratic increase in computational complexity as video resolution grows, we design local attention mechanism to partition the video latents into nonoverlapping spatial windows and then conduct self-attention in local windows. However, this partition makes it difficult to model fine-grained relationships at the boundaries between adjacent local windows. To address this issue, we propose Cross-window Hierarchical Local Attention, which can effectively model local dependencies within each window and captures interactions between neighboring windows.

Local Attention. Concretely, we first reshape the video latent zv into a new tensor of shape B × T × H × W × D. We then partition the spatial dimensions (H,W) into K × K non-overlapping local windows, resulting in a set of local video latent groups {vi,j}K,Ki=1,j=1, where each vi,j ∈ RB×T×KH×WK ×D corresponds to a spatiotemporal sub-volume of the original video latent:

H K

H K

W K

W K

,:]. (6)

vi,j = zv[:,:,i·

: (i+1)·

, j·

: (j+1)·

For each local video latent vi,j, we apply self-attention within the it to model the spatiotemporal dependencies:

vi,j′ = Self-Attention(WQsvi,j,WKs vi,j,WVs vi,j). (7) By applying self-attention only within each local window, the computational complexity is reduced from O((TWH)2 ·D) to O(K2 ·(TWHK2 )2 ·D) = O((TWH)2 · D/K2). As the number of windows increases (i.e., window size decreases), the complexity decreases accordingly. In the extreme case, it reduces the complexity to O(TWH·D) when each token forms an independent local group, enabling high-resolution video generation at significantly reduced cost.

After computing self-attention within each local win-

dow, we aggregate all locally updated features {vi,j′ } and restore them to the original video latent resolution:

zl = Rearrange({vi,j′ }Ki,j=1). (8)

[Figure 3]

Figure 3. Cross-window Attention.

This rearranged zl preserves the original spatial-temporal resolution of the video while significantly reducing the computation required during attention modeling, which ensures that local details are efficiently captured.

Cross-window Attention. Despite incorporating global information modeling, direct communication between local attention windows remains limited, especially at the boundaries, where discontinuities frequently occur. To address this, we propose a Cross-window Local Attention to enhance inter-window interaction across local attention windows.

Given that the model is composed of multiple layers of transformer blocks, we apply alternating local attention schemes at adjacent layers, where adjacent layers have different partition blocks and window boundaries. For an even-numbered layer i (i mod 2 = 0), the spatial domain of the video latent is partitioned into non-overlapping K × K windows. For an odd-numbered layer i (i mod 2 = 1), we apply a shifted window strategy with (K+1)×(K+1) partitions that partially overlap with the even-layer windows. This cross-window local attention strategy enables hierarchical interaction across neighboring windows between adjacent transformer layers.

As a result, boundary information in the K × K windows at layer i is propagated through overlapping regions in the (K + 1) × (K + 1) windows at layer i + 1, and vice versa. This enhances continuity across local attention boundaries and improves consistency in the generated outputs. Formally, the attention computation in layer i can be described as:

zcro(i) = LocalAttn(K+(i mod 2))×(K+(i mod 2))(z(i)). (9)

Hierarchical Local Attention. While the proposed crosswindow local attention enhances information exchange across adjacent local attention windows, it may still be hard to capture fast-moving small objects, which can simultaneously span multiple local windows between frames. In such cases, the limited overlapping in cross-window attention is insufficient, and global attention lacks the resolution to model fine-grained local details. To address this, we introduce a Hierarchical Local Attention (HLA) mechanism, which divides the full attention into (K/2) × (K/2) coarse windows (each twice the size as the local window), and performs local attention within each coarse window at an intermediate scale. This approach effectively compensates for

the inability of global attention to capture fine-grained details, while also overcoming the limited receptive field inherent in conventional local attention mechanisms.

Specifically, we first compress the latent features within each local window using a strategy similar to our spatialcompressed global attention. The local latent zchla within each coarse window of size 2KH × 2KW is downsampled via strided convolution. To effectively model the hierarchical attention, we apply a domain-aware LoRA adaptation to the pretrained attention weights (including WQ, WK, WV , and FFN) to ensure they are appropriately adapted for hierarchical attention computation:

##### Whla = Wlocal + ∆WHLA, (10)

where ∆WHLA is the domain-specific LoRA adaptation for hierarchical attention.

Similar to the cross-window local attention design, we employ an alternating shift mechanism between adjacent transformer layers to ensure information flow across hierarchical attention windows. That is, for layer i, hierarchical attention is computed with non-overlapping (K/2)×(K/2) windows; for layer i+1, we partition the spatial domain into (K/2 + 1) × (K/2 + 1) non-overlapping windows, making the windows of adjacent layers intersect with each other and thus enabling boundary information propagation. The attention operation at each layer can be described as:

zhla(i) = HierAttn(K

2 +(i mod 2))×(K2 +(i mod 2))(z(i)),

(11) where HierAttnk×k(·) denotes attention over a k × k partitioned hierarchical window.

This hierarchical structure, combined with cross-layer shift design and domain-aware adaptation, enables efficient fine-grained motion modeling of fast-moving small objects and enhances the robustness of local attention modeling in dynamic video scenes. To fuse the results from both the Cross-window Local Attention zcro and the Hierarchical Local Attention zhla, we employ a time-aware alpha αlocal to fuse the two results, which is the same as the Time-aware Global-Local Composition.

### 5. Experiments

#### 5.1. Implementation Details

Baselines. We compare our model with state-of-the-art methods, including Wan [32], HunyuanVideo [21], and CogVideo-X [37]. For each method, we generate two sets of videos: 1) one by directly generating videos at the target resolution, and 2) the other by first generating videos at the default resolution and then applying a super-resolution method [40] to upscale them to the target size. Note that CogVideoX cannot support HD video generation; therefore, we directly combine it with video super-resolution.

4K:A snowy village at dusk, with soft snowflakes drifting down and smoke curling from chimneys.

1080P: a person walking in the snowstorm.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

CogVideoX +SR

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Hunyuan Video+SR

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

Wan+SR

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Hunyuan Video

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Wan

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Ours

[Figure 50]

[Figure 51]

Figure 4. Comparison results of existing state-of-the-art video generation methods on 1080P video generation. The red boxes highlight zoomed-in regions, where our model produces the clearest high-resolution videos with the most fine-grained details.

Evaluation Metrics. Conventional metrics such as FVD [30] are inadequate for evaluating the quality of highresolution video generation, as they rely on pretrained low-resolution video encoders that fail to capture highresolution features. To address this limitation, we introduce three novel metrics specifically designed for highresolution video evaluation: 1) HD-FVD measures the similarity between generated and real high-resolution videos, while 2) HD-MSE and 3) HD-LPIPS assess the finegrained pixel-level and semantic-level [38] details of the generated videos, respectively. Additional CLIP score [27] and temporal consistency [18] are included for a more comprehensive evaluation. Further details and more Vbench metrics [18] are provided in the appendix.

Resolution

HDFVD ↓

HDMSE ↑

HDLPIPS ↑

Temporal Consis ↑

Method SR

CLIP-L ↑

|CogVideoX<br><br>HunyuanV Wan|✔ ✔ ✔<br><br>|394.82 97.21 0.3060 0.2834 0.9468 238.75 126.68 0.3590 0.2883 0.9614 309.10 163.86 0.3499 0.2747 0.9750<br><br>|
|---|---|---|
|HunyuanV Wan Ours|✘ ✘ ✘<br><br>|237.89 207.68 0.4911 0.2636 0.9752 821.54 42.93 0.4290 0.2528 0.9768 214.12 390.19 0.5455 0.2654∗ 0.9827<br><br>|

1080P

|CogVideoX<br><br>HunyuanV Wan|✔ ✔ ✔<br><br>|574.10 68.94 0.2645 0.2436 0.9449 453.41 276.76 0.4066 0.2576 0.9684 471.56 77.67 0.2782 0.2455 0.9697|
|---|---|---|
|HunyuanV Wan Ours|✘ ✘ ✘<br><br>|805.42 102.36 0.3858 0.2151 0.9679 1272.08 29.45 0.4270 0.2123 0.9705<br><br>424.61 386.01 0.6450 0.2444∗ 0.9710|

4K

Table 1. Quantitative comparisons. Our UltraGen demonstrates superior high-quality HD video generation capabilities. Bold indicates the best performance and ∗ indicates the best performance among all the non-SR methods.

#### 5.2. Comparison Results

Qualitative Comparison. We compare our model with state-of-the-art methods on both 1080P and 4K video generation tasks. The comparison results are shown in Fig. 4. As can be seen, the Wan model is unable to directly generate 1080P videos, resulting in blurry outputs with little to no semantic content. HunyuanVideo is capable of generating 1080P videos, but often produces results with incorrect semantics that are inconsistent with the given prompt. Methods that combine super-resolution models can generate text-aligned videos; however, the outputs after superresolution tend to be overly smooth and lack fine details. Among these, only HunyuanVideo+SR produces relatively

good results, but the level of detail is still significantly lower than that of our model, as highlighted in the zoomed-in red boxes. Therefore, our model is able to generate highresolution videos with fine-grained details while faithfully following the given prompt, demonstrating its superior performance in high-resolution video generation. Moreover, additional results generated by our model are presented in Fig. 5. It can be seen that our model consistently produces high-quality HD videos across various prompts.

Quantitative Comparison. We compare our method with state-of-the-art approaches in Tab. 1. For HD evaluation metrics, our model achieves the lowest HD-FVD scores on both 1080P and 4K video generation, indicating superior

A close-up of a goldfish swimming in a round glass bowl, bubbles rising to the surface

A lighthouse by the sea, waves gently crashing against the rocks.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Without Global Attention

###### 1080P4K

[Figure 58]

[Figure 59]

[Figure 60]

A close-up of a boy with round glasses, reading an old book by candlelight.

[Figure 61]

[Figure 62]

[Figure 63]

Without Cross-window Attention

[Figure 64]

[Figure 65]

[Figure 66]

Without Hierarchical Attention

A timelapse of the northern lights dancing above a snowy mountain landscape.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Without Domain-aware LoRA

A woman sits solemnly on a chair, looking left and right.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

###### Swin Local Attention

Figure 5. More generated HD videos (1080P & 4K).

[Figure 79]

[Figure 80]

[Figure 81]

Full Model (Ours)

Resolution HunyuanVideo Wan UltraGen (Ours) Speedup (Ours)

1080P 43 min 35 min 13 min ×2.69 4K 11h 36min 8h 46min 1h 50min ×4.78

Figure 6. Ablation study on the proposed modules.

Table 2. Comparison of inference time. Our model archives a 4.78 × speedup compared to the baseline Wan model.

out global attention tends to generate disjoint content, exemplified by the isolated 16 golden fishes in the rightmost case. Models lacking either cross-window local attention or hierarchical attention can capture global relationships only coarsely and still exhibit inconsistencies at window boundaries. The model without domain-aware LoRA alleviates boundary inconsistency but suffers from reduced generation quality, producing somewhat blurry results. This is due to the limited capacity of a single set of attention weights to model three distinct attention mechanisms (global, local, and hierarchical). Moreover, when replacing hierarchical cross attention with Swin-Attention for local attention modeling, we observe that although adjacent windows can be connected smoothly, Swin-Attention struggles to effectively capture hierarchical features. As a result, the model often generates semantically inconsistent content across windows. For example, it may produce two goldfish in adjacent windows where only one should appear, indicating a lack of semantic coherence. In contrast, the full model generates high-quality videos, effectively resolves boundary inconsistencies, and captures global semantics well, validating the effectiveness of all our proposed modules. More quantitative ablation studies are shown in the appendix.

quality and diversity in the generated videos. Furthermore, we obtain the best HD-MSE and HD-LPIPS, demonstrating that our generated videos contain the most fine-grained details and validating the effectiveness of our HD video generation ability. Our model also achieves the best temporal consistency, which demonstrates the smoothness of the generated videos and the coherence across frames. In terms of prompt following, we observe that directly generating HD videos without super-resolution leads to a relatively lower CLIP score due to the difficulty in high-resolution video generation. Since our model is based on Wan 1.3B, its CLIP score cannot surpass that of Wan+SR. Nevertheless, we still achieve the best CLIP score among methods that natively generate high-resolution videos, highlighting the strong prompt-following capability of our model.

Time Comparison. Finally, we compare the inference time of our model with HunyuanVideo and Wan at different resolutions, as shown in Tab. 2. Our model achieves a 2.7× speedup for 1080P generation and a 4.78× speedup for 4K generation compared to the baseline Wan model, demonstrating the high efficiency of our approach for highresolution video generation.

### 6. Conclusion

#### 5.3. Ablation Studies

In this work, we propose UltraGen, a novel framework for efficient, end-to-end native high-resolution video generation. By leveraging a hierarchical dual-branch attention architecture, UltraGen effectively decouples local and global attention, enabling the synthesis of high-fidelity regional details while maintaining overall semantic consis-

We conduct ablation studies on five variants: (1) without global attention, (2) without hierarchical attention, (3) without domain-aware LoRA, (4) without cross-window local attention, and (5) employing Swin-Attention [25] for local attention modeling. As shown in Fig. 6, the model with-

tency. Our spatially compressed global modeling and hierarchical cross-window local attention mechanisms further reduce computational complexity, making high-resolution video generation (up to 4K) feasible for both training and inference. Extensive experiments demonstrate that UltraGen not only scales pre-trained low-resolution models to 1080P and 4K resolutions, but also consistently outperforms existing state-of-the-art methods and super-resolution pipelines in both qualitative and quantitative evaluations.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023. 2, 3
- [3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 11
- [4] Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin Hu, and Shuicheng Yan. Ivebench: Modern benchmark suite for instruction-guided video editing assessment. arXiv preprint arXiv:2510.11647, 2025. 2
- [5] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17702–17711, 2025. 3
- [6] Yuzhen Du, Teng Hu, Ran Yi, and Lizhuang Ma. Ld-bfr: Vector-quantization-based face restoration model with latent diffusion enhancement. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 2852–2860,

2024. 3

- [7] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3
- [8] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [9] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 2, 3

- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2

- [11] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. ArXiv, abs/2210.02303, 2022. 3
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 5
- [13] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 2
- [14] Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. Motionmaster: Training-free camera motion transfer for video generation. arXiv preprint arXiv:2404.15789, 2024. 2
- [15] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation, 2025. 2
- [16] Teng Hu, Zhentao Yu, Zhengguang Zhou, Jiangning Zhang, Yuan Zhou, Qinglin Lu, and Ran Yi. Polyvivid: Vivid multisubject video generation with cross-modal interaction and enhancement. arXiv preprint arXiv:2506.07848, 2025. 2
- [17] Teng Hu, Jiangning Zhang, Ran Yi, Hongrui Huang, Yabiao Wang, and Lizhuang Ma. High-efficient diffusion model fine-tuning with progressive sparse low-rank adaptation. In 13th International Conference on Learning Representations, ICLR 2025, pages 92066–92078. International Conference on Learning Representations, ICLR, 2025. 2
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7, 12
- [19] Ziyao Huang, Fan Tang, Yong Zhang, Xiaodong Cun, Juan Cao, Jintao Li, and Tong-Yee Lee. Make-your-anchor: A diffusion-based 2d avatar generation framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6997–7006, 2024. 2
- [20] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3, 6, 12, 13
- [22] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2
- [23] Sen Liang, Zhentao Yu, Zhengguang Zhou, Teng Hu, Hongmei Wang, Yi Chen, Qin Lin, Yuan Zhou, Xin Li, Qinglin Lu, et al. Omniv2v: Versatile video generation and editing via dynamic content manipulation. arXiv preprint arXiv:2506.01801, 2025. 2

- [24] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment. arXiv preprint arXiv:2502.11079, 2025. 2
- [25] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 8, 12
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [29] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2, 3

- [30] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7, 11
- [31] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 2
- [32] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 6, 11, 12, 13
- [33] Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqiao Luo, Peizhao Zhang, Tingbo Hou, Peter Vajda, Niraj K. Jha, and Xiaoliang Dai. Lingen: Towards high-resolution minute-length text-tovideo generation with linear computational complexity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2578–2588, 2025. 3
- [34] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, pages 1–20, 2024. 3

- [35] Zhucun Xue, Jiangning Zhang, Teng Hu, Haoyang He, Yinan Chen, Yuxuan Cai, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, et al. Ultravideo: High-quality uhd video dataset with comprehensive captions. arXiv preprint arXiv:2506.13691, 2025. 11
- [36] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai Li, and Dacheng Tao. Adavideorag: Omni-contextual adaptive retrieval-augmented efficient long video understanding. In NeurIPS, 2025. 2
- [37] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 6, 13
- [38] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7, 12
- [39] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 3
- [40] Yuehan Zhang and Angela Yao. Realviformer: Investigating attention for real-world video super-resolution. In European Conference on Computer Vision, pages 412–428. Springer,

2024. 3, 6

- [41] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 2
- [42] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporalconsistent diffusion model for real-world video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2535– 2545, 2024. 3

- A. Overview This supplementary material consists of:

- • Efficiency analysis (Sec. B);
- • More implementation details (Sec. C);
- • High-resolution video evaluation metrics (Sec. D);
- • Evaluation on Vbench (Sec. E);
- • Quantitative ablation studies (Sec. F);
- • More visualization results (Sec. G);
- • Limitations (Sec. H).

- B. Efficiency Analysis

Since the primary computational cost of full attention lies in the calculation of the attention map, we approximate the overall computational complexity by analyzing the complexity of the attention map itself. The original full attention mechanism has a computational complexity of O((TWH)2D).

We consider the case without cross-window connections for ease of analysis (note that introducing cross-window connections increases the effective K, which can further reduce the computational cost to some extent), where the input is partitioned into K × K local windows. The computational complexity of our local attention is then O(K2 · (TWHK2 )2 · D) = O((TWH)2D/K2).

In addition to local attention, our model also incorporates global attention and hierarchical attention. However, by leveraging a global latent compression mechanism, we ensure that the attention map sizes for these two modules are consistent with that of local attention. Specifically, the computational complexity for global attention is O((TWHK2 )2D), and for hierarchical attention, it is O((K2 )2· (TWHK2 )2D) = O((TWH)2D/(4K2)).

Therefore, the total computational complexity is expressed as:

(TWH)2D K2

T(n) = O

(TWH)2D 4K2

+ O

5(TWH)2D 4K2

= O

+

+ O

(TWH)2D K4

(TWH)2D K4

,

(12)

Then, the speedup ratio compared to the standard complexity O((TWH)2D) is:

(TWH)2D 5

Speedup =

- 4K2(TWH)2D + K14(TWH)2D

=

1

- 5

(13)

4K2 + K14

- 4K4

- 5K2 + 4

=

where K = 4 is used in our experiments, resulting in an approximate 12-fold speedup. However, in practice, the actual speedup is somewhat lower than 12 due to the additional computation required for generating queries, keys, and values in the global and hierarchical attention modules. Nevertheless, as the resolution increases and the attention map computation becomes the dominant cost, the observed speedup approaches the theoretical value of 12.

### C. More Implementation Details

Training and Inference Details. We perform full finetuning on the pretrained Wan 1.3B model [32], integrating domain-aware LoRA with a rank of 64 for both global and hierarchical attention mechanisms. The training process utilizes the UltraVideo dataset [35], which comprises 42,000 4K-resolution videos, and is conducted over 50 epochs. Training is executed on 32 H20 GPUs with a batch size of 32 and a learning rate of 1 × 10−4. For the 1080P video generation model, we fix the number of frames at 81, following the official configuration of Wan. For the 4K model, due to GPU memory constraints, we are only able to train the model with 29 frames. For inference, we employ 30 denoising steps and set the classifier-free guidance scale to 5.0.

### D. High-resolution Video Evaluation Metrics.

HD-FVD: The standard FVD metric [30] utilizes the I3D network [3] to extract video features, which involves resizing input videos to a low resolution (Hl × Wl) prior to feature extraction and comparison. To enable evaluation at high resolutions, we propose HD-FVD, which decomposes high-resolution videos into patches of size Hl × Wl. Features are then extracted from these patches using the pretrained I3D network, thereby preserving high-resolution information. The Fr´echet Distance is subsequently computed between the features of generated and reference video patches.

HD-MSE: High-resolution videos inherently contain fine details that are absent in their low-resolution counterparts. To quantitatively assess the preservation of such details, we first downsample the videos by set of factors of {2k}, resulting in a set of downsampled videos {vD,2k}. Each downsampled video is then upsampled back to the original resolution, and the mean squared error (MSE) is computed with respect to the original video. This process is formalized as:

##### HD-MSE =

k

∥v − vD,2k∥ (14)

A higher HD-MSE indicates that more fine details are lost during downsampling, thereby reflecting the presence of high-quality, high-resolution content in the generated

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Aesthetic Quality ↑

Imaging Quality ↑

Average ↑

Resolution Method SR

CogVideoX ✔ 0.9456 0.9592 0.9901 0.5138 0.5771 0.7972

HunyuanVideo ✘ 0.9796 0.9839 0.9967 0.5892 0.6237 0.8346 Wan ✘ 0.9770 0.9762 0.9967 0.4317 0.4529 0.7669 Ours ✘ 0.9771 0.9777 0.9961 0.5819 0.7350 0.8536

1080P

CogVideoX ✔ 0.9472 0.9575 0.9895 0.5072 0.5708 0.7944

HunyuanVideo ✘ 0.9964 0.9967 0.9979 0.3973 0.4402 0.7657 Wan ✘ 0.9466 0.9764 0.9952 0.2877 0.3735 0.7159 Ours ✘ 0.9854 0.9894 0.9933 0.5787 0.6832 0.8460

4K

Table 3. Quantitative comparison on selected methods using VBench metrics. Bold denotes the best score.

videos. In our experiment, we enumerate k from 3 to 5 (corresponds downsample factor 8, 16, and 32) to compute the HD-MSE.

HD-LPIPS: Analogous to HD-MSE, HD-LPIPS evaluates the preservation of fine-grained semantic details in high-resolution videos. Here, the MSE in Eq. 14 is replaced with the LPIPS metric [38], which is more sensitive to perceptual differences:

##### HD-LPIPS =

k

LPIPS(v − vD,2k), (15)

where we use k = {3,4,5} to compute HD-LPIPS.

### E. Evaluation on Vbench

To further demonstrate the effectiveness of our approach, we conduct comparisons with several state-of-the-art methods using the VBench evaluation framework [18]. All methods are evaluated under identical resolution and prompt settings. VBench offers a standardized and comprehensive suite of metrics—including subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality—enabling a thorough assessment of video generation quality. It is important to note that VBench is not designed for high-resolution generation; thus, videos must be resized to the standard resolution of the pretrained models for evaluation. As a result, super-resolution-based methods such as Wan [32] and HunyuanVideo [21] are not included in our comparison. When their high-resolution outputs are downsampled to the standard resolution for VBench evaluation, the assessment essentially reflects the performance of the base models (i.e., HunyuanVideo and Wan) rather than their high-resolution generation capabilities, which would not provide a fair comparison in the highresolution video generation setting.

Table 3 reports the quantitative results on the generated videos from different methods. It can be seen that our method achieves the highest overall average score in both 1080P and 4K resolution, demonstrating a balanced and robust performance across diverse aspects of video quality. Notably, our approach attains the best Imaging Qual-

ity score, reflecting its strong ability to mitigate low-level distortions such as blur and noise in high-resolution frames. While it does not outperform all competitors on every individual metric, it consistently ranks near the top across all categories. In comparison, methods that directly generate high-resolution videos, such as Wan and HunyuanVideo, show significantly lower scores on perceptual quality metrics like Aesthetic Quality and Imaging Quality, indicating challenges in preserving fine details and reducing artifacts at scale. These results validate the effectiveness of our model in producing high-fidelity, temporally coherent videos with fewer visual distortions.

### F. Quantitative Ablation Studies.

In the main paper, we have demonstrated the effectiveness of each proposed module in UltraGen. To provide a more comprehensive and rigorous evaluation, we present additional quantitative ablation studies in this section, examining five ablated variants: (1) without global attention, (2) without hierarchical attention, (3) without domainaware LoRA, (4) without cross-window local attention, and (5) replacing our local attention module with SwinAttention [25]. Quantitative comparisons are reported in Table 4 using the HD-FVD, CLIP-L, and VBench metrics. Our model achieves the best performance on both HD-FVD and CLIP-L, indicating superior high-definition generation quality. Furthermore, with respect to the VBench metrics, our model attains the highest scores in motion smoothness and aesthetic quality, as well as the second-best imaging quality, resulting in the highest overall average VBench score. Notably, the variants without global attention or cross-window local attention exhibit severe boundary inconsistencies, leading to the lowest aesthetic and imaging quality. Both hierarchical attention and domain-aware LoRA contribute to improved generation quality; omitting either results in a moderate decrease in performance. Compared to the Swin-Attention variant, our hierarchical cross-layer mechanism demonstrates superior performance in high-resolution video generation. In summary, our model achieves state-of-the-art HD video generation performance,

Subject Consistency↑

Background Consistency↑

Motion Smoothness↑

Aesthetic Quality↑

Imaging Quality↑

Average↑ without global attention 328.98 0.2302 0.9680 0.9692 0.9919 0.4489 0.6386 0.8033

Method HD-FVD ↓ CLIP-L ↑

without cross-window attention 419.15 0.2488 0.9720 0.9725 0.9929 0.4369 0.6964 0.8141 without hierarchical attention 376.49 0.2581 0.9800 0.9821 0.9943 0.5400 0.6784 0.8350 without domain-aware LoRA 284.08 0.2603 0.9790 0.9791 0.9948 0.5541 0.7424 0.8499

swin local attention 458.93 0.2548 0.9789 0.9756 0.9943 0.5308 0.7228 0.8405 full model (ours) 214.12 0.2654 0.9771 0.9777 0.9961 0.5819 0.7350 0.8536

Table 4. Quantitative ablation study.

1080P: an expansive and vast desert landscape. 4K: Blooming roses in a garden,

morning dew sparkling on their petals.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

CogVideoX +SR

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Hunyuan Video+SR

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

Wan+SR

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Hunyuan Video

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Wan

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Ours

[Figure 129]

[Figure 130]

Figure 7. More Qualitative comparisons between our UltraGen and the existing HD video generation methods.

validating the effectiveness of each proposed module.

resolution leads to outputs with reduced detail and texture. In contrast, our UltraGen model not only aligns closely with the given prompts but also achieves superior high-definition video generation quality.

### G. More Visualization Results.

More qualitative comparisons. In this section, we present additional qualitative comparisons between our UltraGen model and several baseline methods, including CogVideoX [37]+SR, HunyuanVideo [21]+SR, Wan [32]+SR, as well as the native HunyuanVideo and Wan models. The supplementary results are illustrated in Fig. 7. As shown, both HunyuanVideo and Wan struggle to generate high-quality native 1080P and 4K videos: HunyuanVideo fails to follow the prompt and introduces significant noise at 4K resolution, while Wan produces videos that are overly smooth and lack detail. Although the superresolution-based models are able to generate videos that are consistent with the prompts, their heavy reliance on super-

Additional 1080P and 4K Results. To further demonstrate the effectiveness and robustness of our model, we present additional examples of generated 1080P videos in Fig. 8 and 4K videos in Fig. 9. As shown, our model consistently produces high-quality videos that faithfully correspond to a diverse range of text prompts. It should be noted that the 4K videos are limited to 29 frames due to GPU memory constraints, which may somewhat restrict their temporal dynamics. Nevertheless, the overall results are still very impressive.

### H. Limitations

While our model is capable of generating high-quality, high-resolution videos, it still inherits certain limitations from the underlying base model, which was originally designed for lower-resolution outputs. As a result, in particularly challenging scenarios—such as those involving rapid or large-scale motions—the model may occasionally encounter difficulties in accurately capturing complex motion dynamics, leading to minor artifacts or less natural motion. Addressing these challenges and further enhancing the model’s robustness in such demanding high-resolution settings will be an important focus of our future work.

A lighthouse by the sea, waves gently crashing against the rocks.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

A surreal underwater sculpture gallery, ancient marble statues covered with glowing coral.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

A pencil sketch of a child with curly hair, gazing thoughtfully before a rain-speckled window.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

A soft-focus portrait of a girl, her expression calm and serene.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Two pandas are playing on the wood.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

A dramatic cityscape during a fireworks festival, colorful explosions lighting up the night sky.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

A magical fairy circle in a moonlit glade, tiny glowing fairies dancing.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Figure 8. More 1080P visualization results generated by our model.

A tranquil monastery on a mountain peak, clouds drifting below, spiritual serenity.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

A tranquil waterfall cascading into a hidden pool, lush greenery, mist rising.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

A vibrant coral reef teeming with marine life, sunlight filtering through the water, high detail.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

A portrait of a young woman with headphones, eyes closed and lost in music.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Two cute kittens are sleeping in the basket.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

A magical snow-covered village with twinkling lights, festive and cozy.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

A magical portal opening in an ancient stone circle, swirling energy, fantasy adventure.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Figure 9. More 4K visualization results generated by our model.

