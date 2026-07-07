[Figure 1]

## EarlyTom : Early Token Compression Completes Fast Video Understanding

Hesong Wang1,2,3,⋆, Xin Jin2,⋆, Lu Lu3,†, Chenhaowen Li3, Jian Chen3, Qiang Liu3, Huan Wang2,† 1Zhejiang University 2Westlake University 3Alibaba Cloud Computing

{wanghesong, jinxin86, wanghuan}@westlake.edu.cn ll200214@alibaba-inc.com

https://viridisgreen.github.io/EarlyTom

[Figure 2]

# arXiv:2605.30010v1[cs.CV]28May2026

[Figure 3]

Better

###### Large Language Model

(Avg. : 58.2)

(Avg. : 58.8)

(Avg. : 57.6)

(Avg. : 56.2)

Spatial selection

(Avg. : 58.2)

Tokenizer

(Avg. : 57.3)

Prompt Input

Projector

<system> Describe this video.

###### Video Input

(Avg. : 58.4)

Vision Encoder Frame merging

[Figure 4]

[Figure 5]

[Figure 6]

(Avg. : 54.1)

...

Figure 1. Left: This paper aims to improve the inference efficiency of video understanding based on video large language models (LLMs). Latency profiling suggests the major speed bottleneck lies in the vision encoder part instead of the LLM. Knowing this, we introduce EarlyTom, a training-free token compression method designed for the early stage (i.e., vision encoder) of video LLMs. EarlyTom features two core components: (1) early-stage visual token compression achieved via inner vision encoder frame merging, and (2) a spatial token selection strategy that further increases compression effectiveness without introducing bias. Right: Scatter plot illustrating the relationship between FLOPs and throughput, along with the average performance across four widely used video understanding benchmarks (MVBench, EgoSchema, LongVideoBench, and VideoMME) for several training-free state-of-the-art methods. EarlyTom achieves state-of-the-art performance while maintaining accuracy comparable to full-token methods.

### Abstract

duce a decoupled spatial token selection strategy that improves the overall compression effectiveness. EarlyTom reduces TTFT by up to 2.65× and FLOPs by up to 61% on a single NVIDIA A100 GPU for the LLaVA-OneVision-7B model, while maintaining accuracy comparable to the fulltoken baseline. These improvements substantially enhance the practicality of deploying Video-LLMs in real-world production scenarios.

Video large language models (Video-LLMs) have demonstrated strong capabilities in video understanding tasks. However, their practical deployment is still hindered by the inefficiency introduced by processing massive amounts of visual tokens. Although recent approaches achieve extremely low token retention ratios while maintaining accuracy comparable to full-token baselines, most of them perform compression only at the late stage of prefilling, leaving the efficiency of the vision encoder unoptimized. In this paper, we first show that vision encoding contributes a large portion to the time-to-first-token (TTFT). Therefore, instead of compressing visual tokens only after the vision encoder, performing compression inside the encoder still leaves substantial room for exploration. Based on this insight, we propose EarlyTom, a training-free token compression framework that performs early-stage visual token compression inside the vision encoder, enabling significantly better TTFT reduction and higher throughput. In addition, we intro-

### 1. Introduction

Video large language models (Video-LLMs) [1, 5, 19, 21, 24, 29, 38, 44, 46, 49] have demonstrated impressive capability in video understanding tasks. However, efficiently processing large volumes of visual tokens is computationally expensive, which significantly limits the practical deployment of Video-LLMs in real-world scenarios. Although existing methods have made notable progress in compressing vision tokens to improve efficiency, most of them overlook the vision encoder itself. As illustrated in Figure 3, the vision encoding stage consumes 36.3% of the total time-to-first-token (TTFT) in the baseline, and this issue becomes even more pronounced in state-of-the-

⋆Equal contribution. †Corresponding author. Work done while Hesong Wang was an intern at Alibaba Cloud Computing.

art methods such as HoliTom and VisionZip, where it rises to 55.8% and 68.4%, respectively. As a result, there is still large room to improve the performance of Video-LLMs.

As summarized in prior works [32, 33, 35], most existing token compression methods operate either after the vision encoder or inside the LLM. Inner-LLM token compression methods, such as FastV [4], SparseVLM [50], and PyramidDrop [40], focus on compressing tokens within the LLM and therefore provide limited reduction in TTFT. On the other hand, outer-LLM strategies (e.g., VisionZip [43] and LLaVAPruMerge [31]) compress tokens before entering the LLM, offering higher but still limited TTFT reduction. Hybrid approaches such as HoliTom [32], FastVID [34], and DyCoke [35] attempt to combine both paradigms but still face constrained acceleration, which fundamentally restricts their practicality in compute-bound applications like largescale video retrieval. Addressing TTFT bottlenecks in video LLMs remains an open challenge.

To better understand the problem, we profile the TTFT composition across several state-of-the-art methods. The results in Figure 3 reveal that vision encoding accounts for a major portion of TTFT, especially in methods already optimized for LLM prefill latency. In addition, existing compression methods introduce non-trivial computational overhead, which further increases TTFT. These observations motivate us to design a token compression mechanism that acts early inside the vision encoder while minimizing extra overhead for faster and efficient inference.

In this paper, we present EarlyTom, an efficient token compression framework designed for extreme performance. Specifically, we propose (1) an inner vision encoder frame merge strategy that compresses redundant visual information during the encoding process, and (2) a decoupled token selection strategy co-designed at the system level to further reduce visual tokens with minimal latency. On LLaVA-OneVision-7B, with only 10% token retention, EarlyTom achieves 2.65× TTFT reduction and 1.3× throughput speedup, while maintaining competitive downstream quality across diverse video understanding benchmarks.

Our main contributions are summarized as follows:

- (a) We propose an inner vision encoder frame merge mechanism that compresses redundant visual information during vision encoding, effectively reducing visual tokens with negligible overhead and significantly reducing time-to-first-token.
- (b) We introduce a decoupled token selection strategy that performs efficient, low-latency token reduction, further shrinking vision tokens and enabling substantial endto-end acceleration without sacrificing accuracy.
- (c) Extensive experiments on LLaVA-OneVision-0.5B/7B demonstrate that EarlyTom achieves state-of-the-art acceleration performance, delivering extremely fast TTFT while maintaining comparable accuracy.

### 2. Related Work

Intra-encoder token compression. Intra-encoder methods perform token compression within the vision encoder or projector, before tokens are fed into the language model. ToMe [2] reduces tokens in the vision encoder depending on the similarity of key tokens, which improves efficiency and acceleration. PiToMe [36] proposes an energy score to preserve informative tokens; large similar clusters are merged, while unique tokens with low energy are retained. LLaVAPruMerge [31] selects cluster centers based on attention scores from the [CLS] tagged tokens, then merges the remaining tokens with lower attention scores through KNN clustering [12] and a weighted cluster center update mechanism. VisionZip [43] retains visual tokens with higher attention scores, then merges the remaining tokens through clustering. FiCoCo [13] integrates multi-dimensional redundant evaluations, tokenadaptive association matching, and weighted fusion strategies through a “filtering-association-compression” process. MustDrop [25] proposes merging similar neighborhood tokens while retaining key tokens in the visual encoder, and by employing dual attention filtering during the prefilling stage to eliminate text-irrelevant tokens. TokenPacker [23] introduces an efficient visual projector with a coarse-to-fine design: it first generates low-resolution point queries via bilinear interpolation, then refines them by injecting highresolution multi-level visual features through a region-topoint module. MergeMix [15] proposes a preference tuning by building augmented samples and training with token merge for efficiency.

Pre-LLM token compression. Pre-LLM methods perform token compression before the language model and after the vision encoder, treating the compression as a plug-andplay module. DyCoke [35] proposes a training-free twostage compression pipeline that merges redundant frame tokens through cross-frame temporal compression, followed by dynamic KV cache pruning during decoding to eliminate spatial redundancy while dynamically preserving key tokens. FastVID [34] analyzes video redundancy from temporal and visual density perspectives, proposing dynamic temporal segmentation and density-driven spatio-temporal pruning. It segments videos and prunes based on local “information density”. PVC [42] proposes a training strategy that progressively encodes each frame and adaptively compresses redundant tokens by leveraging temporal redundancy. VScan [47] conducts systematic empirical research on how LLM handles visual tokens, merging them during visual encoding and introducing fine-grained pruning at intermediate model layers. HoliTom [32] emphasizes global and redundancy-aware holistic compression, reducing tokens by outer-LLM spatio-temporal segmentation and merging while incorporating a robust inner-LLM merging

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

...

...

...

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

VideoFrames

Token id Token id Token id

- Figure 2. The video sink tokens. We visualize videos across datasets to illustrate the video attention sinking phenomenon: certain tokens (specific frames/regions) consistently attract disproportionately high attention (as shown in the attention score heatmaps), revealing that existing top-K-based token compression methods overlook semantic information in other frames and limit video context understanding.

| |889 ms| | | | |
|---|---|---|---|---|---|
| |556 ms<br><br>458 ms<br><br>336 ms<br><br>1.60x<br><br>1.94x<br><br>2.65x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Baseline HoliTom VisionZip Ours

0

200

400

600

800

Latency(ms)

Vision Encoding

Visual Token Processing

LLM Prefill

System Overhead

- Figure 3. Time-to-first-token (TTFT) latency composition. We break down TTFT into four parts: vision encoding, visual token processing, LLM prefill, and system overhead. In the baseline, vision encoding takes 323 ms, accounting for 36.3% of the total, indicating that this stage still has substantial room for optimization. For state-of-the-art methods like HoliTom and VisionZip, vision encoding remains the largest component, occupying 55.8% (324 ms) and 68.4% (325 ms), respectively. In addition, HoliTom introduces extra token-processing overhead, increasing this component by 121.9% (+78 ms) compared to the baseline. In contrast, our method reduces vision encoding time directly inside the encoder, achieving a 2.65× TTFT reduction over the baseline while adding almost no additional overhead, evaluated under 10% token retention on an NVIDIA A100 GPU.

query-aware method for extreme compression.

### 3. Method

In this section, we present EarlyTom, a training-free token compression framework for efficient video LLM inference. The overall pipeline is illustrated in Figure 4 and detailed in the following sections.

#### 3.1. Preliminaries and Analysis

Video-LLM inference. The inference process of video LLMs can be divided into three main stages: vision encoding, LLM prefilling, and decoding. During vision encoding, video frames are transformed into embedding representations, which are then aligned to the LLM embedding space through a projector to form video tokens. These video tokens are subsequently concatenated with text tokens and fed into the LLM during the prefilling stage. Finally, the LLM generates responses in an autoregressive manner during decoding. Our method primarily focuses on optimizing the vision encoding and pre-prefilling stages to reduce latency while preserving accuracy.

Profiling of time-to-first-token. To identify the primary bottlenecks in video LLM inference, we decompose the time-to-first-token latency into four components: vision encoding, visual token processing, LLM prefill, and system overhead. As illustrated in Figure 3, vision encoding occupies a substantial portion of TTFT. In the baseline setting, vision encoding accounts for 36.3% of the total TTFT, and this proportion becomes even more pronounced when applying LLM-prefill–optimized methods such as HoliTom and VisionZip, where it rises to 55.8% and 68.4%, respectively. Meanwhile, HoliTom introduces additional compression overhead during the visual token processing stage, further increasing the first-token latency.

strategy. QueCC [22] analyzes the trade-off between visual tokens and LLM size via inference-time scaling laws, showing that under fixed compute, visual reasoning favors larger LLMs with aggressive token compression, and proposes a

Video sink tokens. To analyze how visual tokens contribute

7 Full Tokens within One Frame

Pruned Single Token

###### Stage I: Inner-Vision Encoder Frame Merging

Large Language Model 𝑓

[Figure 19]

Single Token 7 Compressed Tokens within One Frame

###### ... 1 2 3 4 5 ...

###### Stage II: Decoupled Spatial Selection

Text Tokens

Vision Tokens

1 2 3 4 5 6 7 8

w1 * 2 + w2 * 3

...

Weighted frame merge

###### 2 4 6 7

1 3 5 8

Tokenizer t

Visual Token Spatial Compression

Local window top-K selection

Top-K selection

sim1 > sim2

...

[Figure 20]

📌 📌

[Figure 21]

... ...

Prompt Input

1 2 3 4 5

Dynamic tokens

Static tokens

<system> Describe this video.

Tokens in Frame

sim1 > τ

Middle frame merge

2 4 6 7

1 3 5 8

Video Input (10s~50min)

Projector 𝑝

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

... ...

...

1 2 3 4 5

1 2 3 4 5 6 7 8

Vision Encoder g

Streaming frame segmentation

Divide into dynamic/static

- Figure 4. Overall pipeline of EarlyTom. Our method consists of two main stages for efficient video token compression. Stage I: Innervision encoder frame merging performs temporal compression inside the vision encoder. The video is adaptively segmented based on streaming frame similarity, redundant middle frames are merged using a local-optimal criterion, and merged representations are further refined with weighted fusion to reduce early-stage temporal redundancy. Stage II: Decoupling selection conducts spatial token reduction after vision encoding. Merged frame features are decomposed into dynamic and static token sets: dynamic frames undergo global TopK selection, while static frames use local-window selection to preserve spatial distribution. The selected tokens from both paths are recombined and fed into the LLM for decoding. Together, these two stages enable early temporal compression and balanced spatial sampling, significantly accelerating Video LLM inference while maintaining semantic fidelity.

sion encoder as illustrated in Figure 5. Specifically, we first divide the video into segments according to frame similarity in a streaming manner, which is computed by averaging the cosine similarities of tokens at corresponding spatial positions. For two consecutive frames, we calculate their cosine similarity and update the score with an Exponential Moving Average (EMA) over time. When the similarity score drops below a predefined threshold, we treat this point as a segment boundary, which is described in the equation below:

to cross-frame information, we visualize SigLIP [45] attention maps across video frames. We find that certain spatial patch locations consistently receive unusually high attention, forming vertical stripes across frames even when visual content changes. Some works [6, 9, 11, 16, 39, 51, 53, 54] have shown that these correspond to sink tokens, whose query/key vectors exhibit abnormally large norms.

⊤ √ j

Formally, for attention A(i,j) = QiK

d , sink tokens satisfy |Qsink|2 ≫ |Qp|2, forcing A(sink,j) to dominate regardless of content. Thus, raw attention scores from SigLIP cannot directly indicate token importance, since a portion of attention is absorbed by these structural attractors rather than meaningful visual regions.

sˆt = αst + (1 − α)ˆst−1, break if sˆt < τseg, (1)

where α denotes the EMA smoothing factor, st denotes the cosine similarity between frame t and t − 1, and sˆt is the EMA-smoothed similarity. We split the two frames when the sˆt is smaller than the threshold τseg.

Based on the above analysis, we propose EarlyTom, which consists of two core components: (1) an inner–vision encoder frame compression stage that improves prefill efficiency with minimal overhead, and (2) a decoupled spatial token selection stage that provides additional token compression without introducing bias into the visual features.

Middle frame merge. We adopt a local optimal strategy for the middle frames (i.e., frames within a segment excluding the first and last frames). Two frames are merged if and only if (1) their similarity is higher than a predefined threshold and (2) this similarity is larger than that between the next pair of frames. This process is defined as:

#### 3.2. Inner Vision Encoder Frame Compression

As analyzed in Section 3.1, compressing redundant frames within the vision encoder, which is in the early prefill stage, is crucial for further enhancing model efficiency and performance. Based on this observation, we propose an inner vision encoder frame merge strategy.

si > τmerge si > si+1

, (2)

merge(Fi,Fi+1) iff

where si is the similarity between Fi and Fi+1, and τmerge is the merging threshold. This merging strategy ensures that only the most similar frames are merged, helping remove redundancy while keeping temporal consistency.

Streaming frame segmentation. Given an input video, we perform frame merging at several selected layers in the vi-

CosineSimilarity

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| |[Figure 26]<br><br>|
|---|---|
| | |
| | |
| | |
| | |

Density

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Frame Index

Token Hidden State

(a) Frames Compression (b) Distribution of Tokens

- Figure 5. Frames compression and distribution of features. (a) Illustrates the cosine similarity changes across different frame indices for network layers at indices 6 and 20 during frame compression in the vision encoder. (b) The distribution of raw tokens, top-K sampling, and our method. This subfigure shows that our method is closer to vanilla top-K selection.

Weighted frame merge. To further improve the quality of merged representations, we use a weighted merging scheme as illustrated in the equation below:

siFi + si+1Fi+1 si + si+1

Fˆ =

, (3)

where Fi and Fi+1 are the frame features and si, si+1 are their corresponding similarity scores. Each pair of frames is weighted by its similarity with the following frame. This weighting makes the merged frame representation more concentrated around semantically important content and reduces ambiguity caused by uneven temporal variation.

#### 3.3. Decoupled Spatial Token Selection

In video feature tokens, we observe that certain vision sink tokens, as illustrated in Figure 2, consistently appear across all frames, receive high attention scores, and occupy the same positions along the sequence length. Existing methods, such as FastVID [34] and HoliTom [32], employ Top-K sampling for spatial token merging, which may introduce inherent bias and cause significant distribution shifts across frames as shown in Figure 5. To address this issue, we propose a decoupled sampling strategy that divides all frames into dynamic and static parts and applies distinct sampling schemes for each. Moreover, we adopt a system co-design approach to further enhance efficiency.

Decoupling frames into dynamic and static. After merging frames in the vision encoder, we first divide the merged video frames Fˆ ∈ RN×L×D into a dynamic part Fˆd ∈ RT×L×D and a static part Fˆs ∈ R(N−T)×L×D. The division strategy is similar to the streaming segmentation described in Section 3.2: we designate the head and tail frames within each segment as dynamic frames, while treating the middle frames as static frames, as we empirically observe that head and tail frames possess the highest discriminative

power per segment. Next, we independently compress the dynamic and static frames using their respective strategies.

Global top-K selection. For each dynamic frame, we perform a global Top-K selection based on its per-token attention scores. This process is defined as:

Fˆid = Fˆid[Ii,:], Ii = TopK(Ai,rˆ), i ∈ [1,T], (4)

where Ai denotes the per-token attention scores of frame Fi, Ii represents the indices of the selected tokens, and rˆ is the re-scaled selection ratio used to achieve the predefined compression rate, incorporated with stage 1, defined as:

r (B−BN ) ∗ L

, (5)

rˆ =

where B is the number of initial frames (e.g., 32 for LLaVAOneVision). By performing global importance-based compression, this process further improves the compression ratio while preserving the most motion-sensitive tokens across the entire temporal dimension.

Local window top-K selection. For static frames, our goal is to compress them while preserving their original distribution as much as possible, thereby avoiding unnecessary bias introduced by sink tokens. To this end, we apply a localwindow Top-K selection strategy to the static frames. We first divide them into M local windows of equal size:

L w

L rˆ

. (6)

{W1,W2,...,Wm}, M =

, w =

Within each window Wi, we select the token with the maximum attention score, finally, we observe compressed static

frames Fˆs. With this technique, the compressed static frames exhibit a distribution that is closer to the original one, thereby mitigating the negative effects caused by the bias introduced by vision sinks.

For all dynamic frames Fˆd and static frames Fˆs, we concatenate them according to their initial order:

Fˆ = Gather(Fˆd,Fˆs), (7) which serves as the input for LLM decoding.

System co-design. To further improve execution efficiency, we offload part of the static token selection to the CPU. We empirically observe that dynamic token selection is more time-consuming due to its larger candidate set. As described in Section 3.1, all frames are first divided into similarity-based segments; accordingly, we perform segment-wise static token selection on the CPU, while the GPU determines which dynamic tokens should be preserved. With this CPU–GPU heterogeneous computation, we further leverage otherwise idle CPU computational capacity, thereby increasing processing speed while maintaining overall cost-efficiency.

- Table 1. Performance and accuracy comparison with SoTA methods across benchmarks. Best results are in bold, second-best results are underlined. Time-to-first-token is denoted as TTFT for simplicity. All efficiency results are measured on a single NVIDIA A100 GPU.

EgoSchema ↑

LongVideo Bench ↑

Prefilling FLOPs (T) ↓

Throughput (tokens/s) ↑

Avg. ↑

FLOPs Ratio ↓

TTFT (ms) ↓

VideoMME ↑

MVBench ↑

Before LLM Retained Ratio

Method

Score %

LLaVA-OV-7B 100% 82.6 100% 889.9 24.4 58.3 60.4 56.4 58.6 58.4 100 FastV [4]ECCV’24 100% 51.1 61.9% 820.0 28.4 55.9 57.5 56.7 56.1 56.5 96.7 PyramidDrop [40]CVPR’25 100% 51.8 62.7% 813.4 28.3 56.1 58.0 54.1 56.4 56.2 96.2 DyCoke [35]CVPR’25 25% 50.5 61.1% 905.6 21.1 53.1 59.5 49.5 54.3 54.1 92.6 VisionZip [43]CVPR’25 25% 50.5 61.1% 516.6 29.4 57.9 60.3 56.5 58.2 58.2 99.7 PruneVid [14]ACL’25 25% 50.5 61.1% 703.6 29.4 57.4 59.9 55.7 57.4 57.6 98.6 FastVID [34]NeurIPS’25 25% 50.5 61.1% 581.6 26.9 56.5 58.2 56.3 58.0 57.3 98.1 HoliTom [32]NeurIPS’25 25% 49.0 59.3% 661.3 29.9 58.4 61.2 56.7 58.9 58.8 100.7

EarlyTom 25% 36.5 44.2% 426.3 32.9 57.4 60.5 56.3 58.5 58.2 99.7 VisionZip [43]CVPR’25 20% 48.7 58.9% 495.0 29.8 57.7 59.8 55.2 57.9 57.7 98.8 PruneVid [14]ACL’25 20% 49.0 59.3% 662.1 29.5 57.2 59.7 54.7 56.9 57.1 97.8 FastVID [34]NeurIPS’25 20% 48.7 58.9% 546.6 27.6 56.3 57.9 57.1 57.9 57.3 98.1 HoliTom [32]NeurIPS’25 20% 47.5 57.5% 622.3 30.0 58.7 61.0 57.1 58.6 58.8 100.7

EarlyTom 20% 35.1 42.4% 415.3 33.4 57.8 60.6 55.6 58.0 58.1 99.3 VisionZip [43]CVPR’25 15% 46.9 56.8% 475.9 32.1 56.5 59.8 54.4 56.1 56.7 97.1 PruneVid [14]ACL’25 15% 47.5 57.5% 574.1 27.1 56.8 59.7 55.4 56.6 57.1 97.8 FastVID [34]NeurIPS’25 15% 46.9 56.8% 530.8 28.7 56.0 57.4 56.2 57.7 56.8 97.3 HoliTom [32]NeurIPS’25 15% 46.0 55.7% 572.7 27.5 58.1 61.2 56.4 57.3 58.2 99.7 EarlyTom 15% 33.6 40.7% 390.6 30.4 57.5 60.2 54.4 56.9 57.3 98.1 VisionZip [43]CVPR’25 10% 45.2 54.7% 458.5 28.5 53.5 58.0 49.3 53.4 53.5 91.6 PruneVid [14]ACL’25 10% 45.9 55.6% 592.2 28.6 56.2 59.8 54.5 56.0 56.6 96.9 FastVID [34]NeurIPS’25 10% 45.2 54.7% 502.1 28.3 55.9 56.5 56.3 57.3 56.5 96.7 HoliTom [32]NeurIPS’25 10% 44.6 54.0% 556.6 29.0 57.3 61.2 56.3 56.8 57.9 99.1 EarlyTom 10% 32.2 39.0% 336.2 31.6 56.5 60.1 52.4 55.8 56.2 96.2

### 4. Experiments

#### 4.1. Settings

Benchmarks and metrics. In our paper, we choose four mainstream video understanding tasks for our evaluation: MVBench [20], EgoSchema [30], LongVideoBench [37], and VideoMME [10]. The videos in these tasks vary in length and scenario difficulty, providing a comprehensive perspective for evaluating the effectiveness and generalization of our method. To evaluate the efficiency of our approach, we report time-to-first-token (TTFT), throughput, and TFLOPs. These metrics capture both the latency and compute efficiency of our method, highlighting its practical benefits for large-scale or long-form video processing.

State-of-the-art methods. To evaluate the performance of our method, we compare our method with some mainstream token compression methods in Video-LLMs, i.e., FastV [4], PyramidDrop [40], DyCoke [35], VisionZip [43], FastVid [34], PruneVid [14], and HoliTom [32]. For their accuracy results, we report results from HoliTom.

Implementations. Our method is implemented based on the LLaVA-OneVision-0.5B/7B model [19]. We incorporate the inner-LLM merging technique from HoliTom [32] into our framework and develop a custom Triton kernel to ensure computational efficiency. All experiments are

conducted on NVIDIA A100 and RTX 4090 GPUs. The reported time-to-first-token (TTFT) is measured using the NVIDIA Nsight Systems profiler. For throughput evaluation, we report the average result of ten inference runs after warm-up. The prefilling FLOPs are computed following the HoliTom [32] benchmark protocol, which consists of both vision encoding and LLM prefilling FLOPs. In accordance with the official LLaVA-OneVision configuration, 32 video frames are uniformly sampled as visual inputs, and the vision encoder employs a pretrained SigLIP model [45]. Detailed configurations for hyperparameter selection are provided in Table 8 in the Appendix. All benchmark evaluations are performed using the LMMsEval framework [18, 48].

FLOPs and throughput. In our paper, we evaluate inference performance using FLOPs and throughput. Since both the vision encoder and the LLM decoder are built on Transformer architectures, the computation of FLOPs follows the same formulation. The computational cost mainly comes from the multi-head self-attention (MHA) and the feed-forward network (FFN). Following previous works [4, 32, 35, 40], the FLOPs for processing Li vision tokens at layer i with hidden size D and FFN intermediate size M, can be expressed as 4LiD2 + 2L2iD + 2LiDM. HoliTom [32] reports that only about 2% of FLOPs occur

- Table 2. Cross-backbone comparison on performance and accuracy. Best results are in bold, second-best results are underlined. Timeto-first-token is denoted as TTFT for simplicity. All efficiency results are measured on a single NVIDIA A100 GPU.

Prefilling FLOPs (T) ↓

Throughput (tokens/s) ↑

EgoSchema ↑

LongVideo Bench ↑

Avg. ↑

FLOPs Ratio ↓

TTFT (ms) ↓

MVBench ↑

VideoMME ↑

Before LLM Retained Ratio

Method

Score % LLaVA-OV-0.5B 100% 45.3 100% 413.7 42.7 45.5 26.8 45.8 43.7 40.5 100 FastVID [34]NeurIPS’25 25% 42.4 93.6% 409.9 25.9 44.7 25.3 44.9 42.1 39.3 97.0 VisionZip [43]CVPR’25 25% 42.4 93.6% 368.6 41.1 45.6 27.7 45.9 42.9 40.5 100.0 HoliTom [32]NeurIPS’25 25% 42.3 93.4% 519.4 35.2 45.8 27.6 46.2 44.4 41.0 101.2 EarlyTom 25% 29.9 66.0% 331.5 47.8 45.5 27.4 46.3 43.4 40.7 100.4

FastVID [34]NeurIPS’25 20% 42.3 92.4% 412.6 28.8 43.8 25.7 44.3 41.6 38.9 96.0 VisionZip [43]CVPR’25 20% 42.3 93.4% 368.5 42.3 45.1 27.5 44.8 42.7 40.0 98.8 HoliTom [32]NeurIPS’25 20% 42.2 93.2% 499.4 38.3 45.5 27.7 45.9 44.1 40.8 100.7

EarlyTom 20% 29.8 65.8% 313.1 40.6 45.2 27.5 44.7 43.7 40.3 99.5 FastVID [34]NeurIPS’25 15% 42.1 92.9% 411.3 29.4 43.1 25.3 44.7 40.7 38.5 95.1 VisionZip [43]CVPR’25 15% 42.1 92.9% 367.1 37.8 44.6 26.9 44.9 42.3 39.7 98.0 HoliTom [32]NeurIPS’25 15% 42.1 92.9% 473.9 34.1 45.4 27.6 46.4 43.4 40.7 100.4

EarlyTom 15% 29.7 65.6% 311.1 35.1 44.8 27.0 44.9 42.3 39.8 98.3 FastVID [34]NeurIPS’25 10% 42.0 92.7% 408.5 31.9 42.7 24.7 44.2 40.7 38.1 94.1 VisionZip [43]CVPR’25 10% 42.0 92.7% 366.1 38.7 43.2 25.8 42.6 40.0 37.9 93.6 HoliTom [32]NeurIPS’25 10% 42.0 92.7% 457.1 39.6 45.0 27.3 44.5 43.3 40.0 98.8 EarlyTom 10% 29.6 65.3% 280.1 43.9 44.3 26.8 44.5 41.8 39.4 97.3

during the decoding stage, and the majority of the computation lies in the prefilling (encoder) stage. However, different from HoliTom, we evaluate not only the LLM decoder but also the effectiveness and efficiency of the vision encoder. Therefore, the FLOPs of the whole inference pipeline are computed according to Equation (8):

Tv

FLOPs =

i=1

Tt

+

i=1

4LiD2 + 2L2iD + 2LiDM Vision Encoder FLOPs per layer

4LiD2 + 2L2iD + 2LiDM LLM Decoder FLOPs per layer

.

(8)

Compared with some outer-LLM token compression methods, performing token compression early within the vision encoder reduces the number of tokens entering the LLM, thereby significantly decreasing FLOPs and improving inference efficiency. For throughput evaluation, we use the same video input for all methods and measure the total runtime r. The throughput is reported as the average generated tokens per second over ten runs (with two warm-up passes): Throughput = Avg( ri=1 tokenstime ).

#### 4.2. Main Results

Performance comparison with state-of-the-art methods. Table 1 presents a comprehensive comparison of EarlyTom against a range of state-of-the-art training-free token compression methods, focusing on FLOPs, TTFT, and throughput. As shown in Table 1, prior methods such as PyramidDrop [40], VisionZip [43], PruneVid [14], and FastVID [34] significantly reduce the FLOPs of the prefill stage. How-

ever, these approaches largely rely on late-stage compression and operate after vision encoding, leaving the vision encoder as a dominant bottleneck. As a result, although their retained-token ratios fall to as low as 10–25%, the corresponding TTFT still ranges from 458 ms to 661 ms, and the throughput fluctuates between 27.5 and 32.1 tokens/s. In contrast, EarlyTom fundamentally shifts the compression point to an early stage inside the vision encoder, thereby optimizing one of the most expensive portions of TTFT. Consequently, EarlyTom achieves the lowest TTFT among all training-free approaches, only 336.2 ms with a 10% retained-token ratio, outperforming all other compared methods by a clear margin.

Meanwhile, EarlyTom maintains a FLOPs budget of 36.5T under a retention ratio of 25%, achieving significantly higher efficiency than the full-token baseline (82.6T) and other token compression methods. The results indicate that EarlyTom not only reduces the computational burden but also fundamentally improves system-level efficiency by co-optimizing both vision-encoding and LLM-prefill costs. Even under more aggressive compression ratios, EarlyTom maintains low TTFT and high throughput simultaneously, outperforming methods such as VisionZip [43], PruneVid [14], and FastVID [34]. This consistent dominance across multiple retention configurations highlights the superiority of EarlyTom in optimizing early-stage token compression and its ability to deliver balanced improvements in both latency and system efficiency. Overall, EarlyTom sets a new benchmark for inference efficiency in video LLMs, significantly outperforming all existing training-free methods in FLOPs, TTFT, and throughput.

- Table 3. Frame merging effectiveness varies across different initial compression layers. We report with a compression ratio of 0.2.

#Layer TTFT ↓ Throughput ↑ MVBench ↑ VideoMME ↑ EgoSchema ↑ Avg. ↑

Layer 4 380.0 31.6 57.4 57.9 60.4 58.6 Layer 6 387.1 32.3 57.8 58.1 60.4 58.9 Layer 8 421.1 30.7 57.5 58.0 60.4 58.6 Layer 10 436.9 31.1 57.4 58.0 60.6 58.7

- Table 4. Ablation study of different token sampling ways. We report the throughput and accuracy of three video tasks. In all results, we set the retain ratio to 0.2.

Sampling Throughput ↑ MVBench ↑ VideoMME ↑ EgoSchema ↑ Avg. ↑

Random 35.3 57.0 56.6 59.8 57.8 Top-K 31.5 57.5 57.3 60.4 58.4 EarlyTom 33.4 57.8 58.1 60.6 58.8

Accuracy comparison with state-of-the-art methods. Although EarlyTom is designed for improving efficiency, it also maintains accuracy comparable to a full-token baseline across multiple benchmarks. Table 1 shows results across four widely used video understanding benchmarks. Under all configurations, EarlyTom achieves an average accuracy of more than 96% compared with the full-token baseline, which is competitive with other training-free state-of-theart methods. Meanwhile, EarlyTom achieves this accuracy while reducing TTFT by up to 2.65×, demonstrating that the substantial efficiency gains do not come at the cost of model performance. In more challenging compression scenarios such as the 15% and 10% settings, other methods often show noticeable degradation in benchmark performance. For instance, VisionZip [43] suffers a noticeable accuracy drop under aggressive pruning, whereas EarlyTom maintains stable performance, with only a 4% decrease compared to the full-token output, while VisionZip [43] drops by nearly 9%. This indicates that EarlyTom preserves relevant features more effectively than late-stage pruning strategies. In summary, EarlyTom achieves near-baseline accuracy while significantly outperforming all prior approaches in computational efficiency, proving its practical value for real-world, latency-sensitive deployments.

Comparison across different backbones. To evaluate the robustness and generality of EarlyTom, we apply EarlyTom to a smaller backbone, LLaVA-OV-0.5B, and report results in Table 2. Similar to the observations on the 7B model, EarlyTom achieves substantial TTFT and FLOPs reductions and throughput improvements across all compression settings, while maintaining benchmark accuracy within a narrow margin of the full-token baseline across the four benchmarks, demonstrating that early-stage compression generalizes well even under lightweight vision-encoder architectures. This robustness is further evidenced across different retained-token settings: EarlyTom yields consistent improvements in efficiency with stable accuracy regardless of the size of the backbone. These results confirm that Ear-

Table 5. Ablation study on the compression module of our method. The stage-1 retention ratio is averaged due to its sample-dependent behavior, as the redundancy strongly depends on the input sample.

Retained

Method

Ratio MVBench ↑ VideoMME ↑ EgoSchema ↑ Avg. ↑ Vanilla 100% 58.3 58.6 60.4 59.1

- Only stage-1 73.9% 57.9 57.0 60.3 58.4

- Only stage-2 20% 57.3 57.6 60.4 58.4 EarlyTom 20% 57.8 58.1 60.6 58.8

lyTom is architecture-agnostic and capable of delivering strong acceleration without sacrificing quality.

#### 4.3. Ablation Studies

Contribution of compression modules. As shown in Table 5, the temporal merge achieves 98.8% of the baseline accuracy while retaining approximately 73.9% of the tokens. The spatial token selection module reaches the same accuracy with a retention rate of 20%. When both frame merge and spatial selection are jointly applied, our method further improves performance, achieving an accuracy of 58.8, surpassing either individual module.

Impact of different frame merging layers. Table 3 demonstrates that initiating the merging process from layer

- 4 yields the lowest TTFT, but results in a noticeable accuracy degradation. In contrast, starting from layer 6 achieves the best balance between accuracy and throughput. Since our frame merging primarily depends on the hidden states produced by the vision encoder, this also explains why throughput and TTFT do not scale proportionally.

Effectiveness of the proposed local window sampling. Table 4 shows that the top-K selection is slower than random sampling because it requires a global ranking over all tokens with complexity O(N log K), whereas random sampling only generates K indices with complexity O(K). As a result, top-K selection incurs extra computational and memory overhead, while random sampling cannot retain the most informative tokens. Our local window sampling combines the strengths of both methods, achieving a better trade-off between efficiency and accuracy.

- 5. Conclusion

In this paper, we propose EarlyTom, a training-free token compression framework for fast Video LLM inference. Benefiting from early-stage frame merging within the vision encoder and a further decoupled spatial token selection strategy, EarlyTom achieves up to a 2.65× reduction in TTFT and a 61% reduction in FLOPs, while keeping comparable accuracy to the full-token baseline. These results demonstrate the effectiveness and efficiency of EarlyTom, revealing its strong potential in video understanding tasks and laying a solid foundation for the deployment of Video LLMs in real-world production environments.

### Acknowledgement

This paper is supported by Young Scientists Fund of the National Natural Science Foundation of China (NSFC) (No. 62506305), Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (No. 2024R01007), Key Research and Development Program of Zhejiang Province (No. 2025C01026), Scientific Research Project of Westlake University (No. WU2025WF003). It is also supported by the research funds of the National Talent Program, Hangzhou Municipal Talent Program and Alibaba Innovative Research Program.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1
- [2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In ICLR, 2023. 2
- [3] Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, and Honggang Chen. Variation-aware vision token dropping for faster large vision-language models. arXiv preprint arXiv:2509.01552, 2025. 3
- [4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In ECCV, 2024. 2, 6
- [5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

2024. 1

- [6] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In ICLR,

2024. 4

- [7] Wenjie Du, Li Jiang, Keda Tao, Xue Liu, and Huan Wang. Which heads matter for reasoning? rl-guided kv cache compression. arXiv preprint arXiv:2510.08525, 2025. 3
- [8] Sicheng Feng, Gongfan Fang, Xinyin Ma, and Xinchao Wang. Efficient reasoning models: A survey. Transactions on Machine Learning Research, 2025. 3
- [9] Wenfeng Feng and Guoying Sun. Edit: Enhancing vision transformers by mitigating attention sink through an encoder-decoder architecture. In OCSA, 2026. 4
- [10] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025. 6
- [11] Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In ICLR, 2025. 4

- [12] Gongde Guo, Hui Wang, David Bell, Yaxin Bi, and Kieran Greer. Knn model-based approach in classification. In OTM,

2003. 2

- [13] Yuhang Han, Xuyang Liu, Zihan Zhang, Pengxiang Ding, Junjie Chen, Honggang Chen, Donglin Wang, Qingsen Yan, and Siteng Huang. Filter, correlate, compress: Training-free token reduction for mllm acceleration. In AAAI, 2026. 2
- [14] Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video large language models. In ACL, 2025. 6, 7
- [15] Xin Jin, Siyuan Li, Siyong Jian, Kai Yu, and Huan Wang. Mergemix: A unified augmentation paradigm for visual and multi-modal understanding. arXiv preprint arXiv:2510.23479, 2025. 2
- [16] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. In ICLR, 2025. 4
- [17] Zicheng Kong, Dehua Ma, Zhenbo Xu, Alven Yang, Yiwei Ru, Haoran Wang, Zixuan Zhou, Fuqing Bie, Liuyu Xiang, Huijia Wu, et al. Omni-rrm: Advancing omni reward modeling via automatic rubric-grounded preference synthesis. arXiv preprint arXiv:2602.00846, 2026. 3
- [18] Bo Li, Peiyuan Zhang, Kaichen Zhang, Fanyi Pu, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Accelerating the development of large multimoal models, 2024. 6
- [19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. TMLR, 2025. 1, 6
- [20] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024. 6
- [21] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. Science China Information Sciences, page 200102, 2025. 1
- [22] Kevin Y. Li, Sachin Goyal, Jo˜ao Dias Semedo, and J. Zico Kolter. Inference optimal vlms need fewer visual tokens and more parameters. In International Conference on Learning Representations, 2024. 3
- [23] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. IJCV, pages 1–19, 2025. 2
- [24] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In ECCV,

2024. 1

- [25] Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision token dropping: Towards efficient multimodal large language model. arXiv preprint arXiv:2411.10803, 2024. 2
- [26] Xuyang Liu, Xiyan Gui, Yuchao Zhang, and Linfeng Zhang. Mixing importance with diversity: Joint optimization for kv cache compression in large vision-language models. arXiv preprint arXiv:2510.20707, 2025. 3

- [27] Xuyang Liu, Yiyu Wang, Junpeng Ma, and Linfeng Zhang. Video compression commander: Plug-and-play inference acceleration for video large language models. In EMNLP, 2025.
- [28] Xuyang Liu, Ziming Wang, Junjie Chen, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Siteng Huang, and Honggang Chen. Global compression commander: Plugand-play inference acceleration for high-resolution large vision-language models. In AAAI, 2026. 3
- [29] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12585–12602, 2024. 1
- [30] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. In NeurIPS, 2023. 6
- [31] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In ICCV, 2025. 2
- [32] Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Holitom: Holistic token merging for fast video large language models. In NeurIPS, 2025. 2, 5, 6, 7
- [33] Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. arXiv preprint arXiv:2507.20198, 2025. 2
- [34] Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Sicheng Zhao, Guiguang Ding, et al. Fastvid: Dynamic density pruning for fast video large language models. In NeurIPS, 2025. 2, 5, 6, 7
- [35] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In CVPR, 2025. 2, 6
- [36] Chau Tran, Duy MH Nguyen, Manh-Duy Nguyen, TrungTin Nguyen, Ngan Le, Pengtao Xie, Daniel Sonntag, James Y Zou, Binh Nguyen, and Mathias Niepert. Accelerating transformers with spectrum-preserving token merging. In NeurIPS, 2024. 2
- [37] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. In NeurIPS, 2024. 6
- [38] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1
- [39] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR, 2025. 4
- [40] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In CVPR, 2025. 2, 6, 7

- [41] Minhao Xiong, Zichen Wen, Zhuangcheng Gu, Xuyang Liu, Rui Zhang, Hengrui Kang, Jiabing Yang, Junyuan Zhang, Weijia Li, Conghui He, et al. Prune2drive: A plug-andplay framework for accelerating vision-language models in autonomous driving. arXiv:2508.13305, 2025. 3
- [42] Chenyu Yang, Xuan Dong, Xizhou Zhu, Weijie Su, Jiahao Wang, Hao Tian, Zhe Chen, Wenhai Wang, Lewei Lu, and Jifeng Dai. Pvc: Progressive visual token compression for unified image and video processing in large vision-language models. In CVPR, 2025. 2
- [43] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In CVPR, 2025. 2, 6, 7, 8
- [44] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 1
- [45] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 4, 6
- [46] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 1
- [47] Ce Zhang, Kaixin Ma, Tianqing Fang, Wenhao Yu, Hongming Zhang, Zhisong Zhang, Haitao Mi, and Dong Yu. Vscan: Rethinking visual token reduction for efficient large vision-language models. TMLR, 2025. 2
- [48] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024. 6
- [49] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1
- [50] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In ICML, 2025. 2
- [51] Shiyu Zhao, Zhenting Wang, Felix Juefei-Xu, Xide Xia, Miao Liu, Xiaofang Wang, Mingfu Liang, Ning Zhang, Dimitris N Metaxas, and Licheng Yu. Accelerating multimodal large language models by searching optimal vision token reduction. In CVPR, 2025. 4
- [52] Junhan Zhu, Hesong Wang, Mingluo Su, Zefang Wang, and Huan Wang. Obs-diff: Accurate pruning for diffusion models in one-shot. arXiv preprint arXiv:2510.06751, 2025. 3
- [53] Jiedong Zhuang, Lu Lu, Ming Dai, Rui Hu, Jian Chen, Qiang Liu, and Haoji Hu. St3: Accelerating multimodal large language model by spatial-temporal visual token trimming. In AAAI, 2025. 4
- [54] Zayd MK Zuhri, Erland Hilman Fuadi, and Alham Fikri Aji. Softpick: No attention sink, no massive activations with rectified softmax. arXiv preprint arXiv:2504.20966, 2025. 4

[Figure 27]

## EarlyTom : Early Token Compression Completes Fast Video Understanding Supplementary Material

### Overview

Due to page limitations in the main paper, we present additional quantitative experiments, detailed latency analyses, qualitative visualizations, and implementation details in this supplementary material. The content is organized as follows:

- • Section A evaluates the generalizability of our method on a different video-LLM architecture. Specifically, we provide extensive efficiency and accuracy results on the LLaVA-Video-7B benchmark to verify the robustness of EarlyTom across different backbones. Furthermore, we extend our evaluation to the Qwen2.5-VL architecture, comparing EarlyTom against two native token reduction baselines. We also conduct fine-grained ablation studies to investigate the individual contribution of each component within our framework on Qwen architecture.
- • Section B presents a fine-grained decomposition of the time-to-first-token (TTFT) latency. We analyze the specific contributions of vision encoding, visual token processing, and LLM prefilling to the total latency on both LLaVA-OneVision-7B and 0.5B models across different settings.
- • Section C provides additional visualizations of the attention sink phenomenon. By visualizing attention heatmaps from the vision encoder, we further substantiate the motivation behind our decoupled spatial token selection strategy.
- • Section D details the implementation of our framework, providing the pseudocode for the two core components: the inner-vision encoder frame merging and the decoupled spatial token selection.
- • Section E presents the future work of EarlyTom, including potential directions for system co-design, heterogeneous inference optimization, and acceleration for the decoding stage in multimodal models.

### A. Generalizability Analysis on LLaVA-Video and Qwen2.5-VL

To further verify the effectiveness and broad applicability of our framework, we extend our evaluation to the LLaVA-Video-7B model and Qwen2.5-VL-7B model.

Efficiency analysis. As detailed in Table 6, EarlyTom consistently delivers substantial improvements in computational efficiency across all tested token retention settings. By performing frame merging directly within the vision encoder, our method effectively reduces the prefilling FLOPs.

For instance, at a 15% retention rate, EarlyTom reduces the FLOPs ratio to 35.1% and achieves a time-to-first-token of 947.4 ms, representing a 6.8× speedup compared to the full-token baseline (6429.3 ms). The efficiency advantages are also corroborated on the Qwen2.5-VL-7B backbone (Table 7). Specifically, while trivial baselines like Average Pooling and Uniform Subsampling result in a 16.6% FLOPs ratio, EarlyTom further optimizes this to 12.2% (67.7T), achieving a significantly faster TTFT (3667 ms) than both the full model and the native token reduction baselines.

Accuracy and trade-off. Table 6 presents a comprehensive comparison of accuracy and efficiency. EarlyTom maintains competitive performance on standard video understanding benchmarks, achieving an average score of 56.43% while operating with significantly reduced computational overhead. These results demonstrate that EarlyTom can successfully generalize to the LLaVA-Video architecture, providing an efficient inference solution that balances high throughput with reliable model performance. This robust generalizability is further evidenced by our results on the Qwen2.5-VL-7B backbone (Table 7). At a 15% token retention ratio, EarlyTom achieves an average score of 62.2%, which significantly outperforms the Uniform Subsampling and Average Pooling baselines. Notably, EarlyTom maintains higher accuracy than these trivial baselines while utilizing even fewer FLOPs, demonstrating a superior Pareto frontier in the accuracy-efficiency trade-off.

Ablation studies. To investigate the individual contribution of our proposed modules, we conduct fine-grained ablation studies on the Qwen2.5-VL architecture, as summarized in Table 7. We observe that both decoupled spatial token selection and weighted frame merging are essential for maintaining optimal performance under aggressive compression. Specifically, removing the spatial selection module leads to a performance drop from 62.2% to 61.4%, indicating its critical role in identifying and preserving informative regions across frames. Similarly, excluding the weighted merging strategy results in a decline to 61.3%, underscoring the importance of our importance-aware aggregation. These results confirm that the synergy between spatial selection and temporal merging is the key to EarlyTom’s ability to preserve high-fidelity visual information while reducing token redundancy.

- Table 6. Efficiency comparison with SoTA methods on the LLaVA-Video-7B model. Best results are in bold, second-best results are underlined. Time-to-first-token is denoted as TTFT for simplicity. All efficiency results are measured on a single NVIDIA A100 GPU.

Model Method

Before LLM Retained Ratio

Prefilling FLOPs (T) ↓

FLOPs Ratio ↓

TTFT (ms) ↓

Throughput (tokens/s) ↑

MVBench ↑

EgoSchema ↑

LongVideo Bench ↑

VideoMME ↑

Avg. ↑

Score %

LLaVAVideo-7B

Vanilla 100% 246.2 100% 6429.3 8.1 60.4 57.2 58.9 64.3 60.2 100 FastV [4]ECCV’24 100% 158.2 64.3% 3494.3 10.0 54.3 54.1 55.0 58.8 55.6 92.4 PyramidDrop [40]CVPR’25 100% 159.4 64.7% 3494.8 10.1 55.9 54.3 54.7 61.9 56.7 94.2 VisionZip [43]CVPR’25 15% 159.4 64.7% 3241.4 14.2 56.7 54.7 54.7 60.7 56.7 94.2 HoliTom [32]NeurIPS’25 15% 156.6 63.6% 1669.5 17.1 57.7 54.8 56.2 62.1 57.7 95.8 EarlyTom 15% 86.4 35.1% 947.4 16.7 55.8 54.7 53.9 61.3 56.4 93.7

- Table 7. Experiment results on trivial baselines and ablation studies. All results are obtained on Qwen2.5-VL-7B with a maximum of 768 frames and a retain ratio of 15%. Efficiency metrics are measured under a 23k-token context length on a single NVIDIA A100 GPU.

Method

Prefilling FLOPs (T) ↓

FLOPs Ratio ↓

TTFT (ms) ↓ MVBench ↑

VideoMME ↑ Avg. Short Medium Long Average Score

Qwen2.5-VL-7B 554.7 100% 6842 67.1 76.0 66.0 55.1 65.7 66.4 Average Pooling 91.9 16.6% 4609 56.8 66.4 57.3 51.1 58.3 57.6 Uniform Subsampling 91.9 16.6% 4578 57.7 68.6 59.6 55.0 60.8 59.3 EarlyTom w/o Decoupled Spatial Token Selection 67.7 12.2% 3667 60.7 71.0 61.6 53.6 62.0 61.4 EarlyTom w/o Weighted Frame Merging 67.7 12.2% 3667 60.7 70.5 62.3 52.7 61.8 61.3 EarlyTom 67.7 12.2% 3667 62.5 70.7 61.6 53.6 61.9 62.2

- Table 8. Details of the hyperparameters on LLaVA-OneVision.

sionZip effectively reduce the LLM prefill latency through token reduction, they fail to address the high computational cost of the vision encoder. Moreover, HoliTom introduces significant computational overhead during the Visual Token Processing stage, which partially offsets the gains from reduced prefill time. In contrast, EarlyTom directly compresses redundancy within the vision encoder, achieving a substantial reduction in encoding latency. Consequently, our method achieves the lowest total TTFT across all settings, delivering a speedup of up to 2.65× compared to the baseline at a 10% retention rate.

EgoSchema ↑

LongVideo Bench ↑

VideoMME ↑ LLaVA-OneVision-7B

MVBench ↑

w. Inner LLM

Retained Ratio

EMA factor α

τseg=0.8 τseg=0.7 τseg=0.8 τseg=0.6 [6,14,20] [10,21,23] [6,21,23] [10,21,23] 20% ✓ 0.9

25% ✓ 0.9

τseg=0.8 τseg=0.6 τseg=0.6 τseg=0.5 [6,14,20] [10,21,23] [10,21,23] [8,21,23] 15% ✓ 0.9

τseg=0.8 τseg=0.5 τseg=0.5 τseg=0.4 [6,14,20] [10,21,23] [10,21,23] [8,21,23] 10% ✓ 0.9

τseg=0.65 τseg=0.3 τseg=0.3 τseg=0.3

[8,14,20] [10,21,23] [10,21,23] [10,21,23] LLaVA-OneVision-0.5B

τseg=0.8 τseg=0.7 τseg=0.8 τseg=0.6 [8,21,23] [10,21,23] [6,21,23] [8,21,23] 20% ✓ 0.9

25% ✓ 0.9

Analysis on LLaVA-OneVision-0.5B. The advantages of our approach are consistent across model scales. Figure 8 presents the results on the smaller 0.5B backbone. A notable observation is that on this lightweight model, the computational overhead introduced by comparison methods becomes more detrimental. Specifically, HoliTom exhibits a higher total latency than the Baseline (e.g., 0.90× speedup at 10% retention) because the time saved in the LLM prefill stage is insufficient to outweigh the extra cost of its token processing module. Conversely, EarlyTom maintains its superiority by minimizing both vision encoding time and processing overhead. Even with the smaller potential for prefill acceleration in the 0.5B model, our method achieves a robust speedup of 1.48× (at 10% retention), validating the effectiveness of our early-stage compression strategy.

τseg=0.8 τseg=0.6 τseg=0.6 τseg=0.5 [8,21,23] [10,21,23] [10,21,23] [8,21,23] 15% ✓ 0.9

τseg=0.8 τseg=0.5 τseg=0.5 τseg=0.4 [8,21,23] [10,21,23] [10,21,23] [8,21,23] 10% ✓ 0.9

τseg=0.65 τseg=0.3 τseg=0.3 τseg=0.3

[8,21,23] [10,21,23] [10,21,23] [8,21,23]

### B. Detailed Analysis of TTFT Latency Decomposition

In this section, we provide a fine-grained visualization of the Time-to-First-Token (TTFT) latency composition for both LLaVA-OneVision-7B (Figure 7) and LLaVAOneVision-0.5B (Figure 8) under varying token retention rates (10%, 15%, 20%, and 25%). The total latency is decomposed into four components: Vision Encoding, Visual Token Processing, LLM Prefill, and System Overhead.

### C. Visualization of the Attention Sink Phenomenon Across Diverse Video Samples

Analysis on LLaVA-OneVision-7B. As illustrated in Figure 7, the vision encoding stage constitutes a dominant portion of the total latency for the Baseline, HoliTom, and VisionZip. While existing methods like HoliTom and Vi-

In this section, we provide additional visualizations to further substantiate the analysis of the “Attention Sink” phe-

nomenon discussed in the main paper. Figure 6 displays the attention heatmaps extracted from the SigLIP vision encoder across a diverse set of video samples.

Observation. A consistent pattern emerges across all examples: distinct vertical stripes appear in the heatmaps, indicating that certain spatial tokens maintain exceptionally high attention scores throughout the entire video sequence. These tokens, often referred to as “attention sinks,” act as static attractors within the feature space, dominating the attention distribution regardless of the changing visual content in dynamic frames.

Motivation for Our Design. This visualization highlights a critical insight for token compression: simply ranking tokens by attention magnitude might bias the selection towards these static sink tokens, potentially overlooking less prominent but semantically rich dynamic features. Recognizing this inherent distribution characteristic, EarlyTom adopts a Decoupled Spatial Token Selection strategy. By distinguishing between static frames (where sinks are stable) and dynamic frames, and applying tailored selection mechanisms for each, our method ensures that the compressed token set preserves both the necessary structural information (sinks) and the crucial motion-sensitive details, leading to a more robust and balanced video representation.

### D. Pseudocode of EarlyTom

In this section, we provide the detailed pseudocode for the two core components of EarlyTom to facilitate implementation. Algorithm 1 outlines the inner-vision encoder frame merging process, which performs adaptive streaming segmentation and weighted merging to reduce temporal redundancy. Algorithm 2 illustrates the decoupled spatial token selection strategy, describing how dynamic and static frames are processed via distinct selection mechanisms to ensure balanced spatial information preservation.

### E. Future Work

EarlyTom reveals that the inference budget is mainly dominated by the prefill stage in VLMs. Although existing methods [3, 7, 17, 26–28, 41, 52] have proposed various techniques for efficient inference, they primarily focus on algorithm-level improvements rather than system-level optimizations. How to jointly leverage system design and algorithmic techniques in a heterogeneous manner remains an open problem. Meanwhile, recent reasoning models [8] have exhibited strong scene understanding capabilities, yet they still suffer from lengthy generation steps during the decoding stage. Therefore, accelerating inference and improving efficiency via system–algorithm co-design is essential and worthy of further exploration.

- Algorithm 1 Inner-Vision Encoder Frame Merging

Input: Frame features F ∈ RB×L×D, hyperparameters α,τseg,τmerge. Output: Merged frame features Fˆout ∈ RN×L×D. Streaming Frame Segmentation in Equation (1) S ← SegmentBySimilarity(F,α,τseg), Fmerged list ← [] for each segment Sseg = {F0,...,Fk} in S do

Fmid ← [], i ← 1 Iterate over Middle Frames within the Segment while i < k do

Compute Pairwise Frame Similarities si ← Sim(Fi,Fi+1), si+1 ← Sim(Fi+1,Fi+2) Middle Frame Merge Condition in Equation (2) if si > τmerge and si > si+1 then

Weighted Frame Merge in Equation (3) Fˆm ← WeightedMerge(si,Fi,si+1,Fi+1) Fmid.append(Fˆm); i ← i + 2

else

Fmid.append(Fi); i ← i + 1

end if end while Assemble Merged Segment

Fseg out ← Concat(F0,Fmid,Fk) Fmerged list.append(Fseg out)

end for Concatenate All Merged Segments Fˆout ← Concatenate(Fmerged list) Return Fˆout

- Algorithm 2 Decoupled Spatial Token Selection

Input: Features Fˆ and attentions A from vision encoder, segment list S, target ratio r. Output: Final compressed features Fˆ. Decouple Frames into Dynamic and Static Sets

Fˆd,Ad ← [], [] Fˆs,As ← [], [] for each segment Sseg = {Fˆ0,...,Fˆk} in S do

Fˆd.append(Fˆ0,Fˆk); Ad.append(A0,Ak) Fˆs.append(Fˆ1:k−1); As.append(A1:k−1)

end for Compute Re-scaled Retention Ratio in Equation (5)

rˆ = ( r

B−N

B )∗L Compress Dynamic Frames via Global Top-K Selection Fˆd ← GlobalTopKSelection(Fˆd,Ad,rˆ) Compress Static Frames via Local-window Selection Fˆs ← LocalWindowSelection(Fˆs,As,rˆ) Gather and Reorder Selected Tokens in Temporal Order Fˆ ← GatherAndReorder(Fˆd,Fˆs) Return Fˆ

[Figure 28]

[Figure 29]

[Figure 30]

...

| |[Figure 31]|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

VideoFrames

Token id

[Figure 32]

[Figure 33]

[Figure 34]

...

|[Figure 35]|
|---|

VideoFrames

Token id

[Figure 36]

[Figure 37]

[Figure 38]

...

|[Figure 39]|
|---|

VideoFrames

Token id

[Figure 40]

[Figure 41]

[Figure 42]

...

|[Figure 43]|
|---|

VideoFrames

Token id

[Figure 44]

[Figure 45]

[Figure 46]

...

|[Figure 47]|
|---|

VideoFrames

Token id

[Figure 48]

[Figure 49]

[Figure 50]

...

|[Figure 51]|
|---|

VideoFrames

Token id

- Figure 6. Additional visualizations of attention score distributions. We present the attention heatmaps from the SigLIP vision encoder across six randomly selected videos. The consistent vertical stripes (highlighted in bright colors) indicate that specific spatial tokens accumulate disproportionately high attention scores throughout the temporal sequence. This observation confirms that attention “sinks” are a widely existing structural characteristic in the vision encoder, motivating the design of our decoupled token selection strategy.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |889 ms| | | | |
|---|---|---|---|---|---|
| |556 ms<br><br>458 ms<br><br>336 ms<br><br>1.60x<br><br>1.94x<br><br>2.65x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

800

600

Latency(ms)

400

200

0

Baseline HoliTom VisionZip Ours

(a) 10% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |889 ms| | | | |
|---|---|---|---|---|---|
| |622 ms<br><br>495 ms<br><br>415 ms<br><br>1.43x<br><br>1.80x<br><br>2.14x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

800

600

Latency(ms)

400

200

0

Baseline HoliTom VisionZip Ours

(c) 20% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |889 ms| | | | |
|---|---|---|---|---|---|
| |572 ms<br><br>475 ms<br><br>390 ms<br><br>1.55x<br><br>1.87x<br><br>2.28x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

800

600

Latency(ms)

400

200

0

Baseline HoliTom VisionZip Ours

(b) 15% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |889 ms| | | | |
|---|---|---|---|---|---|
| |661 ms<br><br>516 ms<br><br>426 ms<br><br>1.34x<br><br>1.72x<br><br>2.09x| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

800

600

Latency(ms)

400

200

0

Baseline HoliTom VisionZip Ours

(d) 25% token retention rate.

- Figure 7. Time-to-first-token (TTFT) comparison on the LLaVA-OneVision-7B model. We report the latency breakdown (vision encoding, token processing, LLM prefill, and system overhead) across different methods.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |413 ms<br><br>457 ms<br><br>0.90x| | | | |
|---|---|---|---|---|---|
| |366 ms<br><br>280 ms<br><br>1.13x<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

400

1.48x

300

Latency(ms)

200

100

0

Baseline HoliTom VisionZip Ours

(a) 10% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |413 ms<br><br>499 ms<br><br>0.83x| | | | |
|---|---|---|---|---|---|
| | | | | | |
| |368 ms<br><br>313 ms<br><br>1.12x<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

500

400

1.32x

Latency(ms)

300

200

100

0

Baseline HoliTom VisionZip Ours

(c) 20% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |413 ms<br><br>473 ms<br><br>0.87x| | | | |
|---|---|---|---|---|---|
| |367 ms<br><br>311 ms<br><br>1.13x<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

400

1.33x

300

Latency(ms)

200

100

0

Baseline HoliTom VisionZip Ours

(b) 15% token retention rate.

Vision Encoding

LLM Prefill

Visual Token Processing

System Overhead

| |413 ms<br><br>519 ms<br><br>0.80x| | | | |
|---|---|---|---|---|---|
| | | | | | |
| |368 ms<br><br>331 ms<br><br>1.12x<br><br>| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

500

400

1.25x

Latency(ms)

300

200

100

0

Baseline HoliTom VisionZip Ours

(d) 25% token retention rate.

- Figure 8. Time-to-first-token (TTFT) comparison on the LLaVA-OneVision-0.5B model. We report the latency breakdown (vision encoding, token processing, LLM prefill, and system overhead) across different methods.

