# arXiv:2506.15745v2[eess.IV]24Oct2025

## InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding

##### Minsoo Kim1† Kyuhong Shim2 Jungwook Choi1‡ Simyung Chang3‡

1Hanyang University 2Sungkyunkwan University 3Qualcomm AI Research, Qualcomm Korea YH§

{minsoo2333, choij}@hanyang.ac.kr khshim@skku.edu simychan@qti.qualcomm.com

### Abstract

Modern multimodal large language models (MLLMs) can reason over hour-long video, yet their key–value (KV) cache grows linearly with time—quickly exceeding the fixed memory of phones, AR glasses, and edge robots. Prior compression schemes either assume the whole video and user query are available offline or must first build the full cache, so memory still scales with stream length. InfiniPot-V is the first training-free, query-agnostic framework that enforces a hard, lengthindependent memory cap for streaming video understanding. During video encoding it monitors the cache and, once a user-set threshold is reached, runs a lightweight compression pass that (i) removes temporally redundant tokens via Temporal-axis Redundancy (TaR) metric and (ii) keeps semantically significant tokens via Value-Norm (VaN) ranking. Across four open-source MLLMs and four long-video and streaming-video benchmarks, InfiniPot-V cuts peak GPU memory by up to 94%, sustains real-time generation, and matches or surpasses full-cache accuracy—even in multi-turn dialogues. By dissolving the KV cache bottleneck without retraining or query knowledge, InfiniPot-V closes the gap for on-device streaming video assistants.

### 1 Introduction

Recent advances in multimodal large language models (MLLMs) have dramatically expanded the scope of visual reasoning. Vision–language instruction tuning now allows a single backbone to answer open-ended questions over long video sequences [27, 30, 48], while context-extension techniques such as FlashAttention-2 and RingAttention push the effective window into the million-token regime [7, 26, 33]. These breakthroughs underpin a new generation of streaming video assistants and humanoid robots that promise continuous, real-time scene understanding on mobile phones, AR glasses and edge robots [14, 31, 42, 37].

Streaming video understanding (SVU) diverges from conventional offline video understanding (OVU). Offline models see the entire clip and user query before inference, so they can tailor every compression or retrieval step. In streaming, frames arrive incrementally and future queries are unknown, forcing all pre-query processing to be query-agnostic. In addition, device memory is fixed, yet the transformer emits hundreds of tokens per frame, so the key–value (KV) cache grows linearly. For example, a 15-min, 10 fps clip processed by LLaVA-Next-Video-7B already needs demands ∼100GB of KV storage, far beyond the tens of gigabytes available on mobile or robotic platforms [52, 19].

†Work done during an internship at Qualcomm Technologies, Inc. ‡Corresponding authors. §Qualcomm AI Research, an initiative of Qualcomm Technologies, Inc.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Prior work tackles long-video memory constraints at three stages (Fig.1). Frame Sampling [10] drops frames before encoding, reducing memory but severely degrading temporal coverage and accuracy. Input-Vision Compression (IVC) [36, 40] prunes redundant vision tokens after encoding, lowering Prefill load but still requiring the full vision token set to be stored in memory. KV cache Compression (KVC) [24, 12] selects query-relevant tokens after the Prefill step, offering the highest accuracy but only after materializing the full KV cache. The challenge intensifies in streaming scenarios: memory usage for Frame Sampling, IVC, and KVC grows almost linearly with video length, eventually exceeding device limits. KV cache offloading (e.g., ReKV [35]) expands memory space yet incurs costly data transfer, repeated for each query. Thus, no existing approach delivers the key property SVU needs: a length-independent and query-agnostic streaming video compression.

A natural approach to address memory constraints in streaming video is to exploit the strong spatiotemporal redundancy of video streams. We introduce InfiniPot-V, the first framework specifically designed for memory-constrained SVU. InfiniPot-V is training-free, query-agnostic, and operates continuously during inference. When the KV cache reaches a user-defined memory threshold M, it performs an in-place compression that frees space for new frames while preserving the semantic essence of prior context. This compression is guided by two lightweight and complementary metrics. Temporal-axis Redundancy (TaR) models Key embeddings as a 3D tensor over time and removes tokens with high cosine similarity to recent frames, effectively pruning static or repetitive content. Value-Norm Importance (VaN) ranks the remaining tokens by the ℓ2 norm of their Value vectors—a strong, model-agnostic proxy for semantic salience—and applies a layer-adaptive pooling strategy. This compression is highly efficient, adding negligible latency while strictly enforcing memory limits.

Extensive evaluation confirms the effectiveness of this design. Across four open-source MLLMs (3B and 7B) and six long-video benchmarks—covering both offline (VideoMME, EgoSchema, MLVU, LongVideoBench) and streaming (RVS-Ego/Movie, OVO-Bench, StreamingBench) tasks—InfiniPotV reduces input context length usage to as low as 6K for 50K-token contexts, with accuracy matching or exceeding full-cache baselines. It maintains real-time performance at 14 frames per second with only 0.5% compression overhead. Additionally, its query-agnostic nature offers clear benefits in multiturn dialogue settings (Appendix. C). By eliminating the KV cache bottleneck without retraining or query dependency, InfiniPot-V paves the way for practical, on-device multimodal assistants.

### 2 Background

We aim to deploy streaming video understanding (SVU) applications [49, 35] in memory-constrained environments. Unlike offline video understanding (OVU) [54, 11, 45], which assumes access to the entire video, SVU must process arbitrarily long video streams and answer questions at any time step using only the frames observed up to that point. Given a video stream VT := [v1,v2,...,vT] with T frames and a set of questions Q = q1,q2,...,qN, SVU answers each question qi at time t (1 ≤ t ≤ T) using only the observed frames Vt := [v1,v2,...,vt].

As SVU deals with unbounded video streams, memory-efficient processing is essential. In this section, we describe how multimodal large language models (MLLMs) handle long videos, review prior approaches to memory reduction in OVU, and analyze their limitations when applied to SVU. (See Appendix. F for a detailed discussion of related work.)

##### 2.1 Preliminary: Offline Long Video Understanding

Video Processing in MLLMs. Multimodal Large Language Models (MLLMs) [52, 48, 43] process offline videos through a structured pipeline (Fig. 1(a)). Given a video VT := [v1,v2,...,vT] of T uniformly sampled frames, a vision encoder fViT transforms each frame into visual tokens:

X = fViT(VT) = [x1,x2,...,xN] ∈ RN×D, (1) where N = P × T denotes the total number of sampled tokens, where P is the number of tokens per frame (determined by input resolution and ViT patch size), and D is the token embedding dimension.

The token sequence X is then passed to the LLM in two phases: Prefill and Decoding. During the Prefill phase (Fig. 1(a), step 2), all tokens are processed at once to construct the initial KV cache. The attention operation computes:

QK⊤ √

Q = XWq, K = XWk, V = XWv, Oattn = Softmax

+ M V, (2)

D

###### ★: Full KV

[Figure 1]

Offline Video

Offline Video Vision Tokens

Offline Video

- 1. Vision Encoding
- 2. Prefill (LLM)
- 3. Decoding (LLM)

Vision Tokens

Vision Tokens

KV Cache

KV Cache

(a) Offline Video Understanding

(b) Input Vision Compression (IVC)

(c) KV Cache Compression (KVC)

(e) OVU Accuracy / Memory with Video Compression

[Figure 2]

Stream Video

KV Cache-1

KV Cache-2

Constrained KV Cache-3 Memory

[Figure 3]

...

(d) Continual KV Cache Compression (CKV) for Streaming Video Understanding

(f) SVU Memory Consumption with # Frames

- Figure 1: MLLMs Video Understanding and Compression. (a) OVU pipeline; (b) IVC: compresses vision tokens after encoding; (c) KVC: compresses KV cache after prefill; (d) CKV: iteratively processes and compresses KV caches to constrain memory usage; (e) Accuracy vs. GPU memory consumption for compression across four token reduction ratios (50%, 25%, 12.5%, 6.25%) on MLVU using Qwen-2-VL-7B. LongVU[36] is used for IVC, SnapKV[24] for KVC; (f) GPU memory usage as input video stream length increases. IVC/KVC/CKV target a 6K cache; Sampling uses 1/4 of input frames. Measured on A100 80GB single GPU.

where Wq,Wk,Wv ∈ RD×D are projection matrices and M is a causal mask enforcing autoregressive decoding.

In the Decoding phase (Fig. 1(a), step 3), the model generates tokens one at a time using cached keys and values from the prefill phase. To avoid redundant computation, the KV cache C = (K,V ) is updated incrementally:

Ct+1 = ({K,kt+1},{V,vt+1}), (3) where kt+1,vt+1 correspond to the KV embeddings of the newly processed token.

##### 2.2 Offline Long-Video Compression Strategies

Long videos produce extremely long token sequences X, leading to prohibitive GPU memory and latency during decoding. Prior works tackle this bottleneck in the offline setting through three classes of methods (Fig. 1a–c):

- (1) Frame Sampling [10]. Uniformly sampling a shorter clip VT′′ ⊆VT reduces the input length and, hence, memory usage is also reduced proportional to compression rate.
- (2) Input-Vision Compression (IVC) [36, 40]. After vision encoding, IVC aggressively prunes redundant vision tokens, keeping only a salient subset X′⊆X (Fig. 1b) to shrink the context fed into the language decoder for memory-compressed Prefill.
- (3) KV cache Compression (KVC) [24, 12, 4]. Conduct compression after prefill: KVC computes

importance scores ut= Ni=N−wAttn(xi→xt) over the last w tokens and retain top-M entries for the memory budget M by applying eviction policy π, yielding a compressed cache C′ = π(C) for memory-compressed Decoding. (Fig. 1c). Note that the π eviction policy is highly dependent on the content of the last w tokens, reflecting the user query, and is thus referred to as query-dependent cache compression method (see Appendix. D for further analysis).

These techniques are effective when the entire video is available upfront, but they implicitly assume (i) unconstrained memory for compression and (ii) a known or easily approximated query.

##### 2.3 Challenges in Streaming Video Understanding

Fig. 1(e) compares three offline compression methods on a fixed 50 K-token video at four compression ratios (darker shades indicating higher ratios: 50%, 25%, 12.5%, 6.25%), revealing a fundamental trade-off between memory usage and accuracy. Frame sampling skips frames to save memory, but severely degrades recognition accuracy. Increasing the sample ratio improves accuracy but quickly inflates memory usage. IVC starts with a large memory footprint for all vision tokens before selecting which to discard. KVC, which operates on more expressive key–value features, achieves the highest accuracy but requires the largest Prefill cache. Notably, even under a favorable offline setting—with full video access and an offline query—none of the methods achieve both high accuracy and low memory usage.

This trade-off becomes more severe in the streaming video understanding (SVU) setting. As shown in Fig.1(f), peak GPU memory usage increases with stream length. KVC exhibits near-linear memory growth, as it must materialize all vision tokens and build the full KV cache before compression. Furthermore, due to its query-dependent nature, KVC must re-execute the memory-intensive prefill stage whenever the user query changes. Frame sampling and IVC also grow linearly, albeit more slowly, eventually exceeding the memory capacity of practical edge devices (e.g., 32GB[19]) as the stream continues. ReKV [35], a recent KVC method, addresses this by offloading the KV cache to CPU memory, but this introduces substantial offloading overhead and compression latency.

These findings highlight two core requirements for SVU: (1) a fixed memory budget that does not grow with stream length, and (2) query-agnostic token retention strategies. Existing methods fail to meet at least one of these, limiting their suitability for SVU. To overcome this, we propose Continual KV cache compression (CKV), illustrated in Fig. 1(d). CKV processes frames in small blocks and compresses the cache whenever the fixed memory limit is reached, ensuring constant memory usage throughout streaming. Additionally, for query-agnostic token retention, our approach employs lightweight spatiotemporal metrics to identify and preserve semantically significant tokens without relying on future queries. As a result, despite operating under strict memory constraints, CKV achieves accuracy on par with or better than KVC (Fig.1(e)), while consuming far less memory than IVC or frame sampling (Fig. 1(f)). The algorithmic details are described in Sec.3.

Algorithm 1 Continual KV cache Compression (CKV) with InfiniPot-V Require: Memory budget |M|, target cache size |C|, TaR ratio α

Initialize K, V ← ∅ ▷ Empty KV cache while video stream continues do

(1) Process: Knew, Vnew ← Process new frame; K ← [K; Knew], V ← [V ; Vnew] ▷ Append new tokens if len(K) ≥ |M| then ▷ Memory budget exceeded

- (2) Extract: Krecent, Vrecent ← recent r frames from K, V ▷ len(K) = len(V ) = |M|
- (3) TaR: sTaR ← ComputeTaRScores(K); ITaR ← TopK(sTaR, α|C| − len(Krecent)) ▷ Sec. 3.1
- (4) VaN: sVaN ← ComputeAdaptiveVaNScores(V ); IVaN ← TopK(sVaN, (1 − α)|C|) ▷ Sec. 3.2
- (5) Combine: I ← ITaR ∪ IVaN ∪ Indices(Krecent); K ← K[I], V ← V [I] ▷ Compress to |C| size

end if if user query arrives then

Generate response using current K, V end if

end while

- 3 InfiniPot-V: Memory-Constrained Streaming Video Understanding We present InfiniPot-V, a CKV framework designed for memory-constrained SVU. As shown in

- Fig. 1(d) and Algorithm 1, InfiniPot-V processes video streams by applying continual KV cache compression within a fixed memory budget. In this framework, KV embeddings from incoming frames are stored until the memory limit |M| is reached. At that point, compression reduces the cache to a smaller target size |C| (|M| ≫ |C|), retaining only the most essential vision tokens based on two criteria. The freed space (|M| − |C|) accommodates new frames. This process repeats continuously, enabling efficient stream processing under strict memory constraints. When a user query is issued, the model answers using the compressed cache that summarizes visual context from all prior frames. Notably, compression adds only 0.5% overhead relative to input frames processing time.

𝑓 frames

Recent Frame

Past Frame Past Frame

[Figure 4]

Constrained KV Cache (|𝑀|)

Vision Token

[Figure 5]

[Figure 6]

...

[Figure 7]

[Figure 8]

3D Reshaped KV Cache Compression

𝑝patches

[Figure 9]

| |
|---|

| |
|---|

Static Patch (Evict)

Moving Patch (Retain)

[Figure 10]

Time (𝑓 frames)

(a) Temporal-axis Redundancy Reduction (TaR)

[Figure 11]

| | | | | |Key Value| |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

[Figure 12]

TaR (temporal) selected

Similarity

Height

0.5

VaN (spatial)

[Figure 13]

selected Recent

frames (𝑟)

0.0

Width

0 5 10 15 20 25

Layer Index

(b) Static Patch Similarity Comparison

(c) InfiniPot-V (Ours): Temporal-Spatial Scoring

- Figure 2: Spatio-Temporal KV cache Compression (TaR and VaN). (a) Temporal redundancy across adjacent frames, showing static patches that can be evicted from past frames; (b) Layerwise cosine similarity of Key/Value embeddings for static patches between consecutive frames in LLaVA-Next-Video-7B; (c) InfiniPot-V performs query-agnostic spatiotemporal compression, reducing temporal redundancy with TaR and selecting tokens via VaN spatial scoring.

InfiniPot-V leverages two token eviction criteria: Temporal-axis Redundancy (TaR) and Value Norm (VaN) for identifying crucial tokens for compressing KV cache. In the following subsections, we detail each criterion, and finally describe how to effectively combine them.

##### 3.1 Temporal-axis Redundancy (TaR) Reduction via Patch-wise Similarity

Video streams exhibit inherent spatiotemporal redundancy across frames [44, 36, 40]. In this section, we focus on exploiting temporal redundancy, as illustrated in Fig. 2(a) where static patches5 (e.g., background) persist across frames. For MLLMs processing videos with fixed memory usage, identifying this redundancy in KV caches is crucial. Our analysis in Fig. 2(b) reveals that Key embeddings effectively capture temporal redundancy, exhibiting higher cosine similarity for static patches between adjacent frames compared to Value embeddings, across all layers.

Building on this insight, we propose TaR, a technique that performs a patch-wise comparison of Key embeddings along the temporal axis to detect and reduce redundant tokens. As shown in Fig. 2(c), we introduce a 3D reshaping of Key embeddings to enable direct comparison of corresponding patches across frames. Based on this structured KV cache, the TaR implementation starts with a memory constraint of |M| tokens, processing f consecutive video frames, each containing p = |M|/f vision tokens. To maintain temporal continuity, we designate the r latest frames as recent frames and retain them in full. The older past frames (f − r frames) are selectively compressed based on their patch-wise similarity to recent frames.

To measure the patch-wise similarity between frames, we divide the current Key embeddings K ∈ RH×(f×p)×D into Krecent ∈ RH×r×p×D and Kpast ∈ RH×(f−r)×p×D, representing the recent and past frames respectively. For each spatial coordinate (i,j), we compute the ℓ2-normalized cosine similarity between recent and past frames of the same patch coordinate:

r

1 r

′,i,j)

sTaR(t,i,j) = −

cos Kpast(t,i,j),K(t

recent . (4)

t′=1

Here, sTaR(t,i,j) is the importance score of the patch in t-th frame at (i,j) coordinate. The negative sign is applied so that a higher computed score indicates lower redundancy (i.e., the token is more distinctive). This ensures that tokens with less temporal similarity to recent frames are prioritized.

5In MLLMs, each vision patch corresponds to a single token, so we use these terms interchangeably.

We then select the least redundant tokens (i.e., higher score) in past frames using the Top-K operator: ITaR = TopK(sTaR,|C| − |Krecent|), (5)

where |C| is the target cache compression size and |Krecent| = rp accounts for the recent frame tokens that are always retained. The compressed key-value pairs are formed by concatenating the selected key frame tokens with all recent frame tokens:

K˜TaR = Concat K[:,ITaR,:],Krecent , V˜TaR = Concat V [:,ITaR,:],Vrecent . (6) By fully preserving the most recent frames, we maintain complete information on rapidly changing or newly introduced content, while selectively retaining distinctive visual elements from the past.

##### 3.2 Spatial Semantic Importance Preserving with Value Norm (VaN)

While TaR focuses on reducing temporal redundancy, VaN serves a complementary role: identifying and preserving semantically salient regions within each video frame, independent of the query. To achieve this, we employ Value embeddings (V ), which inherently capture semantic information in transformer attention [41]. Specifically, we introduce Value Norm (VaN) as a metric for token-level semantic importance: sVaN = ∥V (t,i,j)∥2.

Analysis of Value Norm. We hypothesize that tokens with higher VaN contain richer semantic information, making them more valuable for video understanding. To quantify semantic importance, we project vision token representations from each layer into the vocabulary space [29] and compute the entropy of the resulting word probability distribution, where the higher entropy implies greater informativeness [9, 3]. As shown in Fig. 3 (a), tokens with higher VaN consistently exhibit higher entropy, confirming their semantic significance. This advantage translates to improved performance: Fig. 3 (b) shows that retaining high-VaN tokens achieves substantially higher video understanding accuracy across various compression ratios compared to low-VaN (VaN Reverse) tokens.

15

Entropy

10

5

High-VaN

Low-VaN

0

0 5 10 15 20 25 Layer Index

(a) Entropy Analysis of Vision Tokens

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | |V V<br><br>|aN aN Rev|erse| |
| | | | | | |

7.5

Center-Dist.

Accuracy(%)

60

Locality()

CV

0.2

5.0

55

2.5

0.1

0.0

50

1 1/2 1/4 1/8 Compression Ratio

0 10 20 Layer Index

(b) VideoMME Accuracy

(c) Spatial Locality Analysis

Figure 3: Value Norm (VaN) Analysis. (a) Entropy analysis of vision token representations grouped by their VaN scores. (b) VideoMME performance under varying cache compression ratios using either VaN or reverse-VaN for token selection. (c) Layer-wise locality of VaN, measured by center distance and coefficient of variation (CV); lower values indicate stronger spatial consistency. LLaVA-Next-7B with Video-MME used.

Layer-wise Adaptive Pooling. An analysis of VaN distributions reveals strong spatial locality patterns in early to middle layers, which gradually diminish in deeper layers as shown in Fig. 3(c). To measure spatial locality patterns across layers, we employ two methods: (1) compute the average distance between the center point and surrounding points within a 3 × 3 window spanning the VaN values of each frame (center-dist.), and (2) measure the Coefficient of Variance (CV) to quantify dispersion of VaN distributions. Lower values in both metrics—smaller center-dist. and CV—indicate that VaN scores are closely clustered, implying high spatial locality, whereas higher values reflect greater dispersion and lower locality.

As shown in Fig. 3 (c), both metrics consistently indicate strong locality in early to middle layers, while gradually diminishing in deeper layers. Based on this observation, we design an adaptive spatial pooling mechanism that adjusts the average pooling kernel size per layer. To implement this, we design a mapping function g that assigns kernel sizes in inverse relation to each layer’s CV:

PoolSize(CVl) = g(CVl) where g : R+ → 1,3,5,7 This approach assigns larger pooling kernels (e.g., 7) to lower layers with smaller CV values (higher spatial locality), and smaller kernels (e.g., 1, implying no pooling) to upper layers with larger CV values, thus preserving fine-grained details where needed.

Method Size # Frames Budget EgoSchema MLVU VideoMME LVB Max Duration |M| 3min 120min 60min 60min GPT4-V* – 1fps – 55.6 – 60.7 – GPT4-o* – 1fps – 72.2 66.2 77.2 66.7

LLaVA-OV* 7B 32 8K 60.1 64.7 58.2 – LongVU* 7B 1fps 8K 67.6 65.4 60.6 – LongVU* 3B 1fps 8K 59.1 55.9 51.5 –

Qwen-2-VL 7B 768 50K 65.2 65.8 63.9 58.8 Qwen-2-VL + Ours 7B 768 6K 65.6 65.8 62.8 58.4 LLaVA-Next 7B 128 25K 67.6 68.7 62.8 63.5 LLaVA-Next + Ours 7B 128 6K 65.8 65.2 61.1 60.9 Qwen-2.5-VL 3B 768 50K 64.4 63.3 60.3 59.9 Qwen-2.5-VL + Ours 3B 768 6K 61.8 62.1 59.3 56.5

- Table 1: Comparison of various MLLMs accuracy on four Offline Video Understanding (OVU) benchmarks. * denotes the numbers from official paper.

Compression Budget Video MME MLVU Method |M| Short Med Long Holistic Single Multi Avg. FullKV 50K 74.7 62.1 55.0 76.3 73.9 43.3 64.2 TTC [40] 3K 66.8 51.2 47.9 72.1 58.8 33.2 54.8

- (IVC) 6K 72.6 55.0 51.7 76.3 60.9 36.7 58.4 STC [36] 3K 67.9 51.0 49.3 71.5 58.6 33.9 55.0

- (IVC) 6K 72.6 56.2 51.6 74.3 61.1 35.9 57.9

InfiniPot-V 3K 73.9 57.8 51.8 77.7 70.4 43.2 63.1 (CKV) 6K 74.1 60.8 53.4 77.2 72.3 44.8 64.3

- Table 2: Comparison under memory-constrained settings (3, 6K memory-budget) with Input Video Compression (IVC) methods: TTC from DyCoke [40] and STC from LongVU [36]. Qwen-2-VL-7B used across VIdeoMME and MLVU benchmarks. Results comparing memory-unconstrained IVC methods (without KV cache compression) with InfiniPot-V are provided in Tab. A6.

For KV cache compression, we select tokens using VaN scores processed through our adaptive pooling mechanism, retaining the Top-|C| tokens with highest pooled VaN values as described in

- Fig. 2(c): IVaN = TopK(VaNpool,|C|) K˜VaN = K[:,IVaN,:], V˜VaN = V [:,IVaN,:]. (7)

##### 3.3 Design Space Exploration

Combining TaR and VaN for Token Selection. TaR and VaN capture complementary aspects of spatio-temporal redundancy in streaming video. To integrate them, we prioritize TaR-based selection by first allocating α|C| tokens to TaR, then filling the remaining (1−α)|C| with VaN-selected tokens. This two-stage selection strategy effectively balances temporal and feature importance. A detailed hyperparameter exploration, including sweeps over α and the size of the recent frame window r, is provided in Appendix. A.2.

Comparison with Memory-Constrained Alternatives. A natural question is whether IVC or KVC can be adapted for SVU under memory constraints. To explore this, we apply query-agnostic methods such as spatial token compression (STC) and token temporal merging (TTC) from LongVU [36] and DyCoke [40]. InfiniPot-V outperforms all these baselines by a notable accuracy margin, demonstrating the strength of continual compression over expressive key-value embeddings (details in Tab. 2).

LLaVA-Next-7B - VideoMME

LLaVA-Next-7B - MLVU_dev

LLaVA-Next-7B - LVB_dev

64

62

67

62

60

65

Accuracy(%)

60

57

62

58

55

60

Uniform Select

56

###### SnapKV InfiniPot InfiniPot-V (Ours)

52

57

54

50

55

52

52

47

1/16 1/8 1/4 1/2 1

1/16 1/8 1/4 1/2 1

1/16 1/8 1/4 1/2 1

Qwen-2-VL-7B - VideoMME

Qwen-2-VL-7B - MLVU_dev

Qwen-2-VL-7B - LVB_dev

64

66

- 54

- 55

- 56

- 57

- 58

- 59

64

62

Accuracy(%)

62

60

Uniform Select

60

###### SnapKV InfiniPot InfiniPot-V (Ours)

58

58

56

56

1/16 1/8 1/4 1/2 1

1/16 1/8 1/4 1/2 1

1/16 1/8 1/4 1/2 1

Compression Ratio

Compression Ratio

Compression Ratio

Figure 4: KV cache Compression (KVC) methods evaluation results with offline long video understanding tasks under Continual KV Cache Compression (CKV) framework. Performance across four compression ratios (1/16, 1/8, 1/4, 1/2) for LLaVA-Next-7B (top row) and Qwen-2-VL-7B (bottom row) on VideoMME, MLVUdev, and LongVideoBenchdev (LVBdev) tasks. The full evaluation results are shown in Table A5.

### 4 Experiments

##### 4.1 Experimental Setup

Benchmarks. We evaluate our InfiniPot-V on both offline video understanding (OVU) and streaming video understanding (SVU) tasks. For OVU, we utilize representative multiple-choice based long video understanding benchmarks (ranging from 3 minutes to over 2 hours): VideoMME [11], MLVU [54], LongVideoBench (LVB) [45], and EgoSchema [28].

For SVU, we employ RVS-Ego/Movie streaming video QA benchmark [49], featuring open-ended questions paired with timestamps, and evaluate the answers using GPT-3.5-turbo-0125 following [49, 35]. We further extend multiple-choice based SVU evaluation, OVO-Bench [23] and StreamingBench [25]. OVO-Bench evaluates temporal reasoning under three scenarios—backward tracing, real-time visual perception, and forward active responding—while StreamingBench evaluates real-time visual comprehension of streaming videos.

Models. We apply our method on four state-of-the-art MLLMs capable of long-video understanding: Qwen-2-VL [43], Qwen-2.5-VL [48], LLaVA-OV-7B [22], and LLaVA-Next-Video [52]. Details on input video sampling settings and benchmark details are provided in Appendix. B.

##### 4.2 Evaluatal Results

Offline Video Understanding. To assess the absolute compression capability of our method, we evaluate InfiniPot-V with both commercial MLLMs (GPT-4V [30], GPT-4o [31]) and state-ofthe-art public models designed for offline video understanding, including LLaVA-OV [22], and LongVU [36]. Unlike these specialized, fully trained models, InfiniPot-V is a training-free, plugin framework compatible with MLLMs of various scales, enabling high performance under fixed memory budgets. As shown in Tab. 1, InfiniPot-V reduces memory usage to just 25% (6K tokens) for LLaVA-Next (originally 25K tokens) and 12.5% for Qwen-VL series (6K vs. 50K), with minimal performance loss. Notably, it achieves comparable or better accuracy than LongVU at the 7B scale and significantly outperforms it at 3B, demonstrating both efficiency and scalability.

Comparison with IVC under Memory Constraints. To evaluate recent query-agnostic IVC methods under memory-constrained CKV, we adopt a unified setup on VideoMME and MLVU: token temporal merging (TTC) from DyCoke [40] and spatial token compression (STC) from LongVU [36] are applied to compress vision tokens to fit the target memory budget |M|, while KV cache is

RVS-Ego RVS-Movie Execution Time Total Memory Usage LLaVA-OV-7B Acc Score Acc Score Video Enc. (msec/Frame) GPU CPU ReKV 60.1 3.9 53.4 3.8 285.7 37.5 GB + 18.8GB/h

ReKV w/o off. 55.8 3.3 50.8 3.4 74.6 27.2 GB 0 InfiniPot-V 57.9 3.5 51.4 3.5 76.3 27.8 GB 0

- Table 3: Streaming benchmark comparison to offloading-based KV cache control method. (ReKV) Video Enc. shows execution time per frame, GPU indicates peak memory usage, and CPU denotes the size of video KV-Cache offloaded to CPU per hour. Results based on an 1-hour video processed with a 0.5 fps sampling rate in streaming mode. LLaVA-OV-7B is used.

Qwen-2.5-VL Scale OVO-BW OVO-Real OVO-FW OVO-Avg StreamingBench

Uniform Select 7B 44.5 61.1 47.8 51.7 75.2 InfiniPot-V 7B 47.6 65.9 47.9 53.6 76.4

- Table 4: Comparison on OVO-Bench (BW: Backward, Real: Realtime, FW: Forward) and StreamingBench (Real-time visual understanding) using Qwen-2.5-VL-7B under 4K memory budget.

managed using a sliding window attention (SWA) [2]. When operated under such constraints, these

- IVC methods suffer from notable accuracy degradation. In contrast, InfiniPot-V performs KV cache compression using TaR and VaN, leveraging expressive key-value representations to achieve superior average accuracy under a 6K memory budget—corresponding to an 88% lossless compression rate.

Comparison with KVC under Memory Constraints. Fig.4 evaluates KVC methods within our CKV framework under constrained memory across offline video understanding tasks. Compression ratios (1/16, 1/8, 1/4, 1/2) are defined based on each model’s maximum frame capacity (e.g., 128 frames for LLaVA-Next, 768 for Qwen-2-VL). Our InfiniPot-V consistently outperforms all baseline methods (Uniform Select, SnapKV, InfiniPot) across all tasks for both LLaVA-Next-7B and Qwen-2VL-7B, demonstrating superior video understanding performance. Under CKV constraints—where actual query access is not available—query-dependent methods like SnapKV[24] degrade significantly. In contrast, InfiniPot-V maintains strong accuracy even at high compression ratios (e.g., 1/16), thanks to its query-agnostic selection via TaR and VaN.

Streaming Video Understanding. We first evaluate InfiniPot-V on streaming video understanding (SVU) using two StreamingVQA benchmarks, RVS-Ego and RVS-Movie, with LLaVA-OV-7B. As a baseline, we compare against ReKV [35], a state-of-the-art SVU method, under two configurations: (1) CPU–GPU with CPU offloading, which allows spilling KV cache to CPU memory, and (2) CPU–GPU without offloading, simulating shared-memory devices where CPU memory is unavailable or pre-occupied [19]. Tab. 3 reports SVU accuracy, compression time, and memory usage. While CPU offloading enables ReKV to retain the full KV cache, it incurs substantial transfer overhead; without offloading, ReKV suffers from severe accuracy degradation due to limited local memory. In contrast, InfiniPot-V operates entirely within GPU memory, eliminating offloading latency and achieving higher accuracy, making it a practical solution for memory-constrained systems.

To further evaluate accuracy in streaming scenarios, we extend the analysis to two additional benchmarks: OVO-Bench [23] and StreamingBench [25], as summarized in Tab. 4. Compared to the uniform token selection baseline, InfiniPot-V demonstrates consistent gains across all metrics, notably improving recall of past visual information on the OVO-BW (backward) task (44.5 → 47.6) and achieving higher real-time understanding accuracy on the OVO-Real (Realtime) (61.1 → 65.9) and StreamingBench score (75.2 → 76.4), demonstrating the effectiveness of our TaR–VaN compression in eliminating temporal redundancy while preserving spatially salient semantics, showing its capability under challenging SVU settings.

##### 4.3 Ablation Study

Tab. 5 validates our design decisions for TaR and VaN. Reversed strategies (TaR Reverse and VaN Reverse) significantly degrade performance by discarding distinctive or semantically important tokens. Within TaR, patch-wise similarity proves more effective than frame-level similarity (64.5 vs. 62.9).

MLVUdev Holistic Reasoning Single Detail Multi Detail Ablation Study Topic Anomaly Plot Needle Ego AO AC Avg Full KV 85.2 67.5 72.7 83.9 65.1 54.1 32.5 65.9 Uniform Select 83.7 66.5 67.9 76.1 58.5 51.0 27.2 61.5 TaR Reverse 79.0 64.5 56.9 65.6 55.1 45.2 21.8 55.5 TaR Frame 82.9 66.0 67.0 78.9 63.6 51.0 31.1 62.9 TaR 85.9 66.5 71.8 78.0 62.2 51.7 35.4 64.5 VaN Reverse 78.3 66.5 56.2 66.8 53.4 46.3 17.5 55.0 VaN 84.4 68.0 68.6 76.6 61.9 52.5 29.1 63.0 VaN + Pool 85.2 68.0 71.4 77.5 63.1 52.1 31.5 64.1 TaR + VaN + Pool 86.3 68.0 72.7 80.3 63.9 54.1 35.4 65.8

- Table 5: Ablation study of TaR, VaN, and their combination. Experiments conducted on MLVU using Qwen-2-VL-7B with a 6K memory budget.

Method Length Context VRAM FPS Throughput Power

(sec) Length (GB) ↓ (frame/sec) ↑ (tok/sec) ↑ (J) ↓ Full KV 100 18K 13.8 6.2 5.0 1.6

- InfiniPot-V 9.2 (1.5×) 6.4 (1.0×) 9.2 (1.8×) 1.2 (1.4×) Full KV 300 54K 26.6 5.0 2.0 4.7

- InfiniPot-V 10.1 (2.6×) 6.4 (1.3×) 9.2 (4.6×) 2.4 (2.0×) Full KV 500 90K 39.0 4.1 1.2 7.5

- InfiniPot-V 10.7 (3.6×) 6.3 (1.5×) 9.1 (7.3×) 3.7 (2.0×) Full KV 600 108K OOM OOM OOM OOM

- InfiniPot-V 11.3 6.4 9.2 4.3

- Table 6: Streaming video processing performance results on NVIDIA Jetson AGX Orin. Streaming 10-minute video samples (FPS 0.2–1) processed with Qwen-2.5-VL-3B, measuring memory, FPS (frames processed per second), throughput, and power consumption.

VaN alone surpasses the baseline, and its performance improves further with adaptive pooling (64.1 vs. 63.0). Combining TaR and VaN yields the highest accuracy, significantly outperforming the baseline. Additional integration explorations are discussed in Appendix. A.2.

##### 4.4 Edge Device Deployment Results

We evaluate our CKV framework—combining continual KV compression with TaR and VaN scoring—on the NVIDIA Jetson AGX Orin [19] using Qwen-2.5-VL-3B and 10-minute StreamingBench [25] videos (0.2–1 FPS). As shown in Tab. 6, our method achieves consistent efficiency across all metrics. Peak memory remains nearly constant (9.2–10.7GB) while Full KV grows linearly (13.8→39.0GB), yielding a 3.6× reduction at 500 seconds streaming video. Prefill speed, measured in FPS (frames processed per second during prefill), stays stable at 6.3–6.4FPS, whereas Full KV drops to 4.1FPS. Generation throughput also increases up to 7.3× (1.2→9.1tok/sec) under a fixed memory budget. Power usage scales proportionally with compute, improving energy efficiency by nearly 2× (7.5→3.7J). Notably, CKV enables continuous inference on 600-second streams where Full KV fails with OOM, confirming its practicality for real-time multimodal reasoning on memory-constrained edge devices.

### 5 Conclusion

In this paper, we proposed InfiniPot-V, a training-free KV cache control framework for streaming video processing in memory-constrained environments. Built around practical constraints—unavailable queries and strict memory budgets during compression—InfiniPot-V employs two novel token eviction criteria, TaR and VaN, achieving significant improvements in long video understanding under streaming scenarios.

### References

- [1] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S. Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models, 2024.
- [2] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv:2004.05150, 2020.
- [3] David Chan, Rodolfo Corona, Joonyong Park, Cheol Jun Cho, Yutong Bai, and Trevor Darrell. Analyzing the language of visual tokens, 2025.
- [4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models, 2024.
- [5] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. LongVILA: Scaling long-context visual language models for long videos. In The Thirteenth International Conference on Learning Representations, 2025.
- [6] Giulio Corallo, Orion Weller, Fabio Petroni, and Paolo Papotti. Beyond rag: Task-aware kv cache compression for comprehensive knowledge reasoning, 2025.
- [7] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.
- [8] Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. A simple and effective l_2 norm-based strategy for KV cache compression. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18476–18499, November 2024.
- [9] Sebastian Farquhar, Jannik Kossen, Livia Kuhn, et al. Detecting hallucinations in large language models using semantic entropy. Nature, 630:625–630, 2024.
- [10] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition, 2019.
- [11] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [12] Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning. In The Thirteenth International Conference on Learning Representations, 2025.
- [13] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive KV cache compression for LLMs. In The Twelfth International Conference on Learning Representations, 2024.
- [14] Google DeepMind. Project ASTRA. https://deepmind.google/technologies/ project-astra/, 2024.
- [15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina González, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jáchym Koláˇr, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano,

- Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbeláez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, June 2022.
- [16] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Monishwaran Maheswaran, June Paik, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. Squeezed attention: Accelerating long context length llm inference. arXiv preprint arXiv:2411.09688, 2024.
- [17] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020.
- [18] Yuxiang Huang, Binhang Yuan, Xu Han, Chaojun Xiao, and Zhiyuan Liu. Locret: Enhancing eviction in long-context llm inference with trained retaining heads on consumer-grade devices, 2025.
- [19] Leela S. Karumbunathan. NVIDIA Jetson AGX Orin Series Technical Brief. Technical Report TB_10749-001_v1.2, NVIDIA Corporation, July 2022. Version 1.2.
- [20] Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, and Hyun Oh Song. Kvzip: Query-agnostic kv cache compression with context reconstruction, 2025.
- [21] Minsoo Kim, Kyuhong Shim, Jungwook Choi, and Simyung Chang. InfiniPot: Infinite context processing on memory-constrained LLMs. In Yaser Al-Onaizan, Mohit Bansal, and YunNung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16046–16060, Miami, Florida, USA, November 2024. Association for Computational Linguistics.
- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [23] Yifei Li, Junbo Niu, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, and Jiaqi Wang. Ovo-bench: How far is your video-llms from real-world online video understanding?, 2025.
- [24] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [25] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024.
- [26] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In The Twelfth International Conference on Learning Representations, 2024.
- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc., 2023.
- [28] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

- [29] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting visual information processing in vision-language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [30] OpenAI. Gpt-4v(ision) system card, 2023.
- [31] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.
- [32] Junyoung Park, Dalton Jones, Matthew J Morse, Raghavv Goel, Mingu Lee, and Chris Lott. Keydiff: Key similarity-based kv cache eviction for long-context llm inference in resourceconstrained environments, 2025.
- [33] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations, 2024.
- [34] Sundar Pichai, Demis Hassabis, and Koray Kavukcuoglu. Introducing gemini 2.0: our new ai model for the agentic era. https://blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024, 2024.
- [35] Zhelun Yu Shangzhe Di. Streaming video question-answering with in-context video KV-cache retrieval. In The Thirteenth International Conference on Learning Representations, 2025.
- [36] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding, 2024.
- [37] Morgan Stanley. Humanoids: Investment implications of embodied ai. Technical report, Morgan Stanley, June 2024. Accessed via Future Management Group: https://www.futuremanagementgroup.com/wp-content/uploads/240626-HumanoidRobots-Morgan-Stanley.pdf.
- [38] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.
- [39] Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Shikuan Hong, Danning Ke, Yiwu Yao, and Gongyi Wang. Razorattention: Efficient KV cache compression through retrieval heads. In The Thirteenth International Conference on Learning Representations, 2025.
- [40] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models, 2024.
- [41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
- [42] Ethan Waisberg, Joshua Ong, Mouayad Masalkhi, Nasif Zaman, Prithul Sarker, Andrew G Lee, and Alireza Tavakkoli. Meta smart glasses—large language models and the future for assistive glasses for individuals with vision impairments. Eye, 38(6):1036–1038, 2024.
- [43] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [44] T. Wiegand, G.J. Sullivan, G. Bjontegaard, and A. Luthra. Overview of the h.264/avc video coding standard. IEEE Transactions on Circuits and Systems for Video Technology, 13(7):560– 576, 2003.

- [45] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for longcontext interleaved video-language understanding. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.
- [46] Mingze Xu, Mingfei Gao, Shiyu Li, Jiasen Lu, Zhe Gan, Zhengfeng Lai, Meng Cao, Kai Kang, Yinfei Yang, and Afshin Dehghan. Slowfast-llava-1.5: A family of token-efficient video large language models for long-form video understanding, 2025.
- [47] Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu, Aojun Zhou, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Think: Thinner key cache by query-driven pruning. In The Thirteenth International Conference on Learning Representations, 2025.
- [48] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [49] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memory-based real-time understanding for long video streams, 2024.
- [50] Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster, 2024.
- [51] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. Sparsevlm: Visual token sparsification for efficient vision-language model inference, 2024.
- [52] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024.
- [53] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [54] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

Algorithm 2 InfiniPot-V Algorithm

Require: Video stream V , memory constraint |M|, target KV cache size |C|, recent frame count r, CV thresholds {τ1, τ2, τ3}, TaR ratio α ∈ [0, 1], f frames corresponding to |M| tokens, vision token number per single frame p = |M|/f

Ensure: Compressed KV cache {K˜l, V˜l}Ll=1

- 1: Let CTaR = α|C| be the TaR selection budget
- 2: Let CVaN = (1 − α)|C| be the VaN selection budget
- 3: Initialize empty KV cache for each layer l ∈ {1, . . . , L}
- 4: while processing video stream V do
- 5: Accumulate KV embeddings until reaching |M|
- 6: for each layer l do
- 7: // Temporal-axis Redundancy (TaR)
- 8: Reshape Kl into Krecent,l ∈ RH×r×p×D and Kpast,l ∈ RH×(f−r)×p×D
- 9: for each patch (t, i, j) in past frames do
- 10: s(t, i, j) = −r1 rt′=1 cos(Kpast(t,i,j,l ), K(t

′,i,j) recent,l )

- 11: end for
- 12: Il ← TopK(Sl, CTaR) ▷ Select least redundant tokens
- 13: // Value Norm (VaN) with Adaptive Pooling
- 14: VaNl ← ∥Vl∥2
- 15: // Compute CV for adaptive pooling
- 16: µl ← mean(VaNl)
- 17: σl ← std(VaNl)
- 18: CVl ← σl/µl
- 19: // Determine pooling size using mapping function g
- 20: pool_sizel ← g(CVl) ▷ Using thresholds {τ1, τ2, τ3}
- 21: where g(CV) =

 



7, if CV < τ1 5, if τ1 ≤ CV < τ2 3, if τ2 ≤ CV < τ3 1, if CV ≥ τ3

- 22: VaNpool,l ← AveragePool2d(VaNl, pool_sizel)
- 23: // Combine TaR and VaN by prioritizing TaR-selected tokens
- 24: VaNpool,l[Il] ← max(VaNpool,l) ▷ Prioritize TaR tokens
- 25: Jl ← TopK(VaNpool,l, |C|) ▷ Final token selection
- 26: K˜l ← Kl[:, Jl, :], V˜l ← Vl[:, Jl, :] ▷ Update layer KV cache with compressed KV cache
- 27: end for
- 28: end while

### A InfiniPot-V Algorithm and Configuration

##### A.1 Algorithm Description

Algorithm 2 presents the complete process of InfiniPot-V’s cache control framework along with its compression formulation. InfiniPot-V processes video streams by continuously pre-filling and compressing the KV cache using two token selection strategies: Temporal-axis Redundancy (TaR) and Value Norm (VaN). For TaR, the algorithm splits video frames into recent frames (the latest r frames) and past frames, then computes cosine similarities between corresponding patches to identify and remove redundant visual tokens. (Line 10)

For spatial semantic importance token selection, a layer-wise adaptive pooling mechanism based on VaN is employed. The pooling size is dynamically determined by the Coefficient of Variation (CV) of the VaN, (Line 18) where a higher CV indicates a sparser or more distinct feature distribution. Precomputed model-specific CV thresholds {τ1,τ2,τ3} determine pooling sizes from the set {1,3,5,7}, selecting larger windows for uniform (low CV) VaN distributions and smaller ones for sparse (high

- CV) VaN distributions (Line 21).

To integrate both criteria, TaR-selected tokens are prioritized by assigning them the maximum VaN score before the final token selection. Specifically, by setting VaNpool,l[Il] = max(VaNpool,l) (Line 24) and then applying a TopK selection, the algorithm ensures that temporally distinctive tokens are preserved while allowing VaN to select additional tokens based on semantic importance.

VideoMME |M| = α|TaR| + (1 − α)|VaN|, |M| = 6K Qwen-2-VL-7B α = 0 (VaN) α = 0.2 α = 0.4 α = 0.6 α = 0.8 α = 1 (TaR)

Short (-3 min) 74.4 74.3 74.1 74.9 74.4 73.8 Medium (3 - 30 min) 59.9 59.3 61.3 61.4 61.2 58.2

- Long (30 - 120 min) 51.9 51.0 52.4 53.1 53.4 53.2 Average 62.1 61.6 62.6 63.1 63.0 61.7 LLaVA-Next-7B α = 0 (VaN) α = 0.2 α = 0.4 α = 0.6 α = 0.8 α = 1 (TaR)

Short (-3 min) 69.8 69.8 72.0 71.1 71.9 68.8 Medium (3 - 30 min) 59.3 59.3 59.2 58.7 57.7 57.3

- Long (30 - 120 min) 52.1 52.1 52.7 52.0 51.4 51.0 Average 60.4 60.4 61.3 60.6 60.3 59.0

- Table A1: TaR and VaN Combination Ratio: Sweep over combination ratio α in TaR and VaN combination under a 6K memory budget (|M|) on VideoMME. α=0 and α=1 correspond to VaNonly and TaR-only, respectively. The best-performing configurations are shown in bold, while the second-best results are underlined.

Qwen-2-VL-7B MLVU VideoMME

|M| = 6K Holistic Single Multi Avg. Short Medium Long Avg. r = f × 0.125 77.6 66.2 43.9 63.1 68.7 57.3 51.0 59.0 r = f × 0.25 77.8 67.1 43.5 63.4 68.0 57.2 51.1 58.8 r = f × 0.50 78.6 64.6 39.0 61.3 65.1 56.7 49.7 57.2 |M|/|C| = 0.75 78.2 71.7 44.6 65.8 74.4 59.9 51.9 62.1 |M|/|C| = 0.50 76.8 68.2 42.3 63.3 74.1 60.3 52.9 62.4 |M|/|C| = 0.25 73.2 65.9 39.1 60.3 73.6 56.9 50.4 60.3

- Table A2: Recent Frame and Compression Ratio Exploration: Top: Sweep over recent frame numbers r determined by multiplying various ratios (0.125,0.25,0.5) to f (frame number corresponding to memory budget |M|) in TaR. Bottom: Performance under varying compression ratios |M|/|C| across MLVU and VideoMME with Qwen-2-VL-7B. TaR performs best with r ≤ 0.25 and compression ratio ≥ 0.5. The highest values are shown in bold, with the second-highest underlined.

##### A.2 Hyper-Parameter Exploration

InfiniPot-V involves three main hyper-parameters: the TaR and VaN budget allocation ratio α, the number of recent frames r used in TaR, and the target compression size C applied at each continual KV cache compression step. This section presents comparative experiments exploring each hyper-parameter.

TaR and VaN Budget Ratio (α) We compare the accuracy of offline video understanding (OVU) task across different values of α, which determines the budget allocation between TaR and VaN under a fixed memory budget (|M| = 6K), for both Qwen-2-VL-7B and LLaVA-Next-7B models. As shown in Tab. A1, performance peaks when α is between 0.4 and 0.6, outperforming the use of either VaN-only (α = 0) or TaR-only (α = 1). This confirms the effectiveness of our approach, which jointly considers both spatial and temporal dimensions for KV cache compression.

Recent Frames (r) and Compression Ratio (|M|/|C|) Tab. A2 presents exploration experiments for two key hyperparameters: the recent frames number r, which determines the proportion of recent frames within the memory budget in TaR, and the compression ratio |M|/|C|, which defines what proportion of the compression size |C| to maintain relative to the memory budget |M| in continual KV cache compression (CKV).

For the recent frame number r (Tab. A2, Top), we observe optimal performance on both MLVU and VideoMME benchmarks when r ≤ 0.25f. Setting r = 0.5f results in an excessive number of frames being designated as the latest frames for temporal redundancy measurement, which limits the effectiveness of redundancy reduction. This limitation is reflected in the decreased performance

metrics. (61.3 vs 63.4 in MLVU and 57.2 vs 59.0 in VideoMME) Note that the r sweep experiments are conducted using TaR-only settings (α = 1).

For the compression ratio (|M|/|C|), we conduct comparative experiments across three ratios (0.75, 0.50, and 0.25). As shown in Tab. A2 Bottom, an excessive compression ratio such as 0.25 in CKV results in noticeable performance degradation. These findings confirm that a ratio of 0.5 or higher represents an appropriate configuration for CKV.

Based on these explorations, we standardize the hyperparameter values at α = 0.5, r = 0.125, and |M|/|C| = 0.75 for all main experimental results when evaluating InfiniPot-V.

### B Experimental Setting Details

##### B.1 MLLMs Video Sampling Details.

For all benchmarks, we employ a consistent uniform frame sampling strategy to ensure maximized long video understanding performance across all settings. For Qwen-2-VL [43], which supports dynamic image resizing based on the number of frames, we use the hyper-parameter configuration reported to yield the best performance in their original work: FPS_MAX_FRAMES = 768, VIDEO_MIN_PIXEL = 128 × 28 × 28 and VIDEO_MAX_PIXEL = 768 × 28 × 28. Although theoretically larger token budgets could be set, we adopt this configuration to match the optimal context length of 50K as reported in the original paper [43], on top of which we applied KV cache compression. For LLaVA, we set the number of sampled frames to 128 to ensure it remained within the model’s trained context length (<32K). With this video sampling configuration, Qwen-2-VL [43] uses 384 frames with 130 tokens per frame, resulting in a total context length of 49,920 tokens, while LLaVA-Next [52] and LLaVA-OV [22] use 128 frames with 196 tokens per frame, yielding a total of 25,088 for offline long video inputs.

##### B.2 Long Video Understanding Benchmark Details

Offline Video Understanding(OVU) We evaluate our method on four multiple-choice based offline video question answering benchmarks: Video-MME [11], MLVU [54], EgoSchema [28], and LongVideoBench [45]. For MLVU and EgoSchema, we use the development sets for evaluation.

For Video-MME, we report results without subtitles version. This is because prepending subtitles for all video frames as a single context block directly before the question represents an unrealistic setting that is incompatible with streaming scenarios, where subtitles are typically unavailable during real-time video processing and would not be accessible as complete context in advance.

Streaming Video Understanding (SVU) For SVU evaluation, we use two benchmarks: RVSEgo and EVS-Movie [49]. RVS-Ego is constructed from 10 videos from the Ego4D [15] dataset, while RVS-Movie uses 22 long videos from MovieNet [17]. Each benchmark consists of a QA set containing open-ended generation questions and their corresponding timestamps indicating when each question should be presented during video streaming.

The evaluation process works as follows: during CKV processing, when the video stream reaches the timestamp of a given question sample, we present the question and generate an answer based on the compressed KV cache accumulated up to that point. The generated answers are then compared against ground-truth answers using GPT-3.5-turbo-0125 to produce accuracy and score metrics.

##### B.3 Baseline Settings

Input Video Compression (IVC) Details. For the comparison with Input Video Compression (IVC) methods in Tab. 2 and A6, we implement LongVU [36] and DyCoke [40] as follows: For LongVU, we apply Spatial Token Compression (STC) every 8 frames as specified in the original paper. STC compresses vision token embeddings by identifying temporally redundant patches using cosine similarity between patches. We adjust the similarity threshold to control the compression rate while maintaining the original methodology. For DyCoke, we implement Token Temporal Merging (TTM) which, similar to LongVU, compresses vision encoder output features. TTM calculates cosine similarity between patches in adjacent frames to eliminate redundant patch embeddings. Following

the original paper, we process compression every 4 frames and adjust the similarity threshold to control compression rates.

For fair comparison in the Continual KV cache compression (CKV) framework in Tab. 2, we adapt both methods to work within memory constraint |M|. Specifically, we compress each input video stream to size |C| and implement sliding window attention [2] to evict older KV cache entries once the cache size reaches the predefined memory limit (|M|). This adaptation ensures all methods operate under identical memory constraints for a fair comparison with InfiniPot-V. For benchmark comparing InfiniPot-V with IVC methods that use full vision encoding without cache compression, see Tab. A6.

KV Cache Compression (KVC) Details. In Fig. 2, we compare three KV cache compression methods within Continual KV cache compression (CKV). First, Uniform Select, inspired by uniform video sampling approaches, selects frames at regular intervals and retains all KV cache tokens corresponding to those frames. For SnapKV [24], we follow the original method configuration under the CKV process, using the last 32 tokens of the given budget |M| tokens as an observation window (w) to calculate attention scores for token selection (see Eq. 1 in Appendix. D.1). Additionally, we apply 1D pooling with a kernel size of 7 to these scores, as done in the original implementation6. For InfiniPot [21], we design a proxy prompt for video compression: "Provide a detailed description of this video." This prompt is utilized in the CaP method to generate attention scores and apply KV cache compression. Detailed experimental results are provided in Tab. A5.

FastV Hyper-Parameter Settings. To provide additional performance comparison with compression methods specialized for MLLMs, we also include performance comparisons with FastV [4] in Tab.A5. FastV requires two hyper-parameters, L and R, which specify the layer where token pruning begins and the percentage of tokens to prune. For a fair comparison, we adjust the R of FastV to ensure that the total number of KV cache entries across layers matches the total entry count of other baselines that maintain the same number of KV-cache entries across each layer. Specifically, for Qwen-2-VL, the (L,R) pairs corresponding to memory budgets of 3K, and 6K are set to (2, 2.8%) and (2, 5.8%) respectively.

##### B.4 Positional Encoding Details.

MLLM backbone LLMs utilize positional encoding to differentiate vision token positions. LLaVANext [52] and LLaVA-OV [22] use standard 1D RoPE [38], while Qwen-2-VL [43] employs 3D RoPE for multimodal encoding. For Offline Video Understanding(OVU), we apply KV cache compression after positional encoding (i.e., Post RoPE). However, Streaming Video Understanding(SVU) presents a challenge: continuous video stream processing can exceed the model’s maximum positional range. For example, in LLaVA models with 196 tokens per frame, streaming more than 6 minutes of video at 0.5 FPS exceeds the 32K context window (note that RVS-Ego and RVS-Movie average over 60 minutes).

To address this, we adopt strategies from InfiniPot [21] and ReKV [35], re-ordering positional indices to fit within the memory budget |M| at each CKV step. Specifically, we cache the pre-positional encoded KV’s hidden states and re-assign positional indices during decoding, ensuring they never exceed |M| position regardless of video length. While this enables SVU for arbitrarily long videos, it discards the original positional information of vision tokens. In particular, additional handling is required for Qwen-2-VL’s 3D RoPE. Developing methods that preserve the original spatial and temporal position encoding while supporting streaming video lengths beyond the model’s positional capacity remains an open direction for future work.

### C Multi-Turn Video Understanding Analysis

Fig. A1 presents a qualitative comparison between query-dependent (SnapKV)[24] and query-agnostic (InfiniPot-V) KV cache compression approaches in multi-turn conversations with streaming video input. When SnapKV performs compression based on Q1, it generates answers almost identical to the Full KV cache for that specific query (Q1), answering that the butter was placed in the refrigerator.

6https://github.com/FasterDecoding/SnapKV

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

#### ... ... ... ...

[Figure 18]

|Q1: Where did I put the butter after eating? [Single-Detail]|
|---|
|Full-KV: After eating, the person put the butter in the refrigerator. 🙆 SnapKV: The person put the butter in the fridge after eating. 🙆 InfiniPot-V: The person in the video put the butter in the fridge after eating. 🙆<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]|
|Q2: When turning off the stove, what kind of kitchen utensil were on the stove? [Single-Detail]|
|Full-KV: There was a kettle on the stove. 🙆 SnapKV: There were two spoons on the stove. ❌ InfiniPot-V: There was a kettle on the stove. 🙆<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|Q3. How many pieces of bread appear in total during the cooking process? [Multi-Detail]|
|Full-KV: The video shows that there are two pieces of bread in the pan and two pieces of bread on the plate, total of 4 pieces of bread. 🙆 SnapKV: The video shows that there are two pieces of bread in total during the cooking process. ❌ InfiniPot-V: The video shows a total of 4 pieces of bread being used during the cooking process. 🙆<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]|

- Figure A1: Qualitative Results of Multi-Turn Conversation: Full-KV uses 16K cache while InfiniPot-V and SnapKV employ 3K compressed KV cache. SnapKV performs query-guided cache compression based on Q1 before proceeding with multi-turn conversation. The video sample is from the MLVU ego reasoning subtask, using the Qwen-2-VL-7B model. 128 frame sampling is used.

However, this query-guided compression strategy reveals significant limitations when handling different types of queries (Q2, Q3) about the same video content. Specifically, SnapKV makes critical errors in subsequent queries - misidentifying a kettle as "two spoons" in Q2 and incorrectly counting the total number of bread pieces in Q3.

In contrast, InfiniPot-V maintains accurate answers consistently across all three queries using the same 3K compressed KV cache. It correctly identifies that the butter was placed in the fridge (Q1), recognizes the kettle on the stove (Q2), and counts all 4 pieces of bread throughout the cooking process (Q3), demonstrating the effectiveness of query-agnostic compression for multi-turn streaming video scenarios.

### D Why Query-Agnostic KV Cache Compression Matters for SVU?

In this section, we provide a detailed analysis of why query-agnostic compression is essential for Streaming Video Understanding (SVU), building upon the requirements discussed in Sec. 2. To demonstrate how these SVU-specific constraints impact existing KV cache compression methods, we present a case study across three representative scenarios.

##### D.1 Preliminary: Attention-based KV Cache Compression

Eviction-based KV cache compression reduces cache size by removing tokens with the lowest importance scores. Employing attention scores for computing token importance scores is the predominant approach in previous methods [24, 4, 12, 16].

In methods such as SnapKV [24], the importance scores ut of a token xt are computed by aggregating attention scores from the last w tokens (i.e., observation window) which contain the user instruction:

N

Attn(xi → xt), (1)

ut =

i=N−w

where N is the current sequence length. Using these scores, the KV cache is compressed by retaining the top-M tokens with the highest aggregated attention scores. Here, M defines the memory budget: I = TopK(u,M) and u = [u1,··· ,uN] indicates the importance scores of all tokens. The compressed Key and Value caches are then formed by extracting tokens at indices I:

###### K˜ = K[:,I,:], V˜ = V [:,I,:] (2)

where K,V ∈ RH×N×D are the uncompressed Key and Value caches with H heads, N tokens, and per-head dimension D. This approach has two characteristics: (1) it requires computing the full KV cache for all tokens before compression, and (2) it requires the user query to be present at the end

[Figure 28]

| |
|---|

| |
|---|

Q. How many people are watching the sunset?

Text Token

Vision Token

KV Cache

Wrong!

|2|
|---|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

|\n\n..|
|---|

Compressed Memory (|𝐶|)

Miss GT Frame!

LLMBackbone

Wrong!

Generation Stage

+

|\n\n..|
|---|

[Figure 33]

KV Compression w/o Q

KV Compression w/ Q

KV Compression w/o Q

...

###### Constrained Memory (|𝑀|)

Prefill Stage

GT Evicted!

[Figure 34]

+

...

...

SYS Token Vision Token INST Token

SYS Token Vision Token

SYS: System INST: Instruction

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

Vision Encoder

GT Frame

GT Frame

GT Frame

|[Figure 35]|
|---|

[Figure 36]

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|[Figure 42]|
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

...

...

𝑡 = 0 𝑡 = 1 𝑡 = 𝑁 Streaming Video

Close-ended Video Close-ended Video

(a) Case 1. Non-Streaming Scenario

(b) Case 2. Non-Streaming Scenario

###### (c) Case 3. Streaming Scenario

Memory-Constrained? ✓ Query Agnostic? ✓

Memory Constrained? X Query Agnostic? X

Memory Constrained? X Query Agnostic? ✓

- Figure A2: KV Cache Compression Case Study with SVU: Illustration of cache control strategies under three conditions, differing in the presence of two core requirements for Streaming Video Understanding (SVU): memory constrained (MC) and query agnostic (QA). (a) Case 1: Queryguided compression retains relevant (GT) frames for accurate responses. (b) Case 2: Without query guidance, compression fails to preserve critical frames, resulting in inaccurate responses. (c) Case 3 (Streaming scenario): In streaming video processing, where frames arrive continuously, continual KV cache compression (CKV) is necessary, but queries are unavailable during compression.

of the context. We refer to these approaches as query-guided or attention-based cache compression methods.7

##### D.2 Case Study: Towards Streaming Video Understanding with CKV

To investigate the applicability of attention-based KV cache compression methods to streaming video understanding, we examine three cache control strategies (Fig. A2).

- Case 1. Recent KV cache compression methods [24, 12] assume full access to context and queries at compression time, as shown in Fig. A2(a). In this memory-unconstrained setting, the model observes the full input before compression. Previous works [24] have demonstrated that attention scores effectively identify query-relevant tokens KV cache (orange box corresponding to GT Frame in Fig. A2(a)), enabling compression that retains critical information while discarding less important tokens. As shown in Tab. A3, this approach maintains performance comparable to the uncompressed cache setup (68.01 vs 68.75) at the cost of large memory usage at compression, detailed in Fig. 1.
- Case 2. Fig. A2(b) illustrates how attention-based cache compression fails when user queries are unavailable during compression. Under this scenario, although the memory budget is assumed unconstrained, the KV cache is compressed without consideration of (future) queries, causing important visual tokens (orange tokens cache corresponding to the GT Frame) lost during compression. To quantify this degradation and explore alternatives, we test compression with generic queries (q′: "What is happening in this video?", q′′ :"What are the key events in this video?") and the last vision tokens (qv) for importance scoring:

Gen. |M| = 6K Full KV n/a 25K 68.75 (↑)

xt Attention Scoring

Prefill |M|

Gen. |M| = 3K

Case

- Case 1 Attn(q → xt) 25K 68.01 68.40

- Case 2

Attn(q′ → xt)

25K

60.35 63.42 Attn(q′′ → xt) 60.60 63.50 Attn(qv → xt) 60.32 62.28

- Case 3 Attn(qv → xt) 3K/6K 57.55 59.98

Table A3: Case study of Attention Scoring: conducted on MLVU benchmark with LLaVA-Next-Video-7B. Note that memoryconstrained setting (Case 3) shares the same budget during prefill and generation stages.

7Throughout this paper, "query" refers to the user’s instruction or question related to the given video.

- q1, q2

- q1, q3

0.9

q2, q3

0.8

JaccardSimilarity

0.7

0.6

0.5

0.4

0 5 10 15 20 25 Layer Index

Figure A3: Jaccard Similarity between KV Caches: Compare KV cache sets selected by different queries (q1,q2,q3) across layers.

Context Case 1, 2 Case 3 (Ours) Length Mem (GB) TTFT (s) Mem (GB) TTFT (s)

5K 21.29 0.98 20.93 1.08 25K 33.76 1.21 21.60 1.12 50K 58.55 2.12 22.16 1.17

100K 79.38 3.27 22.85 1.20

Table A4: Peak GPU Memory and TTFT: Comparison of peak memory usage and TimeTo-First-Token(TTFT) across different context lengths for memory-unconstrained (Case 1, 2) and memory-constrained (Case 3) approaches.

ualtt = Attn(qalt → xt), qalt ∈ {q′,q′′,qv} (3)

Tab. A3 and Fig. A2(a) show that these alternatives significantly degrade performance (60.32 vs 68.75), even with unconstrained memory.

Case 3: Streaming Scenario. Beyond the query-agnostic challenge in Case 2, deploying streaming video understanding on resource-constrained devices requires fixed memory usage for the KV cache. For input video streams, these constraints necessitate continual compression when new frames arrive and memory capacity is reached, as shown in Fig. A2(c). To evaluate this scenario, we use the query-agnostic approach from Case 2 with vision tokens (qv) for importance scoring, while compressing the KV cache whenever memory limits are reached. As shown in Tab. A3, this combined constraint further degrades performance (57.55 vs 60.32), highlighting the challenge of preserving key information under both query-agnostic and memory-constrained settings.

This case study reveals two key challenges for KV cache compression in streaming video: 1) the need for query-agnostic compression due to continuous incoming video, and 2) the requirement to maintain fixed memory constraints. These challenges cause significant performance drops in previous methods [24, 21, 4], motivating Continual KV cache compression (CKV) specifically designed for memory-constrained streaming video.

Attention Scoring Analysis We further analyze the query-dependent nature of attention-based KV cache compression using the VideoMME benchmark. To investigate why performance varies with different queries, we compute the Jaccard similarity between token sets selected for different queries across layers using attention scores at each layer. For this analysis, q1,q2,q3 represent three distinct questions associated with the same video sample in the VideoMME benchmark. As shown in Fig. A3, the similarity between token sets decreases significantly in the middle-to-late layers, dropping to around 0.4. This indicates that each query selects a different set of tokens, particularly in deeper layers. This analysis highlights that attention-based scoring methods inherently select query-specific tokens, explaining the performance degradation when query information is unavailable or changes during streaming video scenarios.

### E Memory and Latency Measurement Results

- Table A4 presents measurements of peak memory consumption and Time-To-First-Token (TTFT) during the prefill stage, conducted on a single NVIDIA A100-80GB GPU using PyTorch. The experiments averaged over five runs with three warmup iterations, compare the performance of memory-unconstrained (Case 1, 2) and memory-constrained (Case 3) approaches across various context lengths. For memory-unconstrained methods, we observe a linear growth in memory requirements, escalating from 21.29 GB at 5K tokens to 79.38 GB at 100K tokens, accompanied by a proportional increase in TTFT from 0.98 to 3.27 seconds.

Our memory-constrained continual KV cache compression (Case 3) exhibits remarkably different behavior. Despite the increasing context length, the peak memory usage shows only minimal growth, rising modestly from 20.93 GB at 5K tokens to 22.85 GB at 100K tokens. Similarly, the TTFT remains relatively stable, increasing from 1.08 to 1.20 seconds across the same range. These detailed measurements demonstrate that our approach effectively maintains near-constant resource utilization while processing extended video frames.

### F Related Work

##### F.1 MLLMs for Long Video Understanding

Recent advances in long-context MLLMs have attracted significant attention. Notable examples include Gemini-2.0 [34], supporting streaming video; LongVILA [5], capable of handling up to 6,000 video frames; LLaVA-Next-Video [52], which leverages high-quality synthetic instruction data; and Qwen-2-VL [43], enabling hour-long video analysis via multimodal RoPE.

##### F.2 Input-Vision Compression (IVC)

To address the computational demands of long-form video processing, several approaches have been proposed to compress redundant visual information before it enters the backbone LLM.

LongVU [36] adopts query-dependent input frame sampling and redundant pixel removal for finegrained video understanding, but the two-tower vision encoding results in high latency during input sampling, making it impractical for streaming scenarios. Additionally, this approach requires training specialized models to operate in the proposed manner, limiting its applicability to existing pre-trained models.

DyCoke [40] reduce redundancies between adjacent frames at the input video level and dynamically updates query-related tokens in the KV cache from external storage. Slow-Fast-LLaVA-1.5 [46] proposes dividing input video processing into separate slow and fast pathways, using different projection methods to reduce input vision tokens. However, this approach still suffers from the limitation of requiring all input vision tokens to be processed simultaneously and necessitates additional model training.

##### F.3 KV Cache Compression (KVC)

Understanding the long context in MLLMs demands efficient KV cache control to manage memory growth and latency overhead. KV cache compression methods can be broadly categorized into query-dependent and query-agnostic approaches.

Query-Dependent KV cache Compression. Methods like SnapKV [24], H2O [53], HeadKV [12] and ThinK [47] leverage query-to-context attention scores to identify crucial KV entries but require the full context to be prefilled before compression, making them impractical under memory constraints. In the multimodal domain, FastV [4] accelerates prefill by pruning vision tokens at certain layers based on their attention scores from the final query token. SparseVLM [51] selects visual tokens relevant to user queries via cross-attention. Overall, query-dependent methods effectively compress context but struggle to handle diverse queries for the given context after compression [39]. ReKV [35] addresses streaming video scenarios by offloading video-related KV cache to CPU memory and retrieving query-dependent cache entries on demand. This approach relies on external storage and suffers from data transfer overhead, making it unsuitable for memory-constrained streaming video understanding.

Query-Agnostic KV cache Compression. Recent works pursue query-agnostic KV cache compression to eliminate reliance on future queries [13, 8, 16, 20, 6, 32]. In particular, SqueezedAttention [16] uses key-based clustering but requires full-context encoding, limiting its applicability to memoryconstrained settings. InfiniPot [21] compresses context by approximating potential user queries through a task-specific proxy prompt, but it’s fixed prompt restricts flexibility. In the vision domain, HiRED [1] and FasterVLM [50] utilize [CLS] token attention scores for compression decisions. However, their reliance on special tokens restricts their application to recent MLLMs that lack such tokens [48, 52], limiting their broader applicability.

Streaming Compression Prefill Decoding VideoMME MLVU

Case

LVB Avg.

MC QA Method Budget Budget Short Medium Long Avg. Holistic Single Multi Avg.

###### Qwen-2-VL-7B

- - Full KV 50K 50K 74.68 62.11 55.00 63.93 76.34 73.91 43.29 65.85 58.77 62.85

- Case1

✗ ✗

FastV [4] 48/3K (R = 2.8) 54.11 50.11 48.67 50.96 69.59 59.40 33.84 55.01 47.94 51.30

(L = 2) 48/6K (R = 5.8) 59.67 54.55 50.78 55.00 72.00 64.08 33.47 57.60 50.53 54.38 SnapKV [24]

50K 3K 74.00 61.00 54.22 63.07 77.08 67.49 39.07 62.11 59.06 61.42 50K 6K 74.22 60.55 54.33 63.03 77.59 73.91 42.90 66.10 58.80 62.64

- Case2

✗ ✓

Uniform

50K 3K 70.33 54.67 49.55 58.18 72.29 59.06 33.51 55.54 59.80 57.84 50K 6K 72.00 58.78 52.11 60.96 77.08 67.49 39.07 62.11 59.11 60.73

SnapKV† 50K 3K 69.00 54.00 50.67 57.89 75.88 63.48 35.35 58.99 56.70 57.86 50K 6K 72.11 57.56 52.22 60.63 76.46 66.43 36.22 60.66 56.72 59.34

- Case3(CKV)

3K 3K 66.00 52.44 48.00 55.48 72.54 59.00 33.51 55.59 55.21 55.43 6K 6K 72.33 53.33 48.67 58.11 72.55 62.19 33.67 57.00 55.82 56.98

Uniform

12K 12K 74.00 55.33 51.44 60.26 75.94 65.53 37.01 60.36 57.91 59.51 24K 24K 74.22 59.22 53.22 62.22 77.22 71.10 40.78 64.18 58.60 61.67

3K 3K 66.67 52.22 49.89 56.26 75.88 63.48 35.35 58.99 54.91 56.72 6K 6K 72.00 55.33 51.33 59.55 76.46 66.43 36.22 60.66 55.15 58.45

SnapKV‡

12K 12K 74.44 58.89 52.89 62.07 75.71 68.61 35.98 61.31 56.89 60.09 24K 24K 74.22 61.00 53.78 63.00 77.66 71.82 39.90 64.37 59.09 62.15

✓ ✓

3K 3K 67.11 54.55 51.00 57.55 74.94 61.80 36.60 58.36 54.00 56.64 6K 6K 72.89 57.33 51.33 60.52 75.02 63.18 37.09 59.11 54.64 58.09

InfiniPot [21]

12K 12K 74.00 57.78 53.22 61.67 74.46 66.46 38.30 60.70 56.94 59.77 24K 24K 74.22 60.55 53.56 62.78 76.03 71.11 40.29 63.71 57.85 61.44

3K 3K 73.89 57.78 51.78 61.11 77.73 70.38 43.15 64.70 57.64 61.15 6K 6K 74.11 60.78 53.44 62.78 77.16 72.31 44.75 65.82 58.40 62.33

InfiniPot-V

12K 12K 74.22 62.68 53.89 63.59 76.90 73.41 43.97 65.99 59.18 62.92

24K 24K 74.22 63.22 53.11 63.52 76.91 73.97 42.18 65.73 58.94 62.73

LLaVA-Next-Video-7B

- - Full KV 25K 25K 74.33 60.11 54.11 62.85 80.60 73.73 49.43 68.75 63.55 65.05

- Case1

✗ ✗

Uniform

25K 3K 74.33 62.33 55.00 63.89 80.29 72.38 49.19 68.01 62.35 64.75

- 25K 6K 73.89 62.00 54.78 63.56 80.66 72.25 49.62 68.19 62.55 64.76

SnapKV [24]

25K 3K 74.44 59.89 53.78 62.70 80.41 73.01 49.67 68.46 62.34 64.50

- 25K 6K 74.44 60.11 53.78 62.78 80.60 73.45 49.48 68.64 62.34 64.59

- Case2

✗ ✓

Uniform

25K 3K 66.33 54.00 49.67 56.67 75.12 59.65 38.55 58.04 59.14 57.95 25K 6K 71.00 56.33 51.55 59.63 77.84 65.60 43.92 62.90 61.69 61.41

SnapKV† 25K 3K 64.00 54.55 51.11 56.55 78.53 59.73 41.69 59.94 56.19 57.56 25K 6K 69.55 58.44 52.78 60.26 80.86 63.65 45.07 63.26 59.90 61.14

- Case3(CKV)

1.5K 1.5K 56.22 46.89 44.00 49.04 69.72 52.53 36.53 52.87 54.92 52.28 3K 3K 59.22 51.55 47.44 52.74 74.30 57.25 36.48 56.19 54.40 54.44 6K 6K 64.89 55.67 49.78 56.78 76.71 61.14 34.55 57.99 57.72 57.50

Uniform

12K 12K 72.67 59.89 53.00 61.85 80.03 67.33 44.31 64.38 61.04 62.42

1.5K 1.5K 52.40 58.00 51.33 47.89 74.92 56.89 32.62 55.11 53.65 52.22 3K 3K 62.11 54.55 48.55 55.07 76.94 59.18 35.71 57.55 54.71 55.78 6K 6K 66.33 56.11 51.11 57.85 79.60 62.15 37.12 59.98 57.81 58.55

SnapKV‡

12K 12K 72.11 58.00 53.11 61.07 79.71 67.99 44.89 64.74 58.83 61.55

✓ ✓

1.5K 1.5K 53.22 51.11 47.55 53.11 69.89 56.44 30.54 52.88 52.14 52.71 3K 3K 58.22 51.78 49.33 54.22 72.42 55.88 34.45 54.48 52.43 53.71 6K 6K 62.44 53.89 51.11 55.81 76.46 57.97 37.07 57.28 55.58 56.22

InfiniPot [21]

12K 12K 70.55 59.22 52.55 60.77 79.84 67.81 45.57 64.89 59.23 61.63 1.5K 1.5K 63.89 52.55 47.11 54.52 77.08 57.32 34.64 56.49 56.48 55.83

3K 3K 67.78 56.22 50.33 58.11 77.88 65.74 40.31 61.94 58.37 59.47 6K 6K 72.44 59.55 51.33 61.11 80.03 69.41 43.93 65.16 60.86 62.38

InfiniPot-V

12K 12K 73.89 58.67 52.11 61.55 80.91 71.16 51.57 68.35 61.84 63.91

- Table A5: InfiniPot-V vs KVC Offline long video understanding evaluation results under memoryconstrained scenario (case 3), with MC (Memory-Constrained) and QA (Query-Agnostic) conditions marked. Results are reported on (1) Video-MME - Short: -3min, Medium: 3-30min, Long: 30min-2h,

(2) MLVU - Holistic, Single-Detail, Multi-Detail LVU, and (3) LVB (LongVideoBenchmark).

### G Experimental Results Data

##### G.1 Comparison between InfiniPot-V and KVC

Tab. A5 provides a detailed performance comparison between KV cache compression (KVC) methods and InfiniPot-V across offline video understanding (OVU) benchmarks under various compression ratios for two models: Qwen-2-VL and LLaVA-Next.

In Case 1, where the full prefill is conducted and the final query is accessible at compression time, FastV demonstrates significantly inferior performance at similar compression ratios due to its aggressive token-pruning strategy. In contrast, SnapKV shows robust performance at high compression ratios across both models by utilizing the full context KV cache and retaining vision tokens that are highly correlated with the given query.

Qwen-2-VL Vision Decoding MLVU VideoMME IVC Methods Budget Budget Holistic Single Multi Avg. Short Med Long Avg. Avg. Full KV 50K 50K 76.3 73.9 43.3 65.9 74.7 62.1 55.0 63.9 64.2 Uniform 50K 6K 77.7 69.8 41.6 64.0 74.9 58.0 52.8 61.9 62.5 TTM [40] 50K 6K 78.2 70.0 42.7 64.5 74.9 59.2 52.7 62.3 62.9 STC [36] 50K 6K 77.9 71.5 44.7 65.7 74.3 59.6 54.6 62.8 63.8 InfiniPot-V 6K 6K 77.2 72.3 44.7 65.8 74.1 60.8 53.4 62.8 63.8 Uniform 50K 3K 75.7 66.5 38.6 61.1 72.2 53.4 50.0 58.6 59.4 TTM [40] 50K 3K 77.3 67.8 39.5 62.4 72.7 56.2 52.2 60.4 61.0 STC [36] 50K 3K 76.9 68.2 41.7 63.1 71.2 55.9 53.7 60.3 61.3 InfiniPot-V 3K 3K 77.7 70.4 43.2 64.7 73.9 57.8 51.8 61.1 62.5

- Table A6: InfiniPot-V vs IVC: Performance comparison between Input-Vision Compression (IVC) methodology and InfiniPot-V. Vision budget denotes the vision token length before IVC, while decoding budget refers to the input token length used during decoding. Evaluated using Qwen-2-VL with MLVU and VideoMME datasets.

Case 2 examines the query-agnostic setting, where, as explored in our earlier case study in Appendix. D.2, SnapKV exhibits notable performance degradation across both models when applied in a query-agnostic manner, showing performance comparable to uniform selection baseline.

In Case 3, which represents the CKV framework scenario where the constrained memory budget is used for both prefill and decoding stages, InfiniPot-V significantly outperforms all three baselines across various compression ratios on both models, as showcased in Fig. 2.

##### G.2 Comparison between InfiniPot-V and IVC

Table A6 presents a performance comparison between Input-Vision Compression (IVC) methods and InfiniPot-V on the MLVU and Video-MME benchmarks using the Qwen-2-VL model. Under a 6K decoding budget, the IVC methods demonstrate robust overall performance by utilizing the full vision encoding budget (50K tokens). InfiniPot-V achieves comparable or slightly superior performance to these methods while operating under constrained memory budgets for both vision encoding and decoding stages (6K tokens).

When the decoding budget is compressed to 3K tokens, the IVC methods exhibit performance degradation, with LongVU’s STC methodology achieving the highest performance among the IVC approaches. Notably, InfiniPot-V demonstrates both efficiency and effectiveness by achieving higher accuracy than IVC methods that utilize the full vision encoding budget, while operating under constrained budgets (3K) for both vision encoding and decoding stages.

### H Limitation and Future Work

InfiniPot-V introduces the first training-free, query-agnostic framework for memory-constrained streaming video understanding, enabling length-independent KV cache compression with minimal accuracy loss across long-form, real-time scenarios. However, several avenues exist for further advancement. Current approaches focus primarily on vision tokens, yet real-world streaming applications involve multiple modalities including speech, text, and video simultaneously. Future work could extend our framework to unified multimodal compression, enabling more realistic and comprehensive streaming understanding systems that efficiently manage diverse input types within fixed memory constraints.

Additionally, our current fixed budget allocation between TaR and VaN components could benefit from adaptive mechanisms that dynamically adjust compression ratios based on input characteristics—allocating more resources to temporal redundancy reduction for static scenes or prioritizing spatial importance for content-rich frames. Furthermore, while InfiniPot-V’s training-free nature ensures broad applicability, end-to-end learning approaches could optimize models specifically for continual compression scenarios, potentially enabling more aggressive compression ratios through learned token importance estimation [18] tailored to streaming video understanding tasks.

