# Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache

Xiaoran Liu1,2*, Siyang He1*, Qiqi Wang1*, Ruixiao Li1,2*, Yuerong Song1,2, Zhigeng Liu1, Mianqiu Huang1, Linlin Li3, Qun Liu3, Zengfeng Huang1,2, Qipeng Guo2,4, Ziwei He2†, Xipeng Qiu1,2† 1School of Computer Science, Fudan University, 2Shanghai Innovation Institute, 3Huawei Noah’s Ark Lab, 4Shanghai AI Lab xrliu24@m.fudan.edu.cn, xpqiu@fudan.edu.cn, ziwei.he@sjtu.edu.cn

arXiv:2506.11886v1[cs.CL]13Jun2025

## Abstract

Large Language Models struggle with memory demands from the growing Key-Value (KV) cache as context lengths increase. Existing compression methods homogenize head dimensions or rely on attention-guided token pruning, often sacrificing accuracy or introducing computational overhead. We propose FourierAttention, a training-free framework that exploits the heterogeneous roles of transformer head dimensions: lower dimensions prioritize local context, while upper ones capture long-range dependencies. By projecting the long-contextinsensitive dimensions onto orthogonal Fourier bases, FourierAttention approximates their temporal evolution with fixed-length spectral coefficients. Evaluations on LLaMA models show FourierAttention achieves the best longcontext accuracy on LongBench and Needle-InA-Haystack (NIAH). Besides, a custom Triton kernel, FlashFourierAttention, is designed to optimize memory via streamlined read-write operations, enabling efficient deployment without performance compromise.

## 1 Introduction

Large Language Models (LLMs) have transformed natural language processing with breakthroughs in text generation, comprehension, and reasoning (OpenAI, 2023; Sun et al., 2024; OpenAI, 2024; Guo et al., 2025). However, their autoregressive decoding relies heavily on a memory-intensive KeyValue (KV) cache, leading to significant memory allocation as context lengths scale (Vaswani et al., 2017; Fu, 2024; Liu et al., 2025). This overhead limits LLM deployment in resource-constrained environments. While approaches like quantization and sparse attention have been explored to reduce memory needs, they often compromise accuracy or add complexity (Liu et al., 2024c; Hooper et al.,

* Equal contribution. † Corresponding Author.

2024; Yuan et al., 2025). Developing memoryefficient methods that preserve performance remains critical for broader LLM applicability.

Existing training-free KV cache compression methods, like token eviction strategies (Xiao et al., 2024; Zhang et al., 2023; Li et al., 2024b), prune sequence subsets but overlook the heterogeneous roles of head dimensions, leaving dimension-aware allocation largely unexplored. Similarly, quantization methods (Liu et al., 2024c; Hooper et al., 2024; Duanmu et al., 2024) reduce memory by fixed bitwidths, and hidden dimension compression (Chang et al., 2024; Saxena et al., 2024) methods apply uniform ratios, both neglect their distinct contribution across dimensions (Liu et al., 2024b; Peng et al., 2024). These approaches treat head dimensions as homogeneous, static units rather than dynamically allocating resources based on their importance.

Another critical limitation of existing methods lies in their reliance on attention-guided strategies (Zhang et al., 2023; Li et al., 2024b). While these approaches enable selective token pruning with minimal accuracy degradation, they impose prohibitive memory and latency overheads due to attention score recalculation. We address this challenge by adapting the HiPPO framework (Gu et al., 2020), a mathematically grounded approach for long-sequence modeling. HiPPO approximates infinite-length sequences as compact finite states by projecting inputs onto finite-order orthogonal basis functions, such as polynomial bases or Fourier bases (Gu et al., 2020; He et al., 2023). This retains global critical and contextually vital patterns while filtering out redundant signals. By leveraging HiPPO’s theoretical foundations, we can bypass attention recomputation entirely, achieving both memory efficiency and computational efficiency.

Building on these insights, we introduce FourierAttention, a training-free KV cache compression framework using a translated Fourier transform. Departing from prior methods that uniformly

KV Cache

KV Cache

Preserve initial & local tokens

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |

,  / , 

,  / , 

FourierT

| | | | |
|---|---|---|---|
| | | | |

Interleave

Forward

/

/

Concat

/

Inv. FT

New Token

| | | |
|---|---|---|
| | | |

FourierT

/ Prefilling Phase

/

/

Update Decoding Phase

Update

Figure 1: Overview of FourierAttention.

process all head dimensions, FourierAttention identifies localized, context-insensitive dimensions in KV states and approximates their temporal evolution via a fixed set of orthogonal Fourier basis functions. By retaining only the dominant Fourier coefficients (k ≪ L, where L is the sequence length), our method projects sequences into a compact spectral representation. Unlike polynomial bases that are widely used in HiPPO, which require recurrent state updates, FourierAttention exploits the shift-invariance and temporal parallelism of Fourier transforms, allowing for efficient computation in a single pass. During decoding, a customized Triton kernel FlashFourierAttention is used to decompose KV cache states during attention calculation, minimizing memory overhead via streamlined readwrite operations.

Our contributions can be summarized as follows:

- • We reveal a bifurcation in Transformer head dimensions: lower dimensions prioritize local context, while upper ones capture long-range dependencies. This inspires us to compress long-context-insensitive dimensions without sacrificing contextual awareness.
- • We introduce FourierAttention, which optimizes KV cache by projecting its temporal evolution onto a fixed set of orthogonal Fourier bases. This method efficiently eliminates redundant components while preserving contextual fidelity, achieving a balance between memory and computational efficiency.
- • We evaluate FourierAttention’s performance on the LLaMA Series using LongBench and NIAH. Our FourierAttention achieves the best

long-context performance on average while maintaining lower memory consumption.

## 2 Related Work

KV cache optimization is a crucial technique for enhancing efficiency in attention-based LLMs (Fu, 2024; Liu et al., 2025). As context length increases, the KV cache in LLMs grows linearly, creating substantial memory overhead that becomes a bottleneck for long-context applications. Beyond architectural modifications during pretraining (Ainslie et al., 2023; Liu et al., 2024a), existing training-free optimization approaches mainly involve token eviction or compression. The former discards tokens based on positional or attention patterns, including StreamingLLM (Xiao et al., 2024), H2O (Zhang

- et al., 2023), SnapKV (Li et al., 2024b), and PyramidKV (Cai et al., 2024), while the latter compresses KV cache through quantization or lowrank projection, such as KIVI (Liu et al., 2024c), KVQuant (Hooper et al., 2024), and Palu (Chang
- et al., 2024). However, these methods lack finegrained consideration of different head dimensions in KV cache, applying uniform optimization across all dimensions. In contrast, our FourierAttention compresses most dimensions to a fixed length while preserving long-context-sensitive dimensions, effectively reducing KV cache size while maintaining the original long-context capabilities.

## 3 Methodology

Due to considerations of different head dimensions of KV cache, we propose FourierAttention. In FourierAttention, most dimensions of the KV cache are compressed to a fixed length through translated Fourier transform, as shown in Figure 1.

Layer 2 Dim [0, 127]

Layer 31 Dim [0, 127]

Avg Score 100.00

0 2 4 6 8

0 2 4 6 8

[Figure 1]

[Figure 2]

[Figure 3]

0

20

DocumentDepth

40

60

10 12 14

10 12 14

80

100

0 2 4 6 8 10 12 14

0 2 4 6 8 10 12 14

4k 8k 16k 24k 32k

Context Length

Context Length

Context Length

Layer 2 Dim [0, 69]

Layer 31 Dim [0, 69]

Avg Score 97.02

0

0 2 4 6 8

[Figure 4]

[Figure 5]

[Figure 6]

0

- 2 4 6 8

20

DocumentDepth

40

60

10 12 14

10 12 14

80

100

0 2 4 6 8 10 12 14

0 2 4 6 8 10 12 14

4k 8k 16k 24k 32k

Context Length

Context Length

Context Length

0

[Figure 7]

- 2 4 6 8

10 12 14

Layer 2 Dim [70, 127]

[Figure 8]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 31 Dim [70, 127]

[Figure 9]

4k 8k 16k 24k 32k

Context Length

100

80

60

40

20

0

DocumentDepth

Avg Score 53.20

(a) LLaMA3.1-8B

[Figure 10]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 2 Dim [0, 127]

[Figure 11]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 27 Dim [0, 127]

[Figure 12]

4k 8k 16k 24k 32k

Context Length

100

80

60

40

20

0

DocumentDepth

Avg Score 100.00

[Figure 13]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 2 Dim [0, 69]

[Figure 14]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 27 Dim [0, 69]

[Figure 15]

4k 8k 16k 24k 32k

Context Length

100

80

60

40

20

0

DocumentDepth

Avg Score 87.46

[Figure 16]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 2 Dim [70, 127]

[Figure 17]

0 2 4 6 8 10 12 14

Context Length

0 2 4 6 8

10 12 14

Layer 27 Dim [70, 127]

[Figure 18]

4k 8k 16k 24k 32k

Context Length

100

80

60

40

20

0

DocumentDepth

Avg Score 37.62

(b) LLaMA3.2-3B

- Figure 2: Visualization of the average attention score and its components in LLaMA3.1-8B (Dubey et al., 2024) and LLaMA3.2-3B (Meta, 2024a) over 32 sentences, each with a length of 16. The component of the lower dimensions corresponds to the local branch in StreamingLLM (Xiao et al., 2024), while that of the upper dimensions corresponds to the global branch. This reveals the different functions of different dimensions in the attention mechanism. It can be further validated that adding Gaussian noise to the lower dimensions has little effect on NIAH performance, but adding noise to the upper dimensions will harm the performance remarkably.

- 3.1 Head Dimension Specialization

0 2 4 6 8 10 12 14

Context Length

We analyze the heterogeneous sensitivity of transformer head dimensions to varying context lengths. By visualizing attention scores across 128 dimensions in LLaMA architecture (Figure.2), we identify a bifurcation in attention patterns: the first 70 dimensions (0–69) exhibit sharp focus on shortrange context, with score distributions concentrated on recent tokens, while the latter 58 dimensions (70–127) maintain a persistent bias toward initial "sink tokens"—positional embeddings that serve as static reference points. This divergence suggests distinct contextual roles encoded within head dimensions, where specialized subsets prioritize local versus global signal retention.

To further validate this hypothesis, we evaluate the model on a Needle-In-A-Haystack retrieval task across sequences of up to 32,000 tokens. As shown in Figure 1(a), the baseline model achieves perfect retrieval accuracy (100.0). Introducing Gaussian noise to the first 70 dimensions, performance remains robust (97.02), confirming their limited role in long-range dependency resolution. Conversely, perturbing the latter 58 dimensions catastrophically reduces accuracy to 53.20 on average, with failures consistent across all tested depths and context

lengths (Figure 1(b) mirrors this trend). This stark contrast empirically demonstrates that upper dimensions in transformer are indispensable for retaining long-range information, while lower dimensions specialize in local context encoding. These findings provide critical insights for optimizing memoryefficient architectures, as strategically prioritizing dimensions specialized in long-range retention enhances contextual awareness within memory limits. For more details on dimension selection, please refer to Section 3.4.

### 3.2 Preliminary: HiPPO Framework

Inspired by HiPPO (Gu et al., 2020), we compress these less context-sensitive dimensions into fixedlength states to reduce KV cache storage. Under the HiPPO framework, an infinitely long sequence, f1···L, can be approximated by finite-length states, c ∈ Rk, as the combining coefficients of finiteorder basis functions. HiPPO designs different state update equations for various basis functions under different measure functions, such as LegT based on Legendre Polynomial in a translated fixed window size. Among these methods, FourierT measure based on Translated Fourier Transform is most suitable for token-wise parallelism in transformers, because it can be expressed in matrix form and

performed independently in different order states. Therefore, we adopt FourierT to compress cache, K,V ∈ RL×d, which also achieves better downstream performance in Section 5.1.

### 3.3 Online Compression via HiPPO-FourierT

We set the translated window length in FourierT to the maximum context length, ensuring effective compression within valid input-output ranges. In the prefilling phase, we preserve all dimensions of the initial Linit and the local Llocal tokens,

Ki,Kl = K[: Linit],K[−Llocal :] V i,V l = V [: Linit],V [−Llocal :]

(1)

and distinguish the dimension indices Dku, Dkc,Dvu, Dvc in KV cache for uncompressing and compressing, to enable training-free integration.

Kmn = K[Linit : −Llocal,Dkc],

V mn = V [Linit : −Llocal,Dvc], Kmu = K[Linit : −Llocal,Dku],

V mu = V [Linit : −Llocal,Dvu].

(2)

We preserve Kmu,V mu, compress Kmn,V mn to fix-sized Kmc ∈ R2k×|Dkc|,V mc ∈ R2k×|Dvc| and use the original KV for forward propagation.

Kmc = FKmn, V mc = FV mn, O = flash_attention(Q,K,V ).

(3)

When it comes to the compression matrix in FourierT, originally, F ∈ Ck×L, where Fnt = ei2πntT . However, since caches in mainstream LLMs are real-valued, we convert complex numbers to corresponding 2D vectors, transforming k-order complex states into 2k-order real states. Therefore, the real compression matrix in FourierT is F ∈ R2k×L as shown in Equation 4.





1 1 ··· 1

- 0 0 ··· 0
- 1 cos 2Tπ ··· cos 2π(LT−1)

- 0 sin 2Tπ ··· sin 2π(LT−1)

. .

... .

- 1 cos 2π(Tk−1) ··· cos 2π(k−1)(T L−1) 0 sin 2π(Tk−1) ··· sin 2π(k−1)(T L−1)

F =

.

 

 

(4) In the decoding phase, FourierAttention com-

presses tokens out of the local range individually,

Kmc ← Kmc + ft+1Kl[0,Dkc] V mc ← V mc + ft+1V l[0,Dvc]

⊤

ft+1 = 0 1 ··· cos 2π(kT−1)t sin 2π(kT−1)t

(5) and reconstruct intermediate cache K˜ m,V˜ m via inverse Fourier transform in attention computation with the current query vector qt+1.

1 kFTKmc

K˜ m[Dku] ← Kmu, K˜ m[Dkc] ←

K˜ = cat(Ki,K˜ m,Kl) V˜ m[Dvu] ← V mu, V˜ m[Dvc] ←

1 kFTKmc

V˜ = cat(V i,V˜ m,V l) ot+1 = flash_attention(qt+1,K˜ ,V˜ ).

(6) To eliminate intermediate read-write cost in decompression, we try to implement a custom kernel, FlashFourierAttention, using Triton (Tillet et al., 2019), integrating the decompression into standard FlashAttention2 (Dao, 2024) and FlashDecoding (Dao et al., 2023). FlashFourierAttention loads compressed intermediate states once and decompresses at corresponding sequence positions during iterative KV cache loading. FlashFourierAttention is still in progress to achieve better computational efficiency compared with standard attention.

### 3.4 Fine-Grained Compression Schema

In FourierAttention, a crucial point lies in how to select the dimension to be compressed. To address this, we directly compress and decompress all KV caches, prioritizing dimensions with smaller meansquared error in reconstruction to a fixed length. Based on further observations of the KV cache, we adopt a fine-grained compression schema, where more dimensions of the V cache and lower-layer caches are compressed to a fixed length. We analyze the standard deviation of KV cache dimensions along the temporal direction across different layers and find that for both LLaMA3.1-8B and LLaMA3.2-3B, as shown in Figure 3. The standard deviation of the K cache is consistently higher than that of the V cache, and the standard deviation in upper layers exceeds that of lower layers. Consequently, we compress more dimensions of the smoother V cache and lower-layer caches to a fixed length, while retaining more K cache and upperlayer caches to extend with sequence length. Thus,

[Figure 19]

[Figure 20]

(a) K cahce in LLaMA3.1-8B (b) V cahce in LLaMA3.1-8B

[Figure 21]

[Figure 22]

(c) K cahce in LLaMA3.2-3B (d) V cahce in LLaMA3.2-3B

- Figure 3: Visualization of standard deviation of KV cache in different layers in LLaMA3.1-8B (Dubey et al.,

2024) and LLaMA3.2-3B (Meta, 2024a). The feature dimensions are sorted based on the indices in each head.

FourierAttention exhibits an asymmetric, invertedpyramid compression pattern.

Interestingly, this differs from most KV cache compression approaches. Works like Cai et al. (2024) and Xing et al. (2024) suggest preserving more KV caches in lower layers, as attention becomes sparser in upper layers. However, in FourierAttention, the optimization criterion is whether the dimension can be well reconstructed. Since caches in upper layers exhibit more oscillatory features due to more deterministic predictions, we retain more dimensions to maintain output stability.

- 4 Experiment

### 4.1 Setup

We conduct experiments on LLaMA3.1-8B (Dubey et al., 2024) and LLaMA3.2-3B (Meta, 2024a). For all models, we set the length of initial tokens Linit to 4, the length of local tokens Llocal to 1024, and the number of states k = 512. We evaluate the reconstruction loss using the prompt portion of the 32k Needle-In-A-Haystack benchmark in OpenCompass (Contributors, 2023). As mentioned earlier, we employ an asymmetric inverted pyramid compression strategy: for the first 4 layers, we compress 90% of K dimensions and 95% of V dimensions; for the last 8 layers, 50% of K and 70% of V; and for the remaining layers, 80% of both K

and V. Overall, 76% KV caches are compressed to a fixed length. All experiments are performed on an NVIDIA H100 GPU with FP16 precision and accelerated with FlashAttention2 (Dao, 2024).

### 4.2 Long-Context Evaluation

We evaluate our method against other KV cache optimization approaches with two long-context benchmarks in OpenCompass (Contributors, 2023), LongBench (Bai et al., 2023) and Needle-In-AHaystack (NIAH) (Kamradt, 2023; Li et al., 2024a), with a truncation context length of 32K. We compare with StreamingLLM (Xiao et al., 2024), SnapKV (Li et al., 2024b), Palu (Chang et al., 2024), and KIVI (Liu et al., 2024c), covering both token eviction and feature compression. For fair comparison, we retain 4 initial tokens and 1024 local tokens in StreamingLLM, additionally keep 1024 recalled middle tokens, matching our compressed dimension count, in SnapKV, compress KV feature dimensions to 70% in Palu, and apply 4-bit quantization, 75% compression, in KIVI.

For LongBench as shown in Tables 1 and 2, our FourierAttention achieves performance closest to the original model on LLaMA3.2-3B and is slightly inferior to SnapKV on LLaMA3.1-8B. For the NIAH task as shown in Figure 4 and 5, we similarly achieve performance closest to the original pretrained models at 32k context length. While SnapKV is theoretically suitable for retrieval tasks like NIAH, it still exhibits recall errors. Though Palu and KIVI maintain stable attention approximation under moderate compression, 30-50%, they show significant performance degradation at 75% compression due to insufficient granular analysis of KV cache features. In contrast, our FourierAttention optimizes compression by identifying and preserving KV dimensions insensitive to compression, thereby maximally retaining the long-context capabilities and demonstrating superiority across both models and benchmarks.

In addition to comparisons in downstream performance, we will also conduct efficiency experiments. Since our custom kernel FlashFourierAttention is still in progress, we will report this in detail in the next version of the paper.

5 Discussion

### 5.1 Choice of Basis Functions

Although FourierAttention employs HiPPOFourierT for compression, Gu et al. (2020)

##### Single-Doc QA Multi-Doc QA Summarization Few-shot Learning

NQA Qsp MulF HQA 2WQA MSQ QRpt QSum MulN TREC TrQA SSum

- LLaMA3.1-8B 13.22 20.24 32.81 11.97 13.60 8.72 29.70 25.09 0.90 73.50 90.97 47.26

+ StreamingLLM 7.87 13.86 15.59 7.79 10.08 4.49 19.93 21.52 9.88 61.50 84.66 43.48 + SnapKV 12.67 19.84 32.48 12.00 13.76 8.59 29.19 24.90 12.56 73.00 90.97 46.53 + Palu 4.50 18.03 21.58 9.47 11.32 5.23 17.25 6.90 8.98 68.50 83.38 32.29 + KIVI 12.73 20.94 32.79 11.51 13.93 8.77 30.33 25.18 2.41 73.50 10.83 45.98 + Ours 14.94 18.47 30.47 13.46 14.29 9.64 22.34 22.69 5.00 72.50 89.18 43.80

- LLaMA3.2-3B 10.27 21.69 35.52 9.58 12.78 6.75 30.15 23.77 28.24 70.00 87.24 38.19

+ StreamingLLM 9.14 17.59 21.57 7.06 9.78 3.99 19.02 21.32 23.23 53.00 84.33 39.76 + SnapKV 8.92 21.04 34.97 9.50 12.77 6.62 29.45 23.36 27.77 69.50 86.39 38.33 + Palu 1.98 19.17 20.37 5.84 10.28 2.65 13.37 4.10 13.97 57.00 47.39 21.48 + KIVI 10.21 22.17 35.08 9.68 12.29 6.94 30.84 23.52 23.21 70.00 64.09 43.46 + Ours 8.91 21.27 31.20 11.99 17.77 8.26 23.41 22.16 23.76 69.00 86.18 37.12

Table 1: Results of LLaMA Series (Dubey et al., 2024; Meta, 2024b) on LongBench (Bai et al., 2023). Our FourierAttention achieves a superiority over StreamingLLM (Xiao et al., 2024), Palu (Chang et al., 2024), and KIVI (Liu et al., 2024c) and shows closest performance with LLMs with SnapKV (Li et al., 2024b).

Synthetic Code

Avg. PsgC PsgR LCC Re-P

- LLaMA3.1-8B 0.75 26.75 72.00 69.27 39.49

+ StreamingLLM 1.25 6.18 58.36 56.35 31.52 + SnapKV 0.75 26.75 59.97 59.74 37.07

- + Palu 0.60 14.58 56.68 54.57 30.68

+ KIVI 0.75 30.00 29.43 19.86 23.18 + Ours 2.25 16.07 67.32 63.02 36.98

LLaMA3.2-3B 0.00 7.00 70.01 66.38 38.04 + StreamingLLM 1.37 6.47 55.47 53.53 31.19 + SnapKV 0.00 6.75 58.20 56.36 34.83

- + Palu 1.45 3.08 55.10 49.54 25.53

+ KIVI 0.00 6.00 40.01 37.87 28.98 + Ours 1.62 8.32 67.29 59.91 36.33

Table 2: Continuous table of Table 1.

MK1 MK2 MK3 MV Avg.

LLaMA3.2-3B 99.00 100.00 99.00 100.00 99.50 + FourierT 99.00 99.00 99.00 96.00 98.25 + LegT 89.00 93.00 50.00 93.75 81.44

+ uniform 100.00 99.00 98.00 90.50 96.88 + KV inv. 98.00 100.00 98.00 93.00 97.25 + layer inv. 99.00 98.00 93.00 86.00 94.00

Table 3: Validation of basis function and compression schema in LLaMA3.2-3B (Meta, 2024a)

proposes and claims polynomial basis functions like LegT with superior performance. While maintaining identical sliding window sizes, we compare LegT and FourierT in reconstructing KV caches from LLMs. As illustrated in Figure 6, we evaluate their reconstruction effects on 4 randomly selected KV cache dimensions from layer 0 of LLaMA3.2. Under equivalent state

dimensions1, FourierT consistently achieves lower reconstruction loss than LegT.

We further evaluate FourierT and LegT compression on LLaMA3.2-3B using more discriminative NIAH variants, Multi-Key NIAH (MK) and MultiValue NIAH (MV) (Hsieh et al., 2024) in 4k context length. For fair comparison, we employ the same method to identify dimensions suitable for LegT compression and apply an identical compression schema. Results in Table 3 show FourierT still performs better, demonstrating that FourierT offers better parallelizability for compression efficiency and performance in downstream evaluation.

### 5.2 Ablation on Compression Schema

As mentioned in Section 3.4, we propose a more fine-grained compression scheme based on additional observations of the KV cache. As shown in the Table 3, we compare three approaches: uniform compression across all layers and between KV (uniform), inverted KV compression schema by K-priority over V (KV inv.), and inverted layerwise compression schema by upper-layer priority over lower-layer (layer inv.). Results demonstrate that our original V-priority and lower-layerpriority compression schema achieves superior performance on discriminative NIAH variants. This further illustrates that frequency-based sequencewise KV cache compression exhibits fundamentally different optimization characteristics compared to conventional KV token eviction methods (Cai et al., 2024; Xing et al., 2024).

1FourierT uses lower-order basis functions since FourierAttention’s state size is twice the number of states

Overall Score: 100.00

100

[Figure 23]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(a) Pretrained

Overall Score: 17.57

100

[Figure 24]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(b) StreamingLLM

Overall Score: 88.35

100

[Figure 25]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(c) SnapKV

Overall Score: 68.99

100

[Figure 26]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(d) Palu

Overall Score: 26.35

100

[Figure 27]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(e) KIVI

Overall Score: 93.17

100

[Figure 28]

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

80

DepthPercent

60

Score

40

20

Average Depth Score

100.0

0

4000 8000 16000 24000 32000

Token Limit

(f) FourierAttention (ours)

- Figure 4: Results of LLaMA3.1-8B (Dubey et al., 2024) on Needle-In-A-Haystack (Kamradt, 2023). FourierAttention achieves a highest average score over StreamingLLM (Xiao et al., 2024), SnapKV (Li et al., 2024b), Palu (Chang et al., 2024), and KIVI (Liu et al., 2024c) and shows closest performance with LLMs with full attention.

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 100.00

[Figure 29]

0

20

40

60

80

100

Score

Average Depth Score

(a) Pretrained

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 17.42

[Figure 30]

0

20

40

60

80

100

Score

Average Depth Score

(b) StreamingLLM

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 90.29

[Figure 31]

0

20

40

60

80

100

Score

Average Depth Score

(c) SnapKV

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 35.99

[Figure 32]

0

20

40

60

80

100

Score

Average Depth Score

(d) Palu

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 26.48

[Figure 33]

0

20

40

60

80

100

Score

Average Depth Score

(e) KIVI

4000 8000 16000 24000 32000

Token Limit

0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0

100.0

DepthPercent

Overall Score: 94.04

0

20

40

60

80

100

Score

[Figure 34]

Average Depth Score

(f) FourierAttention (ours)

- Figure 5: Results of LLaMA3.2-3B (Meta, 2024a) on Needle-In-A-Haystack (Kamradt, 2023). FourierAttention achieves a highest average score over StreamingLLM (Xiao et al., 2024), SnapKV (Li et al., 2024b), Palu (Chang et al., 2024), and KIVI (Liu et al., 2024c) and shows closest performance with LLMs with full attention.

### 5.3 Compressed Dimension Distribution

Finally, we analyze the compressed dimensions selected by our FourierAttention. We count the number of each dimension selected for compression, averaged across attention heads in different layers, grouped every 16 dimensions. Results in Figure 7 show that in both LLaMA3.1-8B and LLaMA3.23B, starting from layer 2, lower dimensions are more frequently compressed while the upper dimensions tend to preserve complete temporal information in our FourierAttention. This phenomenon is more evident in upper layers, where fewer dimen-

sions are chosen to be compressed. As illustrated in Figure 2, these uncompressed upper dimensions primarily contribute to forming attention sinks and capturing long-context semantic relationships, thus requiring complete retention, whereas other dimensions can be stored with limited length.

## 6 Conclusion

We propose FourierAttention, a novel KV cache optimization approach that compresses long-contextinsensitive dimensions without sacrificing contextual awareness based on an interesting phenomenon

| |origin<br><br>LegT| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

- 1

- 0
- 1
- 2

MSE: 0.4075

0 500 1000 1500 2000 2500 3000 3500

Token Index

2.0

1.5

1.0

0.5

0.0

0.5

1.0

1.5

2.0

MSE: 0.2884

origin

LegT

0 500 1000 1500 2000 2500 3000 3500

Token Index

- 2

2

3

0 500 1000 1500 2000 2500 3000 3500

Token Index

###### MSE: 0.2707

###### MSE: 0.2024

origin

1.5

- 0

- 1

- 2

LegT

1.0

0.5

0.0

0.5

1.0

1

1.5

origin

2.0

LegT

0 500 1000 1500 2000 2500 3000 3500

Token Index

(a) Cache reconstruction by HiPPO-LegT

###### MSE: 0.1105

###### MSE: 0.0293

2.0

1.00

| |origin<br><br>FourierT| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

origin

FourierT

1.5

0.75

1.0

0.50

0.5

0.25

0.0

0.00

0.25

0.5

0.50

1.0

0.75

1.5

1.00

0 500 1000 1500 2000 2500 3000 3500

0 500 1000 1500 2000 2500 3000 3500

Token Index

Token Index

###### MSE: 0.0324

MSE: 0.0732

1.0

| |origin<br><br>FourierT| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

origin

FourierT

1.0

0.5

0.5

0.0

0.0

0.5

0.5

1.0

1.0

1.5

0 500 1000 1500 2000 2500 3000 3500

0 500 1000 1500 2000 2500 3000 3500

Token Index

Token Index

(b) Cache reconstruction by HiPPO-FourierT

- Figure 6: Visualization of KV cache reconstruction in LLaMA3.2-3B (Meta, 2024a) for different basis functions, LegT and FourierT under HiPPO framework (Gu et al., 2020). FourierT outperforms LegT in cache reconstruction.

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

AverageCount

(a) Layer 0 in LLaMA3.1-8B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(b) Layer 2 in LLaMA3.1-8B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(c) Layer 15 in LLaMA3.1-8B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(d) Layer 30 in LLaMA3.1-8B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(e) Layer 0 in LLaMA3.2-3B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(f) Layer 2 in LLaMA3.2-3B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

16.0

AverageCount

(g) Layer 13 in LLaMA3.2-3B

0 16 32 48 64 80 96 112 128 Head Dimension

0.0

2.0

4.0

6.0

8.0

10.0

12.0

14.0

AverageCount

(h) Layer 26 in LLaMA3.2-3B

- Figure 7: The statistics of each dimension selected for compression, averaged across attention heads in different layers, grouped every 16 dimensions, in LLaMA3.1-8B (Dubey et al., 2024) and LLaMA3.2-3B (Meta, 2024a).

in transformer head dimensions, that lower dimensions capture local features, while upper ones capture long-context dependencies. Inspired by HiPPO, we optimize the long-context-insensitive KV cache through a translated Fourier transform into fixed-length states in the prefilling phase and reconstruct the KV cache in the decoding phase. FourierAttention shows the best performance on the LLaMA Series in LongBench and NIAH on average. We are trying to improve the efficiency of FourierAttention through a customized Tritonbased kernel, FlashFourierAttention, eliminating intermediate read-write operations and effectively reducing memory overhead.

## Limitations

We will continue optimizing our customized Triton kernel FlashFourierAttention, which injects KV cache decomposition in FlashAttention and FlashDecoding (Dao, 2024), minimizing memory overhead via streamlined read-write operations. Moreover, our performance still shows gaps compared to the pre-trained model. These aspects will be thoroughly investigated in future work.

## Acknowledgments

We thank Jian Yuan from Shanghai Jiao Tong University for assisting with experimental verification.

## References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, and 1 others. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and 1 others. 2024. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint

- arXiv:2406.02069.

Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-

- Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S Abdelfattah, and Kai-Chiang Wu. 2024. Palu: Compressing kvcache with low-rank projection. arXiv preprint

arXiv:2407.21118.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

Tri Dao. 2024. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations.

Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. 2023. Flash-decoding for long-context inference.

Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. 2024. Skvq: Sliding-window key and value cache quantization for large language models. arXiv preprint arXiv:2405.06219.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

- Yao Fu. 2024. Challenges in deploying long-context transformers: A theoretical peak performance analysis. arXiv preprint arXiv:2405.08944.

Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. 2020. Hippo: Recurrent memory with optimal polynomial projections. Advances in Neural Information Processing Systems, 33:1474– 1487.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Ziwei He, Meng Yang, Minwei Feng, Jingcheng Yin, Xinbing Wang, Jingwen Leng, and Zhouhan Lin. 2023. Fourier transformer: Fast long range modeling by removing sequence redundancy with fft operator. arXiv preprint arXiv:2305.15099.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024. Kvquant: Towards 10 million context length llm inference with kv cache quantization. arXiv preprint arXiv:2401.18079.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Greg Kamradt. 2023. Needle in a haystack - pressure testing llms. https://github.com/gkamradt/ LLMTest_NeedleInAHaystack.

Mo Li, Songyang Zhang, Yunxin Liu, and Kai Chen. 2024a. Needlebench: Can llms do retrieval and reasoning in 1 million context window? arXiv preprint arXiv:2407.11963.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024b. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, and 1 others. 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

Xiaoran Liu, Ruixiao Li, Mianqiu Huang, Zhigeng Liu, Yuerong Song, Qipeng Guo, Siyang He, Qiqi Wang, Linlin Li, Qun Liu, and 1 others. 2025. Thus spake long-context large language model. arXiv preprint arXiv:2502.17129.

Xiaoran Liu, Hang Yan, Chenxin An, Xipeng Qiu, and Dahua Lin. 2024b. Scaling laws of rope-based extrapolation. In The Twelfth International Conference on Learning Representations.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024c. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750.

AI Meta. 2024a. Introducing meta llama 3: The most capable openly available llm to date. Meta AI.

AI Meta. 2024b. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Meta AI.

- OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.
- OpenAI. 2024. O1: Openai’s first model. Accessed: 2024-12-25.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. Yarn: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations.

Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. 2024. Eigen attention: Attention in low-rank space for kv cache compression. arXiv preprint arXiv:2408.05646.

Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Xiangyang Liu, Hang Yan, Yunfan Shao, Qiong Tang, Shiduo Zhang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, and 5 others. 2024. Moss: An open conversational large language model. Machine Intelligence Research.

Philippe Tillet, Hsiang-Tsung Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and 1 others. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. 2025. Native sparse attention: Hardwarealigned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, and 1 others. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710.

