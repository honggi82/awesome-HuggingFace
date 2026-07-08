# arXiv:2603.08982v1[cs.CV]9Mar2026

### SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing

Xuanyi Zhou∗, Qiuyang Mang∗, Shuo Yang, Haocheng Xi, Jintao Zhang, Huanzhi Mao, Joseph E. Gonzalez, Kurt Keutzer, Ion Stoica, Alvin Cheung

###### UC Berkeley

Diffusion Transformers (DiTs) have become a leading backbone for video generation, yet their quadratic attention cost remains a major bottleneck. Sparse attention reduces this cost by computing only a subset of attention blocks. However, prior methods often either drop the remaining blocks which incurs information loss, or rely on learned predictors to approximate them, introducing training overhead and potential output distribution shifting. In this paper, we show that the missing contributions can be recovered without training: after semantic clustering, keys and values within each block exhibit strong similarity and can be well summarized by a small set of cluster centroids. Based on this observation, we introduce SVG-EAR, a parameter-free linear compensation branch that uses the centroid to approximate skipped blocks and recover their contributions. While centroid compensation is accurate for most blocks, it can fail on a small subset. Standard sparsification typically selects blocks by attention scores, which indicate where the model places its attention mass, but not where the approximation error would be largest. SVG-EAR therefore performs error-aware routing: a lightweight probe estimates the compensation error for each block, and we compute exactly the blocks with the highest error-to-cost ratio while compensating for skipped blocks. We provide theoretical guarantees that relate attention reconstruction error to clustering quality, and empirically show that SVG-EAR improves the quality-efficiency trade-off and increases throughput at the same generation fidelity on video diffusion tasks. Overall, SVG-EAR establishes a clear Pareto frontier over prior approaches, achieving up to 1.77× and 1.93× speedups while maintaining PSNRs of up to 29.759 and 31.043 on Wan2.2 and HunyuanVideo, respectively.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

WAN2.2 720P, Text -to-Video

[Figure 6]

[Figure 7]

Full Attention | Latency = 29 min

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

SVG -EAR |PSNR = 25.808 |Latency = 16 min

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

A narrow, cobblestone alleyway bathed in the soft glow of twilight, flanked by quaint, ivy -covered brick buildings with rustic w ooden shutters. The scene is serene, with a gentle breeze rustling the leaves of potted

[Figure 23]

[Figure 24]

[Figure 25]

plants and hanging flower baskets adorning the windowsills. Warm, golden light spills from vintage lanterns, casting intricat e shadows on the cobblestones. A solitary cat, sleek and graceful, meanders down the alley, pausing occasionally to sniff the air. The distant sound of a violin playing a melancholic tune adds to the tranquil a mbiance, creating a timeless, peaceful moment in this hidden urban gem.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

HunyuanVideo 720P, Text -to-Video

[Figure 35]

[Figure 36]

Full Attention | Latency=27 min

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

SVG -EAR |PSNR = 30.101 |Latency = 14 min

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A joyful Corgi with a fluffy coat and perky ears frolics in a sunlit park, the golden hues of sunset casting a warm glow on t he scene. The camera zooms in on the Corgi's expressive face, capturing its bright eyes and wide, happy grin. As it bounds through the grass, its short legs move with surprising speed, and its tail wags energetica lly. The park's lush greenery and the soft, amber light create a picturesque backdrop.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

The Corgi pauses to playfully chase a fluttering butterfly, its excitement palpable, before the camera focuses closely on its delighted expression, highlighting the pure joy of the moment.

Figure 1 SVG-EAR significantly accelerates video generation for Wan2.2 and HunyuanVideo. On a single NVIDIA H100 GPU, achieving 1.81× and 1.93× speedups with 26 and 30 PSNR, respectively.

*Equal Contribution

- 1 Introduction

Diffusion transformers (DiTs) have become a dominant paradigm for high-fidelity image and video generation (Kong et al., 2024; Wan et al., 2025). However, in video generation settings, the token sequence length grows rapidly with resolution and frame count, while the quadratic cost of attention quickly becomes a primary bottleneck. As a result, accelerating attention without noticeable quality degradation is increasingly critical for long-form and high-resolution video diffusion (Xi et al., 2025; Yang et al., 2025b; Zhang et al., 2025f).

A large body of work tackles this bottleneck via sparse attention. A particularly practical family exploits structured redundancy in attention maps: tokens are semantically clustered and permuted so that similar tokens become contiguous in memory, turning the attention matrix into a block structure over query clusters and key clusters. Then, only a subset of query–key blocks are computed exactly (block-sparse), while the rest are ignored or approximated, yielding hardware-friendly sparsity patterns and substantial speedups (Yang et al., 2025b). This “cluster–permute–route” pipeline is appealing in video DiTs because it aligns well with efficient kernels and preserves important structure.

Despite these advances, existing methods face a fundamental tension in block selection and the treatment of unselected blocks. Many approaches select blocks by approximated attention scores (e.g., top-k/top-p) and drop low-score blocks entirely. While intuitive, low-score blocks can still collectively carry important global context (e.g., background consistency, long-range semantic coupling, and weak but numerous dependencies). Consequently, naively discarding low-score blocks incurs non-trivial information loss and can manifest as perceptible quality degradation in video generation. Recent works such as SLA (Zhang et al., 2025c) and SLA2 (Zhang et al., 2026c) address this by pairing sparse exact attention with a learned linear branch that approximates the contributions of dropped blocks, recovering much of the lost context. However, this approach introduces additional trainable parameters and requires fine-tuning, limiting its plug-and-play applicability and sacrificing the fidelity.

A natural solution is to avoid pure dropping by leveraging the high within-block similarity revealed by clustering. For blocks that are not selected for exact computation, one can apply a parameter-free linear compensation using centroids: replace queries and keys within a cluster with their centroid and approximate the contribution of an entire block with a shared interaction. This branch requires no training and no additional parameters, and it mitigates information loss from dropped blocks.

However, linear compensation alone does not resolve the core issue. Crucially, once a compensation branch exists, conventional score-based routing becomes misaligned with the objective of controlling the final approximation error. A high-score block may be highly coherent within the cluster and thus well approximated by its centroid, making exact computation unnecessary. Conversely, a low-score block can contain diverse key-value interactions where centroid-based compensation induces substantial error. Therefore, under a fixed compute budget, the goal should not be to preserve the highest-score blocks, but to minimize the reconstruction error between full attention and its compensated counterpart, allocating exact computation to the blocks where compensation fails.

Motivated by this insight, we propose SVG-EAR, a sparse attention method that combines parameter-free linear compensation with error-aware routing. SVG-EAR first clusters queries and keys to form a block structure based on similarity. It then computes exact attention for a subset of blocks under the density budget assigned by users, and approximates the remaining blocks using linear compensation with key/value cluster centroids. Unlike prior score-based selection, SVG-EAR estimates the compensation error of each block using a lightweight probing procedure and greedily selects blocks with the highest error-to-cost ratio (i.e., the predicted compensation error normalized by the block size) under a density budget.

To make routing practical at inference time, we exploit intra-cluster similarity and use query centroids as proxies for individual queries, reducing the estimation cost from quadratic O(NqNkd) to near-linear O(CqNkd). We further implement a fused, streaming kernel to avoid materializing intermediate logits and to keep the routing overhead negligible in practice.

We validate the proposed design both theoretically and empirically. On the theory side, we provide an upper bound that relates the true attention-map error to our estimated error, characterizing its dependence on clustering quality and sequence length. On the empirical side, SVG-EAR achieves a superior error–density trade-off on video generation: at the same density, it reduces attention reconstruction error and improves generation quality, and at the same quality target it operates at lower density and delivers higher throughput

(see §5). Specifically, SVG-EAR establishes a clear Pareto frontier over prior approaches (Yang et al., 2025b; Xi et al., 2025; Zhang et al., 2025f), reaching up to 1.77× and 1.93× speedups while maintaining PSNRs of up to 29.759 (Wan2.2 (Wan et al., 2025)) and 31.043 (HunyuanVideo (Kong et al., 2024)), respectively. Overall, our results suggest that when compensation is available, the key to high-fidelity sparse attention is not “selecting high-score blocks”, but identifying where compensation breaks and prioritizing those blocks for exact computation.

We summarize our key contributions as follows:

- 1. We identify two fundamental misalignments in score-based sparse attention: (i) naively dropping low-attention-score blocks can cause substantial information loss; and (ii) once an approximation/compensation branch is introduced, block selection should not prioritize high-score blocks, but instead prioritize blocks that would otherwise incur large approximation error.
- 2. We prototype and implement a compensation-and-routing sparse attention mechanism: a parameter-free linear compensation branch that recovers contributions from uncomputed blocks via cluster means, and an error-aware routing strategy that, under a fixed budget, identifies and computes the blocks with the largest induced error, yielding a markedly improved error–density trade-off.
- 3. We further translate the design into an end-to-end system with efficient kernels and execution flow that keep the overhead negligible in practice, delivering substantial and consistent speedups on real video generation workloads while maintaining generation fidelity.

- 2 Related Work

Sparse Attention for Video Generation. Sparse attention is a leading direction for accelerating video Diffusion Transformers (Zhang et al., 2025b) because 3D spatiotemporal self-attention scales quadratically with the huge token sequence length. Sparse attention mechanisms for video generation can be roughly divided into static and dynamic approaches, based on whether the sparsity pattern is fixed offline or determined online at inference time. Static approaches typically exploit recurring structural patterns (Xi et al., 2025; Li et al., 2025b; Zhao et al., 2025). Dynamic approaches infer masks or critical-token sets per sample and timestep, commonly introducing an identification/proxy-scoring step whose overhead is usually designed to be negligible (Yang et al., 2025b; Xia et al., 2025; Chen et al., 2025a; Liu et al., 2026; Sun et al., 2025a; Wu et al., 2025a; Cai et al., 2025; Zhang et al., 2025f; Li et al., 2026, 2025a). Beyond training-free methods, newer “trainable sparse attention” lines push sparsity higher with fine-tuning and additional branch compensation (Zhang et al., 2025h,c, 2026c,b; Wu et al., 2025b; Liang et al., 2026; Agarwal et al., 2026).

Linear Attention for Diffusion Models. Linear attention and state-space alternatives are important for video diffusion because quadratic attention quickly dominates latency and memory as spatiotemporal token counts grow. Representative families can be grouped into kernelized linear attention (Chen et al., 2025b), state-space/SSM-centered designs that use structured recurrence for global mixing (Gao et al., 2024; Wang et al., 2025; Huang et al., 2025a), and hybrid local-global designs that explicitly balance a cheap linear surrogate with selective higher-fidelity pathways (Fang et al., 2026; Zhang et al., 2025c). Though these methods enable predictable memory scaling and improved feasibility of long-context generation under fixed budgets, they still suffer from low quality and long-term dependencies.

Other Optimization Techniques. Cache-based acceleration exploits redundancy across denoising steps or between conditional/unconditional branches in classifier-free guidance (Ma et al., 2024b,a; Lv et al., 2024; Kahatapitiya et al., 2025; Ma et al., 2025; Chu et al., 2025a; Bu et al., 2025; Liu et al., 2025a). Parallelization becomes essential when targeting deployment-scale throughput or higher resolutions (Li et al., 2024a; Feng et al., 2025; Fang et al., 2024a,b; Jacobs et al., 2023; Liu et al., 2023; Chu et al., 2025b). Quantization reduces memory bandwidth and leverages low-precision tensor cores to accelerate further the linear module (Wu et al.,

- 2024; Li et al., 2025c, 2024b) or attention module (Zhang et al., 2025e,a,g,d, 2026a). Distillation and few-step sampling compress long diffusion trajectories into a small number of steps, usually leading to more than 10× speedup (Lu and Song, 2025; Yin et al., 2024b,a; Zheng et al., 2025).

Autoregressive and Streaming Video Generation. Autoregressive (streaming) video diffusion generates frames (or chunks) sequentially and enables efficient KV caching, usually accompanied by unbounded temporal

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Error:

Error:

Error:

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Density: 22%

Density: 22%

Density: 17%

| | | | |[Figure 75]<br><br>| |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]| |
|---|---|---|---|---|---|
|[Figure 81]|[Figure 82]|[Figure 83]|[Figure 84]|[Figure 85]| |
|[Figure 86]|[Figure 87]| |[Figure 88]|[Figure 89]|[Figure 90]|
| |[Figure 91]|[Figure 92]|[Figure 93]|[Figure 94]|[Figure 95]|
|[Figure 96]|[Figure 97]|[Figure 98]| |[Figure 99]|[Figure 100]|
|[Figure 101]|[Figure 102]|[Figure 103]<br><br>[Figure 104]|[Figure 105]|[Figure 106]| |

|[Figure 107]|[Figure 108]|[Figure 109]|[Figure 110]|[Figure 111]| |
|---|---|---|---|---|---|
|[Figure 112]|[Figure 113]|[Figure 114]| |[Figure 115]|[Figure 116]|
|[Figure 117]|[Figure 118]| |[Figure 119]|[Figure 120]|[Figure 121]|
| |[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]| |
|[Figure 126]|[Figure 127]| |[Figure 128]|[Figure 129]|[Figure 130]|
| |[Figure 131]|[Figure 132]|[Figure 133]|[Figure 134]|[Figure 135]|

[Figure 136]

[Figure 137]

|[Figure 138]|
|---|

StrategyError

[Figure 139]

Compute Block

[Figure 140]

[Figure 141]

[Figure 142]

- (a) Original attention map

[Figure 143]

- (b) Permuted attention map

[Figure 144]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | |[Figure 145]| | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | |[Figure 146]|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | |[Figure 147]| | | |

Compensate Block

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | |[Figure 148]<br><br>|
| | | | | | |
| | | | | | |
| | | | | | |

[Figure 149]

[Figure 150]

| |
|---|

[Figure 151]

Ignore Block

[Figure 152]

| |
|---|

| |
|---|

| |
|---|

- Figure 2 Existing methods and top-p selection fall short: dropping "low-score" blocks and error-unaware block selection degrade the error-density trade-off. (a) Original attention map. (b) Permuted map after semantic-aware clustering. (c) Ignoring low-score blocks causes a large, sparse attention error. (d) Linear compensation with cluster means still yields high error due to naive top-p selection. (e) Our method improves both error and density by routing based on the gap between full computation and compensation.

horizons and interactive control. CausVid (Yin et al., 2025) and Self-Forcing (Huang et al., 2025b) exemplify this shift by converting bidirectional video diffusion into causal generation that better matches interactive inference and is consolidated through further algorithmic innovations to mitigate anti-drifting (Liu et al.,

- 2025b; Chen et al., 2026; Zhu et al., 2026; Lv et al., 2026; Yang et al., 2025a). World Models (Xi et al.,
- 2026; HunyuanWorld, 2025b; Sun et al., 2025b; HunyuanWorld, 2025a; Bruce et al., 2024; Gao et al., 2025) objectives overlap with streaming video generation but extend it toward closed-loop simulation, where an agent or user can intervene at each step.

#### 3 Motivation

- 3.1 Structured Redundancy in Attention Computation

Previous works exploit the inherent sparsity in attentions to accelerate DiTs (Xi et al., 2025; Yang et al., 2025b; Zhang et al., 2025f; Zhao et al., 2025). We first cluster semantically similar tokens and permute them so that tokens within each cluster are laid out contiguously. This allows the attention map to be partitioned into blocks that often exhibit high inner-block similarity.

As shown in figure 2 (c), the existing methods then select a subset of these blocks for exact attention ranked by their approximated attention scores, while simply ignoring the low score blocks. However, this approach can lead to significant information loss, as the ignored blocks can also contain important contextual information, and the limited budget may not allow for their inclusion.

Fortunately, the similarities within each block naturally provide an opportunity to efficiently approximate the skipped blocks’ contributions without any training and additional parameters. As shown in figure 2 (d), a simple strategy here is to use the mean of the remaining blocks in each cluster to fill in the gaps left by the ignored blocks. Given a block to compensate, suppose the mean of the key-value tokens is k¯. The attention logits for all entries in the block interacted with a query qi can be then computed as qik¯T/

√

d, where d is the dimensionality of the token vectors.

- 3.2 Existing Block Selection Fails to Control Output Error

While linear compensation improves the error-density trade-off, its effectiveness ultimately depends on how the blocks are selected. Existing block selection strategies are score-based and thus misaligned with the

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | |[Figure 153]| | | |
| | | | | | |
| | | | | | |

|| |
|---|
<br><br>|| |
|---|
|
|---|
<br><br>| |
|---|
| |
| |
| |
| |
<br><br>| | | | | | | |
|---|---|---|---|---|---|---|
<br><br>|| |
|---|
|
|---|
<br><br>| | | | | | | |
|---|---|---|---|---|---|---|
<br><br>[Figure 154]<br><br>Error<br><br>| | | |
|---|---|---|
| | | |
<br><br>[Figure 155]<br><br>Key tokens<br><br>| |
|---|
<br><br>[Figure 156]<br><br>Querytokens<br><br>| |
|---|
<br><br>[Figure 157]<br><br>Error -cost<br><br>[Figure 158]<br><br>[Figure 159]<br><br>| |
|---|
| |
<br><br>| |
|---|
|
|---|

[Figure 160]

Budget

Querytokens

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

(a) Permuted attention map (b) Block error estimation

(c) Error -to-size ratio map

- Figure 3 Overview of SVG-EAR (a) The attention map after semantic-aware clustering. (b) Block error estimation. Using the cluster mean as a proxy for individual queries, the block error is computed via the sum of squared differences

between exponentiated logits of individual keys kj and the key mean k¯, then normalized by the block area to determine the error-to-size ratio. (c) Blocks with the highest error-to-size ratio are greedily selected for exact attention within the budget, while the rest are assigned to linear compensation.

objective of controlling the final output error when linear compensation is introduced. For example, a block with high attention scores may exhibit strong inner-block similarity, making it well suited for linear compensation. In contrast, a block with low attention scores can contain diverse key-value interactions, where linear compensation induces substantial approximation error. To maximize fidelity under a fixed compute budget, the goal should be to minimize the reconstruction error of the attention map. As shown in figure 2 (e), our proposed approach routes computation based on the gap between the full attention map and its linearly compensated counterpart. As a result, by directly optimizing an error-aware objective and prioritizing the blocks that contribute most to the reconstruction error, we achieve a superior error-density trade-off, as detailed below.

- 4 Methodology

In this section, we first formulate the problem based on our parameter-free linear compensation. We then introduce SVG-EAR, an error-aware routing method with theoretical guarantees and low overhead. Our key insight is that the error between the full attention map and its linearly compensated counterpart can be accurately estimated via a lightweight probing algorithm, which enables precise attention routing. We further show how the linear compensation leverages value-cluster means to produce the final output efficiently. Finally, we implement a fused kernel that estimates the error using a streaming update.

- 4.1 Problem Formulation

k×d. We cluster Q and K using flash k-means (Yang et al., 2025b), and obtain Cq and Ck clusters, respectively. For each query qi and key kj, we use q¯i and k¯j to denote the mean token in the cluster that contains qi and kj, respectively. We denote Q¯ ∈ RN

Given a transformer attention layer with query tokens Q ∈ RN

q×d and key tokens K ∈ RN

q×d and K¯ ∈ RN

k×d as the matrices of per-token cluster means for Q and K, respectively; that is, the i-th row of Q¯ equals q¯i (the centroid of the cluster containing qi), and similarly the j-th row of K¯ equals k¯j.

Our core idea here is to approximate part of the attention computation linearly with cluster means. For a query qi, when approximating its interaction with a cluster of keys kc = {kj}, we replace each key kj in the logit computation with the corresponding cluster mean k¯j. This gives an approximate logit qik¯jT/

√

d, which is

shared across all keys within the same cluster. We use a binary routing mask M ∈ {0,1}N

q×Nk for each query qi. If Mj(i) = 1, we compute the interaction

between qi and kj exactly. If Mj(i) = 0, we approximate it using the cluster mean. To match ML acceleratorfriendly sparsity, where the tokens in the same cluster will be permutated continuously and processed together,

the mask should be applied on the query-key block level: All queries in the same cluster share the same mask. For a fixed query, all keys in the same cluster share the same mask value. With this mask, we write the sparse

attention map as

A˜(sparseM) = softmax

QKT √

d ⊙ M +

QK¯T √

d ⊙ 1 − M (1)

Given a budget ρ (i.e., density), our goal is to find a mask M that minimizes the mean squared error (MSE) between the full attention map and the sparse attention map, subject to the constraint that the number of non-zero entries in M is at most ρNqNk. Formally, we formulate the optimization problem as:

∥A˜(sparseM) − Afull∥2F s.t. ∥M∥2F ≤ ρNqNk, (2)

min

M

T

where ∥ · ∥F denotes the Frobenius norm and Afull = softmax QK

d is the full attention map.

√

- 4.2 Error-aware Routing

Relaxed optimization problem. The above optimization problem is difficult to solve efficiently at routing time because the softmax normalizer couples all keys for each query, and Afull and A˜(sparseM) use different normalization terms. In particular, for any query, altering the routing decision for any single key cluster changes the normalizer and thus changes the attention weights assigned to all other keys. Consequently, selecting an optimal block-level mask cannot be reduced to independent per-block choices; in the worst case it requires searching over a combinatorial space of size up to O(2C

q·Ck), which is intractable.

To obtain a practical and block-separable proxy, we therefore relax the objective by focusing on the exponentiated logits as follows:

min

M

i,j

qik¯jT √

qikjT √

1 − Mj(i) exp(

) − exp(

d

) 2 s.t. ∥M∥2F ≤ ρNqNk. (3)

d

Intuitively, this relaxation shifts the objective from finding a mask M that minimizes the discrepancy between two normalized attention maps to minimizing the discrepancy between their unnormalized attention weights. Although this relaxation may fail to identify masks whose near-optimality only emerges after normalization, it still provides a useful proxy for routing decisions. This is because, in our setting, the variation in the softmax normalizer is practically limited for two reasons: (1) semantic clustering ensures that each kj is close to its centroid k¯j, reducing deviation in the logits; (2) the unmasked tokens typically receive high attention scores, so they dominate the normalization term, thereby limiting the impact of the remaining masked tokens on the normalizer.

Oracle solution. For the relaxed optimization problem, an oracle solution can in principle be obtained by selecting the query–key blocks that incur the largest squared errors under the budget constraint, which reduces to a tractable 0–1 knapsack problem. Specifically, let

ϵ2i,j = exp

qik¯jT √

d − exp

qikjT √

d

2 (4)

denote the squared error for the entry (i,j). Each query-key block (qc,kc) can then be treated as an item in the knapsack formulation, where the value is given by (i,j)∈(q

c,kc) ϵ2i,j, and the weight corresponds to its size |qc||kc|. After that, the problem can be solved using standard knapsack algorithms or well approximated by greedy methods that select blocks in descending order of error-to-cost ratio (i,j)∈(qc,kc) ϵ2i,j

|qc||kc| until the budget is exhausted.

The oracle solution is fully error-aware; however, it is not practical. Computing the exact per-entry error requires evaluating the full attention logits for all token pairs, which incurs the same O(NqNkd) quadratic cost as full attention itself. In other words, the oracle depends on precisely the information that sparse attention is designed to avoid computing.

Error estimation. To achieve error-aware routing without incurring the prohibitive cost of computing the exact error, we propose to estimate the error using a lightweight probing algorithm. Again, we leverage the inherent similarity within each query token cluster, which allows us to use the cluster mean as a proxy for the individual queries in the error estimation. Specifically, we define the following estimated error for each entry (i,j):

q ¯ik¯jT √

q ¯ikjT √

2 (5)

ϵˆ2i,j = exp

d − exp

d

Because queries within the same cluster share a common estimated error, the computational complexity reduces to O(CqNkd), which is negligible in practice when an optimized kernel implementation is used.

We then use the estimated errors to determine whether each query-key block is assigned to exact attention or linear compensation. Specifically, we first aggregate the estimated errors within each block. We then greedily select the blocks with the highest error-to-size ratio (i,j)∈(q

c,kc) ϵˆ2i,j/|qc||kc| for exact attention until the budget is exhausted, while assigning the remaining blocks to linear compensation.

###### Proposition 1. Let δq2 = N1

q i ∥qi − q¯i∥22 denote the average squared ℓ2 error between each query and its cluster mean, and let Kmax denote the maximum ℓ2 norm of the key tokens. Given any mask M that does not significantly perturb the attention normalizers, we have:

1 NqNk ∥A˜(sparseM) − Afull∥2F ≤

ϵˆ2i,j Zi2

2 NqNk i,j

(1 − Mj(i))

+

8δq2Kmax2 Nkd

(6)

, where Zi denotes the softmax normalizer for query i.

See Appendix A for the proof and the formal statement of the normalizer stability assumption. The left-hand side is the true MSE on the attention map, while the right-hand side consists of the estimated MSE (using ϵˆ2i,j) scaled by a constant factor, plus a residual term. The residual 8δ

2 qKmax2

Nkd scales linearly with the clustering error δq2 and decreases inversely with the sequence length Nk. Therefore, as clustering improves (i.e., δq2 → 0) or the sequence length increases, the bound becomes asymptotically tight, establishing that our error estimation is both theoretically safe and quantitatively controllable.

- 4.3 Linear Compensation

After determining the mask M, we compute the final attention output by combining exact and approximated contributions. Let V ∈ RN

k×d denote the value tokens, which follow the same clustering assignment as K, and let V¯ ∈ RN

k×d denote the matrix of mean value tokens, where each v¯j is the average of V over the key cluster containing vj. The sparse attention output is then given by:

O˜sparse(M) = A ˜(sparseM) ⊙ M V + A ˜(sparseM) ⊙ (1 − M) V ¯ (7)

For blocks where Mj(i) = 1, the output is computed with the original values V . For blocks where Mj(i) = 0, since all keys within the cluster share the same approximate attention weight, the contribution of a key cluster

kc reduces to |kc| · A˜(sparseM) (i,j) · v¯j. This significantly improves computational efficiency: for a given query and masked block, we only need to compute a single logit using the cluster mean k¯j and a single scalar-vector product with the mean value v¯j. Hence, the overall time complexity of SVG-EAR’s attention computation is reduced to O (NqCk + ρNkNq) · d .

Moreover, since the linear compensation also considers the value tokens, we can further improve the error estimation above by incorporating each vj and v¯j into the analysis. A value-aware error estimation is thus given by:

q ¯ik¯jT √

q ¯ikjT √

ϵ˜2i,j = ∥exp

vj∥22 (8)

v ¯j − exp

d

d

which can be computed in the same complexity O(CqNkd) as Equation 5 while providing a practically more accurate estimation.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Querypositions

Querypositions

Querypositions

Querypositions

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Key positions Key positions Key positions Key positions

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Querypositions

Querypositions

Querypositions

Querypositions

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Key positions Key positions Key positions Key positions

[Figure 192]

[Figure 193]

[Figure 194]

Error -aware routing w/ (d) mean compensation

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

(a) Original attention map

(b) Permuted attention map (c) Top -p selection (SVG2)

[Figure 201]

- Figure 4 Visualization of attention maps from specific attention heads in Wan2.2 during text-to-video generation. (a) Original attention maps with sparse patterns, using heatmaps to represent attention weights. (b) Permuted attention maps. Following SVG2, we permute the attention maps such that attention begins to cluster into specific blocks. (c) Applying SVG2’s top-p selection to these permuted attention maps, which ignores the sparse attention mechanisms of the remaining blocks. (d) Applying our Error-aware Routing and Mean Compensation mechanisms to the permuted attention maps, achieving higher attention map similarity.

- 4.4 Efficient Kernel Implementation

The error estimation procedure is inherently memory I/O-bound, as naively materializing all logits requires repeatedly writing intermediate results to HBM. To address this bottleneck, we design a custom kernel based on a streaming update scheme. Specifically, instead of materializing all logits in HBM and computing the squared errors afterward, we fuse these steps by expanding the squared terms and maintaining the required running statistics on the fly. To ensure numerical stability, we subtract the running maximum from the logits prior to exponentiation. This design reduces HBM accesses to O(NkCqd · S−1), where S denotes the SRAM size, while incurring only a small constant-factor increase in FLOPs within the same O(NkCqd) complexity. For the linear compensation part, we reuse SVG2’s (Yang et al., 2025b) customized attention kernel that accepts dynamic block sizes as inputs. See Appendix B for more implementation details.

- 5 Experiments

Models. We evaluate SVG-EAR on open-sourced state-of-the-art video generation models, including Wan2.2I2V/T2V-A14B (Wan et al., 2025), and HunyuanVideo-T2V-13B (Kong et al., 2024) to generate videos with 720p resolution. After being tokenized by 3D-VAE, Wan2.2 generates 21 frames with 3,600 tokens per frame, while HunyuanVideo processes 33 frames with 3,600 tokens per frame.

Metrics. We assess the similarity of the generated video compared to full attention using the following metrics: Peak Signal-to-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS), and Structural Similarity Index Measure (SSIM). We use VBench (Huang et al., 2024) to evaluate the video quality. To quantify the efficiency of sparse attention mechanisms (i.e., computational budget), we use density, which is defined as the sparse attention computation divided by the full attention computation. To assess end-to-end efficiency, we test the Speedup Ratio, which is calculated as the inference time of the full attention divided by the inference time of the sparse attention.

###### Table 1 Quality and efficiency benchmarking results of SVG-EAR and baselines, where the best results are highlighted, and the second-best results are underlined.

Config PSNR↑ SSIM↑ LPIPS↓ ImgQual↑ SubCons↑ Density↓ FLOP↓ Speedup↑ Wan 2.2 14B, 720P, I2V - - - 0.704 0.960 100% 658.46 PFLOPS 1×

SpargeAttn 27.140 0.883 0.116 0.703 0.958 30.15% 396.83 PFLOPS 1.58× SVG 25.297 0.844 0.139 0.703 0.958 30.25% 397.20 PFLOPS 1.58× SVG2 27.668 0.888 0.117 0.701 0.958 29.38% 393.95 PFLOPS 1.61× SVG-EAR 29.759 0.918 0.093 0.704 0.959 23.64% 378.88 PFLOPS 1.61× SVG-EAR-Turbo 28.344 0.900 0.108 0.702 0.958 20.42% 363.85 PFLOPS 1.77× Wan 2.2 14B, 720P, T2V - - - 0.706 0.916 100% 658.46 PFLOPS 1×

SpargeAttn 20.872 0.708 0.242 0.708 0.916 30.15% 396.83 PFLOPS 1.58× SVG 19.455 0.654 0.292 0.712 0.912 30.25% 397.20 PFLOPS 1.59× SVG2 23.556 0.802 0.183 0.705 0.914 32.30% 404.88 PFLOPS 1.57× SVG-EAR 24.995 0.841 0.153 0.706 0.915 25.95% 387.53 PFLOPS 1.59× SVG-EAR-Turbo 23.940 0.814 0.174 0.705 0.915 22.25% 370.71 PFLOPS 1.75× Hunyuan 13B, 720P, T2V - - - 0.665 0.904 100% 612.38 PFLOPS 1×

SpargeAttn 24.589 0.796 0.232 0.629 0.908 40.09% 389.76 PFLOPS 1.38× SVG 27.325 0.880 0.140 0.665 0.905 29.92% 351.97 PFLOPS 1.57× SVG2 29.445 0.911 0.112 0.654 0.901 26.21% 299.02 PFLOPS 1.89× SVG-EAR 31.043 0.928 0.092 0.659 0.903 22.17% 281.86 PFLOPS 1.93×

Datasets. For text-to-video generation, we adopt the prompt in Penguin Benchmark after prompt optimization provided by the VBench team. For image-to-video generation, we adopt the prompt-image pairs provided by VBench with prompt enhancement. We evaluate the video quality on a subset of 50 samples randomly selected from text-to-video and image-to-video datasets, respectively.

Baselines. We compare SVG-EAR against state-of-the-art sparse attention algorithms, including static method Sparse VideoGen (SVG) (Xi et al., 2025), and dynamic methods Sparse VideoGen 2 (SVG2) (Yang et al., 2025b) and SpargeAttention (Zhang et al., 2025f). We do not include SLA (Zhang et al., 2025c, 2026c) because it requires extra training, which makes similarity metrics not comparable. We provide the details of each method in Appendix D.

Implementation. We implement SVG-EAR on top of SVG2 (Yang et al., 2025b), inheriting its budget allocation strategy and default hyperparameters. Since SVG2 assigns an adaptive compute budget to each query cluster, our routing algorithm operates at the granularity of individual query clusters accordingly. By reducing the number of centroids and the density budgets, we developed SVG-EAR-turbo that matches the efficiency of SVG2-turbo while consistently outperforming SVG2 in quality.

- 5.1 Quality Evaluation

We present the attention maps of specific heads during the generation process of Wan2.2 Text-to-Video on prompts from VBench, exhibiting various sparse patterns as shown in Fig. 4. Compared to the original top-p selection, our method covers a broader range and allocates attention to marginal tokens with relatively small attention weights. The attention maps generated by our method are more similar to the permuted maps, particularly in cases where the attention weight distribution is relatively uniform.

The quantitative results summarized in Table 1 demonstrate that SVG-EAR consistently outperforms all baselines across PSNR, LPIPS, and SSIM, while simultaneously achieving the highest speedup. Specifically, SVG-EAR

Torch Implementation Ours

Clustering/Permutation Error-aware routing/Compensation

Others Linear

Attention

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

| | | | |
|---|---|---|---|
|5.69×<br><br>8.37×<br><br>13.74×<br><br>dense| | | |
| | | | |
| | | | |
| | | | |

- 11.8% 88.2%
- 12.9% 19.3% 67.7% 6.5%

DENSE

- 103
- 104

Time(s)

SVG2

12.3% 19.0% 62.2%

OURS

(100, 300) (200, 500) (300, 1000)

0 200 400 600 800 1000 1200 1400 1600

(# of Q Cluster, # of K Cluster)

Time (s)

(a) Generation latency breakdown

(b) Efficiency of Custom Kernels

- Figure 5 Efficiency evaluation. (a) illustrates the latency breakdown of different components during a single complete inference process of Wan2.2 T2V 720p, compared with the vanilla implementation and SVG2. (b) presents an end-to-end latency comparison between our efficient Triton implementation and the native PyTorch version.

achieves an average PSNR of 24.995 on Wan2.2 T2V and 31.043 on HunyuanVideo. Furthermore, compared to the existing SOTA SVG2 (Yang et al., 2025b), SVG-EAR establishes a new Pareto frontier, representing a superior trade-off between generation quality and inference efficiency. For a more comprehensive evaluation, see Appendix E. Moreover, for reference-free metrics, we present representative VBench metrics in Table 1, while the comprehensive evaluation is detailed in Appendix C.

- 5.2 Efficiency Evaluation

End-to-end speedup evaluation. We evaluate the efficiency of SVG-EAR by measuring the end-to-end inference time speedup compared to full attention and total FLOPS, as shown in Table 1. SVG-EAR achieves a speedup of 1.59× on Wan2.2 T2V and 1.93× on HunyuanVideo, higher than all baselines. Notably, SVG-EAR-turbo achieves a speedup of 1.75× on Wan2.2 T2V, while maintaining superior quality compared to all baselines.

Kernel efficiency evaluation. We evaluated the computational overhead of our method. As shown in Fig. 5 (a), our approach accounts for 6.5% of the total end-to-end latency during Wan2.2 text-to-video (720p) inference. While the latency reduction is minor, the results are noticeably better in visual fidelity and overall quality. The efficiency is driven by our custom Triton kernel, which achieves up to a 13.74× speedup compared to the original PyTorch implementation as shown in Fig. 5 (b). This allows our method to enhance generation quality while maintaining a high inference speed.

- 5.3 Error Analysis

Strategy of block selection. To access the error-density trade-off of the three methods illustrated in Fig. 2, i.e., top-p selection (SVG2), top-p selection with linear compensation, and error-aware routing with linear compensation, we compute the mean squared error between each method’s attention map and the full attention map, i.e., N 1

q×Nk ∥Asparse − Afull∥2F. Following the primary experimental settings with QC = 300 and KC = 1,000, we conduct this evaluation by sampling a representative set of attention maps across the generation process. As shown in Fig. 6 (a), our error-aware routing yields the attention map most consistent with the full attention, while the top-p-based compensation only achieves a marginal error reduction against SVG2.

Effect of clustering. We provide a theoretical guarantee that relates attention reconstruction error to clustering quality of query tokens in §4. To evaluate the effect, we conduct inference on an additional sample with KC = 1,000 and p = 0.85. As shown in Fig. 6 (b), we compute the clustering error δq2 and the attention MSE across varying numbers of query centroids (i.e., QC). We observe a significant reduction in attention error as QC increases, demonstrating that improved clustering quality directly enhances the approximation

Top-p w/o Compensation

Top-p w/ Compensation Error-aware

- 1 × 10−8

- 2 × 10−8

- 3 × 10−8

- 4 × 10−8

AttentionMSE

0

10 30 50 70 90

Density

(a) Attention MSE vs. Density

Clustering error δq2 Attention MSE

2δClusteringerrorq

##### AttentionMSE

10−8

110

8 × 10−9

105

6 × 10−9

100

4 × 10−9

95

2 × 10−9

90

100 200 300 400 500

Number of Q clusters

(b) Clustering Error

- Figure 6 Error Analysis. (a) Comparison of attention MSE versus density (i.e., number of sparse computation normalized by total computation) for top-p selection, top-p selection with linear compensation, and error-aware routing

with linear compensation. (b) Both the clustering error δq2 (solid black line) and attention MSE (dashed red line) decrease as the number of Q clusters increases, demonstrating that higher clustering quality leads to a tighter error bound and better approximation.

accuracy. This empirical observation aligns with our theoretical guarantees, where the attention error relates to the clustering quality.

#### 6 Conclusion and Limitation

We presented SVG-EAR, a training-free approach to accelerate DiT-based video generation under sparse attention. SVG-EAR recovers the contribution of skipped attention blocks via a parameter-free centroid compensation branch, leveraging the strong similarity structure of keys and values after semantic clustering. To avoid the small fraction of blocks where compensation is inaccurate, we further introduced error-aware routing that prioritizes blocks with the highest error-to-cost ratio under a fixed density budget. We provided theoretical guarantees linking attention reconstruction error to clustering quality, and demonstrated empirically that SVG-EAR improves the quality–efficiency trade-off. The main limitation of this paper is the lack of discussion and evaluation of whether the proposed method extends to attention mechanisms beyond DiTs.

#### References

Krish Agarwal, Zhuoming Chen, Cheng Luo, Yongqi Chen, Haizhong Zheng, Xun Huang, Atri Rudra, and Beidi Chen. Monarchrt: Efficient attention for real-time video generation. arXiv preprint arXiv:2602.12271, 2026.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

Jiazi Bu, Pengyang Ling, Yujie Zhou, Yibin Wang, Yuhang Zang, Dahua Lin, and Jiaqi Wang. Dicache: Let diffusion model determine its own cache. arXiv preprint arXiv:2508.17356, 2025.

Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.

Aiyue Chen, Yaofu Liu, Junjian Huang, Guang Lian, Yiwu Yao, Wangli Lan, Jing Lin, Zhixin Ma, Tingting Zhou, and Harry Yang. Rainfusion2. 0: Temporal-spatial awareness and hardware-efficient block-wise sparse attention. arXiv preprint arXiv:2512.24086, 2025a.

Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025b.

Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026.

Huanpeng Chu, Wei Wu, Guanyu Feng, and Yutao Zhang. Omnicache: A trajectory-oriented global perspective on training-free cache reuse for diffusion transformer models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16302–16312, 2025a.

Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18475–18486, 2025b.

Jiarui Fang, Jinzhe Pan, Aoyu Li, Xibo Sun, and Jiannan Wang. Pipefusion: Patch-level pipeline parallelism for diffusion transformers inference. arXiv preprint arXiv:2405.14430, 2024a.

Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024b.

Tongcheng Fang, Hanling Zhang, Ruiqi Xie, Zhuo Han, Xin Tao, Tianchen Zhao, Pengfei Wan, Wenbo Ding, Wanli Ouyang, Xuefei Ning, et al. Salad: Achieve high-sparsity attention via efficient linear attention tuning for video diffusion transformer. arXiv preprint arXiv:2601.16515, 2026.

Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song

- Han, et al. Streamdiffusionv2: A streaming system for dynamic and interactive video generation. arXiv preprint arXiv:2511.07399, 2025.

Jianxiong Gao, Zhaoxi Chen, Xian Liu, Junhao Zhuang, Chengming Xu, Jianfeng Feng, Yu Qiao, Yanwei Fu, Chenyang Si, and Ziwei Liu. Longvie 2: Multimodal controllable ultra-long video world model. arXiv preprint arXiv:2512.13604, 2025.

Yu Gao, Jiancheng Huang, Xiaopeng Sun, Zequn Jie, Yujie Zhong, and Lin Ma. Matten: Video generation with mamba-attention. arXiv preprint arXiv:2405.03025, 2024.

Jiancheng Huang, Gengwei Zhang, Zequn Jie, Siyu Jiao, Yinlong Qian, Ling Chen, Yunchao Wei, and Lin Ma. M4v: Multi-modal mamba for text-to-video generation. arXiv preprint arXiv:2506.10915, 2025a.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025b.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Team HunyuanWorld. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint, 2025a.

Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025b.

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Chenyang Zhang, Michael S Ryoo, and Tian Xie. Adaptive caching for faster video generation with diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15240–15252, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Haopeng Li, Shitong Shao, Wenliang Zhong, Zikai Zhou, Lichen Bai, Hui Xiong, and Zeke Xie. Pisa: Piecewise sparse attention is wiser for efficient diffusion transformers. arXiv preprint arXiv:2602.01077, 2026.

Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7183–7193, 2024a.

Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024b.

Xiaolong Li, Youping Gu, Xi Lin, Weijie Wang, and Bohan Zhuang. Psa: Pyramid sparse attention for efficient video understanding and generation. arXiv preprint arXiv:2512.04025, 2025a.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, and Song Han. Radial attention: o(n log n) sparse attention with energy decay for long video generation, 2025b. https://arxiv.org/abs/2506.19852.

Zhiteng Li, Hanxuan Li, Junyi Wu, Kai Liu, Haotong Qin, Linghe Kong, Guihai Chen, Yulun Zhang, and Xiaokang Yang. Dvd-quant: Data-free video diffusion transformers quantization. arXiv preprint arXiv:2505.18663, 2025c.

Cheng Liang, Haoxian Chen, Liang Hou, Qi Fan, Gangshan Wu, Xin Tao, and Limin Wang. Vmonarch: Efficient video diffusion transformers with structured attention. arXiv preprint arXiv:2601.22275, 2026.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7353–7363, 2025a.

- Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025b.

Yuxi Liu, Yipeng Hu, Zekun Zhang, Kunze Jiang, and Kun Yuan. Mixture of distributions matters: Dynamic sparse attention for efficient video diffusion transformers. arXiv preprint arXiv:2601.11641, 2026.

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models, 2025. https:

//arxiv.org/abs/2410.11081.

Chengtao Lv, Yumeng Shi, Yushi Huang, Ruihao Gong, Shen Ren, and Wenya Wang. Light forcing: Accelerating autoregressive video diffusion via sparse attention. arXiv preprint arXiv:2602.04789, 2026.

Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. arXiv preprint arXiv:2410.19355, 2024.

Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching. Advances in Neural Information Processing Systems, 37:133282–133304, 2024a.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15762–15772, 2024b.

Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, and Qi Tian. Magcache: Fast video generation with magnitudeaware cache. arXiv preprint arXiv:2506.09045, 2025.

Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Zhao Jin, Jingyi Liao, Shunyu Liu, and Dacheng Tao. Vorta: Efficient video diffusion via routing sparse attention. arXiv preprint arXiv:2505.18809, 2025a.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world model. arXiv preprint, 2025b.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Hongjie Wang, Chih-Yao Ma, Yen-Cheng Liu, Ji Hou, Tao Xu, Jialiang Wang, Felix Juefei-Xu, Yaqiao Luo, Peizhao Zhang, Tingbo Hou, et al. Lingen: Towards high-resolution minute-length text-to-video generation with linear computational complexity. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2578–2588, 2025.

Jianzong Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengfei Wan, Di Zhang, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025a.

Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Ptq4dit: Post-training quantization for diffusion transformers. Advances in neural information processing systems, 37:62732–62755, 2024.

Xinjian Wu, Hongmei Wang, Yuan Zhou, and Qinglin Lu. Usv: Unified sparsification for accelerating video diffusion models. arXiv preprint arXiv:2512.05754, 2025b.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, Jianfei Chen, Ion Stoica, Kurt Keutzer, and Song Han. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity, 2025. https://arxiv.org/abs/2502.01776.

Haocheng Xi, Shuo Yang, Yilong Zhao, Muyang Li, Han Cai, Xingyang Li, Yujun Lin, Zhuoyang Zhang, Jintao Zhang, Xiuyu Li, et al. Quant videogen: Auto-regressive long video generation via 2-bit kv-cache quantization. arXiv preprint arXiv:2602.02958, 2026.

Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Training-free and adaptive sparse attention for efficient long video generation, 2025. https://arxiv.org/abs/2502.21079.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025a.

Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025b.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024b.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. In International Conference on Machine Learning (ICML), 2025a.

Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Haoxu Wang, Pengle Zhang, Huiqiang Jiang, Haofeng Huang, Chendong Xiang, et al. Efficient attention methods: Hardware-efficient, sparse, compact, and linear attention. 2025b.

Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, et al. Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention. arXiv preprint arXiv:2509.24006, 2025c.

Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training. arXiv preprint arXiv:2505.11594, 2025d.

Jintao Zhang, Jia Wei, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In International Conference on Learning Representations (ICLR), 2025e.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025f.

Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, and Jianfei Chen. Sageattention2++: A more efficient implementation of sageattention2. arXiv preprint arXiv:2505.21136, 2025g.

Jintao Zhang, Marco Chen, Haoxu Wang, Kai Jiang, Ion Stoica, Joseph E. Gonzalez, Jianfei Chen, and Jun Zhu. Sagebwd: A trainable low-bit attention. arXiv preprint arXiv:2603.02170, 2026a.

Jintao Zhang, Kai Jiang, Chendong Xiang, Weiqi Feng, Yuezhou Hu, Haocheng Xi, Jianfei Chen, and Jun Zhu. Spargeattention2: Trainable sparse attention via hybrid top-k+ top-p masking and distillation fine-tuning. arXiv preprint arXiv:2602.13515, 2026b.

Jintao Zhang, Haoxu Wang, Kai Jiang, Kaiwen Zheng, Youhe Jiang, Ion Stoica, Jianfei Chen, Jun Zhu, and Joseph E Gonzalez. Sla2: Sparse-linear attention with learnable routing and qat. arXiv preprint arXiv:2602.12675, 2026c.

Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389, 2025h.

Tianchen Zhao, Ke Hong, Xinhao Yang, Xuefeng Xiao, Huixia Li, Feng Ling, Ruiqi Xie, Siqi Chen, Hongyu Zhu, Yichong Zhang, et al. Paroattention: Pattern-aware reordering for efficient sparse and quantized attention in visual generation models. arXiv preprint arXiv:2506.16054, 2025.

Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

## Appendix

- A Proofs

- Proposition 2. Let δq2 = N1

q i ∥qi−q¯i∥22 denote the average squared ℓ2 error between each query and its cluster mean, and let Kmax denote the maximum ℓ2 norm of the key tokens. Given any mask M that does not significantly perturb the attention normalizers. Here, we say a mask M does not significantly perturb the attention normalizers if, for all queries i, the normalizer remains approximately unchanged when replacing any masked

T

k¯j−kj)

T √ j

d ) (where Mj(i) = 0) with its interpolated counterpart exp (1−t)qi+t1(¯qi−qi) (1−t)kj+t2(

entry exp(qik

√

d for any t1,t2 ∈ [0,1]. We have the following error bound:

ϵˆ2i,j Zi2

8δq2Kmax2 Nkd

1 NqNk ∥A˜(sparseM) − Afull∥2F ≤

2 NqNk i,j

(1 − Mj(i))

+

, where Zi denotes the softmax normalizer for query i.

Proof. Given the normalizer stability assumption, for entries i,j with Mj(i) = 0 we have:

1 Zi

∆Ai,j ≈

exp

qikjT √

d − exp

qik¯jT √

d

fj(qi) Zi

=

(9)

, and for entries i,j with Mj(i) = 0 we have ∆Ai,j ≈ 0. We then have:

fj2(qi) = fj(qi) − fj(¯qi) + fj(¯qi) 2 ≤ 2 fj(qi) − fj(¯qi) 2 + 2 fj(¯qi) 2 (10)

≤ 2 fj(qi) − fj(¯qi) 2 + 2ˆϵ2i,j (11) By applying Cauchy-Schwarz inequality on the integral over dt, we obtain:

2 fj(qi) − fj(¯qi) 2 ≤ 2∥qi − q¯i∥22

1

∥∇fj q(t) ∥22 dt (12)

0

, where q(t) = (1 − t)qi + tq¯i is the linear interpolation between qi and q¯i. Given that

q(t)kjT √

q(t)kjT √

1 √

k ¯j (13)

kj − exp

∇fj(q(t)) =

exp

d

d

d

, by applying the triangle inequality, we have:

Kmax √

∥∇fj(q(t))∥2 ≤

d

Combining (12) and (14), we obtain:

exp

q(t)kjT √

d

+ exp

q(t)k¯jT √

d

(14)

2 fj(qi) − fj(¯qi) 2 ≤ 2∥qi − q¯i∥22

1

Kmax2 d

0

exp

q(t)kjT √

d

+ exp

q(t)k¯jT √

d

2

dt (15)

Summing over all keys j with Mj(i) = 0, we obtain:

2 Zi2

fj(qi) − fj(¯qi) 2 (16)

j,Mj(i)=0

q(t)k¯jT √

q(t)kjT √

1

Kmax2 d

2

/Zi2 + exp

/Zi2

≤ 2∥qi − q¯i∥22

dt (17)

exp

d

d

0

j,Mj(i)=0

###### d /Zi2 and exp q(t)k¯

T √ j

T √ j

By the normalizer stability assumption, exp q(t)k

d /Zi2 can be regarded as pesudoprobabilities Pj(t) and P¯j(t), respectively. This leads to:

2 Zi2

fj(qi) − fj(¯qi) 2 (18)

j,Mj(i)=0

1

Kmax2 d j

Pj(t) + P¯j(t) 2dt (19)

≤ 2∥qi − q¯i∥22

0

8δq2Kmax2 d

1

4Kmax2 d

≤ 2∥qi − q¯i∥22

(20)

dt =

0

By summing up over all queries i, we obtain: 1 NqNk ∥A˜(sparseM) − Afull∥2F ≤

ϵˆ2i,j Zi2

2 NqNk i,j

(1 − Mj(i))

+

8δq2Kmax2 Nkd

(21)

, which completes the proof.

| |
|---|

- B Kernel Implementation Details

To maximize hardware efficiency and minimize memory overhead, both the error estimation (Algorithm 1) and fused attention (Algorithm 2) algorithms are implemented as highly optimized GPU kernels using Triton. For clarity, the batch and head dimensions are omitted from the pseudocode, as they can be trivially parallelized across the computation grid.

- Algorithm 1 Error Estimation Kernel

Require: Q¯ ∈ RQ

c×D, K,¯ V¯ ∈ RK

c×D

Require: K,V,Sj Ensure: E ∈ RQ

c×Kc

- 1: Si,j ← Q¯iK¯

⊤ √ j

D ,mi ← maxj Si,j

- 2: Yi,j ← exp(Si,j − mi)V¯j
- 3: Ei,j ← ∥Yi,j∥22,mlocali,j ← mi
- 4: for each i ∈ {1,...,Qc} do
- 5: for each j ∈ {1,...,Kc} do
- 6: for (k,v) ∈ Sj do
- 7: s ← Q¯

√ik⊤ D

- 8: mnew ← max(mlocali,j ,s)
- 9: α ← exp(mlocali,j − mnew)
- 10: Ei,j ← Ei,j · α2,Yi,j ← Yi,j · α
- 11: Ei,j ← Ei,j + ∥exp(s − mnew)v − Yi,j∥22
- 12: mlocali,j ← mnew
- 13: end for
- 14: Ei,j ← Ei,j · exp(2 · (mi − mlocali,j ))
- 15: end for
- 16: end for
- 17: return E

- Algorithm 2 Fused Attention Kernel

Require: Q ∈ RS×D,K,¯ V¯ ∈ RK

c×D Require: M ∈ RQ

c×Kc,W ∈ RK

,Cc

c

Require: O ∈ RS×D,LSE ∈ RS Ensure: Out ∈ RS×D

- 1: for each cluster c ∈ {1,...,Qc} do
- 2: for each query i ∈ Cc do
- 3: mi ← LSEi, li ← 1.0
- 4: for each j s.t. Mc,j = False do
- 5: Si,j ← QiK¯

⊤ √ j

D + ln(Wj)

- 6: mnew ← max(mi,Si,j)
- 7: α ← exp(mi − mnew)
- 8: p ← exp(Si,j − mnew)
- 9: li ← li · α + p
- 10: acci ← acci · α + pV¯j
- 11: mi ← mnew
- 12: end for
- 13: Outi ← acci/li
- 14: end for
- 15: end for
- 16: return Out

- C Vbench Results

To evaluate our model via the VBench framework, we focus on five key dimensions: Subject Consistency (SubConsis), Background Consistency (BackConsis), Motion Smoothness (MotionSmooth), Aesthetic Quality (AesQual), and Imaging Quality (ImagQual). All scores are averaged across our evaluation dataset, and the quantitative results are presented in Table 2.

- Table 2 VBench result of SVG-EAR.

Config SubConsis BackConsis MotionSmooth AesQual ImagQual

Wan 2.2 14B, 720P, I2V 0.960 0.960 0.987 0.628 0.704 SVG 0.958 0.959 0.989 0.627 0.703 Sparge 0.958 0.957 0.987 0.627 0.703 SVG2 0.958 0.957 0.986 0.624 0.701 SVG-EAR 0.959 0.959 0.986 0.627 0.703 SVG-EAR-Turbo 0.958 0.958 0.987 0.625 0.702 Wan 2.1 14B, 720P, Text-to-Video 0.916 0.941 0.973 0.650 0.706 SVG 0.912 0.940 0.974 0.651 0.712 Sparge 0.916 0.942 0.974 0.653 0.708 SVG2 0.914 0.943 0.973 0.651 0.705 SVG-EAR 0.915 0.941 0.973 0.652 0.706 SVG-EAR-Turbo 0.915 0.941 0.973 0.651 0.705 Hunyuan 13B, 720P, Text-to-Video 0.904 0.945 0.993 0.625 0.666 SVG 0.905 0.947 0.993 0.627 0.665 Sparge 0.908 0.948 0.994 0.601 0.629 SVG2 0.901 0.944 0.993 0.624 0.654 SVG-EAR 0.903 0.944 0.993 0.626 0.659

#### D Config

- Table 3 summarizes the detailed hyperparameter configurations for SVG-EAR and the baseline methods across different tasks and model backbones. To align all implementation details except for the self-attention mechanism, we set the time warm-up of SVG to 1 to achieve a dense attention pattern. The specific parameters are defined as follows:

- • Time-warm denotes the number of initial diffusion steps during which sparse attention is disabled in favor of full attention. For the majority of baseline methods, this is set to 20% of the total timesteps

. For SparseAttn and SVG on HunyuanVideo, we increase this value to 30% to mitigate performance degradation.

- • Layer-warm specifies the number of model layers that utilize full dense attention during the warmup phase. For the Wan2.2 model, we apply dense attention exclusively to the first layer. Similarly, for HunyuanVideo, dense attention is restricted to the first layer across both single-stream and dual-stream blocks.
- • Top-p is utilized to regulate the computational budget for both SVG2 and SVG-EAR.
- • QC and KC specify the number of query and key clusters, respectively, for SVG2 and our proposed SVG-EAR.
- • Density specifies the density of the attention mask, applied exclusively to SpargeAttn and SVG.

- Table 3 Detailed configuration and hyperparameter settings for SVG-EAR and baseline methods. The symbol “ – ” indicates that a specific parameter is not applicable to the corresponding model.

Config Time-warm Layer-warm Qc Kc Top-p Density Wan 2.2 A14B, 720P, image-to-video

SpargeAttn 10/50 1/40 – – – 0.3 SVG 10/50 1/40 – – – 0.3 SVG2 10/50 1/40 300 1000 0.9 – SVG2-turbo 15/50 1/40 300 1000 0.7 – SVG-EAR 10/50 1/40 300 1000 0.85 – SVG-EAR-Turbo 10/50 1/40 200 500 0.8 –

Wan 2.2 A14B, 720P, text-to-video

SpargeAttn 10/50 1/40 – – – 0.3 SVG 10/50 1/40 – – – 0.3 SVG2 10/50 1/40 300 1000 0.9 – SVG2-turbo 15/50 1/40 100 500 0.7 – SVG-EAR 10/50 1/40 300 1000 0.85 – SVG-EAR-Turbo 10/50 1/40 200 500 0.8 –

Hunyuan 13B, 720P, text-to-video

SpargeAttn 15/50 1/20, 1/40 – – – 0.4 SVG 15/50 1/20, 1/40 – – – 0.3 SVG2 10/50 1/20, 1/40 400 1000 0.9 – SVG-EAR 10/50 1/20, 1/40 400 1000 0.85 –

- E Full Table Result

In Table 4, we additionally include SVG2-Turbo, which demonstrates our clear Pareto frontier. Specifically, while SVG2-Turbo further improves acceleration at the cost of some generation quality, our turbo variant aligns with its high inference speed but achieves even higher generation quality than the baseline SVG.

- Table 4 Quality and efficiency benchmarking results of SVG-EAR and baselines, where the best results are highlighted, and the second-best results are underlined.

Config PSNR↑ SSIM↑ LPIPS↓ ImgQual↑ SubCons↑ Density↓ FLOP↓ Speedup↑ Wan 2.2 14B, 720P, I2V - - - 0.704 0.960 100% 658.46 PFLOPS 1×

SpargeAttn 27.140 0.883 0.116 0.703 0.958 30.15% 396.83 PFLOPS 1.58× SVG 25.297 0.844 0.139 0.703 0.958 30.25% 397.20 PFLOPS 1.58× SVG2 27.668 0.888 0.117 0.701 0.958 29.38% 393.95 PFLOPS 1.61× SVG2-Turbo 27.536 0.884 0.124 0.700 0.957 16.69% 385.41 PFLOPS 1.72× SVG-EAR 29.759 0.918 0.093 0.704 0.959 23.64% 378.88 PFLOPS 1.61× SVG-EAR-Turbo 28.344 0.900 0.108 0.702 0.958 20.42% 363.85 PFLOPS 1.77× Wan 2.2 14B, 720P, T2V - - - 0.706 0.916 100% 658.46 PFLOPS 1×

SpargeAttn 20.872 0.708 0.242 0.708 0.916 30.15% 396.83 PFLOPS 1.58× SVG 19.455 0.654 0.292 0.712 0.912 30.25% 397.20 PFLOPS 1.59× SVG2 23.556 0.802 0.183 0.705 0.914 32.30% 404.88 PFLOPS 1.57× SVG2-Turbo 23.173 0.772 0.212 0.703 0.910 18.77% 392.23 PFLOPS 1.71× SVG-EAR 24.995 0.841 0.153 0.706 0.915 25.95% 387.53 PFLOPS 1.59× SVG-EAR-Turbo 23.940 0.814 0.174 0.705 0.915 22.25% 370.71 PFLOPS 1.75× Hunyuan 13B, 720P, T2V - - - 0.665 0.904 100% 612.38 PFLOPS 1×

SpargeAttn 24.589 0.796 0.232 0.629 0.908 40.09% 389.76 PFLOPS 1.38× SVG 27.325 0.880 0.140 0.665 0.905 29.92% 351.97 PFLOPS 1.57× SVG2 29.445 0.911 0.112 0.654 0.901 26.21% 299.02 PFLOPS 1.89× SVG-EAR 31.043 0.928 0.092 0.659 0.903 22.17% 281.86 PFLOPS 1.93×

- F Visualization of the Generated Videos

We provide visualization comparison between SVG-EAR and full Attention on HunyuanVideo and Wan 2.2 as shown in Figure 7, Figure 8 and Figure 9. Our generated results achieve an extremely high pixel-level similarity to full attention, achieving an excellent visual quality that is indistinguishable from the full attention results. This provides further evidence of our method’s high fidelity in reproducing full attention outputs while significantly reducing computational costs.

[Figure 206]

[Figure 207]

SVG-EAR

[Figure 208]

###### Dense Attention

[Figure 209]

SVG-EAR

[Figure 210]

###### Dense Attention

[Figure 211]

SVG-EAR

[Figure 212]

###### Dense Attention

[Figure 213]

SVG-EAR

[Figure 214]

###### Dense Attention

[Figure 215]

SVG-EAR

###### Figure 7 Comparion of Dense Attention and SVG-EAR on Wan 2.2 Text-to-Video generation

[Figure 220]

[Figure 221]

SVG-EAR

[Figure 222]

###### Dense Attention

[Figure 223]

SVG-EAR

[Figure 224]

###### Dense Attention

[Figure 225]

SVG-EAR

[Figure 226]

###### Dense Attention

[Figure 227]

SVG-EAR

[Figure 228]

###### Dense Attention

[Figure 229]

SVG-EAR

###### Figure 8 Comparion of Dense Attention and SVG-EAR on Wan 2.2 Image-to-Video generation

[Figure 234]

[Figure 235]

SVG-EAR

[Figure 236]

###### Dense Attention

[Figure 237]

SVG-EAR

[Figure 238]

###### Dense Attention

[Figure 239]

SVG-EAR

[Figure 240]

###### Dense Attention

[Figure 241]

SVG-EAR

[Figure 242]

###### Dense Attention

[Figure 243]

SVG-EAR

###### Figure 9 Comparion of Dense Attention and SVG-EAR on HunyuanVideo Text-to-Video generation

