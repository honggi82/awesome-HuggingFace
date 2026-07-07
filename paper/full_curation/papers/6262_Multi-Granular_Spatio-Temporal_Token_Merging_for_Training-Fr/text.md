# arXiv:2507.07990v1[cs.CV]10Jul2025

## Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs

Jeongseok Hyun1* Sukjun Hwang2 Su Ho Han1 Taeoh Kim3 Inwoong Lee3

Dongyoon Wee3 Joon-Young Lee4 Seon Joo Kim1† Minho Shim3† 1Yonsei University 2Carnegie Mellon University 3NAVER Cloud 4Adobe Research

### Abstract

Video large language models (LLMs) achieve strong video understanding by leveraging a large number of spatiotemporal tokens, but suffer from quadratic computational scaling with token count. To address this, we propose a training-free spatio-temporal token merging method, named STTM. Our key insight is to exploit local spatial and temporal redundancy in video data which has been overlooked in prior work. STTM first transforms each frame into multigranular spatial tokens using a coarse-to-fine search over a quadtree structure, then performs directed pairwise merging across the temporal dimension. This decomposed merging approach outperforms existing token reduction methods across six video QA benchmarks. Notably, STTM achieves a 2× speed-up with only a 0.5% accuracy drop under a 50% token budget, and a 3× speed-up with just a 2% drop under a 30% budget. Moreover, STTM is queryagnostic, allowing KV cache reuse across different questions for the same video. The project page is available at https://www.jshyun.me/projects/sttm.

### 1. Introduction

Integrating visual understanding capabilities into Large Language Models (LLMs) has led to significant advances in multimodal systems [1, 11, 17, 45, 47, 50, 56]. However, Video LLMs, which extend these capabilities to video understanding, face unique computational challenges due to the inherently large number of visual tokens required to represent spatio-temporal information [7, 26, 32, 35, 39, 58].

Early video LLMs [26, 27, 39, 58] reduce token count by training abstractors (e.g., Q-Former [25]) to compress visual information. FlashAttention (FA) [9, 10] reduces attention’s quadratic memory cost to linear, enabling the training of long-context LLMs. Building on this, LLaVAstyle [29] video LLMs use linear projectors that preserve

*This work was done during an internship at NAVER Cloud. †Co-corresponding authors.

###### (a) Rel. Accuracy (50%)

###### (b) Rel. Accuracy (30%)

VNBench

VNBench

NExT-QA

EgoSchema NExT-QA

100

98

98

92

EgoSchema

VideoMME

VideoMME

96

86

94

80

92

74

LongVideoBench

LongVideoBench

MLVU

MLVU

Q. Aware Q. Agnostic Dataset

FastV ToMe < 3min

DyCoke

FrameFusion STTM (Ours) Needle

DyCoke-stage1

< 1hour

###### (c) Avg. Rel. Performance (50%)

###### (d) Avg. Rel. Performance (30%)

99.5

Rel. TTFT.

| |
|---|

Rel. Acc.

97.8

98.6

98.1 98.0

96.2

97.5

94.6

96.8

93.6

92.8

92.5

34.1 30.9 29.5 36.8 35.7 33.1

51.2 46.3 47.9 52.7 49.6 48.4

FastVDyCokeFrame Fusion

ToMeDyCoke stage1

FastVDyCokeFrame Fusion

ToMeDyCoke stage1

STTM (Ours)

STTM (Ours)

Figure 1. Comparison of training-free token reduction methods using LLaVA-Video-7B under 50% and 30% pre-filling token budgets. Query-aware (Q. Aware) methods require re-computation for each new query, whereas query-agnostic methods support KVcache reuse. The evaluated video QA datasets cover short (<3 min), long (<1 hr), and needle-in-a-haystack (NIAH) videos. (a, b): Per-dataset accuracy. (c, d): Average results across all.

high-resolution spatial features and deliver strong performance [59, 60]. More recently, Ring Attention [31] extends FA [9] across GPUs, enabling very long-context models, such as LWM [30] and LongVILA [7], to process hour-long videos with high spatial and temporal resolution features.

Long latency remains a key bottleneck for deploying video LLMs. Attention FLOPs still scale quadratically with token count, and long video contexts must be processed in full to pre-fill Key-Value (KV) states before answering any question [34]. In practice (e.g., Gemini [16]), KV states for the video are cached to avoid recomputation for subse-

quent queries, supporting efficient multi-turn querying over a shared video context. However, existing training-free token reduction methods [6, 15, 20, 43], which avoid the overhead of retraining, overlook this scenario and instead focus on query-aware strategies that discard video tokens based on attention scores between the video and the query, without preserving KV cache reusability.

Designing a query-agnostic token reduction method is essential for enabling KV cache reuse, but it is challenging due to the lack of signals to guide token selection. VideoMAE [12, 46] demonstrates that videos can be reconstructed from a lower token ratio than images, highlighting the inherently redundant nature of video content due to its spatio-temporal continuity. However, existing trainingfree token reduction methods do not explicitly leverage this spatio-temporal structure [6, 15, 38, 43, 49]. To address this, we propose a novel spatio-temporal token merging method, named STTM, which is applied at an early layer of the LLM. STTM performs decomposed merging: it first merges tokens along the spatial dimension, followed by merging along the temporal dimension.

To perform spatial merging, we build a multi-level quadtree for each video frame, linking each parent node to its four child nodes. A parent node’s token is retained at the coarsest level when its similarity to all four child nodes exceeds a threshold. Regions containing fine-grained details – indicated by low similarity – are subdivided to the next finer level to preserve high-frequency information. Consequently, each frame is represented by multi-granular spatial tokens, capturing both coarse and fine detail.

Extending spatial merging to the spatio-temporal domain is not trivial, due to varying spatial granularities across frames. To handle this challenge, we exploit spatiotemporal locality by restricting merging candidates to spatially overlapping tokens between consecutive frames. Unlike prior temporal-only methods that compare singlegranularity tokens across adjacent frames [15], our approach fully leverages the spatial structure and temporal continuity in video data to guide token merging.

After comparing spatially overlapping tokens across time, similar token pairs are chained into spatio-temporal graphs. Within each graph, nodes are merged into a single token representing a token tracklet, with merging directed toward the earlier frame. This strategy accumulates temporal changes into the token where the content first appears.

The same region may be represented at different spatial levels across frames, leading to two cases: many-toone (fine-to-coarse) and one-to-many (coarse-to-fine). The many-to-one case is straightforward: merging fine tokens into a single coarse token in an early frame. The one-tomany case is complicated. Ideally, we would select the most similar fine-grained token as the destination, but this requires per-region comparisons that are difficult to vector-

ize and inefficient on GPUs. To address this, we approximate the merge direction by selecting the top-left token among candidates. This approximation allows us to implement temporal merging using a vectorized union-find algorithm [44], leading to efficient parallel processing.

Comprehensive experimental results across six video QA benchmarks demonstrate that our proposed method effectively reduces video tokens while maintaining high performance. As shown in Fig. 1, STTM outperforms both query-aware and query-agnostic methods overall. In the average results Fig. 1 (c, d), it achieves the highest accuracy with competitive latency under both 50% and 30% token budgets. A closer look at the per-dataset accuracy in Fig. 1 (a, b) further highlights its robustness, particularly on the challenging NIAH and long video datasets. Moreover, we validate STTM’s generalization by applying it to other LLMs and a large-scale 72B Video LLM.

### 2. Related Work

#### 2.1. Spatio-Temporal Redundancy in Videos

Videos exhibit strong spatio-temporal locality – information within nearby regions in space and time is often redundant. This property has long been exploited across various domains. In deep learning, CNNs [18, 23] leverage spatial locality and are extended to 3D convolutions [4, 48] to model spatio-temporal correlations in video. In transformers, MAE [19] and VideoMAE [12, 46] show that reconstructing masked regions using local context is an effective self-supervised learning objective, implicitly benefiting from redundancy.

Similarity, video compression methods explicitly eliminate redundancy. Quadtree-based partitioning [13, 37] adapts block resolution based on regional complexity [41, 42, 52]. Temporal redundancy is also reduced, for example through the Skip mode in H.264/AVC [52], which omits re-encoding regions that remain unchanged across frames. Inspired by these principles, we propose a multi-granular spatio-temporal token merging method that explicitly compresses redundant video tokens by exploiting local similarity in both space and time.

#### 2.2. Long Context Modeling in Video LLMs

In the early stages, LLMs had a limited maximum context length; for instance, LLaMA [47] supported only 2048 tokens. To handle extensive video context, multimodal LLMs integrate abstractor architectures [5, 24, 25] that reduce visual tokens before passing them into the language model. For video LLMs, Q-Former [25] based architecture has been widely adopted to reduce the number of visual tokens from a video [26–28, 32, 35, 36, 39, 51, 58], instead of using linear projectors, which preserve the original token count.

However, with the introduction of FlashAttention [9, 10],

###### System Step 1: Spatial Merging Step 2: Temporal Merging

Video

Question

σ NST𝑖 Tokens

|| |
|---|
| |
|---|---|
| | |

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>...|
|---|

|| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

[Figure 1]

[Figure 2]

[Figure 3]

HW Tokens NST0 Tokens

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

[Figure 9]

| |
|---|

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

| |
|---|

| |
|---|

[Figure 27]

[Figure 28]

[Figure 29]

Reshape

[Figure 30]

[Figure 31]

T0 T1 T2

T×H×W×C

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 32]

NST Tokens

[Figure 33]

Spatio-Temporal

[Figure 34]

Token Merging

[Figure 35]

[Figure 36]

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

[Figure 37]

NST×C

| |
|---|

| |
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|| |
|---|
<br><br>Lv.1<br><br>| |
|---|
<br><br>Lv.2<br><br>| |
|---|
<br><br>Lv.3|| |
|---|
<br><br>Merging Direction Empty<br><br>| |Src|
|---|---|
| | |
<br><br>|Dst|
|---|
<br><br>|
|---|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Transformer ith layer

[Figure 51]

[Figure 52]

T0 T1 T2

- Figure 2. (Left) Our spatio-temporal token merging method is a training-free, plug-and-play module that produces spatio-temporally multigranular tokens. (Middle) In step 1, tokens are merged based on spatial locality, where similar tokens within a 2D grid are combined into a single token. (Right) In step 2, spatially multi-granular tokens are further merged along the temporal dimension, where similar tokens across frames are consolidated into their earliest occurrence. The arrows indicate the direction of token merging. The green lines indicate merging over one timestep, and magenta lines are merging over two timesteps. The scale and the number of tokens are set for illustration.

which reduces the memory complexity of attention from quadratic to linear in sequence length, linear projectorbased approaches have recently gained more attention and showed better performance than abstractor-based approaches. Furthermore, recent methods such as RingAttention [31] have enabled video LLMs [7, 30] to process up to one million video tokens. Following this trend, handling thousands of frames without downsampling is expected to become common in the future. However, such a large number of tokens significantly increases latency, highlighting the need for effective video token compression.

#### 2.3. Training-Free Token Reduction in Video LLMs

Token reduction aims to identify and reduce highly redundant or less informative tokens, and has been researched in the architecture of Vision Transformer [2, 8]. Due to high computational costs in long video LLMs [7], studying token reduction in the regime of multimodal LLMs is being spotlighted [6, 15, 20, 21, 38, 43, 49, 55].

In the initial works [6, 38], visual tokens are reduced without considering the visual structure such as the 2D locality. One-dimensional token reduction methods treating visual tokens as a sequential stream, similar to text tokens, without considering spatial or temporal relationships. FastV [6] reduces low-rank visual tokens based on attention patterns in the early layers. Additionally, it employs text queries to search attention weights, requiring a token reduction to be performed again for each new query. In contrast, our method is query-agnostic, allowing KV cache reuse in LLMs, making it more efficient for conversational and multi-question scenarios.

Temporal token reduction methods exploit the redundancy between video frames. FrameFusion [15] demonstrates that the token similarity distribution condenses in deeper layers while preserving ranking consistency. It merges similar tokens between frames using similarity metrics and prunes unnecessary tokens based on attention scores. We argue that leveraging spatio-temporal character-

istics, rather than operating in a single dimension, enables more effective and efficient token reduction.

### 3. Methodology

We propose Spatio-Temporal Token Merging (STTM) that merges video tokens along spatial and temporal dimensions, producing multi-granular video tokens. This module is training-free and can be easily plugged into an intermediate layer of an LLM. As the STTM operates along the spatiotemporal dimension and outputs multi-granular tokens, it is designed as a single-pass operation, inserted into a single early layer of the transformer. Furthermore, our token reduction method is question-agnostic, enabling the reuse of KV caches across different questions for the same video.

#### 3.1. Overview of STTM Module

As illustrated in Fig. 2, we use the video tokens from the previous layer’s output to perform token merging. After the merging process, the video tokens, ZV ∈ RT×H×W×C, are reduced into spatio-temporally merged tokens, ZST ∈ RN

ST×C, where NST ≪ T × H × W.

To exploit spatial locality, we first merge tokens that are similar within the 2D grid of each frame. As labeled by Lv.1, Lv.2, and Lv.3 in Fig. 2, we define the different scales of the 2D grid for merging unit. Based on these multi-scale merging grids, a large redundant region can be represented by a single coarse-grained token, achieving a high token reduction ratio. On the other hand, fine-grained tokens are used for representing the region with large variation. This spatial merging process results in spatially multi-granular tokens, ZT

Ti

S ×C, where NT

##### S ≪ H × W.

S ∈ RN

i

i

In videos, locality further extends to the spatio-temporal dimension. Thus, we merge tokens by comparing their similarities at the same 2D region across consecutive frames. When tokens remain similar over time, we chain them along a connected path and merge them into the earliest occurrence. Tokens with different spatial granularities can also be

[Figure 53]

[Figure 54]

| |
|---|

| |
|---|

- Lv.1

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

- Lv.2
- Lv.3

[Figure 67]

[Figure 68]

Stop

Lv.1

0.9

0.5

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

[Figure 69]

[Figure 70]

Stop

[Figure 71]

[Figure 72]

Lv.2

[Figure 73]

[Figure 74]

[Figure 75]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

|Affinity<br><br>Score<br><br>| | | |
|---|---|---|
| | | |
<br><br>High<br><br>Low|
|---|

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Coarse-to-Fine Search

- Figure 3. A coarse-to-fine spatial search is performed using a quadtree structure. If all four fine child nodes exhibit high similarity with the coarse parent node, the search process terminates, and the parent node is used to represent the corresponding region. Otherwise, the search continues until the finest level is reached. Here, the scale for each level is an example for illustration.

connected across frames when they overlap. As depicted in step 2 of Fig. 2, some fine-grained tokens at T1 are merged into a coarser token at T0, while others remain separate due to low similarity. Our spatial and temporal hybrid merging allows multi-granular tokens in both dimensions.

#### 3.2. Spatial Token Merging

As illustrated in Fig. 3, our approach employs a coarseto-fine hierarchical search based on a quadtree data structure [13, 37]. At every level, each region requires only four comparisons for the granularity decision process; this process determines whether features at the current level sufficiently represent the region. If not, finer-grained features from the next level are adopted to represent the region without losing details. This hierarchical method effectively balances computational efficiency and representation flexibility. The quadtree structure organizes spatial tokens into a hierarchical representation, where each coarse-level parent node corresponds to four (2×2) finer-level child nodes within its respective 2D grid region. This hierarchical organization enables a structured approach to spatial token merging, preserving 2D locality, which is essential for subsequent temporal merging.

Building the quadtree follows three steps. First, we initialize leaf nodes of the quadtree based on the initial token feature map, ZT

lv ∈ RH×W×C. Second, we construct a multi-scale representation by recursively downsampling the feature map until the feature map size reaches 2×2, which we denote as Lv.1, representing the coarsest spatial resolution. At each level, the feature map is downsampled by averaging spatially neighboring 2×2 feature patches, yielding a coarser-level representation, ZT

i

lv−1 ∈ RH2 ×W2 ×C. During this downsampling operation, we connect four leaf nodes from ZT

i

lv to a single parent node from ZT

lv−1. Third, using this precomputed densely connected quadtree, we perform a quadtree search as described in Fig. 3 and keep only the necessary nodes to represent each frame.

i

i

For each level, we compute the cosine similarity between the current level’s nodes and their child nodes. When the similarities of four child nodes are higher than the spatial threshold value, τS, we can regard this region has low spatial detail and terminate the further searching process for the corresponding node. As a result, all child nodes for this node are pruned in the quadtree. In contrast, when any of the child nodes show low similarity, we prune the current node and instead use the child nodes to represent the corresponding region. Thus, this quadtree search process can be regarded as the granularity decision process.

The computational complexity of our quadtree-based spatial merging algorithm follows O(HW) per frame. The quadtree search is an iterative process that performs four comparisons for each node at a given level. In the worst case scenario, where all nodes are subdivided down to the deepest level, the computational cost of spatial token merging, CSTM, is the geometric series following:

Lv

HW 4i × 4 (1)

CSTM =

i=1

Asymptotically, CSTM is bounded by the sum of infinite geometric series:

HW 1 − 14

4HW 3

(2)

lim

CSTM =

=

Lv→∞

Notably, we implement the quadtree search algorithm in parallel at each level, ensuring that the actual running time remains linear in complexity.

#### 3.3. Temporal Token Merging

To capture the spatio-temporal redundancy, we compute the similarity of tokens representing the same region in neighboring two frames and make the connection between pairs of tokens exhibiting similarity higher than τT. As we merge into the earlier tokens, this process results in directed graphs, where the source node is from T + 1, and the destination node is at T. Based on these directed graphs built from consecutive frames, we obtain connected graphs and identify the common root node for all nodes in the connected graph. The common root node and other nodes are set as destination and source, respectively, for merging. So, the nodes can be merged not only in consecutive twoframe sequences but also in the far distance. As shown in Fig. 2 (right), a token representing the top-left token at T2 is merged into the token at T0, while the top-right token at T2 is merged into the token at T1. This merging process results in spatio-temporally multi-scale tokens.

Since spatial merging results in varying scales of tokens, we cannot directly merge the tokens at the same exact spatial location across frames. Instead, we merge the tokens under their spatio-temporally overlapping regions. For example, in Fig. 2 step 2, top-left region is expressed in varying scales across frames. We compare the similarity be-

Table 1. Comparison of training-free token reduction methods using LLaVA-Video-7B under 50% and 30% pre-filling token budgets. Token-reduced results are reported relative to the result with 100% .

tween a single Lv.1 token at T0 and 2×2 Lv.2 tokens at T1. Two Lv.2 tokens at T1 with high similarity are merged into the token at T0, while the other two remain to represent a change in T1’s top-left region.

During temporal merging, we set the direction of merging as tokens at the earlier frame. In the case of top-left at T1 and T2, the token at T2 is coarser than T1 and has four possible destination paths for merging. When there are multiple connections with similarity higher than τT, we provide the priority to the token located towards top-left. We simplify the choice of choosing the top-left merging token for such multiple destination cases primarily for parallel indexing implementation, although merging with the most similar token is the naive method.

The computational complexity of our temporal merging algorithm follows O(THW) per video. Suppose the worst case is that spatial tokens of all frames are in the finest level, incurring (T −1)×H ×W comparisons. However, we can expect more efficiency gain since spatial merging results in fewer tokens at every frame, reducing the comparisons. Computing the common root node can also be achieved in O(THW) by adapting the union-find algorithm [44] into a vectorized form.

#### 3.4. Token Reordering after Merging

After spatio-temporal token merging, the resulting video tokens form a graph structure. To serve as input for an LLM, this graph must be linearized into a one-dimensional sequence. In our reordering strategy, we prioritize tokens based on two criteria. First, spatial ordering follows a Zshaped scan based on each token’s top-left coordinate. Second, temporal ordering gives precedence to tokens from earlier frames over those from later ones.

Once reordered into a 1D sequence, handling positional embeddings becomes crucial. Recent LLMs typically use rotary positional embeddings (RoPE) [40], which apply a rotation matrix computed from each token’s position index. We evaluate three strategies for assigning RoPE after merging: (1) Merged RoPE, which averages the RoPEs of

merged tokens; (2) Survived RoPE, which retains the original RoPEs of the surviving tokens; (3) Reassigned RoPE, which reassigns position IDs based on the new token order for alignment with the original positional encoding.

### 4. Experiments

#### 4.1. Evaluation Setting

Datasets. To focus on visual understanding, we adopt an evaluation setting without subtitles. We evaluate methods on six diverse video QA benchmarks. These include short-form datasets, EgoSchema [33] and NExT-QA [54], as well as long-form datasets with hour-long videos: VideoMME [14], LongVideoBench [53], and MLVU [62]. To rigorously evaluate fine-grained spatio-temporal understanding, we use VNBench [61], a synthetic dataset simulating the needle in a haystack task [22]. It introduces subtle visual or textual “needles” into short segments of a video – irrelevant to the original content but relevant to the question. Evaluation Metrics. We report standard accuracy for multiple-choice question answering. For practicality, we include the average running time (in seconds) and the number of visual tokens (NV ). Since our focus is on the pre-filling stage, we measure the time-to-first-token (TTFT). In addition to absolute values, we provide relative values (R.) with respect to performance without token reduction.

Implementation Details. We sample video frames at 1 FPS, but uniformly sample frames if a video exceeds the maximum frame limit. To ensure coverage of injected needles in VNBench, we set the maximum number of frames to 180; for other datasets, we use a limit of 128 frames. A single and four A100 80G GPUs are used for 7B and 72B models, respectively. Our merging threshold values (τS and τT) are empirically adjusted to approximately meet specific token budget for fair comparisons with other methods.

#### 4.2. Comparison with Existing Methods

We compare other methods [2, 6, 15, 43] under the same experimental setup. ToMe [2] is applied within an LLM rather than its original ViT-based setting. DyCoke-stage1 refers to

Table 2. Comparison of training-free token reduction methods using LLaVA-OneVision-7B. Relative to 100% result .

|Token Budget<br><br>|Method|VNBench|VideoMME|LongVideoBench|Avg.|
|---|---|---|---|---|---|
| | |Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|

|100%|Qwen2VL 7B|66.4 2.438 22025<br><br>|61.8 10.745 74982<br><br>|56.8 10.597 72109<br><br>|n.a n.a n.a<br><br>|
|---|---|---|---|---|---|
|50%<br><br>|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|95.5 46.3 50.0 98.1 43.0 49.6<br><br>105.2 29.8 48.9<br><br>|100.1 42.0 50.0<br><br>101.2 38.8 48.1 101.8 44.3 52.3<br><br><br>|101.4 41.9 50.0 101.2 39.1 48.8 101.1 43.5 51.5<br><br>|99.0 43.4 50.0<br>100.2 40.3 48.8 102.7 39.2 50.9<br><br><br>|
|30%|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|85.6 30.3 30.0 81.6 29.0 33.4<br><br>100.4 17.2 30.6<br><br>|99.3 26.4 30.0 99.9 25.2 31.4<br><br>101.0 25.7 32.9<br><br>|99.5 26.3 30.0<br><br>101.2 25.4 32.3<br><br>100.1 23.3 27.7<br><br><br>|94.8 27.7 30.0 94.2 26.5 32.4<br><br>100.5 22.1 30.4<br><br>|

Table 3. Comparison using Qwen2VL-7B. Relative to 100% result .

|Token Budget<br><br>|Method|VideoMME|
|---|---|---|
| | |Acc ↑ TTFT ↓ NV|

|100%|LLaVA-Video 72B|70.5 17.698 22086<br><br>|
|---|---|---|
|50%|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|100.1 47.6 50.0<br><br>99.8 45.5 47.9<br><br>101.3 44.2 44.2<br><br><br>|
|30%|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|97.1 29.3 30.0<br>98.3 30.2 31.2<br><br>99.1 30.5 30.5<br><br><br>|

Table 4. Comparison using LLaVAVideo-72B. Relative to 100% result .

a technique for pre-filling stage [43]. We report relative values here, with absolute values provided in Appendix.

Performance of LLaVA-Video-7B. Our method outperforms both query-aware and query-agnostic methods on average across benchmarks (Tab. 1). On average , it incurs only 0.5% and 2.2% relative accuracy drops under the 50% and 30% budgets, respectively. On VNBench (30% budget), other Q.Agn methods exhibit large drops (about 18%), but it shows only a 2.0% drop. These results indicate that our method better preserves fine-grained spatio-temporal details. All methods apply token reduction with O(N) complexity; thus, TTFT is dominated by attention in the LLM layers and scales with the number of retained tokens. Since tokens are reduced at the ipnut or in the early LLM layers, all methods exhibit similar TTFT at each token budget.

Generalization to Other MLLMs. We also evaluate our method with LLaVA-OneVision (Tab. 2) and Qwen2VL (Tab. 3), and it continues to outperform other methods on both MLLMs. Notably, it even improves accuracy while using fewer tokens (under both 50% and 30% budgets). We observe accuracy improvements of 1.1% and 0.5%, along with 3.1× and 4.5× speed-ups, for OneVision and Qwen2VL, respectively. Qwen2VL consumes more visual tokens than LLaVA-based models, resulting in a greater latency reduction due to the quadratic complexity of attention. Scaling to 72B. Our method continues to outperform other methods on the 72B LLM under both 50% and 30% token budgets (Tab. 4). Notably, it even improves accuracy by 1.3% while using only 44.2% of the tokens. The 72B model requires substantial processing time per request (e.g., an average of 17.7 seconds). As a query-agnostic method,

it performs token reduction without relying on the question, enabling reuse of the KV cache for the same video across different questions. This improves deployment efficiency in multi-turn or multi-query scenarios.

#### 4.3. Ablation Study

We conduct ablation studies to validate effectiveness of the proposed components using LLaVA-Video-7B.

Multi-Granularity Spatial Tokens. To evaluate the effectiveness of our hierarchical spatial token merging method, we compare it against a single-granularity method using bilinear interpolation (Tab. 5). On VideoMME, the singlegranularity method performs competitively, even surpassing FastV [6] while using only 41.3% of the tokens. However, its accuracy drops significantly on VNBench which requires fine-grained spatial information. In contrast, our quadtreebased multi-granularity merging maintains accuracy across both datasets. At τS of 0.80, it incurs only 1.3% and 0.8% drop in relative accuracy with a token budget of approximately 50%. This highlights the advantage of incorporating multi-granularity spatial tokens for balancing compression efficiency and accuracy retention.

Root Node Spatial Resolution. As shown in Tab. 5, using a coarser spatial scale (2×2) for root node initialization leads to a lower relative NV at the same spatial threshold. This indicates that more tokens are merged and each frame is represented by coarser spatial tokens. However, this higher compression ratio comes at the cost of a significant accuracy drop compared to using root nodes with a 4×4 spatial scale. The effect is particularly pronounced at lower τS, where the accuracy degradation is more severe,

|Token Granularity|Lv.1 Scale<br><br>τS<br><br>|VNBench|VideoMME|
|---|---|---|---|
| | |R.Acc ↑ R.NV ↓<br><br>|R.Acc ↑ R.NV ↓|

Single 13×13 82.1 86.2 99.2 86.2 Multi 2×2 0.85 88.1 54.2 98.1 58.2 Multi 4×4 0.85 99.9 82.0 99.5 83.0 Single 11×11 80.0 61.7 98.4 61.7 Multi 2×2 0.80 73.1 27.0 96.2 31.5 Multi 4×4 0.80 98.7 57.2 99.2 59.9 Single 9×9 78.9 41.3 97.4 41.3 Multi 2×2 0.75 62.6 13.3 92.5 15.8 Multi 4×4 0.75 93.0 35.8 97.5 38.9

FastV 93.7 50.0 96.7 50.0

Table 5. Ablation study on the proposed multi-granular spatial token merging. Lv.1: spatial scale of root nodes.

|STM TTM|VNBench|VideoMME|
|---|---|---|
| |R.Acc. ↑ R.NV ↓<br><br>|R.Acc. ↑ R.NV ↓|

✓ ✗ 98.7 57.2 99.2 59.9 ✗ ✓ 100.0 53.7 99.9 56.7 ✓ ✓ 98.0 25.8 98.8 25.8

✓ 73.3 27.0 95.5 32.0

Table 6. Effectiveness of spatial and temporal merging modules. Sequentially combining these two modules results in a synergistic effect. Last row: joint merging.

Method R.Acc ↑ R.TTFT ↓ R.NV ↓

Optimal 99.6 58.1 47.7 Top-left 99.2 50.1 47.1 Optimal 97.7 39.6 27.7 Top-left 98.8 31.4 25.8

- Table 7. Temporal merging ablation

|Positional Embedding|VNB|VidMME|
|---|---|---|
| |R.Acc ↑<br><br>|R.Acc ↑|

Merging 96.0 95.7 Survival 96.5 96.0

Reassignment 98.0 98.8

- Table 8. Ablation study on positional embedding after merging.

|LLM Layer|VNBench|VideoMME|
|---|---|---|
| |R.Acc ↑ R.TTFT ↓ R.NV ↓|R.Acc ↑ R.TTFT ↓ R.NV ↓|

1 95.8 27.9 28.3 96.0 26.5 26.4 3 98.0 30.9 25.8 98.8 31.3 25.8 7 98.2 40.6 24.8 97.9 41.6 25.6

19 99.5 75.5 31.7 99.2 71.6 22.6

Table 9. Ablation study of token merging position.

G-Token VNB VidMME LVB MLVU EgoS NExT Avg. (∆) ✓ 78.2 62.6 58.9 70.6 58.5 82.2 68.5

✗ 77.6 63.1 59.6 70.9 58.7 82.9 68.8 (+0.3)

Table 10. Effect of grid token removal on accuracy.

especially on VNBench. This is due to the loss of spatial details, as quadtree subdivision from a 2×2 root may terminate prematurely, producing tokens that are too coarse to retain essential visual information. To address this, we adopt a 4×4 spatial scale for root nodes, which enables lower τS for higher compression while preserving accuracy.

Decomposed Spatio-Temporal Merging. In our algorithm, we first perform spatial merging, followed by temporal merging in sequence. Tab. 6 presents the impact of spatial and temporal merging, individually. While both methods effectively reduce the number of tokens, applying them sequentially yields a synergistic effect, achieving a high token reduction ratio with only a marginal accuracy drop. Additionally, we experiment with a joint spatio-temporal merging method based on an octree data structure, which partitions a long video into predefined cubic segments and recursively subdivides them at a 2×2×2 finer scale. While this method shows promising results on VideoMME, it leads to a significant accuracy drop on VNBench, where rapid spatial changes occur within short time intervals. This suggests that rigid hierarchical partitioning across spatiotemporal dimensions is not effective in dynamic scenarios. Therefore, we adopt a decomposed merging strategy.

Approximated Temporal Matching. As shown in Tab. 7, selecting the most similar destination node does not consistently improve performance. In contrast, using a top-left approximation enables a vectorized union-find algorithm [44], avoiding costly similarity checks, and leads to significant runtime improvement without drop in accuracy.

Positional Embedding. Tab. 8 shows the results of dif-

###### (a) VNBench

###### (b) VideoMME

80

63.1

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>77.6<br><br>53.3<br><br>64.1<br><br>74.1<br><br>63.9<br><br>73.2 75.4 74.7<br><br>76.0<br><br>77.4 77.2<br><br>Original<br><br>ToMe<br><br>DyCoke-stage1<br><br>STTM (Ours)| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

- 58

- 59

- 60

- 61

- 62

- 63

62.6

62.3 61.9

75

62.0

Accuracy(%)

70

61.4

| |
|---|

61.0

65

60.3

60

60.0

59.2

| |
|---|

55

58.3

| |
|---|

50

100 50 40 30 20 10

100 50 40 30 20 10

Visual Token Ratio (%)

Figure 4. Trade-off of accuracy and visual token retention ratio.

ferent strategies for handling positional embeddings during token merging within LLM layers. We observe that preserving the original positional embeddings yields better performance than merging them. Since Qwen2VL [50] uses MRoPE, which computes positions based on (t, y, x), we cannot apply the reassignment strategy and instead adopt the survival strategy.

Merging Layer Position. We evaluate the impact of different LLM layer positions for video token merging (Tab. 9). As observed in recent works [3, 6, 55, 57], performing token merging at later layers yields higher accuracy but lower speed-up, due to increased full-token attention computations. Merging tokens before the third LLM layer provides a good trade-off between accuracy and latency; we adopt this configuration for 7B models, while for the 72B model, we merge tokens before the first LLM layer.

Grid Token. As shown in Tab. 10, removing the special grid token used for video data in LLaVA-Video does not degrade accuracy across benchmarks. To simplify the handling of video tokens during spatio-temporal merging, we omit such token in LLaVA-Video and OneVision.

Accuracy vs. Compression. Fig. 4 illustrates the trade-off between accuracy and visual token retention ratio. On both datasets, our method consistently outperforms other queryagnostic token reduction methods [2, 43] across token retention levels, ranging from 50% to 15%. This demonstrates the robustness of our method in achieving a favorable balance between compression efficiency and accuracy.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

(a) (b)

[Figure 145]

[Figure 146]

Figure 5. Visualization of spatial token merging results. Each image patch within a green box represents a single token.

[Figure 147]

- (a)

original

merged

- (b)

[Figure 148]

original

merged

Figure 6. Visualization of spatio-temporal token merging results on VideoMME. (a) The first eight consecutive frames are sampled. (b) Intermediate frames are sampled for illustration purpose. Empty regions indicate areas that have been merged with early tokens.

#### 4.4. Visualization of Token Merging

almost entirely merged. Fig. 6 (b) illustrates an example with dynamic scene changes. Similarly, within each scene, later frames tend to be merged into the first frame of the segment (e.g., frames 4 and 5). Notably, subtle but semantically meaningful changes – such as facial expressions or hand gestures – are preserved in frames 6 and 7.

Unlike uniform reduction methods, STTM adaptively reduces tokens based on the redundancy present in each video. It allocates more tokens to regions that require fine-grained detail, while aggressively reducing tokens in redundant areas. On VideoMME (30% budget), its token retention ranges from 3.3% to 51.2% across videos, demonstrating its adaptability to varying content complexity.

### 5. Conclusion

- Fig. 5 presents the results of spatial merging before ap-

plying temporal merging. Fig. 5 (a) illustrates a case where semantically similar tokens are effectively merged, reducing 196 visual tokens to 63 in a frame. In contrast, Fig. 6 (b) shows a worst-case scenario for spatial compression. In this frame, fine-grained tokens are preserved to retain small text details. Although the compression rate is low, OCR accuracy on VideoMME is maintained after token merging. While using all tokens yields an OCR accuracy of 67.6, ToMe, DyCoke-stage1, and our method achieve 58.3, 64.7, and 65.5, respectively, under token reduction.

- Fig. 6 shows the results of spatio-temporal merging. In

This is the first work to explore multi-granular video token merging in a training-free manner for video LLMs. We propose a decomposed spatio-temporal token merging method, called STTM. Its effectiveness is validated across six video QA benchmarks. Notably, it largely outperforms others on VNBench which requires fine-grained understanding. Since our method operates independently of user instructions, the KV cache can be reused across different questions for the same video, improving computational efficiency. While our method successfully minimizes spatio-temporal redundancy, its performance currently depends on manually adjusted threshold values. Exploring adaptive threshold selection is a promising direction, enabling automatic adjustment of token merging based on the given token budget.

Fig. 6 (a), regions with newly emerging content remain intact, while redundant regions across consecutive frames are merged. For instance, letter regions that appear in the second and sixth frames are preserved, while background tokens, such as the black areas, are merged into the earlier frames. Duplicated frames (e.g., frames 3, 4, 7, and 8) are

Acknowledgements. This work was partly supported by the NAVER Cloud Corporation.

### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1
- [2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In ICLR, 2023. 3, 5, 7
- [3] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024. 7
- [4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, pages 6299–6308, 2017. 2
- [5] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In CVPR, pages 13817–13827, 2024. 2
- [6] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In ECCV, pages 19–35. Springer, 2024. 2, 3, 5, 6, 7
- [7] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Yihui He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, and Song Han. Longvila: Scaling long-context visual language models for long videos. In ICLR, 2025. 1, 3
- [8] Joonmyung Choi, Sanghyeok Lee, Jaewon Chu, Minhyuk Choi, and Hyunwoo J Kim. vid-tldr: Training free token merging for light-weight video transformer. In CVPR, pages 18771–18781, 2024. 3
- [9] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In ICLR, 2024. 1, 2
- [10] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022. 1, 2
- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [12] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. NeurIPS, 35:35946–35958, 2022. 2
- [13] Raphael A Finkel and Jon Louis Bentley. Quad trees a data structure for retrieval on composite keys. Acta informatica, 4:1–9, 1974. 2, 4
- [14] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, pages 24108–24118, 2025. 5
- [15] Tianyu Fu, Tengxuan Liu, Qinghao Han, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang.

- Framefusion: Combining similarity and importance for video token reduction on large visual language models. arXiv preprint arXiv:2501.01986, 2024. 2, 3, 5
- [16] Google. Caching — google ai, 2024. Accessed: 2025-07-09. 1
- [17] Gemini Team Google. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 2
- [19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 16000–16009, 2022. 2
- [20] Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video large language models. arXiv preprint arXiv:2412.16117, 2024. 2, 3
- [21] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In CVPR, pages 13700–13710, 2024. 3
- [22] G Kamradt. Needle in a haystack – pressure testing llms. https://github.com/gkamradt/LLMTest_ NeedleInAHaystack, 2023. 5
- [23] Yann LeCun, Bernhard Boser, John Denker, Donnie Henderson, Richard Howard, Wayne Hubbard, and Lawrence Jackel. Handwritten digit recognition with a backpropagation network. NeurIPS, 2, 1989. 2
- [24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022. 2
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 1, 2
- [26] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 1, 2
- [27] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 1
- [28] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024. 2
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2023. 1
- [30] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268,

2024. 1, 3

- [31] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. In ICLR, 2024. 1, 3

- [32] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL,

2024. 1, 2

- [33] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. NeurIPS, 36:46212– 46244, 2023. 5
- [34] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. MLSys, 5:606–624, 2023. 1
- [35] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In CVPR, pages 14313– 14323, 2024. 1, 2
- [36] Michael S Ryoo, Honglu Zhou, Shrikant Kendre, Can Qin, Le Xue, Manli Shu, Silvio Savarese, Ran Xu, Caiming Xiong, and Juan Carlos Niebles. xgen-mm-vid (blip-3video): You only need 32 tokens to represent a video even in vlms. arXiv preprint arXiv:2410.16267, 2024. 2
- [37] Hanan Samet. The quadtree and related hierarchical data structures. ACM CSUR, 16(2):187–260, 1984. 2, 4
- [38] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388,

2024. 2, 3

- [39] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, pages 18221–18232, 2024. 1, 2
- [40] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 5

- [41] Gary J Sullivan and Richard L Baker. Efficient quadtree coding of images and video. TIP, 3(3):327–331, 1994. 2
- [42] Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the high efficiency video coding (hevc) standard. TCSVT, 22(12):1649–1668, 2012. 2
- [43] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In CVPR, 2025. 2, 3, 5, 6, 7
- [44] Robert Endre Tarjan. Efficiency of a good but not linear set union algorithm. JACM, 22(2):215–225, 1975. 2, 5, 7
- [45] OpenAI Team. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1
- [46] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. NeurIPS, 35:10078– 10093, 2022. 2
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2

- [48] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In ICCV, pages 4489–4497,

2015. 2

- [49] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Lookonce optimization in kv cache for efficient multimodal longcontext inference. arXiv preprint arXiv:2406.18139, 2024. 2, 3
- [50] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 7
- [51] Yuxuan Wang, Cihang Xie, Yang Liu, and Zilong Zheng. Videollamb: Long-context video understanding with recurrent memory bridges. arXiv preprint arXiv:2409.01071,

2024. 2

- [52] Thomas Wiegand, Gary J. Sullivan, Gisle Bjontegaard, and Ajay Luthra. Overview of the h.264/avc video coding standard. TCSVT, 13(7):560–576, 2003. 2
- [53] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. NeurIPS, 37:28828–28857,

2025. 5

- [54] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In CVPR, pages 9777–9786, 2021. 5
- [55] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In CVPR, 2025. 3, 7
- [56] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [57] Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. arXiv preprint arXiv:2405.12532, 2024. 7
- [58] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. EMNLP Demo Track, 2023. 1, 2
- [59] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 1
- [60] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 1
- [61] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue, Longteng Guo, Bingning Wang, Weipeng Chen, and Jing Liu. Needle in a video haystack: A scalable synthetic evaluator for video mllms. arXiv preprint arXiv:2406.09367, 2024. 5

[62] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In CVPR, pages 13691–13701, 2025. 5

### Appendix

Tabs. 11 to 14 show the absolute values for the main comparison results.

|Token Budget|Method<br><br>|Q. Agn.|VNBench|VideoMME|LongVideoBench|MLVU|EgoSchema|NExT-QA|Avg.|
|---|---|---|---|---|---|---|---|---|---|
| | | |Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|

|100%|LLaVA-Video 7B|✓|77.6 0.962 11149<br><br>|63.1 2.039 22086<br><br>|59.6 1.805 19624<br><br>|70.9 2.343 25088<br><br>|58.7 2.312 25069<br><br>|82.9 0.659 8116<br><br>|68.8 1.687 18522<br><br>|
|---|---|---|---|---|---|---|---|---|---|
|50%<br><br>|+ FastV<br><br>+ DyCoke<br><br>+ FrameFusion| |72.7 0.503 5575 72.1 0.458 5386 76.2 0.471 5529|61.0 1.034 11043<br><br>61.5 0.912 10555<br><br>62.0 0.961 10739<br><br><br>|57.4 0.918 9812<br>58.1 0.820 9393<br>59.2 0.853 9616<br>|68.3 1.164 12544<br><br>69.5 1.046 11978<br><br><br>69.4 1.083 12138|57.6 1.166 12535<br><br>58.6 1.049 11969<br><br><br>57.7 1.088 12298<br><br>|82.4 0.353 4058 82.1 0.330 3957 82.6 0.336 4032<br><br>|66.6 0.856 9261<br>67.0 0.769 8873 67.9 0.799 9059<br>|
| |+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|✓ ✓ ✓<br><br>|74.1 0.518 5575 73.2 0.495 5386 77.4 0.455 4804<br><br>|61.4 1.043 11043<br><br>62.0 0.981 10555 62.6 1.021 10771<br><br><br>|58.0 0.949 9812<br><br>58.2 0.877 9393<br><br>59.6 0.895 9183<br><br><br>|69.7 1.192 12544 69.7 1.123 11978 69.9 1.152 12187<br><br>|58.7 1.199 12535 58.7 1.118 11969 58.6 1.045 10737<br><br>|82.6 0.370 4058<br><br>82.4 0.350 3957<br><br>82.5 0.322 3452<br><br><br>|67.4 0.878 9261<br><br>67.4 0.824 8873<br><br>68.4 0.815 8522<br><br><br>|
|30%<br><br>|+ FastV<br><br>+ DyCoke<br><br>+ FrameFusion| |61.6 0.336 3345 55.9 0.308 3548 72.4 0.292 3157<br><br>|59.2 0.683 6626<br>60.7 0.598 6878 60.7 0.581 6018<br>|54.7 0.610 5887 57.1 0.544 6131 57.1 0.520 5462<br><br>|69.3 1.620 17562<br><br>67.3 0.684 7798 67.5 0.650 6768<br><br>|56.9 0.771 7520 58.3 0.693 7792<br>57.4 0.657 6870<br>|81.6 0.240 2435 81.5 0.229 2631 81.9 0.218 2313<br><br>|63.9 0.710 7229 63.5 0.509 5797 66.2 0.486 5098<br><br>|
| |+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|✓ ✓ ✓<br><br>|64.1 0.364 3345 63.9 0.358 3548 76.0 0.299 2649<br><br>|59.2 0.720 6626<br><br>60.0 0.700 6878 62.3 0.640 5929<br><br><br>|56.3 0.658 5888<br><br>56.5 0.631 6131<br><br>57.0 0.616 5702<br><br><br>|67.0 0.821 7527<br><br>68.6 0.796 7798 68.5 0.769 7337<br><br><br>|57.4 0.834 7521<br>58.7 0.800 7792 58.0 0.773 7285<br><br><br>|81.6 0.265 2436<br><br>81.7 0.256 2631 82.0 0.235 2168<br><br><br>|64.3 0.610 5557 64.9 0.590 5797 67.3 0.555 5179<br><br>|

Table 11. Comparison of training-free token reduction methods using LLaVA-Video-7B under 50% and 30% pre-filling token budgets.

|Token Budget<br><br>|Method|Q. Agn.|VNBench|VideoMME|LongVideoBench|MLVU|EgoSchema|NExT-QA|Avg.|
|---|---|---|---|---|---|---|---|---|---|
| | | |Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓|

|100%|LLaVA-OV 7B|✓|68.8 0.922 11149<br><br>|59.0 1.904 22086<br><br>|56.3 1.712 19624<br><br>|67.5 2.224 25088<br><br>|61.3 2.216 25069<br><br>|80.6 0.630 8116<br><br>|65.6 1.601 18522<br><br>|
|---|---|---|---|---|---|---|---|---|---|
|50%<br><br>|+ FastV<br><br>+ DyCoke<br><br>+ FrameFusion| |65.6 0.467 5575 64.9 0.418 5386 67.8 0.436 5568<br><br>|58.4 0.934 11043 60.1 0.831 10555<br>59.8 0.865 10888<br>|56.2 0.842 9812<br><br>57.9 0.740 9393<br><br><br>56.6 0.778 9719|67.4 1.071 12544<br><br>68.7 0.941 11978<br><br><br>68.2 0.994 12379<br><br>|61.0 1.080 12535 61.8 0.948 11969 60.8 0.997 12511<br><br>|80.2 0.330 4058 80.4 0.301 3957 80.8 0.312 4056<br><br>|64.8 0.787 9261<br>65.6 0.697 8873 65.7 0.730 9187<br>|
| |+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|✓ ✓ ✓<br><br>|67.1 0.483 5575 65.1 0.459 5386 71.2 0.404 4573<br><br>|59.8 0.963 11043<br><br>59.3 0.911 10555<br><br>60.7 0.773 8579<br><br><br>|57.7 0.865 9812<br><br>58.2 0.821 9393 57.7 0.804 9011<br><br><br>|68.8 1.101 12544<br><br>69.1 1.039 11978 69.7 1.020 11692<br><br><br>|62.3 1.108 12535 61.8 1.049 11969 61.7 0.972 10944<br><br>|80.1 0.343 4058 80.6 0.327 3957 80.5 0.312 3628<br><br>|66.0 0.811 9261<br><br>65.7 0.767 8873<br><br>66.9 0.714 8071<br><br><br>|
|30%<br><br>|+ FastV<br><br>+ DyCoke<br><br>+ FrameFusion| |59.2 0.303 3345 53.9 0.280 3548 66.1 0.261 3263<br><br>|58.1 0.597 6626<br>59.8 0.537 6878 59.0 0.499 6154<br>|55.3 0.543 5887<br><br>55.9 0.488 6131<br><br>56.5 0.455 5566<br><br><br>|65.6 0.681 7526 68.3 0.598 7798<br>66.7 0.570 6914<br>|60.9 0.694 7520 62.1 0.607 7792 60.3 0.582 7251<br><br>|79.6 0.218 2435<br>79.7 0.208 2631 79.9 0.195 2400<br>|63.1 0.506 5556<br><br>63.3 0.453 5797<br><br>64.7 0.427 5258<br>|
| |+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|✓ ✓ ✓<br><br>|59.1 0.332 3345 58.4 0.324 3548 70.4 0.277 2773<br><br>|59.8 0.649 6626<br><br>60.3 0.633 6878 60.6 0.601 6264<br><br><br>|56.5 0.587 5888<br><br>56.2 0.577 6131<br><br>56.6 0.570 6022<br><br><br>|69.1 0.740 7527 68.0 0.723 7798 68.4 0.735 7989<br><br>|62.1 0.750 7521 62.3 0.731 7792 61.8 0.570 5621<br><br>|79.5 0.240 2436 79.7 0.236 2631 79.7 0.240 2577<br><br>|64.4 0.550 5557 64.1 0.537 5797 66.2 0.499 5208<br><br>|

Table 12. Comparison of training-free token reduction methods using LLaVA-OneVision-7B.

|Token Budget<br><br>|Method|VNBench|VideoMME|LongVideoBench|Avg.|
|---|---|---|---|---|---|
| | |Acc ↑ TTFT ↓ NV ↓|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓<br><br>|Acc ↑ TTFT ↓ NV ↓|

|100%|Qwen2VL 7B|66.4 2.438 22025<br><br>|61.8 10.745 74982<br><br>|56.8 10.597 72109<br><br>|61.7 7.927 56372<br><br>|
|---|---|---|---|---|---|
|50%|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|63.4 1.130 11013 65.1 1.049 10645 69.8 0.726 7228<br><br>|61.9 4.509 37491<br><br>62.6 4.166 36057 62.9 4.761 39217<br><br><br>|56.7 4.348 36054<br><br>57.5 4.148 34735 57.4 4.610 37315<br><br><br>|60.7 3.329 28186<br>61.7 3.121 27146 63.4 3.366 27920<br><br><br>|
|30%<br><br>|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|56.9 0.740 6608 54.2 0.706 6980 66.7 0.420 6748<br><br>|61.4 2.835 22496<br><br>61.8 2.710 23479<br><br>62.4 2.766 23143<br><br><br>|54.7 2.600 21633 57.5 2.694 22649 56.9 2.472 20022<br><br>|57.6 2.058 16912 57.8 2.037 17703 62.0 1.886 16638<br><br>|

Table 13. Comparison using Qwen2VL-7B. Relative to 100% result .

|Token Budget<br><br>|Method<br><br>|VideoMME|
|---|---|---|
| | |Acc ↑ TTFT ↓ NV|

|100%<br><br>|LLaVA-Video 72B|70.5 17.698 22086<br><br>|
|---|---|---|
|50%|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)<br><br>|70.6 8.424 11043<br><br>70.4 8.052 10555<br><br>71.4 7.821 10082<br><br><br>|
|30%<br><br>|+ ToMe<br><br>+ DyCoke-stage1<br><br>+ STTM (Ours)|68.5 5.186 6626<br><br>69.3 5.353 6878 69.9 5.405 6897<br><br><br>|

Table 14. Comparison using LLaVAVideo-72B. Relative to 100% result .

