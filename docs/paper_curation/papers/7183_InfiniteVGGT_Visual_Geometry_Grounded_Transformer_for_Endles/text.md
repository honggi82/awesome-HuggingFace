# arXiv:2601.02281v1[cs.CV]5Jan2026

## InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams

Shuai Yuan1 Yantai Yang1,2 Xiaotian Yang1 Xupeng Zhang1 Zhonghao Zhao1 Lingming Zhang Zhipeng Zhang1

1AutoLab, School of Artificial Intelligence, Shanghai Jiao Tong University 2 Anyverse Dynamics

Frame Attention

[Figure 1]

### Abstract

[Figure 2]

[Figure 3]

CausalAttention

Global Attention

Total frame

The grand vision of enabling persistent, large-scale 3D visual geometry understanding is shackled by the irreconcilable demands of scalability and long-term stability. While offline models like VGGT achieve inspiring geometry capability, their batch-based nature renders them irrelevant for live systems. Streaming architectures, though the intended solution for live operation, have proven inadequate. Existing methods either fail to support truly infinitehorizon inputs or suffer from catastrophic drift over long sequences. We shatter this long-standing dilemma with InfiniteVGGT, a causal visual geometry transformer that operationalizes the concept of a rolling memory through a bounded yet adaptive and perpetually expressive KV cache. Capitalizing on this, we devise a training-free, attentionagnostic pruning strategy that intelligently discards obsolete information, effectively “rolling” the memory forward with each new frame. Fully compatible with FlashAttention, InfiniteVGGT finally alleviates the compromise, enabling infinite-horizon streaming while outperforming existing streaming methods in long-term stability. The ultimate test for such a system is its performance over a truly infinite horizon, a capability that has been impossible to rigorously validate due to the lack of extremely long-term, continuous benchmarks. To address this critical gap, we introduce the Long3D benchmark, which, for the first time, enables a rigorous evaluation of continuous 3D geometry estimation on sequences about 10,000 frames. This provides the definitive evaluation platform for future research in long-term 3D geometry understanding. Code is available at: https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT

Token

- (a) offline & total frames (e.g. VGGT)
- (b) online & implicit memory (e.g. CUT3R, TTT3R)
- (c) online & unbounded memory (e.g. StreamVGGT)
- (d) online & rolling memory (ours InfiniteVGGT)

Hidden state

St-1 St

Sequence

fast

Xt

[Figure 4]

frame

Cache

- Frame 1
- Frame 2

Token Cache

OOM

Causal Attention

[Figure 5]

memory

- T-1

Cache

- T-2

T-1

T-1

frame

Token 3

[Figure 6]

T-1 add

memory

Diversity Cache

Token

Frame T-1

- T=1
- T=2
- T=3

- T=4

Frame Attention

Diversity

frame

[Figure 7]

###### +

###### +

slow

Redundant

Frame T

frame

Causal Attention

Causal Attention

Redundant

Figure 1. Paradigm Comparison between previous online and offline 3D geometry understanding and our InfiniteVGGT.

Causal add

Cache

Token

Diversity

T

- 1

- 2

- 3

ing as the bedrock for critical applications such as augmented reality (AR) [15, 18, 44] and embodied AI [3, 16, 20, 23, 41, 43, 46]. Historically, the domain has been dominated by classical methods rooted in Structure-fromMotion (SfM) [1, 9, 22, 28, 32, 39] and Multi-View Stereo (MVS) [10, 13]. While capable of high-fidelity geometric optimization, these approaches are characterized by fragmented, multi-stage pipelines that are prohibitively slow, and prone to cascading errors. A paradigm shift has been catalyzed by the advent of end-to-end deep learning frameworks, which transcend these limitations by holistically inferring 3D structure from raw image data. Models such as DUSt3R [36], VGGT [34], and their derivatives [19, 42] have reshaped the landscape, championing fully data-driven methodologies that achieve globally consistent reconstructions with unprecedented efficiency.

Cache

- Layer1

- Layer2

- Layer3

- T-1

Cache

- T-2

Token

Diversity Cache

Redundant

- 1
- 2
- 3

Redundant

### 1. Introduction

The dense reconstruction of 3D scenes from 2D images constitutes a cornerstone problem in geometric vision, serv-

As these end-to-end models mature, the contemporary

Corresponding Author.

landscape has become defined by a fundamental dichotomy between offline batch processing [7, 30, 34] and online streaming paradigms [4, 17, 35, 45]. As illustrated in Fig. 1(a), offline methods masterfully exploit multi-view geometric constraints to achieve superior geometric fidelity, rendering them ideal for short-term reconstructions where data is fully pre-captured. This batch-centric paradigm, however, is fundamentally ill-suited for online applications or unbounded sequences due to its prohibitive GPU memory footprint [30]. Conversely, streaming architectures are conceptually tailored for online scenarios, such as robotics, by processing inputs sequentially to provide immediate perceptual feedback. Their theoretical appeal lies in handling infinite-length scene flows. Yet, this promise is largely unrealized in practice. One paradigm, namely explicit history accumulation frameworks like StreamVGGT [45], betrays its online intent by accumulating unbounded KeyValue (KV) stores (Fig. 1(c)), a path that inevitably leads to crippling memory and computational overheads. The other one, namely implicit state compression mechanisms, such as those in CUT3R [35] and TTT3R [4] (Fig. 1(b)), make a Faustian bargain, where they compress history into a simple RNN hidden state to guarantee bounded resources, but in doing so, discard critical information, thereby exacerbating long-term drift and compromising robustness. Then, a question naturally arises that is it possible to selectively retain critical historical information to ensure temporal consistency, while still operating within the bounded resources required for a truly online system?

The key to escaping this dilemma lies not in a more complex model, but in a pivotal insight into the nature of the data itself. We observe that in contiguous camera trajectories, minimal viewpoint shifts create massive token-level redundancy within the KV cache. This is not a trivial matter, as each frame adds approximately 1,000 tokens, the cache rapidly explodes to a scale (O(105) tokens within 100 frames) that necessitates hardware-optimized kernels like FlashAttention just to remain functional. Herein lies a fundamental paradox that these kernels achieve their speed by circumventing the materialization of the full O(N2) attention matrix, yet traditional pruning methods rely on accessing these very weights to gauge token importance. Consequently, the tool required to manage the size of the cache prevents us from intelligently shrinking it. To resolve this impasse, we introduce an elegant solution by leveraging key cosine similarity as an efficient, attention-independent proxy for token importance. This allows us to identify and discard redundant tokens before the costly attention computation, thereby preserving the efficiency of optimized kernels while surgically shrinking the cache and finally paving the way for truly scalable streaming reconstruction.

Building upon this, we introduce InfiniteVGGT, which embodies a novel “rolling memory” paradigm for online 3D

geometry understanding. It avoids the unbounded memory growth inherent in explicit history accumulation frameworks while simultaneously mitigating the information drift that plagues implicit state compression methods. Our rolling memory achieves this by continuously and dynamically refreshing its contents through a deeply integrated, multi-level retention strategy. At its foundation, the strategy abandons intuitive and coarse frame-wise deletion, selectively preserving individual tokens to maintain crucial longterm context. This granular process is then governed by a dynamic budget that is intelligently structured across the model’s architecture. It functions layer-wise by assigning a unique token budget to each layer, resulting in layer-specific and specialized KV caches. This systematic control system operates without materializing attention weights, ensuring full compatibility with FlashAttention, and ultimately enables a system with a strictly bounded GPU memory footprint capable of processing infinite sequences.

The ultimate test for such a system is its performance over a truly infinite horizon, a capability that has been impossible to rigorously validate due to the lack of continuous, long-term benchmarks. To address this gap, we introduce the Long3D benchmark, which, for the first time, enables a rigorous evaluation of continuous 3D geometry estimation on sequences about 10,000 frames. This provides the definitive evaluation platform for future research in long-term 3D scene understanding and reconstruction.

Our contributions are threefold: ♠ An unbounded memory architecture InfiniteVGGT for continuous 3D geometry understanding, built on a novel, dynamic, and interpretable explicit memory system. ♠ State-of-the-art performance on long-sequence benchmarks and a unique capability for robust, infinite-horizon reconstruction without memory overflow. ♠ The Long3D benchmark, a new dataset for the rigorous evaluation of long-term performance, addressing a critical gap in the field.

### 2. Related Work

Classical Offline and Online Reconstruction. Traditional 3D vision methods fall into two primary paradigms distinguished by their operational constraints, namely offline batch processing and online streaming. The cornerstone of offline reconstruction is Structure-from-Motion (SfM). SfM pipelines [1, 9, 22, 28, 32, 39], epitomized by COLMAP [28], perform a global Bundle Adjustment (BA) [14] across all views and points to achieve maximum global accuracy. While computationally demanding, this batch optimization produces highly precise results that subsequently serve as a foundation for Multi-View Stereo (MVS) [11, 12, 29, 37] algorithms to generate dense models. In stark contrast, online streaming methods, prominently represented by Simultaneous Localization and Mapping (SLAM), prioritize online performance. These sys-

tems incrementally estimate the camera trajectory, employing a range of techniques that include feature-based [24], direct [8], and dense [25] approaches.

Learning-based Offline 3D Reconstruction. Recent advances in offline 3D reconstruction have seen classical multi-stage pipelines give way to unified, feedforward architectures. Early works like DUSt3R [36] and MASt3R [19] formulate reconstruction as a pairwise pointmap regression problem, imposing a computationally expensive global alignment stage to aggregate multi-view information. VGGT [34] addresses this by introducing a large transformer that jointly predicts camera poses, depth, and feature tracks in a single forward pass. More recently, π3 [38] refines VGGT to operate independently of a fixed reference frame. However, the input length of such large models remains a bottleneck. To extend scalability, VGGTLong [7] decomposes long trajectories into sub-maps at the cost of single-pass simplicity. In a different approach, SailRecon [6] enhances scene regression using a subset of anchor images to create a global neural representation for efficient localization of all other images. Focusing instead on computational efficiency, FastVGGT [30] accelerates the forward process through a training-free token merge mechanism. This method exploits attention redundancy to preserve key geometric cues, achieving a 4× speedup on 1000frame sequences while reducing drift.

Learning-based Online 3D Reconstruction. The early transformer-based methods, such as Spann3R [33] and Point3R [40], pioneered online forward-pass reconstruction using explicit spatial or pointer memory. This paradigm was refined by StreamVGGT [45] and Stream3R [17], which apply causal attention and a KV cache to process sequences on-the-fly. However, the reliance on an evergrowing KV cache leads to prohibitive increases in memory and computation, rendering these models impractical for truly long streaming inputs. To circumvent this scaling issue, WinT3R [21] employs a sliding-window mechanism to balance reconstruction quality with latency. This design inherently limits the temporal receptive field and can cause drift. Although WinT3R attempts to mitigate this with a global camera-token pool, it still falls short of supporting infinite-length reconstruction. Seeking to overcome these limitations, another line of research adopts RNNbased architectures. CUT3R [35], for example, uses continuously updated states to accommodate arbitrary-length image streams. Building on this foundation, TTT3R [4] introduces test-time training rules to improve length generalization, enabling the online processing of thousands of frames. Nevertheless, catastrophic forgetting caused by transitionally compressed memory remains a fundamental challenge, limiting the ability of capturing long-range temporal dependencies. To mitigate these limitations, we propose an online

streaming framework capable of infinite-length 3D geometry reconstruction by introducing a hierarchical, dynamic rolling memory that preserves long-term dependencies to reduce both drift and catastrophic forgetting.

### 3. Method

#### 3.1. Preliminaries

From Offline to Online 3D Reconstruction. The offline model VGGT [34] processes a batch of N images {Ii ∈ RH×W×3}Ni=1 in a single forward pass. It alternately applies frame (Fθ) and global (Gθ) interaction across 24 selfattention layers to jointly estimate a set of 3D quantities,

(gi,Di,Pi,Ti)Ni=1 = ϕ(Fθ {Ii}Ni=1 ,Gθ {Ii}Ni=1 ),

(1) where gi ∈ R9 represents the camera parameters, Di ∈ RH×W is the depth map, Pi ∈ RH×W×3 is the point map, and Ti ∈ RH×W×C are point-tracking features.

To adapt this architecture for online and streaming usage, models like StreamVGGT [45] substitute the global interaction Gθ with a causal temporal attention module Tθ. This allows the model to process frames incrementally. At any given timestep t, the module generates the output for the current frame It by leveraging a KV cache, Ct−1, that stores the context from all previous frames,

(gt,Dt,Pt,Tt) = ϕ(Fθ (It),Tθ(It,Ct−1)) (2)

NL l=1

The KV cache Ct = (Kt(l),Vt(l))

, where NL is the total number of causal attention layers, is contiguously updated by combining the new keys and values. The cache size grows linearly (O(t)) with the sequence length t.

#### 3.2. Motivation and Analysis

As previously discussed, the KV cache, which functions as explicit memory, grows linearly with each new frame, leading to unsustainable memory demands over time. The central challenge is thus to maintain a fixed-size rolling memory. This requires an intelligent eviction strategy that preserves valuable information while discarding redundancy.

Are Attention Scores a Feasible Eviction Criterion? An intuitive approach is to use attention scores as a proxy for token importance. This idea is initially compelling because sequential input frames in 3D reconstruction possess high spatio-temporal redundancy due to significant viewpoint overlap. We empirically validate this by extracting the patch-embedded tokens from the backbone of StreamVGGT [45] for adjacent frames, finding their cosine similarity consistently exceed 0.95. This high similarity stems from the DINO [26] backbone being trained as a semantic encoder with high invariance to slight changes in viewpoint. It prioritizes “what” is seen over “from where”

𝐹𝑟𝑎𝑚𝑒29 to 𝐹𝑟𝑎𝑚𝑒1 𝐹𝑟𝑎𝑚𝑒29 to 𝐹rame2 Layer = 6 Head = 0 Layer = 9 Head = 0 Layer = 12 Head = 0

𝐹𝑟𝑎𝑚𝑒29 to 𝐹rame3

[Figure 8]

[Figure 9]

[Figure 10]

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | |[Figure 11]<br><br>|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | |[Figure 12]<br><br>| |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

[Figure 13]

2ndprinciplecomponent

2ndprinciplecomponent

2ndprinciplecomponent

Layer=5

[Figure 14]

[Figure 15]

[Figure 16]

Layer=15

1st principle component 1st principle component 1st principle component

|(a) Attention Map from 𝐹𝑟𝑎𝑚𝑒cur to 𝐹𝑟𝑎𝑚𝑒cache|
|---|

(b) PCA Embedding of Keys and Queries

- Figure 2. Visualization Results. (a) Attention maps from the current frame to adjacent historical cached frames, demonstrating nearidentical distributions due to minimal viewpoint shifts in online streaming camera motion. (b) PCA embeddings of query (Q) and key (K) vectors for representative layers and heads, revealing clustering and redundancy in the feature space.

it is seen, which further confirms the extensive redundancy present in the tokens fed into the subsequent aggregator module. As a result, the query frame It often assigns nearidentical attention weights to historical frames that share a similar perspective (Fig. 2 (a)). This observation suggests that an attention-based eviction strategy could effectively prune the cache. However, this approach introduces a critical computational dilemma. The KV cache in these architectures must scale to hundreds of thousands or even millions of tokens (|Ct| ≫ 105). To manage this scale online, the causal attention mechanism (Tθ) fundamentally depends on hardware-optimized kernels, such as FlashAttention, which mitigate memory bandwidth bottlenecks by never explicitly materializing the full attention matrix. This reliance creates an irreconcilable conflict that any token filtering strategy predicated on attention scores requires the materialization of the full attention weight matrix, which is the very operation that optimized kernels are designed to bypass. Executing this operation would be computationally prohibitive and would negate the low-latency inference essential for streaming systems. We therefore argue that this paradigm is suboptimal and motivate the need for an alternative approach to construct an efficient rolling memory.

pendent of the current query. Tokens with higher diversity scores correspond to those most dissimilar from the global mean, and are thus retained during cache compression. As a result, the cache preserves the most informative subset of tokens while maintaining a minimal memory footprint.

[Figure 17]

#### 3.3. Diversity-aware Rolling Memory

Immutable Anchor Token. As illurstrated in Fig. 3, our rolling memory pipeline commences by establishing an immutable set of anchor tokens, defined as the complete KV cache derived from the initial input frame. This design choice is motivated by the architectural foundation of VGGT [34], wherein all subsequent 3D predictions are rigidly aligned to the coordinate system of the first frame, which serves as the canonical global reference. Any alteration or pruning of these initial tokens would irreversibly compromise geometric consistency across the entire reconstruction. Accordingly, we designate the first-frame cache as the immutable anchor set Canc(l,h) and exclude it from all subsequent compression operations. For any given layer l and head h, the total cache Ct(l,h) is thus partitioned into the anchor set and a mutable candidate set, Ct,cand(l,h) , which contains all tokens from t = 2 onwards.

Key Diversity as a Redundancy Proxy. Instead of estimating token salience through attention weights, we measure redundancy in the key space. As illustrated in Fig. 2(b), PCA visualizations of the key and query spaces reveal that queries (qt) from the current frame and cached key vectors (kt−1) consistently occupy distinct, nearly orthogonal subspaces across layers. This geometric separation persists over time, confirming that key-space similarity provides a stable measure of redundancy. Therefore, distinct keys will be more aligned with the query, and in turn, can be preserved as more salient keys. Building on this, we define the negative cosine similarity as diversity score to quantify this dispersion, hypothesizing that keys are the principal components and provide the most effective mechanism for quantifying redundancy. This metric efficiently captures the dispersion of key representations in feature space and is inde-

Diversity-quantified Token Retention. Then, we apply our retention strategy π exclusively to the candidate set

Ct,cand(l,h) to retain the most informative tokens. This process is performed independently for each layer l and head h to

account for their heterogeneous redundancy profiles. Our strategy begins by establishing a reference vector for each head’s key space. This is achieved by computing the mean key µ(l,h). To ensure this metric captures directional variance exclusively, we operate on L2-normalized keys, where kˆi = ki/||ki||. The mean key is thus the expectation over the set of normalized candidate keys Kˆt,cand(l,h) ,

###### [kˆ] (3)

µ(l,h) = Ekˆ∈Kˆ(l,h)

t,cand

Anchor

Sequence

|DecoderLayer| |
|---|---|
| | |

[Figure 18]

[Figure 19]

Encoder

Encoder Image

DecoderLayer

DecoderLayer

DecoderLayer

ScoresDecoderLayer

Image

Image

K&V

K&V

K&V

- Frame 1
- Frame 2

[Figure 20]

[Figure 21]

Encoder Image

Encoder Image

Image

Layer l-th

Head x Layer/Head-wise Rolling Memory 1-st

Expectation

[Figure 22]

cache current

[Figure 23]

Encoder Image

Encoder Image

Importance score function

Diversity

Frame T-1

|Topk|
|---|

Averagescore

Topk

[Figure 24]

| |
|---|

[Figure 25]

Encoder

Encoder

| |
|---|

retain redundant

Frame T

- Figure 3. Overview of the InfiniteVGGT, illustrating a rolling memory paradigm that prunes KV cache contents to prevent VRAM accumulation over time, employing key cosine similarity and adaptive layer-wise allocation for 3D geometry understanding.

Next, we define a diversity score sdiv for each individual key kˆi to quantify its dissimilarity from this mean vector. Based on our previous analysis, we employ the negative cosine similarity as our metric,

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

(b)Dormitory(a)Library

[Figure 30]

[Figure 31]

[Figure 32]

s(divl,h)(kˆi) = −CosSim(µ(l,h),kˆi) (4) This formulation ensures that keys with the lowest cosine similarity to the mean, which represent the most geometrically distinct features, are assigned the highest scores. Consequently, a high sdiv score signifies high informational salience, guiding the retention of the most valuable tokens.

Figure 4. Long3D Examples. Views and global point clouds of different scenes.

where τ is a temperature hyperparameter. The total budget for layer l is Bl = plbud · Btotal. This budget B(l,h) is then enforced via a TopK selection. The final compressed cache C˜t is the union of all retained candidate tokens C˜t,cand(l,h) and the immutable anchor set Canc.

#### 3.4. Layer-wise Adaptive Budget Allocation

To optimize the KV cache, we introduce an adaptive, layer-wise budget allocation mechanism that assigns a nonuniform storage budget to each layer in proportion to its measured information diversity. This strategy is motivated by the observation that informational diversity is unevenly distributed across the model. Our analysis reveals that shallow layers, which amplify subtle inter-frame differences for spatial reasoning, exhibit high diversity. In contrast, both the initial layer, processing low-level statistics like color and brightness, and the deep layers, where representations converge towards a holistic semantic understanding, demonstrate significantly less diversity. To implement this principle, we first define a layer-wise average diversity

### 4. Long3D Benchmark

To address the critical lack of benchmarks for evaluating continuous, long-term 3D geometry estimation, we propose Long3D. Prior to this work, rigorously assessing a system’s performance over extended, uninterrupted periods was infeasible, as existing benchmarks are either restricted to short sequences (≤ 1000 frames) or, like the 7Scenes [31], are merely collections of discontinuous clips, which prevents a proper assessment of long-term, uninterrupted performance. Long3D fills this critical void by providing the first framework for evaluating model robustness on truly continuous video streams. In total, our dataset features 5 challenging sequences captured in diverse indoor and outdoor environments, with each individual sequence ranging from about 2,000 to 10,000 frames. Fig. 4 shows

score sldiv as the mean of all token diversity s(divl,h) within that layer. The budget proportion plbud for each layer is then calculated via a softmax normalization of these scores,

exp(sldiv/τ)

plbud =

(5)

L j=1 exp(sjdiv/τ)

###### Table 1. 3D Reconstruction Results on 7-Scenes [31] and NRGBD [2].

7-Scenes NRGBD Acc. ↓ Comp. ↓ NC ↑ Acc. ↓ Comp. ↓ NC ↑ Mean Med. Mean Med. Mean Med. Mean Med. Mean Med. Mean Med. VGGT (Offline) [34]

Method Input

OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM StreamVGGT [45] OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM CUT3R [35] 0.135 0.091 0.071 0.032 0.543 0.562 0.224 0.126 0.074 0.012 0.579 0.624 Point3R [40] 0.047 0.027 0.029 0.011 0.563 0.596 0.076 0.043 0.014 0.005 0.618 0.695 TTT3R [4] 0.041 0.025 0.024 0.005 0.565 0.599 0.103 0.045 0.025 0.005 0.608 0.673 InfiniteVGGT 0.040 0.015 0.025 0.005 0.570 0.607 0.051 0.032 0.022 0.005 0.649 0.756

300

VGGT (Offline) [34]

OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM StreamVGGT [45] OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM CUT3R [35] 0.162 0.114 0.093 0.050 0.532 0.546 0.315 0.215 0.101 0.032 0.551 0.572 Point3R [40] 0.049 0.023 0.026 0.009 0.559 0.589 0.093 0.045 0.024 0.005 0.613 0.685 TTT3R [4] 0.052 0.031 0.027 0.005 0.558 0.587 0.140 0.070 0.058 0.014 0.599 0.657 InfiniteVGGT 0.043 0.016 0.026 0.005 0.565 0.599 0.069 0.040 0.034 0.005 0.653 0.763

400

VGGT (Offline) [34]

OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM StreamVGGT [45] OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM CUT3R [35] 0.183 0.130 0.091 0.033 0.530 0.543 0.326 0.243 0.132 0.042 0.556 0.582 Point3R [40] 0.063 0.026 0.031 0.015 0.555 0.583 0.113 0.048 0.037 0.005 0.613 0.684 TTT3R [4] 0.062 0.036 0.029 0.005 0.552 0.577 0.165 0.084 0.095 0.015 0.594 0.648 InfiniteVGGT 0.043 0.018 0.025 0.005 0.561 0.593 0.080 0.054 0.037 0.008 0.643 0.746

500

an example of our benchmark. This data was collected using a handheld 3D spatial scanner equipped with an IMU, a 3D LiDAR (360° horizontal by 59° vertical FOV), and an RGB camera (800 × 600 at 10 Hz, 90° FOV). For each scene, the data consists of a global ground-truth point cloud and the corresponding uninterrupted sequence of RGB images. On our benchmark, we evaluate dense-view streaming reconstruction, where models process the entire image stream to generate a complete global point cloud. For evaluation, predicted and ground-truth point clouds are aligned using the Iterative Closest Point (ICP) algorithm, consistent with prior methods [35, 45]. Performance is quantified using three established metrics, including Accuracy (Acc.), Completion (Comp.), Chamfer Distance(CD) and Normal Consistency (NC).

### 5. Experiments

#### 5.1. Experiments Setup

We conduct a comprehensive evaluation of our proposed method across three demanding tasks of 3D reconstruction, video depth estimation, and camera pose estimation. Initially, we leverage the longest contiguous sequences from established public datasets, benchmarking InfiniteVGGT against leading long-term streaming baselines, namely CUT3R [35] and TTT3R [4]. Our method is a training-free optimization designed to overcome the memory bottlenecks inherent in long-sequence reconstruction. We implement and evaluate this approach on the StreamVGGT [45], concentrating our analysis on long-sequence scenarios where the benefits of our memory-efficient design are most pro-

nounced. On shorter sequences, where the baseline’s GPU memory is not exceeded, performance differences are negligible (see Sec. 6). Subsequently, we introduce and evaluate on our novel large-scale Long3D benchmark to probe stability across extensively prolonged inputs. All experiments were executed on a single NVIDIA A100 GPU.

#### 5.2. 3D Reconstruction

Evaluation on 7-Scenes and NRGBD Datasets. Following the previous work [35], we evaluate scene-level 3d reconstruction on 7-Scenes [31] and NRGBD [2] datasets. But unlike the evaluation of extremely sparse-view reconstruction protocol before, for the long-term streaming, we sampled images with stride = 2 in each sequence and use the first 300 to 500 images as input like TTT3R [4]. As shown in Tab. 1, the offline method VGGT [34] and the online method StreamVGGT [45] fail on long sequences input as the memory constraints. As for the other runnable online method, while TTT3R maintains robust performance on the 7-Scenes dataset, its reconstruction capability on NRGBD degrades significantly as the number of input frames increases. Despite affording greater robustness on varied datasets, Point3R’s explicit pointer mechanism [40] suffers from perpetually increasing memory usage, rendering it incompatible with long-sequence reconstruction (evidenced in [4]). Our method InfiniteVGGT exhibits minimal temporal error accumulation as the sequence length increases, allowing it to consistently maintain the state-of-art reconstruction accuracy. Concurrently, its strong performance across varied datasets highlights its high robustness.

###### Evaluation on Long3D Benchmark. As Sec. 4 states,

##### GT CUT3R TTT3R InfiniteVGGT

| |
|---|

| |
|---|

| |
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

300Frames500Frames

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

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

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

300Frames500Frames

| |
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|

| |
|---|

| |
|---|

Figure 5. Qualitative Results of 3D Reconstruction. Table 2. 3D Reconstruction Results on Long3D.

#### 5.3. Video Depth Estimation

Acc. ↓ Comp. ↓ NC ↑ CD ↓ Mean Med. Mean Med. Mean Med.

Method Scene/Input

Evaluation on Bonn Datasets. Video depth estimation evaluates per-frame depth quality and inter-frame depth consistency. Since most existing datasets only contain a limited number of continuous frames, to show the long-term performance, we evaluate InfiniteVGGT on the longest available continuous sequences from Bonn [27]. Specifically, we select continuous sequences ranging from 200 to 500 frames, beginning after the initial 30 frames like TTT3R. As shown in Tab. 3, the performance of InfiniteVGGT is benchmarked against CUT3R [35] and TTT3R [4], showing the effectiveness of our method.

CUT3R [35]

0.496 0.374 0.085 0.036 0.520 0.525 0.291

Classroom 2128

- TTT3R [4] 0.396 0.319 0.081 0.035 0.530 0.540 0.239

- InfiniteVGGT 0.357 0.298 0.057 0.033 0.576 0.612 0.207

CUT3R [35]

Dormitory 4208

1.800 1.372 0.404 0.090 0.501 0.495 1.102 TTT3R [4] 1.965 1.749 0.329 0.100 0.515 0.509 1.147

- InfiniteVGGT 1.438 1.159 0.575 0.089 0.526 0.538 1.007

- TTT3R [4] 2.175 1.484 0.430 0.095 0.494 0.481 1.303 InfiniteVGGT 1.121 0.821 0.571 0.077 0.508 0.514 0.846

1.907 1.437 0.193 0.079 0.504 0.507 1.050

CUT3R [35]

Library 4726

2.489 2.432 5.802 5.071 0.495 0.483 4.146 TTT3R [4] 2.791 2.392 3.160 2.673 0.509 0.502 2.975 InfiniteVGGT 1.843 1.555 1.854 0.816 0.510 0.509 1.848

CUT3R [35]

Badminton Court 6067

8.062 5.650 0.673 0.251 0.496 0.491 4.638 TTT3R [4] 7.710 5.793 6.192 5.159 0.513 0.519 6.951 InfiniteVGGT 5.733 4.603 1.206 0.251 0.495 0.490 3.470

CUT3R [35]

Academic Building 9545

we evaluated our method alongside other models capable of processing extended-length inputs on sequences of approximately 2,000, 4,500, 6000 and nearly 10,000 frames on Long3D dataset. The results demonstrate that our approach achieves robust performance across diverse scenes and varying sequence lengths, outperforming existing models like CUT3R [35] and TTT3R [4] on most metrics. More importantly, although temporal drift inevitably accumulates with increasing input frames, our method effectively limits this error propagation compared to baselines. However, we observed that our method underperforms on the mean of Comp. metric compared to these baselines. We identify this as a key area for optimization in our future work.

#### 5.4. Ablation Study

Crucial Token Selection Policy. We conduct a comparative analysis of attention weight-based and cosine similaritybased token selection policies on the 7-Scenes dataset [31]. In addition to evaluating reconstruction quality via chamfer distance (CD), and normal consistency (NC) metrics, chamfer distance is computed as the average of accuracy and completeness. We also profile the per-frame inference time and peak GPU memory consumption. As summarized in Tab. 4, the cosine similarity-based approach yields more accurate point cloud reconstruction. Moreover, standard attention weight-based methods can introduce an additional 120ms of inference latency per frame, while our model’s

Table 3. Video Depth Estimation on Bonn [27].

Bonn Abs Rel ↓ δ < 1.25 ↑ VGGT (Offline) [34]

Method Input

OOM OOM StreamVGGT [45] OOM OOM CUT3R [35] 0.072 0.947 Point3R [40] 0.069 0.954 TTT3R [4] 0.068 0.953 InfiniteVGGT 0.063 0.964

200

VGGT (Offline) [34]

OOM OOM StreamVGGT [45] OOM OOM

300

- CUT3R [35] 0.089 0.938 Point3R [40] 0.081 0.946 TTT3R [4] 0.079 0.949 InfiniteVGGT 0.072 0.958 VGGT (Offline) [34]

400

OOM OOM StreamVGGT [45] OOM OOM

- CUT3R [35] 0.090 0.934

- Point3R [40] 0.081 0.945 TTT3R [4] 0.078 0.951 InfiniteVGGT 0.070 0.958 VGGT (Offline) [34]

500

OOM OOM StreamVGGT [45] OOM OOM CUT3R [35] 0.084 0.939

- Point3R [40] 0.081 0.946 TTT3R [4] 0.076 0.953 InfiniteVGGT 0.069 0.960

compatibility with FlashAttention [5] mitigates this bottleneck, enabling significantly faster inference speeds and a reduced peak GPU memory footprint.

Initial Budget Per-head. We further ablate the budget B(l,h) using the 7-Scenes dataset, comparing the results for 300 and 500 input with a stride of 2, where B(l,h) denotes the initial maximum storage capacity for tokens per head in each layer. As shown in Tab. 5, a smaller token storage budget B(l,h) significantly degrades the reconstruction quality, while this impact diminishes as the budget increases, eventually becoming negligible.

Layer-wise Allocation Mechanism. To demonstrate the effectiveness of our layer-wise allocation mechanism for token selection, we conducted an ablation study on the 7Scenes. The input frames are 500. Given an initial budget

B10000(l,h) , we compare maintaining a uniform, fixed storage limit across all layers against our dynamic layer-wise allocation scheme. As shown in Tab. 6, dynamically allocating the budget across layers further improves the resulting point cloud accuracy and normal consistency.

Anchor Frame. The VGGT [34] architecture fundamentally depends on the first frame as a global reference and establishes the canonical coordinate system for the entire sequence. Given this pivotal role, we posit that applying

Table 4. Ablation on Attention and Cosine Similarity Method.

Method CD ↓ NC ↑ Time (s) ↓ Peak Memory (GB) ↓

Attention weight 0.036 0.567 0.288 17.30 Cosine similarity 0.032 0.570 0.168 14.49

###### Table 5. Ablation Study on Initial Budget Per-Head.

Input 300 500 CD ↓ NC ↑ CD ↓ NC ↑

Initial Budget

B10000l,h 0.062 0.565 0.075 0.555 B25000l,h 0.032 0.570 0.033 0.560 B50000l,h 0.032 0.570 0.031 0.562

###### Table 6. Ablation Study on Layer-wise Allocation Mechanism.

Acc. ↓ Comp. ↓ NC ↑ Mean Med. Mean Med. Mean Med. w/o layer-wise allocation

Method Input

0.098 0.058 0.057 0.008 0.554 0.582 w/ layer-wise allocation 0.093 0.053 0.056 0.008 0.555 0.583

500

###### Table 7. Ablation Study on Anchor Frame Mechanism.

Acc. ↓ Comp. ↓ NC ↑ Mean Med. Mean Med. Mean Med.

Method

w/o anchor frame 0.047 0.020 0.027 0.006 0.570 0.606 w/ anchor frame 0.040 0.015 0.025 0.006 0.570 0.607

token pruning to the initial state could lead to irreversible information loss. Therefore, we design a strategy where the tokens of the first frame are fully retained as an anchor frame, effectively bypassing the diversity-based selection mechanism applied to subsequent frames. To validate the necessity of this design, we conduct an ablation study on the 7-Scenes [31] dataset, using 300-frame inputs sampled with a stride of 2, to assess the impact of this anchor frame strategy on 3D reconstruction accuracy. As evidenced by the results in Tab. 7, preserving the complete reference information of the first frame prevents error accumulation and leads to a significant improvement in reconstruction quality.

### 6. Discussion

The primary objective of this work is to enable online, infinite-horizon 3D geometry estimation for streaming scenes through a novel rolling memory mechanism. Given that our method, InfiniteVGGT, is a training-free modification of StreamVGGT [45], its performance on shorter sequences relative to the baseline warrants clarification. We therefore begin by confirming that our approach achieves comparable performance in these less demanding scenarios. On input sequences from 50 to 100 frames, which is a range where the baseline operates without memory constraints, our comparison of CD and NC metrics reveals negligible performance differences. As shown in Fig. 6, InfiniteVGGT even achieves a slight precision advantage in the NC metric. This advancements arise from our diversity-aware rolling memory mechanism, which refines the model’s historical context. By preserving a more diverse set of informa-

VGGT (offline) StreamVGGT

CUT3R Point3R

TTT3R InfiniteVGGT

CD NC

NormalConsistency(NC)

ChamferDistance(CD)

Number of Input Views

Figure 6. Comparison of 3D Reconstruction. CD and NC metrics on NRGBD [2] dataset.

tion from early stages, the mechanism enhances robustness against noisy data encountered as the sequence grows. It also prevents the memory from being saturated by redundant subsequent inputs. For long sequences, these benefits become critical. InfiniteVGGT not only resolves the outof-memory (OOM) errors that plague the baseline but also curtails the accumulation of temporal error. In line with our primary goal of addressing the challenges of long-sequence reconstruction, our evaluation is therefore concentrated on these demanding scenarios.

### 7. Conclusion

We present InfiniteVGGT, a novel rolling memory paradigm for streaming 3D geometry understanding that mitigates the trade-off between unbounded memory growth and long-term drift. Our training-free strategy achieves this by identifying memory redundancy via key cosine similarity and applying an adaptive, layer-wise budget allocation. This mechanism, fully compatible with FlashAttention, ensures bounded memory and computational efficiency for online streaming over infinite-horizon sequences. As a result, InfiniteVGGT surpasses existing explicit- and implicitstate methods in reconstruction accuracy and robustness. We also introduce the Long3D benchmark to support rigorous evaluation of extended-sequence performance.

### References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54

(10):105–112, 2011. 1, 2

- [2] Dejan Azinovi´c, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In CVPR, pages 6290–6301, 2022. 6, 9
- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom,

Karol Hausman, Brian Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1

- [4] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 2, 3, 6, 7, 8
- [5] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, pages 16344–16359,

2022. 8

- [6] Junyuan Deng, Heng Li, Tao Xie, Weiqiang Ren, Qian Zhang, Ping Tan, and Xiaoyang Guo. Sail-recon: Large sfm by augmenting scene regression with localization. arXiv preprint arXiv:2508.17972, 2025. 3
- [7] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggtlong: Chunk it, loop it, align it – pushing vggt’s limits on kilometer-scale long rgb sequences, 2025. 2, 3
- [8] J. Engel, T. Sch¨ops, and D. Cremers. LSD-SLAM: Largescale direct monocular SLAM. In ECCV, 2014. 3
- [9] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup, Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, et al. Building rome on a cloudless day. In ECCV, pages 368–381. Springer, 2010. 1, 2
- [10] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. 32(8):1362–1376, 2009. 1
- [11] Yasutaka Furukawa, Carlos Hern´andez, et al. Multi-view stereo: A tutorial. Foundations and Trends® in Computer Graphics and Vision, 9(1-2):1–148, 2015. 2
- [12] Silvano Galliani, Katrin Lasinger, and Konrad Schindler. Massively parallel multiview stereopsis by surface normal diffusion. In ICCV, pages 873–881, 2015. 2
- [13] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In CVPR, pages 2495–2504, 2020. 1
- [14] Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press,

2000. 2

- [15] Jin Gyu Hong, Seung Young Noh, Hee Kyung Lee, Won Sik Cheong, and Ju Yong Chang. 3d clothed human reconstruction from sparse multi-view images. In CVPRW, pages 677–687, 2024. 1
- [16] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1
- [17] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893, 2025. 2, 3
- [18] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In CVPR, 2025. 1
- [19] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r, 2024. 1, 3

- [20] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, Xiaofan Wang, Bei Liu, Jianlong Fu, Jianmin Bao, Dong Chen, Yuanchun Shi, Jiaolong Yang, and Baining Guo. Cogact: A foundational vision-languageaction model for synergizing cognition and action in robotic manipulation, 2024. 1
- [21] Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, and Tong He. Wint3r: Window-based streaming reconstruction with camera token pool, 2025. 3
- [22] Shaohui Liu, Yidan Gao, Tianyi Zhang, R´emi Pautrat, Johannes L Sch¨onberger, Viktor Larsson, and Marc Pollefeys. Robust incremental structure-from-motion with hybrid features. In ECCV, pages 249–269. Springer, 2025. 1, 2
- [23] Hidenobu Matsuki, Gwangbin Bae, and Andrew Davison. 4dtam: Non-rigid tracking and mapping via dynamic surface gaussians. In CVPR, 2025. 1
- [24] Ra´ul Mur-Artal, J. M. M. Montiel, and Juan D. Tard´os. ORBSLAM: a versatile and accurate monocular SLAM system. Submitted to IEEE Transaction on Robotics. arXiv preprint arXiv:1502.00956, 2015. 3
- [25] Richard A. Newcombe, Steven J. Lovegrove, and Andrew J. Davison. Dtam: Dense tracking and mapping in real-time. In ICCV, pages 2320–2327, 2011. 3
- [26] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [27] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Gigu`ere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7855–7862, 2019. 7, 8
- [28] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, 2016. 1, 2
- [29] Johannes L Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In ECCV, pages 501–518. Springer, 2016. 2
- [30] You Shen, Zhipeng Zhang, Yansong Qu, and Liujuan Cao. Fastvggt: Training-free acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560, 2025. 2, 3
- [31] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In CVPR, pages 2930–2937, 2013. 5, 6, 7, 8
- [32] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: Exploring photo collections in 3d. In ACM SIGGRAPH 2006 Papers, pages 835–846, 2006. 1, 2
- [33] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024. 3
- [34] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 1, 2, 3, 4, 6, 8

- [35] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 2, 3, 6, 7, 8
- [36] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 1, 3
- [37] Yuesong Wang, Zhaojie Zeng, Tao Guan, Wei Yang, Zhuo Chen, Wenkai Liu, Luoyuan Xu, and Yawei Luo. Adaptive patch deformation for textureless-resilient multi-view stereo. In CVPR, pages 1621–1630, 2023. 2
- [38] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025. 3
- [39] Changchang Wu. Towards linear-time incremental structure from motion. In 2013 International Conference on 3D Vision (3DV), pages 127–134. IEEE, 2013. 1, 2
- [40] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863, 2025. 3, 6, 8
- [41] Yuqi Wu, Wenzhao Zheng, Sicheng Zuo, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Embodiedocc: Embodied 3d occupancy prediction for vision-based online scene understanding. In ICCV, 2025. 1
- [42] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In CVPR, 2025. 1
- [43] Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, and Linfeng Zhang. Efficientvla: Training-free acceleration and compression for vision-language-action models. arXiv preprint arXiv:2506.10100, 2025. 1
- [44] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In CVPR, 2024. 1
- [45] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 2, 3, 6, 8
- [46] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 1

