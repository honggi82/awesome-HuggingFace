## Efficient Autoregressive Video Diffusion with Dummy Head

Hang Guo12 Zhaoyang Jia23 Jiahao Li2 Bin Li2 Yuanhao Cai4 Jiangshan Wang1 Yawei Li5 Yan Lu2

# arXiv:2601.20499v1[cs.CV]28Jan2026

### Abstract

The autoregressive video diffusion model has recently gained considerable research interest due to its causal modeling and iterative denoising. In this work, we identify that the multi-head selfattention in these models under-utilizes historical frames: approximately 25% heads attend almost exclusively to the current frame, and discarding their KV caches incurs only minor performance degradation. Building upon this, we propose Dummy Forcing, a simple yet effective method to control context accessibility across different heads. Specifically, the proposed heterogeneous memory allocation reduces head-wise context redundancy, accompanied by dynamic head programming to adaptively classify head types. Moreover, we develop a context packing technique to achieve more aggressive cache compression. Without additional training, our Dummy Forcing delivers up to 2.0× speedup over the baseline, supporting video generation at 24.3 FPS with less than 0.5% quality drop. Project page is available at https://csguoh.github.io/ project/DummyForcing/.

### 1. Introduction

Video diffusion models (Brooks et al., 2024; Peebles & Xie,

- 2023; Wan et al., 2025) have achieved remarkable progress in recent years, with the state-of-the-art models now being able to generate realistic shots with complex motions. Early works typically rely on bidirectional attention in diffusion transformers to generate all video frames at once. Under this paradigm, users have to wait until the model processes all frames before they can watch the video, which is slow and prevents interactive generation. Recently, autoregressive video diffusion models (Chen et al.,
- 2024; Yin et al., 2025; Huang et al., 2025) have shifted the

1Tsinghua University 2Microsoft Research Asia 3University of Science and Technology of China 4Johns Hopkins University 5ETH Z¨urich. †Work done during Hang Guo and Zhaoyang Jia’s internship at Microsoft Research Asia.

Preprint. January 29, 2026.

cache of past T-1 frames evicted caches current frame features

- head1
- head2 head3

AttentionHeadIndex

Non-dummy Heads

use historical frames

Dummy Heads

headN

weak past frame usage

headN-1

frame1 frame2 frame3 frame T-1 frame T

Frame Index

Figure 1. Weak context utilization in the multi-head attention of existing methods, e.g., Diffusion Forcing (Chen et al., 2024), Self Forcing (Huang et al., 2025). Naively pruning all KV caches of 25% heads results in only a marginal performance drop (84.0 vs. 83.78) while speedup inference from 17.6FPS to 19.6FPS.

paradigm to frame-by-frame generation, where each autoregressive step iteratively denoises to produce clean frames. Unlike previous bidirectional counterparts, autoregressive video diffusion supports a caching mechanism and can aggregate historical context stored in the KV cache through self-attention. By combining the strengths of autoregressive modeling and diffusion denoising, current methods allow sequential frame modeling while maintaining high visual quality. However, existing methods still face efficiency challenges when processing long visual token sequences. For instance, the KV cache length for past frames increases significantly in computation-dense tasks such as long videos or high-resolution videos. Although most current works (Yang et al., 2025a; Liu et al., 2025) improve efficiency at the input level by employing a sliding window strategy to restrict the model’s attention only to recent frames, the model’s internal utilization on contextual frames still remains a black box and has been largely unexplored.

To this end, we delve deep into the multi-head self-attention layers of existing models, as they are responsible for context aggregation. Surprisingly, we found most autoregressive video diffusion pipelines (Chen et al., 2024; Huang et al., 2025; Liu et al., 2025) are inefficient in context utilization: about 25% attention heads disproportionately allocate large attention scores (over 80%) to the current frame, even with access to historical frames (Sec. 3.2 gives a detailed discussion). Furthermore, as shown in Fig. 1, we remove the KV cache corresponding to these heads during inference, and this simple modification causes only a 0.26% performance drop. These observations suggest that the pre-trained

###### Efficient Video Generation Long-context Video Generation

- 0.78 FPS speedup

WAN2.1 Ours

22.5× 6.58×longer cache size

Self Forcing 17.56 FPS 24.30 FPS

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- 1.0×baseline speedup 31.2×

shot1 shot2 shot3 shot4 speedup speedup speedup

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

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

High-resolution Video Generation

480P Video 720P video 1080P video 1.4×speedup 1.6×speedup 2.0×speedup

(compared with the original model)

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

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

Figure 2. The proposed Dummy Forcing can be applied to (1) efficiently generate videos, (2) overcome quadratic complexity in high-resolution video generation, and (3) enlarge context lengths without increasing computational overhead.

models have learned a shortcut to prioritize certain attention heads to perform context aggregation, while leaving the other heads to mainly refine the current frame. Since the latter heads do not actually work on context aggregation, we refer to them as “dummy heads” in this paper.

Based on the above insight, we propose Dummy Forcing to compress redundant contextual information for efficient autoregressive video diffusion models. Specifically, we develop the Heterogeneous Memory Allocation (HMA), which assigns adaptive context lengths based on distinct head types to eliminate redundant historical frames. Subsequently, we introduce Dynamic Head Programming (DHP), which maximizes an information retention objective and uses a dynamic programming algorithm to derive optimal head classification results. Finally, we employ Packed Attention Forward (PAF) to achieve more aggressive dummy head numbers through adjust the classification boundary between dummy and non-dummy heads.

In short, we make the following key contributions:

- I. We identify that current autoregressive video diffusion models exhibit inefficient context utilization. Furthermore, we observe dummy heads in these models whose attentions are exclusively focused on the current frame, even though past frames are available.
- II. We introduce Dummy Forcing to efficiently compress the redundant context of dummy heads. The proposed method enables adaptive cache management and head classification while achieving high compression ratios.
- III. As shown in Fig. 2, we apply Dummy Forcing to multiple video generation applications. Without any training, the proposed method achieves up to 2.0× end-to-end acceleration compared to the baseline and can generate videos at speeds exceeding 24 FPS.

- 2. Related Work

rectional attention during the denoising process. With largescale training, these billion-parameter models have achieved impressive results (Yang et al., 2024; Wan et al., 2025; Kong

- et al., 2024). However, bidirectional attention (Vaswani et al., 2017) incurs significant quadratic complexity, making it challenging to support long video generation where the token sequence can be very long. Recently, autoregressive video diffusion models (Chen et al., 2024; Henschel et al., 2025; Teng et al., 2025; Chen et al., 2025a; Yin et al., 2025; Gu et al., 2025) have made substantial progress. By enabling streaming generation and KV caching, these methods significantly reduce the computational cost of video synthesis. CausVid (Yin et al., 2025) distills a bidirectional WAN (Wan
- et al., 2025) into an autoregressive one for fast generation. Self Forcing (Huang et al., 2025) addresses the train–test mismatch by conditioning on frames from the model’s outputs. RealTime (Millon, 2025) scales the parameters in Self Forcing further to 14B. LongLive (Yang et al., 2025a) introduces KV re-caching to enable interactive video generation. SANA-Video (Chen et al., 2025b) proposes linear attention for efficient long video modeling. Rolling Forcing (Liu et al., 2025) designs a joint denoising scheme with varying noise levels to reduce error accumulation.

KV Cache Compression in LLMs. KV cache pruning has been extensively studied in LLMs due to the inherently autoregressive nature of language (Wan et al., 2024; Oren et al., 2024; Qin et al., 2025; Cai et al., 2025; Jiang et al., 2024; Li et al., 2024; Cai et al., 2024). For instance, StreamingLLM (Xiao et al., 2023) observes that attention scores frequently concentrate on the initial tokens and proposed retaining sink tokens during long-context modeling. H2O (Zhang et al., 2023) and its variants highlight the importance of intermediate tokens, utilizing token importance scores derived from attention maps to preserve pivotal tokens. There are also studies analyzing cache utilization across different heads in LLMs. For instance, DuoAttention (Xiao et al., 2024) categorizes retrieval heads and streaming heads, compressing the KV cache of streaming heads to retain only recent tokens and attention sinks. FastGen (Ge et al., 2023) further proposes a finer-grained head classification where each head handles tokens of varying lengths. However, these methods focus on token-level com-

Video Diffusion Models. Early video diffusion models (Polyak et al., 2024; Chen et al., 2023; Brooks et al.,

- 2024; Zhang et al., 2025a; Zhang & Agrawala, 2025) typically feed all frames into the model at once and employ bidi-

pression and do not consider frame-level redundancy in video data. Furthermore, we find that KV cache compression for video models can be more aggressive, with the cache of dummy heads potentially all removed.

Efficient Video Generation. Early video generation models typically adopt DiT (Peebles & Xie, 2023) architectures with bidirectional attention. Since these methods generate all video frames in a single pass, they do not support KV caching. As a result, prior acceleration methods for video generation often rely on sparse attention to speed up fixedlength attention computation (Zhang et al., 2025b; Xia et al.,

- 2025; Zhang et al., 2025c; Sun et al., 2025). For example, SVG (Xi et al., 2025; Yang et al., 2025b) introduces spatial and temporal heads to capture distinct intra-frame and interframe sparsity patterns. However, in autoregressive video diffusion models, attention mask shifts from bidirectional to causal, and the context length varies with autoregressive steps. This makes existing video sparse attention methods difficult to apply directly, as variable-length attention would require recompiling kernels at each step.

### 3. Method 3.1. Preliminary

Autoregressive video diffusion models decompose video synthesis into a frame-by-frame process, where each autoregressive (AR) step generates a clean context via an iterative denoising process. Formally, given a video sequence consisting of T frames X = {x1,x2,··· ,xT}, an autoregressive diffusion model is trained to learn the joint distribution under the following causal factorization:

p(x1,x2,··· ,xT) =

T

p(xi | x1,x2,··· ,xi−1), (1)

i=1

where the conditional distribution p(xi|x1,x2,··· ,xi−1) is learned by denosing from a Gaussian distribution using a diffusion model (Ho et al., 2020; Song et al., 2020; Lipman et al., 2022). Notably, some works (Huang et al., 2025; Yang et al., 2025a) model xi as a chunk consisting of multiple frames to encourage temporal consistency. For presentation clarity, we keep xi as a single-frame format in the following part of this paper.

Thanks to the autoregressive property, the condition from previous AR steps (x1,x2,···xi−1) can be modeled through the KV cache. To mitigate the quadratic computational complexity induced by increasing cache length, existing methods (Yang et al., 2025a; Millon, 2025; Liu et al., 2025) typically adopt a sliding-window strategy: a cache window of size L is maintained, consisting of 1 sink frame and L − 1 neighboring frames. In detail, the sink frame occupies only a small number of frames and remains fixed when the context window changes. For instance, in short

0 sum HW sum LHW sum

(L+1)HW

| | |
|---|---|
| |[Figure 61]|

[Figure 62]

[Figure 63]

mean

[Figure 64]

Sink frames Neighbor frames Current frames

Figure 3. We compute the frame attention score by summing across rows and averaging across columns within the sink/neighbor/current frame group.

video generation, most methods commonly select the first frame as the sink frame. The sink frame primarily serves as a global anchor via the attention sink mechanism (Xiao et al., 2023; Gu et al., 2024), thereby encouraging temporal consistency across videos. In contrast, the neighbor frames are dynamically updated to capture temporal dependencies as the sliding window moves.

Under this cache management, the multi-head self-attention in existing autoregressive video diffusion models is trained to aggregate cross-frame information. Formally, denote the frame index of sink frame as s, at the i-th AR step, the self-attention layers work as follows:

softmax(Qi[Ks,Ki−L+1:i−1,Ki]⊤)[Vs,Vi−L+1:i−1,Vi],

(2) where [·] denotes concatenation along token sequence.

##### 3.2. Motivation

Current autoregressive video diffusions face efficiency challenges when handling long cache sequences. In this section, we delve into the multi-head self-attention layer to explore how the i-th frame interacts with its previous L frames.

Profiling Setups. To quantitatively depict how queries from the current frame Qi interact with keys from the sink frames Ks, the neighbor frames Ki−L+1:i−1, and the current frame Ki, as shown in Fig. 3, we define the following “frame attention score”:

HW

1 HW

αr =

Auv,

(3)

u=1 v∈Jr

r ∈ {sink,neighbor,current},

where A ∈ RHW×(L+1)HW denotes the attention map, Jsink = [0,HW), Jneighbor = [HW,LHW), Jcurrent = [LHW,(L+1)HW), and HW represents the total number of visual tokens of one frame. Note that r αr = 1. In the main paper, we use the Self Forcing model (Huang et al., 2025) as a representative, and provide results on other models in Appendix. Based on the above setup, we give the following three key observations.

Observation 1: Certain heads under-utilize past context. As shown in Fig. 4(c), we find there are about 25% attention heads that assign over 80% attention weights to the

(a) Attn Distribution on Sink Frames (b) Attn distribution on Neighbor Frames (c) Attn distribution on Current Frames

[Figure 65]

[Figure 66]

- 2

4

6

8

10

12

density

[Figure 67]

attention value

- 0.0 0.2 0.4 0.6 0.8 1.0
- 1
- 2

4

- 6

density

0

3

5

- 7

[Figure 68]

attention value

- 0.0 0.2 0.4 0.6 0.8 1.0
- 1
- 2

density

0

- 3
- 4

|Text Prompts|
|---|

|Denoising Timesteps|
|---|

|AR Steps|
|---|

Conditions

core set ratio

(d) Position Stability of Dummy Heads

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0.92

0.85

0.76

Current frame takes over 80%

attention score in some heads

- Figure 4. (a)-(c): We gather attention maps from all heads and use Eq. (3) to compute the frame attention scores on the sink/neighbor/current frames. (d): The core set ratio under different conditions. For each bar, we change the corresponding condition while keeping the others fixed. We provide more implementation details of the observation experiment in the Appendix.

current frame. In other words, the Qi in these heads attends almost exclusively on Ki, even though Ks and Ki−L+1:i−1 is available. Since these heads fail to work on frame aggregation, we refer to them as “dummy heads” in this paper. To identify a given number of N dummy heads, we calculate the frame attention scores on current frames, i.e., αcurrent, and then select the topN largest values as dummy heads.

- Observation 2: Dummy head exhibits position stability. We further study the dummy heads’ position when conditions change, e.g., text prompts, AR steps, and denoising

timesteps. Denote I = {(ln,hn)}Nn=1 as the dummy head location in the ln-th layer and hn-th head. We obtain Ic, where c = 1,2,··· ,C, under C varying conditions by computing αcurrent and select the topN. We then employ the core set ratio to quantify the variation of Ic. The core set ratio is defined as N1 |I1 ∩I2 ∩···∩IC| ∈ [0,1]. As shown in Fig. 4(d), the position of dummy heads remained largely unchanged as condition changes. For example, 92% head index repeatedly appears across all given AR steps.

- Observation 3: Pruning cache of dummy heads incurs only slight drop. Since little attention is paid to past frames in the dummy head, we attempt to remove their KV caches. Specifically, we sample one single condition and use TopN selection described above to drive I consisting of the location index of N dummy heads, where N is set to 25% of the total number of heads. We then fix I and apply it throughout the evaluation on VBench (Huang et al., 2024a). We remove all KV caches for dummy heads while leaving non-dummy heads intact. As shown in Tab. 1, this naive compression strategy works with a 0.26% drop. In contrast, random cache eviction severely degrades performance.

- 3.3. Dummy Forcing

0

0.0 0.2 0.4 0.6 0.8 1.0

attention value

Table 1. Results of pruning dummy heads’ KV cache. “Random” denotes randomly selecting 25% heads and evicting their caches.

setups FPS speedup Quality Semantic Total Self Forcing 17.6 1.0× 84.73 81.03 84.00 Random 19.6 1.1× 79.58 78.15 79.30 Dummy Head 19.6 1.1× 84.42 81.22 83.78

Head Programming adaptively classify different head types by formalizing an optimization problem; (c) Packed Attention Forward extends context of dummy heads to mitigate the classification boundary between classes for aggressive dummy head numbers. More details are given below.

Heterogeneous Memory Allocation. Beyond compressing the context of the dummy head as shown in Sec. 3.2, we demonstrate in Sec. 4.5 that certain non-dummy heads exhibit high attention scores toward sink frames in the KV cache while utilizing few other historical frames. To further compress cache length, we subdivide non-dummy heads into “sink heads” and “neighbor heads”. For sink heads, we remove redundant L − 1 recent frames and restrict attention queries to only observe 1 sink frame and 1 current frame. For neighbor heads, we retain a sliding local window of size L − 1 along with 1 current frame. Note that the sink frame is not included in the context of neighbor heads since it is exclusively modeled through sink heads. As shown in Fig. 5(a), we categorize heads into three types: sink head, neighbor head, and dummy head, each possessing specific contextual information. Formally, at the i-th AR step, the self-attention is as follows:

sink : softmax(Qi[Ks,Ki]⊤)[Vs,Vi], neighbor : softmax(QiKi−L+1:i)Vi−L+1:i, dummy : softmax(QiKi⊤)Vi.

(4)

Based on the observation in Sec. 3.2, we propose Dummy Forcing to achieve more aggressive acceleration and better performance retention.

Overview. As shown in Fig. 5, the proposed method comprises three components: (a) Heterogeneous Memory Allocation assigns head-specific KV cache lengths to achieve compression of redundant historical frames; (b) Dynamic

Since attention heads are partitioned into three types, the original multi-head self-attention is modified as follows: we first extract three head groups through indexing along the head dimension. Then, we use Eq. (4) to aggregate context according to different head types. Finally, we place the attention outputs back into their original positions. We use Triton (OpenAI, 2021) to reduce the additional overhead caused by the above head indexing and placement.

sink neighbor frames current

[Figure 69]

[Figure 70]

h = max( h,0, h,1)

[Figure 71]

[Figure 72]

[Figure 73]

Launch Attention Kernel

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]<br><br>[Figure 77]|
|---|

|0.25|0.25|0.5|
|---|---|---|
|0.05|0.05|0.9|
|0.15|0.05|0.8|
|0.05|0.25|0.7|
|0.1|0.4|0.5|
|0.3|0.1|0.6|
|0.05|0.15|0.8|
|0.1|0.3|0.6|

[Figure 78]

[Figure 79]

<dummy>

0.25

0.25 0.05

Sink head

<dummy>

[Figure 80]

[Figure 81]

Sink Head

[Figure 82]

[Figure 83]

<dummy>

AttentionHeads

| |PACK|
|---|---|

0.15 0.25

0.15 0.25 0.40 0.30

Dummy head

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]<br><br>[Figure 87]|
|---|

[Figure 88]

[Figure 89]

<local>

[Figure 90]

[Figure 91]

<local> <sink>

Neighbor Head

Launch Attention Kernel

[Figure 92]

[Figure 93]

0.30

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]|
|---|

[Figure 98]

[Figure 99]

<dummy>

0.15 0.30

0.15 0.30

Neighbor Head

[Figure 100]

[Figure 101]

<local>

Dummy Head

(a) Heterogeneous Memory Allocation (b) Dynamic Head Programming (c) Packed Attention Forward

- Figure 5. (a) We assign different contextual receptive fields to different head types. (b) A toy example of classifying different head types, with num head=8 and N=4 in this case. (c) We fuse different heads by context packing for more aggressive compression.

Dynamic Head Programming. As shown in Fig. 4(d), the dummy head index I is not perfectly invariant under different conditions, e.g., there is about 25% discrepancy when using varying text prompts. Intuitively, an adaptive head classification scheme would yield better performance. To achieve this, we first use Eq. (3) to compute each head’s frame attention score [αsink,αneighbor,αcurrent] ∈ R3, which is then gathered across all heads to drive global frame attention score denoted as F ∈ Rnum head×3. Since F depends on the attention map, for computation efficiency, we uniformly sample a small portion of query tokens and compute an approximate attention map against all key tokens. We then formalize the adaptive head classification as the following optimization problem. Since different head classes incur corresponding frame attention loss, e.g., sink heads lack αneighbor due to the removal of neighbor frames, our objective is thus to maximize the remaining attention values while forcing N dummy heads. Precisely, let ch ∈ {sink,neighbor,dummy} be the head type at the h-th head, the optimization problem can be formularized as:

num head

fh(ch),

max

h=1

(5)

num head

s.t.

I(ch = dummy) = N,

h=1

where fh is the value function at the h-th attention head:

 

- Fh,0 + Fh,2 ch = sink,
- Fh,1 + Fh,2 ch = neighbor,
- Fh,2 ch = dummy.

(6)

fh(ch) =



We point out that Eq. (5) is a classic dynamic programming problem and proof in Appendix that the following greedy algorithm achieves optimality: for each head h, we compute ℓh = max(Fh,0,Fh,1) to obtain the opportunity cost of forcing the h-th head to be dummy. Then we sort {ℓh}numh=1 head and assign the N smallest elements as dummy heads. For the remaining heads, we set ch = sink if Fh,0 ≥ Fh,1 else ch = neighbor.

Packed Attention Forward. As shown in Tab. 1, although removing KV caches of 25% heads results in minimal performance loss, the resulting speedup remains limited. Intuitively, further increasing the number of dummy heads N should bring additional acceleration. However, as shown in Sec. 4.5, naively enlarging N causes noticeable performance drops due to boundary mis-classification between non-dummy and dummy heads, pruning caches on contextcritical heads. To address this, we propose to shift the classification boundary by slightly extending the context of dummy heads to additionally include some boundary heads. Specifically, given the temporal dependency decay in video frames, we additionally append the (i − 1)-th frame as a packing frame into the context of dummy heads. After this improvement, the attention in dummy heads is as follows:

pack dummy : softmax(Qi[Ki−1,Ki]⊤)[Vi−1,Vi]. (7)

As an additional benefit, dummy heads and sink heads now share the same effective context length. This enables us to pack both head types into a single attention call. As a result, the number of attention kernel launches is reduced from three to two. Empirically, the reduction in kernel launches largely compensates for the slightly enlarged context length, while enabling more than 50% of heads to be configured as dummy heads without noticeable quality degradation.

### 4. Experiments

##### 4.1. Experimental Setup

In the main paper, we apply the proposed method on state-ofthe-art (SoTA) autoregressive diffusion methods, including Self Forcing (Huang et al., 2025) and LongLive (Yang et al., 2025a). We also provide results of other models in the Appendix. For tasks, we evaluate on classic video generation, including 5s short video and 30s long video in Sec. 4.2, highresolution 720P&1080P video generation in Sec. 4.3, and long-context video generation in Sec. 4.4. Unless specified, we set the dummy head number to 50% of total heads as the default. Due to page limit, we provide more implementation details in the Appendix.

- Table 2. Quantitative comparison on efficiency and quality on 5 second short video generation with VBench. The marks ◦, ⋄,• denote bidirectional models, autoregressive models, and inference acceleration methods, respectively. “FPS” denotes the number of frames generated per-second tested with one single H100 GPU. “Dummy Forcing” is our proposed method.

Methods

Inference Efficiency VBench Score

#Param Resolution FPS↑ Speedup↑ Quality Score↑ Semantic Score↑ Total Score ↑

- ◦LTX-Video 1.9B 768 × 512 8.98 - 82.30 70.79 70.79
- ◦Wan2.1 1.3B 832 × 480 0.78 - 84.26 80.09 80.09 ⋄SkyReels-V2 1.3B 960 × 540 0.49 - 82.67 84.70 74.53 ⋄MAGI-1 4.5B 832 × 480 0.19 - 79.18 82.04 67.74 ⋄CausVid 1.3B 832 × 480 17.56 - 81.20 84.05 69.80 ⋄RealTime 14.0B 832 × 480 2.89 - 85.24 81.69 84.53 ⋄Self Forcing 1.3B 832 × 480 17.56 1.0× 84.73 81.03 84.00

- •+ Infinipot-V 1.3B 832 × 480 19.54 1.1× 82.72 79.05 81.99

- •+ R-KV 1.3B 832 × 480 18.41 1.1× 83.46 79.60 82.69

- •+ TeaCache 1.3B 832 × 480 21.82 1.2× 84.22 80.47 83.47

- •+ Dummy Forcing 1.3B 832 × 480 24.30 1.4× 84.63 80.98 83.90 ⋄LongLive 1.3B 832 × 480 17.57 1.0× 83.64 81.08 83.13

- •+ Infinipot-V 1.3B 832 × 480 19.54 1.1× 82.70 80.92 82.35

- •+ R-KV 1.3B 832 × 480 18.41 1.1× 82.92 81.26 82.59

- •+ TeaCache 1.3B 832 × 480 21.82 1.2× 83.10 80.73 82.63

- •+ Dummy Forcing 1.3B 832 × 480 24.30 1.4× 83.28 80.89 82.80

- Table 3. Quantitative comparison on efficiency and quality on 30s long video generation with VBench-Long.

method FPS↑ Speedup↑ Quality↑ Semantic↑ Total↑ SkyReels-V2 0.49 - 80.77 53.37 75.29 FramePack 0.92 - 83.61 75.32 81.95 Self-Forcing 17.56 1.0× 84.36 80.18 83.53 +TeaCache 21.82 1.2× 83.82 79.74 83.00 +Ours 24.30 1.4× 84.28 78.85 83.19 LongLive 17.57 1.0× 83.34 80.34 82.74 +TeaCache 21.82 1.2× 82.87 80.20 82.33 +Ours 24.30 1.4× 83.33 79.56 82.57

- 4.2. Classic Video Generation

Table 4. Quantitative VBench results on high-resolution video generation at 720P (1280×720) and 1080P (1920×1088).

Methods resolution FPS↑ speedup↑ Quality↑ Semantic↑ Total↑ Self Forcing 720P 5.6 1.0× 84.77 81.93 84.20 +Ours 720P 9.1 1.6× 84.77 81.61 84.14 LongLive 720P 5.6 1.0× 83.82 81.50 83.36 +Ours 720P 9.1 1.6× 83.21 81.91 82.95 Self Forcing 1080P 1.3 1.0× 84.90 78.68 83.66 +Ours 1080P 2.6 2.0× 84.15 78.76 83.07 LongLive 1080P 1.3 1.0× 81.86 80.78 81.65 +Ours 1080P 2.6 2.0× 81.97 80.39 81.65

Self Forcing, our Dummy Forcing achieves 1.4× speedup, generating video at 24.3FPS in real time with only a 0.1% drop in quality. In Appendix, we provide further discussion on cache compression ratios, speedup effects on a single attention layer and qualitative visualization results.

Comparison on Short Video Generation. Since few studies have explored accelerating autoregressive video diffusion models, we carefully select SoTA methods from related domains for comparison with other inference speedup approaches. Specifically, R-KV (Cai et al., 2025) employs token-level KV cache compression for LLM reasoning, Infinipot-V (Kim et al., 2025) prunes caches for streaming video understanding, and TeaCache (Liu et al., 2024) skips denoising timesteps for DiT acceleration. We report results in Tab. 2. For previous KV cache pruning methods, R-KV and Infinipot-V, since they calculate token importance at each AR step, the additional time introduced by token selection algorithm undermines the benefits gained from reduced cache length, resulting in a marginal 1.1× overall speedup ratio. For diffusion step skipping method TeaCache, the acceleration gain is limited given that current base models are already few-step diffusion models. In contrast, our method achieves the most aggressive acceleration while maintaining the best performance. For instance, in

Comparison on Long Video Generation. Tab. 3 gives the VBench-Long (Huang et al., 2024b) evaluation results on generating 30s long videos. Our Dummy Forcing achieves better efficiency and performance than competitive benchmarks, allowing 1.4× end-to-end speedup with up to a 0.4% quality drop. In Fig. 6, we present quantitative comparison results, and our method achieves faster generation while maintaining strong motion and high visual quality.

##### 4.3. High-resolution Video Generation.

As resolution increases, the number of cached visual tokens grows quadratically, leading to inefficiency in existing methods. In this section, we explore applying the proposed Dummy Forcing in generating videos at higher resolutions such as 720P and 1080P. Specifically, we discover that cur-

LongLive

0-6s 6-12s 12-18s 18-24s 24-30s

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

1.4× speedup

+Ours

Prompt: A dynamic skateboarding scene captured mid-air, showing a young skateboarder losing control ...

- Figure 6. Quantitative comparison on 30s long video generation . Our Dummy Forcing achieves 1.4× end-to-end acceleration without compromising output quality. More results are provided in the Appendix.

rent autoregressive video diffusion models exhibit strong zero-shot capabilities for low-to-high resolution video generation. To leverage this, we modify the shape of the initial Gaussian noise in each AR step to allow high-resolution generation. Experimental results are presented in Tab. 4. As the length of tokens to be processed increases, the speedup achieved by our method becomes more pronounced. For instance, ours achieves even 2.0× acceleration without quality drop on 1080P video with LongLive model. This result demonstrates the generalizability of the proposed method.

##### 4.4. Long-context Video Generation

Current autoregressive video generation models typically employ sliding windows to avoid excessively long sequences. For instance, SoTA methods (Yang et al., 2025a) can only observe the past 36 frames. Consequently, during shot transitions in narrative scenarios, e.g., a character disappears and reappears, current methods inevitably produce inconsistent videos. Here, we explore long-context video generation to evaluates different methods’ historical context length under similar runtime cost. Detailed experimental settings are given in Appendix. For our Dummy Forcing, we allocate cache budget saved from the dummy&sink heads to the neighbor heads, thereby enabling longer effective cache. For comparative baselines, we consider the LongLive model with and without the sliding window strategy. As shown in Tab. 5, LongLive without a local window suffers from computational complexity when processing long visual tokens. In contrast, the proposed Dummy Forcing can achieve similar cache length while delivering a 1.93× speedup. On the other hand, compared with LongLive using a sliding window, our Dummy Forcing achieves better generation quality (68.45 v.s. 69.48) thanks to our 6.58× longer cache size. Furthermore, we provide a quantitative comparison in Fig. 7.

- As can be seen, current methods tend to “re-generate” a new identity due to limited context, while ours can accurately reproduce previously vanished characters and backgrounds.
- At last, we also evaluate on 60s interactive video generation, see the Appendix for results.

- Table 5. Quantitative comparison on long-context video generation. #cache denotes the number of past cached frames. “sw” means using the local sliding window strategy.

setups

Context Efficiency

Score↑ #cache↑ ctx ratio↑ FPS↑ speedup↑

LongLive w/o sw 237 6.58× 9.36 1.00× 69.38 LongLive w/ sw 36 1.00× 17.57 1.87× 68.45 Dummy Forcing 237 6.58× 18.14 1.93× 69.48

- Table 6. Ablation experiments of different components on VBench. setups FPS Quality Semantic Total

- (0)Self Forcing (baseline) 17.6 84.73 81.03 84.00
- (1)combine sink&neighbor 22.1 84.23 81.09 83.60
- (2)w/o packing in dummy head 24.2 83.91 80.86 83.30
- (3)Dummy Forcing (ours) 24.3 84.63 80.98 83.90

##### 4.5. Ablation Studies

Effectiveness of Different Components. To verify different design choices in the proposed method, we ablate on other alternative configurations in Tab. 6. (1) We combine sink and neighbor head class into a unified non-dummy head type, whose context consists of 1 sink frame and L − 1 recent frames. Results show this variant achieves reasonable performance, but its acceleration is suboptimal. This is because although both sink head and neighbor head are non-dummy heads, they focus on distinct parts of the context, and simply merging them would result in redundancy. (2) We remove the packing frame in dummy heads and call three separate attention. Results show that it degrades performance while offering negligible speed gains. This is because naively increasing dummy head number without adjusting the classification boundary between dummy and non-dummy impairs the information aggregation of contextcritical heads. And the runtime from additional attention calls negates the benefits from reduced cache length.

Different Dummy Ratios. In the proposed Dummy Forcing, the number of dummy heads N serves as a critical hyperparameter to balance efficiency and performance. Here, we

###### Shot Transition Shot Transition

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

LongLive

inconsistent with shot1

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

| |
|---|

6.58× longer

+Ours

consistent with shot1

A wide shot of Gwen Stacy sitting comfortably in a cozy library or study room. She is seated in a cushioned chair,

|Shot1:|
|---|

Gwen Stacy stretches her arms

A middle-aged bookstore owner standing inside a small independent

|Shot2:|
|---|

|Shot3:|
|---|

upward in a relaxed, full-body stretch,

clearly indicating the moment after finishing reading.

bookstore on a rainy city street at dusk.

holding a book gently with both hands ...

He carefully stacks and aligns ...

- Figure 7. Quantitative comparison on long-context video generation. We design a set of text prompts with a storytelling narrative that involves two scene transitions, where each shot is interactively generated through the corresponding prompt. See more results in Appendix.

[Figure 136]

[Figure 137]

150 200 250 300 350

Number of dummy heads

VBenchTotalScore

68

70

72

74

76

78

80

82

84

TimeperARstep(ms)

Number of dummy heads

150 200 250 300 350

540

560

580

600

620

Vbench with different #dummy head Runtime with different #dummy head

- Figure 8. Ablation experiments on Vbench scores and average runtime per AR step across different numbers of dummy heads.

Neighbor head distribution across layers Dummy head distribution across layers

10

[Figure 138]

[Figure 139]

- 0
- 1
- 2
- 3
- 4

Numberofheads

Numberofheads

8

6

4

2

0

Layer index Layer index

Figure 9. Distribution of the number of neighbor heads and dummy heads across different layers.

Table 7. Results of combining our method and TeaCache.

###### Self-Forcing LongLive

methods

set different N to investigate its impact. We employed the Self Forcing model, which consists of a total of 360 heads. As shown in Fig. 8, when N is set below 240, the proposed method maintains relatively stable performance compared to N = 0. This phenomenon indicates that nearly 2/3 heads do not fully utilize past frames. In contrast, reducing the dummy head count from 300 to 360 causes significant degradation. In this case, KV cache pruning harms neighbor heads, which are crucial for context aggregation.

FPS speedup VBench FPS speedup VBench

original models 17.6 1.0× 84.00 17.6 1.0× 83.13 +Ours&TeaCache 30.5 1.7× 83.26 30.5 1.7× 82.47

gates information from the current frame in shallow layers to abstract high-level features. Subsequently, it queries past frames in this high-level semantic space in intermediate layers for information aggregation. In the last few layers, the model refines the current frame and returns low-level space for subsequent decoding.

### 5. Discussion

Compatibility with other Acceleration Methods. Since our Dummy Forcing works across attention heads, we further combine it with the previously best-performing method TeaCache, yielding a variant “Ours+TeaCache”. As shown in Tab. 7, the combination of these two orthogonal methods achieves larger generation speeds, e.g., over 30FPS, demonstrating the potential for integration with other methods.

Head Distribution across Layers. Since the proposed method can automatically identify dummy head positions, it is interesting to observe the distribution of different head classes across layers. In Fig. 9, we present the number of neighbor and dummy heads per layer averaged across 100 prompts. Roughly, dummy heads primarily appear in the first and last few layers, while neighbor heads cluster in intermediate layers. To explain this, the model first aggre-

### 6. Conclusion

In this work, we identify the dummy heads in existing autoregressive video diffusion models. Building on this observation, we propose Dummy Forcing, which employs heterogeneous memory allocation to expose different attention heads with varying context lengths. We further introduce dynamic head programming that derives an optimal greedy policy to perform online head classification. For aggressive dummy head numbers, we propose packed attention forward, which extends context length without incurring overhead. We apply Dummy Forcing to multiple downstream tasks, including efficient video generation, high-resolution video generation, and long-context video generation. Extensive experiments demonstrate the generality of our method.

### Impact Statement

The primary goal of this work is to advance the efficiency and scalability of machine learning models, enabling faster video generation and more effective use of computational resources without additional training. Since this work is developed based on pre-trained video generation models, potential risks associated with synthetic video generation, such as misinformation or misuse, are inherited from the broader class of generative video models. We expect the broader impact of this work to be largely positive by facilitating more efficient and scalable deployment of advanced video generation models while remaining aligned with existing ethical considerations in generative modeling.

Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.

Chen, J., Zhao, Y., Yu, J., Chu, R., Chen, J., Yang, S., Wang, X., Pan, Y., Zhou, D., Ling, H., et al. SANA-Video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025b.

Ge, S., Zhang, Y., Liu, L., Zhang, M., Han, J., and Gao, J. Model tells you what to discard: Adaptive KV cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.

### References

Beaumont, R., Schuhmann, C., and Contributors. LAION-AI/aesthetic-predictor: A linear estimator on top of CLIP to predict the aesthetic quality of images. https://github.com/LAION-AI/ aesthetic-predictor, 2022.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.

Cai, Z., Zhang, Y., Gao, B., Liu, Y., Li, Y., Liu, T., Lu, K., Xiong, W., Dong, Y., Hu, J., et al. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.

Cai, Z., Xiao, W., Sun, H., Luo, C., Zhang, Y., Wan, K., Li, Y., Zhou, Y., Chang, L.-W., Gu, J., et al. R-KV: Redundancy-aware KV cache compression for trainingfree reasoning models acceleration. arXiv preprint arXiv:2505.24133, 2025.

Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021.

Chen, B., Mart´ı Mons´o, D., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion Forcing: Nexttoken prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081– 24125, 2024.

Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al. Skyreelsv2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025a.

- Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., Wang, Y., and Lin, M. When attention sink emerges in language models: An empirical view. arXiv preprint arXiv:2410.10781, 2024.
- Gu, Y., Mao, W., and Shou, M. Z. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

Henschel, R., Khachatryan, L., Poghosyan, H., Hayrapetyan, D., Tadevosyan, V., Wang, Z., Navasardyan, S., and Shi, H. StreamingT2V: Consistent, dynamic, and extendable long video generation from text. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2568–2577, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. Self Forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al. Vbench: Comprehensive benchmark suite for video generative models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024a.

Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024b.

Jiang, H., Li, Y., Zhang, C., Wu, Q., Luo, X., Ahn, S., Han, Z., Abdi, A. H., Li, D., Lin, C.-Y., et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

Kim, M., Shim, K., Choi, J., and Chang, S. Infinipot-v: Memory-constrained kv cache compression for streaming video understanding. arXiv preprint arXiv:2506.15745, 2025.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. SnapKV: LLM knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947– 22970, 2024.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, F., Zhang, S., Wang, X., Wei, Y., Qiu, H., Zhao, Y.,

- Zhang, Y., Ye, Q., and Wan, F. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108, 2024.

Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. Rolling Forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

Millon, E. Krea Realtime 14B: Real-time video generation,

2025. URL https://github.com/krea-ai/ realtime-video.

OpenAI. Triton: Open-source gpu programming for neural networks. https://openai.com/index/ triton/, 2021.

Oren, M., Hassid, M., Yarden, N., Adi, Y., and Schwartz, R. Transformers are multi-state RNNs. arXiv preprint arXiv:2401.06104, 2024.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Qin, Z., Cao, Y., Lin, M., Hu, W., Fan, S., Cheng, K., Lin, W., and Li, J. CAKE: Cascading and adaptive KV cache eviction with layer preferences. arXiv preprint arXiv:2503.12491, 2025.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural

language supervision. In International Conference on Machine Learning, pp. 8748–8763. PmLR, 2021.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Sun, W., Tu, R.-C., Ding, Y., Jin, Z., Liao, J., Liu, S., and Tao, D. VORTA: Efficient video diffusion via routing sparse attention. arXiv preprint arXiv:2505.18809, 2025.

Team, Q. et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024.

Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I., et al. Attention is all you need. Advances in Neural Information Processing Systems, 30(1):5998–6008, 2017.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. WAN: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wan, Z., Wu, X., Zhang, Y., Xin, Y., Tao, C., Zhu, Z.,

- Wang, X., Luo, S., Xiong, J., Wang, L., et al. D2O: Dynamic discriminative operations for efficient long-context inference of large language models. arXiv preprint arXiv:2406.13035, 2024.
- Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al. Internvid: A largescale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.

Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al. Sparse VideoGen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Xia, Y., Ling, S., Fu, F., Wang, Y., Li, H., Xiao, X., and Cui, B. Training-free and adaptive sparse attention for efficient long video generation. arXiv preprint arXiv:2502.21079, 2025.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Fu, Y., and Han, S. Duoattention: Efficient long-context LLM inference with retrieval and streaming heads. arXiv preprint arXiv:2410.10819, 2024.

Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al. Longlive: Realtime interactive long video generation. arXiv preprint arXiv:2509.22622, 2025a.

Yang, S., Xi, H., Zhao, Y., Li, M., Zhang, J., Cai, H., Lin, Y., Li, X., Xu, C., Peng, K., et al. Sparse VideoGen2: Accelerate video generation with sparse attention via semanticaware permutation. arXiv preprint arXiv:2505.18875, 2025b.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. CogvideoX: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast autoregressive video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22963–22974, 2025.

Zhang, D. J., Wu, J. Z., Liu, J.-W., Zhao, R., Ran, L., Gu, Y., Gao, D., and Shou, M. Z. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. International Journal of Computer Vision, 133(4):1879– 1893, 2025a.

Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., and Chen, J. Spargeattn: Accurate sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025b.

Zhang, L. and Agrawala, M. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.

Zhang, P., Chen, Y., Su, R., Ding, H., Stoica, I., Liu, Z., and Zhang, H. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025c.

- Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., R´e, C., Barrett, C., et al. H2O: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.

### A. Proof of Optimality for Greedy Strategy

In the proposed Dynamic Head Programming, we demonstrate that the dummy head assignment is in fact a dynamic programming problem that can be efficiently solved with O(nlog n) complexity using a greedy algorithm. This section provides a proof of optimality.

Recall that the original objective is

num head

max

fh(ch),

h=1

(8)

num head

s.t.

I(ch = dummy) = N,

h=1

where fh is the value function at the h-th attention head:

 

- Fh,0 + Fh,2 ch = sink,
- Fh,1 + Fh,2 ch = neighbor,
- Fh,2 ch = dummy.

(9)

fh(ch) =



Denote I∗ as the set of dummy head indices obtained from our greedy strategy. This solution achieves a total value of

num head

V ∗ =

Fh,2. (10)

max(Fh,0,Fh,1) +

h/∈I∗

h=1

Consider any alternative dummy head assignment solution I ̸= I∗, where |I| = N. Since I ̸= I∗, there exist head index i ∈ I/ I∗ and j ∈ I∗/ I, i.e., i-th head is selected as dummy head in I but not in I∗, and vice versa for j-th head. According to the definition of our greedy algorithm, we can derive ℓi ≥ ℓj, i.e.,

max(Fi,0,Fi,1) ≥ max(Fj,0,Fj,1). (11)

The value of a given policy I is VI = h/∈I max(Fh,0,Fh,1) + h num=0 head Fh,2. Consider a swap operation on I which excludes i-th head and includes j-th head, i.e. Iswap = I \ {i} ∪ {j}. Then the total value change due to this exchange is:

VIswap − VI = max(Fi,0,Fi,1) − max(Fj,0,Fj,1) = ℓi − ℓj ≥ 0. (12)

Therefore, VIswap ≥ VI. By repeatedly applying such a swap operation, we can transform any solution I into our greedy solution I∗ without decreasing the total objective value. Since each exchange either strictly increases the value or leaves it unchanged, and the process terminates at I∗, the greedy algorithm is thus optimal. □

### B. Dummy Head in Other Models

In the main paper, we present the dummy head observation based on the Self Forcing (Huang et al., 2025) model. In this section, we extend to other popular autoregressive video diffusion frameworks, including CausVid (Chen et al., 2024) and Rolling Forcing (Liu et al., 2025). Results are as follows.

CausVid (Yin et al., 2025) trains video models employing the Diffusion Forcing (Chen et al., 2024) framework, which conditions the current frame on previous frames with varying noise levels. Under this paradigm, the self-attention layer has to additionally include the noise distribution discrepancy across frames. Similar to Sec. 3.2, we present the frame attention score on the current frame in Fig. 10. It can be observed that under different conditions, there are heads exhibiting attention scores larger than 0.8 on the current frame, indicating the existence of dummy heads. For further validation, we apply the proposed Dummy Forcing to this model by setting 50% heads as dummy heads. The generated videos are shown in Fig. 15. It can be seen that removing 50% of the total KV cache does not cause significant quality degradation, demonstrating the applicability of dummy heads within the Diffusion Forcing framework.

Rolling Forcing (Liu et al., 2025) proposes a joint denoising scheme that simultaneously denoises multiple frames within a rolling window, where each frame exhibits progressively increasing noise levels. During each model forward, Rolling

###### ARstep2_DenoiseTime0 ARstep2_DenoiseTime522 ARstep6_DenoiseTime0 ARstep6_DenoiseTime522

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Attention Score Attention Score Attention Score Attention Score

- Figure 10. Frame attention scores on the current frame in the CausVid model (Yin et al., 2025). We present the distribution across different AR steps and denoising timesteps.

[Figure 144]

[Figure 145]

[Figure 146]

Attention Score Attention Score Attention Score Attention Score

[Figure 147]

ARstep6_DenoiseTime1000 ARstep6_DenoiseTime522 ARstep12_DenoiseTime1000 ARstep12_DenoiseTime522

- Figure 11. Frame attention scores on the current frame in the Rolling Forcing model (Liu et al., 2025). We present the distribution across different AR steps and denoising timesteps.

Forcing performs denoising on frames within the window using conditions from previous clean frames. In Fig. 11, we present the frame attention distribution on the current frame, and one can see that the dummy head also exists. We then use Dummy Forcing on this model with 50% dummy heads and provide the generated results in Fig. 16. As can be seen, there is no noticeable degradation in video quality, demonstrating the versatility of the dummy head in the Rolling Forcing framework.

### C. More Implementation Details

Setup for Observation Experiments. In this section, we provide detailed settings for the profiling experiments described in Sec. 3.2. Specifically, for Fig. 4(a)-(c), we collect attention maps across all attention heads of the model. We use the average results from 100 text prompts sampled from VBench. Additionally, we select the third AR step because the lengths of attention key for the sink/neighbor/current frames are identical at this step. We employ the last denoising step to compute attention maps to avoid the impact from noise. For Fig. 4(d), given the core-set ratio N1 |I1 ∩ I2 ∩ ··· ∩ IC| monotonically decreases with increasing C, we set C = 5 since existing models typically perform diffusion denoising for up to five steps. In this way, conditions including text prompts, AR steps and denoising time are controlled to exhibit the same value scales. For the results in Tab. 1, we employ a single text prompt “ a bicycle accelerating to gain speed” along with the third AR step and the last denoising step to obtain the attention map for all heads. Based on this attention map, we select the TopN largest αcurrent values to determine the fixed dummy head positions during Vbench evaluations.

Setup for Dummy Forcing Implementation. Here, we provide the specific implementation details of the proposed Dummy Forcing algorithm. Specifically, since the frame attention score F in our method requires attention maps, we uniformly sampled 25% of the original query tokens for efficient estimation. These tokens were then used to compute attention maps with all key tokens. We find this approximation achieves good performance while taking less than 10ms in practice. Additionally, the proposed method relies on the head classification results for head-specific KV cache management. To achieve this, we select the third AR step and the last denoising timestep across all attention heads to classify head types. Thanks to the stability of the dummy head as demonstrated in Sec. 3.2, we then keep this head classification result fixed during subsequent AR steps and denoising time steps. In other words, our method requires only a single head classification computation call which can complete within 100ms. In contrast, previous KV cache methods require recalculation at each AR steps to update the KV cache tokens, incurring additional computational overhead.

- Table 8. Results on 60s interactive long video generation. Quality scores are reported based on the whole video. CLIP semantic scores are reported on every 10-second video clip.

CLIP Semantic Score↑ 0–10s 10–20s 20–30s 30–40s 40–50s 50–60s Avg.

Methods FPS↑ Speedup↑ Quality Score↑

Self-Forcing 17.56 1.0× 82.77 0.3431 0.3353 0.3303 0.3245 0.3175 0.3219 0.3288

- +Dummy Forcing 24.30 1.4× 83.38 0.3389 0.3266 0.3157 0.3226 0.3108 0.3193 0.3223 LongLive 17.57 1.0× 84.63 0.3460 0.3370 0.3355 0.3298 0.3216 0.3223 0.3320

- +Dummy Forcing 24.30 1.4× 84.38 0.3367 0.3264 0.3300 0.3197 0.3119 0.3260 0.3251

4.7

[Figure 148]

Vanilla Self-Attn

+ Ours

- 1

- 2
- 3
- 4

1.7x

3.7

AverageRunTime(ms)

2.8

2.6

2.4

2.1

0

9HW 15HW 21HW

Context Length

- Figure 12. Runtime profiling on a single self-attention layer under different context lengths. HW denotes the total number of visual tokens of one latent frame. The attention runtime is averaged across all layers of the model.

### D. More Experimental Results

Comparison on 60s Interaction Video Generation. Following Yang et al. (2025a), we generate a 60-second-long video by sequentially feeding the model with 6 text prompts, each responsible for a 10-second video segment. Since VBench does not include corresponding tasks, we adopt the 12 prompt suites released by LongLive. For evaluation, we assess the generated videos on VBench-Long using dimensions that support customized-prompt videos, including subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality. Moreover, we compute the CLIP (Radford et al., 2021) similarity between each video segment and its corresponding text prompt to evaluate semantic adherence. We employ the KV re-caching (Yang et al., 2025a) technique to mitigate temporal shift in long video generation. The experimental results are summarized in Tab. 8. Our method achieves comparable video quality and semantic consistency with higher generation speed.

Speedup on Single Attention Layer. The main paper mainly demonstrates the speedup effect from end-to-end profiling. Notably, the proposed method accelerates only self-attention computations while sharing the same computational costs on other modules as the baseline model. To provide module-level acceleration results, Fig. 12 shows the runtime of a single self-attention layer under different context lengths. Compared to previous baselines, which exhibit quadratic computational complexity with respect to visual token length, the acceleration effect of our Dummy Forcing becomes more pronounced as the sequence length increases. For instance, when self-attention processes interactions across 15 frames, the proposed Dummy Forcing achieves a 1.7× speedup.

Results on 14B Models. In this section, we explore the effectiveness of the proposed method on large-scale autoregressive video diffusion. Specifically, we evaluate the proposed method on the RealTime-14B (Millon, 2025) model. RealTime-14B employs the Self Forcing training pipeline for post-training on the WAN2.1-14B (Wan et al., 2025) model to obtain autoregressive video diffusion. The model contains a total of 1600 heads, and we set 800 heads as dummy heads. We evaluate the proposed method on 5-second short video generation on VBench (Huang et al., 2024a) and 30-second long video generation on VBench-long (Huang et al., 2024b). Results are presented in Tab. 10. It can be observed that the proposed method scales well to larger models. For instance, on VBench-long, pruning all KV cache of 50% heads achieves even better total scores while delivering acceleration. The above experiments demonstrate the generalizability of the proposed method on large-scale models.

Comparison on Cache Compression Ratio. Since our approach works by compressing the length of the KV cache, it is

- Table 9. Quantitative comparison on cache compression ratios. We use Self Forcing as the baseline model and evaluate the KV cache reduction capabilities of different methods.

methods cache reduction ratio FPS speedup Semantic Score Quality Score Total Score

Self Forcing 100.0% 17.56 1.0× 84.73 81.03 84.00 + Infinipot-V 16.7% 19.54 1.1× 82.72 79.05 81.99 + R-KV 16.7% 18.41 1.1× 83.46 79.60 82.69 + Dummy Forcing 27.8% 24.30 1.4× 84.63 80.98 83.90

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames

[Figure 158]

NeighborHeadDummyHeadSinkHead

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Layer0_Head10 Layer1_Head6 Layer4_Head8

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Layer5_Head10 Layer1_Head9 Layer3_Head9

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames Sink Frames Neighbor Frames Current Frames

[Figure 177]

[Figure 178]

Layer0_Head7 Layer0_Head4 Layer4_Head4

Figure 13. Attention maps visualization of sink, neighbor, and dummy heads. Zoom in for better effects.

interesting to explore how many caches are compressed. In Tab. 9, we present the KV cache compression ratios of different methods. For the previous KV cache compression methods Infinipot-V (Kim et al., 2025) and R-KV (Cai et al., 2025), we set the cache budget to 1.5 frames, including 1 sink frame and 0.5 neighbor frames, which reduces the cache length to 16.7% of the original baseline. For the proposed method, we consider the dummy & non-dummy binary classification as a compression lower bound. This includes 1 packing frames for 50% heads and 4 cached frames (1 sink + 3 neighbor frame) for the remaining 50% heads, reducing the cache length to 27.8% of the baseline. Notably, previous KV cache compression methods require recalculating token importance scores to select tokens at each AR step. This additional computational overhead undermines the benefits of shorter context length, ultimately resulting in suboptimal end-to-end speed. Furthermore, previous methods are not specifically designed for autoregressive video diffusion models, leading to noticeable performance degradation when affecting context-critical heads. In contrast, the proposed Dummy Forcing requires only a single head classification per shot, followed by cache reduction based on predefined rules, significantly reducing additional overhead of the pruning algorithm. This enables our method to achieve faster runtime while preserving more tokens, striking a good balance between generation quality and speed.

Attention Map Visualization. In this section, we present the visualizations of attention maps. We employ the Self Forcing model as the base model, which utilizes the chunk-by-chunk generation scheme where each chunk comprises 3 latent frames. We select the third AR step because the equal lengths of the sink, neighbor, and dummy frames facilitate visualization. Fig. 13 shows representative attention maps for the three head types: sink, neighbor, and dummy heads. As observed, different head types exhibit corresponding attention distribution patterns. For instance, the sink head shows significantly higher attention score to the sink frame compared to the other two head types, while the dummy head allocates most attention scores to the current frame. These visualizations further validate the proposed head-specific KV cache management strategy.

- E. Details of Long-context Generation In this section, we provide details on the experimental setup for long-context video generation.

- Table 10. Quantitative comparison with RealTime14B model in 5-second short video generation on Vbench and 30-second long video generation on VBench-long.

setups duration FPS Quality Semantic Total Score

RealTime14B (Millon, 2025) 5s 2.8 85.24 81.69 84.53 + Dummy Forcing (Ours) 5s 3.2 84.61 81.73 84.04 RealTime14B (Millon, 2025) 30s 2.8 84.23 72.91 81.97 + Dummy Forcing (Ours) 30s 3.2 84.56 76.42 82.94

Setups. In multi-shot video generation scenarios, it is easy for humans to evaluate the consistency of characters across multiple shots. However, designing automated quantitative evaluation protocols remains challenging. To this end, we design the following “A-B-A” like long-context video generation task. Specifically, we force the video generation model to generate 15s interactive videos driven by 3 text prompts P = {P1,P2,P3}, with each prompt controlling a 5s video segment. We use all prompts from VBench as P1. To induce a shot transition in P2, we prompt the Qwen2.5-72B-Instruct (Team et al., 2024) model to generate a new shot that is different from P1 to yield P2. The related prompt is given in Fig. 14. Next, we set P3 = P1 to reconstruct the exact same scene. By evaluating the similarity between the first and the third video segments, we can automatically obtain quantitative scores that reflect the context utilization capability of different models.

Metrics Calculation. Following the VBench (Huang et al., 2024a) setup, we include the following video generation evaluation dimensions: subject consistency, background consistency, image quality, aesthetic quality, and overall consistency. The subject consistency evaluates consistency between frames by calculating similarity in the DINO (Caron et al., 2021) feature space. Since the second video segment is prompted to generate a totally different transition scene, we extract the first and third segments to form a 10-second video clip for DINO similarity computation. The background consistency calculates similarity between two frames in the CLIP (Radford et al., 2021) image encoder space. Similarly, we extract the first and third segments and compute similarity on the concatenated 10-second video clip. The image quality and aesthetic quality scores are obtained using the MUSIQ (Beaumont et al., 2022) image quality predictor and LAION aesthetic predictor, respectively, and evaluated across all three video segments. Overall consistency is computed using video-text consistency calculated by ViCLIP (Wang et al., 2023) to reflect both semantic and stylistic consistency, where the text prompts contain distinct semantics and styles. Finally, we average the scores across all dimensions to obtain the total score.

### F. Limitation and Future Works

Our Dummy Forcing effectively addresses the limited contextual utilization in current autoregressive video diffusion models by pruning redundant context from the dummy head. Nonetheless, our work can be further improved in the future in the following aspects. First, the proposed method is currently a training-free pipeline. We note that post-training could potentially achieve further performance gains or improved compression rates. Specifically, we can first identify the dummy heads in pre-trained models using our proposed Dummy Forcing. Subsequently, by fine-tuning on a small dataset, we can completely remove the KV cache for dummy heads during training, forcing the model to concentrate its context aggregation capabilities on the few non-dummy heads. Due to computational constraints and the heavy workload involved, we reserve dummy head fine-tuning for future work. Second, although we thoroughly analyze dummy heads in this paper, including their distribution across layers and commonality in existing models, further investigation into why they emerge in autoregressive video diffusion models remains an intriguing topic. This may involve analyzing training dynamics. Given that autoregressive video diffusion models represent a recent emerging generative paradigm, we believe future work will provide answers regarding the causes of dummy heads. Third, since this work extensively explores the applicability of the proposed method to autoregressive video generation models, such as Diffusion Forcing, Self Forcing, and Rolling Forcing, further investigation into inference acceleration for other model categories, such as world models, represents a meaningful direction, and we leave it as future work.

#### G. More Generation Results In this section, we provide more visual results, which are organized as follows:

- • In Fig. 14, we give the user prompt for generating the shot transition in our long-context video generation evaluation.

- • In Fig. 15 and Fig. 16, we give the results of applying Dummy Forcing to other autoregressive diffusion pipleines including Diffusion Forcing and Rolling Forcing, respectively.
- • In Fig. 17 and Fig. 18, we give more results on long-context video generation using the proposed context probing setup.
- • Fig. 19 and Fig. 20 gives more quantitative comparison results on single prompt 5s short video generation.
- • In Fig. 21 and Fig. 22, we give more visualization on single prompt 30s long video generation task.

You are a professional video-prompt generation specialist. Your task is to generate three text prompts for a multi-stage video generation system. These prompts will be fed into a video generation model in sequence to evaluate its scene transition behavior and its ability to preserve long-term visual memory.

Your input: you will receive one ORIGINAL_PROMPT describing the first shot. Your output: return one python list containing three prompts in the format of list(str), where

- Prompt 1: exactly the ORIGINAL_PROMPT with no changes.
- Prompt 2: a new prompt that introduces a required scene transition.
- Prompt 3: a prompt that is exactly identical to the ORIGINAL_PROMPT. Rules for Prompt 2:

- - The setting or environment must be completely different from the ORIGINAL_PROMPT.
- - Keep the mood, realism level, descriptive tone, and overall style consistent with the ORIGINAL_PROMPT.
- - Do not use phrases that reference previous content, such as "still," "as before," "continues," or similar.
- - Use clear mid-level English and avoid rare or overly literary vocabulary. Other rules:
- - Prompt 2 must have 80-100 words.
- - All prompts must be in English language.
- - Prompt 1 and Prompt 3 must be exact copies of the ORIGINAL_PROMPT.
- - Output format: ["PROMPT_1", "PROMPT_2", "PROMPT_3"]

Do not include any explanations, headings, markdown, or extra text. Directly output the list of prompts.

Figure 14. System prompt used for shot transition in long-context video generation evaluation.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

CausVidCausVid+Ours+Ours

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Prompt: A handheld camera captures a dog running through a park with a joyful exploration, the camera following the dog closely and bouncing ...

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Prompt: A dramatic post-apocalyptic scene in the style of a horror film, featuring a skeleton wearing a colorful flower hat and oversized sunglasses dancing...

- Figure 15. Quantitative comparison between CausVid (Yin et al., 2025) and our Dummy Forcing. We prune 50% heads’ KV cache to serve as dummy heads in the proposed method.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

RollingForcingRollingForcing+Ours+Ours

Prompt: A cinematic scene from a classic western movie, featuring a rugged man riding a powerful horse through the vast Gobi Desert...

Prompt: A dynamic tracking shot following a skateboarder performing a series of fluid tricks down a bustling city street...

- Figure 16. Quantitative comparison between Rolling Forcing (Liu et al., 2025) and our Dummy Forcing. We prune 50% heads’ KV cache to serve as dummy heads in the proposed method.

shot transition shot transition

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

LongLive

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Ours

- Shot1: A close-up, slow-motion video of a small brown bird meticulously building a nest using twigs and leaves...
- Shot2: A wide shot of a bustling city street at dusk. The street is lined with tall buildings and lit by ...

Shot3: A close-up, slow-motion video of a small brown bird meticulously building a nest using twigs and leaves...

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

LongLive

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Ours

- Shot1: A close-up shot of a middle-aged man carefully counting a large pile of cash on a wooden table. He has a focused ...
- Shot2: A panoramic view of a bustling city park on a sunny afternoon. People of all ages are enjoying the day ...

Shot3: A close-up shot of a middle-aged man carefully counting a large pile of cash on a wooden table. He has a focused ...

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

LongLive

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Ours

- Shot1: A still frame showing a vintage red phone booth with a glass door and a payphone inside
- Shot2: A serene beach at sunrise, with gentle waves lapping against the shore.

Shot3: A still frame showing a vintage red phone booth with a glass door and a payphone inside

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

LongLive

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Ours

- Shot1: A whimsical animated scene featuring a pair of scissors and a teddy bear.
- Shot2: A serene animated scene of a bustling city park. Children play on a colorful playground,
- Shot3: A whimsical animated scene featuring a pair of scissors and a teddy bear.

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

LongLive

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Ours

- Shot1: A still scene featuring an orange and a vintage analog wall clock. The orange is bright and juicy, ...
- Shot2: A tranquil outdoor scene in a forest clearing. Dappled sunlight filters through the dense canopy of trees ...
- Shot3: A still scene featuring an orange and a vintage analog wall clock. The orange is bright and juicy, ...

Figure 17. More quantitative comparison results in long-context video generation using context probing (Part1).

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

LongLive

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Ours

- Shot1: A detailed, close-up view of a traditional Chinese vase, intricately decorated with vibrant blue and white ...
- Shot2: A serene, wide shot of a peaceful forest clearing at dawn. Mist rises from the ground, creating a mystical ...

Shot1: A close-up view of a large, ornate red clock with intricate detailing. The clock face is circular, ... Shot2: A wide shot of a bustling city street at dusk. The street is lined with tall buildings and illuminated ...

Shot1: A beautifully crafted blue ceramic vase placed on a wooden table. The vase has a sleek, modern design ... Shot2: A bustling city street at dusk, with the warm glow of streetlights illuminating the cobblestone pavement...

- Shot1: Astronaut floating in the vastness of space, depicted in a Van Gogh-inspired style. The astronaut wears a sleek,
- Shot2: A bustling cityscape at night, depicted in a Van Gogh-inspired style. Tall skyscrapers with twinkling lights rise...

- Shot1: A breathtaking fantasy landscape featuring rolling hills covered in vibrant, luminescent flowers and towering, ...
- Shot2: A bustling medieval marketplace filled with vibrant stalls selling colorful fabrics, spices, and handmade crafts...

shot transition shot transition

- Shot3: A detailed, close-up view of a traditional Chinese vase, intricately decorated with vibrant blue and white ...

Shot3: A close-up view of a large, ornate red clock with intricate detailing. The clock face is circular, ...

Shot3: A beautifully crafted blue ceramic vase placed on a wooden table. The vase has a sleek, modern design ...

- Shot3: Astronaut floating in the vastness of space, depicted in a Van Gogh-inspired style. The astronaut wears a sleek,

- Shot3: A breathtaking fantasy landscape featuring rolling hills covered in vibrant, luminescent flowers and towering, ...

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

LongLive

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Ours

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

LongLive

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

Ours

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

LongLive

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Ours

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

LongLive

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Ours

Figure 18. More quantitative comparison results in long-context video generation using context probing (Part2).

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Self Forcing

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

Ours

A cheerful and playful Corgi running and frolicking in a sunlit park during the golden hour of sunset. The Corgi has a fluffy tail ...

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

Self Forcing

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Ours

A high-speed action scene featuring a sleek black motorcycle accelerating down a winding asphalt road...

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Self Forcing

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Ours

A vibrant and energetic cheerleader performing a routine. The cheerleader is a young woman with curly blonde hair tied up in a ponytail...

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

Self Forcing

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Ours

A playful polar bear is strumming a guitar in a snowy wilderness. The polar bear is standing upright, holding the guitar...

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

Self Forcing

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Ours

A dynamic snowboarding sequence set against a stunning winter landscape...

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Self Forcing

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Ours

A large semi-truck turning a corner on a busy city street during rush hour. The truck has a typical design with multiple mirrors,...

Figure 19. More quantitative comparison results in 5s short video generation task (Part1).

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Self Forcing

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Ours

A person with warm, gentle facial expressions is petting a friendly, medium-sized dog...

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

Self Forcing

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

Ours

A dynamic and intense scene of two medieval knights engaging in sword fighting in a spacious, dimly lit arena...

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

Self Forcing

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

Ours

The swimmer is fully submerged, with only their head breaking the water's surface, taking deep breaths...

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

Self Forcing

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

Ours

In a photorealistic style, an astronaut in a sleek, modern spacesuit is riding a horse floating ...

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

Self Forcing

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

Ours

A high-definition, cinematic scene featuring Tony Stark as Iron Man soaring through the sky...

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Self Forcing

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

Ours

A dramatic and intense scene featuring an erupting volcano. The volcano is spewing lava and ash into the air...

Figure 20. More quantitative comparison results in 5s short video generation task (Part2).

0-6s 6-12s 12-18s 18-24s 24-30s

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

LongLive

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Ours

A dynamic action shot in the style of a professional cycling magazine, capturing a cyclist in mid-stride as ...

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

LongLive

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

Ours

A close-up shot of a young boy energetically unscrewing a bottle cap. The boy has short, messy hair and ...

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

LongLive

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

Ours

A dynamic racing scene captured in the style of a high-speed sports photography,...

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

LongLive

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

Ours

A dynamic and chaotic scene in a dense forest during a heavy rainstorm, capturing a real girl frantically running ...

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

LongLive

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

Ours

A vibrant and dynamic illustration in the style of a futuristic sci-fi comic, depicting two playful dogs, ...

Figure 21. More quantitative comparison results in 30s long video generation task (Part1).

0-6s 6-12s 12-18s 18-24s 24-30s

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

LongLive

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

Ours

A dynamic action shot in the style of a high-energy sports magazine spread, featuring a golden retriever sprinting ...

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

LongLive

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

Ours

A vibrant and dynamic illustration in a cartoon style, depicting a porcupine wearing a colorful tutu ...

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

LongLive

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Ours

A dynamic street scene in a rainy city, capturing a couple running through a sudden downpour, ...

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

LongLive

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

Ours

A dynamic tracking shot in the style of a classic Hollywood film, capturing a steam locomotive chugging ...

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

LongLive

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

Ours

A subtle and elegant photograph in a Japanese style, capturing a woman with gentle ...

Figure 22. More quantitative comparison results in 30s long video generation task (Part2).

