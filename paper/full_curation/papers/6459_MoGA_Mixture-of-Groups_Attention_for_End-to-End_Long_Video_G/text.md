# arXiv:2510.18692v1[cs.CV]21Oct2025

[Figure 1]

## MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation

#### Weinan Jia1,∗, Yuning Lu2,†, Mengqi Huang1, Hualiang Wang3,∗, Binyuan Huang4,∗, Nan Chen1, Mu Liu2, Jidong Jiang2, Zhendong Mao1,†

1University of Science and Technology of China, 2FanqieAI, ByteDance China, 3Hong Kong University of Science and Technology, 4Wuhan University

∗Work done at FanqieAI, ByteDance China, †Corresponding authors

### Abstract

Long video generation with Diffusion Transformers (DiTs) is bottlenecked by the quadratic scaling of full attention with sequence length. Since attention is highly redundant, outputs are dominated by a small subset of query–key pairs. Existing sparse methods rely on blockwise coarse estimation, whose accuracy–efficiency trade-offs are constrained by block size. This paper introduces Mixtureof-Groups Attention (MoGA), an efficient sparse attention that uses a lightweight, learnable token router to precisely match tokens without blockwise estimation. Through semantic-aware routing, MoGA enables effective long-range interactions. As a kernel-free method, MoGA integrates seamlessly with modern attention stacks, including FlashAttention and sequence parallelism. Building on MoGA, we develop an efficient long video generation model that end-to-end produces minute-level, multi-shot, 480p videos at 24 fps, with a context length of approximately 580k. Comprehensive experiments on various video generation tasks validate the effectiveness of our approach.

Date: October 22, 2025 Project Page: MoGA

### 1 Introduction

A growing body of research indicates that scaling laws are a primary driver of progress toward artificial general intelligence [4, 24, 33]. As model parameters and data scale to billions, Transformer-based foundation models [35] often exhibit emergent capabilities [24, 30, 39]. In video generation, given the inherently temporal nature, progress requires not only scaling parameters and data but, more critically, scaling the effective context length. This need is especially salient for long-form video generation (e.g., movies), where persistent memory is essential for maintaining consistency of environments and characters [46].

The main challenge of vanilla attention [35] for long sequences is its computational cost, which grows quadratically with the context length. To mitigate the challenge, prior work [17, 34, 37, 42, 56] adopts a multi-stage pipeline that first generates key frames and then synthesizes intermediate frames. However, this design yields disjoint objectives that are not directly optimized for the end task, leading to error accumulation across stages. It also introduces hand-crafted inductive biases, hindering scalability.

For end-to-end long video generation, one line of work compresses historical content to accommodate longer contexts (e.g., via recurrent layers [7] or FramePack [50]), which inevitably results in information loss. A

Input Video Tokens

Input Video Tokens

Input Video Tokens

| |
|---|

Input Video

[Figure 2]

[Figure 3]

[Figure 4]

Pooling

Token Router

[Figure 5]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 6]

Q K V

|[Figure 7]<br><br>✅| | | |
|---|---|---|---|
| | |[Figure 8]<br><br>✅| |
| |[Figure 9]<br><br>✅| | |
| |[Figure 10]<br><br>❌|[Figure 11]<br><br>❌| |

| | |
|---|---|
| | |

[Figure 12]

Group Self-Attention

SimilarityScore

| | |
|---|---|
| | |

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

Token Blocks

… … … … Group 1 Group 2 Group M

[Figure 24]

Softmax Weight

[Figure 25]

[Figure 26]

[Figure 27]

Attention Score

| | |
|---|---|
| | |

⋅

|[Figure 28]| |
|---|---|
| | |

[Figure 29]

Token Rearranger

Blocks Similarity Map

[Figure 30]

[Figure 31]

[Figure 32]

Similarity Confusion

Dense Compute

Efficient Sparse

（a）Full Attention

（c）Mixture-of-Groups Attention

（b）Block Sparse Attention

- Figure 1 Illustration of our motivation. (a) Full attention suffers from dense computing when dealing with long sequences. (b) Block sparse attention [27] may fail when block-level similarity is confused, resulting in unreliable attention. (c) Mixture-of-Groups attention uses a lightweight token router (i.e., a single linear layer) that assigns tokens to specialized groups, enabling groupwise attention and efficient long-context modeling.

complementary direction exploits sparse attention [49] by restricting computation to a selected subset of salient query–key pairs. Existing selection strategies generally fall into two categories: (i) static selection, i.e., prior-driven heuristics that emphasize local spatiotemporal neighborhoods, which is efficient but limited in capturing dynamic long-range dependencies [12, 26, 32, 41]; and (ii) coarse-to-fine dynamic selection, which first estimates block-level importance scores, routes query tokens to the top-k blocks, and then applies fine-grained attention within the selected blocks [5, 27, 40, 43, 47]. As shown in Fig.1(b), the latter introduces an efficiency-performance trade-off: using larger blocks with a small top-k reduces the computational cost of the coarse stage but reduces selection performance.

In this work, we reveal that such coarse-grained estimation is unnecessary and each token should be precisely allocated. To achieve this, we propose Mixture-of-Groups Attention (MoGA), a simple and efficient dynamic token routing solution for end-to-end long video generation. A lightweight router (i.e., a single linear layer) is employed to assign tokens to specific groups, as illustrated in Fig. 1(c), inspired by the Mixture-of-Experts (MoE) [21]. Full attention is then performed within each group, where the groupwise attention integrates seamlessly with modern attention kernels, e.g., FlashAttention [8]. Intuitively, the linear router’s weights can be viewed as implicit cluster centers, enabling direct assignment of tokens to learnable anchors, without global similarity estimation. Furthermore, to balance long-range coherence and local fidelity, we couple MoGA with the spatiotemporal window attention [12], which can be considered as groupwise attention with static, predefined groups. In addition, extended context alone is insufficient because a single global prompt cannot reliably control scene transitions or orchestrate events at precise time points in long videos. We therefore introduce shot-level textual conditioning via cross-modal attention, where each shot is guided by a concise description [14, 38]. To support this, we build a data pipeline that produces minute-level video samples with dense, multi-shot captions and reliable shot segmentation.

Our contributions: We propose MoGA, an effective sparse attention mechanism that replaces block-level scoring with precise group assignment via a lightweight token router, enabling effective modeling of long contexts. Building on MoGA, we introduce a video generation model capable of producing minute-level, multi-shot, 480p videos at 24 fps with a context length of about 580k tokens. Fig. 7 illustrates a one-minute video generated by our model. Extensive evaluations show consistent improvements over state-of-the-art (SoTA) sparse attention baselines and a multi-shot video generation model.

### 2 Related Work

#### 2.1 Long Video Generation

Previous work on long video generation beyond typical duration limits has converged on three main paradigms. Multistage methods decompose long video generation into multiple steps [17, 34, 37, 42, 44, 56]. For example, Captain Cinema [42] adopts hierarchical planning with top-down keyframe generation and bottom-

up synthesis for narrative coherence. Multistage approaches introduce hand-crafted inductive biases and pose challenges for end-to-end optimization. Autoregressive approaches generate videos through sequential segment synthesis [1, 6, 14, 16, 19, 45]. Diffusion Forcing [6] adapts denoising schedules for variable sequence lengths. CasusVid [45] distills bidirectional models into an efficient autoregressive model. StreamingT2V [16] combines short- and long-term memory for streaming video extension. FAR [14] introduces hierarchical causal representations for multiscale dependencies. MAGI-1 [1] demonstrates the scaling capability of this paradigm. Context compression methods address computational constraints by compressing historical content [7, 23, 50]. TTT [7] compresses long context via a bidirectional recurrent layer. FramePack [50] employs importance-based frame compression to maintain a fixed computational budget. However, these methods either produce videos of limited duration [1, 6, 19] or fail to generate multi-shot videos in real-world scenes [7, 14, 16, 23, 45, 50]. A closely related line of work is LCT [15], which models interleaved multi-shot prompts and videos within a local context window using full attention. While pioneering end-to-end multi-shot long video generation, LCT remains constrained by the quadratic cost of full attention.

- 2.2 Sparse Attention for Video Generation

Attention–based foundation models unify many domains and consistently exhibit a common sparsity structure [9, 27, 47]. In video generation, given the inherent sparsity, a natural approach to efficient generation is to select important query-key pairs. Prior work broadly falls into two categories: static priors [26, 41, 51] and coarse-to-fine dynamic routing [40, 43, 52]. Among static approaches, STA [51] employs 3D sliding windows with a hardware-aware implementation. SVG [41] uses online pattern selection to classify attention heads as spatial or temporal sparse attention. Radial Attention [26] introduces a static attention mask to perform spatiotemporal attention with O(nlog n) complexity. However, these methods have difficulty modeling evolving long-range dependencies, which are crucial for maintaining cross-shot consistency. Another line of work adopts dynamic token routing for sparse attention. VSA [52] first obtains compressed representations of contiguous spatiotemporal blocks, and then selects the top-k blocks for fine-grained attention. Similarly, VMoBA [40] introduces an improved MoBA [27] tailored to video generation. In such methods, the block size presents a trade-off between expressiveness and efficiency. Smaller blocks yield more accurate coarse-grained attention estimates but reduce efficiency. In addition, SVG2 [43] is a training-free dynamic sparse attention method that performs online k-means clustering over tokens during inference and selects the top-k clusters based on their centroids. It shares a similar motivation with MoGA, i.e., tokens can be grouped into semantically coherent clusters. However, online clustering in SVG2 introduces additional k-means computations during the forward pass and is not straightforward to differentiate through. In contrast, MoGA employs trainable cluster centroids to enable simple and efficient routing with minimal computational overhead, making it suitable for end-to-end training.

- 3 Method

- 3.1 Preliminary

Vanilla self-attention [35] plays a crucial role in video generation with Diffusion Transformers (DiTs) [29]. Consider an input sequence X ∈ RN×d, where N = h × w × t is the total number of tokens across the latent spatial dimensions (h × w) and the latent temporal dimension (t), and d denotes the model’s hidden dimension. For simplicity, we consider a single query case, where x is a token from the input sequence and q is its corresponding query. Vanilla self-attention (SA) is computed as:

##### qK⊤

SA(q,K,V ) = softmax(

√

) · V , (1)

d

where K and V denote the keys and values. While self-attention excels at capturing long-range dependencies via global information aggregation, it incurs quadratic computational complexity of O(N2). The computational burden becomes particularly prohibitive in long-video generation. For example, generating a 1-minute video at 480p with approximately 1,600 tokens per frame across 961 frames (16 fps) yields a total token count of about 384k1. Performing full attention on such a long sequence is intractable.

1Following Wan [36], the VAE downsampling factors for (t, h, w) are (4, 8, 8) and patchify sizes are (1, 2, 2).

(a) Visual Attention

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>shot shot<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>…<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>shot|
|---|

Model Architecture

Full + TemporalSpatial

Spatial-Temporal Group Attention

VAE Decoder

TokensVideo ⨁ OutputTokens

Mixture-of-Groups Attention

Projection & Unpatchify

…

…

DiTBlocksN×

Cross-Modal Attention

(b) Mixture-of-Groups Attention

Visual Attention

[Figure 43]

- Group 1

Group Attention

- Group 2

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| | | |
|---|---|---|
| |Router| |
| |Token| |
| | | |

Text Encoder

shot

… … … …

TokenRearranger

TokenRouter

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Patchify

shot

VAE Encoder

Caption n Caption 1 Caption 0

…

…

…

Group M

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

…

A man is sitting at a table ...

shot shot shot

shot

- Figure 2 Left: Our model adopts a DiT architecture with interleaved Visual Attention and Cross-Modal Attention blocks. Visual Attention exclusively processes visual content, whereas Cross-Modal Attention enables shot-level text conditioning, instantiated via either cross-attention [36] or multi-modal attention [10, 25]. Top-right (a): Visual Attention combining MoGA with Spatial-Temporal Group Attention for global-local consistency. Bottom-right (b): MoGA, where a router groups tokens and performs intra-group attention, enabling long-range global interactions.

In this section, we introduce MoGA for efficient long video generation. The overall architecture is shown in Fig. 2. We first present the preliminaries, then detail MoGA, and finally describe the pipeline for constructing multi-shot long-video training data.

Beyond computational cost, full attention is not ideally aligned with the structure of videos. In videos, softmax attention is inherently sparse [41] because nearby tokens exhibit strong local spatiotemporal correlation, while only a few globally shared, dynamic semantics persist across frames. Most query–key pairs contribute little, whereas a small subset dominates [13]. For long videos, attention should leverage this sparsity by prioritizing important query–key interactions to reduce redundancy.

#### 3.2 Mixture-of-Groups Attention (MoGA)

MoGA addresses the above challenge via efficient token routing, where a lightweight, trainable router assigns correlated tokens to groups and performs self-attention within each group. Specifically, the router is a linear projection followed by softmax gating, similar to MoE [11]. Given a token x ∈ Rd and a predetermined number of groups M, the router computes routing scores r ∈ RM as:

r = Router(x). (2) The group assignment probabilities are computed as:

p(i | x) = softmax(r)i, (3) and the token is assigned to the group with the highest probability:

p(i | x). (4)

g(x) = arg max

i∈[M]

Following group assignment, we apply self-attention independently within each group. The MoGA output is: MoGA(x) = p(g(x) | x) · SA(q,Kg(x),Vg(x)), (5)

A group visualization

A group visualization

A group visualization

Shot 2 Shot 3

Shot 1

|[Figure 71]<br><br>[Figure 72]|
|---|

|[Figure 73]<br><br>[Figure 74]|
|---|

|[Figure 75]<br><br>[Figure 76]|
|---|

|[Figure 77]<br><br>[Figure 78]|
|---|

[Figure 79]

[Figure 80]

|[Figure 81]<br><br>[Figure 82]|
|---|

|[Figure 83]<br><br>[Figure 84]|
|---|

|[Figure 85]<br><br>[Figure 86]|
|---|

|[Figure 87]<br><br>[Figure 88]|
|---|

[Figure 89]

[Figure 90]

Figure 3 Visualization of dynamic router grouping.

Algorithm 1 MoGA Pseudocode with FlashAttention Require: Q,K,V are the query, key and value of tokens X

- 1: g = router(X) ▷ MoGA routing results
- 2: Qˆ,Kˆ ,Vˆ ,cu_seqlen,max_seqlen,permute_index = permute(Q,K,V ,g)
- 3: Oˆ = flash_attn(Qˆ,Kˆ ,Vˆ ,cu_seqlen,max_seqlen)
- 4: O = repermute(Oˆ,permute_index) ▷ MoGA recovers the original token positions

where Kg(x) and Vg(x) are the keys and values of the group g(x), and q is the query feature of x. This grouped attention mechanism reduces computational complexity from O(N2) to a theoretical minimum of O(N2/M) under uniform group assignment.

As illustrated in Fig. 3, we extract the grouping assignments from an intermediate-layer router during the video generation process and visualize one representative group. After end-to-end training, the router assigns the man’s head, hands, and portions of his clothing to the same group, indicating its ability to capture semantically coherent structures that span shot boundaries.

MoGA builds on groupwise attention and remains compatible with high-performance kernels such as FlashAttention [8] (see Alg. 1). Beyond sparse attention, a second pillar of long-context modeling is sequence parallelism [22], with which MoGA is also compatible. Before the sequence gather and head scatter step in each attention layer, MoGA computes routing scores over tokens (with whole heads) and then aggregates the routing results across all tokens.

Group Balancing Loss. A potential issue with token assignment is that the router may collapse by routing most tokens to only a few groups, which would degenerate MoGA into full attention To encourage adaptive token allocation across groups, we introduce an auxiliary group balancing loss, inspired by the load balancing loss [11] used in MoE. The loss is defined as:

M

FiPi, (6)

Lgb = α · M

i=1

where α is a loss weight and Fi is the fraction of tokens allocated to group i,

1 N x

1(g(x) = i), (7)

Fi =

where 1 is the indicator function, and Pi is the mean routing probability allocated for group i,

1 N

p(g(x) | x). (8)

Pi =

g(x)=i

Minimizing Lgb encourages uniform token assignment across groups, as this objective attains its minimum under a uniform distribution [11].

Spatial-Temporal Group Attention. Although MoGA captures long-range coherence, it lacks local continuity. We complement it with local spatiotemporal group attention (STGA) [12, 53], which restricts self-attention

Raw Videos

Training Samples

Video-Level Shot-Level

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Video Score

Video Filter

Shot Split

Shot Score

Shot Filter

Shot Crop

Shot Caption

Shot Merge

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Figure 4 Multi-shot long video data pipeline.

to local windows in latent video space, as shown in Fig. 2(a). This captures short-range dependencies with bounded compute.

We first partition the latent video into fixed spatial windows and then group frames along the temporal axis. Frames from different shots are assigned to distinct temporal groups. We empirically find that completely removing inter-shot interactions causes flicker in the first frame after a shot cut. To mitigate this, when computing group attention, we augment the keys and values with two latent frames from adjacent shots (without augmenting the queries). This preserves continuity at shot boundaries with negligible additional compute. To enable intra-frame information exchange, we also perform per-frame attention by grouping tokens within each latent frame. Each token therefore receives outputs from multiple groups (one dynamic and two static), and we take their mean as the final output.

#### 3.3 Data Pipeline

We construct a pipeline that converts raw long videos into one-minute, multi-shot clips with dense annotations for long video generation. The pipeline has two stages: a video-level phase and a shot-level phase (Fig. 4).

Video-Level. We first analyze raw videos using visual quality assessment (VQA) models (e.g., aesthetics [31], clarity, exposure) and simple operators (e.g., black-border detection) to obtain metadata and quality scores. We then filter raw videos with source-specific, calibrated thresholds to remove low-quality content. Because long video samples require temporal coherence, we relax clip-level filtering [25, 54] while applying stricter filtering at the source (raw-video) level. Next, we segment each video into single-shot clips using AutoShot [55] and PySceneDetect [3]. AutoShot shows higher sensitivity to fades and gradual transitions. Combining predictions from both tools allows us to label whether a boundary is clean or affected by transition overlap. This stage yields a pool of single-shot clips.

Shot-Level. We process single-shot clips using VQA and optical character recognition (OCR) models and discard low-quality clips. Based on OCR results, we compute a maximum-area crop that excludes watermarks and subtitles while preserving the original aspect ratio. Clips with insufficient retained area are discarded. Next, we generate captions for cropped clips using a multimodal large language model [2]. Finally, we merge temporally adjacent single-shot clips into multi-shot training samples (up to 65 seconds) and trim a few frames from clips affected by transition overlap to ensure clean boundaries.

### 4 Experiments

Training Settings. We fine-tune MoGA on existing DiT-based short video generation models with the rectified flow objective [10]. For a fair comparison with baselines, we train MoGA on the open-source Wan2.1 models (1.3B and 14B) [36]. The resulting model stably generates 477 frames at 16 fps (30 seconds) and 480p resolution, with a context length of 187k. We use a constant learning rate of 1e-5. The loss weight α is set to 0.1. We set the number of groups to M = 5 and partition the spatial grid into 2 × 2 groups. We adopt a multistage training strategy: 3k steps on 10-second clips followed by 1k steps on 30-second clips.

Because MoGA is a general sparse attention, we also apply it to a video generation model built on MMDiT [10, 12, 25]. Unlike Wan, this model replaces cross-attention with MMDiT to perform cross-modal attention. It partitions space into 4 × 4 groups and sets the router’s group number to M = 20, enabling a much longer context length. The MMDiT-based model generates 1,441 frames at 24 fps (60 seconds) at 480p, with a context length of 578k.

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Aesthetic Quality ↑

Image

Method Base Model

Quality ↑ Sparsity

↑ Wan (Original) Wan2.1-14B 0.9611 0.9560 0.9936 0.5807 0.6680 0%

DiTFastAttn (Training-based) Wan2.1-14B 0.9456 0.9394 0.9924 0.5269 0.6466 50.00% SVG (Training-free) Wan2.1-14B 0.9002 0.8926 0.9870 0.5370 0.6357 50.00% VMoBA (Training-free) Wan2.1-14B 0.8605 0.8876 0.9789 0.5369 0.6111 31.00%

MoGA (Ours) Wan2.1-14B 0.9699 0.9542 0.9927 0.5810 0.6994 71.25%

Table 1 Quantitative comparison for 5-second single-shot video generation.

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Aesthetic Quality ↑

Image Quality ↑

Cross-Shot DINO ↑

Cross-Shot CLIP ↑

Method Base Model

IC-Lora+Wan Wan2.1-1.3B 0.9476 0.9538 0.9901 0.5237 0.6684 0.4669 0.7169 EchoShot Wan2.1-1.3B 0.9544 0.9518 0.9939 0.5718 0.6534 0.5961 0.8469

MoGA (Ours) Wan2.1-1.3B 0.9549 0.9597 0.9919 0.5890 0.6729 0.6623 0.8654

Table 2 Quantitative comparison for 10-second multi-shot video generation.

Subject Consistency ↑

Background Consistency ↑

Motion Smoothness ↑

Aesthetic Quality ↑

Image Quality ↑

Method Base Model

IC-Lora+Wan Wan2.1-14B 0.8946 0.9169 0.9872 0.5759 0.6835 MoGA (Ours) Wan2.1-14B 0.9572 0.9475 0.9893 0.5789 0.6993 MoGA (Ours) MMDiT 0.9305 0.9301 0.9895 0.5881 0.6996

Table 3 Quantitative comparison for 30-second multi-shot long video generation.

Baselines. To evaluate our method, we compare with multiple baselines. For multi-shot long video generation, we include the keyframe-based pipeline IC-LoRA+Wan [18, 36] and EchoShot [38], which natively supports multi-shot generation. For sparse video generation, we compare against sparse attention methods, including the training-based DiTFastAttn [48] and the training-free methods SVG [41] and VMoBA [40].

Evaluation Metrics. Following prior work, we evaluate all methods using the metrics introduced by VBench [20]. Specifically, subject consistency and background consistency measure how well the main subjects and backgrounds of sampled frames are preserved throughout the video. Motion smoothness measures motion fluidity, penalizing jitter and abrupt transitions. We also report aesthetic quality and image quality to quantify the visual appeal and technical fidelity of each frame. To compute cross-shot consistency, we first sample a fixed number of frames from different shots. We then compute feature similarities across shots using CLIP [30] and DINOv2 [28], referred to as Cross-Shot CLIP and Cross-Shot DINO. For single-shot 5-second video generation, we constructed a diverse test set comprising 300 prompts. For multi-shot 10-second video generation, we use the 100 multi-shot prompt sets from [38]. For long video generation, we evaluate on a test set of 11 scripts comprising 105 prompts. Each script contains 8–10 shots to produce a 30-second video.

#### 4.1 Quantitative Results

First, we compare MoGA with prior sparse attention methods for single-shot, short video generation, following their evaluation settings to ensure fairness. As shown in Tab. 1, despite higher sparsity, MoGA achieves consistent improvements over existing sparse baselines across metrics. It is worth noting that although our method is highly sparse, it can still match or surpass the original Wan (full attention) on multiple metrics.

Next, we compare MoGA with other multi-shot video generation methods. Tab. 2 reports quantitative comparisons among MoGA, IC-LoRA+Wan and EchoShot. Despite relying on sparse attention, our method outperforms the full attention baseline (EchoShot) on most metrics, indicating that preserving interactions among salient tokens not only reduces FLOPs but also suppresses noise from irrelevant content. This leads to stronger character identity consistency and improved temporal scene coherence.

Finally, we benchmark long video generation against the baseline. Because few open-source methods can produce 30-second, multi-shot videos, we compare MoGA (with two backbones) to IC-LoRA+Wan. As

|[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]|
|---|

| |[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]|
|---|---|
| | |

|[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]|
|---|

|[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]|
|---|

T2I + I2V (IC-Lora)

###### Inconsistent

|[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]|
|---|

|[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]|
|---|

|[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]|
|---|

(Wan2.1)

[Figure 165]

[Figure 166]

|[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]|
|---|

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

| |[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]|
|---|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

EchoShot

|[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]|[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>|
|---|---|

[Figure 205]

|[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>Inconsistent| |
|---|---|

|[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]|
|---|

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

| |
|---|
| |
|[Shot2] aa manmen withwith plaid shirt<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]|
| |
| |

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Shot1] a woman with basket [Shot3] the woman smile [Shot4] panoramic shot of them

MoGA (Ours)

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Shot6] woman full face smile [Shot8] woman looks to left

[Shot5] woman smiles at man [Shot7] man looks at woman

- Figure 5 Qualitative results of MoGA and other methods. We present eight representative shots, demonstrating long-content coherence, character consistency, and visual quality.

- Figure 6 Computational efficiency. The x-axis denotes the generated video duration (s). As the number of groups (M) increases, MoGA’s FLOPs decrease substantially.

shown in Tab. 3, MoGA substantially outperforms IC-LoRA+Wan under the same backbone, highlighting the benefits of end-to-end modeling over multistage pipelines. Notably, even under aggressive sparsity, MoGA with MMDiT maintains high visual fidelity, indicating a scalable path to longer context lengths.

#### 4.2 Qualitative Results

Visual Comparison. In this subsection, we present qualitative results on 30-second videos across representative baselines. Because EchoShot cannot natively produce 30-second outputs, we concatenate video clips generated by EchoShot to form the full sequence. As shown in Fig. 5, the IC-LoRA+Wan pipeline is constrained by its per-iteration image cap (typically three frames), which limits its ability to cover a larger number of shots. Consequently, it often exhibits subject drift and background inconsistency as the sequence progresses.

𝑺𝒉𝒐𝒕𝟏 𝑺𝒉𝒐𝒕𝟑

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

𝟗𝟎𝒕𝒉 𝟏𝟐𝟎𝒕𝒉 𝟏𝟓𝟎𝒕𝒉 𝟏𝟖𝟎𝒕𝒉 𝟐𝟏𝟎𝒕𝒉 𝟐𝟒𝟎𝒕𝒉 𝟐𝟕𝟎𝒕𝒉

𝟏𝒔𝒕 𝟑𝟎𝒕𝒉 𝟔𝟎𝒕𝒉

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

𝟑𝟎𝟎𝒕𝒉 𝟑𝟑𝟎𝒕𝒉 𝟑𝟔𝟎𝒕𝒉 𝟑𝟗𝟎𝒕𝒉 𝟒𝟐𝟎𝒕𝒉 𝟒𝟓𝟎𝒕𝒉 𝟒𝟖𝟎𝒕𝒉 𝟓𝟏𝟎𝒕𝒉 𝟓𝟒𝟎𝒕𝒉 𝟓𝟕𝟎𝒕𝒉

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

𝟔𝟎𝟎𝒕𝒉 𝟔𝟑𝟎𝒕𝒉 𝟔𝟔𝟎𝒕𝒉 𝟔𝟗𝟎𝒕𝒉 𝟕𝟐𝟎𝒕𝒉 𝟕𝟓𝟎𝒕𝒉 𝟕𝟖𝟎𝒕𝒉 𝟖𝟏𝟎𝒕𝒉 𝟖𝟒𝟎𝒕𝒉 𝟖𝟕𝟎𝒕𝒉

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

𝟗𝟎𝟎𝒕𝒉 𝟗𝟑𝟎𝒕𝒉 𝟗𝟔𝟎𝒕𝒉 𝟗𝟗𝟎𝒕𝒉 𝟏𝟎𝟐𝟎𝒕𝒉 𝟏𝟎𝟓𝟎𝒕𝒉 𝟏𝟎𝟖𝟎𝒕𝒉 𝟏𝟏𝟏𝟎𝒕𝒉 𝟏𝟏𝟒𝟎𝒕𝒉 𝟏𝟏𝟕𝟎𝒕𝒉

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

…

𝑺𝒉𝒐𝒕𝟐𝟐

𝟏𝟐𝟎𝟎𝒕𝒉 𝟏𝟐𝟑𝟎𝒕𝒉 𝟏𝟐𝟔𝟎𝒕𝒉 𝟏𝟐𝟗𝟎𝒕𝒉 𝟏𝟑𝟐𝟎𝒕𝒉 𝟏𝟑𝟓𝟎𝒕𝒉 𝟏𝟑𝟖𝟎𝒕𝒉 𝟏𝟒𝟏𝟎𝒕𝒉

Figure 7 One-minute video generated by MoGA.

EchoShot scales to more shots but still shows notable cross-shot inconsistencies on long temporal ranges. In contrast, MoGA maintains stable, coherent content over extended durations. For example, even without repeated or explicit specification across shots, the woman’s hat remains consistently preserved. Since STGA lacks explicit cross-shot information exchange, this consistency can be attributed to MoGA, which effectively selects and maintains shot-spanning identity and context.

One-Minute Video of 1,441 Frames. In Fig. 7, we present the generated results of MoGA on an ultra-long video of over one minute, using the MMDiT-based MoGA model (M = 20). MoGA maintains strong long-range contextual consistency. The 1st and 22nd shots remain highly coherent, and fine details such as the woman’s hairpin and earrings are preserved across shots. Moreover, even with multiple faces appearing across different shots, the model avoids identity confusion.

Emergence of Background Consistency. As shown in Fig. 8, we demonstrate MoGA’s ability to maintain background consistency. After training on long, multi-shot videos, MoGA exhibits emergent, implicit control over consistency in both the environment and the characters. Even without explicit specification of details (e.g., the cabinet shape and the position of intravenous drip bottle), different shots automatically maintain coherent, temporally consistent depictions.

Multi-Style Video Generation. Fig. 9 illustrates MoGA’s multi-style generation capability. MoGA not only performs strongly in realistic spaces but also excels in stylized domains such as animation. It can produce high-quality, long-form 2D videos while maintaining temporal coherence, identity consistency, and scene continuity across diverse styles.

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

𝑺𝒉𝒐𝒕𝟏

𝟏𝒔𝒕 𝟑𝟎𝒕𝒉 𝟔𝟎𝒕𝒉 𝟗𝟎𝒕𝒉 𝟏𝟐𝟎𝒕𝒉 𝟏𝟓𝟎𝒕𝒉 𝟏𝟖𝟎𝒕𝒉 𝟐𝟏𝟎𝒕𝒉 𝟐𝟒𝟎𝒕𝒉 𝟐𝟕𝟎𝒕𝒉

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

𝑺𝒉𝒐𝒕𝟒

𝟑𝟎𝟎𝒕𝒉 𝟑𝟑𝟎𝒕𝒉 𝟑𝟔𝟎𝒕𝒉 𝟑𝟗𝟎𝒕𝒉 𝟒𝟐𝟎𝒕𝒉 𝟒𝟓𝟎𝒕𝒉 𝟒𝟖𝟎𝒕𝒉 𝟓𝟏𝟎𝒕𝒉 𝟓𝟒𝟎𝒕𝒉 𝟓𝟕𝟎𝒕𝒉

###### Figure 8 Emergence of background consistency.

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

𝟏𝒔𝒕 𝟑𝟎𝒕𝒉 𝟔𝟎𝒕𝒉 𝟏𝟓𝟎𝒕𝒉 𝟏𝟖𝟎𝒕𝒉

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

𝟐𝟏𝟎𝒕𝒉 𝟐𝟒𝟎𝒕𝒉 𝟐𝟕𝟎𝒕𝒉 𝟑𝟎𝟎𝒕𝒉 𝟑𝟑𝟎𝒕𝒉

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

𝟑𝟔𝟎𝒕𝒉 𝟑𝟗𝟎𝒕𝒉 𝟒𝟐𝟎𝒕𝒉 𝟒𝟓𝟎𝒕𝒉 𝟒𝟖𝟎𝒕𝒉

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

𝟓𝟏𝟎𝒕𝒉 𝟓𝟒𝟎𝒕𝒉 𝟓𝟕𝟎𝒕𝒉 𝟔𝟎𝟎𝒕𝒉 𝟔𝟑𝟎𝒕𝒉

Figure 9 Animation style generation for MoGA.

#### 4.3 Ablation Study

Computational Efficiency. Fig. 6 plots the relationship between the number of groups (M) and FLOPs for the Wan2.1-1.3B model. Our experiments show that even with a relatively small group count (M=5) for 30-second videos, MoGA achieves substantial computational savings compared to full attention (2.26 PFLOPs vs. 6.94 PFLOPs). It also delivers a 1.7× speedup in both training and inference. Notably, unlike alternative sparse attention such as VMoBA, which incur additional memory overhead due to their block-based mechanisms, our approach maintains memory efficiency without additional memory consumption.

Routing Group Number M. We conduct an ablation study on the number of groups under a fixed computational budget (Tab. 4). Cross-shot DINO and CLIP scores exhibit a rise-then-fall trend as the number of groups increases. This suggests that a moderate level of grouped sparsity strikes a balance between global consistency and efficiency, yielding near-optimal consistency while maintaining computational efficiency.

Controllability of Subject Consistency. Fig. 10 visualizes a comparison between MoGA and full attention. Both models are trained on 10-second data with Wan2.1-14B. The left panel illustrates MoGA’s ability to maintain subject identity across multiple scenes, while the right panel demonstrates its robustness to appearance changes (e.g., clothing) when preserving identity consistency. Despite 71.25% sparsity, MoGA achieves narrative coherence and content editability on par with full attention, and in some cases delivers superior performance.

Group Numbers Cross-Shot DINO ↑ Cross-Shot CLIP ↑ Sparsity PFLOPs

- 1 0.8206 0.5919 0% 0.88
- 2 0.8589 0.6761 41.25% 0.59 4 0.8672 0.6853 66.25% 0.42 8 0.8606 0.6910 78.75% 0.36

16 0.8569 0.6896 81.25% 0.35

###### Table 4 Results of consistency for MoGA with Wan2.1-1.3B on 10-second videos.

|[Figure 381]<br><br>[Figure 382]<br><br>[Figure 383]<br><br>[Figure 384]<br><br>[Figure 385]<br><br>[Figure 386]|
|---|

|[Figure 387]<br><br>[Figure 388]<br><br>[Figure 389]<br><br>[Figure 390]<br><br>[Figure 391]<br><br>[Figure 392]|
|---|

|[Figure 393]<br><br>[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]<br><br>[Figure 398]|
|---|

|[Figure 399]<br><br>[Figure 400]<br><br>[Figure 401]<br><br>[Figure 402]<br><br>[Figure 403]<br><br>[Figure 404]|
|---|

|[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]<br><br>[Figure 410]|
|---|

|[Figure 411]<br><br>[Figure 412]<br><br>[Figure 413]<br><br>[Figure 414]<br><br>[Figure 415]<br><br>[Figure 416]|
|---|

Full Attn

|[Figure 417]<br><br>[Figure 418]<br><br>[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]|
|---|

|[Figure 423]<br><br>[Figure 424]<br><br>[Figure 425]<br><br>[Figure 426]<br><br>[Figure 427]<br><br>[Figure 428]|
|---|

|[Figure 429]<br><br>[Figure 430]<br><br>[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]|
|---|

|[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]<br><br>[Figure 440]|
|---|

|[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]|
|---|

|[Figure 447]<br><br>[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]<br><br>[Figure 452]|
|---|

MoGA

|[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]<br><br>[Figure 460]|
|---|

|[Figure 461]<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]|
|---|

|[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]|
|---|

| |
|---|
|[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]|
| |
| |

|[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]|
|---|
| |

|[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]<br><br>[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]|
|---|

Full Attn

|[Figure 495]<br><br>[Figure 496]<br><br>[Figure 497]<br><br>[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]|
|---|

|[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]|
|---|

|[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]|
|---|

|[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]|
|---|

|[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]|
|---|

|[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]|
|---|

MoGA

###### Figure 10 Visual comparison of MoGA vs. full attention for multi-shot generation with a single subject. The left column shows the subject wearing the same outfit across different shots, while the right column shows the subject changing outfits at shot transitions according to the text instructions.

Only MoGA Only STGA MoGA + STGA

|[Figure 531]<br><br>[Figure 532]| | |[Figure 533]<br><br>[Figure 534]|[Figure 535]<br><br>[Figure 536]|
|---|---|---|---|---|

| |[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]|| |
|---|
| |[Figure 541]<br><br>[Figure 542]|
|---|---|---|---|---|

| |
|---|
|[Figure 543]<br><br>[Figure 544]<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]<br><br>[Figure 548]|
| |

- Shot 1
- Shot 2

###### Inconsistent Consistent

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

| |[Figure 553]<br><br>[Figure 554]<br><br>[Figure 555]<br><br>[Figure 556]<br><br>| |
|---|---|---|
|[Figure 557]<br><br>[Figure 558]| | |
| | | |

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

Figure 11 Visual ablation of MoGA and STGA.

Effectiveness of MoGA and STGA. As shown in Fig. 11, MoGA and STGA play complementary roles in enabling context-consistent long video generation. Using MoGA alone lacks local information exchange and fails to produce meaningful visual content. Conversely, using only STGA limits long-range shot interactions, leading to poor cross-shot consistency and weakened narrative coherence. When combined, the model achieves strong cross-shot consistency. These results indicate that MoGA effectively routes and preserves shot-spanning identity and context at relatively low computational cost.

### 5 Conclusion

This paper introduces MoGA, a sparse attention mechanism that replaces coarse block-level scoring with precise, learned group assignments via a lightweight token router. By routing tokens into coherent groups, MoGA improves attention efficiency and fidelity for very long contexts. Building on MoGA, we propose the video generation model that produces minute-level, multi-shot videos at 480p resolution and 24 fps. Diverse experiments in video generation further demonstrate the effectiveness of our approach.

Acknowledgments. We thank the ByteDance Seedance team and Wenfeng Lin for their support.

### References

- [1] Sand. ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. Magi-1: Autoregressive video generation at scale. arXiv:2505.13211, 2025.

- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv:2502.13923, 2025.

- [3] Breakthrough and Contributors. Pyscenedetect: Video scene cut detection tool, 2014.
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.

- [5] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv:2508.21058, 2025.

- [6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024.

- [7] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In CVPR, 2025.

- [8] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv:2307.08691, 2023.

- [9] DeepSeek-AI. Deepseek-v3.2-exp: Boosting long-context efficiency with deepseek sparse attention, 2025.
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

- [11] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR, 2022.

- [12] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv:2506.09113, 2025.

- [13] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv:2310.01801, 2023.

- [14] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv:2503.19325, 2025.

- [15] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv:2503.10589, 2025.

- [16] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In CVPR, 2025.

- [17] Kaiyi Huang, Yukun Huang, Xintao Wang, Zinan Lin, Xuefei Ning, Pengfei Wan, Di Zhang, Yu Wang, and Xihui Liu. Filmaster: Bridging cinematic principles and generative ai for automated film generation. arXiv:2506.18899, 2025.

- [18] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv:2410.23775, 2024.

- [19] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In NeurIPS, 2025.

- [20] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

- [21] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural Computation, 1991.

- [22] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv:2309.14509, 2023.

- [23] Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv:2507.12952, 2025.

- [24] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020.

- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv:2412.03603, 2024.

- [26] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(n log n) sparse attention with energy decay for long video generation. arXiv:2506.19852, 2025.

- [27] Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. arXiv:2502.13189, 2025.

- [28] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023.

- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, 2023.

- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

- [31] Christoph Schuhmann, Romain Vencu, Romain Beaumont, Radu Kaczmarczyk, Charlie Mullis, Ashish Katta, Jenia Jitsev Coombes, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.

- [32] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv:2504.08685, 2025.

- [33] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023.

- [34] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di ZHANG, et al. Videotetris: Towards compositional text-to-video generation. In NeurIPS, 2024.

- [35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

- [36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314, 2025.

- [37] Bo Wang, Haoyang Huang, Zhiying Lu, Fengyuan Liu, Guoqing Ma, Jianlong Yuan, Yuan Zhang, Nan Duan, and Daxin Jiang. Storyanchors: Generating consistent multi-scene story frames for long-form narratives. arXiv:2505.08350, 2025.

- [38] Jiahao Wang, Hualian Sheng, Sijia Cai, Weizhan Zhang, Caixia Yan, Yachuang Feng, Bing Deng, and Jieping Ye. Echoshot: Multi-shot portrait video generation. arXiv:2506.15838, 2025.

- [39] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv:2206.07682, 2022.

- [40] Jianzong Wu, Liang Hou, Haotian Yang, Xin Tao, Ye Tian, Pengfei Wan, Di Zhang, and Yunhai Tong. Vmoba: Mixture-of-block attention for video diffusion models. arXiv:2506.23858, 2025.

- [41] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv:2502.01776, 2025.

- [42] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv:2507.18634, 2025.

- [43] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv:2505.18875, 2025.

- [44] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv:2303.12346, 2023.

- [45] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

- [46] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv:2506.03141, 2025.

- [47] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv:2502.11089, 2025.

- [48] Zhihang Yuan, Hanling Zhang, Lu Pu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattn: Attention compression for diffusion transformer models. In NeurIPS, 2024.

- [49] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. In NeurIPS, 2020.

- [50] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv:2504.12626, 2025.

- [51] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv:2502.04507, 2025.

- [52] Peiyuan Zhang, Haofeng Huang, Yongqi Chen, Will Lin, Zhengzhong Liu, Ion Stoica, Eric P Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. arXiv:2505.13389, 2025.

- [53] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Zehuan Yuan, and Bingyue Peng. Waver: Wave your way to lifelike video generation. arXiv:2508.15761, 2025.

- [54] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv:2412.20404, 2024.

- [55] Wentao Zhu, Yufang Huang, Xiufeng Xie, Wenxian Liu, Jincan Deng, Debing Zhang, Zhangyang Wang, and Ji Liu. Autoshot: A short video dataset and state-of-the-art shot boundary detection. In CVPR, 2023.

- [56] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In CVPR, 2024.

## Appendix

### A Details of the Computational Complexity

As shown in Tab. 5, it reports the computational cost under varying number of groups (M) and video duration. As the generation video duration increases, the computational complexity of STGA exhibits approximately linear growth and the computational complexity of MoGA is approximately 1/M of that of Full Attention.

Video Duration 5s 10s 15s 20s 30s Frames 77 157 237 317 477 Sequence Length 31200 62400 93600 124800 187200

Full Attention 0.28 0.88 1.85 3.19 6.94 MoGA (M = 5) 0.093 0.34 0.67 1.09 2.26 MoGA (M = 10) 0.065 0.25 0.48 0.78 1.56 MoGA (M = 20) 0.051 0.21 0.39 0.61 1.22

PFLOPs

Table 5 Compute (PFLOPs) versus group number M and video duration on Wan2.1-1.3B.

### B Analysis of Group Balancing Loss

As shown in Fig. 12, we validate the effectiveness of the group balancing loss, which measures the balance of the router’s token-to-group assignments. A higher value indicates that tokens concentrate in a few groups, whereas a lower value indicates more balanced grouping. When we include this loss during training, the metric rapidly converges to around 1, reflecting globally balanced assignments. In contrast, without it, the metric increases as the router funnels tokens into a few groups to gain short-term advantages in the diffusion MSE loss. Because our goal is to separate weakly related tokens and maintain balanced grouping, the additional group balance loss is necessary to enforce the desired assignments.

[Figure 567]

w/o group balance loss

with group balance loss

Figure 12 Group balancing loss curves of MoGA.

