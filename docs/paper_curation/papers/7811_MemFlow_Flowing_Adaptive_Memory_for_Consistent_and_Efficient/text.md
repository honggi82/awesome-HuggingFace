[Figure 1]

## MemFlow: Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives

Sihui Ji1,† Xi Chen1 Shuai Yang3 Xin Tao2 Pengfei Wan2 Hengshuang Zhao1

1HKU 2Kling Team, Kuaishou Technology 3HKUST(GZ) https://github.com/KlingTeam/MemFlow

# arXiv:2512.14699v1[cs.CV]16Dec2025

|[Figure 2]<br><br>Clip 1: A child plays with ball on a beach.<br><br>MemFlow 12s<br><br>|
|---|

|[Figure 3]<br><br>Clip 2: A dog runs towards the ball.<br><br>24s|
|---|

|[Figure 4]<br><br>Clip 3: A dog runs on the beach<br><br>36s|
|---|

|[Figure 5]<br><br>Clip 4: The child huggs the dog.<br><br>48s|
|---|

|[Figure 6]<br><br>Clip 5: The child stands besides the dog.<br><br>60s|
|---|

|[Figure 7]|
|---|

|[Figure 8]<br><br>LongLive|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]<br><br>MemFlow<br><br>Clip 1: A man decorates a Christmas tree.|
|---|

|[Figure 13]<br><br>Clip 3: They place the star on the tree.|
|---|

|[Figure 14]<br><br>Clip 4: There are many gifts under the tree.|
|---|

|[Figure 15]<br><br>Clip 5: They continue to decorate the tree.|
|---|

[Figure 16]

Clip 2: A woman brings a star decoration.

|[Figure 17]<br><br>LongLive|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]<br><br>MemFlow<br><br>Clip 1: A man walks in the forest.|
|---|

|[Figure 23]<br><br>Clip 3: The man petts the deer.|
|---|

|[Figure 24]<br><br>Clip 4: The man holds the deer.|
|---|

|[Figure 25]<br><br>Clip 5: The man walks back.|
|---|

[Figure 26]

Clip 2: The man meets a deer.

|[Figure 27]|
|---|

|[Figure 28]<br><br>LongLive|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

Figure 1. Existing streaming interactive text-to-video models such as LongLive [35] often fail to maintain consistency after prompt switching (suffering from redundant subjects or inter-clip inconsistency). MEMFLOW addresses this by maintaining dynamic memory for long-term consistency, enabling narrative coherence even if new subjects appear or scenario switches.

ent to-generate video chunks should refer to different historical cues, which is hard to satisfy with fixed strategies. In this work, we propose MEMFLOW to address this problem. Specifically, before generating the coming chunk, we dynamically update the memory bank by retrieving the most relevant historical frames with the text prompt of this chunk. This design not only accurately sources the context needed to maintain visual consistency, but also ensures semantic coherence even as new events unfold or scenes transition. In addition, during generation, we only activate the most

### Abstract

The core challenge for streaming video generation is maintaining content consistency over long context, which poses high requirement for the memory design. Most existing solutions maintain the memory by compressing historical frames with predefined strategies. However, differ-

† This work was conducted during the author’s internship at Kling Team, Kuaishou Technology.

Corresponding Author.

relevant tokens in the memory bank for each query in the attention layers, which effectively guarantees the generation efficiency. In this way, MEMFLOW achieves outstanding long-context consistency with negligible computation burden (7.9% speed reduction compared with the memory-free baseline) and keeps the compatibility with any streaming video generation model with KV cache.

### 1. Introduction

Video generation has attained remarkable quality [20, 21, 24, 29, 37], making its extension to long durations critical for advancing creative and cinematic applications. While Diffusion Transformer (DiT) models [26] leverage bidirectional attention to capture complex spatiotemporal dependencies, their inherent computational costs and GPU memory limits constrain them to short video generation. Autoregressive (AR) diffusion models [3, 15, 28, 39] offer a promising alternative by decomposing long videos into sequential clips, which alleviates the computation bottleneck through a reduced attention window.

Interactive video generation has emerged as a crucial application for enabling users to guide narratives with streaming prompt inputs. Most existing works conduct chunklevel autoregressive generation, where new video segments are streamingly generated based on previously generated content and newly-provided text prompts. This interactive paradigm with dynamic prompt transitions allows the introduction of new elements and scene switches across extended temporal horizons. However, it also poses difficulties in effectively preserving memory for long-range content consistency due to complex inter-clip dependencies. First, since different to-generate video chunks should refer to different historical cues, the memory is required to adaptively provide relevant context according to streaming prompts; Second, the capacity of stored memory must be highly constrained, a necessity dictated by both the hardware limits of GPU memory and the demands of generation efficiency.

While the necessity of such adaptive and efficient memory module is evident, many existing approaches have been overly simplistic, failing to fully address the dual challenges outlined above. They preserve memory in predefined paradigms, some only employ the first video chunk as memory sink [35], some attempt to store more historical frames through fixed compression schemes [8, 32, 42], some try to bake context implicitly with trainable memory modules [6, 12, 18]. However, those rigid strategies struggle to dynamically provide historical content corresponding to different prompt inputs, especially for new element emergence or scenario switches in prompt transitions.

Thus, we innovatively design Narrative Adaptive Memory (NAM), a memory mechanism that adaptively retrieves relevant historical content for interactive streaming video generation. Specifically, we introduce a memory bank ag-

gregating historical visual token (KV cache) from streamingly generated chunks. During the sequential generation of each chunk, we first retrieve the context which aligns with the current prompt most, by calculating the attention score between textual token of the prompt and visual token from memory. The context frame with higher score is considered to be semantically relevant with current chunk generation, and will be integrated to update the memory along with a condensed representation of the immediately preceding chunk. This design enables the current chunk to utilize historical cues, which have truly relevant content with the new prompt. Our NAM is effective in preserving narrative coherence even if new event happens or scenario switches, which is hard to satisfy with fixed memory strategies.

However, the introduction of memory inevitably brings an extra computation burden, which hinders real-time generation. Thus, we propose Sparse Memory Activation (SMA), which strategically activates only the most relevant tokens in attention layers according to the attention scores calculated from query (current chunk) and key (context in memory) by top-k selection. Subsequent attention is then applied within these selected tokens, which effectively accelerates inference by reducing computation cost while preserving quality.

In this way, our MEMFLOW effectively maintains contextual consistency over long durations and adeptly balances the memory–efficiency trade-off. It achieves stateof-the-art quality for interactive video generation with only 7.9% speed reduction compared with memory-free baseline. Our framework sustains 18.7 FPS on a single NVIDIA H100, demonstrating a clear advantage in producing narrative-coherent, long-term consistent videos with complex character and scene switching.

### 2. Related Works

Long Video Generation. Prior efforts to extend video generation to longer durations can be broadly categorized into three approaches. Autoregressive-diffusion hybrid approaches generate long videos by iteratively predicting frames [3, 15, 28, 35, 39, 42]. Diffusion-Forcing [2] mitigates error propagation by adjusting denoising schedules. CausVid [39] distills bidirectional models into efficient fewstep causal models, with Self Forcing [15] further addressing the train–test gap. MAGI-1 [28] and SkyReels-V2 [3] successfully scale up this AR-diffusion paradigm. Multistage methods [13, 31, 32, 47] decompose a long video into multiple clips to be generated separately. They either first synthesize a sequence of coherent keyframes followed by video infilling for each clip [14, 32, 46], or draft sequential prompts and use a T2V model to synthesize individual segments [23, 45]. A fundamental limitation of these approaches is the isolated nature of clip generation, which often leads to a lack of temporal coherence over long hori-

###### Autoregressive Generation

Mem Local Window

###### Narrative Adaptive Memory

| | | |
|---|---|---|

| | |
|---|---|
| | |

| | | |
|---|---|---|

| | |
|---|---|
| | |

Prompt N Prompt N+1 Prompt N+2

Previous Memory Bank Previous Chunk

Current Prompt

###### Autoregressive Diffusion Model Mem

Clip N Clip N+1 Clip N+2

[Figure 32]

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Memory Retrieval

Memory Utilization

[Figure 33]

[Figure 34]

[Figure 35]

The man is hugging the dog…

[Figure 36]

N+1N+2NClipClipClip

Current Chunk

[Figure 37]

AR-Diffusion Framework

[Figure 38]

Sparse Memory Activation Textual KV

| | | |
|---|---|---|

| | | |
|---|---|---|

Updated Memory

Frame Sink

Memory Updating

Memory Selection

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

Pruned KV

Prototypical KV

Relevance Gated

[Figure 39]

| | | |
|---|---|---|

| | | |
|---|---|---|

Updated Memory Bank

Selection

- Figure 2. Overall Framework of MEMFLOW, which is designed for interactive long video generation with both long-term consistency and efficiency. In autoregressive generation, we conduct memory retrieval, updating, and selection sequentially before finally use it for synthesizing the incoming video chunk (Sec 3.1). Given the current prompt and KV cache memory bank, we first use textual token from the prompt to query the memory for retrieving semantically aligned KV cache. After adding the prototypical KV cache of previous chunk to inject the latest context, the memory is updated for current chunk generation (Sec 3.2). The updated memory is then combined with ”Frame Sink”-the KV cache from the first chunk-for the following Sparse Memory Activation. SMA filters the most relevant context for attention computation, improving the generation efficiency without sacrificing visual quality. (Sec 3.3).

ing (Sec 3.2); then (2) Sparse Memory Activation (SMA) is employed for memory selection to address memoryefficiency trade-off (Sec 3.3). The refreshed memory is subsequently utilized by AR-diffusion model to synthesize the current video chunk, after which this process continues to roll out over extended temporal horizon. The entire framework is trained end-to-end using a streaming long-tuning strategy, enabling the model to learn how to effectively manage its memory during long-duration rollouts. Figure 2 provides a high-level illustration of our framework.

zons. The third category applies efficient architectures to manage computational costs. TTTVideo [6] and LaCT [43] learn context using neural networks with linear attention. TokensGen [25] represents video clips with condensed tokens. Mixture of Contexts [1] dynamically selects relevant context for attention computation. These methods often sacrifice visual fidelity for efficiency.

Memory Mechanisms in Video Generation. Effective memory mechanisms are crucial for maintaining consistency in long video generation. Action-guided video generation often relies on geometric and spatial dependencies [5, 33, 41]. Worldmem [33] and Context as memory [40] conduct memory retrieval based on Field of View (FOV) overlap between conditioned camera poses. VMem [22] introduces Surfel-Indexed View Memory for efficient retrieval by indexing past views with 3D surface elements. These methods, however, are highly dependent on spatial priors, thus lack generalizability. General video generation primarily maintains memory through context compression [8, 12, 42]. FramePack [42] compresses input frames into a fixed-size context to manage memory and efficiency. FAR [8] and StreamingT2V [11] combine short- and long-term memory via multiscale compression and learnable modules, respectively. These methods often maintain memory without adaptive retrieval, making it challenging to build dynamic connections between relevant context and the currently generated clip.

#### 3.1. Overall Framework

Baseline. Our work builds upon a hybrid autoregressivediffusion framework that integrates autoregressive chunkwise video generation with denoising diffusion [15, 28, 39]. At each generation iteration, the model produces a chunk of T frames, conditioned on the n immediately preceding frames. This autoregressive process naturally produces Key-Value (KV) cache from previous iterations, which is leveraged as the foundational structure for our memory bank. This design allows us to store historical context efficiently without incurring extra computational overhead. In a standard setup, the autoregressive attention mechanism operates over (n + T) local frames (the last n preceding frames and current T generated frames). By integrating our memory bank containing B frames, the attention operation is extended to cover (n+B +T) frames, seamlessly blending short-term dependencies and long-term memory.

### 3. Method

Training Mechanism. We train our memory-augmented AR-diffusion model using a distillation-based approach, specifically adopting the Self-Forcing [15] paradigm. Specifically, we adopt Distribution Matching Distillation (DMD) loss [38] that minimizes the gap between the student and teacher generator’s output distribution, to distill a pretrained bidirectional model into a few-step causal

MEMFLOW enhances long-video narrative consistency by incorporating a novel dynamic memory bank into a streaming video generation framework (Sec 3.1). To dynamically recall relevant historical context according to current prompt, we first adopt (1) Narrative Adaptive Memory (NAM) mechanism for memory retrieval and updat-

model. To equip the model with long-context capabilities, we employ a streaming long-tuning strategy [35]. During this phase, the generator samples a short video clip (e.g., 5s) in each round conditioned on previous clips, and the teacher provides reliable supervision on this newly generated short clip via DMD. We can repeat this rolling extension for generating long sequences until the video reaches a preset maximum length, with supervision applied throughout the entire rollout. Crucially, we integrate our memory mechanism (NAM and SMA) into this tuning process: employing NAM in the streaming tuning allows the model to learn how to retrieve relevant history from self-generated frames during training, aligning training with inference and improving long-range consistency; while SMA mitigates the computational overhead introduced by memory, incurring only 7.9% efficiency loss compared to the memory-free baseline.

#### 3.2. Narrative Adaptive Memory (NAM)

We first formulate the components of our memory bank in NAM. At each iteration, a new chunk is generated autoregressively, during which it is processed by the DiT to produce key-value (KV) representations {Kml ,Vml }Ll=1 at each transformer layer l, where m denotes the index of chunk generation iteration. At the beginning of next iteration, the memory is updated as {Kl

′

′

m}Ll=1 for subsequent computation. Our memory mechanism aims to provide contentaligned context for incoming generation, which necessitates its ability to retrieve history relevant to incoming prompt and incorporate the most recently generated content for updating. To avoid excessive expansion of the memory bank as generation proceeds, we introduce two synergistic techniques: (i) Semantic Retrieval, which retrieve most informative context based on the cross-attention relevance between textual queries and visual keys, and (ii) Redundant Removal, which leverages temporal redundancy to select the KV feature of first latent frame as prototype for the entire local chunk.

m,V l

Semantic Retrieval. During generation, each transformer layer l produces key–value representations for the current chunk while attending across the present sequence, the KV cache in local window, and the global memory bank. The retrieval criterion is derived from cross-attention scores between the textual tokens as query and visual tokens from KV cache as key, which has proven effective in prior works [4, 30, 36] of large vision-language models. In our design, the textual tokens are computed from the prompt of chunk to be generated, thus the visual tokens with higher scores are semantically-aligned with this chunk. By retrieving those KV cache in the memory bank, we expect the model to attend to content-relevant visual features.

Let Qltext ∈ Rd be the textual query of the current text prompt at layer l . For each of the b frames stored in the memory bank, represented by its key of KV cache Km,il ∈

Rn×d where i = 1,...,b, we compute a semantic relevance score, Sm,il :

Sm,il = Aggregate Softmax

Qltext(Km,il )⊤

√

d

, (1)

where the Softmax(·) computes attention weights, and Aggregate(·) is mean pooling here to produce a scalar score Sm,il ∈ R. Then we can identify the top-k most semantically aligned frames to be retained.

Redundant Removal. Following Semantic Retrieval, the immediately preceding chunk is consolidated into a representative prototype before being integrated into the memory. Instead of adopting computationally intensive contextmerging techniques [10, 36, 44] that rely on importance weighting, we propose a highly efficient heuristic. We leverage the high temporal redundancy inherent in short video chunks, where visual information exhibits strong similarity across consecutive frames. We posit that a single frame is sufficient to encapsulate the core visual content of the entire chunk. Therefore, we simply select the KV pair of the first frame from the preceding chunk to serve as its compact prototype. The updated memory bank {Kml′ ,Vml′} is then formed by concatenating the selected historical frames with the newly consolidated local prototype. The two strategies ensure that the memory is semantically relevant and real-time updated, enabling the model to build long-term and short-term dependencies crucial for narrative coherence.

#### 3.3. Sparse Memory Activation (SMA)

Directly extending local context window to incorporate a memory bank introduces computational burden, as attention complexity scales with context size. While rigidly compressing the context can improve efficiency, it often compromises memory quality, as critical historical cues may be discarded indiscriminately. To address this memory–efficiency trade-off, we introduce Sparse Memory Activation, a relevance-gated memory selection technique for dynamic memory pruning before attention computation.

Our approach operates on the principle of selective attention, where query token from the current video chunk attends only to a subset of the most relevant historical frames in memory. Formally, we first partition key (Kml ) and value (Vml ) of the memory bank into b frames. We then compute a compact descriptor for both the query (Qlvis) from current chunk and key of each frame using mean pooling over the token dimension, which is highly sufficient and expressive for generation tasks and has demonstrated by prior works [1]. This yields a single query descriptor, q¯vis ∈ R1×d, and a set of frame-wise key descriptors, {k¯j}bj=1 ∈ R1×d, the chunk index m is omitted here for simplicity. The relevance between the current query and

- Table 1. Quantitative comparison under multi-prompt 60-second setting with representitive long video generation models, where SkyReels-V2 [3], Self Forcing [15] and FramePack [42] are adapted for the task by directly switching prompts. All scores are measured over the whole sequence, except for the CLIP score, which is computed at intervals aligned with the prompt switching.

Quality Score ↑

Consistency Score ↑

Aesthetic Score ↑

CLIP Score ↑ 0–10s 10–20s 20–30s 30–40s 40–50s 50–60s

Method

SkyReels-V2 [3] 81.55 94.72 56.83 25.31 23.40 22.50 21.62 21.67 20.91 Self Forcing [15] 83.94 95.74 58.45 26.24 24.87 23.46 21.92 22.05 21.07 LongLive [35] 84.28 96.05 59.89 26.63 25.77 24.65 23.99 24.52 24.11 FramePack [15] 84.40 96.77 59.44 26.51 22.60 22.18 21.53 21.98 21.62 MEMFLOW 85.02 96.60 61.07 26.31 24.70 23.94 24.13 24.90 24.22

frame-wise key in memory is then determined by the inner product of their respective descriptors:

##### sj = q¯vis⊤ k¯j, for j = 1,...,b (2)

Based on these relevance scores, we identify the set of indices Ik corresponding to the top-k most relevant frames:

sj (3)

Ik = arg max

I⊆{1,...,b},|I|=k j∈I

This formulation selects the subset of indices I of size k that maximizes the sum of relevance scores. Finally, the attention computation for query Qlvis is restricted to the keyvalue pairs belonging to the selected top-k chunks:

Attn(Qlvis,Kml ,Vml ) ≈ Attn(Qlvis,Km,l I

,Vm,l I

) (4)

k

k

where Km,l I

are the concatenated key and value tensors from the chunks indexed by the set Ik.

and Vm,l I

k

k

By activating part of the memory bank, SMA reduces computational latency while retaining the most pertinent historical information. This strategy enables the model to selectively recall the right context at the right time, thereby preserving long-range dependencies and narrative coherence. Moreover, by implicitly filtering out less relevant or potentially erroneous information from the history, our approach mitigates error accumulation. This allows MEMFLOW to achieve both robust memorization and computational efficiency, ensuring the generation of coherent long videos without the degradation of visual quality over time.

### 4. Experiment

#### 4.1. Implementation Details

We build MEMFLOW on Wan2.1-T2V-1.3B [29], following the training and inference pipeline from LongLive [35], while enabling our memory bank and sparse activation. We perform Self Forcing [15] DMD pipeline with streaming long tuning on a 60s sequence by using the switchprompt dataset from LongLive constructed by Qwen2-72BInstruct [34]. We conduct streaming long tuning equipped

with the memory bank for 3000 steps. During training, each iteration continues the model’s own rollout by generating the next 5s video clip until a maximum length of 60s is reached.

#### 4.2. Comparisons for Multi-prompt Generation

Since our memory mechanism is designed for interactive long-form videos with multiple prompts, we first compare MEMFLOW’s abilities with representative long video generation models. For fair comparison, we adapt existing methods - including SkyReels-V2 [3], FramePack [42], and Self Forcing [15] - for multi-prompt video generation by switching prompts during the autoregressive synthesis. Note that LongLive [35] inherently supports generating videos with interactive instructions. Following LongLive [35], we customize 100 groups of narrative scripts, with each consisting of 6 successive 10-second prompts for a total of 100 videos lasting for 60 seconds.

We use metrics from VBench-Long [17] for assessing visual quality of all generated videos, among which the two dimensions of consistency and aesthetic are highlighted for comparison of long-range consistency in subjects, background and visual aesthetics. Table 1 shows that MEMFLOW achieves the best quality score among all methods, verifying its comprehensive competitiveness in perceptual quality. In terms of consistency score, our method outperforms all other models except Framepack [42], with Framepack tending to synthesize videos with reduced interframe dynamics, which definitely shows superiority in “consistency”. Thus the result can still demonstrate our superior performance in global consistency with the specially designed memory. Our advantage in aesthetic score also highlights the effectiveness of our method in mitigating error accumulation during long rollouts.

For text alignment with interactive prompts, videos are segmented according to prompt boundaries for evaluating clip-wise semantic adherence. The CLIP score [27] between each video clip and its corresponding text is calculated at 10-second intervals. The results demonstrate outstanding prompt adherence and narrative coherence, particularly when the video is extrapolated to longer durations,

- Table 2. Quantitative comparisons under single-prompt 5-second setting with representative open-source video generation models of similar parameter sizes and resolutions. Our MEMFLOW performs on par with other models on overall quality and has a clear advantage in semantic score attributed to textual retrieval-based memory. † denotes the scores reproduced by us.

Evaluation scores ↑ Total Quality Semantic Diffusion models

Throughput (FPS) ↑

Model #Params Resolution

LTX-Video [9] 1.9B 768×512 8.98 80.00 82.30 70.79 Wan2.1 [29] 1.3B 832×480 0.78 84.26 85.30 80.09

Autoregressive models

SkyReels-V2 [3] 1.3B 960×540 0.49 82.67 84.70 74.53 MAGI-1 [28] 4.5B 832×480 0.19 79.18 82.04 67.74 CausVid [39] 1.3B 832×480 17.0 81.20 84.05 69.80 NOVA [7] 0.6B 768×480 0.88 80.12 80.39 79.05 Pyramid Flow [19] 2B 640×384 6.7 81.72 84.74 69.62 Self Forcing, chunk-wise [15] 1.3B 832×480 17.0 84.31 85.07 81.28 Self Forcing, frame-wise [15] 1.3B 832×480 8.9 84.26 85.25 80.30 LongLive [35] 1.3B 832×480 20.3† 84.87 86.97 76.47 MEMFLOW 1.3B 832×480 18.7 85.14 85.95 81.90

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

12s 24s 36s 48s 60s

SkyReels-V2

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

FramePack

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Self-Forcing

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

LongLive

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Ours

- Figure 3. Qualitative comparisons under multi-prompt 60-second setting with representative long video generation models, where MEMFLOW outperforms other alternatives in narrative coherence and subject consistency, without drifting or duplicated characters.

owing to the model’s ability to establish long-term contextual associations. Such superiority is further corroborated by the qualitative results in Figure 3. Our Narrative Adaptive Memory successfully links description in the prompt, “a woman in a casual sweater”, with the exact person in

previous frames, thus maintaining the subject consistency. While our baseline, LongLive [35], fails to make connections between visual cues and semantic instructions, therefore continuously introduces new characters after prompt switching, exhibiting less temporal coherence and prompt

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

12s 24s 36s 48s 60s

w/o Memory

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Frame Sink

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

NAM

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

NAM+SMA

- Figure 4. Qualitative analysis of different memory mechanisms under multi-prompt 60-second setting. “w/o Memory” means only attending to the local attention window, “Frame Sink” refers to keeping KV cache from the first chunk as memory, “NAM” adopts the whole memory bank without filtering, and “NAM+SMA” is our full model which compresses memory by relevance-gated selection.

compliance. Other approaches exhibit more severe error accumulation, with subject inconsistency in SkyReels-V2 [3] and color drifting in FramePack. Self Forcing [15] also suffers from similar problems with LongLive, showing misalignment between prompt scripts and narrative progression across clips as characters are repeatedly introduced into the ongoing scene. Additionally, We conducted a user study with 20 participants to compare our method against the aforementioned models, and report the result in supplementary material. It includes human preference rates in visual quality, instruction following, and global consistency, further supporting the effectiveness of our approach. In terms of speed, MEMFLOW is more than 38× faster than SkyReels-v2 [3], slightly faster than Self Forcing [15], while slightly slower than LongLive [35] due to memory updating and activation.

#### 4.3. Comparisons for Single-prompt Generation

Although not specifically trained for single-prompt generation, our model shows superior performance compared to state-of-the-art models for durations of 5s and 30s on VBench [16] official prompt set.

Short Video Generation. We evaluate MEMFLOW ’s short-video generation and compare it with relevant open-source models of similar scale, including LTXVideo [9], Wan2.1 [29], SkyReels-V2 [3], MAGI-1 [28], CausVid [39], NOVA [7], Pyramid Flow [19], Self Forcing [15], and LongLive [15]. For 5-second videos, MEMFLOW achieves strong overall quality with the highest to-

tal score, compared with state-of-the-art models as in Table 2. By retrieving relevant context from Narrative Adaptive Memory via prompt query, our model surpasses all other models in semantic score. Due to computation cost in memory updating and activation, our MEMFLOW sacrifices inference speed by 8.6% compared with LongLive, but still outperforms other methods with 18.7 FPS for real-time inference. The results also show that our framework does not degrade short-clip generation capability.

Long video generation. The superiority of our method becomes more pronounced in long-horizon single prompt generation for 30-second videos. We observe consistent improvements across quality and semantic metrics in Table 4, leading to an outstanding overall performance than SkyReels-V2 [3], FramePack [42], Self Forcing [15], and LongLive [35]. It verifies that our Narrative Adaptive Memory provides more semantic-consistent context for video generation over long duration compared with using local context window only (SkyReels-V2, Self Forcing), context compression (FramePack) or retaining the first chunk as memory (LongLive). Moreover, the retrieval-based memory updating strategy interrupts the error propagation in memorization implicitly-only context with the highest level of semantic adherence is included for attention computation-thus alleviating the degradation of visual quality due to error accumulation over time. MEMFLOW maintains an advantage in long video generation quality with comparable performance on efficiency.

- Table 3. Quantitative analysis of different memory mechanisms under multi-prompt 60-second setting. “w/o Memory” means only attending to the local attention window, “Frame Sink” refers to keeping KV cache from the first chunk as memory [35], “NAM” adopts the whole memory bank without filtering, and “NAM+SMA” is our full model which compresses memory by relevance-gated selection.

Memory Mechanism

Subject Consistency ↑

Background Consistency ↑

Throughput (FPS) ↑

CLIP Score ↑ 0–10s 10–20s 20–30s 30–40s 40–50s 50–60s

w/o Memory 94.41 95.15 23.5 26.74 25.10 24.60 23.61 24.23 24.14 Frame Sink [35] 97.66 96.20 20.3 26.63 25.77 24.65 23.99 24.52 24.11 NAM+SMA 98.01 96.70 18.7 26.31 24.70 23.94 24.13 24.90 24.22 NAM 98.05 96.57 17.6 26.50 25.30 24.42 24.23 24.96 24.28

- Table 4. Quantitative comparisons for single-prompt 30second setting with representative long video generation models, showing more pronounced superiority than 5-second setting on all metrics with efficiency comparable to state-of-the-art models.

Sink is utilized by LongLive [35], allowing direct comparison with its model. The configuration of w/o memory is also implemented on the baseline of LongLive by removing the sink latent frames. Table 3 highlights the effectiveness of our NAM, which consistently outperforms others in maintaining temporal consistency and semantic coherence. The retrieval-based memory establishes intrinsic dependencies across contexts, enabling stable narrative transitions even under subject insertion or switching. With SMA, inference efficiency improves from 17.6 to 18.7 frames per second with minimal quality degradation. As shown in Figure 4, removing memory causes abrupt scene transitions, while Frame Sink preserves continuity only for initial subjects but collapses on later ones. In contrast, our model captures the relations between existing and emerging subjects, achieving superior semantic alignment on switched prompts, especially as the video extends beyond 30 seconds.

Total Score ↑

Quality Score ↑

Semantic Score ↑

Throughput (FPS) ↑

Model

SkyReels-V2 [3] 75.29 80.77 53.37 0.49 FramePack [42] 81.95 83.61 75.32 0.92 Self Forcing [15] 81.59 83.82 72.70 17.0 LongLive [35] 83.52 85.44 75.82 20.3 MEMFLOW 84.51 85.92 78.87 18.7

0.28

0.27

0.26

CLIPScore

0.25

Memory Capacity. In Figure 5, we ablate the impact of our two key components under the 60-second setting: the capacity of origin memory in NAM and activated memory after SMA. The left panel analyzes NAM with varying capacities b = {3,6,9} against two baselines “w/o Memory” and “Frame Sink”. The results reveal that a larger memory capacity does not guarantee better performance. Notably, NAM (b=6) consistently underperforms the baseline, while NAM (b=9) exhibits significant performance instability. We attribute this to an imbalance within the attention’s receptive field: as memory capacity b increases, the proportion of global context from memory significantly outweighs that of the local window. This over-reliance on global context can disrupt short-term narrative flow, leading to the observed fluctuations in CLIP score. Therefore, we select NAM (b=3), a capacity equivalent to half the size of our local context window, as it provides the most stable balance between local and global context and effective enhancement in semantic coherence over baseline.

0.24

0.23

0.22

0-10s 10-20s 20-30s 30-40s 40-50s 50-60s

Time(s)

None Frame Sink NAM (b=3) NAM (b=6) NAM (b=9)

- Figure 5. Quantitative analysis of different memory capacity under multi-prompt 60-second setting. “w/o Memory” means only attending to the local attention window, “Frame Sink” refers to keeping KV cache from the first chunk as memory [35], “NAM” adopts the whole memory bank including b latent frames.

#### 4.4. Ablation Studies

We perform an ablation study on the core designs of our framework, Narrative Adaptive Memory and Sparse Memory Activation, in a 60-second interactive multi-prompt video generation setting with five prompt switches.

Memory Mechanism. In Table 3, we ablate different memory mechanism by comparing (i) w/o memory: conditioned only on local context window; (ii) Frame Sink: addtionally retain the KV cache from the first chunk; (iii) Narrative Adaptive Memory (NAM): maintain our dynamic memory bank; and (iiii) Narrative Adaptive Memory and Sparse Memory Activation (NAM+SMA): our full model. Frame

### 5. Conclusion

In this work, we introduce MEMFLOW, a memory mechanism for equipping interactive long video generation with long-range consistency without severe efficiency degrada-

tion. For maintain long-term memory for narrative coherence, we design Narrative Adaptive Memory for dynamic retrieval of semantic-aligned context via textual query. We also introduce Sparse Memory Activation for balancing the memory-efficiency trade-off by relevance-gated memory filtering. Our model achieves 18.7 FPS inference on a single NVIDIA H100 GPU, and supports interactive video generation while maintaining consistency, visual quality and narrative coherence even under complex narrative transitions and character switching.

### References

- [1] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025. 3, 4
- [2] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2025. 2
- [3] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreels-v2: Infinite-length film generative model. CoRR, abs/2504.13074, 2025. 2, 5, 6, 7, 8
- [4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024. 4
- [5] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025. 3
- [6] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In CVPR, pages 17702–17711, 2025. 2, 3
- [7] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025. 6, 7
- [8] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame prediction. CoRR, abs/2503.19325, 2025. 2, 3
- [9] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. CoRR, abs/2501.00103, 2025. 6, 7
- [10] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam

- Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13504–13514, 2024. 4
- [11] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2568–2577, 2025. 3
- [12] Yining Hong, Beide Liu, Maxine Wu, Yuanhao Zhai, KaiWei Chang, Linjie Li, Kevin Lin, Chung-Ching Lin, Jianfeng Wang, Zhengyuan Yang, et al. Slowfast-vgen: Slowfast learning for action-driven long video generation. arXiv preprint arXiv:2410.23277, 2024. 2, 3
- [13] Kaiyi Huang, Yukun Huang, Xintao Wang, Zinan Lin, Xuefei Ning, Pengfei Wan, Di Zhang, Yu Wang, and Xihui Liu. Filmaster: Bridging cinematic principles and generative ai for automated film generation. arXiv preprint arXiv:2506.18899, 2025. 2
- [14] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 2
- [15] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. CoRR, abs/2506.08009,

2025. 2, 3, 5, 6, 7, 8

- [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 7
- [17] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. In arXiv, 2024. 5
- [18] Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025. 2
- [19] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025. 6, 7
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv:2412.03603, 2024. 2
- [21] Kuaishou. Kling video model. https://kling. kuaishou.com/en, 2024. 2
- [22] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 3

- [23] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videostudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, pages 468–485. Springer, 2024. 2
- [24] OpenAI. Sora: Creating video from text, 2024. 2
- [25] Wenqi Ouyang, Zeqi Xiao, Danni Yang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Tokensgen: Harnessing condensed tokens for long video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18197–18206, 2025. 3
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4172–4182, 2023. 2
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 5
- [28] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. MAGI-1: autoregressive video generation at scale. CoRR, abs/2505.13211, 2025. 2, 3, 6, 7
- [29] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 5, 6, 7
- [30] Xiao Wang, Qingyi Si, Shiyu Zhu, Jianlong Wu, Li Cao, and Liqiang Nie. Adaretake: Adaptive redundancy reduction to perceive longer for video-language understanding. In Findings of the Association for Computational Linguistics: ACL 2025, pages 5417–5432, 2025. 4
- [31] Xunzhi Xiang, Yabo Chen, Guiyu Zhang, Zhongyu Wang, Zhe Gao, Quanming Xiang, Gonghu Shang, Junqi Liu, Haibin Huang, Yang Gao, et al. Macro-from-micro planning for high-quality and parallelized autoregressive long video generation. arXiv preprint arXiv:2508.03334, 2025. 2
- [32] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint arXiv:2507.18634, 2025. 2
- [33] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025. 3
- [34] An Yang, Jinze Bai, et al. Qwen2 technical report. arXiv,

2024. 5

- [35] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive

long video generation. arXiv preprint arXiv:2509.22622,

2025. 1, 2, 4, 5, 6, 7, 8

- [36] Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, and Mengye Ren. Streammem: Query-agnostic kv cache memory for streaming video understanding. arXiv preprint arXiv:2508.15717,

2025. 4

- [37] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv:2408.06072,

2024. 2

- [38] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, pages 6613–6623, 2024. 3
- [39] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 2, 3, 6, 7
- [40] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141,

2025. 3

- [41] Shangjin Zhai, Zhichao Ye, Jialin Liu, Weijian Xie, Jiaqi Hu, Zhen Peng, Hua Xue, Danpeng Chen, Xiaomeng Wang, Lei Yang, et al. Stargen: A spatiotemporal autoregression framework with video diffusion model for scalable and controllable scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26822– 26833, 2025. 3
- [42] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. CoRR, abs/2504.12626, 2025. 2, 3, 5, 7, 8
- [43] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. In arXiv, 2025. 3
- [44] Yiming Zhang, Zhuokai Zhao, Zhaorun Chen, Zenghui Ding, Xianjun Yang, and Yining Sun. Beyond training: Dynamic token merging for zero-shot video understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22046–22055, 2025. 4
- [45] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequences. In ICLR, 2025. 2
- [46] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024. 2
- [47] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8806–8817, 2024. 2

