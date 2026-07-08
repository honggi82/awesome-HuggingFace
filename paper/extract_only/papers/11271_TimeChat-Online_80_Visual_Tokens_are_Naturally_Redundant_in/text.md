# arXiv:2504.17343v1[cs.CV]24Apr2025

## TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos

Linli Yao♥,∗ Yicheng Li♥,∗ Yuancheng Wei⋄,∗ Lei Li♠ Shuhuai Ren♥

Yuanxin Liu♥ Kun Ouyang♥ Lean Wang♥ Shicheng Li♥ Sida Li♥ Lingpeng Kong♠ Qi Liu♠ Yuanxing Zhang‡ Xu Sun♥

♥Peking University, ⋄South China University of Technology, ♠The University of Hong Kong, ‡Kuaishou Technology https://timechat-online.github.io

###### Video Streams (1 fps)

[Figure 1]

Original

[Figure 2]

t=1-8s

Dropped

t=9-16s

[Figure 3]

t=17-24s

t=25-32s

Figure 1: This paper presents TimeChat-Online for efficient Streaming Video Understanding [6]. Its core design is the Differential Token Dropping (DTD) module that selectively preserves only significant temporal changes across video streams. The DTD eliminates 82.8% of redundant video tokens without any user-query guidance, while achieving a 1.76× speedup in response latency and maintaining over 98% of original accuracy. Furthermore, it naturally monitors video scene transitions, facilitating online Proactive Responding.

### Abstract

in video tokens while maintaining 98% performance on StreamingBench, revealing that over 80% of visual content in streaming videos isnaturallyredundantwithoutrequiringlanguageguidance.Toenable seamless real-time interaction, we present TimeChat-Online-139K, a comprehensive streaming video dataset featuring diverse interaction patterns including backward-tracing, current-perception, and futureresponding scenarios. TimeChat-Online’s unique Proactive Response capability,naturallyachievedthroughcontinuous monitoring ofvideo scene transitions via DTD, sets it apart from conventional approaches. Our extensive evaluation demonstrates TimeChat-Online’s superior performance on streaming benchmarks (StreamingBench and OvOBench) and maintaining competitive results on long-form video tasks such as Video-MME and MLVU. Notably, when integrated with Qwen2.5VL-7B, DTD achieves a 5.7-point accuracy improvement on the challenging VideoMME subset containing videos of 30-60 minutes, while reducing video tokens by 84.6%. Our work establishes a new paradigm for efficient streaming video understanding and reveals the potential of leveraging natural video redundancy for future VideoLLMs development.

The rapid growth of online video platforms, particularly live streaming services, has created an urgent need for real-time video understanding systems. These systems must process continuous video streams and respond to user queries instantaneously, presenting unique challenges for current Video Large Language Models (VideoLLMs). While existing VideoLLMs excel at processing complete videos, they face significant limitations in streaming scenarios due to their inability to handle dense, redundant frames efficiently. We introduce TimeChat-Online, a novel online VideoLLM that revolutionizes real-time video interaction. At its core lies our innovative Differential Token Drop (DTD) module, which addresses the fundamental challenge of visual redundancy in streaming videos. Drawing inspiration from human visual perception’s Change Blindness phenomenon, DTD preserves meaningful temporal changes while filtering out static, redundant content between frames. Remarkably, our experiments demonstrate that DTD achieves an 82.8% reduction

*Equal contribution. linliyao@stu.pku.edu.cn.

### CCS Concepts

• Computing methodologies → Natural language generation.

### Keywords

Video Large Language Models, Streaming Video Understanding, Video Token Pruning

### 1 Introduction

The proliferation of online video platforms (e.g., live broadcasts) and real-world applications (e.g., domestic robots and surveillance systems) has driven researchers to focus on online video understanding with continuous streams, commonly referred to as Streaming VideoQA [6, 12, 44, 62, 75, 78]. In streaming video scenarios, video assistants must continuously process incoming frames while simultaneously enabling real-time user interaction, remaining responsive to user queries posed at any moment.

Streaming video tasks introduce two fundamental challenges: Firstly, Long-form High-redundant Video Context: successive video frames are received at high frame rates (typically 1-10 FPS in real-world applications [6]), with neighboring frames exhibiting substantial similarity in backgrounds and static objects. Additionally, video streams are potentially infinite, creating extensive temporal context that must be maintained across the timeline. Secondly, Realtime Interaction with Proactive Responding: streaming video tasks involve backward tracing, current-time perception, and forward responding. When presented with a user query at a specific moment, a VideoLLM must efficiently access past and current visual context to generate immediate responses with minimal latency. For questions requiring future visual cues not yet available, the model must possess Proactive Responding capabilities to automatically trigger responses at a prospective timestamp when relevant visual cues become available.

Despite recent advancements in Video Large Language Models (VideoLLMs) [3, 5, 26, 27, 35, 46, 73, 78, 80], these models struggle with online video understanding. They are primarily designed for offline video processing, where they receive and process entire videos at once. In online scenarios, they cannot implement proactive responses and face difficulties with long-form high-redundancy video streams: short-context VideoLLMs that uniformly sample sparse video frames [34, 35, 80], such as 32 or 64 frames, suffer from significant visual context loss; conversely, long-context VideoLLMs that densely sample video frames at 1 FPS (frame-per-second), such as Qwen2.5VL-7B [5], incur substantial response delays when processing computationally intensive video tokens. Recent approaches have proposed Average Pooling [5, 24] or Resampler-based mechanisms [4, 11, 31, 46, 74] to compress redundancy in long-form videos. However, these compression methods impose a fixed number of tokens per frame, failing to adapt to the variable redundancy inherent in dynamic video streams. Concurrently, language-guided approaches [11, 28, 48] prove inefficient for streaming scenarios, as they necessitate reprocessing of all historical dense frames whenever a new user query is received.

Toaddressthesechallenges,thispaperpresentsTimeChat-Online, a novel online VideoLLM that efficiently enables real-time interaction with streaming video content. To tackle the long-form and high-redundancy challenges of video streams, we propose a

Differential Token Drop (DTD) mechanism. Inspired by human visual perception phenomena such as Change Blindness [47, 49], our approach mimics how the human visual system processes continuous video streams: not by capturing every detail in each frame, but by selectively focusing on salient spatial-temporal changes while filtering out static, redundant content. As Figure 1 illustrates, our DTD mechanism adaptively preserves only the changed visual tokens between successive frames from a joint spatial-temporal perspective. This approach significantly reduces video token count by 82.8% in streaming videos purely at the visual level without requiring any textual information. Notably, it maintains VideoQA accuracy at comparable levels to full-token processing, indicating that more than 80% of the visual context in video streams is naturally redundant.

While DTD effectively addresses the challenge of processing long-form high-redundant video streams, we must also tackle the real-time interaction requirements of online video understanding. To this end, we introduce a new training dataset TimeChat-Online-139K that addresses the scarcity of data specifically designed for streaming VideoQA. We collect long-form videos averaging 11.1 minutes in length and utilize GPT-4o [41] to annotate them with diverse streaming VideoQA pairs that encompass backward tracing, currenttime understanding, and forward responding task types. To enable proactive responding for future-oriented questions, we construct negative samples in which the questions cannot be answered using the currently available streaming video content. For future-responding question interactions, TimeChat-Online is designed to be triggered at video scene transition timestamps, generating new responses with the updated video context. As Figure 1 illustrates, scene transitions are naturally indicated by frames with few tokens dropped, signifying significant visual differences from previous frames.

To summarize, our contributions are three-fold: I) We propose TimeChat-Online with a novel Differential Token Drop module that significantly reduces spatial-temporal redundancy in streaming videos (eliminating 82.8% of tokens, achieving a 1.76× speed-up while maintaining 98% accuracy on StreamingBench). The DTD can also be seamlessly integrated with general VideoLLMs such as Qwen2.5VL to substantially enhance its efficiency in long video tasks (reducing 87.2% of video tokens and yielding a 5.6 absolute accuracy gain on the VideoMME long set). II) We develop a new instruction-tuning dataset, TimeChat-Online-139K, specifically designed to facilitate more flexible streaming VideoQA interactions. III) Experimental results demonstrate that TimeChat-Online achieves state-of-the-art performance on streaming video benchmarks, including StreamingBench (58.0) and OVO-Bench (47.6). Furthermore, it maintains competitive performance on general long-form video understanding tasks such as VideoMME (63.3), MLVU (65.4), and LongVideoBench (57.7).

### 2 Related Work

Streaming Video Understanding. Streaming video understanding, which processes continuously updating video frames, was first introduced by VideoLLM-online [6]. Recent advancements follow two primary directions. The first focuses on efficient encoding of dense video streams. Memory-bank-based approaches like VideoStreaming [44], Flash-VStream [75], StreamChat [61], and VideoChatonline [21] utilize dynamic memory banks to retain informative

video tokens. ReKV [12] and Inf-MLLM [39] optimize KV Cache management for video context, while VideoLLM-MoD [60] employs Mixture-of-Depth to reduce token count. Most of these methods require language guidance from user queries to select relevant video content. In contrast, our proposed DTD reduces video tokens by over 80% purely-visually as an early step to lighten the computational burden of video tokens before language model processing. The second direction enhances real-time interaction experiences in streaming scenarios. IXC2.5-Omni [77] and Qwen2.5-Omni [62] incorporate audio modality, MMDuet [57] refines video-text duet interaction formats, and Dispider [43] introduces a disentangled Perception-Decision-Reaction paradigm. Our work differs by implementing a novel proactive response paradigm that leverages our synthetic streaming training dataset TimeChat-Online-139K and is intelligently triggered by scene transitions naturally revealed by DTD’s token drop ratio curves.

Efficient Video Token Pruning. Recent advancements in long video processing have led to diverse approaches for video token pruning [28, 38, 45, 66–69]. A branch of methods [28, 46, 54, 63, 66] compress frames or clips to a fixed number, disregarding the dynamic visual redundancy inherent in different videos. For instance, LLamavid [32] represents each frame with fixed two tokens. To address the limitations, several methods [22, 45, 53, 68] design adaptive token merging in either spatial or temporal dimensions. While they offer improved flexibility, they blur the vanilla spatial-temporal positions which will hurt the fine-grained video perception such as spatial localization, action ordering, or temporal counting task. Furthermore, selection-based methods without merging [58] primarily consider spatialredundancywhileneglectingtemporalredundancy.Incontrast, our DTD adaptively drops video tokens via dynamic temporal redundancy while preserving the related spatial-temporal positions. Another category of approaches [7, 48, 81] leverages language guidance through user queries or vision-language cross-attention mechanisms for token pruning. However, these language-guided methods are inefficient for streaming scenarios, as they require reprocessing all historical frames for each new user query. This significantly increases computational burden and introduces response delays, making them impractical for real-world applications. In contrast, our DTD efficiently processes video streams by calculating redundancy only for newly-arriving frames with faster speed.

### 3 TimeChat-Online Framework

Thispaperaddressesthestreamingvideoquestion-answering(Streaming VideoQA) task [33, 61] and introduces an online video assistant, TimeChat-Online, capable of providing real-time responses to user queries posed at specific timestamps within video streams. In Section 3.1, we first formulate the Streaming VideoQA task and distinguish it from offline video understanding tasks. Then, in Section 3.2, we present the design of the Differential Token Drop, which eliminates over 80% of video tokens by retaining significant temporal changes while removing static redundancy. Next, we introduce the curation process of TimeChat-Online-139K for training streaming video tasks in Section 3.3 and describe the training strategy and inference procedure in Section 3.4.

- 3.1 Streaming VideoQA Formulation Given a video consisting of𝑇 frames in total, the streaming task will

consistently update the incoming video frames F = [𝑓1, 𝑓2, ..., 𝑓𝑡] where 1 ≤ 𝑡 ≤ 𝑇 along the timeline rather than given all video frames at once. When a user question 𝑄𝑡 is proposed at a Current timestamp 𝑡, the goal of Streaming-VideoQA task is to leverage the Current frame content 𝑓𝑡 and the Past video streams F1:𝑡−1 to answer the 𝑄𝑡. Proactive Responding. When the historical video content F1:𝑡 is insufficient to answer the user question 𝑄𝑡, the model should be able to proactively respond at a Future timestamp 𝑡′ (𝑡 ≤ 𝑡′ ≤ 𝑇) when the updated video content F𝑡+1:𝑡′ is sufficient to answer 𝑄𝑡. Moreover, for questions that can be partially answered with current content, a satisfactory video assistant should also be able to provide new responses at Future timestamps as more visual information becomes available. For example, as illustrated in Figure 2, when the user asks "How does the man cook the cauliflower?" at 𝑡 = 10𝑠, the video assistant can proactively respond at future timestamps 𝑡′ = 25𝑠 (“He chops the cauliflower”) and 𝑡′ = 35𝑠 (“He puts the cauliflower into the frying pan”) with newly visual content about “the cooking man” as the video progresses over time.

Online VideoLLMs vs. Offline VideoLLMs. For streaming video content F1:𝑡 and a specific user question 𝑄𝑡, it can be converted to an offline VideoQA task by inputting the historical frames F1:𝑡 as the whole video content along with the user question 𝑄𝑡 to existing offline VideoLLMs. However, these offline VideoLLMs have two main limitations: Firstly, existing offline VideoLLMs cannot efficiently handle the dense frames of high-FPS video streams (e.g., 1-10 FPS). Vanilla dense-sampling VideoLLMs such as Qwen2VL-7B [5] that can take 1 FPS frames as input incur unsatisfactory response delays when processing long-form video streams. Conversely, sparse sampling VideoLLMs such as LLava-Video-7B [80] and VILA [34], which uniformly sample a fixed number of frames (64 and 14 frames)

- as input, suffer from significant visual information loss. Secondly, existing offline VideoLLMs are not able to provide proactive responses
- at future timestamps for questions that require upcoming visual context, which significantly limits the user interaction experience. Therefore, this paper aims to propose an online video assistant that can both efficiently handle high-FPS sampled video streams and provide proactive interaction for future-responding questions.

- 3.2 Differential Temporal Token Drop

Inspired by the Change Blindness phenomenon in human visual perception [49], we propose a Differential Token Drop (DTD) module to efficiently reduce video redundancy by only preserving the significant temporal changes in video streams. As Figure 2 shows, DTD consists of three main steps: (a) Patchify and Encoding, (b) Static Redundancy Calculation, and (c) Position-aware Token Dropping.

3.2.1 Patchify and Encoding. We use a ViT [13] to split each video frame 𝑓𝑡 into a sequence of visual patches P𝑡 = [𝑝1,𝑝2, ...,𝑝𝐻×𝑊 ] and encode the related spatial tokens as V𝑡 = [𝑣1,𝑣2, ...,𝑣𝐻×𝑊 ], where 𝐻 and𝑊 denotes the maximum position index in height and width respectively. For video streams F1:𝑡 received at timestamp 𝑡, we can get the temporal sequence of patches P1:𝑡 and the related visual tokens V1:𝑡.

Efficient Video Stream Encoding

Past

Current Time

###### Future Time

1s 2s 3s t -1 = 38s

t = 39s

40s 41s

|[Figure 4]|
|---|

|[Figure 5]<br><br>(38,0,0) (38,0,1) (38,0,2)<br>(38,1,0)<br>(38,2,0)<br><br><br>(38,1,1) (38,1,2)<br>(38,2,1) (38,2,2)<br>|
|---|

|[Figure 6]<br><br>(39,0,0) (39,0,1) (39,0,2)<br>(39,1,0)<br>(39,2,0)<br><br><br>(39,1,1) (39,1,2)<br>(39,2,1) (39,2,2)<br>|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

(a) Patchify & Encoding

Invisible to the model at the current timestamp

|(38,0,0)|
|---|

|(38,0,1)|
|---|

|(38,0,2)|
|---|

|(39,0,0)|
|---|

|(39,0,1)|
|---|

(b) Static Redundancy Calculation

|(38,1,0)|
|---|

|(38,1,1)|
|---|

|(38,1,2)|
|---|

|(39,1,1)|
|---|

(

|(38,2,0)|
|---|

|(38,2,1)|
|---|

|(38,2,2)|
|---|

|(39,2,1)|
|---|

M-ROPE (t, h, w) Position Reservation

sim( , ) ≥ τ sim( , ) < τ Reserve

Drop

(c) Position-aware Token Dropping

~ 82.5% tokens dropped

(39,0,0) (39,0,1) (39,1,1) (39,2,1)

Past Current

Future Frames

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

##### Real-Time Response

10s 25s

35s

[Figure 20]

Proactive Response

Trigger Time

Drop Ratio

How does the man cook the cauliflower?

[Figure 21]

[Figure 22]

: He puts

[Figure 23]

the cauliflowe r into the frying pan.

Timeline

[Figure 24]

[Figure 25]

[25s] : He puts the cauliflower

: He chops the cauliflower.

[35s]

[Figure 26]

[10s] : He washes the cauliflower.

into the frying pan.

Figure 2: The core of TimeChat-Online lies in the Differential Token Dropping (DTD) design for efficiently encoding video streams. DTD captures significant temporal changes through three steps: (a) patchifying and encoding dense video frames, (b) calculating static redundancy between temporally-consecutive and spatially-identical video tokens, (c) dropping temporally-redundant video tokens while preserving the (temporal, height, width) positions of remaining tokens. DTD dynamically eliminates visual redundancy in the temporal dimension, yielding an adaptive drop ratio for each frame. During Real-Time Interaction, frames with low drop ratios in the timeline indicate video scene transitions, triggering TimeChat-Online to achieve Proactive Responding at these scene-oriented timestamps.

- 3.2.2 Static Redundancy Calculation. Intuitively, if temporally consecutive frames (𝑓𝑡−1, 𝑓𝑡) are visually similar, we determine the latter frame 𝑓𝑡 as a redundant frame, because it contains the same static visual content as the former frame 𝑓𝑡−1. From a more fine-grained perspective, we can formulate static redundancy by comparing temporally-consecutive patches P or visual tokens V at pixel-level and feature-level, respectively. Pixel-level Redundancy. Inspired by the RLT [10] method, we select two temporally consecutive and spatially identical patches

𝑣ℎ𝑤𝑡−1 · 𝑣ℎ𝑤𝑡 𝑣ℎ𝑤𝑡−1

𝑆𝑖𝑚(𝑣ℎ𝑤𝑡−1,𝑣ℎ𝑤𝑡 ) =

> 𝜏𝑓 𝑒𝑎𝑡 (2)

𝑣ℎ𝑤𝑡

2

2

If the selected two visual tokens are visually similar, they will have similar feature embeddings, leading to a large cosine similarity. We define a hyperparameter 𝜏𝑓 𝑒𝑎𝑡 to determine the threshold of feature-level redundancy. The values of 𝜏𝑝𝑖𝑥𝑒𝑙 and 𝜏𝑓 𝑒𝑎𝑡 can control the overall drop ratio of video tokens. They measure how different two temporally consecutive patches/tokens are, depending only on the inherent nature of videos. Empirically, we verify that the 𝜏 values are consistent across different datasets. For example, 𝜏𝑓 𝑒𝑎𝑡 = 0.25 corresponds to around 85% of video tokens being dropped across three datasets, while𝜏𝑓 𝑒𝑎𝑡 = 0.5 corresponds to around 45%. Notably, unlike feature-level token pruning approaches [7] that operate inside LLMs, our pixel-level and feature-level calculations are purely visionbased strategies that operate before the Large Language Model (LLM) [14, 65] module, which achieves greater efficiency.

(𝑝ℎ𝑤𝑡−1,𝑝ℎ𝑤𝑡 ), where ℎ and 𝑤 are the height and width indices in the spatial dimension. Next, we calculate the pixel similarity between

- them using the L1 distance [1] as:

𝑆𝑖𝑚(𝑝ℎ𝑤𝑡−1,𝑝ℎ𝑤𝑡 ) = 𝑝ℎ𝑤𝑡−1 − 𝑝ℎ𝑤𝑡 1 < 𝜏𝑝𝑖𝑥𝑒𝑙 (1)

If the selected two patches are visually similar, they will have close pixel values, leading to a small L1 distance. We define a hyperparameter𝜏𝑝𝑖𝑥𝑒𝑙 to determine the threshold of pixel redundancy. Feature-level Redundancy. Besides pixel-level redundancy, we can also calculate the feature-level visual redundancy. The cosine similarity [51] between temporally-consecutive spatially-aligned visual tokens 𝑣ℎ𝑤𝑡−1 and 𝑣ℎ𝑤𝑡 is defined as:

3.2.3 Position-aware Token Dropping. Based on the calculated similarity between token (or patch) pairs, we identify redundant tokens (or patches) in the Current frame 𝑓𝑡 and generate a binary

- Table 1: Performance comparison on StreamingBench focusing on Real-Time Visual Understanding tasks. Real-Time Visual Understanding encompasses Object Perception (OP), Causal Reasoning (CR), Clips Summarization (CS), Attribute Perception (ATP), Event Understanding (EU), Text-Rich Understanding (TR), Prospective Reasoning (PR), Spatial Understanding (SU), Action Perception (ACP), and Counting (CT). “VTokens(%)” represents the percentage of video tokens remaining after dropping, where 100% indicates no dropping, and ↓ 82.6% signifies an 82.6% reduction in video tokens. The bold values indicate the best performance and underlined values indicate the second best.

Model #Frames VTokens(%) OP CR CS ATP EU TR PR SU ACP CT All Human - - 89.47 92.00 93.60 91.47 95.65 92.52 88.00 88.75 89.74 91.30 91.46 Proprietary MLLMs

Gemini 1.5 pro [18] 1 fps - 79.02 80.47 83.54 79.67 80.00 84.74 77.78 64.23 71.95 48.70 75.69 GPT-4o [41] 64 - 77.11 80.47 83.91 76.47 70.19 83.80 66.67 62.19 69.12 49.22 73.28 Claude 3.5 Sonnet [2] 20 - 73.33 80.47 84.09 82.02 75.39 79.53 61.11 61.79 69.32 43.09 72.44

###### Open-source Offline VideoLLMs

Video-LLaMA2-7B [9] 32 - 55.86 55.47 57.41 58.17 52.80 43.61 39.81 42.68 45.61 35.23 49.52 VILA-1.5-8B [34] 14 - 53.68 49.22 70.98 56.86 53.42 53.89 54.63 48.78 50.14 17.62 52.32 Video-CCAM-14B [16] 96 - 56.40 57.81 65.30 62.75 64.60 51.40 42.59 47.97 49.58 31.61 53.96 LongVA-7B [79] 128 - 70.03 63.28 61.20 70.92 62.73 59.50 61.11 53.66 54.67 34.72 59.96 InternVL-V2-8B [8] 16 - 68.12 60.94 69.40 77.12 67.70 62.93 59.26 53.25 54.96 56.48 63.72 Kangaroo-7B [36] 64 - 71.12 84.38 70.66 73.20 67.08 61.68 56.48 55.69 62.04 38.86 64.60 LLaVA-NeXT-Video-32B [35] 64 - 78.20 70.31 73.82 76.80 63.35 69.78 57.41 56.10 64.31 38.86 66.96 MiniCPM-V-2.6-8B [19] 32 - 71.93 71.09 77.92 75.82 64.60 65.73 70.37 56.10 62.32 53.37 67.44 LLaVA-OneVision-7B [24] 32 - 80.38 74.22 76.03 80.72 72.67 71.65 67.59 65.45 65.72 45.08 71.12 Qwen2.5-VL-7B [5] 1 fps 78.32 80.47 78.86 80.45 76.73 78.50 79.63 63.41 66.19 53.19 73.68

Open-source Online VideoLLMs Flash-VStream-7B [75] - - 25.89 43.57 24.91 23.87 27.33 13.08 18.52 25.20 23.87 48.70 23.23 VideoLLM-online-8B [6] 2 fps - 39.07 40.06 34.49 31.05 45.96 32.40 31.48 34.16 42.49 27.89 35.99 Dispider-7B [43] [CVPR 2025] 1 fps - 74.92 75.53 74.10 73.08 74.44 59.92 76.14 62.91 62.16 45.80 67.63 TimeChat-Online-7B 1 fps ↓44.2% 80.76 79.69 80.76 83.33 74.84 78.82 78.70 64.23 68.75 57.98 75.28 TimeChat-Online-7B 1 fps ↓82.6% 79.13 81.25 78.86 80.77 70.44 77.26 77.78 67.07 66.19 53.72 73.64 TimeChat-Online-7B 1 fps 100% 80.22 82.03 79.50 83.33 76.10 78.50 78.70 64.63 69.60 57.98 75.36

mask 𝑀drop𝑡 that indicates whether each token (or patch) should be dropped or preserved. For feature-level dropping, we directly

identify which visual tokens in V𝑡 to drop. For pixel-level dropping, we map the dropped patch indices to their corresponding visual tokens, leveraging the one-to-one correspondence maintained by the ViT encoder between input patches and output visual tokens. We

- then eliminate redundant static tokens from the current frame 𝑓𝑡 by

applying the binary mask 𝑀drop𝑡 to the visual tokens V𝑡 and related position embeddings P𝑡:

V𝑡 = V𝑡 ◦ 𝑀drop𝑡 , P𝑡 = P𝑡 ◦ 𝑀drop𝑡 (3)

Spatial-Temporal Position Reserving. After dropping the marked redundant tokens, the vanilla spatial and temporal position relations of the remaining tokens will be disrupted. This disruption would harm fine-grained spatial/temporal perception tasks such as Spatial Localization [25] or Action Recognition [76]. To preserve the relative spatial-temporal positions, we leverage Multi-modal Rotary Position Embedding (M-ROPE) derived from QwenVL-series models [5, 54]. M-ROPE indexes the 3D {temporal, height, width} position 𝑝 = {𝑡,ℎ,𝑤} of each video token. We calculate the 3D M-ROPE positions before dropping to record the original spatial-temporal structure of video tokens. When dropping visual tokens, we simultaneously drop their M-ROPE position embeddings as shown in the equation 3.

As depicted in Figure 2, the remaining visual tokens V𝑡 are aligned with their vanilla 3D position embeddings calculated before dropping. Finally, we can parallelize the position-aware token dropping operation for all temporal-consecutive frame pairs in F1:𝑡 and get the final dropped video token sequence V1:𝑡 across the timeline.

Discussion: Advantages of Differential Token Dropping. DTD is a purely vision-based approach to reduce video tokens by preserving significant temporal changes across video frames. Compared with existing token pruning methods [39, 48, 53, 61, 63], DTD offers three key advantages: (1) video-aware dynamic pruning that adaptively reduces video tokens from a holistic video perspective, well-suited for both high-speed and slow-motion videos; (2) positional reservation that maintains the fine-grained spatial-temporal positions of retained tokens; and (3) streaming-friendly operation that calculates visual redundancy only for newly incoming frames without re-processing historical video content. Moreover, DTD is orthogonal to memorybased [75] or KV-Cache Retrieval streaming methods [12, 44], serving as an initial efficient step to reduce the computational video burden and being complementary to these approaches.

### 3.3 TimeChat-Online-139K Collection

To better apply the DTD design, we introduce TimeChat-Online139K, a comprehensive synthetic streaming VideoQA dataset specifically designed for training online-VideoLLMs. Existing works [6,

- Table 2: Evaluation results on OVO-Bench [30] comprising three categories: i) Real-Time Visual Perception (OCR: Optical Character Recognition, ACR: Action Recognition, ATR: Attribute Recognition, STU: Spatial Understanding, FPD: Future Prediction, OJR: Object Recognition), ii) Backward Tracing (EPM: Episodic Memory, ASI: Action Sequence Identification, HLD: Hallucination Detection), and iii) Forward Active Responding (REC: Repetition Event Count, SSR: Sequential Steps Recognition, CRR: Clues Reveal Responding).

Real-Time Visual Perception Backward Tracing Forward Active Responding

Model #Frames

Overall OCR ACR ATR STU FPD OJR Avg. EPM ASI HLD Avg. REC SSR CRR Avg. Avg.

Human Agents - 94.0 92.6 94.8 92.7 91.1 94.0 93.2 92.6 93.0 91.4 92.3 95.5 89.7 93.6 92.9 92.8

###### Proprietary Multimodal Models

Gemini 1.5 Pro 1fps 87.3 67.0 80.2 54.5 68.3 67.4 70.8 68.6 75.7 52.7 62.3 35.5 74.2 61.7 57.2 65.3 GPT-4o 64 69.1 65.1 65.5 50.0 68.3 63.7 63.6 49.8 71.0 55.4 58.7 27.6 73.2 59.4 53.4 58.6

###### Open-source Offline VideoLLMs

LLaVA-NeXT-Video-7B 64 69.8 59.6 66.4 50.6 72.3 61.4 63.3 51.2 64.2 9.7 41.7 34.1 67.6 60.8 54.2 53.1 LLaVA-OneVision-7B 64 67.1 58.7 69.8 49.4 71.3 60.3 62.8 52.5 58.8 23.7 45.0 24.8 66.9 60.8 50.9 52.9 Qwen2-VL-7B 64 69.1 53.2 63.8 50.6 66.3 60.9 60.7 44.4 66.9 34.4 48.6 30.1 65.7 50.8 48.9 52.7 InternVL-V2-8B 64 68.5 58.7 69.0 44.9 67.3 56.0 60.7 43.1 61.5 27.4 44.0 25.8 57.6 52.9 45.4 50.1 LongVU-7B 1fps 55.7 49.5 59.5 48.3 68.3 63.0 57.4 43.1 66.2 9.1 39.5 16.6 69.0 60.0 48.5 48.5

###### Open-source Online Video-LLMs

Flash-VStream-7B 1fps 25.5 32.1 29.3 33.7 29.7 28.8 29.9 36.4 33.8 5.9 25.4 5.4 67.3 60.0 44.2 33.2 VideoLLM-online-8B 2fps 8.1 23.9 12.1 14.0 45.5 21.2 20.8 22.2 18.8 12.2 17.7 - - - - -

TimeChat-Online-7B 1fps (↓44.6%) 74.5 48.6 68.1 48.3 69.3 59.8 61.4 56.9 64.9 11.8 44.5 31.8 38.5 40.0 36.8 47.6 (+14.4) TimeChat-Online-7B 1fps (↓84.8%) 69.8 48.6 64.7 44.9 68.3 55.4 58.6 53.9 62.8 9.1 42.0 32.5 36.5 40.0 36.4 45.6 (+12.4)

TimeChat-Online-7B 1fps (100%) 75.2 46.8 70.7 47.8 69.3 61.4 61.9 55.9 59.5 9.7 41.7 31.6 38.5 40.0 36.7 46.7

21, 46] convert dense video narrations [6] or timestamp-related tasks [37] into streaming dialogue datasets. However, these transformed data samples are limited in question-answer diversity and fail

- to mimic real-world interactions. Instead, our dataset encompasses diverse online tasks across backward tracing, real-time perception, and forward active responding to facilitate more flexible Real-time Interaction. Specifically, we collect long-range and visually informative videos, annotate scene-oriented dense captions, and produce diverse streaming question-answer samples using GPT-4o [41].

- Step 1: Visually Informative Video Collection. The quality of video content is crucial for the Streaming VideoQA task. Videos with monotonous or static scenes provide limited information for multiple-turn real-time interactions. To address this, we collect visually informative videos characterized by diverse scene changes. First, we source long-form videos [46] and segment them into successive scenes using PySceneDetect1. We then extract key frames from each scene segment and eliminate redundant frames based on visual similarity measured by DINO-v2 [42]. To ensure sufficient visual diversity and information richness, we select videos containing more than 5 distinct scenes.
- Step 2: Scene-oriented Detailed Caption Generation. Following the collection of videos and extraction of key frames, we employ GPT-4o [41] to generate scene-oriented dense captions for each frame. These comprehensive descriptions encompass both static and dynamic visual elements, including shot types, object appearances and actions, environmental and background variations, and camera movements. To enhance the model’s ability to discern scene transitions, we provide preceding frame caption as contextual information.
- Step 3: Streaming VideoQA Generation. We generate streaming VideoQA samples by providing the dense captions of the current scene and the previous scenes as context. We utilize GPT-4o to generate diverse question-answer pairs, including Temporal-enhanced,

1https://www.scenedetect.com/

Backward Tracing, Real-Time Visual Perception, and Forward Active Responding QAs following previous works[30, 33].

Step 4: Negative Samples for Future-Response Training. To facilitate the training of proactive responding capabilities, we construct negative samples that intentionally cannot be answered using the currently available streaming video content. We carefully select irrelevant video frames before the question timestamp and generate corresponding answers labeled as “unanswerable”.

Data Statistics. We collect 11,043 visually informative videos ranging from 5 minutes to several hours in length, with an average duration of 11.1 minutes per video. From each video, we extract an average of 87.8 key frames with approximately 7.14 seconds between consecutive frames. Each key frame is annotated with a detailed description averaging 176 words. Based on the dense descriptions, we generate 139K question-answer pairs. Details regarding data statistics, task types, and GPT-4o prompts are provided in the Appendix.

### 3.4 Training and Real-Time Inference

We implement TimeChat-Online based on the long-context Qwen2.5VL [5] architecture to support high-FPS dense frame processing. We incorporate DTD to efficiently reduce visual redundancy in video streams. Our model is trained on a combination of our streaming dataset TimeChat-Online-139K and offline video understanding datasets (LLaVA-Video-178K [80], Tarsier2 [70] and VideoChatFlash [29]) to ensure robust performance. During training, we densely sample frames at 1 FPS to simulate streaming video input with a maximum sequence length of N frames. For DTD, we apply token dropping with 50% probability to each training batch.

During inference, TimeChat-Online processes video frames at 1 FPS and uses DTD to drop approximately 85% of video tokens. A first-in-first-out memory bank stores the slimmed historical video tokens. It is worth noting that this memory design is not the emphasis of this paper, and can be replaced with alternatives [12, 75].

- Table 3: Results on offline long video benchmarks. We report the accuracy on the MLVU [82], LongVideoBench [59] and VideoMME(w/o subtitles) [17]. † indicates the reproduced results.

Model #Frames MLVU LongVideoBench

VideoMME

overall long

Video Length - 3∼120 min 8 sec∼60 min 1∼60 min 30∼60 min Open-Source Offline VideoLLMs LLaMA-VID-7B 1fps 33.2 - - MovieChat-7B 2048 25.8 - 38.2 33.4 LLaVA-Next-Video-7B 32 - 43.5 46.6 VideoChat2-7B 16 47.9 39.3 39.5 33.2 LongVA-7B 128 56.3 - 52.6 46.2 Kangaroo-7B 64 61.0 54.2 56.0 46.6 Video-CCAM-14B 96 63.1 - 53.2 46.7 VideoXL-7B 128 64.9 - 55.5 49.2 Qwen2.5-VL-7B† 1fps (100%) 66.9 61.5 63.2 50.4 Qwen2.5-VL-7B w/ DTD 1fps (↓46.2%) 68.6 61.6 63.4 51.9 Qwen2.5-VL-7B w/ DTD 1fps (↓84.6%) 68.8 59.3 64.9 56.1

Open-source Online VideoLLMs Dispider-7B [CVPR 2025] 1fps 61.7 - 57.2 VideoChat-Online-8B [CVPR 2025] 2fps - - 52.8 44.9 TimeChat-Online-7B 1fps (100%) 62.6 55.4 62.4 48.4 TimeChat-Online-7B 1fps (↓46.3%) 62.9 57.1 63.3 52.4 TimeChat-Online-7B 1fps (↓85.0%) 65.4 57.7 62.5 49.2

Table 4: Ablation study of different token dropping strategies on StreamingBench (Real-Time Visual Understanding). For fair comparison, all methods are evaluated with Qwen2.5VL-7B (Vanilla) without supervised fine-tuning (SFT). Δ represents accuracy retention rate (%) relative to the full token setting.

Method w/o SFT

StreamingBench

(3s∼24min) Δ Vanilla, Avg 22K Video Tokens per Video (1 fps) Qwen2.5VL-7B 73.7 100% Drop 44.1%, Avg 12K Video Tokens per Video

+ VisionZip [CVPR 2025] 72.3 98.1% + Pixel-level drop 72.8 98.8% + Feature-level drop (frame-aware) 73.0 99.1% + Feature-level drop (video-aware) 73.4 99.6%

Drop 82.5%, Avg 4K Tokens per Video

+ VisionZip [CVPR 2025] 68.8 93.4% + Pixel-level drop 68.8 93.4% + Feature-level drop (frame-aware) 72.0 97.7% + Feature-level drop (video-aware) 72.0 97.7%

Proactive Response with Trigger Time. For Future-responding questions, the model must decide the optimal timestamps to generate a proactive response. We define these decision time points as Trigger times, which always occur when the video transitions to a new scene [43]. These Trigger times can be effectively monitored through the token drop ratio curve along the timeline, as illustrated in Figure 2 (bottom). The valleys in this curve indicate moments when the current frame content changes significantly from previous frames, revealing a scene transition. We implement proactive response by generating responses at each trigger time using the most recent video content. Through training with the negative samples in TimeChat-Online139K (Section 3.3), the model learns to generate an “unanswerable” responsewhentheavailablevideocontentisinsufficientataparticular trigger time, effectively instructing the system to wait for the next trigger time to respond again.

- 4 Experiments

pixel-level dropping as shown in Figure 1. We set 𝜏𝑓 𝑒𝑎𝑡 = 0.25/0.5 corresponding to around 85% / 45% token dropping rate by default. For real-time interaction, the most recent 6K slimmed video tokens after dropping are preserved in a first-in-first-out memory bank, introducing a maximum latency of 2 seconds for responding.

### 4.2 Results on Streaming Video Benchmarks

Wefirstevaluateourmodel’sperformanceontwostreamingVideoQA benchmarks: StreamingBench [33] and OVO-Bench [30]. For these evaluations, VideoLLMs processes the historical video content received before the Current timestamp, which represents the moment when a user question is posed.

StreamingBench. Table 1 shows that TimeChat-Online achieves state-of-the-art performance with a score of 75.28 on the Real-time VisualUnderstandingsubtaskofStreamingBench,representinga7.65 improvement over the recent online model Dispider-7B [43] scored 67.63. This demonstrates that TimeChat-Online effectively combines the superior VideoQA capabilities of offline VideoLLMs with the real-time streaming inference capabilities of online VideoLLMs. Moreover, this 75.28 score surpasses both the best offline VideoLLM Qwen2.5VL-7B (73.68) and proprietary models including GPT-4o (73.28) and Claude-3.5-Sonnet (72.44).

### 4.1 Implementation Details

We implement our model using the Qwen2.5VL 7B architecture. For training, we sample frames at 1 FPS with a maximum sequence length of 64 frames for streaming VideoQA samples. We configure the model with a maximum input resolution of 448×448 pixels, threshold parameters 𝜏𝑓 𝑒𝑎𝑡 = 0.7, batch size 128, and learning rate 1e-5. Our training procedure combines both offline and online datasets, including subsets of LLaVA-Video-178K (100K samples), Tarsier2 (100K samples), VideoChat-Flash (3K samples) and TimeChat-Online-139K, among which negative samples with the “unanswerable” label are 20K. We keep the vision encoder frozen while fine-tuning the full-parameter projector and language model for 1 epoch. All experiments are performed on 8×A800 80G GPUs. The hard drop ratio to determine Trigger Time is 60%. During inference, we maintain the 1 FPS frame processing rate while extending the maximum frame length to 1016. We use feature-level dropping for all experiments by default since it performs consistently better than

When compared to Qwen2.5VL-7B with 1fps full token inputting, TimeChat-Online achieves superior performance (75.28 vs. 73.68) while requiring 44.2% fewer video tokens. Even at an extreme token dropping ratio of 82.8%, TimeChat-Online maintains comparable results to the full token setting of Qwen2.5VL-7B (73.64 vs. 73.68). These findings highlight the substantial redundancy in 1 fps video streams and the effectiveness of our DTD approach.

OVO-Bench. Table 2 presents the results on OVO-Bench, which comprehensively evaluates Backward Tracing and Forward Active Responding capabilities across 12 diverse subtasks. TimeChat-Online substantially outperforms existing online VideoLLMs, achieving a

Table 5: Impact of training datasets on StreamingBench (RealTime Visual Understanding) performance across different token drop ratios. Ours refers to our proposed dataset TimeChatOnline-139K.

Token Drop Ratio ↓44% ↓82.6%

Dataset

Qwen2.5-VL-7B (Vanilla) 73.4 72.0 + LLaVA-Video100K 73.4 72.3 + LLaVA-Video100K + Tarsier2129K 73.5 72.5 + LLaVA-Video100K + Tarsier2129K + VideoChat-Flash3K 74.0 72.8 + Above Offline Datasets + Ours 75.3 73.6

final score of 47.6, which represents a significant 14.4-point absolute improvement over Flash-VStream [75] and VideoLLM-online [6]. Notably, with 84.8% of video tokens dropped, TimeChat-Online maintains robust performance with a score of 45.6.

Case Study. Figure 4 visualizes a proactive responding case of TimeChat-Online. When a user proposes a question “What specifically did the woman in red do?” that can also be answered by future moments, TimeChat-Online proactively generates responses at future trigger times (i.e., the video scene transition timestamps), which are indicated by frames with low token drop ratios.

### 4.3 Results on Offline Long Video Tasks

We also evaluate TimeChat-Online on three offline long-form video understanding benchmarks: VideoMME [17], MLVU [82], and LongVideoBench [59]. In the offline setting, the entire video is provided as input to the VideoLLMs. Table 3 demonstrates that TimeChat-Online exhibits superior offline video understanding capabilities compared to recent state-of-the-art online VideoLLMs, including VideoChat-Online [21] and Dispider-7B. Leveraging the efficiency of DTD, TimeChat-Online performs particularly well on extremely long-form videos, such as the long subset of VideoMME. Compared to VideoChat-Online, TimeChat-Online achieves a 7.5point improvement (from 44.9 to 52.4) on the long subset of VideoMME which contains videos ranging from 30 to 60 minutes in length. We also report Qwen2.5-VL-7B w/ DTD zeroshot results, as DTD can be integrated directly without additional training. Surprisingly, increasing token drop ratio from 46.2% to 84.6% consistently improves Qwen2.5-VL-7B’s performance on MLVU and VideoMME, making it superior to the 100% full token setting. For VideoMME’s long subset (30-60min), accuracy rises from 50.4 to 56.1 using only 15.4% of retained video tokens. This indicates substantial vision redundancy in long videos, and reducing this redundancy can simplify VideoLLM’s vision perception with shorter context, thereby enhancing overall performance.

### 4.4 Ablation Study

Effectiveness of DTD Design. We compare different token dropping methods in Table 4. VisionZip [68] represents a similar purely-vision spatial token selection method, while pixel-level and feature-level dropping approaches are introduced in this paper (Section 3.2.2). Zero-shot results with Qwen2.5-VL-7B demonstrate that Featurelevel (video-aware) Dropping, i.e., the final design of DTD, achieves the best performance under the same dropping ratio. The frame-aware

80

0%

75

70

78.5%

| |
|---|

| |
|---|

| |
|---|

65

###### AccuracyonVideoMME

60

87.2%

55

50

45

97.5%

40

35

30

Short videos (<2min)

Short baseline w/o drop

Medium videos (4~15min)

Medium baseline w/o drop

25

Long videos (30~60min)

Long baseline w/o drop

20

0 10 20 30 40 50 60 70 80 90 100

Video Token Drop Ratio (%)

###### Figure3:VideoredundancyofdifferentvideolengthonVideoMME[17].

dropping approach applies a fixed dropping ratio to each frame, in contrast to the video-aware approach that dynamically selects tokens across the entire video. These results reveal that joint spatio-temporal dynamic token pruning proves most effective.

Effectiveness of TimeChat-Online-139K dataset. Table 5 demonstrates the impact of our training strategy that combines online and offline datasets. The results clearly show that integrating TimeChatOnline-139K with existing offline VideoQA datasets significantly enhances streaming performance.

### 5 Analysis of DTD

Performance-Drop Ratio Tradeoff. As shown in Figure 1(right top), feature-level dropping consistently outperforms pixel-level dropping. Meanwhile, training with DTD can make VideoLLMs better adapt to dropped token distribution and perform better on extreme dropping ratio such as 80%+ token dropping. Overall, the DTD approach maintains robust performance even with substantial token reduction. At an extreme 82.8% drop ratio, the model achieves comparable results to the full token setting (73.64 vs. 73.70).

Redundancy Across Different Video Lengths. Figure 3 demonstrates that longer videos contain higher redundancy, permitting more aggressive token dropping. While short videos show obvious performance drops at high dropping rates, long videos (30-60 minutes) maintain performance integrity even with extreme dropping rates up to 97.5%.

Efficiency Improvements. Figure 1(right down) quantifies the computational benefits of token dropping. At 81.1% drop ratio, inference latency decreases from 3220ms to 1820ms (1.76× faster) while accuracy improves from 50.4% to 56.3%. This suggests that focusing on fewer, more informative tokens not only accelerates inference but can enhance model performance by reducing visual noise.

### 6 Conclusion

This paper introduces TimeChat-Online, a novel online VideoLLM that incorporates Differential Token Dropping, a simple yet efficient approach addressing long-form high-redundancy challenges in

###### Video Streams

Time (seconds)

113s 115s

114s

###### 116s 117s

118s

119s 120s

|[Figure 27]<br><br>[Figure 28]|
|---|

|[Figure 29]<br><br>[Figure 30]|
|---|

|[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|
|---|

|[Figure 36]<br><br>[Figure 37]|
|---|

|[Figure 38]<br><br>[Figure 39]|
|---|

|[Figure 40]<br><br>[Figure 41]|
|---|

|[Figure 42]<br><br>[Figure 43]|
|---|

|[Figure 44]|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

【116 second】

###### [116 second] What specifically did the woman in red do?

[Figure 52]

[Figure 53]

She adjusted the white blouse of the woman in black.

161s

162s

163s

164s 165s 166s

179s 180s

|[Figure 54]<br><br>[Figure 55]|
|---|

|[Figure 56]<br><br>[Figure 57]|
|---|

|[Figure 58]<br><br>[Figure 59]|
|---|

|[Figure 60]<br><br>[Figure 61]|
|---|

|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]|
|---|
|[Figure 65]|

|[Figure 66]<br><br>[Figure 67]|
|---|

|[Figure 68]<br><br>[Figure 69]|
|---|

|[Figure 70]<br><br>[Figure 71]|
|---|

… …

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

【165 second】

[Figure 79]

[Figure 80]

She adjusted the black corset and helped put on a yellow satin robe.

194s 195s

196s

202s

204s

228s 229s 230s

|[Figure 81]<br><br>[Figure 82]|
|---|

|[Figure 83]<br><br>[Figure 84]|
|---|

|[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|
|---|

|[Figure 88]<br><br>[Figure 89]|
|---|

|[Figure 90]<br><br>[Figure 91]|
|---|

|[Figure 92]<br><br>[Figure 93]|
|---|

|[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]|
|---|

###### …

… …

|[Figure 99]|
|---|

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

【196 second】

###### 【230 second】

[Figure 107]

[Figure 108]

[Figure 109]

She adjusted the sleeve of another woman's dress and then held up a dark garment.

[Figure 110]

She adjusted the black dress and styled the hair of another woman.

[Figure 111]

[Figure 112]

Past Current Future User Query Time Trigger Time w/ Scene Change

- Figure 4: Case study of TimeChat-Online on StreamingBench. When a user proposes a question “What specifically did the woman in red do?” that can also be answered by the future moments, TimeChat-Online will proactively generate responses at the future trigger time (i.e., the video scene transition timestamps), which are indicated by the frames with low token drop ratios.

### References

streaming video understanding. Comprehensive experiments demonstrate that TimeChat-Online achieves state-of-the-art performance on streaming video benchmarks while eliminating up to 82.8% of visual tokens. Additionally, our analysis reveals that reducing temporal redundancy is particularly critical for hours-long videos, which can eliminate over 95% of tokens via the DTD strategy without performance degradation. For real-time interaction in streaming scenarios, we also propose a synthetic TimeChat-Online-139K to endow TimeChat-Online with backward-tracing, real-time perception, and proactive future-responding capabilities. This work not only highlights the substantial redundancy in video streams but also establishes a promising direction for computationally efficient video encoding in streaming VideoQA tasks.

- [1] Charu C Aggarwal, Alexander Hinneburg, and Daniel A Keim. 2001. On the surprising behavior of distance metrics in high dimensional space. In International conference on database theory. Springer, 420–434.
- [2] Anthropic. 2024. Claude 3.5 Sonnet. https://www.anthropic.com/news/claude-35-sonnet
- [3] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. 2024. MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens. ArXiv preprint abs/2404.03413 (2024).
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shĳie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. ArXiv preprint abs/2308.12966

(2023).

- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shĳie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025).

- [6] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. 2024. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18407–18418.
- [7] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models.

- arXiv:2403.06764 [cs.CV]

[8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. 2024. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites.

- arXiv:2404.16821 [cs.CV] https://arxiv.org/abs/2404.16821

- [9] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476 (2024).
- [10] Rohan Choudhury, Guanglei Zhu, Sihan Liu, Koichiro Niinuma, Kris Kitani, and László Jeni. 2024. Don’t Look Twice: Faster Video Transformers with Run-Length Tokenization. Advances in Neural Information Processing Systems 37 (2024), 28127–28149.
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.).
- [12] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Tao Zhong, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. 2025. Streaming video question-answering with in-context video kv-cache retrieval. arXiv preprint arXiv:2503.00540 (2025).
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly, et al. 2020. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations.
- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024. The Llama 3 Herd of Models. ArXiv preprint abs/2407.21783 (2024).
- [15] Bernard Ghanem Fabian Caba Heilbron, Victor Escorcia and Juan Carlos Niebles.

2015. ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 961–970.

- [16] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. 2024. Video-ccam: Enhancing video-language understanding with causal cross-attention masks for short and long videos. arXiv preprint arXiv:2408.14023 (2024).
- [17] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2024. Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. ArXiv preprint abs/2405.21075 (2024).
- [18] Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv preprint abs/2403.05530 (2024).
- [19] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395 (2024).

- [20] Gabriel Huang, Bo Pang, Zhenhai Zhu, Clara Rivera, and Radu Soricut. 2020. Multimodal Pretraining for Dense Video Captioning. arXiv:2011.11760 [cs.CV] https://arxiv.org/abs/2011.11760
- [21] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. 2024. Online video understanding: A comprehensive benchmark and memory-augmented method. arXiv preprint arXiv:2501.00584 (2024).
- [22] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. 2024. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13700–13710.
- [23] Jie Lei, Tamara L. Berg, and Mohit Bansal. 2021. QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries. arXiv:2107.09609 [cs.CV] https://arxiv.org/abs/2107.09609
- [24] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-OneVision: Easy Visual Task Transfer. ArXiv preprint abs/2408.03326 (2024).
- [25] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. 2025. LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding. arXiv preprint arXiv:2501.08282

(2025).

- [26] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023. VideoChat: Chat-Centric Video Understanding. ArXiv preprint abs/2305.06355 (2023).
- [27] Lei Li, Yuanxin Liu, Linli Yao, Peiyuan Zhang, Chenxin An, Lean Wang, Xu Sun, LingpengKong,andQiLiu.2025. TemporalReasoningTransferfromTexttoVideo. In ICLR 2025. OpenReview.net. https://openreview.net/forum?id=sHAvMp5J4R
- [28] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. 2024. Videochatflash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574 (2024).
- [29] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. 2024. Videochatflash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574 (2024).
- [30] Yifei Li, Junbo Niu, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, and Jiaqi Wang. 2025. OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? arXiv:2501.05510 [cs.CV] https://arxiv.org/abs/2501.05510
- [31] Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision. Springer, 323–340.
- [32] Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision. Springer, 323–340.
- [33] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. 2024. StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding. arXiv preprint arXiv:2411.03628

(2024).

- [34] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26689–26699.
- [35] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.
- [36] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. 2024. Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input. arXiv preprint arXiv:2408.15542

(2024).

- [37] Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen.

2024. Et bench: Towards open-ended event-level video-language understanding. Advances in Neural Information Processing Systems 37 (2024), 32076–32110.

- [38] Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, and Hongtao Xie. 2025. Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models. arXiv preprint arXiv:2503.16036 (2025).
- [39] Zhenyu Ning, Jieru Zhao, Qihao Jin, Wenchao Ding, and Minyi Guo. 2024. InfMLLM: Efficient streaming inference of multimodal large language models on a single GPU. arXiv preprint arXiv:2409.09086 (2024).
- [40] Andreea-Maria Oncescu, Joao F Henriques, Yang Liu, Andrew Zisserman, and Samuel Albanie. 2021. Queryd: A video dataset with high-quality text and audio narrations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2265–2269.
- [41] OpenAI. 2024. GPT-4o System Card.
- [42] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision.

- arXiv preprint arXiv:2304.07193 (2023).
- [43] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. 2025. Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction. arXiv preprint arXiv:2501.03218 (2025).
- [44] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. 2024. Streaming long video understanding with large language models. Advances in Neural Information Processing Systems 37 (2024), 119336–119360.
- [45] Shuhuai Ren, Sishuo Chen, Shicheng Li, Xu Sun, and Lu Hou. 2023. TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding. ArXiv abs/2310.19060 (2023).
- [46] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14313–14323.
- [47] Ronald A. Rensink. 2002. Change detection. Annual Review of Psychology 53

(2002), 245–277.

- [48] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. 2024. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434 (2024).
- [49] Daniel J Simons and Ronald A Rensink. 2005. Change blindness: Past, present, and future. Trends in cognitive sciences 9, 1 (2005), 16–20.
- [50] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. 2015. TVSum: Summarizingwebvideosusingtitles.In2015IEEEConferenceonComputerVision and Pattern Recognition (CVPR). 5179–5187. doi:10.1109/CVPR.2015.7299154
- [51] Harald Steck, Chaitanya Ekanadham, and Nathan Kallus. 2024. Is cosine-similarity of embeddings really about similarity?. In Companion Proceedings of the ACM Web Conference 2024. 887–890.
- [52] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. 2019. COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis. arXiv:1903.02874 [cs.CV] https://arxiv.org/abs/ 1903.02874
- [53] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2024. DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models. arXiv preprint arXiv:2411.15024 (2024).
- [54] Peng Wang, Shuai Bai, Sinan Tan, Shĳie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191 (2024).
- [55] Weiying Wang, Jieting Chen, and Qin Jin. 2020. VideoIC: A Video Interactive Comments Dataset and Multimodal Multitask Learning for Comments Generation. In Proceedings of the 28th ACM International Conference on Multimedia (Seattle, WA, USA) (MM ’20). Association for Computing Machinery, New York, NY, USA, 2599–2607. doi:10.1145/3394171.3413890
- [56] Weiying Wang, Yongcheng Wang, Shizhe Chen, and Qin Jin. 2019. YouMakeup: A Large-Scale Domain-Specific Multimodal Dataset for Fine-Grained Semantic Comprehension. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-ĲCNLP). 5136–5146.
- [57] Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Jiansheng Wei, Huishuai Zhang, and Dongyan Zhao. 2024. VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format. arXiv:2411.17991 [cs.CV] https://arxiv.org/abs/2411.17991
- [58] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weĳia Li, Conghui He, and Linfeng Zhang. 2025. Stop Looking for Important Tokens in Multimodal Language Models: Duplication Matters More. arXiv preprint arXiv:2502.11494 (2025).
- [59] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding. ArXiv preprint abs/2407.15754 (2024).
- [60] Shiwei Wu, Joya Chen, Kevin Qinghong Lin, Qimeng Wang, Yan Gao, Qianli Xu, Tong Xu, Yao Hu, Enhong Chen, and Mike Zheng Shou. 2024. Videollm-mod: Efficient video-language streaming with mixture-of-depths vision computation. Advances in Neural Information Processing Systems 37 (2024), 109922–109947.
- [61] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. 2025. Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge. arXiv preprint arXiv:2501.13468

(2025).

- [62] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. 2025. Qwen2. 5-Omni Technical Report. arXiv preprint arXiv:2503.20215 (2025).
- [63] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. 2024. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841 (2024).
- [64] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. 2022. Advancing High-Resolution Video-Language

- Representation with Large-Scale Video Transcriptions. arXiv:2111.10337 [cs.CV] https://arxiv.org/abs/2111.10337
- [65] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shĳie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. 2024. Qwen2 Technical Report. ArXiv preprint abs/2407.10671 (2024).
- [66] Chenyu Yang, Xuan Dong, Xizhou Zhu, Weĳie Su, Jiahao Wang, Hao Tian, Zhe Chen, Wenhai Wang, Lewei Lu, and Jifeng Dai. 2024. PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models. arXiv preprint arXiv:2412.09613 (2024).
- [67] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. 2025. TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model. arXiv preprint arXiv:2503.18278 (2025).
- [68] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2024. Visionzip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467 (2024).
- [69] Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. 2024. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985 (2024).
- [70] Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. 2025. Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding. arXiv preprint arXiv:2501.07888

(2025).

- [71] Zihao Yue, Qi Zhang, Anwen Hu, Liang Zhang, Ziheng Wang, and Qin Jin. 2023. Movie101: A New Movie Understanding Benchmark. arXiv:2305.12140 [cs.CV] https://arxiv.org/abs/2305.12140
- [72] Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oğuz, Yashar Mehdad, and Mohit Bansal. 2023. Hierarchical Video-Moment Retrieval and Step-Captioning. In CVPR.
- [73] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. 2025. VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding. arXiv preprint arXiv:2501.13106 (2025).
- [74] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Yansong Feng and Els Lefever (Eds.). 543–553. doi:10.18653/ v1/2023.emnlp-demo.49
- [75] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. 2024. Flash-vstream: Memory-based real-time understanding for long video streams. arXiv preprint arXiv:2406.08085 (2024).
- [76] Jianrui Zhang, Mu Cai, and Yong Jae Lee. 2024. Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos. arXiv preprint arXiv:2410.02763 (2024).
- [77] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, et al. 2024. Internlm-xcomposer2. 5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions. arXiv preprint arXiv:2412.09596 (2024).
- [78] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, Qipeng Guo, Haodong Duan, Xin Chen, Han Lv, Zheng Nie, Min Zhang, Bin Wang, Wenwei Zhang, Xinyue Zhang, Jiaye Ge, Wei Li, Jingwen Li, Zhongying Tu, Conghui He, Xingcheng Zhang, Kai Chen, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. InternLM-XComposer2.5OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions. arXiv preprint arXiv:2412.09596 (2024).
- [79] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024. Long Context Transfer from Language to Vision. ArXiv preprint abs/2406.16852 (2024).
- [80] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li.2024. VideoInstructionTuningWithSyntheticData. arXiv:2410.02713[cs.CV] https://arxiv.org/abs/2410.02713
- [81] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems 36 (2023), 34661–34710.
- [82] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding. ArXiv preprint abs/2406.04264

(2024).

[83] Luowei Zhou, Chenliang Xu, and Jason J Corso. 2018. Towards Automatic Learning of Procedures From Web Instructional Videos. In AAAI Conference on Artificial Intelligence. 7590–7598. https://www.aaai.org/ocs/index.php/AAAI/ AAAI18/paper/view/17344

### A Appendix

In the appendix, we provide more results and analysis and summarize them as follows:

- • In Section A.1, we report extensive experimental results.
- • In Section A.2, we present the training hyperparameter tables.
- • In Section A.3, we present details of TimeChat-Online-139K.
- • In Section A.4, we introduce diverse visualization cases for the Differential Token Drop design.

### A.1 More Experimental Results

StreamingBench Full-set Results. Table 11 presents the comprehensive results across all three categories of StreamingBench. TimeChatOnline achieves state-of-the-art performance among open-source models with an overall score of 58.11, outperforming the recent online model Dispider-7B [43] by 4.99 points (58.11 vs. 53.12). While proprietary model Gemini 1.5 pro leads with 67.07, TimeChatOnline surpasses Claude-3.5-Sonnet (57.68) and all open-source offline VideoLLMs. For Omni-Source and Contextual Understanding, TimeChat-Online scores 37.80 and 35.30 respectively, consistently surpassing Dispider-7B.

Most importantly, with 82.6% token reduction, TimeChat-Online maintains 97.3% of its original performance (56.56 vs. 58.11), confirming that most visual tokens in streaming videos are redundant whileourapproachpreservesessentialinformationforcomprehensive understanding.

###### Selection of Hyperparameters 𝜏. The values of 𝜏𝑝𝑖𝑥𝑒𝑙 and 𝜏𝑓 𝑒𝑎𝑡

control the overall drop ratio of video tokens by dynamically measuring the visual difference between temporally consecutive frames. This approach identifies redundancy based solely on the video’s inherent temporal characteristics, without requiring language guidance. Remarkably, we observe that the relationship between 𝜏 values and drop ratios remains consistent across diverse datasets, as shown in Table 6. For example, 𝜏𝑓 𝑒𝑎𝑡 = 0.25 consistently yields approximately 85% token reduction across StreamingBench, OVOBench, LongVideoBench, and MLVU, while 𝜏𝑓 𝑒𝑎𝑡 = 0.5 results in approximately 45%-55% reduction. More 𝜏-dropratio correspondence can also be found in Figure 1 (top right) in the main paper.

This consistency demonstrates that visual redundancy is an intrinsic property of videos regardless of content type or task domain. Based on our comprehensive experiments, we conclude that over 80% of visual information in long-form videos is naturally redundant. Therefore, we recommend using 𝜏𝑓 𝑒𝑎𝑡 = 0.25 as the optimal setting for video token dropping, as it maintains high performance while significantly reducing computational requirements.

Breakdown Analysis of Fine-grained subtasks on MLVU. Table 8 showstheperformancebreakdownonfine-grainedsubtasksofMLVU. Remarkably, even with 89.5% token reduction, our model not only maintains but improves performance across multiple challenging fine-grained visual understanding tasks compared to the full token baseline. The needle-in-haystack task (NQA) improves from 80.0 to 82.0, plot reading (PQA) from 66.4 to 68.8, temporal ordering (AO)

Table 6: Consistency of drop ratios (%) across different datasets with the same 𝜏𝑓 𝑒𝑎𝑡 values, demonstrating that our method captures intrinsic video redundancy independent of dataset characteristics.

Drop Ratio (%)

𝜏𝑓 𝑒𝑎𝑡 StreamingBench OVOBench LongVideoBench MLVU

0.5 44.1% 44.6% 56.5% 46.3% 0.25 82.6% 84.8% 85.1% 84.7%

Hyper-parameter Value Visual Encoder

Frame Sampling Rate FPS=1.0 Input Resolution 448*448 Visual Tokens per Image 128 Max Image per Sequence 64 Patch Size 14x14

Large Language Model

Number of Layers 80 Hidden Size 8192 Vocabularyy Size 152064 Number of Attention Heads 64 Number of KV Heads 64 Number of KV Heads 8

Model Training

Max Context Length 11264 Batch Size 128 Learning Rate 1e-5 Warmup Ratio 0.05 Training epoch1 1 LR Scheduler Type Cosine

###### Table 7: Training hyper-parameters for TimeChat-Online.

from 40.9 to 49.4, and counting (AC) from 34.0 to 42.2, resulting in an overall improvement from 62.0 to 64.1.

These results demonstrate that DTD effectively preserves essential visual details while discarding redundant information. Despite dropping nearly 90% of tokens, the model maintains the capability to recognize small objects (needle task), interpret plots, track temporal relationships, and count objects accurately. This is enabled by our spatial-temporal aware design, which preserves the original video’s spatiotemporal structure even after significant token reduction. The improved performance on the counting task (AC) is particularly noteworthy, showing an 8.2 point gain, which confirms that our approach selectively retains detailed visual information critical for complex analysis tasks.

### A.2 Training Hyperparameter

We provide detailed hyper-parameters in Table 7.

- Table 8: Breakdown analysis on fine-grained subtasks of MLVU. Our approach with feature-level token dropping (89.5% reduction) improves performance across all subtasks. ↓ indicates lower than human performance, while ↑ indicates comparable or better than human performance.

Drop Ratio (%) NeedleQA PlotQA) Action Order Action Counting M-Avg

No drop 100% 80.0 66.4 40.9 34.0 62.0 𝜏𝑓 𝑒𝑎𝑡 = 0.5 46.4% 80.8 67.7 44.0 32.0 61.9 𝜏𝑓 𝑒𝑎𝑡 = 0.25 89.5% 82.0 68.8 49.4 42.2 64.1 Δ - +2.0 +2.4 +8.5 +8.2 +2.1

3000

| |(21.1%) 2331<br><br>(15.0%) 1655<br><br>(21.9%) 2412<br><br>(8.8%) 966<br><br>(5.4%) 592<br><br>(16.4%) 1811<br><br>(4.9%) 536<br><br>(3.6%) 394<br><br>(0.9%) 101<br><br>(2.0%) 225<br><br>6min<br><br>Other durations| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

2500

2000

NumberofVideos

1500

1000

500

0

6min >6min >7min >8min >9min>10min>15min>20min>30min>60min

Video Duration

- Figure 5: Distribution of video durations across the 11,043 videos in our dataset. The minimum video length in our dataset is 5 minutes.

with diverse scenes have less redundancy, while monotonous videos with static scenes display high redundancy.

- • Figure11demonstrateshowdifferenthyperparameter𝜏 values lead to different levels of redundancy control.
- • Figure 8 and Figure 9 illustrate the Trigger Time naturally monitored in the drop ratio-timeline curve, where valleys of low drop ratio reveal video scene transitions. TimeChatOnline leverages this scene change detection without requiring additional perception modules as in Dispider [43], thereby reducing response delay.
- • Figure 10 shows a case study of TimeChat-Online on StreamingBench with specific drop ratio curve. When a user proposes a question “What specifically did the woman in red do?” that canalsobeansweredbythefuturemoments,TimeChat-Online will proactively generate responses at the future trigger time (i.e., the video scene transition timestamps), which are indicated by the frames (165 seconds, 196 seconds and 230 seconds) with low token drop ratios.
- • Figure 12 shows the average drop ratio across different video durations. Since temporal redundancy is inherent to a video, it has little correlation with video length. The analysis reveals that spatial-related tasks may have more visual redundancy, while temporal-reasoning tasks typically have less visual redundancy as they require more informative visual content for reasoning and response.
- • Figure 13 presents the average drop ratio across different video subtask types.

### A.3 Dataset Statistics

Our dataset comprises 11,043 videos sampled from 12 publicly available source datasets, with an average duration of 11.1 minutes per video. The composition of the video sources and the distribution of video durations are shown in Table 9 and Figure 5, respectively. Based on these videos, we generate a total of 139K question-answer (QA) pairs, following the procedure outlined in main Section 3.3. These QA pairs are categorized into four high-level types—Temporalenhanced, Backward Tracing, Real-Time Visual Perception, and Forward Active Responding—encompassing eleven fine-grained subcategories. Representative prompt examples for each subtask are provided in Table 10, while the prompt templates for Step 2: Sceneoriented Detailed Caption Generation and Step 3: Streaming QA Pair Construction are shown in Table 12 and Table 13, respectively.

#### A.4 Diverse Visualization Cases We visualize qualitative results in the following figures:

• Figure 6 and Figure 7: Visualization of Feature-level token dropping and Pixel-level token dropping. These demonstrate that DTD performs video-aware dynamic pruning based on specific video content. With identical 𝜏 hyperparameters, different frames within the same video exhibit varying drop ratios due to adaptive temporal redundancy. Different videos also show significant redundancy differences: informative videos

Dataset #Videos Dataset #Videos Dataset #Videos COIN [52] 151 QV-Highlights [23] 1778 ActivityNet [15] 12 HD-VILA [64] 695 YouCook2 [83] 710 TVSum [50] 10 ViTT [20] 2000 QuerYD [40] 566 YouMakeup [56] 1801 VideoIC [55] 2649 Movie101 [71] 202 HiREST [72] 469

Total 11,043

- Table 9: Source datasets and number of unique videos sampled from each. Dataset names link to their respective original publications.

Main Category Subcategory Prompt Example

Event order What is the correct order of events in this video? Key Attribute change What are the key changes in the main character’s attire from the beginning to the

end of the video? Camera transitions How does the camera angle change between the beginning and middle frames? Event relationship Which event directly leads to [Cui Hu] aiming his gun at an adversary? Key Frame Focus Which frame captures the progression from indoor reflection to outdoor contem-

Temporal-enhanced

plation for Grandma Li Ailian?

Episodic Memory Which event occurred before the individual pointed towards the forest? Action Sequence Identification Which action occurred first in the makeup tutorial? Hallucination Detection Did the presenter ever apply a green eyeshadow during the tutorial?

Backward Tracing

Real-Time Visual Perception Future Prediction What is likely to happen next after the presenter shows the highlighter stick?

Sequential Steps Recognition What are the steps involved in applying bronzer as shown in the tutorial? Clues Reveal Responding What accessory was visible when the person was blending the contour product?

Forward Active Responding

###### Table 10: Task taxonomy and corresponding prompt examples.

[Figure 113]

[Figure 114]

(a) Feature-level: 𝜏𝑓 𝑒𝑎𝑡 = 0.4, drop ratio = 58.3% (b) Pixel-level: 𝜏𝑝𝑖𝑥𝑒𝑙 = 0.1

###### Figure 6: Visualization of (a) Feature-level token dropping and (b) Pixel-level token dropping for the video case 752 from StreamingBench.

###### Table 11: Performance comparison on StreamingBench full set including three categories: Real-Time Visual Understanding, OmniSource Understanding and Contextual Understanding.

Real-Time Visual Understanding Omni-Source Understanding Contextual Understanding

Model Params Frames

Overall OP CR CS ATP EU TR PR SU ACP CT All ER SCU SD MA All ACU MCU SQA PO All

Human† - - 89.47 92.00 93.60 91.47 95.65 92.52 88.00 88.75 89.74 91.30 91.46 88.00 88.24 93.60 90.27 90.26 88.80 90.40 95.00 100 93.55 91.66

###### Proprietary MLLMs

Gemini 1.5 pro - 1 fps 79.02 80.47 83.54 79.67 80.00 84.74 77.78 64.23 71.95 48.70 75.69 46.80 39.60 74.90 80.00 60.22 51.41 40.73 54.80 45.10 48.73 67.07 GPT-4o - 64 77.11 80.47 83.91 76.47 70.19 83.80 66.67 62.19 69.12 49.22 73.28 41.20 37.20 43.60 56.00 44.50 41.20 38.40 32.80 56.86 38.70 60.15 Claude 3.5 Sonnet - 20 80.49 77.34 82.02 81.73 72.33 75.39 61.11 61.79 69.32 43.09 72.44 31.60 34.00 32.80 48.80 36.80 38.40 34.80 34.40 64.71 37.70 57.68

###### Open-Source Video MLLMs

LLaVA-OneVision 7B 32 80.38 74.22 76.03 80.72 72.67 71.65 67.59 65.45 65.72 45.08 71.12 40.80 37.20 33.60 44.80 38.40 35.60 36.00 27.27 29.55 32.74 56.36 Qwen2-VL 7B 0.2-1 fps 75.20 82.81 73.19 77.45 68.32 71.03 72.22 61.19 61.47 46.11 69.04 41.20 22.00 32.80 43.60 34.90 31.20 26.00 39.60 22.73 31.66 54.14 MiniCPM-V 2.6 8B 32 71.93 71.09 77.92 75.82 64.60 65.73 70.37 56.10 62.32 53.37 67.44 40.80 24.00 34.00 41.20 35.00 34.00 31.60 41.92 22.22 34.97 53.85 LLaVA-NeXT-Video 32B 64 78.20 70.31 73.82 76.80 63.35 69.78 57.41 56.10 64.31 38.86 66.96 37.69 24.80 34.40 42.80 34.90 29.20 30.40 35.35 18.18 30.79 52.77 InternVL-V2 8B 16 68.12 60.94 69.40 77.12 67.70 62.93 59.26 53.25 54.96 56.48 63.72 37.60 26.40 37.20 42.00 35.80 32.00 31.20 32.32 40.91 32.42 51.40 Kangaroo 7B 64 71.12 84.38 70.66 73.20 67.08 61.68 56.48 55.69 62.04 38.86 64.60 37.60 31.20 28.80 39.20 34.20 32.80 26.40 33.84 16.00 30.06 51.10 LongVA 7B 128 70.03 63.28 61.20 70.92 62.73 59.50 61.11 53.66 54.67 34.72 59.96 39.60 32.40 28.00 41.60 35.40 32.80 29.60 30.30 15.91 29.95 48.66 VILA-1.5 8B 14 53.68 49.22 70.98 56.86 53.42 53.89 54.63 48.78 50.14 17.62 52.32 41.60 26.40 28.40 36.00 33.10 26.80 34.00 23.23 17.65 27.35 43.20 Video-CCAM 14B 96 56.40 57.81 65.30 62.75 64.60 51.40 42.59 47.97 49.58 31.61 53.96 33.60 22.00 28.40 34.80 29.70 27.60 24.40 16.67 22.73 22.88 42.53 Video-LLaMA2 7B 32 55.86 55.47 57.41 58.17 52.80 43.61 39.81 42.68 45.61 35.23 49.52 30.40 32.40 30.40 36.00 32.40 24.80 26.80 18.67 0.00 21.93 40.40

###### Streaming MLLMs

Flash-VStream 7B - 25.89 43.57 24.91 23.87 27.33 13.08 18.52 25.20 23.87 48.70 23.23 25.91 24.90 25.60 28.40 26.00 24.80 25.20 26.80 1.96 24.12 24.04 VideoLLM-online 8B 2 fps 39.07 40.06 34.49 31.05 45.96 32.40 31.48 34.16 42.49 27.89 35.99 31.20 26.51 24.10 32.00 28.45 24.19 29.20 30.80 3.92 26.55 32.48 Dispider 7B 1 fps 74.92 75.53 74.10 73.08 74.44 59.92 76.14 62.91 62.16 45.80 67.63 35.46 25.26 38.57 43.34 35.66 39.62 27.65 34.80 25.34 33.61 53.12 TimeChat-Online-7B 7B 1 fps (↓44.2%) 80.76 79.69 80.76 83.33 74.84 78.82 78.70 64.23 68.75 57.98 75.28 41.60 29.60 34.80 52.00 39.50 41.20 30.40 42.80 18.80 33.30 58.00 TimeChat-Online-7B 7B 1 fps (↓82.6%) 79.13 81.25 78.86 80.77 70.44 77.26 77.78 67.07 66.19 53.72 73.64 40.00 32.80 36.80 52.00 40.40 38.40 26.80 40.40 14.40 30.00 56.56

TimeChat-Online-7B 7B 1fps (100%) 80.22 82.03 79.50 83.33 76.10 78.50 78.70 64.63 69.60 57.98 75.36 38.40 26.80 35.60 50.40 37.80 39.20 31.60 41.60 28.80 35.30 58.11

[Figure 115]

[Figure 116]

(a) Feature-level: 𝜏𝑓 𝑒𝑎𝑡 = 0.4, drop ratio = 89.5% (b) Pixel-level: 𝜏𝑝𝑖𝑥𝑒𝑙 = 0.1

###### Figure 7: Visualization of (a) Feature-level token dropping and (b) Pixel-level token dropping for the video case 671 from StreamingBench.

[Figure 117]

1.0

|[Figure 118]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

1.0

0.9

0.9

0.8

0.8

0.7

0.7

0.6

0.6

0.5

0.5

0.4

0.4

0.3

Threshold (0.6)

0.3

0 10 20 30 40 50 60 70

(a) Scene Transition Point w/ Trigger Time

(b) Drop Ratio - Timeline Curve

###### Figure 8: Visualization of monitored Trigger Time via drop ratio curve. The colored highlighted frames correspond to trigger times that reveal video scene transitions. Our model utilizes a temporal patch size of 2.

[Figure 119]

(a) Scene Transition Point w/ Trigger Time

1.00

|[Figure 120]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

1.00

0.95

0.95

0.90

0.90

0.85

0.85

0.80

0.80

0.75

0.75

Threshold (0.85)

0.70

0 20 40 60 80 100

(b) Drop Ratio - Timeline Curve

###### Figure 9: Visualization of monitored Trigger Time via drop ratio curve. The colored highlighted frames correspond to trigger times that reveal video scene transitions. Our model utilizes a temporal patch size of 2.

###### Video Streams

Time (seconds)

113s 115s

114s

###### 116s 117s

118s

119s 120s

|[Figure 121]<br><br>[Figure 122]|
|---|

|[Figure 123]<br><br>[Figure 124]|
|---|

|[Figure 125]<br><br>[Figure 126]|
|---|

|[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|
|---|

|[Figure 130]<br><br>[Figure 131]|
|---|

|[Figure 132]<br><br>[Figure 133]|
|---|

|[Figure 134]<br><br>[Figure 135]|
|---|

|[Figure 136]<br><br>[Figure 137]|
|---|

Past Future

Current

[Figure 138]

User Query Time

|[Figure 139]|
|---|

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Trigger Time w/ Scene Change

【116 second】

[116 second] What specifically did the woman in red do?

[Figure 148]

[Figure 149]

She adjusted the white blouse of the woman in black.

Frames w/ low drop ratio indicate Video Scene Transitions

Drop Ratio

161s

162s

163s

164s 165s 166s

179s 180s

[Figure 150]

|[Figure 151]<br><br>[Figure 152]|
|---|

|[Figure 153]<br><br>[Figure 154]|
|---|

|[Figure 155]<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]|
|---|

|[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]|
|---|
|[Figure 162]|

|[Figure 163]<br><br>[Figure 164]|
|---|

|[Figure 165]<br><br>[Figure 166]|
|---|

|[Figure 167]<br><br>[Figure 168]|
|---|

… …

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

【165 second】

[Figure 176]

[Figure 177]

[Figure 178]

She adjusted the black corset and helped put on a yellow satin robe.

194s 195s

196s

202s

204s

228s 229s 230s

|[Figure 179]<br><br>[Figure 180]|
|---|

|[Figure 181]<br><br>[Figure 182]|
|---|

|[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]|
|---|

|[Figure 186]<br><br>[Figure 187]|
|---|

|[Figure 188]<br><br>[Figure 189]|
|---|

|[Figure 190]<br><br>[Figure 191]|
|---|

|[Figure 192]<br><br>[Figure 193]|
|---|

|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]|
|---|

[Figure 197]

###### …

… …

Times (s) 116s

|[Figure 198]|
|---|

|[Figure 199]|
|---|

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

【196 second】

###### 【230 second】

| | |
|---|---|
| | |

[Figure 206]

[Figure 207]

[Figure 208]

She adjusted the sleeve of another woman's dress and then held up a dark garment.

[Figure 209]

She adjusted the black dress and styled the hair of another woman.

###### Figure 10: Case study of TimeChat-Online on StreamingBench with drop ratio curve. When a user proposes a question “What specifically did the woman in red do?” that can also be answered by the future moments, TimeChat-Online will proactively generate responses at the future trigger time (i.e., the video scene transition timestamps), which are indicated by the frames with low token drop ratios.

[Figure 210]

[Figure 211]

[Figure 212]

(a) Pixel-level: 𝜏𝑝𝑖𝑥𝑒𝑙 = 0.01 (b) Pixel-level: 𝜏𝑝𝑖𝑥𝑒𝑙 = 0.05 (c) Pixel-level: 𝜏𝑝𝑖𝑥𝑒𝑙 = 0.1

[Figure 213]

[Figure 214]

[Figure 215]

(d) Feature-level: 𝜏𝑓 𝑒𝑎𝑡 = 0.7 (e) Feature-level: 𝜏𝑓 𝑒𝑎𝑡 = 0.6 (f) Feature-level: 𝜏𝑓 𝑒𝑎𝑡 = 0.5

###### Figure 11: Comprehensive visualization of feature-level and pixel-level token dropping with varying threshold values (𝜏𝑓 𝑒𝑎𝑡) for the same video case.

Total drop ratios of videos in StreamingBench grouped by DURATION

80

70

60

50.5

48.9

50

44.7 44.7

Dropratio(%)

42.4 43.5

42.3

41.6

40.8 39.9

40

32.6

30

20

10

0

[0,1)min [1,2)min [2,3)min [3,4)min [5,6)min [4,5)min [7,8)min [6,7)min [8,9)min [9,10)min >=10min

(a) StreamingBench

Total drop ratios of videos in VideoMME grouped by DURATION

80

70

60

50

46.5

45.5

44.0 43.5 44.1

Dropratio(%)

42.8 43.6

42.3

41.5

40.9 40.6

38.7

40

36.1

34.0

27.3 28.1

30

27.0

20

10

0

(60,90]s(90,120]s(30,60]s(0,30]s(2,4]min(6,8]min(8,10]min(4,6]min(12,15]min(10,12]min(15,30]min(50,55]min(45,50]min(30,35]min(40,45]min(35,40]min(55,60]min

(b) VideoMME

###### Figure 12: Average token drop ratio across different video durations using the same threshold parameter 𝜏.

Total drop ratios of videos in StreamingBench grouped by TASK TYPE

80

67.1

70

61.8 60.7

60

52.8

50.1 49.0

47.0 46.3 45.6 44.6

50

Dropratio(%)

42.4

38.6 38.5 37.9 37.6

40

33.7 32.6

30.7

30

20

10

0

Prospective ReasoningMisleading Context RecognitionCountingProactiveOutputSpatial UnderstandingSource DiscriminationEmotion RecognitionAction PerceptionMultimodal AlignmentText-Rich UnderstandingAttribute PerceptionCausal ReasoningEvent UnderstandingScene UnderstandingAnomaly Context UnderstandingObject PerceptionSequential Question AnsweringClipsSummarize

(a) StreamingBench

Total drop ratios of videos in VideoMME grouped by TASK TYPE

80

70

60

51.4

47.3

50

44.6 44.6

Dropratio(%)

42.2 42.0 41.9 41.8 41.1 40.7 39.6 38.6

40

30

20

10

0

Spatial PerceptionSpatial ReasoningTemporal ReasoningTemporal PerceptionInformation SynopsisObject RecognitionAction RecognitionOCRProblemsCounting ProblemAttribute PerceptionObject ReasoningAction Reasoning

(b) VideoMME

###### Figure 13: Average token drop ratio across different task types using same hyperparameter 𝜏 settings.

###### GPT-4 Prompt for Detailed Video Frame Caption Generation:

You are an advanced AI visual assistant tasked with describing frames extracted from a video clip. When provided with a frame, describe it in detailed and accurate terms, focusing on both static and dynamic elements visible within, as well as the shot type, object appearances and actions, environment and background variations, and camera movements. While your primary focus should be the current frame, you may reference the provided caption of the preceding frame to describe any relevant relationships between the two frames, particularly regarding camera movements, which often require contrasts between frames. Ensure your description reflects only the contents that can be determined in the current frame without analysis or speculation. Do not include any elements of this prompt in your response. Your caption should be in a narrative format, avoiding list-like itemizations. Begin with "This frame".

###### Here’s an output example:

This frame shows an arrangement of six small white bowls placed on a dark granite countertop. Each bowl contains different ingredients key to the cooking recipe. Starting from the top left and moving clockwise, the first bowl has a white powder which seems like cornstarch. A hand is seen above this bowl, with a finger pointing towards it, indicating either an instruction or a choice selection. The top middle bowl is empty, noticeably different from the rest. The top right bowl is filled with what appears to be brown sugar, granulated and slightly heaped. In the bottom row, the bowl on the left is also empty, mirroring the top middle bowl. The bottom middle bowl contains a light yellow-brown liquid, which could be oil or a prepared marinade. The bottom right bowl is filled with a dark liquid, possibly soy sauce. The lighting in this frame is uniform, ensuring all the details and texture of the ingredients are clearly visible. No background elements distract from the central subject. This static shot is focused on presenting the different ingredients required, providing a clear visual reference for the cooking process.

The current frame is uploaded as a .jpg file, and here’s some relevant information: <video_meta_information>

<video_name> <video_id> <video_subtitle> <video_category> <video_duration>

</video_meta_information> The caption of the preceding frame is provided below: <caption_of_preceding_frame>

###### Table 12: The prompt template used for scene-oriented key frame caption generation.

GPT-4 Prompt for Streaming VideoQA Generation: Given the following video frame descriptions, generate 5 different question-answer pairs that focus on temporal understanding and visual perception. These questions should cover the following cognitive abilities:

- 1. Episodic Memory: Questions about where objects were located before events or recalling details from earlier frames
- 2. Action Sequence Identification: Questions about what actions occurred before others or the order of events
- 3. Hallucination Detection: Questions requiring verification of visual elements that actually appeared in the video
- 4. Future Prediction: Questions predicting what might happen next based on the video context
- 5. Sequential Steps Recognition: Questions about step-by-step processes shown in the video
- 6. Clues Reveal Responding: Questions requiring attention to specific details or clues spread across frames

Video Description: {description} Question Requirements:

- 1. Create multiple-choice questions (A, B, C, D format) with one correct answer
- 2. Include at least one question from each of the six cognitive abilities listed above
- 3. Ensure questions require temporal understanding across frames
- 4. Include questions that require connecting information from different parts of the video
- 5. Create at least 2-3 questions that specifically require comparing frames with long distances between them CRITICAL REQUIREMENTS:

- 1. NEVER reference specific frame numbers/indices in questions or answers (e.g., DO NOT say "In Frame 201" or "From Frame 10 to Frame 20")
- 2. Instead, describe the content and events directly (e.g., "When the person was in the kitchen" or "After picking up the cup")
- 3. You MAY reference timestamps if needed (e.g., "At 1:20 in the video")
- 4. Questions should focus on the content and temporal relationships, not on arbitrary frame numbering

Target Format: Please format your response as a JSON object using the following structure: [

{

"question": <question>, "options": <options>, "answer": <answer>, "answer_text": <answer_text>, "category": <category>, "rationale": <rationale>, "rationale_frames": <rationale_frames>

}

] Ensure that your response includes 5 multiple-choice question-answer pairs in this JSON format, with each pair addressing different cognitive abilities from the six categories listed. **The specific frame numbers (index) or timestamps should NOT appear in the questions or options themselves.**

###### Table 13: The prompt template used for generating streaming question-answer pairs.

