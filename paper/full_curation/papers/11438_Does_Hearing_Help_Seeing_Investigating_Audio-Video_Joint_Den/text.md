# arXiv:2512.02457v2[cs.CV]3Dec2025

## Does Hearing Help Seeing? Investigating Audio–Video Joint Denoising for Video Generation

Jianzong Wu1,2 Hao Lian1 Dachao Hao1 Ye Tian1 Qingyu Shi1 Biaolong Chen2 Hao Jiang2 Yunhai Tong1

1 Peking University 2 Alibaba Group Project Page: https://jianzongwu.github.io/projects/does-hearing-help-seeing/ Email: jzwu@stu.pku.edu.cn

|Video Noise|
|---|

Supervise

|Video<br><br>Prompt| |
|---|---|
| | |

|Video Prediction|
|---|

|Video GT|
|---|

|Video Prompt| |
|---|---|
| | |

T2AV model

###### Supervise

|Audio GT|
|---|

|Video Prediction|
|---|

|Video GT|
|---|

|Video Noise|
|---|

|Audio Prediction|
|---|

|Audio Noise|
|---|

|Audio Prompt| |
|---|---|
| | |

T2V model

(a) T2V Train

(c) T2AV Train

|…stick appears and<br><br>starts hitting the seat of<br><br>the chair repeatedly…| |
|---|---|
| | |

|…stick appears and starts hitting the<br><br>seat of the chair<br><br>repeatedly…| |
|---|---|
| | |

|Video Noise|
|---|

Chair: Stick: Hitting: Motion with click sound!

Chair: Stick:

T2AV model

Hitting: Wave closer?

|Video<br><br>Noise| |
|---|---|
| | |

T2V model

|…distinct, sharp thumping sounds…|
|---|

|Audio<br><br>Noise|
|---|

[Figure 1]

[Figure 2]

video output

(d) T2AV Inference

(b) T2V Inference

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Figure 1. Conceptual comparison between T2V and T2AV diagrams. (a) T2V training denoises video latents with video-only supervision. (b) In inference, T2V may misinterpret motion because there is insufficient evidence linking visual appearances to world physics. (c) T2AV training jointly denoises audio and video latents with audio–video supervision. (d) In inference, T2AV produces physically plausible motion with synchronized audio. Audio helps video generation models understand the world.

### Abstract

Recent audio-video generative systems suggest that coupling modalities benefits not only audio–video synchrony but also the video modality itself. We pose a fundamental question: Does audio–video joint denoising training improve video generation, even when we only care about video quality? To study this, we introduce a parameterefficient Audio–Video Full DiT (AVFullDiT) architecture that leverages pre-trained text-to-video (T2V) and text-toaudio (T2A) modules for joint denoising. We train (i) a T2AV model with AVFullDiT and (ii) a T2V-only counterpart under identical settings. Our results provide the first systematic evidence that audio–video joint denoising can deliver more than synchrony. We observe consistent improvements on challenging subsets featuring large and object contact motions. We hypothesize that predicting audio acts as a privileged signal, encouraging the model to inter-

nalize causal relationships between visual events and their acoustic consequences (e.g., collision × impact sound), which in turn regularizes video dynamics. Our findings suggest that cross-modal co-training is a promising approach to developing stronger, more physically grounded world models. Code and dataset will be made publicly available.

### 1. Introduction

Recent closed-source audio–video (A/V) generators, such as Sora 2 [29] and Veo 3 [44], demonstrate not only striking audio–video synchronization but also marked advances in video-only fidelity, particularly for large, fast motions and physically grounded dynamics that have challenged prior text-to-video (T2V) systems [6, 9, 10, 35, 43]. These observations raise a fundamental question for the community: Does joint text-to-audio-video (T2AV) training improve video generation even when only the video output is

evaluated? Answering this question matters. If the answer is negative, T2AV may primarily serve as a coordination mechanism that integrates existing T2V and text-to-audio (T2A) capabilities, yielding better synchrony without advancing the video model’s ceiling as a world model. Conversely, suppose hearing helps seeing, i.e., adding an audio denoising objective directly improves video learning. In that case, we may get a promising insight: Audio can act as a privileged signal that strengthens a model’s grasp of physical causality (e.g., contact → sound), thereby regularizing motion and interactions, as shown in Fig. 1. Intuitively, natural creatures rarely rely on vision alone: seeing a closed door does not reveal whether it will open next, but the sound of footsteps and a turning handle makes that prediction likely. From this perspective, audio is a key component for agents to understand the physical world. While recent related open-sourced efforts (e.g., JavisDiT [27], UniVerse1 [40]) build capable T2AV systems with large A/V corpora, they primarily benchmark synchrony and overall generation quality. However, what remains missing is a controlled, head-to-head assessment that isolates the effect of joint A/V denoising and solely video denoising. In contrast, we address this gap through a systematic comparison of T2AV versus T2V under matched pre-trained models, data, and optimization settings, providing valuable insights for our conclusions.

To enable a clean study, we introduce AVFullDiT, a parameter-efficient framework that inherits knowledge from pre-trained T2V and T2A backbones. Its AVFull-Attention performs symmetric self-attention over concatenated audio and video tokens, encouraging rich cross-modal information exchange while preserving unimodal inductive biases. We further propose AVSyncRoPE, which rescales audio rotary positional encodings (RoPE) to align real-time token spacing with video, thereby improving video learning and, incidentally, A/V synchrony. Using two curated datasets—one smaller, category-focused set for fast iteration and a larger open-domain set for generalization—we train matched T2V and T2AV models and evaluate them on video-only evaluation sets. Our study yields three findings: 1) despite a slightly lower validation loss for T2V, T2AV consistently improves video metrics; 2) T2AV produces more realistic, tempered motion, avoiding exaggerated or frozen dynamics; and 3) T2AV enhances physical commonsense, yielding more plausible object interactions and mitigating failures such as contact avoidance. We argue that these improvements arise from the world knowledge embedded in the audio modality, and our experimental results in Sec. 4 offer supporting evidence for this. Our contributions can be summarized as:

- • We pose and systematically answer a fundamental question: Does hearing help seeing for video generation?
- • We present AVFullDiT, with AVFull-Attention and

AVSyncRoPE, to efficiently co-train audio and video.

• Through extensive controlled experiments, we demonstrate that audio supervision enhances video generation, clarifying where and how T2AV improves T2V performance and providing actionable insights for future multimodal world models.

### 2. Related Work

Multimodal co-training mutual benefit. Recent work has extended LLMs [3, 18, 25, 38, 52] to Vision–Language Models (VLMs) for perception tasks such as image captioning [4, 12, 26, 55, 56, 58]. In parallel, diffusion-based generative models have triggered rapid advances in visual synthesis and editing [6, 9, 10, 16, 23, 31, 32, 39, 41, 43, 54]. A growing body of research is investigating unified architectures that co-train understanding and generation, expecting potential mutual benefits [11, 13, 14, 33, 45, 50]. Empirically, perception-oriented objectives enhance controllability and visual quality in visual generation and editing [1, 2, 11, 13, 14, 34, 36, 37, 45–50]. Beyond image–text, multimodal co-training has also been leveraged for video generation. VideoJAM [7] demonstrates that pairing video synthesis with optical-flow joint denoising benefits scenes with significant motion, and UDPDiff [53] shows that mask–video co-generation enhances generation fidelity. These findings suggest that multimodal co-training can induce positive transfer across modalities. We argue that audio is a comparably important modality for enabling video generation systems to understand the world. However, prior work remains limited in this direction. In this work, we present the first systematic investigation of audio–video co-training for T2V models, finding that cross-modal cotraining delivers consistent gains and enhances the shortterm capabilities of current video generation models.

Audio-video joint denoising. Recent closed-source systems Veo 3 [44] and Sora 2 [29] demonstrate striking advances in text-to-video quality, while jointly generating temporally aligned audio. These successes surface two questions for the community: (i) how to train audio–video generative models at comparable quality, and (ii) whether their superior video realism is attributable, in part, to the co-generation architecture itself. Early explorations tackle the first question from different angles. SVG [21] and Wang et al. [42] propose co-generation frameworks that leverage pre-trained T2V and T2A components and fine-tune on small datasets, demonstrating the feasibility of the task. In contrast, JavisDiT [27] and UniVerse-1 [40] develop largescale data pipelines and training recipes to create generalpurpose A/V generators. While these efforts deliver compelling end-to-end results, they do not systematically address the second question: beyond synchrony, does introducing the audio modality directly enhance video generation quality? Our work, to our knowledge, is the first to

study this gap by isolating the effect of joint denoising versus a T2V-only objective. We present evidence that joint denoising yields consistent gains in video generation, indicating that the audio modality provides valuable information for video generation models.

### 3. Method

In this section, we present AVFullDiT, an architecture that maximally reuses the weights of pre-trained T2V and T2A models while introducing minimal additional parameters. A new AVFull-Attention module bridges the two modalities, enabling rapid adaptation to a unified audio–video joint denoising objective. The overall architecture is shown in Fig. 2(a), and the AVFull-Attention design is detailed Fig. 2(b). To further enhance video quality, we introduce AVSyncRoPE, a simple but effective modification to vanilla audio RoPE that synchronizes audio and video along the token timeline. This synchronization enhances video learning, and as a secondary benefit, improves audio–video alignment. Its architecture is illustrated in Fig. 3.

#### 3.1. AVFullDiT

Overall architecture. As shown in Fig. 2(a), AVFullDiT consists of two unimodal early towers, which are precisely the early layers from the pre-trained T2V and T2A models. In later layers, there are stacked audio–video joint blocks, which compute AVFull-Attention (detailed in Sec. 3.1). We train the whole Transformer with a unified flow-matching velocity-prediction objective.

Latent representation and noising. Given an input video Dv and its paired audio Da, we first obtain latents with their corresponding pre-trained VAEs:

x0v = Ev(Dv), x0a = Ea(Da), (1)

where Ev and Ea are the video and audio VAE encoders, respectively. Then, for a sampled timestep t and two Gaussian noises ϵv,ϵa∼N(0,I), we construct noisy latents by

xtv = (1 − t)x0v + tϵv, xta = (1 − t)x0a + tϵa, (2) and define the velocity targets as

vvt = ϵv − x0v, vat = ϵa − x0a. (3) Additionally, we obtain video and audio caption representations cv and ca using frozen text encoders from the pretrained video and audio checkpoints.

Unimodal DiT blocks on earlier layers. The noisy latents are synchronized with AVSyncRoPE along the time dimension (detailed in Sec. 3.2). Tokens are then processed by stacks of unimodal DiT blocks, which are initialized from the pre-trained models:

hnv = Vn hnv−1, cv, t , n ∈ {1,2,...,Nv} hna = An hna−1, ca, t , n ∈ {1,2,...,Na}

(4)

where n denotes the layer number. Vn and An are the pretrained video and audio DiT blocks. hnv and hna are the hidden states. Nv and Na are the number of unimodal layers for each modality. We define them separately because the total number of layers for pre-trained video and audio DiTs is usually different. We set an equal number of later layers to compute the AVFull-Attention, leaving the earlier layers as unimodal. The unimodal earlier layers utilize pre-trained knowledge on each modality to obtain high-quality video and audio priors for further cross-modal fusion.

AVFull-Attention on later layers. On top of the two early towers, we stack Nav Audio–Video Full DiT Blocks that exchange information across modalities via AVFull-Attention. Specifically, we replace the Self-Attention for both pretrained models with AVFull-Attention, leaving the CrossAttention and FFN unchanged as unimodal.

a×Ca be the hidden states entering the AVFull-Attention, where B, L(·), and C(·) are batch size, sequence length, and channel dimension, respectively. We take the video channel width Cv as the joint attention width.1 For the video branch, we reuse the pre-trained T2V projections to get query, key, and value:

Let hv ∈ RB×L

v×Cv and ha ∈ RB×L

Qv = hvWvq, Kv = hvWvk, Vv = hvWvv, Qv,Kv,Vv ∈ RB×L

(5)

v×Cv.

Because the audio channel width Ca may differ from Cv, we expand the audio projections to the same width by concatenating newly introduced, learnable projection matrices alongside the pre-trained T2A weights:

Qa = ha Waq;Wanq , Ka = ha Wak;Wank , Va = ha Wav;Wanv ,

Qa,Ka,Va ∈ RB×L

a×Cv, (6)

where Wa(·) ∈ RC

a×Ca are initialized from the T2A checkpoint and Wan(·) ∈ RC

a×(Cv−Ca) are newly added parameters. · denotes concatenation along the channel dimension. This design enables the reuse of all pre-trained parameters while introducing only a minimal set of additional parameters to align channel sizes.

We now form a single sequence by concatenating audio and video tokens along the sequence dimension and perform standard multi-head attention:

Qav = Concat(Qv,Qa), Kav = Concat(Kv,Ka), Vav = Concat(Vv,Va), Aav = MHSA(Qav,Kav,Vav).

(7)

Because the attention is computed over the union of audio and video tokens, information flows bidirectionally and can

1In practice Cv ≥ Ca for common T2V/T2A models. For training stability, we choose to expand the audio channel.

Newly Added Weight

T2A Pre-trained Weight

T2V Pre-trained Weight

|Predicted Video Velocity|
|---|

|Predicted Audio Velocity|
|---|

|Video Loss|
|---|

###### Audio Loss

|𝐶𝑎|
|---|

|[𝐵,𝐿𝑣,𝐶𝑣]|
|---|

|[𝐵,𝐿𝑎,𝐶𝑎]|
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

| |
|---|

× 𝑁a𝑣 layers

###### Audio-Video Full DiT Blocks

| | |
|---|---|
|𝐶𝑣| |

FFN

|GT Video Velocity|
|---|

|GT Audio Velocity|
|---|

FFN

𝑾𝑣𝑜 𝑶𝑣 = 𝑨𝑣𝑾𝑣𝑜 𝑾𝑎𝑜 𝑾𝑎𝑛𝑜 𝑶𝑎 = 𝑨𝑎[𝑾𝑎𝑜,𝑾𝑎𝑛𝑜 ]

Cross-Attention

###### Audio-Video Full-Attention

MHSA(𝑸𝑎𝑣,𝑲𝑎𝑣,𝑽𝑎𝑣)

| |[𝐵,𝐿𝑎 + 𝐿𝑎,𝐶𝑣]|
|---|---|
| | |
| | |

|𝑸𝑎𝑣,𝑲𝑎𝑣,𝑽𝑎𝑣|
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

× 𝑁𝑎 layers

× 𝑁𝑣 layers

|𝑸𝑣,𝑲𝑣,𝑽𝑣|
|---|

Concat [𝐵,𝐿𝑣,𝐶𝑣]

𝑸𝑎,𝑲𝑎,𝑽𝑎

[𝐵,𝐿𝑎,𝐶𝑣]

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

Audio DiT Block

Video DiT Block

𝑾𝑎𝑞 𝑾𝑎𝑛𝑞 𝑸𝑎 = 𝒉𝑎[𝑾𝑎𝑞,𝑾𝑎𝑛𝑞 ]

𝑾𝑣𝑞 𝑸𝑣 = 𝒉𝑣𝑾𝑣𝑞

|Video Noise|
|---|

|Audio Noise|
|---|

Audio

Video VAE

𝑾𝑣𝑘 𝑲𝑣 = 𝒉𝑣𝑾𝑣𝑘

𝑾𝑎𝑘 𝑾𝑎𝑛𝑘 𝑲𝑎 = 𝒉𝑎[𝑾𝑎𝑘,𝑾𝑎𝑛𝑘 ]

VAE

| | |
|---|---|
|[Figure 9]<br><br>| |

𝑾𝑣𝑣 𝑽𝑣 = 𝒉𝑣𝑾𝑣𝑣

𝑾𝑎𝑣 𝑾𝑎𝑛𝑣 𝑽𝑎 = 𝒉𝑎[𝑾𝑎𝑣,𝑾𝑎𝑛𝑣 ]

|[Figure 10]|
|---|

|A wooden board with a reddishbrown background is<br><br>shown…|
|---|

|The audio features a series of distinct, sharp clicks…|
|---|

|[𝐵,𝐿𝑣,𝐶𝑣]|
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

[𝐵,𝐿𝑎,𝐶𝑎]

𝐶𝑎

| |
|---|

Input Video Tokens Input Audio Tokens

Video Caption Input Video Input Audio Audio Caption

(a) AVFullDiT (b) AVFull-Attention

Figure 2. The architecture of AVFullDiT and Audio-Video Full Attention. (a) AVFullDiT reuses pre-trained T2V/T2A early towers and stacks joint blocks that predict video/audio velocities under a unified flow-matching loss. (b) AVFull-Attention performs symmetric MHSA over the concatenated audio–video token sequence using the video width as the joint dimension; audio projections are expanded with small adapter matrices. The attended sequence is split and projected back per modality.

propagate across the two modalities within a single operation. In the overall architecture, AVFull-Attention provides the core and only multimodal computation.

Finally, we split the attended sequence back to each modality and apply modality-specific output projections:

Av = Aav[:,: Lv,:], Aa = Aav[:,Lv :,:], (8)

ov = AvWvo, oa = Aa,[Wao;Wano ], (9) where Wvo ∈ RC

v×Cv is initialized from the T2V checkpoint, Wao ∈ RC

a×Ca comes from T2A, and Wano ∈ R(C

v−Ca)×Ca is newly added. Hence, we transfer the audio latent back into its original channel width Ca.

After Nav AVFull blocks, the model predicts video and audio velocities v˜v and v˜a. The loss is computed as:

Lv = (v˜v − vv)2, La = (v˜a − va)2, L = λvLv + λaLa,

(10)

where λv and λa are video and audio loss weights.

Discussion. Compared with modeling cross-modal information by Cross-Attentions, AVFull-Attention has three advantages: (i) symmetry: audio and video participate in the same attention graph; (ii) multi-hop fusion: tokens from one modality can interact via tokens of the other within the same

|𝜏 =𝐶𝑜𝑛𝑠𝑒𝑞𝑢𝑒𝑛𝑡𝐶𝑜𝑛𝑠𝑒𝑞𝑢𝑒𝑛𝑡𝑨𝒖𝒅𝒊𝒐𝑽𝒊𝒅𝒆𝒐𝐿𝑎𝑡𝑒𝑛𝑡𝐿𝑎𝑡𝑒𝑛𝑡 𝐹𝑟𝑎𝑚𝑒𝐹𝑟𝑎𝑚𝑒𝑇𝑖𝑚𝑒𝑇𝑖𝑚𝑒𝐷𝑢𝑟𝑎𝑡𝑖𝑜𝑛𝐷𝑢𝑟𝑎𝑡𝑖𝑜𝑛Δ𝑡Δ𝑡𝑣<br><br>𝑎|
|---|

Video Latent Frame

Audio Latent Frame

VideoLatentSpatialDimension

|(0, 0)|(1, 0)|(2, 0)|
|---|---|---|
|(0, 1)|(1, 1)|(2, 1)|
|(0, 2)Vanil|la(1,Video2)Ro|PE(2, 2)|

|(0)|(1)|(2)|(3)|(4)|…|
|---|---|---|---|---|---|

(b) Audio RoPE

|(0𝜏)<br><br>|(1𝜏)<br><br>|(2𝜏)<br><br>|(3𝜏)<br><br>|(4𝜏)<br><br>|…|
|---|---|---|---|---|---|

(a) Video RoPE

(c) AVSyncRoPE (Ours)

Figure 3. The architecture of AVSyncRoPE. (a–b) Vanilla RoPEs of the pre-trained video and audio DiTs live on different temporal scales. (c) We rescale audio positions so that video and audio tokens are aligned in real-time. This improves video learning, along with the side benefit of tighter A/V synchrony.

layer; and (iii) parameter efficiency: only a small set of adapter weights Wanq ,Wank ,Wanv ,Wano is added to align channel sizes, while all pre-trained weights are reused and fine-tuned end-to-end.

#### 3.2. AVSyncRoPE

Motivation. The pre-trained T2V and T2A DiTs utilize RoPEs that are defined in their own latent spaces. Because

###### Table 1. Datasets used in experiments. #Clips and Duration are those after pre-processing.

Dataset Source #Clips Duration (s) Domain AVSync15 [57] VGGSound [8] 1,499 2 Motion and Related Sound Effects Landscape [24] Youtube 778 2 Nature Scenes and Sounds The Greatest Hits [30] Self-Collected 977 2 Clicking Various Surfaces VGGSound [8] Youtube 107,648 5 Open-World Sounded Video

those latent frames cover different time durations, the two positional systems are on different time scales. When we concatenate audio and video tokens for AVFull-Attention, a token pair that occurs at the same real time can carry very different RoPE phases, which makes it ambiguous for the model to learn which audio token should align with which video token in the real time dimension.

AVSyncRoPE. Let ∆tv and ∆ta denote the duration covered by the consequent video and audio latent frame, respectively (determined by the VAEs’ temporal scale factors, video fps, and audio sampling rate). We define the constant

∆tv ∆ta

. (11)

τ =

Let r = RoPE(p) denote the standard RoPE that rotates the time dimension by angles proportional to the temporal position index p. In the pre-trained models, integer indices p ∈ {0,1,...} are used: pv for video and pa for audio. AVSyncRoPE is a parameter-free, drop-in change that rescales the audio position index so that both modalities are expressed in the same time scales. Concretely, we divide the audio index by τ:

pa τ

). (12)

rv = RoPE(pv), ra = RoPE(

After rescaling, a video token at index p and an audio token at index τp share the same RoPE value, which corresponds to the fact that they are at the same point in time. We apply AVSyncRoPE among all attention layers in the unimodal audio tower and in the AVFull-Attention blocks.

#### 3.3. Data Collection

Source. To thoroughly evaluate the influence of the T2AV training diagram on T2V models, we curate a heterogeneous corpus that balances open-domain everyday scenes, natural environments, and human performance content. Specifically, we draw from four sources: AVSync15 [57], a curated set that emphasizes tightly synchronized audiovisual events. Landscape [24]: nature/ambient footage with slowly varying dynamics. The Greatest Hits [30]: videos containing a stick clicking on various surfaces, which is very compact in motion and sound. And VGGSound [8]: open-domain audio–video clips. This is a large dataset for training at scale. This mixture exposes the model to a wide range of motion statistics, acoustic textures, and event densities, which we think is adequate for stable and convincing conclusions. The statistics are shown in Tab. 1.

Pre-precessing and annotation. After downloading the full raw dataset, we apply data pre-processing to remove degenerate content. Specifically, we apply (i) duplicate video removal using metadata; (ii) silence video removal based on average audio volume; and (iii) portrait resolution video removal. To maximize reuse of pre-trained unimodal token–text interactions in our T2V and T2A models, we annotate two captions per clip: a silent video-focused description cv and an audio-focused description ca. Video captions are generated using Tarsier2 [56], which is prompted with silent videos only, aiming to describe visible entities, actions, camera motion, and scene context. Audio captions are produced with Qwen2.5-Omni [51], prompted with video and audio. It describes the audio while ignoring purely visual attributes. The average token count for video and audio captions is approximately 116 and 47, which is suitable for the pre-trained T2V and T2A text encoder checkpoints.

### 4. Experiments

#### 4.1. Settings

Dataset. Based on the curated corpus described in Sec. 3.3, we partition the data into two distinct datasets. The first, a smaller and more specialized dataset we term ALT-Merge, aggregates clips from AVSync15 [57], Landscape [24], and The Greatest Hits [30]. We set aside a total of 300 clips for evaluation, ensuring a balanced contribution of approximately 100 clips from each of the three sources. The remaining 2,954 clips constitute the training set. The second utilizes the large-scale VGGSound dataset [8]. We split 106,717 clips for training and 731 clips for evaluation. For the evaluation set, we perform balanced sampling across their semantic categories.

Implementation details. We utilize Wan2.2-TI2V-5B [39] as the pre-trained video diffusion model and TangoFlux [19] as the pre-trained audio diffusion model. These models were selected for their academically manageable scale, state-of-the-art performance, and their shared use of a flow matching objective. During training, we fine-tune all parameters of AVFullDiT while keeping the VAEs and text encoders frozen. A consistent set of hyperparameters is used across all experiments, featuring a learning rate of 1e-5 and the AdamW optimizer [28]. All models are trained at a fixed resolution of 192 × 320. For the ALT-Merge dataset, we train for 7,500 steps with a batch size of 8, using video clips of 2-second duration. For the VGGSound dataset, training is extended to 100,000 steps with a batch size of 24, using 5-second video clips to ensure joint convergence of both audio and video. Training was conducted on NVIDIA H20 141G GPUs. In inference, we use 50 denoising steps. For classifier-free guidance (CFG) in the T2AV model, the positive branch is conditioned on both video and audio text prompts, while the negative branch is conditioned on neg-

Video Prompt

Audio Prompt

…The stick continues to move back and

…a series of distinct,

forth over the keyboard and mouse,

###### sharp snapping

simulating conducting motions…

sounds…

[Figure 11]

[Figure 12]

[Figure 13]

T2AV

[Figure 14]

[Figure 15]

[Figure 16]

T2V

- Figure 4. Example of wrong video prompt annotation in TheGreatestHits. The bolded video prompts are the incorrect part, while the audio prompts indicate the correct generation result for T2AV.

ative prompts for both modalities. The guidance scales for video and audio are set to 5.0 and 4.5, respectively, inheriting the settings from the pre-trained models.

Metrics. To comprehensively evaluate the capabilities of video generation, we employ a suite of automated metrics. For general video quality, we adopt five dimensions from VBench [17]: Background Consistency, which assesses the stability of static regions; Dynamic Degree, measuring the magnitude of motion; Image Quality, for the aesthetic appeal of frames; Subject Consistency, ensuring the coherent appearance of primary subjects over time; and Text Consistency, which evaluates the semantic alignment with the silent video prompt. To test our central hypothesis that the audio modality aids in understanding the physical world, we incorporate a metric for physical commonsense. Specifically, we use the pre-trained model from Videophy-2 [5] to score the physical plausibility of generated content. In our ablation studies, we introduce additional metrics for audio quality and audio-video synchrony. Audio fidelity is measured by Fr´echet Audio Distance (FAD) [22]. We use the CLAP [15] score to evaluate the semantic similarity between the generated audio and the audio prompt. Finally, a pre-trained Synchformer model [20] is utilized to calculate the temporal offset between the audio and video tracks.

#### 4.2. Quantitative Results

Metrics comparison. We conduct a quantitative comparison between the T2AV model and the T2V counterpart under identical training settings. To further validate our findings, we also evaluate performance on representative subsets curated from each evaluation set. The comprehensive results are presented in Tab. 2.

On the full ALT-Merge evaluation set, the T2AV model demonstrates consistent improvements over the T2V baseline across all metrics except for Dynamic Degree. This provides strong initial evidence that joint audio-video training benefits the video generation process. It is important

to note that a higher Dynamic Degree, which measures the magnitude of motion, does not necessarily correlate with better video quality. We observe that the T2V model exhibits higher variance in this metric, often producing videos with either exaggerated motion or excessive stillness. In contrast, the T2AV model generates motion that is more stable and realistic, with a more concentrated distribution for the Dynamic Degree score. This is evident in the subset results: on The Greatest Hits, a dataset featuring rapid stick-hitting motions, T2V scores 91.00 while T2AV scores a more tempered 76.00. Conversely, on the less dynamic Landscape dataset, T2V scores a low 16.25, whereas T2AV achieves 31.25, indicating a tendency towards more plausible motion levels. These nuances will be further detailed in Sec. 4.3. The most significant gains are observed in the Physics metric, which measures physical commonsense. On the motion- and contact-intensive subset of The Greatest Hits, the T2AV model achieves the largest improvement of 3.14%. The AVSync15 and Landscape subsets show gains of 2.59% and 1.03%, respectively. This trend strongly suggests that for scenes involving substantial motion and object interactions, joint training with audio helps the model better grasp real-world physical laws, validating our core hypothesis that “hearing helps seeing.” Another noteworthy result is the lower Text Consistency on The Greatest Hits. Upon inspection, we find this is attributable to systematic errors in the video prompts generated by the Tarsier2 [56] model for silent videos. As shown in Fig. 4, the silent video captioner often mislabels a stick striking an object as a stick merely waving in the air, as it cannot perceive the acoustic cues of impact. Our T2AV model, however, correctly leverages the audio prompt describing “snapping sounds” to generate the appropriate contact-rich video. This explains the lower score and highlights a key weakness of the T2V training paradigm: its susceptibility to inaccuracies in silent video annotation. Notably, even with accurate prompts, the T2V model often generates videos that avoid contact, a failure mode we demonstrate in Sec. 4.3.

On the large-scale VGGSound dataset, our findings are consistent. The T2AV model outperforms the T2V baseline on most metrics in the full set, confirming that the introduction of an audio objective enhances performance in the general domain. To investigate whether this improvement stems from learned real-world correlations, we partitioned the dataset into two subsets: AV-Tight (e.g., chopping wood, church bell ringing), where audio and visual events are highly coupled, and AV-Loose, where they are not. The results in Tab. 2 show that the AV-Tight subset not only achieves the highest scores but also registers the most significant percentage improvements across all metrics (excluding Dynamic), particularly in Text Consistency (+2.70%) and Physics (+2.51%). This strongly supports the hypothesis that the model’s enhanced understanding is de-

- Table 2. Video generation comparison between T2AV and T2V training. Improvements are highlighted in green, while decreases are highlighted in red. Metrics with a † are not definitely better when higher. Numbers with the largest percentage improvements are bolded.

Method Dataset Evaluation Set BGConsis Dynamic† ImageQual SubjQual TextConsis Physics

Full Set 97.44 52.67 59.04 95.79 9.91 4.27 |— AVSync15 [57] 98.14 45.00 61.10 97.69 9.78 4.25 |— Landscape [24] 98.11 16.25 54.29 96.59 8.68 4.85 ⌞ — TheGreatestHits [30] 96.05 91.00 60.39 92.87 11.05 3.82

ALT-Merge

T2V

Full Set 94.19 77.70 59.57 88.72 9.34 4.34 |— AV-Tight 94.44 79.71 60.18 89.07 9.62 3.99 ⌞ — AV-Loose 94.06 77.08 59.68 88.55 9.24 4.43

VGGSound [8]

Full Set 97.93 (+0.50%) 49.67 (-5.70%) 59.87 (+1.40%) 96.30 (+0.53%) 10.00 (+0.91%) 4.36 (+2.11%) |— AVSync15 98.42 (+0.28%) 40.00 (-11.11%) 62.85 (+2.86%) 97.88 (+0.19%) 9.98 (+2.04%) 4.36 (+2.59%) |— Landscape 98.33 (+0.22%) 31.25 (+92.31%) 54.26 (-0.01%) 97.10 (+0.53%) 8.95 (+3.11%) 4.90 (+1.03%) ⌞ — TheGreatestHits 97.01 (+1.00%) 76.00 (-16.48%) 60.78 (+0.64%) 93.78 (+0.98%) 10.86 (-1.72%) 3.94 (+3.14%)

ALT-Merge

T2AV

Full Set 94.27 (+0.08%) 77.98 (+0.36%) 58.78 (-1.33%) 88.92 (+0.22%) 9.58 (+2.57%) 4.37 (+0.69%) |— AV-Tight 94.67 (+0.24%) 78.98 (-0.92%) 60.93 (+1.25%) 89.68 (+0.68%) 9.88 (+2.70%) 4.09 (+2.51%) ⌞ — AV-Loose 94.17 (+0.12%) 77.44 (+0.47%) 58.45 (-2.06%) 88.77 (+0.25%) 9.46 (+2.38%) 4.44 (+0.22%)

VGGSound

rived from learning the intrinsic coupling between modalities as they occur in the physical world.

[Figure 17]

[Figure 18]

(a) Validation loss on ALT-Merge. (b) Validation loss on VGGSound.

- Figure 5. Validation loss comparison between T2AV and T2V.

Validation loss comparison. Beyond the metrics, we also track the validation loss, as illustrated in Fig. 5. We observe a consistent trend across both the ALT-Merge and VGGSound datasets: the video validation loss for the T2Vonly model remains slightly lower than that of the T2AV model, with the difference being less than 0.01. This finding suggests that a slightly lower validation loss does not directly equate to superior generative quality. Furthermore, we note that the audio loss in the T2AV model converges more slowly than the video loss. We attribute this to the newly introduced parameters in the audio branch, which must be learned from scratch. This outcome aligns with our design to prioritize video performance, leveraging the audio modality as a regularizer to enhance video generation.

#### 4.3. Qualitative Results

Fig. 6 presents the qualitative comparison. The results visually substantiate the high variance observed in the T2V model’s Dynamic Degree metric. For instance, when prompted to generate a person playing the violin (a) or a windmill in the wind (b), the T2V model produces motion that is either unnaturally agitated (jittering bow) or physically implausible (frantically spinning blades). In contrast, the T2AV model generates more controlled and realistic actions, such as a distinct up-and-down bow stroke and a

reasonably paced windmill rotation, which are coherently synchronized with corresponding audio cues. Similarly, in the waterfall scene (c), T2AV renders dynamic water flow, while the T2V generation is nearly static. These cases demonstrate that T2AV produces more stable and natural movements, thereby avoiding the extremes of exaggerated or minimal motion commonly found in the T2V baseline. In the knife-sharpening example (d), the T2V model fails to depict contact, showing the knife gliding above the stone. This “contact avoidance” is a known failure mode for video generation models [6, 9, 10, 16, 43]. The T2AV model, however, correctly generates the physical interaction, with the blade making contact with the stone, accompanied by a synchronized metallic scraping sound. This suggests that the audio signal acts as a strong regularizer, encouraging the model to generate videos that are more consistent with the physical laws of our world. Given that the motion and audio cannot be fully conveyed through static images, we strongly recommend viewing the complete audio-video results provided in the appendix.

#### 4.4. Ablation Study

Table 3. Ablation studies. IQ, SC, and PH are Image Quality, Subject Consistency, and Physics. FAD is in units of 104. All ablations are done on ALT-Merge.

(a) T2AV cross-modal attention.

Method FAD ↓ CLAP ↑ IQ ↑ SC ↑ PH ↑ Sync ↓

Cross-Attention 11.37 27.29 58.54 95.68 4.35 0.27 AVFull-Attention (Ours) 9.36 38.75 59.87 96.30 4.36 0.29

(b) T2AV RoPE design.

Method FAD ↓ CLAP ↑ IQ ↑ SC ↑ PH ↑ Sync ↓

Vanilla 8.93 44.46 57.76 94.05 4.22 0.32 Expand Video 8.81 43.09 56.34 93.57 4.20 0.26 Shrink Audio (Ours) 9.36 38.75 59.87 96.30 4.36 0.29

##### Cross-modal attention mechanism. We ablate the mod-

(a)

(b)

The first note The second note

T2AV Audio Track

T2AV Audio Track

Wind noise and rustling leaves

Windmill and trees swaying to the sound of the wind

T2AV Video Track

Up bow Down bow

T2AV Video Track

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

T2V Video Track Jittering bow

T2V Video Track Windmill spins with unnatural, frantic speed

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

(d)

T2AV Audio Track

The rasp of metal on stone

(c)

T2AV Audio Track

The roar of the falls

T2AV Video Track

Sharpening a knife on a stone

T2AV Video Track

Rushing waterfall

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|

T2V Video Track Knife glides above the sharpening stone

T2V Video Track Almost frozen video

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

- Figure 6. Qualitative comparison between T2AV and T2V. Video and audio track content are described using text. Motion in (c) is highlighted for clarity. Please refer to the appendix for real audio-video results.

eling of audio-video interaction. As an alternative to our proposed AVFull-Attention, we designed a Cross-Attention baseline. In this setup, following the unimodal selfattention in the later Nav layers, a cross-attention module is added, where each modality receives the other as its keyvalue input. As shown in Tab. 3a, AVFull-Attention surpasses the Cross-Attention baseline on all audio and video quality metrics, with only a minor trade-off in the synchronization score. This result demonstrates that AVFullAttention is a more effective approach for learning audiovideo joint denoising. We hypothesize this is because AVFull-Attention maximally preserves the architecture of the pre-trained models, elegantly and symmetrically unifying them through a single, powerful attention operation.

AVSyncRoPE design. We ablate the design of AVSyncRoPE. An alternative to shrinking the audio RoPE is to expand the video RoPE to match the audio’s. Tab. 3b compares these two approaches against the vanilla RoPE baseline. The results on video-related metrics (Image Quality, Subject Consistency, and Physics) show that our proposed method is clearly superior. Conversely, the “Expand Video” method performs best on audio metrics, followed by the vanilla baseline. We infer that modifying a modality’s RoPE shifts its features away from the distribution familiar to the pre-trained model, thereby degrading that modality’s performance. To maximize video quality, our primary objec-

tive, we choose to shrink the audio RoPE. Notably, both the “Expand Video” and “Shrink Audio” variants achieve significantly better synchronization than the vanilla baseline, confirming the validity of our motivation: to temporally align the audio and video positional encodings.

### 5. Conclusion

In this work, we pose a fundamental question: Does hearing help seeing? Specifically, we investigated whether joint audio-video denoising offers benefits to video generation. To systematically address this, we introduced AVFullDiT, a parameter-efficient architecture designed to maximize the leverage of knowledge from pre-trained T2V and T2A models. Our design incorporates a novel AVFullAttention mechanism and an AVSyncRoPE module, both validated for their effectiveness. Through rigorous experiments on two datasets of varying size and distribution, we obtain three main conclusions. 1) Despite a marginally higher video validation loss, T2AV training consistently enhances video generation across multiple metrics. 2) The T2AV model generates motion that is more realistic and tempered, avoiding the extremes of exaggerated or static scenes. 3) The T2AV model demonstrates an improved understanding of physical commonsense, more reliably generating plausible object interactions and avoiding common failure modes, such as “contact

avoidance.” In summary, our findings confirm that hearing does help seeing. Audio-video cross-modal joint denoising yields consistent improvements for video generation, paving the way for the development of stronger multimodal world models.

Acknowledgement. This work is supported by the National Key Research and Development Program of China (No. 2023YFC3807600).

### References

- [1] Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025. 2
- [2] Inclusion AI, Bowen Ma, Cheng Zou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, et al. Ming-flash-omni: A sparse, unified architecture for multimodal perception and generation. arXiv preprint arXiv:2510.24821, 2025. 2
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2
- [5] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025. 6
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Yufei Guo Will DePue, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Zeroscope, 2023. 1, 2, 7
- [7] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025. 2
- [8] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP, 2020. 5, 7, 15
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 1, 2, 7
- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024. 1, 2, 7

- [11] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2

- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

- 2024. 2

[13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583,

- 2025. 2

- [14] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2
- [15] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP, 2023. 6
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2024. 2, 7

- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 6
- [18] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024. 2
- [19] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization. arXiv preprint arXiv:2412.21037, 2024. 5, 13
- [20] Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP, 2024. 6
- [21] Masato Ishii, Akio Hayakawa, Takashi Shibuya, and Yuki Mitsufuji. A simple but strong baseline for sounding video generation: Effective adaptation of audio and video diffusion models for joint generation. arXiv preprint arXiv:2409.17550, 2024. 2
- [22] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fr\’echet audio distance: A metric for evaluating music enhancement algorithms. arXiv preprint arXiv:1812.08466, 2018. 6
- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2

- [24] Seung Hyun Lee, Gyeongrok Oh, Wonmin Byeon, Chanyoung Kim, Won Jeong Ryoo, Sang Ho Yoon, Hyunjun Cho, Jihyun Bae, Jinkyu Kim, and Sangpil Kim. Sound-guided semantic video generation. In ECCV, 2022. 5, 7
- [25] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 2
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023. 2
- [27] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025. 2
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [29] OpenAI. Sora 2, 2025. 1, 2
- [30] Andrew Owens, Phillip Isola, Josh McDermott, Antonio Torralba, Edward H Adelson, and William T Freeman. Visually indicated sounds. In CVPR, 2016. 5, 7
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [32] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 2
- [33] Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, et al. Muddit: Liberating generation beyond text-to-image with a unified discrete diffusion model. arXiv preprint arXiv:2505.23606, 2025. 2
- [34] Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, and Xiangtai Li. Dreamrelation: Bridging customization and relation generation. In CVPR, 2025. 2
- [35] Qingyu Shi, Jianzong Wu, Jinbin Bai, Jiangning Zhang, Lu Qi, Xiangtai Li, and Yunhai Tong. Decouple and track: Benchmarking and improving video diffusion transformers for motion transfer. In ICCV, 2025. 1
- [36] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 2
- [37] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, 2024. 2
- [38] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 2
- [39] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 5

- [40] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025. 2
- [41] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [42] Xihua Wang, Ruihua Song, Chongxuan Li, Xin Cheng, Boyuan Li, Yihan Wu, Yuyue Wang, Hongteng Xu, and Yunfeng Wang. Animate and sound an image. In CVPR, 2025. 2
- [43] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 1, 2, 7
- [44] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 1, 2
- [45] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2
- [46] Jianzong Wu, Xiangtai Li, Henghui Ding, Xia Li, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Betrayed by captions: Joint caption grounding and generation for open vocabulary instance segmentation. In ICCV, 2023.
- [47] Jianzong Wu, Xiangtai Li, Chenyang Si, Shangchen Zhou, Jingkang Yang, Jiangning Zhang, Yining Li, Kai Chen, Yunhai Tong, Ziwei Liu, et al. Towards language-driven video inpainting via multimodal large language models. In CVPR, 2024.
- [48] Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong. Diffsensei: Bridging multimodal llms and diffusion models for customized manga generation. In CVPR, 2025.
- [49] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [50] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 2
- [51] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 5
- [52] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2
- [53] Lehan Yang, Lu Qi, Xiangtai Li, Sheng Li, Varun Jampani, and Ming-Hsuan Yang. Unified dense prediction of video diffusion. In CVPR, 2025. 2

- [54] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. CoRR, 2024. 2
- [55] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025. 2
- [56] Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. Tarsier2: Advancing large vision-language models from detailed video description to comprehensive video understanding. arXiv preprint arXiv:2501.07888,

2025. 2, 5, 6

- [57] Lin Zhang, Shentong Mo, Yijing Zhang, and Pedro Morgado. Audio-synchronized visual animation. In ECCV, 2024. 5, 7
- [58] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2

### Does Hearing Help Seeing? Investigating Audio–Video Joint Denoising for Video Generation Supplementary Material

##### Overview.

- • Sec. A: Introduction video.
- • Sec. B: More qualitative results.
- • Sec. C: More implementation details.
- • Sec. D: User study between T2AV and T2V results.
- • Sec. E: More ablation studies.
- • Sec. F: Limitations and failure cases.
- • Sec. G: Broader impacts.
- • Sec. H: VGGSound subset details.

### A. Introduction Video

To help readers quickly grasp the primary idea of our work, we provide a 6-minute introduction video. Please refer to “introduction video.mp4” in the supplementary file.

### B. More Qualitative Results

Due to the large amount of qualitative results, we chose to present them in a separate, anonymous local HTML page. Please refer to “demo/index.html” in the supplementary file. On this project page, we provide extensive qualitative results, making a comprehensive comparison between T2AV results (with audio) and T2V results.

### C. More Implementation Details

The audio and video loss weights are set to 1:1 in all our experiments, except for ablation experiments. For all generations, we use the same random seed for T2AV and T2V. We observe that their results from the same prompts share similar visual layouts.

[Figure 43]

48.9% 17.0% 34.0%

55.1% 10.2% 34.7%

15.6% 68.9% 15.6%

42.0% 32.0% 26.0%

Figure 7. User Study. In all aspects of evaluation, T2AV is generally better than T2V.

### D. User Study

To provide a comprehensive qualitative assessment of video generation performance, we conduct a subjective user study

comparing T2AV against the T2V baseline. We randomly sampled 50 generation pairs from the ALT-Merge and VGGSound evaluation sets. Participants are asked to evaluate the videos based on four distinct criteria: Overall Preference, a general impression of the video without restriction to specific dimensions. Video Quality, the visual fidelity and clarity of the imaging. Instruction Following, the degree of alignment between the generated content and the video prompt. Physics & Commonsense, the plausibility of physical dynamics and interactions within the scene. To strictly isolate the visual improvements gained from joint training and exclude auditory bias, we remove the generated audio tracks from the T2AV outputs. Consequently, both T2AV and T2V samples were presented as silent videos. The presentation order is randomized and blinded to prevent evaluators from identifying the source model. We recruit a diverse panel of 10 evaluators varying in gender, age, and occupation; the majority had no professional background in AI, ensuring the feedback reflects a general user perspective. Results are aggregated by averaging responses across all participants. The results are summarized in Fig. 7. The T2AV model matches or outperforms the T2V baseline across all four dimensions. Notably, T2AV demonstrates a significant advantage in Overall Preference, Instruction Following, and Physics & Commonsense. This superiority in physical plausibility and prompt adherence strongly supports our hypothesis that audio serves as a strong regularizer for learning world dynamics. These subjective findings corroborate our quantitative metrics, confirming that hearing indeed helps seeing in video generation.

### E. More Ablation Studies

Table 4. Ablation on adjusting audio and video loss weights. λa and λv are audio and video loss weights, respectively. FAD is in units of 104. All ablations are done on ALT-Merge.

Loss Weights FAD ↓ CLAP ↑ IQ ↑ SC ↑ PH ↑ Sync ↓

λa = 0.1,λv = 1 10.01 35.09 56.11 95.05 4.37 0.34 λa = 0.3,λv = 1 9.95 36.56 58.87 95.71 4.36 0.35 λa = 1,λv = 1 (Ours) 9.36 38.75 59.87 96.30 4.36 0.29

Audio and video loss weight. To further enhance video learning, we investigate adjusting the relative loss weight. We test audio loss weights λa of 0.3 and 0.1, increasing the proportion of video loss in the total loss. The results in Tab. 4 indicate that reducing the audio loss weight leads to a significant degradation in audio-related metrics, including FAD and CLAP. Conversely, video quality metrics such

as IQ and SC do not show meaningful improvement and are highest when the weights are balanced. This finding aligns with our conclusion that incorporating audio in training aids video quality enhancement. Notably, the Physics metric experiences a marginal increase of 0.01 at an audio weight of 0.1. Still, it remains unchanged at 0.3, suggesting that the relative loss weight has a minimal impact on the model’s understanding of physical principles. The crucial factor appears to be the presence of audio information itself. Furthermore, the Synchronization metric deteriorates substantially as the audio loss weight is reduced, indicating that balanced loss weights are essential for achieving optimal audio-visual synchronization.

Table 5. Ablation on audio guidance scale in inference.

Scale FAD ↓ CLAP ↑ IQ ↑ SC ↑ PH ↑ Sync ↓

- 4.0 9.47 38.54 59.86 96.31 4.35 0.29

4.5 (Ours) 9.36 38.75 59.87 96.30 4.36 0.29

- 5.0 9.55 39.67 59.69 96.31 4.33 0.29

Guidance scale. We conduct an ablation study on the audio guidance scale for classifier-free guidance during inference. As shown in Tab. 5, varying the scale from 4.0 to 5.0 leads to only marginal fluctuations across all evaluation metrics. This finding highlights the stability of our model’s performance with respect to this inference hyperparameter. We ultimately adopt a scale of 4.5, which aligns with the default setting of the pre-trained TangoFlux model [19].

[Figure 44]

[Figure 45]

[Figure 46]

- (a)

Car shifting inaccurately

[Figure 47]

[Figure 48]

[Figure 49]

- (b)

Box shape changing

- Figure 8. Failure cases. (a) The car drifts to the side, instead of forward or backward. (b) The shape of the box changes.

### F. Limitations

While our work presents compelling evidence for the benefits of audio-video joint denoising, we acknowledge several limitations that offer avenues for future research. First, a notable limitation of our current T2AV model is its inability to generate clear and coherent human speech. While it can produce sounds indicative of human vocalization, the output lacks articulatory clarity. This is primarily due to the nature of our training data and annotations; the audio captions describe acoustic events (e.g., ”a woman speaking”) but do not contain verbatim transcriptions of spoken content. Consequently, the model learns to associate the visual

of a person talking with the general sound of speech, rather than specific words. Interestingly, despite this, we observe that the model sometimes generates muffled speech containing words semantically related to the visual context. This suggests that the model has begun to build a rudimentary association between specific scenes and relevant vocabulary, forming a partial world model of spoken language. Second, although our T2AV model demonstrates improved physical commonsense, it is not immune to generating content that occasionally violates physical laws. It still produces failure cases, as shown in Fig. 8. This is a recognized and persistent challenge across the broader field of video generation. However, it is crucial to emphasize that these instances of physical implausibility are less frequent and less severe than in the T2V-only counterpart, as supported by our quantitative and qualitative results. Therefore, this limitation does not undermine our central finding that incorporating an audio objective serves as a strong regularizer, leading to a significant overall improvement in physical realism compared to a video-only baseline.

### G. Broader Impacts

Our research demonstrates that incorporating an audio generation objective significantly improves the quality of video generation, particularly in depicting physically plausible dynamics and interactions. This finding has two main broader implications.

Informing the development of world models. The central finding—that “hearing helps seeing” suggests that building more comprehensive world models may depend on integrating a wider range of sensory inputs. Our work indicates that audio can serve as a privileged signal, providing crucial information about causality and physical interactions (e.g., collisions that produce sound) that is less ambiguous than visual data alone. This provides a compelling rationale for incorporating audio and other modalities into the development of more robust and physically grounded AI systems that can better comprehend and reason about the world.

Advancing unified multimodal architectures. This study contributes to the growing body of research on unified multimodal models. By demonstrating that one modality (audio) can serve as a regularizer to enhance the quality of generation of another (video), our work supports the hypothesis that different data streams can provide complementary supervisory signals. This insight can guide the development of more effective and efficient unified architectures, enabling knowledge transfer across modalities to achieve a more holistic understanding and generative capability.

### H. VGGSound Subset Details

As outlined in our experimental setup, we partition the VGGSound dataset into two distinct subsets, “AV-Tight”

and “AV-Loose,” to better evaluate our model’s handling of different degrees of audio-visual correspondence. This division, detailed in Tab. 6, is based on the causal relationship between the visual events and their corresponding sounds. The AV-Tight subset includes categories where the audio is directly and immediately linked to the visible action. In these instances, there is a strong, predictable correlation between what is seen and what is heard. For example, classes such as chopping wood, typing on computer keyboard, and striking pool feature sounds that are intrinsically linked to and caused by specific, observable physical interactions. In contrast, the AVLoose subset contains categories where the relationship between the audio and video is more ambient or contextual, rather than strictly causal and synchronized. While the sound and visuals co-occur, the sound is not necessarily produced by a single, discrete visual event. For instance, in categories like airplane flyby, raining, or wind noise, the audio represents the general acoustic environment rather than a sound tied to a specific, synchronized action. This distinction allows us to test the hypothesis that joint audio-video denoising provides the most significant benefits when the modalities are tightly coupled and mutually informative.

Table 6. List of AV-Tight and AV-Loose subsets in VGGSound [8].

AV-Tight Classes

arc welking bathroom ventilation fan running beat boxing cattle, bovinae cowbell chainsawing trees child speech, kid speaking chopping food chopping wood church bell ringing cutting hair with electric trimmers disc scratching driving motorcycle eletric grinder grinding eletric shaver, electric razor shaving eletric blender running firing cannon firing muskets forging swords metronome missile launch motorboat, speedboat acceleration ocean burbling opening or closing car eletric windows planing timber playing guiro playing saxophone printer printing ripping paper rope skipping sharpen knife singing bowl skateboarding spraying water strike lighter striking pool toilet flushing typing on computer keyboard

###### AV-Loose Classes

air conditioning noise air horn airplane airplane flyby alligators, crocodiles hissing ambulance siren baby crying baby laughter baltimore oriole calling barn swallow calling basketball bounce bee, wasp, etc, buzzing bird chirping, tweeting bird squawking blowtorch igniting bouncing on trampoline bowling impact bull bellowing canary calling cap gun shooting car engine idling car engine starting car passing by cat caterwauling cat mewwing cat purring cattle mooing cell phone buzzing cheetah chirrup chicken clucking chicken crowing child singing chimpanzee pant-hooting chinchilla barking chipmunk chirping civil defense siren cow lowing coyote howling cricket chirping crow cawing cuckoo bird calling cupboard opening or closing dinosaurs bellowing dog barking dog bow-bow dog growling dog howling donkey, ass braying driving buses driving snowmobile duck quacking eagle screaming eating with cutlery elephant trumpeting elk bugling engine accelerating, revving, vroom female singing female speech, woman speaking ferret dooking fire truck siren fly housefly buzzing foghorn footsteps on snow francolin calling frog croaking gibbon howling man speech, man speaking mosquito buzzing mouse pattering mouse squeaking mynah bird singing opening or closing drawers orchestra otter growling owl hooting parrot talking penguins braying people burping people hiccup people running prople screaming people sneezing pheasant crowing pig oinking pigeon, dove cooing plastic bottle crushing playing flute playing harpsichord playing washboard police car(siren) police radio chatter popping popcorn pumping water race car, auto racing railroad car, train wagon raining rapping reversing beeps roller coaster running rowboat, canoe,-kayak rowing running electric fan sailing scuba diving sea lion barking sea waves sheep bleating shot football singing choir skidding skiing sliding door sloshing water slot machine smoke detector beeping snake hissing snake rattling splashing water squishing water stream burbling striking bowling subway, metro, underground swimming tap dancing tapping guitar telephone bell ringing thunder tornado roaring tractor digging train horning train wheels squealing train whistling turkey gobbling underwater bubbling using sewing machines vacuum cleaner cleaning floors vehicle horn, car horn, honking volcano explosion warbler chirping waterfall burbling whale calling wind noise wood thrush calling woodpecker pecking tree writing on blackboard with chalk yodelling zebra braying

