## ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling

# arXiv:2603.25746v1[cs.CV]26Mar2026

Yawen Luo1,† Xiaoyu Shi2, Junhao Zhuang1 Yutian Chen1 Quande Liu2 Xintao Wang2 Pengfei Wan2 Tianfan Xue1,3,

1MMLab, CUHK 2Kuaishou Technology 3CPII under InnoHK Corresponding author {luoyw0207@gmail.com, xiaoyushi@link.cuhk.edu.hk, tfxue@ie.cuhk.edu.hk} https://luo0207.github.io/ShotStream/

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

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

Figure 1. Multi-Shot Video results of ShotStream. ShotStream is an autoregressive multi-shot video generation model enabling interactive storytelling and on-the-fly synthesis at 16 FPS on a single GPU. Each case presented here (rows 1–4) illustrates a generated sequence comprising five consecutive shots and 405 total frames, demonstrating the model’s capacity to maintain narrative and visual consistency across scene transitions. The expanded bottom row details the high visual quality achieved within a single shot. We highly encourage readers to view our project page for the video results.

### Abstract

Multi-shot video generation is crucial for long narrative storytelling, yet current bidirectional architectures suffer from limited interactivity and high latency. We propose ShotStream, a novel causal multi-shot architecture that en-

† Work done during internship at Kling Team, Kuaishou Technology.

ables interactive storytelling and efficient on-the-fly frame generation. By reformulating the task as next-shot generation conditioned on historical context, ShotStream allows users to dynamically instruct ongoing narratives via streaming prompts. We achieve this by first fine-tuning a text-to-video model into a bidirectional next-shot generator, which is then distilled into a causal student via Distribution

Matching Distillation. To overcome the challenges of intershot consistency and error accumulation inherent in autoregressive generation, we introduce two key innovations. First, a dual-cache memory mechanism preserves visual coherence: a global context cache retains conditional frames for inter-shot consistency, while a local context cache holds generated frames within the current shot for intra-shot consistency. And a RoPE discontinuity indicator is employed to explicitly distinguish the two caches to eliminate ambiguity. Second, to mitigate error accumulation, we propose a twostage distillation strategy. This begins with intra-shot selfforcing conditioned on ground-truth historical shots and progressively extends to inter-shot self-forcing using selfgenerated histories, effectively bridging the train-test gap. Extensive experiments demonstrate that ShotStream generates coherent multi-shot videos with sub-second latency, achieving 16 FPS on a single GPU. It matches or exceeds the quality of slower bidirectional models, paving the way for real-time interactive storytelling. Training and inference code, as well as the models, are available on our project page.

### 1. Introduction

While current text-to-video models [2, 19, 29, 34] excel at synthesizing high-fidelity single-shot videos, the field is rapidly advancing toward long-form narrative storytelling [10] akin to traditional film and television. This evolution necessitates multi-shot video generation, which enables the creation of sequential shots that maintain subject and scene consistency while advancing the narrative through varied content. For instance, cinematic techniques like the shot-reverse shot [11] create cohesive interactions by cutting back and forth between characters, effectively guiding the viewer’s attention through dynamic perspectives. Driven by the growing demand for such complex cinematic narratives, multi-shot video generation [1, 4, 27, 35, 42, 50] has gained increased attention.

Existing multi-shot video generation methods [4, 10, 14, 24, 27, 35, 37, 42] mainly rely on bidirectional architectures to model intra-shot and inter-shot dependencies, ensuring temporal and narrative consistency. Although effective, these bidirectional architectures suffer from two main limitations: 1) Lack of interactivity: Current methods require all prompts upfront to generate the entire multishot sequence at once, making it difficult to adjust individual shots without a complete re-generation. A more userfriendly approach would accept streaming prompt inputs at runtime, enabling users to interactively guide the narrative and adapt the current shot based on previously generated content. 2) High latency: The computational cost of bidirectional attention grows quadratically with context length, posing a major challenge for long sequences. Even with the

integration of sparse attention mechanisms to reduce overhead and accelerate generation, these models still exhibit prohibitive latency. For instance, HoloCine [24] requires approximately 25 minutes to generate a 240-frame multishot video.

To overcome these limitations, we propose ShotStream, a novel causal multi-shot architecture that enables interactive storytelling and efficient on-the-fly frame generation. To achieve interactivity, we reformulate multi-shot synthesis as an autoregressive next-shot generation task, where each subsequent shot is generated by conditioning on previous shots. This reformulation allows ShotStream to accept streaming prompts as inputs and generate videos shot-byshot, empowering users to dynamically guide the narrative at runtime by adjusting content, altering visual styles, or introducing new characters, as shown in Fig. 2.

To achieve this efficient causal architecture, we first train a bidirectional teacher model for next-shot prediction, conditioned on historical context. Because past shots comprise hundreds of frames, retaining the entire history introduces severe temporal redundancy and becomes memoryprohibitive. To address this, we condition the model on a sparse subset of historical frames rather than the entire sequence. Specifically, we introduce a dynamic sampling strategy that selects frames based on the number of preceding shots and specific conditional constraints, effectively preserving historical information within a strict frame budget. We then inject these sampled context frames by concatenating their context tokens with noise tokens along the temporal dimension to form a unified input sequence. This concatenation-based injection mechanism is highly parameter-efficient and eliminates the need for additional architectural modules.

Subsequently, we distill this slow, multi-step bidirectional teacher model into an efficient, 4-step causal student model via Distribution Matching Distillation [47, 48]. However, transitioning to this causal architecture introduces two primary challenges: 1) maintaining consistency across shots, and 2) preventing error accumulation to sustain visual quality during autoregressive generation. To address the first challenge, we introduce a novel dual-cache memory mechanism. A global context cache stores sparse conditional historical frames to ensure inter-shot consistency, while a local context cache retains frames generated within the current shot to preserve intra-shot continuity. Naively combining these caches introduces ambiguity, as the causal model struggles to differentiate between historical and current-shot contexts. To resolve this, we propose a RoPE discontinuity indicator that explicitly distinguishes between the global and local caches.

The second challenge, error accumulation, stems primarily from the train-test gap [12]. We mitigate this challenge by aligning training with inference through a pro-

|[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]| |
|---|---|
| | |

|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]| |
|---|---|
| | |

|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]| |
|---|---|
| | |

|[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]|[Figure 78]|
|---|---|
| | |

[Shot 1] A shielded superhero stands with a black-suited heroine in a city street.

[Shot 2] Close-up of the black-suited heroine with flames in the background.

[Shot 3] Rear view of the black-suited heroine facing toward the street chaos.

[Shot 4] Close-up of the shielded superhero with the burning street behind him.

##### Real-time, long, multi-shot video generation driven by streaming prompts.

| |Multi-Shot|Long Gen.|Interactive|Real-Time|
|---|---|---|---|---|
|Bidirectioal Multi-Shot Models|[Figure 79]|[Figure 80]|[Figure 81]|[Figure 82]|
|Long Autoregressive Models|[Figure 83]|[Figure 84]|[Figure 85]|[Figure 86]|
|ShotStream (Ours)|[Figure 87]|[Figure 88]|[Figure 89]|[Figure 90]|

- Figure 2. Overview of the ShotStream workflow, which enables real-time, long, multi-shot video generation from streaming prompts.

sistency, coupled with a RoPE discontinuity indicator to explicitly distinguish between the two caches.

posed two-stage progressive distillation strategy. We begin with intra-shot self-forcing conditioned on ground-truth historical shots, where the generator rolls out the current shot causally, chunk-by-chunk, to establish foundational next-shot capabilities. We then progressively transition to inter-shot self-forcing using self-generated histories. In this stage, the model rolls out the multi-shot video shot-byshot, while generating the internal frames of each individual shot chunk-by-chunk. This strategy bridges the train-test gap and significantly enhances the quality of autoregressive multi-shot generation.

• We propose a two-stage distillation strategy that effectively mitigates error accumulation by bridging the gap between training and inference to enable robust, longhorizon multi-shot generation.

### 2. Related Work

###### 2.1. Multi-Shot Video Generation

Driven by interest in narrative video generation, multi-shot video synthesis has advanced rapidly [4, 10, 14, 24, 27, 35, 37, 42, 52]. Current methods generally fall into two categories. Keyframe-based approaches [42, 51, 52] generate the initial frames of each shot and extend them using imageto-video models. However, they often struggle with global coherence, as consistency is enforced only at the keyframe level while intra-shot content remains isolated. The second paradigm, unified sequence modeling [4, 10, 14, 24, 27, 35], jointly processes all shots within a sequence. For instance, LCT [10] applies full attention across all shots using interleaved 3D position embeddings to distinguish them. While efficiency-focused variants like MoC [4] and HoloCine [24] employ dynamic or sparse attention patterns to reduce computational burden, they still suffer from high latency. Furthermore, their bidirectional architectures and unified modeling inherently limit interactivity, complicating the adjustment of specific shots within a generated sequence.

Extensive evaluations demonstrate that ShotStream generates long, narratively coherent multi-shot videos (as shown in Fig. 1) while achieving an efficient 16 FPS on a single NVIDIA H200 GPU. Quantitatively, our method achieves state-of-the-art performance on the test set regarding visual consistency, prompt adherence, and shot transition control. To complement these metrics with subjective evaluation, we conduct a user study involving 54 participants. Users are asked to compare 24 multi-shot videos generated by our method against those from baselines. The results reveal a decisive user preference for ShotStream in terms of visual consistency, overall visual quality, and prompt adherence.

In summary, our main contributions are as follows:

- • We present ShotStream, a novel causal multi-shot long video generation architecture that enables interactive storytelling and on-the-fly synthesis.
- • We reformulate multi-shot synthesis as a next-shot generation task to support interactivity, allowing users to dynamically adjust ongoing narratives via streaming prompts.
- • We design a novel dual-cache memory mechanism for our causal model to ensure both inter-shot and intra-shot con-

###### 2.2. Autoregressive Long Video Generation

Driven by next-token prediction objectives, autoregressive models naturally support the gradual rollout required for long video generation [3, 39, 40]. Recently, integrating autoregressive modeling with diffusion has emerged as a

Step 1: Bidirectional Next-Shot Teacher Training

Target Shot

DiT Block x N

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Shot 1]

- [Shot 1]

Sparse Conditional Context Frames

[Figure 96]

[Figure 97]

[Shot 1] [Shot 2]

[Figure 98]

[Figure 99]

- [Shot 2] [Shot 3] [Shot 3]

[Figure 100]

[Figure 101]

[Figure 102]

Patchify&Temporal TokenConcat

3D VAE Encoder

[Figure 103]

3DSelfAttention

CrossAttention

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Decoder

[Figure 109]

[Figure 110]

3DVAE

𝑧

FFN

[Figure 111]

[Figure 112]

[Figure 113]

Target Shot

[Figure 114]

[Figure 115]

𝑧

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Shot4] [Shot4] [Shot4] 3D VAE Encoder

add noise

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

𝑧

|Global Caption: A young boy and a girl share a sweet moment while planting a tree in the garden. [Shot 1]: A boy in a plaid shirt holds a shovel. [Shot 2]: A close-up of a girl smiling shyly. [Shot 3]: The boy and girl kneel to plant a tree. [Shot 4]: A close-up of the boy smiling warmly at her.|
|---|

Trainable Module Frozen Module 𝑧 Context Latent 𝑧 Noise Latent

[Figure 126]

[Figure 127]

[Figure 128]

umT5 Encoder

- Figure 3. Architecture of the Bidirectional Next-Shot Teacher Model. To realize ShotStream, we first fine-tune a text-to-video model into a bidirectional next-shot model, which generates subsequent shots conditioned on sparse context frames from preceding shots. These conditional context frames are encoded into latents via a 3D VAE and injected by concatenating them with noise latents along the temporal dimension. Notably, only the 3D spatial-temporal attention layers within the DiT Blocks are optimized during fine-tuning. A 4-shot example is shown here for illustration.

###### 3.2. Self Forcing

promising paradigm for causal, high-quality video synthesis [6, 9, 12, 20, 22, 33, 43, 45, 46, 49]. Methods like CausVid [49] achieve low-latency streaming by distilling multi-step diffusion into a 4-step causal generator. To mitigate exposure bias from the train-test sequence length discrepancy, Self Forcing [12] and Rolling Forcing [20] condition generation on self-generated outputs and progressive noise levels, respectively, to suppress error accumulation. Additionally, LongLive [43] enables dynamic runtime prompting via a KV-recache mechanism. Despite these advancements, existing techniques are largely confined to single-scene generation and struggle with multi-shot narratives. Our method addresses this gap, extending autoregressive modeling to generate cohesive, multi-shot narrative videos.

Error accumulation [25, 30] remains a persistent challenge in autoregressive video generation, caused by the discrepancy between using ground-truth data during training and relying on imperfect predictions during inference. To bridge this train-test gap, self forcing [12] introduces a training paradigm that explicitly unrolls the autoregressive process. By conditioning each frame on previously generated outputs rather than ground-truth frames, the model is compelled to navigate and recover from its own inaccuracies. Consequently, self forcing [12] effectively mitigates exposure bias and stabilizes long generation.

### 4. Method

This section details the architecture and training methodology of ShotStream. We first fine-tune a text-to-video model into a bidirectional next-shot model (Sec. 4.1). This model is subsequently distilled into an efficient, 4-step causal model via Distribution Matching Distillation. We also propose a novel dual-cache memory mechanism and a two-stage distillation strategy to enable efficient, robust, and long-horizon multi-shot generation (Sec. 4.2).

### 3. Preliminary

###### 3.1. Distribution matching distillation

Distribution Matching Distillation (DMD) [47, 48] distills slow, multi-step diffusion models into fast, few-step student generators while maintaining high quality. The key objective is to match the student and teacher at the distribution level by minimizing the reverse KL divergence between the smoothed data distribution, pdata, and the student generator’s output distribution, pgen. This optimization is performed across random timesteps t, where the gradient is approximated by the difference between two score functions: one trained on the true data distribution and another trained on the student generator’s output distribution using a denoising loss. Detailed in the Sec. 8 of the Supplementary Material.

###### 4.1. Bidirectional Next-Shot Teacher Model

The objective of the next-shot model is to generate a subsequent shot conditioned on historical shots. Since historical shots contain hundreds of frames with high visual redundancy, retaining the entire history is unnecessary and impractical with a limited conditional budget. Therefore, we condition the model on sparse context frames extracted via a dynamic sampling strategy. Specifically, given Shist historical shots and a maximum conditional context budget of fcontext frames, we sample ⌊fcontext/Shist⌋ frames from each historical shot, where ⌊·⌋ denotes the floor function. Any re-

#### Step 2.1: Intra-Shot Self-Forcing Distillation

|Real Score<br><br>[Figure 129]<br><br>Fake Score<br><br>[Figure 130]<br><br>DMD Loss|
|---|

sample

Conditional Context Frames

|GT Historical| |
|---|---|
|Shots| |

[Figure 131]

[Figure 132]

[Figure 133]

add noise

Causal Generator

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Noise

|[Figure 141]|
|---|

𝑧

|Global Cache (Conditional Frames)<br><br>| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
<br><br>Local Cache (Target Shot Frames)<br><br>Unmasked Chunk Masked Chunk<br><br>| |
|---|
<br><br>Dual-Cache Mechanism|
|---|

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

#### Step 2.2: Inter-Shot Self-Forcing Distillation

|Iteration N<br><br>Conditional Context Frames<br><br>[Figure 146]<br><br>[Figure 147]<br><br>Noise<br><br>|Self-Generated (N-1)th Shot|
|---|
<br><br>update Nth Shot<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>DMD Loss<br><br>|[Figure 151]|
|---|
<br><br>[Figure 152]<br><br>[Figure 153]<br><br>Lora<br><br>[Figure 154]<br><br>|Causal Generator|
|---|
|
|---|

||
|---|

||
|---|

Self-Forcing

|Conditional Context Frames<br><br>[Figure 157]<br><br>[Figure 158]<br><br>Noise<br><br>|Self-Generated Nth Shot|
|---|
<br><br>update (N+1)th Shot<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>Iteration N+1<br><br>DMD Loss<br><br>|[Figure 162]|
|---|
<br><br>[Figure 163]<br><br>[Figure 164]<br><br>Lora<br><br>[Figure 165]<br><br>|Causal Generator|
|---|
|
|---|

Gradient Computation

|Iteration N+1<br><br>Sparse Global Context Frames<br><br><br><br><br><br>Noise<br><br>|Causal Generator Lora<br><br>|
|---|
<br><br>|(N+1)th Shot| |
|---|---|
| | |
<br><br>Gradient Computation<br><br>|Self-Generated Nth Shot|
|---|
<br><br>update|
|---|

- Figure 4. Causal Architecture and Two-Stage Distillation Pipeline. We distill a slow, multi-step bidirectional teacher into an efficient, few-step causal generator. To maintain visual coherence, we propose a novel dual-cache memory mechanism: a global context cache stores conditional frames to ensure inter-shot consistency, while a local context cache retains generated frames within the target shot to guarantee intra-shot consistency. To prevent error accumulation, we employ a progressive two-stage distillation strategy. In the first stage, intra-shot self-forcing distillation (Step 2.1), the model is conditioned on ground-truth historical shots to causally generate the current shot chunk-bychunk. In the second stage, inter-shot self-forcing distillation (Step 2.2), the model is conditioned on its own previously generated shots, rolling out the video shot-by-shot while iteratively generating the frames of each individual shot chunk-by-chunk.

Step 2: Causal Few-step Next-Shot Student Distillation

maining budget is allocated to the most recent shot to fully utilize the budget, which is set to 6 frames in our experiments.

binds past visual information to textual descriptions. This binding facilitates the extraction of necessary context for generating the subsequent shot. Therefore, we also inject the specific captions corresponding to each conditional context frame into the model, i.e., the frames of each shot attend to both the global caption and the corresponding local shot caption via cross-attention. Specifically, as shown in Fig. 3, our next-shot model reuses the 3D VAE ε from the base model to transform Vcontext into conditioning latents,

Stage 1: Bidirectional Next-Shot Teacher Training

To condition the model on sampled sparse context frames Vcontext, we employ a temporal token concatenation mechanism, an injection technique proven effective across multi-control generation [15], editing [44], and camera motion cloning [23]. Although effective, these methods do not distinguish between the captions of condition frames and target frames; instead, they uniformly apply the target frame’s caption to the condition frames. Directly adopting this approach for next-shot generation is problematic, as the captions of previous shots contain crucial information that

- [Shot 1]

Sparse Global Context Frames

[Shot 1] [Shot 2]

- [Shot 2] [Shot 3] [Shot 3]

Patchify Token

3D VAE Encoder

zcontext = ε(Vcontext), (1) where zcontext ∈ Rf

context×c×h×w comprises fcontext frames, c

channels, and a spatial resolution of h × w. Building upon this shared latent space, we first patchify the condition latent zcontext and the noisy target latent zt ∈ Rf×c×h×w with f frames into tokens:

xj = Patchify(zj),zj ∈ {zcontext,zt}. (2)

The resulting condition tokens xcontext and noisy video tokens xt are then concatenated along the frame dimension to form the input for the DiT blocks:

xinput = FrameConcat(xcontext,xt). (3)

The notation FrameConcat denotes that the condition tokens are concatenated with the noise tokens along the frame dimension. Given that the token sequences xcontext ∈ Rb×f

context×s×d and xt ∈ Rb×f×s×d share the same batch size b, spatial token count s of each frame, and feature dimension d, this temporal concatenation yields a combined tensor xinput ∈ Rb×(f

context+f)×s×d. During the training process, noise is added exclusively to the target video tokens, keeping the context tokens clean. This design enables the DiT’s native 3D self-attention layers to directly model interactions between the condition and noise tokens, without introducing new layers or parameters to the base model.

###### 4.2. Causal Architecture and Distillation

The bidirectional next-shot teacher model (detailed in Sec. 4.1) requires approximately 50 denoising steps, resulting in high inference latency. To enable low-latency generation, we distill this multi-step teacher into an efficient 4step causal generator. However, transitioning to this causal architecture introduces two primary challenges: 1) maintaining consistency across shots, and 2) preventing error accumulation to sustain visual quality during autoregressive generation. To address these issues, we propose two key innovations: a dual-cache memory mechanism and a twostage distillation strategy, respectively.

Dual-Cache Memory Mechanism. To maintain visual coherence, we introduce a novel dual-cache memory mechanism (Fig. 4): a global cache stores sparse conditional frames to preserve inter-shot consistency, while a local cache retains recently generated frames to ensure intra-shot consistency. However, querying both caches simultaneously within our chunk-wise causal architecture introduces temporal ambiguity, as the model struggles to distinguish historical from current-shot contexts. To address this, we propose a discontinuous RoPE strategy that explicitly decouples the global and local contexts by introducing a discrete temporal jump at each shot boundary. Specifically, for the t-th latent zt within the k-th shot, its temporal rotation angle is formulated as Θt = ϕt + kθ, where ϕ denotes the base temporal frequency and θ serves as the phase shift representing the shot-boundary discontinuity.

Two-stage Distillation Strategy. A major challenge in autoregressive multi-shot video generation is error accumulation caused by the training-inference gap [12]. To mitigate this, we propose a two-stage distillation training strategy. In the first stage, intra-shot self-forcing (Fig. 4, Step 2.1), the model samples global context frames from groundtruth historical shots while the chunk-wise causal generator produces the target shot via a temporal autoregressive rollout. Specifically, the local cache utilizes previously selfgenerated chunks from the current target shot rather than ground-truth data. Although this stage establishes foundational next-shot generation capabilities, a training-inference gap remains: during inference, the model must condition on its own potentially imperfect historical shots instead of the ground truth. To bridge this gap, we introduce the second stage: inter-shot self-forcing (Fig. 4, Step 2.2). Specifically, the causal model generates the initial shot from scratch and applies DMD. For all subsequent iterations, the generator synthesizes the next shot conditioned entirely on prior self-generated shots. During each iteration, the model continues to employ intra-shot self-forcing to generate each new shot chunk by chunk, applying DMD exclusively to the newly generated shot. This autoregressive unrolling continues until the entire multi-shot video is generated. By closely mirroring the inference-time rollout, this stage aligns training and inference, effectively mitigating error accumulation and enhancing overall visual quality.

Inference. The inference procedure of ShotStream identically aligns with its training process. ShotStream generates multi-shot videos in a shot-by-shot manner. As each new shot is generated, the global context frames are updated by sampling from previously synthesized historical shots. Within the current shot, video frames are generated sequentially chunk by chunk, leveraging our causal few-step generator and KV caching to ensure computational efficiency.

### 5. Experiments

###### 5.1. Experiment Setup

Implement Details. We build ShotStream upon Wan2.1T2V-1.3B [34] to generate 832 × 480 video clips. The bidirectional next-shot teacher is trained on an internal dataset of 320K multi-shot videos. For causal adaptation, the student model is initialized via regression on 5K teachersampled ODE solution pairs [49]. Distillation proceeds in two stages: intra-shot self-forcing using ground-truth historical shots from the dataset, followed by inter-shot selfforcing using captions from a subset of 5-shot videos. Architecturally, the model operates with a chunk size of 3 latent frames, utilizing a global cache of 2 chunks and a local cache of 7 chunks. We refer readers to the Sec. 9 in the Supplementary Material for further details.

Evaluation Set. To comprehensively evaluate multi-shot

- Table 1. Quantitative results for multi-shot video generation. The best results are highlighted in boldface, while the second best are underlined. Here, Sub., Bg., Cons., and Align. denote Subject, Background, Consistency, and Alignment, respectively.

Intra-shot Cons. Inter-shot Cons. Trans.

Text Align.↑

Aesthetic Quality ↑

Dynamic

Method Architecture FPS

Degrees ↑ Sub. ↑ Bg. ↑ Semantic ↑ Sub. ↑ Bg. ↑

Control↑

Mask2DiT [27] Bidirectional 0.149 0.646 0.679 0.711 0.612 0.534 0.513 0.184 0.520 48.91 EchoShot [35] Bidirectional 0.643 0.772 0.739 0.596 0.392 0.396 0.664 0.186 0.543 65.92 CineTrans [41] Bidirectional 0.413 0.776 0.797 0.459 0.412 0.459 0.572 0.170 0.513 59.47

Self Forcing [12] Causal 16.36 0.737 0.707 0.738 0.542 0.445 0.633 0.214 0.512 55.45 LongLive [43] Causal 16.55 0.758 0.792 0.722 0.594 0.565 0.693 0.216 0.565 58.45 Rolling Forcing [20] Causal 15.32 0.725 0.781 0.758 0.561 0.473 0.684 0.223 0.523 62.26 Infinity-RoPE [45] Causal 16.37 0.752 0.738 0.622 0.453 0.407 0.715 0.209 0.513 63.40

ShotStream (Ours) Causal 15.95 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.56

video generation capabilities, following previous work [24, 35, 37, 41], we leverage Gemini 2.5 Pro [8] to generate 100 diverse multi-shot video prompts. To ensure a fair comparison, we tailor these text prompts to match the specific input style of each baseline model. These test prompts cover a wide range of themes, enabling a robust measurement of the models’ ability to maintain consistency across different scenes.

Evaluation Metrics. Before computing metrics, we use the pretrained TransNet V2 [32] to detect shot boundaries in each video. We evaluate the model’s multi-shot performance across five key dimensions: 1) Intra-Shot Consistency: Following HoloCine [24], we compute Subject Consistency using DINO [5] cosine similarities and Background Consistency using CLIP [28] similarities across frames. 2) Inter-Shot Consistency: Following MultiShotMaster [37], we isolate subjects and backgrounds from keyframes using YOLOv11 [16] and SAM [18], then assess consistency via DINOv2 [26]. Semantic Consistency is measured by computing the cosine similarity of ViCLIP [38] features extracted from each shot. 3) Transition Control: We utilize the Shot Cut Accuracy (SCA) metric [24] to evaluate the model’s transition control capabilities by measuring the accuracy of cut counts and their temporal precision. 4) Prompt Following: We use Text Alignment [36, 37] to measure video-text consistency. 5) Overall Quality: We report Aesthetic Quality and Dynamic Degrees from VBench [13] to assess visual quality.

Baselines. We compare our model with relevant opensource video generation models of similar scale, including two categories: 1) Bidirectional Multi-Shot Video Generation Model: Mask2DiT [27] employs symmetric binary masks within the MMDiT architecture to isolate text annotations per segment while maintaining temporal coherence for multi-scene generation. EchoShot [35] targets portrait

customization by using shot-aware position embeddings to model inter-shot relationships. CineTrans [41] designs a mask-based control mechanism for shot transitions. 2) Autoregressive and Interactive Long Video Generation Model: Baselines include Self Forcing [12], LongLive [43], Rolling Forcing [20], and Infinity-RoPE [45]. LongLive [43] utilizes a KV-recache mechanism to refresh states with new prompts, enabling interactive generation. InfinityRoPE [45] introduces a training-free RoPE Cut for multiscene transitions in continuous rollouts. Although these autoregressive models can generate videos of several hundred frames, their multi-shot generation capabilities remain limited.

Quantitative Results. We validate ShotStream using the evaluation set described in Sec. 5.1. As shown in Table 1, our model outperforms competing methods across major metrics. It achieves the highest visual consistency while maintaining precise control over shot transitions, reflected by higher Consistency and Transition Control scores. Additionally, our method demonstrates superior prompt alignment for individual shots and higher overall aesthetic quality. We also evaluate inference efficiency of all methods using a single H200 GPU. Compared to bidirectional models, our approach yields more than 25× improvement in throughput (FPS). It also enables autoregressive long multishot video generation with minimal speed degradation relative to causal long-video models.

Qualitative Results. In Fig. 5, we provide a qualitative comparison on a complex, narrative-driven multi-shot prompt to illustrate the superiority of our method. Baseline methods, including Mask2DiT [27], CineTrans [41], Self Forcing [12], and Rolling Forcing [20] fail to generate shots that align with their respective prompts. While EchoShot [35] and Infinity-RoPE [45] successfully adapt to individual shot instructions, they struggle with inter-shot

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

Mask2DiT

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

EchoShot

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

CineTrans

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

SelfForcing

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

LongLive

RollingForcing

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

Infinite-RoPE

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

Ours

[Shot 1] A serious blonde woman in red glasses speaks while looking away.

[Shot 2] An East Asian woman wearing a large necklace speaks and smiles confidently.

[Shot 3] The woman in glasses looks up with a surprised expression.

[Shot 4] Both women stand across a table in a office guarded by security men.

[Shot 5] The woman in glasses continues speaking to make her point.

- Figure 5. Qualitative Comparison. We present the initial frames of each shot generated by all compared methods. Our approach not only adheres strictly to the prompts and maintains high visual coherence, but also produces natural transitions between shots.

###### 5.2. User Study

consistency. LongLive [43] confuses the identities of the two women appearing throughout the sequence. In contrast, our method faithfully adheres to multi-shot prompts while achieving high visual consistency and coherence with smooth transitions.

Due to the subjective nature of evaluating multi-shot video generation, we conducted a user study to compare different methods and validate the perceptual advantages of our proposed ShotStream. We randomly sampled 24 multi-shot

- Table 2. User Preference Rate. Participants select their preferred video from a randomized set of results generated by all methods. Multiple selections are allowed.

Method Mask2DiT EchoShot CineTrans Self Forcing LongLive Rolling Forcing Infinity-RoPE Ours

Visual Consistency 3.08% 12.31% 6.21% 1.54% 12.31% 15.38% 16.92% 87.69% Prompt Following 0.83% 3.08% 1.54% 10.77% 16.15% 16.15% 14.62% 76.15% Visual Quality 7.69% 18.46% 16.92% 10.77% 18.46% 23.08% 15.38% 83.08%

Table 3. Ablation Study on the Bidirectional Next-Shot Teacher Model Design.

Intra-shot Cons. Inter-shot Cons. Trans.

Dynamic Sub. ↑ Bg. ↑ Semantic ↑ Sub. ↑ Bg. ↑ Degrees ↑

Text Align. ↑

Aesthetic Quality ↑

Method

Control ↑

Context Frames Sampling Strategy

First Only 0.789 0.793 0.671 0.618 0.612 0.956 0.212 0.502 61.55 First & Last 0.809 0.827 0.709 0.629 0.638 0.969 0.223 0.528 62.08 Dynamic (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

Condition Frames Captioning Strategy

Target Caption 0.804 0.818 0.681 0.609 0.572 0.937 0.194 0.422 62.86 Multi-Captions (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

Condition Injection Mechanism

Channel Concat 0.814 0.802 0.743 0.628 0.608 0.912 0.223 0.509 61.43 Frame Concat (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

Training Strategy

Full 0.816 0.810 0.749 0.631 0.624 0.969 0.227 0.546 60.85 Only 3D (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

prompts from the evaluation set (detailed in Sec. 5.1). During the test, participants are simultaneously presented with eight videos displayed in a randomized order: one generated by our method and seven from the competing baselines. Participants are asked to evaluate the videos from three aspects: Visual Consistency, Prompt Following, and Visual Quality. Multiple selections are allowed for each question. The user study involves 54 participants, and the results in Table 2 indicate that our method is consistently preferred by most users.

###### 5.3. Ablation Studies

We perform ablation studies to validate the key design choices and training strategies of the bidirectional next-shot teacher and causal student models.

Bidirectional Next-Shot Teacher Model Design. As shown in Table 3, we validate our design choices across four key aspects: 1) Context Frame Sampling Strategy: To ex-

tract sparse context frames from historical shots, our model adopts a dynamic sampling strategy based on the number of historical shots to fulfill the conditional context budget. This aims to maximize the retention of historical information within a constrained budget. This dynamic approach outperforms naive baselines, such as sampling only the first frame or the first and last frames of each historical shot. 2) Condition Frames Captioning Strategy: To validate the necessity of injecting specific prompts for historical shots, we compare our approach against the widely used baseline [15, 23, 44] of using the same caption for both condition and target frames. Results indicate that injecting the corresponding prompts for condition frames is necessary and benefits in-context learning by effectively binding historical visual content with its respective text. 3) Condition Injection Mechanism: We evaluate our approach of injecting condition frames via temporal token concatenation against the commonly used channel concatenation method [31].

Table 4. Ablation Study on the Causal Student Model Design and Training Strategy.

Intra-shot Cons. Inter-shot Cons. Trans.

Dynamic Sub. ↑ Bg. ↑ Semantic ↑ Sub. ↑ Bg. ↑ Degrees ↑

Text Align. ↑

Aesthetic Quality ↑

Method

Control ↑

Dual-Cache Distinction Strategy

w/o Indicator 0.813 0.816 0.728 0.507 0.465 0.967 0.203 0.549 63.09 Learnable Emb. 0.811 0.809 0.737 0.518 0.588 0.972 0.204 0.523 61.19 RoPE Offset (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

Causal Distillation Training Strategy

- Stage 1 Only 0.803 0.809 0.758 0.604 0.622 0.976 0.224 0.568 59.66

- Stage 2 Only 0.819 0.814 0.704 0.583 0.547 0.975 0.218 0.467 52.86 Two Stage (Ours) 0.825 0.819 0.762 0.654 0.645 0.978 0.234 0.571 63.06

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

Indicator Learnable

w/o

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

Emb.

- Stage1

Only

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

- Stage2

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

Only

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

Ours

[Shot 1] A winged creature looms from a truck at dusk.

[Shot 2] A truck faces a distant winged creature in the ruins.

[Shot 3] From inside the car, the creature approaches with wings spread.

[Shot 4] Inside the car, a passenger watches the threat outside.

[Shot 5] The winged creature stands against the sunset skyline.

Figure 6. Qualitative Ablation Results for the Causal Student Model. Please refer to project page for video comparisons.

Causal Student Model Design. Table 4 presents ablations on the model design and distillation strategy of our causal model. 1) Dual-Cache Distinction Strategy: To justify the need to separate global from local caches, we compare our proposed RoPE offset (Row 3) against a baseline with no

This proves superior by allowing the 3D self-attention layers to directly model the relationships between condition and target tokens. 4) Training Strategy: We demonstrate that fine-tuning only the 3D self-attention layers outperforms full-parameter fine-tuning.

distinction (Row 1) and a variant using a learnable embedding applied to the target video’s first chunk (Row 2). The results demonstrate that explicit distinction is essential (Row 1 vs. Row 3), and our training-free RoPE offset outperforms the learnable embedding approach (Row 2 vs. Row 3). 2) Causal Distillation Training: We evaluate our two-stage distillation strategy against single-stage baselines (Rows 4 and 5). Both stages prove indispensable: stage 1 establishes foundational next-shot generation capabilities, while stage 2 faithfully simulates inference to bridge the train-test gap. Furthermore, the qualitative results in Fig. 6 reinforce the necessity of both the RoPE offset and the two-stage distillation. Notably, the inter-shot self-forcing distillation significantly improves long-term consistency in visual style and color across the video (Stage 1 Only vs. Ours).

### 6. Conclusion

In this paper, we introduce ShotStream, a novel causal multi-shot video generation architecture that enables interactive long narrative storytelling while achieving 16 FPS on a single GPU. Our core contributions include reformulating the next-shot generation task for streaming, training a bidirectional next-shot teacher model, and distilling it into a causal architecture via a proposed two-stage distillation strategy. Additionally, we introduce a novel dual-cache memory mechanism to ensure visual consistency. Compared to existing bidirectional multi-shot models, ShotStream significantly reduces generation latency and supports streaming prompt inputs at runtime. This empowers users to interactively guide the narrative, adapting upcoming shots based on previously generated content. Furthermore, ShotStream advances the capabilities of autoregressive long video generation models by extending their ability to generate cohesive multi-shot sequences, paving the way for real-time, interactive, long-form storytelling.

Limitations and Future Work. While ShotStream is effective for autoregressive multi-shot video generation, we identify two primary limitations. First, we observe visual artifacts and inconsistencies when scenes and text prompts are highly complex. This primarily stems from the limited capacity of our backbone; because our current models are relatively small, we expect that scaling up the base model will improve performance and stability in challenging scenarios. Second, although our method is efficient, it still has room for acceleration to provide better interactive experiences. Techniques such as sparse attention [14, 53] and attention sink [31, 43] could be integrated into our model to achieve faster generation. We leave these extensions to future research.

### References

- [1] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, Chenyang Zhang, Tao Xiang, Fanny Yang, Serge Belongie, and Tian Xie. Onestory: Coherent multi-shot video generation with adaptive memory, 2025. 2
- [2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators,

2024. 2

- [3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 3
- [4] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, and Gordon Wetzstein. Mixture of contexts for long video generation, 2025. 2, 3
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 7
- [6] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreelsv2: Infinite-length film generative model, 2025. 4
- [7] Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151,

2023. 1

- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 7
- [9] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 4
- [10] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17281–17291, 2025. 2, 3
- [11] Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Qiao Yu, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next shot via in-context tuning. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025. 2

- [12] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 4, 6, 7
- [13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7
- [14] Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao. Moga: Mixture-of-groups attention for end-toend long video generation, 2025. 2, 3, 11
- [15] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907, 2025. 5, 9
- [16] Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024. 7
- [17] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 7
- [19] Kuaishou. Kling video model. https://kling. kuaishou.com/en, 2024. 2
- [20] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 4, 7
- [21] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. 1
- [22] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678,

2025. 4

- [23] Yawen Luo, Xiaoyu Shi, Jianhong Bai, Menghan Xia, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Camclonemaster: Enabling reference-based camera control for video generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–10, 2025. 5, 9
- [24] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, Yujun Shen, and Huamin Qu. Holocine: Holistic generation of cinematic multi-shot long video narratives, 2025. 2, 3, 7
- [25] Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in diffusion models. arXiv preprint arXiv:2308.15321, 2023. 4
- [26] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 7
- [27] Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, and Yongdong Zhang. Maskˆ 2dit: Dual mask-based diffusion transformer for multi-scene long video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18837–18846, 2025. 2, 3, 7
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7
- [29] RunwayML. Gen-3 alpha. https://runwayml.com/ research/introducing-gen-3-alpha, 2024. 2
- [30] Florian Schmidt. Generalization in generation: A closer look at exposure bias. arXiv preprint arXiv:1910.00292, 2019. 4
- [31] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025. 9, 11
- [32] Tom´as Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218–11221, 2024. 7
- [33] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 4
- [34] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 6, 1
- [35] Jiahao Wang, Hualian Sheng, Sijia Cai, Weizhan Zhang, Caixia Yan, Yachuang Feng, Bing Deng, and Jieping Ye. Echoshot: Multi-shot portrait video generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2, 3, 7
- [36] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 7
- [37] Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. arXiv preprint arXiv:2512.03041,

2025. 2, 3, 7

- [38] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 7

- [39] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 3
- [40] Dirk Weissenborn, Oscar T¨ackstr¨om, and Jakob Uszkoreit. Scaling autoregressive video models. arXiv preprint arXiv:1906.02634, 2019. 3
- [41] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models. arXiv preprint arXiv:2508.11484, 2025. 7
- [42] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation, 2025. 2, 3
- [43] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622,

2025. 4, 7, 8, 11

- [44] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 5, 9
- [45] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649,

2025. 4, 7

- [46] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025. 4
- [47] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 2, 4, 1
- [48] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2, 4
- [49] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast causal video generators. arXiv eprints, pages arXiv–2412, 2024. 4, 6, 1
- [50] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory, 2025. 2
- [51] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: Step-by-step generating multi-shot video with minimal manual intervention. arXiv preprint arXiv:2412.02259, 2024. 3

- [52] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. Advances in Neural Information Processing Systems, 37: 110315–110340, 2024. 3
- [53] Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards realtime diffusion-based streaming video super-resolution. arXiv preprint arXiv:2510.12747, 2025. 11

## ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling Supplementary Material

In this supplementary file, we provide the following materials:

- • Text-to-Video base model.
- • Distribution Matching Distillation objective function.
- • Training details.

### 7. Text-to-Video Base Model

Our proposed model, ShotStream, builds upon the transformer-based latent diffusion architecture, Wan2.1T2V-1.3B [34]. The architecture integrates a 3D Variational Auto-Encoder (VAE) [17] for latent feature mapping alongside a sequence of transformer blocks responsible for modeling temporal dynamics. Each basic transformer block is composed of 3D spatial-temporal attention, cross-attention, and feed-forward network (FFN). The input text is encoded by the umT5 encoder εumT5 [7] and injected into the architecture through cross-attention layers. The model utilizes the Rectified Flow [21] framework to train the diffusion transformer, such that we can generate a data sample x from a starting Gaussian sample z ∈ N(0,I). Specifically, for a data point x, its noised version xt at timestep t is constructed as xt = (1 − t)x + tz. The training objective is a simple MSE loss:

LRF(θ) = Et,x,z ∥vθ(xt,t,ctext) − (z − x)∥22 , (4) where the velocity vθ is parameterized by a network θ.

### 8. Distribution Matching Distillation Objective

As discussed in the main text, DMD minimizes the reverse KL divergence between the smoothed data distribution and the student generator’s output distribution. The gradient for this optimization is approximated by the difference between two score functions:

∇ϕLDMD ≜ Et ∇ϕKL pgen,t∥pdata,t

≈ −Et sdata Ψ Gϕ(ϵ),t ,t

− sgen,ξ Ψ Gϕ(ϵ),t ,t

dGϕ(ϵ) dϕ

dϵ ,

(5)

where Ψ represents the forward diffusion process, ϵ is random Gaussian noise, Gϕ is the student generator parameterized by ϕ, and sdata and sgen,ξ denote the score functions trained on the data and the student generator’s output distributions, respectively, using a denoising loss (Eq. 4).

### 9. Training Details

Dataset and Preprocessing. We utilize an internally curated dataset of 320K multi-shot videos, with each video comprising 2 to 5 shots and totaling up to 250 frames. Each sample is annotated with hierarchical prompts: a global caption describing the narrative arc, characters, and visual style, and shot-level captions detailing specific actions and content within each segment.

Bidirectional Teacher Training. We optimize only the 3D self-attention layers within the DiT blocks. Training is conducted for 10,000 steps using the Adam optimizer with a learning rate of 1e − 5 and a batch size of 64.

Causal Adaptation Initialization. Following the CausVid protocol [49], we initialize the student model with teacher weights and adapt it to a causal attention architecture. We sample 5K ODE solution pairs from the teacher and train all model parameters for 2,000 steps with a learning rate of 1e − 6 and a batch size of 64. This alignment on ODE trajectories bridges the gap between the bidirectional teacher and causal student, stabilizing subsequent distillation.

Causal Distillation. Distillation proceeds in two stages:

- • Stage 1: Intra-shot Self-forcing. This stage employs ground-truth historical shots from our dataset. The model is trained to predict the immediate next shot conditioned on perfect history. This stage converges quickly around 500 steps with a batch size of 32. Following DMD2 [47], we set the learning rates to 2e − 6 for the generator and 4e − 7 for the critic, with a 1:5 update ratio.
- • Stage 2: Inter-shot Self-forcing. Using a 5-shot subset of our dataset, the model is trained on its own multishot rollouts. For each iteration, the model generates a sequence of 5-second shots; when a shot boundary is reached, the global context cache is updated with the generated content while the local cache is reset. We apply LoRA tuning for 1,000 steps using the same learning rates as Stage 1.

All experiments are conducted on a cluster of 32 NVIDIA H800 GPUs.

