## WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling

#### Wenqiang Sun*13 Haiyu Zhang*23 Haoyuan Wang*3 Junta Wu3 Zehan Wang3 Zhenwei Wang3 Yunhong Wang2 Jun Zhang1 Tengfei Wang3 Chunchao Guo3

|[Figure 1]<br><br>a) Real World|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2512.14614v2[cs.CV]9Jun2026

W W S S

|[Figure 6]<br><br>b) Stylized World|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]<br><br>c) Third-Person|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

W D A S

|[Figure 17]<br><br>d) 3D Reconstruction|
|---|

Point Cloud

[Figure 18]

[Figure 19]

[Figure 20]

Smoke Spread

|[Figure 21]<br><br>e) Promptable Event|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

W W W W

Figure 1. WorldPlay is a real-time, interactive world model that achieves long-term geometric consistency. It responds to user navigation commands in a streaming fashion, while maintaining scenes remain coherent when revisiting (shown in red boxes). Our model shows remarkable generalization across diverse scenes, including (a) real world, (b) stylized world, and (c) third-person agent control. Furthermore, it supports (d) 3D scene generation via reconstruction and (e) dynamic world events triggered by text-based manipulation.

### Abstract

methods. WorldPlay draws power from three key ingredients. 1) We use a Dual Action Representation to enable robust action control in response to the user’s keyboard and mouse inputs. 2) To enforce long-term consistency, our Reconstituted Context Memory dynamically rebuilds context from past frames and uses temporal reframing to keep geometrically important but long-past frames accessible, effectively alleviating memory attenuation. 3) We also propose Context Forcing, a novel distillation method designed for memory-aware model. Aligning memory context between the teacher and student preserves the student’s capacity to use long-range information, enabling real-time speeds while

This paper presents WorldPlay, a streaming video diffusion model that enables real-time, interactive world modeling with long-term geometric consistency, resolving the trade-off between speed and memory that limits current

*Equal contribution 1Hong Kong University of Science and Technology 2Beihang University 3Tencent Hunyuan. Correspondence to: Jun Zhang <eejzhang@ust.hk>, Tengfei Wang <tengfeiwang12@gmail.com>, Chuncaho Guo <chunchaoguo@gmail.com>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

preventing error drift. Taken together, WorldPlay generates long-horizon streaming 720p video at 24 FPS with superior consistency, comparing favorably with existing techniques and showing strong generalization across diverse scenes. Project page and online demo can be found: https://3d-models.hunyuan.tencent.com/world/

and https://3d.hunyuan.tencent.com/sceneTo3D.

### 1. Introduction

World models are driving a pivotal shift in computational intelligence, moving beyond language-centric tasks towards visual and spatial reasoning. By simulating dynamic 3D environments, these models empower agents to perceive and interact with complex surroundings, opening up new possibilities for embodied robotics and game development.

At the forefront of world modeling is real-time interactive video generation, which aims at autoregressively predicting future video frames (or chunks) to deliver instant visual feedback in response to every user’s keyboard command. Despite significant progress, a fundamental challenge persists: how to simultaneously achieve real-time generation (speed) and long-term geometric consistency (memory) in interactive world modeling. One class of methods (Decart,

- 2024; Parker-Holder et al., 2024; He et al., 2025b) prioritizes speed with distillation but neglects memory, resulting in inconsistency where scenes change upon revisit. The other class preserves consistency with explicit (Li et al., 2025b; Ren et al., 2025) or implicit (Xiao et al., 2026; Yu et al.,
- 2025b; Chen et al., 2025) memory, but complex memory makes distillation non-trivial (Sec. 3.4). As summarized in Table 1, the simultaneous achievement of both low latency and high consistency remains an open problem.

To tackle this challenge, we develop WorldPlay, a real-time and long-term consistent world model for general scenes. We consider this problem as a next chunk (16 frames) prediction task for generating streaming videos conditioned on actions from users. Building upon autoregressive diffusion models, WorldPlay draws power from the model’s three key ingredients below.

The first is Dual Action Representation for control over agent and camera movement. Previous works (Decart, 2024; Parker-Holder et al., 2024; He et al., 2025b) typically rely on discrete keyboard inputs (e.g., W, A, S, D) as action signals, which afford plausible, scale-adaptive movement but suffer from ambiguity for memory retrieval that requires revisiting exact locations. Conversely, continuous camera poses (R,T) provide spatial locations but cause training instability due to scene scale variance in training data. To combine the best of both worlds, we convert action signals into continuous camera poses and discrete keys, achieving

robust control and accurate location caching.

The second key design is Reconstituted Context Memory for maintaining long-term geometric consistency. We actively reconstitute the memory through a two-stage process, moving beyond simple retrieval (Yu et al., 2025b; Xiao et al., 2026). It first dynamically rebuilds a context set by querying past frames based on spatial and temporal proximity. To overcome the long-range decay (the fading influence of distant tokens in Transformers (Su et al., 2024)), we propose temporal reframing to rewrite positional embeddings of these retrieved frames. This operation effectively “pulls” geometrically important but long-past memories closer in time, forcing the model to treat them as recent. This process keeps the influence of relevant long-range information preserved, enabling robust free extrapolation with strong geometric consistency.

The final key ingredient is Context Forcing, a novel distillation method designed for memory-aware models to enable real-time generation. Existing distillation methods (Chen

- et al., 2024a; Huang et al., 2025c; Yin et al., 2024a) fail to keep long-term memory as there is a fundamental distribution mismatch: training a memory-aware autoregressive student to mimic a memory-less bidirectional teacher. Even when augmenting teacher with memory, mismatched memory context will cause distribution diverge. We solve this by aligning the memory context for teacher and student during distillation. This alignment facilitates effective distribution matching, enabling real-time speed without eroding the memory while alleviating error accumulation over long sequences.

Taken together, WorldPlay achieves real-time, interactive video generation at 24 FPS (720p) while maintaining longterm geometric consistency under streaming user control. The model is built on a large-scale, curated dataset of 320K real and synthetic videos with a custom rendering and processing platform. As shown in Fig. 1, WorldPlay shows superior generation quality and remarkable generalization across diverse scenes including first- and third-person real and stylized worlds, and supports applications ranging from 3D reconstruction and promptable events.

2. Related Work

Video Generation. Diffusion models (Ho et al., 2020; Lipman et al., 2023; Song et al., 2021) have emerged as the stateof-the-art approach in video generative modeling. (Chen

- et al., 2024b; Guo et al., 2024; Yang et al., 2024) adopt the latent diffusion model (LDM) (Rombach et al., 2022) to learn video distribution in the latent space, achieving efficient video generation. Recently, autoregressive video generation models (Chen et al., 2024a; Henschel et al., 2025; Kim et al., 2024) theoretically enable one to generate unlim-

- Table 1. Comparison with recent interactive world models. WorldPlay distinguishes itself as a general-domain model that simultaneously achieves long-horizon video generation, flexible action control, real-time interactivity, and long-term geometric consistency. ’Con.’ and ’Dis.’ represent continue and discrete action, respectively.

| |Yume (Mao et al., 2025b)|Matrix-Game 2.0 (He et al., 2025b)<br><br>|GameGenX (Che et al., 2025)|GameCraft (Li et al., 2025a)|WorldMem (Xiao et al., 2026)<br><br>|VMem (Li et al., 2025b)|WorldPlay|
|---|---|---|---|---|---|---|---|
|Resolution<br><br>|544p|360p|720p<br><br>|720p<br><br>|360p<br><br>|576p|720p<br><br>|
|Action Space<br><br>|Text|Dis.<br><br>|Dis.<br><br>|Con.<br><br>|Dis.|Con.<br><br>|Con. + Dis.|
|Real-time<br><br>|✔<br><br>|✔|✗|✗<br><br>|✗<br><br>|✗<br><br>|✔|
|Long-term Consistency|✗<br><br>|✗|✗|✗|✔|✔<br><br>|✔|
|Long-Horizon<br><br>|✗|✔<br><br>|✔|✔<br><br>|✗<br><br>|✗|✔<br><br>|
|Domain<br><br>|General|General<br><br>|General<br><br>|General|Minecraft|Static Scene<br><br>|General|

ited length videos, laying the foundation for world models. With the advancement of powerful architectures (Peebles & Xie, 2023) and sophisticated data pipelines, (Deepmind, 2025; Wan et al., 2025; Kuaishou, 2024; Minimax, 2024; Gao et al., 2025; Kong et al., 2024a), which are trained on web-scale datasets, have demonstrated emergent zero-shot capabilities to perceive, model, and manipulate the visual world (Wiedemer et al., 2025), making it feasible to simulate the physical world.

Interactive and Consistent World Models. World models aim to predict future states based on current and past states. Studies such as (Alonso et al., 2024; Bar et al., 2025; Valevski et al., 2025; Yu et al., 2025c; Sun et al., 2025b; He et al., 2025a; Wang et al., 2024b; Miyato et al., 2024; Kong et al., 2024b; Li et al., 2025c; Bahmani et al., 2025; Sun et al., 2025a; Mao et al., 2025b;a; Xiang et al., 2025; Tang et al., 2025) adopt discrete, continuous action signals or text instructions to enable agents to navigate and interact with virtual environments. (Yesiltepe et al., 2026) proposes a training-free framework for instruction-controllable video generation. Subsequent works that aim to achieve geometric consistency can be categorized into two types: explicit 3D reconstruction and implicit conditioning. (Li et al., 2025b; Yu et al., 2025a; Ren et al., 2025; Cao et al., 2025; Yu et al., 2025; YU et al., 2025; Liu et al., 2026a) ensure spatial consistency by explicitly reconstructing 3D representations and rendering condition frames from these representations. However, they heavily rely on reconstruction quality, making it challenging to maintain long-term consistency. Recent work (HunyuanWorld, 2025) constructs 3D world models explicitly, without relying on video generation models. Although achieving promising 3D generation results, they can not be performed in real-time. In contrast, (Xiao et al., 2026; Yu et al., 2025b) achieve implicit conditioning by leveraging field-of-view (FOV) to retrieve relevant context from historical frames. Concurrent work (Hong et al., 2025) achieves interactive generation with fixed-length consistency through context compression (Zhang et al., 2025c). However, developing a real-time world model with long-horizon geometric consistency remains unsolved.

Distillation. For video diffusion models, existing ap-

proaches typically employ distillation (Salimans & Ho, 2022; Geng et al., 2025; Frans et al., 2025; Li et al., 2026; Zhang et al., 2025a) to achieve few-step inference, achieving faster generation. (Sauer et al., 2024a;b; Kang et al., 2024; Lin et al., 2025a; 2024; 2025b) adopt adversarial training strategies to enable few-step inference, however, they often suffer from training instability and mode collapse. (Yin et al., 2024b;a; Lu et al., 2025; Shin et al., 2026) utilize Variational Score Distillation (Wang et al., 2023) to achieve outstanding few-step generation performance in various tasks. In addition, CausVid (Yin et al., 2025) proposes distilling a causal student model from a bidirectional teacher diffusion model to achieve real-time autoregressive generation. Furthermore, Self-Forcing (Huang et al., 2025c) mitigates exposure bias by refining the rollout strategy of CausVid. Our method proposes context forcing to preserve both the interactivity and geometric consistency while achieving realtime generation.

### 3. Method

Our goal is to construct a geometry-consistent and realtime interactive world model Nθ(xt|Ot−1,At−1,at,c) parameterized by θ, which can generate next chunk xt (a chunk is a few frames) based on past observations Ot−1 = {xt−1,...,x0}, action sequences At−1 = {at−1,...,a0}, and current action at. Here, c is a text prompt or image that describes the world. For simplicity of notation, we omit A,a,c in following sections. We first introduce the relevant preliminaries in Sec. 3.1. In Sec. 3.2, we discuss the action representation for control. Sec. 3.3 describes our reconstituted context memory to ensure long-term geometric consistency, followed by Sec. 3.4 covering our context forcing, which mitigates exposure bias and enables few-step generation while maintaining long-term consistency. Finally, Sec. 3.5 details additional optimizations for real-time streaming generation. The pipeline is shown in Fig. 2.

#### 3.1. Preliminaries

Full-sequence Video Diffusion Model. Current video diffusion models (Kong et al., 2024a; Wan et al., 2025) typically

[Figure 26]

- Figure 2. Method overview. Given a single image or text prompt to describe a world, WorldPlay performs a next chunk (16 video frames) prediction task to generate future videos conditioned on action from users. For the generation of each chunk, we dynamically reconstitute context memory from past chunks to enforce long-term temporal and geometric consistency.

[Figure 27]

- Figure 3. Detailed architecture of our autoregressive diffusion transformer. The discrete key is incorporated with time embedding, while the continuous camera pose is injected into causal self-attention through PRoPE (Li et al., 2025c).

chitecture, which limits its ability for infinite-length interactive generation. Inspired by Diffusion Forcing (Chen et al., 2024a), we finetune it into a chunk-wise autoregressive video generation model. Specifically, for video latent z0 ∈ RC×T×H×W, we divide it into T4 chunks {z0i ∈ RC×4×H×W|i = 0,..., T4 − 1}, and thus each chunk (4 latents) can be decoded into 16 frames. During training, we add different noise levels ki for each chunk and modify the full-sequence self-attention to block causal attention. The training loss is similar to Eq. 1.

#### 3.2. Dual Action Representation for Control

Existing methods use keyboard and mouse inputs as action signals and inject the action control via MLP (Decart, 2024; Xiao et al., 2026) or attention blocks (He et al., 2025b; Yu et al., 2025b). This enables the model to learn physically plausible movements across scenes with diverse scales (e.g.very large and small scenes). However, they struggle to provide precise previous locations for spatial memory retrieval. In contrast, camera poses (rotation matrix and translation vector) provide accurate spatial locations that facilitate precise control and memory retrieval, but training only with camera poses faces challenges in training stability due to the scale variance in the training data. To address this, we propose a dual action representation that combines the best of both worlds as shown in Fig. 3. This design not only caches spatial locations for our memory module in Sec. 3.3, but also enables robust and precise control. Specifically, we employ PE and a zero-initialized MLP to encode discrete keys and incorporate it into the timestep embedding, which is then used to modulate the DiT blocks. For continuous camera pose, we leverage relative positional encoding, i.e., PRoPE (Li et al., 2025c), which offers greater generalizability than commonly used raymaps, to inject complete

consist of a causal 3D VAE (Kingma & Welling, 2013) and a Diffusion Transformer (DiT) (Peebles & Xie, 2023), where each DiT block is composed of 3D self-attention, crossattention, and feedforward network (FFN). The diffusion timestep is processed by positional embedding (PE) and a Multi-Layer Perceptron (MLP) to modulate the DiT blocks. The model is trained using flow matching (Lipman et al.,

- 2023). Specifically, given a video latent z0 encoded by the 3D VAE, a random noise z1 ∼ N(0,I), and a diffusion timestep k ∈ [0,1], an intermediate latent zk is obtained through linear interpolation. The model is trained to predict the velocity vk = z0 − z1,

2

. (1)

LFM(θ) = Ek,z

0,z1 Nθ(zk,k) − vk

Chunk-wise Autoregressive Generation. However, the full-sequence video diffusion model is a non-causal ar-

[Figure 28]

[Figure 29]

[Figure 30]

(a) Full context (b) Absolute indices (c) Relative indices

- Figure 4. Memory mechanism comparisons. The red and blue blocks represent the memory and current chunk, respectively. The number in each block represents the temporal index in RoPE. For simplicity of illustration, each chunk only contains one frame.

camera frustums into self-attention blocks. The original self-attention computation is as follows,

Attn1 = Attn(R⊤ ⊙ Q,R−1 ⊙ K,V ), (2) where R represents the 3D rotary PE (RoPE) (Su et al.,

- 2024) for video latents. To encode frustum relationships between cameras, we utilize an additional attention,

Attn2 =Dproj ⊙ Attn((Dproj)⊤ ⊙ Q, (Dproj)−1 ⊙ K,(Dproj)−1 ⊙ V ),

(3)

K 0 0 1

here, Dproj =

Tcw is derived from the camera’s

intrinsic K and extrinsic parameters Tcw, as described in (Li et al., 2025c). The result of each self-attention block is Attn1 + zero init(Attn2).

#### 3.3. Reconstituted Context Memory for Consistency

Maintaining long-term geometric consistency requires recalling past frames, ensuring content remains unchanged when revisiting to a previous location. However, naively using all past frames as context (Fig. 4a) is computationally intractable and redundant for long sequences. To address this, we rebuild a memory context Ct from past chunks Ot−1 for each new chunk xt. Our approach advances beyond prior work (Xiao et al., 2026; Yu et al., 2025b; Chen et al., 2025) by combining both short-term temporal cues and long-range spatial references: 1) A temporal memory (CtT) comprises L most recent chunks {xt−L,...,xt−1} to ensure short-term motion smoothness. 2) A spatial memory (CtS) samples from non-adjacent past frames to prevent geometric drift over long sequences, where CtS ⊆ Ot−1 − CtT. This sampling is guided by geometric relevance scores that incorporate both FOV overlap and camera distance.

Once memory context is rebuilt, the challenge shifts to applying them to enforce consistency. Effectively using retrieved context requires overcoming a fundamental flaw in positional encodings. With standard RoPE (Fig.4b), the distance between the current chunk and past memory grows

[Figure 31]

Figure 5. Context forcing is a novel distillation method that employs memory-augmented self-rollout and memory-augmented bidirectional video diffusion to preserve long-term consistency, enable real-time interaction, and mitigate error accumulation.

unbounded over time. This growing relative distance can eventually exceed the trained interpolation range in RoPE, causing extrapolation artifacts (Su et al., 2024). More critically, the growing perceived distance to these long-past spatial memory would weaken their influence on the current prediction. To resolve this, we propose Temporal Reframing (Fig.4c). We discard the absolute temporal indices, and dynamically re-assign new positional encodings to all context frames, establishing a fixed, small relative distance to the current, irrespective of their actual temporal gap. This operation effectively “pulls” important past frames closer in time, ensuring their sustained influence and enabling robust extrapolation for long-term consistency.

#### 3.4. Context Forcing

Autoregressive models often suffer from error accumulation during long video generation, leading to degraded visual quality over time (Huang et al., 2025c; Yin et al., 2025). Moreover, the multi-step denoising of diffusion models is too slow for real-time interaction. Recent methods (Huang et al., 2025c; Yang et al., 2026; Liu et al., 2026b; Cui et al., 2026a) address these challenges by distilling a powerful bidirectional teacher diffusion model into a fast, few-step autoregressive student. These techniques force the student’s output distribution pθ(x0:t) to align with the teacher’s, thereby improving generation quality by employing a distribution matching loss (Yin et al., 2024a):

∇θLDMD = Ek(∇θKL(pθ(x0:t)||pdata(x0:t))), (4)

where the gradient of the reverse KL can be approximated by the score difference derived from teacher model.

However, these methods are incompatible with memoryaware models due to a critical distribution mismatch. Standard teacher diffusion models are trained on short clips and are inherently memory-less. Even if a teacher is augmented with memory, its bidirectional nature inevitably differs from the student’s causal, autoregressive process. This means that without a meticulously designed memory context to mitigate this gap, the difference in memory context will make their conditional distributions p(x|C) misaligned, which in turn causes distribution matching to fail.

- Table 2. Quantitative comparisons. We compare against both methods without memory, i.e., CameraCtrl (He et al., 2025a), SEVA (Zhou et al., 2025), ViewCrafter (Yu et al., 2025), Matrix-Game-2.0 (He et al., 2025b), and GameCraft (Li et al., 2025a), and methods with memory, i.e., Gen3C (Ren et al., 2025), VMem (Li et al., 2025b). Our method achieves superior results, particularly in long-term settings, which more clearly demonstrate the long-term consistency.

Short-term (61 frames) Long-term (≥ 250 frames) Real-time PSNR ↑ SSIM ↑ LPIPS ↓ Rdist ↓ Tdist ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Rdist ↓ Tdist ↓

CameraCtrl (He et al., 2025a) ✗ 17.93 0.569 0.298 0.037 0.341 10.09 0.241 0.549 0.733 1.117 SEVA (Zhou et al., 2025) ✗ 19.84 0.598 0.313 0.047 0.223 10.51 0.301 0.517 0.721 1.893 ViewCrafter (Yu et al., 2025) ✗ 19.91 0.617 0.327 0.029 0.543 9.32 0.277 0.661 1.573 3.051 Gen3C (Ren et al., 2025) ✗ 21.68 0.635 0.278 0.024 0.477 15.37 0.431 0.483 0.357 0.979 VMem (Wang et al., 2024b) ✗ 19.97 0.587 0.316 0.048 0.219 12.77 0.335 0.542 0.748 1.547 Matrix-Game-2.0 (He et al., 2025b) ✔ 17.26 0.505 0.383 0.287 0.843 9.57 0.205 0.631 2.125 2.742 GameCraft (Li et al., 2025a) ✗ 21.05 0.639 0.341 0.151 0.617 10.09 0.287 0.614 2.497 3.291

Ours (w/o Context Forcing) ✗ 21.27 0.669 0.261 0.033 0.157 16.27 0.425 0.495 0.611 0.991 Ours (full) ✔ 21.92 0.702 0.247 0.031 0.121 18.94 0.585 0.371 0.332 0.797

We thus propose context forcing as shown in Fig. 5, which alleviates the memory context misalignment between teacher and student for distillation. For the student model, we self-rollout 4 chunks conditioned on the memory con-

text pθ(xj:j+3|x0:j−1) =

j+3

i=j

pθ(xi|Ci). To construct our

teacher model Vβ, we augment a standard bidirectional diffusion model with memory, and structure its context by masking xj:j+3 from student’s memory context,

pdata(xj:j+3|x0:j−1) = pβ(xj:j+3|Cj:j+3 − xj:j+3), (5)

where Cj:j+3 denotes all context memory chunks corresponding to student’s self-rollout xj:j+3. By aligning the memory context with the student model, we enforce the distributions represented by the teacher to be as close as possible to the student model, which enables more effectively distribution matching. Moreover, this avoids training Vβ on long videos and redundant context, facilitating the learning of long-term visual distribution. Additionally, we introduce a progressive distillation strategy that incrementally increases the number of self-rollout latents. This facilitates distillation across varying sequence lengths, thereby enhancing long-horizon video generation. Through context forcing, we preserve long-term consistency in real-time generation with 4-denoising steps, and mitigate error accumulation.

- 3.5. Streaming Generation with Real-Time Latency

while maintaining generation quality.

Streaming Deployment and Progressive Decoding. To minimize time-to-first-frame and enable seamless interaction, we adopt a streaming deployment architecture using NVIDIA Triton Inference Framework and implement a progressive VAE decoding strategy that decodes and streams frames in smaller batches, allowing users to observe generated content while subsequent frames are still being processed. This streaming pipeline ensures smooth, low-latency interaction even under varying computational loads.

Quantization and Efficient Attention. Furthermore, we employ a comprehensive suite of quantization strategies. Specifically, we adopt Sage Attention (Zhang et al., 2025b), float quantization, and matrix multiplication quantization to improve the inference performance. Additionally, we use KV-cache mechanisms for attention modules to eliminate redundant computations during autoregressive generation.

### 4. Experiments

Implementation Details. WorldPlay is trained on a comprehensive dataset comprising approximately 320K highquality video samples derived from both real-world footage and synthetic environments. Details regarding the dataset processing pipeline and the training/inference are provided in Appendix B and Appendix A, respectively.

Evaluation Protocol. Our test set comprises 600 cases sourced from DL3DV, game videos, and AI-generated images spanning a range of styles. For the short-term setting, we utilize the camera trajectories from the test videos as the input pose. The generated frames are directly compared against the ground truth to assess visual quality and action precision. For the long-term setting, we test the longhorizon generation ability and long-term consistency using various custom cycle camera trajectories designed to enforce revisiting. Each model generates frames along a customize trajectory and then returns along the same path, metrics are evaluated on the return path by comparing the generated

We augment context forcing with a suite of optimizations to minimize latency, unlocking an interactive streaming experience at 24 FPS and 720p resolution on 8×H800 GPUs.

Mixed Parallelism Method for DiT and VAE. Unlike the conventional parallelism method that replicates the entire model or adapting sequence parallelism on the temporal dimension, our parallelism method combines sequence parallelism (Li et al., 2023) and attention parallelism, which partitions the tokens of each chunk across devices. This design ensures that the computational workload is distributed evenly, substantially reducing per-chunk inference time

|[Figure 32]<br><br>W A S D<br><br>|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Ours

W

A

D

D

S

S

D

S

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Gen3C

W

A

D

D S

S

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Ours

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

GameCraft

W A S D

|[Figure 58]<br><br>W A S D<br><br>|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Ours

D D D A A A

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

GameCraft

D D D A A A

|[Figure 71]<br><br>W A S D<br><br>|
|---|

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Ours

W S

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Matrix-Game 2.0

W S

|[Figure 84]<br><br>W A S D<br><br>|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Ours

W W D D D D

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Matrix-Game 2.0

W W D D D D

- Figure 6. Qualitative comparisons with existing methods. WorldPlay achieves the state-of-the-art long-term consistency (shown in red boxes) and visual quality across diverse scenes, including both first- and third-person real and stylized worlds.

frame to the corresponding frame generated during the initial pass. We employ LPIPS, PSNR, and SSIM to measure visual quality and Rdist and Tdist to quantify action precision. Specifically, we use the ground truth poses as the input for each model to generate videos. Then, we utilize ViPE to estimate the camera pose of the generated videos. Following the previous works (He et al., 2025a; Yu et al., 2025), we compute the relative poses of the ground truth and generated camera poses by setting the extrinsic matrix of first frame as an identity matrix and normalize the translation scale using the furthest frame. The rotation distance Rdis is calculated by comparing the predicted rotation matrices Rgen and ground truth rotation matrices Rgt:

Rdis = arccos

tr(RgenR⊤gt) − 1 2

, (6)

where tr(·) denotes the trace of the matrix. The translation distance Tdis is computed between the predicted tgen and

ground truth translation vectors tgt:

Tdis = ∥tgt − tgen∥22. (7)

Baselines. We conduct comprehensive comparisons against various baselines, which mainly fall into two categories: 1) Action-controlled diffusion models without memory: CameraCtrl (He et al., 2025a), SEVA (Zhou et al., 2025), ViewCrafter (Yu et al., 2025), Matrix-Game 2.0 (He et al., 2025b) and GameCraft (Li et al., 2025a); 2) Actioncontrolled diffusion models with memory: Gen3C (Ren et al., 2025) and VMem (Li et al., 2025b). More evaluation results can be found in Appendix.

#### 4.1. Main Result

Quantitative Results. As shown in Table 2, in the shortterm regime, our approach achieves superior visual fidelity and maintains competitive control accuracy. Although methods leveraging explicit 3D representations (i.e.ViewCrafter (Yu et al., 2025), Gen3C (Ren et al., 2025)) realize more accurate rotation, they suffer from issues such

###### Table 3. Quantitative Comparison of 3D Structural Consistency using MEt3R (Asim et al., 2025).

Method MEt3R Score (↓)

Gen3C (Ren et al., 2025) 0.187 Matrix-Game-2.0 (He et al., 2025b) 0.367 GameCraft (Li et al., 2025a) 0.305 Ours (full) 0.133

###### Table 4. Quantitative evaluation of inference acceleration strategies. ”P” and ”Q” denote Parallelism and Quantization, respectively.

###### DiT P&Q VAE P&Q Streaming Decoding FPS Improvement

2.80 —

✓ 3.80 1.35× ✓ ✓ 16.0 5.71× ✓ ✓ ✓ 24.0 8.57×

as the inaccurate depth estimation and inconsistent scale when performing translations. For more challenging longterm scenarios, where action accuracy generally degrades, our method remains more stable and achieves the best performance. Regarding long-term geometric consistency, MatrixGame-2.0 (He et al., 2025b) and GameCraft (Li et al., 2025a) exhibit poor performance due to the lack of memory mechanism. Although VMem (Li et al., 2025b) and Gen3C (Ren et al., 2025) employ explicit 3D cache to maintain consistency, they are constrained by depth accuracy and alignment, making it difficult to achieve robust long-term consistency. Benefiting from Reconstituted Context Memory, we achieve improved long-term consistency. Moreover, through context forcing, we further prevent error accumulation, resulting in better visual quality and action accuracy. To rigorously evaluate the long-horizon 3D structural consistency of our model, we incorporate a more advanced metric, MEt3R (Asim et al., 2025), which explicitly models the multi-view correspondence by leveraging pre-trained 3D geometric priors (e.g., DUSt3R (Wang et al., 2024a)). We evaluate the generated long-horizon videos by pairing frames from the initial trajectory with the corresponding frames in the revisiting trajectory, which are then fed into the MEt3R model to quantify their 3D structural alignment. As shown in Table. 3, it clearly demonstrates the superiority of our approach in maintaining precise 3D consistency.

To further validate the efficiency of our optimization strategies proposed in Sec. 3.5, we conduct experiments on inference speed as detailed in Table 4, demonstrating that the tailored parallelization and quantization applied to DiT and VAE significantly boost the inference throughput of our model. Crucially, WorldPlay concurrently achieves the requisite real-time interactivity for immersive simulation.

Qualitative Results. We provide qualitative comparisons with baselines in Fig. 6. The explicit 3D cache used in Gen3C (Ren et al., 2025) is highly sensitive to the quality of intermediate output and limited by the accuracy of depth

Table 5. Ablation for action representation.

Action PSNR↑ SSIM↑ LPIPS↓ Rdist ↓ Tdist ↓ Discrete 21.47 0.661 0.248 0.103 0.615 Continuous 21.93 0.665 0.231 0.038 0.287 Full 22.09 0.687 0.219 0.028 0.113

Table 6. Ablation for positional encoding design in memory. The results are evaluated on the long-term test data.

PSNR↑ SSIM↑ LPIPS↓ Rdist ↓ Tdist ↓

RoPE 14.03 0.358 0.534 0.805 1.341 Reframed RoPE 16.27 0.425 0.495 0.611 0.991

estimation. Conversely, our reconstituted context memory guarantees long-term consistency with more robust implicit prior, achieving superior generalizability. Matrix-Game2.0 (He et al., 2025b) and GameCraft (Li et al., 2025a) fail to support free exploration due to the lack of memory. Furthermore, they do not generalize well to third-person scenarios, making it difficult to control agents and limiting their applicability. In contrast, WorldPlay successfully extends its efficacy to these scenarios and maintains high visual fidelity and long-term geometric consistency.

#### 4.2. Ablation

Action Representation. Table 5 validates the effectiveness of the proposed dual-action representation. When using only discrete keys as action signals, the model struggles to achieve fine-grained control, such as the distance of movement or the degree of rotation, resulting in poor performance on Rdist and Tdist metrics. Using continuous camera poses yields better results but converges more difficult due to scale variance. By employing the dual-action representation, we achieve the best overall control performance.

RoPE Design. Table. 6 presents the quantitative results of different RoPE designs within the memory mechanism as detailed in Sec. 3.3, showing that reframed rope outperforms naive counterparts, especially on visual metrics. As illustrated in the upper part of Fig. 7, RoPE is more prone to error accumulation. It also increases the distance between memory and predicted chunk due to absolute temporal indices, resulting in weaker geometric consistency, as shown in the lower part of Fig. 7.

Context Forcing. To verify the importance of memory alignment, we train the teacher model following (Yu et al., 2025b), where the memory is selected at latent level rather than at chunk level. Although this may reduce the number of memory context in the teacher model, it also introduce misaligned context between the teacher and student model, leading to collapsed results as shown in Fig. 8a. Moreover, utilizing a memory-less bidirectional model as the teacher induces a distribution mismatch, which hinders long-horizon video generation and significantly compromises long-term consistency, as illustrated in Fig. 8b. Additionally, for the

[Figure 97]

###### WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling

|a) RoPE|
|---|
|b) Reframed RoPE (Ours)|

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

[Figure 103]

Point Cloud

Error Accumulation

| |
|---|

|[Figure 104]|
|---|

[Figure 105]

W

D

A

|[Figure 106]|
|---|

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

S

| |
|---|

[Figure 111]

[Figure 112]

W

A

D

|[Figure 113]|
|---|

[Figure 114]

[Figure 115]

[Figure 116]

- a) RoPE Geometric Inconsistency

- b) Reframed RoPE (Ours)

[Figure 117]

S S

[Figure 118]

[Figure 119]

|[Figure 120]|
|---|

| |
|---|

A colossal stone Buddha statue dominates the center of the image, nestled between towering, mistshrouded cliffs. A burning incense burner sits in the foreground on a snow-covered ground.

W

A

S

|[Figure 121]|
|---|

[Figure 122]

[Figure 123]

[Figure 124]

W

| |
|---|

[Figure 125]

[Figure 126]

S

W

A

- Figure 7. RoPE design comparisons. Upper: Our reframed RoPE avoids exceeding the the positional range in standard RoPE, alleviating error accumulation. Bottom: By maintaining a small relative distance to long-range spatial memory, it achieves better long-term consistency.

|[Figure 127]<br><br>a) Misaligned Context|
|---|
|[Figure 128]<br><br>b) Memory-less Teacher|
|[Figure 129]<br><br>c) Self-rollout Context|
|[Figure 130]<br><br>d) Ours|

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

W S S

[Figure 137]

[Figure 138]

[Figure 139]

W S S

W S S

[Figure 140]

[Figure 141]

[Figure 142]

W S S

- Figure 8. Ablation for context forcing. a) When the teacher and student have misaligned context, it leads to distillation failure, resulting in collapsed outputs. b) Leveraging memory-less teacher introduces a distribution mismatching. c) Self-rollout historical context can introduce artifacts. Zoom in for details.

A W

- Figure 9. 3D reconstruction results. We first utilize our model to autoregressively generate videos. The videos are then processed by a 3D reconstruction model to produce the final point clouds.

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Balloon Appears

A wide open meadow stretches to the horizon, covered with colourful wildflowers swaying gently in the wind. The scene feels bright and calm under the midday sun.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Rainbow Arches

W W W W

W W W W

- Figure 10. Promptable event. Our method supports text-based manipulation during streaming.

at any time to responsively alter the ongoing stream.

### 5. Conclusion

WorldPlay is a powerful world model with real-time interaction and long-term geometric consistency. It empowers users to customize unique worlds from a single image or text prompt. While focused on navigation control, its architecture has shown potential for richer interaction like dynamic, text-triggered events. By providing a systematic framework for control, memory, and distillation, WorldPlay marks a critical step toward creating consistent and interactive virtual worlds.

past chunks x0:j−1, we attempt to self-rollout historical chunks as context following (Yang et al., 2026). However, this may cause the bidirectional diffusion model to provide inaccurate score estimation, as it is trained using clean chunks as memory. Consequently, this discrepancy introduces artifacts as illustrated in Fig. 8c. We obtain historical chunks by sampling from real videos, which yields superior results as shown in Fig. 8d.

Limitations. While WorldPlay demonstrates strong performance, several avenues remain open for exploration and improvement. First, our model can generawting videos of approximately 30 seconds, efficiently scaling this framework to longer durations, such as minutes or hours (Cui et al., 2026a;b), remains a significant challenge. Second, although distillation is utilized to mitigate error accumulation, fundamentally averting this phenomenon during the training of autoregressive diffusion models remains a critical open challenge. Moreover, expanding the action types to a broader set with multi-agent interaction and complex physical dynamics is another promising direction. Finally, retrieval mechanisms based on the FOV may fail to accurately identify memory context when faced with significant occlusions (Yu et al., 2025b).

#### 4.3. Application

3D Reconstruction. Benefiting from the long-term geometric consistency, we can integrate a feed-forward 3D reconstruction model (Liu et al., 2025) to produce high-quality point clouds from the generated videos, as presented in Fig. 1 (d) and Fig. 9.

Promptable Event. Beyond navigation control, WorldPlay supports text-based interaction to trigger dynamic world events (i.e., environmental transitions and object appearances). As shown in Fig. 10 and Fig. 1 (e), users can prompt

### Acknowledgement

This work was supported by the Hong Kong Research Grants Council under the Areas of Excellence scheme grant AoE/E-601/22-R and NSFC/RGC Collaborative Research Scheme grant CRS HKUST603/22.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A. J., Pearce, T., and Fleuret, F. Diffusion for world modeling: Visual details matter in Atari. In Advances in Neural Information Processing Systems, volume 37, pp. 58757–58791, 2024. 3

Asim, M., Wewer, C., Wimmer, T., Schiele, B., and Lenssen, J. E. MEt3R: Measuring multi-view consistency in generated images. In CVPR, pp. 6034–6044, 2025. 8

Bahmani, S., Skorokhodov, I., Qian, G., Siarohin, A., Menapace, W., Tagliasacchi, A., Lindell, D. B., and Tulyakov, S. AC3D: Analyzing and improving 3D camera control in video diffusion transformers. In CVPR, pp. 22875–22889, 2025. 3

Bar, A., Zhou, G., Tran, D., Darrell, T., and LeCun, Y. Navigation world models. In CVPR, pp. 15791–15801,

2025. 3

Cao, C., Zhou, J., Li, S., Liang, J., Yu, C., Wang, F., Xue, X., and Fu, Y. Uni3C: Unifying precisely 3D-enhanced camera and human motion controls for video generation. In ACM SIGGRAPH Asia, 2025. 3

Che, H., He, X., Liu, Q., Jin, C., and Chen, H. GameGen-X: Interactive open-world game video generation. In ICLR,

2025. 3

Chen, B., Mart´ı Mons´o, D., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion Forcing: Nexttoken prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, volume 37, pp. 24081–24125, 2024a. 2, 4

Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., and Shan, Y. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pp. 7310–7320, 2024b. 2

Chen, T., Hu, X., Ding, Z., and Jin, C. Learning world models for interactive video generation. In Advances in

Neural Information Processing Systems, volume 38, pp. 154456–154483, 2025. 2, 5

Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., and Hsieh, C.-J. Self-Forcing++: Towards minute-scale high-quality video generation. In ICLR, 2026a. 5, 9

Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., and Hsieh, C.-J. LoL: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026b. 9

Decart, E. Oasis: A universe in a transformer. https: //oasis-model.github.io/, 2024. 2, 4

Deepmind, G. Veo3 video model, 2025. https://deep mind.google/models/veo/. 3

Duan, H., Yu, H.-X., Chen, S., Fei-Fei, L., and Wu, J. WorldScore: A unified evaluation benchmark for world generation. In ICCV, pp. 27713–27724, 2025. 22

Frans, K., Hafner, D., Levine, S., and Abbeel, P. One step diffusion via shortcut models. In ICLR, 2025. 3

Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. 3

Geng, Z., Deng, M., Bai, X., Kolter, Z., and He, K. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems, volume 38, pp. 75460–75482, 2025. 3

Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., and Dai, B. AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning. In ICLR, 2024. 2

He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., and Yang, C. CameraCtrl: Enabling camera control for text-to-video generation. In ICLR, 2025a. 3, 6, 7

He, X., Peng, C., Liu, Z., Wang, B., Zhang, Y., Cui, Q., Kang, F., Jiang, B., An, M., Ren, Y., et al. Matrix-Game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025b. 2, 3, 4, 6, 7, 8

Henschel, R., Khachatryan, L., Poghosyan, H., Hayrapetyan, D., Tadevosyan, V., Wang, Z., Navasardyan, S., and Shi, H. StreamingT2V: Consistent, dynamic, and extendable long video generation from text. In CVPR, pp. 2568– 2577, 2025. 2

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020. 2

Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., HoldGeoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al. RELIC: Interactive video world model with longhorizon memory. arXiv preprint arXiv:2512.04040, 2025. 3

Huang, J., Zhou, Q., Rabeti, H., Korovko, A., Ling, H., Ren, X., Shen, T., Gao, J., Slepichev, D., Lin, C.-H., et al. VIPE: Video pose engine for 3D geometric perception. arXiv preprint arXiv:2508.10934, 2025a. 16

Huang, T., Zheng, W., Wang, T., Liu, Y., Wang, Z., Wu, J., Jiang, J., Li, H., Lau, R., Zuo, W., et al. Voyager: Longrange and world-consistent video diffusion for explorable 3D scene generation. ACM TOG, 44(6):1–15, 2025b. 22

Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. Self Forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems, volume 38, pp. 167283–167308, 2025c. 2, 3, 5, 15

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al. VBench: Comprehensive benchmark suite for video generative models. In CVPR, pp. 21807–21818, 2024. 22

HunyuanWorld, T. HunyuanWorld 1.0: Generating immersive, explorable, and interactive 3D worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025. 3

Kang, M., Zhang, R., Barnes, C., Paris, S., Kwak, S., Park, J., Shechtman, E., Zhu, J.-Y., and Park, T. Distilling diffusion models into conditional gans. In ECCV, pp. 428–447. Springer, 2024. 3

Kim, J., Kang, J., Choi, J., and Han, B. FIFO-Diffusion: Generating infinite videos from text without training. In Advances in Neural Information Processing Systems, volume 37, pp. 89834–89868, 2024. 2

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4

- Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024a. 3, 15
- Kong, X., Liu, S., Lyu, X., Taher, M., Qi, X., and Davison, A. J. EscherNet: A generative model for scalable view synthesis. In CVPR, pp. 9503–9513, 2024b. 3

Kuaishou. Kling video model, 2024. https://klinga i.com/global/. 3

Li, J., Tang, J., Xu, Z., Wu, L., Zhou, Y., Shao, S., Yu, T., Cao, Z., and Lu, Q. Hunyuan-GameCraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025a. 3, 6, 7, 8

Li, R., Torr, P., Vedaldi, A., and Jakab, T. VMem: Consistent interactive video scene generation with surfel-indexed view memory. In ICCV, 2025b. 2, 3, 6, 7, 8

- Li, R., Yi, B., Liu, J., Gao, H., Ma, Y., and Kanazawa, A. Cameras as relative positional encoding. In Advances in Neural Information Processing Systems, volume 38, pp. 15984–16009, 2025c. 3, 4, 5
- Li, S., Xue, F., Baranwal, C., Li, Y., and You, Y. Sequence parallelism: Long sequence training from system perspective. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2391–2404, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.134. URL https: //aclanthology.org/2023.acl-long.134/. 6

Li, X., Wang, T., Gu, Z., Zhang, S., Guo, C., and Cao, L. FlashWorld: High-quality 3D scene generation within seconds. In ICLR, 2026. 3

Li, Z., Li, C., Mao, X., Lin, S., Li, M., Zhao, S., Xu, Z., Li, X., Feng, Y., Sun, J., et al. Sekai: A video dataset towards world exploration. In Advances in Neural Information Processing Systems, volume 38, 2025d. 16

Lin, S., Wang, A., and Yang, X. SDXL-Lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 3

Lin, S., Xia, X., Ren, Y., Yang, C., Xiao, X., and Jiang, L. Diffusion adversarial post-training for one-step video generation. In ICML, 2025a. 3

Lin, S., Yang, C., He, H., Jiang, J., Ren, Y., Xia, X., Zhao, Y., Xiao, X., and Jiang, L. Autoregressive adversarial post-training for real-time interactive video generation. In Advances in Neural Information Processing Systems, volume 38, pp. 41061–41086, 2025b. 3

Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al. DL3DV-10K: A largescale scene dataset for deep learning-based 3D vision. In CVPR, pp. 22160–22169, 2024. 16

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In ICLR,

2023. 2, 4

Liu, F., Sun, W., Wang, H., Wang, Y., Sun, H., Ye, J., Zhang, J., and Duan, Y. ReconX: Reconstruct any scene from sparse views with video diffusion model. IEEE Transactions on Image Processing, 2026a. 3

Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. Rolling Forcing: Autoregressive long video diffusion in real time. In ICLR, 2026b. 5, 15

Liu, Y., Min, Z., Wang, Z., Wu, J., Wang, T., Yuan, Y., Luo, Y., and Guo, C. WorldMirror: Universal 3D world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726, 2025. 9

Lu, Y., Ren, Y., Xia, X., Lin, S., Wang, X., Xiao, X., Ma, A. J., Xie, X., and Lai, J.-H. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In ICCV, pp. 16818–16829, 2025. 3

Mao, X., Li, Z., Li, C., Xu, X., Ying, K., He, T., Pang, J., Qiao, Y., and Zhang, K. YUME-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025a. 3

Mao, X., Lin, S., Li, Z., Li, C., Peng, W., He, T., Pang, J., Chi, M., Qiao, Y., and Zhang, K. YUME: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025b. 3

Minimax. Hailuo video model, 2024. https://hailuo ai.video. 3

Miyato, T., Jaeger, B., Welling, M., and Geiger, A. GTA: A geometry-aware attention mechanism for multi-view transformers. In ICLR, 2024. 3

Parker-Holder, J., Ball, P., Bruce, J., Dasagi, V., Holsheimer, K., Kaplanis, C., Moufarek, A., Scully, G., et al. Genie 2: A large-scale foundation world model. 2024. URL https://deepmind.google/discover/blo g/genie-2-a-large-scale-foundation-w orld-model/. 2

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In ICCV, pp. 4195–4205, 2023. 3, 4

Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. You only look once: Unified, real-time object detection. In CVPR, pp. 779–788, 2016. 16

Ren, X., Shen, T., Huang, J., Ling, H., Lu, Y., NimierDavid, M., M¨uller, T., Keller, A., Fidler, S., and Gao, J. Gen3C: 3D-informed world-consistent video generation with precise camera control. In CVPR, pp. 6121–6132, 2025. 2, 3, 6, 7, 8

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022. 2

Runway. Introducing GEN-3 alpha: A new frontier for video gneration. 2024. URL https://runwayml .com/research/introducing-gen-3-alpha. 22

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 3

Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., and Rombach, R. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In ACM SIGGRAPH Asia, pp. 1–11, 2024a. 3

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. In ECCV, pp. 87–103. Springer, 2024b. 3

Shin, J., Li, Z., Zhang, R., Zhu, J.-Y., Park, J., Shechtman, E., and Huang, X. MotionStream: Real-time video generation with interactive motion controls. In ICLR, 2026. 3

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 2024. 2, 5

Sun, W., Chen, S., Liu, F., Chen, Z., Duan, Y., Zhang, J., and Wang, Y. DimensionX: Create any 3D and 4D scenes from a single image with controllable video diffusion. In ICCV, 2025a. 3

Sun, W., Wei, F., Zhao, J., Chen, X., Chen, Z., Zhang, H., Zhang, J., and Lu, Y. From virtual games to real-world play. arXiv preprint arXiv:2506.18901, 2025b. 3

Tang, J., Liu, J., Li, J., Wu, L., Yang, H., Zhao, P., Gong, S., Yuan, X., Shao, S., and Lu, Q. Hunyuan-GameCraft2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025. 3

Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. Diffusion models are real-time game engines. In ICLR, 2025. 3

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 15

Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. DUSt3R: Geometric 3D vision made easy. In CVPR, pp. 20697–20709, 2024a. 8

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In Advances in Neural Information Processing Systems, volume 36, pp. 8406–8441, 2023. 3

Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., and Shan, Y. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, pp. 1–11, 2024b. 3, 6

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 3

Wu, J. Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M. Z., Fidler, S., Gojcic, Z., and Ling, H. Difix3D+: Improving 3D reconstructions with single-step diffusion models. In CVPR, pp. 26024–26035, 2025. 16

Xiang, J., Gu, Y., Liu, Z., Feng, Z., Gao, Q., Hu, Y., Huang,

- B., Liu, G., Yang, Y., Zhou, K., et al. PAN: A world model for general, interactable, and long-horizon world simulation. arXiv preprint arXiv:2511.09057, 2025. 3

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In ICLR, 2024. 15

Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., and Pan, X. WorldMem: Long-term consistent world simulation with memory. In Advances in Neural Information Processing Systems, volume 38, pp. 49632–49652, 2026. 2, 3, 4, 5

Xu, J., Zou, X., Huang, K., Chen, Y., Liu, B., Cheng, M., Shi, X., and Huang, J. EasyAnimate: A high-performance long video generation method based on transformer architecture. In ACM MM, 2025. 22

Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al. LongLive: Real-time interactive long video generation. In ICLR, 2026. 5, 9, 15, 22

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. CogVideoX: Text-to-video diffusion models with an expert transformer. In ICLR, 2024. 2, 22

Yesiltepe, H., Meral, T. H. S., Akan, A. K., Oktay, K., and Yanardag, P. Infinity-RoPE: Action-controllable infinite video generation emerges from autoregressive self-rollout. In CVPR, 2026. 3

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, B. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, volume 37, pp. 47455–47487, 2024a. 2, 3, 5

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In CVPR, pp. 6613– 6623, 2024b. 3

Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, pp. 22963–22974, 2025. 3, 5

Yu, H.-X., Duan, H., Hur, J., Sargent, K., Rubinstein, M., Freeman, W. T., Cole, F., Sun, D., Snavely, N., Wu, J., et al. WonderJourney: Going from anywhere to everywhere. In CVPR, pp. 6658–6667, 2024. 22

Yu, H.-X., Duan, H., Herrmann, C., Freeman, W. T., and Wu, J. WonderWorld: Interactive 3D scene generation from a single image. In CVPR, pp. 5916–5926, 2025a. 3, 22

Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., and Liu, X. Context as Memory: Scene-consistent interactive long video generation with memory retrieval. In ACM SIGGRAPH Asia, 2025b. 2, 3, 4, 5, 8, 9

Yu, J., Qin, Y., Wang, X., Wan, P., Zhang, D., and Liu, X. GameFactory: Creating new games with generative interactive videos. In ICCV, 2025c. 3

YU, M., Hu, W., Xing, J., and Shan, Y. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025. 3

Yu, W., Xing, J., Yuan, L., Hu, W., Li, X., Huang, Z., Gao, X., Wong, T.-T., Shan, Y., and Tian, Y. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 3, 6, 7

Zhang, H., Chen, X., Wang, Y., Liu, X., Wang, Y., and Qiao, Y. AccVideo: Accelerating video diffusion model with synthetic dataset. arXiv preprint arXiv:2503.19462, 2025a. 3

Zhang, J., Wei, J., Zhang, P., Zhu, J., and Chen, J. SageAttention: Accurate 8-bit attention for plug-and-play inference acceleration. In ICLR, 2025b. 6

Zhang, L., Cai, S., Li, M., Wetzstein, G., and Agrawala, M. Frame context packing and drift prevention in nextframe-prediction video diffusion models. In Advances in Neural Information Processing Systems, volume 38, pp. 30546–30566, 2025c. 3

Zhou, J., Gao, H., Voleti, V., Vasishta, A., Yao, C.-H., Boss, M., Torr, P., Rupprecht, C., and Jampani, V. Stable Virtual Camera: Generative view synthesis with diffusion models. In ICCV, pp. 12405–12414, 2025. 6, 7

Zhou, Y., Wang, Q., Cai, Y., and Yang, H. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024. 22

### A. Training and Inference Details

We adopt the pretrained DiT-based video diffusion models (Wan et al., 2025; Kong et al., 2024a) as the backbone. For the chunk-wise autoregressive diffusion transformer, we group 4 latents into a chunk. For the memory context, we set the temporal memory length to 3 chunks and the spatial memory length to 1 chunk. Moreover, inspired by (Yang et al., 2026; Liu et al., 2026b; Xiao et al., 2024), we observe that preserving the first chunk as attention sink enhances long-term consistency and further mitigates error accumulation. For the bidirectional teacher model Vβ, we also adopt the dual-action representation and construct the memory context as described in Sec. 3.4. The training consists of three stages.

Stage One: Action Control. In the first stage, we focus on injecting action control into the pretrained model. We employ the dual action representation to the pretrained model and train the bidirectional action model for 30K iterations. Then, we replace the 3D self-attention with block causal attention and train for an additional 30K iterations as our AR action model. We find that this enables the AR action model to converge more easily. In this stage, the model is trained on 61 frames (4 chunks) using the Adam optimizer with a learning rate of 1e − 5 and a batch size of 64.

Stage Two: Memory. In the second stage, we train the bidirectional action model and the AR action model with context memory as described in Sec.3.3 and Sec.3.4, respectively. For the bidirectional action model, the generation sequence consists of 4 chunks xj:j+3 (16 latents, 61 frames). We utilize Cj:j+3 − xj:j+3 as a variable-length memory context for the teacher. Following the Flow Matching framework, we apply noise from [0,1] to xj:j+3, while the memory context Cj:j+3 − xj:j+3 undergoes uniform noise sampling from [0,0.2] to enable robust conditioning. Crucially, the training loss are computed strictly on the generation sequence. In stage two, both the bidirectional and AR models are trained on sequences of up to 160 latents (637 frames). Other settings remain the same as in the first stage.

Stage Three: Context Forcing. In the final stage, we use the bidirectional model as the teacher and the AR model as the student for distillation. To stabilize the distillation process, we employ a progressive training strategy that gradually increases the maximum length of the generated latents. For the student model, the learning rate is set to 1e − 6, while for the bidirectional model, which is used to compute the fake score, the learning rate is set to 2e − 7. The models are trained for 2K iteration with a batch size of 64. All other hyperparameters follow (Huang et al., 2025c). For the details of context forcing, see Algorithm 1.

Finally, our AR model can produce multiple chunks in a streaming fashion with KV cache as shown in Algorithm 2. When the user provides only camera poses, we first compute the relative translations and rotations between consecutive poses, and then apply a thresholding mechanism to identify and convert them into discrete actions. Conversely, when only discrete actions are available, we use the predefined relative translations and rotations associated with each action to convert them into camera poses.

Algorithm 1 Context Forcing Training

Algorithm 2 Inference with KV Cache

Require: Number of inference chunks nc Require: Denoise timesteps {k1, . . . , kd} Require: Number of inference chunks nc Require: AR diffusion model Nθ (returns KV embeddings via

Require: Number of denoising timesteps d and chunks n = 4 Require: Dataset D (encoded by 3D VAE) Require: AR diffusion model Nθ Require: Bidirectional diffusion model Vβfake and V real

NθKV)

- 1: loop
- 2: Progressively increase maximum chunk length m
- 3: Sample chunk length j ∼ Uniform(0, 1, . . . , m)
- 4: Sample context x0:j−1 ∼ D
- 5: for i = j, . . . , j + n − 1 do
- 6: Initialize xiniti ∼ N(0, I)
- 7: Reconstitute context memory Ci ⊆ {x0, . . . , xi−1}
- 8: Sample s ∼ Uniform(1, 2, . . . , d)
- 9: Self-rollout xi using Nθ with Ci and s denoising steps
- 10: end for
- 11: Align context memory Ctea ← Cj:j+n−1 − xj:j+n−1
- 12: Sample diffusion timestep k ∼ [0, 1]
- 13: xˆj:j+n−1 ← AddNoise(xj:j+n−1, k)
- 14: Fake score Sfake ← Vβfake(ˆxj:j+n−1, Ctea, k)
- 15: Real score Sfake ← V real(ˆxj:j+n−1, Ctea, k)
- 16: Update θ via distribution matching loss
- 17: Update β via flow matching loss (Huang et al., 2025c)
- 18: end loop

- 1: Initialize model output Xθ ← []
- 2: Initialize KV cache KV ← []
- 3: for i = 0, . . . , nc − 1 do
- 4: Initialize xi ∼ N(0, I)
- 5: Reconstitute context memory Ci ⊆ {x0, . . . , xi−1}
- 6: for s = d, . . . , 1 do
- 7: if s = d and i > 1 then
- 8: Reset KV ← NθKV(Ci, 0)
- 9: end if
- 10: Denoise xi ← Nθ(xi, KV, ks)
- 11: end for
- 12: Add output Xθ.append(xi)
- 13: end for
- 14: return Xθ

- Table 7. Data organization. The table details the four categories of data, their sources, the availability of action annotations (discrete and continuous), the number of clips, and their corresponding ratio in the final dataset.

Category Data Source Annotation (discrete, continuous) # Clips Ratios Real-World Dynamics Sekai (Li et al., 2025d) (✗, ✗) 40K 12.5% Real-World 3D Scene DL3DV (Ling et al., 2024) (✔, ✔) 60K 18.75% Synthetic 3D Scene UE Rendering (✗, ✔) 50K 15.625% Simulation Dynamics Game Video Recordings (✔, ✗) 170K 53.125%

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Figure 11. Camera trajectories included in our collected dataset.

### B. Dataset

- Table 7 provides a comprehensive breakdown of our dataset. We deliberately curate a diverse and high-quality collection, encompassing data from the simulation engine and real world, as well as static and dynamic environments, to guarantee the strong generalization of our model.

For Real-World Dynamics, we employ the Sekai dataset (Li et al., 2025d). However, the original videos often suffer from scene clutter and high dynamics. To address these issues, we implement a rigorous filtering pipeline. Specifically, we apply a state-of-the-art object detection model (YOLO (Redmon et al., 2016)) to identify the presence of crowds and vehicles. By setting an empirical threshold, we filter out clips with high densities of moving objects, thereby ensuring annotation accuracy and stable training.

Regarding the Real-World 3D Scene data (DL3DV (Ling et al., 2024)), the original videos lack diversity in camera movement speed and trajectory complexity. To overcome this, we implement a sophisticated processing workflow: 3D Scene Reconstruction → Customized Trajectory Rendering → Visual Quality Filtering → Video Repair Post-processing (using Difix3D+ (Wu et al., 2025)). This procedure yields additional 60K high-quality real video clips featuring balanced movement speed. During the customized trajectory rendering stage, we deliberately design diverse revisit trajectories to facilitate the learning of long-term geometric consistency. The discrete actions and continuous camera poses in these rendered data are highly accurate, which helps the model learn well-structured action patterns.

For Synthetic 3D Scene (UE Rendering) data, we collect hundreds of UE scenes and obtain 50K video clips by rendering complex, customized trajectories. For Simulation Dynamics (Game Video Recordings), we establish a dedicated game recording platform and invite players to record 170K video clips from 1st/3rd-person AAA games.

We segment the original long videos into 30 to 40 seconds clips and employ a vision-language model to produce descriptive text annotations for every clip. Subsequently, we leverage VIPE (Huang et al., 2025a) to generate high-quality camera poses for clips without camera annotations. However, given the long duration and high scene diversity of our dataset, we observe that pose estimation could be inaccurate, i.e., pose collapse. Therefore, we filter out videos whose adjacent frames exhibit erratic camera positions or rotation angles. Specifically, we utilize the Peak-to-Median Ratio (PMR, the ratio of the maximum inter-frame motion to the median) of inter-frame motions as a detection metric. We subsample the predicted poses every four frames and compute the relative transformations to approximate the camera’s instantaneous velocity. By evaluating the PMR, we can robustly identify impulsive pose jumps. Clips exhibiting a PMR above a conservative threshold (set to 5.0 in our experiments) are classified as unacceptable and are discarded. Finally, for clips lacking discrete action annotations, we derive them from the continuous camera poses: we project the rotation and translation components onto the x,y,z axes and apply a threshold to map these continuous values into corresponding discrete action states.

- Table 8. Left: comparison of Models under Context Forcing. The results are evaluated on the long-term test data. Student (AR) denotes the AR model before distillation, Teacher (bidirectional) refers to the memory-augmented bidirectional video diffusion model, and Final (distilled) represents the AR model after distillation. NFE denotes the number of function evaluations. Right: ablation for memory size. Spa. and Tem. denote the number of chunks in spatial memory and temporal memory, respectively.

NFE PSNR↑ SSIM↑ LPIPS↓ Rdist ↓ Tdist ↓

###### Spa. Tem. PSNR↑ SSIM↑ LPIPS↓ Rdist ↓ Tdist ↓

Student (AR) 100 16.27 0.425 0.495 0.611 0.991 Teacher (Bidirectional) 100 19.31 0.599 0.383 0.209 0.717 Final (Distilled) 4 18.94 0.585 0.371 0.332 0.797

3 1 16.41 0.418 0.502 0.634 1.054 1 3 16.27 0.425 0.495 0.611 0.991

- Fig. 11 illustrates the camera trajectories. Our dataset contains complex and diverse trajectories, including a large number of revisit trajectories, which enables our model to learn precise action control and long-term geometric consistency.

C. Additional Experimental Results

C.1. More Qualitative Results

- Fig. 12 illustrates the results of WorldPlay under various actions and virtual environments. As shown in the first three rows, we can interact with complex composite actions, e.g., various combinations of movements. Moreover, WorldPlay can follow intricate trajectories, such as complex rotations and alternating sequences of rotations and movements as demonstrated in the middle six rows. This enhanced control capability is enabled by our dual action representation, which allows for more precise and reliable action guidance. Furthermore, WorldPlay exhibits strong generalization, enabling it to control different types of agents, e.g., human or animals, to roam within the scenes as shown in the last two rows in Fig. 12 and the last two cases in Fig. 13. For more intuitive perspectives, please refer to the supplementary videos.

#### C.2. Long Video Generation

VBench metrics and comparison with previous state-of-the-art methods. Ours

Gen3C

Fig. 14 presents long video generation results from WorldPlay, we maintain long-term consistency, e.g., frame 1 and frame 252 in the top two examples, and preserve high visual quality throughout the entire sequence. Moreover, our context memory ensures that the generation time for each chunk remains constant and does not increase as the video length grows, enabling real-time interactivity and enhancing the user’s immersive experience. Furthermore, the first three rows of Fig. 13 illustrate the generated results spanning 637 frames.

ViewCrafter

Dynamic Degree

Aesthetic Quality

GameCraft

|Class<br><br>Imaging Quality<br><br>|Color<br><br>Temporal Flickering<br><br>Spatial<br><br>20<br><br>40<br><br>60<br><br>80|
|---|---|
|Motion Smoothness<br><br>Action<br><br>|Overall<br><br>Temporal Style<br><br>|

Matrix-Game-2.0

al Relationship

Object Cla

Multiple Objects

Subject Consistency

Human Ac

ll Consistency

#### C.3. Comparison of Models under Context Forcing

M

We provide a comprehensive comparison of different models under context forcing in Table 8 and Fig. 16. The teacher model exhibits better control capability and visual quality due to the bidirectional nature, which provides reliable guidance during distillation. However, this limits its real-time interactivity. Through context forcing, we mitigate error accumulation while maintaining and even surpassing long-term consistency of the student model, yielding improved overall performance. In addition, context forcing reduces the student model’s inference steps, enabling real-time interaction.

Appearance Style Background Consistency Scene

Figure 15. VBench evaluation.

#### C.4. Ablation for Memory Size

- Table 8 evaluates the effect of different memory sizes. Using a larger spatial memory size leads to slightly better PSNR metric, while a larger temporal memory size better preserves the pretrained model’s temporal continuity, resulting in better overall performance. Moreover, a larger spatial memory size may significantly increase the teacher model’s memory size, as the spatial memory of adjacent chunks may completely differ, while their temporal memory overlaps. This not only increases the difficulty of training the teacher model but also poses challenges for distillation.

|[Figure 156]<br><br>W A S D<br><br>|
|---|

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

W D W D W D S A S A

|[Figure 162]<br><br>W A S D<br><br>|
|---|

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

W A W A S D S D S D

|[Figure 168]<br><br>W A S D<br><br>|
|---|

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

W D W D W A W A W A

|[Figure 174]<br><br>W A S D<br><br>|
|---|

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

|[Figure 180]<br><br>W A S D<br><br>|
|---|

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

|[Figure 186]<br><br>W A S D<br><br>|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

A D

|[Figure 192]<br><br>W A S D<br><br>|
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

W S

|[Figure 198]<br><br>W A S D<br><br>|
|---|

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

W W W

|[Figure 204]<br><br>W A S D<br><br>|
|---|

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

W W W

|[Figure 210]<br><br>W A S D<br><br>|
|---|

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

W W W D D

|[Figure 216]<br><br>W A S D<br><br>|
|---|

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

W W W W W

###### Figure 12. More qualitative results.

|[Figure 222]<br><br>W A S D<br><br>|
|---|

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

|[Figure 232]<br><br>W A S D<br><br>|
|---|

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

|[Figure 242]<br><br>W A S D<br><br>|
|---|

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

|[Figure 252]<br><br>W A S D<br><br>|
|---|

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

W W A A

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

D D D S S

|[Figure 262]<br><br>W A S D<br><br>|
|---|

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

W W A A

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

D D S S S

###### Figure 13. More visualization results.

|[Figure 272]|
|---|

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Frame 1 Frame 14 Frame 28 Frame 42 Frame 56 Frame 70 Frame 84

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Frame 98 Frame 112 Frame 126 Frame 140 Frame 154 Frame 168 Frame 182

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Frame 196 Frame 210 Frame 224 Frame 238 Frame 252 Frame 266 Frame 280

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Frame 294 Frame 308 Frame 322 Frame 336 Frame 350 Frame 364 Frame 378

|[Figure 300]|
|---|

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Frame 1 Frame 14 Frame 28 Frame 42 Frame 56 Frame 70 Frame 84

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Frame 98 Frame 112 Frame 126 Frame 140 Frame 154 Frame 168 Frame 182

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Frame 196 Frame 210 Frame 224 Frame 238 Frame 252 Frame 266 Frame 280

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Frame 294 Frame 308 Frame 322 Frame 336 Frame 350 Frame 364 Frame 378

|[Figure 328]|
|---|

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

W W W W W W

Frame 1 Frame 14 Frame 28 Frame 42 Frame 56 Frame 70 Frame 84

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

W W W W W

Frame 98 Frame 112 Frame 126 Frame 140 Frame 154 Frame 168 Frame 182

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

W W

W W W W

Frame 196 Frame 210 Frame 224 Frame 238 Frame 252 Frame 266 Frame 280

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

W W W

Frame 294 Frame 308 Frame 322 Frame 336 Frame 350 Frame 364 Frame 378

###### Figure 14. Long video generation.

|[Figure 356]<br><br>a) Student (AR)|
|---|

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

|[Figure 361]<br><br>b) Teacher (Bidirectional)|
|---|

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

|[Figure 366]<br><br>c) Final (Distilled)|
|---|

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Figure 16. Visualization of different models under context forcing.

Promptable Event

Sky Darkens Flame Erupts

|[Figure 371]|
|---|

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

S S S S S

Traveler Walks Dog Roams

|[Figure 377]|
|---|

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

W W W W W

Video Continuation

|[Figure 383]|
|---|

[Figure 384]

|[Figure 385]|
|---|

[Figure 386]

Figure 17. Visualization of promptable event and video continuation.

- Table 9. Quantitative comparison on the WorldScore. Bold and underline presents the 1st, Bold indicates the 2nd, and underline means the 3rd.

|Method<br><br>|WorldScore Average<br><br>|Camera Control<br><br>Object Control<br><br>Content Alignment|3D Consistency<br><br>Photometric Consistency<br><br>Style Consistency<br><br>Subjective Quality<br><br>|
|---|---|---|---|
|WonderJourney (Yu et al., 2024) WonderWorld (Yu et al., 2025a) EasyAnimate (Xu et al., 2025) Allegro (Zhou et al., 2024) Gen-3 (Runway, 2024) CogVideoX-I2V (Yang et al., 2024) Voyager (Huang et al., 2025b)|63.75 72.69 52.85 55.31 60.71 62.15 77.62<br><br>|84.60 37.10 35.54 92.98 51.76 71.25<br><br>26.72 54.50 50.76 24.84 57.47 51.48 29.47 62.92 50.49 38.27 40.07 36.73<br><br>85.95 66.92 68.92<br><br><br>|80.60 79.03 62.82 66.56 86.87 85.56 70.57 49.81<br><br>67.29 47.35 73.05 50.31 70.50 69.89 65.60 47.41<br>68.31 87.09 62.82 63.85 86.21 88.12 83.22 62.44<br><br><br>81.56 85.99 84.89 71.09<br>|
|Ours<br><br>|79.74<br><br>|88.76 69.05 66.51<br><br>|86.43 89.07 85.17 73.16<br><br>|

#### C.5. Evaluation on VBench

We evaluate our model on VBench (Huang et al., 2024) across diverse metrics. For each baseline, we provide the same image and action to generate long-horizon videos. The results presented in Fig. 15 demonstrate the superior performance of WorldPlay. Notably, our method achieves outstanding results in key aspects such as consistency, motion smoothness, and scene generalizability.

#### C.6. Evaluation on WorldScore

We conduct a comprehensive evaluation using the WorldScore (Duan et al., 2025) benchmark, which consists of 2,000 diverse test cases encompassing various styles and scenarios for the static setting. Each case requires generating content based on an input image, a text prompt, and a specific camera trajectory. To assess the effectiveness of our model, we compare it against several state-of-the-art 3D and video generation baselines. Following the evaluation protocol established by Voyager (Huang et al., 2025b), we focus on metrics that reflect controllability and generation quality across novel views. As demonstrated in Table 9, our method achieves the highest average score among all compared models, highlighting our model’s superior performance in both precise controllability and high-fidelity generation quality.

### D. User Study

We conduct a comprehensive user study across multiple dimensions, including visual quality, control accuracy, and long-term consistency. In our setup, users are presented with two videos, generated from the same initial image and action inputs, and asked to select their preference based on the specified criteria. To ensure the robustness of our evaluation, we select 300 cases from diverse benchmarks such as VBench (Huang et al., 2024) and WorldScore (Duan et al., 2025), and 300 customized trajectories. The final results are then evaluated by a panel of 30 assessors. As shown in Fig. 18, compared to other baselines, our distilled model achieves superior generation quality across all aforementioned evaluation metrics, clearly demonstrating our model’s capability for both real-time interaction and long-term consistency.

### E. Additional Applications

#### E.1. Promptable Event

Due to the autoregressive nature of WorldPlay, we can modify the text prompt at any time to control the subsequent generated content. Specifically, inspired by LongLive (Yang et al., 2026), we employ a KV-recache technique to refresh the cached key–value states whenever the text prompt is modified. This effectively erases residual information from the previous prompt while preserving the motion and visual cues necessary to maintain temporal continuity. As shown in the upper part of Fig. 17, we can change the weather and trigger a fire eruption, or introduce new objects and characters. Through promptable event, we can generate various complex and

Preference Study Results

| |48.1%<br><br>72.9%<br><br>78.4|92.1%<br><br>%<br><br>88.5%|51.9%|27.1%<br><br>7.9%<br><br>21.6%<br><br>11.5%|
|---|---|---|---|---|
| | | | | |

GameCraft

Ours

Matrix-Game-2.0

Ours

ViewCrafter

Ours

Gen3C

Ours

Teacher (Bidirectional)

Ours

0 20 40 60 80 100

Preference Rate (%)

Figure 18. Human evaluation.

uncommon scenarios, which can benefit agent learning by enabling agents to handle these unexpected situations.

#### E.2. Video Continuation

As shown at the bottom of Fig. 17, WorldPlay can generate follow-up content that remains highly consistent with a given initial video clip in terms of motion, appearance, and lighting. This enables stable video continuation, effectively extending the original video while preserving spatial-temporal consistency and content coherence, which opens up new possibilities in creative video generation and virtual environment construction.

