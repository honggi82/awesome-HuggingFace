## MagicWorld: Towards Long-Horizon Stability for Interactive Video World Exploration

#### Guangyuan Li1,2 Bo Li2 Jinwei Chen2 Xiaobin Hu3 Lei Zhao1,* Peng-Tao Jiang2,* 1College of Computer Science and Technology, Zhejiang University 2vivo BlueImage Lab, vivo Mobile Communication Co., Ltd. 3National University of Singapore Project Page: vivocameraresearch.github.io/magicworld

# arXiv:2511.18886v2[cs.CV]18Mar2026

|[Figure 1]<br><br>30|
|---|

|[Figure 2]<br><br>63|
|---|

|[Figure 3]<br><br>91|
|---|

|[Figure 4]<br><br>106|
|---|

|[Figure 5]<br><br>W A S D<br><br>1|
|---|

OursYUME1.5

⋯

|[Figure 6]<br><br>305|
|---|

|[Figure 7]<br><br>346|
|---|

|[Figure 8]<br><br>391|
|---|

|[Figure 9]<br><br>287<br><br>W A S D|
|---|

Subject Motion Drift Long-Horizon Drift

[Figure 10]

[Figure 11]

|[Figure 12]<br><br>W A S D<br><br>1|
|---|

|[Figure 13]<br><br>30|
|---|

|[Figure 14]<br><br>63|
|---|

|[Figure 15]<br><br>91|
|---|

|[Figure 16]<br><br>106|
|---|

⋯

|[Figure 17]<br><br>287<br><br>W A S D|
|---|

|[Figure 18]<br><br>391|
|---|

|[Figure 19]<br><br>305|
|---|

|[Figure 20]<br><br>346|
|---|

Natural Subject Motion Long-Horizon Stability

[Figure 21]

[Figure 22]

Figure 1. We present MagicWorld, which addresses subject motion drift and long-horizon instability in video world models. Existing methods often exhibit unrealistic subject dynamics and unstable scene evolution as generation progresses. MagicWorld instead preserves natural subject motion and enables stable long-horizon generation, producing coherent dynamics and consistent scene structure over time. Yellow dashed circles highlight the subject in the scene, while the numbers in the top-right corner indicate the frame index.

### Abstract

drift, we incorporate a flow-guided motion preservation constraint that mitigates motion degradation in dynamic subjects, encouraging realistic motion patterns and stable interactions during scene evolution. To mitigate error accumulation in long-horizon interactions, we design two complementary strategies, including a history cache retrieval strategy and an enhanced interactive training strategy. The former reinforces historical scene states by retrieving past generations during interaction, while the latter adopts multi-shot aggregated distillation with dual-reward weighting for interactive training, enhancing long-term stability and reducing error accumulation. In addition, we construct RealWM120K, a real-world dataset with diverse city-walk videos and multimodal annotations to support dynamic perception and long-horizon world modeling. Experimental results demonstrate that MagicWorld improves mo-

Recent interactive video world model methods generate scene evolution conditioned on user instructions. Although they achieve impressive results, two key limitations remain. First, they exhibit motion drift in complex environments with multiple interacting subjects, where dynamic subjects fail to follow realistic motion patterns during scene evolution. Second, they suffer from error accumulation in long-horizon interactions, where autoregressive generation gradually drifts from earlier scene states and causes structural and semantic inconsistencies. In this paper, we propose MagicWorld, an interactive video world model built upon an autoregressive framework. To address motion

*Corresponding author. This work was completed by Guangyuan Li during internship at vivo.

tion realism and alleviates error accumulation during longhorizon interactions.

### 1. Introduction

Video world models [1, 8, 10, 25, 30, 40, 41] learn the evolution of visual scenes conditioned on actions, enabling agents to understand, predict, and plan in open environments. By capturing spatiotemporal structure, motion dynamics, and object interactions, they have shown strong potential in applications such as autonomous driving [4, 21, 43], embodied intelligence [23, 31, 39], and virtualworld generation [1, 18, 49]. Furthermore, through generative video prediction and action-conditioned modeling, video world models [8, 10, 11, 29, 37, 49] can construct interactive virtual environments for low-cost experimentation and policy optimization, enabling stronger generalization and long-term planning.

Currently, interactive video world models [11, 29, 33, 37, 44, 49] have become mainstream, and they are typically built using diffusion models combined with autoregressive generation. However, these methods still face two main challenges, as shown in Fig. 1. (1) They suffer from motion drift in generated scenes, where dynamic subjects fail to follow realistic motion patterns during scene evolution. In real-world settings, interactions between dynamic subjects and static structures are highly coupled and nonlinear. Consequently, subjects that should undergo motion during scene transitions frequently remain static or exhibit degraded motion, undermining the realism of dynamic scene evolution. (2) They lack effective strategies to mitigate error accumulation during long-horizon interactions. As generation proceeds autoregressively, small prediction errors gradually compound over time, leading to drift from earlier observations and causing structural and semantic inconsistencies in extended interactive scenarios.

Based on the above analysis, we propose MagicWorld, an interactive video world model. Our method focuses on resolving motion drift in dynamic environments and alleviating error accumulation across extended interactive generation. Specifically, we introduce a flow-guided motion preservation constraint to prevent motion drift and degradation of dynamic subjects. By enforcing temporal coherence in dynamically evolving regions, the constraint promotes realistic motion patterns and stable interactions with surrounding scene structures across frames. To mitigate error accumulation during long-horizon interactive generation, we design a history cache retrieval strategy and an enhanced interactive training strategy. The former preserves historical scene information across autoregressive steps by storing generated frame latents in a cache and retrieving recent historical representations. The retrieved representations are incorporated into subsequent generation steps,

reinforcing previously established scene structures, maintaining view consistency across viewpoint changes, and reducing progressive drift. The latter adopts an enhanced interactive training strategy based on multi-shot aggregated distribution matching distillation (DMD) with dual-reward weighting. Specifically, unlike prior interactive DMD methods [14, 42] that update the generator after each interaction step, we aggregate the DMD losses across all interaction steps before performing optimization. This design allows the generator to evaluate the overall quality of the generated trajectory rather than optimizing each step independently. Furthermore, we introduce a dual-reward mechanism that weights the distillation objective using both visual quality and motion quality signals. Finally, we construct the RealWM120K dataset, specifically designed to support dynamic object modeling in real-world scenes. It consists of city-walk videos collected from multiple cities worldwide and is accompanied by multimodal annotations, providing a comprehensive benchmark for studying dynamic perception and long-horizon world consistency. Our contributions to the community are fourfold:

- • We propose MagicWorld, an autoregressive interactive video world model, designed to address motion drift in dynamic environments and mitigate error accumulation in long-horizon interactive generation.
- • We introduce a flow-guided motion preservation constrain that enforces temporal coherence in dynamic regions to prevent motion drift and ensure realistic motion evolution of dynamic subjects.
- • We design a history cache retrieval strategy to preserve historical scene states during autoregressive rollout, and an enhanced interactive training strategy based on multi-shot aggregated DMD with dual-reward weighting, jointly improving long-horizon stability and reducing error accumulation.
- • We build the RealWM120K dataset with diverse citywalk videos and multimodal annotations for real-world video world modeling. Extensive experiments demonstrate that MagicWorld outperforms state-of-the-art methods in both VBench metrics and visual quality.

### 2. Related Work

##### 2.1. Video World Models

Recently, video world models [7, 8, 10, 11, 16, 29, 32, 33, 37, 39, 44, 47, 49, 50] have attracted increasing attention, with the goal of learning world dynamics directly from visual data and enabling controllable world evolution and state rollout. For instance, Matrix-Game 2 [49] introduced an interactive world model that enabled controllable game world generation and long-horizon video synthesis. Yume [29] introduced a video world model that generated dynamically explorable worlds from a single input image

and supported text-controlled world evolution. Guo et al. [10] proposed MineWorld, an interactive video world model designed for the Minecraft environment. HY-World 1.5 [16] introduced a systematic interactive world modeling framework that maintained geometric consistency while satisfying real-time latency requirements. Dai et al. [7] proposed FantasyWorld, which jointly modeled video latent representations and implicit 3D fields within a single forward pass. StableWorld [44] introduced long interactive video generation, with a focus on improving generation stability and temporal consistency. LingBot-World [33] introduced an open-source world simulation framework based on video generation, supporting long-horizon video generation with low-latency real-time interaction. However, existing video world models suffer from motion drift in dynamic subjects, where objects that should move remain static or exhibit degraded motion. To address these issues, we introduce a flow-guided motion preservation constraint to prevent motion degradation.

##### 2.2. Long Video Generation

Current long video generation methods [9, 12, 14, 19, 22, 27, 38, 42, 45–47] typically combine diffusion models with autoregressive (AR) prediction, forming a middle-ground paradigm between purely diffusion-based and purely ARbased approaches. For example, CausVid [46] reconstructed video diffusion into a causal AR process and reduced inference steps through distribution-matching distillation. MAGI-1 [35] scaled AR video generation via chunk-wise prediction. Self-forcing [14] narrowed the gap between training and inference in AR video diffusion by simulating inference during training. LongLive [42] performed causal frame-level AR, introducing KVrecache, long-sequence training to enhance long-term consistency and accelerate generation. Reward Forcing [27] adopted reward-driven distribution matching, enabling realtime video generation with improved motion quality.

Existing methods [39, 42, 47] mitigate error accumulation in autoregressive generation by either retrieving historical information or adopting interactive training strategies. For example, some approaches [39, 47] retrieve historical frames based on camera trajectories and FOV overlap, while others maintain a memory bank of past frames and select relevant ones using a greedy matching strategy based on FOV overlap and temporal distance. However, these methods rely heavily on geometric information and may miss semantically relevant historical states when viewpoints differ. In contrast, our method performs retrieval directly in the latent space using similarity matching, without requiring camera information. This allows the model to retrieve historical states that are most relevant to the current generation in terms of structure or semantics, leading to more stable long-horizon and cross-interaction generation. In addition,

during interactive training, instead of updating parameters of the generator after each interaction step [14, 42], we aggregate the DMD losses across all interaction steps before performing optimization. This allows the model to evaluate the overall quality of the generated video rather than optimizing each step independently, further improving longhorizon consistency and mitigating error accumulation.

### 3. Methodology

##### 3.1. Overview

Our method aims to build an interactive video world model that allows users to explore a dynamically evolving world constructed from a single scene image through continuous interactions. We focus on addressing two key challenges in interactive world generation, the motion drift of dynamic subjects and instability in long-horizon interactions, enabling more realistic motion and stable scene evolution over extended rollouts. The overall single interaction pipeline of MagicWorld is illustrated in Fig. 2(a). Given a single scene image I0 ∈ RH×W×3, as the initial observation for world construction, the model generates f frames at each interaction step. At the n-th step interaction, the user provides an action an ∈ A, where A denotes the action space, such

- as keyboard commands (e.g., combinations of W, A, S, and D). This action specifies the intended movement or viewpoint change within the virtual world. Since generation at interaction step n+1 depends on the last frame produced
- at step n, we denote this frame as I(nf). Thus, generation at step n+1 can be formulated as: Vn+1 = G(I(nf),an+1),

where Vn+1 = {I(1)n+1,I(2)n+1,...,I(nf+1) } represents the fframe video segment generated at step n+1. G(·) denotes

the video DiT. After generation, the final frame I(nf+1) is used as the initial state for the next step to support subsequent interaction. Ultimately, under the action sequence {a1,...,an,...}, MagicWorld produces the corresponding sequence of video segments {V1,...,Vn,...}, thereby progressively constructing an explorable world.

##### 3.2. Motion and Geometry Preservation

In video world generation, we observe that existing methods often suffer from motion drift during scene evolution. Dynamic subjects, such as pedestrians or vehicles, are expected to move consistently with scene transitions. However, in practice, they frequently remain static or exhibit incorrect motion patterns, resulting in motion collapse and unrealistic dynamics. This issue stems from the absence of explicit motion-aware constraints that couple dynamic behavior with scene evolution. To address this limitation, we introduce a flow-guided motion preservation constraint that enforces temporal coherence in dynamically evolving regions, ensuring realistic motion patterns and stable interactions throughout interactive generation. In addition, main-

(a) Single Interaction Pipeline

(b) Motion Preservation

[Figure 23]

Latent Pred x (    )

|Latent Pred x (  ) Flow 𝟊(    )→(  )<br><br>|[Figure 24]<br><br>f-1|
|---|
<br><br>|[Figure 25]<br><br>f|
|---|
|
|---|

Current Frame

Current Frame

Generated Latent

|[Figure 26]<br><br>f-1|
|---|

|[Figure 27]| |
|---|---|
| | |

Noise

[Figure 28]

[Figure 29]

Recover ℒ

Action-Guided Geometry

[Figure 30]

[Figure 31]

f

VideoDiT

Encoder

⋮

2

1

Warp

f

###### C

⋮

2 f

1

[Figure 32]

[Figure 33]

[Figure 34]

OpticalFlow

Compression

[Figure 35]

Update

[Figure 36]

[Figure 37]

Estimation

[Figure 38]

###### W A S D

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Retrieval

|[Figure 43]| |
|---|---|
| | |

Action

f

f

f

Cache

⋮

2

⋮

2

⋮

2

1

1

1

Geometric Prior

Current Frame

Prior Video

RGB Video Optical Flow

(c) Enhanced Interactive Training

Trainable Frozen Latent C Concatenate

[Figure 44]

[Figure 45]

Interaction 1 Interaction 2 Interaction m

Output Sequence 1

First frame Action

Current frame Action

Current frame

Action

+ Noise

|[Figure 46]| |
|---|---|
| | |

|[Figure 47]| |
|---|---|
| | |

|[Figure 48]| |
|---|---|
| | |

W A S D

W A S D

###### W A S D

Weighted

OutputVideo Visual&Motion

Real Fake

Average

Reward

⋮

Teacher Gradient

Output Sequence m

⋮

Weighted

History

History

Causal Student Video DiT

Causal Student Video DiT

Causal Student Video DiT

+ Noise

[Figure 49]

[Figure 50]

[Figure 51]

Real Fake

[Figure 52]

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

⋮

⋮

⋮

Decoder

f

⋮

Output Sequence 1 Output Sequence 2 Output Sequence m

2

1

- Figure 2. Overview of MagicWorld. (a) Single interaction pipeline with history cache retrieval. (b) Flow-guided motion preservation enforces temporal coherence in dynamic regions. (c) Enhanced interactive training with multi-shot aggregated reward DMD, jointly optimizing multi-step rollouts with visual and motion rewards to improve motion realism and mitigate error accumulation.

Motion supervision is applied on xpred0 , ensuring that the motion regularization targets the denoised content rather than the noisy intermediate. Since motion supervision is applied in the latent space, the optical flow F is compressed

taining structural stability is equally critical. To preserve structural consistency, we incorporate an action-guided geometry strategy that grounds scene evolution in geometric constraints, thereby stabilizing spatial layouts during action-driven transitions.

to align with xpred0 in dimension. Let x(0fˆ−1) and x(0fˆ) denote two consecutive latent frames from xpred0 , and let F(fˆ−1)→(fˆ) be the forward flow. Here, fˆ = f//4 denotes the number of frames in the latent space, 4 denotes the temporal compression ratio. We warp x(0fˆ) back to latent frame fˆ− 1 using warping operator W(·):

###### 3.2.1. Flow-Guided Motion Preservation

To prevent motion drift in dynamic subjects during scene evolution, we introduce a flow-guided motion preservation constraint that regularizes temporal coherence in motionaware regions, as shown in Fig. 2(b). During training, optical flow is estimated online from ground-truth videos using an optical flow estimation model [34]. To avoid the out-ofmemory issue caused by optical-flow supervision in RGB space, motion supervision is performed in the latent space. Following a velocity-based flow-matching formulation, we construct intermediate latent states via linear interpolation between the clean latent and noise: xt = (1 − t)x0 + tx1, where x1 ∼ N(0,I). The model predicts the velocity field vθ(xt,t), whose the target is defined as x1 − x0. Given the predicted velocity, we reconstruct the clean latent as:

x˜(0fˆ−1) = W x(0fˆ),F(fˆ−1)→(fˆ) . (2)

This warping aligns the content of latent frame fˆ to the coordinate system of latent frame fˆ − 1, enabling direct temporal consistency measurement. We define a motionaware weighting based on the magnitude of the optical flow:

, where the L2 norm measures the flow magnitude and assigns larger weights to regions undergoing stronger motion. The flow-guided motion preser-

w(fˆ−1) = F(fˆ−1)→(fˆ)

2

xpred0 = xt − tvθ (xt,t). (1)

vation constraint is defined as:

1 N ˆ

w(fˆ−1)(i) x(0fˆ−1)(i) − x˜(0fˆ−1)(i) ,

Lmotion =

f i

(3) where i denotes the spatial location index, and N represents the total number of elements involved in the summation. This formulation encourages temporal consistency in motion-aware regions while avoiding excessive constraints on static background areas. By explicitly regularizing regions with significant motion, the model mitigates motion drift and degraded motion behaviors, ensuring that dynamic subjects maintain coherent movement during scene evolution. Finally, the overall training objective combines the velocity flow-matching loss and the motion preservation loss: Lstage1 = LFM + Lmotion.

###### 3.2.2. Action-Guided Geometry

To preserve structural consistency during action-driven scene evolution, we employ an action-guided geometry (AGG) strategy. Specifically, we reconstruct a 3D scene representation from the first frame and transform it according to action-induced viewpoint changes. The reconstructed geometry is projected into multiple views to encode actionconditioned scene transformations into a unified geometric prior. This prior serves as an explicit structural constraint for the world model, guiding scene evolution and stabilizing spatial layouts across interaction steps. AGG consists of two steps.

Action mapping. To convert user interactions into camera trajectories, each discrete action an ∈ A is mapped to a camera extrinsic sequence of length f (i.e., the trajectory at step n):

f k=1

T (an;Θ) = R(nk),t(nk)

,

(4)

R(0)n ,t(0)n ≡ R(nf−)1,t(nf−)1 ,

where Θ denotes tunable parameters (e.g., step size, rotation angle, interpolation steps), k = 1,...,f indexes the discretized camera poses along the trajectory, and R,t are rotation and translation matrices. Given the first frame I(1)n , we estimate its depth D(x) [5] and back-project pixels using camera intrinsics K to obtain a 3D geometric prior: xˆ = K−1x, Xc = D(x)ˆx. The representation is then transformed into the world coordinate system to form the scene prior P.

Geometry projection. At step n, the user provides an action an ∈ A, which is mapped by the action controller to a corresponding camera pose (R(nk),t(nk)). Based on this camera pose, the geometric prior P in the world coordinate system is projected into the new viewpoint, producing the action-guided geometric representation:

Paction,n (k) = Π(P,K,R(nk),t(nk)), (5)

where Π(·) denotes the geometric projection operator. The resulting Paction,n (k) serves as an explicit geometric prior. We render the action-guided geometric sequence into a geometric prior video: Vnprior = R Pactionn ,(k) fk=1 , where R(·) denotes the rendering function. The prior video Vnprior is combined with the current first frame and the noise input, and fed into the Video DiT.

##### 3.3. Long-Horizon Interactive Learning

Interactive video world generation follows an autoregressive process in which each step depends on previously generated results. As interactions continue, prediction errors accumulate and gradually cause scene drift, leading to structural instability and semantic deviations in longhorizon scenarios. To address this issues, we introduce two complementary strategies, a history cache retrieval strategy to reinforce historical scene information during rollout and an enhanced interactive training strategy that explicitly simulates multi-step interaction during training to enhance long-term consistency.

###### 3.3.1. History Cache Retrieval

The goal of history cache retrieval is to store the clean latent features generated at each autoregressive step into a history cache. During the subsequent inference step, the current input frame latent is matched against the cached history via similarity retrieval to obtain the most relevant reference frames, which are then fed into the model as auxiliary conditions. History cache retrieval consists of three steps.

Cache update. Given the latent sequence {L(1)n ,...,L(nfˆ)} generated at step n. We retain only its last-frame latent L(nfˆ) for the next interaction, while the remaining frame latents are appended to the cache pool. We set the cache capacity to 20 latent frames. When updating the history cache, we keep the first-frame latent fixed and only update the remaining entries. The first latent is obtained from the initial interaction and directly corresponds to the latent derived from the input scene image, thus carrying the most essential and stable information about the environment. Once the cache reaches its capacity, the non-fixed entries are replaced in a first-in, first-out manner.

Cache retrieval. At interaction step n, we take the latent of the first frame in the current step as qn. To perform retrieval, we apply spatial pooling to both the query and all cached latents to obtain vector representations: q = pool(qn),ci = pool(hi). Following previous works [17, 26], we compute the cosine similarity:

si = ⟨q,ci⟩ ∥q∥∥ci∥

. (6)

All cached latents are ranked by similarity, and the top-3 entries are selected. Notably, retrieval is independent of

temporal distance. By computing similarity in the latent space, the model retrieves historical states that best match the current iteration in structure or semantics, even if they are from much earlier steps. This design avoids bias toward short-term neighbors and provides the model with relevant contextual cues that help mitigate scene drift.

Cache injection. The retrieved top-3 latents are embedded as history tokens and concatenated with the input tokens along the sequence dimension, providing explicit reference information. History cache retrieval mitigates autoregressive error accumulation by caching and retrieving historical information, supporting stable scene evolution and improved viewpoint consistency across interactions.

###### 3.3.2. Enhanced Interactive Training Strategy

To mitigate error accumulation and enhance long-horizon stability, we adopt an enhanced interactive training strategy, as shown in Fig. 2(c). The overall optimization procedure consists of two stages, ordinary differential equation (ODE) initialization and multi-shot aggregated reward distribution matching distillation (DMD) training.

ODE initialization. We start from a pretrained bidirectional video DiT trained with the Lstage1 objective and reformulate it as a deterministic generative process via ODE initialization. This conversion enables causal generation while preserving the distribution learned by the original diffusion model. Based on the ODE-initialized model, we construct a causal student generator Gˆ(·).

Multi-shot aggregated distillation. During training, we explicitly simulate interactive rollout for M steps. At interaction step m, given the current state Lm and action am, the student generates a latent segment: Xˆ m = Gˆ(Lm,am), and updates the state via Lm+1 = Xˆ (mf). This process exposes the model to its own predictions and aligns with the interactive inference procedure. For each step, we compute a DMD loss L(DMDm) . However, unlike prior DMD-based approaches [14, 42] that update the generator immediately, we aggregate the distillation losses over the entire M-step interaction rollout before performing optimization:

LaggDMD =

M

1 M

L(DMDm) . (7)

m=1

This aggregated optimization allows the generator to perceive the overall quality of the trajectory rather than optimizing each step independently. By delaying parameter update until the full rollout is completed, the generator can evaluate later interaction outcomes and adjust earlier predictions accordingly, improving long-horizon consistency and mitigating error accumulation.

Dual-reward quality guidance. To further enhance longhorizon stability, we introduce reward weighting at the interaction level. For each generated segment Xˆ m, we decode

it into RGB and evaluate it using a pretrained VideoReward model [24], which produces visual quality score rmVQ and motion quality score rmMQ. These scores assess visual consistency and motion fidelity in multi-step interactive video generation. We combine them as: rm = rmVQ + rmMQ. The reward-weighted multi-shot aggregated DMD objective becomes:

1 M

Lstage2 =

M

exp(rm)L(DMDm) . (8)

m=1

By jointly optimizing over full interaction rollouts and weighting distillation with both visual and motion quality signals, the proposed strategy aligns training with interactive inference and enhances long-horizon stability. Unlike prior distillation approaches [14, 27, 42], our enhanced interactive training strategy aggregates multi-step supervision before parameter updates, enabling the model to perceive and optimize overall interaction quality across extended sequences.

##### 3.4. Dataset Construction

Recent advances in video world models [8, 10, 11, 39, 44, 49] have demonstrated strong performance in game environments and relatively static or weakly dynamic scenes. However, these models still struggle in real-world settings, especially in street scenes involving dense dynamic agents and non-trivial camera motion. To bridge this critical gap, we construct RealWM120K, a video dataset specifically designed for real-world environments, with a focus on citywalk street scenes collected across the globe. The dataset covers major cities from multiple countries and regions, spanning diverse conditions across different times of day and seasons, thereby providing a rich and realistic distribution of real-world city street scenarios. The dataset construction consists of four steps:

Video collection. We collect city-walk videos from YouTube, which naturally capture unconstrained camera motion, diverse city layouts, and real-world dynamic agents. A subset of the video URLs is obtained from the Sekai [20] collection. In total, we collect 2,659 videos with over 4,000 hours of raw footage.

Clip extraction. Raw videos are segmented into 140K short clips, each with a duration of 10 seconds at 30 FPS and a resolution of 720×1280, balancing temporal continuity and computational feasibility for world model training.

Clip filtering. To ensure semantic coherence and visual quality, we employ Qwen2.5-VL-72B [36] to assess both semantic quality and aesthetic quality for each clip. Starting from approximately 140K candidate clips, we retain the top 85%, resulting in about 120K video clips used in the final dataset.

Multi-modal annotation. Each clip is annotated with a caption generated by Qwen2.5-VL-72B [36], focusing on scene

W A S D

W A S D

W A S D

Scene Image

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

110 140 210 240 370 400

2.0 Wan2.2

Camera

1.5 Matrix-Game

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

110 140 210 240 370 400

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

110 140 210 240 370 400

1.5 YUME

Base HY-World

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

110 140 210 240 370 400

Ours LingBot-World

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

110 140 210 240 370 400

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

110 140 210 240 370 400

- Figure 3. Qualitative comparison of different methods on the same scene image under long-term interactions. Our method maintains stable scene geometry and natural motion dynamics while significantly reducing error accumulation, leading to more consistent generation during long-horizon interactions. Please zoom in for details.

layout, spatial structure, dynamic objects, lighting, weather, and overall atmosphere, enabling semantic grounding for world models. We extract synchronized geometric representations, including camera trajectories, point clouds, and object masks extracted using ViPE [13], and depth maps estimated by Video Depth Anything [6]. These modalities provide explicit supervision for spatial consistency, geometry-aware reasoning, and dynamic object understanding in real-world environments.

- 4. Experiments

- 3, 9, and 3, respectively. We evaluate on RealWM120KVal, which contains 100 real-world images covering diverse scenes, generating videos at 480×832 resolution with 81 frames at 16 FPS during inference.
- 4.2. Comparison with SOTA Methods

We compare our method with recent interactive video world models as well as camera-based video generation models. For a fair comparison, we adopt the autoregressive inference strategy for all camera trajectory methods, where the last frame of each interaction serves as the first frame of the subsequent one. We use VBench [15] metrics for evaluation, covering temporal and visual quality.

##### 4.1. Implementation Details

Qualitative Comparison. Fig. 3 presents qualitative comparisons of long video sequences generated through continuous interactions within the same scene. Compared with existing methods, our approach demonstrates superior geometric preservation and motion consistency, while exhibiting significantly reduced error accumulation over extended interactions. We further evaluate viewpoint consistency, as shown in Fig. 4. Our method maintains strong scene consistency during viewpoint changes. Specifically, when the camera leaves a particular viewpoint and returns after multiple interaction steps, the original scene structure and dynamic subjects are still well preserved. Additional video comparisons are provided in the supplementary material.

We adopt the pretrained weights from Wan2.1-Fun-V1.11.3B [2] as the foundational model. All training experiments are conducted on 56 NVIDIA A800 GPUs. For bidirectional video DiT training, the batch size is set to 4 per GPU. The input video resolution is 480×832, and each training sample contains 81 frames. We train the bidirectional model for 60k steps with a learning rate of 2×10−5. For distillation training, we first collect 40k ODE pairs and fine-tune the causal student model for 10k steps. This is followed by an additional 10k steps of DMD-based training. The learning rate is set to 6×10−6. The multi-shot DMD rollout length is 4. The chunk size of latent frames, the local attention window size, and the sink size are set to

[Figure 104]

[Figure 105]

[Figure 106]

LingBot-WorldBaseOurs

[Figure 107]

[Figure 108]

W A S D

W A S D

W A S D

W A S D

1 152 187 243 286

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

W A S D

W A S D

W A S D

W A S D

1 152 187 243 286

- Figure 4. Visual comparison of viewpoint consistency. Our method maintains consistent scene appearance across viewpoint changes, ensuring the scene remains consistent when revisiting the same viewpoint after multiple transitions. Please zoom in for details.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

W A S D

W A S D

W A S D

W A S D

1 120 219 247 307

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

W A S D

W A S D

W A S D

W A S D

1 120 219 247 307

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

W A S D

W A S D

W A S D

W A S D

1 120 219 247 307

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

W A S D

W A S D

W A S D

W A S D

1 120 219 247 307

[Figure 134]

Geometric Inconsistency View Inconsistency

[Figure 135]

View Inconsistency

[Figure 136]

Motion Drift

[Figure 137]

Motion Drift

[Figure 138]

Motion Drift

[Figure 139]

Flow

[Figure 140]

Geometry

[Figure 141]

Geometry History

[Figure 142]

[Figure 143]

Geometry History

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- Figure 5. Qualitative comparison of different component variants in the ablation study. The first row shows results from the bare model. As each component is progressively incorporated, the generated results show clear improvements in scene consistency, viewpoint consistency, and subject motion quality. Please zoom in for details.

Quantitative Comparison. Tab. 1 reports the quantitative comparison on the RealWM120K-Val benchmark using VBench metrics. Our method achieves the best overall score (0.8547), outperforming previous methods across efficiency, temporal quality, and visual quality. In terms of efficiency, our method achieves competitive latency (15s), ranking second while being significantly faster than most existing world models. For temporal quality, our method obtains the best results across all metrics, demonstrating strong temporal stability and dynamic preservation during interactive generation. For visual quality, our method ranks second in aesthetic quality and achieves the best image

quality score, indicating that the proposed geometry and motion preservation mechanisms improve temporal coherence without sacrificing visual fidelity.

##### 4.3. Ablation Study

To validate the effectiveness of each component and strategy, we conduct ablation studies on both the bidirectional model and the DMD training scheme. All variants are evaluated under the same settings as the main experiments.

Effect of history cache retrieval. To evaluate the role of the history cache retrieval, we compare the geometry-based model with and without the history cache. As shown in

- Table 1. Comparison of interactive video world models and camera-based video generation methods under VBench [15] on the RealWM120K-Val dataset. The best and second-best results are highlighted in red and blue, respectively. Latency (s) is measured with a batch size of 1 on a 480×832×81 video using each model’s default inference steps for a fair comparison, and is evaluated on the NVIDIA H20 GPU.

Efficiency Temporal Quality Visual Quality Latency(s)

Methods

Overall Score ↑

Temporal Flick. ↑

Motion smooth. ↑

Subject Cons. ↑

Background Cons. ↑

Aesthetic Qua. ↑

Image Qua. ↑

ViewCrafter [48] 302 0.7807 0.9569 0.9790 0.8188 0.8748 0.5001 0.5543

- Wan2.1-Camera [2] 39 0.8172 0.9586 0.9801 0.8778 0.9173 0.5018 0.6674

- Wan2.2-Camera [3] 94 0.7935 0.9573 0.9846 0.8508 0.8982 0.4861 0.5837 Matrix-Game 2.0 [11] 11 0.8082 0.9457 0.9814 0.8476 0.8990 0.4971 0.6784 YUME [29] 1918 0.8314 0.9491 0.9865 0.9098 0.9264 0.5239 0.6926 YUME 1.5 [28] 19 0.8334 0.9502 0.9821 0.9115 0.9221 0.5270 0.7077 HY-World 1.5 [16] 163 0.8452 0.9719 0.9918 0.9409 0.9348 0.5303 0.7014 LingBot-World-Base [33] 1920 0.8364 0.9653 0.9755 0.9423 0.9261 0.5406 0.6687 Ours 15 0.8547 0.9752 0.9921 0.9627 0.9408 0.5394 0.7182

Multi-shot DMD

Long-Horizon Instability

Degraded Visual and Motion Quality

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

1 120 320 350 410

W A S D

W A S D

W A S D

W A S D

Degraded Visual and Motion Quality

Multi-shot aggregated DMD

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

1 120 320 350 410

W A S D

W A S D

W A S D

W A S D

[Figure 164]

Multi-shot aggregated reward DMD

Stable and High-Quality Long-Horizon Generation

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

1 120 220 350 410

W A S D

W A S D

W A S D

W A S D

- Figure 6. Visual comparison of different DMD training schemes. Multi-shot aggregated reward DMD produces more stable and higherquality long-horizon video generation. Red dashed circles indicate low-quality regions, while yellow dashed circles highlight high-quality regions. Please zoom in for details.

Tab. 2, incorporating the history cache leads to further performance improvement, indicating that accessing historical scene states helps preserve previously generated structures and maintain long-horizon scene consistency during interactive generation. Visual results in Fig. 5 further show that the history cache improves viewpoint consistency during scene transitions.

dicating that enforcing temporal coherence in dynamic regions effectively prevents motion drift and promotes more realistic motion of dynamic subjects.

Effect of the base model architecture. To evaluate the impact of different architectures, we compare the base bidirectional model with a geometry-based model. As shown in Tab. 2, the geometry-based model achieves better performance, indicating that geometric priors help stabilize scene structure. In particular, geometric guidance provides explicit spatial cues for scene evolution, thereby reducing structural inconsistencies during viewpoint changes.

Effect of flow-guided motion preservation. To examine the impact of the flow-guided motion preservation constraint, we compare models trained with and without the proposed motion supervision, as shown in Tab. 2 and Fig. 5. The results show clear improvements in subject consistency, in-

###### Effect of DMD training scheme. To investigate the im-

- Table 2. Quantitative comparison of different variants using the RealWM120K-Val dataset. Overall score denotes the average over all VBench [15] metrics.

Temporal Quality Visual Quality Temporal Flick. ↑

Methods Overall Score

Motion smooth. ↑

Subject Cons. ↑

Background Cons. ↑

Aesthetic Qua. ↑

Image Qua. ↑

Base Model 0.8238 0.9599 0.9807 0.9075 0.9182 0.5023 0.6742 Geometry ✓ 0.8391 0.9690 0.9897 0.9361 0.9288 0.5174 0.6936 Geometry ✓ History ✓ 0.8412 0.9701 0.9901 0.9373 0.9294 0.5258 0.6945 Geometry ✓ History ✓ Flow ✓ 0.8471 0.9709 0.9912 0.9599 0.9334 0.5278 0.6995

Multi-shot DMD 0.8463 0.9717 0.9889 0.9585 0.9326 0.5261 0.7001 Multi-shot aggregated DMD 0.8496 0.9736 0.9903 0.9604 0.9357 0.5312 0.7066 Multi-shot aggregated reward DMD 0.8547 0.9752 0.9921 0.9627 0.9408 0.5394 0.7182

pact of different DMD optimization schemes, we compare three variants: (1) Multi-shot DMD, performs gradient updates immediately after each interaction step, (2) Multishot aggregated DMD, aggregates the DMD losses over M interaction steps and updates the generator, (3) Multishot aggregated reward DMD, further incorporates reward weighting into the aggregated DMD objective. As shown in Tab. 2 and Fig. 6, multi-shot aggregated DMD consistently improves performance compared with the step-wise optimization scheme. Further incorporating reward guidance achieves the best performance across all metrics, with clear improvements in visual quality and motion quality. These results suggest that using an aggregated DMD loss during interactive training is more effective than immediate gradient updates after each step, as it enables the generator to perceive the quality of later interaction outcomes. When combined with reward guidance, the training process is further encouraged to favor higher-quality trajectories, thereby effectively mitigating error accumulation during long-horizon interactions.

### 5. Conclusion

We present MagicWorld, an autoregressive interactive video world model designed to address motion drift of dynamic subjects and error accumulation in long-horizon interactions. To mitigate motion degradation, we introduce a flowguided motion preservation constraint enforcing temporal coherence in dynamic regions. To reduce error accumulation, we design a history cache retrieval strategy with an enhanced interactive training scheme. We also construct RealWM120K, a real-world multimodal dataset for interactive video world modeling. Extensive experiments show that MagicWorld consistently outperforms existing methods in both VBench metrics and visual quality.

### 6. Acknowledgement

We thank Siming Zheng and Shuolin Xu for their initial support and suggestions to construct our basic framework.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2
- [2] alibaba pai. Wan2.1-fun-v1.1-1.3b-control-camera, 2025. 7, 9
- [3] alibaba pai. Wan2.2-fun-5b-control-camera, 2025. 9
- [4] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15791–15801, 2025. 2
- [5] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. In International Conference on Learning Representations, 2025. 5
- [6] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv:2501.12375, 2025. 7
- [7] Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, and Yonggang Qi. Fantasyworld: Geometry-consistent world modeling via unified video and 3d prediction. In The Fourteenth International Conference on Learning Representations, 2026. 2, 3
- [8] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024. 2, 6
- [9] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 3

- [10] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025. 2, 3, 6
- [11] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 2, 6, 9
- [12] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025. 3
- [13] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 7
- [14] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 6
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7, 9, 10
- [16] Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025. 2, 3, 9
- [17] Guangyuan Li, Jun Lv, Yapeng Tian, Qi Dou, Chengyan Wang, Chenliang Xu, and Jing Qin. Transformer-empowered multi-scale contextual matching and aggregation for multicontrast mri super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20636–20645, 2022. 5
- [18] Ruihuang Li, Caijin Zhou, Shoujian Zheng, Jianxiang Lu, Jiabin Huang, Comi Chen, Junshu Tang, Guangzheng Xu, Jiale Tao, Hongmei Wang, et al. Hunyuan-game: Industrialgrade intelligent game creation model. arXiv preprint arXiv:2505.14135, 2025. 2
- [19] Zongyi Li, HU Shujie, LIU Shujie, Long Zhou, Jeongsoo Choi, Lingwei Meng, Xun Guo, Jinyu Li, Hefei Ling, and Furu Wei. Arlon: Boosting diffusion transformers with autoregressive models for long video generation. In The Thirteenth International Conference on Learning Representations. 3
- [20] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025. 6
- [21] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12037–12047, 2025. 2

- [22] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350,

2025. 3

- [23] Huaping Liu, Di Guo, and Angelo Cangelosi. Embodied intelligence: A synergy of morphology, action, perception and learning. ACM Computing Surveys, 57(7):1–36, 2025. 2
- [24] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025. 6
- [25] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 2
- [26] Liying Lu, Wenbo Li, Xin Tao, Jiangbo Lu, and Jiaya Jia. Masa-sr: Matching acceleration and spatial adaptation for reference-based image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6368–6377, 2021. 5
- [27] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678,

2025. 3, 6

- [28] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025. 9
- [29] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. 2, 9
- [30] J Parker-Holder, P Ball, J Bruce, V Dasagi, K Holsheimer, C Kaplanis, A Moufarek, G Scully, J Shar, J Shi, et al. Genie 2: A large-scale foundation world model. 2024. 2
- [31] Lei Ren, Jiabao Dong, Shuai Liu, Lin Zhang, and Lihui Wang. Embodied intelligence toward future smart manufacturing in the era of ai foundation model. IEEE/ASME Transactions on Mechatronics, 2024. 2
- [32] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world model. arXiv preprint, 2025. 2
- [33] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026. 2, 3, 9
- [34] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field

- transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020. 4
- [35] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 3
- [36] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [37] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284,

2025. 2

- [38] Xunzhi Xiang, Yabo Chen, Guiyu Zhang, Zhongyu Wang, Zhe Gao, Quanming Xiang, Gonghu Shang, Junqi Liu, Haibin Huang, Yang Gao, et al. Macro-from-micro planning for high-quality and parallelized autoregressive long video generation. arXiv preprint arXiv:2508.03334, 2025. 3
- [39] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025. 2, 3, 6
- [40] Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In NeurIPS 2023 Workshop on Generalization in Planning. 2
- [41] Sherry Yang, Jacob C Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Position: video as the new language for real-world decision making. In Forty-first International Conference on Machine Learning, 2024. 2
- [42] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622,

2025. 2, 3, 6

- [43] Xuemeng Yang, Licheng Wen, Tiantian Wei, Yukai Ma, Jianbiao Mei, Xin Li, Wenjie Lei, Daocheng Fu, Pinlong Cai, Min Dou, et al. Drivearena: A closed-loop generative simulation platform for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26933–26943, 2025. 2
- [44] Ying Yang, Zhengyao Lv, Tianlin Pan, Haofan Wang, Binxin Yang, Hubery Yin, Chen Li, Ziwei Liu, and Chenyang Si. Stableworld: Towards stable and consistent long interactive video generation. arXiv preprint arXiv:2601.15281, 2026. 2, 3, 6
- [45] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649,

2025. 3

- [46] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From

- slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 3
- [47] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141,

2025. 2, 3

- [48] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 9
- [49] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025. 2, 6
- [50] Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. Versecrafter: Dynamic realistic video world model with 4d geometric control. arXiv preprint arXiv:2601.05138, 2026. 2

