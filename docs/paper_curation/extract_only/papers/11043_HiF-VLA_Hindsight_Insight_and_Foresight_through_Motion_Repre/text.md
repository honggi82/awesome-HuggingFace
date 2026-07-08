# arXiv:2512.09928v2[cs.RO]9Apr2026

## HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models

Minghui Lin1, Pengxiang Ding1,2†, Shu Wang1, Zifeng Zhuang1,2, Yang Liu1, Xinyang Tong1, Wenxuan Song3, Shangke Lyu4, Siteng Huang2† , Donglin Wang1,5 , 1Westlake University 2Zhejiang University 3HKUST(GZ) 4Nanjing University 5Westlake Robotics

linminghui@westlake.edu.cn, siteng.huang@gmail.com

https://hifvla.github.io https://github.com/OpenHelix-Team/HiF-VLA

##### Abstract

Vision-Language-Action (VLA) models have recently enabled robotic manipulation by grounding visual and linguistic cues into actions. However, most VLAs assume the Markov property, relying only on the current observation and thus suffering from temporal myopia that degrades long-horizon coherence. In this work, we view motion as a more compact and informative representation of temporal context and world dynamics, capturing inter-state changes while filtering static pixel-level noise. From this perspective, HiF-VLA equips a motion-centric world model for the VLA, enabling agents to reason about temporal dynamics for future evolution during action generation. Building on this idea, we propose HiF-VLA (Hindsight, Insight, and Foresight for VLAs), a unified framework that leverages motion for bidirectional temporal reasoning. HiFVLA encodes past dynamics through hindsight priors, anticipates future motion via foresight reasoning, and integrates both through a hindsight-modulated joint expert to enable a “think-while-acting” paradigm for long-horizon manipulation. As a result, HiF-VLA surpasses strong baselines on LIBERO-Long and CALVIN ABC-D benchmarks, while incurring negligible additional inference latency. Furthermore, HiF-VLA achieves substantial improvements in realworld long-horizon manipulation tasks, demonstrating its broad effectiveness in practical robotic settings.

##### 1. Introduction

Vision-Language-Action models (VLAs) [4, 15, 16, 19, 53, 56] have emerged as a promising framework for robotic manipulation, leveraging powerful Vision-Language Models (VLMs) [1, 2, 10, 18, 34] to map visual and language representations to the action space. Despite progress, most

† Project leads. Corresponding authors.

VLAs [19, 20, 35] implicitly assume a Markov property, predicting actions solely from the current observation without explicitly modeling temporal dependencies. This simplification leads to a form of temporal myopia. However, long-horizon manipulation requires reasoning that extends beyond the present, maintaining temporal continuity across visual, language, and motor modalities. Without such temporal reasoning, dependencies between consecutive actions deteriorate, resulting in fragmented trajectories and diminished task-level coherence.

Recent efforts [26, 35, 54] have sought to alleviate this temporal myopia by incorporating historical context, most commonly by stacking multiple past observation frames. However, this approach is fundamentally limited. Stacking raw frames is not only computationally prohibitive and increases inference latency, hindering real-time control (see Tab. 3), but it also introduces substantial pixel-level redundancy. This deluge of static information often obscures the salient, task-relevant dynamics, making it difficult for the model to distinguish meaningful changes from background noise. We argue that a more precise and efficient representation of history is not the raw visual content of the past, but the motion that transpired between states. Motion serves as a direct and compact proxy for temporal memory and environment dynamics, faithfully capturing the dynamics of interactions (such as an object being moved or a drawer being closed), while discarding redundant static information. This makes motion an ideal primitive for representing history in a way that is both expressive and computationally efficient.

Building on the principle of motion as a compact representation of history, we argue that robust decision-making requires bidirectional temporal reasoning, connecting the past to the future. An intelligent agent must possess hindsight, the ability to interpret recent dynamics that led to its current state, grounding its decisions in verified past outcomes. At the same time, it must exhibit foresight, the

[Figure 1]

Ours

Past Work

| |(a) Comparison with Existential Methods| |
|---|---|---|
| | | |

- (b) Inference Process
- (c) Strong Performance

Input Output

Chunk i-1 Chunk i Chunk i+1

History Frames Current Observation Predicted Subgoal Actions

[Figure 2]

[Figure 3]

 Redundancy

frame t-3 frame t-2

frame t-1 frame t

frame t+n

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

𝑎𝑡: 𝑎𝑡+𝑛

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

T

O

O O

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Vanilla VLA

 Inefficiency

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

A A A A A A A A A A A A

 Unclear Structure

(e.g. OpenVLA-OFT)

[∆𝑥,∆𝜃,∆𝐺𝑟𝑖𝑝]

𝑯 × 𝑾 × 𝟑

[Figure 44]

[Figure 45]

Video Encoding (e.g. H.264,MPEG-4)

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

T O O

O

VLA

𝑎𝑡: 𝑎𝑡+𝑛

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

|[Figure 66]<br><br>[Figure 67]<br><br>× (𝑾//𝟏𝟔)|
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

with History Frames

 Conciseness  Efficiency  Clear Structure

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

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A A A A A A A A A A A A

[Figure 98]

(e.g. RoboVLMs)

(𝑯//𝟏𝟔) ?) × 𝟐

[∆𝑥,∆𝜃,∆𝐺𝑟𝑖𝑝]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Hindsight Insight

T

VLA O O with Subgoal

O

𝑀𝑉𝑡:𝑡+𝑛

[Figure 108]

Foresight Joint

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

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

(e.g. CoT-VLA)

A A A A A A A A A A A A

Hindsight

[Figure 133]

Expert

Action

[Figure 134]

Hindsight + Insight + Foresight

[Figure 135]

[Figure 136]

###### Inference Latency

|[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>O<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>| |
|---|---|
| | |

###### Calvin ABC-D

###### Real-world

[Figure 145]

###### LIBERO-Long

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

O

O

T

HiF-VLA (Ours)

(cover&stack)

4.35

8×History

96.4

4.28

57.9

[Figure 152]

[Figure 153]

[Figure 154]

94.0

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

A A A A A A A A A A A A

[Figure 180]

[Figure 181]

4.1

4×History

-58.3%

[Figure 182]

33.3

-29.3%

89.2

[Figure 183]

3.92

Current Observation i.e. Insight

87.7

[Figure 184]

Hindsight

Predicted

T O A Task Instruction

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

or Foresight

Actions

Multi-frames History HiF-VLA

Seer OpenVLA-OFT

HiF-VLA (Ours)

𝝅𝟎

- Figure 1. (a-b) Comparison with existing methods: VLAs rely on instantaneous observations [19, 20] (a-top), stack multiple past frames [26, 37] (a-second), or generate pixel-level subgoals [49, 52] (a-third), suffering from redundancy, high inference cost, and weak structure. In contrast, HiF-VLA (a-bottom) jointly models Hindsight, Insight, and Foresight, expanding the temporal receptive field bidirectionally for compact, structured, and efficient reasoning. (c) HiF-VLA reduces inference latency and achieves state-of-the-art performance on LIBERO-Long and CALVIN ABC-D, significantly outperforming the baseline in real-world experiments.

系列 1 系列 2

2002/1/5

80

60

40

20

0

2002/1/7 2002/1/6

capability to anticipate plausible future dynamics, enabling proactive, goal-directed behavior rather than purely reactive responses. Such bidirectional reasoning over past and future dynamics is also central to world models. We identify motion as the natural bridge unifying these two temporal dimensions. Consequently, we propose HiF-VLA (Hindsight, Insight, and Foresight for VLA Models), a unified framework that leverages motion to structure this reasoning process. As detailed in Fig. 2, HiF-VLA comprises three components: 1) Hindsight prior acquisition: encodes historical frames into structured, low-dimensional motion vectors [17] as hindsight, preserving dynamics without redundant pixels. 2) Foresight reasoning with insight: interprets task instructions and current observations to anticipate plausible foresight motions and latent action tokens. 3) Hindsight-modulated joint expert: applies hindsight as a top-down constraint on foresight and action streams, which interact within the expert decoder to generate temporally coherent actions. As shown in Fig. 1, compared to methods that rely solely on frame stacking or subgoal prediction, HiF-VLA more effectively bridges perception, dynamics, and control within a unified representational space, forming a motion-centric World Action Model [46]. This design enables an embodied “think-whileacting” paradigm, enhancing robustness, temporal continuity, and causal consistency in long-horizon manipulation tasks.

Our main contributions are summarized as follows:

[Figure 195]

[Figure 196]

[Figure 197]

- • We propose HiF-VLA, a VLA framework endowed with bidirectional spatio-temporal completion. By incorporating motion vectors as structured, low-dimensional temporal primitives, HiF-VLA explicitly expands the temporal receptive field, enabling temporally consistent and efficient action prediction while reducing redundancy.
- • We propose a hindsight-modulated joint expert that unifies temporal and action representations within a unified space, enabling a “think-while-acting” paradigm for causally consistent and temporally coherent long-horizon motion generation.
- • Extensive experiments demonstrate that our approach achieves substantial performance gains on widely adopted long-horizon benchmarks, while exhibiting strong temporal scalability and high inference efficiency.

##### 2. Related Work

Vision-Language-Action Models. VLA models learn endto-end mappings from language instructions and visual observations to low-level actions. Prior works can be broadly grouped by their action policies. Diffusion-based methods such as RDT-1B [27], CogACT [23], and DexVLA [40] employ iterative denoising to synthesize continuous control trajectories; autoregressive approaches like RT-2 [56] and OpenVLA [19] discretize continuous actions into tokens and predict them sequentially using VLM backbones;

regression-oriented systems such as OpenVLA-OFT [20], VLA-Adapter [38] and SF [22] adopt bidirectional attention to directly learn continuous actions via L1 regression. However, most VLA models do not explicitly model temporal dependencies. They often neglect the importance of historical information and reasoning capabilities through VLMs, both of which have been shown to substantially improve performance in long-horizon video tasks [9, 12, 32].

Temporal Modeling and Inference in Robotics. Temporal modeling and foresight reasoning have been extensively studied in long video understanding and generation, but remain relatively underexplored in robotic manipulation. Existing approaches largely focus on one-sided reasoning. One line of work incorporates historical frames as visual prompts within the VLM input context, as in TraceVLA [54], Octo [35], GR-2 [8], RoboVLMs [26]. Another emphasizes foresight by predicting future visual subgoals to guide action generation, such as CoT-VLA [52], Seer [37], UniVLA [39], and UP-VLA [49], which typically condition an inverse dynamics model (IDM) on predicted future subgoals to infer actions. However, frame-level historical encoding and pixel-level future prediction introduce substantial computational overhead and redundancy, while failing to capture the importance of bidirectional temporal information. To address these, we propose HiF-VLA, which efficiently unifies hindsight, insight, and foresight within a single framework, enabling temporally coordinated decision-making for action generation.

##### 3. Method

###### 3.1. Preliminary

We begin by defining the setting and notation for VLA reasoning. Robot expert demonstrations are denoted as Dr = (l,a1...T,o1...T). At each time step t, the model receives an observation ot and a textual instruction l, and the objective is to predict an action chunk at:t+n over a horizon of length n. A vanilla VLA Pθ [20] builds on visionlanguage models (VLMs), learning from diverse manipulation demonstrations to transfer vision-language understanding and generation capabilities to embodied scenarios. Formally:

a˜t:t+n ∼ Pθ at:t+n | ot,l . (1) Our key insight is to expand the temporal receptive field available prior to action execution by compensating sparse visual observations. To this end, we propose HiF-VLA, as illustrated in Fig. 2, a unified framework built upon a vanilla VLA architecture, which incorporates compact historical information prior mhist−h:t of length h and additionally predicts future motion mt:t+n conditioned on the current insight. Formally, the inference process can be expressed as:

(˜at:t+n,m˜ t:t+n)∼Pθ′(at:t+n,mt:t+n|ot,l,mhist−h:t). (2)

During training, we augment the input with historical information and jointly predict both the future n-step motion and the corresponding actions. During inference, motion decoding is optional and can be omitted depending on downstream task requirements.

###### 3.2. Hindsight Prior Acquisition

Relying solely on instantaneous visual observations for action decision-making often fails to provide stable perception, especially under challenging conditions such as occlusion or repeated executions. Hence, capturing consistent dynamic patterns of the manipulator from historical states is critical. Existing approaches [8, 35, 37] typically maintain a sliding window of past observations and concatenate them with the current observation to form a global representation. However, this frame-stacking strategy introduces significant redundancy: the high similarity between adjacent frames makes it difficult for the model to focus on taskrelevant dynamics, potentially dispersing its attention.

To address this, we introduce a compressed historical prior: Motion Vectors (MVs) [17]. In video codec standards like H.264 [42] and MPEG-4 [21], MVs predict the displacements of macroblocks between adjacent frames to avoid pixel-wise redundancy. Importantly, MVs are not a coarse approximation: combined with keyframes, they enable near-lossless video reconstruction and maintain high compression. This property provides a natural, efficient, and faithful solution for capturing historical dynamics. Formally, let (x,y) denote the position of a macroblock in an image (size H × W × 3). The motion vector is defined as:

###### MVt−1:t(x,y) = (xt − xt−1,yt − yt−1), (3)

where (xt,yt) and (xt−1,yt−1) denote the positions of the macroblock in consecutive frames ot and ot−1. We adopt MPEG-4 to extract keyframes and motion information, regarding the current observation ot at time step t as the keyframe, while maintaining a historical window of length m, forming GOP (Group of Pictures) units, GOP = [MVt−m:t−m+1,...,MVt−2:t−1,MVt−1:t,ot]. Here, MVs follow the MPEG-4 16×16 macroblock layout and are represented as a tensor of size h × (H//16) × (W//16) × 2, compactly encoding the historical motion trajectories of entities in the scene. Compared to raw frames, this representation significantly reduces redundancy while retaining taskrelevant dynamics, thereby providing structured spatiotemporal priors to the decision layer.

Then, we adopt a lightweight ViT-based [11] hindsight encoder, combined with shallow 3D convolutions, to encode hindsight motions into compact hindsight tokens Mh ∈ RK

h×d.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

(c) Hindsight-Modulated Joint Expert

Current

History Frames Observation

History Mvs Current Observation

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

[Figure 204]

- (a) Hindsight Prior

Acquisition

- (b ) Foresight

Reasoning with Insight

- (c) HindsightModulated Joint Expert details

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

|[Figure 209]<br><br>[Figure 210]|
|---|

|[Figure 211]<br><br>[Figure 212]|
|---|

|[Figure 213]<br><br>[Figure 214]|
|---|

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Video Encoding

Motions Actions

(e.g. H.264,

MPEG-4)

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

MLP MLP

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Modulation Modulation

Capture Observations

Task Instructions

###### Current Observation/Insight

Foresight Tokens

Action

Tokens

[Figure 229]

Joint Attention

Q K V

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Linear Linear

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

VLM

Modulation Modulation

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

|[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>|Execute Actions|
|---|---|
| | |

[Figure 264]

AdaLN AdaLN

[Figure 265]

[Figure 266]

Action Prediction

𝑎𝑡:𝑎𝑡+𝑛

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

| | | |
|---|---|---|
| | | |

[Figure 271]

Visual Hindsight

[Figure 272]

[Figure 273]

[Figure 274]

Hindsight Encoder

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Visual Foresight

𝑀𝑉𝑡:𝑡+𝑛

- Figure 2. HiF-VLA Pipeline. (a) In Hindsight Prior Acquisition (see Sec. 3.2), HiF-VLA encodes dense historical frame sequences into compact Motion Vector (MV) streams, forming structured hindsight primitives that capture temporal dynamics without pixel redundancy. (b) In Foresight Reasoning with Insight (see Sec. 3.3), the VLM interprets the task instruction and current observation to infer plausible foresight motions and corresponding latent action tokens. (c) Finally, the Hindsight-Modulated Joint Expert (see Sec. 3.4) fuses hindsight, foresight, and action representations within a unified latent space, producing temporally consistent and causally coherent action predictions.
- 3.3. Foresight Reasoning with Insight

in foresight motion tokens Mf ∈ RK

f×d and action latent tokens Af ∈ RK

a×d. Formally, the reasoning process can be expressed as: (Mf,Af) = Fθ(ot,l).

Beyond historical cues, foresight is equally critical for robot tasks. It allows the agent to evaluate the consequences of its actions before execution, promoting more reliable action decision-making. Existing approaches, such as CoTVLA [52] and UP-VLA [49], achieve visual CoT reasoning by generating future visual subgoals. However, these approaches depend on pixel-level predictions prone to local distortions and semantic drift. The resulting dense and redundant subgoals obscure task-relevant dynamics, while the reliance on discrete frame prediction (without continuous temporal modeling) further undermines temporal consistency in complex environments.

We aim for the model to reason about visual foresight and action execution in parallel, and subsequently integrate and refine information from these distinct reasoning streams. This design enriches the diversity of the VLM’s internal thought process and unlocks the potential of parallel reasoning.

###### 3.4. Hindsight-Modulated Joint Expert

Conventionally, the action expert receives high-level representations from the VLM and translates them into low-level control signals. However, predicting actions in isolation lacks reasoning about the future dynamics. Motion, as the physical manifestation of actions in the visual space, provides complementary information: jointly predicting both motion and actions enables VLA models to better align semantic understanding with underlying dynamics.

To address these limitations, we propose an efficient foresight reasoning mechanism guided by current insight, as shown in Fig. 2(b). Instead of predicting raw future pixels, we employ more general and structured MVs as spatiotemporal targets for future action execution. MVs compactly encode the trajectory evolution of the manipulator within the scene, effectively reducing pixel redundancy while providing a structured spatial prior.

To achieve such synergistic reasoning, we propose the Hindsight-Modulated Joint Expert, which jointly models action and motion as two complementary streams within a shared temporal latent space and where historical information is introduced as a conditional prior to guide future inference and suppress erroneous replay of historical trajectories. Importantly, we integrate historical motion priors as adaptive temporal conditions, rather than embedding them directly into the VLM input. Injecting extra history motion into the VLM risks disrupting the established alignment be-

Specifically, we introduce Kf learnable foresight query tokens {q1f,q2f,...,qKf

}, together with Ka empty action tokens {q1a,q2a,...,qKa

f

}, into the VLM embedding space. These tokens are concatenated with the original inputs (including the task instruction l, the current observation ot), and then fed into VLM Fθ, enabling parallel reasoning over continuous visual dynamics and action generation, resulting

a

tween visual and language modalities [51] (see Fig. 4). Instead, historical motion tokens Mh are encoded into a compact conditional representation and modulate the joint reasoning process via Adaptive Layer Normalization (AdaLN) conditioning, where layered modulation and regularization jointly constrain future motion-action patterns.

Formally, we define three types of sequence representations: hindsight motion tokens Mh ∈ RK

h×d (used only as conditional input), foresight motion latent tokens Mf ∈ RK

a×d. As shown in Fig. 2(c), the foresight motion tokens and action tokens form two parallel streams, which interact via cross-stream joint attention while retaining separate FFNs to ensure complementary yet disentangled representations. The hindsight tokens Mh are projected through a linear layer to obtain the conditioning vectors hc, which is then injected into each joint expert module via AdaLN to modulate both foresight and action representations z ∈ {Mf,Af}:

f×d, and action tokens Af ∈ RK

M˜f,A˜f = JointExpert(Mf,Af | hc), (4)

z − µ(z) σ(z)

+ β(hc), (5)

AdaLN(z;hc) = γ(hc) ·

where σ(z) and µ(z) are the mean and standard deviation of z, γ(hc) and β(hc) are modulation parameters from hc. The Joint Attention employs non-causal self-attention over a sequence formed by concatenating Mf and Af, from which Q, K, and V are jointly projected.

Finally, the fused motion and action representations are projected through their respective heads to generate the future motions m˜ t:t+n and the actions a˜t:t+n. In general, the hindsight-modulated joint expert not only explicitly models the complementary relationship between action and motion, but also employs historical conditioning regularization to guide temporally consistent and physically plausible behaviors, thereby enhancing the stability and causal consistency of action.

###### 3.5. Unified HiF-VLA Model

Input Representation. Following OpenVLA-OFT [20], we feed the current observation into both the DINOv2 [29] and SigLIP [47] visual encoders to obtain a hybrid visual embedding. At the same time, we initialize a set of learnable foresight motion queries and empty action tokens. These tokens are concatenated with the instruction embedding and the current observation embedding to form the multi-modal input sequence. In parallel, the historical motion sequence is compressed via spatial-temporal convolution, aggregated by a ViT encoder, and projected into the latent space of the joint expert as the hindsight prior.

Parallel Decoding and Joint Expert. The VLM adopts a non-causal attention mask, enabling joint prediction of future motion latents and action latents. The three streams of

tokens (hindsight, foresight, predicted action) are then fused via the Hindsight-Modulated Joint Expert.

Training Objective. We now describe in detail how we train the model to predict both actions and motion. To ensure both action and motion predictions remain wellcalibrated, we define two L1-loss objectives:

n

n

1 n

1 n

LMV =

|mt+j − m˜ t+j|,LA =

|at+j − a˜t+j|.

j=1

j=1

(6) The overall loss combines the two objectives:

Lall = LA + λ · LMV , (7)

where λ is a balancing factor between action accuracy and motion reconstruction quality, set to 0.01.

##### 4. Experiments

In this section, we design experiments to address the following research questions (RQs):

- RQ1: How does HiF-VLA perform compared to SOTA methods on challenging long-horizon benchmarks?
- RQ2: Does HiF-VLA effectively mitigate the redundancy and inefficiency issues in conventional approaches?
- RQ3: How well does HiF-VLA maintain inference scalability as the temporal horizon increases?
- RQ4: Through ablation studies, how do different components contribute to HiF-VLA’s overall performance?
- RQ5: Can HiF-VLA successfully handle long-horizon tasks on real-world robotic platforms? 4.1. Overall Performance

Experimental Setups. We evaluate HiF-VLA on two long-horizon benchmarks. LIBERO-Long [25] comprises ten multi-subgoal manipulation tasks across diverse scenes. CALVIN ABC-D [28] includes four indoor environments (A-D); policies are trained on A-C and evaluated on the unseen D to assess generalization on consecutive tasks. All experiments are conducted under two settings following [20]: a third-view setup using the primary camera, or a multiview setup using both the primary and wrist cameras.

Implementation Details. We adopt Prismatic-7B [18] as the VLM backbone, and initialize it with weights from OpenVLA [19], which were pretrained on OXE [30]. All other modules are randomly initialized. Training is performed on 8 NVIDIA A100 GPUs with a global batch size of 64. A fixed temporal chunk of n=8 is adopted for both action and foresight modeling, while the hindsight window is variable (default 8). Fine-tuning is conducted for 150k steps on LIBERO and 80k on CALVIN. The main baseline is OpenVLA-OFT [20], which shares the same pretrained initialization. Additional comparisons include Seer [37], VPP [13], and π0 [4], etc.

Put mugs on left and right plates

Put mug on plate, pudding right

Put mug in

Put soup and box in basket

Put box and butter in basket

Turn on stove and put pot

Put bowl in drawer and close

Pick book and place it in back

Put soup and sauce in basket

Put both pots on stove

Method Avg.SR

microwave

and close

Policy inputs: third-view image, language instruction

OpenVLA [19] 54.0 35.0 95.0 65.0 45.0 40.0 80.0 60.0 45.0 20.0 55.0 UniVLA* [24] 63.0 64.0 82.0 76.0 96.0 58.0 98.0 24.0 74.0 32.0 26.0 MemoryVLA [31] 93.4 92.0 96.0 96.0 100 100 100 96.0 96.0 62.0 96.0 OpenVLA-OFT* [20] 91.0 82.0 96.0 96.0 94.0 90.0 96.0 92.0 100 70.0 94.0 HiF-VLA(Ours) 94.4 94.0 98.0 100 100 94.0 100 90.0 98.0 76.0 94.0

Policy inputs: third-person image, wrist-view image, language instruction

Seer (scratch) [37] 78.7 80.0 90.0 91.7 81.7 85.0 65.0 86.7 88.3 51.7 66.7 Seer [37] 87.7 91.7 90.0 98.3 100 91.7 93.3 85.0 88.3 61.7 71.7 UniVLA* [24] 90.0 100 92.0 94.0 98.0 86.0 100 80.0 100 70.0 82.0 OpenVLA-OFT* [20] 94.0 90.0 98.0 98.0 98.0 96.0 100 92.0 100 72.0 96.0 HiF-VLA(Ours) 96.4 88.0 98.0 100 100 100 100 96.0 100 82.0 100

Table 1. Performance comparison on the LIBERO-Long benchmark. We report the average success rate (%) across 10 tasks with “Avg. SR”. Results marked with “∗” were reproduced using the official open-source code. Bold indicates the best performance.

Result Analysis. 1) LIBERO-Long: As shown in Tab. 1, we present the detailed performance of HiF-VLA across 10 tasks in the LIBERO-Long benchmark. We evaluate both third-view and multi-view inputs over 500 trials. Compared to the baseline third-view method, our approach achieves a 94.4% success rate, representing a 3.4% absolute improvement. Notably, our third-view variant performs on par with multi-view baselines, underscoring HiF-VLA’s robust temporal reasoning capability. Moreover, our method achieves a 96.4% success rate under the multi-view setting, consistently outperforming other state-of-the-art VLA models. 2) CALVIN-ABC-D: We further evaluate the generalization capability of HiF-VLA, which was trained on the ABC dataset and evaluated in the D environment. As shown in Tab. 2, our method surpasses the baseline by 0.25 in terms of the average task length metric and achieves superior performance under both third-view and multi-view settings. These results clearly answer RQ1. This improvement can be attributed to the bidirectional temporal perception and reasoning architecture incorporated in our model, which enables a more effective understanding of long-term action dependencies and consequently leads to enhanced adaptability and robustness in complex long-horizon tasks.

###### 4.2. Efficiency and Redundancy Analysis

Experimental Setup. We conduct experiments exclusively on LIBERO-Long to evaluate the efficiency and redundancy aspects, and the results are presented in Tab. 3. For the baseline (1), we adopt the method proposed by [20]. (2) “+ Subgoal” and (4) “+ History Frames” correspond to variants that incorporate RGB-based prediction and historical information following [8, 52]. (3) “+ Foresight (Ours)” and (5) “+ Hindsight (Ours)” denote variants that introduce Motion-

Method 1 2 3 4 5 Avg. Len. ↑ Policy inputs: third-view image, instruction

SuSIE [5] 87.0 69.0 49.0 38.0 26.0 2.69 OpenVLA [19] 91.3 77.8 62.0 52.1 43.5 3.27 CLOVER [6] 96.0 83.5 70.8 57.5 45.4 3.53 VPP [13] 90.9 81.5 71.3 62.0 51.8 3.58

- π0 [4] 93.7 83.2 74.0 62.9 51.0 3.65 UniVLA [7] 95.5 85.8 74.8 66.9 56.5 3.80 HiF-VLA(Ours) 93.5 87.4 81.4 75.9 69.4 4.08 Policy inputs: third-person and wrist-view image, instruction GR-1 [44] 85.4 71.2 59.6 49.7 40.1 3.06 Vidman [41] 91.5 76.4 68.2 59.2 46.7 3.42

- π0 [4] 93.8 85.0 76.7 68.1 59.9 3.92 UP-VLA [49] 92.8 86.5 81.5 76.9 69.9 4.08 OpenVLA-OFT [20] 96.3 89.1 82.4 75.8 66.5 4.10 RoboVLMs [26] 98.0 93.6 85.4 77.8 70.4 4.25 Seer [37] 96.3 91.6 86.1 80.3 74.0 4.28 VPP [13] 96.5 90.9 86.6 82.0 76.9 4.33 HiF-VLA(Ours) 98.5 94.1 88.1 81.4 73.1 4.35

Table 2. Performance comparison on the CALVIN ABC-D benchmarks. We report the average number of successfully completed tasks across five consecutive instructions. Bold indicates the best performance.

based foresight and historical information, respectively. (6) “+ Hindsight + Foresight (Ours)” integrates both types of Motion-based information simultaneously. Note that we used third-person input, a hindsight length of 4, and a batch size of 4 in all experiments.

Result Analysis. As shown in Tab. 3(2) and (3), incorporating subgoal or foresight prediction improves task suc-

- 91
- 92
- 93
- 94
- 95
- 96
- 97

[Figure 279]

96.2 96.4

- 91
- 92
- 93
- 94
- 95
- 96
- 97

- 91
- 92
- 93
- 94
- 95
- 96
- 97

95.8 96.0

[Figure 280]

96.2 96.4

[Figure 281]

2000

96.2 96.4

|[Figure 282]| | |
|---|---|---|
| | | |

95.8 96.0

HiF-VLA(ours) Baseline+H-frames Baseline+H-F-frames

95.8 96.0

2000

SuccessRate(%)

2000

[Figure 283]

HiF-VLA(ours)

|| | |
|---|---|---|
| | | |

HiF-VLA(ours) Baseline+H-frames Baseline+H-F-frames

SuccessRate(%)

SuccessRate(%)

94.4

Baseline+H-frames

1500

Latency(ms)

94.4

94.4

1500

Latency(ms)

1500

93.6

Baseline+H-F-frames

Latency(ms)

93.2

93.4

93.6

93.6

93.2

93.4

1000

93.2

93.4

1000

1000

500

Thrid-view Multi-view

500

500

Thrid-view Multi-view

Thrid-view Multi-view

4 8 16 32

4 8 16 32

0

4 8 16 32

Length of Hindsight

0

0

[Figure 285]

[Figure 286]

0 4 8 16 32

Length of Hindsight

Length of Hindsight

[Figure 287]

0 4 8 16 32

0 4 8 16 32

Length of Hindsight

Length of Hindsight

Length of Hindsight

|HiF-VLA(o| |urs)<br><br>DecoderDecoderDecoder| | |
|---|---|---|---|---|
| |[Figure 288]<br><br>Baseline+His.-frames<br><br>Baseline+His.+Subg96.2 96.4<br><br>[Figure 289]<br><br>Third-view<br><br>VLMVLMVLM<br><br>| |.-frames95<br><br>Multi-v<br><br>|[Figure 290]<br><br>OOM<br><br>.8 96.0<br><br>iew|

Put the black bowl in the bottom drawer of the cabinet and close it

98

2000

Decoder

Decoder

[Figure 291]

[Figure 292]

Put the black bowl in the bottom drawer of the cabinet and close it

98

2000

Put the black bowl in the bottom drawer of the cabinet and close it

98

2000

Decoder

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Third-view Multi-view

|[Figure 301]<br><br>HiF-VLA(ours)<br><br>Baseline+His.-frames<br><br>Baseline+His.+Subg.-frames|<br><br>OOM|
|---|---|
| | |

HiF-VLA(ours)

Third-view Multi-view

[Figure 303]

OOM

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Baseline+His.-frames

VLM

###### VLM

[Figure 315]

[Figure 316]

96.2 96.4

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

96.2 96.4

###### VLM

95.8 96.0

95.8 96.0

[Figure 321]

96

1500

Baseline+His.+Subg.-frames

[Figure 322]

[Figure 323]

96

1500

96

1500

Latency(ms)

SuccessRate(%)

Latency(ms)

SuccessRate(%)

Latency(ms)

SuccessRate(%)

94.4

94.4

94.4

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

[Figure 324]

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

94

1000

[Figure 325]

[Figure 326]

94

1000

93.6

94

1000

93.6

93.6

93.4

93.2

93.4

93.4

93.2

93.2

t-32 t-16

t-32 t-16

t-32 t-16

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

| | |
|---|---|
|[Figure 339]<br><br>92.892.892.8| |
|0 4<br><br>Len<br><br>4 8<br><br>Lengt|8 16 32<br><br>gth of Hindsight<br><br>16 32<br><br>h of Hindsight<br><br>94.494.494.4|

92

500

92

500

92

500

To VLM

To VLM

To VLM

[Figure 340]

[Figure 341]

90

0

90

0

90

0

To Expert

To Expert

4 8 16 32

To Expert

4 8 16 32

0 4 8 16 32

0 4 8 16 32

Length of Hindsight

Length of Hindsight

Length of Hindsight

Length of Hindsight

t-8 t

t-8 t

t-8 t

(a) Historical Frame Example

(b) Inference Efficiency

(c) Effect of Hindsight

Figure 3. Effect of hindsight length on performance and efficiency. (a) Example of historical frames. (b) HiF-VLA maintains low inference s in third-view and multi-view perspectives.

2000

2000

98

98

2000

[Figure 342]

[Figure 343]

[Figure 344]

Put the black bowl in the bottom drawer

Put the black bowl in the bottom drawer of the cabinet and close it

|HiF-VLA(ours)<br><br>Baseline+H-frames<br><br>Baseline+H-F-frames| |
|---|---|
| | |

[Figure 345]

|Put the black bowl in the bottom drawer of the cabinet and close it<br><br>98<br><br>96.2 96.4<br><br>95.8 96.0 1500<br><br>latency as hindsight length increases. (c) Perfor|96.2 96.4<br><br>HiF-VLA(ours)<br><br>Baseline+H-frames<br><br>Baseline+H-F-frames<br><br>mance of hindsigofthe| |95.8 96.0<br><br>htcabinetof differentandcloseit lengths<br><br>|
|---|---|---|---|
|[Figure 346]<br><br>[Figure 347]<br><br>96<br><br>94.4<br><br>Methods<br><br>Peak GPU Memory(GB) ↓<br><br>Latency (ms) ↓| |[Figure 348]<br><br>[Figure 349]<br><br>Avg. SR↑| |

[Figure 350]

[Figure 351]

HiF-VLA(ours)

Baseline+H-frames

[Figure 353]

96.2 96.4

94.4

94.4

95.8 96.0

94.4

Baseline+H-F-frames

1500

96

96

1500

SuccessRate(%)

SuccessRate(%)

SuccessRate(%)

SuccessRate(%)

SuccessRate(%)

SuccessRate(%)

Decoder Decoder

Decoder Decoder

Decoder Decoder

94.4

94.4

92.8

92.8

92.8

1000

94

1000

94

- (1) Baseline 30.8 (1.00×)
- (2) + Subgoal 38.2 (1.24×)

- (3) + Foresight (Ours) 31.8 (1.03×)

- (4) + History frames 63.6 (2.06×) 229.5 (3.15×) 90.4

- (5) + Hindsight (Ours) 31.4 (1.02×) 117.7 (1.61×) 92.2

- (6) + Hindsight + Foresight (Ours) 32.2 (1.05×) 121.6 (1.67×) 93.2

|72.9 (1.00 )|91.0|
|---|---|
|93.2<br><br>×<br><br>115.9 (1.59×) 82.7 (1.13×)<br><br>|93.6<br><br>93.4<br><br>t-32 t-16<br><br>91.8 92.2<br><br>|

1000

94

93.6

93.6

93.2

93.2

93.4

93.4

###### VLM VLM

###### VLM VLM

###### VLM VLM

To Expert

To Expert

To Expert

t-32 t-16

t-32 t-16

[Figure 354]

[Figure 355]

[Figure 357]

500

92

500

92

[Figure 358]

[Figure 359]

500

92

To VLM

To VLM

To VLM

(a) (b)

(a) (b)

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

Instruction𝑀ℎ RGB 𝑀𝑓 𝐴𝑓 Instruction RGB 𝑀𝑓 𝐴𝑓 𝑀ℎ

Thrid-view Multi-view

Thrid-view Multi-view 0

Thrid-view Multi-view 0

0

90

90

90

(a) Injecting VLM

(b) Conditioning Decoder

(c) Results

0 4 8 16 32

4 8 16 32

0 4 8 16 32

4 8 16 32

0 4 8 16 32

4 8 16 32

Length of Hindsight

Length of Hindsight

Length of Hindsight

Length of Hindsight

t-8 t

t-8 t

Length of Hindsight

Length of Hindsight

Figure 4. Performance comparison on different hindsight embedding locations. (a) represents direct injection into the VLM, and (b) represents conditional embedding as an expert decoder. (c) shows the performance of both on LIBERO-Long. Mh denotes the hindsight tokens, Mf represents the foresight tokens and Af is the action tokens.

t-8 t

Table 3. Performance comparison between multi-frame baselines variants and HiF-VLA variants. Peak GPU memory (GB) during training and inference latency (ms) are reported. The history and hindsight lengths are fixed to 4 for a fair and computationally feasible comparison across all methods.

cess rates; however, subgoal-based methods incur substantial latency overhead (1.59× slower than the baseline). In contrast, our foresight head introduces negligible additional cost (0.13× latency and 0.03× GPU Memory). Moreover, as illustrated in Tab. 3(4), dense multi-frame inputs significantly slow down inference (229.5 ms, 3.15× slower than the baseline) and even degrade performance, suggesting that redundant pixel information dilutes task-relevant temporal cues and may cause overfitting to visually irrelevant details. To address this, HiF-VLA replaces dense RGB inputs with compact motion representations, allowing the model to focus on dynamic and task-relevant cues, thereby improving both efficiency and accuracy. Furthermore, unified integration of hindsight and foresight further enhances overall performance. These results highlight HiF-VLA’s clear advantages in reducing redundancy and improving inference efficiency, effectively addressing RQ2.

inference runs on an NVIDIA A100 GPU and report the average latency (defined as the time required to generate one action chunk) in the LIBERO-Long simulation.

Results Analysis. As shown in Fig. 3b, the latency of multi-frame baselines increases almost linearly with the length of history, reflecting their high computational sensitivity to frame stacking. For instance, at a history length of 8, the baseline incurs over 4.5× higher latency compared to the vanilla VLA. In contrast, HiF-VLA maintains a consistently low computational overhead across all context lengths, with latency increasing only marginally as history grows, demonstrating superior scalability with respect to temporal context length, thereby directly addresses RQ3.

###### 4.4. Ablation Studies

Experimental Setup. To address RQ4 regarding the optimal integration of historical information, the effects of hindsight length and historical embedding positions are investigated on the LIBERO-Long.

###### 4.3. Inference Scalability

Experimental Setup. We evaluate the inference efficiency of HiF-VLA against two multi-frame baselines [20]: (i) multi-frame extension with stacked past observations, and (ii) multi-frame history combined with single-frame subgoal prediction. For each model variant, we conduct 100

Hindsight Length. We examine the impact of hindsight length in both third-view and multi-view settings. As illustrated in Fig. 3c, the model achieves peak performance 94.4% and 96.4% when the hindsight length is set to 8.

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Place blocks on the plates. Cover block and stack bowls.

Press buttons in order.

AgileX PiPER Robotic Arm

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

Realsense D435 Scene Camera

USB Wrist Camera





 

 



 

####  

[Figure 380]

[Figure 381]

[Figure 382]

62.5

33.3

17.4

[Figure 383]

[Figure 384]

[Figure 385]

65

57.9

34.2

61 62 63 64 65 66

0 20 40 60 80

0 5 10 15 20 25 30 35 40

(a) Robotic Arm Platform (b) Performance of OpenVLA-OFT vs. HiF-VLA

Figure 5. Real-world long-horizon tasks. (a) We deploy our system on the AgileX Piper robotic arm equipped with an external scene camera (Intel RealSense D435) and a wrist-mounted camera. (b) We design three long-horizon tasks covering diverse primitives such as pick, put, cover, stack, and press, emphasizing temporal consistency in action generation.

Pick up the white block. Put it on the white plate.

 

Pick up the pink block.

Pick up the green bowl. Cover the white block. Pick up the pink bowl. Stack it on the green bowl.

Put it on the pink plate.

Press the yellow button. Press the blue button.

ciate time states. As shown in Fig. 5(b), the baseline model (OpenVLA-OFT) performs poorly across these longhorizon tasks — for example, achieving only 17.4% success on Press-Buttons-Order, often failing to complete required button presses. This is likely due to the minimal visual difference between pressed and unpressed states, making it difficult for the baseline to detect successful actuation. In contrast, HiF-VLA benefits from its broad temporal receptive field, enabling reliable detection of subtle state transitions and robust execution of long-horizon tasks, resulting in superior performance across real-world tasks.

We believe, in the LIBERO-Long benchmark, most manipulation sequences exhibit moderate temporal dependencies, where 8 hindsight lengths are sufficient to capture meaningful causal cues without introducing redundant information. Hindsight Embedding Position. We study how the embedding strategy of hindsight information affects model performance. Specifically, we compare two variants: (i) na¨ıvely concatenating hindsight with language and vision as direct inputs to the VLM, and (ii) conditioning hindsight within the expert module, which is adopted in our implementation. As shown in Fig. 4, the expert-conditioned embedding consistently achieves higher success rates than VLM-based embeddings. We attribute this to the fact that motion-based hindsight may interfere with the pretrained alignment between visual and language representations. In contrast, injecting hindsight at the decoding stage provides a more direct, residual-like path for motion information. This allows the low-level dynamics encoded in the hindsight to guide future action prediction without passing through, and potentially being corrupted by the VLM’s semantic fusion layers.

Press the purple button.

##### 5. Conclusion

This work introduces HiF-VLA, an efficient and unified framework for temporal perception and reasoning built upon low-dimensional, structured motion vectors. By integrating hindsight, insight, and foresight cues, HiFVLA establishes a bidirectional temporal expansion over a sparse visual receptive field, enabling the robot to capture task-critical dynamics at minimal computational cost and thereby improving temporal consistency and causal coherence in long-horizon tasks. Across both simulated and realworld long-horizon manipulation benchmarks, HiF-VLA demonstrates strong performance. Limitations. The current motion representation remains dependent on estimation accuracy and may be sensitive to noise in highly dynamic scenes. We leave the exploration of large-scale pretraining on internet videos to enhance motion understanding and generation capabilities for future work.

###### 4.5. Real-world Experiments

Experiment setups. To evaluate the effectiveness of our approach in real-world applications, we conduct real-world experiments using the AgileX Piper robot. As shown in Fig. 5(a), a RealSense D435 camera captures the scene from a third-view, while an additional USB camera is mounted on the robot’s wrist for egocentric observations. We collect three long-horizon tasks, each with 100 demonstrations involving diverse manipulation primitives, including pick, place, stack, cover, and press.

##### Acknowledgements

Real-world evaluation. For real-world environments, we train each model separately for every task and evaluate performance by averaging success rates over 20 trials per task. These long-horizon tasks require the model to maintain action consistency across stages, correctly asso-

This work was supported by the National Science and Technology Major Project (Grant No. 2022ZD0208800) and the National Natural Science Foundation of China General Program (Grant No. 62573362).

##### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [2] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024. 1
- [3] Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2
- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 5, 6, 2
- [5] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Rich Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pre-trained image-editing diffusion models. In Proceedings of the International Conference on Learning Representations,

2024. 6

- [6] Qingwen Bu, Jia Zeng, Li Chen, Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma, and Hongyang Li. Closed-loop visuomotor control with generative expectation for robotic manipulation. In Proceedings of the Advances in Neural Information Processing Systems, pages 139002–139029, 2024. 6
- [7] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. UniVLA: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025. 6, 2
- [8] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. GR-2: A generative video-languageaction model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3, 6, 1
- [9] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion Forcing: Next-token prediction meets full-sequence diffusion. Proceedings of the Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [10] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. On scaling up a multilingual vision and language model. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14432–14444, 2023. 1
- [11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition

- at scale. In Proceedings of the International Conference on Learning Representations, 2021. 3, 1
- [12] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 3
- [13] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. In Proceedings of the International Conference on Machine Learning, 2025. 5, 6, 1
- [14] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, YuChiang Frank Wang, and Fu-En Yang. ThinkAct: Visionlanguage-action reasoning via reinforced visual latent planning. In Proceedings of the Advances in Neural Information Processing Systems, 2025. 2
- [15] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail,

Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 1

- [16] Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, Fan Wang, Deli Zhao, and Xin Li. RynnVLA-001: Using human demonstrations to improve robot manipulation. arXiv preprint arXiv:2509.15212, 2025. 1
- [17] Yang Jin, Zhicheng Sun, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, et al. Video-LaVIT: Unified video-language pretraining with decoupled visual-motional tokenization. In Proceedings of the International Conference on Machine Learning, 2024. 2, 3
- [18] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visuallyconditioned language models. In Proceedings of the International Conference on Machine Learning, 2024. 1, 5
- [19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1, 2, 5, 6
- [20] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 1, 2, 3, 5, 6, 7
- [21] Didier Le Gall. MPEG: A video compression standard for multimedia applications. Communications of the ACM, 34

(4):46–58, 1991. 3

- [22] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial Forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276, 2025. 3, 2
- [23] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu,

- Yizhong Zhang, et al. CogACT: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024. 2
- [24] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025. 6
- [25] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Proceedings of the Advances in Neural Information Processing Systems, pages 44776–44791, 2023. 5, 2
- [26] Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2025. 1, 2, 3, 6
- [27] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1B: a diffusion foundation model for bimanual manipulation. In Proceedings of the International Conference on Learning Representations, 2025. 2
- [28] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. CALVIN: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3): 7327–7334, 2022. 5
- [29] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 5, 1
- [30] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open X-Embodiment: Robotic learning datasets and RT-X models : Open X-Embodiment collaboration. In 2024 IEEE International Conference on Robotics and Automation, pages 6892–

6903. IEEE, 2024. 5

- [31] Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025. 6, 2
- [32] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 3
- [33] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 1

- [34] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1

- [35] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 1, 3, 2
- [36] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Proceedings of the European Conference on Computer Vision, pages 402–419. Springer,

2020. 3

- [37] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. In Proceedings of the International Conference on Learning Representations, 2025. 2, 3, 5, 6
- [38] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. VLA-Adapter: An effective paradigm for tiny-scale vision-language-action model. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025. 3
- [39] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025. 3
- [40] Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. DexVLA: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025. 2
- [41] Youpeng Wen, Junfan Lin, Yi Zhu, Jianhua Han, Hang Xu, Shen Zhao, and Xiaodan Liang. VidMan: Exploiting implicit dynamics from video diffusion model for effective robot manipulation. Proceedings of the Advances in Neural Information Processing Systems, 37:41051–41075, 2024. 6, 1
- [42] Thomas Wiegand, Gary J Sullivan, Gisle Bjontegaard, and Ajay Luthra. Overview of the H. 264/AVC video coding standard. IEEE Transactions On Circuits and Systems For Video Technology, 13(7):560–576, 2003. 3
- [43] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pretraining for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023. 1
- [44] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In Proceedings of the International Conference on Learning Representations, 2024. 6
- [45] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. Understanding and improving layer normalization. Advances in neural information processing systems, 32,

2019. 1

- [46] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang,

- Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi ”Jim” Fan, and Joel Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 2
- [47] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 5, 1
- [48] Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, et al. 4d-vla: Spatiotemporal visionlanguage-action pretraining with cross-scene calibration. In Proceedings of the Advances in Neural Information Processing Systems. 2
- [49] Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen. UP-VLA: A unified understanding and prediction model for embodied agent. arXiv preprint arXiv:2501.18867, 2025. 2, 3, 4, 6
- [50] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. DreamVLA: a visionlanguage-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025. 2
- [51] Zongzheng Zhang, Haobo Xu, Zhuo Yang, Chenghao Yue, Zehao Lin, Huan-ang Gao, Ziwei Wang, and Hao Zhao. Tavla: Elucidating the design space of torque-aware visionlanguage-action models. arXiv preprint arXiv:2509.07962,

2025. 5

- [52] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Tsung-Yi Lin, Gordon Wetzstein, Ming-Yu Liu, and Donglai Xiang. CoT-VLA: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1702–1713,

2025. 2, 3, 4, 6

- [53] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3DVLA: A 3D vision-language-action generative world model. In Proceedings of the International Conference on Machine Learning, 2024. 1
- [54] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In Proceedings of the International Conference on Learning Representations, 2025. 1, 3, 2
- [55] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, and Haoang Li. FlowVLA: Thinking in motion with a visual chain of thought. arXiv preprint arXiv:2508.18269, 2025. 2
- [56] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the Conference on Robot Learning, pages 2165–2183, 2023. 1, 2

## HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models

### Supplementary Material

Motions Actions

Spatial-Temporal

MLP

Attention

##### 6. More Implementation Details

Motions Actions

| | | |
|---|---|---|
| | | |

Beyond the SigLIP [47] and DINOv2 [29] image encoders and the Prismatic VLM [18] backbone described in the main text, we provide further additional implementation details for the two core modules used in HiF-VLA: the Hindsight Encoder and the Hindsight-Modulated Joint Expert. These details complement the high-level description given in the main text.

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

MLP MLP

| | | |
|---|---|---|
| | | |

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

Modulation Modulation

MLP MLP

[Figure 404]

[Figure 405]

Self-Attention

Modulation Modulation

[Figure 410]

Joint Attention

Q K V

Self-Attention

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

q q k k v v

Joint Attention

Q K V

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

q q k k v v

Linear Linear

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 450]

[Figure 451]

Modulation Modulation

Linear Linear

[Figure 452]

[Figure 453]

RoPE RoPE

Modulation Modulation

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

RoPE RoPE

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

q k v q k v

[Figure 494]

q k v q k v

###### 6.1. Hindsight Encoder

AdaLN AdaLN

AdaLN AdaLN

[Figure 496]

[Figure 497]

Hindsight-Modulated Joint

We employ a 4-layer Vision Transformer (ViT) [11] as the hindsight motion encoder. The hindsight motion sequence of length h is first partitioned into (h//2) × (H//2) × (W//2) spatiotemporal blocks using a 3D convolution, which simultaneously reduces temporal redundancy and preserves local motion continuity. These blocks, together with an additional [CLS] token that aggregates global temporal context, are fed into the Transformer. The resulting historical features are subsequently projected via a linear layer into the joint expert embedding space, ensuring dimensional compatibility with the foresight and action pathways. This design allows the hindsight encoder to provide structured historical priors that can effectively regularize downstream decision-making.

Hindsight Encoder

Expert

Hindsight

History-Modulated Joint Expert

Encoder

Figure 6. Architecture of the hindsight-modulated joint expert.

##### 7. Comparison with Video-Generation VLAs

Compared to VLA approaches that rely on video generation [8, 13, 41, 43], our method differs fundamentally in how it models temporal dynamics. A large body of recent work [8, 13, 41, 43] employs general-purpose video generative models to predict future frames, using these predictions either for inverse dynamics computation or as conditional representations to assist action policy generation. While these approaches have made substantial progress, they face some limitations when contrasted with HiF-VLA, which leverages sparse motion representations.

###### 6.2. Hindsight-Modulated Joint Expert

The hindsight-modulated joint expert serves as the fusion module that integrates hindsight, foresight, and action representations. All tokens are projected into a shared embedding dimension of 1024, including: hindsight motion tokens from the hindsight encoder, foresight motion tokens and latent action tokens produced by the VLM backbone. Within this space, the hindsight tokens serve as adaptive conditioning signals that modulate both foresight and action streams via AdaLN [45] scaling and shifting. This conditioning mechanism allows the model to dynamically adjust its predictions based on temporally grounded historical cues, enabling causal alignment between past observations and future behavior. Positional information is provided through Rotary Positional Embedding (RoPE) [33], allowing efficient encoding of both spatial and temporal ordering. The joint expert consists of 6 Transformer layers, each capable of cross-stream attention, enabling the model to capture complementary relationships between predicted dynamics and actions.

Specifically, video-generation-based methods typically require multi-frame historical input and high-resolution future video synthesis, resulting in substantial computational overhead. Moreover, pixel-level prediction is prone to local artifacts and distortions, introducing uncontrollable noise. In contrast, HiF-VLA replaces video-level temporal modeling with low-dimensional, structured motion representations, enabling a more efficient and stable way to encode historical visual states and predict future dynamics. In addition, HiF-VLA, by jointly predicting foresight motion and action trajectories as two complementary information streams, allows the model to anticipate how the physical world may evolve while generating the corresponding actions—effectively enabling thinking while acting.

Methods LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Average TraceVLA [54] 84.6 85.2 75.1 54.1 74.8

Octo [35] 78.9 85.7 84.6 51.1 75.1 CoT-VLA [52] 81.1 87.5 91.6 87.6 69.0 SpatialVLA [22] 88.2 89.9 78.6 55.5 78.1 ThinkAct [14] 88.3 91.4 87.1 70.9 84.4

Seer [37] - - - 87.7 87.7 FlowVLA [55] 93.2 95.0 91.6 72.6 88.1

4D-VLA [48] 88.9 95.2 90.9 79.1 88.6 DreamVLA [50] 97.5 94.0 89.5 89.5 92.6

CogACT [23] 97.2 08.0 90.2 88.8 93.2

π0 [4] 96.8 98.8 95.8 85.2 94.2 GR00T N1 [3] 94.4 97.6 93.0 90.6 93.9

UniVLA [7] 96.5 96.8 95.6 92.0 95.2 MemoryVLA [31] 98.4 98.4 96.4 93.4 96.5 OpenVLA-OFT [20] 97.6 98.4 97.9 94.5 97.1 HiF-VLA(ours) 98.8 99.4 97.4 96.4 98.0

- Table 4. Performance on the LIBERO benchmark. The table compares our method against a wide range of state-of-the-art approaches, with the best performance highlighted in bold.

Ablation Parameters SR

- (a)

Foresight Motion Loss Weight λ

0.1 94.4

0.05 95.2 0.01* 96.4 0.001 95.6

- (b) Joint Expert Depth

2 95.2 4 95.6

6* 96.4 8 95.2

- (c)

Length: (Hindsight, Foresight)

(8,8)* 96.4 (8,16) 94.6

(16,16) 95.2

- Table 5. Ablation on hyper-parameters. * denotes the submission setting. (multi-view)

ness and general applicability across diverse task scenarios.

###### 8.2. More Hyper-parameters Ablation

The Impact of Weight Factor λ. The hyperparameter λ balances the contributions between foresight motion prediction and action prediction. To systematically investigate this trade-off, we evaluate a range of λ values. Our analysis reveals that an appropriate weighting allows the motion prediction to effectively support the model’s reasoning, thereby enhancing planning capabilities; however, an unsuitable value destabilizes the VLA architecture and impedes policy generation. Our results in Tab. 5(a) identify an optimal performance point at λ = 0.01. This key finding indicates that a modest, well-calibrated contribution from the motion-prediction branch is essential for achieving balanced and robust model behavior.

Depth of the Joint Expert. We investigated the optimal interaction depth within the Joint Expert by performing an ablation study over depths of {2,4,6,8} (see Tab. 5(b)). We observed that a depth of 6 yields the best performance. Shallower networks facilitate the interaction between motion and action modalities, whereas increasing the depth further yields negligible performance gains, indicating depth is not highly critical beyond moderate depth.

##### 8. More Experimental Results

###### 8.1. Comprehensive Evaluation on the LIBERO Benchmark

We report detailed evaluation results on all four suites of the LIBERO benchmark [25] and compare our method against a broad set of baseline models, as summarized in Tab. 4. While achieving its greatest margin of superiority under the most challenging LIBERO-Long suite, HiF-VLA also delivers competitive or superior performance across the remaining three suites when compared to prior state-of-the-art approaches. As a result, HiF-VLA attains the best average performance over all four suites, demonstrating its robust-

Foresight and Foresight Horizon. In Tab. 5 bottom, increasing foresight to n = 16 hurts performance due to error accumulation in long-term prediction. In contrast, extending hindsight to 16 (keeping foresight at 16) improves results by providing richer context, supporting our choice to decouple hindsight and foresight lengths.

We further investigate whether the ordering of motion and action tokens affects performance. Since bidirectional attention provides a global receptive field within each segment, the token order should not influence the learned representation. Consistent with this expectation, swapping the token order (M|A → A|M) results in less than 0.5% performance change.

###### Ablation Variant SR

- (a)

Motion or State as (Hindsight+Foresight)

S+M 92.6 S+S 92.0

M+M* 94.4

- (b) Causal or Bidirectional

Causal-[M|A] 87.4 Bi-[A|M] 94.0 Bi-[M|A]* 94.4

- (c) Motion Representation

Different Motion Representations. We further study the impact of different motion representations by replacing motion vectors (MVs) with optical flow, as reported in Tab. 6(c). Optical flow provides dense motion estimation and serves as a commonly used alternative motion cue. Specifically, optical flow is estimated using RAFT [36]. The results show that both representations achieve nearly identical success rates, indicating that the HiF architecture is robust to the specific choice of motion representation. This suggests that the performance gains primarily stem from the proposed modeling framework rather than a particular motion signal. However, the two representations differ significantly in computational cost. Computing optical flow introduces substantial preprocessing overhead. With a 4-frame history, flow-based inference requires 186.8 ms, whereas our MV-based approach takes only 121.6 ms. The efficiency gap becomes larger when the history length increases, resulting in up to a 78% reduction in latency overhead with MVs for 8-frame histories. These results indicate that motion vectors provide a more efficient motion representation while maintaining comparable performance.

Flow 94.2 MVs 94.4

- Table 6. Ablation on different variants. M: Motion, S: State, A: Action. * denotes the submission setting. (third view)

###### 8.3. More Variants Ablation

State replaces motion. To better understand the role of motion representations, we conduct an ablation study replacing codec motion vectors (M) with robot proprioceptive states (S). This experiment is motivated by the observation that motion vectors might appear to primarily reflect robot arm movement. If this were the case, robot states—which explicitly encode the robot configuration—should provide similar information and lead to comparable performance. However, motion vectors describe inter-frame visual changes and therefore capture not only robot motion but also the visual consequences of interactions with the environment, such as object displacement, contact-induced motion, or changes in scene structure. These effects are not fully observable from proprioceptive signals alone. As shown in Tab. 6(a), replacing motion vectors with robot states (M+M → S+S) leads to a clear performance drop. This result indicates that motion vectors provide complementary dynamic information beyond robot states and play an important role in capturing interactiondriven visual dynamics.

[Figure 500]

0.06

|[Figure 501]<br><br>w/o action prediction w/ action prediction<br><br>|
|---|

[Figure 502]

0.05

0.04

L1Loss

Bidirectional Interaction vs. Causal Separation. To evaluate the benefit of joint foresight–action modeling, we compare our design with a decoupled variant that removes the bidirectional interaction between motion and action tokens. In this variant, motion and action tokens are processed using causal attention and decoded with separate prediction heads. This modification preserves parallel computation but prevents motion and action representations from interacting during inference. In contrast, our Joint Expert employs bidirectional attention to model motion and action synchronously, allowing the two representations to influence each other during the forward pass. As shown in Tab. 6(b), the decoupled causal variant achieves a success rate of 87.4%, whereas our joint formulation improves performance to 94.4%. This result suggests that enabling bidirectional interaction between foresight motion and action representations is important for effective decision making.

0.03

0.02

0.01

0 5k 10k 15k 20k

Training Step

Figure 7. Convergence of the foresight-motion L1 loss during training. The curve “w/o action prediction” corresponds to a variant where the action-prediction branch is removed and only the foresight-motion pathway is trained. The curve “w/ action prediction” represents the full HiF-VLA architecture, where both streams in the joint expert (foresight motion and action) are retained.

[Figure 503]

[Figure 504]

###### 8.4. Analysis of “Think-while-Acting” Paradigm

Foresight Motion Prediction. To more comprehensively evaluate the interaction between foresight motion and action prediction, we additionally removed the actionprediction branch and trained the model using only the motion prediction pathway. We then compared the evolution of the motion L1 loss during training, as shown in Fig. 7. The results indicate that incorporating action prediction leads to faster convergence and a noticeably more stable training curve for motion prediction. This indicates that the action branch provides effective complementary information to the motion branch, enabling a synergistic interaction between the two. Such coupling supports the model’s ability to reason about future dynamics while simultaneously making action decisions—thereby achieving a genuine “thinkwhile-acting” capability.

Foresight and Action Prediction Visualization. We present qualitative visualizations of the foresight motion predictions and action predictions across several tasks in the LIBERO-Long dataset, as shown in Fig. 8. The visualizations demonstrate how our model’s anticipated motion sequences closely align with its generated action plans. Across diverse long-horizon tasks, this close alignment ensures that the agent’s actions are consistently grounded in a coherent forecast of future states. This visually corroborates that HiF-VLA maintains temporal coherence and successfully executes the proposed “think-while-acting” paradigm by dynamically adapting its policy based on foresighted reasoning.

##### 9. Real-World Experiments

###### 9.1. Real-World Experimental Setup

We evaluate our method on a series of long-horizon realworld tasks using an AgileX Piper robot, which is equipped with a 6-DoF manipulator and a 1-DoF gripper. A single Intel RealSense D435 camera provides third-person observations, while an additional USB wrist-mounted camera provides egocentric input. Data are collected at 20 Hz. All models are trained under the standard HiF-VLA configuration and deployed on a single NVIDIA RTX 4090 GPU.

###### 9.2. Task Descriptions

Place blocks on the plates. The robot must place the white block onto the white tray and the pink block onto the pink tray. This task primarily evaluates the model’s basic visual recognition ability and precise object placement. Its clear goal structure and low action complexity make it suitable for assessing whether the model can reliably establish accurate observation–action mappings in simple environments.

Cover block and stack bowls. The robot first covers a white block with a green bowl and then stacks a pink bowl on top of the green one. This task stresses the model’s abil-

ity in multi-step sequential reasoning, spatial relation understanding, and long-horizon dependency modeling. The explicit hierarchical dependency structure (cover −→ stack) makes it an effective test for the model’s capacity to handle layered object interactions and long-term constraints.

Press buttons in order. The robot must press three colored buttons in a specified order. This task assesses the model’s ability to distinguish visually similar states, maintain correct action ordering, and perform time-sensitive decision making. Because pre-press and post-press observations can look visually similar, the task naturally introduces visual ambiguity and demands strong historical information integration and temporal reasoning.

###### 9.3. Real-World Execution Visualization

Fig. 9 provides visualizations of HiF-VLA’s rollout across the real-world tasks. The figure presents the model’s generated actions at key time steps. As shown, HiF-VLA demonstrates stable behavior planning capabilities in real operation: in the block-grasping task, the robot consistently maintains reliable object recognition and performs precise grasp-and-place actions; in the multi-step bowl covering and stacking task, the system robustly executes cross-object sequential operations and preserves coherent action structure throughout stages with clear hierarchical dependencies; in the button-pressing task, the model correctly distinguishes between visually similar states and adheres to the required action order. These visualizations illustrate that HiF-VLA exhibits robust visual understanding, coherent action organization, and strong execution of long-horizon task structures in real scenarios, thereby validating its reliability in performing complex manipulation tasks.

###### 9.4. Failure Cases

Despite HiF-VLA outperforming in both simulation benchmarks and real-world experiments, several failure cases still occur, as shown in Fig. 10. In the first task, the model prematurely opened the gripper based on an incorrect spatial judgment, resulting in a placement failure. In the second task, the stacking failed because the robotic arm did not lift the bowl to an appropriate height. In the third task, an erroneous depth estimation caused the gripper to descend insufficiently, leading to an incomplete button press.

These failure modes highlight the critical role of spatial geometry and 3D perception. They suggest that future work may benefit from integrating richer 3D representations into our framework to further enhance robustness in real-world manipulation.

Task: put both the alphabet soup and the cream cheese box in the basket.

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

- (a)
- (b)
- (c)

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

Task: put the white mug on the plate and put the chocolate pudding to the right of the plate.

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

Task: put both moka pots on the stove.

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

Figure 8. Example rollouts of three tasks in LIBERO-Long, illustrating the close alignment between the predicted foresight motion and the observed action execution over a short inference window.

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

- (a) Place blocks on the plates.

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

- (b) Cover block and stack bowls.

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

(c) Press buttons in order.

- Figure 9. Example rollouts of real-world tasks.

[Figure 619]

|[Figure 620]<br><br>Failed to place!<br><br>| | | |[Figure 621]<br><br>Failed to press!| |
|---|---|---|---|---|---|
|[Figure 622]| |[Figure 623]<br><br>|[Figure 624]<br><br>[Figure 625]<br><br>| |[Figure 626]<br><br>[Figure 627]|

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

Failed to press!

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

Failed to stack!

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

- (a) Place blocks on the plates.

[Figure 657]

|[Figure 658]|
|---|

|[Figure 659]<br><br>|
|---|

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

Failed to place!

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

Failed to stack!

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

- (b) Cover block and stack bowls.

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

Failed to place!

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

Failed to press!

[Figure 705]

|[Figure 706]<br><br>[Figure 707]|
|---|

|[Figure 708]<br><br>[Figure 709]|
|---|

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

Failed to stack!

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

(c) Press buttons in order.

- Figure 10. Failure cases of real-world tasks.

