[Figure 1]

## MIND: Benchmarking Memory Consistency and Action Control in World Models

Yixuan Ye1∗, Xuanyu Lu1∗, Yuxin Jiang2∗, Yuchao Gu2, Rui Zhao2, Qiwei Liang3, Jiachun Pan2, Fengda Zhang4, Weijia Wu2†, Alex Jinpeng Wang 1† 1 CSU-JPG, Central South University 2 National University of Singapore 3 Hong Kong University of Science and Technology (Guangzhou) 4 Nanyang Technological University Project Page: https://csu-jpg.github.io/MIND.github.io/

# arXiv:2602.08025v2[cs.CV]11Feb2026

[Figure 2]

[Figure 3]

1-st State 𝑛-th State

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

First-Person Third-Person First-Person Third-Person

1-st State (Memory, Prior Context) 𝑛-th State (Prediction) Would the scene you see later still match what you saw before?

Figure 1. Evaluation for Memory Consistency and Action Control with MIND. The first open-domain closed-loop revisited benchmark at 1080p/24 FPS for evaluating world models from both first-person and third-person perspectives.

### Abstract

different action spaces under shared scenes. To facilitate future performance benchmarking on MIND, we introduce MIND-World, a novel interactive Video-toWorld baseline. Extensive experiments demonstrate the completeness of MIND and reveal key challenges in current world models, including the difficulty of maintaining long-term memory consistency and generalizing across action spaces.

World models aim to understand, remember, and predict dynamic visual environments, yet a unified benchmark for evaluating their fundamental abilities remains lacking. To address this gap, we introduce MIND, the first open-domain closed-loop revisited benchmark for evaluating Memory consIstency and action coNtrol in worlD models. MIND contains 250 high-quality videos at 1080p and 24 FPS, including 100 (first-person) + 100 (third-person) video clips under a shared action space and 25 + 25 clips across varied action spaces covering eight diverse scenes. We design an efficient evaluation framework to measure two core abilities: memory consistency and action control, capturing temporal stability and contextual coherence across viewpoints. Furthermore, we design various action spaces, including different character movement speeds and camera rotation angles, to evaluate the action generalization capability across

### 1. Introduction

Recent advances in video generation technology have significantly improved the creation of high-fidelity, realistic content, laying a solid foundation for developing sophisticated world models [8, 13, 28, 44, 47]. These models have accelerated advancements across diverse domains, including autonomous driving [21, 25, 29, 43, 49], embodied intelligence [2, 3, 7, 27, 35], and interactive game environments [4, 23, 38, 42, 44], by enabling the generation of complex, diverse, and controllable virtual worlds. Despite these advances,

∗ Equal contribution. † Corresponding author.

Table 1. Comparison of World Model Benchmarks. ‘Avg.’ denotes the average number of frames used for memory context and predicted segment in each benchmark. ‘1st-P.’ and ‘3rd-P.’ refer to first person and third person perspectives, respectively. ∅ denotes the benchmarks without action-based generation (e.g., text–video gen). ‘CharPos.’ refer to the character position. MIND is the first open-domain closed-loop revisited benchmark for evaluating video consistency across both first- and thirdperson perspectives.

|Benchmark|CharPos.<br><br>|Fixed Act. (Avg.)|Generalized Act. (Avg.)<br><br>|Res./FPS Scenario (image /video )|
|---|---|---|---|---|
| | |1st-P. 3rd-P.|1st-P. 3rd-P.| |
|WorldSimBench [31] WorldModelBench [22] WorldScore [9] World-in-World [48] GameWorld [50] Lian et al. [26]|✗ ✗ ✗ ✗ ✗ ✓<br><br>|1 / - ∅ 1 / - ∅ 1 / - ∅<br><br>∅ ∅ 1 / - ∅<br><br>65 / 436 ∅<br><br>|∅ 1 / ∅ ∅ ∅ ∅ ∅ ∅ ∅ ∅ ∅ ∅|-/- : Minecraft, Driving...<br><br>-/- : Humans, Natural...<br><br>-/- : Dining , Passageways...<br><br><br>576p/- : Interior environment...<br><br>720p/- : Minecraft<br><br>360p/20 , : Minecraft|
|MIND (Ours)<br><br>|✓|1.1k/3.4k 1.2k/3.6k<br><br>|1.3k/3.8k 1.2k/3.7k|1080p/24 , : Landscape, SciFi, Stylized, Ancient, Urban, Industrial, Interior, Aquatic|

building a reliable world model remains challenging. Beyond visual realism, such models must maintain long-term memory consistency and exhibit accurate action control and robust action generalization across diverse scenarios. Yet, current evaluations mainly focus on visual quality or physical realism, overlooking these essential aspects. Consequently, the field still lacks a comprehensive benchmark to systematically assess memory consistency and action controllability in open-domain environments.

Existing benchmarks primarily focus on evaluating the quality and realism of generated videos, often limited to first-person perspective data collected within a single action space. For instance, WorldScore [9] decomposes scene generation into specific camera motion trajectories to assess video quality, while WorldModelBench [22] evaluates adherence to physical laws to measure world modeling capabilities in application-driven domains. Although Lian et al. [26] introduced a world model memory benchmark, it is limited to Minecraft scenes, lacks open-domain diversity, and depends on loop-based agent data that poorly reflects human behavior. Furthermore, the existing world model benchmark predominantly features first-person perspectives [9, 26], making it challenging to evaluate the ability of world models to simulate motion and poses. In summary, as shown in Table 1, existing benchmarks focus mainly on firstperson settings and image-level evaluation, lacking memory consistency assessment and scene diversity. Establishing a comprehensive world model benchmark remains an open and unresolved challenge.

We present MIND, the first closed-loop revisited open-domain benchmark for evaluating memory consistency and action control from both first-person and third-person perspectives across diverse scenarios. MIND focuses on two key abilities of world models: 1) Memory consistency refers to the ability of model to maintain coherent spatial layouts, object identities, and scene attributes over long temporal contexts, ensuring that generated frames remain consistent with past observations. 2) Action control measures how accurately the model executes given control

inputs and generalizes these dynamics to new motion ranges or unseen action spaces, reflecting its capacity for precise and adaptable interaction within dynamic environments. Furthermore, the provided videos include frame-level aligned actions, character and camera positions, and image labels, collected from multiple volunteers to capture diverse human behaviors. The dataset contains 250 high-quality 1080p / 24 FPS, frame-level action-aligned videos spanning eight major scene categories, enabling comprehensive evaluation of world models.

To summarize, the contributions of this paper are:

- • Open-Domain Benchmark for World Models. We introduce MIND, the first closed-loop revisited open-domain benchmark at 1080p / 24 FPS for evaluating world models from both first-person and third-person perspectives.
- • Evaluation for Memory Consistency and Action Control. We design an efficient framework to assess memory consistency and action control, capturing temporal stability and contextual coherence across viewpoints.
- • Evaluation for Cross-Action Space Generalization. We design various action spaces, including different character movement speeds and camera rotation angles, to evaluate the action space generalization capability.
- • The novel Video-to-World baseline, MINDWorld. Extensive experiments demonstrate the completeness of MIND and expose key challenges in current world models, such as limited longterm memory consistency and limited generalization across action spaces.

### 2. Related Work

#### 2.1. Video Generation

Recently, video generation models represented by SVD [1], Hunyuanvideo [20], Wan [39] and Sora 2 [30] have achieved remarkable breakthroughs. While significantly enhancing the visual realism, temporal consistency and controllability of generated videos, these models have extended their generation

[Figure 8]

[Figure 9]

[Figure 10]

Input（Action） Unreal Engine 5（3D Scene） Output（Data）

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Frame-Level Alignment Label

Actor Position

Open Domain

Camera Position

Third Person

First Person

Context Memory Predicted Evaluation

[Figure 18]

Camera

|[Figure 19]<br><br>[Figure 20]|
|---|

Recorder

Video.mp4

[Figure 21]

[Figure 22]

Control

Collector

[Figure 23]

Action.json

[Figure 24]

GPT - 40

Actor

Description.txt

[Figure 25]

[Figure 26]

[Figure 27]

Memory Consistency

Action Sequence Context Memory

[Figure 28]

World Model

###### Predicted Video

Action Controllability

[Figure 29]

Visual Quality

Ground Truth Video

Temporal Quality

- Figure 2. Overview of the MIND. We build and collect the first open-domain closed-loop revisited benchmark using Unreal Engine 5, supporting both first-person and third-person perspectives with 1080 p resolution at 24 FPS.

videos via self-supervision, thus enabling novel action learning under limited conditions. Real-time interaction is a core characteristic of this field, and relevant training paradigms lay a foundation for real-time streaming generation of diffusion-based world models. For instance, Diffusion-Forcing [5] trains diffusion models to denoise token sets with independent per-token noise levels and Self-Forcing [17] performs autoregressive inference with KV caching during training, conditioning the generation of each frame on the model’s own prior outputs. Together, these advances mark a shift from static video synthesis to interactive, temporally consistent world models.

capabilities to long-sequence and physically plausible scenarios. Benchmarks including VBench [18] and VBench-2.0 [52] have established dedicated evaluation systems for video generation models, covering key dimensions such as human fidelity, physical plausibility and commonsense consistency.

#### 2.2. World Model

Recent advances in world models have broken down the technical barriers between visual generation and embodied simulation, enabling agents or users to interact in temporally consistent virtual environments. Unlike traditional text-to-video models, world models emphasize long-term memory consistency [12, 15, 24, 40, 41, 46, 51], action-conditioned controlled generation [11, 14, 34, 44, 47] and real-time response [2, 13, 17, 33, 36], evolving into three core research directions. To ensure long-term memory consistency, mainstream existing studies adopt three strategies: pose frame retrieval, context memory compression, and explicit 3D memory representation. Specifically, CAM [46] retrieves context frames based on the field-of-view coverage of pose perspectives, Infinite-World [40] designs a hierarchical pose-free memory compression module to autonomously anchor generated content to distant historical information, and SPMem [41] achieves explicit 3D memory representation by virtue of geometrically anchored long-term spatial memory. For the optimization of action-conditioned controlled generation, GameFactory [47] proposes a multi-stage training strategy integrated with domain adapters, which decouples game style learning from action control to realize scene-generalizable action control and AdaWorld [11] embeds action information into the pre-training process and extracts implicit actions from

#### 2.3. Evaluation for World Model

The rapid rise of world models has spurred new benchmarks, yet most primarily emphasize scene quality or physical plausibility. WorldScore [9] standardizes camera-trajectory layouts to rate generated video quality. WorldModelBench [22] targets adherence to physical laws in application-driven settings. And WorldSimBench [31] assesses visual realism. However, these efforts under represent two core abilities of world models: long-context memory consistency and action-space generalization across varied controls. In contrast, MIND introduces the first opendomain closed-loop revisited benchmark at 1080p and 24FPS from both first-person and third-person views, providing unified, efficient protocols to evaluate memory consistency and action control.

### 3. MIND Benchmark 3.1. Video Source and Environment Settings

To comprehensively evaluate world models across diverse interactive contexts, we construct a large-scale

Original Position ∆ = 0.6x ∆ = 1.0x ∆ = 1.4x

###### Distribution of Scenario Counts by Category

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### Train Test

60

50

40

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

30

- 19%

18%

Train Actions W

S A

D

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

43%

7% 3%6%

2%

2%

18%

19%

Test Actions

W

[Figure 42]

S A

[Figure 43]

D

[Figure 44]

[Figure 45]

|48| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|38| | | | | | | | | | | | | | | | | | | | | | |
|24 26<br><br>19<br><br>24 26<br><br>20| | | | | | | | | | |24 25<br><br>29<br><br>21| | | | | | | | | | | |
| | | | | | |17| | | | | | | |10| | | | | |9 10| | |
| | | | | | | | | | | | | | | | | | | | | | | |

0

10

- 20

- (a) Camera Angle Increment ∆ for One Action

FirstPersonThirdPerson

- (b) Movement Increment ∆ for One Action (Walk Forward

Original Position ∆ = 150cm/s ∆ = 200cm/s ∆ = 250cm/s

Landscape Scifi Stylized Ancient UrbanIndustrial Interior Aquatic

PersonPersonThirdFirst

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Walk Forward (First Person)

)

42%

Figure 4. Action Generalization from MIND. Different generalization settings for ∆p (movement increment) and ∆r (camera angle increment) are derived from both firstperson and third-person perspectives. Each image is captured after the action has been executed for 24 frames.

Walk Forward

2%

2%

6% 7% 4%

- Figure 3. Distribution for Scene Categories and Action Space in MIND. MIND supports open-domain scenarios with diverse and well-balanced action spaces.

on the selected movement direction:

video corpus rendered within Unreal Engine 5. As shown in Figure 6, the benchmark spans 8 categories covering over 40 open-domain environments, designed to reflect a wide spectrum of visual and physical dynamics. These include natural (e.g., forest, desert, mountain, ocean), urban (e.g., downtown, residential, industrial), indoor, vehicle, sci-fi, stylized, fantasy, and abstract scenes. As shown in Figure 2 , we construct a systematic data generation pipeline and recruit multiple volunteers to perform both scripted and free-form actions within these environments. We collect 250 frame-aligned videos at 1080 p / 24 FPS. Among them, 200 videos (100 firstperson and 100 third-person, evenly split for training and testing) share the same action space, while the remaining 50 videos (25 per perspective) feature distinct action spaces, providing high-quality and controllable ground truth for evaluation.

#### 3.2. Basic Actions Modeling

In this section, we define a basic action set for modeling both agent translation and camera rotation, which are essential for evaluating action control and scene consistency in world models.

Action Space Definition. We define the action space A as follows:

A = {W,A,S,D,↑,↓,←,→}, where:

- • W,A,S,D correspond to forward, left, backward, and right movement,
- • ↑,↓,←,→ correspond to camera pitch and yaw rotations.

Translational Motion. For translational actions, the position of agent pt = [xt,yt,zt]⊤ is updated based

pt+1 = pt + ∆p · va,

where va is the direction of movement corresponding to the action (e.g., vW = [0,0,1]⊤ for forward) and ∆p is the step size.

Rotational (Camera) Motion. For camera rotation, the orientation rt = [θt,ϕt]⊤ is updated by a small angular increment ∆r:

rt+1 = rt + ∆r · ua,

where ua corresponds to the direction of camera rotation (e.g., u↑ = [0,+1]⊤ for pitch up).

#### 3.3. Action Space Generalization

In the action modeling framework, the values of ∆p (for translational motion) and ∆r (for rotational motion) are not fixed, but can be generalized to accommodate a range of action scales. The action set can be customized to represent different motion scales, as shown in Figure 4. For example, an action with a 0.7-degree rotation and 150 units of translation (∆r = 0.7◦,∆p = 150) allows for precise control. Larger movements, such as a 1.4-degree rotation with 280 units of translation (∆r = 1.4◦,∆p = 280), represent broader actions. Conversely, smaller steps like 0.4 degrees of rotation with 100 units of translation (∆r = 0.4◦,∆p = 100) enable more subtle adjustments, useful for tasks requiring high precision. This flexibility in ∆p and ∆r allows the system to adapt to varying levels of control and task requirements. Action generalization enhances the flexibility of model and realism across diverse scenarios. Thus, world models must adapt to varied action spaces. To assess this adaptability, we collect high-quality, frame-aligned videos from different action spaces within the same scene. Specifically, we configure five

[Figure 54]

[Figure 55]

[Figure 56]

Down

combinations of character movement speeds ∆p and camera rotation speeds ∆r to generate datasets with diverse action spaces. The setting includes a total of 25 first-person and 25 third-person clips, thereby comprehensively and systematically assessing the generalization capability of world models.

[Figure 57]

[Figure 58]

1 2 3 4

Down Shift Right

Up

Shift Left

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

[Figure 71]

[Figure 72]

[Figure 73]

5 6 7

Prediction

[Figure 74]

Prediction

Forward

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Memory

[Figure 81]

[Figure 82]

[Figure 83]

#### 3.4. Temporal and Memory Consistency

Forward & Turn Left

Forward & Turn Right

[Figure 84]

[Figure 85]

[Figure 86]

Turn Right

To evaluate the ability of world models to maintain memory and temporal consistency over time, we introduce a memory revisit strategy, as illustrated in Figure 2. In our setup, a human operator performs predefined actions (W,A,S,D,↑,↓,←,→) within a

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Turn Left

8 9 10

[Figure 91]

Backward & Turn Right

Backward & Turn Left

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Camera Direction

[Figure 96]

[Figure 97]

[Figure 98]

- 3D Unreal Engine 5 environment. The resulting firstperson and third-person videos are frame-aligned with action logs and used as ground-truth supervision. Memory Setup. We define a memory segment as a observed video sequence M = {f1,f2,...,fT}, where each frame ft encodes both visual appearance and scene layout. The memory provides contextual grounding that the model must retain when generating subsequent predictions. After observing M, the model receives an action sequence A = {aT+1,...,aT+k} and is required to predict the fu-

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Starting Point Pose

[Figure 106]

[Figure 107]

[Figure 108]

Backward

Figure 5. The 10 Symmetric Motion Paths. The blue line represents the original path, and the red line represents the corresponding mirrored path. Each action lasts 24 frames.

the mean squared error (MSE) between predicted and ground-truth frames: Llcm = k1 ki=1 ∥fˆT+i−fT+i∥22 where a lower Llcm indicates stronger long-term memory retention and reconstruction fidelity.

ture video frames Vˆ = {fˆT+1,...,fˆT+k}.

Consistency Objective. The model is evaluated on whether the predicted frames fˆT+i remain temporally and spatially consistent with the memorized scene. This includes:

Generated Scene Consistency. To quantify the world model’s ability to maintain consistency in generated scenes, we introduce a generated scene consistency metric based on 10 symmetric motion paths (Figure 5), each involving simple translations or rotations lasting 24 frames. The model moves forward and then retraces the same path in reverse; ideally, frames from the forward (fwd) and reverse (rev) paths should match exactly. We measure this consistency using

- • Memory Consistency: previously observed objects, layouts, and textures should remain unchanged when revisited through new actions (e.g., returning to the same location should reproduce the same scene appearance);
- • Temporal Consistency: predicted frames should exhibit smooth transitions and coherent dynamics over time, avoiding flickering or sudden structural changes.

MSE: Lgsc = k1 ki=1 ∥fˆTfwd+i − fˆTrev+i∥22, where fˆTfwd+i and fˆTrev+i denote the predicted frames from the forward and reverse trajectories, respectively. A lower Lgsc indicates stronger scene generation consistency and geometric stability.

Formally, given a revisiting trajectory Aloop that leads back to a previously seen state, the consistency error can be defined as:

Action Accuracy. The accuracy of action feedback in world models is central to their precise instruction execution and reliable completion of complex sequential tasks. To evaluate this capability fairly, we unify the predefined action sequences for all models, recover camera trajectories from generated videos via ViPE [16], eliminate scale and coordinate system discrepancies through Sim(3) Umeyama alignment [37], and then calculate translational and rotational relative pose errors. This metric quantifies the accuracy of each model in generating expected frames based on action commands, and is independent of the model’s internal velocity parameters.

Lmem = ∥fˆt − ft′∥22,

where ft′ corresponds to the ground-truth frame at the revisited scene.

#### 3.5. Evaluation

Long Context Memory Consistency. Long context memory evaluates the ability of world model to reconstruct previously observed content from contextual memory, reflecting its understanding of scene dynamics and physical laws. Given a full memory sequence M = {f1,...,fT} and an action sequence A = {aT+1,...,aT+k}, the model generates predicted frames Vˆ = {fˆT+1,...,fˆT+k}. Ideally, the predicted sequence should match the real sequence V = {fT+1,...,fT+k} obtained under the same actions. We quantify the long-context memory ability by

Action Space Generalization. World models serve as simulators for domains such as autonomous driving and robotics, where understanding spatial regularities is crucial. We evaluate action space generalization by computing the MSE between generated and ground-

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Landscap e

SciFi Stylized Ancient

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

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Interior Aquatic

Urban Industri al

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

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

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

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

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

Figure 6. Eight Scene Categories and Action Visualization in MIND. Each category covers multiple representative environments designed to evaluate action-following controllability and history consistency in world models.

[Figure 215]

truth frames under diverse action settings. Ideally, the model should learn action-space constraints from context and generate videos that follow them with zeroshot consistency.

[Figure 216]

[Figure 217]

Noise

N × ️DiT Block

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

DiT Block

DiT Block

[Figure 222]

[Figure 223]

Generated Video

Image Encoder

Visual Quality. We evaluate visual quality from two complementary perspectives: 1) Aesthetic quality. LAION [32] aesthetic prediction model is used to quantitatively evaluate the visual attractiveness of each frame. Trained on large-scale human preference data, it scores composition, color, lighting, realism, and style consistency. Higher scores indicate closer alignment with human aesthetic judgments. 2) Imaging Quality. MUSIQ [19] evaluates perceptual fidelity by detecting artifacts like overexposure, noise, compression, and blur. Trained on the SPAQ [10] dataset, it quantifies image clarity and sharpness as an objective measure of visual quality.

A person is walking through a art gallery with large paintings.

Text Encoder

Description

Self Attention

Multi-head Cross-Attention

###### FFN

Action Embedding Action Embedding

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

M L P

Timestep Embedding

Figure 7. MIND-World model framework. The parameterized action injection mechanism enables frame-level alignment with input videos and streamlines the process, forming an efficient baseline for Video-to-World training and inference. Furthermore, it allows for unlimited-length inference based on action sequences.

### 4. Experiment 4.1. MIND-World

We present MIND-World, a high-dynamics, realtime, interactive autoregressive video generation pipeline (see Figure 7). Following [13], our training pipeline has three stages: (i) a bidirectional, actionconditioned teacher model, (ii) student initialization from the teacher’s ODE trajectories [45], and (iii) selfforcing DMD distillation [17] into a few-step, framewise causal pipeline. Unlike [13, 47], which relies on heavy action blocks, we inject actions directly into the

timestep embeddings, yielding a simpler and effective conditioning mechanism. At inference time, we maintain a context cache and generate frames autoregressively conditioned on both the cached context and incoming actions, enabling continuous, low-latency streaming generation. We evaluate under two settings: 1) With context memory: a window of w frames is cached as clean world context in working memory, conditioning subsequent frame generation. 2) Without context memory: generation cold-starts from the initial image and proceeds autoregressively.

- Table 2. Model Performance for the First Person on MIND-First 50. All videos underwent identical processing and were evaluated at 720p resolution. ↓ indicates lower values are better; ↑ indicates higher values are better.

Model

Long Context Mem.↓

Generated Scene Consis.↓

Action Space Generalization↓

Aesthetic Quality ↑

Image Quality↑

Action Accuracy (RPE↓)

Trans Rot w/o Context Memory (Image-to-World)

MIND-World (Ours) 0.1091 0.0359 0.1200 0.4583 0.5655 0.0356 0.4395 Matrix-Game 2.0 [13] 0.1188 0.0306 0.1084 0.4302 0.5180 0.0265 0.6914

w Context Memory (Video-to-World) MIND-World (Ours) 0.1035 0.0309 0.1226 0.4590 0.5702 0.0384 0.5534

- Table 3. Model Performance for the Third Person on MIND-Third 50. All videos underwent identical processing and were evaluated at 720p resolution. ↓ indicates lower values are better; ↑ indicates higher values are better.

Model

Long Context Mem.↓

Generated Scene Consis.↓

Action Space Generalization↓

Aesthetic Quality ↑

Image Quality↑

Action Accuracy (RPE↓)

Trans Rot w/o Context Memory (Image-to-World)

MIND-World (Ours) 0.1066 0.0327 0.0677 0.5204 0.5672 0.0271 0.2587 Matrix-Game 2.0 [13] 0.1404 0.0372 0.0777 0.4236 0.4857 0.0622 0.9031

w Context Memory (Video-to-World) MIND-World (Ours) 0.1042 0.0316 0.0685 0.5300 0.5673 0.0321 0.3338

- 4.2. Experiment Setting

ther confirm the benefits of memory. Additionally, Matrix-game-2.0 performs poorly in thirdperson generation; human evaluation verifies that its metrics accurately reflect this limitation—the model fails to generate controllable third-person characters.

MIND-World. We initialize the foundation model with SkyReels-V2-I2V-1.3B [6] and fine-tune it with action injection for 3K steps with batch size 8. For distillation, we initialize the 4-step causal student based 1K teacher’s ODE trajectories and train for 3K steps, followed by 2K steps via DMDbased Self-Forcing. The student is strictly per-frame causal (chunk size = 1) with a local attention window of 25 frames. All experiments are conducted on

Action Accuracy. As shown in Tables 2 and 3, even when inputting context memory with the same action space as the fine-tuning phase, the world model’s action control performance still deteriorates. This indicates limitations in the current action injection mechanism, and how to design more effective action injection strategies to enhance the world model’s action control capability remains an important research problem worthy of in-depth exploration.

- 4 × NVIDIA H100 GPUs. MIND Dataset. As illustrated in Figure 6, the dataset covers 8 major categories, comprising a total of 100 first-person and 100 third-person videos in the same action space, along with 25 first-person and 25 third-person videos in different action spaces. All videos are long-term, open-domain, high-quality, and frame-aligned. As illustrated in Figure 3, during the finetune training and test split, action distribution consistency is ensured, and the distribution across scene categories is balanced as much as possible. Among them, 50 first-person and 50 third-person videos from the shared action space are used for training MIND-World, while the remaining 150 videos are reserved for the MIND evaluation.

Action Space Generalization. Tables 2 and 3 show that injecting context memory impairs world model inference, since inconsistent action spaces disrupt reasoning in models without action generalization.

Visual Quality. Tables 2 and 3 collectively show that world models with memory produce videos of superior visual quality and better alignment with human aesthetic preferences. This is due to the memory mechanism leveraging richer contextual information, ensuring high consistency in style and coherence with the given segment.

#### 4.3. Per-Dimension Evaluation

For each dimension, we compute the MIND score using the evaluation suite described in Section 3.5, with results presented in Tables 2 and 3. To advance future research, we introduce MIND-World as an open-domain video-to-world baseline with memoryaugmented world modeling capabilities.

Memory Consistency. Tables 2 and 3 show that, on the long context memory metric, models with context memory outperform those without by more than 4%. generated scene consistency results fur-

#### 4.4. Insights and Discussions

This section details the observations and insights derived from our comprehensive evaluation experiments. Open Domain. As illustrated in Challenge 1 of Figure 8, MIND-World trained on easily collected Minecraft datasets struggle to generalize to opendomain inference, whereas those trained on the highquality dataset provided by MIND exhibit significantly improved generalization. However, acquiring such data is challenging; thus, effectively leveraging readily available large-scale data to achieve open-

Ground Truth Train on Minecraft Train on MIND

Action Space Generalization Average

[Figure 232]

[Figure 233]

[Figure 234]

Context Memory 0.8 × 1.0 × 1.2 ×

||0.0912<br><br>|0.0910|0.1003<br><br>|
|---|---|---|---|
||0.1032|0.0850<br><br>|0.1028|

MIND-World

Challenge 2: Action Space Generalization.

Challenge 1: Open Domain

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

1st Frame 24th Frame 48th Frame

1st Frame 24th Frame 48th Frame

MIND-World

Matrix-Game 2.0

Challenge 3: Precise Action Control

Changes in Avg. LCM metric per Frame over 1 Second (24 Frames)

[Figure 241]

[Figure 242]

[Figure 243]

0.095

0.085

without Memory

0.075

with Memory

w/oMemory Ground Truth wMemory

0.065

Challenge 4: Long Context Memory

1 5 10 15 20

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

24th Frame 48th Frame 96th Frame

Matrix-Game 2.0 Matrix-Game 2.0 MIND-World

Challenge 5: Generated Scene Consistency

Challenge 6: Third-person Perspective

Figure 8. Summary of insights from the challenges in MIND. For each challenge, a representative example is visualized.

domain generalization remains a key open problem.

Action-Space Generalization. Tables 2 and 3 reveal that, in action space generalization, video-toworld models with memory capabilities underperform image-to-world models without memory. As shown in Challenge 2 of Figure 8, further analysis indicates that memory-enabled models outperform memoryless ones within the original action space; however, their performance drops significantly when the action space changes. This suggests that context memory tied to an action space inconsistent with training disrupts model inference. Therefore, accurately detecting the action space from context memory and achieving precise prediction remains a major challenge.

##### Precise Action Control. In the experiment of Path

- 5 in Figure 5, where the agent first moves left and then right to return to the starting position—the generated result is shown in Challenge 3 of Figure 8, Matrix-game-2.0 [13] fails to move left as expected, instead remaining stationary and ultimately stopping far to the right of the origin. In contrast, MIND-World correctly moves left but fails to return to the initial position after moving right. Repeated experiments reveal that the visual prompt (i.e., the input image or video) significantly affects action following. Therefore, separating visual prompts from action dynamics is key to achieving precise action control in world models. Long-horizon Memory Consistency. Tables 2 and 3 show that, in long-horizon rollouts, models with memory significantly outperform those without. The visualization in Challenge 4 of Figure 8 further confirms this: memory-enabled generations remain largely consistent with ground truth, while memory-less outputs

deviate substantially. Moreover, current world models can only capture short-term memory; effectively maintaining and leveraging long-context memory remains a critical open problem.

Generated Scene Consistency. By conducting a mirroring experiment on Matrix-game-2.0 [13] along Path 5 in Figure 5, the results as shown in Challenge 5 of Figure 8 reveal that when the camera revisits previously generated scenes, the content is clearly inconsistent with prior generations. Therefore, ensuring consistent generation of scenes continues to pose a significant difficulty.

Third-person Perspective. As shown in Challenge 6 of Figure 8, Matrix-game-2.0 [13] fails to control the third-person character and execute movement, causing the generated video to pass through the character and eventually lose it. In contrast, MIND-World can control the character but fails to properly handle the relationship between the foreground character and the background, resulting in the character passing directly through buildings. Therefore, accurately perceiving and modeling the interactions between characters and backgrounds remains a major challenge in world models.

### 5. Conclusion

We introduced MIND, the first open-domain closedloop revisited benchmark for evaluating memory consistency and action control in world models from both first-person and third-person perspectives. Built on Unreal Engine 5 with diverse action spaces, MIND enables systematic assessment of long-term scene memory, temporal coherence, and action space generalization. Experiments with MIND-World reveal that the

challenges remain in generalizing across action spaces and maintaining long-horizon coherence. MIND establishes a unified foundation for advancing interactive, temporally consistent open domain world model.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127,

2023. 2

- [2] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024. 1, 3
- [3] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 1
- [4] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024. 1
- [5] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37: 24081–24125, 2024. 3
- [6] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. SkyReels-V2: Infinite-length film generative model, 2025. 7
- [7] Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, and Jian Tang. Wow: Towards a world omniscient world model through embodied interaction, 2025. 1
- [8] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu,

- Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025. 1
- [9] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li FeiFei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. ICCV,

2025. 2, 3

- [10] Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang. Perceptual quality assessment of smartphone photography. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3677–3686,

2020. 6

- [11] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions,

2025. 3

- [12] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction, 2025. 3
- [13] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Cyrus Wu, Wei Li, Xuchen Song, Yang Liu, Eric Li, and Yahui Zhou. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009,

2025. 1, 3, 6, 7, 8

- [14] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick HoldGeoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025. 3
- [15] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft,

2025. 3

- [16] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 5
- [17] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 3, 6
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang,

- Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807– 21818, 2024. 3
- [19] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 6
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603,

2024. 2

- [21] Bohan Li, Zhuang Ma, Dalong Du, Baorui Peng, Zhujin Liang, Zhenqiang Liu, Chao Ma, Yueming Jin, Hao Zhao, Wenjun Zeng, and Xin Jin. Omninwm: Omniscient driving navigation world models, 2025. 1
- [22] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, and Yao Lu. Worldmodelbench: Judging video generation models as world models. 2025. 2, 3
- [23] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuangamecraft: High-dynamic interactive game video generation with hybrid history condition,

2025. 1

- [24] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory, 2025. 3
- [25] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025. 1
- [26] Kewei Lian, Shaofei Cai, Yilun Du, and Yitao Liang. Toward memory-aided world models: Benchmarking via spatial consistency. arXiv preprint arXiv:2505.22976, 2025. 2
- [27] Qi Lv, Weijie Kong, Hao Li, Jia Zeng, Zherui Qiu, Delin Qu, Haoming Song, Qizhi Chen, Xiang Deng, and Jiangmiao Pang. F1: A visionlanguage-action model bridging understanding and generation to actions. arXiv preprint arXiv:2509.06951, 2025. 1
- [28] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation

- model. arXiv preprint arXiv:2507.17744, 2025. 1
- [29] Arian Mousakhan, Sudhanshu Mittal, Silvio Galesso, Karim Farid, and Thomas Brox. Orbis: Overcoming challenges of long-horizon prediction in driving world models. arXiv preprint arXiv:2507.13162, 2025. 1
- [30] OpenAI. Sora 2 is here: our latest video generation model. https://openai.com/ index/sora-2/, 2025. Accessed: 2025-10-

29. 2

- [31] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. ICML, 2025. 2, 3
- [32] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Mehdi Cherti, Clayton Mullis, Andreas K¨opf, Theo Coombes, and Jenia Jitsev. Laion-aesthetic predictor. https:// github.com/LAION- AI/aestheticpredictor, 2022. LAION-AI, GitHub repository. 6
- [33] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards longterm geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025. 3
- [34] Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, and Qinglin Lu. Hunyuan-gamecraft-2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025. 3
- [35] PAN Team, Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, Davit Abrahamyan, Arif Ahmad, Ganesh Bannur, Junrong Chen, Kimi Chen, Mingkai Deng, Ruobing Han, Xinqi Huang, Haoqiang Kang, Zheqi Li, Enze Ma, Hector Ren, Yashowardhan Shinde, Rohan Shingre, Ramsundar Tanikella, Kaiming Tao, Dequan Yang, Xinle Yu, Cong Zeng, Binglin Zhou, Zhengzhong Liu, Zhiting Hu, and Eric P. Xing. Pan: A world model for general, interactable, and long-horizon world simulation, 2025. 1
- [36] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026. 3
- [37] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on pattern analysis

- and machine intelligence, 13(4):376–380, 2002. 5
- [38] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837,

2024. 1

- [39] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [40] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, and Ming-Ming Cheng. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory, 2026. 3
- [41] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory, 2025. 3
- [42] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory, 2025. 1
- [43] Zhenjie Yang, Xiaosong Jia, Qifeng Li, Xue Yang, Maoqing Yao, and Junchi Yan. Raw2drive: Reinforcement learning with aligned world models for end-to-end autonomous driving (in carla v2). arXiv preprint arXiv:2505.16394, 2025. 1
- [44] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, et al. Yan: Foundational interactive video generation. arXiv preprint arXiv:2508.08601, 2025. 1, 3
- [45] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 6
- [46] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval, 2025. 3
- [47] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025. 1, 3, 6
- [48] Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal M Patel, Paul Pu Liang, et al. World-in-world: World models in a closed-loop world. arXiv preprint arXiv:2510.18135, 2025. 2

- [49] Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving,

2025. 1

- [50] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrixgame: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025. 2
- [51] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory, 2025. 3
- [52] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 3

