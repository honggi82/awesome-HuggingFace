# arXiv:2603.22212v1[cs.CV]23Mar2026

## Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models

Meiqi Wu‡1,2,5, Zhixin Cai‡3,5, Fufangchen Zhao‡4,5, Xiaokun Feng2, Rujing Dang5, Bingze Song5, Ruitian Tian5, Jiashu Zhu5, Jiachen Lei5, Hao Dou5, Jing Tang5, Lei Sun5, Jiahong Wu∗5, Xiangxiang Chu5, Zeming Liu3, Kaiqi Huang∗2 1School of Computer Science and Technology, UCAS 2The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, CASIA

- 3School of Computer Science and Engineering, Beihang University
- 4State Key Laboratory of Networking and Switching Technology, BUPT 5AMAP, Alibaba Group

### Abstract

Video-based world models have emerged along two dominant paradigms: video generation and 3D reconstruction. However, existing evaluation benchmarks either focus narrowly on visual fidelity and text–video alignment for generative models, or rely on static 3D reconstruction metrics that fundamentally neglect temporal dynamics. We argue that the future of world modeling lies in 4D generation, which jointly models spatial structure and temporal evolution. In this paradigm, the core capability is interactive response: the ability to faithfully reflect how interaction actions drive state transitions across space and time. Yet no existing benchmark systematically evaluates this critical dimension. To address this gap, we propose Omni-WorldBench, a comprehensive benchmark specifically designed to evaluate the interactive response capabilities of world models in 4D settings. OmniWorldBench comprises two key components: Omni-WorldSuite, a systematic prompt suite spanning diverse interaction levels and scene types; and Omni-Metrics, an agent-based evaluation framework that quantifies world modeling capabilities by measuring the causal impact of interaction actions on both final outcomes and intermediate state evolution trajectories. We conduct extensive evaluations of 18 representative world models across multiple paradigms. Our analysis reveals critical limitations of current world models in interactive response, providing actionable insights for future research. Omni-WorldBench will be publicly released to foster progress in interactive 4D world modeling.

### 1 Introduction

The world models aim to characterize the temporal evolution of environmental states under given interaction conditions, providing a foundation for counterfactual reasoning, planning, and decisionmaking [1]. Taking advantage of recent advances in video generation, this paradigm has increasingly adopted video synthesis as a core implementation pathway. By leveraging high-quality generalpurpose video representations to model world dynamics, video-based world models have been widely applied to autonomous driving, embodied intelligence, and game agents, substantially accelerating progress in these domains.

Unlike rapid progress in world model design, the development of dedicated evaluation benchmarks appears to be somewhat lagging. Existing evaluation methods largely rely on conventional video

‡Work done during the internship at AMAP, Alibaba Group. ∗Corresponding author.

Preprint. Under review.

<First_Frame>

<First_Frame>

LEVEL1 INTERACTION

###### LEVEL3 INTERACTION

[Figure 1]

[Figure 2]

| |[Figure 3]| |
|---|---|---|
| | | |

[Figure 4]

Text Prompts

Entity

Entity Static

Entity Entity

AgenticSc ore

[Figure 5]

[Figure 6]

Static

MLLM

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

<Prompt>

State

Video

Action

<Prompt>

In an indoor workspace, a robotic arm with a black gripper hovers above a table covered with a white cloth. The table holds a green panda-shaped pen holder, a black marker, and a dark bottle. The background shelves hold various items, and a camera is mounted above the arm; the right robotic arm then grasps the black marker and places it into the green panda-shaped pen holder on the left.

Action

Static

at rest no movement

In a still shot, an apple rests on a table.

[Figure 11]

[Figure 12]

W1

<First_Frame>

Interaction Effect Fidelity

Action

LEVEL2 INTERACTION

OmniMetric

[Figure 13]

Entity Entity

Entity

Entity

Right robotic arm state changes Black marker changes motion Relation changes No explicit changes mentioned

Action

- W2
- W3

[Figure 14]

[Figure 15]

Generated Video Quality

OmniWorldBench

<Prompt> A baseball player, standing on the field, throws a baseball as high and as far as he can with all his might.

State

baseball`s position/velocit y changes

Camera-Object Controllability

Omni-WorldSuite

- Figure 1: Overview of Omni-WorldBench. Left: Omni-WorldSuite defines three levels of interactions, each specified by an initial frame and a prompt. Right: Omni-Metrics comprises an evaluation pipeline that measures interaction effect fidelity, generated video quality, camera-object controllability, and spatiotemporal causal coherence, and then employs an MLLM to adaptively fuse these signals into the final AgenticScore.

generation metrics, such as FID and FVD, or adopt general-purpose evaluation benchmarks (e.g., VBench [2]). Although these metrics are effective in measuring visual fidelity and text–video alignment [3], they do not adequately capture the core capability of world models—the ability to generate consistent and plausible responses under varying interaction conditions.

To comprehensively evaluate the interactive response capabilities of world models, we propose a novel benchmark, Omni-WorldBench (Fig. 1). At its core, we construct a systematic prompt suite, Omni-WorldSuite, designed to thoroughly assess model performance across diverse interaction levels and scenario types. Specifically, interaction conditions can produce effects confined to a single object, extend to the local environment, or induce global environmental changes. These progressively increasing interaction scopes impose distinct representational and dynamic modeling requirements on world models. Consequently, the evaluation prompts in Omni-WorldSuite are systematically organized around these three hierarchical interaction levels. Furthermore, since world modeling is a broad and application-dependent research paradigm, existing studies are often grounded in specific domains such as autonomous driving, embodied robotics, and gaming environments. To ensure that Omni-WorldSuite is applicable to both general-purpose video generation models and scenario-specific world models, our evaluation prompts also encompass real-world physical settings as well as representative application domains.

To complement Omni-WorldSuite, we establish a comprehensive and effective evaluation protocol, Omni-Metric, designed to holistically assess the fidelity and consistency of world state representations. Distinct from prior works that predominantly focus on static visual fidelity [4, 5], Omni-Metrics explicitly extends the evaluation toward dynamic, controllable, and interaction-aware generation, which are essential to world models. Specifically, Omni-Metrics evaluates models from three complementary aspects. First, Generated Video Quality extends evaluation beyond static appearance to dynamic perceptual quality, measuring temporal flickering, motion smoothness, content alignment, and dynamic degree to capture the visual coherence of generated sequences over time. Second, Camera-Object Controllability assesses whether a model can follow explicit camera instructions while maintaining coherent object behavior, and further evaluates long-horizon continuity through a novel scene transition metric, Transitions Detect. Third, Interaction Effect Fidelity targets the core capability of interactive world modeling by examining whether actions induce the expected effects on intervened objects in a physically plausible and causally consistent manner, supported by quantitative indicators of action-effect correspondence, physical principles, and spatial logic. Since these dimensions are heterogeneous yet complementary, we further introduce an agent-based aggregation framework that adaptively combines outputs from multiple evaluation tools according to prompt semantics, yielding a unified overall metric, AgenticScore, for more reliable evaluation.

Finally, we conduct a systematic evaluation of 18 representative world models, and the results comprehensively reveal the performance boundaries and limitations of current models in interactive response capabilities. Further human alignment studies demonstrate that Omni-Metric aligns well

with human preferences, validating its effectiveness in assessing world model performance. Our key contributions are as follows:

- 1. To address the critical absence of standardized evaluation protocols, we introduce OmniWorldBench. To the best of our knowledge, this is the first benchmark dedicated to assessing the interactive response capabilities of world models, offering a comprehensive and holistic evaluation framework rather than a narrow capability test.
- 2. We establish a rigorous evaluation infrastructure comprising Omni-WorldSuite, a hierarchical prompt suite spanning diverse interaction levels and scenario types, and Omni-Metric, an agent-based evaluation protocol that quantitatively measures the impact of actions on both final outcomes and intermediate state transitions.
- 3. We conduct a comprehensive evaluation of 18 video generation models and world models, systematically analyzing their performance. Our findings unveil critical limitations in the 4D interactivity capabilities of current world models, highlighting key areas for improvement. Additionally, we propose a curated benchmark, offering to guide and accelerate future advancements in 4D world model generation.

### 2 Related Works

#### 2.1 World Models Design

World models characterize how environment states evolve over time under given interaction conditions, thereby providing effective support for tasks such as counterfactual simulation, planning, and decision-making [1]. Early world models primarily relied on multimodal large language models (MLLMs) [6, 7, 8, 9] that represent world states through textual abstractions [10, 11]. Recent advances in video generation [12, 13, 14, 15, 16] have driven a shift toward video-based world models, which offer a more expressive and grounded representation of complex environments and have emerged as a dominant paradigm in the field [17, 18, 19]. In this work, we focus on world models built upon video generation.

Across different application domains, video-based world models have followed distinct yet intrinsically related technical trajectories. In autonomous driving, world models primarily focus on long-horizon traffic scene evolution and the decision-making of vehicle agents [20]. Representative works such as GAIA [21], DriveDreamer [22], DrivingWorld [23], and Vista [24] leverage actionconditioned future frame prediction to support planning and simulation. In embodied intelligence and robotics, world models place greater emphasis on object-centric dynamics and manipulation control [25]. Methods such as IRASim [26], Cosmos [27], RoboScape [28] and LVP [29] tightly integrate perception, action, and physical reasoning to simulate interaction-driven environment changes. In game environments, works including Genie [30, 31], Matrix-Game [32, 33], WorldPlay [34], and Hunyuan-GameCraft [35, 36] aim to construct highly interactive and playable virtual worlds. Despite differences in input modalities, action spaces, and domain-specific constraints, these methods share a common objective: learning how the environment responds coherently to different interaction instructions. This highlights interaction as a core capability of world modeling [17, 27]. Motivated by this, our benchmark takes interaction as the central axis for evaluating world models.

#### 2.2 World Models Evaluation

Despite the rapid progress of video-based world models, the development of corresponding evaluation benchmarks has remained relatively limited [17]. Early studies [37, 38, 39, 40, 41] primarily rely on generic metrics, such as FID [42], IS [43], FVD [44], which often exhibit significant deviations from human perceptual judgments [45, 46, 47, 48]. Subsequently, several evaluation tools originally designed for video generation, such as VBench [2], have been introduced [49, 36, 47, 50, 16]. While these benchmarks play an important role in assessing overall visual quality and text–video alignment [3, 51], they struggle to adequately characterize the core interactive capabilities of world model tasks. As a result, such metrics provide only limited insights for the design and analysis of interactive world models. Moreover, WorldScore [4] has been proposed as a benchmark specifically tailored to world models. It focuses on evaluating a model’s ability to generate geometrically consistent 3D scenes under viewpoint changes, emphasizing spatial coherence and geometric realism. Although this represents an important step toward world-model-aware evaluation, the considered form of interaction

###### Dataset-grounded Prompt Generation Concept-driven Prompt Generation

[Figure 16]

Caption Generation & Human Verity

[Figure 17]

###### Open Dataset Prototype Extraction

Camera Motion

First Frame

[Figure 18]

First Frame

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

DriveLM InternData-A1 Sekai

Object

Scene Action

GPT5 Deepseek-R1 Gemini Human

Caption Generation & Human Verity

Interaction Hierarchy Self-state

Image Generation & Human Editing

[Figure 24]

Local target

Text Prompts

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Camera Motion

[Figure 32]

[Figure 33]

Text Prompts

[Figure 34]

FLUX.1

Global Cascade

Qwen-VL Human Qwen-Image GPT-5 Human

(a) (b) (c) (d)

|Benchmark|Prompt Presentation| | |Task-Oriented Scene| | |General Scene| | | |Inter.|
|---|---|---|---|---|---|---|---|---|---|---|---|
| |Text|Image|Traj|AD|EAI|Game|PP|LCC|Cau.|CS| |
|VBench|[Figure 35]<br><br>✅|[Figure 36]<br><br>❌|[Figure 37]<br><br>❌|[Figure 38]<br><br>❌|[Figure 39]<br><br>❌|[Figure 40]<br><br>❌|[Figure 41]<br><br>❌|[Figure 42]<br><br>❌|[Figure 43]<br><br>❌|[Figure 44]<br><br>❌|[Figure 45]<br><br>❌|
|WorldScore|[Figure 46]<br><br>✅|[Figure 47]<br><br>✅|[Figure 48]<br><br>❌|[Figure 49]<br><br>❌|[Figure 50]<br><br>❌|[Figure 51]<br><br>❌|[Figure 52]<br><br>❌|[Figure 53]<br><br>✅|[Figure 54]<br><br>❌|[Figure 55]<br><br>❌|[Figure 56]<br><br>❌|
|WorldModel Bench|[Figure 57]<br><br>✅|[Figure 58]<br><br>✅|[Figure 59]<br><br>❌|[Figure 60]<br><br>✅|[Figure 61]<br><br>✅|[Figure 62]<br><br>✅|[Figure 63]<br><br>✅|[Figure 64]<br><br>❌|[Figure 65]<br><br>❌|[Figure 66]<br><br>✅|[Figure 67]<br><br>❌|
|OmniWorldBench (Ours)|[Figure 68]<br><br>✅|[Figure 69]<br><br>✅|[Figure 70]<br><br>✅|[Figure 71]<br><br>✅|[Figure 72]<br><br>✅|[Figure 73]<br><br>✅|[Figure 74]<br><br>✅|[Figure 75]<br><br>✅|[Figure 76]<br><br>✅|[Figure 77]<br><br>✅|[Figure 78]<br><br>✅|

[Figure 79]

[Figure 80]

[Figure 81]

###### OmniWorldBench

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

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

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

[Figure 112]

[Figure 113]

OmniWorldSuite

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

- Figure 2: Omni-WorldSuite Construction Pipeline and Analysis. (a) Dataset-grounded prompt generation. Prompts are generated from open-source data using first-frame and camera-motion cues, refined through VLM captioning, and finally verified by human annotators. (b) Conceptdriven prompt generation. Prompts are derived from interaction prototypes using LLM/VLM-based generation and human curation, together with generated or edited first frames. (c) Suite taxonomy across indoor scenes, including diffusion (Diff.), sliding, and building-related (Buil.) scenarios; outdoor scenes, including natural, projectile motion (Proj.), and urban scenarios (Urban); and taskoriented settings, including robotics (Robot), autonomous driving (Driv.), and gaming (Game). (d) Coverage comparison by prompt modality and capability axes. Abbr.: Traj (camera trajectory); AD (autonomous driving), EAI (embodied AI); PP (physical principles); LCC (loop-closure consistency); Cau. (causality); CS (common sense); Inter. (Interaction).

is largely restricted to camera motion. In contrast, contemporary world models increasingly emphasize a broader range of interaction types [36, 29]. Motivated by this gap, we introduce Omni-WorldBench, an interaction-centric evaluation benchmark that systematically covers multiple levels of interaction complexity. We hope that Omni-WorldBench can serve as a comprehensive tool for characterizing the interactive expressiveness of world models.

- 3 Omni-WorldSuite

To enable a comprehensive analysis of the interactive response capabilities of world models, OmniWorldSuite constructs targeted evaluation prompts across diverse interaction levels and scenario types. In this section, we detail the construction pipeline of Omni-WorldSuite, provide representative examples, and present its statistical analysis.

#### 3.1 Construction Pipeline

The prompts in Omni-WorldSuite are designed along two primary dimensions. The first dimension is scene coverage, spanning both general daily-life scenarios and task-oriented environments such as autonomous driving, embodied AI, and gaming. Collectively, these scenarios cover key aspects of world modeling, including physical laws, commonsense reasoning, causality, camera motion, closedloop dynamics, and spatial constraints. The second dimension is a three-level interaction hierarchy that characterizes the scope of interaction effects (Fig. 1 (Left)). Level 1 involves actions whose effects are confined to the acting object, without altering other objects or the surrounding environment. Level 2 includes localized interactions where one object directly affects another. Level 3 captures more complex interactions that influence multiple objects and lead to broader environmental changes. Each prompt is defined by a textual description of interaction-driven world-state evolution and an initial frame image specifying the starting world state. For a subset of prompts that require explicit camera control, we additionally provide camera trajectories to constrain the viewpoint transition during generation. Fig. 2(a) and (b) illustrate two prompt construction strategies.

Dataset-grounded Prompt Generation. As shown in Fig. 2(a), we introduce a dataset-grounded prompt construction strategy to address the limited realism, complexity, and robustness of synthetic images. We first extract the camera motion trajectory and the first video frame from open-source datasets to serve as the motion and visual prompts, respectively. Next, we employ Qwen-VL [7] to generate an initial caption for the sequence. To mitigate potential errors in spatial relations and object attributes, all generated captions are manually verified and refined to ensure consistency with the source sequence. The final evaluation prompt consists of the verified caption, the initial frame, and, when available, the original camera trajectory, serving as the grounded input for benchmark evaluation. Specifically, Omni-WorldSuite covers three domains:

- • Autonomous Driving, which uses sequences from DriveLM [52]. We extract the first-frame ego-view image together with recorded camera trajectories to evaluate the model’s ability to extrapolate road dynamics under realistic driving conditions.
- • Embodied Robotics, which uses manipulation-oriented tasks from InternData-A1 [53] to examine physical causality arising from robot–object interactions.
- • Gaming and Simulation, which uses Sekai [54] to test whether the model can preserve coherent motion patterns in highly dynamic and non-photorealistic environments.

Concept-driven Prompt Generation. As shown in Fig. 2(b), we introduce a concept-driven prompt construction strategy featuring a generate–verify–refine pipeline to synthesize text, first frames (representing the initial world state), and camera motion trajectories. Specifically, we first build a set of prototype concepts spanning scene domains, target objects, and actions under different interaction levels. We then randomly sample an interaction level, scene type, target entity, and action from the resulting taxonomy. Conditioned on these attributes, ChatGPT-5.2 [55] generates a textual prompt and a camera trajectory. Both outputs are further cross-checked by Gemini [56] and DeepSeek-R1 [57], followed by careful human verification and refinement. This manual revision process eliminates linguistic ambiguity and ensures the clarity, motion plausibility, and overall consistency of the evaluation cases. Finally, we adopt a multi-stage image generation pipeline to ensure high-fidelity initial frames. We use FLUX.1-dev [58] to generate 3 candidates per prompt with a CFG scale of 3.5 and 50 sampling steps. All candidates are manually screened for physical plausibility, instruction adherence, and visual quality. If no valid result is obtained, we rewrite the prompt with ChatGPT-5.2 and, when necessary, apply Qwen-Image [59] for refinement or artifact correction. Only minor localized in-painting is allowed during post-processing. All final images must satisfy quality control requirements, including a minimum resolution of 1024 × 1024, consistency with the prompt, and clear visibility of the target interactive objects.

Omni-WorldSuite Examples. As Fig. 3 illustrates, we pair initial frames with action-driven prompts to demonstrate the three-level interaction hierarchy, visually anchoring relevant entities with red boxes.

- • Level 1: Actions are confined to the acting object without altering other objects or the environment. General Scenes evaluate phenomena like physical optics (e.g., viewing fields through a crystal ball), while Task-Oriented Scenes test continuous spatial navigation (e.g., moving along a riverside path).
- • Level 2: One object directly affects another. Examples include testing thermodynamics in General Scenes (e.g., heating a metal rod in a campfire) and complex ego-vehicle navigation alongside dynamic traffic in Task-Oriented Scenes (e.g., autonomous driving).
- • Level 3: Actions influence multiple objects and lead to broader environmental changes. Prompts cover physical causality in General Scenes (e.g., snapping spaghetti, tidying a room) and multi-stage manipulation in Task-Oriented Scenes (e.g., a robotic arm grasping a bottle and handing it to a person).

#### 3.2 Omni-WorldSuite Analysis and Statistics

Concept Set Analysis. As shown in Fig. 2(c), the set of prototype concepts mainly covers two broad scene categories, namely indoor and outdoor scenes, as well as task-oriented scenarios such

- as autonomous driving, embodied robotics, and gaming. Within each broad category, we further include several representative interaction types. Overall, these prompts span multiple dimensions,

[Figure 150]

[Figure 151]

General Scene Task-Oriented Scene

- LEVEL 1 INTERATION
- LEVEL 2 INTERATION
- LEVEL 3 INTERATION

[Figure 152]

[Figure 153]

[Figure 154]

###### Optics

[Figure 155]

###### Outdoor Camera Trajectory

###### Game Scene

Indoor Camera Trajectory

###### <First_Frame>

<First_Frame> <Trajectory>

<Prompt> The shot opens on a sunlit park path along the water, dotted with shade from trees and lampposts. On the right, weeping willows hang over the water, and distant buildings are partially visible through the trees; the camera then continuously pushes forward along the riverside path, with the row of trees between the path and the waterway accompanying the movement.

###### <First_Frame>

<Trajectory>

<First_Frame>

[Figure 156]

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

|[Figure 161]|
|---|

<Prompt> Distant fields and houses are seen through a transparent crystal ball.

<Prompt> Completing a full rectangular circuit in a silent sci-fi corridor, passing repeating wall panels and steady indicator lights before returning to the original viewpoint.

<Prompt> From a high-altitude rooftop, the camera rotates left in place to scan a calm city skyline during the transition to golden hour.

[Figure 162]

[Figure 163]

[Figure 164]

###### Fluid

[Figure 165]

###### Energy Conservation

###### Thermodynamics

###### Common Sense

[Figure 166]

Autonomous Driving

<Prompt> On a campus-like road on an overcast day, an orange and black "NUS" shuttle bus is driving. Our vehicle begins a rectangular path motion: first, it pushes forward, drawing the view into the depth of the road; then, it dollies right, sweeping across the scenery on the right; next, it reverses to pull back; finally, it dollies left to complete a closed loop, returning to the starting point

###### <First_Frame>

###### <First_Frame>

###### <First_Frame>

###### <First_Frame>

<First_Frame>

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

<Prompt> A small steel ball sits on top of a vertical spring. The spring is compressed forcefully downward and then released, launching the ball upward.

<Prompt> On the African savanna, an adult elephant and a domestic cat stand side by side.

<Prompt> One end of a long metal rod is held in the flames of a campfire.

<Prompt> In the early morning, a dewdrop rests on the surface of a lotus leaf.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### Causality

[Figure 176]

###### Dynamics

###### Spatial Constraints

###### Object Properties

Embodied AI

<First_Frame>

###### <First_Frame>

###### <First_Frame>

###### <First_Frame>

###### <First_Frame>

<Prompt> In an indoor lab scene, a white robotic arm with a black gripper hovers above a white tabletop, with the table and background shelf layout remaining unchanged; the arm slowly adjusts the gripper's posture and opens/closes it slightly, then grasps a bottle of Dongfang Shuye and hands it to a person nearby.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

|[Figure 181]|
|---|

<Prompt> A skier descends from the top of a snowy mountain in an S-s haped path.

<Prompt> Looking down at their phone, a person walks briskly toward a very clean, transparent closed glass door.

<Prompt> A dry strand of spaghetti is held at both ends, and a finger presses down on the middle.

<Prompt> Tidy the roomput the clothes in the laundry basket and stack the books neatly on the bookshelf.

- Figure 3: Omni-WorldSuite examples across three interaction levels. Left: Examples from the General Scene domain. Right: Examples from the Task-Oriented Scene domain, including optics/camera trajectories, game, physics/common sense, autonomous driving, and embodied AI. Each example pairs a first-frame grounding (and trajectory, when applicable) with an action prompt, with red boxes indicating interaction-relevant entities.

ranging from natural environments, urban scenes, and architectural spaces to fundamental physical motion, fluid and thermal phenomena, optical effects, material deformation, commonsense reasoning, object affordance, robotic manipulation, and embodied interaction, thereby forming a comprehensive prompt set that balances scene diversity, physical realism, and task interactivity. Beyond static scene descriptions, the collection also includes a large number of dynamic processes, causally driven events, and goal-oriented manipulation tasks, enabling a systematic evaluation of a model’s capabilities in scene understanding, physical consistency, spatial constraint reasoning, and embodied task execution.

To facilitate the computation of evaluation metrics, we further provide auxiliary metadata for each prompt. (i) First, we enumerate all entity objects appearing in the prompt and categorize them into affected and unaffected sets according to the interaction actions. For affected entities, we additionally annotate the expected coarse motion direction and magnitude. (ii) Next, based on the world evolution described in the textual prompt, we extract a list of key events ordered by their temporal occurrence. (iii) Finally, to evaluate camera motion and spatial consistency, we annotate expected camera motions for a subset of prompts, including the motion direction and magnitude. We also incorporate a challenging return-to-origin setting, where the model is required to return the camera to its original position after completing a motion cycle. Video frames in which the camera revisits the same spatial position are referred to as revisit frames.

Compare with other Benchmarks. As shown in Fig. 2(d), compared with prior benchmarks such as VBench [60], WorldScore [4], and WorldModelBench [61], Omni-WorldBench supports the most comprehensive set of prompt modalities, encompassing text, image, and trajectory inputs. Moreover, it evaluates both task-oriented and general scenes, rather than focusing on only a narrow subset of scenarios. Specifically, it covers a diverse range of scene and reasoning types, including physical regularities, loop-closure motion, causal reasoning, and commonsense reasoning, thereby achieving the broadest coverage of scenario types among existing benchmarks. Furthermore, OmniWorldBench is the first benchmark to explicitly account for interaction types as a core evaluation dimension. This comprehensive design provides a reliable testbed for the development and evaluation of next-generation 4D world models.

Statistics. Omni-WorldSuite contains 1,068 evaluation prompts, making it a comparatively large evaluation suite for video generation assessment. As shown in Fig. 4(a), the suite exhibits a multi-label

(a) Overall Distribution (b) Physics Principles (PP) (c) Commonsense (CS) (d) Causality (Cau)

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

(e) Camera Motion (CM) (f) Loop-Closure Consistency (LCC) (g) Spatial Constraints (SC)

[Figure 187]

[Figure 188]

[Figure 189]

(h) Interaction Level Distribution

(i) World Cloud of Objects (j) Word Cloud of Actions (k) Word Cloud of Scenes

[Figure 190]

[Figure 191]

[Figure 192]

- Figure 4: Statistics of Omni-WorldSuite. (a) Overall Distributions; (b–g) Distributions of core principles; (h) prompt counts by interaction level; (i–k) word clouds of objects, actions, and scenes. NM (Newtonian Mechanics), FM (Fluid Mechanics), MP (Material Properties), WO (Waves and Optics), MC (Momentum and Collision), TP (Thermodynamics and Phase Transition), EC (Energy Conversion and Conservation); SEK (Scene/Event Knowledge), OFK (Object Function Knowledge), HAK (Human Action Knowledge); C2B (Condition-to-Behavior), A2M (Action-to-Motion), C2O (Collision-to-Outcome); TFS (Tracking / Follow Shot), OAS (Orbit / Arc Shot), HHS (Handheld / Shaky); ART (Axial Round-Trip Motion), ODC (Optical / Dynamic Consistency Closure), SCC (Spiral / Composite Closure), CDR (Curved / Diagonal Return Motion), PCP (Planar Closed-Path Motion), ORC (Orbital / Rotational Closure), UNC (Uncategorized); MKC (Mechanical / Kinematic Constraints), CSS (Contact & Support Stability), OAC (Occlusion & Accessibility Constraints), CL (Containment & Leakage), GFSC (Geometric Fit & Size Compatibility), DMC (Deformation & Material Constraints).

distribution over six major annotation dimensions, namely Physics Principles (PP), Commonsense (CS), Causality (Cau), Camera Motion (CM), Loop-Closure Consistency (LCC), and Spatial Constraints (SC). Among these dimensions, Physics Principles appears most frequently, followed by Causality and Commonsense. Fig. 4(b–g) further present the subcategory distributions within each dimension. Specifically, NM and FM are the most frequent categories in Physics Principles; SEK dominates the Commonsense dimension; C2B is the most common causal type; Pan and Tilt are the most frequent camera motion patterns; ART and ODC are the most common loop-closure categories; and MKC appears most frequently among the spatial constraints. Fig. 4(h) further shows that Level 2 contains the largest number of prompts, followed by Level 3 and Level 1. In addition, the word clouds in Fig. 4(i–k) highlight the diversity of objects, actions, and scenes covered by the suite. Overall, these statistics indicate that Omni-WorldSuite is not only large in scale, but also broad in semantic and structural coverage, providing a diverse testbed for evaluating interactive world modeling under physical, causal, spatial, and motion-related constraints.

### 4 Omni-Metric

To facilitate an omni-directional assessment of world models, we introduce Omni-Metric, a framework designed to deliver a truly comprehensive evaluation experience. Omni-Metric delineates three pivotal dimensions: Generated Video Quality, which quantifies both static and dynamic visual fidelity; Camera-Object Controllability, which scrutinizes scene coherence and object controllability in the absence of external interventions; and Interaction Effect Fidelity, which evaluates adherence to physical laws, event interactions, and temporal sequence logic within realistic scenarios. Collectively, these

dimensions establish a rigorous paradigm for benchmarking the perceptual quality, environmental stability, and causal reasoning capabilities inherent to advanced world models.

#### 4.1 Structured Information Extraction

Given a world model to be evaluated, it generates a video v conditioned on an evaluation prompt P (optionally with an initial frame I). Before computing the metrics, we extract structured representations from V ∈ RT×H×W, where T, H, and W are the number of frames, height, and width.

Entity Trajectories. We employ GroundingDINO and SAM to extract temporally consistent segmentation mask sequences for each entity in the video, denoted as {trajk}Nk=1. Here, trajk represents the mask sequence of the k-th entity (among N entities), which is treated as its trajectory representation. Optical Flow. We use RAFT to estimate the optical flow field F of the video, capturing regional motion intensity and dynamic variations.

Relative Camera Motion. Following [62], we approximate the relative camera motion between consecutive frames using optical flow variations, thereby estimating the corresponding camera motion direction and magnitude.

#### 4.2 Generated Video Quality

This section details the Generated Video Quality dimension of Omni-Metrics. For this evaluation, we leverage established metrics from prior benchmarks: specifically, imaging quality, temporal flickering, motion smoothness, and dynamic degree are sourced from VBench [2], while content alignment is adopted from WorldScore [4]. To effectively balance static and dynamic video attributes during assessment, we employ AgenticScore to perform adaptive weight allocation across these indicators. Comprehensive details regarding the AgenticScore mechanism are provided in Section

- 4.5.

#### 4.3 Camera-Object Controllability

Camera-Object Controllability evaluates the coherence of static elements, such as scene layouts and object identities, within generated videos. Specifically, in view-following sequences governed by camera trajectories, this metric assesses whether the scene undergoes anomalous variations or if objects remain strictly consistent with the prompt specifications. Furthermore, acknowledging that scene transitions in generated content often induce camera trajectory discontinuities, we incorporate a dedicated transition assessment module to mitigate evaluation biases arising from such interruptions. This dimension comprises three independent metrics: Camera Control, Object Control, and Transition Detection. Below is a detailed introduction to each metric.

Camera Control. To quantitatively analyze camera motion errors in videos, we employ the camera control metric proposed in WorldScore [4]. This metric evaluates discrepancies in camera trajectories by separately assessing rotational and translational components. These error measurements are subsequently normalized to yield a final score, where a higher value indicates superior performance.

Object Control. To evaluate object generation consistency, existing approaches (e.g., WorldScore) assess the object control metric by detecting objects in videos using models such as GroundingDino and subsequently performing rule-based matching against the prompt text. However, given the inherent limitations in detection accuracy and the susceptibility of rule-based matching to semantic errors arising from synonymy, we propose an improved formulation for this metric. Specifically, we reframe object control as a direct visual question answering (VQA) problem: given a small set of uniformly sampled frames, a multimodal model is asked whether each target object is present

in the video, with a constrained binary response. For a video with object list O = {oi}Ki=1 we query the model independently for each oi and obtain binary predictions yi ∈ ˆ{0,1}. The final score is computed as K1 Ki=1 yˆi, reflecting the proportion of prompt-specified objects that are visually grounded in the generated content. This formulation eliminates brittle rule-based matching and leverages the semantic robustness of large VLMs to synonyms and compositional cues. In addition,

uniform temporal sampling offers a lightweight yet effective summary of the video, providing a practical trade-off between computational cost and coverage of object occurrences.

Transitions Detect. We determine whether a video contains scene transitions using a content-based scene boundary detector. Specifically, we apply PySceneDetect’s ContentDetector [63], which computes frame-to-frame visual dissimilarity (in HSV space) and flags a boundary when the change exceeds a threshold τ, subject to a minimum scene length constraint L (in frames) to suppress spurious detections. Given an input video, we first optionally downsample for efficiency, then perform scene

detection to obtain a scene list tstarti ,tendi Ni=1. The number of scenes N provides a direct indicator of transitions (a transition exists if N > 1). Consistent with the implementation, we map this to a

binary score

##### 1, N = 1 0, N > 1

(1)

strans =

so that videos without scene cuts receive a full score, while any detected transition yields zero. This formulation provides a simple and robust assessment of temporal continuity by penalizing scene breaks while remaining computationally lightweight.

#### 4.4 Interaction Effect Fidelity

As a core contribution of Omni-Metric, the Interaction Effect Fidelity dimension aims to quantitatively assess challenging aspects of video generation, including long-term content consistency and stability, the causal logical ordering of events, and adherence to the physical laws of the real world. To address these challenges, we propose four comprehensive evaluation metrics: InterStab-L, InterStab-N, InterCov, and InterOrder, which are detailed below.

InterStab-L. To rigorously quantify long-horizon temporal coherence, we introduce InterStabL, which assesses the consistency of visual content across user-specified temporal revisit pairs R = {(ta,tb)}. Formally, continuous timestamps are discretized to frame indices within the video sequence of length T. For any frame pair (i,j) corresponding to a revisit pair, we define a composite similarity metric s(i,j) that integrates both low-level structural fidelity and high-level semantic consistency:

- 1

- 2

(SSIMgray (Ii,Ij) + cos(ϕ(Ii),ϕ(Ij))), (2)

s(i,j) =

where SSIMgray denotes the grayscale Structural Similarity Index [64], and ϕ(·) represents a pretrained vision encoder (e.g., the visual tower of CLIP) that maps frames to semantic feature vectors f. To mitigate the degeneracy of trivial static sequences (where high similarity arises from lack of motion rather than stability), we incorporate a dynamics gating mechanism. Specifically, we evaluate similarity across four canonical anchor intervals spanning the video duration; if the average similarity of these anchors exceeds a static threshold τstatic, the metric is penalized to zero to enforce content dynamics. Otherwise, InterStab-L is defined as the mean similarity over the revisit set:

##### 1 |R|

InterStab-L =

s(i(ta),i(tb)) · Idynamic, (3)

(ta,tb)∈R

where Idynamic is the validity indicator derived from the anchor check. A higher InterStab-L score reflects robust long-term consistency at designated temporal intervals, balancing structural preservation with semantic stability.

InterStab-N. Specifically, InterStab-N is used to assess the stability of non-target regions. Given the entity masks extracted in Sec. 4, removing the target masks yields the non-target spatial region N. We then use the flow magnitudes in these regions over the entire video duration T as a measure of their motion energy:

T

1 T

1 |N| x∈N

∥Flowt(x)∥, (4)

Enon(s) =

t=1

where Flowt(x) denotes the optical flow vector at location x in frame t. The resulting motion energy is then mapped to a bounded stability score:

Enon(s) β × min(H,W)

InterStab-N(s) = exp −

, (5)

where β is a scaling factor that, together with the frame resolution, normalizes InterStab-N to [0,1]. Higher InterStab-N values indicate greater stability in the non-target regions.

InterCov. InterCov quantifies object-level causal faithfulness in generated videos by verifying whether interaction-affected entities exhibit semantically consistent responses while unaffected entities maintain temporal stability. This metric complements low-level flow-based coverage with high-level semantic validation, leveraging the reasoning capabilities of Vision-Language Models (VLMs) to assess interaction fidelity. Formally, let O = {o1,··· ,oN} denote the set of target entities subject to causal constraints. We employ a VLM-based semantic verifier to evaluate the video sequence, yielding a binary validity signal vo ∈ {0,1} for each entity o ∈ O, where vo = 1 indicates that the entity’s behavior aligns with the prescribed interaction logic (e.g., dynamic response for affected objects, stationarity for others). The metric is defined as the semantic recall of consistent interactions:

1 |O| o∈O

InterCov =

I(vo = 1), (6)

where I(·) is the indicator function. Consequently, InterCov serves as a rigorous measure of objectlevel semantic consistency, ensuring that generated dynamics adhere to the underlying causal structure.

InterOrder. This metric quantifies the alignment between the chronology of propagated events and the ground-truth sequence E = {ei}Ki=1. Specifically, for any distinct event pair (em,en) satisfying m < n, we employ a pre-trained Vision-Language Model (VLM) as an automated verifier to assess both the occurrence of the events and their relative temporal precedence via a structured query protocol. An event pair is deemed temporally consistent if the generated sequence preserves the ground-truth ordering. Formally, InterOrder is defined as the ratio of consistent event pairs Ks to the total number of possible pairs:

2Ks K(K − 1)

, (7)

InterOrder =

where InterOrder ∈ [0,1]. A higher score indicates superior capability in maintaining temporal coherence and logical event progression.

#### 4.5 AgenticScore

To accommodate diverse application scenarios and capture different aspects of interactive representation ability, the prompts in Omni-WorldSuite emphasize different evaluation focuses. Therefore, when aggregating the metrics to obtain the final score, each prompt should assign different weights to different evaluation dimensions rather than simply averaging all metrics. Inspired by agent-based frameworks, we treat each evaluation metric as an independent evaluation agent. Each metric agent first produces a score for its corresponding dimension, after which an aggregation agent adaptively combines these results according to the semantic content of the prompt to produce the final score.

Specifically, the three interaction-centered evaluation agents—interaction effect fidelity AI, generate video quality AG, and camera-object controllability AC—each compute their scores by averaging the results of their respective sub-metrics. For example, AI = (InterStab − L + InterStab − N + InterCov + InterOrder)/4. The aggregation agent then analyzes the relative importance of these three evaluation dimensions using an MLLM conditioned on the evaluation prompt, and maps the resulting ranking to predefined weight coefficients w1,w2,w3.

The final score, AgenticScore, is defined as:

AgenticScore = w1AI + w2AG + w3AC. (8)

### 5 Experiments

#### 5.1 Models and Evaluation Protocol

Evaluated Models. Across distinct generation tasks—namely Text-to-Video (T2V; Director3D [65], OpenSoraPlan [66], T2V-Turbo [67], HunyuanVideo [68]), Image-to-Video (IT2V; Matrix Game2.0 [33], Wan2.1 [13], Wan2.2 [13], CogVideo [69], OpenSora [70], Cosmos [27],

LargeVideoPlanner [29]), and camera-controlled generation (HunyuanWorld [71], HunyuanGameCraft [35], ViewCrafter [72], Gen3C [73], Lingbot [74], FantasyWorld [75], WonderWorld [76])–we evaluate a total of 18 representative world models encompassing diffusion-based, autoregressive, and hybrid paradigms.

Evaluation Protocol. We comprehensively evaluate the generative capabilities of world models using our proposed benchmark, Omni-WorldBench. The evaluation protocol is driven by OmniMetric (defined in Sec. 4), which encompasses 15 metrics across three distinct dimensions: (1) generated video quality, (2) interaction effect fidelity, and (3) camera and object controllability. To compute these metrics, we introduce a custom test set, Omni-WorldSuite. Specifically, we evaluate T2V and IT2V models using 410 diverse prompts from this suite, while camera-conditioned models are assessed using a dedicated set of 120 prompts equipped with explicit camera trajectories.

#### 5.2 Implementation Details

All inference experiments are conducted using NVIDIA H20 GPUs. To ensure optimal performance and fair comparison, the software environments—specifically the Python and PyTorch versions—are strictly configured according to the official guidelines provided by each model’s respective codebase.

Text-to-Video (T2V) Models. For the T2V generation paradigm, models are conditioned solely on text prompts. Specifically, HunyuanVideo generates 91 frames at a 1280 × 720 resolution using 50 inference steps at 16 FPS. OpenSoraPlan (v1.0.0) employs the T5-v1.1-XXL text encoder, producing 65 frames at 512 × 512 resolution with 250 sampling steps and a classifier-free guidance (CFG) scale of 7.5 at 24 FPS. T2V-Turbo (v2-no-MG V) generates 40 frames at 8 FPS, utilizing 32 inference steps and a CFG scale of 7.5. Notably, Director3D relies on its self-predicted camera trajectories for novel view rendering, outputting 960 × 960 resolution videos.

Image-to-Video (IT2V) Models. IT2V models utilize both a starting frame and text prompts as conditioning inputs. Both Wan2.1 (14B-720P) and Wan2.2 (A14B) generate 81 frames (5 seconds) at 1280 × 720 resolution operating at 16 FPS; however, Wan2.1 uses 50 steps with a 5.0 guidance scale, whereas Wan2.2 uses 40 steps with a 3.5 guidance scale. Cosmos (Cosmos-predict-14B) operates at the same resolution and frame rate but outputs 77 frames using 35 steps and a guidance scale of 7. CogVideo (CogVideoX-5b-I2V) generates 49 frames at 720 × 480 (8 FPS, 50 steps, CFG scale

###### 6). OpenSora (v2) is configured to a 256px (16:9) resolution, yielding 129 frames at 24 FPS with 50 steps and a 7.5 CFG scale. LargeVideoPlanner leverages a base model for 832 × 480 resolution (81 frames, 16 FPS, 40 steps) with customized history and language guidance scales of 1.5 and 2.5, respectively. Finally, Matrix-Game2.0 (universal mode) outputs 650 × 352 videos at 16 FPS, relying on randomly generated camera trajectories.

Camera-Conditioned Models. To evaluate camera controllability, these models require explicit camera parameters or trajectories. HunyuanWorld (v1.5, Autoregressive-480P-I2V) generates videos at 800 × 496 resolution and 16 FPS. Hunyuan-GameCraft utilizes complete pose information to generate 132 frames at 704 × 1216 (24 FPS). ViewCrafter adopts an equidistant camera pose sampling strategy, producing 25 frames at 576 × 1024 (8 FPS). Both Gen3C and Lingbot operate at 720 × 1280 resolution, with their outputs consistently truncated to the first 121 frames. Furthermore, FantasyWorld (832 × 480) and WonderWorld (512 × 512) employ a frame-subsampling strategy, compressing the original 132-frame camera trajectories down to 81 frames. It is worth noting that for WonderWorld, large-scale camera motions in the dataset may occasionally result in blank frames during rendering due to incomplete point cloud coverage.

#### 5.3 Quantitative Evaluation Results and Analysis

This section presents a comprehensive automatic evaluation of various advanced video generation models on the proposed benchmark. As shown in Tab. 1, different model categories exhibit clear trade-offs among interaction fidelity, video quality, and controllability.

Overall Performance: Overall, the Image-to-Video (IT2V) paradigm, which incorporates richer conditional inputs like images, demonstrates the highest performance potential on the current benchmark. Wan2.2 achieves the highest overall AgenticScore across all models at 75.92%, closely followed

Table1:Quantitativeevaluationresultsofvariousmodelsontheproposedbenchmark.ThemetricsaregroupedintoInteractionEffectFidelity,GeneratedVideo

Quality,Camera-ObjectControllability,andtheoverallAgenticScore.Thebestresultswithineachgrouparehighlightedinbold.Avg.=average.

InteractionEffectFidelityGeneratedVideoQualityCamera-ObjectControllabilityAgenticScore

CogVideo—79.0379.5154.8048.9865.5861.4798.0479.1998.8429.0273.3197.5687.3392.4573.27

HunyuanGameCraft27.9664.7851.2846.7437.5050.0867.0996.1247.2998.6791.6780.1795.0084.5569.1767.39

100.00ViewCrafter42.9181.154.2243.1941.1142.4261.3791.0149.0395.4079.3695.0086.1774.6965.88

95.76OpenSoraPlan—68.7840.2936.5960.3653.5598.6782.1699.2216.8370.0998.2970.6584.4768.10

99.02MatrixGame2.0—47.3819.9655.2748.4142.7658.8595.3762.4498.1782.7753.4187.8570.6360.33

90.45FantasyWorld42.2972.6655.3448.4041.9454.5964.8796.3256.3298.6873.3377.9093.3375.3669.49

57.5053.75Gen3c48.0775.9038.4056.3958.5595.7863.8498.8698.3383.0785.8384.5572.8271.61

67.6598.33Lingbot33.9774.8466.5945.2835.2855.5096.9352.8398.6745.8372.3889.7674.0267.16

86.0984.13OpenSora—66.6869.9062.5448.1761.8257.4098.2999.0979.7695.1290.5192.8274.71

98.7894.90Cosmos—79.5579.6353.8951.8166.2266.3098.2980.9399.1744.1577.7791.0175.42

99.4899.6899.75Director3D—73.4989.2444.4838.4161.4148.9089.8747.3177.0572.0785.9171.00

64.5254.1991.93Wan2.1—70.9858.5362.0665.8996.7581.5698.0470.9882.6481.9586.9473.21

82.1587.4398.9999.36LargeVideoPlanner—42.8445.1564.3966.6077.6732.4475.0197.3289.8493.5873.42

84.96100.0085.1896.1285.80WonderWorld24.8951.2643.8451.2460.4092.2674.2299.0273.9587.3374.02

82.9863.4890.2447.8079.54T2V-Turbo—66.1843.8336.7057.4297.6498.5499.0273.7586.3969.85

67.3466.8375.92Wan2.2—79.6879.9856.9952.7098.3679.6799.0946.8378.1696.8391.1894.01

67.9262.2297.6576.1099.1474.36HunyuanWorld55.4077.4955.3148.1564.2767.3280.9096.1087.5279.67

53.0246.7864.8885.3091.9273.96HunyuanVideo—77.3582.3761.6098.6781.1899.3144.8877.1398.54

Control Avg.(%)↑

Detect Object

Control Transitions

Camera

Degree Avg.

Smoothness Dynamic

Alignment Motion

Flickering Content

WithCamera

IT2V

T2V

Quality Temporal

ModelInterStab-LInterStab-NInterCovInterOrderAvg. Imaging

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

cogvideo

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

cosmos Predict2.5

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Direct or3d

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Hunyuan Video

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Large-VideoPlanner

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

MatrixGame2.0

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

OpenSora

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Open-SoraPlan

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

T2vTurboWan2.2Wan2.1

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

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

First_frame:

Prompt: ” A baseball player, standing on the field, throws a baseball as high and as far as he can with all his might.”

- Figure 5: Non-camera-controlled Interaction Comparison. Qualitative comparison of generated videos from different models under the same prompt and first-frame condition. Representative frames illustrate differences in interaction effect fidelity, motion dynamics, and scene coherence during the throwing action.

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

fantasy world

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

gen3c

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Hunyuan gamecraft

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Lingbot world

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

viewcraft

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Wander world

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Hunyuan world

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

Prompt: "Camera View Trajectory-Left Strafe: Shot moving left through the window of a moving train, fields outside rushing by."

First_frame:

- Figure 6: Camera-Controlled Interaction Comparison. Qualitative Comparison of Generated Videos from Different Models under the Same Prompt, First-Frame, and Camera Trajectory Condition.

by Cosmos (75.42%). Among pure Text-to-Video (T2V) models, HunyuanVideo performs the best, reaching 73.96%. In the group supporting explicit camera control (With Camera), hunyuanworld (74.36%) and wonderworld (74.02%) take the lead.

Interaction Effect Fidelity: This dimension evaluates the stability of models in handling complex physical and logical interactions. The IT2V group shows high consistency, with Wan2.2 achieving the highest average score of 67.34%. Notably, some models in the “With Camera” group exhibit a significant trade-off across different interaction sub-metrics. For instance, wonderworld scores an impressive 84.96% on InterStab-L but drops sharply to 24.89% on InterStab-N. This indicates that maintaining consistent underlying interaction logic while introducing complex camera scheduling remains a challenge for current models.

Generated Video Quality: In terms of basic visual quality, the vast majority of evaluated models have reached extremely high levels in Temporal Flickering and Motion Smoothness (mostly exceeding 95.00%). However, there is a substantial variance in the Dynamic Degree across models, which constitutes a core differentiator in generation capabilities. ViewCrafter and WonderWorld achieve a perfect score of 100.00%, while other models in the same group vary significantly. Therefore, the major differences across models no longer mainly come from temporal smoothness, but rather from content alignment and dynamic responsiveness.

This metric directly reflects the models’ ability to precisely control specific elements. Camera-aware methods show clear advantages here. WonderWorld demonstrates an overwhelming advantage with an explicit Camera Control score of 96.12%, far surpassing other models in the same category. Meanwhile, HunyuanWorld obtains the best average controllability score of 79.67% in its group. Furthermore, regarding Object Control, Cosmos (94.90%) and Wan2.2 (94.01%) excel in the IT2V group.

Summary: Current models are already strong in conventional video quality metrics, but still show clear limitations in action-conditioned world evolution, causal interaction consistency, and joint camera-object control. These results highlight the importance of evaluating world models beyond passive video quality and toward agent-centric interactive generation.

#### 5.4 Qualitative Evaluation

Visual Comparison of T2V and IT2V Models. To provide a concrete illustration of our evaluation on interaction effect fidelity and motion dynamics, we present a qualitative comparison in Fig. 5. The models are evaluated under a challenging Level-2 interaction prompt that requires generating a baseball player performing a powerful throw. As shown in the visual sequences, Wan2.2 [13] demonstrates superior performance in this scenario; it successfully synthesizes a complete, anatomically reasonable pitching motion while maintaining the athlete’s structural integrity and scene coherence throughout the video. In stark contrast, Matrix-Game2.0 [33] struggles significantly with this complex physical interaction. The generated action is not only incomplete but also suffers from severe temporal degradation, culminating in the catastrophic collapse and complete disappearance of the human figure in the final frames. These qualitative observations—particularly the stark disparities in physical interaction and temporal consistency—are highly consistent with the quantitative results presented in Sec. 5.3, further validating the effectiveness of our Omni-Metric evaluation framework.

Visual Comparison of Camera-Conditioned Models. In our qualitative analysis, we categorize this example as a Level-1 interaction (camera view trajectory control: left strafe). As shown in Fig. 6, HunyuanWorld [71] exhibits relatively stable performance throughout the sequence. In contrast, ViewCrafter [72] introduces a spurious building that appears out of nowhere, degrading visual consistency and leading to a lower score. This qualitative observation is consistent with our quantitative evaluation results presented in Sec. 5.3, further validating the effectiveness of our Omni-Metric evaluation framework.

### 6 Conclusion

Summary. In this work, we introduce Omni-WorldBench, the first benchmark dedicated to evaluating the interactive response capabilities of video world models. Unlike existing benchmarks that mainly focus on visual quality or motion realism, Omni-WorldBench emphasizes action-driven scene evolution, intermediate state transitions, and causal consistency under interactive prompts, providing a more comprehensive and holistic evaluation perspective. To support this goal, we establish a rigorous evaluation framework consisting of Omni-WorldSuite, a hierarchical prompt suite spanning diverse interaction levels, physical principles, and task-oriented scenarios, and Omni-Metric, an agent-based evaluation protocol that quantitatively measures the impact of actions on both final outcomes and intermediate state transitions, while also assessing non-intervention consistency, spatiotemporal causal coherence, and visual quality, and aggregating them into an overall AgenticScore. Through a systematic evaluation of 18 video generation models and world models, we reveal substantial gaps between visual realism and true interactivity in current systems: although many models achieve strong visual fidelity and motion smoothness, their ability to maintain causally grounded interaction dynamics remains limited. Our results further show that Omni-Metric can effectively capture these differences. We hope Omni-WorldBench can serve as a standardized testbed for diagnosing current limitations and advancing research on more interactive and causally consistent world models, while being continuously refined and extended through community feedback.

Limitations. Despite its broad coverage, Omni-WorldBench still has several limitations. Although Omni-WorldSuite spans diverse physical principles, task-oriented scenarios, and interaction levels, it cannot fully capture the complexity of open-world interactive environments, especially longhorizon and highly dynamic settings. In addition, while Omni-Metric provides a unified protocol for evaluating action-conditioned outcomes and intermediate state transitions, we plan to release human-aligned evaluation results in the future to further complement and validate the assessment of interaction quality.

### References

- [1] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, vol. 2, no. 3, 2018.
- [2] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit et al., “Vbench: Comprehensive benchmark suite for video generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21807–21818.
- [3] X. Liu, X. Xiang, Z. Li, Y. Wang, Z. Li, Z. Liu, W. Zhang, W. Ye, and J. Zhang, “A survey of ai-generated video evaluation,” arXiv preprint arXiv:2410.19884, 2024.
- [4] H. Duan, H.-X. Yu, S. Chen, L. Fei-Fei, and J. Wu, “Worldscore: A unified evaluation benchmark for world generation,” arXiv preprint arXiv:2504.00983, 2025.
- [5] Z. Huang, F. Zhang, X. Xu, Y. He, J. Yu, Z. Dong, Q. Ma, N. Chanpaisit, C. Si, Y. Jiang, Y. Wang, X. Chen, Y.-C. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “VBench++: Comprehensive and versatile benchmark suite for video generative models,” arXiv preprint arXiv:2411.13503, 2024.
- [6] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [7] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
- [8] S. Bai, M. Li, Y. Liu, J. Tang, H. Zhang, L. Sun, X. Chu, and Y. Tang, “Univg-r1: Reasoning guided universal visual grounding with reinforcement learning,” arXiv preprint arXiv:2505.14231, 2025.
- [9] X. Chu, H. Huang, X. Zhang, F. Wei, and Y. Wang, “Gpg: A simple and strong reinforcement learning baseline for model reasoning,” arXiv preprint arXiv:2504.02546, 2025.
- [10] Z. Yang, J. Liu, P. Chen, A. Cherian, T. K. Marks, J. Le Roux, and C. Gan, “Rila: Reflective and imaginative language agent for zero-shot semantic audio-visual navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16251– 16261.
- [11] M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, and M. Hausknecht, “Alfworld: Aligning text and embodied environments for interactive learning,” arXiv preprint arXiv:2010.03768, 2020.
- [12] A. Open, “Sora. creating video from text,” Computer Software]. https://openai. com/sora.
- [13] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang,

- X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan,
- Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z.-F. Wu, and Z. Liu, “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

- [14] F. Mao, A. Hao, J. Chen, D. Liu, X. Feng, J. Zhu, M. Wu, C. Chen, J. Wu, and X. Chu, “Omni-effects: Unified and spatially-controllable visual effects generation,” arXiv preprint arXiv:2508.07981, 2025.
- [15] M. Wu, B. Song, R. Lin, C. Zhu, X. Feng, J. Wu, X. Chu, and K. Huang, “Latent temporal discrepancy as motion prior: A loss-weighting strategy for dynamic fidelity in t2v,” arXiv preprint arXiv:2601.20504, 2026.
- [16] C. Zhu, J. Zhu, Y. Li, M. Wu, B. Song, C. Chen, J. Wu, X. Chu, and Y. Wang, “Artifact-aware evaluation for high-quality video generation,” arXiv preprint arXiv:2601.20297, 2026.
- [17] J. Ding, Y. Zhang, Y. Shang, Y. Zhang, Z. Zong, J. Feng, Y. Yuan, H. Su, N. Li, N. Sukiennik et al., “Understanding world or predicting future? a comprehensive survey of world models,” ACM Computing Surveys, vol. 58, no. 3, pp. 1–38, 2025.

- [18] Z. Zhu, X. Wang, W. Zhao, C. Min, B. Li, N. Deng, M. Dou, Y. Wang, B. Shi, K. Wang et al., “Is sora a world simulator? a comprehensive survey on general world models and beyond,” arXiv preprint arXiv:2405.03520, 2024.
- [19] Y. Zheng, L. Zhong, Y. Wang, R. Dai, K. Liu, X. Chu, L. Lv, P. Torr, and K. Q. Lin, “Code2world: A gui world model via renderable code generation,” arXiv preprint arXiv:2602.09856, 2026.
- [20] T. Feng, W. Wang, and Y. Yang, “A survey of world models for autonomous driving,” arXiv preprint arXiv:2501.11260, 2025.
- [21] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, “Gaia-1: A generative world model for autonomous driving,” arXiv preprint arXiv:2309.17080, 2023.
- [22] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu, “Drivedreamer: Towards real-worlddrive world models for autonomous driving,” in European conference on computer vision. Springer, 2024, pp. 55–72.
- [23] X. Hu, W. Yin, M. Jia, J. Deng, X. Guo, Q. Zhang, X. Long, and P. Tan, “Drivingworld: Constructing world model for autonomous driving via video gpt,” arXiv preprint arXiv:2412.19505, 2024.
- [24] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li, “Vista: A generalizable driving world model with high fidelity and versatile controllability,” Advances in Neural Information Processing Systems, vol. 37, pp. 91560–91596, 2024.
- [25] X. Long, Q. Zhao, K. Zhang, Z. Zhang, D. Wang, Y. Liu, Z. Shu, Y. Lu, S. Wang, X. Wei et al., “A survey: Learning embodied intelligence from physical simulators and world models,” arXiv preprint arXiv:2507.00917, 2025.
- [26] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong, “Irasim: Learning interactive real-robot action simulators,” arXiv preprint arXiv:2406.14540, 2024.
- [27] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical ai,” arXiv preprint arXiv:2501.03575, 2025.
- [28] Y. Shang, X. Zhang, Y. Tang, L. Jin, C. Gao, W. Wu, and Y. Li, “Roboscape: Physics-informed embodied world model,” arXiv preprint arXiv:2506.23135, 2025.
- [29] B. Chen, T. Zhang, H. Geng, K. Song, C. Zhang, P. Li, W. T. Freeman, J. Malik, P. Abbeel, R. Tedrake et al., “Large video planner enables generalizable robot control,” arXiv preprint arXiv:2512.15840, 2025.
- [30] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in Forty-first International Conference on Machine Learning, 2024.
- [31] J. Parker-Holder, P. Ball, J. Bruce, V. Dasagi, K. Holsheimer, C. Kaplanis, A. Moufarek, G. Scully, J. Shar, J. Shi et al., “Genie 2: A large-scale foundation world model,” URL: https://deepmind. google/discover/blog/genie-2-a-large-scale-foundation-world-model, 2024.
- [32] Y. Zhang, C. Peng, B. Wang, P. Wang, Q. Zhu, F. Kang, B. Jiang, Z. Gao, E. Li, Y. Liu et al., “Matrix-game: Interactive world foundation model,” arXiv preprint arXiv:2506.18701, 2025.
- [33] X. He, C. Peng, Z. Liu, B. Wang, Y. Zhang, Q. Cui, F. Kang, B. Jiang, M. An, Y. Ren et al., “Matrix-game 2.0: An open-source real-time and streaming interactive world model,” arXiv preprint arXiv:2508.13009, 2025.
- [34] W. Sun, H. Zhang, H. Wang, J. Wu, Z. Wang, Z. Wang, Y. Wang, J. Zhang, T. Wang, and C. Guo, “Worldplay: Towards long-term geometric consistency for real-time interactive world modeling,” arXiv preprint arXiv:2512.14614, 2025.
- [35] J. Li, J. Tang, Z. Xu, L. Wu, Y. Zhou, S. Shao, T. Yu, Z. Cao, and Q. Lu, “Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition,” arXiv preprint arXiv:2506.17201, 2025.
- [36] J. Tang, J. Liu, J. Li, L. Wu, H. Yang, P. Zhao, S. Gong, X. Yuan, S. Shao, and Q. Lu, “Hunyuan-gamecraft-2: Instruction-following interactive game world model,” arXiv preprint arXiv:2511.23429, 2025.

- [37] R. Feng, H. Zhang, Z. Yang, J. Xiao, Z. Shu, Z. Liu, A. Zheng, Y. Huang, Y. Liu, and H. Zhang, “The matrix: Infinite-horizon world generation with real-time moving control,” arXiv preprint arXiv:2412.03568, 2024.
- [38] J. Yu, Y. Qin, X. Wang, P. Wan, D. Zhang, and X. Liu, “Gamefactory: Creating new games with generative interactive videos,” arXiv preprint arXiv:2501.08325, 2025.
- [39] J. Guo, Y. Ye, T. He, H. Wu, Y. Jiang, T. Pearce, and J. Bian, “Mineworld: a real-time and open-source interactive world model on minecraft,” arXiv preprint arXiv:2504.08388, 2025.
- [40] R. Chen, L. Sun, J. Tang, G. Li, and X. Chu, “Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 3517–3526.
- [41] M. Li, R. Wang, L. Sun, Y. Bai, and X. Chu, “Next token is enough: Realistic image quality and aesthetic scoring with multimodal large language model,” arXiv preprint arXiv:2503.06141, 2025.
- [42] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.
- [43] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” in Advances in Neural Information Processing Systems, 2016.
- [44] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “FVD: A new metric for video generation,” in International Conference on Learning Representations Workshop, 2019.
- [45] M. Ding, W. Zheng, W. Hong, and J. Tang, “Cogview2: Faster and better text-to-image generation via hierarchical transformers,” Advances in Neural Information Processing Systems, vol. 35, pp. 16890–16902, 2022.
- [46] M. Otani, R. Togashi, Y. Sawai, R. Ishigami, Y. Nakashima, E. Rahtu, J. Heikkilä, and S. Satoh, “Toward verifiable and reproducible human evaluation for text-to-image generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14277–14286.
- [47] X. Ling, C. Zhu, M. Wu, H. Li, X. Feng, C. Yang, A. Hao, J. Zhu, J. Wu, and X. Chu, “Vmbench: A benchmark for perception-aligned video motion generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 13087–13098.
- [48] M. Wu, J. Zhu, X. Feng, C. Chen, C. Zhu, B. Song, F. Mao, J. Wu, X. Chu, and K. Huang, “Imagerysearch: Adaptive test-time search for video generation beyond semantic dependency constraints,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 13, pp. 10700–10708, Mar. 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/38044
- [49] H. Che, X. He, Q. Liu, C. Jin, and H. Chen, “Gamegen-x: Interactive open-world game video generation,” arXiv preprint arXiv:2411.00769, 2024.
- [50] X. Feng, H. Yu, M. Wu, S. Hu, J. Chen, C. Zhu, J. Wu, X. Chu, and K. Huang, “Narrlv: Towards a comprehensive narrative-centric evaluation for long video generation,” arXiv preprint arXiv:2507.11245, 2025.
- [51] H. Huang, Y. Wang, Z. Huang, H. Li, T. Huang, X. Chu, and R. Zhang, “Mmgenbench: Fully automatically evaluating lmms from the text-to-image generation perspective,” arXiv preprint arXiv:2411.14062, 2024.
- [52] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, “Drivelm: Driving with graph visual question answering,” in European conference on computer vision. Springer, 2024, pp. 256–274.
- [53] J. Cai, Z. Cai, J. Cao, Y. Chen, Z. He, L. Jiang, H. Li, H. Li, Y. Li, Y. Liu et al., “Internvlaa1: Unifying understanding, generation and action for robotic manipulation,” arXiv preprint arXiv:2601.02456, 2026.
- [54] Z. Li, C. Li, X. Mao, S. Lin, M. Li, S. Zhao, Z. Xu, X. Li, Y. Feng, J. Sun et al., “Sekai: A video dataset towards world exploration,” arXiv preprint arXiv:2506.15675, 2025.

- [55] W. C. Choi and C. I. Chang, “Chatgpt-5 in education: New capabilities and opportunities for teaching and learning,” 2025.
- [56] G. DeepMind, “Gemini 3: Next-generation multimodal models,” 2025, technical Report. [Online]. Available: https://deepmind.google/technologies/gemini/
- [57] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature, vol. 645, no. 8081, pp. 633–638, 2025.
- [58] S. Lee, T. Ebbecke, E. Millon, W. Beddow, L. Zhuo, I. García-Ferrero, L. Esparraguera, M. Petrescu, G. Saß, G. Menezes, and V. Perez, “Flux.1 krea [dev],” https://github.com/krea-ai/ flux-krea, 2025.
- [59] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. ming Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, Z. Zhang, Z. Wang, A. Yang, B. Yu, C. Cheng, D. Liu, D. Li, H. Zhang, H. Meng, H. Wei, J. Ni, K. Chen, K. Cao, L. Peng, L. Qu, M. Wu, P. Wang, S. Yu, T. Wen, W. Feng, X. Xu, Y. Wang, Y. Zhang, Y. Zhu, Y. Wu, Y. Cai, and Z. Liu, “Qwen-image technical report,”

2025. [Online]. Available: https://arxiv.org/abs/2508.02324

- [60] D. Zheng, Z. Huang, H. Liu, K. Zou, Y. He, F. Zhang, Y. Zhang, J. He, W.-S. Zheng, Y. Qiao, and Z. Liu, “VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness,” arXiv preprint arXiv:2503.21755, 2025.
- [61] D. Li, Y. Fang, Y. Chen, S. Yang, S. Cao, J. Wong, M. Luo, X. Wang, H. Yin, J. E. Gonzalez et al., “Worldmodelbench: Judging video generation models as world models,” arXiv preprint arXiv:2502.20694, 2025.
- [62] S. Li, X. Wu, Y. Cao, and H. Zha, “Generalizing to the open world: Deep visual odometry with online adaptation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13184–13193.
- [63] B. Castellano, “PySceneDetect: Scene detection and video splitting library,” https: //www.scenedetect.com/, 2024, accessed: 2024-05-21. [Online]. Available: https: //www.scenedetect.com/
- [64] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
- [65] S. Wu, Y. Lin, F. Zhang, Y. Zeng, J. Xu, P. Torr, X. Cao, and Y. Yao, “Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer,” 2024. [Online]. Available: https://arxiv.org/abs/2405.14832
- [66] B. Lin, Y. Ge, X. Cheng, Z. Li, B. Zhu, S. Wang, X. He, Y. Ye, S. Yuan, L. Chen et al., “Open-sora plan: Open-source large video generation model,” arXiv preprint arXiv:2412.00131, 2024.
- [67] J. Li, W. Feng, T.-J. Fu, X. Wang, S. Basu, W. Chen, and W. Y. Wang, “T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback,” Advances in neural information processing systems, vol. 37, pp. 75692–75726, 2024.
- [68] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang et al., “Hunyuanvideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.
- [69] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “Cogvideo: Large-scale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022.
- [70] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You, “Open-sora: Democratizing efficient video production for all,” arXiv preprint arXiv:2412.20404, 2024.
- [71] T. HunyuanWorld, “Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency,” arXiv preprint, 2025.
- [72] W. Yu, J. Xing, L. Yuan, W. Hu, X. Li, Z. Huang, X. Gao, T.-T. Wong, Y. Shan, and Y. Tian, “Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis,” arXiv preprint arXiv:2409.02048, 2024.

- [73] X. Ren, T. Shen, J. Huang, H. Ling, Y. Lu, M. Nimier-David, T. Müller, A. Keller, S. Fidler, and J. Gao, “Gen3c: 3d-informed world-consistent video generation with precise camera control,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 6121–6132.
- [74] R. Team, Z. Gao, Q. Wang, Y. Zeng, J. Zhu, K. L. Cheng, Y. Li, H. Wang, Y. Xu, S. Ma et al., “Advancing open-source world models,” arXiv preprint arXiv:2601.20540, 2026.
- [75] Y. Dai, F. Jiang, C. Wang, M. Xu, and Y. Qi, “Fantasyworld: Geometry-consistent world modeling via unified video and 3d prediction,” 2025. [Online]. Available: https://arxiv.org/abs/2509.21657
- [76] H.-X. Yu, H. Duan, C. Herrmann, W. T. Freeman, and J. Wu, “Wonderworld: Interactive 3d scene generation from a single image,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5916–5926.

