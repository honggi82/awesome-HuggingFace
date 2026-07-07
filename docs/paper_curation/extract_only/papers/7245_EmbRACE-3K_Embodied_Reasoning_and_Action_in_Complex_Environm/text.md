# arXiv:2507.10548v1[cs.CV]14Jul2025

## EmbRACE-3K: Embodied Reasoning and Action in Complex Environments

Mingxian Lin1* Wei Huang1* Yitang Li2 Chengjie Jiang3 Kui Wu4 Fangwei Zhong4 Shengju Qian3‡ Xin Wang3 Xiaojuan Qi1†

1The University of Hong Kong 2Tsinghua University 3LIGHTSPEED 4Beijing Normal University

https://mxllc.github.io/EmbRACE-3K/

### Abstract

Recent advanced vision-language models (VLMs) have demonstrated strong performance on passive, offline image and video understanding tasks. However, their effectiveness in embodied settings– which require online interaction and active scene understanding– remains limited. In such scenarios, an agent perceives the environment from a first-person perspective, with each action dynamically shaping subsequent observations. Even state-of-the-art models such as GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro struggle in open-environment interactions, exhibiting clear limitations in spatial reasoning and long-horizon planning. These limitations are further emphasized by our empirical analysis of modern VLMs, which reveals consistent failure modes when applied to embodied tasks. To address this gap, we introduce EmbRACE-3K, a dataset of over 3,000 language-guided tasks situated in diverse, photorealistic environments constructed using Unreal Engine and the UnrealCV-Zoo framework. The tasks encompass a wide range of embodied challenges, including navigation, object manipulation, and multi-stage goal execution. Each task unfolds as a multi-step trajectory, pairing first-person visual observations with high-level instructions, grounded actions, and natural language rationales that express the agent’s intent at every step. This design results in fine-grained, temporally grounded annotations that closely align perception with decision-making. In total, the dataset contains approximately 26,000 decision steps, each annotated with multimodal context and step-wise reasoning. Using EmbRACE-3K, we establish a benchmark to evaluate the embodied reasoning capabilities of VLMs such as GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-7B, across three key dimensions: Exploration, Dynamic Spatial-Semantic Reasoning, and Multi-stage Goal Execution. In zero-shot settings, all models achieve success rates below 20%, underscoring the challenge posed by our benchmark and the current limitations of VLMs in interactive environments. To demonstrate the utility of EmbRACE-3K, we further fine-tune Qwen2.5-VL-7B using supervised learning followed by reinforcement learning. This approach yields substantial improvements across all three challenge categories, highlighting the dataset’s effectiveness in enabling the development of embodied reasoning capabilities.

### 1 Introduction

Recent advances in vision-language models (VLMs) have led to strong performance across a wide range of offline, passive understanding tasks, including image captioning, video summarization,

* Equal Contribution, ‡ Project Lead, † Corresponding Author.

Preprint. Under review.

[Figure 1]

- Figure 1: EmbRACE-3K dataset. The dataset contains over 3k language-guided tasks and 26k decision steps set in diverse, photorealistic environments. Each task involves high-level natural language instructions, grounded actions (e.g., move, turn, look, interact), egocentric visual observations, and step-wise reasoning. Agents interpret visual inputs and follow instructions to execute multi-step decision trajectories, with each step annotated with natural language rationales—forming a coherent and interpretable decision process over a long horizon that involves spatial reasoning.

and visual question answering. State-of-the-art models such as GPT-4o [17], Gemini 2.5 Pro [6, 7], Claude-3.5-Sonnet [2], and Qwen2.5-VL [3] demonstrate impressive capabilities in aligning visual and linguistic information, particularly when operating on pre-recorded image or vision sequences in static, non-interactive settings.

However, these models often fall short when applied to embodied tasks, where an agent must actively perceive, reason, and act within an interactive environment [5, 12]. Unlike passive vision benchmarks, embodied scenarios involve a closed-loop perception-action cycle: what the agent sees next is determined by the actions it takes now. A single turn or misstep can dramatically alter subsequent observations. In this dynamic, egocentric setting, agents must follow high-level instructions, adapt to constantly shifting visual inputs, and make temporally coherent decisions under partial observability. This tight coupling between perception and action introduces challenges that go far beyond object recognition or static scene understanding, requiring reasoning over how decisions shape future inputs– posing a fundamentally different learning problem.

Despite this fundamental shift, existing VLMs are often deployed in embodied settings without structural adaptation. In practice, they process short video clips or image sequences as static input, ignoring the dynamic nature of egocentric interaction. This results in a training-deployment mismatch. In our preliminary experiments with Qwen2.5-VL and GPT-4o in simulated embodied environments, we observed consistent failure patterns: the models tend to overfit to immediate visual cues, fail to adjust spatial reasoning as the viewpoint changes, and struggle to maintain attention on objects that briefly exit the field of view. These issues underscore the limitations of passive pretraining when applied to sequential, interactive decision-making tasks.

To address this gap, we introduce EmbRACE-3K: Embodied Reasoning and Action in Complex Environments (shown in Figure 1). This dataset comprises over 3,000 language-guided tasks collected in diverse, photorealistic Unreal Engine environments and controlled via the UnrealCV-Zoo framework. Each task unfolds as a multi-step trajectory in which the agent receives a high-level instruction and interacts with the environment through vision and action. In total, EmbRACE-3K includes approximately 26,000 decision steps, each annotated with egocentric visual observations, the selected action, and a natural language “thinking” rationale that explains the agent’s intent.

Unlike prior datasets, EmbRACE-3K is built to capture the causal structure of embodied interaction by explicitly modeling how decisions affect perception, and how perception guides subsequent reasoning. It provides fine-grained, step-level annotations that align not only with what the agent observes and does, but also with why it acts. These annotations include intermediate reasoning steps, enabling models to learn perception-conditioned decision making rather than relying solely on end-to-end action prediction. Crucially, EmbRACE-3K supports online, closed-loop interaction, where each action taken by the agent dynamically changes the environment and influences future observations. This setup allows for realistic, temporally extended evaluation under partial observability, supporting spatial-semantic consistency and long-horizon goal pursuit. Compared to prior work such as Octopus [28], which formulates reasoning at the level of code generation without step-wise visual grounding, or datasets like ALFRED [23] that rely on pre-defined trajectories, EmbRACE-3K enables fine-grained, multimodal alignment between perception, language, reasoning, and action. This makes it a strong foundation for evaluating and training embodied agents that not only act, but also interpret, reason, and adapt over time in photo-realistic simulated environments.

By shifting the focus from passive visual comprehension to instruction-guided, step-wise reasoning, EmbRACE-3K enables the training and evaluation of vision-language agents in goal-oriented embodied scenarios. First, we use EmbRACE-3K to establish a benchmark targeting three core embodied capabilities: Exploration, Dynamic Spatial-Semantic Reasoning, and Multi-stage Goal Execution. In zero-shot evaluations, state-of-the-art models– GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-7B—achieve success rates below 20% on all three tasks, revealing a significant gap between current VLM capabilities and the demands of embodied reasoning. Next, we fine-tune Qwen2.5-VL-7B on EmbRACE-3K using a two-stage approach: supervised fine-tuning (SFT) followed by reinforcement learning (RL). This significantly boosts the model’s performance across all tasks, resulting in higher success rates and lower goal distance error (GDE) compared to GPT-4o and Gemini 2.5 Pro. These results validate the effectiveness of EmbRACE-3K in enabling VLMs to acquire embodied reasoning capabilities. Moreover, we observe that models trained with SFT alone perform well on in-domain tasks, but suffer marked performance degradation on out-of-domain scenarios. This limited generalization highlights the importance of reinforcement-based adaptation for improving robustness in unfamiliar environments. It also underscores the pressing need for highquality, interaction-centric datasets to support training in such settings– an area where EmbRACE-3K is particularly well-positioned to contribute.

### 2 Related Work

Vision-Language Models Recent advancements in vision-language models (VLMs) have demonstrated remarkable progress in both architectural innovations and performance. For example, GPT4o [18] has achieved significant improvements in visual understanding by leveraging enhanced reasoning capabilities. Similarly, Gemini Pro 1.5 [24] has extended the context length to an unprecedented 1 million tokens, positioning itself as a leader on long video benchmarks [9]. The open-source VLM ecosystem has also seen substantial growth, driven by improvements in model architecture and training methodologies. Notable contributions include state-of-the-art models such as InternVL3 [34], Qwen2.5-VL [26], LLaVA-OneVision [13], and Llama3.2-Vision [16]. These advancements have significantly narrowed the performance gap between open-source and proprietary VLMs, with many open-source models now achieving competitive results.

Embodied Tasks for VLM Recent efforts in embodied vision-language modeling have targeted diverse forms of perception, control, and supervision. Octopus [28] leverages code synthesis to bridge language and action planning, incorporating environmental feedback in simulation. SayCan [1] grounds language in executable actions using affordance-based filtering. Ego4D [10] focuses on passive egocentric video understanding without action execution, while TEACh [19] centers on dialog-driven task planning in multi-agent settings. Octopi [31] extends grounded reasoning to the tactile modality. Despite these advances, many benchmarks lack fine-grained supervision, online interaction, or visual realism. EmbRACE-3K addresses these limitations by providing a fully stepwise, spatio-temporally grounded, and closed-loop evaluation framework built in photo-realistic Unreal Engine environments. Each decision step is paired not only with egocentric observations and grounded actions, but also with explicit reasoning annotations that capture the agent’s intent and intermediate thinking process. This enables more interpretable and diagnostic evaluation of vision-language models in long-horizon embodied tasks.

- Table 1: Comparison of Embodied Benchmarks for VLM Evaluation across Multiple Dimensions Benchmark Input Modality Scene Fidelity Level Step-wise Spatio-Temporal Aware Online Closed-loop

ALFRED [23] Language&Vision Indoor Photo-Realistic ✓ × × ✓ Octopus [28] Language&Vision Indoor&Outdoor Game-based × ✓ ✓ ✓ HabitatNav [25] Vision Indoor Real-world ✓ ✓ ✓ × MindCube [30] Vision Indoor Real-world × ✓ × × V-IRL [27] Language&Vision Outdoor Real-world ✓ ✓ ✓ × VSI [29] Vision Indoor Real-world × ✓ × × MCU [14] Language&Vision Indoor&Outdoor Game-based ✓ × × ✓

EmbRACE-3K Language&Vision Indoor&Outdoor Photo-Realistic ✓ ✓ ✓ ✓

### 3 Pilot Study

We conducted a set of preliminary evaluations using GPT-4o and Qwen2.5-VL in simulated embodied environments. Despite architectural differences, both models exhibited similar failure patterns when tasked with step-wise, instruction-driven tasks. These observations highlight fundamental limitations of current video-trained VLMs in embodied settings.

- (i) Short-sighted exploration. The models tend to fixate on immediate visual cues, lacking the ability to plan toward long-term goals. In the task "Locate and approach the red car", for instance, agents briefly glance left, find no immediate evidence of the target, then glance right with the same result, and promptly proceed forward without conducting a broader search. This behavior suggests that each movement is selected based on local visual feedback rather than an integrated exploration strategy. In "Find the plant near the shelf", agents abandon shelf traversal after only a few head turns, missing nearby objects outside their immediate field of view. This limitation can be traced to how VLMs are typically trained. In conventional video datasets, the model passively receives visual input and learns to summarize or answer questions based on full-sequence observations. There is no need to actively decide where to look or how to explore, so these models never acquire the capacity for information-seeking behavior.
- (ii) Dynamic spatial-semantic drift. The interpretation of spatial relations becomes unstable as the agent moves, due to a lack of egocentric pose awareness. In tasks like "Approach the second trash can" or "Go to the white house in front", agents initially respond correctly to spatial cues but fail to adapt as their viewpoint changes. Ordinal and directional terms such as "second" and "front" become detached from the agent’s current orientation, leading to consistent semantic misalignment. This issue arises because most VLMs are trained on static or loosely time-linked image-text data. Even in video-based pretraining, spatial reasoning is often limited to temporal QA, captioning, or event ordering, where egocentric position and spatial frame shifts are rarely encoded. As a result, these models rely on static geometric assumptions that do not update as the agent moves, causing a gradual drift in language grounding.
- (iii) Target forgetting. Models often fail to retain intent beyond the current frame, leading to target loss when objects briefly leave the field of view. For instance, in "Go near the red car", a temporary disappearance of the car results in its permanent omission. Similarly, in multi-stage instructions like "First reach the trash can, then go to the red car", the agent frequently completes the first task but fails to recall the second goal after unrelated actions. This issue stems from the pretraining objectives of most video-language datasets, which emphasize frame-level recognition, counting, or sequence-level QA, rather than persistent target awareness over time. As a result, sudden appearances or disappearances of objects are not treated as meaningful, leaving the model unable to track unseen but relevant entities.

These behaviors expose critical gaps in current VLM capabilities and motivate the design of EmbRACE-3K to support better spatial reasoning, goal continuity, and instruction-grounded action planning.

### 4 Data Collection and Benchmark Construction

To support systematic investigation of embodied vision-language reasoning, we construct EmbRACE-3K: Embodied Reasoning and Action in Complex Environments. This benchmark is designed to expose the structural mismatch between passively trained VLMs and the demands of step-wise, instruction-driven embodied tasks. Informed by our pilot study, which highlighted issues

[Figure 2]

- Figure 2: Multi-stage Embodied Task Data Collection Pipeline. The EmbRACE-3K dataset is built in four stages: (1) sampling diverse 6-DoF agent poses with ego views in virtual environments, (2) generating grounded task instructions using Gemini, (3) collecting human demonstrations, and (4) annotating each action with step-wise natural language reasoning to explain agent decisions and enhance interpretability.

such as short-sighted action selection, spatial-semantic misalignment, and goal forgetting, we develop a multi-stage data pipeline (shown in Figure. 2) that provides fine-grained supervision aligned with closed-loop reasoning.

#### 4.1 Simulation Platform and Environmental Diversity

All data in EmbRACE-3K are collected within the UnrealCV-Zoo framework [33], which extends Unreal Engine with first-person control and low-level API access. From 100 available photorealistic environments, we select 24 diverse maps that span indoor and outdoor settings, varying in object density, spatial topology, lighting, and navigational complexity. This diversity enables robust evaluation of generalization across scene types and task variations.

#### 4.2 Multi-stage Embodied Task Data Collection

Our data collection proceeds in four structured stages, designed to capture the full perceptionreasoning-action loop required for interactive embodied tasks. This process emphasizes both scene diversity and reasoning complexity to facilitate robust agent training and evaluation.

- Stage 1: Environment Sampling and Pose Selection We begin by sampling diverse agent poses across selected simulation maps using a hybrid strategy that combines automated scripts and manual inspection. The automated component leverages the Navigation Movement API provided by Unreal Engine to uniformly explore traversable regions. To ensure data quality, sampled positions undergo manual validation to filter out visually trivial locations (e.g., texture-less walls) or physically unreachable areas (e.g., positions blocked by obstacles or un-navigable terrain). Each selected pose is recorded with full 6-DoF (degrees of freedom) coordinates—including position and orientation as well as the corresponding egocentric RGB image captured from the agent’s first-person perspective.
- Stage 2: Task Instruction Generation For each selected agent pose, we retrieve object-level metadata within a 1000 meter radius, including the semantic names and spatial positions of nearby objects. This contextual information, along with the egocentric RGB view captured at the pose, is provided to the Gemini 2.5 pro model to generate natural language task instructions. The model is explicitly conditioned on the spatial layout and visual context to ensure semantic grounding, producing instructions that are both plausible and solvable within the local environment.

We also inform the model of the desired task type prior to generation, guiding it to produce tasks aligned with one of five categories identified in our pilot analysis of embodied reasoning challenges:

[Figure 3]

[Figure 4]

(a)

(b)

[Figure 5]

[Figure 6]

400

800

300

600

200

400

100

200

0

0

5 10 15 20 25 30

10 21 32 43 54 65 76 87 98 109 122

Thinking Word-Cloud of EmbRACE-3K

Thinking Token Number Distribution

Action Trajectory Length Distribution Task Word-Cloud of EmbRACE-3K

- Figure 3: The Token Number and Word-Cloud Distribution of EmbRACE-3K. (a)Token number distribution of reasoning and the length distribution of action trajectories. (b)The word clouds of task instructions and agent thinking processes in EmbRACE-3K.

- 0. Basic: Target is clearly visible and immediately reachable, requiring minimal reasoning.
- 1. Exploration: Target is initially out of view, prompting the agent to perform an active search.
- 2. Dynamic Spatial-Semantic: Target is described using relative or ordinal spatial references.
- 3. Multi-stage: Task requires completing a series of subgoals in a specific order.
- 4. Interaction: Task requires direct manipulation (e.g., open a door, pick or drop an object).

To ensure quality and diversity, all generated instructions undergo a post-processing stage that includes manual verification and targeted manual authoring. Annotators inspect generated instructions to verify consistency with visual and spatial context, correct ambiguous phrasing, and supplement the dataset with novel, human-authored tasks for underrepresented cases. This hybrid generation-and-curation pipeline ensures both scalability and high-quality alignment with embodied agent capabilities.

- Stage 3: Human Demonstration and Trajectory Capture Each generated instruction is performed by a human player controlling the agent in real time. We record all egocentric frames, executed actions, and precise pose trajectories. These demonstrations provide high-quality behavioral examples that encode closed-loop dependencies between perception, action context, and intent. The resulting action sequences are typically sparse and efficient, reflecting realistic strategies for exploration and goal completion.
- Stage 4: Step-wise Reasoning Annotation To improve interpretability and facilitate cognitive supervision, we annotate each step in the demonstration trajectory with natural language explanations of the chosen actions. Unlike traditional chain-of-thought (CoT) [21] methods that focus on isolated frames, our annotations are grounded in the agent’s egocentric perspective and the full task context. Gemini receives the task instructions, complete egocentric views, and the entire action trajectory, enabling holistic reasoning about how each action contributes to the final goal and influences future observations. These explanations capture not only the action taken but also its relevance to the spatial structure, task dynamics, and overall intent. This approach ensures that CoT traces provide decision-level supervision tightly aligned with the perception-action cycle.

#### 4.3 Data Curation

To ensure high-quality and interpretable data, we apply a series of post-processing and analysis steps to refine the raw collection. First, we filter out trajectories with more than 32 steps to simplify training and evaluation, ensuring consistent sequence length across tasks. Second, we categorize all instructions into five high-level task types based on reasoning demands: Basic, Exploration, Spatial-Relational, Multi-stage, and Interaction. The Interaction class is further subdivided into two subtypes, Open the Door and Pick and Drop the Object, based on UnrealZoo’s interaction primitives. The resulting type distribution is shown in Figure 4, with Basic tasks accounting for approximately half of the dataset.

In addition to type-based categorization, we conduct a series of corpus-level analyses to characterize the dataset. Figure 3(a) illustrates the distributions of action trajectory lengths and reasoning token counts, showing that most tasks involve under 15 steps and 80 thinking tokens. Figure 3(b) presents word clouds derived from both task instructions and CoT annotations, revealing distinct vocabularies for goal specification and intermediate reasoning.

Finally, we standardize all trajectories into a unified format, including ordered egocentric frames, discrete action sequences, 6-DoF poses, and aligned language fields such as instruction, thinking trace, and step-level justification. Visual content is normalized for resolution and field of view to

###### (a) (b)

Scene 1 Scene 2 Scene n

Scene 1

|[Figure 7]<br><br>[Figure 8]|[Figure 9]<br><br>[Figure 10]|
|---|---|
| | |

|[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]<br><br>[Figure 14]| |
|---|---|
| | |

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]|
|---|

| | | |…|…|
|---|---|---|---|---|
| |Vis|ion Languag|e Model| |
|thinking<br><br>| | |
|---|---|
<br><br>action| |thinking action|…|thinking action|

|system prompt| |
|---|---|
| | |
|task prompt| |
| | |
| | |

Accuracy Reward

Format Reward

system prompt

Group Samples

| | | |
|---|---|---|
| | | |

###### Policy Model

| | |
|---|---|

| | |
|---|---|

task prompt

thinking

action

###### Policy Model

[Figure 19]

❄

| | |
|---|---|

| | |
|---|---|

| | | |
|---|---|---|
| | | |

Policy Model

thinking

action

[Figure 20]

🔥

###### Reference Model

…

…

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

[Figure 21]

❄

th

logits

thinking

action

Figure 5: Two-stage Training framework for Embodied Agent on EmbRACE-3K. (a) SFT training pipeline for agent in open-environment; (b) GRPO training pipeline for agent in open-environment

ensure consistency across samples. These steps ensure that EmbRACE-3K offers not only scale and diversity, but also the structural coherence required for training and evaluating embodied visionlanguage models.

### 5 Reasoning Training Pipeline

As illustrated in Figure 5, we designed two distinct training frameworks within the EmbRACE-3K dataset to enhance the spatial understanding and action planning capabilities of embodied agents. Specifically, we proposed a supervised fine-tuning (SFT) pipeline based on spatial reasoning, which leverages memory learning to strengthen the agent’s reasoning abilities for scenes and actions. Additionally, we developed an exploratory reasoning framework grounded in reinforcement learning, where a rule-based reward function enables the agent to autonomously learn reasoning skills.

|50%<br><br>12%<br><br>13%<br><br>8%<br><br>5%<br><br>12%<br><br>17%<br><br>Task Type Distribution<br><br>Type 0: Basic<br><br>Type 1: Exploration<br><br>Type 2: Dynamic Spatial-Semantic<br><br>Type 3: Multi-Stage<br><br><br>Type 4.1: Open Door<br><br>Type 4.2: Pick & Drop<br>|
|---|

Figure 4: Distribution of task types.

#### 5.1 Supervised Fine-tuning for Reasoning Memory

We utilized Qwen2.5-VL-7B as the foundational models and trained with 2,344 high-quality reasoning trajectories from EmbRACE-3K, encompassing a total of 10k trainable actions. This training phase was designed to endow the model with enhanced capabilities for understanding new visual scenes and reasoning for action decision-making. As shown in Figure 5(a), we directly deployed Qwen2.5-VL7B into the Llama-Factory [32] framework for instruction-based SFT through multi-turn dialogues. The supervised training outputs consisted of two key components: the reasoning process enclosed in the <think></think> tag and the final action decision enclosed in the <action></action> tag. The SFT process was conducted on 8 GPUs.

#### 5.2 Reinforcement Learning for Reasoning Exploring

Recent advancements [22, 15] suggest that reinforcement learning fine-tuning can lead to breakthroughs in reasoning capabilities across domains such as mathematics and coding. Additionally, reinforcement learning has achieved promising results in multimodal understanding tasks involving images and videos. Building on the observed advancements of the group relative policy optimization (GRPO) algorithm in DeepSeek-R1 [11] and prior explorations of multimodal reasoning training [8, 20], we further adopt the standard GRPO framework, as illustrated in Figure. 5(b), to investigate the impact of reinforcement learning on the reasoning abilities of embodied agents using the EmbRACE-3K dataset. For a given question q, the policy model generates a group of candidate responses {o1,o2,...,oG} using the previous policy πθ

. Each candidate response is associated with a corresponding reward {r1,r2,...,rG}, which is computed based on rule-based reward functions, such as those evaluating format and accuracy. The updated model πθ is subsequently trained by maximizing the following objective function:

old

- Table 2: Performance comparison on Type Basic, Exploration and Dynamic Spatial-Semantic tasks.

Method Basic Exploration Dynamic Spatial-Semantic

SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓ SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓ SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓

###### In-Domain

GPT-4o 53.6% 484.7 0.396 17.9 37.1% 14.3 % 1178.3 0.086 28.5 75.0% 62.9% 374.1 0.521 12.5 25.7% Gemini 2.5 Pro 76.4% 232.2 0.649 10.1 13.6% 39.3 % 1068.9 0.264 24.3 57.1% 71.4% 238.1 0.589 13.1 25.7% Qwen2.5-VL-origin 26.4% 531.6 0.176 23.4 65.7% 0.0 % 991.8 0.000 28.7 85.7% 14.3% 527.9 0.079 28.0 80.0% Qwen2.5-VL-no-thinking 79.3% 232.4 0.775 4.3 0.7% 28.6 % 652.1 0.268 8.3 3.6% 68.6% 298.5 0.580 6.0 0.0% Qwen2.5-VL-sft-only 72.9% 237.9 0.647 7.9 4.3% 71.4 % 279.3 0.594 15.1 7.1% 68.6% 245.0 0.557 10.1 2.9% Qwen2.5-VL-sft-rl 81.4% 215.7 0.766 6.2 3.6% 60.7 % 391.8 0.578 11.9 7.1% 68.6% 238.4 0.612 7.3 2.9%

###### Out-of-Domain

GPT-4o 20.8% 1278.6 0.163 20.9 45.4% 3.6 % 4017.8 0.011 30.9 90.9% 10.2% 2144.2 0.078 26.2 67.8% Gemini 2.5 Pro 38.0% 643.8 0.336 12.4 16.2% 9.1 % 2166.5 0.077 25.3 60.0% 20.3% 971.9 0.176 19.2 37.3% Qwen2.5-VL-origin 10.6% 4276.2 0.083 25.3 73.1% 0.0 % 9978.3 0.000 31.2 94.5% 8.5% 7844.0 0.079 26.0 74.6% Qwen2.5-VL-no-thinking 45.8% 595.4 0.446 6.9 0.9% 10.9 % 1340.4 0.105 8.2 0.0% 27.1% 907.5 0.268 7.6 3.4% Qwen2.5-VL-sft-only 49.1% 594.2 0.424 10.7 5.6% 22.8 % 1239.7 0.224 19.3 18.2% 35.6% 839.7 0.333 12.1 5.1% Qwen2.5-VL-sft-rl 53.2% 520.9 0.513 7.9 2.8% 30.9 % 1162.8 0.288 13.4 7.3% 42.4% 824.6 0.405 9.8 5.1%

Table 3: Performance comparison on Type Multi-stage, Open Door and Pick&Drop tasks.

Method Multi-stage Interaction - Open Door Interaction - Pick and Drop

SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓ SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓ SR ↑ GDE ↓ SSPL ↑ Steps ↓ TR ↓

###### In-Domain

GPT-4o 27.3% 478.9 0.194 23.0 40.9% 28.6 % 275.2 0.160 25.1 62.6% 23.5% 2124.6 0.213 22.8 55.9% Gemini 2.5 Pro 40.9% 362.5 0.383 19.8 40.9% 53.8 % 179.8 0.474 10.5 12.1% 39.8% 1615.3 0.255 26.2 73.5% Qwen2.5-VL-origin 0.0% 643.7 0.000 28.3 86.4% 9.9 % 354.34 0.043 29.7 84.6% 0.0% 8882.8 0.000 32.0 100.0% Qwen2.5-VL-no-thinking 72.7% 278.8 0.708 9.7 4.5% 68.1 % 113.2 0.648 7.1 1.1% 64.7% 3069.3 0.587 13.3 0.0% Qwen2.5-VL-sft-only 81.8% 205.2 0.707 13.9 4.5% 80.2 % 99.9 0.723 8.4 3.3% 44.1% 2645.5 0.388 15.2 0.0% Qwen2.5-VL-sft-rl 81.8% 283.6 0.771 10.6 0.0% 73.6 % 107.9 0.687 7.6 1.1% 50.0% 2665.5 0.447 13.4 0.0%

###### Out-of-Domain

GPT-4o 2.7% 1312.9 0.027 25.8 64.9% 16.7 % 1804.6 0.069 29.1 83.3% 29.2% 5787.5 0.270 25.8 70.3% Gemini 2.5 Pro 16.2% 1207.4 0.156 17.6 32.4% 33.3 % 430.3 0.318 17.8 30.3% 25.0% 6746.2 0.192 23.1 54.2% Qwen2.5-VL-origin 0.0% 8788.9 0.000 30.7 94.6% 7.6 % 9761.5 0.037 28.8 83.3% 0.0% 9999.0 0.000 32.0 100.0% Qwen2.5-VL-no-thinking 10.8% 1059.9 0.108 11.3 2.7% 30.3 % 316.1 0.290 8.3 0.0% 33.3% 8431.9 0.457 11.8 0.0% Qwen2.5-VL-sft-only 18.9% 1122.5 0.170 18.7 21.6% 42.4 % 283.2 0.352 12.9 9.1% 32.8% 5147.6 0.319 19.6 8.3% Qwen2.5-VL-sft-rl 27.0% 1265.7 0.253 16.5 16.2% 45.5 % 268.9 0.381 11.8 6.1% 37.5% 4527.3 0.341 14.5 4.2%

JGRPO(θ)=Eq,{o

i}

1 G

θ(oi|q)

θ(oi|q)

G i=1 min π

πθold(oi|q)Ai,clip π

πθold(oi|q),1−ϵ,1+ϵ Ai −βDKL(πθ||πref)

where, ϵ and β are hyperparameters. Given that EmbRACE-3K contains action trajectories of up to 32 steps in length and corresponding input vision, we set G = 6 to normalize the sampled re-

i−mean({r1,r2,...,rG})

wards, thereby computing the advantage, Ai = r

std({r1,r2,...,rG}) , for updating the model. This approach aims to guide the embodied agent in freely exploring reasoning strategies within open environments. The rule-based supervision incorporates a reward format in the form of <think></think> and <action></action>, directly evaluating the content of the actions. We conducted GRPO training on the R1V [4] framework using 8 GPUs.

### 6 Experiments of Embodied Agent

#### 6.1 Experiment Setup

To evaluate the effectiveness of EmbRACE-3K and its contribution to embodied reasoning, we conduct experiments on both in-domain and out-of-domain tasks sampled from UnrealZoo environments. Test scenarios cover six task types defined in our benchmark: Basic, Exploration, Dynamic Spatial-Semantic, Multi-stage, Interaction - Open Door, and Interaction

- - Pick and Drop. This diverse coverage allows us to assess model behavior under different spatial, semantic, and sequential reasoning requirements.

Each test prompt includes a structured input consisting of the task instruction, a brief description of the current scene, and a history of previously executed actions. For the visual input, we provide the first-person egocentric views at the current time step, as well as the five most recent frames and the initial frame. This limited-frame strategy strikes a balance between temporal context and computational tractability. Including the full trajectory often leads to excessive latency and model timeout. Then, we evaluate a range of models as baselines: - GPT-4o, Gemini 2.5 Pro, and the original Qwen2.5-VL-origin serve as zero-shot or few-shot baselines without task-specific tuning.

- - Qwen2.5-VL-sft-rl: Our fully fine-tuned variant, which begins with SFT on EmbRACE-3K and is further trained using reinforcement learning with trajectory-level reward shaping. - Qwen2.5VL-sft-only: A model trained only with SFT on our dataset, without additional RL optimization. -

Qwen2.5-VL-no-thinking: An ablated variant trained via SFT, where all chain-of-thought (<think>) reasoning annotations are removed from the input. This model isolates the contribution of explicit reasoning supervision to decision-making performance.

All Qwen variants in our experiments are based on the 7B architecture. All models are tested under consistent inference conditions, using the same evaluation protocol and task sets to ensure fair comparison.

#### 6.2 Evaluation Metrics

We evaluate agent performance mainly using the following five metrics:

Success Rate (SR): This metric measures the proportion of tasks that the agent completes successfully. A task is considered successful if the agent reaches the goal under task-specific spatial and behavioral constraints, such as reaching within 300 meters of the target and issuing a Finish action.

Goal Distance Error (GDE): GDE quantifies the Euclidean distance (in centimeters) between the agent’s final position and the assigned target. For multi-stage tasks, GDE is computed as the sum of distances to each subgoal, with special handling to account for missing or inaccurate midway targets.

Step-based Success weighted by Path Length (SSPL): SSPL evaluates the efficiency of successful episodes. It is defined as the ratio of the optimal number of steps to the actual steps taken, weighted by success. Specifically, for each task i,

sopti max(si,sopti )

SSPLi = Si ·

where Si indicates task success, sopti is the length of the shortest ground-truth action sequence, and si is the number of actions executed by the agent.

Steps: This metric reports the average number of discrete actions (e.g., MoveForward, TurnLeft) taken by the agent per episode, regardless of success or failure, reflecting the behavioral cost.

Timeout Rate (TR): Timeout Rate measures the proportion of episodes in which the agent exceeds a maximum step threshold (e.g., 32 steps) without completing the task. A high TR indicates frequent inefficiencies or failures to terminate appropriately.

#### 6.3 Experiment Analysis

We analyze the results presented in Table 2 and Table 3 to assess how different models perform across task types in both in-domain and out-of-domain settings.

Challenge difficulty. Across all models without fine-tuning, performance remains low on exploration, spatial-relational, and multi-stage tasks. For example, the success rate (SR) of GPT-4o is only 3.6% (Exploration), 10.2% (Dynamic Spatial-Semantic), and 2.7% (Multi-stage) on out-of-domain tasks. Similar patterns are observed for Gemini 2.5 Pro and Qwen2.5-VL-origin. These results confirm that EmbRACE-3K poses substantial challenges for zero-shot models, particularly in tasks requiring long-horizon planning and egocentric spatial reasoning.

Supervised and RL fine-tuning improves performance. Fine-tuning Qwen2.5-VL with our dataset leads to strong gains across all task types. In Table 2, the sft-rl variant achieves 30.9% SR on Exploration tasks and 42.4% SR on Spatial-Semantic tasks out-of-domain, both higher than GPT-4o and Gemini 2.5 Pro. Notably, its GDE drops from over 9978.3 to 1162.8 on Exploration, and from 7844.0 to 824.6 on Spatial-Semantic. Similar improvements are observed in Table 3: on Multi-stage tasks, SR improves from 0.0% (Qwen2.5-VL-origin) to 27.0%, and GDE reduces from 8788.9 to 1265.7. These gains show that environment-aligned supervision paired with RL reward shaping substantially improves success rates and path efficiency.

Reasoning annotation improves decision quality. Comparing sft-only and no-thinking models isolates the contribution of chain-of-thought reasoning. In Dynamic Spatial-Semantic (in-domain), SR improves from 27.1% (no-thinking) to 42.4%, and SSPL increases from 0.268 to 0.405. Similarly, in Multi-stage tasks, the SR gap between no-thinking (10.8%) and sft-only (18.9%) indicates more stable sequential behavior when decision steps are paired with language rationale. These results suggest that step-wise reasoning supervision helps maintain spatial grounding and task context.

Generalization remains a key challenge. The sft-only model performs well in-domain but shows large performance drops out-of-domain. For instance, its SR on Exploration drops from 71.4% to 22.8%, and on Multi-stage from 68.6% to 35.6%. In contrast, the RL-augmented model generalizes better: Exploration SR is 30.9%, and Multi-stage is 42.4%. This supports our hypothesis that trajectory-level reinforcement signals promote policy robustness in unseen scenes, where spatial layout and object configuration differ from training environments.

### 7 Conclusion

This work introduces EmbRACE-3K, a novel dataset and benchmark designed to address the limitations of current VLMs in embodied, interactive scenarios. Featuring diverse environments and multi-actions, EmbRACE-3K fosters research in dynamic, goal-oriented contexts within open environments. High-quality CoT annotations enhance agent actions by incorporating reasoning into spatial planning. This approach bridges the gap between instructional tasks and visual inputs, enabling more robust and logical decision-making. Benchmarking experiments reveal significant challenges in spatial reasoning, long-term planning, and causal understanding, underscoring the dataset’s value in advancing embodied reasoning. Notably, fine-tuning VLMs like Qwen2.5-VL-7B with EmbRACE-3K achieves superior performance compared to GPT-4o and Gemini 2.5 Pro. By enabling temporal generalization and integrating perception with language-guided behavior, EmbRACE-3K establishes a foundation for developing intelligent agents capable of real-world applications.

### References

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.
- [2] Anthropic. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/ introducing-claude-3-5-sonnet, 2024. Accessed: 2025-05-16.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [4] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V, 2025. Accessed: 2025-02-02.
- [5] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024.
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [7] DeepMind. Gemini 1.5: Unlocking multimodal understanding across millions of tokens. arXiv preprint arXiv:2403.05530, 2024.
- [8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.
- [10] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [12] Runyu Jiao, Alice Fasoli, Francesco Giuliari, Matteo Bortolon, Sergio Povoli, Guofeng Mei, Yiming Wang, and Fabio Poiesi. Free-form language-based robotic reasoning and grasping. arXiv preprint arXiv:2503.13082, 2025.
- [13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [14] Haowei Lin, Zihao Wang, Jianzhu Ma, and Yitao Liang. Mcu: A task-centric framework for open-ended agent evaluation in minecraft. arXiv preprint arXiv:2310.08367, 2023.
- [15] Changshu Liu, Shizhuo Dylan Zhang, Ali Reza Ibrahimzada, and Reyhaneh Jabbarvand. Codemind: A framework to challenge large language models for code reasoning. arXiv preprint arXiv:2402.09664, 2024.
- [16] Meta. Llama 3. 2024.
- [17] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [18] OpenAI. Gpt-4o. 2025.
- [19] Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. Teach: Task-driven embodied agents that chat. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 2017–2025, 2022.
- [20] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [21] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024.
- [22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [23] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020.

- [24] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [25] Karmesh Yadav, Jacob Krantz, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Jimmy Yang, Austin Wang, John Turner, Aaron Gokaslan, Vincent-Pierre Berges, Roozbeh Mootaghi, Oleksandr Maksymets, Angel X Chang, Manolis Savva, Alexander Clegg, Devendra Singh Chaplot, and Dhruv Batra. Habitat challenge 2023. https://aihabitat.org/challenge/2023/, 2023.
- [26] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [27] Jihan Yang, Runyu Ding, Ellis Brown, Xiaojuan Qi, and Saining Xie. V-irl: Grounding virtual intelligence in real life. In European conference on computer vision, pages 36–55. Springer, 2024.
- [28] Jingkang Yang, Yuhao Dong, Shuai Liu, Bo Li, Ziyue Wang, Haoran Tan, Chencheng Jiang, Jiamu Kang, Yuanhan Zhang, Kaiyang Zhou, et al. Octopus: Embodied vision-language programmer from environmental feedback. In European Conference on Computer Vision, pages 20–38. Springer, 2024.
- [29] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.
- [30] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025.
- [31] Samson Yu, Kelvin Lin, Anxing Xiao, Jiafei Duan, and Harold Soh. Octopi: Object property reasoning with large tactile-language models. arXiv preprint arXiv:2405.02794, 2024.
- [32] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics.
- [33] Fangwei Zhong, Kui Wu, Churan Wang, Hao Chen, Hai Ci, Zhoujun Li, and Yizhou Wang. Unrealzoo: Enriching photo-realistic virtual worlds for embodied ai. arXiv preprint arXiv:2412.20977, 2024.
- [34] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

