[Figure 1]

## Nav-R1: Reasoning and Navigation in Embodied Scenes

Qingxiang Liu1∗ Ting Huang1∗ Zeyu Zhang2∗† Hao Tang2‡ 1Shanghai University of Engineering Science 2Peking University

∗Equal contribution. †Project lead. ‡Corresponding author: bjdxtanghao@gmail.com.

“The division of labor between System 1 (fast) and System 2 (slow) is highly efficient: it minimizes effort and optimizes performance.” — Daniel Kahneman (Nobel Prize in Economics)

[Figure 2]

###### Task Type

Start position

bathtab

# arXiv:2509.10884v1[cs.RO]13Sep2025

[Figure 3]

[Figure 4]

Dialogue Reasoning

Viewpoint

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Target object

Planning

Navigation

[Figure 11]

Step 3 Step 2

Step 1: Step 2: Step 3: · · ·

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

sofa

|[Figure 16]<br><br>[Figure 17]|
|---|

TV

Step 1

bed

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Embodied Dialogue Embodied Reasoning Embodied Planning Embodied Navigation

[Figure 23]

Situation: Standing in the bathroom,

Human: I want to watch TV on the sofa.

Instruction: Navigate to the bedroom and

May I ask you some questions?

facing the shower.

What should I do?

find a bed.

[Figure 24]

|[Figure 25]|
|---|
|[Figure 26]|
|[Figure 27]|

[Figure 28]

Of course! I'm here to help.

Question: Can I walk straight to the

[Figure 29]

###### Nav-R1:

###### Nav-R1:

bathtub?

- Step 1. Turn left
- Step 2. Move forward
- Step 3. Turn right
- Step 4. …

- Step 1. Turn on the TV…
- Step 2. Walk to the sofa…
- Step 3. Sit on the sofa …

[Figure 30]

How many pillows are on the bed?

Nav-R1: No. You need to turn left to reach

[Figure 31]

I can see two pillows on the bed.

the bathtub.

Fig. 1. Nav-R1 is an embodied foundation model that integrates dialogue, reasoning, planning, and navigation capabilities to enable intelligent interaction and task execution in 3D environments.

Abstract—Embodied navigation requires agents to integrate perception, reasoning, and action for robust interaction in complex 3D environments. Existing approaches often suffer from incoherent and unstable reasoning traces that hinder generalization across diverse environments, and difficulty balancing long-horizon semantic reasoning with low-latency control for real-time navigation. To address these challenges, we propose Nav-R1, an embodied foundation model that unifies reasoning in embodied environments. We first construct Nav-CoT-110K, a large-scale dataset of step-by-step Chains-of-Thought (CoT) for embodied tasks, which enables cold-start initialization with structured reasoning. Building on this foundation, we design a GRPO-based reinforcement learning framework with three complementary rewards: format, understanding, and navigation, to improve structural adherence, semantic grounding, and path fidelity. Furthermore, we introduce a Fast-in-Slow reason-

ing paradigm, decoupling deliberate semantic reasoning from low-latency reactive control for efficient yet coherent navigation. Extensive evaluations on embodied AI benchmarks demonstrate that Nav-R1 consistently outperforms strong baselines, with over 8% average improvement in reasoning and navigation performance. Real-world deployment on a mobile robot further validates its robustness under limited onboard resources. Code: https://github.com/AIGeeksGroup/Nav-R1. Website: https://aigeeksgroup.github.io/Nav-R1.

I. INTRODUCTION

Embodied scene understanding is a central problem in embodied AI, robotics, and intelligent agents, requiring an agent to perceive, reason, and act within complex 3D environments [4], [29]. A robust understanding of embodied

navigation not only supports goal-directed tasks such as object search, instruction following, and trajectory planning, but also enables higher-level embodied interactions including dialogue, reasoning, and decision-making. Such capabilities are fundamental for service robots, augmented reality assistants, and intelligent embodied systems deployed in realworld environments [41], [42].

Recent advances in large vision language models (LVLMs) have extended the success of 2D perception into the 3D embodied domain, giving rise to unified frameworks that couple perception, language, and action. As illustrated in

- Fig. 1, these advances lay the foundation for tackling four embodied tasks such as embodied dialogue [10], [20], [15], embodied reasoning [15], [10], [12], embodied planning [13], [10], [21], and embodied navigation [4], [29], [54], [36], which are the central focus of this work.

Despite such progress, significant challenges remain. First, existing approaches often suffer from incoherent and unstable reasoning traces that fail to align with navigation instructions, leading to brittle generalization and semantically inconsistent output. Second, embodied navigation requires balancing long-horizon semantic reasoning with low-latency reactive control for real-time execution, a dual requirement that remains largely unaddressed in current methods.

Motivated by these limitations, there is a critical need for embodied foundation models that can jointly address semantic reasoning and embodied action, rather than treating them as separate problems. To this end, our work aims to develop a unified framework that balances long-horizon reasoning with real-time responsive control, thereby enabling robust generalization and reliable execution in diverse 3D environments.

To overcome these limitations, we introduce Nav-R1, an embodied foundation model that integrates reasoning, planning, dialogue, and navigation into a unified framework. Specifically, we construct a large-scale dataset Nav-CoT110K, synthesizing high-quality Chains-of-Thought (CoT) for embodied tasks by prompting a strong VLM with egocentric observations, instructions, and action options, followed by rule-based filtering for consistency. This dataset is used for a cold-start stage that equips Nav-R1 with structured reasoning and alignment of instructions. Building upon this initialization, we design a reinforcement learning stage with three complementary rewards: (i) a format reward ensuring structural adherence, (ii) an understanding reward capturing semantic correctness and visual grounding, and (iii) a navigation reward optimizing path fidelity and endpoint accuracy. To further address the tension between semantic fidelity and real-time control, we propose a Fast-in-Slow dual system reasoning scheme inspired by cognitive science [27]: a slow module aggregates long-term semantics from visual histories, while a fast module executes short-horizon actions with low latency, coordinated asynchronously for robust yet efficient navigation.

We evaluate Nav-R1 extensively on the embodied benchmarks. Nav-R1 consistently outperforms prior state-of-theart methods, achieving improvements in navigation success

TABLE I STATISTICS OF THE PUBLIC 3D-VL DATASETS THAT WE DRAW ON WHEN SYNTHESISING THE NAV-COT-110K DATASET. NS DENOTES THE SCENE NUMBER. Modality IS THE MODALITY WITHIN

INSTRUCTIONS, WHERE [V,L,P] DENOTE [vision,language, point]. NT DENOTES THE TASK NUMBER. DE,CE DENOTE THE DISCRETE AND CONTINUOUS ENVIRONMENTS.

Scenes Instruction Capability

Dataset

NT Env

MP3D HM3D NS ObjNav VLN Modality

R2R [4] ✓ ✗ 90 ✗ ✓ L 22K DE R2R-CE [29] ✓ ✗ 90 ✗ ✓ L 4.5K CE RxR-CE [30] ✓ ✗ 90 ✗ ✓ L - CE SOON [57] ✓ ✗ 90 ✓ ✗ L 30K DE OVON [52] ✗ ✓ 181 ✓ ✗ L 53K CE

Nav-CoT-110K (Ours) ✓ ✓ 342 ✓ ✓ V,L,P 110K CE

rates and reasoning accuracy. Beyond simulation, we deploy Nav-R1 on a WHEELTEC R550 mobile robot equipped with a Jetson Orin Nano, LiDAR, and RGB-D camera, demonstrating robust real-world performance under limited onboard computation by designing cloud-based inference. These results validate that Nav-R1 achieves a strong balance of reasoning capability, semantic grounding, and embodied real-time control.

Our contributions are summarized as follows:

- • We introduce Nav-R1, an embodied foundation model equipped with a GRPO-based reinforcement learning framework to enhance reasoning and navigation in 3D environments. Specifically, we design three complementary reward functions, format reward, understanding reward, and navigation reward, to improve structural adherence, semantic grounding, and path fidelity. In addition, we construct the large-scale Nav-CoT-110K dataset through a CoT data engine, which provides highquality step-by-step reasoning trajectories to bootstrap the model via cold-start initialization.
- • We propose a novel Fast-in-Slow reasoning paradigm that decouples long-horizon semantic reasoning from short-horizon reactive control. And this dual-system design ensures semantic coherence for planning while maintaining low-latency responses in dynamic environments.
- • We conduct comprehensive evaluations on both embodied AI benchmarks and real-world robot deployment. Nav-R1 achieves an average improvement of 8% compared to strong baselines across dialogue, reasoning, planning, and navigation tasks, and further demonstrates robust performance when deployed on a physical robot with limited on-board resources.

II. RELATED WORK

a) Embodied understanding: Embodied understanding seeks to equip agents with the ability to perceive, reason, and act in 3D environments by tightly integrating multimodal sensory data with linguistic instructions. Early work, limited to 2D visual abstractions, lacked the spatial expressiveness required for complex 3D reasoning. Recent advances address this gap by using large language models (LLMs) as

[Figure 32]

[Figure 33]

- : 2D image token

: Depth token

[Figure 34]

- : 3D point cloud token

Asynchronous Sampling 3D Scene

Egocentric Views

[Figure 35]

[Figure 36]

RGB Image

Depth Map

|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]| |
|---|---|
| |1 / n|

###### Dual system

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

: Language token

[Figure 46]

- : System 1 forward pass

- : System 2 forward pass

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Sample Images

Image Encoder

Depth Encoder

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

|[Figure 59]| |
|---|---|
| | |

[Figure 60]

Point Encoder

|Navigation Action<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>or or or|
|---|

[Figure 65]

Image Encoder

Fast system (System 1)

RGB Image

[Figure 66]

[Figure 67]

| | |
|---|---|
| | |

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

| |
|---|

|[Figure 72]| |
|---|---|
| | |

[Figure 73]

Depth Encoder

Depth Map

𝐿

###### LoRA

[Figure 74]

Blockm+1

Blockm+2

Blockm

Block1

Block2

Slow system (System 2)

[Figure 75]

· · ·

Instruction

|[Figure 76]<br><br>[Figure 77]<br><br>Reasoning: The best exploratory action from the current position is to turn left...<br><br>Language Response|
|---|

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

𝐿

Instruction: Walk across living room to kitchen bar. Stop at bar stools.

Text Encoder

System 2 System 1

[Figure 82]

Fig. 2. Architecture of Nav-R1. Nav-R1 designs a Fast-in-Slow reasoning paradigm that processes egocentric RGB-D views, scene point cloud, and language instructions. The slow system performs long-horizon semantic reasoning, while the fast system executes real-time navigation, enabling coherent reasoning and low-latency control in embodied environments.

the connective tissue between perception, grounding, and action planning [24], [23], [25]. LEO [21] exemplifies this trend, introducing an embodied generalist agent trained in two stages - 3D vision-language alignment and subsequent vision-language-action instruction tuning, thus achieving unified competence in captioning, question answering and embodied reasoning without task-specific sub-modules. Complementing LEO, GaussianVLM [15] adopts a scene-centric paradigm that embeds linguistic features directly into 3D Gaussian splats. Its dual sparsification mechanism distills dense scene representations into task-aware tokens, obviating external object detectors and enabling zero-shot generalization across embodied reasoning tasks. Together, these studies underscore that unified architectures and language-aligned 3D representations are crucial to robust scene comprehension and action-oriented cognition.

video-conditioned vision-language-action backbone, achieving real-time inference trajectories. Incoming generalist agents, OctoNav [13] introduces a hybrid training paradigm that integrates the chain-of-thought of Think-Before-Action with GRPO and RL online, enabling faithful compliance with free-form commands and continuous control. Most recently, MTU3D [58] bridges visual grounding and active exploration by treating unexplored regions as frontier queries within a unified vision-language-exploration objective; pre-training on one million trajectories lifts success rates by 14–23% on HM3D-OVON, GOAT-Bench, SG3D, and A-EQA while supporting language, category, and image goals alike. Collectively, these advances showcase the growing capacity of LVLMs to fuse perception, reasoning, and action for adaptive embodied navigation.

III. DATASETS

b) Embodied navigation: Embodied navigation tasks require agents to translate multimodal instructions into smooth, continuous motion through unstructured 3D scenes, which mainly include tasks such as object goal navigation (ObjectNav) and vision language navigation (VLN). Traditional pipelines rely on discrete topological graphs, which limit path flexibility. Recent research instead embraces end-to-end paradigms powered by large vision language models. VLN-R1 [36] couples reinforcement fine-tuning with a time-decayed reward, using GRPO-style training and the VLN-Ego dataset to shrink the instruction–action gap. Uni-NaVid [54] further unifies the navigation subtasks object search, instruction follow, and more - within a single

We build our framework on public 3D vision language datasets and the newly constructed Nav-CoT-110K dataset. Table I summarizes the statistics of all datasets considered in this work, including scene coverage, task types, instruction modalities, and environment settings.

A. Public Datasets

We draw on several widely used embodied AI benchmarks to ensure diversity and comparability. R2R [4] and its continuous extension R2R-CE [29] provide natural language navigation instructions in Matterport3D scenes. RxR-CE [30]

extends this setup to a multilingual setting with dense temporal grounding. For object-goal navigation, SOON [57] introduces category-conditioned search in indoor environments, while HM3D-OVON [52] further supports open-vocabulary object navigation under a zero-shot setting. Together, these benchmarks cover both instruction-following and objectcentric navigation tasks across discrete and continuous environments.

(1) Instruction Part

Nav-CoT

[Figure 83]

System Instruction

Imagine you are a robot designed for navigation tasks in 3D scenes. Your instruction is ‘Exit the bedroom, enter the bathroom, wait at the toilet.’

<think>…My instruction is to go to the bathroom. I saw the bathroom entrance in Image 3. My current location (Image 6) is in a corner of the room, facing a window. This is away from the bathroom entrance I previously saw. To get back on track, I need to turn away from this window and reorient myself towards the interior of the suite where the bathroom is located. Turning left will pan my view away from the corner and back towards the main living area, from where I can navigate towards the bathroom entrance shown in Image 3. Turning right would just face me more into the corner wall. Therefore, turning left is the correct action to find the path to the bathroom again.</think>

- (2) Vision Part

- (3) Action Part

- (4) Format Part

[Figure 84]

[Figure 85]

[Figure 86]

###### Text Prompt

History View

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Current View

You are provided with a sequence of egocentric observations from previous timesteps(Images 1-5) and one egocentric observation from the current timestep(Image 6).

Gemini 2.5 Pro

Images 1,2,3,4,5 Image 6

Large VisionLanguage Model

###### Action Options

[Figure 92]

Your task is to select the next action, and action options include:

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Move forward 25cm Turn left 30° Turn right 30° Stop

B. Synthetic Dataset

[Figure 97]

Output Format Based on the instruction and egocentric observations, reason step by step and provide your response. Output format: <think>…the reasoning content…</think> <action>...the next action…</action>. Strictly adhere to the output format and do not output any other information.

<action>turn left 30°</action>

Building upon these resources, we introduce Nav-CoT110K, a large-scale dataset consisting of 110K step-bystep Chain-of-Thought trajectories. Unlike prior datasets that primarily provide instructions and target locations, NavCoT-110K explicitly includes structured reasoning aligned with multimodal observations, thereby bridging perception, language, and action. This dataset serves as the foundation for the cold-start stage of Nav-R1, enabling it to acquire structured reasoning capabilities before reinforcement learning.

Fig. 3. CoT Data Engine. We construct the Nav-CoT dataset by defining navigation instructions, integrating egocentric visual inputs, providing action options and specifying the output format. These components are fed into Gemini 2.5 Pro, which generates step-by-step reasoning and action decisions aligned with navigation goals.

environment constraints, and instruction semantics, and to produce structured step-by-step CoT sequences. The outputs follow a standardized format, with reasoning enclosed in <think>...</think> tags and the chosen action in <action>...</action> tags, ensuring transparent alignment between observations, reasoning, and decisions.

IV. THE PROPOSED METHOD

- A. Overview

The proposed Nav-R1 framework is designed as an embodied foundation model that unifies multimodal perception, structured reasoning, and embodied control. As illustrated in Fig. 2, Nav-R1 adopts a Fast-in-Slow reasoning paradigm, where the slow system performs long-horizon semantic reasoning while the fast system ensures low-latency navigation. The framework follows a two-stage training pipeline: a CoT data engine is first used to construct the NavCoT-110K dataset, which provides high-quality step-by-step reasoning trajectories. Based on this dataset, a cold-start stage initializes the model’s reasoning ability, followed by a reinforcement learning stage with multi-dimensional rewards to refine semantic grounding and navigation fidelity.

- B. CoT Data Engine

Running this pipeline across diverse environments yields approximately 115K CoT examples, each consisting of a scene ID, navigation instruction, visual inputs, structured reasoning, and corresponding action. These raw outputs are then refined through a two-stage filtering pipeline: (i) rulebased checks to discard incomplete or logically inconsistent responses, and (ii) quality verification by cross-validating actions against feasible navigation paths. After refinement, the 110K examples form the Nav-CoT-110K dataset, which serves as the cold-start initialization corpus for Nav-R1, providing rich reasoning trajectories that tightly couple perception, instruction following, and navigation decision making.

C. Cold Start Stage

We introduce a CoT data engine designed to construct high-quality Chains of Thought (CoT) for embodied navigation and reasoning tasks. This engine harnesses the reasoning abilities of vision-language models (VLMs) to generate coherent step-by-step rationales that inform navigation decisions in complex 3D environments.

Although reinforcement learning has shown remarkable effectiveness in reasoning-intensive models such as DeepSeekR1 [12], directly applying RL to large 3D vision language models often leads to unstable optimization. In particular, the policy tends to generate semantically incoherent CoT sequences or produce actions that fail to align with navigation instructions, making it difficult for the model to converge from scratch.

As shown in Fig. 3, the process begins with egocentric visual observations extracted from 3D scenes, providing rich contextual views aligned with the agent’s perspective. In parallel, we incorporate navigation instructions drawn from standard embodied AI benchmarks - VLN tasks rely on R2R [4], R2R-CE [29], and RxR-CE [30] datasets, while ObjectNav tasks use instructions from SOON [57] and OVON [52].

To address this issue, we adopt a cold-start stage based on supervised fine-tuning. Specifically, the NavCoT-110K dataset generated by our CoT data engine is employed to initialize Nav-R1. This supervised training step equips the policy with essential capabilities to produce structured reasoning sequences in the format of <think>...</think><action>...</action> and ground them to corresponding navigation actions.

To guide reasoning, we design a composite prompt containing four essential components: (1) navigation instruction, (2) egocentric visual input, (3) set of feasible actions at each step, and (4) explicit formatting requirements. This prompt directs Gemini 2.5 Pro [43] to reason over spatial relations,

By bootstrapping the model with coherent reasoning and action patterns, the cold-start stage stabilizes subsequent RL optimization and provides a smooth transition to the

###### Input Data Reward Function

###### Understanding Reward Navigation Reward

[Figure 98]

[Figure 99]

[Figure 100]

###### Question:

###### Instruction:

How many white chairs in the room?

Walk straight past the glass table with white chairs then turn left and go into the room on the left. Wait at the entrance.

Text Image

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

[Figure 114]

[Figure 115]

Policy Model

[Figure 116]

𝜊 𝜊 ··· 𝜊

RGB Image Depth Image

Historical Observations

Current Observation

Answer Reward

Endpoint & Path Reward

Calculate the difference between the predicted answer and the true answer.

[Figure 117]

𝒑 Predicted end point 𝒑 GT end point 𝑫𝑭(⋅) Path distance 𝑻 Predicted trajectory 𝑻 GT trajectory 𝒌 Decay coefficient

[Figure 118]

Reference Model

1, 𝑖𝑓 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝑎𝑛𝑠𝑤𝑒𝑟 = 𝑔𝑟𝑜𝑢𝑛𝑑 𝑡𝑟𝑢𝑡ℎ 0, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒

𝑹ans =

Semantic Reward

KL Divergence

𝑹sem = CLIPScore(𝑖𝑚𝑎𝑔𝑒,𝑎𝑛𝑠𝑤𝑒𝑟)

[Figure 119]

[Figure 120]

𝐴 𝐴

𝑟 𝑟

[Figure 121]

𝑖𝑚𝑎𝑔𝑒 Current perspective image 𝑎𝑛𝑠𝑤𝑒𝑟 Generated answers

𝑹path=exp( − 𝒌(𝑫𝑭(𝑻, 𝑻)) 𝑹end=exp( − 𝒌 𝒑 − 𝒑 𝟐) 𝑹navigation = 𝑹path + 𝑹end

···

···

𝐴 GRPO

𝑹understanding = 𝑹ans + 𝑹sem

𝑟

###### Format Reward

[Figure 122]

Group Computation

1, 𝑖𝑓 𝑎𝑛𝑠𝑤𝑒𝑟 𝑎𝑑ℎ𝑒𝑟𝑒𝑠 𝑡𝑜 𝑓𝑜𝑟𝑚𝑎𝑡 0, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒

Output format: <think>…the reasoning content…</think><answer>… the final answer…</answer><action>… the next action…</action>

[Figure 123]

𝑹format =

Fig. 4. The pipeline of RL Policy. The policy model generates N outputs from text-image input. Then understanding reward (answer correctness and semantic alignment), navigation reward (path fidelity and endpoint accuracy), and format reward (structure adherence) are computed, grouped, and combined with a KL term to a frozen reference model to update the policy.

reinforcement learning stage, where multi-dimensional rewards further refine semantic understanding, path fidelity, and structural adherence.

responses are then evaluated using the following reward functions.

b) Format reward: To guarantee structural consistency, format reward RFormat verifies whether each output strictly adheres to the reasoning-decision template: <think>...</think><answer>...</answer> or <think>...</think><action>...</action>.

- D. Reinforcement Learning

Group Relative Policy Optimization (GRPO) [40] has recently demonstrated strong effectiveness in reasoningintensive tasks, such as DeepSeek R1 [12]. Its core principle is to refine the policy through group-based feedback: multiple candidate responses are sampled from the current policy, scored by task-specific reward functions, and updated according to their relative advantages.

Formally,

1, if output adheres to format 0, otherwise

RFormat =

. (1)

This constraint not only ensures machine-parseable outputs but also disentangles reasoning from final predictions.

As depicted in Fig. 4, extending this framework to embodied 3D vision-language tasks, Nav-R1 introduces three complementary rewards that jointly supervise training: Format Reward, which enforces the structural validity of the outputs; Understanding Reward, which ensures semantic correctness and visual grounding in 3D scene reasoning; and Navigation Reward, which evaluates path fidelity and goal-reaching accuracy in navigation. Together, these rewards provide multidimensional feedback that balances linguistic structure, semantic understanding, and action execution.

c) Understanding reward: The understanding reward Runderstanding assesses whether the model demonstrates genuine comprehension of the 3D scene. It consists of two components: Answer Reward and Semantic Reward. The answer reward measures exact correctness against ground truth:

1, if predicted answer equals ground truth 0, otherwise

Rans =

, (2)

while the semantic reward measures alignment between the generated answer aˆ and the paired RGB-D image I:

a) Policy samples: For an input state (x,q), where x encodes multi-modal scene representations and q denotes the instruction or question, Nav-R1 generates N candidate responses {o1,o2,··· ,oN} from policy πθ. Each candidate corresponds to either an answer prediction for scene understanding or an action prediction for navigation. These

Rsem = CLIPScore(I,aˆ). (3) The overall understanding reward is defined as:

Runderstanding = Rans +Rsem. (4)

This design prevents both factual errors and semantically irrelevant outputs.

d) Navigation reward: For navigation tasks, the navigation Reward Rnavigation evaluates whether the agent follows the instruction and successfully reaches the target. It integrates two components: a path reward, which measures the fidelity of the trajectory with respect to the reference path Tˆ, and an endpoint reward, which enforces the precision of the final location pˆ.

Given a predicted trajectory T and a ground-truth trajectory Tˆ, the path reward is defined as:

Rpath = exp −kDF(T,Tˆ) , (5)

where DF(·) denotes a trajectory distance metric and k is a decay coefficient.

Furthermore, the endpoint reward penalizes deviations between the predicted endpoint p and the ground-truth endpoint pˆ:

Rend = exp −k∥pˆ− p∥2 . (6) The final navigation reward is the combination of both:

Rnavigation = Rpath +Rend. (7)

This formulation ensures that both trajectory-level alignment and goal-reaching ability are optimized.

e) Policy optimization: Inspired by Group Relative Policy Optimization (GRPO) [40], we sample multiple candidate responses {o1,o2,··· ,oN} from the current policy πθ, and compute their corresponding rewards r = {r1,r2,··· ,rN} from the above functions. Each reward is normalized to compute the relative advantage:

ri −mean(r) std(r)

Aˆi =

, (8)

where Aˆi denotes the advantage of the i-th response. The policy is then updated by maximizing the clipped GRPO objective with KL regularization:

JGRPO(θ) =Ec

1 G

clip

G

πθ(oi|q) πθold(oi|q)

### ∑

Aˆi,

min

i=1

πθ(oi|q) πθold(oi|q)

,1−ε,1+ε A ˆi

−β ·DKL(πθ∥πref) .

(9)

- E. Fast-in-Slow Reasoning

Inspired by dual system theories of human cognition [27], we propose a Fast-in-Slow paradigm for Nav-R1, which tightly couples deliberate semantic reasoning with rapid action execution. As shown in Fig. 2, this design addresses the tension between accurate long-horizon planning and lowlatency control in dynamic embodied environments.

###### TABLE II OBJECT GOAL NAVIGATION RESULTS ON HM3D-OVON [52]. ∗ DENOTES ZERO-SHOT METHODS.

Val-Seen Val-Seen-Synonyms Val-Unseen

Method

###### SR↑ SPL↑ SR↑ SPL↑ SR↑ SPL↑

BC [35] 11.1 4.5 9.9 3.8 5.4 1.9 DAgger [38] 11.1 4.5 9.9 3.8 5.4 1.9 RL [39] 18.1 9.4 15.0 7.4 10.2 4.7 BCRL [47] 39.2 18.7 27.8 11.7 18.6 7.5 DAgRL [6] 41.3 21.2 29.4 14.4 18.3 7.9 VLFM∗ [51] 35.2 18.6 32.4 17.3 35.2 19.6 DAgRL+OD [52] 38.5 21.1 39.0 21.4 37.1 19.8 Uni-NaVid∗ [54] 41.3 21.1 43.9 21.8 39.5 19.8 MTU3D∗ [58] 55.0 23.6 45.0 14.7 40.8 12.1

Nav-R1 (Ours) 58.4 26.3 48.1 23.1 42.2 20.1

- a) Slow reasoning: The slow system (System 2) oper-

ates at a lower frequency and processes multimodal observations, including egocentric RGB-D frames and language instructions. It aggregates the historical context into compact memory states and outputs latent features ht that encode scene semantics, temporal dependencies, and global navigation goals. These structured features provide high-level guidance to ensure coherent decision-making. In practice, the slow system aggregates visual history into compact memory states, enabling Nav-R1 to maintain semantic consistency at the scene level while avoiding excessive computation.

- b) Fast reasoning: The fast system (System 1) runs at

a higher frequency and ensures real-time responsiveness in dynamic environments. Instead of independently reasoning, it reuses the final transformer blocks of Nav-R1 to inherit pretrained knowledge from System 2 while keeping its computation lightweight. At each step, the fast system fuses high-frequency egocentric multimodal inputs, including RGB frames, depth maps, and point cloud tokens (ot+1,...,ot+H), with the latent feature ht from the slow system to predict a short-horizon sequence of actions:

{at+1,...,at+H} = πfast(ot+1:t+H,ht), (10)

where πfast denotes the high-frequency policy model that integrates visual, depth, and 3D geometric cues for real-time control.

c) Asynchronous coordination: To balance efficiency and accuracy, we design an asynchronous update mechanism with a frequency ratio of 1 : n between the slow and fast systems. An update from the slow system provides latent guidance for n consecutive steps of fast execution. This decoupling strategy ensures that global semantics remain stable, while local control is executed with low latency. Empirically, we find that n ≈ 3 achieves the best balance between semantic fidelity and responsiveness in embodied navigation tasks, yielding both robust long-horizon reasoning and efficient real-time control.

V. EXPERIMENTS A. Benchmarks and Metrics

a) Benchmarks: To thoroughly evaluate Nav-R1, we adopt a diverse set of embodied AI benchmarks that span

TABLE III COMPARISON WITH STATE-OF-THE-ART METHODS ON THE VAL-UNSEEN SPLIT OF R2R-CE [4] AND RXR-CE [30]. ∗ INDICATES METHODS USING THE WAYPOINT PREDICTOR FROM [16]. NAV-R1 OUTPERFORMS ALL METHODS THAT DO NOT RELY ON SIMULATOR PRE-TRAINED WAYPOINT PREDICTORS, EVEN WHEN THOSE METHODS LEVERAGE ADDITIONAL INPUTS SUCH AS DEPTH, PANORAMIC VIEWS, AND ODOMETRY.

Observation R2R-CE Val-Unseen RxR-CE Val-Unseen S.RGB Pano. Depth Odom. NE↓ OS↑ SR↑ SPL↑ NE↓ SR↑ SPL↑ nDTW↑

Method

CMA∗ [16] ✗ ✓ ✓ ✓ 6.20 52.0 41.0 36.0 8.76 26.5 22.1 47.0 Sim2Sim∗ [28] ✗ ✓ ✓ ✓ 6.07 52.0 43.0 36.0 - - - GridMM∗ [49] ✗ ✓ ✓ ✓ 5.11 61.0 49.0 41.0 - - - Ego2-Map∗ [17] ✗ ✓ ✓ ✓ 5.54 56.0 47.0 41.0 - - - DreamWalker∗ [45] ✗ ✓ ✓ ✓ 5.53 59.0 49.0 44.0 - - - Reborn∗ [2] ✗ ✓ ✓ ✓ 5.40 57.0 50.0 46.0 5.98 48.6 42.0 63.3 ETPNav∗ [1] ✗ ✓ ✓ ✓ 4.71 65.0 57.0 49.0 5.64 54.7 44.8 61.9 HNR∗ [48] ✗ ✓ ✓ ✓ 4.42 67.0 61.0 51.0 5.50 56.3 46.7 63.5

AG-CMTP [8] ✗ ✓ ✓ ✓ 7.90 39.0 23.0 19.0 - - - R2R-CMTP [8] ✗ ✓ ✓ ✓ 7.90 38.0 26.0 22.0 - - - InstructNav [32] ✗ ✓ ✓ ✓ 6.89 - 31.0 24.0 - - - LAW [37] ✓ ✗ ✓ ✓ 6.83 44.0 35.0 31.0 10.90 8.0 8.0 38.0 CM2 [14] ✓ ✗ ✓ ✓ 7.02 41.0 34.0 27.0 - - - WS-MGMap [9] ✓ ✗ ✓ ✓ 6.28 47.0 38.0 34.0 - - - AO-Planner [7] ✗ ✓ ✓ ✗ 5.55 59.0 47.0 33.0 7.06 43.3 30.5 50.1 Seq2Seq [29] ✓ ✗ ✓ ✗ 7.77 37.0 25.0 22.0 12.10 13.9 11.9 30.8 CMA [29] ✓ ✗ ✓ ✗ 7.37 40.0 32.0 30.0 - - - NaVid [55] ✓ ✗ ✗ ✗ 5.47 49.0 37.0 35.0 - - - Uni-NaVid [54] ✓ ✗ ✗ ✗ 5.58 53.5 47.0 42.7 6.24 48.7 40.9 NaVILA [11] ✓ ✗ ✗ ✗ 5.22 62.5 54.0 49.0 6.77 49.3 44.0 58.8 VLN-R1 [36] ✓ ✗ ✗ ✗ 7.00 41.2 30.2 21.8 9.10 22.7 17.6 OctoNav [13] ✓ ✗ ✗ ✗ - 42.9 37.1 33.6 - - - StreamVLN [50] ✓ ✗ ✗ ✗ 4.98 64.2 56.9 51.9 6.22 52.9 46.0 61.9 CorrectNav [53] ✓ ✗ ✗ ✗ 4.24 67.5 65.1 62.3 4.09 69.3 63.3 75.2

Nav-R1 (Ours) ✓ ✗ ✓ ✗ 3.86 74.1 72.5 68.8 3.98 71.3 66.3 79.4

navigation, dialogue, reasoning, and planning tasks. For embodied navigation, we benchmark on R2R-CE [29] and RxR-CE [30], where agents navigate to goal locations in unseen environments, as well as HM3D for standard object goal navigation. To further test open-vocabulary generalization, we adopt HM3D-OVON [52], which introduces novel categories under a zero-shot setting. Embodied dialogue and planning are evaluated on 3D-LLM [18], which requires generating natural responses and coherent multi-step action plans. Embodied reasoning is assessed on SQA3D [33], which involves spatial question answering over complex 3D scenes.

b) Metrics: For embodied navigation, we follow standard metrics [4] including the navigation error (NE), success rate (SR), oracle success rate (OS), success weighted by path length (SPL) [3], and normalized dynamic time warping (nDTW) [26]. For embodied dialogue, planning, and reasoning tasks, we report widely used language generation metrics such as CIDEr (C) [44], BLEU-4 (B-4) [34], METEOR (M) [5], and ROUGE-L (R) [31].

- B. Implementation Details

a) Data synthesis: We first construct the Nav-CoT110K dataset using our CoT data engine. The instructions are sampled from R2R [4], R2R-CE [29], RxR-CE [30], SOON [57], and HM3D-OVON [52]. For each scene, we

provide RGB-D egocentric input, candidate action sets, and explicit output formatting to Gemini 2.5 Pro [43], which generates reasoning traces in <think> tags and the corresponding actions in <action> tags or answers in <answer> tags. A two-stage filtering process is applied: (i) rule-based checks discard incomplete or logically inconsistent responses, and (ii) trajectory verification ensures action feasibility against ground-truth paths. After filtering, 110K high-quality trajectories remain and are used for cold-start training.

- b) Cold-start initialization: We initialize Nav-R1 from

the pre-trained 3D-R1 model [23], which already provides strong 3D reasoning and vision-language alignment. On top of this initialization, we perform supervised fine-tuning (SFT) on Nav-CoT-110K for 2 epochs with a batch size of 8. The AdamW optimizer is used with weight decay 0.01, and a cosine annealing learning rate schedule decays from 10−4 to 10−5. This stage equips Nav-R1 with the ability to generate coherent reasoning-action sequences of the form <think>...</think><action>...</action>, ensuring structural adherence and semantic grounding before reinforcement learning.

- c) Reinforcement learning: After cold-start initializa-

tion, we fine-tune the model with Group Relative Policy Optimization (GRPO) [40]. For each input, multiple responses are sampled and scored with three complementary rewards:

###### TABLE IV EMBODIED DIALOGUE AND PLANNING RESULTS ON 3D-LLM [18]. EMBODIED REASONING RESULTS ON SQA3D [33].

Embodied dialogue Embodied planning Embodied reasoning C↑ B-4↑ M↑ R↑ C↑ B-4↑ M↑ R↑ C↑ B-4↑ M↑ R↑

Method

LL3DA [10] 190.01 23.95 23.50 40.61 128.80 12.95 17.05 39.25 - - - Spatial 3D-LLM [46] - - - - 195.92 14.65 18.95 36.93 - - - LSceneLLM [56] 104.98 - 21.26 36.00 214.63 - 21.05 47.05 - - - LEO [22] - - - - - - - - 124.70 9.40 25.50 48.40 3D-R1 [23] 280.34 39.45 66.89 55.34 230.50 25.45 48.34 55.67 138.67 23.56 35.45 60.02

###### Nav-R1 (Ours) 281.20 39.34 67.53 55.12 230.52 25.98 47.11 56.23 139.98 23.20 36.15 59.50

[Figure 124]

TABLE V QUANTITATIVE RESULTS OF REAL-WORLD EXPERIMENTS ACROSS THREE DISTINCT INDOOR ENVIRONMENTS. WE REPORT NAVIGATION ERROR (NE) AND SUCCESS RATE (SR) FOR NAV-R1 AND FOUR BASELINES. NAV-R1 CONSISTENTLY OUTPERFORMS ALL COMPETING METHODS.

###### (a) Hardware Platform

###### (b) Deployment Process

On-board PC

[Figure 125]

Visual Input

LiDAR

Jetson Orin Nano

[Figure 126]

M10P Astra Pro

Camera

###### RGB & Depth

[Figure 127]

STM32

Camera

On-board PC

Task Description Egocentric Views

Microcontroller

[Figure 128]

Embodied Foundation Model

Meeting Room Lounge Corridor NE↓ SR↑ NE↓ SR↑ NE↓ SR↑

Reasoning Navigation Decisions

Method

Speed PWM signals

On-board PC STM32 Motors

NaVILA [11] 2.06 0.45 2.21 0.45 1.97 0.50 NaVid [55] 1.88 0.55 2.22 0.50 1.94 0.55 Uni-NaVid [54] 1.76 0.64 2.13 0.64 1.87 0.65 MTU3D [58] 1.64 0.73 1.98 0.81 1.62 0.70

Fig. 5. Real-world robot setup and deployment pipeline. (a) Hardware platform: the WHEELTEC R550 robot equipped with Jetson Orin Nano (on-board PC), M10P LiDAR for mapping, Astra Pro RGB-D camera for perception, and STM32 microcontroller for motor control. (b) Deployment process: egocentric visual inputs are transmitted to the embodied foundation model Nav-R1, which performs reasoning and navigation. The decisions are then sent back to the on-board PC and converted into low-level motor commands by the STM32 controller.

Nav-R1 (Ours) 1.23 1.03 0.98 1.12 1.24 1.02

(i) a format reward enforcing structural validity, (ii) an understanding reward that combines exact match correctness with CLIP-based semantic alignment, and (iii) a navigation reward measuring trajectory fidelity and endpoint accuracy. RL training runs for 2 epochs with batch size 12, fixed learning rate 10−5, and a KL penalty β = 0.02 against the frozen SFT policy.

- c) Embodied planning: For embodied planning, the results in Table IV indicate that Nav-R1 performs on par with prior methods, generating coherent multi-step action sequences. This confirms that our Fast-in-Slow design and GRPO training preserve planning skills while primarily optimizing navigation.
- d) Embodied navigation: As shown in Table III and Table II, Nav-R1 consistently outperforms prior methods on both instruction-following and object-goal navigation benchmarks. It achieves higher success rates and trajectory efficiency while reducing navigation errors, demonstrating superior generalization across unseen environments.

d) Parameter efficient tuning: To reduce training cost, we adopt parameter-efficient fine-tuning by injecting LoRA adapters [19] into the last 8 transformer blocks of the backbone. Each adapter is configured with rank r = 6 and scaling factor α = 8, introducing ∼12M trainable parameters. In total, about 142M parameters are updated, reducing trainable parameters by ∼98% compared to full fine-tuning. All experiments are conducted on 4×NVIDIA H20 GPUs.

D. Real World Evaluation

- C. Main Results

a) Robot type settings: As shown in Fig. 5, we use the WHEELTEC R550 as the mobile platform for realworld evaluation. The robot is equipped with a Jetson Orin Nano as the on-board computing unit, an M10P LiDAR for environmental mapping, and an Astra Pro camera for RGB-D perception. Considering the limited edge computing resources of the platform, the embodied foundation model Nav-R1 is deployed on a cloud server rather than running locally. The system operates in a closed loop manner: the robot transmits egocentric RGB inputs to the cloud, where NavR1 performs reasoning and generates navigation decisions. These commands are then transmitted back to the on-board

- a) Embodied dialogue: Table IV shows that Nav-R1

maintains strong dialogue capability, achieving results close to 3D-R1 while outperforming previous baselines. This indicates that incorporating navigation-oriented reasoning does not weaken interaction quality.

- b) Embodied reasoning: For embodied reasoning, as

shown in Table IV, Nav-R1 performs comparably to 3D-R1. This is consistent with our design choice, as we do not train additional understanding modules but instead preserve the same scene reasoning ability while prioritizing navigation improvements.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

###### Meeting Room

###### Lounge

###### Corridor

[Figure 135]

[Figure 136]

[Figure 137]

Go to the chair with wooden legs on the left side.

Head straight, then turn left when you reach the couch, and stop in front of it.

Go to the black chair on your left and pause, then move forward to the front-right and stop at the blue umbrella.

|[Figure 138]<br><br>[Figure 139]<br><br>|
|---|

|[Figure 140]<br><br>[Figure 141]<br><br>|
|---|

|[Figure 142]<br><br>[Figure 143]<br><br>|
|---|

BEVTrajectoryEgo-Video

BEVTrajectoryEgo-Video

BEVTrajectoryEgo-Video

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

···

###### ··· ···

Fig. 6. Qualitative results from the real-world deployment of Nav-R1. We evaluate the agent in three indoor scenarios: meeting room, lounge, and corridor. Each scene illustrates the BEV trajectory and ego-centric video frames, showing the model’s ability to generalize to diverse layouts and object configurations in real-world environments.

TABLE VI AVERAGE INFERENCE LATENCY COMPARISON. COMPARISON OF AVERAGE PER-FRAME INFERENCE LATENCY (MS) FOR NAVID, UNI-NAVID, AND NAV-R1 ON JETSON ORIN NANO AND A REMOTE SERVER.

system, where an STM32 microcontroller converts them into PWM signals that directly control the robot’s motors.

- b) Scene setup and task types: To assess generalizabil-

ity under various real-world conditions, we evaluate NavR1 in three distinct indoor scenes: a meeting room, a lounge, and a corridor, each characterized by unique spatial layouts and object distributions. Across these settings, the robot is instructed to perform navigation-oriented tasks that vary in difficulty, ranging from short-horizon paths with clear line of sight to long-horizon trajectories involving clutter, obstacles, and occlusions. This setup enables a comprehensive assessment of the robustness of Nav-R1 in handling heterogeneous layouts and task complexities.

- c) Quantitative real-world evaluation: To quantify per-

formance, we compare Nav-R1 against previous navigation models including NaVILA [11], NaVid [55], UniNaVid [54], and MTU3D [58]. Each model is tested in the meeting room, lounge, and corridor with both simple and complex instructions. As summarized in Table V, NavR1 consistently achieves the best results, with significantly reduced navigation error (NE) and higher success rate (SR) across all three environments.

- d) Qualitative real-world evaluation: As shown in

Method On-board (ms) Server (ms)

NaVid [55] 320 85 Uni-NaVid [54] 410 90

Nav-R1 (Ours) – 95

server-side execution reduces latency to below 100ms. NavR1 achieves comparable efficiency with ∼95ms latency on the server, which is only slightly slower than NaVid and UniNaVid due to its dual-system reasoning overhead, yet this marginal gap does not affect real-time embodied navigation.

To ensure stable transmission, all experiments adopt a high-speed WiFi 6E (802.11ax, 6GHz band) network with 1.2Gbps peak bandwidth and <10ms access latency, covering the entire 200m2 indoor test area.

VII. CONCLUSION

In this paper, we presented Nav-R1, an embodied foundation model designed to enhance both reasoning coherence and real-time navigation. To overcome the instability of reasoning traces and the difficulty of balancing long-horizon semantics with real-time responsiveness, we introduced the large-scale Nav-CoT-110K dataset for cold-start initialization, a GRPO-based reinforcement learning framework with format, understanding, and navigation rewards, and a Fastin-Slow dual-system paradigm that decouples semantic reasoning from reactive control. Comprehensive experiments on VLN, ObjectNav, embodied dialogue, planning, and reasoning benchmarks show that Nav-R1 achieves consistent improvements in navigation success, trajectory fidelity, and reasoning coherence, while maintaining dialogue and planning performance on par with 3D-R1. Moreover, real-world deployment on a Jetson Orin Nano-powered mobile robot further validates its robustness under limited edge resources, with cloud-assisted inference enabling real-time closed-loop control.

Fig. 6, Nav-R1 exhibits coherent trajectories across meeting room, lounge, and corridor scenes. It reliably reaches diverse targets such as chairs, sofas, and umbrellas, demonstrating robustness to clutter, narrow passages, and long-horizon paths in real-world settings.

VI. TEST-TIME EFFICIENCY

Test-time efficiency is critical for embodied models in practical robotics. As illustrated in Fig. 5, the on-board Jetson Orin Nano faces strict resource limits, making largescale inference prohibitively slow. To address this, Nav-R1 adopts a cloud-assisted design, where egocentric inputs are streamed to a remote server for reasoning, and only compact navigation commands are returned for execution.

We benchmark NaVid [55] and Uni-NaVid [54] on both on-board and server inference. As shown in Table VI, both baselines incur substantial delays on the Orin Nano, while

APPENDIX VIII. ABLATION STUDY

a) Dual-system vs single-system: We compare the full dual-system Nav-R1 with single-system variants that only retain either the slow semantic reasoning system or the fast reactive control system. As shown in Table VII, both variants perform worse: the slow-only version struggles with realtime execution, while the fast-only version fails to maintain global semantic consistency. The dual-system achieves the best trade-off, confirming the effectiveness of asynchronous coordination.

TABLE VII ABLATION ON DUAL-SYSTEM DESIGN. EVALUATION ON R2R-CE VAL-UNSEEN.

Method NE↓ OS↑ SR↑ SPL↑

Slow-only 5.12 68.3 61.2 54.3 Fast-only 5.47 65.1 58.7 50.1

Dual-system (Ours) 3.86 74.1 72.5 68.8

b) Reward decomposition: To verify the effectiveness of the proposed reward design, we ablate the three rewards in RL training: format reward RFormat, understanding reward RUnderstanding, and navigation reward RNavigation. Table VIII shows that removing any reward leads to performance degradation. Without RFormat, the model often generates unstructured outputs. Removing RUnderstanding reduces semantic grounding, while dropping RNavigation severely hurts trajectory fidelity. The full combination achieves the best results, demonstrating that the three rewards are complementary.

TABLE VIII REWARD DECOMPOSITION ON HM3D-OVON VAL-UNSEEN. ✓DENOTES INCLUSION OF THE REWARD.

RFormat RUnderstanding RNavigation SR↑ SPL↑

- ✗ ✗ ✗ 34.5 15.2

✓ ✗ ✗ 36.1 16.0

- ✗ ✓ ✗ 37.0 16.7
- ✗ ✗ ✓ 36.5 15.9

✓ ✓ ✗ 39.4 18.2 ✓ ✗ ✓ 38.7 17.5

- ✗ ✓ ✓ 39.9 18.7

###### ✓ ✓ ✓ 42.2 20.1

c) Hyper-parameters: We further study the sensitivity of Nav-R1 to hyper-parameters on the RxR-CE Val-Unseen split. Tables IX report the ablation result. For the KL penalty β, too small values cause divergence from the reference policy, while too large values limit exploration. The best trade-off is achieved at β = 0.02.

IX. LIMITATION AND FUTURE WORK

Although Nav-R1 achieves strong results, it still has several limitations. The Nav-CoT-110K dataset, though large, is mainly synthesized from existing benchmarks and does not

TABLE IX ABLATION ON KL PENALTY β. EVALUATION ON RXR-CE VAL-UNSEEN.

β NE↓ SR↑ SPL↑ nDTW↑ 0.005 5.12 64.3 59.5 70.1

- 0.01 4.56 66.3 60.7 72.9

- 0.02 (Ours) 3.98 71.3 66.3 79.4

- 0.03 4.23 68.9 64.2 75.1 0.05 4.36 67.8 62.4 74.2

fully capture real-world complexity. Our model also relies on RGB-D and language inputs, without integrating richer modalities such as audio or tactile signals. Moreover, current deployment depends on cloud inference, limiting real-time scalability on edge devices. Future work will explore expanding data coverage, incorporating multimodal perception, improving efficiency for on-board deployment, and extending to longer-horizon and more diverse embodied tasks.

X. VISUALIZATION

- a) CoT example: To better illustrate how structured

reasoning traces are generated, Fig. 7 visualizes a representative Nav-CoT-110K example. It shows how natural language instructions, egocentric observations, and candidate actions are transformed into step-by-step reasoning content and navigation decisions, highlighting the role of CoT supervision in stabilizing model training.

- b) Results visualization: To further illustrate the real-

world performance of Nav-R1, we provide qualitative visualizations across multiple embodied tasks and diverse indoor environments. As shown in Fig. 8–10, we first evaluate navigation-oriented behaviors in three distinct scenes. Each example depicts the natural language instruction, egocentric RGB observations, depth maps, and LiDAR-based top-down maps, complemented with third-person views and BEV trajectories for reference.

Beyond navigation, we further demonstrate the versatility of Nav-R1 in dialogue, reasoning, and planning tasks. Fig. 11–13 present real-world qualitative results on embodied dialogue, reasoning, and planning, respectively. These visualizations highlight that Nav-R1 not only executes goaldirected trajectories robustly, but also maintains coherent interaction, safe reasoning, and multi-step planning capabilities under complex real-world layouts.

In addition to real-world deployment, we also provide benchmark visualizations on simulation datasets. Fig. 14 and Fig. 15 highlight the VLN-CE R2R benchmark, where Nav-R1 successfully grounds long-horizon instructions into coherent navigation trajectories. Similarly, Fig. 16 presents results on HM3D ObjectNav, showing that the agent can robustly explore large-scale 3D layouts and accurately localize target objects. These benchmark results further confirm that the proposed model generalizes across navigation paradigms.

#### Prompt

[Figure 174]

Instruction Part

Imagine you are a robot designed for navigation tasks in 3D scenes. Your instruction is ‘Walk out of the door with the round mirror towards a dresser, and walk into the bathroom.’.

[Figure 175]

Vision Part

You are provided with a sequence of egocentric observations from previous timesteps(Images 1-5) and one egocentric observation from the current timestep(Image 6).

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

Action Part

Your task is to select the next action, and action options include 'move forward 25cm', 'turn left 30°', 'turn right 30°' and 'stop'.

[Figure 189]

###### Format Part

Based on the instruction and egocentric observations, reason step by step and provide your response. Output format: <think>…the reasoning content…</think> <action>...the next action…</action>. Strictly adhere to the output format and do not output any other information.

#### Response

[Figure 190]

Reasoning Content <think>

- Step 1: Analyze the user instruction. The instruction is a sequence of two main goals: "Walk out of the door with the round mirror towards a dresser" and then "walk into the bathroom."
- Step 2: Review the image history to understand the progress. Image 3, 4, and 5 show the interior of a room with tufted walls and a round mirror on the door. This is the room I needed to exit. Image 1 shows the main entrance of the suite, with a sunburst mirror and a dark dresser to the left. This is the target dresser mentioned in the instruction. The sequence of images implies that I have already exited the room with the round mirror and have moved towards the dresser.
- Step 3: Analyze the current observation (Image 6). The current view is directly facing an open doorway. Inside the doorway, a toilet is visible, indicating this is the bathroom. To the right of the doorway, the sunburst mirror and the dresser are visible, confirming I have successfully navigated "towards a dresser". The current position is aligned with the entrance to the bathroom.
- Step 4: Determine the next action based on the instruction and current state. The first part of the instruction ("Walk out of the door...towards a dresser") is complete. The second part is to "walk into the bathroom." Since I am directly in front of the open bathroom door, the correct action is to move forward to enter it. 'turn left' or 'turn right' would move me away from the goal. 'stop' is incorrect because I am not yet inside the bathroom. Therefore, the next action is 'move forward 25cm'. </think>

[Figure 191]

Navigation Action

<action>move forward 25cm</action>

Fig. 7. Nav-CoT-110K CoT data example.

Depth Scene(MeetingRoom)

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Start from the initial position and walk straight toward the chair on the left front. Pause briefly near the chair, then continue moving forward in a straight line toward the glass door on the right front, and stop at the door.

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Ego-viewMap VLNTask

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Ego-viewDepthMap VLNTask

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Steps

[Figure 221]

Walk straight down, then turn right and approach the row of black chairs. Stop beside the nearest chair.

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

Ego-viewDepthMap VLNTask

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Steps

[Figure 246]

Walk straight ahead, keeping the big wooden table on your left. Go all the way to the side of the table and then turn right. After that, just head straight until you reach the wall in front of you and stop there.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Ego-viewMapDepth ObjectNavTask

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Steps

[Figure 271]

Find the blue dustpan on the left side near the chairs.

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Ego-viewDepthMap ObjectNavTask

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Steps

[Figure 296]

Go to the chair with wooden legs on the left side.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Steps

###### Fig. 8. Real-world qualitative results of Nav-R1 on VLN and ObjectNav tasks in meeting room.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Scene(Lounge)

[Figure 325]

Head straight, then turn left when you reach the couch, and stop in front of it.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

Ego-viewMap VLNTask

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Depth

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Walk straight down, then turn right and approach the row of black chairs. Stop beside the nearest chair.

Steps

[Figure 350]

Walk straight ahead, then turn when you reach the couch on your right, and stop next to it.

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

Ego-viewMap VLNTask

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

Ego-viewMapDepth VLNTask

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

[Figure 380]

Steps

[Figure 381]

Start from the beginning, walk to the side table on your right and pause there. Then go straight towards the front-left and stop at the wall.

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

Ego-viewDepthMap ObjectNavTask

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

Steps

[Figure 412]

Head towards the front-right area and look for the white mouse near the sofa and side table.

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

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

Ego-viewMapDepth ObjectNavTask

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

Steps

[Figure 437]

Move straight ahead and look for the keyboard along the wall in front.

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Depth

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

Steps

###### Fig. 9. Real-world qualitative results of Nav-R1 on VLN and ObjectNav tasks in lounge.

Depth Scene(Corridor)

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

Walk straight to the black chair on your left. At the chair, turn left and stop facing the wall.

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

Ego-viewMap VLNTask

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Steps

[Figure 491]

Walk straight to the black chair on your right. Then turn right and stop facing the chair.

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

Ego-viewMap VLNTask

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

Ego-viewMapDepth VLNTask

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

Steps

[Figure 516]

Go to the black chair on your left and pause, then move forward to the front-right and stop at the blue umbrella.

[Figure 517]

[Figure 518]

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

Ego-viewDepthMap ObjectNavTask

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

Steps

[Figure 541]

Move straight ahead and look for the blue umbrella in front.

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

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

Ego-viewDepthMap ObjectNavTask

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

Steps

[Figure 566]

Head towards the front-left and find the black chair.

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

Depth

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

Steps

###### Fig. 10. Real-world qualitative results of Nav-R1 on VLN and ObjectNav tasks in corridor.

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

Ego View

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

Depth

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

Point Map

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

What do you see around you?

[Figure 623]

I see a shiny tiled floor stretching ahead, and an open corridor leading forward.

[Figure 624]

Go forward to the left, what do you see?

[Figure 625]

OK, go left and forward. I see a broom and dustpan.

[Figure 626]

How far are you from the broom now? Stop when you are almost to it.

[Figure 627]

Understood, I have stopped near the broom.

[Figure 628]

Bird's Eye View

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

Ego View

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

Depth

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

Point Map

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

Where are you now and what do you see?

[Figure 661]

I am in a lounge. I see two sofas in front of me.

[Figure 662]

Go forward to the right, what do you see?

[Figure 663]

OK, go right and forward. I see a coffee table.

[Figure 664]

###### Fig. 11. Real-world qualitative results of Nav-R1 on embodied dialogue task.

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

Ego View

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

Depth

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

Point Map

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

Where are you now and what do you see?

[Figure 697]

I am in a lounge. I see a coffee table and two sofas in front of me.

[Figure 698]

Is the path to the right coffee sofa accessible? If so, go to the side of sofa.

[Figure 699]

The path to the right sofa is passable. And I will go to stop there.

[Figure 700]

Can you walk directly to the wall now? If not, please give a path to the wall. If you can, walk directly to it.

[Figure 701]

No. The path to get to the wall is turn left and go straight. I will go to there.

[Figure 702]

Bird's Eye View

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

Ego View

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

Depth

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

Point Map

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

What objects are near you?

[Figure 735]

I observe chair legs on the left side and an umbrella placed further ahead.

[Figure 736]

What reasoning supports safe navigation?

[Figure 737]

The chair legs narrow the space, but there is a gap between them and the umbrella that I can use to pass through safely.

[Figure 738]

###### Fig. 12. Real-world qualitative results of Nav-R1 on embodied reasoning task.

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

Ego View

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

Depth

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

Point Map

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

Please plan a path to stop in front of the chair on your left.

[Figure 771]

- Step 1: Turn left.
- Step 2: Move forward.
- Step 3: Stop in front of the chair.

[Figure 772]

Okay. Please continue to plan a path to the blue umbrella in front of you on the right and stop.

[Figure 773]

- Step 1: Turn right.
- Step 2: Move forward slowly.
- Step 3: Stop in front of the blue umbrella and don't touch it.

[Figure 774]

Bird's Eye View

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

Ego View

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

Depth

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

Point Map

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

Please plan a path to stop in front of the sofa on your right.

[Figure 807]

- Step 1: Turn right.
- Step 2: Slowly move forward and turn slightly left.
- Step 3: Continue forward and stop in front of the sofa.

[Figure 808]

Okay. Please continue to plan a path to wall and stop there.

[Figure 809]

- Step 1: Turn left.
- Step 2: Slowly move forward and turn slightly right.
- Step 3: Continue forward and stop near the wall.

[Figure 810]

###### Fig. 13. Real-world qualitative results of Nav-R1 on embodied planning task.

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

Ego-viewBEVMap

- VLNTaskⅠ

Walk past the striped area rug. Make a right onto the marble floor. Walk past the bathroom on the left. Make a left opposite the zebra painting. Walk through the open door, and wait at the mirror.

[Figure 819]

Steps

[Figure 820]

Ego-viewBEVMapDepth

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

- VLNTaskⅡ

Go through the archway with the white windows, go passed the 4 pointed star on the floor and through the doorway, go into the bedroom towards the bed, make a slight right and stop at the intersection.

[Figure 844]

Steps

[Figure 845]

Ego-viewDepthBEVMap

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

- VLNTaskⅢ

Turn around and take a right. Enter into the bedroom on the left and wait there.

[Figure 869]

Steps

[Figure 870]

Ego-viewDepthBEVMap

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

- VLNTaskⅣ

[Figure 894]

[Figure 895]

Walk past staircase, turn left at dining table and stop in front of desk.

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

Depth

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

Ego-viewBEVMap

- VLNTaskⅤ

Go down the hallway passed the stairs and living area with the wood table. Exit out into the backyard through the doorway.

[Figure 919]

Steps

[Figure 920]

Ego-viewBEVMapDepth

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

- VLNTaskⅥ

Walk through the door by the sink into the middle of the next room. Turn right and walk down the hallway and enter the third door on your right.

[Figure 944]

Steps

[Figure 945]

Ego-viewBEVMapDepth

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

- VLNTaskⅦ

Go straight past the couches. Exit the room using the two double doors on the right. Turn right and wait across the zebra painting on the left wall.

[Figure 969]

Steps

[Figure 970]

Ego-viewDepthBEVMap

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

- VLNTaskⅧ

[Figure 994]

[Figure 995]

Walk past brown leather recliner. Walk through open french doors. Make hard left opposite zebra painting. Wait at mirror.

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

Depth

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

Ego-viewBEVMap

- ObjectNavTaskⅠ

Search for a tv_monitor.

[Figure 1019]

Steps

[Figure 1020]

Ego-viewBEVMapDepth

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

- ObjectNavTaskⅡ

Search for a chair.

[Figure 1044]

Steps

[Figure 1045]

Ego-viewBEVMapDepth

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

- ObjectNavTaskⅢ

Search for a plant.

[Figure 1069]

Steps

[Figure 1070]

Ego-viewDepthBEVMap

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

- ObjectNavTaskⅣ

[Figure 1094]

[Figure 1095]

Search for a toilet.

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

Depth

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

Steps

###### Fig. 16. Visual results of ObjectNav on HM3D.

REFERENCES

- [1] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “Etpnav: Evolving topological planning for vision-language navigation in continuous environments,” PAMI, 2024.
- [2] D. An, Z. Wang, Y. Li, Y. Wang, Y. Hong, Y. Huang, L. Wang, and J. Shao, “1st place solutions for rxr-habitat vision-and-language navigation competition,” in CVPRW, 2022.
- [3] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.
- [4] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf,

I. Reid, S. Gould, and A. Van Den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3674–3683.

- [5] S. Banerjee and A. Lavie, “METEOR: An automatic metric for MT evaluation with improved correlation with human judgments,” in Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, J. Goldstein, A. Lavie, C.-Y. Lin, and C. Voss, Eds. Ann Arbor, Michigan: Association for Computational Linguistics, June 2005, pp. 65–72. [Online]. Available: https://aclanthology.org/W05-0909/
- [6] H. Chen, A. Suhr, D. Misra, N. Snavely, and Y. Artzi, “Touchdown: Natural language navigation and spatial reasoning in visual street environments,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 12530–12539.
- [7] J. Chen, B. Lin, X. Liu, X. Liang, and K.-Y. K. Wong, “Affordancesoriented planning using foundation models for continuous visionlanguage navigation,” arXiv preprint arXiv:2407.05890, 2024.
- [8] K. Chen, J. K. Chen, J. Chuang, M. Vázquez, and S. Savarese, “Topological planning with transformers for vision-and-language navigation,” in CVPR, 2021.
- [9] P. Chen, D. Ji, K. Lin, R. Zeng, T. Li, M. Tan, and C. Gan, “Weaklysupervised multi-granularity map learning for vision-and-language navigation,” in NeurIPS, 2022.
- [10] S. Chen, X. Chen, C. Zhang, M. Li, G. Yu, H. Fei, H. Zhu, J. Fan, and T. Chen, “Ll3da: Visual interactive instruction tuning for omni-3d understanding, reasoning, and planning,” in CVPR, 2024, pp. 26418– 26428.
- [11] A.-C. Cheng, Y. Ji, Z. Yang, Z. Gongye, X. Zou, J. Kautz, E. Bıyık, H. Yin, S. Liu, and X. Wang, “Navila: Legged robot vision-languageaction model for navigation,” arXiv preprint arXiv:2412.04453, 2024.
- [12] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [13] C. Gao, L. Jin, X. Peng, J. Zhang, Y. Deng, A. Li, H. Wang, and S. Liu, “Octonav: Towards generalist embodied navigation,” arXiv preprint arXiv:2506.09839, 2025.
- [14] G. Georgakis, K. Schmeckpeper, K. Wanchoo, S. Dan, E. Miltsakaki, D. Roth, and K. Daniilidis, “Cross-modal map learning for vision and language navigation,” in CVPR, 2022.
- [15] A.-M. Halacheva, J.-N. Zaech, X. Wang, D. P. Paudel, and L. V. Gool, “Gaussianvlm: Scene-centric 3d vision-language models using language-aligned gaussian splats for embodied reasoning and beyond,” arXiv preprint arXiv:2507.00886, 2025.
- [16] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the gap between learning in discrete and continuous environments for vision-andlanguage navigation,” in CVPR, 2022.
- [17] Y. Hong, Y. Zhou, R. Zhang, F. Dernoncourt, T. Bui, S. Gould, and H. Tan, “Learning navigational visual representations with semantic map supervision,” in ICCV, 2023.
- [18] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3dLLM: Injecting the 3d world into large language models,” in Thirtyseventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=YQA28p7qNz
- [19] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9
- [20] H. Huang, Y. Chen, Z. Wang, R. Huang, R. Xu, T. Wang, L. Liu, X. Cheng, Y. Zhao, J. Pang, and Z. Zhao, “Chat-scene: Bridging 3d scene and large language models with object identifiers,” in The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024.

- [21] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” in ICLR 2024 Workshop: How Far Are We From AGI, 2024. [Online]. Available: https://openreview.net/forum?id=ltX3S0juSa
- [22] ——, “An embodied generalist agent in 3d world,” in ICLR 2024 Workshop: How Far Are We From AGI, 2024. [Online]. Available: https://openreview.net/forum?id=ltX3S0juSa
- [23] T. Huang, Z. Zhang, and H. Tang, “3d-r1: Enhancing reasoning in 3d vlms for unified scene understanding,” arXiv preprint arXiv:2507.23478, 2025.
- [24] T. Huang, Z. Zhang, Y. Wang, and H. Tang, “3d coca: Contrastive learners are 3d captioners,” arXiv preprint arXiv:2504.09518, 2025.
- [25] T. Huang, Z. Zhang, R. Zhang, and Y. Zhao, “Dc-scene: Datacentric learning for 3d scene understanding,” arXiv preprint arXiv:2505.15232, 2025.
- [26] G. Ilharco, V. Jain, A. Ku, E. Ie, and J. Baldridge, “General evaluation for instruction conditioned navigation using dynamic time warping,” arXiv preprint arXiv:1907.05446, 2019.
- [27] D. Kahneman, Thinking, Fast and Slow. Allen Lane, 2011.
- [28] J. Krantz and S. Lee, “Sim-2-sim transfer for vision-and-language navigation in continuous environments,” in ECCV, 2022.
- [29] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16. Springer, 2020, pp. 104–120.
- [30] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Roomacross-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding,” in EMNLP, 2020.
- [31] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, July 2004, pp. 74–81. [Online]. Available: https://aclanthology.org/W04-1013/
- [32] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “Instructnav: Zero-shot system for generic instruction navigation in unexplored environment,” arXiv preprint arXiv:2406.04882, 2024.
- [33] X. Ma, S. Yong, Z. Zheng, Q. Li, Y. Liang, S.-C. Zhu, and S. Huang, “Sqa3d: Situated question answering in 3d scenes,” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=IDJx97BC38
- [34] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ser. ACL ’02. USA: Association for Computational Linguistics, 2002, p. 311–318. [Online]. Available: https://doi.org/10.3115/1073083.1073135
- [35] D. A. Pomerleau, “Alvinn: an autonomous land vehicle in a neural network,” in Proceedings of the 2nd International Conference on Neural Information Processing Systems, ser. NIPS’88. Cambridge, MA, USA: MIT Press, 1988, p. 305–313.
- [36] Z. Qi, Z. Zhang, Y. Yu, J. Wang, and H. Zhao, “Vln-r1: Visionlanguage navigation via reinforcement fine-tuning,” arXiv preprint arXiv:2506.17221, 2025.
- [37] S. Raychaudhuri, S. Wani, S. Patel, U. Jain, and A. Chang, “Languagealigned waypoint (law) supervision for vision-and-language navigation in continuous environments,” in EMNLP, 2021.
- [38] S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, G. Gordon, D. Dunson, and M. Dudík, Eds., vol. 15. Fort Lauderdale, FL, USA: PMLR, 11–13 Apr 2011, pp. 627–635.
- [39] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.
- [40] Z. Shao, P. Wang, ihao Zhu, R. Xu, J. Song, M. Zhang, Y. W. Y.K. Li, and D. Guo, “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” CoRR, vol. abs/2402.03300,

2024. [Online]. Available: https://arxiv.org/abs/2402.03300

- [41] Z. Song, G. Ouyang, M. Fang, H. Na, Z. Shi, Z. Chen, F. Yujie, Z. Zhang, S. Jiang, M. Fang, et al., “Hazards in daily life? enabling robots to proactively detect and resolve anomalies,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 7399–7415.

- [42] Z. Song, G. Ouyang, M. Li, Y. Ji, C. Wang, Z. Xu, Z. Zhang, X. Zhang, Q. Jiang, Z. Chen, et al., “Maniplvm-r1: Reinforcement learning for reasoning in embodied manipulation with large visionlanguage models,” arXiv preprint arXiv:2505.16517, 2025.
- [43] L. Team, A. Modi, A. S. Veerubhotla, A. Rysbek, A. Huber, A. Anand, A. Bhoopchand, B. Wiltshire, D. Gillick, D. Kasenberg, E. Sgouritsa, G. Elidan, H. Liu, H. Winnemoeller, I. Jurenka, J. Cohan, J. She, J. Wilkowski, K. Alarakyia, K. R. McKee, K. Singh, L. Wang, M. Kunesch, M. Pîslar, N. Efron, P. Mahmoudieh, P.-A. Kamienny,

- S. Wiltberger, S. Mohamed, S. Agarwal, S. M. Phal, S. J. Lee,
- T. Strinopoulos, W.-J. Ko, Y. Gold-Zamir, Y. Haramaty, and Y. Assael, “Evaluating gemini in an arena for learning,” arXiv preprint arXiv:2505.24477, 2025.

- [44] R. Vedantam, C. L. Zitnick, and D. Parikh, “Cider: Consensus-based image description evaluation,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 4566–4575.
- [45] H. Wang, W. Liang, L. Van Gool, and W. Wang, “Dreamwalker: Mental planning for continuous vision-language navigation,” in ICCV, 2023.
- [46] X. Wang, Z. Li, Y. Xu, J. Qi, Z. Yang, R. Ma, X. Liu, and C. Zhang, “Spatial 3d-llm: Exploring spatial awareness in 3d vision-language models,” arXiv preprint arXiv:2507.16524, 2025.
- [47] X. Wang, Q. Huang, A. Celikyilmaz, J. Gao, D. Shen, Y.-F. Wang, W. Y. Wang, and L. Zhang, “Reinforced cross-modal matching and self-supervised imitation learning for vision-language navigation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 6622–6631.
- [48] Z. Wang, X. Li, J. Yang, Y. Liu, J. Hu, M. Jiang, and S. Jiang, “Lookahead exploration with neural radiance representation for continuous vision-language navigation,” in CVPR, 2024.
- [49] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “Gridmm: Grid memory map for vision-and-language navigation,” in ICCV, 2023.
- [50] M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen, X. Liu, and J. Pang, “Streamvln: Streaming vision-and-language navigation via slowfast context modeling,” arXiv preprint arXiv:2507.05240, 2025.
- [51] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “Vlfm: Visionlanguage frontier maps for zero-shot semantic navigation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 42–48.
- [52] N. Yokoyama, R. Ramrakhya, A. Das, D. Batra, and S. Ha, “Hm3dovon: A dataset and benchmark for open-vocabulary object goal navigation,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5543–5550.
- [53] Z. Yu, Y. Long, Z. Yang, C. Zeng, H. Fan, J. Zhang, and H. Dong, “Correctnav: Self-correction flywheel empowers visionlanguage-action navigation model,” arXiv preprint arXiv:2508.10416, 2025.
- [54] J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang, “Uni-navid: A video-based vision-languageaction model for unifying embodied navigation tasks,” Robotics: Science and Systems, 2025.
- [55] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and W. He, “Navid: Video-based vlm plans the next step for vision-and-language navigation,” in RSS, 2024.
- [56] H. Zhi, P. Chen, J. Li, S. Ma, X. Sun, T. Xiang, Y. Lei, M. Tan, and C. Gan, “Lscenellm: Enhancing large 3d scene understanding using adaptive visual preferences,” arXiv preprint arXiv:2412.01292, 2024.
- [57] F. Zhu, X. Liang, Y. Zhu, Q. Yu, X. Chang, and X. Liang, “Soon: Scenario oriented object navigation with graph-based exploration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12689–12699.
- [58] Z. Zhu, X. Wang, Y. Li, Z. Zhang, X. Ma, Y. Chen, B. Jia, W. Liang, Q. Yu, Z. Deng, S. Huang, and Q. Li, “Move to understand a 3d scene: Bridging visual grounding and exploration for efficient and versatile embodied navigation,” International Conference on Computer Vision (ICCV), 2025.

