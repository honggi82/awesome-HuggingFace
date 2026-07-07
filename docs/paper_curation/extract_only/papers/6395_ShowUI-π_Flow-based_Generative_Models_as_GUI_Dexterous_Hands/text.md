### ShowUI-π: Flow-based Generative Models as GUI Dexterous Hands

Siyuan Hu* Kevin Qinghong Lin* Mike Zheng Shou Show Lab, National University of Singapore https://showlab.github.io/showui-pi

# arXiv:2512.24965v1[cs.CV]31Dec2025

[Figure 1]

[Figure 2]

[Figure 3]

Query: Rotate top-center Title starting with ‘Summary’ counterclockwise by 135 degrees.

[Figure 4]

Initialize

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

(a) PowerPoint: Resize the textbox diagonally. (b) Captcha: Solve the rotate captcha. (c) Premiere: Apply the effect to the clip.

ShowUI-π

[Figure 8]

[Figure 9]

[Figure 10]

|[3]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]<br><br>[Figure 14]|
|---|

|[Figure 15]<br><br>[Figure 16]|
|---|

Drag = press-and-hold + move cursor along a trajectory continuously

|ShowUI-π: 26.98 (+4.8)| | |
|---|---|---|
|Gemini-CUA: 22.18| | |
|Operator: 13.27| | |

(e) PowerPoint: Resize the textbox horizontally.

(f) OS Desktop: Sort the file into a folder.

(d) Handwriting: Write on the canvas.

Figure 1. Drag refers to a continuous interaction where the cursor maintains contact with the UI element while moving along a trajectory, rather than a single discrete click. Left: Visualization of ScreenDrag data domains. Right: ShowUI-π is a lightweight flow-based generative model for GUI Automation that handles dragging actions requiring on-the-fly observation, such as drawing and Captcha solving. Given a query, ShowUI-π efficiently generates corresponding continuous trajectory from streaming visual observations.

Digital Dex. Hand

#### Abstract

sual observations via a lightweight action expert, ensuring smooth and stable trajectories; (iii) Drag Training data and Benchmark, where we manually collect and synthesize 20K drag trajectories across five domains (e.g., PowerPoint, Adobe Premiere Pro), and introduce ScreenDrag, a benchmark with comprehensive online and offline evaluation protocols for assessing GUI agents’ drag capabilities. Our experiments show that proprietary GUI agents still struggle on ScreenDrag (e.g., Operator scores 13.27, and the best Gemini-2.5-CUA reaches 22.18). In contrast, ShowUI-π achieves 26.98 with only 450M parameters, underscoring both the difficulty of the task and the effectiveness of our approach. We hope this work advances GUI agents toward human-like dexterous control in digital world. The code is available at https://github. com/showlab/showui-pi.

Building intelligent agents capable of dexterous manipulation is essential for achieving human-like automation in both robotics and digital environments. However, existing GUI agents rely on discrete click predictions (x,y), which prohibits free-form, closed-loop trajectories (e.g., dragging a progress bar) that require continuous, on-the-fly perception and adjustment. In this work, we develop ShowUIπ, the first flow-based generative model as GUI dexterous hand, featuring the following designs: (i) Unified Discrete–Continuous Actions, integrating discrete clicks and continuous drags within a shared model, enabling flexible adaptation across diverse interaction modes; (ii) Flowbased Action Generation for drag modeling, which predicts incremental cursor adjustments from continuous vi-

*Equal contribution. Corresponding author.

#### 1. Introduction

Building intelligent assistants capable of dexterous manipulation is essential for achieving human-like automation in both physical and digital environments [18, 21, 44]. For example, in the physical world, robotic dexterous hands are deployed for manipulation tasks e.g.,, sorting objects on physical desktops [3, 4, 30, 52]. Analogously, in the digital world, Graphical User Interfaces (GUI) serve as the primary medium through which people interact with computers, and automating GUI interactions holds substantial promise for enhancing productivity [11], reducing workload [45], and improving accessibility [10].

Recent advances in large language models (LLMs) [1, 12, 16, 27, 50] and vision-language models (VLMs) [15, 35–37] have accelerated progress in GUI agents [6, 21]. These emerging visual-centric agents directly perceive screen observations and output actions such as clicking or typing. However, most existing GUI agents [38, 42] are obtained by fine-tuning foundation VLMs without architecture adaptation, which represent action’s coordinate in a discrete, tokenized form. This representation fundamentally restricts agents from executing complex highdegree-of-freedom dragging, such as creative drawing or Captcha-solving by rotation i.e., tasks inherently demanding continuous, real-time visual observation and responsible for incremental trajectory adjustment, By contrast, Vision-Language-Action models [3, 9, 30] in robotics leverage flow-based generative methods (e.g., diffusion policy, flow matching) to enable achieving such continuous, finegrained control on the fly.

Motivated by how humans control the mouse for finegrained cursor movement, i.e., continuously perceive and adjust actions, we wonder: can we construct such a digital dexterous hand in GUI? We propose ShowUI-π, the first flow-based GUI model designed for continuous trajectory. Specifically, ShowUI-π highlights the following architecture: (i) Unified Discrete-Continuous Actions: ShowUI-π casts discrete clicks as drags with negligible movements, and integrates them with continuous drags into a unified modeling. Under this formulation, both action types are represented by a sequence of (x,y,m) triplets, where (x,y) are cursor coordinates and m ∈ {down,up} is the mouse button state. This unified design allows ShowUI-π to handle both drag and click tasks with a single shared model, adapting without task-specific head selection. (ii) Flow-based Action Generation: Different from existing GUI agents predicting discrete, tokenized actions from language decoding, e.g.,, click(x,y) and drag(start, end), ShowUI-π is a flow-based generative model, and employs a lightweight action expert to incrementally predict cursor adjustments from continuous visual observations. Built on a transformer backbone, the action expert is trained with flow matching to generate stable and precise action trajecto-

ries; (iii) ScreenDrag benchmark: We construct a benchmark specially for continuous GUI tasks, including 505 real-world drag tasks across five domains covering both professional control and daily usage, such as PowerPoint, OS Desktop and file manager, Handwriting on canvas, Adobe Premiere Pro, and Captcha solving, with 101 tasks for each domain. Moreover, to fully evaluate continuous trajectories, we introduce two complementary modes: an offline openloop evaluation using average trajectory error and endpoint accuracy, and an online closed-loop evaluation using task success rate. (iv) Continuous Trajectory Training Data: To advance the model training, we construct a training dataset of 20K manually collected and synthesized drag trajectories across the five aforementioned domains and 11 categories of tasks, and all trajectories have recorded UI states and dense coordinates.

Our experiments show that proprietary GUI agents still struggle on ScreenDrag (e.g., Operator scores 13.27, and the best Gemini-2.5-CUA reaches 22.18). In contrast, ShowUIπ achieves 26.98 with only 450M parameters, underscoring both the difficulty of the task and the effectiveness of our approach. Moreover, our ablation studies further reveal the superficial nature of standard flow-matching training and highlight the substantial impact of individual designs. Our contributions are three-fold:

- • The first flow-based GUI agent for continuous trajectories. To the best of our knowledge, we are the first work to tackle continuous trajectory-based drags in GUI automation, revealing core limitations of discrete, tokenized actions. We propose ShowUI-π, a lightweight 450M flow-based VLA that unifies discrete clicks and continuous drags within a shared modeling.
- • ScreenDrag training dataset and benchmark suite. We introduce ScreenDrag, a benchmark tailored for continuous GUI manipulation, with 505 real-world drag tasks across five domains and 20K manually collected and synthesized drag trajectories over 11 task categories. ScreenDrag supports both offline open-loop metrics and online closed-loop success evaluation.
- • Comprehensive evaluation and key insights. Experiments show that proprietary GUI agents still struggle on ScreenDrag (e.g.,, Operator 13.27, best Gemini-2.5-CUA 22.18), while ShowUI-π reaches 26.98 with only 450M parameters, highlighting both task difficulty and model effectiveness. Ablations further demonstrate the impact of our individual design choices for achieving robust continuous control.

#### 2. Related Work

##### 2.1. Digital GUI Automation.

LLM-based agents in digital environments have evolved from reasoning and tool-use paradigms, such as Chain-of-

Stage 1: Element Parsing Stage 2: Task Proposal Stage 3: Execute in Env.

[Figure 17]

|[Figure 18]<br><br>[Figure 19]|
|---|

|[Figure 20]<br><br>[Figure 21]|
|---|

[Figure 22]

|[Figure 23]|[Figure 24]|
|---|---|
| | |

|[Figure 25]<br><br>|1|
|---|
<br><br>|2|
|---|
<br><br>|3|
|---|
<br><br>[656,315, 435,251]<br><br>[656,588, 435,181]<br><br>[1098,315 ,458,454]|
|---|

Instruction: Reduce the image width by 0.4 from its right

[Figure 26]

PyAutoGUI.drag([(1327,542),(15 56,542),(1510,542),(1463,542), (1417,542),(1373,542)])

BoundingBox:[656,315,435 ,251],[656,588,435,181], [1098,315,275,454]

[Figure 27]

Code w. Dense Trajectory List of coordinates

Desktop Software (e.g., Powerpoint)

Position, Dimension Bounding Boxes

Record Data UI state changes & metadata

Environment

UI Element Metadata

Reverse Instruction to Code Execute Code in Environment

Figure 2. ScreenDrag Automated Data Collection Pipeline. ScreenDrag automated data generation pipeline for continuous trajectorybased GUI interaction data. The pipeline includes three stages: (i) Element Parsing: The software application UI is parsed with UI Automation of Windows SDK in order to retrieve the UI element metadata. (ii) Task Proposal: Given the UI element metadata, an LLM will be prompted to generate a drag instruction, the expected metadata change and the drag code with dense trajectory. (iii) Trajectory Synthesis: The drag code will be executed in the software environment. A rule-based verifier will check the parsed metadata from UI states before and after the drag to ensure that the metadata change s .

|Input: xxx ~ Trajectory<br><br>satisfies the expectation.<br><br>mization ternative|
|---|

Thought [39], ReAct [46], and related prompting frameworks [8, 43, 47], toward task-oriented GUI automation. Existing methods include training-free pipelines which plan with a VLM and execute via external tools [13, 49], and training-based models jointly learning perception and action from screenshots and instructions [20, 35, 37]. Existing GUI agents, including ShowUI, decode actions into discrete text tokens, simplifying integration with VLM planners but limiting control to simple clicks or short drags. ShowUI-π departs from these methods by directly modeling continuous spatial trajectories for long, smooth, temporally coherent manipulations.

and high-dimensional handling [9, 17, 34, 51]. Alframeworks such as (ii) Flow Matching and Recti-

[(0,1327,542),(1,1556,542),(1,1510,542),(1,1463,542),(1,1417,542),(1,1373 ,542)] [(1327,542), (1556,542), (1510,542), (1463,542), (1417,542), (1373,542)]

directly regress a time-conditioned velocity field predefined probability path, eliminating explicit

|Output: Reduce the<br><br>fied Flow along a score estimation|
|---|

n and iterative denoising for deterministic ODE-based sampling. Rectification further simplifies trajectory modeling and enhances sample efficiency [22, 23]. Such properties have inspired their integration into roboticsoriented flow-matching VLA policies, combining a flowbased action head with VLM backbones for smooth, realtime continuous control [3, 30]. The learning objective is:

T

1 T

##### 2.2. Physical Robotic Manipulation.

∥vθ(aˆt,t | ot,Q) − ut∥2, (1)

Lflow matching =

t=1

Generalist Vision-Language-Action (VLA) models map natural-language instructions and visual observations to action policies by leveraging large multimodal backbones and robotics data at scale. Representative lines include RT-1 [4] and successors [52] which transfer knowledge from webscale vision-language pretraining to robot control, opensource VLA backbones and compact variants such as OpenVLA [18], PaliGemma [2], and TinyVLA [40], as well as recent works enriching training signals with visual traces and language alignment [19, 26].

where vθ(aˆt,t | ot,Q) denotes the predicted velocity field at time t conditioned on the current observation ot and task instruction Q, ut is the target velocity along the probability path, aˆt is the intermediate action state, and T is the number of training timesteps.

#### 3. ScreenDrag Dataset

##### 3.1. Problem Definitions

Discrete Language Models formulations predict low-level controls from decoded discrete action tokens. Representative models include RT-2 [52] and OpenVLA [18], which discretize continuous actions into 256 bins per dimension and overwrite the least-used tokens in the language model vocabulary to represent these action bins as autoregressively generated tokens; and Magma [44], which applies discrete action tokenization across both UI navigation and robotic manipulation tasks. Language-modeling maintains compatibility with instruction-tuned VLMs but suffers from temporal quantization when fine motor control is needed [29].

Unlike discrete actions such as clicking, which can be completed with one command based on an initial observation, continuous actions—such as dragging or rotation—require on-the-fly visually grounded trajectories in real time. Formally, given an initial observation o0 and a language instruction Q, a GUI agent policy π sequentially predicts a continuous trajectory τ = {at}Tt=0, where each action at = [x,y] denotes the spatial coordinates at time step t.

However, existing benchmarks reduce drag-like interactions to discrete start–end point pairs and provide only a single screenshot per atomic action, ignoring the rich intermediate state changes that occur during continuous dragging. In addition, many drag scenarios allow multiple valid trajectories to complete a task, yet this flexibility is not captured in current formulations. To address these limitations, we introduce ScreenDrag, a benchmark designed to evaluate GUI

Continuous Generative Models parameterize a timeindexed transport from a base distribution to action trajectories. Within this paradigm, (i) Diffusion Policies instantiate the transport via score-based denoising, demonstrating strong capabilities in visuomotor control due to stable opti-

[Figure 28]

agents’ capabilities in performing continuous actions.

[Figure 29]

[Figure 30]

Solve the rotate Captcha.

PowerPoint

Captcha

##### 3.2. Dataset Construction

Rotate top-center Title starting with "Pitch" clockwise by 90 degrees.

Data Source. ScreenDrag comprises a diverse set of realworld drag tasks including (1) PowerPoint element manipulation such as rotation, (2) File drag sorting in OS desktop and File manager, (3) Handwriting on canvas, (4) Premiere Pro asset manipulation between workbench and tracks, as well as (5) Captcha-solving. These tasks cover both daily usage and professional control, requiring precise and continuous spatial control, and observation on-the-fly during the task execution.

Adobe Premiere Pro

[Figure 31]

Train Data & Benchmark 5 domains & 11 categories of tasks (20K + 505)

Move the timeline to align with the beginning of "Temple

Drag Q1Plan.xlsx to StudioFile.

[Figure 32]

[Figure 33]

OS Desktop & File Manager

Handwriting

Data Curation. Notably, high-quality drag data must capture screen state changes, dense cursor trajectories, and diverse task instructions. A straightforward approach is to manually record human demonstrations, but this is laborintensive, thus not scalable. To overcome this limitation, we propose a scalable data collection pipeline that leverages automated drag execution, as illustrated in Fig. 2. Our data curation approach includes three stages: (i) Element Parsing: We first extract the element bounding box metadata using UI Automation of Windows SDK [48], to collect a set of candidate elements to be dragged, e.g.,, an image in a PowerPoint slides. (ii) Task Proposal: Then we use an LLM (Qwen-2.5-72B [31]) to propose candidate instructions, e.g.,, reduce the image width by 0.4 from its right. (iii) Trajectory Synthesis: Next, we synthesize drag PyAutoGUI code containing dense trajectory coordinates and sending it to a code executor, which performs the drag actions on the real operating system and software applications. To complement this automated process, we also collect human demonstrations of drag interactions. Each human demonstration includes a high-resolution 60 FPS screen recording that captures UI state changes, the executed dense trajectory, and the corresponding instruction.

Draw "Hello World" on the canvas.

Figure 3. ScreenDrag Data Distribution. The inner ring indicates the five equally distributed domains. The outer ring demonstrates per-category breakdowns with shares of the full dataset.

start–end linear drag and instead offer a dynamic environment in which the model must perform drag actions incrementally. To this end, ScreenDrag introduces two complementary evaluation modes for drag tasks: offline evaluation and online evaluation.

Offline Evaluation is performed as an open-loop [41] assessment of the policy, measuring the policy’s ability to replicate the ground-truth stepwise behavior without compounding errors. Given a static environment where screenshot, task and oracle previous state are provided, offline evaluation assesses whether the policy can successfully reach the next trajectory point, and the amount of the error, which can be represented by two metrics:

- (i) Average Trajectory Error evaluates the mean square error of the entire trajectory (aligned with Eq.1), defined as the average Euclidean distance between predicted and ground-truth coordinates across all timesteps:

ATE =

1 T

T

t=0

a ˆt − agtt 2, (2)

providing a detailed assessment of the spatial accuracy and smoothness of interactions step by step throughout trajectory execution.

- (ii) Trajectory Endpoint Accuracy emphasizes endpoint precision, measuring the proportion of predicted trajectories whose endpoints fall within a predefined spatial tolerance radius ϵ, e.g.,, 20 pixels from the ground-truth endpoints. This metric highlights the agent’s ability to accurately reach the intended final interaction points. Defined as:

Data Analysis. As shown in Fig. 3, ScreenDrag comprises a training dataset of 20K manually collected and synthesized drag trajectories, with high resolution screen recording and metadata of the UI changes, as well as dense trajectories during the drag. The training dataset spans across five aforementioned domains and 11 categories of tasks. Moreover, ScreenDrag includes a benchmark of 505 trajectories for evaluation (101 tasks for each domain). The average duration of screen recordings is 9.62 seconds, and the average number of frames is 577 frames per recording. Refer to the Supplementary Material for more detailed analysis.

##### 3.3. Evaluation Metrics

T

1 T

Besides preparing training data, an effective evaluation protocol for model-generated drag trajectories is also essential. As noted earlier, the benchmark must expose UI state changes to the model throughout the drag process. Moreover, evaluation should go beyond assessing a simple

1 a ˆt ∈ Rt , (3)

TEA =

t=0

where aˆt is the predicted intermediate action state, agtt is the ground-truth intermediate action state, Rt is the corre-

[Figure 34]

[Figure 35]

Task: Rotate top-center Title starting with 'Summary' counterclockwise by 135 degrees.

[Figure 36]

[Figure 37]

[Figure 38]

N

[Figure 39]

1 N

i ∈ Rgoali , (4)

Success Rate =

1 a ˆi,T

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

i=1

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

where N is the number of online rollouts (tasks), aˆi,T

is

i

the final action state of the i-th rollout, and Rgoali is the goal acceptance region for that task (e.g., all end states within ϵ

Policy Policy Policy

Policy

pixels of the ground-truth goal). See Supp. for illustrations.

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

[Figure 61]

鼠标连起来

#### 4. ShowUI-πModel

Pred: [1179,251] GT: [1185,254] Δ: 22.50

Pred: [1121,308] GT: [1126,310] Δ: 14.50

Pred: [1110,376] GT: [1114,385] Δ: 48.50

Pred: [(1179,251), (1121,308), (1110,376)]

[Figure 62]

Overview. As outlined in Fig. 6, ShowUI-π is built upon the SmolVLA-450M [33] architecture, with tailored designs for streaming GUI control. ShowUI-π consists of two primary integrated components: (i) a pretrained VLM, initialized by SmolVLM-2 [25] and (ii) a flow matching action expert. The VLM efficiently processes multimodal inputs, encoding screenshot o into visual token representations in a unified embedding space together with projected action states and language instructions Q. The action expert is built on a transformer with the same number of layers as the VLM backbone (i.e. 16). During action prediction, the corresponding layers of the VLM backbone and the action expert perform interleaved self-attention and cross-attention. The action state a from the previous step is projected back into the VLM backbone to condition subsequent predictions; it is initialized as a0 = [−1,−1] and later updated using the last action ak. The action expert is trained with flow matching, executed k times per trajectory, where each step refines noisy actions into clean predictions, enabling smooth and deterministic trajectory generation. We next illustrate our three designs for effectively modeling: (i) Unified action representation; (ii) Flow-based trajectory generation; (iii) Directional regularization for stable trajectory prediction.

Reach the correct final state

Offline Eval (MSE)

Online Eval (Succ. Rate)

Figure 4. Comparison between offline evaluation and online evaluation pipelines of ScreenDrag: (i) Offline evaluation is based on the distance of prediction and ground-truth in independent trunks; (ii) Online evaluation is incremental over sequential trunks, and based on the final outcome.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

sponding acceptance region, t is the timestamp, and T is the number of timesteps.

Online Evaluation. Offline evaluation metrics are inherently limited as they solely measure static stepwise trajectory precision, neglecting dynamic closed-loop [41] interactions and UI state changes in real-world GUI drag tasks, which are missing from offline evaluation, or naive endpoint-pair evaluation in existing benchmarks. Notably, online evaluation naturally captures real-world trajectory variance, where agents can successfully achieve task goals despite diverse trajectory patterns.

However, constructing realistic online evaluation environments [24] is non-trivial, such environments should provide diversity across multiple domains, while preserving the reproducibility. One naive approach is to set up the OS and software applications as environment, however, such designs either by manual setup or virtual machine image are costly and difficult to reproduce as the initial state of drag tasks are highly diverse. Therefore, we design a datadriven approach to enable policy closed-loop rollouts. For each drag task, we store the video recording, task specification, and dense drag trajectory, providing extensive possible GUI states encountered during dragging. During rollouts, the model’s predicted action is matched to the nearest recorded state if it falls within a tolerance ϵ (e.g., within 20 pixels of a ground-truth waypoint), upon which the corresponding next observation is retrieved. See the Supp. for more details. (iii) Task Success Rate: In this data-driven online environment, the agent receives the current visual observation and task instruction, generates an action, and subsequently obtains an updated visual observation from the approximated corresponding state from the extensive states. We measure performance primarily through binary task success (success or failure) per rollout, computed as the number of successful tasks against the total number of tasks.

##### 4.1. Unified Continuous and Discrete Actions

Clicks and drags exhibit different temporal and spatial dynamics, making it nontrivial to integrate them within a unified model. Yet such unification is crucial: a single model capable of handling both interaction types can flexibly adapt to diverse GUI tasks that require switching between discrete and continuous actions. Building upon our trajectory generation, we aim to unify discrete clicks and continuous drags under a unified action modeling framework. Observing that “a click is essentially a drag with negligible movement”, we thus unify click or drag action interactions as sequences A of (x,y,m) triplets, where (x,y) are cursor coordinates and m is an atomic up and down mouse button state.

A = [a1,...,aH],ak = (xk,yk,mk),mk ∈ {down,up}. (i) Within this representation, clicks become minimal twostep trajectories: [(x1,y1,down), (x1,y1,up)] while (ii)

Predicted 1-th action: a1,a2,a3,…,ak

Predicted 2-th action ak+1,ak+2,ak+3,…,a2k

[Figure 69]

Environment Textual token Action token

Visual token

[Figure 70]

[Figure 71]

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

LLM ActionExpert

## LLM

Action Expert

Vision Encoder

Vision Encoder

[Figure 72]

Rotate title starting with 'Summary' counter-clock wise by 135 degrees.

Rotate title starting with 'Summary' counter-clock wise by 135 degrees.

Noisy action Noisy action

|[Figure 73]|
|---|

Prev. action state: ak

|[Figure 74]|
|---|

Init. action state: (-1,-1,0)

Update Environment

Task Query 1st Observation

[Figure 75]

[Figure 76]

[Figure 77]

Task Query 2nd Observation

[Figure 78]

v

KV QKV KV

Cross-Attention Self-Attention

- Figure 5. Overview of ShowUI-π. Given a task query and visual observations, the model first processes them through the VLM to obtain intermediate hidden states, which are then attended by the action expert. During interaction, the predicted actions update the environment, the next observation is encoded, and a new action chunk is produced—enabling fine-grained, closed-loop cursor control.

ShowUI-π

Rotate title starting with 'Summary' counter-clock wise by 135 degrees.

| | |
|---|---|
| |t|
| | |
| | |

LLM

Task Query 1st Observation

[Figure 79]

[Figure 80]

Vision Encoder

Init. action state: (-1,-1,0)

[Figure 81]

Action Expert

Cross-Attention Self-Attention

KV QKV KV

Self-Attention Self-Attention

Noisy Action a

Cross-Attention

v

|[Figure 83]<br><br>4.3. Directional<br><br>GUI actions|
|---|

[Figure 84]

Cross-Attention Self-Attention

Noisy Action a

Self-Attention Cross-Attention

KV

Visual token

Environment Textual token Action token

[Figure 85]

Update Environment

LLM

Predicted 1-th action: a1,a2,a3,…,ak

Rotate title starting with 'Summary' counter-clock wise by 135 degrees.

Task Query 2nd Observation

[Figure 86]

[Figure 87]

Prev. action state: ak

|[Figure 88]|
|---|

Predicted 2-th action ak+1,ak+2,ak+3,…,a2k

QKV KV

Predicted action v

Action Chunk [a1,a2,a3,…,ak]

Action Expert Vision Encoder

Noisy action Noisy action

| | |
|---|---|
| | |
| | |
| | |

LLM ActionExpert

Vision Encoder

- Figure 6. Architecture of ShowUI-π. ShowUI-π uses an LLM with cross-attention to a lightweight action expert to generate unified action chunks that handle both discrete clicks and continuous drag segments.

Action Chunk [a1,a2,a3,…,ak]

ShowUI-π

Cross-Attention

Predicted action v

Noisy Action a

daˆ(s) ds

KV

= vθ a ˆ(s),s | ot,Q , (5)

Cross-Attention Self-Attention

Self-Attention Self-Attention

QKV KV

where ot is the current visual observation, Q is the task instruction, and aˆ(s) = [x,y,m] represents the predicted action trajectory as in § 4.1. Moreover, s ∈ [0,1] is a continuous parameter of the flow, smoothly moving the cursor from the segment’s start where s = 0 to its end where s = 1.

Self-Attention Cross-Attention

Action Expert

LLM

Noisy Action a

Vision Encoder

Instead of treating all trajectory steps equally as in the naive flow-matching loss (Eq. 1), we emphasize the steps that matter most for successful GUI interaction. In particular, early movements must be conditioned on the start point, and the final steps must land precisely on the intended endpoint. To capture this, we introduce a reweighting scheme that increases the contribution of both the initial and terminal segments of the trajectory:

Drags, become naturally represented as extended incremental press-hold trajectories:

T

###### [(x1,y1,down), (x2,y2,down), ..., (xT,yT,up)].

wt L(flowt) matching, (6)

Lweighted =

=1

This unified representation greatly simplifies the action space, removing the need for rigid, predefined action formats used in prior GUI agents (e.g.,language tokens). By modeling all interactions as continuous sequences, the policy naturally supports flexible multi-dataset co-training, seamlessly integrating both clicking [7, 21] and dragging within a single unified framework.

where wt is a constant that assigns larger weights to critical steps. We adopt a simple scheme that assigns a weight of 10 to the start and end points, otherwise 1.

##### al Regularization

inherently demand smooth, directionally coherent cursor movements. Standard flow-based methods tend to optimize trajectory magnitude without explicitly enforcing directional consistency. This directional misalignment can cause critical failures in GUI tasks, such as incorrect cursor orientation or jitter. To explicitly regularize directional alignment, we introduce a directional regular-

##### 4.2. Flow-based Continuous Trajectory

To achieve real-time interactions—which demand smooth and efficient trajectory generation—we adopt a flowbased incremental generation framework [22] driven by a lightweight conditional vector field vθ:

Offline Endpoint Accuracy (%, ↑) Offline Trajectory Error (↓)

Methods

OS PPT Premiere Captcha Handwriting Overall OS PPT Premiere Captcha Handwriting Overall Action as Language — Proprietary models

Operator 51.49 3.96 0.00 0.00 0.00 11.09 46.85 244.25 602.00 0.00 795.58 422.17 UI-TARS-1.5-7B 60.40 0.99 0.00 3.96 0.00 13.07 224.57 235.50 564.89 172.53 703.54 380.21 Seed-1.6-Vision 71.29 1.98 0.99 4.95 0.00 15.84 62.17 233.92 467.09 81.64 824.55 333.87 Gemini-2.5-CUA 84.16 11.88 0.00 3.96 0.00 20.00 12.11 121.94 508.19 114.34 0.00 189.15 Action as Language — Open-source models

OpenCUA-32B 96.04 6.93 0.00 0.00 0.00 20.59 10.95 165.70 907.41 62.06 791.88 387.60 OpenCUA-7B 99.01 4.95 0.00 3.96 0.00 21.58 4.42 186.65 762.77 214.81 959.10 425.55 Qwen3-VL-32B 71.29 1.98 0.99 0.00 0.00 14.85 149.77 212.02 527.23 236.83 646.73 354.52 Qwen3-VL-8B 8.91 1.98 0.00 0.00 0.00 2.18 660.33 228.47 532.71 257.44 662.37 468.26 Qwen3-VL-2B 3.96 0.00 0.99 0.00 0.00 0.99 708.68 248.29 551.15 258.07 562.62 465.76 Flow-based action (Ours)

ShowUI-π-450M 61.39 85.06 56.92 96.30 93.07 78.55 350.55 57.68 195.96 136.47 54.57 159.05

Table 1. Main Results on ScreenDrag offline evaluation. Left: Endpoint Accuracy; Right: Trajectory Error.

ization loss term, defined as

1 T

Lreg =

T

(1 − cos(aˆt,ut)), (7)

t=1

where aˆt and ut represent predicted and ground-truth point. Total Objective. By combining the reweighting flowmatching objective with directional regularization, the final training objective is defined as:

Ltotal = Lweighted + λLreg, (8)

where λ controls the balance between trajectory accuracy and directional consistency. We set λ to 0.1 to ensure comparable magnitudes of loss terms.

Online Success Rate (%, ↑)

Methods

OS PPT Premiere Captcha Handwriting Overall Action as Language — Proprietary models

Operator 53.47 9.90 2.97 0.00 0.00 13.27 UI-TARS-1.5-7B 73.27 1.98 1.98 7.92 0.00 17.03 Seed-1.6-Vision 77.23 3.96 1.98 8.91 2.97 19.01 Gemini-2.5-CUA 86.14 20.79 0.00 3.96 0.00 22.18 Action as Language — Open-source models

OpenCUA-32B 97.03 6.93 0.00 0.00 0.00 20.79 OpenCUA-7B 99.01 4.95 0.00 5.94 0.00 21.98 Qwen3-VL-32B 83.17 4.95 3.96 2.97 0.00 19.01 Qwen3-VL-8B 24.75 5.94 6.93 0.00 0.00 7.52 Qwen3-VL-2B 13.86 6.93 2.97 0.00 0.00 4.75 Flow-based action (ours) ShowUI-π-450M 13.11 22.93 8.64 55.91 34.32 26.98

- Table 2. Main Results on ScreenDrag (Online Success Rate).

#### 5. Experiments

Our experiments systematically investigate the following research questions: Q1: How does ShowUI-π compare to existing mainstream GUI agents on performing dexterous operations? Q2: What factors contribute to the performance

of ShowUI-π? Q3: How well does ShowUI-π generalize to different task domains under multi-dataset training?

Baselines. We compare ShowUI-π with several representative baselines, including both proprietary models and opensource models, including OpenAI Operator [28], Gemini2.5-CUA [14], Seed 1.6-Vision [5], Qwen3-VL [32], UITARS 1.5 [6], OpenCUA [38], etc. Because baseline models output actions as language, we use only their first predicted action for offline evaluation. For online evaluation, they interact with our data-driven closed-loop environment for fair comparison, and a rollout is marked successful if the goal is reached at any step. To control computational cost, we limit each baseline to three interaction steps.

##### 5.1. Main Results

Online Evaluation. In Tab. 2, ShowUI-π attains an overall online closed-loop success rate of 26.98%, surpassing the SOTA proprietary model Gemini-2.5-CUA by 4.8%, and the SOTA open-source model OpenCUA-7B by 6.19%. It is worth noting that ShowUI-π performs well on tasks involving free-form drag actions, e.g.,, PowerPoint where circular drag is required to rotate elements, and Handwriting where trajectories are non-linear. Moreover, ShowUI-π performs well on Captcha tasks where observation on-thefly is required. Interestingly, Operator failed all Captcha tasks as it refused to solve Captcha due to its safety policy, and Gemini-2.5-CUA always mistakenly calls openbrowser tool when executing handwriting tasks. Furthermore, OpenCUA-7B reaches the second highest performance among all baseline models, demonstrating that increasing parameter size cannot always improve the action performance, i.e., drags and clicks, of computer use agents, and lightweight models such as OpenCUA-7B and ShowUIπ are worth further development.

Offline Evaluation. As shown in Tab. 1, endpoint accuracy and trajectory error are computed in offline evaluation

Chunk Size

Exec. Steps Offline eval.

before next observation Endpoint Acc. (↑) Traj. Error (↓)

length(aˆ)

- 1 75.45 176.43
- 2 72.28 193.81 5 66.34 241.60

10

- 1 78.55 159.05
- 2 76.35 177.87 5 79.22 205.23

20

- Table 3. Ablation on Chunk Size and Execution Steps. Two chunk sizes (n ∈ {10, 20}); each evaluated at execution steps {1, 2, 5}. We report offline Endpoint Acc. and Traj. Error.

w

Online Success Rate (%, ↑)

OS PPT Premiere Captcha Handwriting Overall

1 12.87 11.49 10.77 7.41 9.90 10.49 5 11.69 17.30 7.84 19.40 16.22 14.49 10 13.11 22.93 8.64 55.91 34.32 26.98 15 26.73 11.49 7.69 33.33 24.75 20.80

- Table 4. Effects of Temporal Weights. Removing temporal weights reduces online success across domains.

Method #Params

Drag

Online SR. Offline Acc.

Separate Heads 550M 23.25 79.22 Unified Head 450M 26.98 78.55

- Table 5. Unified vs. Separate Action Heads. Unified head provides comparable performance to separate heads, while being more elegant and practical.

Directional Reg.

Online Success Rate (%, ↑)

OS PPT Premiere Captcha Handwriting Overall

w/o. (λ = 0) 10.83 14.37 8.26 14.92 14.78 12.63 w. (λ = 0.1) 13.11 22.93 8.64 55.91 34.32 26.98

- Table 6. Effects of Directional Regularization. Directional regularization improves online success across all domains, especially in domains that are sensitive to drag direction, e.g.,, Captcha.

to evaluate whether the trajectory closely aligns with the ground-truth trajectory. For baseline models which output actions as language, only the endpoints are used to compute the trajectory error. For ShowUI-π, all waypoints are used to evaluate whether the model learned to closely follow the trajectory. It can be observed from Tab. 1 that ShowUI-π produces the least trajectory error, showing that the flowbased VLA can follow the learned trajectory pattern well.

Which action modeling approach is most effective? Flow matching delivers the highest endpoint accuracy of 78.55% and lowest trajectory error of 159.05px, surpassing diffusion policy by 31.22% and lower error, and language modeling by 78.15% and much lower error as shown in Tab. 7.

This demonstrates flow matching’s deterministic velocity field effectively captures complex, free-form GUI drags with its training stability.

##### 5.2. Key Ablations

How do chunk size and execution steps impact model performance? As shown in Tab. 3, increasing chunk size, i.e., length of steps in each prediction, from 10 to 20 generally boosts the performance, indicating predicting more action steps is helpful for learning to capture the action distribution. Interestingly, when the execution steps are large, i.e., executing more steps from prediction before the next re-observation, increasing chunk size can help alleviate the degradation. When the execution steps are 5, increasing chunk size from 10 to 20 can improve its accuracy by 12.9% and reduce trajectory error by 36.4px. Moreover, fewer execution steps generally produce the highest accuracy and least error, showing that frequent re-observation enhances precision. Therefore, chunk size 20 with execution step 1 achieves the tradeoff for reliable and precise actions.

How does applying reweighting affect model performance? As shown in Eq. 6, reweighting applies stepdependent coefficients to the flow matching loss across the predicted action horizon, prioritizing accuracy at important steps, i.e., after drag starts and near drag ends. Reweighting at scale 10 yields the highest overall success rate, increasing from 10.49% without weighting to 26.98%, with dramatic gains on Captcha by 48.50%, showcasing the importance of starting and ending steps for successful drags. Interestingly, overweighting, i.e., with scale 15 benefits OS File drag but reduces overall performance, showing that overweighting may harm learning of intermediate actions, and moderate weighting optimally learns the entire drag, shown in Tab. 4.

Unified action head or separate heads for drags and clicks? As shown in Tab. 5, the unified head matches separate heads in offline accuracy at 78.55% versus 79.22%, while improving online drag success by 3.7% and reducing model size by 100M parameters. Though the two designs have comparable performance, the unified design is more elegant, and allows ShowUI-π to handle both drag and click tasks without task-dependent head selection. Conversely, the separate head design introduces impractical head selection and requires an additional 100M parameters, resulting in higher computational overhead.

Is Directional Regularization helpful? It can be observed from Tab. 6 shows that directional regularization consistently improves ShowUI-π across all domains, as GUI drag tasks demand smooth and directionally correct trajectories, e.g.,, Captcha-solving may fail to place the slider or rotate to the wrong angle if the drag direction is inaccurate.

Offline Evaluation. Avg. Endpoint Accuracy (↑) / Trajectory Error (↓).

Trajectory modeling

OS PowerPoint Premiere Captcha Handwriting Overall

SmolVLM 1.98 / 566.98 0.00 / 305.64 0.00 / 493.34 0.00 / 254.14 0.00 / 440.42 0.40 / 412.10 Diffusion Policy 43.47 / 553.66 45.17 / 154.11 29.70 / 347.88 64.44 / 203.46 53.56 / 80.49 47.33 / 267.92 Flow Matching (Our) 61.39 / 350.55 85.06 / 57.68 56.92 / 195.96 96.30 / 136.47 93.07 / 54.57 78.55 / 159.05

- Table 7. Ablation of different modeling approaches under the same SmolVLM backbone. We compare language modeling, diffusion policy, and flow matching, all trained on the same 20K training trajectories. Our flow-matching delivers the most effective performance.

Domain Task Instruction

Trajectory Visualization

PowerPoint

Rotate center Fox clockwise by 45 degrees.

[Figure 90]

Handwriting

Write “Starlit grin” on the canvas

[Figure 91]

Adobe Premiere Pro

Apply Unsharp Mask effect to Space clip

[Figure 92]

Captcha

Solve the rotate Captcha

[Figure 93]

- Table 8. Illustration of ShowUI-π predicted trajectories across four domains. ShowUI-π generates smooth, human-like trajectories that closely follow the instructed paths.

corpus of 20K dense trajectories, ShowUI-π enables precise, real-time cursor adjustment that existing discreteaction GUI agents cannot achieve. To support evaluation, we release ScreenDrag, a comprehensive benchmark covering five domains with both offline and online rollout protocols. Extensive experiments show that current proprietary GUI agents still struggle with continuous manipulation, whereas ShowUI-π sets a leading results with only 450M parameters. We hope this work lays the foundation for building GUI agents with truly human-like dexterity.

Limitations and Future works. We trained ShowUI-π at a small model size and limited training data scale. In our future work, we plan to scale up the model size with more parameters and also larger training data scale from our data collection pipeline and external data. Meanwhile, We will explore text-centric planning integration with ShowUI-π.

#### References

- [1] Anthropic. Claude sonnet 4.5 system card, 2025. Accessed: 2025-11-13. 2
- [2] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 3
- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A Vision–Language– Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024. 2, 3
- [4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2, 3
- [5] ByteDance Seed. Seed1.6, 2025. Accessed: 2025-11-11. 7
- [6] ByteDance Seed. Ui-tars, 2025. Accessed: 2025-11-11. 2, 7
- [7] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317,

2024. 6

- [8] Ziyu Chen, Wei Sun, Junnan Li, Li Yuan, Xiaodong Wang, and Jianwei Yang. Assistgpt: A general multi-modal assis-

##### 5.3. Qualitative Analysis

In Fig.8, we demonstrate how ShowUI-π produces precise trajectory generation aligned with task intent across different domains. In PowerPoint, the model accurately performs the required 45-degree clockwise rotation, producing a trajectory consistent with the instructed geometric transform. In Handwriting, it generates a smooth, continuous stroke that clearly forms the intended “S” shaped curve in “Starlit grin.” For Captcha, the model exhibits fine-grained control, stopping at the exact rotation angle needed to solve the puzzle rather than overshooting. These results collectively show that ShowUI-π can follow continuous, high-degreeof-freedom motions with instruction-following.

#### 6. Conclusion

We introduce ShowUI-π, the first flow-based generative GUI agent designed for continuous trajectory control. By unifying discrete clicks and continuous drags within a shared flow-based formulation, and by training on a large

- tant that can plan, execute, inspect, and learn. arXiv preprint arXiv:2309.11456, 2023. 3
- [9] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 2, 3
- [10] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024. 2
- [11] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, L´eo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024. 2
- [12] Hiroki Furuta, Ofir Nachum, Kuang-Huei Lee, Yutaka Matsuo, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models. arXiv preprint arXiv:2305.11854, 2023. 2
- [13] Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, et al. Assistgui: Task-oriented desktop graphical user interface automation. arXiv preprint arXiv:2312.13108, 2023. 3
- [14] Google AI. Gemini 2.5 computer use (cua), 2025. Accessed: 2025-11-11. 7
- [15] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024. 2
- [16] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A realworld webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856,

2023. 2

- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [18] Youngwoon Lee, Aviral Kumar, Yingjie Miao, Quan Vuong, Xinyang Geng, Wenshuai Zhao, Hao Su, Yevgen Chebotar, Sergey Levine, Stephen Tian, Lisa Lee, Younggyo Seo, Eric Jang, Jie Tan, Jaehoon Lee, Soroush Nasiriany, Katie Kang, Yunshuang Li, Sergey Ioffe, and Kimin Lee. OpenVLA: An open-source vision-language-action model. In The Thirteenth International Conference on Learning Representations, 2024. 2, 3
- [19] Ronghui Li, Jing Huo, Geng Yin, Chaoyue Han, Pan Zhou, Tong Zhang, Jie Sun, Xiaoyong Guo, Rong Zhao, Sicheng Lin, et al. Llara: Language- and layout-driven action reasoning agent for multimodal instruction following. arXiv preprint arXiv:2407.15222, 2024. 3
- [20] Shang-Wen Li, Yujia Li, Haotian Wang, Yizhuo Zhang, Wei Chen, Chen Zhu, Junnan Li, Xiaodong Wang, and Jianwei Yang. Ferret-ui: Grounded mobile ui understanding with multimodal llms. arXiv preprint arXiv:2406.01234, 2024. 3

- [21] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for generalist gui agent. In NeurIPS 2024 Workshop on Open-World Agents, 2024. 2, 6
- [22] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3, 6
- [23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
- [24] Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021. 5
- [25] Andr´es Marafioti, Orr Zohar, Miquel Farr´e, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025. 5
- [26] Yicheng Niu, Jason L Lee, Jinhyuk Lee, Alex Krizhevsky, and Honglak Lee. Llarva: Vision-action instruction tuning for multimodal robot policy learning. arXiv preprint arXiv:2406.11815, 2024. 3
- [27] OpenAI. Gpt-5 system card, 2025. Accessed: 2025-11-13. 2
- [28] OpenAI. Operator, 2025. Accessed: 2025-11-11. 7
- [29] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. arXiv preprint arXiv:2501.09747,

2025. 3

- [30] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: A Vision–Language–Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025. 2, 3
- [31] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. 4
- [32] Qwen Team. Qwen3-vl, 2025. Accessed: 2025-11-11. 7
- [33] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844,

2025. 5

- [34] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using

- nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 3
- [35] Haoyu Wang, Yichen Wang, Qian Li, Ke Xu, Han Zhao, Li Zhang, and Fisher Yu. Seeclick: Learning to act within 3d environments via gui grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 3
- [36] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024.
- [37] Xiaoyang Wang, Tianyi Zhang, Ming Li, Yang Zhou, Hao Wang, Yuchen Guo, Yanjie Sun, Song Zhang, Xing Liu, Bei Jiang, et al. Cogagent: Visual interactive new media large ai model for tools optimization. arXiv preprint arXiv:2312.08914, 2023. 2, 3
- [38] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123,

2025. 2, 7

- [39] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022. 3
- [40] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient visionlanguage-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025. 3
- [41] Norbert Wiener. Cybernetics: or Control and Communication in the Animal and the Machine. The Technology Press, Cambridge, MA, 1948. 4, 5
- [42] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024. 2
- [43] Jianwei Yang, Pengchuan Gao, Chong Zheng, Zhiyu Dou, Linjie Zhang, Xiaodong Wang, Lu Yuan, Jianfeng Wang, Changhu Li, Li Wei, et al. Gpt4tools: Using large language models as tools to think and act. In The Twelfth International Conference on Learning Representations, 2024. 3
- [44] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14203–14214, 2025. 2, 3
- [45] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022. 2
- [46] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Cao, Jun Da, Karthik Kim, Yejin Choi, Denny Yang, Sharan Narang, et al.

- React: Synergizing reasoning and acting in language models. In International Conference on Machine Learning, 2023. 3
- [47] Andy Zeng, Pete Florence, Akhil Viswanathan, Michael Iuzzolino, Yasemin Yilmaz, Brian Ichter, Danny Driess, Deepak Pathak, Kuang-Huei Lee, Igor Mordatch, Pieter Abbeel, Jonathan Tompson, Andrei Rusu, Corey Lynch, Pierre Sermanet, Justin Fu, and Sergey Levine. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2023. 3
- [48] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, et al. Ufo: A ui-focused agent for windows os interaction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 597–622, 2025. 4
- [49] Zihan Zheng, Wenhai Wang, Shaohui Huang, Jinlong Li, Pengchuan Gao, Xiyang Dai, Yutong Chen, Yixin Sun, Yutong Wang, Xiaokang Chen, et al. Seeact: Bridging vision and action for generalist agents on gui tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3
- [50] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023. 2
- [51] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 10838–10845. IEEE, 2025. 3
- [52] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2, 3

#### A. More Results

##### A.1. Grounding Performance of ShowUI-π

To evaluate the grounding performance of ShowUI-π as a VLA unifying both discrete click and continuous drag actions, we evaluate ShowUI-π on one of the most challenging grounding benchmarks, ScreenSpot-Pro. During the grounding evaluation, we use the first coordinate in the generated action chunk as the model prediction for grounding. As demonstrated in Tab. 9, ShowUI-π achieves performance comparable to much larger baseline models. Due to the small parameter size (450M), the vision capacity of ShowUI-π is weaker than other models, thus restricting the grounding performance.

Development Creative CAD Scientific Office OS Avg

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

|Model|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|
|---|---|---|---|---|---|---|---|
|OSAtlas-7B|33.1 1.4 17.7|28.8 2.8 17.9|12.2 4.7 10.3|37.5 7.3 24.4|33.9 5.7 27.4|27.1 4.5 16.8|28.1 4.0 18.9|
|UGround (7B)|26.6 2.1 14.7|27.3 2.8 17.0|14.2 1.6 11.1|31.9 2.7 19.3|31.6 11.3 27.0|17.8 0.0 9.7|25.0 2.8 16.5|
|AriaUI (MOE, 3.9B active)|16.2 0.0 8.4|23.7 2.1 14.7|7.6 1.6 6.1|27.1 6.4 18.1|20.3 1.9 16.1|4.7 0.0 2.6|17.1 2.0 11.3|
|CogAgent (18B)|14.9 0.7 8.0|9.6 0.0 5.6|7.1 3.1 6.1|22.2 1.8 13.4|13.0 0.0 10.0|5.6 0.0 3.1|12.0 0.8 7.7|
|ShowUI (2B)|16.9 1.4 9.4|9.1 0.0 5.3|2.5 0.0 1.9|13.2 7.3 10.6|15.3 7.5 13.5|10.3 2.2 6.6|10.8 2.6 7.7|
|ShowUI-π (450M)|9.7 5.5 7.7|10.1 2.8 7.0|1.0 0.0 0.8|9.7 6.4 8.3|10.2 1.9 8.3|3.7 3.4 3.6|7.5 3.8 6.1|
|OSAtlas-4B|7.1 0.0 3.7|3.0 1.4 2.3|2.0 0.0 1.5|9.0 5.5 7.5|5.1 3.8 4.8|5.6 0.0 3.1|5.0 1.7 3.7|
|MiniCPM-V (7B)|7.1 0.0 3.7|2.0 0.0 1.2|4.1 1.6 3.4|8.3 0.0 4.7|2.8 3.8 3.0|3.7 1.1 2.6|4.5 0.7 3.0|
|Qwen2-VL-7B|2.6 0.0 1.3|1.5 0.0 0.9|0.5 0.0 0.4|6.3 0.0 3.5|3.4 1.9 3.0|0.9 0.0 0.5|2.5 0.2 1.6|
|SeeClick (7B)|0.6 0.0 0.3|1.0 0.0 0.6|2.5 0.0 1.9|3.5 0.0 2.0|1.1 0.0 0.9|2.8 0.0 1.5|1.8 0.0 1.1|
|GPT-4o|1.3 0.0 0.7|1.0 0.0 0.6|2.0 0.0 1.5|2.1 0.0 1.2|1.1 0.0 0.9|0.0 0.0 0.0|1.3 0.0 0.8|
|QwenVL-7B|0.0 0.0 0.0|0.0 0.0 0.0|0.0 0.0 0.0|0.7 0.0 0.4|0.0 0.0 0.0|0.0 0.0 0.0|0.1 0.0 0.1|

Table 9. Performance breakdown of various models across application categories on ScreenSpot-Pro. Model Drag

parable to much larger baseline models, as demonstrated in Tab. 10.

Dist. ↓ Recall ↑

##### A.3. Effects of Co-training with Drag Data and Grounding Data

CogAgent (18B) 44.7 0.0 Qwen-VL-Max (72B) 42.0 0.3 Gemini-Pro-Vision 40.8 0.0 Claude-3-Opus 30.6 1.7 GPT-4-Turbo 31.3 1.4 ShowUI-π (450M) 41.6 1.7 GPT-4o 21.9 2.5

We also evaluate the effects of training ShowUI-π on the combination of Drag data and Grounding data. As shown in Fig. 12 and Fig. 11, training on both types of actions leads to the highest performance on both ScreenDrag online evaluation and Screenshot-Pro benchmarks. Interestingly, after training on ScreenDrag drag data, the performance on the Creative and Office categories in ScreenSpot-Pro are boosted, as there is a large amount of data in ScreenDrag within the corresponding domains, e.g.,, PowerPoint and Premiere Pro. Moreover, training on grounding data can help the model learn a better representation, thus bringing the gain. As the grounding data contains only Click data, the model trained on grounding data solely is unable to generate valid drag trajectories, thus failing the drag tasks.

- Table 10. Drag performance of ShowUI-π and baseline models. Following VideoGUI’s setting, two metrics are used in the evaluation. The distance (Dist) is normalized and then multiplied by 100, reflecting the average offset between the predicted and ground-truth drag endpoints. The recall (Rec) is the percentage of drags whose predicted start and end points both fall within a 100pixel threshold of the corresponding ground-truth endpoints. Note that VideoGUI-Action is image-based and only provides the initial screenshot.

##### A.2. Drag Performance of ShowUI-π on public benchmark

#### B. Setup B.1. Training Details

To evaluate the drag performance of ShowUI-π on public dataset, we evaluate ShowUI-π on the Drag subset of VideoGUI-Action benchmark where drag performance is specifically reported. As the benchmark is image-based, only one screenshot is provided as the visual input. Therefore, we let ShowUI-π generate the entire action chunk at once and use it as the model prediction for dragging, without observations on-the-fly or continuous actions. This restricts ShowUI-π’s performance, as its drag capability is trained on the continuous observations from ScreenDrag only, without the drag data from public datasets, thus leading to a gap between training and public benchmark evaluation. However, ShowUI-π still achieves performance com-

We utilize 4 H200 GPUs for training. The batch size per GPU is set to 64, without gradient accumulation. We use bfloat16 precision for training. The vision encoder and language model part are initialized with the weights from SmolVLM, and other components, e.g.,, action state embedding, action expert, etc., are randomly initialized. The model is trained end-to-end thanks to its small parameter size. To enhance efficiency, we resize the visual observation to (1024,576) from the common (1920,1080), reducing the vision encoder overhead, while maintaining the 16 : 9 scale. We leverage DeepSpeed Zero-2 as the training framework. The learning rate is configured to 1e-4.

###### Development Creative CAD Scientific Office OS Avg

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

|Data Recipe|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg|Text Icon Avg<br><br>|Text Icon Avg|
|---|---|---|---|---|---|---|---|
|Drag Grounding Drag + Grounding<br><br>|1.3 0.0 0.7<br><br>8.4 4.8 6.7<br><br>9.7 5.5 7.7<br><br><br>|0.5 2.8 1.5 8.1 2.1 5.6<br><br>10.1 2.8 7.0<br><br>|1.0 1.6 1.1<br><br>0.5 0.0 0.4<br><br>1.0 0.0 0.8<br><br><br>|0.7 0.0 0.4 12.5 5.5 9.4<br><br>9.7 6.4 8.3<br><br>|0.6 3.8 1.3 8.5 1.9 7.0<br><br>10.2 1.9 8.3<br><br>|2.8 0.0 1.5 7.5 7.9 7.7<br><br>3.7 3.4 3.6<br><br><br>|1.0 1.2 1.1 7.3 4.0 6.0 7.5 3.8 6.1<br><br>|

- Table 11. Performance breakdown of models trained using different data recipe across application categories on ScreenSpot-Pro.

Data Recipe

Online Evaluation. Avg. Success Rate(↑).

OS PowerPoint Premiere Captcha Handwriting Overall

Grounding 0.00 0.00 0.00 0.00 0.00 0.00 Drag 20.79 10.34 4.62 48.15 25.74 21.92 Drag + Grounding 13.11 22.93 8.64 55.91 34.32 26.98

- Table 12. Performance breakdown of models trained using different data recipe across different domains on ScreenDrag online evaluation.

- B.2. Training Data

We use a smaller training corpus for ShowUI-π compared to other models. Specifically, in addition to the training data from ScreenDrag, we use the desktop Click subset of GUIAct, WaveUI, UGround, and ShowUI-Desktop. We did not use any non-Click data from any public dataset, nor did we use any mobile data for training.

- C. Dataset Construction

- C.1. How we collect raw data

We construct raw demonstration data across five domains: PowerPoint, OS Desktop and File Manager, Captcha, Handwriting, as well as Adobe Premiere Pro. Across all domains, data are collected on Windows machines where we record high-frequency mouse events and screen video recordings, using our ScreenDrag data pipeline. To obtain the UI metadata, the DOM is used for Captcha on webpages and the UIA framework is used for other domains, so that each trajectory can be paired with precise element locations and attributes.

PowerPoint. For the PowerPoint domain, we start from the official Microsoft template gallery. We crawl and download a diverse set of slide templates spanning different layouts, color schemes, and typography. For each template, we automatically parse the slide metadata, e.g.,, textboxes, images and shapes, using UIA, and then design manipulation tasks such as rotating and different types of resizing. The position of rotation handle is calculated using heuristics, as its position is not in the UIA metadata.

Captcha. For Captcha tasks, we build the automated data collection pipeline on an open-source library Go-Captcha. We configure the pipeline to generate interactive Captchas such as sliders and puzzle pieces embedded in a webpage,

and the Captcha will be refreshed and regenerated after the previous one is solved. We modify the Captcha library codebase so that the task metadata, e.g.,, puzzle piece position, target position, etc., and the task status, e.g.,, success or failure are accessible at real-time. Therefore, we can filter the successfully solved tasks and collect their trajectories.

Handwriting. For the Handwriting domain, we build upon an open-source handwriting synthesis library. We sample diverse names, short phrases from a Qwen2.5-72B endpoint deployed using VLLM, then the handwriting trajectory from the sampled text will be generated using the handwriting synthesis library with varied stroke styles. Afterwards, the trajectory will be written on the canvas through Win32 mouse interface for fine-grained control.

Adobe Premiere Pro. In the Premiere Pro domain, we capture human demonstration trajectories in creative workflows instead of automation, due to the limitation that Premiere Pro UI metadata is not fully accessible. We recruit two student annotators experienced with video editing to design tasks that reflect common real-world operations, such as trimming clips, adjusting layers on the timeline, arranging clips, and applying effects. The expert demonstrations are recorded using our recorder that can capture highfrequency mouse events with low latency, aligned with the video recording timestamps.

OS Desktop and File Manager. For OS Desktop and File Manager tasks, we build an automated pipeline that creates different types of files and folders on the desktop or file manager windows, then performs drags and records. To increase the task difficulty, we modify the Windows registry so that files and folders created can be placed anywhere on the desktop without automatic arrangement, and each time multiple files and folders with different names will be generated. UIA is used to obtain bounding boxes for desktop

icons, folders, and window controls.

Data Recording. Across all domains, we use OBS for the screen recording, as it is well optimized and has less latency compared to other screen capture methods, e.g.,, FFmpeg.

Data Generation Codebase. To contribute to the community, we will open-source the data generation and recording codebase. Moreover, some of the data crawl sources, e.g.,, the Microsoft PowerPoint Template Gallery, have changed and make the template collection more difficult, i.e., only a limited number of templates will be shown without search queries, therefore, we will also provide our collected raw data.

##### C.2. Data visualization

The visualization of some task trajectories are shown in Tab. 13

##### C.3. Data-driven Closed-loop Online Evaluation

As mentioned in the main paper, a data-driven approach is designed to enable closed-loop rollouts in online evaluation. For each drag task, we store the video recording, task specification, and dense drag trajectory, providing extensive possible GUI states encountered during dragging. During rollouts, the model’s predicted action is matched to the nearest recorded state if it falls within a tolerance ϵ (e.g., within 20 pixels of a ground-truth waypoint), upon which the corresponding next observation is retrieved. As shown in Fig. 7, when the model prediction midway can be mapped to a trajectory point, its corresponding UI state will be retrieved from the extensive pre-collected UI states as the next observation, thus the model can perform continuous actions with observations on-the-fly. This data-driven approach largely reduces the complexity of setting up OS and software applications, enhancing reproducibility, while still enabling a closed-loop rollout manner.

#### D. Failure Cases of Baseline Models

To better illustrate the behavior of baseline models in drag tasks, we analyze eight failure cases observed during evaluation. These qualitative examples highlight problems encountered by existing GUI agents based on languagemodeling.

(i) Know-How but does not have the tool. As shown in Fig. 8, the PowerPoint rotation task expects the model to rotate elements using the white rotation handle on the target elements with an arc trajectory. Interestingly, both OpenCUA-7B and Operator frequently produce the correct first step, successfully locating the handle. However, they are only equipped and trained with linear drag tools, e.g.,, Drag((x1,y1),(x2,y2)) and Linear Drag((x1,y1),(x2,y2)), etc. Therefore, they are unable to perform rotation even if they know how to rotate, as they do not have the corresponding tool.

This highlights a major limitation of GUI agents based on language-modeling: Beyond drag in an arc trajectory, there are various types of free-form drags in real-world scenarios, which are extremely difficult to be fully covered by defining drag tools.

- (ii) Gap between discrete tool use and continuous drag. As shown in Fig. 9, on desktop drag tasks, both the Seed-1.6-Vision baseline and Qwen3-VL-32B frequently produce correct high-level plans such as “drag BudgetNotes.txt into TeamDocs”, successfully locate the icon position, and the resulting trajectory starts in the right direction. However, the cursor often stops midway or lands noticeably short of the folder, leaving a large distance to the target. This case illustrates a gap between discrete tool use and continuous drag: language-modeling baselines are designed to predict one discrete tool call each time, e.g.,, drag files using one Drag() call, instead of continuous actions with on-the-fly observation, thus they cannot finely adjust the cursor along the path, often halting mid-way.
- (iii) Safety over action. As shown in Fig. 10, under the rotate-captcha tasks, Gemini 2.5 computer-use often recognizes the Captcha and starts a drag, but then halts and emits a safety refusal, declaring that it cannot perform the action. For example, when performing Captcha-solving tasks, it will say I see that the next action is to interact with a CAPTCHA. I am unable to solve CAPTCHAs and need you to complete it for me. These behaviors expose an alignment tax: safety filters and RLHF objectives tuned for general-purpose chat misclassify benign UI manipulations, e.g.,, captcha solving as risky or inappropriate, so the agent learns that refusing to act is safer than executing the requested drag. However, such tasks are common and valuable for GUI agents.
- (iv) Semantic misread. As shown in Fig. 11, in handwriting-style episodes such as “Write Alice Brown on canvas”, the Seed-1.6-Vision baseline is instructed to write the name on the canvas, but the trajectory moves toward a window control in the corner instead. Here the instruction is short and unambiguous, the pen tool has already been selected, yet the agent behaves as if the task were to manage the window rather than interact with the canvas. This suggests that strong priors from standard GUI layouts, i.e., menus, close buttons, toolbars, dominate over other UI elements and tasks, leading the model to favor canonical UI elements over the specific target indicated in the prompt.
- (v) Wrong primitive choice. As shown in Fig. 12, for Qwen3-VL-2B/8B, a common failure mode is to approximate a drag as a series of local left click actions, as illustrated in Fig. 12. On both desktop drag and slidercaptcha tasks, the model repeatedly clicks on the element instead of committing to a continuous drag, so the cursor never moves the required distance. This behavior reflects an

Table 13. Examples of task trajectories from five domains. Three frames from the episode are shown for each task.

###### Domain Task Initial Intermediate Final

[Figure 106]

[Figure 107]

[Figure 108]

Solve the Drag-and-Drop Captcha

[Figure 109]

[Figure 110]

[Figure 111]

Captcha

Solve the Rotate Captcha

[Figure 112]

[Figure 113]

[Figure 114]

Solve the Slider Captcha

[Figure 115]

[Figure 116]

[Figure 117]

Drag Analysis.xlsx to MyProject

Desktop

[Figure 118]

[Figure 119]

[Figure 120]

Drag Q1Report to projectDocs

[Figure 121]

[Figure 122]

[Figure 123]

HandwritingWrite “Hello World” on canvas

[Figure 124]

[Figure 125]

[Figure 126]

Rotate the Lion counter-clockwise by 45 degrees

[Figure 127]

[Figure 128]

[Figure 129]

Resize the title Basic Presentation diagonally by 0.5 from top-left corner

PowerPoint

[Figure 130]

[Figure 131]

[Figure 132]

Resize width of the textbox December to 0.2 from its right

[Figure 133]

[Figure 134]

[Figure 135]

Apply Vertical Flip effect to Great Forest clip

Premiere

[Figure 136]

(a) The model predicts a coordinate close to the dense trajectory points.

[Figure 137]

(b) The prediction is mapped to its closest trajectory point.

[Figure 138]

(c) The model receives the next observation at the mapped trajectory point.

- Figure 7. The visualization of the data-driven closed-loop online evaluation process. This approach enables models to perform continuous actions with observations on-the-fly, without the complexity to set up OS and software applications, enhancing reproducibility.

[Figure 139]

- Figure 8. Know-How but does not have the tool. The baseline formulates a correct plan to rotate the textbox by dragging the handle above the textbox ”AGENDA” with an arc trajectory, however, the baseline is not equipped with such a drag tool, it is only trained and equipped with linear drags, thus failing the task.

[Figure 140]

- Figure 9. Gap between discrete tool use and continuous drag. The baseline formulates a correct plan to drag the file icon to the folder and successfully locates the file icon’s initial position, but the execution fails mid-trajectory, leaving the icon stranded far from the target.
- Figure 10. Safety over action. The model initiates a drag on the captcha slider but immediately halts and issues a refusal, misinterpreting the standard UI interaction as a safety violation.

[Figure 141]

- Figure 11. Semantic misread. The model misinterprets the visual instruction, moving the cursor to a non-target corner instead of the canvas, indicating a failure in task understanding.

[Figure 142]

inductive bias inherited from GUI pre-training data, where discrete clicks are the dominant interaction primitive; when transferred to tasks that require continuous actions, the model keeps reaching for the familiar click action and never fully enters the drag regime.

(vi) Direction right, magnitude wrong. As shown in Fig. 13, OpenCUA-32B typically produce PyAutoGUI tool

[Figure 143]

- Figure 12. Wrong primitive choice. Instead of a continuous drag action required for the slider, the baseline issues a series of discrete clicks, failing to execute the task.

[Figure 144]

- Figure 13. Geometric precision. The predicted trajectory follows the correct direction but significantly overshoots the target, highlighting a lack of fine-grained action control.

calls that move in the correct direction but misestimate how far to drag. Handles and icons are moved broadly along the right axis, yet the final position overshoots the target folder.

[Figure 145]

- Figure 14. Dialogue hijacks control. The model pauses execution to ask for unnecessary clarification, halting progress in an embodied setting where autonomous action is expected.

(vii) Dialogue hijacks control. As shown in Fig. 14, under the Captcha-solving tasks, OpenAI Operator baseline often

executes at most one tentative click and then pauses to ask the user to complete the task for it, instead of continuing finishing the task. For example, it will say I see a dragand-drop captcha on the screen. Can you please complete it? This behavior exposes an objective mismatch: RLHF tuning for helpful conversation encourages asking clarifying questions, but benchmarks expect autonomous problem solving with no human in the loop.

[Figure 146]

Figure 15. Early termination. The model terminates the episode immediately with an “Instruction Unclear” error, refusing to attempt the task.

(viii) Early termination. In some handwriting tasks, when the model is instructed to write a phrase on the canvas, UI-TARS-1.5-7B sometimes immediately terminates an episode with an “Instruction Unclear” message and executes no further actions. While this is reasonable as a safety mechanism in open-ended dialogue, in benchmarks it produces deterministic failures on tasks that are diverse but still clearly solvable from visual context.

Why ShowUI-π differs. As a lightweight flow-based VLA, ShowUI-π is capable of generating continuous actions with observations on-the-fly, performing free-form drags without relying on pre-defined tools. Therefore, it can adjust the fine-grained cursor motion along the task execution, addressing the problems encountered by baseline models, e.g.,, problems in Case (i), Case (ii), Case (v), and Case (vi).

#### E. Limitations and Future Work

We trained ShowUI-π at a small model size and limited training data scale. In our future work, we plan to scale up the model size with more parameters and also larger training data scale from our data collection pipeline and external data. Meanwhile, We will explore text-centric planning integration with ShowUI-π.

