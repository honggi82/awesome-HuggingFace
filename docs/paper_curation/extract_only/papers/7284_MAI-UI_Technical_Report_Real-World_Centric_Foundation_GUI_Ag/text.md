2025-12-29

###### MAI-UI Technical Report: Real-World Centric Foundation GUI Agents

Hanzhang Zhou∗, Xu Zhang∗, Panrong Tong, Jianan Zhang, Liangyu Chen, Quyu Kong Chenglin Cai, Chen Liu, Yue Wang() , Jingren Zhou, Steven HOI

[Figure 1]

Tongyi Lab , Alibaba Group

https://github.com/Tongyi-MAI/MAI-UI

###### Abstract

The development of GUI agents could revolutionize the next generation of human-computer interaction. Motivated by this vision, we present MAI-UI, a family of foundation GUI agents spanning the full spectrum of sizes, including 2B, 8B, 32B, and 235B-A22B variants. We identify four key challenges to realistic deployment: the lack of native agent–user interaction, the limits of UI-only operation, the absence of a practical deployment architecture, and brittleness in dynamic environments. MAI-UI addresses these issues with a unified methodology: a self-evolving data pipeline that expands the navigation data to include user interaction and MCP tool calls, a native device–cloud collaboration system routes execution by task state, and an online RL framework with advanced optimizations to scale parallel environments and context length.

# arXiv:2512.22047v1[cs.CV]26Dec2025

MAI-UI establishes new state-of-the-art across GUI grounding and mobile navigation. On grounding benchmarks, it reaches 73.5% on ScreenSpot-Pro, 91.3% on MMBench GUI L2, 70.9% on OSWorld-G, and 49.2% on UI-Vision, surpassing Gemini-3-Pro and Seed1.8 on ScreenSpot-Pro. On mobile GUI navigation, it sets a new SOTA of 76.7% on AndroidWorld, surpassing UI-Tars2, Gemini-2.5-Pro and Seed1.8. On MobileWorld, MAI-UI obtains 41.7% success rate, significantly outperforming end-to-end GUI models and competitive with Gemini-3-Pro based agentic frameworks. Our online RL experiments show significant gains from scaling parallel environments from 32 to 512 (+5.2 points) and increasing environment step budget from 15 to 50 (+4.3 points). Finally, the native device-cloud collaboration system improves on-device performance by 33%, reduces cloud model calls by over 40%, and preserves user privacy.

|MAI-U<br><br>UI-Ven<br><br>Seed1<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]|I<br><br>us<br><br>.8 St<br><br>[Figure 5]<br><br>Qw<br><br>[Figure 6]<br><br>G|ep-GUI<br><br>en3-VL<br><br>TA1| |Scr<br><br>[Figure 7]<br><br>[Figure 8]|73.5<br><br>eenSp|ot-Pro<br><br>[Figure 9]<br><br>72.7<br><br>73.1<br><br>[Figure 10]|
|---|---|---|---|---|---|---|
|62.8<br><br>[Figure 11]<br><br>Gemi|ni-3-Pro|60.0<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>|70.9<br><br>62.6|63.6| |61.9<br><br>[Figure 15]|
| | |50.8|50.1<br><br>[Figure 16]|[Figure 17]|54.9| |
|MAI-U<br><br>[Figure 18]<br><br>UI-Ven Phi-Gr<br><br>[Figure 19]<br><br>Qwen<br><br>[Figure 20]|I<br><br>us ound<br><br>3-VL|[Figure 21]<br><br>|42.4|49|.2 UI-V|ision<br><br>[Figure 22]|
|UI-Tar<br><br>[Figure 23]|31.9<br><br>s|[Figure 24]| | |3<br><br>[Figure 25]|6.8|
| | |26.5<br><br>27.2|[Figure 26]|26.9<br><br>[Figure 27]|2|5.5<br><br>[Figure 28]|

[Figure 29]

76.7

MAI-UI-235B-A22B

[Figure 30]

73.3

MAI-UI-32B

[Figure 31]

[Figure 32]

73.3

UI-Tars-2

GroundingAccuracy(%)

[Figure 33]

70.7

MAI-UI-8B

[Figure 34]

[Figure 35]

70.7 Seed1.8

[Figure 36]

[Figure 37]

69.7

Gemini-2.5-Pro

[Figure 38]

[Figure 39]

- 66.4 65.9 49.1

[Figure 40]

[Figure 41]

28.0

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

41.7

37.3

24.9

[Figure 51]

20.9

[Figure 52]

10.4

9.5

UI-Venus-72B

Qwen3-VL-235B-A22B

Doubao-1.5-UI-TARS

MAI-UI-8B

MAI-UI-32B

MAI-UI-235B-A22B

9.0 Qwen3-VL-32B

5.5 GUI-Owl-32B

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- 67.7 Step-GUI-8B

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

GUI-Owl-7B

9.0 MAI-UI-2B

###### +3.4

+20.8

[Figure 65]

UI-Venus-72B

[Figure 66]

[Figure 67]

MAI-UI-2B

[Figure 68]

Ferret-UI Lite-3B

1 2 4 8 16 32 64 …

Android World (SR)

Mobile World (SR)

Parameter Size (B)

Figure 1: MAI-UI achieves SOTA performance across GUI grounding and navigation benchmarks.

*Lead contributor. All authors made core contributions. Corresponding author. yue.w@alibaba-inc.com

###### 1 Introduction

Graphical user interface (GUI) agent (Wang et al., 2024b) is an agent system that can perceive, reason, and act within GUIs in response to natural language instructions. By translating high-level user intents into concrete UI operations, GUI agents potentially represent a revolution in digital interaction, transforming human-computer interaction from the manual navigation of complex interfaces to goal-oriented natural language control. Motivated by this vision, the community’s increasing efforts have accelerated progress on GUI agents, advancing UI perception, visual grounding, and GUI navigation (Wang et al., 2025a; Ye et al., 2025; Gu et al., 2025c).

Despite rapid technological advances, today’s GUI agents remain insufficient for reliable, robust, and secure deployment in practice. We highlight several open challenges to close this gap. (1) Agent-user interaction: Existing systems are typically optimized for end-to-end execution, however, user instructions in real-world settings are often ambiguous or incomplete. To ensure alignment with user intent, the agent must proactively ask clarifying questions, collect missing details, and seek consent for sensitive operations. Effective agent–user interaction is therefore a critical yet often neglected capability. (2) Beyond UI-only action: Relying solely on UI manipulation poses two issues: (i) the long, multi-step sequences of UI operations increase brittleness to per-step errors and amplify error propagation; and (ii) it limits the agent to tasks that are UI-reachable. Integrating external tools via the Model Context Protocol (MCP) provides structured shortcuts that compress long, fragile UI operation sequences into a few API calls and unlock tasks previously infeasible on mobile. For example, via MCP tools, a mobile agent can manipulate GitHub repositories, bringing traditionally desktop-only workflows to mobile. (3) Native device–cloud collaboration capability: Current GUI agents are typically categorized into lightweight, on-device variants or large models that can only be used as cloud services. However, cloudonly solutions introduce privacy risks, higher costs, and dependence on network connectivity, whereas on-device–only approaches are constrained by model capacity and capability. Consequently, foundation GUI agents lack native device–cloud collaboration capability for privacy-aware and cost-efficient routing and seamless handoff. (4) Robustness to dynamic environments: Agents trained on static, pre-collected trajectories often overfit to specific interface patterns and struggle in out-of-domain scenarios. In practice, real-world GUIs are highly dynamic: layouts vary across app versions and devices, and pop-ups or permission dialogs can appear unexpectedly. Without exposure to dynamic environments in training, agents generalize poorly and remain brittle to real-world unpredictability.

To enhance GUI agent capabilities and address these challenges for realistic deployment, we introduce MAI-UI, a foundational GUI agent for general GUI grounding and mobile navigation.

- • Agent-user interaction and MCP augmentation. To equip GUI agents with user interaction and MCP tool use capability, we introduce a self-evolving data pipeline that incorporates training data for general navigation as well as these two capabilities. The data pipeline iteratively updates both the model and the training corpus using three sources of data: rejection-sampled trajectories, manually-annotated trajectories, and automatic agent rollouts. The action space is also extended to allow the agent to choose among UI manipulation, user engagement, and MCP tool use.
- • Device–cloud collaboration system. For realistic deployment, MAI-UI introduces a pioneering native device-cloud collaborative system, which can dynamically select on-device or cloud execution based on task execution state and data sensitivity. The system consists of a local GUI agent that both act as a GUI agent and as trajectory monitor, a high-capacity cloud GUI agent, and a local unified trajectory memory that maintains consistent information exchange between local and cloud agents.
- • Reinforcement learning in dynamic environments. MAI-UI incorporates online reinforcement learning as a core training component, enabling improvement through interaction with dynamic environments. Our system-level optimizations scale to 500+ GUI environments for parallel rollouts. We further support asynchronous rollout and hybrid parallelism for training, enabling training on long-horizon GUI tasks with up to 50 interactive steps. This training stage yields improved GUI

navigation accuracy and stronger robustness to real-world unpredictability.

MAI-UI includes a full-spectrum of sizes to meet real-world deployment constraints, ranging from efficient 2B on device variants to mid size 8B and 32B models, and large scale 235B-A22B models. Across sizes, our models achieve state-of-the-art performance against strong baselines at comparable scales. Notably, our 2B on-device model achieves a relative improvement of 75.4% over Ferret-UI Lite (Yang et al., 2025b), our mid size 8B and 32B models surpasses GUI-Owl(Ye et al., 2025), Step-GUI (Yan et al., 2025), and UI-Venus(Gu et al., 2025c), and our 235B-A22B variants outperform UI-Tars-2 (Wang et al., 2025a), Seed1.8 (Seed, 2025a), and Gemini-2.5-Pro (DeepMind, 2025a) on AndroidWorld.

MAI-UI sets a new state of the art across diverse evaluation settings, including GUI grounding, offline and online mobile GUI navigation, and a realistic-oriented benchmark that incorporates MCP tool use and agent-user interaction, consistently outperforming prior works.

- • Grounding. Our model establishes new state of the art performance across five well known grounding benchmarks. Notably, MAI-UI achieves 67.9% on ScreenSpot-Pro (73.5% with zoom-in), 91.3% on MMBench GUI L2, 70.9% on OSWorld-G (75.0% on OSWorld-G-Refine), 47.1% on UI-Vision (49.2% with zoom-in), and 96.5% on ScreenSpot-V2, substantially surpassing the strongest counterparts.
- • Offline GUI navigation. On Android Control, MAI-UI attains strong exact match accuracy. On GUI Odyssey, it achieves a success rate of 83.4%, outperforming prior methods.
- • Online GUI navigation. In challenging dynamic real-time environments, MAI-UI-235B-A22B achieves a new state of the art success rate of 76.7% on AndroidWorld, surpassing UI-Tars-2 (Wang et al., 2025a), Gemini-2.5-Pro (DeepMind, 2025a), and Seed1.8 (Seed, 2025a). Additionally, MAI-UI-32B, MAI-UI-8B, and MAI-UI-2B attain success rate of 73.3%, 70.7%, and 49.1% on AndroidWorld, respectively, all outperforming competitive models at comparable scales.
- • Realistic-oriented evaluation. To bridge the gap for more challenging and realistic online assessment, we adopt our MobileWorld benchmark (Kong et al., 2025), which comprises evaluations beyond pure GUI operations. MAI-UI obtains 41.7% success rate, surpassing end-to-end GUI model baselines by

+20.8 and competitive with agentic frameworks with GPT-5 or Gemini-3-Pro as planners. Furthermore, on tasks that require agent-user interaction and MCP tool use, MAI-UI gains a success rate of 37.5% and 51.1%, respectively, representing an absolute increase of +32.1 and +18.7.

###### 2 MAI-UI

This section details the methodology of MAI-UI. Our approach integrates (i) a training paradigm for GUI grounding, (ii) a self-evolving trajectory data pipeline, (iii) training for agent–user interaction and MCP tool augmentation, (iv) online reinforcement learning, and (v) a native device–cloud collaboration system. Figure 2 illustrates a demo trajectory of MAI-UI.

###### 2.1 System Overview

We present a system overview of our model, covering task formulation, the action space, and the model architecture.

- 2.1.1 Task Formulation MAI-UI cover tasks of two categories: general GUI grounding and mobile GUI navigation.

Grounding Task. GUI grounding aims to localize the UI element corresponding to a natural language instruction on a graphical interface (Wang et al., 2024c). Formally, given a GUI screenshot and a natural language instruction I, the GUI agent model predicts a coordinate point P = (x, y) that indicates the location of the target UI element.

Navigation Task. Mobile navigation task can be formulated as a Partially Observable Markov Decision Process (POMDP) (S, O, A, T ) (Qin et al., 2025b), where: S denotes the state space capturing the

[Figure 69]

[Figure 70]

|Action Space<br><br>|[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>… Type<br><br>Click Scroll|
|---|
<br><br>|[Figure 74]<br><br>[Figure 75]<br><br>MCP Tool Call User …|
|---|
<br><br>GUI Action Space<br><br>Extended Actions<br><br>|
|---|

Help me plan a route that departs from the office tomorrow at 8:00 AM. Choose the shortest route, and send a reminder and route details to my colleague Mike.

|[Figure 76]<br><br>[Figure 77]<br><br>I need to gather more information before I can plan a route. The user hasn’t specified the destination, preferred<br><br>transport mode, or how to contact Mike. I’ll prompt the user to provide these details so I can proceed effectively.<br><br>The destination is Shanghai Railway Station. We’ll drive there, and Mike’s email address is mike@alibaba.com<br><br>Initial State<br><br>[Figure 78]<br><br>……<br><br>[Figure 79]<br><br>The user has supplied the destination and contact method. Now I need to generate actionable route options. I’ll use the MCP routing tool to fetch multiple candidate routes. My goal is to compare them by total distance to identify the shortest path<br><br>[Figure 80]<br><br>Call User<br><br>[Figure 81]<br><br>I have the driving route and distances: ……<br><br>MCP Tool<br><br>Final State<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]|
|---|

|Cloud-Device Collaboration<br><br>[Figure 89]<br><br>[Figure 90]<br><br>Cloud Agent<br><br>Local Agent<br><br>GUI Navigation<br><br>GUI Navigation<br><br>Monitoring<br><br>[Figure 91]<br><br>Trajectory Recovery|
|---|

Figure 2: Example trajectory of MAI-UI. MAI-UI completes GUI agent tasks via both UI operations and extended actions, including agent-user interaction and MCP tool use, and integrates a native device-cloud collaboration system.

underlying environment; O represents the observation space, which consists of a natural language instruction I along with one or more screenshots; A defines the action space comprising standard mobile UI operations (e.g., click, swipe or typing); T : S × A → S specifies the state transition of the environment. At time step t, the agent predicts the next action as as at = π(I, ot, ht), where I is the natural language instruction, ot ∈ O is the current observation, and ht = (a1, o1, . . . , at−1, ot−1) denotes the history context of previous actions and observations.

###### 2.1.2 Foundation GUI Agent for Mobile Use

We introduce MAI-UI, a foundation GUI agent with strong GUI grounding and mobile navigation capabilities. It supports effective agent–user interaction for clarification and integrates MCP-based tool augmentation for API use, and features a native device–cloud collaboration system for practical deployment. MAI-UI offers a broad range of model sizes, each specifically trained to deliver robust grounding and navigation capabilities.

Action Space. MAI-UI provides a comprehensive action space for mobile GUI control, including click, swipe, type, wait, etc. By leveraging these actions, MAI-UI can interact with the mobile device through its graphical user interface to complete a wide range of tasks. We also include an answer action to directly respond to user queries in question-answering scenarios. To better support real-world scenarios, we integrate two specialized actions that extend the agent beyond pure GUI operation through active user interaction and MCP tool use. The ask_user action allows the agent to request clarification when the instruction is vague or underspecified. The mcp_call action enables the agent to leverage external MCP tools rather than relying solely on GUI operations. The full action space is shown in Table 1.

Model Architecture. We employ Qwen3-VL (Bai et al., 2025a) as the backbone model. The MAI-UI family spans a full spectrum of sizes, including 2B, 8B, 32B, and 235B-A22B variants, enabling the deployment tailored to hardware constraints and performance requirements. Each model is jointly trained on GUI grounding, perception, and mobile-use navigation data using supervised fine-tuning and

Table 1: Action Space in MAI-UI.

Action Definition click Clicks at coordinates (x, y). long_press Long presses at coordinates (x, y). type Types the specified text content. swipe Swipes in the given direction (up/down/left/right) at optional coordinates. drag Drags from start coordinates (x1, y1) to end coordinates (x2, y2). system_button Presses a system button (back, home, menu, or enter). wait Pauses for a brief moment. terminate Marks the task as complete with status (success or fail). answer Provides an answer with specified text. ask_user Requests user intervention with specified text. mcp_call Provides MCP tool name and corresponding arguments.

reinforcement learning. In addition, MAI-UI integrates a native device–cloud collaboration system that routes computation by task state and data sensitivity.

###### 2.2 GUI Grounding & Perception

GUI grounding and perception are foundational capabilities of GUI agents, enabling the agent to understand screen layouts and to localize the correct UI elements from natural-language instructions. This section presents our data pipeline and training methodology for building these capabilities.

###### 2.2.1 Data Pipeline

As shown in Figure 3, in addition to open-source datasets, our pipeline collects screenshots from real GUI environments to build a robust GUI Agent. This process results in multi-task perception data and multi-perspective grounding data.

Data Collection. To gather diverse GUI data from real-world scenarios, we not only use open-source datasets such as JEDI (Xie et al., 2025a) and OS-Altas (Wu et al., 2024), we also virtualize operating systems in containerized environments. We employed an MLLM-guided exploration strategy to navigate these environments. In every step, we ask MLLMs to identify valid actions from the current state. This procedure continuously changes the interface state and produces new screenshots. Finally, we use the a11y tree or OmniParser V2 (Lu et al., 2024b) to localize UI elements precisely.

Perception Data Generation. To facilitate the diversity in the training data, for each screenshot, we randomly select one to three UI elements as inputs and prompt MLLMs to generate a variety of tasks based on these elements, including question answering, captioning, and state prediction. These diverse tasks allow the model to develop a comprehensive understanding of the interface, specifically enhancing its capabilities in semantic understanding, relation understanding, and layout understanding.

Grounding Data Generation. As demonstrated in our prior work in UI-Ins (Chen et al., 2025), instruction diversity and correctness are critical for GUI grounding but are often neglected:

- • Instruction quality: Approximately 23.3% of instructions in open-source grounding datasets exhibit quality issues, which can actively harm model performance during training.
- • Instruction diversity: Performance improves significantly when training on diverse instruction perspectives, and improves further when the model learns to select appropriate instruction perspective in different scenarios.

Following our Instruction-as-Reasoning paradigm introduced in UI-Ins (Chen et al., 2025), we ask MLLMs to create instructions from four different human-like perspectives, including appearance, function, location and intent. This design is motivated by human UI grounding behavior: people strategically

Source Image Collection

Perception Data Generation

[Figure 92]

Types

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

QA

Virtualize

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Caption

Random Path Selection

Platforms

Container

DataSet Multi-Task Generation

Prediction

Grounding: Instruction-as-Reasoning Data

[Figure 110]

Including

[Figure 111]

[Figure 112]

|[Figure 113]<br><br>[Figure 114]|
|---|

|[Figure 115]<br><br>[Figure 116]|
|---|

Appearance Location

Functionality Intent

Instruction Generation

Verification

OmniParser Detection Random Selection

Figure 3: Overview of grounding and perception data pipeline

switch among instruction perspectives, selecting the most informative one for the task to support effective reasoning. We use these instructions both as inputs and as explicit reasoning pathways, thereby injecting structured reasoning into the model. This training method is introduced in the section below.

###### 2.2.2 Training Paradigm

Algorithm. To build a foundation model with strong grounding capability, we follow our Instructionas-Reasoning paradigm in UI-Ins (Chen et al., 2025). Using the grounding data described above, we first perform supervised fine-tuning (SFT) to instill instruction-as-reasoning capability: utilizing diverse instruction perspectives as explicit analytical reasoning before predicting the coordinates. To encourage dynamic, context-aware selection of the appropriate reasoning perspective across different scenarios, we then conduct a reinforcement learning (RL) stage using the GRPO algorithm.

Reward. For GUI grounding, we use a combination of a format reward and a point-in-box reward. In our experiments, dense reward formulations yield similar performance, so we adopt this simple and effective scheme.

- • Format Reward. We use format reward Rf verify that the model’s response is in a valid format. Specifically, the thinking content and the final answer must be contained in their correct tags, and the coordinates must be extractable for the point-in-box evaluation.
- • Point-in-Box Reward. We utilize a direct point-in-box reward to measure correctness during training. A prediction is considered correct if the predicted coordinate point p = (xp, yp) falls within the

ground-truth bounding box b = (xl, yl, xr, yr), where the (xl, yl) denotes the top-left corner and (xr, yr) represents the bottom-right corner.

 

1 if xl ≤ xp ≤ xr and yl ≤ yp ≤ yr, 0 otherwise.

(1)

Racc =



Zoom-In Strategy. During inference, we introduce an optional zoom-in strategy for complex and high-resolution GUI scenarios. In the first pass, the model predicts a coarse coordinate. We then crop a window centered on this point, with width and height equal to half of the original image dimensions, and resize the crop back to the original resolution. In the second pass, the model refines the prediction by outputting the precise coordinate on the zoomed region.

Navigation Task Generation Trajectory Synthesis

Iterative Rejection Sampling

[Figure 117]

[Figure 118]

###### Diverse Task Expansion

Init Trajectories

training

###### Seed Tasks

APP Manual

[Figure 119]

Human Annotation

###### Model Synthesis

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

SFT-Tuned Model

Expert Design

[Figure 125]

Fine-grained Correctness Judgment

[Figure 126]

Annotation Quality Check

synthesis training

[Figure 127]

[Figure 128]

|Rejection Sampling Trajectories|
|---|

[Figure 129]

[Figure 130]

Open-source Data

Trajectories

###### Data Cleaning

- Figure 4: Overview of the self-evolving data pipeline for trajectory synthesis. The pipeline comprises task generation, trajectory construction via human annotation and autonomous agent rollouts, and iterative rejection sampling that jointly evolve the model and the training corpus.

###### 2.3 Mobile GUI Navigation

Beyond GUI perception and grounding in general setting, MAI-UI excels in mobile GUI navigation tasks. The training pipeline of GUI navigation consists of two main stages: supervised fine-tuning (SFT) and online reinforcement learning (RL). In the first stage, we build a self-evolving data pipeline to collect and synthesize diverse multi-step trajectories, and train the model with these trajectories to obtain strong navigation capability. In the second stage, we enhance the model’s generalization to real-world scenarios through online reinforcement learning in dynamic environments. The following sections provide detailed descriptions of both components.

###### 2.3.1 Supervised Fine-Tuning

MAI-UI employs a self-evolving SFT data pipeline (overview in Figure 4), comprising three key components: Navigation Task Generation, Trajectory Synthesis, and Iterative Rejection Sampling. In the Navigation Task Generation stage, we leverage multiple sources (APP manuals, expert-designed tasks, and open-source data) to construct high-quality seed tasks. In the Trajectory Synthesis stage, we first expand the seed tasks, then combine model-based synthesis and human annotation to generate diverse trajectories. The generated trajectories undergo two quality control steps: fine-grained correctness judgment via automated evaluation and annotation quality checks by human reviewers. This ensures both diversity and quality of the synthesized trajectories. In the final stage, Iterative Rejection Sampling, we initialize training with a set of Stage 2 trajectories as cold-start data to obtain an initial SFT model. We then alternate between fine-tuning the model and deploying the updated policy to rollout new trajectories. Newly generated trajectories are filtered via rejection sampling, retaining only high-quality examples aligned with the model’s evolving capabilities. In parallel, we continually inject new trajectories from Stage 2 to broaden coverage and raise the performance ceiling. This closed loop of training and data synthesis makes both the model and the training corpus self-evolving.

Navigation Task Generation. In this stage, navigation task instructions are derived from three distinct sources: (1) application manuals, from which common usage scenarios are parsed and distilled into intentlevel task descriptions; (2) expert-designed tasks, where human annotators formulate realistic and diverse mobile navigation goals aligned with commonly used scenarios; and (3) open-source datasets, which are filtered by task complexity and reachability. This multi-source strategy expands both task diversity and scale, while rigorous source selection and filtering ensures data quality.

Trajectory Synthesis. In the second stage, we first expand the set of tasks to increase diversity, and then generate trajectory data through parallel pipelines that leverage both model-based rollouts and human annotation.

- • Seed task expansion. Starting from seed tasks, we prompt a multimodal large language model (MLLM) to generate a variety of novel tasks. We categorize this diversity into two levels: L1 adjusts critical parameters of the original task goal, and typical parameters include date/time ranges, numeric thresholds, sorting/filter criteria, etc; L2 replaces the core objects involved in the task while remaining constrained to the same scenario and set of applications.
- • Model synthesis and human annotation. Given the expanded set of tasks, we generate execution trajectories via two parallel pipelines: (1) human annotation: annotators manually perform tasks on an Android emulator and record both screenshots and ground-truth action sequences at each step. (2) model-based synthesis: since human annotation is time-consuming and costly, we also adopt multiple GUI agents to automatically produce valid action sequences for navigation tasks. Notably, for a given task goal, multiple valid execution paths often exist. By combining these complementary sources, we significantly broaden trajectory coverage and enhance dataset robustness.
- • Fine-grained correctness judgment and annotation quality validation. After generating diverse trajectories from both model synthesis and human annotation, we perform quality assessment through two independent validation pipelines:

- 1. Manual quality check: All human-annotated trajectories are reviewed by a second annotator who verifies alignment between the action sequence, screenshots, and the original task goal. Inconsistent or ambiguous demonstrations are either corrected or discarded.
- 2. Fine-grained correctness judgment. Trajectories produced by GUI agents rollouts are examined by an MLLM-as-a-judge module (Gu et al., 2025a). The judge analyzes the task instruction, action history, and screenshots to assess correctness at both trajectory and step levels. Because some expanded tasks are infeasible and rollout models can struggle in complex tasks, many generated trajectories fail to fully complete the intended goal. However, failed trajectories often contain a substantial prefix of correct actions, with errors typically occurring only at intermediate or later steps. Recognizing that not all steps in a failed trajectory are erroneous, we adopt a fine-grained judging approach to identify and retain useful sub-trajectories. The evaluation comprises two components: (1) Overall Trajectory Judgment: The MLLM-as-a-judge assesses end-to-end success, prioritizing visual evidence from screenshots over textual claims generated by the GUI agent; (2) Erroneous Trajectory Reuse: For failed trajectories, the judge identifies the longest prefix of correct actions before the first deviation. This enables reuse of failed trajectories, reducing data waste and enabling the model to learn from partial successes.

Iterative Rejection Sampling. We adopt an iterative self-improvement loop that progressively refines both the model and the trajectory data distribution. Let M(t) denote our model after the t-th round of fine-tuning, and let Iexpansion be the set of diverse task instructions from Diverse Task Expansion. In round t + 1, we use M(t) as the rollout policy to generate new trajectories on Iexpansion:

DRS(t+1) = Rollout(M(t),i) | i ∈ Iexpansion , (2) where each rollout is filtered through the fine-grained correctness judgment module to retain only highquality or partially correct segments. The training set for the next iteration is then constructed by mixing the newly generated rejection sampling data with novel trajectories synthesized from the Trajectory Synthesis stage:

D(t+1) = ·DRS(t+1) ∪ ·Dsynthesis, (3) where Dsynthesis denotes the trajectories generated from the human annotation and the agent rollouts. The model M(t+1) is then fine-tuned on D(t+1), completing the iteration. Rejection-sampled data helps close the gap between pass@1 and pass@N, while novel trajectories introduced in each iteration continually

[Figure 131]

[Figure 132]

[Figure 133]

###### Mobile Session Manager

Environment Manager

Actions

Screenshots

Env1

###### GRPOTrainer

Online Mobile Env1

|Trajectory #1|
|---|

| | |
|---|---|
| | |
| | |

Mobile Agent Loop

Task 1

Multi-turn Online Rollout

Env2

|Trajectory #2|
|---|

Online Mobile Env2

Mobile Agent Loop

Task 2

PolicyÇ√Update

###### …

###### … … …

EnvN

Task N Trajectory

Online Mobile EnvN

Mobile Agent Loop

#N

CPU Worker x P

GPU Worker x M

- Figure 5: Overview of the agentic reinforcement learning framework. The framework alternates between rollout phases where the latest policy interacts with online mobile environments to generate trajectories and training phases that progressively improve the policy using trajectory-level rewards.

raise the pass@N performance ceiling. This process encourages the data distribution to gradually align with the model’s evolving capabilities.

###### 2.3.2 Enabling Agent-User Interaction and MCP Tool Use

To teach the model to interact with users and use MCP tools, we augment our self-evolving data pipeline with trajectories that explicitly cover agent-user interaction and MCP augmentation. These trajectories are mixed into SFT, enabling the model to learn when to elicit missing information from the user and when to use MCP tools to complete tasks efficiently.

Agent–user interaction. We construct tasks with deliberately omitted critical information. When the annotation/rollout reaches to a step that requires missing information, it issues an ask_user action. The query is then routed to a synthetic user agent implemented with a standard LLM that is conditioned on hidden context containing the missing information. The user agent returns concise, context-appropriate replies, including clarifications, corrections, or refusals when applicable. We log the query–response pairs in the history and continue the annotation/rollout so the trajectory can incorporate the returned information to complete the task. We generate both single-turn and multi-turn interactions.

MCP augmentation. We design tasks that require or benefit from external MCP tools (e.g., Amap, Github, Stockstart). During trajectory annotation or rollout, the annotator/agent can issue mcp_call with a tool name and arguments. The MCP server executes the call and returns structured outputs. We record tool schemas, arguments, results, and the subsequent UI actions, and we keep trajectories that demonstrate correct MCP tool selection.

###### 2.3.3 Online Reinforcement Learning

To enhance the model’s reliability in long-horizon tasks and dynamic real-world environments, we employ an agentic reinforcement learning (RL) framework integrated with an online GUI environment, as illustrated in Figure 5. This framework operates through two alternating phases that drive iterative improvement: (1) during the rollout phase, the model executes multi-turn interactions with the GUI environment to complete tasks and collect full execution trajectories; (2) in the training phase, the model performs end-to-end policy updates using trajectory-level rewards. Through this iterative process, each improved policy generates higher-quality rollouts for subsequent training, progressively strengthening the model’s robustness and generalization capability.

[Figure 134]

###### New Task Automatic Data Curriculum Task Statistics

MobileAgentLoop

Execute Async Call Actions

Reset Async Call Env

Evaluate Trajectory

Async Call

System Prompt Instruction IMG#0 Thinking & Tool Calls IMG#1 … Thinking & Tool Calls Reward

|Call| | |
|---|---|---|
| |Generate| |

|Call| | |
|---|---|---|
| |Generate| |

Async Async

Stateful Environment

| |
|---|

Async Rollout Servers

- Figure 6: Detailed rollout process within the MobileAgentLoop. The agent loop asynchronously calls inference servers to generate actions and executes them in stateful environments across multiple turns, with hybrid verifiers evaluating complete trajectories to produce final rewards.

Scalable GUI Environment. A critical bottleneck in scaling agentic RL lies in efficiently scaling stateful environments. Unlike stateless environments for mathematical reasoning or code generation, GUI environments are inherently stateful and resource-intensive, requiring each rollout to operate within an isolated instance. This constraint motivates our use of virtualized environments rather than physical devices.

Inspired by an experimental feature in AndroidWorld (Rawles et al., 2024a), we built a containerized solution that encapsulates the entire GUI environment within a Docker image, comprising a rooted Android Virtual Device (AVD), self-hosted backend services, and a dedicated REST API server for orchestration. This solution is carefully designed to ensure the following three features:

- • Consistency. The unified containerization eliminates external dependencies and guarantees behavioral consistency across heterogeneous host systems.
- • Generalizability. To build a general environment for mobile use, we integrated over 35 applications, encompassing system utilities and open-source software such as Mattermost (enterprise communication), Mastodon (social media), and Mall4Uni (e-commerce). Self-hosting these applications provides full backend access, enabling precise manipulation of initial task states and deterministic verification of execution outcomes.
- • RL-native Design. We employ an AVD snapshot mechanism for reproducible task initialization and expose standard RL primitives (reset, step, get_observation, evaluate, and close) through the containerized API server, enabling parallel deployment of emulator instances.

To further scale environments across distributed infrastructure, we introduce a centralized Environment Manager that coordinates container instances across multiple physical machines. This architecture serves three critical functions: (1) efficient resource utilization: through automatic container reuse, environments are reset and reassigned upon rollout completion instead of being destroyed, (2) cross-machine orchestration: the Manager exposes a unified REST API that provides transparent access to distributed resources across heterogeneous hosts, and (3) fault tolerance: we introduce automatic detection and recovery mechanisms to handle container failures, with failover protocols that seamlessly replace compromised instances from a standby pool. By coordinating just 10 standard Alibaba Cloud ECS servers (ecs.ebmg5s.24xlarge), the Manager supports up to 512 concurrent environment instances for parallel rollout execution, while maintaining high availability essential for continuous online RL training.

Long-Horizon RL. Training RL agents for long-horizon tasks faces two interconnected challenges: traditional synchronous rollout pipelines become inefficient bottlenecks due to extensive multi-turn environment interactions, and the resulting ultra-long trajectory (up to millions of tokens per trajectory) exceed single-GPU memory capacity, necessitating advanced parallelism strategies to enable end-to-end

policy training. To address these challenges, we employ a strict on-policy, asynchronous RL training framework built on top of verl (Sheng et al., 2024), illustrated in Figure 6, with two key optimizations:

- • Asynchronous Rollout for Multi-Turn Efficiency. We implement a custom agent loop that asynchronously dispatches requests to a group of inference servers hosting the latest policy model, thereby mitigating GPU idling during environment interactions. The agent loop further incorporates asynchronous environment interaction with session management, maintaining backup sessions for seamless failover and replacement. On the server side, we employ load balancing and prefill caching to accelerate generation efficiency in multi-turn settings.
- • Hybrid Parallelism for Ultra-Long Sequences. To support end-to-end training of trajectories with millions of tokens, we leverage Megatron’s hybrid multi-dimensional parallelism (TP+PP+CP) to shard each long rollout trajectory across GPUs along the tensor, pipeline, and context dimensions, enabling scalable training while keeping per-GPU memory bounded. Additionally, we resize images to half their original resolution, which significantly improves training efficiency without compromising model performance.

Task and Verifier Design. Effective RL training requires a well-structured task distribution that balances exploration and exploitation. We manually curate a diverse set of over 35 applications spanning simple single-app operations to complex multi-app workflows. Tasks are dynamically stratified into four difficulty levels based on the current policy’s pass@K success rate (SR): frontier tasks (0–25% SR) push model capability boundaries, exploration tasks (25–50% SR) drive skill development, near-mastery tasks (50–75% SR) approach proficiency, and exploitation tasks (75–100% SR) reinforce learned behaviors.

Building on this stratification, we implement an automatic curriculum that progressively adjusts task sampling throughout training. Early stages emphasize simpler tasks to establish foundational skills, while the distribution gradually shifts toward challenging tasks as success rates improve. This adaptive strategy prevents training collapse from excessive difficulty while ensuring continuous learning signals, effectively addressing the exploration-exploitation tradeoff.

To enable scalable evaluation, we develop a hybrid verification approach tailored to task characteristics. Deterministic tasks with clear success criteria use rule-based verifiers with root-level AVD access for precise state verification. For complex tasks where rule-based verification is labor-intensive, we employ an MLLM-as-a-Judge framework to evaluate execution trajectories against task objectives. This hybrid approach achieves 83% agreement with human annotations, enabling reliable large-scale verification without manual bottlenecks.

Training Algorithm. we adopt a tailored GRPO to sample a group of outputs {oi}iG=1 for each task q, and optimizes the policy via the following objective:

|oi|

G

1 ∑cG=1 |oc|

min ri,t(θ)Aˆi,t,clip ri,t(θ),1 − εlow,1 + εhigh A ˆi,t

∑

∑

JGRPO(θ) = E(q)∼D,{o

i}iG=1∼πθold(·|q)

t=1

i=1

G i=1)

θold(oi,t|q,oi,<t) is the importance sampling ratio, and Aˆi,t = Ri−mean({Ri}

where ri,t(θ) = ππθ(oi,t|q,oi,<t)

std({Ri}iG=1) is the normalized advantage. We find a group size of 16 strikes a good balance between effectiveness and efficiency.

We additionally incorporate the following features to encourage exploration and improved stability:

- • Reward Design. The reward signal comprises two components: a task completion reward and an action-level repetition penalty. Task completion is measured as a binary indicator of successful execution, determined by either a rule-based verifier or the MLLM-as-a-Judge framework described above. To discourage unproductive looping, we penalize recurring action sequences (from single repeated actions to cyclic patterns of 3-5 actions). Actions with identical types but different parameters are not penalized, enabling flexible execution while preventing non-progressive behavioral loops.
- • Clip Higher. Following DAPO (Yu et al., 2025), we employ the token-level loss with no KL divergence

Local Agent Cloud Agent

Mobile Environment

[Figure 135]

[Figure 136]

|USERINSTRUCTION|
|---|

[Figure 137]

[Figure 138]

[Figure 139]

Action

Action

[Figure 140]

…

[Figure 141]

[Figure 142]

Email Browser Music Document

USERINSTRUCTION

[Figure 143]

Local

[Figure 144]

|GUI Navigation<br><br>Functions| | |
|---|---|---|
| |Trajectory Recovery| |

Unified Trajectory Memory

|Monitoring<br><br>GUI Navigation<br><br>Functions|
|---|

Monitoring

[Figure 145]

[Figure 146]

User Instruction

Screenshots History Outputs Error Summary

[Figure 147]

[Figure 148]

Device Cloud

[Figure 149]

Task Deviation & Privacy Sensitive

- Figure 7: Overview of device-cloud collaboration architecture. The system adaptively routes computation between device and cloud models based on task context and data sensitivity.

and an asymmetric clipping strategy with a larger upper bound to encourage exploration. Specifically, we set εlow to 0.2 and εhigh to 0.3.

• Experience Replay. We maintain a replay buffer of successful trajectories collected during training. When a rollout group contains no successful completions, we augment it with randomly sampled trajectories from the buffer. The buffer is continuously updated with newly successful experiences, retaining only the most recent eight trajectories per task to maintain near on-policy learning. This mechanism ensures continuous learning signals even during challenging exploration phases, stabilizing training and accelerating convergence.

###### 2.4 Device-Cloud Collaboration

Building on the above training process, we can obtain high-capacity agents for cloud serving and lightweight yet capable on-device agents. However, neither mode alone fully meets the requirements of real-world deployment. On-device solutions are constrained by model size and thus exhibit limited GUI agent capability. Cloud deployments suffer from high latency, privacy risks, and network dependence. To address these limitations, we introduce a native device-cloud collaborative architecture that adaptively routes computation between device and cloud based on task context and data sensitivity.

###### 2.4.1 System Architecture

The overall structure of our device-cloud collaboration system is illustrated in Figure 7. The system consists of a Local GUI Agent, which exhibit the ability of both GUI agent and trajectory monitor, a Cloud GUI Agent, and a Local Unified Trajectory Memory that maintains consistent information exchange between local and cloud agents. The specific function of these modules are demonstrated below:

- • Local Agent. The Local Agent runs on device and functions as both a GUI agent and a monitor. As a GUI agent, it perceives the current screen, and generates actions for each step of the task. As a monitor, it evaluates whether the trajectory so far remains aligned with the user instruction. The monitor checks indicators such as action execution failure, repeated actions without progress, incorrect inputs, or general task deviations. If the trajectory deviates from user instruction and the current context does not contain privacy-sensitive data, the monitor triggers a switch to the Cloud Agent. Additionally, the monitor generates an error summary each time a deviation is detected, which we find highly effective for trajectory recovery.
- • Cloud Agent. The Cloud Agent is called only when the monitor detects trajectory deviation. In addition to the standard GUI agent inputs, it receives an error summary from the monitor that explains why the switch was triggered. Given the trajectory history and error summary, the Cloud Agent

- executes subsequent steps, leveraging its higher capacity to complete the task.
- • Local Unified Trajectory Memory. On device, we maintain a unified history that records the task instruction, historical screenshots, and the model’s past outputs, including thoughts and actions. During action execution, the memory module projects the unified history into the action spaces expected by the device and cloud models, enabling either model can resume from any state without ambiguity.
- • Execution loop. The user provides a task instruction to the Local Agent. At each step, the Local Agent observes the current screenshot, decides an action, and executes it. The environmental observations and model outputs are then written to the Local Unified Trajectory Memory. Every few steps, the local agent assesses alignment between user instruction and the trajectory so far. If alignment is met, the loop continues on device. If deviation is detected and no sensitive data is involved, the system calls the Cloud Agent for task completion.

###### 2.4.2 Local Agent Training

We train a single on-device model that unifies two roles: an agent for GUI navigation, and a trajectory monitor for alignment assessment. Compared with prior work (Jiang & Huang, 2025), our Local Agent introduces two key innovations required for practical device-side deployment:

- • Integrated monitoring capability. In practice, the monitor must handle varied and complex cases, and prompt engineering alone is unlikely to deliver reliable monitoring. The on-device model is explicitly trained to judge whether the trajectory so far remains aligned with the user instruction across diverse apps, layouts, and tasks.
- • Error feedback generation. When deviation is detected, our model further generates a concise error summary to guide trajectory recovery. This signal is crucial for the Cloud Agent to complete the task correctly, since handoffs occur only after the trajectory has already deviated from the user instruction.

Training procedure. We train the Local Agent jointly on two data sources: (i) standard GUI agent data covering perception, grounding, and navigation, and (ii) monitor data that include alignment reasoning, alignment decision, and error summaries. This multi-task training recipe teaches the on-device model to execute and monitor simultaneously, without requiring separate models or fragile prompt engineering.

###### 2.5 MobileWorld Benchmark

Existing benchmarks for mobile GUI agents often fail to capture the complexity of real-world mobile usage. Many evaluations rely on simple applications or limited app categories, and they typically assume idealized interaction models where user instructions are perfectly clear and agents operate solely through GUI manipulation (Rawles et al., 2024b; Xu et al., 2025a). This gap between benchmark performance and real-world utility has become increasingly apparent. To evaluate MAI-UI’s practical capabilities, we adopt our MOBILEWORLD (Kong et al., 2025) benchmark, a comprehensive benchmark designed to bridge this evaluation gap. MOBILEWORLD features over 200 realistic tasks spanning 15+ opensource applications across critical domains including e-commerce (Mall4Uni, mirroring Temu/Amazon), enterprise communication (Mattermost, mirroring Microsoft Teams/Slack), social media (Mastodon, mirroring X/Twitter), and daily productivity tools.

In addition, MOBILEWORLD extends beyond standard GUI manipulation to evaluate two essential real-world capabilities, which directly align with MAI-UI’s design of practical deployment.

- • Agent-User Interaction, where agents must detect ambiguous user requests and proactively seek clarification rather than making incorrect assumptions.
- • MCP Tool Integration, where agents must intelligently decide between manual GUI navigation and API-based operations via MCP (Anthropic, 2024) to optimize efficiency.

- Table 2: Performance comparison on the ScreenSpot-Pro benchmark. We use ‘∗’ to denote the results evaluated by us. The best results are highlighted in bold, and the second-best results are underlined.

CAD Dev. Creative Scientific Office OS

Model

Avg. Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon

Proprietary Models GPT-4o (OpenAI, 2024) 2.0 0.0 1.3 0.0 1.0 0.0 2.1 0.0 1.1 0.0 0.0 0.0 0.8 Claude C. (Anthropic, 2024) 14.5 3.7 22.0 3.9 25.9 3.4 33.9 15.8 30.1 16.3 11.0 4.5 17.1 Gemini-3-Pro (DeepMind, 2025b) - - - - - - - - - - - - 72.7 Seed1.8 (Seed, 2025a) - - - - - - - - - - - - 73.1

Open-Source Models

Qwen3-VL-2B∗ (Bai et al., 2025a) 31.0 15.6 55.2 11.7 59.1 16.1 64.6 22.7 72.3 34.0 59.8 23.6 41.9 InfiGUI-3B (Liu et al., 2025b) 50.8 25.0 64.9 20.0 51.5 16.8 68.8 32.7 70.6 32.1 49.5 19.7 45.2 Ferret-UI Lite (Yang et al., 2025b) - - - - - - - - - - - - 53.3 UI-TARS-7B (Qin et al., 2025a) 20.8 9.4 58.4 12.4 50.0 9.1 63.9 31.8 63.3 20.8 30.8 16.9 35.7 Phi-Ground (Zhang et al., 2025a) 26.9 17.2 70.8 16.7 56.6 13.3 58.0 29.1 76.4 44.0 55.1 25.8 43.2 GUI-Actor-7B (Wu et al., 2025) 47.7 9.4 59.1 15.9 59.6 16.1 70.1 25.5 69.5 41.5 55.1 19.1 44.6 SE-GUI-7B (Yuan et al., 2025) 51.3 14.1 68.2 19.3 57.6 9.1 75.0 28.2 78.5 43.4 49.5 25.8 47.2 GUI-G2-7B (Tang et al., 2025) 55.8 12.5 68.8 17.2 57.1 15.4 77.1 24.5 74.0 32.7 57.9 21.3 47.5 Qwen3-VL-8B∗ (Bai et al., 2025a) 46.7 10.9 79.2 23.4 68.2 14.0 73.6 30.0 76.3 30.2 65.4 21.3 49.9 OpenCUA-7B (Wang et al., 2025b) - - - - - - - - - - - - 50.0 GTA1-7B (Yang et al., 2025a) 53.3 17.2 66.9 20.7 62.6 18.9 76.4 31.8 82.5 50.9 48.6 25.9 50.1 UI-Venus-7B (Gu et al., 2025b) 60.4 21.9 74.7 24.1 63.1 14.7 76.4 31.8 75.7 41.5 49.5 22.5 50.8 InfiGUI-G1-7B (Liu et al., 2025b) 57.4 23.4 74.7 24.1 64.6 18.2 80.6 31.8 75.7 39.6 57.0 29.2 51.9 GUI-Owl-7B (Ye et al., 2025) 64.5 21.9 76.6 31.0 59.6 27.3 79.1 37.3 77.4 39.6 59.8 33.7 54.9

Qwen3-VL-32B∗ (Bai et al., 2025a) 60.4 28.1 69.5 22.1 75.8 25.2 84.7 25.5 85.9 43.4 62.6 15.7 54.9 OpenCUA-32B (Wang et al., 2025b) - - - - - - - - - - - - 55.3 GUI-Owl-32B (Ye et al., 2025) 62.4 28.1 84.4 39.3 65.2 18.2 82.6 39.1 81.4 39.6 70.1 36.0 58.0 GTA1-32B (Yang et al., 2025a) 43.7 23.4 82.5 28.3 69.2 14.7 79.9 31.8 80.8 43.4 70.1 32.6 63.6 UGround-v1-72B (Gou et al., 2025) 16.8 4.7 55.8 4.8 54.0 10.5 70.8 22.7 61.0 18.9 40.2 7.9 34.5 UI-Tars-72B (Qin et al., 2025a) 18.8 12.5 63.0 17.2 57.0 15.4 64.6 20.9 63.3 26.4 42.1 15.7 38.1 UI-Venus-72B (Gu et al., 2025c) 66.5 29.7 84.4 33.1 73.2 30.8 84.7 42.7 83.1 60.4 75.7 36.0 61.9

Ours

MAI-UI-2B 61.4 23.4 76.6 32.4 69.2 21.7 81.2 34.5 85.9 39.6 68.2 41.6 57.4 + Zoom-In 69.5 34.4 75.3 42.8 74.7 30.1 84.0 42.7 85.3 56.6 69.2 47.2 62.8 MAI-UI-8B 72.6 35.9 83.8 52.4 76.3 33.6 79.9 37.3 88.7 60.4 76.6 49.4 65.8 + Zoom-In 80.7 43.8 78.6 58.6 78.8 46.9 86.1 49.1 88.1 81.1 76.6 51.7 70.9 MAI-UI-32B 70.1 45.3 86.4 40.7 82.8 37.8 91.7 46.4 90.4 71.7 78.5 34.8 67.9 + Zoom-In 79.2 53.1 84.4 57.9 87.9 46.2 91.7 54.5 88.1 79.2 80.4 47.2 73.5

###### 3 Experiments

We present comprehensive experiments in this section, including extensive benchmark evaluations and detailed ablations and analyses of the main components of MAI-UI. Specifically, we introduce the experimental setup (Sec. 3.1); report the main benchmark results across GUI grounding, GUI navigation, and real-world-oriented evaluation (Sec. 3.2); present case studies of MCP augmentation and agent–user interaction (Sec. 3.3); demonstrate experiments and case studies on the native device–cloud collaboration system (Sec. 3.4); discuss online RL ablations (Sec. 3.5); and, finally, provide grounding analysis (Sec. 3.6).

###### 3.1 Experimental Setup

Implementation details MAI-UI uses Qwen3-VL (Bai et al., 2025a) as the backbone across multiple model sizes (2B, 8B, 32B, and a 235B-A22B) to meet realistic deployment needs. Training proceeds in four stages: (i) SFT on perception and grounding data, (ii) SFT on mobile-use navigation data with a small portion of grounding data, (iii) RL for grounding to develop robust UI grounding capability, and (iv) online RL for mobile-use navigation in dynamic environments to enhance robustness and generalization. To further enhance the large cloud model variants, we augment the training of the 32B and 235B-A22B models with more real-world trajectories.

- Table 3: Performance comparison on UI-Vision grounding dataset. The best results are highlighted in bold, and the second-best results are underlined.

Models Basic Functional Spatial Avg Proprietary Models

GPT-4o (OpenAI, 2024) 1.6 1.5 1.0 1.4 Claude-3.7-Sonnet (Anthropic, 2024) 9.5 7.7 7.6 8.3

Open-source Models Qwen3-VL-2B∗ (Bai et al., 2025a) 0.0 19.2 0.1 6.2 InfiGUI-G1-3B (Liu et al., 2025b) 31.2 28.0 8.2 22.0 Qwen2.5-VL-7B (Bai et al., 2025b) 1.2 0.8 0.5 0.9 SeeClick (Cheng et al., 2024) 9.4 4.7 2.1 5.4 UGround-V1-7B (Gou et al., 2025) 15.4 17.1 6.3 12.9 OS-Atlas-7B (Wu et al., 2024) 12.2 11.2 3.7 9.0 Qwen3-VL-8B∗ (Bai et al., 2025a) 25.0 27.9 1.2 17.5 UI-TARS-7B (Qin et al., 2025b) 20.1 24.3 8.4 17.6 UI-TARS-1.5-7B (Seed, 2025b) 28.8 27.5 10.7 22.3 InfiGUI-G1-7B (Liu et al., 2025b) 36.2 31.9 11.5 26.1 UI-Venus-7B (Gu et al., 2025b) 36.1 32.8 11.9 26.5 Phi-Ground (Zhang et al., 2025a) 36.8 37.1 7.6 27.2 Qwen3-VL-32B∗ (Bai et al., 2025a) 32.8 34.2 14.7 26.9 UI-TARS-72B (Qin et al., 2025a) 31.4 30.5 14.7 25.5 UI-Venus-72B (Gu et al., 2025b) 45.6 42.3 23.7 36.8

Ours

MAI-UI-2B 41.0 41.2 10.4 30.3 + Zoom-In 43.2 43.0 11.3 31.9 MAI-UI-8B 51.7 49.6 22.5 40.7 + Zoom-In 51.6 50.5 26.6 42.4 MAI-UI-32B 59.1 57.1 26.9 47.1 + Zoom-In 58.7 56.8 33.6 49.2

Benchmarks We evaluate MAI-UI across extensive benchmarks spanning three categories: grounding, mobile-use, and real-world–oriented evaluation.

- • Grounding benchmarks. We evaluate grounding with five complementary benchmarks: ScreenSpotPro (Li et al., 2025) for high-resolution, fine-grained professional layouts, UI-Vision (Nayak et al.,

2025) for diverse applications and reasoning types (e.g., spatial and functional), MMBench-GUI L2 (Xuehui Wang et al., 2025) for hierarchical-instruction following and compositional reasoning, OSWorldG and OSWorld-G Refine (Xie et al., 2025b) for comprehensive skills such as layout understanding, widget matching, and fine-grained manipulation, and ScreenSpot-V2 (Wu et al., 2024) to broaden coverage across different operating systems.

- • Mobile-use benchmarks. We report offline and online results. Offline evaluation includes Android Control (Li et al., 2024), which evaluates planning and action execution capabilities in the mobile setting, and GUI Odyssey (Lu et al., 2024a), which covers cross-app navigation tasks. For online evaluation, AndroidWorld (Rawles et al., 2024a) provides 116 tasks across 20 Android apps in a live Android emulator, and requires continuous interaction with dynamic mobile environment.
- • Real-world–oriented evaluation. We introduce MobileWorld (Kong et al., 2025), a more challenging, dynamic benchmark that closely mirrors production usage. It includes tasks requiring agent–user interaction and MCP tool use, enabling rigorous evaluation of these two capabilities that are critical in real-world settings.

###### 3.2 Main Results

###### 3.2.1 Grounding Capability

Overall Grounding Performance. We evaluate MAI-UI on five comprehensive GUI grounding benchmarks. Across all scales, our 2B, 8B, and 32B variants consistently outperform models of comparable

- Table 4: Performance comparison on the MMBench-GUI L2 benchmark. The best results are highlighted in bold, and the second-best results are underlined.

Windows MacOS Linux iOS Android Web

Model

Avg. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv. Bas. Adv.

Proprietary Models

GPT-4o (OpenAI, 2024) 1.5 1.1 8.7 4.3 1.1 1.0 5.1 3.3 2.5 1.4 3.2 2.9 2.9 Claude-3.7 (Anthropic, 2024) 1.5 0.7 12.5 7.5 1.1 0.0 13.7 10.6 1.4 1.4 3.2 2.3 4.7 Qwen-Max-VL (Yang et al., 2024a) 43.9 36.8 58.8 56.1 53.9 30.1 77.4 59.1 79.5 70.1 74.8 58.8 58.0

Open-Source Models

Qwen3-VL-2B∗ (Bai et al., 2025a) 81.9 0.0 80.3 46.2 67.5 0.0 90.8 0.0 91.0 0.0 88.1 0.0 46.5 OS-Atlas-7B (Wu et al., 2024) 36.9 18.8 44.4 21.7 31.4 13.3 74.8 48.8 69.6 46.8 61.3 35.4 41.4 Aguvis-7B (Xu et al., 2025b) 37.3 21.7 48.1 33.3 33.5 25.0 67.5 65.2 61.0 51.0 61.6 45.5 45.7 UI-TARS-1.5-7B (Seed, 2025b) 68.3 39.0 69.0 44.5 64.4 37.8 88.5 69.4 90.5 69.3 81.0 56.5 64.3 UGround-V1-7B (Gou et al., 2025) 66.8 39.0 71.3 48.6 56.5 31.1 92.7 70.9 93.5 71.0 88.7 64.6 65.7 GUI-Actor-7B∗ (Wu et al., 2025) 80.8 55.1 81.4 60.4 64.9 41.8 94.3 82.7 93.5 79.7 89.7 72.1 76.5 SE-GUI-7B∗ (Yuan et al., 2025) 77.5 57.7 77.1 60.7 68.6 44.9 95.5 80.0 95.5 83.7 89.7 68.8 76.6 Qwen3-VL-8B∗ (Bai et al., 2025a) 88.6 61.8 85.5 69.1 74.9 53.1 95.2 82.4 95.5 84.5 96.8 72.1 81.3 GTA1-7B∗ (Yang et al., 2025a) 76.8 57.4 80.3 63.9 68.6 53.6 93.9 83.3 96.3 84.5 90.3 74.7 78.5 GUI-G2-7B∗ (Tang et al., 2025) 79.7 55.1 79.7 64.7 69.6 50.0 95.2 82.7 96.6 85.4 91.9 75.6 78.8 GUI-Owl-7B (Ye et al., 2025) 86.4 61.8 81.7 64.5 74.4 61.7 94.9 83.0 95.8 83.7 93.2 72.7 80.5 InfiGUI-G1-7B (Liu et al., 2025b) 82.7 61.8 83.8 63.9 72.3 52.0 94.9 89.4 95.2 85.6 93.5 76.3 80.8

GUI-Owl-32B (Ye et al., 2025) 85.6 65.1 84.9 67.1 77.0 63.3 95.2 85.5 96.1 87.0 95.5 80.8 83.0 GTA1-32B∗ (Yang et al., 2025a) 82.3 66.9 89.0 74.0 73.3 52.0 96.2 88.2 95.8 88.5 95.2 79.9 83.4 Qwen3-VL-32B∗ (Bai et al., 2025a) 93.4 71.3 92.8 74.3 78.0 56.1 95.5 88.8 97.2 88.5 92.6 78.6 85.3 UI-TARS-DPO-72B (Qin et al., 2025a) 78.6 51.8 80.3 62.7 68.6 51.5 90.8 81.2 93.0 80.0 88.1 68.5 74.3 InternVL3-78B (Zhu et al., 2025) 70.1 42.6 75.7 52.3 59.2 41.3 93.6 80.6 92.7 78.6 90.7 65.9 72.2

Ours

MAI-UI-2B 84.9 64.0 89.3 72.5 75.4 60.2 95.2 85.2 96.3 84.2 92.9 76.0 82.6 MAI-UI-8B 92.3 74.3 90.7 86.4 81.2 67.3 97.1 90.0 97.5 92.7 95.8 86.0 88.8 MAI-UI-32B 93.0 78.7 92.8 87.6 86.9 77.6 97.1 92.4 98.0 93.2 96.1 92.5 91.3

size and establish new state-of-the-art results. On ScreenSpot-Pro (Li et al., 2025), MAI-UI-32B attains 67.9% accuracy, an 4.1% absolute accuracy gain over the strongest baseline GTA1-32B. With the adaptive zoom-in strategy, performance further increases to 73.5%, surpassing Gemini 3 Pro (DeepMind, 2025b) and Seed1.8 (Seed, 2025a). On OSWorld-G (Xie et al., 2025b), our models show consistent improvements over the best comparable baselines. Specifically, MAI-UI-32B with zoom-in achieves 70.9% on OSWorld-G and 75.0% on OSWorld-G Refine. On UI-Vision (Nayak et al., 2025), MAI-UI-32B achieves 47.1% accuracy, and it increases to 49.2% with the zoom-in strategy, making an absolute gain of +12.4 points over the previous best UI-Venus-Ground-72B. On MMBench-GUI L2 (Xuehui Wang et al., 2025), MAI-UI reaches 91.3%, surpassing the prior best GTA1 by +7.9 points. On ScreenSpot-V2 (Wu et al., 2024), MAI-UI sets a new SOTA at 96.5%. It is also worth noting that MAI-UI-2B, despite its small size, demonstrates strong grounding performance. For example, it achieves 62.8% on ScreenSpot-Pro with the zoom-in operation. This score outperforms GUI-Owl-32B and UI-Venus-72B. We present grounding case studies across different operating systems in Figure 15. Detailed per-benchmark results are discussed below.

Grounding for high-resolution scenario. ScreenSpot-Pro evaluates grounding performance on highresolution professional software with dense and fine-grained UI layouts. As illustrated in Table 2, MAI-UI achieves the best grounding performance across all categories, including CAD, development, creative, scientific, office, and operation systems. Averaged over categories, the MAI-UI’s 2B, 8B, and 32B variants achieve 57.4%, 65.8%, and 67.9%, surpassing the best baselines of comparable size by +4.1, +10.9, and +4.3 points, respectively. With the zoom-in strategy, MAI-UI-32B further attains an average accuracy of 73.5%, significantly surpassing strong open-source baselines such as GTA1-32B (Yang et al., 2025a), UI-Venus (Gu et al., 2025c), and GUI-Owl (Ye et al., 2025), and even outperforming Gemini 3 Pro (DeepMind, 2025b) and Seed1.8 (Seed, 2025a). These results highlight the effectiveness of our model on complex, high-resolution screens.

- Table 5: Performance comparison of state-of-the-art models on the OSWorld-G. The best results are highlighted in bold, and the second-best results are underlined.

Text Matching

Element Recognition

Layout Understanding

Fine-grained Manipulation

Agent Model

Refusal Avg Proprietary Models

Operator (OpenAI, 2025) 51.3 42.4 46.6 31.5 0.0 40.6 Seed1.5-VL (Seed, 2025c) 73.9 66.7 69.6 47.0 18.5 62.9

Open-Source Models

Jedi-3B (Xie et al., 2025a) 67.4 53.0 53.8 44.3 7.4 50.9 Qwen3-VL-2B∗ (Bai et al., 2025a) 61.7 45.8 54.2 39.6 - 45.9 OS-Atlas-7B (Wu et al., 2024) 44.1 29.4 35.2 16.8 7.4 27.7 UGround-7B (Gou et al., 2025) 51.3 40.3 43.5 24.8 0.0 36.4 Aguvis-7B (Xu et al., 2025b) 55.9 41.2 43.9 28.2 0.0 38.7 UI-TARS-7B (Qin et al., 2025b) 60.2 51.8 54.9 35.6 0.0 47.5 UI-TARS-1.5-7B (Seed, 2025b) 36.8 62.7 62.2 50.8 0.0 52.8 Jedi-7B (Xie et al., 2025a) 65.9 55.5 57.7 46.9 7.4 54.1 Qwen3-VL-8B∗ (Bai et al., 2025a) 69.0 55.5 59.7 47.7 - 54.8 GTA1-7B (Yang et al., 2025a) 42.1 65.7 62.7 56.1 0.0 55.1 GUI-Owl-7B (Ye et al., 2025) 64.8 63.6 61.3 41.0 - 55.9 UI-Venus-7B (Gu et al., 2025b) 74.6 60.5 61.5 45.5 - 58.8

OpenCUA-32B (Wang et al., 2025b) - - - - - 59.6 GUI-Owl-32B (Ye et al., 2025) 67.0 64.5 67.2 45.6 - 58.0 Qwen3-VL-32B∗ (Bai et al., 2025a) 72.8 63.3 66.4 51.7 - 60.6 GTA1-32B (Yang et al., 2025a) 63.2 78.4 73.3 65.2 0.0 65.2 UI-TARS-72B (Qin et al., 2025a) 69.4 60.6 62.9 45.6 0.0 57.1 UI-Venus-72B (Gu et al., 2025b) 82.1 71.2 70.7 64.4 - 70.4

Ours

MAI-UI-2B 62.8 56.7 59.3 40.3 - 52.0 + Zoom-In 66.7 59.4 63.2 44.3 - 55.9 MAI-UI-8B 72.0 63.3 66.0 51.0 - 60.1 + Zoom-In 72.8 67.6 71.1 56.4 - 64.2 MAI-UI-32B 73.6 72.4 73.9 57.7 - 67.6 + Zoom-In 78.5 75.2 78.3 62.4 - 70.9

Grounding for diverse and complex instruction. To assess grounding performance in terms of instruction diversity and complexity, we use UI-Vision and MMBench-GUI L2 for evaluation. UI-Vision provides multi-perspective queries (e.g., spatial, functional), and MMBench-GUI L2 includes instructions of Basic (low-level, attribute/appearance) and Advanced (high-level, goal-oriented) categories. Together, these benchmarks reflect realistic usage where user instructions are heterogeneous, high-level, and often implicit. On UI-Vision, our model sets a new state of the art: MAI-UI-32B with zoom-in achieves a 49.2% average accuracy, exceeding the strongest baseline (UI-Venus-Ground-72B 36.8%) by +12.4 points. The 8B and 2B variants of MAI-UI also outperform baselines of similar size by +15.2 and +9.9 points, demonstrating superior grounding performance across diverse instruction perspectives. On MMBench-GUI L2, MAI-UI-32B, 8B and 2B variants attain a 91.3%, 88.8%, 82.6%, setting new state-of-the-art results at each scale. Additionally, the improvement on the high-level Advanced setting is much larger than on Basic setting, demonstrating strong grounding under high-level, implicit instructions.

Grounding for complex desktop scenario. OSWorld-G and OSWorld-G-Refine assess grounding in complex desktop scenarios that require software commonsense, layout understanding, and fine-grained manipulation. As illustrated in Table 5 and Table 11, MAI-UI demonstrate consistent gains across categories and model sizes. On OSWorld-G, MAI-UI-32B achieves an average of 67.6%, and increases to 70.9% with zoom-in. This exceeds the strongest baselines, including UI-Venus-72B and GTA1-32B. MAI-UI-2B, and MAI-UI-8B also outperform baselines of similar scale by 5.0 and 5.4 points, respectively. Category-wise, the MAI-UI shows balanced performance, with the 32B variant achieving 78.5% in Text Matching, 75.2% in Element Recognition, 78.3% in Layout Understanding, and 62.4% in Fine-grained Manipulation. On OSWorld-G Refine, which reduces instruction ambiguity and emphasizes precise manipulation, MAI-UI-32B reaches 73.9% and further improves to 75.0% with zoom-in, exceeding strong

- Table 6: Performance comparison on AndroidWorld Benchmark. The best results are highlighted in bold, and the second-best results are underlined.

MODEL PARAS. SUCCESS RATE Baselines

Qwen3-VL-2B (Bai et al., 2025a) 2B 36.4 ScaleCUA-3B (Liu et al., 2025e) 3B 23.7 Ferret-UI Lite-3B (Yang et al., 2025b) 3B 28.0 UI-Tars-7B (Qin et al., 2025b) 7B 33.0 UI-Tars-1.5-7B (Seed, 2025c) 7B 30.0 UI-Venus-7B (Gu et al., 2025c) 7B 49.1 GUI-Owl-7B (Ye et al., 2025) 7B 66.4 Step-GUI-8B (Yan et al., 2025) 8B 67.7 Qwen3-VL-8B (Bai et al., 2025a) 8B 47.6 Qwen3-VL-32B (Bai et al., 2025a) 32B 57.3 UI-Tars-SFT-72B (Qin et al., 2025b) 72B 46.6 UI-Venus-72B (Gu et al., 2025c) 72B 65.9 Seed1.5-VL (Guo et al., 2025b) - 62.1 Qwen3-VL-235B-A22B (Bai et al., 2025a) 235B 63.7 UI-Tars-1.5 (Seed, 2025c) - 64.2 Gemini-2.5-Pro (DeepMind, 2025a) - 69.7 Seed1.8 (Seed, 2025a) - 70.7 UI-Tars-2 (Wang et al., 2025a) 230B 73.3

Ours MAI-UI-2B 2B 49.1 MAI-UI-8B 8B 70.7 MAI-UI-32B 32B 73.3 MAI-UI-235B-A22B 235B 76.7

baselines such as OpenCUA-32B and GTA1-32B.

Grounding across different operating systems. ScreenSpot-V2 spans mobile, desktop, and web interfaces with both text and icon grounding tasks. As shown in Table 10 in Appendix, MAI-UI-32B achieves a new state of the art with 96.5% average accuracy, demonstrating strong results across all domains (e.g., 99.5% on Desktop-Text and 94.6% on Web-Icon). Notably, MAI-UI-8B attains 95.5% average accuracy, outperforming much larger models such as UI-Venus-72B and GTA1-32B. The MAI-UI-2B on-device variant reaches 92.5%, surpassing many 7B baselines.

###### 3.2.2 Mobile-Use Navigation Capability

In addition to GUI grounding evaluation, we conduct a series of experiments to validate the effectiveness of mobile navigation capabilities. This evaluation comprises two components: static offline benchmarking and challenging dynamic online benchmarking.

Table 7: Performance comparison on Android Control (highlevel instruction) and GUI Odyssey. Baseline results are mainly sourced from scores reported in (Zhang et al., 2025b).

Model AC-High GUI-Odyssey

AgentCPM-GUI-8B (Zhang et al., 2025b) 69.2 75.0 UI-TARS-7B (Qin et al., 2025b) 74.4 67.9 OS-Atlas-7B (Wu et al., 2024) 56.5 76.8 Aguvis-7B (Xu et al., 2024) 54.2 13.5 OdysseyAgent-7B (Lu et al., 2024a) 32.7 73.7 UI-Venus-72B (Gu et al., 2025c) 77.2 72.4

Offline Benchmark We evaluate MAIUI on Android Control (Li et al., 2024) and GUI Odyssey (Lu et al., 2024a), two static offline benchmarks that assess GUI action execution, and grounding. Android Control comprises two instruction types: high-level instructions provide only a natural-language goal, whereas low-level in-

MAI-UI-2B 67.3 72.6 MAI-UI-8B 69.1 80.1 MAI-UI-32B 75.5 83.4

Table 8: Performance comparison on MobileWorld Benchmark (User-Int. is short for User-Interaction). The best results of end-to-end models are highlighted in bold, and the second-best results are underlined.

MODEL GUI-ONLY (116) USER-INT. (45) MCP (40) OVERALL Agentic Framework

Claude-4.5-Sonnet (Anthropic, 2024) + UI-Ins-7B 47.8 37.8 50.0 43.8 Gemini-3-Pro (DeepMind, 2025b) + UI-Ins-7B 55.6 24.4 48.6 46.3 GPT-5 (OpenAI, 2025) + UI-Ins-7B 54.0 62.2 51.6 51.7

End-to-End Model GUI-Owl-7B (Ye et al., 2025) 7.7 - - 4.5 GUI-Owl-32B (Ye et al., 2025) 8.5 - - 5.5 UI-Venus-7B (Gu et al., 2025b) 8.5 2.3 - 5.5 UI-Venus-72B (Gu et al., 2025b) 16.4 4.7 - 10.4 Qwen3-VL-8B (Bai et al., 2025a) 9.4 0.0 0.0 5.5 Qwen3-VL-32B (Bai et al., 2025a) 11.9 6.7 2.7 9.0 Qwen3-VL-235B-A22B (Bai et al., 2025a) 12.8 4.4 5.4 9.5 Doubao-1.5-UI-TARS (Seed, 2025b) 26.3 32.4 - 20.9

Ours

MAI-UI-8B 27.5 22.2 20.0 24.9 MAI-UI-32B 36.2 46.7 30.0 37.3 MAI-UI-235B-A22B 39.7 51.1 37.5 41.7

structions include step-wise action annotations. GUI Odyssey targets cross-application navigation in mobile environments. As shown in Table 7, MAI-UI produces competitive results on Android Control. On GUI Odyssey, MAI-UI-32B achieves a new state of the art on the exact match score, substantially outperforming strong baselines and highlighting superior cross-application navigation capability of our model.

Online Benchmark Online evaluation provides a more realistic assessment of agent capability than static offline tests, as it requires multi-turn adaptive perception, reasoning, and action in dynamic environments. We evaluate MAI-UI on online benchmark AndroidWorld (Rawles et al., 2024a), a live emulator–based benchmark comprising 116 tasks across 20 mobile applications. For a fair comparison, we report results only for end-to-end GUI agents, isolating the intrinsic GUI capability of foundation models without interference from pipeline frameworks or external tools.

As shown in Table 6, MAI-UI achieves new state-of-the-art results on AndroidWorld across model scales. MAI-UI-235B-A22B attains a 76.7% success rate, surpassing UI-TARS-2 (73.3%) and Gemini-2.5-Pro (69.7%) by +3.4 and +7.0 points, respectively. The 32B variant reaches 73.3%, establishing SOTA at its scale and outperforming larger baseline models, including UI-Venus-72B, Gemini-2.5-Pro, and UI-TARS-1.5. MAI-UI-8B achieves 70.7% sucess rate, exceeding strong similar-scale models such as GUI-Owl-7B and Step-GUI-8B. Finally, the 2B model attains 49.1% success rate, improving over the strongest on-device baseline Ferret-UI-Lite-3B (28.0%), by +21.1 points (+75.4% relative), providing a strong foundation for our native device-cloud collaboration system. These results demonstrate the strong GUI capability of MAI-UI across the full spectrum of scales.

Real-world-oriented online evaluation. Beyond AndroidWorld, we assess our models on MobileWorld (Kong et al., 2025), a more challenging and more realistic online benchmark with 201 tasks across 20 applications. MobileWorld emphasizes long-horizon, cross-application tasks and includes evaluation beyond pure UI operation, such as MCP tool use and agent–user interaction. As summarized in Table 8, MAI-UI achieve substantial improvements against end-to-end models: MAI-UI-235B-A22B reaches 41.7% overall success rate, MAI-UI-32B attains 37.3%, and MAI-UI-8B scores 24.9%. These outperform the strongest end-to-end baseline (Doubao-1.5-UI-TARS, 20.9%) by +20.8, +16.4, and +4.0 points, respectively. We also compare against agentic frameworks that use a strong planner such as GPT-5 and an external

[Figure 150]

[Figure 151]

中介给我发了两套房子的信息，我想比较一下哪一套离阿里西溪c园区开车更近，好决定租哪一间。公司 地址是「杭州市余杭区文一西路969号」；把最近那套房子的地址发给我朋友 Mia

###### # User Instruction

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

MCP call

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

amap_maps_geo "杭州市余杭区文一西路969号"

[Figure 161]

[Figure 162]

[Figure 163]

amap_maps_geo "杭州市西湖区文新古墩路453号"

#### …

[Figure 164]

[Figure 165]

[Figure 166]

amap_maps_geo "杭州市西湖区文新文一西路149 号"

[Figure 167]

[Figure 168]

amap_maps_direction_driving "120.103631,30.278430” "120.022842,30.277608"

[Figure 169]

[Figure 170]

"distance": 9618, …

[Figure 171]

[Figure 172]

[Figure 173]

amap_maps_direction_driving "120.105841,30.286372” "120.022842,30.277608"

[Figure 174]

[Figure 175]

[Figure 176]

"distance": 9866, …

Click [412, 1977] Click [331, 377] MCP call MCP details Navigate_back Terminate

(a)

[Figure 177]

Help me check the recent 3 commits summary from the google-research/android_world repository (including author and commit message), format each line as 'author: commit message' and email to mike@gmail.com, email subject as "Recent Commits"

[Figure 178]

###### # User Instruction

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

MCP call

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

gitHub_list_commits {"owner": "google-research”, "repo": "android_world” "perPage": 3}

## …

[Figure 189]

[Figure 190]

"commit”: {

[Figure 191]

[Figure 192]

……

}

[Figure 193]

MCP details Click [149, 1278] Click [818, 2013] Click [809, 209] Terminate

MCP call

(b)

- Figure 8: Case studies of MCP tool using of MAI-UI. (a): Using MCP tools provide shortcuts that compress multiple UI actions into a few API calls; (b): Using MCP tools brings traditionally desktop-only workflows (e.g., GitHub commit search) to mobile. The user instruction for (a) is: “Compare the two apartment listings sent by the agent and determine which has the shorter driving time to Alibaba Xixi Campus (Zone C; 969 Wenyi West Road, Yuhang District, Hangzhou). Send the address of the nearer apartment to my friend Mia”.

grounding model (UI-Ins (Chen et al., 2025)) as executor. These pipelines benefit from strong planning and reasoning ability and an external execution model, making the comparison not strictly fair. However, our pure end-to-end models remain competitive, with MAI-UI-235B-A22B reaching 41.7% overall success rate, close to Gemini-3-Pro+UI-Ins (46.3%) and Claude-4.5-Sonnet+UI-Ins (43.8%).

To quantitatively assess our model’s MCP tool-use and agent–user interaction capabilities, we evaluate on the MobileWorld benchmark’s two relevant subsets (User-Int., 45 tasks; MCP, 40 tasks). Our MAI-UI235B-A22B achieves 51.1% on User-Int. and 37.5% on MCP, outperforming existing end-to-end baselines by +18.7 and +32.1 points, respectively (best prior scores: 32.4% and 5.4%). Compared with agentic frameworks, our end-to-end model is competitive on agent–user interaction, surpassing Gemini-3-Pro+UIIns-7B (24.4%) and Claude-4.5-Sonnet+UI-Ins-7B (37.8%). Overall, these results demonstrate strong

[Figure 194]

[Figure 195]

###### # User Instruction

在Download内找到最近一个月下载的简历文件，把文件发送给我的HR同事，标题为candidates_cv

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

Long_press [419, 930]

Click [399, 1506] Click [378, 2085] Click [766, 195] Click [316, 2136]

Click [411, 1297]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

###### Agent user interaction

[Figure 213]

[Figure 214]

[Figure 215]

请提供要发送的HR同 事的收件人邮箱地址。 是否需要在邮件正文添 加内容？如果需要，请 告诉我正文内容；如果 不添加正文请回复“不 添加”。

[Figure 216]

[Figure 217]

[Figure 218]

### …

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

收件人邮箱地址是 HR_chen@gmail.co m, 不添加正文。

[Figure 223]

Ask_user Click [311, 507] Input_text

HR_chen@gmail.com Click [299, 663] Click [809, 209]

###### User response

- Figure 9: A case study of agent user interaction. The user instruction is: “In the Downloads folder, locate resume file(s) downloaded within one month and send them to my HR colleague with the subject “candidates_cv.”.

MCP tool-use and agent–user interaction capabilities relative to end-to-end models, and competitive performance compared with agentic frameworks.

###### 3.3 MCP Augmentation and Agent-User Interaction

In addition to the quantitative results, we present case studies for the MCP tool use and user-agent interaction capability of MAI-UI.

MCP-Augmented Tasks Figure 8 shows two representative scenarios that benefit from MCP tool use, illustrating two core benefits: compressing multi step GUI operations into a few tool calls and enabling traditionally desktop only workflows on mobile.

- Figure 8a illustrates a realistic cross-application task that aggregates information and compares route distance. The user receives two apartment addresses by SMS and asks the agent to compare driving time from Alibaba Xixi Campus, then send the nearer address to Mia. Traditionally, this would require repeated switching between SMS and a maps app, copying and pasting addresses, and running two separate route searches. With an Amap MCP call, the agent can simply sets the campus as the origin and each candidate address from the SMS as the destination, and retrieves structured travel time and distance. This process compresses multiple GUI operations into a small number of tool calls, significantly reducing GUI interactions and improving end-to-end efficiency.
- Figure 8b shows a workflow that is usually handled on desktop environments. The user asks the agent to

retrieve the author and commit message from an GitHub repository, and send the summary by email. On

###### AndroidWorld Performance Comparison

###### Steps Allocation: Device vs. Cloud

Ferret-UI Lite-3B

28.0

UI-Tars-1.5-7B

30

Device 42.7%

Cloud 57.3%

UI-Venus-7B

49.1

ScaleCUA-32B

47.4

Qwen3-VL-32B

57.3

Seed1.5-VL

62.1

###### Tasks Fully Completed on Device

UI-Tars-1.5

49.1

MAI-UI-2B-DCC w/o ES

58.6

MAI-UI-2B-RS

57.8

Device 40.5%

MAI-UI-2B

49.1

MAI-UI-2B-DCC

65.5

MAI-UI-32B

73.3

10 20 30 40 50 60 70 80

- Figure 10: The native device–cloud collaboration (DCC) capability significantly improves the on-device model’s online performance, surpassing the random switch (RS) baseline, w/o error summary (ES) baseline, and several larger pure-cloud models by a substantial margin. Our native DCC system also improves efficiency, executing 42.7% of steps locally and completing 40.5% of tasks entirely on-device, thereby reducing cloud calls.

mobile, browsing commit history is inconvenient due to limited screen space. Through MCP API calls, the agent directly queries the repository and receives commit metadata in structured form, and extracts the fields required by the task. This case demonstrates that using MCP tools can not only compress UI operations but also expand the capability of mobile GUI agent, providing access to services that are commonly available in desktop applications.

Agent-User-Interaction To evaluate the agent’s ability to interact with the user when necessary, we present a case study of a file-sharing task that requires proactive clarification (Figure 9). The user instructs the agent to locate recent resume files in the Downloads folder and send them to an HR colleague, but several critical parameters are under-specified, including the recipient’s email address and the email body. Detecting these gaps, MAI-UI pauses execution and issues an ask_user action to request the missing details. After receiving the user’s response, the agent resumes the GUI trajectory: it auto-fills and sends the email. This example demonstrates MAI-UI’s proactive agent–user interaction capability, which is essential in realistic GUI tasks where instructions are often ambiguous or incomplete. Overall, the combination of MCP enhancement and Agent-User Interaction enables the model to better handle real-world tasks.

###### 3.4 Device-Cloud Collaboration Analysis

On-device capability gains. On the AndroidWorld online benchmark, we evaluate how device–cloud collaboration (DCC) enhances the on-device model’s performance. Our system uses MAI-UI-2B as the local agent and MAI-UI-32B as the cloud model. We compare our DCC system with four settings: (i) Local GUI agent, which uses the on-device model solely as the GUI agent; (ii) Cloud GUI agent, which relies on large scale cloud GUI agents; and (iii) Random switch (RS) baseline, which calls the cloud model at the same frequency as our DCC method but without monitor-guided switch. As shown in the left part of Figure 10, our DCC system achieves relative improvement of 33.4% over the ondevice model, significantly strengthening its ability to complete mobile tasks. It further surpasses the RS baseline by 7.7%, demonstrating that monitor-driven cloud-model switching effectively triggers cloud assistance and boosts collaboration quality. Finally, compared with several pure cloud GUI agent baselines, our device–cloud system achieves higher scores, demonstrating strong performance under

[Figure 224]

[Figure 225]

###### # User Instruction

Please help me delete all short-sleeve items from the shopping cart in the TaoDian app.

[Figure 226]

Agent user interaction

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Please provide the TaoDian account password for user “mashu”

[Figure 234]

[Figure 235]

The password for the TaoDian app is: xxx

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

[Figure 246]

[Figure 247]

[Figure 248]

Click [539, 1226]

Click [670, 2275] Click [539, 1226] Click [539, 1226]

Click [196, 986] Ask_user

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

##### …

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

Input_text xxx Click [93, 1399] Click [539, 1226] Click [999, 321] Click [67, 491] Scroll down

[Figure 268]

[Figure 269]

###### Local Agent Cloud Agent

- Figure 11: A pilot study demonstrating privacy protection in the device-cloud collaboration system. At step 3, the local agent deviates, as it repeatedly tapping the Login button without entering a password. The trajectory monitor flags the misalignment and proposes switch to the cloud agent. However, the privacy detection module detects sensitive credentials and blocks the switch, keeping execution on device. The local monitor ultimately corrects the trajectory and completes the task.

realistic deployment setting.

Efficiency gains versus cloud-only solution. We analyze cloud usage and on-device completion under our device–cloud collaboration system in the right part of Figure 10. Compared to cloud-only serving, our system reduces cloud model calls by 42.7%, substantially lowering serving cost and latency for using cloud model. On AndroidWorld, over 40% of tasks are completed entirely on device, further confirming that unnecessary cloud model calls are reduced. These results demonstrate that our device–cloud collaboration obtains substantial efficiency gains compared to cloud-only solutions.

Impact of error summary information. To assess the effectiveness of monitor-generated error summaries, we run an ablation that removes the error summary (ES) at switching and compare against our original system. As shown in the left part of Figure 10, compared to the baseline that without error summary, providing the error summary yields a +6.9 increase in task success rate, underscoring its importance for the cloud agent’s trajectory recovery process.

Privacy preservation. We present a pilot case study to showcase the system’s privacy protection capability. We introduce an additional local privacy monitor, which blocks cloud switch whenever privacy-sensitive content is present, even if a trajectory deviation is detected. As shown in Figure 11, for

0.9

| |Rewar|d (EMA, sp|an=6)| | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.8

0.797

Reward

0.7

0.6

0.5

30 60 90 120 150

Training Step

(a)

32 Envs 64 Envs

128 Envs 512 Envs

72

70.7

70

###### AndroidWorld

68

66.8

65.9

66

| |
|---|

65.5

64

| |
|---|

62

| |
|---|

60

58

15 30 60 120

Training Step

(b)

- Figure 12: MAI-UI-8B-RL training details: (a) train set reward trend (b) performance scaling with the number of environments.

a task involving user-sensitive content (password entry), the system continues on-device execution even when the monitor detects trajectory deviation (repeated clicks on the login icon in steps 2–4), thereby adhering privacy constraints. The Local Agent ultimately corrects the trajectory and completes the task. Throughout this case, no privacy-sensitive content is transmitted to the cloud, demonstrating that this device-cloud collaboration system effectively protects user privacy.

###### 3.5 Online RL Analysis

Performance Gains. As shown in Table 9, online RL consistently enhances performance across all model scales. Specifically, the 2B model achieves an absolute improvement of 4.0 percentage points (45.1% → 49.1%), the 8B model gains 6.0 percentage points (64.7% → 70.7%), and the 32B model improves by 3.5 percentage points (69.8% → 73.3%). These results correspond to relative improvements of 8.9%, 9.3%, and 5.0%, respectively, demonstrating that online RL effectively enhances agent performance regardless of model size. Figure 12a further illustrates this improvement, showing that the reward metric steadily increases throughout the training process.

Experimental Analysis. We perform ablation studies to evaluate the proposed key components in online RL, with the results shown in Table 9 and Figure 12.

Table 9: Online RL performance gains and ablations on standard GRPO and interaction budget per trajectory (max_env_steps)

Method AndroidWorld SFT vs. RL Comparison

MAI-UI-2B-SFT 45.1 MAI-UI-2B-RL 49.1(+4.0) MAI-UI-8B-SFT 64.7 MAI-UI-8B-RL 70.7(+6.0) MAI-UI-32B-SFT 69.8 MAI-UI-32B-RL 73.3(+3.5) RL Ablation on 8B

MAI-UI-8B-SFT 64.7 GRPO (max_env_steps=50) 66.5(+1.8) Our method (max_env_steps=15) 66.4(+1.7) Our method (max_env_steps=30) 68.5(+3.8) Our method (max_env_steps=50) 70.7(+6.0)

- • Comprison with standard GRPO. Standard GRPO applied after SFT yields a modest gain of +1.8 percentage points on AndroidWorld. In contrast, our enhanced GRPO with data curriculum, repetition penalty, and experience replay achieves a +6.0 percentage point improvement, delivering an additional

+4.2 percentage points over the baseline.

- • Effect of interaction budget. The maximum environment interaction budget per trajectory substantially influences performance. Extending the budget from 15 to 30 and subsequently to 50 steps yields progressive improvements (+1.7, +3.8, and +6.0 percentage points, respectively). A larger budget enables more extensive rollouts and provides richer exploration opportunities during training.
- • Impact of image resolution. Image resolution critically affects online RL efficiency, as higher resolutions introduce more visual tokens and slow down both training and inference. Leveraging the

# User Instruction

[Figure 270]

[Figure 271]

Create a new contact for Emilia Gonzalez. Their number is +14240925675.

[Figure 272]

###### # User Instruction

Open the file task.html in Downloads in the file manager; when prompted open it with Chrome. Then navigate the X to the bottom-right cell, by using the direction buttons.

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

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

###### …

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Click [551, 1456] Click [940, 2001] Click [324, 920] Input_text (Emilia)

Input_text (Gonzalez)

Click [171, 743] Click [296, 960] Click [728, 1778] Click [546, 2095] Click [296, 961] Click [728, 1778]

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

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Click [206, 814] Click [206, 814] Click [494, 814] Click [494, 814] Click [494, 814] Click [206, 814]

Input_text Click [115, 2323] (+14240925675)

Click [836, 197] Answer (Finished)

Click [324, 1521]

(a)

(b)

- Figure 13: MAI-UI demonstrates robustness to unexpected permission dialogs (a) and pop-ups (b) in AndroidWorld, and reliably resumes the task.

relative coordinate mechanism of Qwen3-VL, we found that 720p resolution achieves performance comparable to 1080p while providing a ∼50.1% speedup per step. Conversely, 540p resolution, despite faster processing, substantially degrades model performance due to insufficient detail for fine-grained UI element perception.

• Scaling parallel environments. Figure 12b shows how parallel environment count affects model performance. Increasing parallel environments from 32 to 512 significantly accelerates learning and improves final performance (65.5% → 70.7%). Training with fewer environments exhibits early saturation, indicating that limited environments constrain policy improvement. These findings highlight that scaling parallel environments to enhance exploration diversity is critical for overcoming performance bottlenecks in GUI agent RL training.

Case Studies: Enhanced Robustness. Figures 13 and 14 illustrate qualitative improvements in robustness after online RL training. Figure 13 demonstrates MAI-UI’s ability to handle unexpected permission dialogs and pop-ups during task execution. When creating a new contact for "Emilia Gonzalez," the agent encounters a notification permission request that was not present during offline training. The RL-trained model successfully dismisses the dialog and continues task execution without deviation, whereas the base model often fails to recover from such interruptions. Figure 14 showcases the agent’s capability to recover from failed actions in a complex expense management task. When instructed to delete duplicate expenses, the agent initially navigates to the wrong application. Nevertheless MAI-UI correct the trajectory and compleete the task in the following steps. These case studies highlight that online RL training substantially improves the agent’s robustness to real-world unpredictability. This is difficult to acquire through offline training alone, where the diversity of edge cases and failure modes is inherently limited.

###### 3.6 Analysis on Grounding

As demonstrated by our prior work in UI-Ins (Chen et al., 2025), our Instruction-as-Reasoning approach offers benefits from several perspectives:

• Reasoning that helps grounding. Prior studies find that general chain-of-thought reasoning can often degrade grounding performance (Lu et al., 2025; Yang et al., 2025a; Zhou et al., 2025). Inspired by how humans approach grounding, we train models to use diverse instruction perspectives as explicit

[Figure 340]

# User Instruction

[Figure 341]

Delete all but one of any expenses in pro expense that are exact duplicates, ensuring at least one instance of each unique expense remains.

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

###### …

[Figure 367]

[Figure 368]

Click [70, 187] Click [70, 187] Click [70, 187] Navigate_home Scroll (left) Click [494, 814]

Click [551, 2142]

Scroll (down) Click [889, 461]

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

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Long_press [551, 1456]

Click [70, 187] Click [296, 747] Click [1010, 187] Click [789, 2004] Click [546, 2227]

Click [901, 221] Click [546, 2227] Finished

- Figure 14: MAI-UI shows robustness in recovering from recovering from failures: the agent initially navigates to the wrong application, but MAI-UI correct the trajectory and complete the task.

reasoning pathways, making reasoning actionable and beneficial for GUI grounding.

- • Mitigating policy collapse in SFT + RL framework. Policy collapse often occurs in grounding after SFT with coordinate-only supervision (Zhang et al., 2025a). Instruction-as-Reasoning stabilizes RL by pretraining the model to generate diverse reasoning pathways, which enhances exploratory behavior and stabilizes policy optimization in the RL phase.
- • Emergent multi-perspective capabilities. After employ our instruction-as-reasoning method, the model can strategically select appropriate reasoning perspectives given different contexts and compose multiple perspectives into a cohesive one. Interestingly, it can also generate novel analytical angles beyond the four trained perspectives.

###### 4 Related Works

###### 4.1 GUI Grounding

GUI grounding is the foundational capability of GUI agents that maps natural language instructions to the locations of target elements in screenshots. Prior GUI grounding methods mainly focus on training in a Supervised Fine-Tuning (SFT) paradigm, such as JEDI (Xie et al., 2025a), OS-Atlas (Wu

- et al., 2024), Aguvis (Xu et al., 2025b), Uground (Gou et al., 2025) and Aria-UI (Yang et al., 2024b). Reinforcement learning methods, particularly GRPO (Guo et al., 2025a) have demonstrated remarkable sucess on various visual-language tasks, including Semantic Segmentation (Liu et al., 2025c), Visual Question-Answering (Liu et al., 2025d; Huang et al., 2025) and Temporal Video Grounding (Wang et al., 2025c). Consequently, recent efforts have increasingly focused on adapting RL for GUI grounding. GUI Grounding methods like GUI-R1 (Luo et al., 2025), GUI-Actor (Wu et al., 2025) and GTA1 (Yang et al., 2025a) play as an pioneer role in pure RL paradigm and surpass SFT-based methods by a large margin. However, a key limitation of a pure RL paradigm is that it overlooks the substantial benefit offered by an initial SFT stage. While InfiGUI-R1 (Liu et al., 2025a) achieved success with an SFT+RL framework by reframing GUI grounding as a trajectory-level task that encourages model reflection, the SFT+RL paradigm remains difficult to implement in practice. It is also demonstrated by Phi-Ground (Zhang

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

|[Figure 397]<br><br>[Figure 398]<br><br># User Instruction: A slideshow control button positioned in the Quick Access bar at the top of the PPT interface, next to the undo/redo buttons.|
|---|

[Figure 399]

[Figure 400]

# User Instruction: Jump to apps starting with the letter K in your installed applications list.

# User Instruction: Select the emoji that shows applause or appreciation for the content about pen techniques.

|[Figure 401]<br><br>[Figure 402]<br><br>| |
|---|
<br><br>|[Figure 403]|
|---|
|
|---|

[Figure 404]

[Figure 405]

[Figure 406]

|[Figure 407]|
|---|

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

# User Instruction: Check your unread GitHub notifications and updates.

||[Figure 412]|
|---|
|
|---|

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

# User Instruction: Make the selected text appear thicker for emphasis in your notes.

# User Instruction: View the explanation of Safari's tab and window behavior settings.

|[Figure 422]|
|---|

|[Figure 423]|
|---|

|:GTBbox : Zoom in for visualization<br><br>| |
|---|
<br><br>[Figure 424]<br><br>: Prediction of MAI-UI|
|---|

Figure 15: Grounding case studies across different operating systems.

- et al., 2025a) that SFT+RL framework is prone to policy collapse. Our grounding method overcomes this issue by using the SFT stage to teach the model diverse reasoning through different instruction perspectives, and then utilizes the RL stage to further incentivize the model to select the appropriate reasoning pathway, thereby establishing an effective example for the SFT+RL training paradigm.

###### 4.2 GUI Navigation

Moving beyond single-step grounding, research on GUI agents has advanced to GUI navigation tasks, where agents must execute multi-step action sequences to complete a user instruction. Early works achieve this goal through agent frameworks that contain multiple components, such as a planner and a grounding module (Wang et al., 2024a; Yang et al., 2024b; Zhou et al., 2024). For practical deployment and cost efficiency, recent efforts increasingly target unified vision–language–action models that jointly learn grounding and navigation (Qin et al., 2025a; Zhang et al., 2025b; Gu et al., 2025b). Most of these models are large-scale to enable stronger reasoning and planning, while a growing line of work builds competitive small on-device models for lower latency, improved privacy, and resilience to connectivity constraints (Yang et al., 2025b). To meet deployment needs, we release both large-scale and on-device models, and further integrate a native device–cloud collaboration capability. With the rapid development in GUI navigation, progress has also been made in real-world deployment of GUI agents (Google, 2025). In practice, however, current agents function as copilots rather than

standalone executors, making effective agent–user interaction a critical yet often overlooked capability. Moreover, most agents remain limited to pure UI manipulation, and only recently have efforts begun to equip them with extended SDK functions (Wang et al., 2025a). Integrating external tools such as the MCP can compress long, brittle UI action sequences into a handful of API calls and unlock desktop workflows that were previously infeasible on mobile.

###### 5 Conclusion

In this work, we present MAI-UI, a family of foundation GUI agents. To address challenges for realworld deployment of GUI agents, we introduce three main components: a self-evolving data pipeline that expands the navigation data to include user interaction and MCP tool calls, a native device–cloud collaboration system that routes execution by task state and data sensitivity, and an online RL framework with advanced system optimizations. Experimental results show that MAI-UI sets a new state of the art on grounding benchmarks and both online and offline evaluation of mobile use navigation. Benchmark results on MobileWorld and qualitative case studies verify effective agent–user interaction and MCPenabled tool use capability of MAI-UI. The device–cloud collaboration substantially improves on-device performance while reducing cloud model calls, yielding performance, privacy and cost benefits. Taken together, these advances make MAI-UI a step further toward practical foundation GUI agent for mobile use.

###### References

Anthropic. What is the model context protocol (mcp)? https://modelcontextprotocol.io/docs/get

ting-started/intro, 2024.

Anthropic. Claude 3.7 sonnet. https://www.anthropic.com/news/claude-3-7-sonnet, 2024. Accessed: 2025-08-02.

Anthropic. Our 3.5 models and computer use. https://www.anthropic.com/news/3-5-models-and-c

omputer-use, sep 2024. Accessed: 2025-09-22.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Liangyu Chen, Hanzhang Zhou, Chenglin Cai, Jianan Zhang, Panrong Tong, Quyu Kong, Xu Zhang, Chen Liu, Yuqi Liu, Wenxuan Wang, et al. Ui-ins: Enhancing gui grounding with multi-perspective instruction-as-reasoning. arXiv preprint arXiv:2510.20286, 2025.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents, 2024. URL https://arxiv.org/ abs/2401.10935.

Google DeepMind. Gemini computer use model. Google Blog, 2025a. URL https://blog.google/tech

nology/google-deepmind/gemini-computer-use-model/. Google DeepMind. Gemini 3 pro, 2025b. URL https://deepmind.google/models/gemini/pro/. Google. Gemini live, 2025. URL https://gemini.google/overview/gemini-live/.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/f orum?id=kxnoqaisCT.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on llm-as-a-judge, 2025a. URL https://arxiv.org/abs/2411.15594.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, Yue Wen, Jingya Dou, Fei Tang, Jinzhen Lin, Yulin Liu, Zhenlin Guo, Yichen Gong, Heng Jia, Changlong Gao, Yuan Guo, Yong Deng, Zhenyu Guo, Liang Chen, and Weiqiang Wang. Ui-venus technical report: Building high-performance ui agents with rft, 2025b. URL https://arxiv.org/abs/2508.10833.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833, 2025c.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. URL https://arxiv.org/abs/2503.06749.

Yangqin Jiang and Chao Huang. Lightagent: Mobile agentic foundation models. arXiv preprint arXiv:2510.22009, 2025.

Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, Zhidan Liu, Steven Hoi, and Yue Wang. Mobileworld: Benchmarking autonomous mobile agents in agent-user interactive, and mcp-augmented environments. arXiv preprint arXiv:2512.19432, 2025.

Kaixin Li, Meng Ziyang, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: GUI grounding for professional high-resolution computer use. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https://openreview.net/forum?id=Xa KNDIAHas.

Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on ui control agents, 2024. URL https://arxiv.org/abs/24 06.03679.

Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners, 2025a. URL https://arxiv.org/abs/2504.14239.

Yuhang Liu, Zeyu Liu, Shuanghe Zhu, Pengxiang Li, Congkai Xie, Jiasheng Wang, Xueyu Hu, Xiaotian Han, Jianbo Yuan, Xinyao Wang, et al. Infigui-g1: Advancing gui grounding with adaptive exploration policy optimization. arXiv preprint arXiv:2508.05731, 2025b.

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025c.

Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. Visionreasoner: Unified visual perception and reasoning via reinforcement learning. arXiv preprint arXiv:2505.12081, 2025d.

Zhaoyang Liu, Jingjing Xie, Zichen Ding, Zehao Li, Bowen Yang, Zhenyu Wu, Xuehui Wang, Qiushi Sun, Shi Liu, Weiyun Wang, Shenglong Ye, Qingyun Li, Xuan Dong, Yue Yu, Chenyu Lu, YunXiang Mo, Yao Yan, Zeyue Tian, Xiao Zhang, Yuan Huang, Yiqian Liu, Weijie Su, Gen Luo, Xiangyu Yue, Biqing Qi, Kai Chen, Bowen Zhou, Yu Qiao, Qifeng Chen, and Wenhai Wang. Scalecua: Scaling open-source computer use agents with cross-platform data. arXiv preprint arXiv:2509.15221, 2025e. URL https://github.com/OpenGVLab/ScaleCUA. Preprint.

Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024a.

Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024b. URL https://arxiv.org/abs/2408.00203.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.

Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.

Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M. Tamer Özsu, Aishwarya Agrawal, David Vazquez, Christopher Pal, Perouz Taslakian, Spandana Gella, and Sai Rajeswar. Ui-vision: A desktop-centric gui benchmark for visual perception and interaction, 2025. URL https://arxiv.org/abs/2503.15661.

OpenAI. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276. OpenAI. Developing a generalist computer-using agent. OpenAI, 2025. URL https://openai.com/ind

###### ex/computer-using-agent/. Accessed: October 22, 2025.

OpenAI. Introducing gpt-5. Technical report, OpenAI, 2025. URL https://openai.com/zh-Hans-CN/in

###### dex/introducing-gpt-5/.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv

- preprint arXiv:2501.12326, 2025a.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv

- preprint arXiv:2501.12326, 2025b.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2024a. URL https://arxiv.org/abs/2405.14573.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024b.

Bytedance Seed. Seed1.8 model card: Towards generalized real-world agency. arXiv preprint, December 2025a. Technical Report.

- ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025b.
- ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025c.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Gui-g2: Gaussian reward modeling for gui grounding, 2025. URL https://arxiv.org/abs/2507.15846.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025a.

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024a.

Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, et al. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024b.

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents, 2025b. URL https://arxiv.org/abs/2508.09123.

Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, Xiangnan Fang, Zewen He, Zhenbo Luo, Wenxuan Wang, Junqi Lin, Jian Luan, and Qin Jin. Time-r1: Post-training large vision language model for temporal video grounding, 2025c. URL https://arxiv.org/abs/2503.13377.

Yiqin Wang, Haoji Zhang, Jingqi Tian, and Yansong Tang. Ponder & press: Advancing visual gui agent towards general computer control, 2024c. URL https://arxiv.org/abs/2412.01268.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents, 2024. URL https://arxiv.org/abs/2410.23218.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025a. URL https: //arxiv.org/abs/2505.13227.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, and Caiming Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025b. URL https: //arxiv.org/abs/2505.13227.

Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. Androidlab: Training and systematic benchmarking of android autonomous agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2144–2166, 2025a.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction, 2025b. URL https://arxiv.org/abs/2412.04454.

JingJing Xie Xuehui Wang, Zhenyu Wu et al. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478, 2025.

Haolong Yan, Jia Wang, Xin Huang, Yeqing Shen, Ziyang Meng, Zhimin Fan, Kaijun Tan, Jin Gao, Lieyu Shi, Mi Yang, et al. Step-gui technical report. arXiv preprint arXiv:2512.15431, 2025.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024a. URL https://arxiv.org/abs/2407.10671.

Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, Ran Xu, Liyuan Pan, Caiming Xiong, and Junnan Li. Gta1: Gui test-time scaling agent, 2025a. URL https://arxiv.org/abs/2507.05791.

Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-ui: Visual grounding for gui instructions, 2024b. URL https://arxiv.org/abs/2412.16256.

Zhen Yang, Zi-Yi Dou, Di Feng, Forrest Huang, Anh Nguyen, Keen You, Omar Attia, Yuhao Yang, Michael Feng, Haotian Zhang, et al. Ferret-ui lite: Lessons from building small on-device gui agents. arXiv preprint arXiv:2509.26539, 2025b.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, et al. Enhancing visual grounding for gui agents via self-evolutionary reinforcement learning. arXiv preprint arXiv:2505.12370, 2025.

Miaosen Zhang, Ziqiang Xu, Jialiang Zhu, Qi Dai, Kai Qiu, Yifan Yang, Chong Luo, Tianyi Chen, Justin Wagle, Tim Franklin, et al. Phi-ground tech report: Advancing perception in gui grounding. arXiv preprint arXiv:2507.23779, 2025a.

Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, et al. Agentcpm-gui: Building mobile-use agents with reinforcement fine-tuning. arXiv preprint arXiv:2506.01391, 2025b.

Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, Huajun Chen, and Yuchen Eleanor Jiang. Symbolic learning enables self-evolving agents. 2024. URL https://arxiv.org/abs/2406.18532.

Yuqi Zhou, Sunhao Dai, Shuai Wang, Kaiwen Zhou, Qinglin Jia, and Jun Xu. Gui-g1: Understanding r1-

zero-like training for visual grounding in gui agents, 2025. URL https://arxiv.org/abs/2505.15810. Jinguo Zhu, Weiyun Wang, et al. Internvl3: Exploring advanced training and test-time recipes for

open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

###### A Additional Grounding Results

We present more grounding results of ScreenSpot-v2 and OSWorld-G Refine in Table 10, and Table 11.

- Table 10: Performance comparison on ScreenSpot-V2. We use ‘∗’ to denote the results evaluated by us. The best results are highlighted in bold, and the second-best results are underlined.

Model

Mobile Desktop Web

Avg. Text Icon. Text Icon. Text Icon.

Open-Source Models

Qwen3-VL-2B∗ (Bai et al., 2025a) 95.5 82.0 95.4 73.6 89.7 76.4 86.7 Phi-ground (Zhang et al., 2025a) 90.2 76.4 93.6 75.9 96.5 62.0 83.8 OS-Atlas-7B (Wu et al., 2024) 95.2 75.8 90.7 63.6 90.6 77.3 85.1 UGround-v1-7B (Gou et al., 2025) 83.6 90.5 85.8 86.3 95.5 83.2 87.7 UI-Tars-1.5-7B (Seed, 2025b) 92.2 81.5 91.0 84.2 95.5 84.5 89.0 SE-GUI-7B∗ (Yuan et al., 2025) 99.3 89.1 96.4 78.6 92.7 81.3 90.8 UI-TARS-7B (Qin et al., 2025a) 96.9 89.1 95.4 85.0 93.6 85.2 91.6 Qwen3-VL-8B∗ (Bai et al., 2025a) 97.9 84.8 95.9 87.9 95.7 83.7 91.7 GUI-Actor-7B∗ (Wu et al., 2025) 97.6 88.2 96.9 85.7 93.2 86.7 92.1 OpenCUA-7B (Wang et al., 2025b) - - - - - - 92.3 GTA1-7B (Yang et al., 2025a) 99.0 88.6 94.9 89.3 92.3 86.7 92.4 GUI-Owl-7B (Ye et al., 2025) 99.0 92.4 96.9 85.0 93.6 85.2 92.8 GUI-G2-7B∗ (Tang et al., 2025) 98.3 91.9 95.4 89.3 94.0 87.7 93.3 InfiGUI-G1-7B∗ (Liu et al., 2025b) 99.0 91.9 94.3 82.1 97.9 89.2 93.5 UI-Venus-7B (Gu et al., 2025b) 99.0 90.0 96.9 90.7 96.2 88.7 94.1

Qwen3-VL-32B∗ (Bai et al., 2025a) 96.2 90.0 97.4 85.0 95.7 89.7 93.0 GUI-Owl-32B (Ye et al., 2025) 98.6 90.0 97.9 87.8 94.4 86.7 93.2 OpenCUA-32B (Wang et al., 2025b) - - - - - - 93.4 GTA1-32B (Yang et al., 2025a) 99.7 90.5 99.0 94.3 95.7 90.1 95.2 UI-Venus-72B (Gu et al., 2025b) 99.7 93.8 95.9 90.0 96.2 92.6 95.3

Ours

MAI-UI-2B 99.3 87.2 97.4 88.6 94.0 84.7 92.5 MAI-UI-8B 99.3 89.1 99.0 92.1 97.9 91.1 95.2 MAI-UI-32B 99.0 92.9 99.5 93.6 97.4 94.6 96.5

- Table 11: Performance comparison of state-of-the-art models on the OSWorld-G-Refine. The best results are highlighted in bold, and the second-best results are underlined.

Text Matching

Element Recognition

Layout Understanding

Fine-grained Manipulation

Agent Model

Refusal Avg

Proprietary Models Operator OpenAI (2025) - - - - - 57.8

Open-Source Models

Qwen3-VL-2B∗ (Bai et al., 2025a) 69.3 60.9 69.2 45.0 - 57.4 Jedi-3B Xie et al. (2025a) - - - - - 61.0 Jedi-7B Xie et al. (2025a) - - - - - 63.8 UI-TARS-1.5-7B Seed (2025b) 52.6 75.4 72.4 66.7 0.0 64.2 Qwen3-VL-8B∗ (Bai et al., 2025a) 73.9 68.2 73.1 54.4 - 64.4 GTA1-7B (Yang et al., 2025a) 63.2 82.1 74.2 70.5 0.0 67.7

Qwen2.5-VL-32B Bai et al. (2025b) 57.9 70.2 73.8 49.2 0.0 59.6 OpenCUA-32B Wang et al. (2025b) 63.2 79.9 84.9 62.1 7.4 70.2 Qwen3-VL-32B∗ (Bai et al., 2025a) 77.4 73.6 76.3 57.7 - 69.0 GTA1-32B (Yang et al., 2025a) 63.2 83.6 84.4 70.5 0.0 72.2

Ours

MAI-UI-2B 70.9 69.1 72.7 47.7 - 63.5 + Zoom-In 71.3 71.8 78.3 51.0 - 66.3 MAI-UI-8B 77.4 73.0 78.3 55.7 - 68.6

- + Zoom-In 79.3 78.8 84.2 59.7 - 72.9 MAI-UI-32B 79.7 79.4 81.0 61.7 - 73.9

- + Zoom-In 79.3 79.7 84.2 64.4 - 75.0

