# arXiv:2504.14239v1[cs.AI]19Apr2025

[Figure 1]

## InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners

Yuhang Liu∗

Zhejiang University liuyuhang@zju.edu.cn

Pengxiang Li∗

Dalian University of Technology lipengxiang@mail.dlut.edu.cn

Congkai Xie

Reallm Labs xieck13@gmail.com

Xavier Hu

Zhejiang University xavier.hu.research@gmail.com

Xiaotian Han

xiaotian.sky.han@gmail.com

Shengyu Zhang†

Zhejiang University sy_zhang@zju.edu.cn

Hongxia Yang

The Hong Kong Polytechnic University hongxia.yang@polyu.edu.hk

Fei Wu

Zhejiang University wufei@zju.edu.cn

### Abstract

Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, twostage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-toerror steps. These approaches enhance the agent’s planning abilities and self-correction capabilities. Experimental results confirm that InfiGUI-R1 achieves strong performance in both cross-platform GUI grounding and trajectory tasks, proving competitive against previous agents, even those with significantly larger parameters. Resources are available at https://github.com/Reallm-Labs/InfiGUI-R1.

∗Both authors contributed equally to this research. †Corresponding author.

### Keywords

GUI Agents, MLLMs, Reinforcement Learning

[Figure 2]

Figure 1: Performance comparison of various GUI agents on the ScreenSpot-Pro benchmark. Our model, InfiGUI-R1-3B marked with a star, demonstrates competitive performance against models with larger parameter counts.

### 1 Introduction

Graphical User Interface (GUI) agents, increasingly powered by advances in Multimodal Large Language Models (MLLMs) [5, 22, 28, 39, 48] hold significant promise for automating a wide range of tasks on computing devices such as mobile phones and computers [9, 42]. These agents interact with digital environments through visual interfaces, aiming to enhance user productivity and broaden the scope of automated task completion.

Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches either rely on manually designed reasoning templates or lack GUI-specific optimization, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, many existing MLLM-based GUI agents continue to operate as

Reactive Actors [10, 29], relying primarily on implicit reasoning. This implicit reasoning often lacks the sufficient depth required for complex, multi-step GUI tasks demanding sophisticated planning and adaptive error recovery. Such tasks necessitate not only precise spatial understanding of dense visual layouts but also the ability to effectively integrate cross-modal information (visual-spatial understanding into textual reasoning) and engage in the deliberative processes crucial for robust, long-horizon task execution.

We argue that fundamentally advancing GUI agent capabilities requires a paradigm shift: moving beyond reactive execution towards agents that function as Deliberative Reasoners. These agents should explicitly incorporate reasoning processes between perception and action (Perception → Reasoning → Action), enabling them to plan ahead, decompose complex goals, understand spatial relationships deeply, and reflect upon past actions to correct mistakes. This transition is crucial for handling the complexities and dynamic nature of real-world GUI environments.

To enablethistransformation,weintroduce theActor2Reasoner framework, a reasoning-centric methodology designed to progressively evolve GUI agents from Reactive Actors to Deliberative Reasoners. Our framework culminates in InfiGUI-R1-3B, an MLLMbased agent demonstrating enhanced reasoning and robustness. The Actor2Reasoner framework tackles two core challenges: 1) reliably injecting foundational reasoning capabilities, particularly bridging the critical cross-modal gap between visual-spatial perception and textual reasoning, to achieve the initial leap from Actor to Reasoner; and 2) refining and elevating the reasoning quality of this foundational Reasoner to instill advanced planning and reflection capabilities, ultimately reaching the Deliberative stage.

The Actor2Reasoner framework unfolds in two distinct stages:

- • Stage 1: Reasoning Injection (Laying the Foundation for the Reasoner): This stage focuses on the pivotal transition from Actor to Reasoner. We employ Spatial Reasoning Distillation, leveraging trajectories from a powerful reasoning teacher model that include explicit spatial reasoning steps. By training the base MLLM on this distilled data via Supervised Fine-Tuning (SFT), we guide it to break the direct Perception → Action link and explicitly incorporate reasoning, especially spatial reasoning crucial for GUI tasks. This establishes the foundational (Perception

→ Reasoning → Action) pattern, overcoming a key limitation of standard MLLMs in integrating visual-spatial understanding into their reasoning flow.

- • Stage 2: Deliberation Enhancement (Refining into a Deliberative Reasoner): Building upon the Reasoner established in Stage 1, this stage uses Reinforcement Learning (RL) to refine its capabilities towards deliberation. This refinement strategically enhances the two core facets of deliberative reasoning: planning and reflection. Two key innovations drive this process:
- • Sub-goal Guidance: To enhance the agent’s forward-looking planning and task decomposition abilities, we guide it to generate explicit intermediate sub-goals during its reasoning process. The alignment of these generated sub-goals with ground truth provides an intermediate reward signal, effectively training the agent’s capacity for proactive planning ("Total Goal → Sub-goal → Action").
- • Error Recovery Scenario Construction: Complementing the planning focus, this innovation cultivates the agent’s ability

to look backward and adjust through reflective self-correction – a hallmark of deliberation. We actively construct scenarios within the RL environment that simulate error states or recovery moments (e.g., having just executed an incorrect action or needing to get "back on track" after an error). Training within these scenarios, using targeted rewards, compels the agent to learn adaptive strategies like escaping error states (e.g., using a ’back’ action) and adjusting plans after recognizing a mistake. This directly shapes the agent’s ability to reflect on its actions and recover from failures, enhancing its robustness.

Together, our framework provides a pathway to imbue GUI agents with the reasoning, planning, and reflection capabilities necessary for task automation. We validate the effectiveness of InfiGUI-R1-3B, trained using our Actor2Reasoner framework, on a suite of challenging benchmarks designed to assess core GUI agent competencies. These include tasks requiring precise GUI element grounding across platforms (e.g., ScreenSpot, ScreenSpot-Pro [21, 24]) and those demanding complex, long-horizon task execution with planning and adaptation (e.g., AndroidControl[26]). Our experimental results demonstrate significant improvements. InfiGUI-R1-3B achieves state-of-the-art cross-platform grounding capabilities (87.5% avg on ScreenSpot) and strong performance in executing complex, long-horizon tasks (71.1% success rate on AndroidControl-High) among models with comparable parameter counts. These findings confirm our framework’s ability to cultivate sophisticated planning and reflection abilities, substantially advancing the agent’s capacity for deliberate, robust, and effective GUI task automation.

Our main contributions are threefold:

- • We propose the Actor2Reasoner framework, a novel twostage training methodology designed to systematically transform MLLM-based GUI agents from Reactive Actors into Deliberative Reasoners by progressively injecting and refining reasoning capabilities.
- • We introduce three key technical innovations within this framework: Spatial Reasoning Distillation to establish foundational cross-modal reasoning, Sub-goal Guidance to enhance planning reasoning, and Error Recovery Scenario Construction to cultivate reflective error correction abilities through targeted RL.
- • We develop InfiGUI-R1-3B, an MLLM-based GUI agent trained via our framework, and demonstrate its effectiveness through comprehensive experiments.

### 2 Related Works 2.1 Multimodal LLMs

Large Language Models (LLMs) [6, 13, 47, 53] have significantly enhanced the capabilities of AI systems in tackling a wide range of tasks [17, 25], thanks to their exceptional ability to process complex semantic and contextual information. The remarkable power of LLMs has also inspired exploration into their potential for processing multimodal data, such as images. Typically, the architecture of Multimodal Large Language Models (MLLMs) consists of three main components: a pre-trained large language model, a trained modality encoder, and a modality interface that connects the LLM with the encoded modality features. Various vision encoders, such as ViT

|InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners Preprint, Under review, April 2025<br><br>[12], CLIP [41], and ConvNeXt [34], extract visual features, which are integrated using techniques like adapter networks [31], crossattention layers [3], and visual expert modules [49]. These methods have facilitated the development of high-performing MLLMs, such as Qwen-VL [7], GPT-4 Vision [36], BLIP-2 [23] and InfiMM [32], thus opening new avenues for LLMs in processing GUI tasks.<br><br>2.2 MLLM-based GUI Agents<br><br>Agents are AI systems that perceive their environments, make decisions, and take actions to complete specific tasks. The emergence of LLMs with human-level reasoning ability has significantly advanced the development of agents. For GUI tasks, earlier systems relied on LLMs to read and interpret structured representations such as HTML code [50]. However, recent works have demonstrated that directly interacting performance [16]. been proposed, lev understanding.<br><br>|with the visual form of GUIs leads to better Consequently, MLLM-based GUI agents have<br><br>leveraging visual perception alongside language<br><br>systems have pioneered this area. ILuLLaVA to enhance general GUI comprehen-<br><br>[58] explored mobile app usage through auCogAgent [15] introduced high-resolution capture UI detail, and Ferret-UI-anyres [57] supresolutions to handle diverse device settings. have introduced modular and lightweight<br><br>GUI Understanding:<br><br>• Screen2Words<br>• Screen Annotation Question Answering:<br>• ScreenQA<br>• Complex QA Instruction Grounding:<br>• RicoSCA<br>• Widget Caption<br>• ...<br><br>• LLaVA-OneVision<br>• PixMo<br><br><br>Stage 1: Reasoning Injection Stage 2: Deliberation Enhancement<br><br>• Glaive Function<br><br>Calling<br><br>Expected to open clock app by tapping icon... (Succeed )<br><br>[Figure 3]<br><br>Task: Set an alarm for 7am.<br><br>[Figure 4]<br><br>Strategic Layer: Summary: Home screen → App Drawer → Clock interface Planning: Alarm tab → Create new → Set 7am → Save<br><br>Tactical Layer: Current Step Reasoning: Need to access alarm tab from current clock screen with multiple function tabs Grounding: Tap alarm icon in top left corner<br><br>{"name": "tap", "arguments": {"point": {"x": 115, "y": 67}}}<br><br>Alarm tab will open showing new alarm option...<br><br>Reflection<br><br>Hierarchical Reasoning<br><br>Action<br><br>Expectation<br><br>[Figure 5]<br><br>[Figure 6]<br><br>History Current Environment<br><br>Reasoning & Action<br><br>Reasoning & Action<br><br>...<br><br>Input<br><br>GUI-Specific<br><br>[Figure 7]<br><br>Instruction<br><br>Answer / Action<br><br>Screenshot<br><br>General Tool Usage<br><br>Instruction<br><br>[Figure 8]<br><br>+ Answer Instruction Action<br><br>Operate<br><br>+<br><br>Prior Expectation<br><br>Subsequent Reflection<br><br>into a Basic Reasoner (Perception Stage 2: Deliberation Enhancement<br><br>ward) for forward-looking task covery Scenario Construction ( self-correction, culminating in<br><br>based GUI agents. The core objective reactive behavior towards deliberative tomation. This framework comprises|
|---|
<br><br>Several representative fine-tuned sion, while AppAgent tonomous interactions. encoders to better c ported flexible screen<br><br>|Expectation-ReflexionvUI [20] Reasoning|
|---|
<br><br>More recent works<br><br>architectures aimed at improving generalization and deployment efficiency. InfiGUIAgent [33] proposed a two-stage approach, combining general pretraining on grounding and QA tasks with synthetic fine-tuning for hierarchical planning and reasoning. UI-TARS [40] extended this by using a unified vision-language interface across mobile, web, and desktop environments, incorporating reflection and milestone tracking mechanisms to boost task success rates. In parallel, AgentS2 [1] adopted a generalist-specialist framework, decoupling high-level reasoning from domain-specific grounding modules and enabling long-horizon planning with Mixture of Grounding mechanisms.<br><br>In terms of input, recent agents prioritize screenshot-level visual understanding, optionally enhanced with layout or OCR-based textual cues. Techniques such as set-of-mark prompting [54] and chain-of-action reasoning [38] have been employed to improve grounding accuracy and task planning. To further improve interaction efficiency, agents such as UI-R1 [35], GUI-R1 [52] replace large-scale supervision with rule-based reinforcement learning, achieving competitive performance with minimal expert data.<br><br>Moreover, to support real-world usability, newer agents are testedonincreasinglycomplexenvironments. UI-TARS and AgentS2 report strong performance on OSWorld and AndroidWorld benchmarks, showing robust cross-platform generalization. GUI-Xplore [44] further introduces a one-shot adaptation setting, encouraging agents to build structural UI maps via autonomous exploration<br><br>[Figure 9]<br><br>[Figure 10]<br><br>GUI Understanding<br><br>Instruction GroundingQuestion Answering<br><br>|Hierarchical Reasoning|
|---|
<br><br>Input<br><br>训练和推理<br><br>|Actor2Reasoner|
|---|
<br><br>Reactive Actor<br><br>[Figure 11]<br><br>Basic Reasoner<br><br>Deliberative Reasoner<br><br>Basic Reasoner<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>Stage 1: Reasoning Injection Stage 2: Deliberation Enhancement<br><br>Supervised Fine-Tuning<br><br>Reinforcement Learning<br><br>|Spatial|
|---|
<br><br>Spatial Info<br><br>Distill<br><br>Teacher Model<br><br>Pinpoint<br><br>Error Escape<br><br>Back on Track<br><br>Error<br><br>Recovery Sub-goal<br><br>[Figure 17]<br><br>Reward Reflect<br><br>Figure 2: Overview of the Actor2Reasoner framework, a twostage methodology for progressively transforming a Reactive Actor into a Deliberative Reasoner. Stage 1: Reasoning Injection uses Supervised Fine-Tuning (SFT) with Spatial Reasoning Distillation—identifying reasoning bottlenecks (Pinpoint) and leveraging a teacher model (Distill)—to instill foundational cross-modal reasoning and transition the agent<br><br>on → Reasoning → Action).<br><br>nt applies RL to refine planning and reflection capabilities, using Sub-goal Guidance (Re-<br><br>decomposition and Error ReReflect) for backward-looking<br><br>a Deliberative Reasoner.<br><br>ve is to transition agents from tive reasoning for GUI task aues two stages designed to first<br><br>establish foundational reasoning and then refine it towards advanced deliberation. Section 3.1 details the methodology for Stage 1, focusing on reasoning injection via Spatial Reasoning Distillation. Subsequently, Section 3.2 details the methodology for Stage 2, where RL is employed to enhance deliberation through Sub-goal Guidance and Error Recovery Scenario Construction.<br><br>3.1 Stage 1: Reasoning Injection<br><br>The primary objective of Stage 1 is to accomplish the fundamental transition from a Reactive Actor (Perception → Action) to a foundational Reasoner (Perception → Reasoning → Action). This transition is critical because standard MLLMs often struggle to effectively integrate the rich visual-spatial information present in GUI screenshots into their textual reasoning processes. This limitation hinders their ability to handle the GUI tasks that demand precise spatial understanding and grounding. To address this, Stage 1 employs Spatial Reasoning Distillation, which is designed to explicitly inject spatial reasoning capabilities into the agent.<br><br>Spatial Reasoning Distillation leverages the reasoning capabilities of a powerful teacher model to generate high-quality reasoning trajectories, which are then used to train the target MLLM (the student). The core idea is to guide the student model to learn not just the correct action, but also the intermediate reasoning steps—particularly those involving spatial logic—that lead to that|
|---|

action. This process is implemented through the following steps:

before task execution.

3.1.1 Pinpointing Reasoning Bottleneck Samples. To maximize the efficiency of distillation, we first identify interaction steps where the base MLLM’s failure is most likely attributable to a lack of reasoning, rather than fundamental perception or action execution deficits. We term these Reasoning Bottleneck Samples. This

### 3 Actor2Reasoner

We introduce the Actor2Reasoner framework, a reasoning-centric, progressive training methodology designed to systematically enhance the capabilities of Multimodal Large Language Model (MLLM)

identification employs a two-step criterion for each step 𝑠 in a given trajectory:

- (i) The base MLLM 𝑀, when given the current GUI screenshot 𝐼𝑠 and the overall task goal 𝐺, fails to predict the correct action. Let 𝑎high = 𝑀(𝐼𝑠,𝐺).
- (ii) However, when provided with the additional ground-truth sub-goal 𝑔𝑠 for that specific step, the same model 𝑀 successfully predicts the correct action. Let 𝑎low = 𝑀(𝐼𝑠,𝐺,𝑔𝑠).

Formally, the set of reasoning bottleneck steps 𝑆bottleneck is defined as:

𝑆bottleneck =

{𝑠 | IsCorrect(𝑎high) = False ∧ IsCorrect(𝑎low) = True}

These samples represent steps where the primary difficulty lies in inferring the immediate task (𝑔𝑠) from the overall goal (𝐺) based on the visual context (𝐼𝑠), making them ideal candidates for reasoning injection. We use a base MLLM such as Qwen2.5-VL-3B-Instruct for this filtering process.

- 3.1.2 Generating Spatial Reasoning Trajectories. For each step 𝑠 ∈

𝑆bottleneck, we generate a detailed reasoning trajectory using a highcapability teacher model. This involves:

Spatial Information Extraction and Compression. We extract relevant structural and spatial information (e.g., element types, text content, coordinates, hierarchy) from the accessibility tree (a11y tree) associated with the GUI screenshot 𝐼𝑠. Irrelevant attributes and elements are filtered out. A powerful MLLM (e.g., Qwen2.5-VL-32BInstruct) is then employed to compress this processed information into a concise textual description 𝐷spatial, which consists of a detailed description of the GUI page, including all relevant elements’ coordinate information and descriptions for the specific step, capturing the essential spatial layout and key element details.

Reasoning Trajectory Generation. The compressed spatial description 𝐷spatial, the available action space description, and the overall goal 𝐺 are fed as input to a powerful large language model with strong reasoning capabilities (e.g., QwQ-32B [46]). This teacher model is prompted to generate both an explicit reasoning text 𝑅teacher and the corresponding action 𝑎teacher. Crucially, 𝑅teacher

is guided to articulate the logical steps, including using the spatial information in 𝐷spatial for element localization, relationship assessment, and action justification.

- 3.1.3 Injecting Reasoning via SFT. The generated pairs (𝑅teacher, 𝑎teacher) are first filtered to ensure quality via rejection sampling

based on the correctness of the predicted action 𝑎teacher. The highquality pairs are then used to fine-tune the base MLLM. The SFT objective trains the student model to predict the teacher’s reasoning and action when given the GUI screenshot and the overall goal: (𝐼𝑠,𝐺) → (𝑅teacher,𝑎teacher). By learning to explicitly generate or implicitly simulate these reasoning steps before outputting the action, the student model internalizes the Perception → Reasoning → Action pattern.

Upon completion of Stage 1, the resulting model is a foundational Reasoner equipped with enhanced spatial understanding and the basic ability to connect perception to action through an intermediate reasoning process.

### 3.2 Stage 2: Deliberation Enhancement

Building upon the foundational Reasoner established in Stage 1, the objective of Stage 2 is to refine its capabilities, transforming it into a Deliberative Reasoner. This stage employs RL with rulebased rewards as the primary mechanism for enhancement. The core idea is to cultivate the agent’s ability for more sophisticated, "deliberative" decision-making by specifically targeting two aspects: forward-looking planning and backward-looking reflection/correction. These aspects are addressed through two key innovations integrated into the RL process: Sub-goal Guidance (detailed in Section 3.2.2) to bolster planning and task decomposition, and Error Recovery Scenario Construction (detailed in Section 3.2.3) to foster self-correction and robustness.

3.2.1 Reinforcement Learning Setup. We utilize RL to further optimize the agent’s policy beyond supervised learning. Specifically, we adopt the REINFORCE Leave-One-Out (RLOO) algorithm [2], which effectively reduces the variance of policy gradient estimates by employing the average reward of other samples within the same batch as a baseline for the current sample. This "leave-one-out" baseline strategy obviates the need for training a separate value or critic model, thereby simplifying the training architecture. The RLOO policy gradient ∇𝜃 𝐽 (𝜃) is estimated as:

∇𝜃 𝐽 (𝜃) ≈

  

  

∑︁

∑︁𝑘

1 𝑘

1 𝑘 − 1

∇𝜃 log𝜋𝜃 (𝑦(𝑖)|𝑥)

𝑅(𝑦(𝑗),𝑥)

𝑅(𝑦(𝑖),𝑥) −

𝑖=1

𝑗≠𝑖

where 𝑘 is the number of output sequences 𝑦(𝑖) sampled from the current policy 𝜋𝜃 given input 𝑥, and 𝑅(𝑦,𝑥) is the reward function evaluating the quality of output 𝑦.

The design of the reward function 𝑅(𝑦,𝑥) is crucial for guiding

the agent’s learning trajectory. Our total reward 𝑅total integrates assessments of both output format correctness and task execution accuracy:

𝑅total = 𝑤𝑓 · 𝑅format + 𝑤𝑎 · 𝑅acc

Here, 𝑅format checks if the model’s output 𝑦 conforms to the expected format (e.g., putting the reasoning process within <think>

</think> tags), yielding 1 if valid and 0 otherwise. 𝑅acc measures the accuracy of the content, and is calculated only if 𝑅format = 1, ensuring the agent first learns to generate structurally valid outputs. 𝑤𝑓 and 𝑤𝑎 are weighting hyperparameters (𝑤𝑓 + 𝑤𝑎 = 1).

The accuracy reward 𝑅acc is tailored to the specific task type: Agent Trajectory Task Reward (𝑅agent): For evaluating sequences

of GUI actions, we provide fine-grained feedback by combining rewards for the action type and its parameters:

𝑅agent = 𝑤𝑡 · 𝑅type + 𝑤𝑝 · 𝑅param

where 𝑤𝑡 + 𝑤𝑝 = 1. 𝑅type grants a reward of 1 if the predicted action type (e.g., ‘click‘, ‘type‘) matches the ground truth, and 0 otherwise. 𝑅param provides a stricter reward, granting 1 only if both the action type and all its parameters match the ground truth, and 0 otherwise. (Note: This reward is further refined by Sub-goal Guidance in Section 3.2.2).

Grounding Task Rewards: For evaluating GUI element localization:

- • Point Localization Reward (𝑅point): Given a predicted point coordinate (𝑥𝑝,𝑦𝑝) and the ground-truth bounding box 𝐵gt of the target element, the reward is 1 if the point falls within the box, and 0 otherwise:

𝑅point = I((𝑥𝑝,𝑦𝑝) ∈ 𝐵gt)

- • Bounding Box Reward (𝑅bbox): We compute the Intersection over Union (IoU) between the predicted bounding box 𝐵pred and the ground-truth box 𝐵gt. To avoid penalizing minor deviations excessively while encouraging significant overlap, we use

a threshold 𝜏IoU. The reward is 1 if the IoU meets or exceeds the threshold, otherwise it is the IoU scaled by the threshold:

1 if IoU(𝐵pred,𝐵gt) ≥ 𝜏IoU IoU(𝐵pred,𝐵gt)

𝑅bbox =

𝜏IoU if IoU(𝐵pred,𝐵gt) < 𝜏IoU

Other Task Rewards (𝑅other): For auxiliary tasks potentially included in the training mix (e.g., VQA, multiple-choice questions), we use Exact Match (EM) or mathematical expression verification against the ground truth 𝑦gt to determine correctness:

𝑅other = I(ExactMatch(𝑦ans,𝑦gt) ∨ MathVerify(𝑦ans,𝑦gt))

To ensure the agent enhances its GUI-specific deliberation skills without compromising its general multimodal understanding and visual grounding foundations, the RL training phase utilizes a diverse mixture of data. This includes the core GUI trajectory data (e.g., from AndroidControl [26]), GUI element grounding data (e.g., from widget captioning datasets [27]), general-purpose multimodal question-answering datasets, and object detection datasets (e.g., from COCO [30]).

Following established practices for eliciting reasoning [43, 56], we employ a system prompt that explicitly instructs the model to first articulate its reasoning process internally before providing the final action. The specific prompt used is:

#### System Prompt for Reasoning

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags.

- 3.2.2 Sub-goal Guidance. To elevate the foundational Reasoner towards a Deliberative Reasoner capable of sophisticated planning, a core aspect of Stage 2 focuses on enhancing its task decomposition ability. Standard MLLMs often falter when required to independently infer the necessary intermediate steps from a high-level objective within a complex GUI environment. Sub-goal Guidance is specifically designed to address this limitation within the RL framework by incentivizing the agent to formulate and pursue accurate sub-goals, thereby fostering more structured and effective planning. This is achieved by assessing the quality of the sub-goal implied within the agent’s reasoning process.

Sub-goal Quality Assessment. We incentivize accurate sub-goal formulation by integrating its assessment into the agent’s reward

structure during RL training. We assess the quality of the implicitly generated sub-goal within the reasoning text.

During training, for each step, we employ a lightweight scoring LLM to analyze the agent’s reasoning output (the text within <think>...</think> tags) and attempt to extract the implied subgoal, denoted as 𝑠𝑔extracted. This extracted sub-goal 𝑠𝑔extracted is then compared against the corresponding ground-truth sub-goal 𝑠𝑔gt (obtained from dataset annotations1). Based on the degree of semantic match between 𝑠𝑔extracted and 𝑠𝑔gt, a raw score 𝑆raw is assigned on a scale of 1 to 10. If the scoring LLM fails to extract a sub-goal from the reasoning text, 𝑆raw is set to 0. This raw score is then normalized to the range [0, 1] to produce the final sub-goal reward:

𝑆raw 10

𝑅subgoal =

This normalized score, 𝑅subgoal, serves as an intermediate reward signal reflecting the quality of the agent’s planning for the current step. To specifically encourage correct planning even when the final action execution fails, we integrate 𝑅subgoal into the Agent Trajectory Task Reward 𝑅agent (introduced in Section 3.2.1). The formulation is modified as follows, incorporating a dedicated weight 𝑤𝑠:

𝑤𝑡 · 𝑅type + 𝑤𝑝 · 𝑅param if 𝑅param = 1 𝑤𝑡 · 𝑅type + 𝑤𝑠 · 𝑅subgoal if 𝑅param = 0

𝑅agent =

where 𝑤𝑡,𝑤𝑝,𝑤𝑠 are non-negative weights, and typically 𝑤𝑠 is set lower than 𝑤𝑝 to prioritize full action correctness when achievable. This conditional reward structure provides targeted feedback on the planning quality when the agent struggles with accurate action execution, thereby guiding the learning process towards better intermediate reasoning and task decomposition.

3.2.3 Error Recovery Scenario Construction. While Sub-goal Guidance enhances forward-looking planning, developing a Deliberative Reasoner also necessitates the ability to reflect upon and recover from errors—a capability often missing in standard GUI agents prone to irrecoverable failures. To cultivate robustness and adaptability, we utilize Error Recovery Scenario Construction, a technique that directly targets the agent’s reflective and corrective reasoning abilities by integrating specific failure-recovery situations into the RL training process. This mechanism complements planning by strengthening the agent’s capacity for backwardlooking adjustment.

Identify Prone-to-error Steps: To maximize training efficiency, we first identify interaction steps where the agent demonstrates instability. For a given step 𝑠, we employ the base model (e.g., Qwen2.5VL-3B-Instruct) to sample 𝑁𝑠𝑎𝑚𝑝𝑙𝑒 action sequences at a heightened temperature𝑇. Steps whose success rate 𝑃success(𝑠) falls between 0 and 1 (0 < 𝑃success(𝑠) < 1) are designated as Prone-to-error Steps, forming the set 𝑆error_prone. These steps signify situations where the agent possesses the capacity for correct action but is also susceptible to errors, presenting optimal opportunities for learning corrective strategies. Training on steps the agent always masters or always fails is less efficient for learning recovery; the former

1https://github.com/google-research/google-research/tree/master/android_control

Accuracy (%)

Model

Avg. Mobile Desktop Web

Text Icon Text Icon Text Icon Proprietary Models

GPT-4o [37] 30.5 23.2 20.6 19.4 11.1 7.8 18.8 Claude Computer Use [4] - - - - - - 83.0 Gemini 2.0 (Project Mariner) [11] - - - - - - 84.0

General Open-source Models

Qwen2-VL-7B [48] 61.3 39.3 52.0 45.0 33.0 21.8 42.9 Qwen2.5-VL-3B [8] - - - - - - 55.5 Qwen2.5-VL-7B [8] - - - - - - 84.7

GUI-specific Models CogAgent [15] 67.0 24.0 74.2 20.0 70.4 28.6 47.4 SeeClick [10] 78.0 52.0 72.2 30.0 55.7 32.5 53.4 UGround-7B [14] 82.8 60.3 82.5 63.6 80.4 70.4 73.3 OS-Atlas-7B [51] 93.0 72.9 91.8 62.9 90.9 74.3 82.5 ShowUI-2B [29] 92.3 75.5 76.3 61.1 81.7 63.6 75.1 Aguvis-7B [18] 95.6 77.7 93.8 67.1 88.3 75.2 84.4 UI-R1-3B [35] - - 90.2 59.3 85.2 73.3 GUI-R1-3B [52] - - 93.8 64.8 89.6 72.1 GUI-R1-7B [52] - - 91.8 73.6 91.3 75.7 UI-TARS-2B [40] 93.0 75.5 90.7 68.6 84.3 74.8 82.3

Ours

InfiGUI-R1-3B 97.1 81.2 94.3 77.1 91.7 77.6 87.5

#### Table 1: Performances on various platforms (Mobile, Desktop, Web) on ScreenSpot. All experiments were conducted using raw screenshot information. Results marked in bold represent the best performance, and those underlined indicate the second-best performance.

needs no correction, while the latter might indicate deeper issues potentially confounded by naive recovery training.

Constructing Recovery Scenarios: For each prone-to-error step 𝑠 ∈ 𝑆error_prone, we construct two distinct types of scenarios for RL training, each designed to teach a specific aspect of error handling: Error Escape Scenario. The primary objective here is to train the agent to recognize it has entered an erroneous state and execute an appropriate "escape" action (e.g., pressing the back button). To simulate this, we select an incorrect action 𝑎𝑠err sampled during the identification phase, which leads to an unintended subsequent observation 𝐼𝑠err+1. The RL agent is then presented with this error observation 𝐼𝑠err+1 alongside a modified history 𝐻𝑠err = 𝐻𝑠−1 ⊕ 𝑎𝑠err (where 𝐻𝑠−1 is the history prior to step 𝑠, and ⊕ denotes concatenation). The desired behavior for the agent in this context is to output a predefined escape action, 𝑎escape. Back on Track Scenario. This scenario aims to train reflective adjustment, enabling the agent to resume the intended task flow after recovering from an error. We assume the agent has just executed the escape action 𝑎escape from the error state, returning it to the original observation 𝐼𝑠 encountered at step𝑠. The agent is presented with this original observation 𝐼𝑠, but its history reflects the recent detour: 𝐻𝑠recover = 𝐻𝑠−1 ⊕ 𝑎𝑠err ⊕ 𝑎escape. The desired behavior in this "back on track" state is for the agent to perform the originally correct action 𝑎𝑠∗ for step 𝑠, demonstrating its ability to re-evaluate the situation and proceed correctly despite the preceding failure.

The constructed ’Error Escape’ and ’Back on Track’ scenario samples are incorporated into the data used for RL training in Stage 2. When the agent encounters these scenarios as input 𝑥 and generates an output 𝑦, its performance is evaluated using the same comprehensive reward function 𝑅total(𝑦,𝑥). By rewarding successful escape actions in the first scenario type and correct subsequent actions in the second, the RL process specifically reinforces the agent’s adaptive strategies for handling failures. This targeted training solidifies the agent’s transition towards a Deliberative Reasoner, together with the task decomposition ability.

### 4 Experiments

In this section, we detail the experimental setup used to train and evaluate our proposed InfiGUI-R1-3B agent. We describe the implementation details, the benchmarks used for evaluation, and present a comprehensive analysis of the results compared to existing state-of-the-art methods.

### 4.1 Setup

Implementation Details. Our model, InfiGUI-R1-3B, is built upon Qwen2.5-VL-3B-Instruct and trained using the proposed Actor2Reasoner Framework, which consists of two main stages. For the RL reward function 𝑅total = 𝑤𝑓 · 𝑅format + 𝑤𝑎 · 𝑅acc, we set the weights 𝑤𝑓 = 0.1 and 𝑤𝑎 = 0.9. Within the agent trajectory accuracy reward 𝑅acc_agent, the weights are 𝑤𝑡 = 0.2 for type matching and 𝑤𝑝 = 0.8 for exact parameter matching. For bounding box

CAD Development Creative Scientific Office OS Avg. Text Icon Avg. Text Icon Avg. Text Icon Avg. Text Icon Avg. Text Icon Avg. Text Icon Avg. Text Icon Avg.

Model

Proprietary Models GPT-4o [19] 2.0 0.0 1.5 1.3 0.0 0.7 1.0 0.0 0.6 2.1 0.0 1.2 1.1 0.0 0.9 0.0 0.0 0.0 1.3 0.0 0.8 Claude Computer Use [4] 14.5 3.7 11.9 22.0 3.9 12.6 25.9 3.4 16.8 33.9 15.8 25.8 30.1 16.3 26.9 11.0 4.5 8.1 23.4 7.1 17.1

General Open-source Models Qwen2-VL-7B [48] 0.5 0.0 0.4 2.6 0.0 1.3 1.5 0.0 0.9 6.3 0.0 3.5 3.4 1.9 3.0 0.9 0.0 0.5 2.5 0.2 1.6 Qwen2.5-VL-3B [8] - - - - - - - - - - - - - - - - - - - - 23.9 Qwen2.5-VL-7B [8] - - - - - - - - - - - - - - - - - - - - 29.0 Kimi-VL [45] - - - - - - - - - - - - - - - - - - - - 34.5

GUI-specific Models SeeClick [10] 2.5 0.0 1.9 0.6 0.0 0.3 1.0 0.0 0.6 3.5 0.0 2.0 1.1 0.0 0.9 2.8 0.0 1.5 1.8 0.0 1.1 CogAgent-18B [15] 7.1 3.1 6.1 14.9 0.7 8.0 9.6 0.0 5.6 22.2 1.8 13.4 13.0 0.0 10.0 5.6 0.0 3.1 12.0 0.8 7.7 Aria-UI [55] 7.6 1.6 6.1 16.2 0.0 8.4 23.7 2.1 14.7 27.1 6.4 18.1 20.3 1.9 16.1 4.7 0.0 2.6 17.1 2.0 11.3 OS-Atlas-4B [51] 2.0 0.0 1.5 7.1 0.0 3.7 3.0 1.4 2.3 9.0 5.5 7.5 5.1 3.8 4.8 5.6 0.0 3.1 5.0 1.7 3.7 OS-Atlas-7B [51] 12.2 4.7 10.3 33.1 1.4 17.7 28.8 2.8 17.9 37.5 7.3 24.4 33.9 5.7 27.4 27.1 4.5 16.8 28.1 4.0 18.9 ShowUI-2B [29] 2.5 0.0 1.9 16.9 1.4 9.4 9.1 0.0 5.3 13.2 7.3 10.6 15.3 7.5 13.5 10.3 2.2 6.6 10.8 2.6 7.7 UGround-7B [14] 14.2 1.6 11.1 26.6 2.1 14.7 27.3 2.8 17.0 31.9 2.7 19.3 31.6 11.3 27.0 17.8 0.0 9.7 25.0 2.8 16.5 UGround-V1-7B [14] - - 13.5 - - 35.5 - - 27.8 - - 38.8 - - 48.8 - - 26.1 - - 31.1 UI-R1-3B [35] 11.2 6.3 - 22.7 4.1 - 27.3 3.5 - 42.4 11.8 - 32.2 11.3 - 13.1 4.5 - - - 17.8 GUI-R1-3B [52] 26.4 7.8 - 33.8 4.8 - 40.9 5.6 - 61.8 17.3 - 53.6 17.0 - 28.1 5.6 - - - GUI-R1-7B [52] 23.9 6.3 - 49.4 4.8 - 38.9 8.4 - 55.6 11.8 - 58.7 26.4 - 42.1 16.9 - - - UI-TARS-2B [40] 17.8 4.7 14.6 47.4 4.1 26.4 42.9 6.3 27.6 56.9 17.3 39.8 50.3 17.0 42.6 21.5 5.6 14.3 39.6 8.4 27.7 UI-TARS-7B [40] 20.8 9.4 18.0 58.4 12.4 36.1 50.0 9.1 32.8 63.9 31.8 50.0 63.3 20.8 53.5 30.8 16.9 24.5 47.8 16.2 35.7

Ours InfiGUI-R1-3B 33.0 14.1 28.4 51.3 12.4 32.4 44.9 7.0 29.0 58.3 20.0 41.7 65.5 28.3 57.0 43.9 12.4 29.6 49.1 14.1 35.7

#### Table 2: Performance comparison of different agent models across various task categories based on Text, Icon, and Average scores on ScreenSpot-Pro. Results marked in bold represent the best performance, and those underlined indicate the second-best performance.

rewards (𝑅bbox), the IoU threshold is 𝜏IoU = 0.7. When using subgoal similarity as a reward (𝑅subgoal) in cases where the action parameters are incorrect (𝑅param = 0), we use a weight 𝑤𝑠 = 0.2.

Training Data. To ensure both strong GUI capabilities and general multimodal understanding, we train InfiGUI-R1-3B on a diverse dataset mixture: AndroidControl (10k trajectories + 2k reflection-focused trajectories), GUI Grounding data (5k samples aggregated from RicoSCA, Widget Caption, etc.), MathV360K (11k samples for general reasoning), and COCO (4k samples for general visual grounding and understanding).

Training Parameters. All experiments were conducted using 16 NVIDIA H800 GPUs. For the SFT stage (Stage 1), we used a learning rate of 2.0e-6, a global batch size of 32, and a warmup ratio of 0.1. For the RL stage (Stage 2), we used a learning rate of 1.0e-6, a batch size of 256 for training updates, a rollout batch size of 256, and generated 16 rollouts per sample during policy exploration.

### 4.2 Evaluation Benchmarks

To comprehensively evaluate InfiGUI-R1-3B, we utilize several key benchmarks targeting different facets of GUI agent capabilities:

ScreenSpot & ScreenSpot-Pro: These benchmarks assess fundamental GUI understanding and element grounding accuracy across

diverse platforms (Mobile, Desktop, Web). ScreenSpot-Pro specifically increases the difficulty with complex desktop applications and high-resolution screens.

AndroidControl: This benchmark evaluate the agent’s ability to execute complex, multi-step tasks within realistic Android environments. They directly test the higher-level reasoning capabilities crucial for a Deliberative Reasoner, including planning, and state tracking over long interaction trajectories. We report results on the Low-level (Low) and High-level (High) splits of AndroidControl.

### 4.3 Results

We compare InfiGUI-R1-3B against a range of state-of-the-art open-source and proprietary GUI agents. The results demonstrate the effectiveness of our Actor2Reasoner framework in advancing GUI agents towards deliberative reasoning.

Performance on ScreenSpot. Table 1 summarizes the results on the ScreenSpot benchmark, evaluating grounding across Mobile, Desktop, and Web platforms. InfiGUI-R1-3B achieves state-of-theart performance among all compared models, including proprietary ones like Gemini 1.5 Pro and Claude, with an impressive average accuracy of 87.5%. It consistently ranks first across all platforms and both text-based and icon-based grounding tasks (Mobile: 97.1/81.2, Desktop: 94.3/77.1, Web: 91.7/77.6). This outstanding performance

0.9

AndroidControl-Low AndroidControl-High Type Grounding SR Type Grounding SR

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | |Overa|ll| |
| | | | | |Low| | |
| | | | | |High Grou|nding| |
| | | | | | | | |

Model

0.8

Claude* 74.3 0.0 19.4 63.7 0.0 12.5 GPT-4o 74.3 0.0 19.4 66.3 0.0 20.8 Aria-UI – 87.7 67.3 – 43.2 10.2 OS-Atlas-4B 91.9 83.8 80.6 84.7 73.8 67.5 Aguvis-7B – – 80.5 – – 61.5 Aguvis-72B – – 84.4 – – 66.4 UI-R1 94.3 82.6 - - - GUI-R1-3B - - - 58.0 56.2 46.6 GUI-R1-7B - - - 71.6 65.6 51.7 UI-TARS-2B 98.1 87.3 89.3 81.2 78.4 68.9

0.7

0.6

Reward

0.5

0.4

0.3

0.2

0.1

0 50 100 150 200 250 300 350

Ours InfiGUI-R1-3B 96.0 93.2 92.1 82.7 74.4 71.1

Training Step

#### Table 3: Performance comparison of different agent models on AndroidControl benchmarks. SR stands for Success Rate. Results marked in bold represent the best performance, and those underlined indicate the second-best performance.

Figure 3: Reward curves during reinforcement learning training. The plot shows the overall reward and the rewards for individual task types (Low-level, High-level, Grounding) over training steps.

underscores the robustness and generalization ability of InfiGUIR1-3B’s visual understanding and grounding capabilities.

Performance on ScreenSpot-Pro. As shown in Table 2, InfiGUIR1-3B achieves competitive performance on the demanding ScreenSpot-Pro benchmark, which focuses on complex, high-resolution desktop GUI grounding. With an overall average score of 35.7, it performs comparably to the larger UI-TARS-7B model (35.7) and significantly outperforms other baselines like OS-Atlas-7B (18.9) and UGround-7B (16.5). Our model shows particular strength in categories like CAD (28.4 avg), Office (57.0 avg) and OS (29.6 avg), demonstrating robust grounding capabilities even in specialized software environments. While not universally outperforming the top model in every category, the strong overall performance validates the effectiveness of our approach.

Performance on AndroidControl. Table 3 presents the results on the AndroidControl benchmark. InfiGUI-R1-3B achieves a high Success Rate (SR) of 92.1% on AndroidControl-Low and 71.1% on AndroidControl-High. This surpasses the previous state-of-the-art model with similar parameters, UI-TARS-2B (SR: 89.3% / 68.9%). Furthermore, it also outperforms larger GUI-specific models such as Aguvis-72B (SR: 84.4% / 66.4%). This highlights the effectiveness of the training focused on planning capabilities in our Stage 2.

In summary, the experimental results across AndroidControl, ScreenSpot-Pro, and ScreenSpot demonstrate that InfiGUI-R13B significantly advances the capabilities of GUI agents. Our Actor2Reasoner framework, combining Spatial Reasoning Distillation and RL-based Deliberation Enhancement (Sub-goal Guidance, Error Recovery), successfully transforms a base MLLM into a more effective Deliberative Reasoner, achieving state-of-the-art among models with similar parameter counts in trajectory-based tasks and element grounding across different platforms and resolutions, even with a relatively small 3B parameter model.

### 4.4 Visualization

Figure 3 illustrates the reward progression throughout the reinforcement learning training process. It displays both the overall reward accumulated by the agent and the specific rewards obtained for different task categories - Low-level, High-level, and Grounding tasks. As observed from the curves, the rewards for the overall agent performance, as well as for each individual task type, exhibit a consistent upward trend as training progresses. This indicates that the agent effectively learns and improves its performance across all GUI tasks during the RL training phase.

### 5 Conclusion

We present InfiGUI-R1-3B, a multimodal GUI agent that bridges the gap between reactive execution and deliberative reasoning. Through the Actor2Reasoner framework, our approach systematically injects and refines reasoning capabilities in two stages: Spatial Reasoning Distillation to build foundational cross-modal reasoning, and Deliberation Enhancement via reinforcement learning to support sub-goal planning and error recovery. Empirical results across diverse benchmarks demonstrate that InfiGUI-R1-3B not only matches or surpasses larger models in grounding accuracy but also excels in long-horizon task execution with robust planning and reflection.

### References

- [1] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang.

2025. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents. arXiv:2504.00906 [cs.AI] https://arxiv.org/abs/2504.00906

- [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740 (2024).

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. In Advances in Neural

- Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). http://papers.nips.cc/paper_files/paper/2022/hash/ 960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html
- [4] Anthropic. 2024. Developing a computer use model. https://www.anthropic.com/ news/developing-computer-use. Accessed: 2025-04-12.
- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. 2023. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390 (2023).

- [6] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenhang Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, K. Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Yu Bowen, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xing Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical Report. ArXiv (2023). https://doi.org/10.48550/arXiv.2309.16609

- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Frontier Large VisionLanguage Model with Versatile Abilities. ArXiv (2023). https://doi.org/10.48550/ arXiv.2308.12966

- [8] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025).

- [9] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al.

2024. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264 (2024).

- [10] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. 2024. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935 (2024).

- [11] Google DeepMind. 2024. Gemini-2.0 (Project Mariner). https://deepmind.google/ technologies/project-mariner. Accessed: 2025-04-12.
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. https://openreview.net/forum? id=YicbFdNTTy

- [13] Luciano Floridi and Massimo Chiriatti. 2020. GPT-3: Its nature, scope, limits, and consequences. Minds and Machines 30 (2020), 681–694.

- [14] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. 2024. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243 (2024).

- [15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14281–14290.

- [16] Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, Yuhuai Li, Shengze Xu, Shawn Wang, Xinchen Xu, Shuofei Qiao, Kun Kuang, Tieyong Zeng, Liang Wang, Jiwei Li, Yuchen Eleanor Jiang, Wangchunshu Zhou, Guoyin Wang, Keting Yin, Zhou Zhao, Hongxia Yang, Fan Wu, Shengyu Zhang, and Fei Wu. 2024. OS Agents: A Survey on MLLM-Based Agents for General Computing Devices Use. Preprints (December 2024). doi:10.20944/preprints202412.2294.v1

- [17] Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, Yao Cheng, Jianbo Yuan, Jiwei Li, Kun Kuang, Yang Yang, Hongxia Yang, and Fei Wu. 2024. InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. arXiv preprint arXiv:2401.05507 (2024).

- [18] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. 2024. Understanding the planning of LLM agents: A survey. arXiv preprint arXiv:2402.02716 (2024).

- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

- [20] Yue Jiang, Eldon Schoop, Amanda Swearngin, and Jeffrey Nichols. 2023. ILuvUI: Instruction-tuned LangUage-Vision modeling of UIs from Machine Conversations. arXiv preprint arXiv:2310.04869 (2023).

- [21] Marko Jurmu, Sebastian Boring, and Jukka Riekki. 2008. ScreenSpot: Multidimensional resource discovery for distributed applications in smart spaces. In Proceedings of the 5th Annual International Conference on Mobile and Ubiquitous Systems: Computing, Networking, and Services. 1–9.

- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. 2024. Llava-onevision: Easy

- visual task transfer. arXiv preprint arXiv:2408.03326 (2024).
- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.

- [24] Kaixin Li, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, Tat-Seng Chua, et al. 2025. Screenspot-pro: Gui grounding for professional highresolution computer use. In Workshop on Reasoning and Planning for Large Language Models.

- [25] Linyi Li, Shijie Geng, Zhenwen Li, Yibo He, Hao Yu, Ziyue Hua, Guanghan Ning, Siwei Wang, Tao Xie, and Hongxia Yang. 2024. InfiBench: Evaluating the Question-Answering Capabilities of Code Large Language Models. arXiv preprint arXiv:2404.07940 (2024).

- [26] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 2024. On the effects of data scale on computer control agents. arXiv e-prints (2024), arXiv–2406.

- [27] Yang Li, Luheng Li, Gangaand He, Jingjie Zheng, Hong Li, and Zhiwei Guan.

2020. Widget Captioning: Generating Natural Language Description for Mobile User Interface Elements. arXiv preprint arXiv:2010.04295 (2020).

- [28] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. 2024. A Survey of Multimodel Large Language Models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering. 405–409.

- [29] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. 2024. Showui: One vision-languageaction model for generalist gui agent. In NeurIPS 2024 Workshop on Open-World Agents.

- [30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13. Springer, 740– 755.

- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.). http://papers.nips.cc/paper_files/paper/2023/hash/ 6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html

- [32] Haogeng Liu, Quanzeng You, Yiqi Wang, Xiaotian Han, Bohan Zhai, Yongfei Liu, Wentao Chen, Yiren Jian, Yunzhe Tao, Jianbo Yuan, Ran He, and Hongxia Yang.

2024. InfiMM: Advancing Multimodal Understanding with an Open-Sourced Visual Language Model. In Annual Meeting of the Association for Computational Linguistics.

- [33] Yuhang Liu, Pengxiang Li, Zishu Wei, Congkai Xie, Xueyu Hu, Xinchen Xu, Shengyu Zhang, Xiaotian Han, Hongxia Yang, and Fei Wu. 2025. InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection. arXiv preprint arXiv:2501.04575 (2025).

- [34] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. 2022. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11976–11986.

- [35] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. 2025. UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning. arXiv preprint arXiv:2503.21620 (2025).

- [36] OpenAI. 2023. GPT-4V(ision) System Card. https://cdn.openai.com/papers/ GPTV_System_Card.pdf
- [37] OpenAI. 2024. GPT-4o. https://openai.com/index/hello-gpt-4o/ Accessed: 2025-01-03.
- [38] Zhenyu Pan, Haozheng Luo, Manling Li, and Han Liu. 2024. Chain-of-action: Faithful and multimodal question answering through large language models. arXiv preprint arXiv:2403.17359 (2024).

- [39] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824 (2023).

- [40] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. arXiv preprint arXiv:2501.12326

(2025).

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

- [42] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. 2024. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573 (2024).

- [43] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv preprint arXiv: 2409.19256 (2024).

- [44] Yuchen Sun, Shanhui Zhao, Tao Yu, Hao Wen, Samith Va, Mengwei Xu, Yuanchun Li, and Chongyang Zhang. 2025. GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration. arXiv preprint arXiv:2503.17709 (2025).

- [45] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. 2025. Kimi-VL Technical Report. arXiv preprint arXiv:2504.07491 (2025).

- [46] Qwen Team. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning. https://qwenlm.github.io/blog/qwq-32b/
- [47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. ArXiv

(2023). https://doi.org/10.48550/arXiv.2302.13971

- [48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024).

- [49] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. 2023. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079 (2023).

- [50] Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2023. AutoDroid: LLMpowered Task Automation in Android. arXiv preprint arXiv:2308.15272 (2023).

- [51] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2024. Osatlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218 (2024).

- [52] Xiaobo Xia and Run Luo. 2025. GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents. arXiv preprint arXiv:2504.10458 (2025).

- [53] Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun. 2021. Lawformer: A pre-trained language model for chinese legal long documents. AI Open 2 (2021), 79–84.

- [54] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao.

2023. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441 (2023).

- [55] Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. 2024. Aria-UI: Visual Grounding for GUI Instructions. arXiv preprint arXiv:2412.16256 (2024).

- [56] Shenzhi Wang Zhangchi Feng Dongdong Kuang Yuwen Xiong Yaowei Zheng, Junting Lu. 2025. EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework. https://github.com/hiyouga/EasyR1.
- [57] Keen You, Haotian Zhang, Eldon Schoop, Floris Weers, Amanda Swearngin, Jeffrey Nichols, Yinfei Yang, and Zhe Gan. 2025. Ferret-ui: Grounded mobile ui understanding with multimodal llms. In European Conference on Computer Vision. Springer, 240–255.

- [58] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771 (2023).

