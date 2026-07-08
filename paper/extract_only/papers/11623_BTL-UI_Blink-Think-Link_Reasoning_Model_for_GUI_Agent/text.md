# arXiv:2509.15566v4[cs.CV]27Oct2025

## BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent

Shaojie Zhang∗ Ruoceng Zhang∗ Pei Fu∗ Shaokang Wang Jiahui Yang Xin Du Shiqi Cui Bin Qin Ying Huang Zhenbo Luo† Jian Luan

MiLM Plus, Xiaomi Inc {zhangshaojie5, zhangruoceng1, fupei1, luozhenbo, luanjian}@xiaomi.com

### Abstract

In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To address this gap, we propose Blink–Think–Link (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward – the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates competitive performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework’s efficacy in developing advanced GUI agents.

### 1 Introduction

Automation of graphical user interface (GUI) interactions constitutes a pivotal milestone in developing genuinely intelligent digital assistants [1, 2, 3]. Recent breakthroughs in large vision-language models (VLMs) [4, 5] and reinforcement learning fine-tuning techniques have substantially improved agents’ capabilities in natural language command interpretation, visual element perception, and multi-step task execution through human-like reasoning [6, 7].

However, current mainstream systems adopt mainly two approaches. The first relies on supervised fine-tuning (SFT) to align model behavior with task objectives, but this method faces two major limitations: a strong dependence on large-scale expert-labeled data and limited generalization capability when faced with out-of-distribution scenarios. The second approach involves rule-based reinforcement fine-tuning (RFT) [8], as shown in Figure 1 (a), which enhances generalization in complex tasks by using a structured reasoning format: intermediate cognitive steps are encapsulated within <think> tags, and final decisions are expressed through <answer> tags. Although effective in improving task performance, these methods [9, 10, 11] exhibit two critical shortcomings: (1)

∗ Equal contribution; † Corresponding author. § https://github.com/xiaomi-research/btl-ui

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

[Figure 1]

|I want to change the Vimeo video quality option to high in order to watch videos in higher quality.<br><br>## User Instruction<br><br>## Action History<br><br>Step 1: open the Vimeo app.<br><br>## System Prompt<br><br>A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>|
|---|

|## System Prompt<br><br>You are a GUI Agent capable of reasoning based on user instructions, action history, and the current screenshot. You should first observe the layout of the screenshot and extract $N$ elements RELATED TO the user instruction, where 0 <= $N$ <=4. Next, think about the reasoning process BASED ON the observations and instructions in your mind, and then provide the user with the answer.The observation process (can be None if $N$ == 0), reasoning process and answer are enclosed within <blink></blink>, <think></think> and <link></link> tags, respectively,<br><br>## User Instruction<br><br>## Action History<br><br>I want to change the Vimeo video quality option to high in order to watch videos in higher quality.<br><br>Step 1: open the Vimeo app.|
|---|

[Figure 2]

- 1

- 2

[Figure 3]

[Figure 4]

|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br><think> The user wants to change the Vimeo video quality to high. This typically involves accessing the settings or options menu, which is often represented by a three-dot icon or similar symbol. The three-dot icon at the top right corner is likely the place to find these settings. </think><br><br><blink><br><br><element><id>1</id><bbox>[1019,170,1068,248]</bbox><caption>dynamic</caption></element><br><element><id>2</id><bbox>[937,768,1052,871]</bbox><caption>dynamic</caption></element> </blink><br><br><br><link> [{"Plan": "Access the settings menu to change the video quality.", "Action": {"function": "Tap", "position": [1042, 207]}}]<br><br></link><br><br>Perception process:<br><br>Reasoning process:<br><br>Determine the action<br><br>|
|---|

|[Figure 9]<br><br>[Figure 10]<br><br><think><br><br>I need to locate the UI element for changing the video quality to high.Typically, such options are found in the video settings or quality dropdown menu, which might be represented by a gear icon or a similar symbol. </think><br><br><answer><br><br>[{"Action": {"function": "Tap"，“position”: [1044, 628]}}] </answer><br><br>Reasoning process<br><br>Determine the action<br><br>[Figure 11]|
|---|

(a) Think-Answer

(b) Blink-Think-Link

- Figure 1: Framework comparison of previous Think-Answer and Blink-Think-Link in GUI tasks for RFT. Specifically, colorful text is supervised by rule-based reinforcement learning. And different colors of text indicate different reward rules. The previous “Think-Answer” framework is optimized by format reward, action type reward, and corresponding args reward. And our Blink-Think-Link framework is optimized by dual format reward, blink reward, and link reward.

significant deviation from natural human-GUI interaction patterns, and (2) excessive focus on interaction outcomes while lacking effective process-oriented reward mechanisms.

Cognitive studies [12, 13, 14] demonstrate that human-GUI interaction achieves remarkable efficiency through three sequential processes: (a) Blink Phase. Rapid target location during saccadic intervals; (b) Think Phase. Multimodal information integration with intentional reasoning; (c) Link Phase. Generation of precise motor execution commands. Building upon this cognitive finding, we innovatively propose a biologically inspired interaction paradigm—the Blink-Think-Link (BTL) paradigm—for GUI agents, and computationally simulate this paradigm through a structured output mechanism (as shown in Figure 1 (b)):

- • <blink>: Where relevant areas of the screen are rapidly located, analogous to saccadic eye movements. The visual attention-related region-of-interest information is encapsulated within <blink></blink> tags.
- • <think>: Where the system engages in high-level reasoning and decision-making, mirroring cognitive task planning. The reasoning processes are recorded in <think></think> tags.
- • <link>: Where actionable commands are generated for precise execution, reflecting human action selection mechanisms. The action commands are output in <link></link> tags.

Specifically, to model human visual localization capabilities during blink intervals, we developed an innovative blink data generation pipeline to automatically produce several region-of-interest (ROI) annotations for training samples. Furthermore, to address the limitations of current reward models in rule-based RFT algorithms that over-rely on outcome-based rewards while neglecting guidance for intermediate interaction processes, We propose the innovative BTL Reward, a Process-Outcome Integrated Reward Mechanism, which comprises three core components: (1) the Dual Format Reward for template and content matching, (2) the Blink Reward for fine-grained guidance of interaction processes, and (3) the Link Reward for action outcome evaluation. By combining the Blink Reward’s granular process supervision with the Link Reward’s precise outcome feedback, this mechanism pioneers the organic integration of process-oriented and outcome-driven approaches. Compared to conventional reward schemes focusing solely on final outcomes, the BTL reward mechanism delivers more sophisticated and multi-dimensional training guidance. Finally, building upon this framework, we develop BTL-UI, a GUI Agent that demonstrates the framework’s effectiveness across multiple GUI tasks.

In general, the main contributions are summarized as follows:

1. We propose BTL (Blink-Think-Link), an innovative framework that simulates the human cognitive process in the human-GUI interaction by explicitly modeling how users perceive, process, and act upon interface elements.

GUI Input

1 Completions Generation

[Figure 12]

Completion 1

[Figure 13]

[Figure 14]

Policy Model

<blink>

[Figure 15]

System prompt + User instruction

[Figure 16]

- <element><id>1</id><bbox>[28,157,139,240]</bbox><caption>dynamic</caption></element>
- <element><id>2</id><bbox>[429,138,667,240]</bbox><caption>dynamic</caption></element>
- <element><id>3</id><bbox>[57,1430,530,2116]</bbox><caption>dynamic</caption></element>
- <element><id>4</id><bbox>[560,1430,1036,1900]</bbox><caption>dynamic</caption></element>
- <element><id>5</id><bbox>[537,1980,1036,2137]</bbox><caption>dynamic</caption></element> </blink> <think> The goal is to find information about Wassily Kandinsky in the Art & Culture app. Since the current screen shows various art pieces but not specifically related to Wassily Kandinsky, I need to use the search function to find him. The magnifying glass icon at the top left corner likely represents the search feature. Tapping this will allow me to input the name 'Wassily Kandinsky' and find relevant information. </think> <link>

[Figure 17]

Prompt

3 Update

[Figure 18]

......

[{"Plan": "Tap the search icon to initiate a search for Wassily Kandinsky.", "Action": {"function": "Tap", "position": [79, 200]}}] </link>

Policy Optimization

Completion N

[Figure 19]

<blink> element><id>1</id><bbox>[37,160,138,239]</bbox><caption>dynamic</caption></element>

[Figure 20]

- <element><id>2</id><bbox>[327,304,612,401]</bbox><caption>dynamic</caption></element>
- <element><id>3</id><bbox>[624,304,830,401]</bbox><caption>dynamic</caption></element>
- <element><id>4</id><bbox>[844,304,1075,401]</bbox><caption>dynamic</caption></element>
- <element><id>5</id><bbox>[34,1146,296,1250]</bbox><caption>dynamic</caption></element> </blink> <think> he user wants to find information about Wassily Kandinsky. To do this, I need to use the search functionality or navigate through categories that might lead to his works. The search icon is visible, which suggests it\'s a good place to start. However, since the goal is to find his name directly, I will focus on the search bar initially. </think> <link> [{"Plan": "Click on the search icon to initiate a search for Wassily Kandinsky.", "Action": {"function": "Tap", "position": [80, 199]}}] </link>

KL Calculation

Advantage Calculation

###### 2 Rule-Based Reward Calculation

[Figure 21]

[Figure 22]

Dual Format Reward Blink Reward Link Reward

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Reference Model

[Figure 27]

IOU Deduplication Complete Action Analysis Hungarian Matching

- 1. Template Format Check
- 2. Content Format Check

[Figure 28]

[Figure 29]

[Figure 30]

Screenshot

- Figure 2: Overall framework of BTL. We adopt Group Relative Policy Optimization (GRPO) to optimize the proposed BTL. Firstly, the base model generates N completions for a given GUI task sample. Furthermore, GRPO computes the relative advantages within a group of completions, eliminating the need for manually annotated data. Finally, the policy model updates parameters under the guidance of relative advantages and the KL divergence constraint.

- 2. We propose two key innovations to jointly advance the learning of GUI agents within this framework: (1) Blink Data Generation—an efficient data annotation pipeline that automatically generates multi-region Regions of Interest (ROIs) for training samples; (2) BTL Reward—the first rule-based Process-Outcome Integrated Reward Mechanism.
- 3. We develop BTL-UI, a GUI agent trained via the BTL framework, and extensive experiment results demonstrate that the model achieves competitive performance across multiple GUI benchmarks.

### 2 Related Work

#### 2.1 GUI Agents

Autonomous agents powered by large language models (LLMs) and VLMs have recently garnered considerable scholarly interest due to their interactive functionalities. For GUI tasks, earlier systems relied on LLMs to read and interpret structured representations such as HTML and accessibility trees. However, since icons, images, diagrams, and their spatial relationships are difficult to express in such structured languages, agents based on LLM often perform poorly [6, 15, 16]. Therefore, VLMbased agents have been introduced to perceive visual GUI signals directly with better performance [2, 17, 18, 19, 20, 21]. For example, UGround [22] developed a specialized GUI grounding model for GUI element localization. OS-Atlas [23] proposed a foundational model for GUI agents by interpreting human intentions and predicting actions in the form of function calls. Aguvis [24] integrated explicit planning and reasoning within the model, enhancing its ability to navigate and interact with complex digital environments autonomously. UI-TARS [9] combines GUI-related pretraining with task-level reasoning fine-tuning to better capture the complexity of GUI interactions. Although research on VLM-based GUI agents has made impressive progress, they mainly follow the SFT training paradigm, which directly mimics the ground-truth actions provided in the curated data.

#### 2.2 Reinforcement Fine-Tuning

With the advent of rule-based reinforcement learning approaches such as OpenAI-o1 [25] and DeepSeek-R1 [8], recent studies have demonstrated that RFT improves the reasoning abilities of the model and provides greater generalizability [26]. Subsequent approaches [26, 27, 28] have introduced this paradigm to VLMs. For example, Vision-R1 [27] combined a vision criterion-driven reward

function and a progressive rule refinement strategy to enhance VLM’s object localization capabilities. Visual-RFT [26] adopted the reinforcement learning strategy to enhance visual perception and grounding ability of VLMs. VLM-R1 [29] demonstrated that RFT with small amounts of high-quality data can enable VLMs to solve complex vision-language tasks.

For GUI tasks, UI-R1 [30] and GUI-R1 [10] introduced rule-based reinforcement learning frameworks that require minimal expert supervision, demonstrating competitive performance. InfiGUI-R1 [11] further advanced the field by bridging reactive execution and deliberative reasoning through the Actor2Reasoner architecture. However, existing RFT-based GUI agents predominantly adopt rulebased reinforcement learning, which focus on final outcomes and lack intermediate process guidance, often overlooking key aspects of human cognition and interaction.

### 3 Method

In this section, we introduce BTL, a new framework grounded in cognitive science theory, with its core concept derived from the Blink-Think-Link paradigm observed in human-GUI interactions. The framework is shown in Figure 2. We detail the implementation details of this framework through the following components: Preliminaries, Blink Data Generation, BTL Reward, and Policy Optimization.

#### 3.1 Preliminaries

The interaction between a GUI agent and its environment can be naturally formulated as a Markov Decision Process (MDP), defined by the tuple ⟨S,A,Z,T ,O⟩. Here, S denotes the state space representing possible screen states; A is the action space that encompasses interaction types such as clicking, typing, and scrolling; Z is the observation space, including screenshots or structured UI representations; T : S × A × S → [0,1] defines the probability of transitioning from one state to another given an action; and O : S × A → Z specifies the probability of receiving a particular observation given a state and an action.

During task execution, at each discrete time step t, the agent receives an input tuple (zt,u,h), where zt ∈ Z is the current state of the screen, u refers to the global task instruction and h is its interaction history. The BTL process can then be formalized as a structured policy function F:

F({zt,u,h}) → ot = {bt,dt,at}, (1)

where ot denotes the BTL output at time t, consisting of: bt—visual attention regions, dt—reasoning and decision trace, at—the final action to be executed. Each action at = (αt,δt) ∈ A is composed of an action type αt (e.g., click) and its corresponding parameters δt (e.g., coordinates, text input). Upon execution of at, the environment transitions to a new state zt+1, and the process repeats until the task is completed or a terminal condition is met.

#### 3.2 Blink Data Generation

One of the core innovations of the BTL framework is its ability to simulate the human mechanism of rapidly locating ROIs in a visual scene during the blink phase. To achieve this, we propose an automated Blink data generation pipeline that annotates ROIs on the screenshot corresponding to the user instruction in the MDP. As illustrated in Figure 3, the pipeline consists of two main stages. A parsing model [31] first processes the raw screenshot to extract semantic UI elements. Then, an analysis model [5] is used to evaluate the visual importance and contextual relevance of these elements, allowing filtering and prioritization to produce the final ROI annotations.

Specifically, in the first stage, we extract individual UI elements such as buttons, icons, and text fields, annotating each with bounding box coordinates, type, and semantic captions. These annotations form a structured representation of the state of the screen, enabling bottom-up human-like interpretation of the GUI. The output of this stage is a comprehensive list of elements, denoted E = {e1,e2,...,en}, where each ek = {idk,bboxk,typek,captionk,interactivityk} represents the attributes related to the element. This foundational representation serves as the input for subsequent filtering and prioritization steps that model instruction-directed visual attention.

In the second stage, we employ Qwen2.5-VL-32B [5] to simulate top-down attention by filtering and ranking elements based on visual saliency and task relevance. Oriented by task instruction u and

Screenshot Input Blink Output

Element Parser Output Filter Prompt

[Figure 31]

[Figure 32]

|<blink><br><br><element> <id>10</id> <bbox>[0.02,0.07,0.27,0.18]</bbox> <caption>dynamic</caption><br><br></element> <element><br><br><id>7</id> <bbox>[0.53,0.07,0.70,0.18]</bbox> <caption>dynamic</caption><br><br></element> <element><br><br><id>9</id> <bbox>[0.77,0.78,0.93,0.85]</bbox> <caption>dynamic</caption><br><br></element> <element><br><br><id>8</id> <bbox>[0.76,0.07,0.93,0.18]</bbox> <caption>dynamic</caption><br><br></element> <element><br><br><id>6</id> <bbox>[0.53,0.78,0.70,0.83]</bbox> <caption>dynamic</caption><br><br></element> </blink>|
|---|

[Figure 33]

|... <element><br><br><id>1</id> <type>icon</type > <bbox>[0.53,0.34,0.70,0.44]</bbox> <interactivity>true</interactivity> <content>Opera</content ><br><br></element><br><br>... <element><br><br><id>10</id> <type>icon</type > <bbox>[0.02,0.07,0.27,0.18]</bbox> <interactivity>true</interactivity> <content>Maps & Navi....</content ><br><br></element><br><br>... <element><br><br><id>21</id> <bbox>[0.70,0.90,0.79,0.95]</bbox> <interactivity>true</interactivity> <content>Dictate</content ><br><br></element><br><br>...|
|---|

System prompt + User instruction + Elements list

|10|
|---|

|7|
|---|

|8|
|---|

[Figure 34]

Screenshot

|6|
|---|

|9|
|---|

<think>I need to filter the n UI elements that are most relevant to the task instruction and the current step...</think>.

[Figure 35]

Filtered Elements List Element Parser

Parsed Elements List

Screenshot with Label

Filtered Elements on Screenshot

- Figure 3: Two-stage data construction pipeline. In the first stage, the basic properties of UI elements are obtained by a parsing model. To eliminate the redundancy of the number and attributes of elements, the analysis model in the second stage simplifies the list to λ elements with their positions (<bbox>), while the reserved <caption> attribute indicates whether the element is interactive. In the example shown in the figure, the instruction for the current step is “Use the GPS to locate a nearby museum and then book a ride with Lyft.” Accordingly, the most relevant element in the Blink output is the “Maps & Navigation” app with <ID>10</ID>.

interaction history h, the model dynamically selects a subset of elements from the filtered list E. This selection process can be formulated as:

EROI = f(E,u,h), (2)

where EROI ⊆ E denotes the resulting set of λ candidate elements that are most relevant to the current task step during blink phases, encapsulated within <blink></blink> tags. The choice of the number of elements after filtering, λ, achieves a trade-off between BTL performance and efficiency. This attention-guided annotation not only mimics human visual focus during blink phases but also provides a high-quality reference for optimizing the agent policy.

#### 3.3 BTL Reward

Effective GUI agents must excel at both interface grounding and long-horizon planning. To this end, we design a three-component rule-based reward scheme, denoted as RBTL that mirrors the human Blink–Think–Link cognitive cycle:

RBTL = Rformat + Rblink + Rlink. (3) Each term provides targeted supervision at a different phase of interaction, as detailed below.

Dual Format Reward. Following previous work [8, 26] that leverages format rewards to encourage predefined templates for easy answer extraction, we introduce a dual format reward to evaluate whether the generated output adheres to both the expected structural template and content. Specifically, the template check function ftemplate is used to check whether the generated completions meet the Blink-Think-Link three-stage grammatical structure. Furthermore, the content check function fcontent is adopted to evaluate whether the blink content complies with the XML format and the link content complies with the JSON format, which facilitates the parsing of trajectory planning and actions with corresponding arguments. We adopt a binary reward scheme, assigning a reward of 1 only when the prediction oi fully satisfies both format and content criteria as follows:

Rformat(oi) =

1, if ftemplate = 1 ∧ fcontent = 1

0. otherwise

(4)

Blink Reward. This component incentivizes the rapid and accurate localization of the interface elements relevant to the instruction u. From the agent’s prediction oi, we extract a set of ROIs Pi = {pxi } and compare them to ground-truth annotations Gi = {gix} (see §3.2). We adopt the Hungarian matcher [32] M(·,·,τ), a classical assignment algorithm used to compute the optimal

one-to-one matching between predicted and ground-truth bounding boxes based on IoU scores, under a given threshold τ.

indexmatch = y | giy ∈ Gi, ∃pxi ∈ Pi s.t. M(pxi ,giy,τ) = 1 . (5)

It is worth noting that in the planning task, the elements related to the instruction u may not be explicitly present in the current screenshot. And the corresponding operation should be to other pages through scrolling or going back. Thus, in the predicted results, Pi = ∅ is allowed. Consequently, the blink reward can be defined as follows:

 

1, if Pi = ∅ ∧ Gi = ∅ ∨ ai ∈ A∗ max s(y) | y ∈ indexmatch , elif Pi ̸= ∅ ∧ Gi ̸= ∅

(6)

Rblink(oi) =



0. otherwise

where A∗ denotes the non-interactive action spaces and s(·) refers to the reward allocation function, which is determined based on the priorities of elements in the annotations.

Link Reward. The link phase assesses the agent’s ability to generate a fully coherent executable command. Recent RFT-based GUI agents [10, 11, 30] always split the reward of the predicted action into a reward for the action type and a reward for the action args (e.g. click coordinates or input text). However, this kind of reward will split an action into two independent contents, which is not in line with human cognition. At the same time, this staged action reward will cause reward hacking, which prevents the agent from understanding the designed action space. Thus, we employ a strict binary criterion: the agent receives a reward only if both the action type and its associated arguments are exactly correct. Formally, the link reward is defined as:

Rlink(oi) =

1, if ftype = 1 ∧ fargs = 1

0. otherwise

(7)

This all-or-nothing scheme ensures that the final command is internally consistent and accurately reflects the intended GUI operation.

#### 3.4 Advantage Computation and Parameter Update

As shown in Figure 2, we adopt Group Relative Policy Optimization (GRPO) to optimize the proposed BTL. Since its supervision is based solely on the final outcome, GRPO is particularly suited for tasks with explicit, objective answers. Furthermore, GRPO significantly reduces memory overhead for VLMs by removing the reward models or value models in other performance optimization methods [33].

Given a base model to be optimized, GRPO starts by initializing a trainable policy model πθ and a frozen reference model πref. For a given GUI task sample {zi,u,h}, the policy model πθ first generates a group of completions {o1,o2,...,oN}. Then, the reward function computes the whole group’s rewards {R1,R2,...,RN}, which are further used to calculate the advantage Ai of each completion within the group by:

Ri − mean({Rj}Nj=1) std({Rj}Nj=1)

. (8)

Ai =

After the reference model computes the logits to output each completion given the task, the policy model πθ is optimized by maximizing the following objective:

N

1 N

JGRPO(θ) =

i=1

πθ(oi|zi,u,h) πθ

Ai − β · KL(πθ(oi|zi,u,h)|πref(oi|zi,u,h)) , (9)

(oi|zi,u,h)

old

where N is the number of completions in one group and β is the hyperparameter to control the KL divergence constraints. This objective motivates the model to tend to produce the completion with a higher advantage within a group, but not to stray too far from the initial model.

### 4 Experiment

#### 4.1 Implementation Details

Experimental Setup. We develop the BTL-UI-3B/7B model based on Qwen2.5-VL-3B/7B and adopt the ms-swift framework [34] for RL training. As shown in Table 1, we train BTL-UI in a mix of grounding and planning data.

Evaluation. To conduct a thorough evaluation of BTL-UI, we employ a range of critical benchmarks that focus on specific aspects of the GUI agent’s grounding and planning capabilities:

Table 1: RFT data for BTL-UI.

Category Source Size Grounding

ShowUI-Web [35] 1K ShowUI-Desktop [35] 1K

Grounding: Screenspot series benchmarks assess fundamental GUI understanding and element grounding accuracy across diverse platforms (Mobile, Desktop, Web). ScreenSpot [7] evaluates the single-step GUI grounding performance across multiple platforms. ScreenSpot-V2 [7], a re-annotated version, addresses annotation errors present in the original ScreenSpot. ScreenSpot-Pro [38] specifically increases the difficulty with complex desktop applications and high-resolution screens.

AndroidControl [36] 500 GUI-Odyssey [37] 500

Low-Level

AndroidControl [36] 500 GUI-Odyssey [37] 500

High-Level

- Table 2: GUI grounding accuracy on ScreenSpot [7]. Bold means the best results, and underline means the second best results. Avg. denotes the average performance on mobile, desktop and web subtasks.

Mobile Desktop Web

Model Method Model Size

Avg. Text Icon Text Icon Text Icon

GPT-4o [39] ZS - 30.5 23.2 20.6 19.4 11.1 7.8 18.8 Qwen2-Vl [4] ZS 7B 75.5 60.7 76.3 54.3 35.2 25.7 55.3 OS-Atlas-Base [23] ZS 7B 93.0 72.9 91.8 62.9 90.9 74.3 82.5 Qwen2.5-VL [5] ZS 3B 90.5 61.1 60.0 43.2 80.9 40.0 65.0 Qwen2.5-VL [5] ZS 7B 86.3 83.8 85.6 67.1 87.4 78.6 84.8 InternVL3 [40] ZS 8B - - - - - - 79.5

CogAgent [6] SFT 18B 67.0 24.0 74.2 20.0 70.4 28.6 47.4 Aria-UI [41] SFT 3.9B 92.3 73.8 93.3 64.3 86.5 76.2 82.4 SeeClick [7] SFT 9.6B 78.0 52.0 72.2 30.0 55.7 32.5 53.4 ShowUI [35] SFT 2B 92.3 75.5 76.3 61.1 81.7 63.6 75.1 Aguvis [24] SFT 7B 95.6 77.7 93.8 67.1 88.3 75.2 84.4 UGround [22] SFT 7B 82.8 60.3 82.5 63.6 80.4 70.4 73.3 UGround-V1 [22] SFT 2B 89.4 72.0 88.7 65.7 81.3 68.9 77.7 UGround-V1 [22] SFT 7B 94.1 79.9 93.8 76.4 90.9 84.0 86.3 UI-TARS [9] SFT 2B 93.0 75.5 90.7 68.6 84.3 74.8 82.3 UI-TARS [9] SFT 7B 94.5 85.2 95.9 85.7 90.0 83.5 89.5

UI-R1 [30] RFT 3B - - 90.2 59.3 85.2 73.3 GUI-R1 [10] RFT 3B - - 93.8 64.8 89.6 72.1 GUI-R1 [10] RFT 7B - - 91.8 73.6 91.3 75.7 -

BTL-UI RFT 3B 96.3 77.3 88.2 57.9 80.0 68.9 80.0 BTL-UI RFT 7B 97.1 83.8 90.2 70.7 88.7 84.5 87.2

Planning: AndroidControl [36] and GUI-Odyssey [37] evaluate the agent’s grounding and planning ability to execute multi-step tasks within realistic Android environments. These benchmarks provide agents with a task instruction, a current screenshot, and previous interaction history, aimed at enabling accurate prediction of the next action. Furthermore, according to the input, the settings on AndroidControl can be divided into low-level tasks and high-level tasks. High-level tasks only input the global instruction to the agent, while low-level tasks will additionally input the single-step action plan. And GUI-Odyssey only adopts the high-level experimental setups.

Evaluation Metrics. For grounding tasks, we use click point prediction accuracy as our evaluation metric. For planning tasks, according to OS-Atlas [23], we report three standard metrics for GUI agents: action type prediction accuracy (Type), click point prediction accuracy (GR) and step success rate (SR). Specifically: Type measures the exact-match accuracy between predicted and ground-truth

- Table 3: GUI grounding accuracy on ScreenSpot-V2 [7]. Bold means the best results, and underline means the second best results. Avg. denotes the average performance on mobile, desktop and web subtasks.

Mobile Desktop Web

Model Method Model Size

Avg. Text Icon Text Icon Text Icon

GPT-4o [39] ZS - 30.5 23.2 20.6 19.4 11.1 7.8 18.8 OS-Atlas-Base [23] ZS 4B 85.7 58.5 72.2 45.7 82.6 63.1 70.1 OS-Atlas-Base [23] ZS 7B 93.0 72.9 91.8 62.9 90.9 74.3 82.5 Qwen2.5-VL [5] ZS 3B 92.1 66.8 72.6 46.8 83.0 44.3 70.4 Qwen2.5-VL [5] ZS 7B 97.9 86.7 87.6 68.6 91.5 79.3 87.1 InternVL3 [40] ZS 8B - - - - - - 81.4

SeeClick [7] SFT 9.6B 78.4 50.7 70.1 29.3 55.2 32.5 55.1 Aguvis [24] SFT 7B 95.6 77.7 93.8 67.1 88.3 75.2 84.4 UI-TARS [9] SFT 2B 95.2 79.1 90.7 68.6 87.2 78.3 84.7 UI-TARS [9] SFT 7B 96.9 89.1 95.4 85.0 93.6 85.2 91.6

BTL-UI RFT 3B 97.9 83.4 88.7 62.1 83.3 69.0 82.9 BTL-UI RFT 7B 98.6 89.6 92.3 70.7 92.3 80.3 89.1

action types (e.g., “click” vs. “swipe”). GR evaluates grounding performance via click point prediction accuracy in specific action types (e.g. “click” and “long press”). SR is the step-wise success rate: a step is counted as successful only if both the predicted action and its associated arguments (e.g., click coordinates or input text) match the ground truth.

#### 4.2 Experimental Results

We evaluate BTL-UI across three key capabilities: grounding, low-level planning, and high-level reasoning. The results demonstrate consistent and significant improvements over existing baselines, validating the effectiveness of the Blink-Think-Link framework.

Grounding Capability. To assess how well the model can localize UI elements, we report grounding accuracy on the ScreenSpot benchmark series in Table 2, 3, and 4. In the original ScreenSpot dataset, BTL-UI-7B achieves an average accuracy of 87.2%, outperforming the baseline Qwen2.5-VL-7B (84.8%) and surpassing the supervised fine-tuned Aria-UI (82.4%). On the corrected ScreenSpotV2, performance further improves to 89.1%. In the ScreenSpot-Pro benchmark, the BTL-UI-3B consistently outperforms other RFT-based models, achieving an average accuracy of 27.1%, which is substantially higher than UI-R1 (17.8%) on the same scale. This suggests that the Blink Phase, which encourages early-stage attention to semantically relevant regions through ROI supervision, enables more precise perception and grounding even under diverse visual layouts. Although the overall grounding performance of BTL-UI remains slightly lower than that of UI-TARS [9], which is a strong GUI Agent developed based on Qwen2-VL [4] with training on 50B tokens, the proposed BTL-UI shows certain advantages in the mobile subtasks.

Planning Capability. As shown in Table 5, BTL-UI exhibits strong generalization and reasoning ability on both low-level and high-level GUI planning benchmarks. In AndroidControl-Low, BTLUI-7B achieves an SR of 88.0%, surpassing the previous best model OS-Atlas-Pro-7B (85.2%) and GUI-R1-7B (66.5%) , while the 3B variant attains a comparable 84.8%, confirming the efficiency of the BTL reinforcement paradigm. For long-horizon tasks in AndroidControl-High, which require multi-step reasoning and contextual grounding, BTL-UI-7B achieves an SR of 69.2%, outperforming GUI-R1-7B (51.7%) and approaching the SFT-based GUI foundation model, OS-Atlas-Pro-7B (71.2%). This improvement reflects the synergy between Blink-phase attention and Link-phase symbolic reward, which jointly stabilize execution and reduce accumulated errors in extended interaction sequences. In GUI-Odyssey, a benchmark that emphasizes hierarchical decision-making and interface switching, BTL-UI-7B reaches an SR of 65.4%, significantly surpassing GUI-R1-7B (38.8%). Although there is still a performance gap compared to UI-TARS, the proposed BTL-UI is comparable to large-scale SFT models such as OS-Atlas, while requiring significantly less training data.

- Table 4: GUI grounding accuracy on Screenspot-Pro [38]. Bold means the best results, and underline means the second best results. Avg. denotes the average performance on all subtasks.

Model Method Model Size

Dev Creative CAD Scientific Office OS

Avg. Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon

GPT-4o [39] ZS - 1.3 0.0 1.0 0.0 2.0 0.0 2.1 0.0 1.1 0.0 0.0 0.0 0.8 Qwen2-VL [4] ZS 7B 0.5 0.0 2.6 0.0 1.5 0.0 6.3 0.0 3.4 1.9 0.9 0.0 1.6 Qwen2.5-VL [5] ZS 3B - - - - - - - - - - - - 23.9 Qwen2.5-VL [5] ZS 7B - - - - - - - - - - - - 29.0 OS-Atlas-Base [23] ZS 7B 33.1 1.4 28.8 2.8 12.2 4.7 37.5 7.3 33.9 5.7 27.1 4.5 18.9

ShowUI [35] SFT 2B 16.9 1.4 9.1 0.0 2.5 0.0 13.2 7.3 15.3 7.5 10.3 2.2 7.7 UGround [22] SFT 7B 26.6 2.1 27.3 2.8 14.2 1.6 31.9 2.7 31.6 11.3 17.8 0.0 16.5 UGround-V1 [22] SFT 7B - - - - - - - - - - - - 31.1 SeeClick [7] SFT 9.6B 0.6 0.0 1.0 0.0 2.5 0.0 3.5 0.0 1.1 0.0 2.8 0.0 1.1 CogAgent [6] SFT 18B 14.9 0.7 9.6 0.0 7.1 3.1 22.2 1.8 13.0 0.0 5.6 0.0 7.7 UI-TARS [9] SFT 2B 47.4 4.1 42.9 6.3 17.8 4.7 56.9 17.3 50.3 17.0 21.5 5.6 27.7 UI-TARS [9] SFT 7B 58.4 12.4 50.0 9.1 20.8 9.4 63.9 31.8 63.3 20.8 30.8 16.9 35.7

UI-R1 [30] RFT 3B 11.2 6.3 22.7 4.1 27.3 3.5 42.4 11.8 32.2 11.3 13.1 4.5 17.8 GUI-R1 [10] RFT 3B 26.4 7.8 33.8 4.8 40.9 5.6 61.8 17.3 53.6 17.0 28.1 5.6 GUI-R1 [10] RFT 7B 23.9 6.3 49.4 4.8 38.9 8.4 55.6 11.8 58.7 26.4 42.1 16.9 -

BTL-UI RFT 3B 47.4 4.8 29.8 11.9 28.9 7.8 44.4 14.5 48.6 11.3 32.7 4.4 27.1 BTL-UI RFT 7B 53.9 7.3 26.7 15.9 35.9 14.6 47.2 13.0 62.7 24.7 55.7 19.7 33.7

- Table 5: GUI planning accuracy on AndroidControl [36] and GUI-Odyssey [37]. Bold means the best results, and underline means the second best results.

Model Method Model Size

AndroidControl-Low AndroidControl-High GUI-Odyssey

Type GR SR Type GR SR Type GR SR

GPT-4o [39] ZS - 74.3 38.7 28.4 63.1 30.9 21.2 37.5 14.2 5.4 OS-Atlas-Base [23] ZS 7B 73.0 73.4 50.9 57.4 54.9 29.8 60.4 39.7 27.0 Qwen2.5-VL [5] ZS 3B 62.0 74.1 59.3 47.8 46.5 38.9 37.4 26.5 26.7 Qwen2.5-VL [5] ZS 7B 83.4 87.1 62.5 68.7 59.7 47.1 55.6 37.8 34.4

SeeClick [7] SFT 9.6B 93.0 73.4 75.0 82.9 62.9 59.1 71.0 52.4 53.9 Aria-UI [41] SFT 3.9B – 87.7 67.3 – 43.2 10.2 – 86.8 36.5 Aguvis [24] SFT 7B – – 80.5 – – 61.5 – – OS-Atlas-Pro [23] SFT 4B 91.9 83.8 80.6 84.7 73.8 67.5 83.5 61.4 56.4 OS-Atlas-Pro [23] SFT 7B 93.6 88.0 85.2 85.2 78.5 71.2 84.5 67.8 62.0 UI-TARS [9] SFT 2B 98.1 87.3 89.3 81.2 78.4 68.9 93.9 86.8 83.4 UI-TARS [9] SFT 7B 98.0 89.3 90.8 83.7 80.5 72.5 94.6 90.1 87.0

UI-R1 [30] RFT 3B 79.2 82.4 66.4 57.9 55.7 45.4 52.2 34.5 32.5 GUI-R1 [10] RFT 3B 83.7 81.6 64.4 58.0 56.2 46.6 54.8 41.5 41.3 GUI-R1 [10] RFT 7B 85.2 84.0 66.5 71.6 65.6 51.7 65.5 43.6 38.8 BTL-UI RFT 3B 95.6 86.1 84.8 84.0 71.4 63.4 84.4 77.2 64.0 BTL-UI RFT 7B 96.8 88.5 88.0 88.2 76.9 69.2 84.6 78.4 65.4

4.3 Ablation Study

As shown in Table 6, to clarify the contributions of each component in our BTL framework, we conduct an ablation study on the AndroidControl-High benchmark. When trained only with SFT, BTL-UI achieves a baseline performance with an SR of 60.6%. While further using the generated Blink data, SFT obtains a 5% improvement. This proves that Blink data is not only suitable for RFT,

- Table 6: Ablation study of BTL-UI. All ablation experiments are evaluated on the AndroidControlHigh benchmark by evaluating the grounding and planning capabilities of the agent.

(a) Ablation study of training method and BTL. Blink Data refers to the data contribution in §3.2. BTL Reward denotes the reward design in §3.3.

Blink BTL AndroidControl-High Data Reward Type GR SR

SFT RFT

- - - - 68.7 59.7 47.1 ✓ 79.4 63.9 60.6 ✓ ✓ 86.4 69.9 65.6

✓ 86.2 71.3 65.4 ✓ ✓ ✓ 88.2 76.9 69.2

(b) Ablation study of Blink Phase ROIs.

AndroidControl-High Type GR SR

λ

- 1 87.0 72.1 66.6
- 2 87.6 72.8 67.4
- 3 88.0 74.2 68.1
- 4 86.8 75.6 68.4
- 5 88.2 76.9 69.2
- 6 89.4 73.1 69.2

but also for SFT. Furthermore, RFT without Blink data achieves an SR of 65.6%. After adopting Blink data and BTL reward, SR is improved by 3.6%.

Moreover, we examine the effect of varying the number of Blink ROIs (λ): increasing λ from 1 to 6 steadily improves success rates from 66.6% to 69.2%, after which gains plateau, suggesting an optimal trade-off between annotation complexity and attention coverage. It is observed that from Table 6, as λ increases, the performance is saturated, so the final λ is selected as 5.

Overall, the ablation results confirm that each element of the BTL framework—Blink Phase for targeted attention, Think Phase for structured reasoning, and the Link Phase for precise validation—plays a crucial role in achieving competitive performance in GUI interaction tasks.

- 4.4 Visualization We present the visualization results and qualitative analysis in the appendix.
- 5 Conclusion and Limitations

We propose the BTL framework, an innovative GUI interaction architecture inspired by the biological cognitive paradigm of Blink–Think–Link. This framework simulates the human closed-loop system of visual perception, cognitive decision-making, and action execution during GUI operations, overcoming the limitations of traditional outcome-driven RFT approaches. Experimental results show that the BTL-UI agent, developed under this framework, achieves significant performance improvements across a variety of GUI interaction tasks.

We believe that the BTL framework proposed in this study establishes a promising and generalizable paradigm for developing digital assistants that are more natural, efficient, and aligned with human cognition. It not only benefits human-GUI interaction but can also be extended to other humancomputer interaction tasks, such as embodied intelligence.

Limitations. The proposed BTL framework introduces <blink> tag outputs compared to conventional Think-Answer structured outputs. Although the blink-generated ROI regions are adaptive and can be empty (zero-length), they typically increase the output sequence length in most cases. While demonstrating performance improvements across various GUI task metrics, this design incurs additional computational processing overhead.

### References

- [1] Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, et al. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024.
- [2] Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for general computing devices use, 2024.
- [3] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al. Gui agents: A survey. arXiv preprint arXiv:2412.13501, 2024.
- [4] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [6] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024.

- [7] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.
- [8] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [9] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.
- [10] Xiaobo Xia and Run Luo. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.
- [11] Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239, 2025.
- [12] Simon P Liversedge and John M Findlay. Saccadic eye movements and cognition. Trends in cognitive sciences, 4(1):6–14, 2000.
- [13] Alejandro Jaimes and Nicu Sebe. Multimodal human–computer interaction: A survey. Computer vision and image understanding, 108(1-2):116–134, 2007.
- [14] Robert JK Jacob. The use of eye movements in human-computer interaction techniques: what you look at is what you get. ACM Transactions on Information Systems (TOIS), 9(2):152–169, 1991.
- [15] Songqin Nong, Jiali Zhu, Rui Wu, Jiongchao Jin, Shuo Shan, Xiutian Huang, and Wenhao Xu. Mobileflow: A multimodal llm for mobile gui agent. arXiv preprint arXiv:2407.04346, 2024.
- [16] Yunpeng Song, Yiheng Bian, Yongtao Tang, Guiyu Ma, and Zhongmin Cai. Visiontasker: Mobile task automation using vision based ui understanding and llm task planning. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, pages 1–17, 2024.
- [17] Xiao Liu, Bo Qin, Dongzhu Liang, Guang Dong, Hanyu Lai, Hanchen Zhang, Hanlin Zhao, Iat Long Iong, Jiadai Sun, Jiaqi Wang, et al. Autoglm: Autonomous foundation agents for guis. arXiv preprint arXiv:2411.00820, 2024.
- [18] Huawen Shen, Chang Liu, Gengluo Li, Xinlong Wang, Yu Zhou, Can Ma, and Xiangyang Ji. Falcon-ui: Understanding gui before following user instructions. arXiv preprint arXiv:2412.09362, 2024.
- [19] Fei Tang, Yongliang Shen, Hang Zhang, Siqi Chen, Guiyang Hou, Wenqi Zhang, Wenqiao Zhang, Kaitao Song, Weiming Lu, and Yueting Zhuang. Think twice, click once: Enhancing gui grounding via fast and slow systems. arXiv preprint arXiv:2503.06470, 2025.
- [20] Filippos Christianos, Georgios Papoudakis, Thomas Coste, Jianye Hao, Jun Wang, and Kun Shao. Lightweight neural app control. arXiv preprint arXiv:2410.17883, 2024.
- [21] Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Vem: Environment-free exploration for training gui agent with value environment model. arXiv preprint arXiv:2502.18906, 2025.
- [22] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.
- [23] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. In International Conference on Learning Representations, 2025.

- [24] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.
- [25] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [26] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [27] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via visionguided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025.
- [28] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. arXiv preprint arXiv:2503.20752, 2025.
- [29] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [30] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.
- [31] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent, 2024.
- [32] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.
- [33] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [34] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scalable lightweight infrastructure for fine-tuning, 2024.
- [35] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for generalist gui agent. In NeurIPS 2024 Workshop on Open-World Agents, 2024.
- [36] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv e-prints, pages arXiv–2406, 2024.
- [37] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024.
- [38] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.
- [39] OpenAI. Gpt-4o, 2024. Accessed: 2025-01-03.
- [40] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [41] Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-ui: Visual grounding for gui instructions. arXiv preprint arXiv:2412.16256, 2024.

### A Prompt

#### System Prompt

You are a GUI Agent capable of reasoning based on user instructions, action history, and the current screenshot. You should first observe the layout of the screenshot and extract N elements RELATED TO the user instruction, where 0 <= N <=5. Next, think about the reasoning process BASED ON the observations and instructions in your mind, and then provide the user with the answer. The observation process (can be None if N == 0), reasoning process and answer are enclosed within <blink></blink>, <think></think> and <link></link> tags, respectively, i.e., <blink>

- <element><id>1</id><bbox>[x0, y0, x1, y1]</bbox><caption>dynamic</caption></element>
- <element><id>2</id><bbox>[x2, y2, x3, y3]</bbox><caption>static</caption></element>
- <element><id>3</id>.....</element>
- <element><id>4</id>.....</element> </blink> <think> reasoning process here </think> <link> answer(["Plan": ..., "Action": "function": ..., ...]) </link>. where captions must be one of [dynamic, static], "dynamic" refers to the interactive area, and "static" refers to the non-interactive areas, such as text and diagrams in the screenshot. And the observation can be <blink> None </blink>, if N == 0. User Instruction Prompt

You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You MUST CHOOSE the next action from the following defined action space.

#### ## Action Space

- Action 1: Back

- - format: {’function’: ’Back’}
- - purpose: back to the previous screen.

- Action 2: Home

- - format: {’function’: ’Home’}
- - purpose: navigate to the home page.

- Action 3: Tap

- - format: {’function’: ’Tap’, ’position’: [x, y]}
- - purpose: tap the specified position.

- Action 4: Type

- - format: {’function’: ’Type’, ’text’: ’str’}
- - purpose: enter specified text at the designated location.

- Action 5: Swipe

- - format: {’function’: ’Swipe’, ’direction’: ’str’}
- - purpose: swipe on the screen in the specified direction.

- Action 6: LongPress

- - format: {’function’: ’LongPress’, ’position’: [x, y]}
- - purpose: long press the specified position ## User Instruction High − Level Instruction ## Action History

- Step 1: ......
- Step 2: ...... ## Screenshots <image>

Table 7: The prompt for the BTL-UI.

The prompt of BTL-UI is shown in Table 7. The system prompt is used to format the output of the model according to the three-phase paradigm of Blink-Think-Link. Moreover, the model outputs according to the format of the system prompt, which is convenient for the calculation of the BTL

reward to adjust the model distribution. As shown in equation 6, because the output of Blink Phase can be ∅, we emphasize that the Blink Phase can be output None in the system prompt.

OS-Atlas [23] has found that blindly mixing data from different sources for multitask fine-tuning can significantly harm performance due to action space conflicts. We unify the action space of GUI-Odyssey [37] and AndroidControl [36]. And we prompt the model to select the corresponding action from the defined action space. Furthermore, we declare the format and purpose in the user instruction prompt, so that the model can better understand each action type. For the grounding and high-level tasks, only the high-level instruction will be fed into the model. For the low-level tasks, both the high-level and the low-level instructions will be given to the model. In the end, the interaction history will be added to the user instruction prompt.

### B Visualization

[Figure 36]

[Figure 37]

|<blink><br><br><element><id>1</id><bbox>[978,161,1064,240]</bbox><caption>dynamic</caption></element><br><element><id>2</id><bbox>[688,170,795,240]</bbox><caption>dynamic</caption></element><br><element><id>3</id><bbox>[832,170,923,240]</bbox><caption>dynamic</caption></element><br><element><id>4</id><bbox>[10,180,110,240]</bbox><caption>dynamic</caption></element><br><element><id>5</id><bbox>[300,2040,378,2130]</bbox><caption>dynamic</caption></element> </blink> <think> The user wants to listen live to Radio GupShup 94.3 FM and search for other radio stations. Currently, we are on the \'Now Playing\' screen of the app. To search for other radio stations, I need to use the search icon which is located at the top right corner of the screen. This will allow me to input the name of another radio station or browse through available options. </think> <link> [{"Plan": "Click on the search icon to search for other radio stations.", "Action": {"function": "Tap", "position": [1010, 207]}}] </link><br>|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

|<blink><br><br><element><id>1</id><bbox>[148,170,1070,262]</bbox><caption>dynamic</caption></element><br><element><id>2</id><bbox>[25,424,228,504]</bbox><caption>dynamic</caption></element><br><element><id>3</id><bbox>[248,424,384,498]</bbox><caption>dynamic</caption></element><br><element><id>4</id><bbox>[404,424,606,498]</bbox><caption>dynamic</caption></element><br><element><id>5</id><bbox>[624,424,767,498]</bbox><caption>dynamic</caption></element> </blink> <think> The user wants to search for other radio stations besides the current one. The search bar is already active, so I need to type the name of another station to find it. The next step is to type the name of the desired radio station into the search bar. </think> <link> [{"Plan": "Type the name of another radio station", "Action": {"function": "Type", "text": "94.3 FM"}}] </link><br>|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

|<blink><br><br><element><id>1</id><bbox>[148,170,1070,262]</bbox><caption>dynamic</caption></element><br><element><id>2</id><bbox>[25,424,228,504]</bbox><caption>dynamic</caption></element><br><element><id>3</id><bbox>[248,424,384,498]</bbox><caption>dynamic</caption></element><br><element><id>4</id><bbox>[404,424,606,498]</bbox><caption>dynamic</caption></element><br><element><id>5</id><bbox>[624,424,767,498]</bbox><caption>dynamic</caption></element> </blink> <think> The user wants to search for other radio stations besides the current one. The search bar is already active, so I need to type the name of another station to find it. The next step is to type the name of the desired radio station into the search bar. </think> <link> [{"Plan": "Select the search result for Radio GupShup 94.3 FM", "Action": {"function": "Tap", "position": [546, 1342]}}] </link><br>|
|---|

[Figure 44]

[Figure 45]

Figure 4: Visualization of the interaction trajectory of the proposed BTL-UI on AndroidControl-High. The corresponding ID of this random case is 19477. And the high-level instruction is ’Listen live to Radio GupShup 94.3 FM and search for other radio stations.’ The tap icon in black is the prediction of BTL-UI, and the other is the ground-truth.

The visualization of the interaction trajectory of our BTL-UI is shown in Figure 4. The high-level instruction is “Listen live to Radio GupShup 94.3 FM and search for other radio stations.” The Blink Phase can locate the ROIs related to the instruction. And the thinking Phase can reason based on the instruction, interaction history, and candidate area. As shown in step 2 of the interaction trajectory in

Figure 4, in the Blink Phase, BTL-UI not only locates the input box to complete the task, but also analyzes the historical search records in the screenshots.

However, since AndroidControl is an offline interaction benchmark, there are some unreasonable labeling data. For instance, step 2 needs to input the text of “94.3 FM” according to the task instruction. But the search box in the screenshot after interaction shows “93.5 FM”, which may affect subsequent interactions. In step 3, the labeled action is to click the search icon. And the search icon is also located in the Blink Phase. Due to the interaction errors in step 2 caused by data noise, the Think Phase of BTL-UI believes that clicking on the “94.3 FM” in historical search records in the screenshot is more reasonable. Therefore, we suppose our BTL-UI has stronger reasoning and error correction abilities.

### C Experiment Statistical Significance

In this section, we report the experiment’s statistical significance. The random factor that affects our results is the sampling of the training process. As shown in Table 1, the training data of our BTL-UI is sampled from various datasets. In the data sampling process, we fix the random seed to 2025 to maintain reproducibility. And the sampled data is further adopted to generate Blink Data, following the pipeline in §3.2. Moreover, BTL-UI adopts the ms-swift [34] framework for RL training. During the training process, we also fix the random seed to 2025 to maintain reproducibility.

