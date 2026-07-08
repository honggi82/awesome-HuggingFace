## Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks

[Figure 1]

Wenqi Zhang1,* Mengna Wang2,3,* Gangao Liu2,3 Xu Huixin2,6,7 Yiwei Jiang2,6,8

Yongliang Shen1 Guiyang Hou1 Zhe Zheng1 Hang Zhang4 Xin Li5 Weiming Lu1 Peng Li2,3,6,† Yueting Zhuang1,†

# arXiv:2503.21696v2[cs.CL]14May2025

1College of Computer Science and Technology, Zhejiang University 2Institute of Software, Chinese Academy of Sciences 3University of Chinese Academy of Sciences

4Alibaba Group 5DAMO Academy, Alibaba Group 6Nanjing Institute of Software Technology

7Nanjing University of Posts and Telecommunications 8Hohai University zhangwenqi@zju.edu.cn, lipeng@iscas.ac.cn Project: https://embodied-reasoner.github.io/

### Abstract

|• Interactive Scenario<br>• Long-horizon Search Task<br>• Image-text Interleaved Trajectory<br>• Spontaneous Reasoning with Coherent and Diverse Thoughts<br>|
|---|

<Self-Reflection> The above places were not found. I need to reflect on my search

plan. I may overlooked some potential locations. Perhaps it is in the drawer …

Recent advances in deep thinking models have demonstrated remarkable reasoning capabilities on mathematical and coding tasks. However, their effectiveness in embodied domains which require continuous interaction with environments through image action interleaved trajectories remains largely -unexplored. We present Embodied Reasoner, a model that extends o1 style reasoning to interactive embodied search tasks. Unlike mathematical reasoning that relies primarily on logical deduction, embodied scenarios demand spatial understanding, temporal reasoning, and ongoing self-reflection based on interaction history. To address these challenges, we synthesize 9.3k coherent Observation-Thought-Action trajectories containing 64k interactive images and 90k diverse thinking processes (analysis, spatial reasoning, reflection, planning, and verification). We develop a three-stage training pipeline that progressively enhances the model’s capabilities through imitation learning, self-exploration via rejection sampling, and self-correction through reflection tuning. The evaluation shows that our model significantly outperforms those advanced visual reasoning models, e.g., it exceeds OpenAI o1, o3-mini, and Claude-3.7 by +9%, 24%, and +13%. Analysis reveals our model exhibits fewer repeated searches and logical inconsistencies, with particular advantages in complex long-horizon tasks. Real-world environments also show our superiority while exhibiting fewer repeated searches and logical inconsistency cases.

<Verification>…<Task Planning> Where is the drawer? I need .. <Action> Observe

Thoughtn Actionn

Embodied-Reasoner

….

###### Task Obs1 Thought1 Action1

Obsn

<Situation Analysis> Let me observe the room and analyze … <Spatial Reasoning> According to the objects and layout of the room ... <Task Planning> I will search for the keychain, which may be on the bedside table, desk, or left on the sofa. Then, I need to find the desk lamp…

<Action> Navigate to sidetable Navigate to desk ….

Q: Could you find the keychain and then put it beside the desk lamp?

[Figure 2]

#0 Init #1 Navigate to sidetable #2 Navigate to desk

[Figure 3]

[Figure 4]

Tried searching multiple areas but didn't find coffee. After reflection, realize that it may be hidden in cabinet

|[Figure 5]|
|---|

|[Figure 6]<br><br>#1 Nav to Countertop|
|---|

|[Figure 7]<br><br>….|[Figure 8]|
|---|---|

|[Figure 9]<br><br>#6 Nav to Cabinet|
|---|

|[Figure 10]<br><br>#1<br><br>#2<br>#3<br>|
|---|

###### #4 …. ….

#0 Init #4 Nav to Countertop #7 Open Cabinet #10 Put in microwave

#8 Pickup…

Real-world Experiment: Can you help me find the coffee and heat it up?

Figure 1. We design an embodied interactive task: searching for objects in an unknown room. Then we propose EmbodiedReasoner, which presents spontaneous reasoning and interaction ability. Before each action, it generates diverse thoughts, e.g., self-reflection or spatial reasoning, forming an image-text interleaved trajectory. It shows consistent reasoning and efficient search behaviors, whereas OpenAI o3-mini often exhibits repetitive searches and logical inconsistencies with higher failure rates.

### 1. Introduction

Recent advancements in deep-thinking models such as OpenAI o1 [35], Gemini 2.0 Flash Thinking [10], DeepSeek R1 [14], and Qwen-QwQ [47] have demonstrated remark-

*The first two authors have equal contributions. †Corresponding author.

[Figure 11]

[Figure 12]

###### Q: Could you first place Bread on the CounterTop, and then place Egg on the SinkBasin for me?

[Figure 13]

A: Observation-Thought-Action-Observation …

[Figure 14]

#1 <Situation Analysis>Ok, let me take a moment to carefully analyze the room. Ah, I see a fridge against the far-right wall. To its left, there’s a counter with a coffee machine and microwave, with cabinets above. Near the left side, a stove burner sits with knobsstands... </ Situation Analysis > #1 <Spatial Reasoning>The overall task involves placing Bread on the CounterTop and Egg on the SinkBasin. My current subtask is to find the Bread and place it on the CounterTop. Let me think, where might the bread be? Based on the image, Bread is likely stored in areas associated with food containment or preparation. The fridge is a logical first search location, as it is designed for ……The other surfaces, like microwave,.....are less likely to store Bread. </ Spatial Reasoning> #1 <Task Planning> Alright, I guess I could start by making a plan first. I will begin my search for the Bread at the fridge, as it is the most probable location for stored food. Once found, I will proceed to place the Bread on the CounterTop as per the subtask requirement. </Task Planning> #1 <Action>Navigate to Fridge

#1 <Situation Analysis> #1 <Spatial Reasoning>

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- #1 <Task Planning>

- #1 <Action>Navigate to Fridge
- #2<Task Planning> Next I should Open fridge and check ..…

- #2<Action> Open Fridge

#2: Open Fridge

- #3 <Situation Analysis>

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

#0: Initialization

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

#1: Navigate to Fridge

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

- #3 <Action> Close Fridge

- #3 <Image>
- #4 <Self-Reflection>

- #4 <Action> Observe

- #3<Situation Analysis>Hmm, it doesn’t seem quite right. I saw that it was nearly empty, with just one egg inside. I need to close the fridge and then reassess my search locations. </Situation Analysis>

- #3 <Action>Close Fridge
- #4<Self-Reflection>Maybe I need to reflect on my search plan now. My initial observation of the room was insufficient…I may have overlooked other potential locations... To proceed effectively, a comprehensive observation of the surroundings is necessary to identify all plausible locations for the Bread. It will guide my search and ensure no viable areas are missed. </Self-Reflection>

- #4<Action>Observe

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Back Left Right

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

#4: Observe

[Figure 56]

#5 <Spatial Reasoning>Let me think this through carefully. From my perspective, CounterTop near the stove and sink does not visibly contain Bread….. The DiningTable, visible in the room, is a logical next choice, as it is a common location for items like Bread…... </ Spatial Reasoning > #5 <Task Planning> Nice, I’ve come up with a new search plan. I will navigate to the DiningTable to examine its surface for the Bread. By focusing on this area, I aim to continue my systematic search , prioritizing likely storage or placement locations. </ Task Planning> #5 <Action>Navigate to DiningTable

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

#5<Spatial Reasoning> #5<Task Planning> #5<Action>Navigate to DiningTable

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

#5: Navigate to DinningTable

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

- #6<Action>Pickup Bread ….
- #7<Action>Navigate to CounterTop ….
- #8<Action>Put in CounterTop

- #8 <Image>
- #9 <Task Planning>

- #9 <Action>Navigate to Fridge …..

[Figure 71]

[Figure 72]

#9 <Task Planning>Okay, now let me continue planning the next steps. From my perspective, I see the CounterTop with the Bread placed on it near the stove and pot. My next subtask is to place the Egg on the SinkBasin. Eggs are typically stored in food containment zones, making the Fridge a logical location to search first. Wait, I remember checking the fridge before, and it seems like the egg was already inside. I will navigate to the Fridge…. </Task Planning> #9 <Action>Navigate to Fridge

[Figure 73]

[Figure 74]

[Figure 75]

#8: Put in CounterTop

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

- #10 <Action> Open Fridge ….
- #11 <Action> Pickup Egg ….
- #12 <Action> Navigate to SinkBasin ...
- #13 <Action> Put in SinkBasin

- #13<Image>
- #14< Double Verification >

- #14<Action> End

[Figure 81]

[Figure 82]

#14 <Double Verification>Let me double-check just to be sure. From my perspective, I see the SinkBasin directly ahead with the Egg now placed on it. The Bread is previously positioned on the CounterTop near the stove. These placements successfully fulfill the specified subtasks. Reviewing the task requirements and actions taken, I conclude that the overall task of placing Bread on the CounterTop and Egg on the SinkBasin is fully accomplished. </Double Verification>

#13: Put in SinkBasin

- Figure 2. Embodied-Reasoner exhibits spontaneous thinking behaviors, e.g., analyzing environmental states (#1,3), reflecting on missed details (#4), reasoning based on the latest observations (#5), and recalling cues for efficient planning (#9). These thoughts remain coherent and logically consistent despite spanning multiple rounds. In contrast, general VLMs lacking thinking abilities struggle with long-horizon interactive tasks and produce unreasonable actions, e.g., forget tasks or repetitive searching.

able reasoning capabilities in domains requiring extensive deliberation. These models, trained through large-scale reinforcement learning (RL)[14, 46] or post-training on elaborate thought trajectories[32, 74], exhibit human-like thinking patterns and self-reflection before arriving at solutions. Their success has led to significant progress in domains requiring deliberate reasoning, particularly in college-level mathematics [13, 32] and coding tasks [20, 72].

Despite these advances, a critical question emerges: Can the o1-style reasoning paradigm be extended beyond these specialized domains to address more complex challenges that require embodied intelligence? Particularly, can these reasoning capabilities be effectively applied to embodied tasks demanding long-horizon planning and deliberate reasoning in interactive environments [44]? This extension is non-trivial due to several fundamental challenges:

Challenge 1: Extended Multimodal Interaction. Compared to most question-answering tasks, which are limited to single-turn dialogues, embodied models operate in an interactive manner over long-horizon task. This means them must continuously interact with the environment, gather real-time feedback, which most appear in visual modality, and then make reasonable actions accordingly (textual modality). In such scenarios, the model need to process lengthy and image-action interleaved context and produce coherent, contextually consistent reasoning. Nevertheless, this remains a challenge for many current multimodal models and visual reasoning models [12, 15, 15, 39, 63, 63]. We observe that even advanced reasoning models like OpenAI o3-mini [36] frequently fail to exhibit robust reasoning in these embodied interactive tasks, leading to repetitive or inconsistent behaviors.

Challenge 2: Diverse Reasoning Modalities. Different from mathematical tasks that primarily rely on professional knowledge and logical deduction, embodied scenarios demand a broader set of capabilities existing in daily life. As shown in Fig. 2, when searching for an object hidden in an unknown room, the model must leverage commonsense knowledge to infer potential search areas (e.g., steps 1, 3), comprehend object spatial relationships to plan efficient exploration paths at steps 1, 5, and employ temporal reasoning to recall relevant cues from previous attempts (step 9) while reflecting on prior failures. These multifaceted reasoning requirements pose challenges for multimodal models.

In this paper, we present Embodied-Reasoner, a novel approach that extends deep-thinking capabilities to embodied interactive tasks. Our key insight is that effective embodied reasoning requires not just the ability to process multimodal inputs, but also to generate diverse thinking processes (analysis, planning, reflection) that adapt to different stages of an interaction. To develop this capability, we develop a data engine that automatically synthesizes coherent Observation-Thought-Action trajectories enriched with diverse, embodied-specific thinking processes, e.g., situational analysis, spatial reasoning, self-reflection, task planning, and verification. These coherent, image-text interleaved trajectories guide the model to learn how to plan and reason based on its interaction history and spatial layout, thereby boosting its spatial and temporal reasoning capabilities. We further introduce a three-stage iterative training pipeline for embodied model that combines imitation, selfexploration, and self-correction. The pipeline begins with imitation learning on synthesized trajectories to develop basic interaction skills, followed by rejection sampling tuning to enhance exploration abilities, and concludes with reflection tuning to foster self-correction.

We evaluate our approach on four high-level embodied tasks in the AI2-THOR simulator [26]: Search, Manipulation, Transportation, and Composite Tasks. These tasks require agents to locate hidden objects in unfamiliar environments through reasoning and planning, then manipulate or transport them to designated areas. Our data engine synthesizes 9.3k task instructions paired with interactive trajectories, containing 64k images and 8M thought tokens, spanning 107 diverse indoor scenes, 2,100 objects, and 2,600 containers. These trajectories are used for model training.

Across 809 tasks in 12 novel scenarios, EmbodiedReasoner significantly outperforms state-of-the-art VLMs and visual reasoning models, exceeding OpenAI o1, o3mini, and Claude-3.7-Sonnet-thinking by +9% in success rate and +12% in search efficiency. The performance gap widens particularly for complex composite tasks, where our model outperforms the second-best model by +39.9%. Our analysis reveals that Embodied-Reasoner demonstrates more consistent reasoning and efficient search behaviors by

spontaneously generating more reasoning tokens for complex tasks and avoiding repetitive and inefficient exploration through temporal reasoning.

Our contributions include: (1) A framework for extending deep-thinking to embodied scenarios that addresses the unique challenges of interactive reasoning; (2) A data engine that synthesizes diverse embodied reasoning trajectories with interleaved observations, thoughts, and actions; (3) A three-stage training pipeline that progressively enhances interaction, exploration, and reflection capabilities; and (4) Extensive evaluation showing significant improvements over state-of-the-art models, particularly for complex long-horizon tasks.

### 2. Observation-Thought-Action Corpora

To develop o1-style reasoning models for embodied scenarios, we first design an embodied task that requires high-level planning and reasoning rather than low-level motor control, i.e., search for hidden objects (Sec. 2.1). Next, we design a data engine in the simulator to synthesize interactive reasoning corpora: task instructions (Sec. 2.2) and corresponding key action sequences (Sec. 2.3). Each action produces a visual observation, forming an interaction trajectory. Lastly, we generate multiple thinking thoughts for each action, e.g., situation analysis, task planning, spatial reasoning, reflection, and verification, creating an interactive reasoning corpus with Observation-Thought-Action contexts (Sec. 2.4).

As shown in Fig. 2, the model requires spatial reasoning abilities to understand the layout of the kitchen and object relations, infer potential locations (fridge, dinning table) based on commonsense knowledge, systematically search unexplored areas, and adapt its plan through real-time observations while avoiding repetitive searches.

#### 2.1. Embodied Interactive Task

Task Environments. We built our embodied task environment using the widely adopted AI2-THOR simulator, which provides physics simulation and real-time vision displays. We employ 120 unique indoor scenes, e.g., kitchens, and 2,100 objects, e.g., credit card and microwave. We control the robot’s movements (e.g., moveahead) and interactions (e.g., pickup object) using AI2THOR’s API, while capturing visual observations at each step.

Task Categories. The robot is initialized in a corner of an unknown room with a limited view, i.e., only part of the room is visible. We design four common interactive tasks in everyday life, with different complexities. ①Search: Searching for an object in an unknown room, e.g., keychain. It may be placed somewhere or hidden inside a container. ②Manipulate: Interacting with objects after searching, such as “finding a lamp and turning on the switch”. ③Transport: After finding a hidden object, transport it to another location. It involves multiple search and manipula-

Instruction Synthesis

###### Trajectory

###### Training Recipe

[Figure 83]

[Figure 84]

[Figure 85]

<Situation Analysis> < Task Planning >

[Figure 86]

Search Manipulate Transport Composite Data Engine

[Figure 87]

[Figure 88]

[Figure 89]

###### Synthesis Trajectory × 𝟏𝟏𝟐𝟖

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

- • Task Templates: Pickup {A} from container and put into {B}
- • Constraint Check A: pickupable ∧ A.Parent: openable ∧ ….

⇒ A∈{Keychain, Book..} B ∈{Sofa, Desk}

- • Diversify Instructions: Style & Difficulty Please help me Pickup Keychain and then put in desk..

#1 Nav to Side Table

- 1st
- 2nd

[Figure 94]

- #1

- #2

[Figure 95]

[Figure 96]

Embodied-Interactor

[Figure 97]

[Figure 98]

Sample

[Figure 99]

[Figure 100]

< Self-Reflection >

[Figure 101]

[Figure 102]

#2 Nav to Desk0000

Data Engine

<Instruction>

Trajectory

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Judgement

[Figure 109]

Action Sequence Synthesis

[Figure 110]

[Figure 111]

< Self-Reflection >

[Figure 112]

Self-exploration Trajectory × 𝟔𝟐𝟒𝟔

[Figure 113]

#3 Nav to Mudroom

[Figure 114]

[Figure 115]

[Figure 116]

InterleavingThoughtwithObservation-Action

[Figure 117]

#4 Nav to Sofa 0 0

[Figure 118]

###### Affiliation Graph

[Figure 119]

[Figure 120]

[Figure 121]

Embodied-Explorer

Root

[Figure 122]

Sample on Previous Dataset

[Figure 123]

###### Bedroom Mudroom

[Figure 124]

<Situation Analysis> < Task Planning >

Trajectory

a o+ t T1 T2 T3

Data Trajectory Engine

Drawer

Desk Sidetable Sofa

- #5 Observe000 00

- #4

- #6

- #7

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

left right

front back

- #5

[Figure 129]

Lamp Plate ….

Keychain

Insert Abnormal State into

<Spatial Reasoning >

[Figure 130]

[Figure 131]

Add Search Processes

[Figure 132]

- #6 Nav to Drawer00

- #7 Open Drawer00

a o- reflection a

o+

t

Key Actions

[Figure 133]

- #1 Nav to Sidetable
- #2 Nav to Desk
- #3 Nav to Mudroom
- #4 Nav to Sofa
- #5 Observe
- #6 Nav to Drawer
- #7 Open Drawer
- #8 Pickup Keychain …

[Figure 134]

[Figure 135]

[Figure 136]

Nav to Mudroom Nav to Drawer Open Drawer Pickup Keychain Nav to Desk Put in Desk

[Figure 137]

T1 T2 T3 reflection T3 Reflect on Erroneous Action in

T4

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

Self-correction Trajectory × 𝟐𝟎𝟏𝟔

[Figure 148]

3rd

[Figure 149]

<Double Verification>

[Figure 150]

Embodied-Reasoner

…..

- Figure 3. Left: Data Engine for <Instruction, Interactive Trajectory> synthesis. First, we synthesize instructions from task templates, and build an affiliation graph from scene’s meta-data. It enables us to derive key actions needed for task. We add exploratory actions and insert thinking thoughts between observation and actions. Right: Three-stage training recipe. ①We fine-tune on synthesized trajectory to develop interaction skills. ②We sample multiple trajectories on novel tasks and evaluate their correctness. The successful ones are used for developing its exploring abilities. ③We continue to sample trajectories using updated model, injecting anomalous states and reflective thoughts in successful cases and correcting errors in failed ones. This self-correction training yields Embodied-Reasoner.

tion steps. ④Composite task: Involving multiple transport tasks in order, e.g., “Place the egg in the microwave, and then put it on the desk after heating. After that, find ..”.

tions into different styles and complexities.

Task Templates with Constraints. We design multiple task templates for each task. Fig. 3 presents a transport task template: pickup {invisible A} and put in {B}, where A denotes a hidden object with pickupable attribute, e.g., keychain, and object B should contain containment properties, such as drawer or desk. It ensures synthesized instruction’s validity. Templates and constraints are in Appendix C.

Action Definition. Although AI2-THOR provides many low-level actions, our task focuses on higher-level planning and reasoning rather than movement control. Besides, lowlevel actions may lead to excessive interactions, so we encapsulate 9 high-level actions built on top of atomic actions: Observe, Move Forward, Navigate to {}, Put in {}, Pickup {}, Toggle {}, Close {}, Open {}, Termination.

Code-based Object Filter. We instruct GPT-4o to select an appropriate task template and generate code for constraint checking based on object’s metadata. It selects objects that satisfy the constraints. We fill the template with matched objects, (A: keychain, B: desk), and synthesize multiple instructions with different object combinations.

#### 2.2. Instruction Synthesis

Our data engine leverages LLMs to generate task instructions automatically. However, unlike previous instruction synthesis [42, 55], embodied task instruction must satisfy the constraints of the scenario, i.e., avoid referencing objects that do not exist in the current scene or involving illegal actions, e.g., “Please move the sofa to corner” is invalid if the scene does not contain a sofa or sofa cannot be moved. Thus, we first design multiple task templates for each task, leverage GPT-4o’s coding capabilities to automatically select objects that meet task constraints, and diversify instruc-

Diversify Instructions. Lastly, it automatically diversifies instructions from two levels: ① Style: We employ GPT4o to rewrite the filled template into multiple human-style instructions, e.g., “I can not find my keychain. Can you help me find them and ...”. ② Difficulty: We sequentially combine multiple simple tasks to create composite tasks.

#### 2.3. Action Sequence Synthesis

###### Count of Five Thinking Patterns

Verification

Task Planning

36,663

Our engine automatically annotates the key actions sequences for synthesized instructions and also produces various action sequences with additional search processes.

Spatial Reasoning

V

26,434

SelfReflection

Reasoning

17,552

6%

100% 3%

Situation Analysis

Spatial

8,882

Affiliation Graph. Firstly, as shown in Fig. 3, we construct an affiliation graph using the simulator’s metadata. In the graph, every node represents an object, and edge denotes an affiliation relation between two objects, e.g., keychain in a drawer is depicted as leaf (keychain) connected to a parent node (drawer) with a “include” relationship.

42%

S

Double Verification

2,131

ACTION

19%

45%

22%

75%

60% 33%

100%

55% 40% R

BEGIN A P

Analysis Planning Reflection

Key Action Sequence. Then we utilize the constructed affiliation graph and synthesized instruction template to derive the minimum required action sequence (key actions) for task completion. For example, “pickup the keychain and place it in desk”, we start from the leaf node (keychain) and trace upward to its parent node (drawer) and grandparent node (mudroom). GPT-4o generates the corresponding action sequence: A1: Navigate to Mudroom, A2: Navigate to Drawer, A3: Open Drawer, A4: Pickup .... All key actions are indispensable for completing tasks.

Conversion Between Five Thinking Patterns

Figure 4. We analyze the frequency of five types of thoughts and their flexible transition relationships in all trajectories.

consistent with the previous thoughts (t1:n-1).

### 3. Training Recipe for Embodied-Reasoner

To incentivize the reasoning ability, we design three training stages, i.e., imitation learning, rejection sampling tuning, and reflection tuning, that bootstrap a general VLM into an embodied interactive model with deep thinking ability.

Add Additional Searching Processes. Beyond the key action sequences, our engine also synthesizes exploratory paths by inserting additional search processes. For instance, as shown in Fig. 3, our engine first inserts three searching actions: Nav to Sidetable, Desk, and Sofa. After failing to find the keychain, it inserts an Observe action until it ultimately locates the keychain in the drawer. These additional search actions make the trajectory more realistic and reasonable, showcasing how robot gradually explores an unfamiliar environment until it successfully locates the target.

Multi-turn Dialogue Format. Considering that the interactive trajectory follows an interleaved image-text format (Observation-Thought-Action), we organize them as multiturn dialogue corpora. In each turn, the observed images and the simulator’s feedback serve as the User Input, while thoughts and actions as Assistant Output. During training, we compute the loss only for thoughts and action tokens.

#### 3.1. Learn to Interact: Imitation Learning

#### 2.4. Interleaving Thought with Observation-Action

In the first stage, we use a data engine to generate a small set of instruction-trajectory, most contain limited searching processes or only consist of key actions (ObservationThought-Key action). Qwen2-VL-7B-Instruct is fine-tuned on this dataset and learns to understand interleaved imagetext context, output reasoning and action tokens.

After running synthesized actions (a1,a2,...an), we obtain an interaction trajectory: o1,a1,o2,a2,...,on,an, where oi denotes first-person perspective images. Then, we generate multiple deep-thinking thoughts (ti) for each action, creating an interleaved context: Observation-Thought-Action.

Diverse Thinking Pattern. Firstly, we define five thinking patterns to simulate human cognitive activities in different situations: Situation Analysis, Task Planning, Spatial Reasoning, Self-Reflection, and Double Verification. We use concise prompts to describe each pattern, guiding GPT4o in synthesizing the corresponding thinking process.

After tuning, we develop Embodied-Interactor, which is capable of interaction within embodied scenarios. However, most synthesized trajectories only include key actions for task completion, without searching processes or observing environments. In most cases, Embodied-Interactor exhibits limited search capabilities, i.e., it does not know how to handle situations where objects cannot be directly found and require further searching. For instance, when it opens a refrigerator for an egg but is empty, it may respond: “egg does not exist” rather than searching for other locations.

Derive Thought from Observation-Action. For each interaction, we instruct GPT-4o to select one or more thinking patterns and then generate detailed thoughts based on interactive context. These thoughts are inserted between observations and actions (on,an→on,t1n,t2n,..tkn,an). Specifically, we prompt GPT-4o with the previous interaction trajectory (o1,t1,a1,...,on), and upcoming action (an), and generate a well-reasoned thinking process (tn). It should consider the latest observation (on) and offer reasonable rationales for next-step action (an), and also remain logically

#### 3.2. Learn to Search: Rejection Sampling Tuning

Self-exploration Trajectory. DeepSeek-R1 reveals that advanced reasoning ability can be acquired through rejection sampling and reward-guided RL on large-scale self-

exploration data. Inspired by this, we employ EmbodiedInteractor to sample massive self-generated trajectories for further training. Specifically, as shown in Fig. 3, we employ data engine to synthesize new task instructions and their key actions, and then use Embodied-Interactor to sample multiple trajectories on each instruction under high-temperature settings. Lastly, we select high-quality trajectories.

Data Engine as the reward model. We use our data engine as a process supervision reward model (PRM) to assess these sampled trajectories. We retain 6,246 successful trajectories—most of which complete the task after several search attempts. We perform second-stage instructiontuning on all collected trajectories, developing EmbodiedExplorer. We observe it exhibits adaptive planning and searching behaviors. For example, when the target object cannot be found directly, it formulates a detailed search plan involving multiple potential areas with different priorities.

#### 3.3. Learn to Self-reflect: Reflection Tuning

Embodied-Explorer occasionally produces unreasonable actions, especially in long-horizon tasks, such as hallucination. Besides, robots often encounter temporary hardware malfunctions. It requires the model to self-reflect on unreasonable behaviors, recognize abnormal states, and correct them on time. As shown in Fig. 3, we employ EmbodiedExplorer to sample massive trajectories on previous tasks. ① For failed trajectories, we locate the first erroneous actions and construct self-correction trajectories. ② For succeeded trajectories, we insert anomaly states to simulate hardware fault.

Insert Abnormal State into Succeeded Trajectory. We simulate two robot anomalies: navigation anomaly, where the robot navigates to an inconsistent location with command (e.g., action: “navigate to the refrigerator”, but instead navigates to table); and manipulation anomaly, where robot arm temporarily fails to perform interaction command. For a succeeded trajectory {.., a, o+, t..}, we insert an abnormal state (o−) after an action (a), and then generate self-reflective thoughts (tr) for this anomalies. Lastly, we retry the same action: {.., a, o−, tr, a, o+ ..}.

Reflect on Unreasonable Action in Failed Trajectory. Using synthesized key actions, we identify the first incorrect action in each failed trajectory (Traj−). Then we generate self-reflective thoughts (tr) for incorrect action and supplement remaining correct trajectories (Traj+t:n), creating a revised trajectory: {Traj−1:t, ttr, Traj+t:n}. We fine-tune the model on synthesized self-correction trajectories. For loss calculation, we mask out erroneous partial trajectory (Traje1:t) and only compute the loss for reflective tokens (ttr) and the correct trajectory (Trajct:n).

Stage #Trajectory Source #Imageall #Imagemax #Actionavg

|Train1st Train2nd Train3rd Total|1,128 Synthesis 4636 11 4.11<br><br>6,246 Self-Explore 45.8k 26 7.33<br><br>2,016 Synthesis 13.8k 29 8.63<br><br><br>9,390 - 64k 29 7.22<br><br>|
|---|---|
|Testset|809 Human 4.9k 29 6.06|

Table 1. We synthesize 9.3k ⟨task, trajectory⟩ for training. Also, we manually annotate key actions for 809 novel testing tasks.

### 4. Dataset Statistics

#### 4.1. Training Corpus

As shown in Tab. 1, we synthesize 9,390 unique task instructions and their Observation-Thought-Action trajectory for three training stages, i.e., ⟨Scene,Inst,Traj⟩. In the first stage, data engine synthesizes 1,128 instruction-trajectory pairs. In the 2nd stage, we remain 6,246 exploratory trajectories through rejection sampling. In the 3rd stage, data engine synthesizes 2,016 self-correction trajectories.

Our dataset spans 107 diverse indoor scenes, e.g., kitchens and living rooms, and covers 2,100 interactive objects (e.g., eggs, laptops) and 2,600 containers (e.g., refrigerators, drawers). All trajectories contain 64K a first-person perspective image from interaction and 8M thought tokens. The distribution of our dataset is presented in Appendix B.

#### 4.2. Thoughts Analysis

The Distribution of Thinking Patterns. We count the frequency of five thinking patterns in all trajectories. As shown in Fig. 4, task planning and spatial reasoning appear most frequently, with 36.6K and 26.4K, respectively. It means each trajectory contains about four planning and three reasoning. Besides, Self-reflection often occurs after a search fails, with about two times per trajectory. These diverse thoughts incentivize the model’s reasoning capabilities.

Conversion between Thinking Patterns. We also compute the transition probabilities between five thinking patterns (see Fig. 4). We find the relationship between them is flexible and depends on the situation. It typically begins with Task Planning, followed by task planning (55%) and spatial reasoning (45%). When navigating unknown regions, it frequently relies on spatial reasoning (Action→S: 42%). If a search attempt fails, it shifts to self-reflection (Action→R: 33%), and once a (sub-)task is completed, it may perform double verification sometimes (Action→V: 3%, S→V: 6%). This diverse structure enables the model to learn spontaneous thinking and flexible adaptability.

#### 4.3. Interactive Evaluation Framework

We cultivate 809 test cases across 12 novel scenarios, which are different from training scenes. We manually design instructions and annotate corresponding key actions and final

states: ⟨Instruction, Key Action, Final state⟩. Notably, our test-set contains 25 carefully designed ultra long-horizon tasks, each involving four sub-tasks and 14-27 key actions.

Metrics We design three metrics to assess the quality of the model-generated trajectories. Success Rate (%): It measures whether a task is successfully completed by evaluating if the key actions align correctly and if the final state meets the task criteria. Search Efficiency: It evaluates task efficiency—more steps indicate lower efficiency. We calculate it as the ratio of key action numbers to predicted action numbers. Task Completeness (%): It computes the proportion of predicted actions that belong to the set of key actions.

### 5. Experiments

#### 5.1. Main Results

Much higher success rate, search efficiency, and task completeness. As shown in Tab. 2, Embodied-Reasoner significantly outperforms all reasoning models and VLMs by a large margin (+9.6% to GPT-o1), including the latest GPT-o3-mini (+24%) and Claude-3.7-Sonnet-thinking (+13%). Besides success rate, our model also demonstrates clear advantages on search efficiency and task completeness over others, e.g., search efficiency is higher than GPT-o1 by +12%. Despite being significantly smaller than advanced reasoning models, Embodied-Reasoner demonstrates stronger interaction and reasoning capabilities in embodied scenarios.

Advantages are more obvious on complex tasks. Analyzing success rate across four different task categories (search, manipulate, transport, and composite tasks), we observe that Embodied-Reasoner demonstrates significantly stronger performance on the more challenging tasks like composite and transport. Notably, on composite tasks, it outperforms the second-best model (GPT-4o) by +39.9%. Interestingly, on the relatively simpler search task, our model lags behind GPT-o3-mini by 13.4%. Our analysis finds that on these simpler tasks, Embodied-Reasoner sometimes over-explores, leading to missed detections of nearby objects.

Our three-stage training progressively incentivize interaction and reasoning capabilities, from 14.7% to 80.9%. Our base model, Qwen2-VL-7B, initially achieved only 14.7%. After the first-stage imitation learning, it improved to 25.4%, mastering simple interaction ability. Subsequently, rejection sampling tuning significantly boosted performance to 65.4%, reaching a level comparable to GPTo1 with exploration abilities. Finally, fine-tuning with selfcorrection trajectories further elevated the model’s success rate to 80.9%. We observe that most baseline models often exhibit repetitive searching behaviors and unreasonable planning, especially when handling long-horizon tasks. In contrast, Embodied-Reasoner performs a strategic search

100

|71.87<br><br>90.33<br><br>56.34<br><br>74.8<br><br>0<br><br>35.71<br><br>0<br><br>69.44<br><br>87.25 90.15 90<br><br>84.21<br><br>64.71 66.67<br><br>49.33<br><br>80<br><br>82.59<br><br>43.1<br><br>33.63<br><br>29.16<br><br>1.5<br><br>61.61 55.19<br><br>27.11<br><br>35.59<br><br>5.88<br><br>0 3.33<br><br>Embodied-Reasoner<br><br>GPT-o1<br><br>Gemini 2.0Flash-thinking<br><br>Qwen2-VL-72B|
|---|

80

SuccessRate(%)

60

40

20

0 00 0 0

- 0

- 0.5
- 1

1.5

2

- 2.5
- 3

- 3.5
- 4

1 2 4 5 6 7 8 ≥9

#OutputTokens(k)

Task Length (#Key Action)

Gemini-2.0-flash-thinking

Qwen2-vl-72B-instruct

Embodied-Reasoner-7B GPT-o1

Task Length (#Key Action)

4.3×

2.4×

2.6×

Figure 5. Relations between task length and success rate, and output token number. As task complexity increases, our model generates more reasoning tokens to maintain high success rates.

and planning after deep thinking and timely self-reflection, significantly reducing those unreasonable cases.

- 5.2. Analysis: How does deep-thinking paradigm enhance embodied search tasks?

More robust to long-horizon tasks. To investigate the impact of the deep-thinking paradigm on embodied search tasks, we count the number of key actions required for each test case. More key actions indicate the task is more complex with more interactions, i.e., long-horizon tasks. As shown in Fig. 5, we visualize the relationship between task length (number of key actions), success rate, and the number of output tokens. We observe that as the number of key actions increases, the success rates of baseline models drop significantly—especially when the task exceeds five actions. In contrast, our model remains robust to complex tasks, achieving over 60% success rate in most scenarios.

Spontaneously scale up reasoning tokens for complex embodied tasks. Fig. 5 (Bottom) shows our EmbodiedReasoner utilizes significantly more reasoning tokens for complex searching tasks, nearly five times that of Gemini2.0-flash-thinking. Besides, as tasks become more complex, the response tokens of our model also grow from 1,000 to 3,500 tokens. We observe that when faced with complex composite searching tasks, Embodied-Reasoner engages in significantly longer analysis processes and more deliberate self-reflection. This deep thinking process enables it to plan more efficient search paths and avoid redundant actions, improving the success rate. In contrast, Gemini-2.0-flashthinking does not show a clear increase in their output tokens, remaining at 1,000 tokens. It suggests it may fail to solve complex embodied tasks by inference time scaling up.

Deep thinking alleviates repetitive searching actions. We observe that baseline models frequently exhibit repetitive search behaviors. For instance, after inspecting a cabinet, the model may still attempt to check the same cabinet

|Model<br><br>| |Success Rate ↑<br><br>|Search Efficiency ↑|Task Completeness ↑|Success Rate for SubTasks ↑| | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |Search<br><br>|Manipulate|Transport<br><br>|Composite|
|Generalpurpose VLMs|Qwen2.5-VL-7B-Instruct [4] Qwen2-VL-7B-Instruct [53] Qwen2.5-VL-72B-Instruct [4] Qwen2-VL-72B-Instruct [53] Claude 3.5-Sonnet [2] Qwen-VL-Max [48] GPT-4o [34]<br><br>|12.38% 10.87% 27.53% 6.45% 23.55% 7.56% 0.95% 14.79% 11.97% 38.67% 23.33% 25.50% 2.82% 0.00% 31.75% 22.61% 50.62% 52.14% 38.89% 21.90% 0.00% 39.00% 28.88% 54.56% 50.00% 52.36% 33.19% 0.00% 45.35% 28.05% 64.12% 54.25% 50.51% 51.22% 3.84% 49.81% 36.28% 68.39% 63.87% 63.21% 45.16% 1.90% 66.67% 41.68% 79.07% 69.03% 79.26% 71.95% 14.42%<br><br>| | | | | | |
|Visual Reasoning Models|QVQ-72B-Preview [39] Kimi-K1.5† [46] GPT-o3-mini [36] Gemini-2.0 Flash Thinking [10] Claude-3.7-Sonnet-thinking [3] GPT-o1 [35]<br><br>|7.54% 6.39% 36.33% 4.35% 7.50% 10.53% 0.00% 46.00% - - - - - 56.55% 26.93% 67.41% 78.57% 59.32% 66.67% 0.00% 56.74% 43.01% 71.70% 71.05% 75.60% 40.67% 8.89% 67.70% 37.95% 78.63% 69.12% 75.88% 71.94% 13.79% 71.73% 43.06% 82.49% 78.42% 79.10% 67.36% 13.16%<br><br>| | | | | | |
| |Embodied-Interactor-7B (ours-1st) Embodied-Explorer-7B (ours-2nd) Embodied-Reasoner-7B (ours-3rd)<br><br>|25.46% 24.75% 53.67% 30.97% 27.09% 29.20% 3.81% 65.39% 46.25% 77.73% 60.00% 75.92% 72.24% 26.67% 80.96% 55.07% 86.30% 65.16% 93.31% 87.20% 54.29%<br><br>| | | | | | |

###### Table 2. We compare the performance of Embodied-Reasoner against advanced VLMs and visual reasoning models. After the three-stage training process, we boost Qwen2-VL-7B from 14.8 to 81. Kimi-K1.5† means we manually evaluate 50 testing cases through the webpage.

[Figure 151]

70 60 50 40 30 20 10

RepetitiveExplorationRate(%)

Embodied-Explorer-7B(2rd) Embodied-Reasoner-7B(3th) Claude 3.5-Sonnet GPT-4o Gemini-2.0 Flash Thinking Qwen-VL-Max GPT-o3-mini Qwen2-VL-72B-Instruct Qwen2.5-VL-72B-Instruct

###### Task Instruction: Can you help me find the coffee and heat it up?

###### ......

Continue to Search

#1 Navigate to Countertop

#8 Pickup Coffee Cup

#2 Navigate to Dinning Table

#6 Navigate to Cabinet #7 Open Cabinet

Potential Places

###### #9 Thought:

#2 Thought: I noticed

#1 Thought: Let me think..., my first step is to find the

Next, I will heat

that there is no coffee on the countertop. It seems that I should

up coffee. I have already seen the microwave on

coffee and then heat it

###### ......

up.Looking around the room, I notice a countertop, a dining table, and a small coffee table. Hmm... I

#9 Navigate to #0 Init Microwave

look for it elsewhere.

Wait, it seems that I have seen a microwave oven. This can be used to heat up.

Embodied-Reasoner

Embodied-

the countertop

Reasoner Success Rate: 56.7%

before...

###### suppose the countertop is

Embodied-Explorer

0

the most likely place to start. I should check there first.

Search Manipulate Transport Composite Overall

After I find coffee.

OpenAI o3-mini

Lacking planning:

###### Repetitive Search:

###### Forgot Task:

Figure 6. Repetitive Exploration Rate measures repetitive search issues, which are often observed in baselines. Our models reduce repetitive searches by recalling and reflecting on past trajectories.

o3-mini first came to the microwave instead of looking for coffee.

o3-mini repeatedly

- o3-mini forgot to
- open the cabinet door and check.

Success Rate:

comes to the same area for inspection.

43.4%

First View Left View

✘ ✘

###### ...... ......

Right View

Back View

#1 Navigate to #14 Navigate to coffee table

#9 Navigate to coffee table

#7 Navigate to microwave #2 observe cabinet #8 observe

Figure 7. Real-world experiments. Our model achieves a higher success rate (56.7%) than OpenAI o3-mini (43.4%) and o1 (50%).

after a few steps. This behavior reflects its weaker temporal reasoning and context awareness abilities in interactive scenarios. To quantify this, we define a repetitive exploration rate (RER), which measures how often the model navigates to the same area within its trajectory. As shown in Fig. 6, our models (Embodied-Reasoner / Explorer) consistently exhibit a significantly lower RER (-50%) compared to baseline models across all four tasks. For example, in composite tasks, Embodied-Explorer achieves the lowest RER of 26%, while GPT-o3-mini and Qwen2-VL-72B reach 54% and 43%, respectively. Compared to EmbodiedExplorer, Reasoner exhibits a slightly higher RER due to its more cautious nature, favoring multiple checks and reflections. In our model’s reasoning thoughts, we observe that it frequently recalls past observations, reflects on previously explored actions, and formulates new plans accordingly. These processes enhance its temporal reasoning ability, thereby reducing repetitive search behaviors.

#### 5.3. Real-world Experiments

To evaluate the generalization of our reasoning model, we design a real-world experiment about object searching, covering 30 tasks across three scenes: 6 kitchen tasks, 12 bathroom tasks, and 12 bedroom tasks. During testing, a human operator holds a camera to capture real-time visual input. The model analyzes each image and generates an action command, which the operator executes the actions.

Fig. 7 illustrates an example: “Can you help me find the coffee and heat it up?” Our model rules out the countertop and dining table after two explorations (steps 1,2), ultimately locating the coffee (#7) in the cabinet and placing it in the microwave for heating (#11). However, we observe that OpenAI o3-mini fails to formulate a reasonable plan, heading to the microwave first instead of searching for the coffee. Besides, it frequently forgets to search and ex-

hibits repetitive searching, aligning with our previous analysis. Please refer to Table A1 for detailed results.

### 6. Related Works

#### 6.1. Large Language Models Reasoning

Recent o1-style models [10, 14, 35, 47, 72, 74] have demonstrated powerful reasoning capabilities, significantly enhancing their ability on college-level benchmarks such as mathematics [16, 40]. Unlike previous efforts to scale up training data and model sizes, these systems involve generating long “thought” tokens during inference time, improving the performance of final answers [13, 32, 67, 69]. From chain-of-thought (CoT) prompting by humans [25, 56] to autonomous thinking derived by LLMs themselves [70], the reasoning abilities have become increasingly internalized and spontaneous. Besides, the long-thought generation process is quite similar to human thinking activities, with diverse thinking patterns, e.g., step-by-step solving, self-reflection, backtracking, and self-validation [20, 37]. These characteristics greatly enhance the complex problemsolving capabilities of LLMs. However, most efforts focus on text-based tasks, and the application of deep-thinking paradigms to embodied scenarios remains largely unexplored. This is the focus of our work.

#### 6.2. Vision-Language Model Reasoning

Recently, many efforts [49, 54, 61, 62, 68], such as QVQ [39] and Kimi-1.5 [46], have extended the deepthinking paradigm to multimodal scenarios by post-training on long-CoT thought or large scale reinforcement learning. However, most visual reasoning models operate in a single-round dialogue setting: processing input images and user’s query and generating textual thoughts for a final answer. This limits their applicability in embodied interactive tasks [1, 5, 11, 24, 30], which require handling multi-image or image-text interleaved contexts while generating coherent and logical thoughts across multiple interactions. Besides, embodied scenes differ from mathematical tasks, as they demand long-horizon planning and deliberate reasoning over previous trajectories. Inspired by image-text interleaved understanding and generation [27, 71], we propose an effective solution to extend general VLMs into embodied reasoning models, including synthesizing interleaved observation-thought-action trajectories and bootstrapping VLMs through iterative training.

#### 6.3. Vision-Language-Action Model

Some researches integrate language models into embodied tasks [19, 28, 41, 45, 77]. For instance, agents like PaLME[11] and other researches[6, 18, 21, 29, 33, 38, 43, 59, 60, 64, 75, 76] leverage the inherent knowledge of large language models (LLMs) and combine robotic data with

general visual-language data for training. Another part of the research[7, 9] emphasizes the integration of language models (LLMs) and visual-anguage models (VLMs) into robotics. Take Chatgpt for robotics[50] as an example. The study first defines a high - level robotic function library, then creates prompts to guide ChatGPT to achieve its goals.

In addition, some research[52, 57, 58, 73] utilizes the capabilities of visual-language models (VLM) to assist traditional strategy training such as reinforcement learning and imitation learning, thus promoting the development of robotic manipulation and navigation[17, 23, 31, 65]. Besides, a large amount of research[8, 22, 51, 66] has been dedicated to developing generalist robotic strategies, which are trained based on continuously expanding robotic learning datasets. However, most of the existing work directly predicts action sequences without enabling robots to have an autonomous thinking process and this is one of the key focuses of our work.

### 7. Conclusions

In this paper, we propose an embodied reasoning model, Embodied-Reasoner, for interactive search tasks that can spontaneously search, reason, and act. To develop this model, we design a data engine that synthesizes 9,390 interactive trajectories in an Observation-Thought-Action interleaved format. It encompasses 64K images and 8M thought tokens featuring diverse thinking patterns. We employ a three-stage training approach—imitation learning, rejection sampling tuning, and reflection tuning—to progressively enhance its interaction and reasoning abilities. Extensive evaluations and real-world experiments demonstrate that our model exhibits superior reasoning capabilities.

### References

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 9
- [2] Anthropic. Claude 3.5-sonnet: Anthropic’s advanced language model, 2024. 8
- [3] Anthropic. Claude 3.7 sonnet: Anthropic’s hybrid reasoning ai model, 2025. 8
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 8
- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 9
- [6] Shaofei Cai, Zihao Wang, Kewei Lian, Zhancun Mu, Xiaojian Ma, Anji Liu, and Yitao Liang. Rocket-1: Mastering

- open-world interaction with visual-temporal context prompting. arXiv preprint arXiv:2410.17856, 2024. 9
- [7] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 9
- [8] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, and Ziran Wang. Drive as you speak: Enabling human-like interaction with large language models in autonomous vehicles. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 902–909, 2024. 9
- [9] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, et al. A survey on multimodal large language models for autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 958–979, 2024. 9
- [10] DeepMind. Flash thinking, 2025. 1, 8, 9
- [11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 9
- [12] Yifan Du, Zikang Liu, Yifan Li, Wayne Xin Zhao, Yuqi Huo, Bingning Wang, Weipeng Chen, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Virgo: A preliminary exploration on reproducing o1-like mllm. arXiv preprint arXiv:2501.01904, 2025. 2
- [13] Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519, 2025. 2, 9
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 2, 9
- [15] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. arXiv preprint arXiv:2412.05237,

2024. 2

- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 9
- [17] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 10608–10615. IEEE, 2023. 9
- [18] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023. 9
- [19] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extract-

- ing actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022. 9
- [20] Zhongzhen Huang, Gui Geng, Shengyi Hua, Zhen Huang, Haoyang Zou, Shaoting Zhang, Pengfei Liu, and Xiaofan Zhang. O1 replication journey–part 3: Inference-time scaling for medical reasoning. arXiv preprint arXiv:2501.06458,

2025. 2, 9

- [21] Pranav Kak and Sushma Jain. Embodied reasoning with self-feedback. In 2024 IEEE International Conference on Electronics, Computing and Communication Technologies (CONECCT), pages 1–5. IEEE, 2024. 9
- [22] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-Cheol Min. Smart-llm: Smart multi-agent robot task planning using large language models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12140–12147. IEEE, 2024. 9
- [23] Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. Goat-bench: A benchmark for multi-modal lifelong navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16373– 16383, 2024. 9
- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 9
- [25] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022. 9
- [26] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474,

2017. 3

- [27] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-ofthought. arXiv preprint arXiv:2501.07542, 2025. 9
- [28] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE, 2023. 9
- [29] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering, pages 405–409, 2024. 9
- [30] Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. Text2motion: From natural language instructions to feasible plans. Autonomous Robots, 47

(8):1345–1365, 2023. 9

- [31] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal,

- Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16488– 16498, 2024. 9
- [32] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024. 2, 9
- [33] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36:25081–25094, 2023. 9
- [34] OpenAI. Gpt-4o: Openai’s multimodal language model,

2024. 8

- [35] OpenAI. Openai o1: Advanced reasoning ai model, 2024. 1, 8, 9
- [36] OpenAI. o3-mini: Openai’s advanced reasoning model,

2025. 2, 8

- [37] Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector Liu, Yuanzhi Li, et al. O1 replication journey: A strategic progress report– part 1. arXiv preprint arXiv:2410.18982, 2024. 9
- [38] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025. 9
- [39] QwenLM. Qvq-72b preview, 2025. 2, 8, 9
- [40] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. arXiv preprint arXiv:2311.12022,

2023. 9

- [41] Gabriel Sarch, Yue Wu, Michael J Tarr, and Katerina Fragkiadaki. Open-ended instructable embodied agents with memory-augmented large language models. arXiv preprint arXiv:2310.15127, 2023. 9
- [42] Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. Taskbench: Benchmarking large language models for task automation. arXiv preprint arXiv:2311.18760, 2023. 4
- [43] Lucy Xiaoyang Shi, Zheyuan Hu, Tony Z Zhao, Archit Sharma, Karl Pertsch, Jianlan Luo, Sergey Levine, and Chelsea Finn. Yell at your robot: Improving on-the-fly from language corrections. arXiv preprint arXiv:2403.12910,

2024. 9

- [44] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020. 2
- [45] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot

- grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2998–3009, 2023. 9
- [46] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. 2, 8, 9
- [47] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, 2024. 1, 9
- [48] Qwen Team. Qwen-vl-max: Alibaba cloud’s advanced large vision language model, 2025. 8
- [49] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, et al. Llamavo1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 9
- [50] Sai H Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. Chatgpt for robotics: Design principles and model abilities. Ieee Access, 2024. 9
- [51] Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, and Katsushi Ikeuchi. Gpt-4v (ision) for robotics: Multimodal task planning from human demonstration. IEEE Robotics and Automation Letters, 2024. 9
- [52] Jiaqi Wang, Enze Shi, Huawen Hu, Chong Ma, Yiheng Liu, Xuhui Wang, Yincheng Yao, Xuan Liu, Bao Ge, and Shu Zhang. Large language models for robotics: Opportunities, challenges, and perspectives. Journal of Automation and Intelligence, 2024. 9
- [53] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 8
- [54] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024. 9
- [55] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022. 4
- [56] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 9
- [57] Junjie Wen, Yichen Zhu, Minjie Zhu, Jinming Li, Zhiyuan Xu, Zhengping Che, Chaomin Shen, Yaxin Peng, Dong Liu, Feifei Feng, et al. Object-centric instruction augmentation for robotic manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4318–

4325. IEEE, 2024. 9

- [58] Licheng Wen, Xuemeng Yang, Daocheng Fu, Xiaofeng Wang, Pinlong Cai, Xin Li, Tao Ma, Yingxuan Li, Linran Xu, Dengke Shang, et al. On the road with gpt-4v (ision):

- Early explorations of visual-language model on autonomous driving. arXiv preprint arXiv:2311.05332, 2023. 9
- [59] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. Embodied task planning with large language models. arXiv preprint arXiv:2307.01848, 2023. 9
- [60] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. Embodied instruction following in unknown environments. arXiv preprint arXiv:2406.11818, 2024. 9
- [61] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason stepby-step, 2024. URL https://arxiv. org/abs/2411.10440. 9
- [62] Haotian Xu, Xing Wu, Weinong Wang, Zhongzhi Li, Da Zheng, Boyuan Chen, Yi Hu, Shijia Kang, Jiaming Ji, Yingying Zhang, et al. Redstar: Does scaling long-cot data unlock better slow-reasoning systems? arXiv preprint arXiv:2501.11284, 2025. 9
- [63] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024. 2
- [64] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplugowl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 13040–13051, 2024. 9
- [65] Naoki Yokoyama, Ram Ramrakhya, Abhishek Das, Dhruv Batra, and Sehoon Ha. Hm3d-ovon: A dataset and benchmark for open-vocabulary object goal navigation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5543–5550. IEEE, 2024. 9
- [66] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 9
- [67] Di Zhang, Jianbo Wu, Jingdi Lei, Tong Che, Jiatong Li, Tong Xie, Xiaoshui Huang, Shufei Zhang, Marco Pavone, Yuqiang Li, et al. Llama-berry: Pairwise optimization for o1like olympiad-level mathematical reasoning. arXiv preprint arXiv:2410.02884, 2024. 9
- [68] Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-ofthought reasoning. arXiv preprint arXiv:2410.16198, 2024. 9
- [69] Wenqi Zhang, Yongliang Shen, Linjuan Wu, Qiuying Peng, Jun Wang, Yueting Zhuang, and Weiming Lu. Self-contrast: Better reflection through inconsistent solving perspectives. arXiv preprint arXiv:2401.02009, 2024. 9
- [70] Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi Tan, Peng Li, Yueting Zhuang, and Weiming Lu. Agent-pro: Learning to evolve via policy-level reflection and optimization. arXiv preprint arXiv:2402.17574, 2024. 9
- [71] Wenqi Zhang, Hang Zhang, Xin Li, Jiashuo Sun, Yongliang Shen, Weiming Lu, Deli Zhao, Yueting Zhuang, and Lidong

- Bing. 2.5 years in class: A multimodal textbook for visionlanguage pretraining. arXiv preprint arXiv:2501.00958, 2025. 9
- [72] Yuxiang Zhang, Shangxi Wu, Yuqi Yang, Jiangming Shu, Jinlin Xiao, Chao Kong, and Jitao Sang. o1-coder: an o1 replication for coding. arXiv preprint arXiv:2412.00154,

2024. 2, 9

- [73] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. 9
- [74] Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for openended solutions. arXiv preprint arXiv:2411.14405, 2024. 2, 9
- [75] Zhonghan Zhao, Wenhao Chai, Xuan Wang, Boyi Li, Shengyu Hao, Shidong Cao, Tian Ye, and Gaoang Wang. See and think: Embodied agent in virtual environment. In European Conference on Computer Vision, pages 187–204. Springer, 2024. 9
- [76] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 9
- [77] Filippo Ziliotto, Tommaso Campari, Luciano Serafini, and Lamberto Ballan. Tango: Training-free embodied ai agents for open-world tasks. arXiv preprint arXiv:2412.10402,

2024. 9

Long-Horizon Tasks

###### Model Success Rate (%)

Exposed Object Search

25

Sequential Object Transfer

108

80

Qwen2.5-VL-72B-Instruct 43.3 OpenAI o1 50.0 OpenAI o3-mini 44.0 Embodied-Reasoner 56.7

Enclosed-to-Enclosed Object Transfer

17

Enclosed Object Search

47

24

Enclosed-to-Exposed Object Transfer

Test Set Distribution

Table A1. The results of real-world experiments.

116

Exposed-to-Enclosed Object Transfer

194

Sequential Object Transfer

956

Exposed Object Grasping

Enclosed-to-Enclosed Object Transfer

127

93

154

Exposed-to-Exposed Object Transfer

Enclosed-to-Exposed Object Transfer

48

57

2529

803

Exposed Object Search

Exposed Object Toggle Enclosed Object Grasping

Exposed-to-Enclosed Object Transfer

Training

- Figure B2. The distribution of the test set with 809 tasks, including 4 task types and 11 sub-task types.

29171

7916

1406

6479

4788 4038

2381 2258

Navigate Pickup Open Close Put in Observe Move Toggle

30,000 25,000 20,000 15,000 10,000

5,000 0

NumberofInteractions(#Trajectory)

Interaction Types

Quantity of Each Interaction Type in Training Dataset

to Forward

- Figure B3. The distribution of the training set interactions, including 8 interaction types in trajectories: navigate to, pickup, open, close, put in, observe, move forward, and toggle.

Dataset Distribution

x

943

Exposed-to-Exposed Object Transfer

297

Enclosed Object Search

1067

2324

190

Exposed Object Toggle

Exposed Object Grasping Enclosed Object Grasping

Figure B1. The distribution of the training dataset with 9,390 samples, including 4 task types and 10 sub-task types.

### A. Experiment Details

- A.1. Real-World Experiments Result

To evaluate the generalization of our reasoning model, we design a real-world experiment about object searching, covering 30 tasks in three scenes. As shown in Tab. A1, Embodied-Reasoner demonstrates notable capabilities in real-world environments. In terms of success rate, it outperforms OpenAI o1 by 6.7%, OpenAI o3-mini by 12.7%, and Qwen2.5-VL-72B-Instruct by 13.4%.

- A.2. Repeat Exploration Rate

tribution of test tasks is shown in Fig. B2, covering 4 task types and 11 corresponding subtask types.

#### B.2. Distribution of Interaction Types

The Repeat Exploration Rate (RER)indicates how often the model revisits the same location within its trajectory and is calculated as the number of revisits to previous locations divided by the total number of explorations.

In the training set, trajectories consist of eight types of interaction actions: navigate to, pickup, open, close, put in, observe, move forward, and toggle. As shown in the Fig. B3, the occurrence frequency of each interaction action across all trajectories is illustrated. Among them, the exploration action navigate to appears the most frequently, occurring over 29k times.

For example, in a task, the model navigated to the following path: Placea, Placeb, Placeb, Placec, Placec. The model revisited Placeb and Placec once each. Thus, the repeat exploration rate was 40% (2/5).

In the test set, we manually design instructions and annotate the corresponding key actions and final states. The test tasks involve six types of interactions: navigate to, pickup, open, close, put in, and toggle. As seen in the Fig. B4, navigate to also appears significantly more frequently than other key actions.

### B. Dataset Details

#### B.1. Distribution of Task Instructions

We synthesize 9,390 unique task instructions along with their Observation-Thought-Action trajectories as the training set. The distribution of training tasks is shown in Fig. B1, encompassing 4 task types(Search, Manipulate, Transport and Composite) and 10 subtask types.

#### B.3. Distribution of Task Length

In the training set, each trajectory consists of an average of 7.2 interactions with the environment (e.g., navigate, pickup). For the four task types: Search, Manipulate, Transport, and Composite. Due to varying task complexity, the

For evaluation, we curate 809 test cases across 12 novel scenarios distinct from the training environments. The dis-

###### Quantity of Each Interaction Type in Test Set

NumberofInteractions(#KeyAction)

1,500

1432

1,250 1,000

750 500 250

747

505

364

138

97

0 Navigate Pickup Put in Open Close Toggle

To

Interaction Types

- Figure B4. The distribution of the test set interactions, including 6 interaction types in key actions: navigate to, pickup, open, close, put in, and toggle.

|Search<br><br>Manipulate<br><br>Transport<br><br>Composite|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23+

Task Length(#Trajectory Action)

Task Length and Type Composition in Training Set

Numberoftasks

1,400

1,200

1,000

800

600

400

200

0

- Figure B5. The quantity distribution of trajectory lengths in the training set and the corresponding task type composition, where Search Task is mainly within 1-9, Manipulate Task within 2-11, Transport Task within 3-14, and Composite Task above 8, extending beyond 23.

corresponding trajectory lengths also differ. As shown in the Fig. B5, Search tasks tend to have shorter trajectories, typically ranging from 1 to 9, while the more complex Composite tasks generate the longest trajectories, usually exceeding 8 and reaching beyond 23.

Similarly, in the test set, as shown in the Fig. B6, the more complex Composite tasks also exhibit the longest key action sequences, usually exceeding 8 and reaching beyond 19.

###### Task Length and Type Composition in Test Set

350

|Search<br><br>Manipulate<br><br>Transport<br><br>Composite|
|---|

300

250

Numberoftasks

200

150

100

50

0

19+

1

2

4

5

6

7

8

9

10

14

16

18

Task Length(#Key Action)

Figure B6. The quantity distribution of key action lengths in the test set and the corresponding task type composition, where Search Task is mainly within 1-2, Manipulate Task within 2, 4-5, Transport Task within 4-7, and Composite Task above 8, extending beyond 19.

#### B.4. Distribution of Object Categories

Our training dataset includes 107 indoor scenes with diverse functions and layouts (kitchens, living rooms, bedrooms, and bathrooms), featuring over 2,100 interactive objects (e.g., eggs, laptops) and 2,600 containers (e.g., refrigerators, drawers). Across the 9,390 unique task instructions, trajectories involve interactions with these objects and containers, with the top 32 most frequently explored and interacted object categories shown in Fig. B7. In the 12 distinct test set scenes, key actions correspond to different objects, with the top 32 most frequently involved containers and interactive objects illustrated in Fig. B8.

#### B.5. Description of Sub-task Types

Our four daily tasks: Search, Manipulate, Transport, and Composite can be further divided into corresponding subtasks based on the types of objects involved. Specifically, the Search Task can be categorized into Exposed Object Search and Enclosed Object Search. The Manipulate Task can be divided into Exposed Object Grasping, Enclosed Object Grasping, and Exposed Object Toggle. The Transport Task can be classified into Exposed-to-Exposed Object Transfer, Exposed-to-Enclosed Object Transfer, Enclosedto-Exposed Object Transfer, and Enclosed-to-Enclosed Ob-

###### Quantity of Each Object Type in Training Trajectories(Top 32)

Cabinet

6826

Drawer

5858

Fridge

5189

SideTable

3778

CounterTop

3736

DiningTable

2412

Desk

1942

CoffeeTable

1862

Dresser

1548

Sofa

1351

GarbageCan

1225

Shelf

- 1078
- 1079

ArmChair

SinkBasin

910

Microwave

829

Toilet

703

Laptop

659

TVStand

613

Sink

572

CellPhone

563

Bed

530

KeyChain

528

RemoteControl

522

CreditCard

513

Box

421

Bowl

381

Egg

- 320
- 321

StoveBurner

Apple

308

Chair

270

Bathtub

259

Others

6692

0 1,000 2,000 3,000 4,000 5,000 6,000 7,000

Figure B7. The quantity distribution of the top 32 object types in the training dataset’s trajectories, with Others representing the remaining 62 categories, such as Bread, Book, DeskLamp, etc.

ject Transfer. Finally, the Composite Task can be divided into Sequential Object Transfer and Long-Term Complex Task.

Exposed Object Search. This task is defined as searching for items within a room environment. The target items are located on the surface and there is no need to open any containers. For example, if there is an apple placed on the table in the room, the corresponding task description is ”Please find the apple in the room”.

Enclosed Object Search. This type of task refers to searching for items in a room where the target items are inside containers, and the containers need to be opened during the task execution. For instance, if there is an egg in the refrigerator in the room, then the task description is ”Please find the egg in the room”.

Exposed Object Grasping. This type of task requires obtaining a specific item in a room, and the target item is on the surface without the need to open any containers. For example, when there is a cup on the table in the room, the corresponding task description is ”Please pick up the cup in the room”.

Enclosed Object Grasping. This task is to obtain an item located inside a container in a room, and the container needs to be opened during the execution. For example, if

###### Quantity of Each Object Type in Test Set (Top 32)

|28<br><br>28<br><br>28<br>29<br><br><br>29<br><br><br>31<br><br>34<br>35<br>36<br>37<br>38<br><br><br>40<br><br>44<br><br>47<br>48<br><br><br>51<br>52<br><br><br>55<br><br>66<br><br>71|88<br><br>90<br><br>97<br><br>100<br><br>113<br><br>119<br><br>122|221|288<br><br>294|410|514| |
|---|---|---|---|---|---|---|

Drawer

Cabinet

CounterTop

Fridge

DiningTable

SideTable

Desk

SinkBasin

Microwave

CoffeeTable

ArmChair

Sofa

GarbageCan

Shelf

Plate

CellPhone

CreditCard

Box

KeyChain

StoveBurner

TVStand

Laptop

Egg

Bowl

Sink

Lettuce

Pillow

Pen

Book

Bed

Mug

others

0 100 200 300 400 500 600

Figure B8. The quantity distribution of the top 32 object types in the test set’s key actions, with Others representing the remaining 44 categories, such as Watch, Pencil, Cup, etc.

there is a bowl in the cabinet in the room, the corresponding task description is ”Please pick up the bowl in the room”.

Exposed Object Toggle. This task is to perform switch operations on items located on the surface in a room, without the need to open any containers. For example, if there is a coffee machine on the table in the room, the corresponding task description is ”Please start the coffee machine”.

Exposed-to-Exposed Object Transfer. his task requires moving an item located on the surface (without opening any containers) to another position in the room. For example, if there is a spoon on the table in the room, the task description is ”Please pick up the spoon in the room and place it in the sink”.

Exposed-to-Enclosed Object Transfer. This task is defined as moving an item located on the surface (without opening any containers) into a closed container in the room. The container needs to be opened to complete the placement. For example, if there is a credit card on the table in the room, the task description is ”Please pick up the credit card in the room and place it in the drawer”.

Enclosed-to-Exposed Object Transfer. This task refers to moving an item located inside a container to another position after opening the container, finding and picking up the item. For example, if there is a dishwashing sponge in the

cabinet, the corresponding task description is ”Please pick up the dishwashing sponge in the room and place it on the table”.

Enclosed-to-Enclosed Object Transfer. This task is defined as moving an item located inside one container to another closed container in the room. Two containers need to be opened during the execution. For example, if there is a loaf of bread in the refrigerator in the room, the task description is ”Please pick up the bread in the room and place it in the cabinet”.

Sequential Object Transfer. This task requires moving different items to specified locations in a room in a specific order. It is composed of two of the three task types: Exposed-to-Exposed Object Transfer, Exposed-to-Enclosed Object Transfer, and Enclosed-to-Exposed Object Transfer. For example, if there are apples on the table and eggs in the cabinet in the room, the task description is ”First, find the apples in the room and place them in the sink (involving Exposed-to-Exposed Object Transfer), then find the eggs and place them on the table (involving Enclosed-to-Exposed Object Transfer)”.

Long Term Complex Task. This task involves performing a series of ordered and complex long-range operations in a room. It is composed of four of the five types: Exposed Object Toggle, Exposed-to-Exposed Object Transfer, Enclosed-to-Exposed Object Transfer, Enclosedto-Enclosed Object Transfer, and Exposed-to-Enclosed Object Transfer. For example, if there are potatoes in the refrigerator and a mug on the table in the room, the task description is: First, please find the potatoes in the room and wash them (this process involves Exposed-to-Enclosed Object Transfer and Enclosed Object Transfer), then put the potatoes in the microwave to heat (involving Exposed-toEnclosed Object Transfer and Exposed Object Toggle), then find the mug and place it on the coffee machine to get coffee and then place the mug with coffee on the table (involving Exposed Object Transfer and Exposed Object Toggle), and finally place the potatoes from the microwave on the table (involving Enclosed Object Transfer).

### C. Detailed Task Templates and Constraints

We design multiple task templates for each task. It ensures synthesized instruction’s validity. Templates and constraints are shown in Appendix C.

### D. Detailed Prompts

We provide detailed interaction prompt designs for our evaluation framework in Appendix D.

### E. Case Study

As shown in Figure E9, Figure E10, and Figure E11, we provide three cases. The first two illustrate different

trajectory performances of Embodied-Reasoner and GPTo1 for the same task, while the third presents EmbodiedReasoner’s performance in a real-world scenario.

#### E.1. Embodied-Reasoner’s Action Trajectories

- As shown in Figure E9, it illustrates Embodied-Reasoner’s performance on the task: ”Could you please first place the Laptop on the Sofa, and then place the CellPhone on the Drawer?”

E.2. GPT-o1’s Action Trajectories

- As shown in Figure E10, it illustrates GPT-o1’s performance on the task: ”Could you please first place the Laptop on the Sofa, and then place the CellPhone on the Drawer?”

In comparison, during task trajectory, GPT-o1 sometimes forgets the task objective after navigating to the target container. For instance, after the 10th interaction, it navigates to the sofa but forgets to place the Laptop. Additionally, it occasionally falls into action loops, such as the repeated move forward actions from steps 13 to 16, 18 to 21.

In contrast, Embodied-Reasoner performs reasoning and planning, searches multiple locations, and retains memory of the current step’s objective, ultimately completing the task successfully.

E.3. Embodied-Reasoner in Real-World

- As shown in Figure E11, it illustrates Embodied-Reasoner’s performance in the real world for the task: ”I’d appreciate it if you could leave the milk on the coffee table when possible”. It shows that Embodied-Reasoner can complete the task in real-world scenarios through step-by-step planning and reasoning.

|Task Types|Sub-Task Types|Templates<br><br>|Constraint Check|Case|Affiliation and Attribute|
|---|---|---|---|---|---|
|Search<br><br>|Exposed Object Search|find A|Pickupable(A) ∧ ¬ Openable(Parent(A))<br><br>|Task: Could you please find the Apple in the room? Key Action Sequences: navigate to CounterTop end|Apple<br><br>Countertop<br><br>Kitchen<br><br>¬openable<br><br>pickupable|
| |Enclosed Object Search<br><br>|find A|Pickupable(A) ∧ Openable(Parent(A))<br><br>|Task: Could you please find the Apple in the room? Key Action Sequences: navigate to Fridge open Fridge end|Apple<br><br>Fridge<br><br>Kitchen<br><br>openable<br><br>pickupable|
|Manipulate|Exposed Object Toggle|toggle A<br><br>|Toggleable(A) ∧ ¬ Openable(Parent(A))|Task: Would you mind powering on the Laptop for me?<br><br>Key Action Sequences: navigate to Desk toggle Laptop, end<br><br>|Laptop<br><br>Desk<br><br>Living Room<br><br>¬openable<br><br>toggleable|
| |Exposed Object Grasping|pickup A|Pickupable(A) ∧ ¬ Openable(Parent(A))|Task: I want to pick up a CreditCard from the room, can you help me? Key Action Sequences: navigate to SideTable pickup CreditCard end<br><br>|CreditCard<br><br>Sidetable<br><br>Living Room<br><br>¬openable<br><br>pickupable|
| |Enclosed Object Grasping|pickup A<br><br>|Pickupable(A) ∧ Openable(Parent(A))<br><br>|Task: Would it be possible for you to pick up a CreditCard from the room? Key Action Sequences: navigate to Drawer open Drawer pickup CreditCard, end|CreditCard<br><br>Drawer<br><br>Living Room<br><br>openable<br><br>pickupable|
|Transport<br><br>|Exposed-toExposed Object Transfer|pickup A put in B|Pickupable(A) ∧ ¬ Openable(Parent(A)) ∧ ¬ Openable(B)<br><br>|Task: Could you please put the AlarmClock on the Shelf? Key Action Sequences: navigate to Sidetable pickup AlarmClock navigate to Shelf put in Shelf, end|AlarmClock<br><br>Sidetable<br><br>Bedroom<br><br>Shelf<br><br>¬openable<br><br>¬openable<br><br>pickupable|
| |Exposed-toEnclosed Object Transfer<br><br>|pickup A put in B<br><br>|Pickupable(A) ∧ ¬ Openable(Parent(A)) ∧ Openable(B)<br><br>|Task: Would you mind<br><br>placing the Bowl in the Cabinet, please? Key Action Sequences:<br><br>navigate to CounterTop pickup Bowl navigate to Cabinet open Cabinet put in Cabinet, end|Bowl<br><br>CounterTop<br><br>Kitchen<br><br>Cabinet<br><br>¬openable<br><br>openable<br><br>pickupable|

|Task Types<br><br>|Sub-Task Types|Templates|Constraint Check<br><br>|Case<br><br>|Affiliation and Attribute|
|---|---|---|---|---|---|
|Transport<br><br>|Enclosed-toExposed Object Transfer|pickup A put in B<br><br>|Pickupable(A) ∧ Openable(Parent(A)) ∧ ¬ Openable(B)<br><br>|Task: Is it okay to put the Candle on the Bathtub? Key Action Sequences: navigate to Cabinet open Cabinet pickup Candle close Cabinet navigate to Bathtub put in Bathtub end|Candle Cabinet<br><br>Bathroom<br><br>Bathtub<br><br>¬openable openable<br><br>pickupable|
| |Enclosed-toEnclosed Object Transfer<br><br>|pickup A put in B<br><br>|Pickupable(A) ∧ Openable(Parent(A)) ∧ Openable(B)|Task: May I ask you to put the Potato in the Microwave? Key Action Sequences: navigate to Fridge open Fridge pickup Potato close Fridge navigate to Microwave open Microwave put in Microwave end|Potato Fridge<br><br>Kitchen<br><br>openable<br><br>Microwave openable<br><br>pickupable|
|Composite task<br><br>|Sequential Object Transfer|first,<br><br>pickup A1<br><br>put in B1 then,<br><br>pickup A2<br><br>put in B2<br><br><br><br><br>|Pickupable(A1) ∧<br><br>¬ Openable(Parent(A1)) ∧<br><br>Openable(B1) ∧<br><br>Pickupable(A2) ∧ ¬ Openable(Parent(A2)) ∧<br><br>Openable(B2) ∧ Different(A1, A2)<br><br><br><br><br><br><br>|Task: Could you please first place the TeddyBear on the CoffeeTable, and then place the Pen on the GarbageCan? Key Action Sequences: navigate to Bed pickup TeddyBear navigate to CoffeeTable put in CoffeeTable navigate to Desk pickup Pen navigate to GarbageCan put in GarbageCan end<br><br>|TeddyBear Bed<br><br>Bedroom<br><br>pickupable<br><br>CoffeeTable<br><br>Pen<br><br>Desk<br><br>pickupable<br><br>GarbageCan ¬openable<br><br>¬openable<br><br>¬openable<br><br>¬openable<br><br>|
| |Long-Term Complex Task|——<br><br>|——|Task: Could you put the bread in the refrigerator and then take the apple out of the refrigerator, wash it, and place it on a plate? Key Action Sequences: navigate to CounterTop, pickup Bread, navigate to Fridge, open Fridge put in Fridge, pickup Apple navigate to SinkBasin, put in SinkBasin toggle Facuet, pickup Apple navigate to Cabinet, open Cabinet put Plate, close Cabinet end| |

System prompt: You are a robot in given room. You need to complete the tasks according to human instructions. We provide an Available_Actions set and the corresponding explanations for each action. Each step, you should select one action from Available_Actions.

Initialization prompt: <image>This is an image from your frontal perspective. Please select an action from the Available_Actions and fill in the arguments. Task: {taskname} Available_Actions: {{ "navigate to <object>": Move to the object. "pickup <object>": Pick up the object. "put in <object>": Put the item in your hand into or on the object. "toggle <object>": Switch the object on or off. "open <object>": Open the object (container), and you will see inside the object. "close <object>": Close the object. "observe": You can obtain image of your directly rear, left, and right perspectives. "move forward": Move forward to see more clearly. "end": If you think you have completed the task, please output "end".}} Before making each decision, you can think, plan, and even reflect step by step, and then output your final action. Your final action must strictly follow format: <DecisionMaking>Your Action</DecisionMaking>,

for example, <DecisionMaking>observe</DecisionMaking>.

Interaction prompt: After executing your previous {action} , you get this new image above. To complete your task, you can think step by step at first and then output your new action from the Available_Actions. Your action must strictly follow format: <DecisionMaking>Your Action</DecisionMaking>, for example, <DecisionMaking>observe</DecisionMaking>.

- Interaction feedback prompt 1: <|feedback|>Action: {action} is illegal, {object} is the most relevant item in this room and

{action}. Object: {object} is not currently navigable, you can try "navigate to <object>" to reach nearby, larger objects for closer observation.

- Interaction feedback prompt 2: <|feedback|>Action: {object} is illegal, Object: {object} is currently unavailable for interaction. Possible situations include: {object} does not exist in your current view; you are too far away from {object}; the {object} cannot perform operation {action}.\nYou can try

"move forward" to approach the target object or "navigate to <object>" to reach nearby, larger objects for closer inspection.

- Interaction feedback prompt 3: <|feedback|>Action: {action} is illegal, the name of the navigated object doesn’t quite match the obejct in the image, please try navigating to another object first.

- Interaction feedback prompt 4: <|feedback|>Action: {action} is illegal, the name of the object doesn’t quite match the obejct in the image, Please try interacting with another object or navigating to another object.

|[Figure 152]<br><br>#1|
|---|
|[Figure 153]<br><br>#2|

Hmm..., let me observe my surroundings first. I see a Sofa positioned near a CoffeeTable at the center of the room….My plan is to search for the Laptop on the SideTable first, as it is likely to hold such an object and aligns with the current subtask requirements. Once located, I will proceed to place the Laptop on the Sofa to complete the first subtask. If the Laptop is found

here, further actions will be planned accordingly based on subsequent observations. So, I think I've come to a decision.<DecisionMaking>navigate to SideTable</DecisionMaking>

Hold on, I need to rethink this.From the images, the room contains several surfaces: a CoffeeTable with objects on top, a Sofa with items placed on it, and other furniture. The Laptop is still unfound despite prior efforts. ,…,Aligning with my goal to locate the Laptop and place it on the Sofa, focusing on the CoffeeTable as a potential location is both practical and consistent with the task requirements. Thus, I will navigate to the CoffeeTable to continue my search.Hmm..., I think I know what to do.<DecisionMaking>navigate to CoffeeTable</DecisionMaking>

|[Figure 154]<br><br>#3|
|---|

Wait up, lemme get my thoughts together. In the updated view, I notice a Laptop placed prominently on a wooden table near the Sofa… The Sofa, where the Laptop needs to be placed, is conveniently adjacent to this location... My next step is to pick up the Laptop from the table. After securing the object, I will proceed to place it on the Sofa to fulfill the current subtask. This sequential approach ensures efficient progress toward achieving the objective. Okay, I've decided.<DecisionMaking>pickup Laptop</DecisionMaking>

|[Figure 155]<br><br>#4|
|---|

Hang on, let me figure out what to do next. From my current view, I confirm that I am holding the Laptop securely…As the Sofa is the designated target for this subtask, moving closer to it will allow me to complete the action efficiently without any complications. But wait, let me create a plan first. I will navigate to the Sofa, ensuring that I position myself optimally to place the Laptop onto it. Completing this action will advance me toward fulfilling the current subtask. Hmm..., I've settled on a choice.<DecisionMaking>navigate to Sofa</DecisionMaking>

|[Figure 156]<br><br>#5|
|---|

Okay, I think I'll start by making a search plan. In the image, I see the Sofa clearly in front of me, and I am holding the Laptop. The Sofa is unobstructed, providing an ideal surface to place the Laptop. To complete the current subtask, I will carefully position the Laptop on the Sofa, ensuring stability and alignment with the task requirements. This action progresses the task efficiently and prepares me for the subsequent objective. Okay, I've decided.<DecisionMaking>put in Sofa</DecisionMaking>

|[Figure 157]<br><br>#6|
|---|

In the image, I observe a well-lit room with various furniture pieces. The Sofa, where I just placed the Laptop, is prominent in the foreground.…For my current subtask placing the CellPhone on the Drawer I need to locate the CellPhone first. Considering the logical placement of small objects, the SideTable appears to be a suitable candidate to explore next. Therefore, I will navigate to the SideTable to search for the CellPhone. Hmm..., I've settled on a choice.<DecisionMaking>navigate to SideTable</DecisionMaking>

|[Figure 158]<br><br>#7|
|---|

Sorry, let me think about this a bit longer.In the image, I observe a SideTable with a plant and other items but no sign of the CellPhone. Reflecting on my existing plan, searching for the CellPhone logically continues with exploring plausible surfaces. The DiningTable, a common location for small objects like phones, stands out as a suitable next step. …Okay, I think I've made up my mind.<DecisionMaking>navigate to DiningTable</DecisionMaking>

|[Figure 159]<br><br>#8|
|---|

Hmm..., I think I'll make a plan first. In the image, I observe a DiningTable with several objects on it, including a CellPhone positioned near the edge. This matches the second target object required for the current subtask. As the CellPhone is clearly visible and accessible, my immediate plan is to pick it up. Securing the object is essential for progressing toward the goal of

placing it on the Drawer. Once picked up, I will determine the optimal path to reach the Drawer and finalize this subtask. Hmm..., I think I know what to do.<DecisionMaking>pickup CellPhone</DecisionMaking>

|[Figure 160]<br><br>#9|
|---|

But wait, let me create a plan first. In the image, I observe a DiningTable with chairs surrounding it and several objects on its surface, including a CellPhone, which I have already picked up,…, I will navigate to the Dresser, as it is a typical location for drawers and aligns with the task requirements. Hmm..., I've settled on a choice.<DecisionMaking>navigate to

###### Drawer</DecisionMaking>

|[Figure 161]<br><br>#10|
|---|

Hmm..., I think I'll make a plan first. In the image, I observe a Dresser with multiple Drawers beneath its surface. The CellPhone I am holding needs to be placed inside the Drawer. To achieve this, I must first open the Drawer to create access for placement. My next step is to interact with the Drawer to open it, enabling me to proceed with placing the CellPhone inside and completing the subtask efficiently. Hmm..., I think I know what to do.<DecisionMaking>open Drawer</DecisionMaking>

|[Figure 162]<br><br>#11|
|---|

Hmm..., let me think about what to do next. In the image, I observe the open Drawer of the Dresser in front of me. ,…,wait, let me create a plan first. My next step is to carefully place the CellPhone inside the open Drawer. This action will conclude the current subtask and ensure proper alignment with the task's objectives. Following this, I will verify that the action has been executed successfully before proceeding to any subsequent tasks. Okay, I think I've made up my mind.<DecisionMaking>put in Drawer</DecisionMaking>

|[Figure 163]<br><br>#12|
|---|

Hmm..., let me think about what to do next.I observe the open Drawer with the CellPhone placed inside, completing the specified subtask. Earlier, I successfully placed the Laptop on the Sofa as required. Both subtasks outlined in the overall objective have been fulfilled. With no additional tasks evident, I conclude that the entire assignment has been accomplished effectively.Okay, I think I've made up my mind.<DecisionMaking>end</DecisionMaking>

Figure E9. Trajectory Case for Embodied Reasoner

|[Figure 164]<br><br>#1|
|---|

- 2.<DecisionMaking>navigate to Laptop</DecisionMaking>

<|feedback|>Action: navigate to Laptop is illegal, Laptop is the most relevant item in this room and navigate to Laptop. Object: Laptop is not currently navigable, you can try "navigate to <object>" to reach nearby, larger objects for closer observation.

- 3.<DecisionMaking>navigate to DiningTable</DecisionMaking>
- 4.<DecisionMaking>pickup Laptop</DecisionMaking>

<|feedback|>Action: pickup Laptop is illegal, Object: Laptop is currently unavailable for interaction. Possible situations include: Laptop does not exist in your current view; you are too far away from Laptop; the Laptop cannot perform operation pickup. You can try "move forward" to approach the target object or "navigate to <object>" to reach nearby, larger objects for closer inspection.

- 5.<DecisionMaking>move forward</DecisionMaking>
- 6.<DecisionMaking>navigate to Sofa</DecisionMaking>
- 7.<DecisionMaking>navigate to CoffeeTable</DecisionMaking>
- 8.<DecisionMaking>pickup Laptop</DecisionMaking>
- 9.<DecisionMaking>put in Sofa</DecisionMaking>

|feedback|>Action: put in Sofa is illegal, Object: Sofa is currently unavailable for interaction. Possible situations include: Sofa does not exist in your current view; you are too far away from Sofa; the Sofa cannot perform operation Put in. You can try "move forward" to approach the target object or "navigate to <object>" to reach nearby, larger objects for closer inspection.

- 10.<DecisionMaking>navigate to Sofa</DecisionMaking>
- 11.<DecisionMaking>move forward</DecisionMaking>
- 12.<DecisionMaking>observe</DecisionMaking>
- 13.<DecisionMaking>move forward</DecisionMaking>
- 14.<DecisionMaking>move forward</DecisionMaking>
- 15.<DecisionMaking>move forward</DecisionMaking>
- 16.<DecisionMaking>move forward</DecisionMaking>
- 17.<DecisionMaking>navigate to Sofa</DecisionMaking>
- 18.<DecisionMaking>move forward</DecisionMaking>
- 19.<DecisionMaking>move forward</DecisionMaking>
- 20.<DecisionMaking>move forward</DecisionMaking>
- 21.<DecisionMaking>move forward</DecisionMaking>
- 22.<DecisionMaking>navigate to Chair</DecisionMaking>
- 23.<DecisionMaking>put in DiningTable</DecisionMaking>
- 24.<DecisionMaking>pickup Laptop</DecisionMaking>
- 25.<DecisionMaking>navigate to Sofa</DecisionMaking>
- 26.<DecisionMaking>move forward</DecisionMaking>
- 27.<DecisionMaking>move forward</DecisionMaking>
- 28.<DecisionMaking>move forward</DecisionMaking>
- 29.<DecisionMaking>move forward</DecisionMaking>
- 30.<DecisionMaking>navigate to CoffeeTable</DecisionMaking>
- 31.<DecisionMaking>move forward</DecisionMaking>
- 32.<DecisionMaking>observe</DecisionMaking>
- 33.<DecisionMaking>navigate to Chair</DecisionMaking>
- 34.<DecisionMaking>navigate to Sofa</DecisionMaking>
- 35. <DecisionMaking>move forward</DecisionMaking>
- 36.<DecisionMaking>move forward</DecisionMaking>

|[Figure 165]<br><br>#3/5|
|---|

|[Figure 166]<br><br>#6/16/24/33|
|---|

|[Figure 167]<br><br>#7|
|---|

|[Figure 168]<br><br>#8/30|
|---|

|[Figure 169]<br><br>#9/10|
|---|

|[Figure 170]<br><br>#11|
|---|

|[Figure 171]|
|---|

#13/14/15/16 /18/19/20/21 /26/27/28/29 /34/35

|[Figure 172]<br><br>#21/23|
|---|

|[Figure 173]<br><br>#22/32|
|---|

|[Figure 174]<br><br>#31|
|---|

|[Figure 175]<br><br>#35/36|
|---|

Figure E10. Trajectory Case for GPT-o1

Task: I'd appreciate if you could leave the milk on the coffee table when possible.

[Figure 176]

- #0: <DecisionMaking> navigate to Fridge </DecisionMaking>

#0:"Hmm..., let me observe my surroundings first. I see a fridge on the far-right side next to wooden cabinets with a countertop beneath them. A round dining table with chairs is positioned centrally, surrounded by yellow chairs and red armchairs. A set of windows is visible at the back wall, allowing light into the room. Hang on, let me figure out what to do next. Based on the observation, the task requires leaving the milk on the coffee table. The most likely place to find the milk would be the fridge, as it is commonly used to store perishable items like milk. If not found there, the next logical location would be a cabinet or shelf where food or drinks might be stored temporarily. Finally, if the milk is not in either of these locations, the dining table itself could be another potential spot since it's a central area where items may have been left. This sequence balances efficiency and probability based on typical storage practices. But wait, let me create a plan first. First, I will check the fridge for the milk, as it is the most probable

location. If the milk isn't there, I will proceed to search any cabinets or shelves that may hold

stored items. Lastly, I will inspect the dining table as it is a prominent surface where someone might have placed the milk. This systematic approach ensures thoroughness while prioritizing the likelihood of finding the milk quickly. Okay, I've decided. <DecisionMaking>navigate to Fridge</DecisionMaking>"

- #1: <DecisionMaking>open Fridge</DecisionMaking>

- #2: <DecisionMaking>pickup Milk</DecisionMaking>

- #3: <DecisionMaking>close Fridge</DecisionMaking>

- #4: <DecisionMaking> Observe </DecisionMaking>
- #5: <DecisionMaking> navigate to CoffeeTable </DecisionMaking>

- #6: <DecisionMaking>put in CoffeeTable</Decision Making>

- #7: <DecisionMaking>end</ DecisionMaking>

#0 Init

[Figure 177]

#1:Navigate to Fridge

[Figure 178]

#2:Open Fridge

[Figure 179]

- #2:“......Upon opening the fridge, I observe several items inside, including a bottle labeled "Milk" prominently displayed on the door. ……My next logical step is to retrieve the milk so I can relocate it to the coffee table, fulfilling the task...... <DecisionMaking>pickup Milk</DecisionMaking> ”

- #3:“……Closing the fridge is a logical and responsible next step before moving toward the coffee table to complete the task of placing the milk there……Once done, I will navigate toward the coffee table, where the milk needs to be placed…… <DecisionMaking>close Fridge</DecisionMaking> ”

#3:Pickup Milk

[Figure 180]

#4:Close Fridge

#4:"Sorry, I want to reconsider this. While I successfully retrieved the milk from the fridge and ensured the area was orderly by closing the fridge, my initial observations lacked thoroughness regarding the room layout and potential placement surfaces. To accurately identify the coffee table, I must reassess my surroundings. Observing again allows me to better understand spatial relationships and confirm the precise location of both the coffee table and any obstacles. This ensures efficient navigation and accurate placement of the milk, aligning with the task's objective.\nSo, I think I've come to a decision.<DecisionMaking>observe</DecisionMaking> "

[Figure 181]

[Figure 182]

[Figure 183]

Left View Right View Back View

#5:Observe

[Figure 184]

#7:"Okay, let me think carefully.From my current perspective, I observe the milk successfully

placed on the white, round coffee table alongside a folded blue umbrella, confirming the task completion. The coffee table is centrally located among colorful chairs, matching prior descriptions. No further actions are required, as the objective to leave the milk on the coffee table has been fulfilled. The surroundings are orderly, and all elements align with the task's requirements. I conclude that the mission has been accomplished effectively.Okay, I've decided. <DecisionMaking>end</DecisionMaking> "

#6:Navigate to Coffee Table

[Figure 185]

#7:Put in Coffee Tbale

Figure E11. Trajectory Case for Embodied Reasoner in Real World

