# arXiv:2509.01106v2[cs.AI]11Sep2025

[Figure 1]

## Robix: A Unified Model for Robot Interaction, Reasoning and Planning

#### Huang Fang∗, Mengxi Zhang∗, Heng Dong∗, Wei Li∗†, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, Hang Li†

ByteDance Seed ∗Equal Contribution, †Project Lead

#### Abstract

We introduce Robix, a unified model that integrates robot reasoning, task planning, and natural language interaction within a single vision-language architecture. Acting as the high-level cognitive layer in a hierarchical robot system, Robix dynamically generates atomic commands for the low-level controller and verbal responses for human interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally with human within an end-to-end framework. Robix further introduces novel capabilities such as proactive dialogue, real-time interruption handling, and context-aware commonsense reasoning during task execution. At its core, Robix leverages chain-of-thought reasoning and adopts a three-stage training strategy: (1) continued pretraining to enhance foundational embodied reasoning abilities including 3D spatial understanding, visual grounding, and task-centric reasoning; (2) supervised finetuning to model human-robot interaction and task planning as a unified reasoning-action sequence; and (3) reinforcement learning to improve reasoning-action consistency and long-horizon task coherence. Extensive experiments demonstrate that Robix outperforms both open-source and commercial baselines (e.g., GPT-4o and Gemini 2.5 Pro) in interactive task execution, demonstrating strong generalization across diverse instruction types (e.g., open-ended, multi-stage, constrained, invalid, and interrupted) and various user-involved tasks such as table bussing, grocery shopping, and dietary filtering.

Date: September 12, 2025 Correspondence: liwei.85@bytedance.com, lihang.lh@bytedance.com Project Page: https://robix-seed.github.io/robix/

#### 1 Introduction

The goal of generalist robots is to assist humans in diverse daily tasks within open, dynamic environments. Realizing this vision requires more than executing isolated commands—it demands the ability to engage in natural human interaction and reason through complex, long-horizon tasks. For example, when cleaning a dining table, a robot must not only recognize dishes and utensils, but also interpret nuanced instructions such as “only clear the plates if people are finished eating”, respond to corrections like “leave that glass”, and adapt to novel scenarios such as properly sorting stacked tableware. To meet these requirements, a general-purpose robot system should adopt a hierarchical architecture in which a high-level cognitive layer handles complex multimodal reasoning, adaptive task planning, and natural human-robot interaction, while a

Human: I’m hungry. Please put the highest-calorie food on the table into the plastic box and fetch me a drink.

###### Human: I’m allergic to caffeine.

Realtime Interruption

1 2 3 4

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Think: The user requests the highestcalorie food along with a drink. There are various types of food and drinks, including hamburger, drumstick, coke, sprite, orange juice, and coffee. Usually, a hamburger contains more calories than a drumstick, and coke is a suitable pairing with hamburger…

Think: I’ve successfully placed the hamburger into the white plastic box. The user also wants a drink, so next I’ll put the coke into the plastic box as well.

Think: I’m currently holding the coke with the left gripper, but the user just informed me that they’re allergic to caffeine. Since coke typically contains caffeine, I need to place it back on the table and select a caffeine-free drink instead.

Think: …Since the user is allergic to caffeine, and both coke and coffee contain caffeine, the remaining options are orange juice and Sprite, which are typically caffeine-free. Orange juice is generally considered the healthier choice, so I will offer that to the user…

Action: put the hamburger into the white plastic box

Action: put the coke into the white plastic box

Action: place the coke back on the table

Action: put the orange juice into the white plastic box

Response: Okay.

Response: Got it.

| | |
|---|---|
|Human: Next, tidy up the entire table. Throw all the trash into the trash bin,<br><br>Veriﬁcation including the caffeinated drinks. Put everything else into the white plastic box. Monitoring & Replanning<br><br>| |

Task Status

5 6 9 10

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Think: I’ve successfully placed the highest-calorie hamburger and the caffeine-free orange juice into the plastic box, completing the user’s task instruction. Next, I can ask the user if there’s anything else they need me to do.

Think: The user asked me to throw away all trash and caffeinated drinks, and place the remaining items into the plastic box. I will start by discarding the coke and coffee, then throw away common trash items like napkins and paper cups. Finally, I’ll place the utensils into the plastic box...

Think: I’ve successfully thrown the napkin into the trash bin. Next, I can start tidying up the utensils. There is a fork and a knife on the plate. To make the subsequent steps easier, I need to put them away ﬁrst. I will begin by placing the fork, which is on top, into the plastic box.

Think: I noticed that the fork is still on the table, indicating that the previous action failed. I should try again and place the fork into the plastic box ﬁrst.

…

Action: put the coke into the rubbish bin

Action: put the fork into the white plastic box

Action: put the fork into the white plastic box

Response: The task has been completed. Is there anything else I can help with?

Response: Alright, I'll clean it up right away.

Human: Yes, go ahead and throw it away.

Proactive Dialogue

Memorization & Veriﬁcation

[Figure 10]

[Figure 11]

11 15

16

19

[Figure 12]

[Figure 13]

Think: The fork has already been successfully placed into the plastic box. There is still a knife on the plate, so next I will place the knife into the plastic box as well.

Think: …Currently, there is still a chicken drumstick on a green plate and a bottle of sprite on the table. The drumstick is food and typically should either be packed or discarded, but the user hasn’t speciﬁed. I should ask the user whether the drumstick should be thrown away as trash.

Think: The user asked me to throw away the chicken drumstick on the green plate. So next, I should discard the drumstick into the trash bin, and then place the plate into the plastic box.

Think: All items on the table have been cleared. As instructed by the user, the caffeinated drinks and trash have been thrown into the bin, the drumstick was also discarded, and all other items have been placed into the plastic box. The task is now complete. I can ask the user if there’s anything…

…

…

Action: put the chicken drumstick into the rubbish bin

Action: put the knife into the white plastic box

Response: I've cleaned up the table as you instructed. Is there anything else you’d like me to do?

Response: Should I throw away the drumstick on the green plate as well?

Response: Okay.

[Figure 14]

Human Instruction

Robot Response

Embodied Reasoning

Low-level Action

Visual Observation

###### Figure 1 A demo of Robix, showcasing (1) complex instruction understanding with commonsense reasoning; (2) real-time interruption handling; (3) task-status monitoring and dynamic replanning; and (4) proactive dialogue to clarify ambiguous instructions or infer user intent.

low-level controller layer executes the atomic motor actions issued by the high-level layer. This division of responsibilities allows the robot to reason at a macro level while acting at a micro level, enabling human-like adaptability in real-world scenarios.

Existing hierarchical approaches typically employ large language models (LLMs) or vision language models (VLMs) as the high-level cognitive layer for task planning, which decompose long-horizon tasks into executable subtasks for the low-level controller [1, 8, 17, 29, 53, 60, 84, 88]. However, these methods focus solely on task decomposition, overlooking human-robot interaction and embodied reasoning, which are essential for general-purpose robotic systems. Taking one step further, recent work [11] constructs modular pipelines that combine reasoning, planning, and interaction through hand-designed workflows. While workflow-based systems are easy to develop, their inflexibility and brittleness remain notable limitations—rooted primarily in rigid modularization and over-reliance on hand-engineered designs. In this work, we introduce Robix, a unified high-level cognitive layer that seamlessly integrates reasoning, task planning, and natural language interaction within a single model. Unlike modular frameworks, Robix adopts an end-to-end vision–language architecture natively designed for interactive task execution. At its core, Robix leverages chain-of-thought reasoning and formulates interactive task execution as a unified reasoning-action sequence, effectively functioning as the “brain” of a generalist robot system. Figure 1 illustrates Robix in an interactive table-organization task, demonstrating flexible capabilities such as understanding complex instructions, handling real-time interruptions, monitoring task progress, and engaging in proactive dialogue to clarify ambiguous commands or infer user intent.

Modeling such complex interactive task execution within a single VLM is challenging. Although general VLMs have achieved strong performance in digital domains, extending them to physical robots is far more demanding: robots must continuously perceive and act in dynamic environments, interpret ambiguous instructions, adapt to real-time feedback, and make sequential decisions under strict physical and temporal constraints. Addressing this gap requires overcoming two major limitations of existing models: (1) limited embodied reasoning—the ability to ground objects and spatial concepts in the physical world and integrate these signals for adaptive planning and task-centric reasoning [64]; (2) lack of flexible multimodal interaction—hindered both by its inherent complexity and by the scarcity of corresponding training data.

To address these challenges, Robix is trained with a three-stage strategy:

- • Continued pretraining on general VLMs to enhance foundational embodied reasoning capabilities. We curate a large-scale dataset covering various robot-relevant tasks, such as 3D spatial understanding, visual grounding, and task-centric reasoning, enabling the model to strengthen its grounded planning and reasoning abilities.
- • Supervised finetuning to endow the model with complex interactive capabilities. We employ comprehensive data synthesis to incorporate chain-of-thought reasoning and model interactive task execution as a unified reasoning-action sequence. The synthetic data covers a full spectrum of capabilities, including complex instruction understanding, long-horizon planning, task status monitoring, dynamic replanning, real-time interruption handling, and human-robot dialogue.
- • Reinforcement learning to further refine the reasoning ability and strengthen the consistency between reasoning and actions, particularly in long-horizon, interactive tasks.

We comprehensively evaluate Robix on embodied reasoning and interactive task execution. Across 31 benchmarks covering robot-relevant abilities (3D spatial understanding, visual grounding, task-centric reasoning) and general-purpose skills (general VQA, multimodal reasoning), Robix achieves obvious improvements on most robot-relevant tasks while maintaining strong general-purpose performance. On a curated interactive-task benchmark spanning in-distribution and out-of-distribution (OOD) settings and diverse instruction types (multi-stage, constrained, open-ended, invalid, interrupted), Robix consistently outperforms commercial (e.g., GPT-4o, Gemini-2.5-Pro) and open-source (e.g., Qwen2.5-VL, RoboBrain-2.0) baselines; notably, Robix-32B exceeds the strongest baseline, Gemini-2.5-Pro, by 3.0 and 11.8 percentage points in accuracy on the two OOD settings. We further assess five real-world scenarios—table bussing, grocery shopping, checkout packing, tableware organization & shipment, and dietary filtering—using task-progress metrics in a hierarchical robot system under two low-level control modes (human teleoperation and an automatic VLA controller). Across

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Please give me the highest-calorie food and fetch a bottle of drink.

[Figure 21]

[Figure 22]

[Figure 23]

I’m allergic to caffeine.

###### Observation 1 Observation 2 Observation 3 Human Interruption

###### Human Instruction

…

…

… … …

### Robix

[Figure 24]

(High-level Planning & Interaction)

…

…

…

Think & Action Think & Action

Think & Action

Response

Okay, no problem. Got it.

###### VLA Model

(Low-level Control)

- Figure 2 Illustration of the hierarchical robot system. Robix serves as the high-level cognitive layer, interpreting tasks and reasoning over multimodal inputs to generate language responses and action plans. The low-level controller layer executes the corresponding atomic commands, enabling interaction with both humans and the physical environment.

both modes, Robix-32B surpasses Gemini-2.5-Pro by 1.6 and 4.3 percentage points on task progress and markedly outperforms all other baselines by 28.1 ∼ 64.6 percentage points. Our experiments demonstrate that Robix couples strong embodied reasoning with flexible high-level planning and interaction, advancing toward general-purpose embodied intelligence.

We summarize Robix’s main features as follows:

- • Unified model. Robix is a single vision-language model that unifies robot reasoning, task planning, and human-robot interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally in an end-to-end manner.
- • Flexible interaction. Within this unified framework, Robix supports proactive dialogue to clarify ambiguity or infer user intent, real-time interruption handling that seamlessly incorporates feedback, and context-aware commonsense reasoning for complex, open-ended tasks.
- • Robust performance. We assess Robix in two setups: (i) on a curated interactive-task benchmark covering both in- and out-of-distribution scenarios with diverse instruction types, and (ii) across five real-world scenarios in a hierarchical robot system with both human teleoperation and an automatic VLA model as the low-level controller. These evaluations demonstrate that Robix consistently delivers strong performance across all settings.

#### 2 The Robix Model

- Figure 2 illustrates the hierarchical robot system, where Robix serves as the high-level cognitive layer responsible for planning and interaction. The low-level controller—typically implemented as a vision-languageaction (VLA) model—executes the atomic commands generated by Robix, enabling the robot system to directly interact with the physical environment.

At each iteration, Robix directly processes visual observations from robot-mounted cameras and user utterances, selectively producing atomic action commands for the low-level controller and appropriate verbal responses. This iterative reasoning-action loop allows Robix to perform deliberate reasoning and generate contextually grounded behaviors. The sequential decision-making process can be formally modeled as:

P tn,an,rn (o1,u1,t1,a1,r1),...,[(on−i,un−i,tn−i,an−i,rn−i)]Ni=1,on,un . (1)

Here, each step involves predicting the next thought tn, action an, and optional verbal response rn, conditioned on the current observations on, optional user instruction un and the interaction history. These intermediate

###### 3D Spatial Understanding

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

###### Multi-view Correspondence

###### 3D Bounding Box Detection

###### Relative Depth Sorting

###### Absolute Depth Estimation

###### Egomotion Prediction

Q：Where is the position corresponding to <point>(566, 224)</point> in the ﬁrst frame located in the second frame?

Q：Detect every chair inside the image and report each as a 3D bounding box. A：3dbbox: <3dbbox>0.74 3.57 0.53 0.82 0.99 0.61 -1.57 -0.00 0.29</3dbbox>…

Q：Between point1: <point>(737, 621)</ point> and point2: <point>(255, 820)</point>, which one is spatially closer to us?

Q：How is the camera rotating between frame 1 and frame 2? Options: A: Rotate left, B: Rotate right, C: No rotation.

Q: Provide the distance between the observer and the center of monitor (red point) in meters. A: The monitor (red point) is approximately 2.0 meters away from your current position.

A： <point>(608, 236)</point>

A：point2 is closer

A：A: Rotate left

###### Visual Grounding

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### 2D Bounding Box Detection

###### Point Detection

###### Count

###### Visual Prompt

Q：What are the bounding box coordinates of the "a small stool in front of the counter" in this image? A：<|box_start|>(682, 679), (813, 955)<|box_end|>

Q：Mark several points on “Park Ave”. A：<point>(762,304)</point>…<point>(875,265)</ point>

Q：Please count the gloves in the image. A：<point>(x,y)<point>…<point>(x,y)<point>The total number is 31.

Q：What object is enclosed with the green polygon? A：It is the tire of the motorcycle.

###### Embodied Task-centric Reasoning General Multimodal Understanding & Reasoning

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Task Status Veriﬁcation Action Aﬀordance Next Action Prediction GUI Agent VQA

…

Q: Determine whether the task “Pass the held pillow with both hands” can be executed. A: <think>The left arm is currently…</think> No, it cannot be executed in the current state.

Q: Determine whether the task "Pick up the cup with the left arm" has been completed. A: <think>According to the image…</think> The task has been completed.

Q: How to get Chrome on my phone? A: <think>… There is a QR code on the right side

Q: What is the most likely next action? A: <think>The left arm is holding up a bottle with the cap opened…</think> The most likely next action is to close the bottle cap with the right arm.

Q：What is the total volume of the measuring cup? A：1000.

of the screen…</think> Scan the QR code on the right side using your phone to download Chrome.

- Figure 3 Overview of Robix’s pretraining data, curated to strengthen core embodied reasoning capabilities (3D spatial understanding, visual grounding, and task-centric reasoning) while also supporting general multimodal understanding and reasoning. The figure showcases the diversity of the data, establishing a solid foundation for embodied models.

thoughts provide a structured reasoning trace that guides decision-making and enables nuanced, context-aware interaction with humans and the environment. To balance memory usage and maintain inference efficiency under token budget constraints (e.g., 32k context length), we retain only the latest N visual observations as explicit input. The full sequence of prior thoughts and actions is stored in short-term memory, allowing Robix to reason over recent history without exceeding capacity limits.

As shown in Figure 1, Robix unifies the entire interactive task execution process—including instruction understanding, task planning, task status monitoring, real-time user feedback integration, proactive dialogue, and dynamic replanning—through grounded, multi-faceted reasoning. Unlike prior modular frameworks for task planning or human-robot interaction, Robix offers significantly greater flexibility, allowing the robot to adapt its behavior in real time to dynamic environmental changes, thereby achieving human-like adaptability.

#### 3 Training Recipe

We develop Robix-7B and Robix-32B by continually training Qwen2.5-VL-7B and 32B [3] on approximately 200 billion tokens using a three-stage training pipeline. First, we perform continued pretraining to enhance the model’s capabilities in robot-relevant perception and reasoning (Section 3.1). Next, we apply supervised finetuning to model the complex human-robot interaction and long-horizon task planning as a sequential decision-making process grounded in chain-of-thought reasoning (Section 3.2). Finally, we leverage reinforcement learning to further improve the embodied reasoning ability and enhance the alignment between reasoning and action in interactive long-horizon tasks (Section 3.3).

##### 3.1 Continued Pretraining

A foundational capability of general-purpose embodied models is embodied reasoning—the ability to ground objects and spatial concepts in the physical world and integrate these signals for downstream robotic tasks [64]. Our objective is to develop a vision-language model with embodied reasoning at its core, capable of generalizing

across diverse embodied scenarios while maintaining strong foundational multimodal understanding. To support this, we construct a large-scale pretraining corpus comprising 200 billion high-quality and diverse tokens, targeting both robot-relevant and general-purpose multimodal capabilities, as illustrated in Figure 3. In particular, we emphasize 3D spatial understanding, visual grounding, and task-centric reasoning, while also incorporating general visual understanding, multimodal reasoning, and instruction tuning data. Below, we detail the data sources and task types used in our continued pretraining.

- 3D Spatial Understanding. Current VLMs generally lack strong spatial understanding capabilities, which is crucial for embodied scenarios such as navigation and manipulation planning. To equip the model with 3D spatial understanding from 2D images, we curate over 30 million instruction pairs (about 40B tokens) spanning five key task types: (1) Multi-view correspondence – learning 2D point correspondences across stereo or multi-view images of the same scene; (2) 3D bounding box detection – predicting metric 3D bounding boxes from monocular images with open-vocabulary object descriptions; (3) Relative depth sorting – inferring the depth ordering of objects within a single image; (4) Absolute depth estimation – estimating absolute object depth using semantic masks and annotated depth maps; (5) Egomotion prediction – modeling camera motion over time to support temporal and spatial reasoning. The majority of the data is derived from the 3D spatial understanding training corpus of Seed-1.5-VL [24]. Additionally, part of the dataset is constructed from publicly available 3D benchmarks, including ScanNet[12], ScanNet++[79], 3RScan[66], CA-1M[35], SUN RGB-D[61], and ARKitScenes[4]. Integrating these five spatial reasoning tasks effectively improves the model’s spatial awareness in embodied tasks.

Visual Grounding. Visual grounding enables multimodal models to interpret user instructions and locate target objects in images. We use two grounding formats—bounding boxes and center points—and train on four types of data: 2D bounding box annotations, point annotations, counting, and visual prompt. We normalize all coordinate values to the range [0,1000], allowing consistent grounding predictions across varying image resolutions. The dataset comprises over 50 million instruction–response pairs (approximately 70 billion tokens), curated from both open-source resources and internal collections within the Seed-1.5-VL corpus. It encompasses a broad range of tasks, including: (1) 2D bounding box annotations: predict bounding boxes from open-vocabulary descriptions, or generate textual descriptions given bounding box coordinates; (2) Point annotations: predict object center points from descriptions or identify objects based on given coordinates. (3) Counting: derived from bounding box and point data, supporting both box- and point-based counting via a two-stage localization and counting pipeline; (4) Visual Prompt: prompts contain both textual instructions and visual annotations (e.g., points, bounding boxes, arrows), enabling the model to learn multimodal fusion and context-aware understanding grounded in visual cues. Together, these tasks significantly enhance the model’s grounding abilities in both language-to-image and image-to-language directions, and improve its capacity for grounded planning in embodied settings.

Task-centric Reasoning. To directly strengthen the model’s reasoning and planning abilities in embodied scenarios, we construct a large-scale embodied task-centric reasoning dataset based on publicly available robot and egocentric datasets, including AgiBot [6], BridgeData V2 [67], Droid [32], Egodex [27], RoboVQA [52], HoloAssist [70], and Ego4D [22]. We curate over 5 million examples (about 10B tokens) targeting three key reasoning functions: (1) Task Status Verification—determining whether a task or subtask has been successfully completed; (2) Action Affordance—assessing whether an action is feasible in the current context; (3) Next Action Prediction—identifying the most plausible next step to achieve the intended goal. To enrich the reasoning process, we further use Seed-1.5-VL-thinking [24] to generate step-by-step thought traces for our QA pairs via carefully designed prompts (details are shown in Section B). This thought-augmented supervision enables the model to learn deliberate, high-level decision-making in dynamic and open-ended environments.

General Multimodal Reasoning. To enhance the model’s general reasoning capabilities, we curate a diverse set of over 6 million multimodal instruction-image pairs (about 10B tokens) spanning STEM problem solving, agent-based decision making, and visual inference tasks. Specifically, we include: (1) STEM Reasoning Data: Multimodal problem-solving examples in mathematics, physics, chemistry, and biology, combining textual questions with diagrams, equations, and visual content. (2) Multimodal Agent Data: GUI-based agent demonstrations that involve step-by-step planning, error correction, and reflective reasoning. (3) Visual Inference Data: Tasks that require grounded visual reasoning, including spotting differences between paired

images and generating HTML/CSS code from user interface screenshots. Together, these datasets equip the model with robust abstract reasoning and cross-modal problem-solving abilities, supporting its generalization to complex tasks in open environments.

General Multimodal Understanding. To preserve and enhance broad vision-language understanding, we curate a large-scale dataset of over 50 million image–text pairs (over 80B tokens) that serve as the foundation for multimodal comprehension. (1) VQA: A diverse set of image- and video-based question answering tasks covering visual perception, factual knowledge, grounding, temporal reasoning, spatial understanding, and counting. (2) Captioning: Dense captions for both images and videos, supporting the model’s understanding of static scenes and multi-frame temporal dynamics. (3) OCR: To improve text recognition, we include large-scale annotated and synthetic datasets covering scene text, documents, tables, charts, and flowcharts. The dataset is filtered from both open-source resources and the Seed-1.5-VL corpus, together forming a robust foundation for training general-purpose vision–language models.

Instruction Tuning. To further enhance the model’s instruction-following and reasoning abilities, we construct a high-quality instruction tuning dataset comprising 1 million examples. These examples span a wide range of tasks and are built by extracting curated subsets from previously collected data, integrating both general instructions and chain-of-thought examples from open-source and internal sources. We refine the instructions using Seed-1.5-VL [24] for quality filtering and ensure better alignment between instructions, images, and responses. This instruction-tuned dataset significantly improves the model’s ability to follow open-world multimodal instructions and engage in multi-turn, grounded reasoning.

We adopt a two-stage training strategy leveraging the large-scale, diverse corpus described above. In stage 1, we continue pretraining Qwen2.5-VL [3] on the full dataset—comprising approximately 5% text-only data—updating all model parameters to enhance general multimodal and embodied reasoning capabilities. Training follows a full cosine learning rate schedule, starting at 1 × 10−5 and decaying to 1 × 10−6, with linear warm-up over the first 10% of total steps. We use a sequence length of 32,768 tokens, with effective batch sizes of 1536× and 3008× the sequence length for the 7B and 32B models, respectively. In stage 2, we perform instruction tuning on curated instruction-following data to align the model with multimodal prompts and improve instruction adherence. The vision encoder is frozen during this phase, while all other parameters remain trainable. The learning rate is fixed to the final value from Stage 1 (1 × 10−6) and remains constant throughout Stage 2. Optimizer states are carried over from Stage 1, and no additional warm-up is used. Both stages are optimized using AdamW [33, 41], with β1 = 0.9, β2 = 0.99, and a weight decay of 0.01. Training on this diverse and comprehensive corpus significantly improves the model’s embodied reasoning, multimodal understanding, and its ability to generalize to long-horizon, interactive tasks in real-world settings.

##### 3.2 Supervised Finetuning

The supervised fine-tuning (SFT) stage adapts the preceding pretrained model into the robot’s high-level cognitive module while retaining its original capabilities. A central challenge lies in the scarcity of largescale, multi-turn egocentric-vision datasets that integrate human-robot interaction with task planning. To address this, we design a data-synthesis pipeline that transforms existing task-planning datasets into human–robot interaction trajectories. Two properties of the resulting SFT data are crucial for out-ofdistribution generalization: (1) diverse human–robot interactions and (2) high-quality reasoning traces. The overall pipeline is shown in Figure 4; the interaction and reasoning synthesis modules are detailed below.

- 3.2.1 Interaction Synthesis The interaction synthesis is mainly based on two data sources:

- • Teleoperated Robot Demonstrations. We utilize both internal teleoperation data (previously used to train GR-3 [7]) and the open-source AGIBot dataset [6]. Each contains episodes of robots performing diverse tasks (e.g., table bussing, breakfast preparation). Human annotators segment each demonstration into clips, where each clip corresponds to an atomic action—e.g., “put the tissue into the trash bin”.
- • Simulation & AIGC Data. We programmatically generate diverse object organization scenarios in our in-house simulator, and further employ state-of-the-art text-to-image models [21] to synthesize complex

[Figure 39]

[Figure 40]

[Figure 41]

###### Source Data Selection

###### Interaction Synthesis Reasoning Synthesis

Labeled Teleoperation Data

###### Multi-Stage Task Instruction

Anytime Interruption

###### Human-Robot Chatting

Human Instruction

Human: Clear the table and pack up the food.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### The table is a mess! Clean up the table and pack the food.

Besides the plastic box, what else is on the table.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

###### 1

2

Robot Reasoning

I still need the bowl.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

###### Scene Understanding

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

put the paper cup into the rubbish bin

put the knife into the plastic box

I am in front of a table. On the table, I can observe a plate of biscuits, a paper cup…

[Figure 69]

[Figure 70]

[Figure 71]

1. put the spoon into the plastic box

2. put the bowl into the plastic box

3. put the goblet into the plastic box

Sure, I will put it back on the table.

There is a plate of food, a plate and a to-go box.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

3 4

Got it.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

###### Constrained Instruction

Invalid & Illegal Instruction

[Figure 82]

###### Task Status Reﬂection

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Clear the trash and leave the food on the table.

Use the plate to hit the plastic box heavily.

…

put the fork into the plastic box put the glass into the plastic box

[Figure 87]

[Figure 88]

My previous action is to put the tissue into the rubbish bin, but I observe that the tissue is still on the table…

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Simulation & AIGC Data

[Figure 93]

[Figure 94]

4. put the tissue into the rubbish bin 5. put the paper cup into the rubbish bin

[Figure 95]

[Figure 96]

9. pour the food into the to-go box

10. close the to-go box

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Sure. I will only clear the trash.

This action is dangerous! I can’t do it.

The task is done! Is there anything else I can help?

Next-Step Analysis

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Open-Ended Instruction

Ambiguous Instruction

…The bowl is on the plate, I should put the bowl into the plastic box now…

Human Instruction

[Figure 109]

I want a tropical fruit, put one into the carton.

I am hungry, put a snack into the carton.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

I don’t like bread.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Robot Response

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Long-Term Instruction Following

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Labeled Action

The green plate is still on the table, and the user previously required me to keep the plate on the table…

[Figure 128]

Put the banana into the carton.

There are potato chips, bagels, and toast on the table. Which one should I put into the carton?

Sure, I am putting the potato chips into the carton.

Got it. I will put the banana into the carton.

- Figure 4 Our data synthesis pipeline. The leftmost panel shows the data sources; the center illustrates how diverse human-robot interactions are synthesized from these sources; the rightmost panel presents example snapshots of the generated reasoning traces.

scenes with items not yet supported in the simulator. To ensure quality, we apply both model-based and human-in-the-loop filtering to discard low-quality generations.

Building on the above sources, we define seven categories of human–robot interaction instructions and develop tailored data-synthesis strategies for each. Training on this dataset equips Robix with flexible interaction capabilities, such as complex instruction understanding, real-time interruption handling, and proactive dialogue.

Multi-Stage Instruction. Teleoperated trajectories are annotated with a task name (e.g., “clean up the table and pack the food”). We select trajectories containing at least ten atomic actions and synthesize a corresponding user instruction from the task name, e.g., “The dining table is a mess. Please clean the table and pack the food on the plate.”

Constrained Instruction. We partition each teleoperation trajectory into non-overlapping segments where feasible. For example, a table-bussing task can be decomposed into subtasks such as garbage collection, tableware collection, and food packing. Based on these subtasks, we synthesize tailored user instructions, e.g., “Clean up the table while leaving the food on the table” and “Remove the trash without moving other items”. Open-Ended Instruction. We generate random scenes in simulation and prompt state-of-the-art LLMs to produce open-ended, commonsense instructions conditioned on each scene (e.g., “Place the drink with the least sugar into the carton” for a scene containing Sprite, Coke, orange juice, and soda water). To include items currently not supported by the simulator (e.g., hamburgers, spaghetti, watermelons), we pair such instructions with images synthesized by advanced text-to-image models. Because current text-to-image models still struggle with instruction following and image quality, we apply automated and human-in-the-loop filtering, remaining only 10% of the original dataset after filtering.

Anytime Interruption. We curate a variety of user interruption utterances (e.g., “Stop!”, “Hold on. I still need it”, “Wait, put the fork into the sink first”) and randomly inject them into task flows. Corresponding robot responses are then synthesized using timing-aware heuristics: if the interruption occurs before grasping, the robot halts or adjusts its plan; if it occurs after grasping, the robot returns the item to the table and replans. Such anytime interruption handling is critical for long-horizon tasks, as mid-task feedback and error

correction substantially enhance system robustness.

Invalid Instruction. To mitigate hallucination and prevent robots from engaging in dangerous actions, we synthesize four types of invalid instructions as follows: (1) Instructions asking the robot to manipulate items that do not exist in the scenario; (2) Instructions requiring physically impossible actions, such as “Put the table into the rubbish bin”; (3) Instructions demanding abilities beyond the robot’s current capabilities, e.g., “Open the coke for me”; (4) Unsafe or dangerous commands, e.g., “Throw the knife onto the sofa”. For these invalid or illegal instructions, we design corresponding response strategies to enable the robot to refuse compliance with the user’s requests.

Ambiguous Instruction. To enable our model to clarify ambiguous instructions, we construct scenes with multiple similar items (e.g., apple, orange, pear) and synthesize underspecified instructions(e.g., “Put a fruit into the basket”). Training with these data enable the model to seek clarification when needed—a capability essential for robust robot systems.

Chat Instruction. We develop some heuristics to randomly insert short human-robot dialogue segments at context-appropriate times. For example, when the robot is collecting trash from the table, the user may ask “I want some fruit. What kind of fruit is on the table?”. This type of instruction requires the robot to respond verbally rather than performing any physical manipulation.

###### 3.2.2 Reasoning Synthesis

To incorporate chain-of-thought reasoning, we prompt state-of-the-art VLMs to generate high-quality reasoning traces emphasizing (1) scene understanding, (2) task status reflection, (3) long-term instruction following, and (4) next-step analysis.

- • Scene understanding. This part of reasoning enables the robot to accurately identify task-relevant, operable objects in the current scene, with emphasis on those within the robot’s field of view.
- • Task status reflection. Robots should be capable of reflecting on their prior actions and repeating tasks when initial attempts fail. Furthermore, they need to identify key milestones in long-horizon tasks and proactively request human assistance upon encountering irrecoverable errors. This capability is also critical for handling user interruptions, as robots must maintain awareness of their current status to plan subsequent actions (e.g., tracking whether a gripper is holding an item).
- • Long-term instruction following. This module is designed to help robot persist the initial goal and intermediate user instructions across long-horizon tasks, ensuring the primary objective is completed and mid-task instructions continue to guide actions many steps later (e.g., “After cleaning the table, grab me a drink from the fridge”).
- • Next-step analysis. In the final phase of reasoning, the robot should analyze potential actions for the next step when the overall task remains incomplete. This analysis include assessing target reachability and whether executing the action advances overall task completion.

Inspired by UI-TARS [50], we adopt ActRe [78] and Thought Bootstrapping [50] to synthesize high-quality reasoning traces. Unlike conventional LLM reasoning, robot reasoning must remain concise to support real-time interaction. Accordingly, we prompt Seed-1.5-VL to produce succinct traces (within 200 tokens). We also apply a model-based filtering pipeline to discard hallucinated or logically inconsistent reasoning. These high-quality, multi-faceted chain-of-thought traces enable Robix to execute robust long-horizon task planning with task-status monitoring and dynamic replanning.

##### 3.3 Reinforcement Learning

Following the supervised fine-tuning (SFT) stage, the model exhibits promising agentic capabilities in adaptive task planning and natural human-robot interaction. However, several limitations in robot reasoning and planning persist, notably: (1) irrational reasoning, such as generating conflicting thoughts, lacking common sense, or partially disregarding user instructions; and (2) thought-action inconsistency, where the model’s proposed plan diverges from its preceding thought in intent or content. For example, in a table-cleaning task,

the SFT model correctly infers that a tissue left on the table should be discarded in a rubbish bin. Yet, in the subsequent plan, it incorrectly suggests handling a paper cup instead. These issues negatively impact the model’s effectiveness in real-world task execution.

To mitigate these problems, we adopt reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO) [23, 54], to enhance both the reasoning capacity and the coherence between thought and action. Our approach is based on two core strategies: (1) co-training with general visual reasoning data, and (2) reward design targeting thought-action consistency.

Co-Training with General Visual Reasoning Data. The RL stage utilizes two primary data sources: robot interaction data and general visual reasoning datasets. Training on robot interaction data improves the model’s robustness and generalization to out-of-distribution (OOD) scenarios. Meanwhile, incorporating general visual reasoning data strengthens the model’s inherent reasoning capabilities. This co-training strategy helps alleviate irrational reasoning and enhances overall task understanding and solving. The general visual reasoning datasets include a wide range of cognitive challenges, such as task completion verification, action affordance evaluation, and object localization—covering a broad spectrum of reasoning skills relevant to real-world robot applications.

Reward Design for Thought-Action Consistency. To explicitly encourage alignment between the model’s thought and action, we introduce a thought-action consistency reward in addition to standard rewards for output formatting and action accuracy. At each decision step, the model’s generated thought and corresponding action are extracted and evaluated by an external LLM (Qwen-2.5-32B [74] in our experiments). This auxiliary reward model is prompted to assess whether the action is logically consistent with the preceding thought. A negative reward is given if the assessment indicates inconsistency. The system prompt of the reward model is listed in Section A.5.

To maximize the effectiveness of RL training, we also employ a data filtering procedure designed to retain only samples that can provide meaningful gradient information for GRPO. The key idea is to discard questions whose candidate answers exhibit low reward variance, as such samples contribute little to policy improvement. Specifically, for each question in the dataset, we generate multiple candidate answers using the SFT model and remove those with low variance in their rewards:

Dnew = (xn,yn∗) ∈ D Var R(yn(i),yn∗)

M i=1

> τ, yn(i) ∼ πSFT(· | xn) , (2)

where D denotes the original dataset, R(yn(i),yn∗) is the reward function assigning a scalar score to the i-th generated answer yn(i) based on the ground-truth yn∗ for question xn, and πSFT is the base policy for RL. The definitions of input xn and output yn∗ follow Equation (1): xn consists of the current observation, instruction, and trajectory, while yn comprises the model’s thought, an optional action, and an optional robot response. In our experiments, we set the number of samples M to 8 and the variance threshold τ to 0. All RL training is performed using the verl framework [55].

Through the combination of co-training with diverse reasoning data and targeted reward design, our reinforcement learning strategy substantially improves the model’s generalization to novel tasks and enhances the consistency between reasoning and planning.

#### 4 Experiments

We conduct extensive experiments to comprehensively evaluate the performance of Robix, focusing on the following key questions:

- • Does Robix enhance fundamental embodied reasoning capabilities?
- • Can Robix effectively model the full process of interactive task execution in an end-to-end manner?
- • Does Robix generalize well to out-of-distribution tasks?
- • How does the full robotic system perform when integrating Robix with a VLA model on real-world tasks?

Robix 7B Base

Robix 32B Base

Qwen 2.5 VL 7B

Seed 1.5-VL Think 3D Spatial Understanding

Qwen 2.5 VL 32B

Cosmos Reason1 7B

RoboBrain 2.0 32B

Gemini 2.5 Pro

OpenAI GPT 4o

Seed 1.5-VL

Benchmark

VSIBench 44.6 50.9 38.2 39.1 33.9 45.2 43.4∗ 42.5∗ 34.0∗ 39.5∗ BLINK 87.6 83.5 86.5 78.3 69.7 82.4 87.6∗ 80.5∗ 79.0∗ 82.0∗ CV-Bench 86.5 87.4 79.0 81.6 76.0 83.7 85.7∗ 79.7∗ 77.9∗ 82.4∗ EmbSpatial 77.4 79.0 70.1 72.3 64.6 76.6 77.2∗ 70.6∗ 68.0∗ 73.5∗ SAT 71.1 79.6 52.8 74.9 60.7 80.3 79.5∗ 62.4∗ 67.5∗ 72.7∗ VSR 83.3 83.7 82.2 83.6 79.9 84.0 82.5∗ 77.5∗ 78.1∗ 80.1∗ SpatialBench 64.7 65.4 61.7 66.8 61.6 68.8 66.0∗ 63.7∗ 64.0∗ 66.3∗ DA-2k 72.2 77.1 65.0 68.6 65.8 56.9 83.0∗ 78.2∗ 86.5∗ 87.5∗

Visual Grounding

LVIS-MG 70.2 79.2 30.6 54.2 18.2 -† 63.8 -† 73.8 72.5 Refcocoval 89.2 91.5 88.9 89.3 77.1 -† 74.6 -† 92.8∗ 92.6∗ Refcoco+val 83.8 86.2 82.6 83.2 69.2 -† -† -† 89.3∗ 89.7∗ Refcocogval 87.1 89.0 85.6 87.3 75.0 -† -† -† 90.1∗ 90.4∗ Refcocouval 87.1 88.6 85.5 86.9 74.2 -† -† -† 88.9∗ 88.5∗ VisualWebBench 63.2 68.9 59.4 66.1 51.5 20.5 87.3 80.2 88.0 87.3 Pixmo-Point 29.5 47.3 43.5 41.3 6.0 46.0 11.3∗ 10.8∗ 13.7∗ 9.8∗ Where2Place 41.9 45.2 33.0 39.9 11.4 73.6 39.9∗ 26.9∗ 18.4∗ 25.4∗

Embodied Task-centric Reasoning

Agibot-ER 61.0 62.6 48.2 55.4 38.0 54.3 67.1∗ 49.9∗ 63.6∗ 60.5∗ EgoTaskQA 28.6 33.6 23.7 31.3 28.4 31.5 37.0∗ 30.0∗ 27.8∗ 31.1∗ OpenEQA-hm3d 48.8 51.1 42.2 46.8 40.0 48.7 63.8∗ 63.4∗ 58.8∗ 60.8∗ OpenEQA-scannet 58.0 58.9 51.3 52.8 50.9 55.9 74.3∗ 71.3∗ 66.1∗ 68.3∗ ERQA 42.5 43.5 38.0 42.8 39.3 46.0 55.0∗ 47.8∗ 39.8∗ 47.5∗ RoboVQA 53.6 48.3 54.0 60.0 55.4 55.7 33.9∗ 34.5∗ 37.0∗ 35.1∗

Multimodal Understanding & Reasoning MME 2332.8 2427.2 2273.1 2425.5 2150.0 2462.3 2491.3∗ 2271.9∗ 2314.4∗ 2470.8∗ MMbench 87.6 89.1 86.8 89.2 85.1 88.6 90.1 84.3 88.0 89.9 RealworldQA 70.7 69.0 65.9 68.4 67.8 63.9 78.0 76.2 77.0 78.4 SimpleVQA 44.7 49.0 47.7 45.2 42.5 41.9 62.0 50.1 63.1 63.4 EgoSchema 66.2 73.4 67.6 74.0 59.8 70.8 74.1∗ 69.8∗ 64.3∗ 77.2∗ VideoMME 63.7 67.6 65.3 68.8 61.9 66.9 86.9 71.9 77.6 77.9 NextQA 82.5 81.6 82.9 81.8 79.5 78.7 83.5∗ 72.7∗ 62.6∗ 68.6∗ MathVista 68.0 69.6 68.3 75.6 62.1 68.7 82.7 63.8 83.0 85.6 MMMU 51.6 58.9 51.9 57.3 47.0 59.2 81.7 70.7 73.6 77.9

∗ Collected by ourselves via API in July 2025. † Invalid results due to failures in following format requirements.

Table 1 Performance of Robix on public vision-language benchmarks compared to prior models. The left side shows Robix and state-of-the-art open-source baselines, while the right side presents closed-source large commercial models. The highest score in each benchmark is highlighted in bold within each group.

We first evaluate fundamental embodied reasoning capabilities of our model and baseline methods on public benchmarks (Section 4.1). Next, we assess the planning and interaction abilities of our model and baseline methods with both offline, pre-defined test sets (Section 4.2) and online robotic tasks (Section 4.3).

##### 4.1 Fundamental Perception & Reasoning Evaluation

We evaluate Robix after continued pretraining (denoted as Robix-Base) on a comprehensive set of public benchmarks against state-of-the-art multimodal models including Qwen-2.5-VL-7B&32B [3], RoboBrain-2.032B [63], Cosmos-Reason1-7B [2], Gemini-2.5-Pro [64], OpenAI GPT-4o [30], Seed-1.5-VL and Seed-1.5-VLThink [24]. The evaluation spans (1) robotics-relevant embodied reasoning (3D spatial understanding, visual grounding, task-centric reasoning) and (2) general multimodal understanding and reasoning. Table 1 presents the detailed results.

- 3D Spatial Understanding. We evaluate Robix’s 3D spatial understanding across 8 spatial reasoning benchmarks: VSIBench [75], BLINK [20], CV-Bench [65], EmbSpatial [16], SAT [51], VSR [37], SpatialBench [16], and DA-2k [76]. Details of these benchmarks are provided in Section A.1. As shown in Table 1, Robix-7B and Robix-32B outperform their backbones (Qwen2.5-VL-7B/32B) on 7 of 8 spatial reasoning tasks, with average accuracies of 73.4 and 75.8 compared to 66.9 and 70.7, respectively. They also surpass embodied

models Cosmos-Reason1-7B (64.0) and RoboBrain-32B (72.2), and exceed the best commercial baseline, Gemini-2.5-Pro, on 5 of 8 tasks.

Visual Grounding. We evaluate Robix’s visual grounding capabilities on eight benchmarks covering both bounding-box and center-point tasks, including LVIS-MG [25], RefCOCO [80], VisualWebBench [39], PixmoPoint [14], and Where2Place [81]. Robix consistently outperforms its backbone across all benchmarks and surpasses state-of-the-art commercial models on most tasks. Notably, Robix-7B and Robix-32B improve the absolute F1 score on LVIS-MG by 39.6 and 25.0 points over Qwen2.5-VL-7B and 32B, respectively. Robix-32B also outperforms commercial models on most tasks. These results highlight Robix’s strong performance in object localization, pointing and fine-grained visual understanding.

Task-centric Reasoning. Embodied reasoning reflects a model’s ability to understand and reason about robotic tasks. We evaluate Robix across 5 diverse open benchmarks, including ERQA [34], RoboVQA [52], OpenEQA (HM3D & ScanNet) [43], and EgoTaskQA [31]. In addition, we introduce Agibot-ER, a real-world task reasoning benchmark derived from the Agibot dataset [6], which includes manually annotated test sets of 97, 120, and 381 samples for the three key reasoning tasks—Task Status Verification, Action Affordance, and Next Action Prediction, respectively. Full details of the benchmark are provided in Section B. We report the average results of the three tasks on this benchmark. Robix consistently outperforms its backbone models as well as Cosmos-Reason1-7B and RoboBrain-2.0-32B across most benchmarks. On Agibot-ER, Robix delivers substantial gains over its backbones, improving absolute accuracy by 12.8 and 7.2 points for the 7B and 32B versions, respectively. It further surpasses Cosmos-Reason1-7B and RoboBrain-2.0-32B by 23 and 8.3 points, demonstrating superior performance in embodied, task-centric reasoning.

General Multimodal Understanding & Reasoning Multimodal understanding is a core capability of vision–language models and a primary focus of VLM development. To evaluate both static image and dynamic video understanding, we assess Robix on a suite of general VQA benchmarks—image-based (MME [18], MMBench [40], RealWorldQA [71], SimpleVQA [9]) and video-based (EgoSchema [44], VideoMME [19], NextQA [73]). We further test general reasoning on MathVista [42] and MMMU [82], which cover complex mathematical and multimodal problem-solving tasks. Robix preserves the performance of its backbone on most benchmarks, demonstrating the benefit of training with diverse, high-coverage multimodal data, but still trails large-scale commercial models—underscoring the need to scale both data and model size for stronger general-purpose multimodal reasoning.

Overall, Robix greatly enhances robotics-relevant perception and reasoning—particularly in 3D spatial understanding and visual grounding—while maintaining strong performance on general multimodal tasks. These gains deepen its understanding of spatial and temporal properties, enabling more effective reasoning and planning in real-world environments.

##### 4.2 Offline Evaluation

The offline evaluation enables fully automated assessment of planning and interaction capabilities using predefined evaluation sets. To thoroughly evaluate both interactive long-horizon planning and out-ofdistribution (OOD) generalization, we design three dedicated evaluation sets:

- • AGIBot Evaluation Set. We manually select 16 high-frequency daily tasks from the AGIBot dataset (e.g., making a sandwich, washing dishes with a dishwasher, arranging a sofa, washing clothes with a washing machine, arranging flowers) and ensure none appear in the training data. This set primarily evaluates the model’s long-horizon task planning capability on OOD tasks. Details are provided in Section A.2.
- • Internal Out-of-Distribution (OOD) Benchmark. We manually design 16 scripts covering task planning and diverse human–robot interaction scenarios, including table organization, dietary filtering, checkout packing, grocery shopping, and shoe cabinet organization. These scripts are enacted by human participants—one acting as the user and the other executing actions via robot teleoperation or a Universal Manipulation Interface (UMI) [10] device—and subsequently annotated by trained annotators. The benchmark includes tasks and items absent from the training data and is intended to evaluate interactive task execution in unseen scenarios.

Internal ID

Internal OOD

AGIBot

Multi-Stage Constrained Interrupt Open-Ended Invalid Replan

# episodes 16 16 9 25 – 15 – – # data 142 225 119 233 110 15 60 100

Table 2 Statistics of the offline evaluation sets.

Internal ID

Internal OOD

AGIBot

Multi. Const. Interrupt Open. Invalid Replan Plan Accuracy F1 score F1 score

Gemini-2.5-Pro 52.6 83.8 79.3 87.1 55.9 60 98.3 83.7 GPT4-o 45.9 77.0 76.1 84.4 44.8 66.7 79.2 73.7 Seed-1.5-VL 37.4 73.2 75.4 76.0 41.1 46.7 100 36.8 Seed-1.5-VL-Think 49.6 80.4 73.9 82.7 42.9 46.7 74.2 75.1 Qwen-2.5-VL-72B 36.7 69.2 71.3 65.1 55.2 26.7 87.0 40.0 Qwen-2.5-VL-32B 43.3 71.6 60.5 62.2 48.0 26.7 70.2 37.0 Qwen-2.5-VL-7B 31.1 54.7 37.5 41.5 20.5 6.7 47.5 10.9 GLM-4.1-9B-Think 34.1 51.7 22.8 45.8 14.0 6.7 86.0 37.2 RoboBrain-2.0-32B 29.6 63.5 58.2 51.7 41.2 0.0 43.6 29.9 RoboBrain-2.0-7B 0.3 31.4 36.0 33.1 25.3 0.0 0.0 22.2

Robix-7B-SFT-wo-R 55.2 69.9 82.5 89.0 91.5 60.0 100 90.5 Robix-7B-SFT 57.8 77.1 85.8 91.1 84.2 86.7 100 88.4 Robix-7B-RL 59.6 85.4 93.2 90.3 78.6 86.7 95.9 87.0 Robix-32B-SFT 64.0 83.5 89.3 93.0 89.7 80.0 100 95.1 Robix-32B-RL 64.4 86.8 96.6 96.0 92.5 93.3 100 96.2

Table 3 Offline evaluation results. Robix-7B-SFT-wo-R refers to our SFT model without chain-of-thought reasoning, while Robix-7B-RL denotes the full trained policy obtained by applying RL after SFT. For AGIBot, Internal OOD (Out-of-Distribution), and Internal ID (In Distribution)–MultiStage/Constrained/Interrupt/OpenEnded, we report plan accuracy; for Internal ID–Invalid/Replan, we report F1 score. The best result for each evaluation set is shown in bold, and the best among baselines is underlined.

• Internal In-Distribution (ID) Benchmark. This evaluation set is randomly sampled from our synthesized data and categorized by task type and user instruction into six groups: (1) multi-stage instructions, (2) constrained instructions, (3) invalid instructions, (4) user interruptions, (5) fail-and-replan, and (6) openended instructions. Each category targets evaluation of the model’s corresponding instruction following and task planning capabilities.

Data Format. The overall statistics of each evaluation set are shown in Table 2. Each episode is structured as a multi-turn dialogue and evaluated using a teacher-forcing approach, i.e., the model observes an error-free interaction and planning history when predicting the next step action. Because an observation–instruction pair may permit multiple valid next actions, we annotate a candidate action list for each step to capture all acceptable options. Examples of the offline evaluation format are provided in Section A.3.

Evaluation metrics. For AGIBot, Internal OOD, Internal ID–MultiStage/Constrained/Interrupt/OpenEnded, we report action prediction accuracy by matching the predicted action against a candidate action list, with similarity judged by Seed-1.5-VL (see Section A.5 for the prompt). For Internal ID–Invalid/Replan, which are binary classification tasks, we report the F1 score.

Baseline methods. We compare against widely used commercial and open-source VLMs, including Gemini-2.5Pro, GPT-4o, Seed-1.5-VL, Seed-1.5-VL-Think, Qwen2.5-VL-7B/32B/72B, GLM-4.1V-9B-Thinking [26], and RoboBrain-2.0-7B/32B. All baselines are adapted to the multi-turn observation–think–action format using the prompts in Section A.5. For each model, we test both English and Chinese prompts and report the better result. Gemini-2.5-Pro and GPT-4o perform better with English prompts, whereas the other models achieve

[Figure 129]

Figure 5 Online evaluation results with a human labeler operating a UMI device as the low-level controller.

higher accuracy with Chinese prompts. All evaluations are conducted using greedy decoding. Results. The offline evaluation results for each model are presented in Table 3. Key observations include:

- • Robix-32B-RL ranks first on all evaluation sets, demonstrating strong task planning and human–robot interaction capabilities, and substantially outperforming all open-source and commercial VLMs on both ID and OOD benchmarks.
- • Chain-of-thought reasoning is critical for both OOD generalization and complex instruction following. Robix-7B-SFT without reasoning (Robix-7B-SFT-wo-R) exhibits a drop of over 7 percentage points in accuracy on the Internal OOD benchmarks compared to its reasoning-enabled counterpart, and suffers a 26.7-point decline on the ID–OpenEnded tasks.
- • RL is critical, boosting Robix-32B’s performance on nearly all evaluation sets. On the challenging Internal OOD benchmarks, Robix-7B-RL and Robix-32B-RL improve accuracy by 8.3 and 3.3 points, respectively, compared to their SFT counterparts. As shown in the case study (Section C), RL primarily enhances the SFT models by (i) reducing irrational reasoning steps, (ii) improving thought-action consistency, and (iii) minimizing formatting errors.
- • Gemini-2.5-Pro is the strongest baseline, ranking first on most evaluation sets among baseline methods. Our evaluation suggests it is currently the leading foundation model for embodied AI applications.

##### 4.3 Online Evaluation

While offline evaluation is cost-effective, it is limited to static environments and cannot assess a model’s ability to interact with the dynamic physical world. To address this, we deploy our model and baselines within a hierarchical robot system across diverse real-world settings—including kitchens, meeting rooms, and grocery stores—and conduct online evaluations to measure their effectiveness as high-level planning and interaction modules for daily tasks. We design two sets of experiments:

- • Online evaluation of VLMs. Assess the planning and interaction capabilities of VLMs in isolation, without the influence of low-level controllers.
- • Online evaluation of the VLM-VLA robot system. Assess the end-to-end system performance by pairing the VLM with an automatic VLA model as the low-level controller.

In the first set of experiments, VLMs serve as the high-level planning and interaction module, while human labelers equipped with a Universal Manipulation Interface (UMI) [10] device act as the low-level controller,

[Figure 130]

Figure 6 Online evaluation on the ByteMini robot with GR-3 model as the low-level controller.

enabling evaluation under a fully reliable control setting. In the second set, we use our in-house VLA model GR-3 [7], as the low-level controller and deploy the integrated VLM–VLA system on the ByteMini robot [7]. Robix is deployed with customized inference optimization techniques [86, 87] to reduce response latency.

###### 4.3.1 Online Evaluation of VLMs

Experimental settings. We designed five tasks—table bussing, checkout packing, dietary filtering, grocery shopping, and tableware organization & delivery—spanning diverse environments such as kitchens, meeting rooms, and grocery stores. To increase realism, some tasks deliberately incorporate user interruptions as well as fail-and-replan scenarios. For details on the initial states and brief descriptions of each task, please refer to Section A.4. We further annotated each task with subtasks to assess completion (e.g., for table bussing: “tissue is in the rubbish bin”, “plate is in the basket”). We compare Robix-32B 1 with four baselines that performed well in the offline evaluation: Gemini-2.5-Pro, GPT-4o, Seed-1.5-VL-Think, and Qwen2.5-VL-32B. To reduce experimental variability, each task–model pair is repeated four times, and we report the average results. Following [57], we use task progress—the percentage of subtasks completed by the end of the task—as the evaluation metric. Trained human annotators assess task progress to ensure reliability and consistency.

Results. The online evaluation results are shown in Figure 5. Both Robix-32B and Gemini-2.5-Pro rank first in 3 of the 5 tasks, with Robix-32B achieving a slightly higher average task progress (92.6% vs. 91%), demonstrating its superior performance in dynamic real-world environments. Robix-32B also outperforms Qwen2.5-VL-32B by a large margin (92.6% vs. 28%), underscoring the effectiveness of our training pipeline. Gemini-2.5-Pro remains the strongest baseline, showing strong capabilities in following complex instructions. However, deploying large foundation models directly for planning and interaction introduces a major challenge—high response latency. In our experiments, Gemini-2.5-Pro sometimes required over 30 seconds to respond. While customized deployment may help reduce latency, we contend that current large-scale commercial VLMs remain too computationally heavy for real-time interaction, even on advanced hardware.

###### 4.3.2 Online Evaluation of the VLM-VLA system

Experimental Settings. We select three tasks from the online evaluation—table cleaning, dietary filtering, and checkout packing—as the evaluation set, excluding the remaining two tasks that require actions beyond GR-3’s current capabilities. To better isolate the high-level cognitive layer’s performance, we also remove particularly challenging items to reduce frequent manipulation failures. Following the VLM online evaluation protocol, each task–model pair is evaluated four times, and we report average results using task progress as the metric. All experiments are conducted with the GR-3 model and the ByteMini robot [7].

Results. We compare Robix-32B with the two strongest baselines from both offline and online evaluations:

1By default, Robix-32B refers to Robix-32B-RL.

Gemini-2.5-Pro and GPT-4o. Figure 6 shows the results across the three real-world tasks. The findings mirror those in Section 4.3.1: Robix-32B achieves an average task progress of 92.5%, exceeding Gemini-2.5-Pro and GPT-4o by 4.3 and 28.1 percentage points, respectively. We find that baseline methods—particularly GPT-4o—sometimes generate actions that are semantically correct but unrecognizable to the VLA. For instance, the VLA can execute “put the Oreo into the shopping basket” but fails to interpret “put the biscuit box into the shopping basket.” Such VLM–VLA misalignment mainly accounts for the online performance drop observed in Gemini-2.5-Pro and GPT-4o.

#### 5 Related Work

Robotic Task Planning. Solving complex, long-horizon tasks in open environments demands robust high-level planning. Vision-Language Models (VLMs) have advanced robotic task planning by grounding high-level instructions in perceptual context [47, 68, 77]. Unlike Large Language Models (LLMs), which often generate ungrounded or physically infeasible plans due to a lack of environmental perception [29, 60], VLMs integrate visual understanding with language reasoning to enable open-vocabulary instruction following and closed-loop planning. Systems such as COME-robot [88], VILA [28], and REPLAN [59] leverage GPT-4V to generate executable plans directly from raw visual observations and iteratively refine them based on environmental feedback, improving robustness through situated reasoning and failure recovery. Despite these advances, VLM-based approaches face persistent challenges: they struggle to maintain long-term consistency, exhibit limited embodied reasoning for grounding objects and spatial concepts in the physical world, and fail to fully integrate these signals for adaptive, task-centric planning. Addressing these issues is essential for scaling VLM-based planning to real-world, long-horizon embodied tasks. Moreover, most existing methods focus solely on task planning while overlooking the human–robot interaction capabilities during task execution that are crucial for a truly generalist robotic system.

Human-Robot Interaction. Existing work on human–robot interaction primarily focuses on enabling seamless, natural communication through real-time feedback and corrections. Early model-based systems grounded language in symbolic environment representations [45, 48, 49, 62], whereas recent learning-based methods adopt hierarchical architectures to directly interpret and act on user feedback [5, 13, 15, 38, 46, 56, 58, 72]. Examples include OLAF [38], which uses GPT-4 to re-label actions and update visuomotor policies from corrections; YAY Robot [56], which integrates feedback into an iterative training loop but is limited by prompt diversity; RT-H [5], which supports language-based intervention but restricts corrections to fixed spatial moves; and RACER [13], which combines a VLM supervisor with physics simulation for recovery guidance. Hi Robot [57] advances these approaches by grounding real-time corrections in the robot’s own observations, enabling interpretation and execution of complex instructions beyond prior systems’ capabilities. However, achieving flexible interaction alongside adaptive task planning requires strong reasoning capacity—a challenge Robix addresses by leveraging chain-of-thought reasoning to unify complex task planning and human–robot interaction within a single model. Robix further introduces novel interaction capabilities, including proactive dialogue to clarify ambiguous instructions or infer user intent, and context-aware commonsense reasoning.

Embodied Reasoning. Embodied reasoning is the capacity of vision–language models (VLMs) to ground objects, spatial concepts, and physical interactions in the real world, and to integrate these signals into downstream robotic tasks [64]. Unlike abstract symbolic reasoning, it is inherently action-oriented, requiring agents to interpret dynamic environments, plan context-aware behaviors, and adapt through feedback. Recent advances span model design, data curation, and task-specific optimization. Embodied-Reasoner [85] learns observation–thought–action trajectories enriched with spatial reasoning and self-reflection for visual search. Gemini Robotics-ER [64] embeds embodied reasoning into its core VLM, achieving strong generalization across tasks such as 3D perception, pointing, state estimation, and affordance prediction. Data-driven approaches include Cosmos-Reason1 [2], which curates datasets emphasizing task-centric reasoning, and RoboBrain-2.0 [63], which synthesizes spatial–temporal reasoning datasets augmented with thought traces for causal chain learning. Task-specific methods include EvolveNav [36], which improves vision–language navigation via formalized CoT fine-tuning and self-reflective post-training; and ECoT [83], which trains vision–language–action models for multi-step reasoning over plans, sub-tasks, motions, and grounded visual features before action generation. However, effectively leveraging embodied reasoning to develop generalist robotic systems capable of interactive,

long-horizon task execution remains underexplored. Robix addresses this gap by integrating robot reasoning, task planning, and natural language interaction to enable seamless interaction with both humans and physical environments, advancing toward general-purpose embodied intelligence.

#### 6 Conclusion

This paper presents Robix, a unified vision-language model that integrates robot reasoning, adaptive task planning, and human-robot interaction. Serving as the high-level cognitive layer of a hierarchical robot system, Robix enables robots to execute interactive, long-horizon tasks in open environments with high flexibility. It demonstrates flexible interaction capabilities, including proactive dialogue to clarify ambiguous instructions or infer user intent, real-time interruption handling, and context-aware commonsense reasoning. Experimental results show that Robix delivers strong performance on real-world robotic tasks and exhibits robust generalization in out-of-distribution settings.

Limitations & Future Work. Similar to other state-of-the-art multimodal models, Robix has several limitations. In highly dynamic tasks with frequent scene transitions, it may produce hallucinations, flawed reasoning, or exhibit gaps in physical commonsense. Additionally, Robix relies on short-term context windows to process interaction history, functioning as a form of short-term memory. Long-term interactive scenarios, however, require more advanced memory mechanisms—specifically, long-term memory with dynamic updates, efficient retrieval, and effective utilization, akin to context engineering in large language models. Addressing these challenges will be a primary focus of future work.

#### 7 Acknowledgements

We thank Wanli Peng, Yongyu Yan, and Tingshuai Yan for their assistance with model deployment and inference optimization. We are also grateful to Baifeng Xie, Lihao Liu, and Yangang Zhang for their support in utilizing the internal simulation platform, and to Xiao Ma for his valuable suggestions on the writing of this paper. We further thank the GR-3 team for providing the teleoperation data, GR-3 model, and ByteMini robot used in our experiments. Finally, we thank the Seed-1.5-VL team for their support with data resources.

#### References

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

- [2] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.

- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, and Elad Shulman. ARKitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.

- [5] Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, et al. Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823, 2024.

- [6] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

- [7] Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

- [8] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024.

- [9] Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. arXiv preprint arXiv:2502.13059, 2025.

- [10] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems (RSS), 2024.

- [11] Simone Colombani, Dimitri Ognibene, and Giuseppe Boccignone. One to rule them all: natural language to bind communication, perception and action. arXiv preprint arXiv:2411.15033, 2024.

- [12] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

- [13] Yinpei Dai, Jayjun Lee, Nima Fazeli, and Joyce Chai. Racer: Rich language-guided failure recovery policies for imitation learning. arXiv preprint arXiv:2409.14674, 2024.

- [14] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv e-prints, pages arXiv–2409, 2024.

- [15] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. 2023.
- [16] Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2024.

- [17] Jiafei Duan, Wentao Yuan, Wilbert Pumacay, Yi Ru Wang, Kiana Ehsani, Dieter Fox, and Ranjay Krishna. Manipulate-anything: Automating real-world robots using vision-language models. arXiv preprint arXiv:2406.18915, 2024.

- [18] Chaoyou Fu, Yi-Fan Zhang, Shukang Yin, Bo Li, Xinyu Fang, Sirui Zhao, Haodong Duan, Xing Sun, Ziwei Liu, Liang Wang, et al. Mme-survey: A comprehensive survey on evaluation of multimodal llms. arXiv preprint arXiv:2411.15296, 2024.

- [19] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025.

- [20] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.

- [21] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Xin Xia, Xuefeng Xiao, Linjie Yang, Zhonghua Zhai, Xinyu Zhang, Qi Zhang, Yuwei Zhang, Shijia Zhao, Jianchao Yang, and Weilin Huang. Seedream 2.0: A native chinese-english bilingual image generation foundation model, 2025.
- [22] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.

- [23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [24] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

- [25] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019.

- [26] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

- [27] Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

- [28] Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, and Yang Gao. Look before you leap: Unveiling the power of gpt-4v in robotic vision-language planning. arXiv preprint arXiv:2311.17842, 2023.

- [29] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

- [30] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [31] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. Advances in Neural Information Processing Systems, 35:3343–3360, 2022.

- [32] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

- [33] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

- [34] Anastasia Kirillova, Eugene Lyapustin, Anastasia Antsiferova, and Dmitry Vatolin. Erqa: Edge-restoration quality assessment for video super-resolution. arXiv preprint arXiv:2110.09992, 2021.

- [35] Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22225–22233, 2025.

- [36] Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, et al. Evolvenav: Self-improving embodied reasoning for llm-based vision-language navigation. arXiv preprint arXiv:2506.01551, 2025.

- [37] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023.

- [38] Huihan Liu, Alice Chen, Yuke Zhu, Adith Swaminathan, Andrey Kolobov, and Ching-An Cheng. Interactive robot learning from verbal correction. arXiv preprint arXiv:2310.17555, 2023.

- [39] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? arXiv preprint arXiv:2404.05955, 2024.

- [40] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

- [41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

- [42] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [43] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16488–16498, 2024.

- [44] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023.

- [45] Cynthia Matuszek, Evan Herbst, Luke Zettlemoyer, and Dieter Fox. Learning to parse natural language commands to a robot control system. In Experimental robotics: the 13th international symposium on experimental robotics, pages 403–415. Springer, 2013.

- [46] Sabrina McCallum, Max Taylor-Davies, Stefano Albrecht, and Alessandro Suglia. Is feedback all you need? leveraging natural language feedback in goal-conditioned rl. In NeurIPS 2023 Workshop on Goal-Conditioned Reinforcement Learning, 2023.

- [47] Aoran Mei, Guo-Niu Zhu, Huaxiang Zhang, and Zhongxue Gan. Replanvlm: Replanning robotic tasks with visual language models. IEEE Robotics and Automation Letters, 2024.

- [48] K Namasivayam, Himanshu Singh, Vishal Bindal, Arnav Tuli, Vishwajeet Agrawal, Rahul Jain, Parag Singla, and Rohan Paul. Learning neuro-symbolic programs for language guided robot manipulation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 7973–7980. IEEE, 2023.

- [49] Siddharth Patki, Andrea F Daniele, Matthew R Walter, and Thomas M Howard. Inferring compact representations for efficient natural language understanding of robot instructions. In 2019 International Conference on Robotics and Automation (ICRA), pages 6926–6933. IEEE, 2019.

- [50] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

- [51] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, et al. Sat: Dynamic spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024.

- [52] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, et al. Robovqa: Multimodal long-horizon reasoning for robotics. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 645–652. IEEE, 2024.

- [53] Rutav Shah, Albert Yu, Yifeng Zhu, Yuke Zhu, and Roberto Martín-Martín. Bumble: Unifying reasoning and acting with vision-language models for building-wide mobile manipulation. arXiv preprint arXiv:2410.06237, 2024.

- [54] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [55] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

- [56] Lucy Xiaoyang Shi, Zheyuan Hu, Tony Z Zhao, Archit Sharma, Karl Pertsch, et al. Yell at your robot: Improving on-the-fly from language corrections. arXiv preprint arXiv:2403.12910, 2024.

- [57] Lucy Xiaoyang Shi, brian ichter, Michael Robert Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, and Chelsea Finn. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. In International Conference on Machine Learning, 2025.

- [58] Utsav Singh, Pramit Bhattacharyya, and Vinay P Namboodiri. Lgr2: Language guided reward relabeling for accelerating hierarchical reinforcement learning. arXiv preprint arXiv:2406.05881, 2024.

- [59] Marta Skreta, Zihan Zhou, Jia Lin Yuan, Kourosh Darvish, Alán Aspuru-Guzik, and Animesh Garg. Replan: Robotic replanning with perception and language models. arXiv preprint arXiv:2401.04157, 2024.

- [60] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2998–3009, 2023.

- [61] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015.

- [62] Agnes Swadzba, Sven Wachsmuth, Constanze Vorwerg, and Gert Rickheit. A computational model for the alignment of hierarchical scene representations in human-robot interaction. In IJCAI, pages 1857–1863, 2009.

- [63] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, et al. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025.

- [64] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

- [65] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

- [66] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. Rio: 3d object instance re-localization in changing indoor environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7658–7667, 2019.

- [67] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

- [68] Beichen Wang, Juexiao Zhang, Shuwen Dong, Irving Fang, and Chen Feng. Vlm see, robot do: Human demo video to robot action plan via vision language model. arXiv preprint arXiv:2410.08792, 2024.

- [69] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.

- [70] Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, et al. Holoassist: an egocentric human interaction dataset for interactive ai assistants in the real world. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20270–20281, 2023.

- [71] xAI. Realworldqa: A benchmark for real-world spatial understanding. 2024. URL https://huggingface.co/ datasets/xai-org/RealworldQA.
- [72] Anxing Xiao, Nuwan Janaka, Tianrun Hu, Anshul Gupta, Kaixin Li, Cunjun Yu, and David Hsu. Robi butler: Remote multimodal interactions with household robot assistant. arXiv e-prints, pages arXiv–2409, 2024.

- [73] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021.

- [74] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [75] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.

- [76] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

- [77] Zhutian Yang, Caelan Garrett, Dieter Fox, Tomás Lozano-Pérez, and Leslie Pack Kaelbling. Guiding long-horizon task and motion planning with vision language models. arXiv preprint arXiv:2410.02193, 2024.

- [78] Zonghan Yang, Peng Li, Ming Yan, Ji Zhang, Fei Huang, and Yang Liu. React meets actre: Autonomous annotation of agent trajectories for contrastive self-training. In First Conference on Language Modeling, 2024.

- [79] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

- [80] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pages 69–85. Springer, 2016.

- [81] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024.

- [82] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- [83] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

- [84] Jianke Zhang, Yanjiang Guo, Xiaoyu Chen, Yen-Jen Wang, Yucheng Hu, Chengming Shi, and Jianyu Chen. Hirt: Enhancing robotic control with hierarchical robot transformers. arXiv preprint arXiv:2410.05273, 2024.

- [85] Wenqi Zhang, Mengna Wang, Gangao Liu, Xu Huixin, Yiwei Jiang, Yongliang Shen, Guiyang Hou, Zhe Zheng, Hang Zhang, Xin Li, et al. Embodied-reasoner: Synergizing visual search, reasoning, and action for embodied interactive tasks. arXiv preprint arXiv:2503.21696, 2025.

- [86] Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang, Tianqi Li, Haojie Duanmu, Renze Chen, Ruifan Xu, et al. Triton-distributed: Programming overlapping kernels on distributed ai systems with the triton compiler, 2025. URL https://arxiv.org/abs/2504.19442.
- [87] Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, et al. Tilelink: Generating efficient compute-communication overlapping kernels using tile-centric primitives. arXiv preprint arXiv:2503.20313, 2025.

- [88] Peiyuan Zhi, Zhiyuan Zhang, Yu Zhao, Muzhi Han, Zeyu Zhang, et al. Closed-loop open-vocabulary mobile manipulation with gpt-4v. arXiv preprint arXiv:2404.10220, 2024.

## Appendix

#### A Experiment details

##### A.1 Fundamental Evaluation Prompts

The prompt templates for all benchmarks are listed below. In each template, {question} is replaced with the sample’s actual question, {options} with its multiple-choice answer options, <image> with the computed ViT embeddings of the input image, and <video> with the ViT embeddings of the video frames (for video benchmarks).

VSIBench. We use the official metric of VSIBench [75].

<video> {question}

Options: {options} Answer with the option’s letter from the given choices directly.

BLINK. Following RoboBrain 2.0 [63], we evaluate BLINK [20] on two sub-tasks—spatial relation understanding and depth perception—and report the average accuracy across them.

<image> {question} {options} Answer with the option’s letter from the given choices directly.

CV-Bench. We follow the official evaluation protocol of CV-Bench [65] and report average accuracy over the three defined sub-tasks.

<image> {question} {options} Answer with the option’s letter from the given choices directly.

EmbSpatial. We use the official metric of EmbSpatial-Bench [16] and report the average accuracy.

<image> {question}

Please answer with a single option letter: {options}

SAT. We use the official metric of SAT [51] and report the average accuracy across its five sub-tasks.

<image> {question} {options}

Please only answer with the option letter.

VSR. We use the official metric of VSR [37] and report the average accuracy.

<image> {question} Answer with a single word yes or no.

SpatialBench. We use the official metric of SpatialBench [16].

<image> {question}

Please answer with a single word: yes or no (for existence task).

DA-2k. Since some VLMs tend to repeatedly output the same option on DA-2k [76], we shuffle the answer choices and report the average accuracy.

<image> {question}

Only provide the answer: point1 or point2.

LVIS. We report results on LVIS [25] using the F1-score 2.

<image> Ground all {object} in this image. Please provide all bounding box coordinates in the format: <|box_start|>(x1, y1),(x2, y2)<|box_end|>.

Each bounding box should contain:

- - (x1, y1): the coordinates of the upper-left corner
- - (x2, y2): the coordinates of the lower-right corner All coordinates must be **normalized to the range [0, 1000]**, where:
- - x refers to the horizontal axis (image width)
- - y refers to the vertical axis (image height) If **no object is detected**, simply output: **No instance found.**

###### RefCOCO & RefCOCO+ & RefCOCOg & RefCOCOu. We use the official metric of RefCOCO [80].

<image> What are the coordinates of the {object} in this image? Please provide the bounding box coordinates of the {object} in the format: <|box_start|>(x1, y1),(x2, y2)<|box_end|>, where (x1, y1) for upper-left, and (x2, y2) for lower-right. All coordinates should be normalized to a [0, 1000] scale, where x corresponds to the horizontal axis (image width), y corresponds to the vertical axis (image height).

Pixmo-Point & Where2Place. For these two benchmarks [14, 81], we compute the proportion of predicted points that fall within the referring objects.

<image> Locate several points for the {object}. You can mark them using <point>x y</point>. Please provide point coordinates in the format: <point>(x1, y1)</point> ... The coordinates of the point (x, y) must be **normalized to the range [0, 1000]**, where:

- x refers to the horizontal axis (image width) - y refers to the vertical axis (image height)

2Since the baseline models (e.g., Qwen-2.5-VL, Cosmos-Reason1, RoboBrain-2.0, etc.) cannot follow a unified output format on the visual grounding benchmarks—including LVIS, RefCOCO, Pixmo-Point, and Where2Place—we revise their prompts to adapt to their output formats.

VisualWebBench. We use the official metric of VisualWebBench [39].

<image> {question}

ERQA. We report accuracy on ERQA [64].

<image>...<image> {question} {options}

Please answer directly with only the letter of the correct option and nothing else.

RoboVQA & EgoTaskQA & OpenEQA-hm3d & OpenEQA-scannet. We compute accuracy by using GPT-4o to compare model responses with the ground-truth labels, and the evaluation prompt is shown below.

<video> {question}

Prompt for GPT-4o (RoboVQA & EgoTaskQA).

You are an AI assistant tasked with evaluating whether a response matches the correct answer to a given question.

Evaluation Rules

- (1) Output 1 if the response matches the answer exactly or with synonymous/equivalent wording.

- - Synonyms, paraphrases, or different surface forms of the same meaning count as matches.
- - Minor wording differences (e.g., “put tomato into fridge” vs. “the person is putting a tomato in the fridge”) count as matches.

- (2) Output 0 if the response is incorrect, contradictory, or refers to a different entity, object, or attribute.

- - If the answer and response describe different objects, actions, or states, mark as 0.
- - If the response introduces additional details that change the meaning of the answer, mark as 0.

Special Cases

- - Similar meaning: Output 1 if the response conveys essentially the same meaning as the answer and does not omit or add critical information (e.g., answer:“put meat on the table”, response:“The person moved meat from the fridge to the counter.”).
- - Partial matches: If the response overlaps but misses or alters essential details (e.g., answer:“put meat and tomato on the table” vs. response:“put meat on the table”), output 0.
- - Granularity differences: If the response is more specific but still semantically equivalent (e.g., answer:“woman”, response:“Jessica”), output 1.
- - Yes/No questions: Only output 1 if the polarity matches (yes <-> yes, no <-> no). Any mismatch outputs 0, regardless of explanation.
- - Ambiguity: If the response cannot be reasonably interpreted as equivalent to the answer, output 0. Examples

- Example 1 Question: Did the attribute of plant changed because of the action getting something from something? Answer: yes Response: Yes, the attribute of plant got watered from no to yes after the action getting something from something. Your output: 1

- Example 2 Question: what status of fork changed while the person do the first action did before he/she put something to something? Answer: cleanliness Response: fork was in drawer before the person put fork to sink.

- Your output: 0

Example 3 Question: What is the person doing before he/she close something? Answer: Put tomato to fridge Response: The person is putting a tomato in the fridge.

- Your output: 1

- Example 4 Question: What is the first action the person did in the video? Answer: Work on sofa Response: The person pulled out a chair.

- Your output: 0

Example 5 Question: How did the person changed the spatial relationships of meat? Answer: Put meat to table Response: The person moved meat from the fridge to the counter.

- Your output: 1

- Example 6 Question: what status of fridge changed while the person do the first action did after he/she point to something? Answer: openess Response: The fridge was closed before the person point to something, and after that the fridge changed to open. Your output: 1
- Example 7 Question: which object changed its status when the person do the last action in the video? Answer: fork Response: spoon

- Your output: 0

- Example 8 Question: What is the action that just happened? Answer: Place can in the tray Response: The person puts the can on the table.

- Your output: 0

- Example 9 Question: current goal is: Please place the fruits in the bowl then place the kitchen supplies into the holder. last 20 steps: 1. put white packet in the bowl 2. put white packet in the bowl 3. put yellow packet in the bowl 4. put blue packet in the bowl 5. put blue packet in the bowl 6. put blue packet in the bowl 7. put yellow packet in the bowl. What’s the immediate next step? Answer: Put duster in the black stand Response: put brush in the holder

- Your output: 0

Your Turn: Question: {question} Answer: {answer} Response: {prediction}

Your output:

Prompt for GPT-4o (OpenEQA-hm3d & OpenEQA-scannet). You are an AI assistant tasked with evaluating whether a response matches the correct answer to a given question, considering both the primary answer and any extra correct answers.

##Evaluation Rules

- (1) Output 1 if the response matches the answer or any of the extra answers exactly or with synonymous/equivalent wording.

- - Synonyms, paraphrases, or different surface forms of the same meaning count as matches.
- - Minor wording differences (e.g., “Wood panel” vs. “Wood) count as matches.

- (2) Output 0 if the response is incorrect, contradictory, or refers to a different entity, object, or attribute than the answer and all extra answers.

- - If the answer and response describe different objects, actions, or states, mark as 0.
- - If the response introduces additional details that change the meaning of the answer, mark as 0. ##Special Cases
- - Similar meaning: Output 1 if the response conveys essentially the same meaning as the answer and does not omit or add critical information (e.g., answer: “A ceiling fan”, response: “fan”).
- - Partial matches: If the response overlaps but misses or alters essential details (e.g., answer: “put meat and tomato on the table” vs. response: “put meat on the table”), output 0.
- - Granularity differences: If the response is more specific but still semantically equivalent (e.g., answer: “woman”, response: “Jessica”), output 1.
- - Yes/No questions: Only output 1 if the polarity matches (yes <-> yes, no <-> no). Any mismatch outputs 0, regardless of explanation.
- - Ambiguity: If the response cannot be reasonably interpreted as equivalent to the answer, output 0. ##Examples:

- Example 1: Question: Is it overcast? Answer: no Extra Answers: ["doesn’t look like it", "no", "it’s sunny"] Response: yes

- Your output: 0

Example 2: Question: Who is standing at the table? Answer: woman Extra Answers: ["a woman", "a lady", "woman"] Response: Jessica

- Your output: 1

- Example 3: Question: Are there drapes to the right of the bed? Answer: yes Extra Answers: ["yes, there are drapes", "yeah", "the drapes are to the right of the king bed"] Response: yes

Your output: 1

- Example 4: Question: What material is the ceiling in the living room? Answer: Wood panel Extra Answers: null Response: wood

- Your output: 1

- Example 5: Question: What is in between the two picture frames on the blue wall in the living room? Answer: The TV Extra Answers: null Response: air conditioner

- Your output: 0

Example 6: Question: Is the house doorway open or closed? Answer: Open Extra Answers: null Response: The house doorway is open.

- Your output: 1

- Example 7: Question: Is my backyard safe to let me dog out in? Answer: Yes, its fenced. Extra Answers: null Response: yes

Your output: 1

- Example 8: Question: What is hanging from the ceiling in the bedroom? Answer: A ceiling fan Extra Answers: null Response: fan Your output: 1
- Example 9: Question: Where is the full body mirror? Answer: In the bedroom by the door Extra Answers: ["next to the bedroom door", "just inside the bedroom", "in the bedroom", "in the bedroom right next to the door"] Response: The full body mirror is in the bedroom. Your output: 1
- Example 10: Question: What is leaning in the corner by the coat rack? Answer: An umbrella Extra Answers: null Response: chair Your output: 0

Your Turn: Question: {question} Answer: {answer} Extra Answers: {extra_answers} Response: {prediction} Your output:

AgiBot-ER. We curate 97, 120, 381 test samples for three sub-tasks, i.e., Task Status Verification, Action Affordance, and Next Action Prediction, respectively. We report the average accuracy of the three tasks.

Task Status Verification

<image><image><image><image><image><image> {question} Your answer can only be "yes" or "no"

Action Affordance

<image><image><image> {question} Your answer can only be "yes" or "no".

Next Task Prediction

<image><image><image><image><image><image> {question} {options} Answer with the option’s letter from the given choices directly.

MME. We report the accuracy+ metric for MME [18].

<image> {question}

Please answer yes or no.

MMBench. We demonstrate the official metric provide by MMBench [40].

<image> {question} {options} Answer with the option’s letter from the given choices directly.

RealWorldQA. We use the official metric of RealWorldQA [71].

<image> {question} {options}

Please answer directly with only the letter of the correct option and nothing else.

SimpleVQA. We compute accuracy by leveraging GPT-4o to compare responses against the ground-truth labels.

<image> {question}

EgoSchema & VideoMME & NextQA. We set the number of video frames to 128 and evaluate accuracy on the three video understanding benchmarks.

<video> {question} {options} Answer with the option’s letter from the given choices directly.

MathVista. We use the official metric of MathVista [42].

<image> {question}

MathVision. We use the official metric of MathVision [69].

<image> {question} {options}

Please only output your final answer or a single letter (if it is a multiple choice question).

MMMU. We use the official metric of MMMU [82].

<image> {question} {options} Answer with the option’s letter from the given choices directly.

##### A.2 The AGIBot offline evaluation set

We select 16 high-frequency daily tasks from the AGIBot dataset for offline evaluation: “Heat food in the microwave oven,” “Make a sandwich,” “Wash dishes with dishwasher,” “Wash the dishes and silverware,” “Discard the trash on the coffee table,” “Pick up the item to wipe away the stain,” “Place items from the meeting room table into the storage box,” “Arrange sofa,” “Wash clothes in the washing machine,” “Remove clothes from the washing machine,” “Arrange flowers,” “Boil water in the kettle,” “Packing in the supermarket,” “Scan and package the goods,” “Restock the hanging basket area,” and “Restock supermarket snacks.” To ensure fairness, the data for these tasks are excluded from both the pretraining and fine-tuning datasets.

###### A.3 Offline evaluation examples We present a snapshot of our offline evaluation sets.

[

... {

"role": "user", "content": "<|system_info_start|>Your current views are<image><image><image><|system_-

info_end|><|human_start|>I want some fruit, put all fruits into the basket.<|human_end|>" }, {

"role": "assistant", "content": "<|think_start|>...<|think_end|><|plan_start|>put the banana into the

plastic box<|plan_end|><|response_start|>...<|response_end|>", "other_actions": [ "put the grapes into the plastic box" ]

}, {

"role": "user", "content": "<|system_info_start|>Your current views are<image><image><image><|system_-

info_end|><|human_start|>Wait, I don’t want banana.<|human_end|>" }, {

"role": "assistant", "content": "<|think_start|>...<|think_end|><|plan_start|>put the banana on the

table<|plan_end|><|response_start|>...<|response_end|>", "other_actions": [ ]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

(a) (b) (c) (d) (e)

Figure 7 Example initial states of the five tasks used in our online evaluation. (a) Table bussing: removing used dishes, utensils, and food items; (b) Checkout packing: organizing and placing purchased items into bags or boxes at checkout; (c) Dietary filtering: selecting or excluding food and drink items based on dietary constraints; (d) Grocery shopping: retrieving and collecting grocery items according to user instructions; (e) Tableware organization & shipment: sorting, packing, and transporting tableware to designated locations.

},

... ]

The above example illustrates two consecutive rounds within a trajectory. We query VLMs in a manner analogous to multi-turn dialogue interactions. For the first round, planning is considered correct if the VLM output semantically matches either: <|plan_start|>put the banana into the plastic box<|plan_end|> or <|plan_start|>put the grapes into the plastic box<|plan_end|>. For the second round, we evaluate models under a teacher-forcing setting, incorporating the ground-truth from the first round into the dialogue history. Since the other_actions list is empty for this step, planning is deemed correct only if the VLM output semantically matches: <|plan_start|>put the banana on the table<|plan_end|>.

##### A.4 Online evaluation tasks

Example initial states of the five tasks used in our online evaluation are shown in Figure 7. Below, we provide an overview of each task.

- • Table bussing: removing used dishes, utensils, and food items. In our experiment, the user requests the robot to clear the table, but interrupts with the command “I still need it” while the robot is placing a glass into the plastic box. We also introduce a failure event when the robot attempts to place a spoon into the box.
- • Checkout packing: organizing and placing purchased items into bags or boxes at checkout. In our experiment, the user asks the robot to move all items on the checkout counter into the shopping basket. An interruption occurs with the command “I am allergic to caffeine” while the robot is placing a coffee into the basket.
- • Dietary filtering: selecting or excluding food and drink items based on dietary constraints (e.g., no caffeine, vegetarian). In our experiment, the user issues a series of instructions: (1) “Put the food with the highest energy into the plastic box, then add a drink. Note that I am allergic to caffeine.”; (2) “Discard all drinks containing caffeine into the rubbish bin”; (3) “Place all fruits and vegetables into the plastic box”; and (4) “Clean up all remaining items on the table”.
- • Grocery shopping: retrieving and collecting grocery items according to user instructions. In our experiment, the user first asks the robot to recommend a snack suitable for a road trip. Following the recommendation, the user requests a less sweet snack for the cart, then asks for a non-alcoholic drink, and finally makes a conditional request: “Are there any nuts on the shelf? If so, put some into the shopping cart.”
- • Tableware organization & shipment : sorting, packing, and transporting tableware to designated locations. In our experiment, the user instructs the robot to discard all trash and transport the tableware to the sink. Specifically, the robot must first dispose of garbage in the rubbish bin, then place all tableware into a plastic box, and finally carry the box to the sink. During this process, the user interrupts with the command “Keep it on the table” while the robot is placing a stainless steel cup into the box.

##### A.5 Offline & Online Evaluation Prompts

We present the prompts used in our experiments, including the judge prompt for offline evaluation, the baseline model prompts, and the reward model prompt for reinforcement learning.

Judge Prompt for Offline Evaluation

# Role You are a judge to decide the similarity of two sentences.

# Inputs You will be provided with two sentences, each sentence represent an action from a robot.

# Output Format Your output should be a single number that is either 0 or 1, where 0 means the two sentences are semantically different, and 1 means the two sentences are semantically the same. Output in the following format: "[SOME ANALYSIS]. The final answer is: [NUMBER]"

# Examples User:

- 1. put the fork into the basket
- 2. put the fork into the basket Assistant: Two sentences are exactly the same. The final answer is: 1.0 User:

- 1. put the iron fork into the woven basket
- 2. put the fork into the basket Assistant: Two sentences are the same semantically. The final answer is: 1.0 User:

- 1. Grab the bottled french fries on the table with the right arm.
- 2. Grab the bottled french fries. Assistant: Two sentences are the same semantically. The final answer is: 1.0 User:

- 1. Move forward slightly
- 2. Move forward Assistant: Two sentences are the same semantically. The final answer is: 1.0 User:

- 1. put the coffee into the plastic box
- 2. put the bottle of coffee into the plastic box Assistant: Two sentences are the same semantically. The final answer is: 1.0 User:

- 1. put the glass into the plastic box
- 2. put the glass cup into the plastic box Assistant: Two sentences are the same semantically. The final answer is: 1.0 User:

- 1. put the fork into the basket
- 2. put the knife into the basket Assistant:

The objects (fork v.s. knife) are different semantically. The final answer is: 0.0 User:

- 1. put the fork into the basket
- 2. pick up the fork Assistant: The actions are different semantically. The final answer is: 0.0 User:

- 1. put the fork into the basket
- 2. navigate to the table Assistant: Neither actions nor objects matches. The final answer is: 0.0 User:

- 1. put the fork into the basket
- 2. pick up the stainless steel cup Assistant: Neither actions nor objects matches. The final answer is: 0.0

English Prompt for Baseline Models.

We use the following prompt to configure VLMs as the interaction and planning modules of the robot system in both offline and online experiments.

# Role You are a robot with two grippers developed by Bytedance, and your name is Roobio. You are deployed in a home environment, and your job is to have natural interactions with users and complete some tasks required by the user. You have basic movement, perception, and manipulation capabilities, and can navigate to the following areas: dining table, sink, refrigerator, dishwasher, microwave, bread maker, cupboard, shoe cabinet, sofa, and washing machine.

# Input In each round, you will be provided with the following information.

- 1. Three images: captured by the cameras from your head/left gripper/right gripper respectively. These images reflect your current visual perception at the current moment.
- 2. User instructions (optional): user may give you some instructions, user instructions are in chinese. # Output To distinguish reasoning, planning, response and query. You must output with the following fields.

- 1. ‘<|think_start|> ... <|think_end|>’: this field is for your reasoning process, it can include

- - The key items in the current scene.
- - Whether the previous action is complete.
- - Reasoning about your next action.

- 2. ‘<|plan_start|> ... <|plan_end|>’ (optional): this field is for task planning, You should plan your next action based on your reasoning process. The action should be helpful to complete user’s task.
- 3. ‘<|response_start|> ... <|response_end|>’ (optional): this field is your response to user, you response must be in *chinese*. You should respond to the user in the following situations.

- - When the user is chatting with you;
- - When the user gives a new instruction on completing a task, you should always respond to

the user whenever you receive a new instruction. The response should be concise, polite, and relevant to the task.

- When you need to ask the user for more information. For example, if the user ask you to give him/her a drink, and you observe both orange juice and apple juice, then you should ask

the user which one he/she prefers. You should avoid asking the user for more information too frequently.

# Guidelines

- 1. The text inside ‘<|think_start|> ... <|think_end|>’ and ‘<|plan_start|> ... <|plan_end|>’ should be in english while the text in ‘<|response_start|> ... <|response_end|>’ must be in chinese.
- 2. When the user interrupt your action, you need to stop the current action and re-plan.
- 3. Your task planning inside ‘<|plan_start|> ... <|plan_end|>’ should be in the following format:

- - put the ... into the ...
- - pick up the ...
- - navigate to the ...
- - open the ...
- - close the ...

- 4. Only include one action at a time, planning such as ‘<|plan_start|> put the A into the B, put the C into the B<|plan_end|>’ is invalid.

# Examples User: <|system_info_start|>Your current views are...<|system_info_end|><|human_start|>Robot, clean up the table for me.<|human_end|> Assistant: <|think_start|>The user asks me to clean up the table, I can observe.... <|think_end|><|plan_start|>put the ... into the ...<|plan_end|><|response_start|>...<|response_end|> User: <|system_info_start|>The views you see after finishing previous action are...<|system_info_end|> Assistant: <|think_start|>I should continue cleaning up the table, there are ... left on the table,

...<|think_end|><|plan_start|>put the ... into the ...<|plan_end|>

...

Chinese Prompt for Baseline Models

We also developed a corresponding Chinese prompt for models that perform better with Chinese.

# 角色

你是字节跳动研发的双臂机器人。你被部署在一个开放式厨房环境中，负责与人类用户进行自然互动并完成日常操 作任务。你具备基础的移动、感知与操控能力，可前往以下区域执行操作：餐桌旁、水池旁、冰箱旁、洗碗机旁、 微波炉旁、烤面包机旁、水池旁的台面、鞋柜旁、沙发旁和洗衣机旁。

# 输入信息 在每轮交互中，你将接收到以下信息：

- （1）三张图像：分别来自你头部摄像头、左夹爪摄像头、右夹爪摄像头，反映你在当前时刻的视觉感知。
- （2）用户指令（可选）：人类用户可能通过语音或文本向你发出任务指令。

# 输出格式 你必须输出以下结构化内容，用于指导任务执行与用户交互。输出字段包括：

- （1）<|think_start|> ... <|think_end|>：你的内部思考与任务状态判断，包括：

- - 当前场景中包含的关键物品；
- - 是否完成上一步操作；
- - 当前场景变化的观察；
- - 下一步应执行的关键动作与理由；
- - 如遇异常（如抓取失败），应在此说明并调整策略。

- （2）<|plan_start|> ... <|plan_end|>（可选）：你对下一步具体动作的操作规划，目标是完成当前用户任 务。操作步骤应使用英文简洁描述。
- （3）<|response_start|> ... <|response_end|>（可选）：你对用户的语音或文本形式的自然语言回复，内容 需简洁、礼貌并与任务状态相关。当当前信息不足以明确执行用户任务时，你可以主动向用户提问以获取更多信

息。但请遵循“非必要不提问”的原则，避免频繁打断用户。 # 任务规则 在于用户的交互与执行任务的过程中，你需要遵循以下规则：

- （1）‘<|think_start|> ... <|think_end|>’和‘<|response_start|> ... <|response_end|>’中的文本以中 文输出，‘<|plan_start|> ... <|plan_end|>’中的任务规划以英文输出；
- （2）用户可能会打断你的操作，你需要基于当前画面和用户给你的最新指令决定下一步的操作；
- （3）‘<|plan_start|>...<|plan_end|>’中的操作推荐遵循以下格式：

- - put the ... into the ...
- - pick up the ...
- - navigate to the ...
- - open the ...
- - close the ...

- （4）‘<|plan_start|>...<|plan_end|>’中的任务规划一次只能规划下一步的内容，不能规划之后多步的内容，例 如：put the A to the B, put the C to the D...这样多个操作不能放到一次规划中，但是它可以出现在你的思 考过程中。

# 例子 User: <|system_info_start|>你当前看到的画面为...<|system_info_end|><|human_start|>帮我清理桌面。<|human_end|> Assistant: <|think_start|>用户让我清理桌面，当前画面中我能看到.... <|think_end|><|plan_start|>put the ... into the ...<|plan_end|><|response_start|>收到。<|response_end|> User: <|system_info_start|>你完成操作后所看到的场景为...<|system_info_end|> Assistant: <|think_start|>我应该继续收拾桌面，当前我能看到桌面上还有...<|think_end|><|plan_start|>put the ... into the ...<|plan_end|>

...

Reward Model Prompt for RL

# Role You are a consistency checker for a robot’s planning. Your task is to evaluate the robot’s thinking process and the resulting final action.

# Input

- 1. The robot’s thinking process, enclosed by <|think_start|> and <|think_end|>.
- 2. The robot’s final action, enclosed by <|plan_start|> and <|plan_end|>. # Task You need to determine:

- 1. Whether the thinking process within <|think_start|> and <|think_end|> is reasonable.
- 2. Whether the final decision within <|plan_start|> and <|plan_end|> is consistent with the decision derived from the thinking process in <|think_start|> and <|think_end|>.

# Output Your output must follow this format: [Your reasoning process] + "The final answer is: [NUMBER]."

Here, [NUMBER] is an integer: ‘1’ indicates consistency; ‘-1’ indicates inconsistency; ‘0’ indicates that it is impossible to judge.

# Notes

- 1. Ignore ambiguity in names. Items that could potentially be consistent should be considered consistent.
- 2. Assign -1 only if the thinking process is completely unreasonable.

- 3. Assign -1 only if it’s completely impossible for the thinking process and the plan to be consistent.
- 4. If the plan is only a part of the strategy outlined in the thinking process, it is still considered consistent (i.e., the plan does not need to reflect all thoughts).

#### B Embodied Task-centric Reasoning

##### B.1 Data Synthesis

Publicly available robot datasets often contain long video demonstrations with temporally annotated clips and corresponding individual actions, or they can be segmented automatically into clips using VLMs. To equip Robix with reasoning and planning capabilities in embodied scenarios, we design three dedicated synthesis pipelines, each targeting a distinct sub-task: task status verification, action affordance, and next action prediction. The details of these pipelines are provided below.

Task Status Verification. The status of a task is categorized as either complete or incomplete. To construct task verification data labeled as complete, we use the entire video clip as the visual input. Reasoning traces are obtained by prompting a strong VLM (Seed-1.5-VL in thinking mode) with the question “Is the {action} complete?”. We then extract the reasoning content from the response and retain only those instances where the predicted status is complete. Conversely, to represent incomplete task status, we randomly truncate each video clip to between one-half and one-third of its original duration, producing partial visual inputs. As before, we retain only those cases where the predicted status is identified as incomplete.

Action Affordance. To obtain high-quality labels for action affordance, we distill both labels and reasoning traces from powerful VLMs. However, we find that relying on a single VLM is insufficient for generating accurate labels reliably. Therefore, we employ various VLMs to produce candidate labels and take the intersection of their outputs as the final annotation. The corresponding reasoning traces are extracted from the Seed-1.5-VL-Think model.

Next Action Prediction. The robot datasets are typically annotated with action sequences. Thus, the ground-truth next action can be directly determined once the previous action is specified. To synthesize the corresponding reasoning trace, we leverage Seed-1.5-VL by prompting it with both the video and the following textual input: “You are an intelligent robot executing {the task}. To complete this task, you are performing the action as shown in the video. Your next action is {next action}. Please reflect on the rationale behind your next action and output your response from the perspective of a robotic agent.”

##### B.2 Agibot-ER Benchmark

We construct an embodied task-centric reasoning benchmark based on the Agibot dataset [6], which provides manual annotations of tasks, atomic action sequences, and the corresponding video clips for each action. Specifically:

- • Task Status Verification: We randomly select 100 samples from the synthesized dataset and further verify the label accuracy, resulting in 97 test samples.
- • Action Affordance: We select 120 video clips and corresponding individual actions from 50 different tasks. To ensure accuracy, we manually annotate these samples rather than relying on VLM-based annotation.
- • Next Action Prediction. The ground-truth next action is obtained directly from the robot datasets. We randomly sample 400 instances and construct distractor options by leveraging a powerful LLM to generate incorrect candidates that differ subtly from the correct one—such as variations in action type, target object, or destination. To ensure quality, we manually verify all synthesized options, resulting in 381 validated test samples.

All test samples are eliminated from the training dataset.

#### C Case studies of RL

Our reinforcement learning stage yields several improvements over the SFT model, including: (1) reduced irrational reasoning, (2) improved thought-action consistency, and (3) fewer formatting errors. In the following, we present comparative case studies between Robix-32B-RL and Robix-32B-SFT to illustrate the effectiveness of reinforcement learning.

Reduced irrational reasoning

# Current task Put all the drinks on the table into the carton.

# Robix-32B-RL output <|think_start|>Based on the current scene, I see that there is already a bottle of soda inside the carton, and the remaining items on the table are milk, Oreos, and gum. Recalling my last action, I have already put the coke into the carton. Now, the only beverage left on the table is milk. To complete the task of putting all beverages into the carton, the next step should be to put the milk into the carton. <|think_end|> <|plan_start|>put the milk into the carton<|plan_end|>

# Robix-32B-SFT output <|think_start|>Based on the current scene, I see that there is already a bottle of soda inside the carton, and the remaining items on the table are milk, Oreos, and gum. Recalling my last action, I have already put the coke into the carton. Now, all the beverages on the table have been put into the carton, and the task goal has been achieved. Next, I should proactively ask the user if there are any other tasks that need assistance. <|think_end|> <|response_start|>The task has been processed. Is there anything else I can help you with? <|response_end|>

Improved thought-action consistency

# Current Task Put the dishes in the sink

# Robix-32B-RL output <|think_start|>I have grasped the storage basket with my dual grippers. The basket contains bowls, forks, and spoons. According to the user’s instructions, the next step is to take these dishes to the sink. I have now finished collecting the tableware. Next, I should navigate to the sink area and deliver the dishes to the designated location.<|think_end|> <|plan_start|>navigate to the sink<|plan_end|>

# Robix-32B-SFT output <|think_start|>I have grasped the storage basket with my dual grippers, ready to transport the dishes. According to the user’s instructions, the dishes need to be taken to the sink. All the dishes that need to be washed have been collected in the storage basket. The next step is to transport the storage basket to the sink area, completing the task of collecting and transporting the dishes.<|think_end|> <|plan_start|>navigate to the cupboard<|plan_end|>

Fewer formatting errors

# Current Task Organize the table

# Robix-32B-RL output <|think_start|>From the current scene, the paper cup is still on the table, near the rubbish bin, and is grasped by my gripper, which indicates that the previous attempt to put the paper cup into the rubbish bin was not successful. Besides the paper cup, there is also a plate

with some leftover food residue on the table. To continue organizing the table, I first need to re-attempt putting the paper cup into the rubbish bin to ensure the trash is handled correctly.<|think_end|> <|plan_start|>put the paper cup into the rubbish bin<|plan_end|>

# Robix-32B-SFT output <|think_start|>From the current scene, the paper cup is still on the table, near the rubbish bin, and is grasped by my gripper, which indicates that the previous attempt to put the paper cup into the rubbish bin was not successful. Besides the paper cup, there is also a plate with some leftover food residue on the table. To continue organizing the table, I first need to re-attempt putting the paper cup into the rubbish bin to ensure the trash is handled correctly.<|think_end|> {

"action": "put", "object": "paper cup", "target": "rubbish bin"

}

