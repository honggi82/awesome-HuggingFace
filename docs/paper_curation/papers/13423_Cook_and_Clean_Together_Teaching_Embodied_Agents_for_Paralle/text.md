## Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution

### Dingkang Liang1∗, Cheng Zhang1*, Xiaopeng Xu1, Jianzhong Ju2, Zhenbo Luo2, Xiang Bai1

1Huazhong University of Science and Technology 2MiLM Plus, Xiaomi Inc. {dkliang, czhang2024, xbai}@hust.edu.cn, {jujianzhong, luozhenbo}@xiaomi.com

##### Abstract

[Figure 1]

# arXiv:2511.19430v1[cs.CV]24Nov2025

Preparing the kitchen for cooking, including washing the sink (4 mins), using the microwave to heat food (15 mins), and wiping the counter (5 mins).

Task scheduling is critical for embodied AI, enabling agents to follow natural language instructions and execute actions efficiently in 3D physical worlds. However, existing datasets often simplify task planning by ignoring operations research (OR) knowledge and 3D spatial grounding. In this work, we propose Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D), a new task that requires the synergy of language understanding, 3D grounding, and efficiency optimization. Unlike prior settings, ORS3D demands that agents minimize total completion time by leveraging parallelizable subtasks, e.g., cleaning the sink while the microwave operates. To facilitate research on ORS3D, we construct ORS3D-60K, a large-scale dataset comprising 60K composite tasks across 4K real-world scenes. Furthermore, we propose GRANT, an embodied multi-modal large language model equipped with a simple yet effective scheduling token mechanism to generate efficient task schedules and grounded actions. Extensive experiments on ORS3D-60K validate the effectiveness of GRANT across language understanding, 3D grounding, and scheduling efficiency.

Total time=24 mins Wash the sink

[Figure 2]

Use the microwave

Wipe the counter

(a) The simple one-by-one completion scheme

Total time=15 mins

[Figure 3]

Start the microwave

Wipe the counter

Wash the sink

Close the microwave

(b) Our OR knowledge-based 3D grounded task scheduling

Figure 1: Comparison of different task completion schemes. An embodied agent is expected to use operations research knowledge to efficiently complete tasks through scheduling.

by leveraging Operations Research (OR) knowledge. This includes identifying which subtasks can be executed concurrently with other subtasks and maximizing the use of waiting time to achieve optimal efficiency. Second, although their setting assumes an agent operating in 3D environments, it is often reduced to textual question answering, without explicitly grounding each step to the target object’s location within the 3D scene. This lack of spatial grounding severely hinders the utility of such plans for downstream embodied executions that require spatial location information (e.g., navigation).

Code — https://github.com/H-EmbodVis/GRANT

### 1 Introduction

Task scheduling is fundamental for embodied agents to efficiently execute human-assigned tasks (Duan et al. 2022; Wang et al. 2024; Huang et al. 2024b; Driess et al. 2023). Achieving this requires the seamless integration of natural language understanding, efficiency optimization, and spatial perception within real-world 3D environments.

To address these limitations and extend the capability of embodied agents for efficient task scheduling, we propose a new and practical task named Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D). In this task, an embodied agent must generate efficient schedules by leveraging OR knowledge and locate the 3D positions of target objects in each action step to complete assigned tasks. As demonstrated in Fig. 2, when we assign a composite task to an embodied agent, we hope it can efficiently complete it by utilizing the waiting periods of subtasks that can be performed concurrently. For example, "Using the microwave" allows the agent to perform other subtasks during its waiting period. To achieve maximum efficiency, the embodied agent must leverage these subtask properties and incorporate OR knowledge to generate an optimal task schedule. Meanwhile, to execute each step in the real world, the agent must accurately localize the target objects within the

Recently, several works (Huang et al. 2024b; Chen et al. 2024; Zhang et al. 2024) have made preliminary attempts on plan generation in 3D environments, allowing models to generate step-by-step plans from human instructions (Fig. 1(a)). Nevertheless, these attempts are oversimplified and exhibit critical limitations that hinder their practical applications. First, they lack consideration of task properties and optimization of efficiency. Under their setting, a model only needs to generate plausible actions in terms of natural language. In contrast, as shown in Fig. 1(b), an embodied agent is assumed to have the capacity to efficiently complete the task

*Equal contribution. Corresponding author. Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

[Figure 4]

2 4 1

5

First, I will start the microwave to cook food. With it set and running, I can perform other tasks during the 35 minutes.

Finally, I will clean the rectangular shaped stainless refrigerator.

While the microwave is operating, I will wash the oblong-shaped sink with a smooth texture, located lower than the kitchen cabinets.

As the 35 minutes come to an end, I will return to finish the cooking and turn off the microwave.

[Figure 5]

[Figure 6]

- 2

- 3

3

Next, I will wipe down the sleek, modern stainless steel coffee maker with a black and silver finish and blue lights, resting on the kitchen counter.

[Figure 7]

Please ensure the kitchen is tidy coffee maker and ready for use, including

[Figure 8]

[Figure 9]

[Figure 10]

- (1) washing the oblong-shaped sink with a smooth texture, located lower than the kitchen cabinets (10 minutes),
- (2) operating the microwave with a silver exterior and modern design, placed on the kitchen counter, to cook food (35 minutes),
- (3) wiping down the sleek, modern stainless steel coffee maker with a black and silver finish and blue lights, resting on the kitchen counter (24 minutes),
- (4) cleaning the rectangularshaped stainless refrigerator (4 minutes).

[Figure 11]

[Figure 12]

[Figure 13]

refrigerator sink

[Figure 14]

[Figure 15]

4

[Figure 16]

[Figure 17]

microwave

[Figure 18]

[Figure 19]

1

[Figure 20]

[Figure 21]

5

[Figure 22]

[Figure 23]

###### 80 47% Quicker!

73

[Figure 24]

TotalTime(mins)

[Figure 25]

[Figure 26]

[Figure 27]

😃

60

[Figure 28]

[Figure 29]

[Figure 30]

39

40

[Figure 31]

[Figure 32]

With task scheduling

[Figure 33]

20

Start subtask 2 Do subtask 3

Recheck subtask 2

Do subtask 4

Do subtask 1

[Figure 34]

0

With task scheduling

Operate one by one

Total time: 35 (subtask 2) + 4 (subtask 4) = 39 mins.

Figure 2: Illustration of the proposed Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D). When assigned a composite task by a human, the embodied agent needs to complete the subtasks efficiently by carefully scheduling using operations research knowledge and simultaneously locating the target objects in each step for navigation and manipulation.

3D scene. Therefore, ORS3D poses significant challenges to existing 3D agents (Huang et al. 2024b,a; Deng et al. 2025; Lin et al. 2023; Chen et al. 2024) in two essential aspects: 1) It requires OR knowledge to identify subtasks that can be performed concurrently and make efficient task schedules. 2) It entangles language and spatial understanding (i.e., an embodied agent is required to simultaneously generate actions and locate the target objects in the 3D scene).

vals of those that can be performed concurrently, producing an optimal execution schedule. During inference, GRANT first predicts subtask properties as constraints, then uses the scheduling token to invoke the solver and generate the optimal schedule, which is subsequently injected back into the model to guide the generation of step-wise action descriptions and target object groundings. Compared to the baseline method (Chen et al. 2024), our approach yields a significant 30.53% improvement in task completion time efficiency, along with notable gains of 1.38% in grounding accuracy and 10.46% in overall performance. As an initial attempt, our method paves the way for further exploration in ORS3D.

To facilitate research on this new task, we construct the ORS3D-60K dataset consisting of 60,825 composite tasks across 4,376 real-world indoor scenes. As shown in Tab. 1, compared to existing 3D understanding and task planningrelated datasets (Wu et al. 2023; Chen et al. 2024; Zhang et al.

In summary, our contributions are as follows: 1) We introduce operations research knowledge-based 3D grounded task scheduling, a new and practical task that meets the common requirement of embodied agents to efficiently complete tasks in the physical world. 2) To support this new task, we construct a large-scale dataset, ORS3D-60K. To the best of our knowledge, we are the first to incorporate operations research knowledge for task scheduling in 3D scenarios. 3) We propose GRANT, an embodied MLLM with a simple yet effective scheduling token mechanism, integrating task scheduling with multimodal understanding to generate efficient, grounded task execution schedules.

- 2024; Huang et al. 2024b; Zhu et al. 2024a), ORS3D-60K is the first to incorporate OR knowledge. It also has the largest number of tasks and presents the most significant challenge by requiring models to generate lengthy textual solutions and provide 3D grounding for target objects. To assess the capability of existing methods in addressing this task, we evaluate several baselines (Huang et al. 2024b; Zhu et al. 2024b, 2023; Chen et al. 2024) from language understanding, efficiency optimization, and spatial perception, where the results show they struggle with this challenging task.

To tackle the ORS3D problem, we further propose a grounded task scheduling agent named GRANT, which is empowered by a Multi-modal Large Language Model (MLLM) and equipped with a simple yet effective Scheduling Token Mechanism (STM) to generate efficient task schedules. Specifically, we introduce a learnable scheduling token that links to an external optimization solver to generate task schedules based on task property constraints provided by the MLLM. The solver employs a dynamic programming algorithm to arrange subtasks within the available time inter-

### 2 Related Works

#### 2.1 Task Planning

Task Planning (Ahn et al. 2022; Choi et al. 2024; Zhang et al. 2024; Chen et al. 2023) is crucial, as it enables embodied agents to execute human instructions efficiently. Wu et al. (Wu et al. 2023) propose TaPA, a vision-language task planning agent that generates executable textual action steps

Dataset Reference #Scenes #Task Avg. length Text output 3D Grounding Planning OR knowledge

TaPA (Wu et al. 2023) arXiv 23 80 15,418 69 ✓ ✗ ✓ ✗ Embodied planning (Chen et al. 2024) arXiv 24 1,319 4,357 37 ✓ ✓ ✓ ✗ SG3D (Zhang et al. 2024) arXiv 24 4,895 22,346 71 ✗ ✓ ✓ ✗ ScanReason (Zhu et al. 2024a) ECCV 24 1,456 12,929 29 ✗ ✓ ✗ ✗ LEO (Task planning) (Huang et al. 2024b) ICML 24 478 13,848 98 ✓ ✗ ✓ ✗ Intent3D (Kang et al. 2025) ICLR 25 1,042 44,990 9 ✗ ✓ ✗ ✗

ORS3D-60K (ours) - 4,376 60,825 311 ✓ ✓ ✓ ✓

- Table 1: Comparison with related datasets. "Avg. length" denotes the average word length of each data item. Our dataset is the only one that introduces Operations Research (OR) knowledge for task scheduling.

### 3 The ORS3D-60K Dataset

for robot navigation and manipulation using multi-view images of the 3D scene. Huang et al. (Huang et al. 2024b) construct a task planning dataset that requires embodied agents to generate step-wise plans from instructions. SG3D (Zhang et al. 2024) proposes task-oriented sequential grounding in 3D scenes, where an agent is required to locate each target object in a given sequence of actions. In contrast to previous works, we focus on more complex scheduling scenarios and the integration of multi-modal information processing.

In this section, we introduce the definition of Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D) and provide details of the proposed ORS3D-60K dataset.

|Non-parallelizable subtask A B C<br><br>Wipe the window to maintain clarity and a bright atmosphere.<br><br>Dust the upper shelf cluttered with books on top of the desk. Wipe the microwave oven placed on the kitchen counter.<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|
|---|

|Parallelizable subtask<br><br>A B A Open the printer placed on the blue desk to print papers.<br><br>Fill the sink inside the cabinet in the bathroom. Use the microwave below the metal shelf for cooking food.<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

#### 2.2 3D Scene Understanding

3D scene understanding is the foundation of embodied AI, enabling it to act in real-world scenes. 3D scene understanding includes depth estimation (Xu et al. 2023, 2025a), 3D object detection (Kolodiazhnyi et al. 2024b; Liang et al. 2025b; Zhou et al. 2025), segmentation (Takmaz et al. 2023; Xu et al. 2025b; Liang et al. 2024, 2025a), and grounding (Chen, Chang, and Nießner 2020; Huang et al. 2024c; Jiang et al. 2024). Mask3D (Schult et al. 2023) is often used as an offthe-shelf object proposal extractor for downstream tasks or as a 3D scene encoder, as the flexible learned instance queries can be easily assembled to Transformer-based LLMs. OneFormer3D (Kolodiazhnyi et al. 2024a) is an end-to-end method that performs instance and semantic segmentation consistently, utilizing a group of learnable instance queries.

Figure 3: Non-parallelizable subtask & parallelizable subtask.

#### 3.1 Design Principles

Understanding human instructions, making efficient schedules to complete human-assigned tasks, and interacting with objects are common and frequent requirements in real-world applications for embodied agents.

As illustrated in Fig. 3, tasks assigned to embodied agents can be categorized into two types from an OR perspective: 1) Non-parallelizable subtask requires continuous attention of the agent to manipulate the target object, such as wiping the table or dusting the shelf. 2) Parallelizable subtask only requires the agent to initiate and recheck the target object upon completion, without continuous attention and manipulation, such as using the microwave to heat food or filling the water sink. The agent needs to exploit the time intervals of parallelizable subtasks to achieve an efficiency objective.

#### 2.3 3D Multi-modal Large Language Models

3D MLLMs (Chen et al. 2024; Wang et al. 2023; Huang et al. 2024a; Kang et al. 2024; Fu et al. 2025; Chen et al.

- 2025; Zhu et al. 2024b, 2025; Hong et al. 2023) narrow the gap between spatial understanding and natural language processing. Several methods (Huang et al. 2024b; Zhu et al. 2023; Kang et al. 2024; Zhu et al. 2024b) utilize point cloud object proposals from off-the-shelf 3D object detectors to extract 3D scene information. Another line of research (Hong et al. 2023; Zhu et al. 2025) leverages pretrained 2D encoders to reconstruct 3D information for the LLMs. In contrast, other approaches like Grounded 3D LLM (Chen et al. 2024) and 3D-LLaVA (Deng et al. 2025) directly process scene point clouds using 3D scene encoders that are jointly trained with LLMs. However, although existing 3D MLLMs excel at scene understanding, they still lack the ability to leverage OR knowledge for efficient task scheduling and completion.

#### 3.2 Problem Formulation

The goal of OR knowledge-based 3D grounded task scheduling is to generate an efficient schedule and accurately locate the target object at each step to complete a composite task.

Specifically, suppose that an embodied agent in a 3D scene is assigned a composite task consisting of n subtasks, denoted as C = {τi}ni=1. Each subtask τi is an operation involving a target object with an expected time, described by a natural language instruction. To achieve efficient task scheduling, the agent needs to generate a time-efficient schedule A = {ai|(τi,li)}si=1 consisting of s steps to accomplish

|GPT-4o<br><br>[Figure 41]<br><br>Subtask meta-info.<br><br>[Figure 42]<br><br>Stage I: Subtask meta-information generation<br><br>3D scene graphs<br><br>Optimal subtask schedule list & subtask meta-info<br><br>3D point cloud<br><br>Optimal schedule generation<br><br>[Figure 43]<br><br>(1) Augmentation<br>(2) Refinement<br>(3) Verification<br><br><br>[Figure 44]<br><br>| |[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>u a x u|btask description rget object label pected time btask type| | |
|---|---|---|---|---|
<br><br>[Figure 48]|
|---|

|[Figure 49]<br><br>Composite task generation<br><br>[Figure 50]<br><br>[Figure 51]<br><br>OR knowledge-based scheduling information generation<br><br>[Figure 52]<br><br>Step-wise solution generation<br><br>Stage II: Structured data generation<br><br>[Figure 53]<br><br>Final structured OR knowledge-based 3D grounded task scheduling dataset<br><br>GPT-4o<br><br>Template<br><br>[Figure 54]<br><br>[Figure 55]|
|---|

(a) Dataset generation pipeline

| |Composite task| |
|---|---|---|
|[Figure 56]<br><br>Please organize and clean the kitchen area while ensuring optimal functionality of the appliances, including: 1) adjusting the wooden chair in front of the round wooden coffee table and to the left of the metallic chair with a folding design (7minutes), 2) sanitizing the stainless steel sink in the kitchen counter (13 minutes), 3) running the dishwasher with a modern design, located within the cabinets, to clean the load (22 minutes),<br><br>4) organizing the sleek cabinets beside the white column and above the dishwasher (11 minutes).<br><br>[Figure 57]<br><br>[Figure 58]| | |

| |Operations research knowledge-based scheduling| |
|---|---|---|
|[Figure 59]<br><br>Subtasks that require continuous attention to operate: subtask 1, subtask 2, subtask 4. Subtasks that do not need continuous attention to operate:<br><br>subtask 3. <SCH> I will complete the subtasks in the optimal order: start subtask 3, subtask 1, subtask 2, recheck subtask 3,<br>subtask 4.<br><br><br>[Figure 60]| | |

[Figure 61]

Step-wise solution with groundings

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

1. First, I will start running the dishwasher with a modern design, located within the cabinets, to clean the load. (start subtask 3)

2. While the dishwasher is running, I will adjust the wooden chair in front of the round wooden coffee table. (subtask 1)

3. In the next step, I will sanitize the surface of the stainless steel sink placed within the counter. (subtask 2)

4. Before 22 minutes are up, I will return to check if the load has been properly cleaned turn off the dishwasher. (recheck subtask 3)

5. Lastly, I will organize the sleek cabinets beside the white column and above the dishwasher. (subtask 4)

(b) An example of the composite task

Figure 4: (a) The ORS3D-60K dataset generation pipeline, which first generates subtask meta-information from 3D scene graphs, then uses this information to generate the structured dataset. (b) A composite task example from ORS3D-60K dataset. The green color mask indicates the ground-truth target object in the corresponding step.

50k

- 11k
- 12k
- 13k
- 14k
- 15k
- 16k
- 17k
- 18k

the composite task. Each step includes a textual action description ai for subtask τi and the 3D location li (e.g., 3D bounding box or point mask) of the target object.

17163

16389

40k

Frequency

Frequency

30k

14372

#### 3.3 Dataset Construction

20k

12901

The dataset construction pipeline is illustrated in Fig. 4(a). In Stage I, we use 3D point clouds from five real-world datasets: ScanNet (Rozenberszki, Litany, and Dai 2022), HM3D (Ramakrishnan et al. 2021), ARKitScenes (Baruch et al. 2021), 3RScan (Wald et al. 2019), and MultiScan (Mao et al. 2022). They are converted into textual 3D scene graphs (Jia et al. 2024) for subtask meta-information generation via GPT-4o. We refine the outputs for correctness and completeness, and perturb subtask expected times by ±10% to generate diverse optimal schedules. In Stage II, we compute the optimal task schedule using an optimization solver, then convert it into step-wise natural language instructions with phrase-level object grounding via GPT-4o. We also generate OR knowledgebased scheduling explanations using templates.

10k

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

0k

(b)

3 8 13 18 23 28

1 2 3 4

The number of subtasks (a) #Subtask in each composite task (b) Expected time of each subtask

Expected time (minutes)

Figure 5: Distributions of (a) subtask number in each composite task, and (b) the expected time of each subtask.

#### 3.4 Dataset Characteristics

The ORS3D-60K dataset exhibits several distinctive characteristics that make it stand out from existing datasets.

First, our dataset is closely aligned with real-world taskcompletion scenarios, extending beyond existing 3D visual grounding and question-answering datasets (Chen, Chang, and Nießner 2020; Kang et al. 2025; Zhu et al. 2024a). It is characterized by the inclusion of OR knowledge, which is not considered in existing related 3D understanding datasets.

- Fig. 4(b) presents a data example from the ORS3D-60K

dataset. The composite task comprises a list of subtasks that the embodied agent must complete. The solution consists of step-by-step actions with target object locations. At each step, the model is required to simultaneously produce an action description of the operation on a subtask and locate the target object in the 3D scene.

Second, as shown in Tab. 1, our dataset has an exceptionally high average text length of 311 words, which poses a

###### Scene queries 𝑄 = 𝑞

[Figure 72]

Task type identification

###### Input scene point cloud

Scene token Text token <SCH> token <GRU> token

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Non-parallelizable subtasks:

[Figure 78]

[Figure 79]

[Figure 80]

Non-parallelizable subtasks: SUBTASK 1, SUBTASK 3, SUBTASK 4. Parallelizable subtasks: SUBTASK 2

[Figure 81]

LoRA

- SUBTASK 1, SUBTASK 3, SUBTASK 4 Parallelizable subtasks:
- SUBTASK 2

[Figure 82]

𝑇

𝐅

[Figure 83]

###### 3D scene encoder

[Figure 84]

- Stage1input

Frozen param. Trainable param.

- Stage2input

[Figure 85]

[Figure 86]

[Figure 87]

…

Input text instruction

[Figure 88]

Optimization solver

[Figure 89]

[Figure 90]

Please organize and clean the kitchen area while ensuring optimal functionality of the appliances, including SUBTASK 1 (10 mins), SUBTASK 2 (35 mins), SUBTASK 3 (24 mins), SUBTASK 4 (4 mins).

Grounding masks

𝑇

[Figure 91]

Large language model

[Figure 92]

[Figure 93]

[Figure 94]

…

|[Figure 95]<br><br>Schedule: 2→1→3→2→4|
|---|

[Figure 96]

[Figure 97]

[Figure 98]

Schedule from solver Step-wise solution

3D grounding head

[Figure 99]

1. First, I will start the microwave <GRU> to cook food…; 2. While the microwave is operating, I will wash the oblong-shaped sink <GRU> …

𝐅

𝑇

I will complete the subtasks in the optimal order: start subtask 2, subtask 1, subtask 3, recheck subtask 2, subtask 4.

1. First, I will start the microwave

[Figure 100]

[Figure 101]

[Figure 102]

to cook food… 2. While the oblong- …

[Figure 103]

𝐺

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

𝑄

- Figure 6: Overview of GRANT. The scene point cloud is processed by a 3D scene encoder into scene tokens. GRANT first infers task properties (stage 1), then uses a scheduling token to generate an optimal schedule (stage 2). The grounding tokens are fed to the 3D grounding head to generate object masks. The input task description is simplified for brevity.

Point cloud tokenization. For a scene point cloud P ∈ RN×6, where each point contains 6-dimensional information [x,y,z,r,g,b] and N is the number of points, a sparse convolutional network is employed to extract point-wise features F ∈ RN×d, where d is the feature dimension. We then use a pre-trained 3D scene encoder to further encode the point cloud features into scene tokens. The 3D scene encoder E employs a fixed set of K learnable scene queries Q = {qi}Ki=1, which interact with point cloud features via cross-attention to produce processed scene queries containing rich semantic information. This process can be formulated as:

significant challenge for the language processing capabilities of embodied agents. Our dataset contains 60,825 composite tasks across 4,376 scenes, representing the largest scale among existing related datasets. Besides, as shown in

- Fig. 5(a), our dataset covers different levels of difficulty, reflected by the varying number of subtasks (4 to 7) in each composite task. The expected time for each subtask (Fig. 5(b)) follows a long-tail distribution, reflecting realworld variability. Furthermore, in our ORS3D setting, 3D grounding is entangled within the text, requiring the model not only to understand what actions to take, but also to accurately identify where each action should occur in the 3D scene. These characteristics make our dataset large-scale, diverse in task complexity, and highly representative of realworld scenarios.

###### Qˆ = E(Q,F), (1)

where Qˆ = {qˆi}Ki=1 is the processed scene queries. To align with the token embedding dimension D of the LLM, the

### 4 Method

The ORS3D task requires integration of language understanding, efficiency optimization, and spatial perception in 3D environments. Therefore, we propose a grounded task scheduling agent, termed GRANT, where the overall architecture is illustrated in Fig. 6. Specifically, GRANT consists of four components: 1) A 3D scene encoder converting point clouds into scene tokens. 2) An LLM processing multimodal inputs for task understanding. 3) A scheduling token mechanism (STM) connecting the LLM to an optimization solver for efficient scheduling. 4) A grounding head generating point masks for object localization. The scene encoder and grounding head integrate with the LLM through specialized tokens, enabling end-to-end training for simultaneous schedule generation and spatial grounding.

processed scene queries are projected into scene tokens Ts = {si}Ki=1 via a simple linear layer, where each si ∈ R1×D represents a scene token.

Text tokenization. We employ a text tokenizer to convert the input composite task description into a sequence of text tokens Tt = {xi}Li=1, where each xi ∈ R1×D and L denotes the length of the input text.

LLM processing. The LLM plays a central role in our model by handling multi-modal inputs and understanding both point clouds and human instructions. It further identifies subtask types, solves complex task scheduling problems, generates descriptive action steps, and provides 3D positions of target objects in a unified manner. As shown in Fig. 6, the scene tokens are prepended to the text tokens and fed to the LLM, which generates output tokens in an auto-regressive manner. The output tokens include specially designed tokens for task scheduling and 3D grounding, which will be elaborated in the following sections.

- 4.1 Multi-modal Input Processing GRANT takes both the scene point cloud and the textual description as input, which are first tokenized and subsequently fed into the LLM for unified understanding.

Algorithm 1: Optimization-based Scheduling Solver (single parallelizable subtask)

- 1: Input: I = {(τi, ci, ti)}ni=1
- 2: Output: Schedule S∗
- 3: Split subtasks into one parallelizable subtask τP (if any) and non-parallelizable set SP

- 4: if no parallelizable subtask then
- 5: S∗ ← SP ▷ Purely sequential

- 6: else
- 7: TP ← duration of τP
- 8: Select Sin ⊆ SP s.t. τ

i∈Sin ti ≤ TP and the sum is maximized

- 9: Sout ← SP \ Sin

- 10: S∗ ← Sout + [τP] + Sin + [τP]
- 11: end if
- 12: return S∗

#### 4.2 Scheduling Token Mechanism

LLMs exhibit strong capabilities in natural language generation but are generally less effective at solving complex mathematical problems. To address this limitation, we introduce a special <SCH> token that connects to an external solver to obtain an optimal scheduling list. This list is then utilized to guide the LLM for step-wise action generation. Specifically, for a composite task description, the LLM first identifies the parallelizable and non-parallelizable subtasks (defined in Sec. 3.1). It then constructs the subtask type information as I = {(τi,ci,ti)}ni=1, where ci ∈ {P,P} denotes the subtask type (P: parallelizable, P: non-parallelizable) and ti is the expected time.

The information I is passed to an external optimization solver via the <SCH> token. As defined in Alg. 1, the solver minimizes the total execution time given the subtask types and their expected times. This is formulated as a 0–1 knapsack problem, where the waiting interval of a parallelizable subtask plays the role of the capacity and the durations of non-parallelizable subtasks serve as item weights and values, so that the solver maximizes the utilization of the waiting time of parallelizable subtasks while minimizing the overall completion time.

A simple dynamic programming algorithm is employed to solve this problem. The solver finally returns the optimal schedule of subtask IDs. This scheduling process can be formulated as:

S∗ = Solver(I), (2) where S∗ is the optimal subtask completion schedule, represented as a list of subtask IDs. Then, S∗ is converted into natural language using predefined templates, then tokenized into Tl by the text tokenizer and concatenated with the preceding tokens to guide the LLM in generating step-wise action descriptions.

#### 4.3 3D Grounding Head

Besides generating action descriptions, the model also needs to simultaneously locate the corresponding target object in order to complete the task in the physical world. To achieve this, we use a special <GRU> token to indicate the target object for grounding in the output of LLM. To align with the

dimension of the processed scene queries, all output <GRU> tokens are passed through a simple MLP head into G = {gj}sj=1, where each gj ∈ R1×d.

The target scene query is selected through max cosine similarity. Specifically, we compute the cosine similarity between gj and each qˆi. The scene query with the highest probability is selected as the best match one, denoted as q∗ ∈ Rd. This process can be formulated as:

gj · qˆi ∥gj∥ · ∥qˆi∥

q∗ = arg max qˆi∈Qˆ

, (3)

The grounding mask is generated by the matched scene query with point cloud features. The mask corresponding to the scene query q∗ is computed by taking the dot product between q∗ and the point cloud features, followed by a sigmoid activation to obtain a point mask, which is expressed as:

m = σ (F · q∗), (4)

where m ∈ RN is the predicted point mask of target object. Training objectives. For language modeling, we use nexttoken prediction with cross-entropy loss. For grounding, we align grounding tokens and scene queries via a similarity matrix and supervise it using a binary correspondence matrix with sigmoid focal loss.

### 5 Experiments

#### 5.1 Implementation Details

The 3D scene encoder of GRANT is initialized with a pretrained CLASP (Chen et al. 2024), freezing all weights except the projection layer used for alignment. We use Tiny-Vicuna1B (Chiang et al. 2023) as the LLM and fine-tune it using LoRA (Hu et al. 2021). We use the AdamW optimizer with a cosine learning rate scheduler (initialized as 8 × 10−4) and a weight decay of 0.1. Models are trained for 10 epochs on the ORS3D-60K training set with a batch size of 1. All experiments are conducted on 8× RTX 4090 GPUs.

#### 5.2 Evaluation Metrics

The model performance is evaluated across three aspects that align with the challenges of the ORS3D task. For output language quality, we use NLP metrics (METEOR & ROUGE). For 3D grounding accuracy on the target object at each step, we adopt the AP@25% detection metric. Considering that the core aspect of the ORS3D task is scheduling, we introduce the Time Efficiency (TE) metric to measure how well the model utilizes the time intervals of parallelizable subtasks. Rather than using raw completion time, TE normalizes the efficiency of each schedule between a naive sequential baseline and the optimal schedule. Formally, the TE of a predicted task schedule is calculated as:

TE = Tworst − Tpred Tworst − Topt

× 100%, (5)

where Tpred is the total time of the predicted task schedule, Topt is the total time of the ground-truth optimal schedule obtained by the OR solver, and Tworst is the total time when all subtasks are executed sequentially without any parallelism.

Language Scheduling 3D Grounding

Method Venue 3D Obj. Det. LLM

Overall ↑ METEOR ↑ ROUGE ↑ TE ↑ Accuracy ↑

Commercial LLM/MLLMs (only text input) Gemini - - Gemini-2.0-flash 41.67 58.48 24.75

31.22 DeepSeek-R1 (Guo et al. 2025) - - DeepSeek-V3 32.40 41.50 72.63 36.63 GPT-4o - - GPT-4o 49.16 62.19 45.27 39.15

Unsupported

Object-level methods (with detected object proposals*) 3D-VisTA (Zhu et al. 2023) ICCV 23 Mask3D -

54.90‡ 13.73 PQ3D (Zhu et al. 2024b) ECCV 24 Mask3D - 56.12‡ 14.03 LEO† (Huang et al. 2024b) ICML 24 Mask3D Vicuna-1B 46.61 60.32 45.63 Unsupported 38.14

Unsupported

###### Scene-level methods

Grounded 3D LLM (Chen et al. 2024) arXiv 24 - Vicuna-1B 41.96 53.71 42.46 34.00 43.03 GRANT (ours) - - Vicuna-1B 42.82 62.78 72.99 35.38 53.49

- Table 2: Experiment results on ORS3D-60K test set. † We adapt LEO by replacing its LLM with Vicuna-1B for a fair comparison.

* indicates that these methods require object point clouds from an external 3D detector like Mask3D (Schult et al. 2023). ‡ Results are produced by directly providing step-wise schedules as input. Overall is the average of METEOR, ROUGE, TE, and Grounding Accuracy (treating unsupported metrics as 0).

Method 3D Obj. Det.

Detection Segmentation AP@0.25 AP@0.50 mIoU

3D-VisTA* Mask3D 54.90 41.88 43.29 PQ3D* Mask3D 56.12 44.01 46.37

Grounded 3D LLM - 34.00 23.93 25.56 GRANT (ours) - 35.38 24.79 26.71

(a) 3D grounding performance comparison

Method Acc.

Para. subtask Non-para. subtask Scheduling

Prec. Recall F1 Prec. Recall F1 TE

Grounded 3D LLM 77.14 73.80 50.15 59.72 95.17 82.66 88.48 42.46 LEO 79.73 78.19 41.57 54.28 95.90 87.37 91.43 45.63 GRANT (ours) 84.65 73.82 54.70 62.84 95.94 90.67 93.23 72.99

(b) Subtask type recognition and scheduling

- Table 3: (a) Comprehensive 3D grounding performance. * indicates that these methods require object point clouds from an external 3D detector like Mask3D (Schult et al. 2023). (b) Impact of subtask type recognition on scheduling efficiency.

Intuitively, the numerator Tworst − Tpred measures the time saved by the model compared to the naive baseline, while

the denominator Tworst − Topt is the maximum possible saving for that task instance. Thus, TE reflects the fraction of the theoretically achievable time savings that the model actually realizes, with TE = 0% indicating purely sequential execution and TE = 100% matching the optimal schedule.

#### 5.3 Main Results

We conduct a comprehensive comparison between our proposed GRANT and existing methods on ORS3D-60K dataset, as reported in Tab. 2. We evaluate commercial LLM/MLLMs by providing only the text part of the instructions, as they only support images as visual input. Notably, DeepSeekR1 demonstrates strong performance on task scheduling (TE 72.63%) due to reinforcement learning on mathematical problems. However, these models cannot process point cloud data or directly locate objects in 3D environments, which limits their applicability in embodied scenarios.

Object-level methods (Zhu et al. 2023, 2024b) process on object point clouds obtained from 3D detectors like Mask3D (Schult et al. 2023). While these methods achieve high grounding accuracy, they are limited by their inability to handle long textual inputs and generate multimodal

outputs. LEO (Huang et al. 2024b) integrates an LLM for enhanced language performance but lacks target object grounding. These methods focus on object-centric 3D understanding, which is insufficient for task-driven embodied scenarios where an agent requires a comprehensive understanding of the entire 3D environment. For scene-level methods, we use Grounded 3D LLM (Chen et al. 2024) as the baseline. By introducing the STM, our model achieves a substantial gain (30.53%) in task scheduling and further boosts 3D grounding by 1.38%. Overall, our method consistently outperforms baseline methods, validating its effectiveness across language understanding, 3D grounding, and scheduling efficiency.

We also compare the 3D target object grounding performance of different models in Tab. 3(a). Object-level methods achieve higher performance due to the use of additional 3D object detectors. However, their performance heavily depends on the detector’s capability, introducing extra complexity in preprocessing 3D point clouds and leading to the loss of full scene information. In contrast, scene-level methods offer a cleaner alternative for 3D grounding by directly processing the entire scene point cloud, making them more suitable for real-world applications.

Accurate recognition of parallelizable subtasks is a prerequisite for effective time scheduling. As shown in Tab. 3(b),

Language Scheduling 3D Grounding METEOR ROUGE TE Accuracy

Subtask number

Setting

Method

Overall Four Five Six Seven

No scheduling content 35.60 48.89 21.03 15.95 + Scheduling content 41.29 55.28 47.04 34.74 + STM (ours) 42.82 62.78 72.99 35.38 GT scheduling content 53.34 75.06 90.29 38.52

PQ3D (Zhu et al. 2024b) 14.82 14.15 13.40 13.73 14.03 LEO (Huang et al. 2024b) 42.14 40.12 36.42 33.91 38.14 Grounded 3D LLM (Chen et al. 2024) 54.35 45.13 36.59 36.04 43.03 GRANT (ours) 60.23 52.98 52.03 48.70 53.49

(a) Effect of scheduling token mechanism

(b) Performance across task difficulty levels

Language Scheduling 3D Grounding METEOR ROUGE TE Accuracy

# LLM Params.

Subtask number 4 5 6 7 10 20 50

1B 42.82 62.78 72.99 35.38 7B 45.19 63.55 73.21 36.25

Runtime (ms) 1.14 1.28 1.31 1.42 1.49 2.01 3.94

(d) Optimization solver runtime

(c) Effect of scaling LLM

- Table 4: Ablation studies. (a) Effect of scheduling token mechanism. (b) Performance across different task difficulty levels. (c) Effect of scaling LLM. (d) Runtime of the optimization solver.

proposed ORS3D-60K dataset.

[Figure 109]

Please prepare the dining area for a family meal, including (1) verifying the kitchen cabinet close to the refrigerator and next to another cabinet is free of dust and dirt (5 minutes), (2) operating the white microwave placed higher than the kitchen counter for cooking (30 minutes), (3) washing the stainless sink located below the kitchen cabinets (9 minutes), (4) brushing down the paper towel dispenser on the kitchen counter and lower than the kitchen cabinets (7 minutes), (5) dusting the wooden kitchen cabinets stacked above the kitchen counter (9 minutes), (6) tending the dining table with a curved top in the middle of the cabinet area (11 minutes), (7) emptying the trash can located at the 11 o'clock direction from another cabinet (3 minutes).

Effect of scaling LLM. As summarized in Tab. 4(c), increasing the size of the LLM improves performance across language understanding, task scheduling, and 3D grounding. To balance performance and training cost, we adopt Vicuna1B as the default setting.

|Subtasks that require continuous attention to operate: subtask 1, subtask 3, subtask 4, subtask 5, subtask 6, subtask 7. Subtasks that do not need continuous attention to operate: subtask 2. <SCH> I will complete the subtasks in the optimal order: subtask 1, start subtask 2, subtask 3, subtask 5, subtask 6, recheck subtask 2, subtask 4, subtask 7.<br><br>[Figure 110]<br><br>First, I will verify the kitchen cabinet <GRU> close to the refrigerator and next to another cabinet is free of dust and dirt. (subtask 1)<br><br>[Figure 111]<br><br>Next, I will start the white microwave <GRU> higher than the kitchen counter for cooking. (start subtask 2)<br><br>While the microwave is running, I will wash the stainless sink <GRU> located<br><br>below the kitchen cabinets. (subtask 3)<br><br>[Figure 112]<br><br>Finally, I will empty the<br><br>trash can <GRU> located at the 11 o'clock direction from another cabinet. (subtask 7)<br><br>[Figure 113]<br><br>Subsequently, I will tune the dining table <GRU> with a curved top in the middle of the cabinet area. (subtask 6)<br><br>[Figure 114]<br><br>Before 30 minutes are up, I will return to check and finish operating the microwave <GRU>. (recheck subtask 2)<br><br>Then, I will brush down the paper towel dispenser <GRU> on the kitchen counter and lower than the kitchen cabinets. (subtask 4)<br><br>[Figure 115]<br><br>Subsequently, I will dust the wooden kitchen cabinet <GRU> stacked above the kitchen counter. (subtask 5)<br><br>Total completion time = 45 mins. 39% quicker compared to sequential completion (74 mins)<br><br>[Figure 116]<br><br>[Figure 117]<br><br>5 6 7 8<br><br>[Figure 118]<br><br>1 2 3 4|
|---|

Solver runtime. As the number of subtasks increases, the optimization solver remains extremely fast (Tab. 4(d)), where even with 50 subtasks the total runtime of the solver stays below 4 ms, introducing virtually no overhead.

Qualitative analysis. Fig. 7 shows a representative example where our model identifies subtask 2 (microwave operation) as parallelizable and schedules other subtasks during its 30-minute waiting period, saving 29 minutes (39% efficiency gain) compared to sequential execution. The model also accurately localizes target objects with high IoU, demonstrating effective integration of OR-based scheduling and spatial grounding.

- Figure 7: A qualitative example of GRANT. In the visualized point clouds, yellow shows correct predictions, red indicates false positives, and green marks missed ground truth regions.

#### 5.5 Limitation

While this work demonstrates strong performance on the ORS3D-60K benchmark, future work will deploy the framework on physical robots to validate robustness in dynamic environments. Additionally, we will explore integrating the external optimization solver directly within the language model to enable end-to-end differentiable reasoning.

our model achieves the highest accuracy in subtask type recognition, with substantial improvements in recall and F1score for parallelizable subtasks, which in turn leads to significantly higher time efficiency. Therefore, robust subtask type recognition is critical for enhancing scheduling performance.

#### 5.4 Ablation Studies

### 6 Conclusion

Effect of STM. As shown in Tab. 4(a), adding scheduling content substantially improves performance, demonstrating the importance of explicit scheduling. The proposed STM further boosts time efficiency by 25.95%, highlighting the critical role of proper scheduling. Using ground-truth constraints indicates that more accurate subtask recognition enables additional performance gains.

In this work, we introduce the ORS3D task that integrates task scheduling from operations research with spatial grounding for embodied agents. We construct the ORS3D-60K dataset, and propose GRANT, an embodied 3D MLLM equipped with the scheduling token mechanism to generate efficient task schedules with grounded actions. Experiments validate the capability of GRANT across language, grounding, and task scheduling. We believe our initial work on the OR knowledgeintensive scenario can inspire future research to further improve the complex planning capabilities and multi-modal integration of embodied agents.

Task difficulty levels. As shown in Tab. 4(b), performance consistently declines as the number of subtasks increases, indicating that all methods degrade with more complex task structures. This trend highlights the increased challenge in task scheduling and reflects the inherent complexity of the

### Acknowledgments

This work was supported by the NSFC (Grant No. 62225603 and 623B2038) and in part by the Hubei Provincial Technology Innovation Program (Grant No. 2024BAA007).

### References

Ahn, M.; Brohan, A.; Brown, N.; Chebotar, Y.; Cortes, O.; David, B.; Finn, C.; Fu, C.; Gopalakrishnan, K.; Hausman, K.; et al. 2022. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691.

Baruch, G.; Chen, Z.; Dehghan, A.; Dimry, T.; Feigin, Y.; Fu, P.; Gebauer, T.; Joffe, B.; Kurz, D.; Schwartz, A.; et al. 2021. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. Proceedings of the Advances in Neural Information Processing Systems.

Chen, D. Z.; Chang, A. X.; and Nießner, M. 2020. Scanrefer: 3d object localization in rgb-d scans using natural language. In Proceedings of European Conference on Computer Vision, 202–221.

Chen, Y.; Ge, Y.; Ge, Y.; Ding, M.; Li, B.; Wang, R.; Xu, R.; Shan, Y.; and Liu, X. 2023. Egoplan-bench: Benchmarking multimodal large language models for human-level planning. arXiv preprint arXiv:2312.06722.

Chen, Y.; Sun, Y.; Chen, X.; Wang, J.; Shen, X.; Li, W.; and Zhang, W. 2025. Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning. arXiv preprint arXiv:2503.06232.

Chen, Y.; Yang, S.; Huang, H.; Wang, T.; Xu, R.; Lyu, R.; Lin, D.; and Pang, J. 2024. Grounded 3d-llm with referent tokens. arXiv preprint arXiv:2405.10370.

Chiang, W.-L.; Li, Z.; Lin, Z.; Sheng, Y.; Wu, Z.; Zhang, H.; Zheng, L.; Zhuang, S.; Zhuang, Y.; Gonzalez, J. E.; Stoica, I.; and Xing, E. P. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality.

Choi, J.-W.; Yoon, Y.; Ong, H.; Kim, J.; and Jang, M. 2024. LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents. Proceedings of the International Conference on Learning Representations.

Deng, J.; He, T.; Jiang, L.; Wang, T.; Dayoub, F.; and Reid, I. 2025. 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer. Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Driess, D.; Xia, F.; Sajjadi, M. S. M.; Lynch, C.; Chowdhery, A.; Ichter, B.; Wahid, A.; Tompson, J.; Vuong, Q.; Yu, T.; Huang, W.; Chebotar, Y.; Sermanet, P.; Duckworth, D.; Levine, S.; Vanhoucke, V.; Hausman, K.; Toussaint, M.; Greff, K.; Zeng, A.; Mordatch, I.; and Florence, P. 2023. PaLM-E: An Embodied Multimodal Language Model. In Proceedings of the International Conference on Machine Learning.

Duan, J.; Yu, S.; Tan, H. L.; Zhu, H.; and Tan, C. 2022. A survey of embodied ai: From simulators to research tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(2): 230–244.

Fu, H.; Zhang, D.; Zhao, Z.; Cui, J.; Liang, D.; Zhang, C.; Zhang, D.; Xie, H.; Wang, B.; and Bai, X. 2025. Orion: A

holistic end-to-end autonomous driving framework by visionlanguage instructed action generation. In Proceedings of IEEE/CVF International Conference on Computer Vision.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; and Xu,

- R. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Nature.

Hong, Y.; Zhen, H.; Chen, P.; Zheng, S.; Du, Y.; Chen, Z.; and Gan, C. 2023. 3d-llm: Injecting the 3d world into large language models. Proceedings of the Advances in Neural Information Processing Systems, 20482–20494.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang,

- S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Huang, H.; Chen, Y.; Wang, Z.; Huang, R.; Xu, R.; Wang,
- T.; Liu, L.; Cheng, X.; Zhao, Y.; Pang, J.; et al. 2024a. Chatscene: Bridging 3d scene and large language models with object identifiers. In Proceedings of the Advances in Neural Information Processing Systems.

- Huang, J.; Yong, S.; Ma, X.; Linghu, X.; Li, P.; Wang, Y.; Li, Q.; Zhu, S.-C.; Jia, B.; and Huang, S. 2024b. An Embodied Generalist Agent in 3D World. In Proceedings of the International Conference on Machine Learning.
- Huang, K.-C.; Li, X.; Qi, L.; Yan, S.; and Yang, M.-H. 2024c. Reason3d: Searching and reasoning 3d segmentation via large language model. arXiv preprint arXiv:2405.17427.

Jia, B.; Chen, Y.; Yu, H.; Wang, Y.; Niu, X.; Liu, T.; Li, Q.; and Huang, S. 2024. SceneVerse: Scaling 3D VisionLanguage Learning for Grounded Scene Understanding. In Proceedings of European Conference on Computer Vision.

Jiang, X.; Lu, L.; Shao, L.; and Lu, S. 2024. Multimodal 3D Reasoning Segmentation with Complex Scenes. arXiv preprint arXiv:2411.13927.

Kang, W.; Huang, H.; Shang, Y.; Shah, M.; and Yan, Y. 2024. Robin3D: Improving 3D Large Language Model via Robust Instruction Tuning. arXiv preprint arXiv:2410.00255.

Kang, W.; Qu, M.; Kini, J.; Wei, Y.; Shah, M.; and Yan, Y. 2025. Intent3D: 3D Object Detection in RGB-D Scans Based on Human Intention. In Proceedings of the International Conference on Learning Representations.

Kolodiazhnyi, M.; Vorontsova, A.; Konushin, A.; and Rukhovich, D. 2024a. Oneformer3d: One transformer for unified point cloud segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20943–20953.

Kolodiazhnyi, M.; Vorontsova, A.; Skripkin, M.; Rukhovich, D.; and Konushin, A. 2024b. UniDet3D: Multi-dataset Indoor 3D Object Detection. arXiv preprint arXiv:2409.04234.

Liang, D.; Feng, T.; Zhou, X.; Zhang, Y.; Zou, Z.; and Bai, X.

- 2025a. Parameter-efficient fine-tuning in spectral domain for point cloud learning. IEEE Transactions on Pattern Analysis and Machine Intelligence. Liang, D.; Hua, W.; Shi, C.; Zou, Z.; Ye, X.; and Bai, X.
- 2025b. Sood++: Leveraging unlabeled data to boost oriented object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Liang, D.; Zhou, X.; Xu, W.; Zhu, X.; Zou, Z.; Ye, X.; Tan, X.; and Bai, X. 2024. Pointmamba: A simple state space model for point cloud analysis. In Proceedings of the Advances in Neural Information Processing Systems, volume 37, 32653– 32677.

Lin, B. Y.; Huang, C.; Liu, Q.; Gu, W.; Sommerer, S.; and Ren, X. 2023. On grounded planning for embodied tasks with language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 13192–13200.

Mao, Y.; Zhang, Y.; Jiang, H.; Chang, A.; and Savva, M. 2022. MultiScan: Scalable RGBD scanning for 3D environments with articulated objects. Proceedings of the Advances in Neural Information Processing Systems, 35: 9058–9071.

Ramakrishnan, S. K.; Gokaslan, A.; Wijmans, E.; Maksymets, O.; Clegg, A.; Turner, J.; Undersander, E.; Galuba, W.; Westbury, A.; Chang, A. X.; et al. 2021. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. Proceedings of the Advances in Neural Information Processing Systems.

Rozenberszki, D.; Litany, O.; and Dai, A. 2022. Languagegrounded indoor 3d semantic segmentation in the wild. In Proceedings of European Conference on Computer Vision, 125–141.

Schult, J.; Engelmann, F.; Hermans, A.; Litany, O.; Tang, S.; and Leibe, B. 2023. Mask3d: Mask transformer for 3d semantic instance segmentation. In 2023 IEEE International Conference on Robotics and Automation, 8216–8223. IEEE.

Takmaz, A.; Fedele, E.; Sumner, R. W.; Pollefeys, M.; Tombari, F.; and Engelmann, F. 2023. OpenMask3D: OpenVocabulary 3D Instance Segmentation. In Proceedings of the Advances in Neural Information Processing Systems.

Wald, J.; Avetisyan, A.; Navab, N.; Tombari, F.; and Nießner, M. 2019. Rio: 3d object instance re-localization in changing indoor environments. In Proceedings of IEEE/CVF International Conference on Computer Vision, 7658–7667.

Wang, T.; Mao, X.; Zhu, C.; Xu, R.; Lyu, R.; Li, P.; Chen, X.; Zhang, W.; Chen, K.; Xue, T.; et al. 2024. EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI. Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wang, Z.; Huang, H.; Zhao, Y.; Zhang, Z.; and Zhao, Z. 2023. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769.

Wu, Z.; Wang, Z.; Xu, X.; Lu, J.; and Yan, H. 2023. Embodied task planning with large language models. arXiv preprint arXiv:2307.01848.

- Xu, G.; Wang, X.; Zhang, Z.; Cheng, J.; Liao, C.; and Yang, X. 2025a. Igev++: Iterative multi-range geometry encoding volumes for stereo matching. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Xu, G.; Wang, Y.; Cheng, J.; Tang, J.; and Yang, X. 2023. Accurate and efficient stereo matching via attention concatenation volume. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4): 2461–2474.

Xu, W.; Shi, C.; Tu, S.; Zhou, X.; Liang, D.; and Bai, X. 2025b. A unified framework for 3d scene understanding. Advances in Neural Information Processing Systems, 37: 59468– 59490.

Zhang, Z.; Zhu, Z.; Li, P.; Liu, T.; Ma, X.; Chen, Y.; Jia, B.; Huang, S.; and Li, Q. 2024. Task-oriented sequential grounding in 3d scenes. arXiv preprint arXiv:2408.04034.

Zhou, X.; Liang, D.; Tu, S.; Chen, X.; Ding, Y.; Zhang, D.; Tan, F.; Zhao, H.; and Bai, X. 2025. Hermes: A unified selfdriving world model for simultaneous 3d scene understanding and generation. In Proceedings of IEEE/CVF International Conference on Computer Vision.

Zhu, C.; Wang, T.; Zhang, W.; Chen, K.; and Liu, X. 2024a. Scanreason: Empowering 3d visual grounding with reasoning capabilities. In Proceedings of European Conference on Computer Vision, 151–168.

Zhu, C.; Wang, T.; Zhang, W.; Pang, J.; and Liu, X. 2025. LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness. In Proceedings of IEEE/CVF International Conference on Computer Vision.

Zhu, Z.; Ma, X.; Chen, Y.; Deng, Z.; Huang, S.; and Li, Q. 2023. 3D-VisTA: Pre-trained transformer for 3D vision and text alignment. In Proceedings of IEEE/CVF International Conference on Computer Vision, 2911–2921.

Zhu, Z.; Zhang, Z.; Ma, X.; Niu, X.; Chen, Y.; Jia, B.; Deng, Z.; Huang, S.; and Li, Q. 2024b. Unifying 3D VisionLanguage Understanding via Promptable Queries. In Proceedings of European Conference on Computer Vision.

