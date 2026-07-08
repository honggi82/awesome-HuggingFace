## MMFactory: A Universal Solution Search Engine for Vision-Language Tasks

Wan-Cyuan Fan1,2 Tanzila Rahman1,2 Leonid Sigal1,2,3 1University of British Columbia 2Vector Institute for AI 3CIFAR AI Chair

{wancyuan, trahman8, lsigal}@cs.ubc.ca

# arXiv:2412.18072v1[cs.CV]24Dec2024

### Abstract

Performance

n

| |Sol 1<br><br>Sol 2<br><br>Sol 4<br><br>Sol 3|
|---|---|
| | |

[Figure 1]

With advances in foundational and vision-language (VLM) models, and effective fine-tuning techniques, a large number of both general and special-purpose models have been developed for a variety of visual tasks. Despite the flexibility and accessibility of these models, no single model is able to handle all tasks and/or applications that may be envisioned by potential users. Recent approaches, such as visual programming and multimodal LLMs with integrated tools aim to tackle complex visual tasks, by way of program synthesis. However, such approaches overlook user constraints (e.g., performance / computational needs), produce test-time sample-specific solutions that are difficult to deploy, and, sometimes, require low-level instructions (e.g., code snippets for similar problems) that maybe beyond the abilities of a naive user. To address these limitations, we introduce MMFactory, a universal framework that includes model and metrics routing components, acting like a solution search engine across various available models. Based on a task description and few sample input-output pairs and (optionally) resource and/or performance constraints, MMFactory can suggest a diverse pool of programmatic solutions by instantiating and combining visio-lingual tools (e.g., detection, segmentation, VLMs) from its model repository. In addition to synthesizing these solutions, MMFactory also proposes metrics and benchmarks performance / resource characteristics, allowing users to pick a solution that meets their unique design constraints. From the technical perspective, we also introduced a committee-based solution proposer that leverages multi-agent LLM conversation to generate executable, diverse, universal, and robust solutions for the user. Experimental results show that MMFactory outperforms existing methods by delivering stateof-the-art solutions tailored to user problem specifications.

x 1

Computation time

Proposing multiple sols

- Sol 1

- Sol 2

- Sol 3

- Sol 4

###### Prompt 1

n selected instances

Grounding models OCR models Depth models Others

General LLMs

###### Sampling

General VLMs Others

Code LLMs Others

Vision models

Language models V-L models

###### (a) Ours

N

…

Main model (LM / VLM)

[Figure 2]

x N

API tools

Proposing only one sol

(b) MLLM w/ tools

…

x N

LM or VLM models

Prompt 1

Selecting one model

All instances in the user task

(c) Model routing

Figure 1. Illustration of MMFactory. Proposed MMFactory framework (a) contrasted with model routing approaches (c) and multimodal LLM with tools (b). Unlike both prior classes of methods, MMFactory proposes a pool of programmatic solutions, composed of series of selected models from the pool, for a given task while also benchmarking their performance and computational characteristics. See Section 1 for full discussion.

these models, a wide range of vision-language (VLM) or multimodal LLMs (MLLMs) [1, 5, 7, 35, 62] have been developed by integrating modality adapters and encoders into their frameworks. This advancement has resulted in stateof-the-art models capable of solving complex visual tasks.

Despite the push for building AGI-like agents, that are all capable, even models like GPT-4o tend to be inferior, or lacking, on specific tasks [19, 31]. At the same time, with the development of fine-tuning techniques, customized or expert models tailored to specific tasks have become easier to develop. With different training data, fine-tuning approaches, and frameworks, models with varying specialties and characteristics are being introduced daily. One can imagine that in near future such models will be ubiquitous,

### 1. Introduction

Large language models (LLMs), such as GPT [2] and Gemini [51, 52], have demonstrated powerful capabilities across various domains, significantly transforming how people approach their tasks and even their daily lives. Building on

creating a marketplace of agents with an overwhelming design choices for users to pick from and build on. In this scenario, routing approaches are needed that can take userdefined tasks, needs, and constraints, acting as a search engine among all types of models, to provide suggested solutions for the user.

Previous works in visual programming [22, 49] and multimodal language models (MLLMs) with tool integration [26, 38, 45] have explored using LLMs as planners to utilize external tools or APIs for solving complex visual tasks or to decompose tasks into sub-tasks. While these approaches have shown promise, there are several limitations to consider. First, existing methods assume a single specialized tool for a given sub-task (e.g., detection [36], segmentation [30], depth estimation [59]). This is overly simplistic, as a variety of tools exist for any one sub-task, inculcating within a particular family of models, that differ by backbone, number of parameters and overall performance. Second, these works generally overlook the user’s specific computation needs and constraints when generating solutions, resulting in inability to tailor solutions to particular hardware or deployment cost (e.g., a user maybe willing to forgo 1% better performance if inference cost is reduced by 50%). Third, the proposed solutions are often tailored per specific example or scenario, which limits their generalization and applicability to other examples in the task, as shown in Fig. 1. Deployment of such solutions is problematic (e.g., no constant code path exist that maybe distilled to a small model executable on an edge device). Addressing these limitations is essential for creating more versatile and user-centric framework for routing the solutions among different kinds of models in order to create custom agents capable of solving specific user problems in accordance to their specification.

To address these challenges, in this work, we introduce MMFactory – a universal framework for automatic and programmatic development of task-specific agents. MMFactory (Fig. 1a) includes a model and metric routing components; that, in combination, act as a solution search engine for non-expert users. Based on a task description (e.g., comparison of depth of points in an image), a few sample input-output pairs (e.g., set of images with labeled points and which point is closest to camera in each), and (optionally) resource and/or performance constraints (e.g., compute limit), MMFactory can suggest a diverse pool of programmatic solutions by instantiating and combining visual, LLM and VLM tools from its repository. In addition to synthesizing these solutions, MMFactory also proposes metrics and benchmarks performance / resource characteristics, allowing users to pick a solution that meets their unique design constraints. From the technical perspective, we also introduced a committee-based solution proposer that leverages multi-agent LLM conversation to generate executable,

diverse, universal, and robust solutions for the user.

Notably, unpublished and concurrent work of [41] also explores the idea of routing, but mainly for choosing a single (most accurate) among the K possible LLM / VLM models (see Fig. 1c). MMFactory framework is considerably more general and provides user with family of solutions and their performance characterization. In addition, our solutions, similar to visual programming [22, 49], are drawn from an exponential set of tools that can work in tandem with one another. Further, the fact that our framework proposes solutions that contain a single executable code path, makes them much easier to deploy.

Our contributions are multiple fold. First, to the best of our knowledge, this work is the first to explore routing across vision, language, and vision-language models. Second, our propose framework can provide multiple solutions in a solution pool for user-defined tasks and constraints. Third, we introduce a novel approach that combines routing and a multi-agent solution proposer to deliver robust results. Fourth, unlike existing approaches, our proposed framework solves all instances of a user-defined task collectively, rather than generating separate solutions for each instance. Fifth, experiments on two benchmarks demonstrate that our framework outperforms the state-of-the-art.

### 2. Related works

Multimodal Large Language Models. Building on the recent success of large language models (LLMs) [4, 20, 42, 53], research trends have shifted toward enhancing these LLMs with multi-modal capabilities. Some of these MLLMs [34, 37, 39, 62] are created for general purpose, while others are designed for specific tasks, including coding [21, 27, 46], video understanding [13, 61], 3D [11, 24, 64], audio or speech [8, 15, 18], math [6, 54], scientific chart [17, 23, 40], and robotics [9, 60], which has demonstrated promising results. However, language-based models alone can’t handle complex tasks very well. Multimodal models, which combine text and images, also face challenges, like misinterpreting context when information is split across text and visuals. They might connect unrelated details or miss important clues, leading to errors. Therefore, researchers are exploring tools and interactive systems to improve their understanding of multimodal information.

Visual programming, LLMs with tools and Routing. As humans, when we face complex tasks, we decompose them into subtasks to understand them better or use tools to make them simpler. These concepts have been extended to neural networks, where previous works [3, 28] suggest that complex vision tasks are fundamentally compositional and can be divided into atomic perceptual units. Following this concept, visual programming [22, 49] and LLMs with tool [35, 44, 45] become prominent research

x 1

Three selected instances

m

Sampling

…

…

User preference > 50% accuracy

[Figure 3]

###### User tasks

[Figure 4]

Prompt 1

ANALYSIS: … THOUGHT:

m instances < 3 sec per image

Sampling

| | |
|---|---|
| | |
| | |

1. Parse the prompt via llm … ACTION:

- def sol_0(...)
- def sol_1(...)
- def sol_2(...)
- def sol_3(...)

n

[Figure 5]

…

|def sol_4(prompt, image_1): # Parse the prompt object_names = LLaMA … # Detect the objects image_boxes = detection… # Using the vlm response = LLaVA_7B_model… return response|
|---|

| | |
|---|---|
| | |
| | |

P

[Figure 6]

Evaluation

Inference

[Figure 7]

GPTScore

Solution router

[Figure 8]

RelaxedAcc

Solution pool

Metric router IT

[Figure 9]

Proposed solution

Proposed metrics

…

Performance curve Model pool

Prompt 1

n instances

###### Vision models Language models V-L models

CLIPscore RelaxedAcc Solution space Others GPT model

BLUE CIDEr

IOU

Metric pool

[Figure 10]

Segmentation

LLaMA Phi-3.5-mini InternLM LLaVA-7B and 13B

InstructBLIP

[Figure 11]

[Figure 12]

Acc GPTscore

[Figure 13]

Detection

[Figure 14]

[Figure 15]

PaliGemma

[Figure 16]

[Figure 17]

[Figure 18]

Depth

OCR tool

[Figure 19]

[Figure 20]

(I) Solution routing (II) Metric routing

Figure 2. Overview of MMFactory. Our framework includes two primary components: Solution Router and Metric Router. The Solution Router generates a pool of potential solutions for the task, while the Metric Router evaluates these solutions, estimating their performance and computational cost to generate a performance curve. This curve enables users to select the model optimal for their task requirements.

is AutoAgents [12], which introduces “observers” to monitor multi-agent conversations, helping ensure quality and coherence in responses. However, AutoAgents provides only one solution per prompt, while our approach offers multiple options with performance and cost details to help users choose the best fit. Additionally, AutoAgents relies on GPT-4’s reasoning, limiting its flexibility with open-source models and restricting it to tasks like open-ended questions and creative writing. Our system, in contrast, supports any open-source model and can handle a wide range of vision tasks. Most importantly, AutoAgents focus on dynamically creating multiple agents based on the task content and planning solutions. Our approach, however, focuses on solving tasks by routing different vision models and incorporating a Python coding environment, which AutoAgents have not explored.

trends. Practically, visual programming focus on leveraging LLMs’ coding ability to decompose complex tasks into multi-step Python code with specialized vision tools. On the other hand, LLMs with tool use focus on teaching LLMs to use various types of tools to achieve image generation/editing [35], accessing web engines [38, 44], operating systems [43], etc. However, these methods have a common problem that the multimodal modules are designed for specific tasks and can’t be reused for similar ones. They also don’t consider user constraints like model size, complexity, or preferences. This gave rise to routing-based approaches. In these approaches [25, 41, 47], a router model can switch between a stronger or weaker LLM during inference to balance cost with model performance. However, this method still needs the router LLM to be trained and can’t offer versatile solutions based on user needs. It also relies on a single model, which isn’t enough to solve a complex task efficiently. In contrast, our framework provides multiple options (i.e. solution pool) for users to choose from, and these options are versatile and can be reused across all instances of the task, rather than being limited to individual instance.

CHRIS: this framework figure needs to be revised...visualize solution pool] change colors of vision, language, vision and language model. gpt logo.

### 3. Methodology

#### 3.1. Overview of MMFactory

We introduce MMFactory, an universal framework designed not only to propose programmatic solutions based on user-defined task examples but also to provide estimated performance and time costs for each solution, allowing users to make informed choices. This framework functions like a solution search engine and interface across various models, enabling access to models for task-solving without requiring extensive background knowledge. MMFactory has several unique features. In addition to proposing multiple solutions with estimated performance and cost plots, the solutions generated are general and can be applied across all examples within the specified task. Specifically, MMFactory consists of two key components: the Solution Router and Metric Router. The former can generate multiple general solutions for solving the task, while

From multi-modal Agents to multi-agent frameworks. Recently, due to the powerful reasoning, tool usage, and other capabilities of LLMs, these models have become essential building blocks in the development of artificial intelligence agents [10, 32, 48, 56] for many real-world applications, such as medicine [50], general tasks [12], and robotics [29]. Given the increasing complexity of tasks, an intuitive approach is to enhance the capabilities of agents by incorporating multiple agents into the task solving. Previ-

- ous works have showcased that multi-agents conversations or debates can improve various capabilities, such as divergent thinking [33], factuality and reasoning abilities [16], and validation [57], and can even achieve automatic agent creation [12]. Among these, the most relevant to our work

#### 3.2. Inputs Structure for Solution Router

|TASK DEFINITION: In this task, you are given a prompt and two images. In the first image, there is only one point labeled with a red circle and REF tag. In the second image, there are four points labeled with red circle and a letter tag of A, B, C, and D. You have to … the second image corresponds to the point in the first image. You may have to know where these points are to answer the question. Here are three examples of the user task.|
|---|

As mentioned in the previous section, our framework is designed to propose multiple solutions that leverage models in the model pool to solve the task. The challenging aspects of this task is that the router must not only understand the task but also comprehend the details of each model in the pool to ensure correct use in the solution. For such complex task, in addition to the initial task prompt, we have to provide extra details for the router, including definitions of the models in the model pool, a requirements list, in-context examples, and the solution pool. For each task, the input prompt P structure is detailed below (examples can also be found in the supplementary.) consisting of task-agnostic information:

|EXAMPLES from the task:<br><br># EXAMPLE 0 #<br># EXAMPLE 1 #<br><br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br># TASK REQUEST PROMPT #: <img src='...'> <img src='...'> … Which point on … (A) Point A (B) Point B (C) Point C…<br><br>The correct answer is: (C)<br><br># TASK REQUEST PROMPT #: <Image> <Image> … Which point … (D) Point.<br><br>The correct answer is: (D)<br><br>…|
|---|

###### …

|(OPTIONAL) USER CONSTRAINTS: For example, execution time need to be less than 5 sec per sample, or models with fewer than 3B parameters…|
|---|

- • Model definitions Pd: Describes the details of each model in the model pool, including functionality, input arguments, return arguments, and example use cases.
- • Requirements Pr: A predefined list of requirements for the router to consider when generating solutions.
- • In-context examples Pe: Following previous work [26], we provide four different output examples as references. Note that the in-context examples are not sampled from the user task O.
- • Solution pool Ps: Showcases all previously generated solutions (Python code only). If no solution exists, ”EMPTY” will be displayed.

Figure 3. Illustration of user specification inputs Pu.

the later evaluate the solutions to estimate their performance and computation cost. The framework is illustrated in Fig. 2.1 Furthermore, we leverage advanced multimodal LLMs (e.g., GPT) as the solution and metric routers. For better understanding, we first introduce the necessary notations, followed by a detailed explanation of these two modules in the following sections.

Problem Formulation and Notations. As shown in Fig. 2, given a user-specified task with N instances, we represent these instances as a set O = {oˆ1,oˆ2,...,oˆn,on+1,...,oN}, where oˆi = (Ii,qi,ai) and oi = (Ii,qi), with Ii, qi, and ai denoting the image set, task request prompt, and ground-truth answer for that instance, respectively. Note that only n instances have groundtruth answers, referred to as example instances Oex = {oˆ1,oˆ2,...,oˆn}, where n ≪ N. The goal of the Solution Router, RS, is to propose programmatic solutions for the task based on the example instances so that the answers for all instances can be inferred by leveraging the proposed solutions. In practice, together with the example instances and predefined task-agnostic prompts (e.g., model definitions), we construct an input prompt P for RS to generate a solution pool S = {s1,s2,...,sl}. Note that we set n ≪ N to enable model routing to perform reasoning to obtain the answer rather than simply memorizing the ground truth answers. Once solutions are obtained, the Metric Router, RM, samples a subset with m instances from O to evaluate the performance of each solution in S. This evaluation yields a set E = {(p1,c1),(p2,c2),...,(pl,cl)}, where pi and ci denote the performance and computation cost of the i-th solution in S. Optionally, other metrics can also be logged.

and user-specified task-specific instructions:

• User-specification Pu: Contains the task definition, example instances Oex sampled from the target task, and (optional) user constraints. Note that the task instances’ input includes images. User input is illustrated in Fig. 3.

#### 3.3. Multi-agent solution router

Taking all the aforementioned information as input, the goal of solution router is to propose novel solutions to solve the task at hand. To achieve that, inspired by multi-agent conversation works [16], we deploy multi-agent system for this complex problem. A conversation is instantiated between two teams: the solution proposer team and the committee team. The proposer team generates ideas and solutions, while the committee team checks for correctness, redundancy, and alignment with requirements, providing feedback. Each team consists of members and a leader. After gathering responses from their members, the leaders of two team exchange responses and collect feedback. By iteratively refining the solution based on this feedback, we achieve robust results. An illustration is provided in Fig. 4. We now detail each component and the conversation process within the multi-agent system. Please refer the supplement for example responses from all the agents in the solution Router.

1Our entire framework is built using Autogen [56], an open-source programming framework for agentic AI design that enables the development of multi-agent communication and Python code execution environments.

Solution Proposing Team. The solution proposing process

ensures that the proposed solution doesn’t duplicate any existing solutions in the current solution pool. If the logic of proposed solution matches an existing one, it rejects the solution to avoid redundancy in the solution pool. Please refer to the supplemental for output examples.

Multi-agent Conversation

[Figure 25]

[Figure 26]

Solution team

Committee team

Code debugger

Repetition checker

Solution proposer

Conversation between solution proposer and committee. The interaction between the Solution Proposing and Solution Committee Teams refines solutions iteratively, as depicted in Fig. 4. As mentioned in the prior work [16], a multi-agent conversation framework enhances reasoning and improves solution accuracy. However, excessive iterations can lead to error propagation. To address this, we require each committee member to deliver a decision at every iteration, either accepting or rejecting the solution with feedback. If all committee members accept the solution, the iteration concludes. Recognizing that convergence is sometimes challenging, we enforce a maximum number of iterations. At the end of the conversation, if the final solution is not redundant (as confirmed by the repetition checker), the most recent iteration’s solution is preserved.

ANALYSIS: … THOUGHT:

Modified code

- 1. …
- 2. …

Code executor

Requirement checker

ACTION: def sol_0 …

- # Step 1…

objs = LLaMA …

- # Step 2…

Execution results

output = LLaVA…

Code checker

Solution engineer

- Figure 4. Illustration of multi-agent conversation. In the solution router, we have two team of agents performing conversation to get the final outputs.

involves three key components: (i) analyzing existing solutions and committee feedbacks, (ii) outlining step-by-step high-level instructions, and (iii) developing Python code implementation. This process integrates analysis, creative problem-solving, and rigorous coding. We employ two agents for this purpose: the solution proposer Asp and the solution engineer Ase (see Fig. 4). The solution proposer Asp begins by reviewing existing solutions and generating a novel approach with clear, high-level instructions, resulting in the ANALYSIS and THOUGHT sections of the output. Following this, the solution engineer Ase builds on the instructions provided to produce executable Python code, documented in the ACTION section. Together, the ANALYSIS, THOUGHT, and ACTION sections form a comprehensive solution for further review. Please refer to the supplementary materials for output examples.

#### 3.4. Metric Router

After model routing, we are able to collect a pool of diverse solutions, S = {s1,s2,...,sm} (see Fig. 2). The evaluation router further assesses these solutions, resulting in a set E = {(p1,c1),(p2,c2),...,(pm,cm)}, where pi and ci represent the performance and computation cost of the i-th solution in S. We introduce an evaluation router, similar to the solution router, which uses the multimodal LLM’s reasoning to select the right metric based on the user’s task and the format of ground truth and predictions. Once the metric is chosen, we can proceed with performance testing and evaluation, estimating both the performance and cost of each solution. The user can also supply a custom metric rendering evaluation router unnecessary; however, the choice of the metric may not itself be trivial for a naive user.

Solution Committee Team. The Solution Committee oversees the quality and robustness of the generated solutions. Its main objectives are to verify that each solution meets predefined requirements, ensure code correctness and functionality, and check for redundancy with existing solutions. A significant challenge is validating code logic beyond mere error-free execution. Therefore, we introduce a code debugger that analyzes intermediate results. Additionally, with the code executor, we can provide the committee with intermediate outputs, enabling a detailed, step-by-step review of the logic. As shown in Fig. 4, we introduce two additional agents with specific roles: a requirement checker and a code checker. The requirement checker evaluates whether the solution aligns with the specified requirements. Meanwhile, the code checker assesses both intermediate and final execution results to verify the accuracy and logical soundness of the code. In the final stage, the repetition checker

Input Structure. We again use MLLM (i.e., GPT-4) as the router to select metrics for evaluation. Below, we detail the input prompt for the router, comprising of task-agnostic:

• Metric Definitions: Provides details for each metric in the metric pool, including use cases, input arguments, return arguments, and examples.

and user-derived task-specific instructions:

• Task Instances: Similar to the solution router, this includes task instructions and n example instances sampled from the target task, along with ground truth answers and predictions from the solutions.

Performance and Computation Cost Curve. For each proposed solution, we apply the aforementioned metric routing. Once a metric is selected, we first choose larger test cases from the user-provided task. As shown in Fig. 2, we

Method Depth Spatial Jigsaw Vis

Sem. Corr.

Art Count Fun. Corr.

Local. Multiview

Refl. Fore. IQ Sim.

corr.

Open-source multimodal LLMs

OpenFlamingo-v2 [5] 54.0 43.4 47.3 25.6 30.2 52.1 21.7 36.2 52.0 41.4 43.3 15.9 23.3 55.2 InstructBLIP-7B [14] 51.6 56.6 52.7 30.8 30.9 47.9 29.2 23.9 44.8 58.7 29.9 29.6 23.3 46.3 InstructBLIP-13B [14] 51.6 65.7 52.7 29.7 32.4 50.4 30.8 22.3 52.0 54.1 46.3 13.6 26.0 46.3 CogVLM [55] 50.8 67.1 52.7 20.9 23.6 49.6 46.3 23.9 43.2 57.1 26.9 24.2 26.7 46.3 LLaVA-v1.5-7B [35] 52.4 61.5 11.3 25.6 23.0 47.9 43.3 21.5 48.8 49.6 36.6 28.0 24.0 46.3 LLaVA-v1.5-13B [35] 53.2 67.8 58.0 29.1 32.4 47.9 50.0 20.8 47.2 41.4 45.5 27.3 28.0 46.3

Ours (LLaVA-7B) 51.6 78.8 56.7 33.1 32.4 54.7 41.2 21.5 56.6 55.6 37.0 26.5 23.3 58.5 Ours (LLaVA-13B) 58.1 69.9 64.0 34.3 34.5 58.1 47.2 23.9 51.6 51.1 45.1 26.5 28.0 45.9

API-based models

Qwen-VL-Max [7] 58.9 77.6 3.3 22.7 29.3 37.6 55.8 28.5 49.6 53.4 49.3 47.7 22.0 51.5 Gemini Pro [20] 50.0 67.1 54.0 37.2 22.1 49.5 65.0 32.3 46.4 41.4 46.3 45.5 27.3 55.9 Claude 3 OPUS [4] 57.3 57.3 32.7 31.4 20.7 60.7 49.2 22.3 46.4 57.9 27.6 62.1 21.3 70.6 GPT-4o [42] 74.2 69.2 55.3 75.0 54.0 82.9 51.7 39.2 56.0 60.2 38.8 85.6 30.0 65.4

GPT-4o (+ SoM + orig.)† 75.0 82.5 - - - - - - - - - - - GPT-4o (+ Visprog)† 46.8 37.8 - - - - - - - - - - - GPT-4o (+ Sketchpad) 83.9† 81.1† 70.7† 80.8† 58.3† 77.19∗ 66.7∗ 42.1∗ 65.4∗ 45.6∗ 33.1∗ 79.0∗ 22.8∗ 84.2∗

Ours (GPT-4o) 80.3 81.8 75.3 85.5 58.3 83.0 61.7 55.4 59.0 60.2 35.1 84.8 28.7 75.3

Table 1. Quantitative results. Experimental results on the BLINK benchmark [19]. † denotes results from the previous work [26], and ∗ represents results collected via official codebase. The best result is highlighted in Bold and the second underlined.

Model Avg. Scene Id Attri. Locat.

InstructBLIP [14] 51.5 58.9 49.7 61.7 35.1 LLaVA-v1.5-7B [35] 57.7 63.7 62.4 66.7 51.3 MiniGPT-4 [63] 45.9 56.3 49.2 45.8 37.9 OpenFlamingo [5] 36.1 46.7 42.3 31.7 33.4 Qwen-VL-Chat [7] 50.9 56.5 47.6 54.8 46.9 CogVLM [55] 42.4 51.7 43.5 38.9 33.8 InternLM [62] 69.2 77.5 73.5 74.8 65.4 GPT-4o [42] 75.6 77.3 79.7 79.2 71.0

Ours (GPT-4o) 75.8 78.3 78.3 79.7 70.1 Model Count. Spatial Inter. Reason. Text InstructBLIP [14] 58.1 34.9 47.4 55.9 61.4 LLaVA-v1.5-7B [35] 60.2 38.5 47.4 59.8 69.0 MiniGPT-4 [63] 45.3 32.6 47.4 57.1 41.8 OpenFlamingo [5] 27.4 29.8 29.9 47.7 35.6 Qwen-VL-Chat [7] 54.2 40.3 55.7 55.0 47.4 CogVLM [55] 29.4 33.6 45.4 53.5 51.5 InternLM [62] 65.8 57.5 71.1 75.8 61.2 GPT-4o [42] 68.1 63.8 78.6 81.2 69.8 Ours (GPT-4o) 67.7 62.8 80.6 84.5 69.9

Table 2. Quantitative results on Seedbench [31].

then perform evaluations, recording both performance and computation cost, and generate a plot. This allows users to select solutions based on their preferences. Please see supplemental for further discussion on metric routing.

### 4. Experiments

Datasets and Evaluation To verify the effectiveness of MMFactory, we conduct experiments on two benchmarks: BLINK [19] and Seedbench [31], and compare our model against previous works. These benchmarks contain various tasks covering visual perception and spatial understanding. BLINK includes 14 visual perception tasks with a total of 3,807 multiple-choice questions, while SeedBench covers 9 classical spatial understanding tasks with a total of 14k image-QA pairs, including scene understanding, instance interaction, and visual reasoning. There are some overlap-

ping tasks between the two benchmarks; however, the main difference is that BLINK focuses on evaluating visual perception, where tasks are designed to be solvable by humans at a glance while hard to answer correctly for MLLMs. In contrast, SeedBench emphasizes models’ visual spatial understanding, involving complex tasks with small objects or intricate descriptive prompts. For evaluation, since the tasks in these datasets are single-choice questions, we follow their protocol by using GPT to map the open-form predictions from MLLMs to the fixed set of choices and perform string matching to report accuracy for each task.

#### 4.1. Quantitative Analysis

In this subsection, we evaluate the effectiveness and performance of our MMFactory. Note that, to ensure a fair comparison with previous SoTA models, we fix the multimodal LLMs to the same ones used in the compared methods for quantitative evaluation. For vision models, we use exactly the same models as those employed in the prior work on Visual Sketchpad [26].

Can MMFactory propose effective solutions? To verify this point, we conducted experiments on BLINK and SeedBench, reporting performance using three different multimodal LLMs (i.e., LLaVA-7B, LLaVA-13B, and GPT-4o) as fixed MLLMs. The results are shown in Tables 1 and 2. Our method demonstrates its ability to propose useful solutions with either comparable or improved performance relative to its own base model. Notably, with the routing approach, very significant performance boosts are observed in certain tasks, such as function correspondence (+15% over GPT-4o) and jigsaw solving (+20% over GPT4o), spatial understanding (+17% over LLaVA-7B), and jigsaw again (+6% over LLaVA-13B). Consistent performance improve-

[Figure 27]

[Figure 28]

Examples from the task User constraints

Task definition

[Figure 29]

Is the car away from the truck?\nSelect from the following choices.\n(A) yes\n(B) no. Answer: (A)

Is the laptop touching the person?\nSelect from the following choices.\n(A) yes\n(B) no. Answer: (A)

- - Only use open source models
- - Less than 10 sec per sample

In this task, you are given a prompt and an image. The prompt will mention two objects of interest and describe a spatial relation … verify whether the prompt accurately reflects the spatial relationship …

Proposed solutions Metric MMFactory

Is the boat on the truck?\nSelect from the following choices.\n(A) yes\n(B) no. Answer: (A)

|# Step 1: Parse the objects from the prompt object_names = llama(f"Identify the two objects mentioned in the following prompt: {prompt}. Please return only the object names separated by a comma.")<br># Step 2: Detect the objects in the image objects = [name.strip() for name in object_names.split(",")] img, image_boxes = ground_dino(image_1, objects)<br># Step 3: Prepare the prompt for the VLM prompt += " Here are the bounding boxes of the objects for reference: " + str(image_boxes) prompt += " The image's origin is … range [0, 1]. Bounding boxes follow the format [x, y, w, h] … width and height, respectively."<br># Step 4: Verify the spatial relationship using VLM response = LLaVA_7B(prompt, image_1)<br><br><br>def sol_1(prompt, image_1):<br><br>return response| |
|---|---|
| | |
| | |

|laptop, person<br><br>[“laptop”, “person”]<br><br>[[0.6208, 0.5451, 0.7514, 0.7983], [0.7446, 0.8226, 0.3865, 0.3487]]<br><br>[Figure 30]<br><br>Is the laptop touching the person? Select from the following choices. (A) yes(B) no Here are the bounding boxes of the objects for reference: [[0.6208, 0.5451, 0.7514, 0.7983], [0.7446, 0.8226, 0.3865, 0.3487]] The image's origin is … . Bounding boxes follow the format [x, y, w, h] … width and height, respectively.<br><br>(A)<br><br>>>> print(object_names)<br><br>>>> print(objects)<br><br>>>> display(img)<br><br>>>> print(image_boxes)<br><br>>>> print(prompt)<br><br>>>> print(response)<br><br>Execution results:|
|---|

[Figure 31]

[Figure 32]

[Figure 33]

… GPTScore

GPTaccuracy

10 sec Time cost (sec)

def sol_4(prompt, image_1):

def sol_0(prompt, image_1):

- # Step 1: Parse the objects of interest objects_of_interest = llama(...) objects_list = [obj.strip() for obj in objects_of_interest.split(",") if obj.strip()]
- # Step 2: Use sliding window detection all_possible_boxes = [] for obj in objects_list:

possible_patches, possible_boxes = sliding_window_detection(image_1, [obj]) all_possible_boxes.append(possible_boxes[0])

- # Step 3: Prepare the prompt prompt += (...)
- # Step 4: Verify the spatial relationship using VLM response = InternVL(prompt, image_1)

- # Step 1: Parse the prompt objects_info = llama(f"Identify the two objects and their spatial relationship in the following prompt: {prompt}. Please return the objects and the relationship clearly.")
- # Step 2: Construct the enhanced prompt enhanced_prompt = f"{prompt} Here are the objects and their spatial relationship: {objects_info}. The image's origin is at the upper-left corner (0, 0), and all coordinates are normalized within the range [0, 1]."
- # Step 3: Use the VLM to analyze the image response = LLaVA_13B(enhanced_prompt, image_1)

return response

return response

- Figure 5. Qualitative examples of MMFactory. MMFactory showcases its abilities to use and combine models by automatically constructing better prompts for MLLMs (in Sol 0) and developing solutions with similar logic but utilizing stronger models (in Sol 4).

ments are also seen on SeedBench, particularly for multiinstance understanding tasks like instance interaction and reasoning, with a ≈ 3% increase, verifying the effectiveness of our proposed solution router.

|Model|Acc Error rate Avg. # sols<br><br>|
|---|---|
|Full model (-) code debugger (-) code checker (-) requirement checker (-) repetition checker<br><br>|50.5 0.0 3.0 40.0 1.7 2.8 33.3 20.8 3.0 48.1 0.5 2.4 40.5 17.8 2.0|

Comparison with augmented frameworks for MLLMs We further compare our framework with other augmentation frameworks for MLLMs, such as SoM [58], Visprog [22], and Visual Sketchpad [26]. Visual Sketchpad [26] allows LMs to adjust their solution based on intermediate visual results from other tools. To demonstrate that our solution proposer with multi-agent cooperation can produce better solution plans than Visual Sketchpad, we fixed the visual tools and the LM as used in their approach and reported the performance of our proposed solutions in Table 1. Benefiting from multi-agent cooperation, our approach achieves comparable or better performance than the previous SoTA, highlighting the effectiveness of the solution proposer. Most importantly, our proposed solutions are general and not limited to specific samples within the task. As a result, we significantly reduce the API calling cost; see Figure 7 for more details. Last but not least, comparing with previous visual programming work of Visporg, we achieve + ≈ 30% over depth and spatial tasks, demonstrating our approach can propose a stronger pre-defined solution.

Table 3. Ablation. of significance of multi-agent conversation.

#### 4.2. Qualitative Analysis

Fig. 5 shows qualitative examples of our proposed MMFactory. It samples a few examples from a given task, defined by the user’s constraints and task details (e.g., image and prompt), and passes them to MMFactory. The “solution proposer” then generates a pool of robust solutions for the task. Simultaneously, the “metric router” generates a performance curve showing the trade-off between time cost and accuracy based on selected metrics (e.g. GPTScore). Unlike existing methods, our approach generates a solution pool from which users can choose the best option based on their constraints. Additionally, our framework provides solutions tailored to the entire task, rather than to individual samples. Additional examples are provided in supplement.

#### 4.3. Model Analysis Ablation studies of the multi-agent corporation. In the solution router, we leverage multi-agent conversation to im-

[Figure 34]

[Figure 35]

Time cost analysis

250

[Figure 36]

|Avg time cost per solution Time cost per iteration Avg time cost per iteration<br><br>|
|---|

0.8

500

225

Timecostperiteration(sec)

Timecostpersolution(sec)

200

400

0.6

175

300

150

###### Accuracy

125

200

0.4

100

100

75

0.2

50

0

1st sol 2ed sol 3rd sol 4th sol

0

Execution cost (sec) Routing cost (sec) Mean Variance Mean Variance

Execution cost (sec) Routing cost (sec) Mean Variance Mean Variance

Model

Model

1st iter 2ed iter 3rd iter 4th iter

5th iter 6th iter

Sketchpad 19.96 43.86 18.20 30.90 Ours 9.74 29.43 ≈ 0.00 ≈ 0.00

Iteration

Sketchpad [26] 19.96 43.86 18.20 30.90 Ours 9.74 29.43 ≈ 0.00 ≈ 0.00

[Figure 37]

- Figure 6. Ablation. Performance analysis with iteration. Lines in different colors represent different runs. Red cross denotes the highest performance in the run.

Figure 7. Computational time. Solution generation cost plot (top). Average execution and routing cost per sample (bottom).

prove the quality and robustness of the generated solutions. We conduct ablation studies on the multi-agent component of the proposer to verify this, with the results shown in Tab. 3. Specifically, we run the solution router on the first five tasks (listed in Tab. 1) in the BLINK dataset, with three runs per task, each allowing a max of six conversation iterations. Without the code debugger, the code checker cannot access the intermediate results of the solution, resulting in a significantly performance accuracy drop of 10%. With-

|Model|Depth Spatial Jigsaw<br><br>Vis. Corr.<br><br>Sem. Corr.<br><br>|
|---|---|
|Sketchpad [26] Ours<br><br>|0.211 0.232 0.224 0.281 0.230 0.064 0.045 0.041 0.034 0.058|

Avg routing API cost per 10 samples (USD)

Table 4. API calling cost analysis per 10 samples (in USD).

|Model|Depth|Spatial|Jigsaw|Vis. Corr.|Sem. Corr.|
|---|---|---|---|---|---|
|Sketchpad Ours<br><br>as the instances,|0.211 0.232 0.224 0.281 0.230 0.064 0.045 0.041 0.034 0.058<br><br>proposed solutions are reusable routing cost per sample is nearly| | | | |

tionally, across all task in zero, significantly less than the on-line routing in previous work.

- out the code checker, there is no feedback on execution results, which not only reduces the performance but also substantially increases the error rate during solution execution. Furthermore, after ablating the requirement checker, we observe both performance and solution correctness degrade compared to the full model. Lastly, without the repetition checker, the average number of proposed solutions decreases significantly by 33%, verifying the effectiveness of the repetition checker in enhancing solution diversity.

Furthermore, as we use a GPT model for the solution router, we report the average API cost and compare it with previous work, Visual Sketchpad [26] (see Tab. 4). A key benefit of our approach is that we perform routing only for a few runs, with the produced solution applicable to all samples, significantly reducing the cost. In contrast, Sketchpad requires an API call for every sample, resulting in almost five times the cost of our approach on the BLINK dataset.

Best answer happen in which run progressive performance analysis. In the solution router, we set a maximum number of conversation iterations for the multi-agent cooperation. As mentioned in previous studies [16], multiagent conversation or debate can lead to error propagation, reducing performance after multiple iterations. To investigate this, we conducted experiments to analyze performance as the number of iterations increased, with results shown in Fig. 6. Specifically, we randomly selected 10 tasks from the BLINK dataset and ran our solution router to generate solutions, setting the maximum number of iterations to six. As shown in the figure, we observe that solutions with the best performance occur around 2–4 iterations.

Routing time and API calling cost. In our solution router, agents iteratively converse to generate the final solutions. As the number of existing solutions in the pool grows, the router may take more time to propose a novel solution. Therefore, we further investigate the routing time cost with varying numbers of solutions in the pool. The average time cost per solution and per iteration is reported in Fig. 7 (top). We observe that the time cost per solution increases as the number of existing solutions grows. We assume this is due to the increasing complexity of the task, requiring the router to utilize the maximum number of iterations to derive the final solution. On average, it takes approximately 8 minutes to generate a solution. Notably, since the generated solutions are applicable to all samples within a task, we only need to perform solution routing once per task, rather than for each sample. We compare execution and routing costs with Visual Sketchpad in Fig. 7 (bottom). Execution cost refers to the time from input prompt to final answer, while routing cost is the time spent coordinating tools (execution time minus tool-calling time). One can find that with the pre-planned solutions, our execution cost is lower. Addi-

### 5. Conclusion

Selecting the right multimodal LLM for a task can be difficult, especially without domain-specific knowledge or clear user requirements. In this paper, we present a framework to help users select the most suitable solution from a solution pool for a given tasks based on their specific constraints. Our approach uses a multi-agent debate mechanism to generate robust and well-reasoned solution. Unlike

sample-specific solutions, our framework provides guidance that applies broadly across all examples for a given task. Through extensive experiments, we demonstrate that our method outperforms current state-of-the-art approaches.

### Acknowledgements

This work was funded, in part, by the Vector Institute for AI, Canada CIFAR AI Chairs, NSERC Canada Research Chair (CRC), and NSERC Discovery and Discovery Accelerator Supplement Grants. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, the Digital Research Alliance of Canada2, companies3 sponsoring the Vector Institute, and Advanced Research Computing at the University of British Columbia. Additional hardware support was provided by John R. Evans Leaders Fund CFI grant and Compute Canada under the Resource Allocation Competition award.

### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 1
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [3] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In CVPR, 2016. 2
- [4] Anthropic. The claude 3 model family: Opus, sonnet, haiku. Technical Report, 2023. 2, 6
- [5] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 1, 6
- [6] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023. 2
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2), 2023. 1, 6
- [8] Zal´an Borsos, Rapha¨el Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al.

2alliance.can.ca 3https://vectorinstitute.ai/#partners

Audiolm: a language modeling approach to audio generation. TASLP, 2023. 2

- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 2
- [10] Harrison Chase. LangChain, 2022. 3
- [11] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024. 2
- [12] Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, B¨orje F Karlsson, Jie Fu, and Yemin Shi. Autoagents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288, 2023. 3
- [13] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023. 2
- [14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. NeurIPS, 2023. 6
- [15] Nilaksh Das, Saket Dingliwal, Srikanth Ronanki, Rohit Paturi, David Huang, Prashant Mathur, Jie Yuan, Dhanush Bekal, Xing Niu, Sai Muralidhar Jayanthi, et al. Speechverse: A large-scale generalizable audio language model. arXiv preprint arXiv:2405.08295, 2024. 2
- [16] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023. 3, 4, 5, 8
- [17] Wan-Cyuan Fan, Yen-Chun Chen, Mengchen Liu, Lu Yuan, and Leonid Sigal. On pre-training of multimodal language models customized for chart understanding. arXiv preprint arXiv:2407.14506, 2024. 2
- [18] Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Junteng Jia, Yuan Shangguan, Ke Li, Jinxi Guo, Wenhan Xiong, Jay Mahadeokar, Ozlem Kalinli, et al. Prompting large language models with speech recognition abilities. In ICASSP, 2024. 2
- [19] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390,

2024. 1, 6

- [20] Google Gemini Team. Gemini: A family of highly capable multimodal models. technical report. Technical Report,

2023. 2, 6

- [21] OpenAI GitHub. Github copilot, 2023. 2
- [22] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In CVPR, 2023. 2, 7
- [23] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A mul-

- timodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023. 2
- [24] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. NeurIPS, 2023. 2
- [25] Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing system. arXiv preprint arXiv:2403.12031, 2024. 3
- [26] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. arXiv preprint arXiv:2406.09403, 2024. 2, 4, 6, 7, 8
- [27] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 2
- [28] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In ICCV, 2017. 2
- [29] Shyam Sundar Kannan, Vishnunandan LN Venkatesh, and Byung-Cheol Min. Smart-llm: Smart multi-agent robot task planning using large language models. arXiv preprint arXiv:2309.10062, 2023. 3
- [30] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 2
- [31] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In CVPR, 2024. 1, 6
- [32] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large language model society. NeurIPS,

2023. 3

- [33] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118, 2023. 3
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2
- [35] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. arXiv preprint arXiv:2311.05437, 2023. 1, 2, 3, 6
- [36] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 2

- [37] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 2

- [38] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. NeurIPS, 2024. 2, 3
- [39] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024. 2
- [40] Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. arXiv preprint arXiv:2401.02384, 2024. 2
- [41] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data. arXiv preprint arXiv:2406.18665, 2024. 2, 3
- [42] OpenAI. Gpt-4 technical report. Technical Report, 2023. 2, 6
- [43] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023. 3
- [44] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023. 2, 3
- [45] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789,

2023. 2

- [46] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, J´er´emy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950,

2023. 2

- [47] Tal Shnitzer, Anthony Ou, M´ırian Silva, Kate Soule, Yuekai Sun, Justin Solomon, Neil Thompson, and Mikhail Yurochkin. Large language model routing with benchmark datasets. arXiv preprint arXiv:2309.15789, 2023. 3
- [48] Significant Gravitas. AutoGPT. 3
- [49] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In ICCV,

2023. 2

- [50] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint arXiv:2311.10537, 2023. 3
- [51] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,

- Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [52] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1
- [53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2
- [54] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. arXiv preprint arXiv:2310.03731, 2023. 2
- [55] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 6
- [56] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155, 2023. 3, 4
- [57] Yiran Wu, Feiran Jia, Shaokun Zhang, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, Qingyun Wu, and Chi Wang. An empirical study on challenging math problem solving with gpt-4. arXiv preprint arXiv:2306.01337, 2023. 3
- [58] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 7
- [59] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 2
- [60] Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and Philip S Yu. Large language models for robotics: A survey. arXiv preprint arXiv:2311.07226, 2023. 2
- [61] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2
- [62] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlmxcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112, 2023. 1, 2, 6
- [63] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 6

[64] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In ICCV, 2023. 2

