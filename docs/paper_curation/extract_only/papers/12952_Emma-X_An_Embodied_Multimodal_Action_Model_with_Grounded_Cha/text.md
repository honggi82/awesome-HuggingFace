[Figure 1]

## EMMA-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning

Qi Sun1*, Pengfei Hong1*, Tej Deep Pala1, Vernon Y.H. Toh1, U-Xuan Tan1, Deepanway Ghosal1†, Soujanya Poria1

1 Singapore University of Technology and Design

[Figure 2]

|VLA backbone<br><br>… …<br><br>Image Tokenizer Text Tokenizer<br><br>Action De-Tokenizer<br><br>…<br><br>7D Robot Actions: [∆𝒙, ∆𝚯, ∆𝑮𝒓𝒊𝒑]<br><br>Text Decoder<br><br>[Figure 3]<br><br>Emma-X<br><br>Grounded Chain of Thought and Spatial Reasoning<br><br>Multimodal Input|
|---|

###### Hierarchical Embodiment Data

Closed-Loop Robot Control Policy

User：Wipe the stove with the towel.

[Figure 4]

[Figure 5]

# arXiv:2412.11974v2[cs.RO]17Dec2024

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Emma-X: Grounded Chain of Thought: .. Spatial Reasoning: ... Robot Actions [∆𝒙, ∆𝚯, ∆𝑮𝒓𝒊𝒑]=…

[Figure 10]

[Figure 11]

###### 60 K Trajectories

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|[Figure 18]|
|---|

[Figure 19]

|Plan: Positioning over pot cover， Grasping the pot cover, Lifting … Subtask: Grasping the pot cover Reason: The robot is closing its gripper around the pot cover's handle to securely pick it up…|
|---|

Downstream Real Robot Manipulation

[Figure 20]

|2D Gripper Position of Future State<br><br>3D Spatial Movements to Future State<br><br>[Figure 21]<br><br>[Figure 22]|
|---|

dataset based on BridgeV2, containing 60,000 robot manipulation trajectories auto-annotated with grounded task reasoning and spatial guidance. Additionally, we introduce a trajectory segmentation strategy based on gripper states and motion trajectories, which can help mitigate hallucination in grounding subtask reasoning generation. Experimental results demonstrate that EMMA-X achieves superior performance over competitive baselines, particularly in real-world robotic tasks requiring spatial reasoning. We make our codes, models and datasets publicly available: https: //declare-lab.github.io/Emma-X/.

#### Abstract

Traditional reinforcement learning-based robotic control methods are often taskspecific and fail to generalize across diverse environments or unseen objects and instructions. Visual Language Models (VLMs) demonstrate strong scene understanding and planning capabilities but lack the ability to generate actionable policies tailored to specific robotic embodiments. To address this, Visual-Language-Action (VLA) models have emerged, yet they face challenges in long-horizon spatial reasoning and grounded task planning. In this work, we propose the Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning, EMMA-X. EMMA-X leverages our constructed hierarchical embodiment

#### 1 Introduction

The robotic policy model aims to generate sequences of low-level action manipulation policies for robots. Traditional reinforcement learningbased robotic control methods often focus on narrowly defined tasks within fixed environments (Ma et al., 2024), hindering their ability to generalize

*Both authors contributed equally to this work. The first authorship was randomly assigned by coin flip.

†Now at Deepmind.

VLAs with an understanding of the current situation and task, they lack long-horizon spatial reasoning on how robots should move next. We hypothesize that the completion of subgoals or subtasks can be enhanced if the VLA incorporates look-ahead spatial reasoning, such as inferring the gripper’s future 2D position and the 3D movement plans necessary for the gripper to reach that position. In particular, we train the VLA model to predict future position gt+k of the gripper as checkpoints and use them to devise a high-level movement plan β(gt,gt+k). This plan informs the immediate action at at the current state st, ensuring decisions are both reactive to the present and aligned with long-term objectives. Similar to a delivery driver planning a route with key landmarks to make purposeful driving decisions, this approach optimizes task completion by balancing foresight and adaptability.

###### Task: put the silver pot cover on the silver pot.

|[Figure 23]<br><br>[Figure 24]<br><br>…|[Figure 25]|[Figure 26]<br><br>[Figure 27]<br><br>…|
|---|---|---|

Task Reasoning of ECoT Our Task Reasoning

|Plan: Move to the silver pot cover, Close the gripper, Put the silver pot cover … Subtask: Move to the silver pot cover Reason: The silver pot cover is the closest to the robot arm and needs to be picked up first.<br><br>[Figure 28]<br><br>[Figure 29]|
|---|

|Plan: Positioning over pot cover，<br><br>Grasping the pot cover, Lifting … Subtask: Grasping the pot cover Reason: The robot is closing its<br><br>gripper around the pot cover's handle to securely pick it up…<br><br>[Figure 30]<br><br>[Figure 31]|
|---|

###### Our Spatial Reasoning

|SUBTASK: Grasping the pot cover REASONING: The robot is closing its gripper around the pot cover's handle to securely pick it up. This requires precision to ensure a stable grip.<br><br>[Figure 32]<br><br>[Figure 33]<br><br>𝒙<br><br>𝒚<br><br>𝒛<br><br>2D Gripper Position of Future State 3D Spatial Movements to Future State|
|---|

- Figure 1: Comparison of our EMMA-X with ECoT in task reasoning. While both approaches utilize Gemini, our method also incorporates image sequence input, whereas ECoT relies solely on text input. We also illustrate an example of spatial reasoning.

Additionally, another limitation in task reasoning provided by ECoT(Zawalski et al., 2024) is the absence of visual grounding when augmenting reasoning data using Gemini. We observe that Gemini frequently hallucinates due to a lack of holistic understanding of the setup and environment. As shown in Figure 1, the image shows that the robot already started to grasp the pot cover, while the task reasoning indicates the subtask is still “Move to silver pot cover", which conflicts with the following reasoning they provided.

beyond task-specific training data and limiting their applicability (Brohan et al., 2023b; Chi et al.).

Recent advancements in foundation models for vision and language have highlighted the remarkable scene-understanding and task-planning capabilities (Radford et al., 2021; Zhai et al., 2023; Touvron et al., 2023). These Visual-Language Models (VLMs) excel at breaking down complex tasks into manageable steps through chain-of-thought reasoning and demonstrate significant potential in planning. Despite their strengths, VLMs are not inherently designed to directly generate policies applicable to specific embodiment configurations in robotics. This limitation has spurred the emergence of Visual-Language-Action (VLA) models, which aim to bridge this gap by leveraging multimodal inputs to produce adaptive and generalized robotic actions for complex, multi-task scenarios (Brohan et al., 2023a; Kim et al., 2024; Octo Model Team et al., 2024).

In this work, we introduce the Embodied Multimodal Action Model with Grounded Chain of Thought Reasoning, EMMA-X. We develop a hierarchical embodiment dataset based on BridgeV2, consisting of 60,000 robot manipulation trajectories. For each state of a given trajectory, we generate detailed spatial reasoning grounded in the environment and task reasoning, such as the plans of how the robot should perform the subtask. As shown in Figure 1, we also generate the 2D gripper position, and 3D spatial movements of the gripper to transit to future states, which enable the VLA model to reason a long-horizon plan for accomplishing the task.

However, most of the existing VLA models often exhibit “muscle memory” response patterns, struggling to perceive scene variation and understand instructions as humans do when handling complex tasks or ambiguous commands. Zawalski et al. (2024) attempts to address this issue through visual and task reasoning, including the bounding box of the object, task segmentation, and the direction of predicted action, etc. Although they equip

Furthermore, we utilize Gemini (Team et al., 2023) to generate grounded task reasoning for each observed state. To avoid the abovementioned reasoning conflict problem of task reasoning in ECoT, we propose a novel trajectory segmentation strategy, which leverages the opening and closing states of the gripper and the motion trajectory of the robot arm to segment the sequence of states into distinct

segments. By grounding, we mean that, unlike ECoT, which prompts Gemini to generate subtask reasoning based solely on textual descriptions, our approach incorporates visual images segmented using the aforementioned strategy. As shown in Figure 1, our method can accurately provide the subtask “Grasping the pot cover" corresponding to the current robotic state. This illustrates that our strategy significantly reduces Gemini’s hallucination issues by requiring it to construct a visual understanding of the environment, rather than relying solely on textual descriptions of the environment. Finally, we train our EMMA-X based on OpenVLA using our constructed hierarchical embodiment dataset.

The main contributions of our work are summarized as follows:

- • We introduce a 7B-parameter embodied multimodal action model, EMMA-X created by fine-tuning OpenVLA with the grounded chain of thought (CoT) reasoning data.
- • We synthetically construct a hierarchical embodiment dataset from the existing robot manipulation dataset, which includes the 3D spatial movements, 2D gripper position, and grounded reasoning.
- • We propose a novel trajectory segmentation strategy that leverages the gripper’s opening and closing states alongside the motion trajectory of the robot arm, facilitating both grounded task reasoning and look-ahead spatial reasoning.
- • Our proposed EMMA-X achieves significant performance improvements over existing competitive baselines on various real-world robot tasks, especially in tasks where spatial reasoning is required.

#### 2 Problem Formulation

- 2.1 Policy Imitation Learning Given a set of expert demonstrations D =

{({st}Tt=1,Ti,{at}Tt=1)}Ni=1, where N is the number of demonstrations in the dataset, T is the number of states (image frames of the environment) for a data sample Di, si = imagei represents the state consisting of an image of the environment, Ti is a natural language task instruction, and ai represents the action taken by the expert in that state, the goal is to learn a policy πθ(a | s,T ) that mimics the expert’s behavior.

The policy πθ is modeled by a Vision-LanguageAction (VLA) model. In line with the OpenVLA setting, the policy outputs a generalized action as a 7-dimensional vector. This vector encodes the endeffector’s (gripper’s) velocity of Cartesian components (x,y,z), orientational components (roll, pitch, yaw), and the gripper’s close-open action.

The goal is to find parameters θ that minimize the difference between predicted action and the expert’s action.

##### 2.2 Hierarchical Policy Imitation

We build on the above formulation by decomposing a general task T into a hierarchical structure consisting of finer-grained components: states, segments, and subtasks.

A state at timestep t, denoted st, represents the scene. The sequence of states for the i-th trajectory is Si = {s1,s2,...,sT}, where T is the number of timesteps. An action at is taken at state st, and the corresponding sequence of actions is Ai = {a1,a2,...,aT}. A segment σ is a series of consecutive states, {st,st+1,...,st+k}, contributing to a subgoal, with Σi = {σ1,σ2,...,σn} representing the segment sequence for the i-th trajectory. In each segment, the robot performs similar actions. A subtask S consists of segments, {σ1,σ2,...,σp}, to achieve a specific subgoal. Finally, a task T is a series of subtasks, {S1,S2,...,Sm}, required to complete the overall objective.

Our Vision-Language-Action (VLA) model πθ(at | st,T ) predicts actions at for each state st by hierarchically decomposing tasks into subtasks. This ensures the end-effector’s motion aligns with subgoal intents, enhancing the model’s ability to execute complex tasks through manageable subtasks. We create a dataset D = {Di}Ni=1, where Di = {Si,Σi,Ti}. Each state st ∈ Si is labeled with its subtask. Without such labeling, chain-ofthought training is infeasible. During inference, the model generates reasoning chains, including subtasks and relevant spatial information derived from visual scenes.

#### 3 Methodology

In this section, we introduce our proposed framework in detail. Our EMMA-X encompasses three crucial designs: (1) Segmenting the trajectory based on the states of the gripper and the motion trajectory of the robotic arm. (2) Generating hierar-

(e) Gripper Position

(d) Movement Generation (f) Grounded Reasoning

(c) Trajectory Segment

(a) Motion Trajectory (b) Gripper State

|[Figure 34]<br><br>[Figure 35]<br><br>REASONING: The robot arm needs to move to the correct location above the pot cover to be able to grasp it… SUBTASK: Positioning over pot cover<br><br>User: analyse the robot actions in each segment and give subtask and reason.<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>SEGMENT: 1 REASONING: The robot arm needs to move to the correct location above the pot cover to.. SUBTASK: Positioning over pot cover<br><br>Task: pick up the pot cover and put it on the silver pot.|
|---|

|[Figure 39]<br><br>End Effector State [x1,y1,z1]<br><br>End Effector Position [x1,y1,z1…]<br><br>Movement to next segment<br><br>1. Move forward 22 steps<br>2. Move left 23 steps<br>3. Move downward 25 steps …<br><br><br>𝒑𝒈𝒐𝒂𝒍 − 𝒑𝒏𝒐𝒘|
|---|

|[Figure 40]<br><br>OWL + SAM|
|---|

|...<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]|
|---|

|opening<br><br>closing<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

Initial State Final State

(g)

|S1: Positioning over pot cover S2: Grasping the pot cover S3: Lifting and moving the pot cover<br><br>|[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]|
|---|
<br><br>|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|
<br><br>|[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]|
|---|
<br><br>|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]|
|---|
<br><br>S4: Positioning and placing the cover on the pot|
|---|

- Figure 2: Construction of our hierarchical embodied dataset. We first segment the trajectory. Then, we generate the

- 3D spatial movement that requires to transition to the end state of the segment. Based on segments, we recognize the 2D gripper position and generate the grounded task reasoning.

chical planning including grounded task reasoning, 2D gripper positions, and 3D spatial movements. (3) Training the our EMMA-X based on OpenVLA with our constructed dataset.

trajectories into sequences of consecutive states where the robot performs semantically similar actions. This segmentation provides richer context, allowing Gemini to assign subtask labels more effectively.

##### 3.1 Trajectory Segmentation

Additionally, segmentation facilitates finding the gripper’s position in a future state and planning its movement. At a given state st, the model predicts the movement plan required to reach the initial state of the next segment, st+k, before determining the policy at for st. Since t + k > t, this approach enables the model to perform look-ahead spatial reasoning, predicting the gripper’s position at a likely future state, planning the motion trajectory, and generating at accordingly.

Why Segment Trajectories? The overarching goal of our work is to enhance Vision-LanguageAction (VLA) models with grounded chain-ofthought (CoT) reasoning. We identified two key limitations in existing VLAs: 1) While existing VLAs improve task decomposition by breaking a task into subtasks and solving each using CoT (Zawalski et al., 2024), their CoT reasoning relies exclusively on textual scene descriptions 1. This limits their reasoning capability for real-world scenarios. 2) They lack robust spatial reasoning abilities, essential for effective task planning and execution.

Our Segmentation Method. As shown in Figure 2(a) and Figure 2(b), we segment observation sequences by integrating the motion trajectory and the gripper states of the end effector. To achieve this, we utilize the Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm (McInnes et al., 2017), which effectively handles noise stemming from small fluctuations caused by imperfections in human demonstration. The flexibility of HDBSCAN enables the discovery of diverse trajectory patterns within the data.

To address these limitations, we propose two key solutions: Incorporating visual scene information: Beyond textual prompts, we integrate visual inputs into Gemini to enable task decomposition into subtasks and generate high-level plans grounded in both visual and textual contexts. Finegrained movement plans: We train the robot to determine where to go and how to reach a potential future state necessary for completing a subtask.

To implement these solutions, every state must be labeled with the subtask the robot is performing. However, our experiments revealed that directly annotating each individual frame via Gemini resulted in noisy labels, likely due to insufficient contextual information. To overcome this, we segment

We define a custom distance measurement to segment the end effector’s trajectory, capturing both spatial and temporal information. Let pi = (xi,yi,zi) denote the 3D position, and ri = (rix,riy,riz) represent the 3D orientation of data point i. Additionally, let ti represent the timestamp of this data point. The distance between two data

1We use “scene” and “environment” interchangeably throughout this paper.

points i and j is given by the following expression:

Hierarchical Embodiment Data

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

d(i,j) = ∥pi−pj∥2+λ∥ri−rj∥2+β|ti−tj| (1)

[Figure 71]

[Figure 72]

60 K Trajectories

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

where λ is a weighting factor for the orientation component, and β controls the influence of the temporal distance 2. This combined distance metric d(i,j) ensures that both spatial movement and temporal separation contribute to the segmentation process. The inclusion of temporal information helps to distinguish trajectories that are spatially similar but occur at different times, while the orientation term captures changes in the end effector’s rotation.

[Figure 77]

[Figure 78]

|Emma-X<br><br>[Figure 79]<br><br>Instruction: pick up the pot cover and put it on the silver pot. 2D Gripper Position of Current State: [112, 105]<br><br>… …<br><br>Image Tokenizer Text Tokenizer<br><br>Action De-Tokenizer<br><br>…<br><br>7D Robot Actions: [∆𝒙, ∆𝚯, ∆𝑮𝒓𝒊𝒑]<br><br>Reasoning: The robot has grasped the pot cover and is now lifting it … Subtask: Lifting and moving the pot cover<br><br>2D Gripper Position of Future State : [70, 103]<br>3D Movement: move forward 19 steps; move left 73 steps; move downward…<br><br><br>Text Decoder<br><br>[Figure 80]<br><br>[Figure 81]|
|---|

Applying the HDBSCAN algorithm with this distance metric allows us to segment the trajectory into meaningful clusters that reflect distinct movement patterns. However, the motion trajectory alone does not fully capture the interaction dynamics of the end effector with the environment. To address this, we incorporate the gripper state gsi, which represents whether the gripper is in a grip (closed) or loose (open) position. A segmentation breakpoint occurs when the HDBSCAN algorithm detects a new cluster or when the gripper state changes between consecutive data points, formally defined as gsi ̸= gsi+1.

Figure 3: The overview of EMMA-X fine-tuned from OpenVLA using our hierarchical embodiment dataset.

Why Look-ahead Spatial Reasoning? Consider the robot as a delivery driver tasked with delivering a package to a specific destination (the goal). The driver has access to a detailed high-level map of the city, which provides potential landmarks or checkpoints (st+k) along the way to the destination. To reach the goal efficiently, the driver performs two tasks: Plans a high-level route: The driver identifies likely landmarks and routes to guide them toward the destination, akin to predicting st+k and the movement plan β(st,st+k). Executes immediate driving decisions: While en route, the driver makes real-time decisions (at), such as turning left or stopping at a traffic signal, informed by the planned route and the current position st.

This dual-segmentation approach effectively combines trajectory-based clustering with interaction-based segmentation, ensuring that the resulting segments capture both the motion patterns and the manipulation actions of the end effector. By integrating these two modalities, we achieve a richer and more accurate segmentation of the policy.

Finally, as a result of the segmentation process, we have a sequence of segments denoted as Σi = {σ1,σ2,...,σn}, where n is the number of segments. Here, a segment is expressed as σ = {st,st+1,...,st+k}, comprising k states.

Without the ability to establish landmarks or checkpoints (future states) and plan routes based on them, the driver would rely solely on reactive decisions, leading to inefficiencies or incorrect paths. By integrating both the high-level plan and immediate feedback, the driver ensures purposeful and adaptive progress toward the goal. Following this analogy, we calculate the look-ahead gripper position and movement plan to reach there.

##### 3.2 Data Generation

After obtaining the segments, we generate hierarchical embodied planning data for each demonstration, as shown in Figure 2. For each segment of a demonstration, we produce the 2D end-effector position and 3D movements for the completion state of the current segment. Additionally, we generate grounded reasoning for the corresponding subtasks.

Look-ahead Gripper Position Generation. Following (Zawalski et al., 2024), we also use OWLv2 (Minderer et al., 2024) and SAM (Kirillov et al., 2023) to detect 2D gripper position, which can be

2We use λ as 1 and β as 0.03 for best segmentation.

seen in Figure 2(e). The difference is that they train the model to output only the gripper position for the current input state, whereas, in our data construction process, we use the current gripper position as input and predict the gripper position for the first state of the next segment. Thus, although both approaches utilize the gripper position, our model focuses more on predicting the gripper position in future states during training, rather than identifying its position in the current state. Let’s consider for every state st, we obtain gt, the gripper position of the first state of the next segment.

Look-ahead Movement Plan Generation. As shown in Figure 2(d), we infer the 3D spatial positions corresponding to the current state and the end state of the current segment using the state policy of the robot. Specifically, we calculate the displacement between these two positions to determine the direction and step size required for the manipulator to move from the current state to the end state. Following the motion language idea in RT-H (Belkhale et al., 2024), we encode our highlevel motion plans using a standardized template in Appendix E. By integrating look-ahead spatial reasoning, the model incorporates both reactive and proactive decision-making. It combines immediate context at the current state st with a high-level plan that predicts likely future states st+k and the corresponding movement strategy β(st,st+k). This dual focus enables the model to align immediate actions with the overarching goal, ensuring purposeful and adaptive task execution. Please note that this data is not directly executed as the robot’s actions. Let’s consider for every state st, we will obtain mt, the movement plan to the first state of the next segment.

Grounded Chain-of-Thought Reasoning. As shown in Figure 2(f) and (g), we utilize Gemini 3 to derive the subtask corresponding to each segment, along with scene understanding and the reasoning behind the series of actions the robot needs to perform the subtask. Specifically, we take sequences of segmented images, and task descriptions as input to guide Gemini in generating the subtask and grounded reasoning for each segment. Compared to (Zawalski et al., 2024) that infer subtasks and their mapping to states solely from textual information, our approach first segments the sequence based on the robot’s motion trajectory and grip-

3We used gemini-1.5-pro-latest for our data generation.

per’s state as explained in Section 3.1. After that, based on the given multimodal information, we generate the corresponding subtasks and the reasoning of each subtask. Note that each subtask can comprise multiple segments. For the i-th trajectory, we obtain the grounded reasoning from Gemini, defined as: GRi = (σk,Sk,Rk) | k = 1,...,n , where: - σk is the k-th segment, - Sk is the subtask label assigned to σk, - Rk is Gemini’s justification for assigning subtask Sk to σk, and - n is the total number of segments in the trajectory. The prompt template can be seen in the Appendix B.

The Final Dataset. The final dataset for the i-th trajectory in the training dataset is defined as: {Di}Ni=1 = {Xi,Yi}Ni=1 =

N i=1, where t =

{(st,Ti),(mt,gt,GRt,at)}Tt=1

1,2,...,T, and T is the total number of timesteps in the trajectory.

##### 3.3 EMMA-X

In this section, we introduce the architecture of our proposed EMMA-X which is a 7B-parameter VLA model fine-tuned from OpenVLA using our constructed hierarchical embodiment data. As shown in Figure 3, we adjust the text prompt with the current gripper position and add chain-of-thought training to enhance the ability of spatial reasoning and scene understanding before predicting the next robot action policy.

During the process of predicting for real robot testing, we input the task description, the current observation image, and the 2D gripper position detected in real-time by OWLv2 (Minderer et al., 2024) and SAM (Kirillov et al., 2023). EMMAX first outputs the subtask and a description of the current scene, including the spatial relationship between the target object in the image and the robotic arm, as well as the operational instructions required for the gripper to reach the goal of the current subtask. Additionally, EMMA-X also predicts the target position the gripper needs to reach after completing the sub-task, including both the 2D location in the image and the 3D spatial movements. Finally, the model outputs the next 7D robot action policy for downstream manipulation.

4 Experiments

##### 4.1 Implementation Details

To create the hierarchical reasoning dataset, we employed our data creation pipeline on full

OpenVLA ECoT EMMA-X (Ours)

Category Task

h_Succ (%) Succ (%) h_Succ (%) Succ (%) h_Succ (%) Succ (%)

SPATIAL RELATION Put the upper half of the carrot in the pot 30 10 35 20 80 60 SPATIAL RELATION Put the left half of the lemon in the pan 30 0 35 10 55 20 SPATIAL RELATION Put the blue cube on the left plate 25 20 5 0 60 60 SPATIAL RELATION Put the blue cube on the right plate 60 60 35 20 90 90

OOD OBJECT Put the banana in pot 70 50 45 40 85 70 OOD OBJECT Put the blue cube on the plate 90 90 20 10 85 70 OOD OBJECT Wipe the stove with towel 70 50 50 30 90 90

OOD INSTRUCTION Pick up any object that is a kind of vegetable 40 30 15 0 75 70 OOD INSTRUCTION Put the inedible object on the towel 0 0 25 0 40 30 OOD INSTRUCTION Put the edible object on the towel 0 0 15 10 35 0

IN DOMAIN Open microwave 50 30 25 0 65 30 IN DOMAIN Close microwave 80 60 45 40 100 100

Average 45.41 33.33 28.75 15.00 71.66 57.50

Table 1: Experimental results of EMMA-X and baselines on 12 real-world WidowX-250 robot manipulation tasks.

100

Avgh_SucceRate()%

ECoT OpenVLA EMMA-X

domain scenarios, out-of-domain (OOD) objects, spatial relationships, and OOD instructions. All policies are assessed on identical real-world setups to ensure consistency in camera angle, lighting conditions, and background. Each task is conducted over 10 trials, adhering to the methodology established by OpenVLA. If the robot can successfully achieve the task specified inside the prompts, it is counted as a success (succ) receiving a score of 1, otherwise, a score of 0 is assigned. Following OpenVLA, we also introduce a "half-success" (hsucc) metric that considers both the task goal and difficulty and assigns a 0.5 score only when the half-success criteria are met (Appendix C).

90

80

70

60

50

40

30

20

10

0

SPATIAL RELATION OOD OBJECT IN DOMAIN OOD INSTRUCTION

Different Categories of Tasks

Figure 4: Experimental results on different categories of real-world robot tasks.

BridgeData-v2, which consists of approximately 60,000 trajectories paired with task instructions, resulting in an augmented dataset.

##### 4.3 Baselines

To train our VLA models, we employed OpenVLA, a 7B vision-language-action (VLA) model built upon the Prismatic vision-language framework and pretrained on the Open X-Embodiment dataset. For autoregressive training, we tokenized our 7-dimensional action policy into discrete policy tokens, consistent with OpenVLA’s methodology. We adhered to OpenVLA’s training procedure and fine-tuned the base model on our augmented dataset for 3 epochs until convergence.

To comprehensively evaluate the performance of our proposed EMMA-X, we conduct extensive experiments across 12 different tasks on the real robot with several competitive methods.

OpenVLA (Kim et al., 2024): A VLA model based on large-scale VLM Prismatic7b and pre-trained on the Open-X-Embodiment dataset(Collaboration et al., 2023).

OpenVLA w/ FT: For a fair comparison, we finetuned the OpenVLA model on the BridgeV2 dataset for the same number of epochs following the same training setting in our method.

##### 4.2 Robot Setup and Metrics

We evaluate our approach using the 6-DoF WidowX robot arm, as introduced in the Bridge V2 paper, which represents a standard benchmark for assessing generalizable robotic policies. The policy takes as input a single third-person camera feed and a natural language instruction, predicting endeffector velocity actions to control the robot.

ECoT (Zawalski et al., 2024): A VLA model fine-tuned from OpenVLA on BridgeV2 dataset (Walke et al., 2023) with their generated chain-ofthought reasoning data.

##### 4.4 EMMA-X Improves Policy Generalization

In this section, we compare EMMA-X with several baselines on 12 real-world robotic tasks. As shown in Table 1, our EMMA-X outperforms the strong

To rigorously test the generalization capabilities of the policies, we develop a suite of challenging evaluation tasks that span multiple aspects: in-

Task: put edible object on the towel Current Gripper : [165, 79]

Task: wipe the stove with the towel Current Gripper : [142, 95]

Task: put the blue cube on the left plate Current Gripper: [104, 52]

|[Figure 82]<br><br>[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Subtask: Reaching for the edible object Reason: The robot arm needs to move towards the pineapple to grasp it. This is the first step towards picking it up Future State Gripper Position:[147, 94] 3D Movement : move backward 1 steps; move left 62 steps; move downward 52 steps

Subtask: Reaching for the towel Reason: The robot needs to grasp the towel to be able to wipe the stove Future State Gripper Position:[147, 132] 3D Movement : move forward 22 steps; move left 22 steps; move downward 72 steps

Subtask: Reaching for the blue cube Reason: The robot arm is moving downwards and positioning its gripper above the blue cube to prepare for grasping. Future State Gripper Position: [128, 124] 3D Movement: move forward 22 steps; move right 32 steps; move downward 142 steps

Figure 5: Qualitative examples of successful and failed cases with EMMA-X on real-world robot testing.

gains in OOD INSTRUCTION tasks, highlighting the efficacy of our grounded task reasoning.

SPATIAL RELATION OOD OBJECT OOD INSTRUCTION h_Succ (%) Succ (%) h_Succ (%) Succ (%) h_Succ (%) Succ (%)

Models

EMMA-X 77 70 88 80 75 70

w/o mt 42 (↓ 35) 37 (↓ 33) 63 (↓ 25) 55(↓ 25) 40 (↓ 35) 30 (↓ 40) w/o gt 32 (↓ 45) 30 (↓ 40) 45 (↓ 43) 35 (↓ 45) 45 (↓ 30) 30 (↓ 40) w/o GRt 22 (↓ 55) 10 (↓ 60) 45 (↓ 43) 40 (↓ 40) 25 (↓ 50) 20 (↓ 50) w/o HDBSCAN 27 (↓ 50) 20 (↓ 50) 53 (↓ 35) 35 (↓ 45) 65 (↓ 10) 40 (↓ 30)

##### 4.5 Analysis

We trained several variants of EMMA-X to evaluate the roles of segmentation, look-ahead spatial reasoning, and grounded chain-of-thought (CoT) reasoning, which collectively constitute the core of EMMA-X. For this evaluation, we sampled 6 prompts across SPATIAL RELATION, OOD OBJECT, and OOD INSTRUCTION (prompts are indicated in magenta color in Section C). For each prompt, we conducted 10 rollouts under the same experimental setup as our main experiments.

OpenVLA 38 30 70 50 40 30 w/ FT 28 (↓ 10) 23 (↓ 17) 65 (↓ 5) 50 (↓ 0) 15 (↓ 25) 0 (↓ 30)

Table 2: Models with different configurations.

baseline OpenVLA, with a 24.17% increase in task success rate and a 26.25% increase in half success rate. This demonstrates the effectiveness of our constructed hierarchical embodiment dataset. In addition, compared to ECOT, our EMMA-X shows significant gains, which can be caused by the following: 1) ECoT suffers from noisy training data, which causes hallucinations when faced with out-of-domain instructions or unfamiliar objects, leading to task failures. Interestingly, even for IN DOMAIN tasks, it performs poorly compared to other models, highlighting its limited reasoning capabilities. Our grounded task reasoning approach addresses this by incorporating the segmented visual images, ensuring more accurate task understanding. 2) EMMA-X enhances spatial reasoning by predicting the 2D gripper position of the end state of the current segment and 3D spatial movements to transit to it before predicting the next robot action policy.

Segmentation Greatly Helps the Policy. To evaluate the effectiveness of our segmentation technique, we conducted an experiment where sequences were segmented solely based on the gripper’s (end effector) open and close positions. The results, reported in Table 2 under the w/o HDBSCAN condition, show a general performance drop of 10% to 50%. Notably, spatial reasoning performance experienced the most significant decline, with a drop of 50%. These findings demonstrate that the distance metric introduced in Eq. 1 is crucial for the segmentation process.

The Impact of Look-ahead Spatial Reasoning. To evaluate the importance of look-ahead spatial reasoning, we conducted two experiments: 1) EMMA-X was trained without explicitly predicting the gripper’s position in the next segment, relying only on the predicted movement plan to reach the future gripper position of that segment (denoted as w/o gt in Table 2). This assumes that EMMA-X implicitly infers the future gripper position. 2) We trained EMMA-X to predict the future end effector’s position but without rolling out a movement

As shown in Figure 4, we also compared the average performance across various categories of robotic tasks. Notably, our method achieved the most significant performance improvement in SPATIAL RELATION tasks, outperforming OpenVLA by 35% and ECoT by 29% in the h_Succ rate. These results strongly validate the effectiveness of our predicted 3D spatial movements. Furthermore, our method demonstrated substantial performance

plan to reach that position (denoted as w/o mt in Table 2). The results reveal significant performance drops in both cases (25%-40% for “w/o mt” and 30% to 45% for “w/o gt”), with a more pronounced decline in spatial reasoning tasks (35% for “w/o mt” and 45% for “w/o gt”). Furthermore, the results suggest that predicting the future end effector’s position is more critical, as the performance drop in the absence of 3D spatial movements to the next segment is less severe. We hypothesize that this may be due to OpenVLA’s inherent spatial reasoning capabilities, which enable it to more easily transition between positions.

The Importance of Grounded CoT Reasoning. Grounded chain-of-thought (CoT) reasoning is a foundational element of EMMA-X. To assess its impact, we trained a variant of EMMA-X without grounded reasoning, while retaining look-ahead spatial reasoning in the data. The results show a marked performance drop by 43%-55%, highlighting that spatial reasoning alone is insufficient. Interestingly, the absence of grounded CoT reasoning resulted in a more severe decline in spatial reasoning performance compared to models where spatial reasoning capabilities were explicitly ablated. This underscores the critical role of grounded CoT in tackling complex reasoning tasks, including spatial reasoning. Therefore, we surmise that for enhancing the generalizable policies of Vision-LanguageAction (VLA) models, it is essential to improve their broader reasoning capabilities, encompassing object recognition, color understanding, abstraction, commonsense knowledge, and more.

Fine-tuning does not Improve OpenVLA. We sought to find whether fine-tuning OpenVLA on BridgeV2 could match the performance of EMMAX. The results, shown in Table 2, reveal that OpenVLA’s performance degrades by 5%-30% after fine-tuning with the worst performance observed for OOD INSTRUCTION. We hypothesize that this decline is due to overfitting, as BridgeV2 is already part of OpenVLA’s pre-training dataset.

Qualitative Analysis on Real-world Robot Task. To qualitatively evaluate the effectiveness of our spatial and task reasoning in guiding robotic actions, we present two successful trajectories and one failed trajectory in Figure 5. From the left case, we find that the predicted gripper position corresponds to the end state of the subtask “reaching for the blue cube”. The 3D movement provides

a detailed path, clearly directed toward the “blue cube”. We also include a failed trajectory where the “hotdog" is mistakenly identified as a “pineapple". This error propagates, impacting the prediction of the gripper’s future position and preventing it from accurately picking up the “hot dog”.

#### 5 Conclusion

We introduce EMMA-X, a 7B-parameter embodied multimodal action model designed to enhance spatial reasoning and task planning for robotic policy generation. We construct a hierarchical embodiment dataset enriched with grounded reasoning, including 2D gripper positions and 3D spatial movements. Furthermore, our proposed trajectory segmentation strategy reduces hallucination in task reasoning by grounding reasoning in visual images. The experimental results demonstrate the effectiveness of EMMA-X, showing significant improvements over existing baselines in tasks requiring long-horizon spatial reasoning.

#### Limitations

While EMMA-X shows promising performance, its latency remains higher compared to OpenVLA. This increased inference time primarily results from the additional tokens generated during the reasoning process. Specifically, EMMA-X generates approximately 10 times more tokens than OpenVLA. To mitigate this, a potential strategy is to predict all policies within a segment and only regenerate the policy if the predicted policy deviates significantly from the expected movement plan. Another limitation is the generalization capability of EMMA-X. Scaling the training process to incorporate a larger subset of the OXE dataset could enhance the model’s ability to handle a broader range of tasks and robotic systems. Lastly, using SAM for detecting the gripper position can lead to inaccuracies. These errors may occur when the gripper is partially occluded by objects or positioned outside the image frame. Employing a more robust model for detecting and segmenting the robot hand could address these challenges and improve reliability.

#### References

Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh.

2024. Rt-h: Action hierarchies using language. Preprint, arXiv:2403.01823.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. 2023a. Rt-2: Visionlanguage-action models transfer web knowledge to robotic control. Preprint, arXiv:2307.15818.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. 2023b. Rt-1: Robotics transformer for real-world control at scale. Preprint, arXiv:2212.06817.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668.

Open X-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Schölkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen,

Deepak Pathak, Dhruv Shah, Dieter Büchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad AbouChakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, João Silvério, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi "Jim" Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick "Tree" Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart’in-Mart’in, Ro-

- han Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xing-
- hao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei,

Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. 2023. Open X-Embodiment: Robotic learning datasets and RT-X models. https://arxiv.org/abs/2310.

08864.

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. 2023. Palm-e: An embodied multimodal language model. Preprint, arXiv:2303.03378.

Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. 2021. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. Preprint, arXiv:2109.13396.

Huy Ha, Pete Florence, and Shuran Song. 2023. Scaling up and distilling down: Language-guided robot skill acquisition. Preprint, arXiv:2307.14535.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. 2024. OpenVLA: An open-source vision-language-action model. In 8th Annual Conference on Robot Learning.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023. Code as policies: Language model programs for embodied control. Preprint, arXiv:2209.07753.

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. 2024. A survey on vision-languageaction models for embodied ai. arXiv preprint arXiv:2405.14093.

Leland McInnes, John Healy, and S. Astels. 2017. hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2:205.

Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. 2024. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. 2024. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Homer Rich Walke, Kevin Black, Tony Z. Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, Abraham Lee, Kuan Fang, Chelsea Finn, and Sergey Levine. 2023. Bridgedata v2: A dataset for robot learning at scale. In 7th Annual Conference on Robot Learning.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. 2024. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11941–11952. IEEE.

#### A Related Work

Generalist Robot Policies. Recent progress in robotics has shifted focus towards developing multi-task "generalist" robot policies capable of handling a wide variety of tasks across diverse robot embodiments (Brohan et al., 2023b,a; Ebert et al., 2021; Walke et al., 2023; Collaboration et al., 2023; Octo Model Team et al., 2024; Kim et al., 2024). For example, Octo (Octo Model Team et al., 2024) utilizes a compositional design to train a generalist policy capable of handling various tasks directly while supporting fine-tuning for new inputs and action spaces. Similarly, OpenVLA (Kim et al., 2024) adopts a streamlined end-to-end approach, fine-tuning vision-language models (VLMs) to produce robot actions by treating these actions as tokens within the language model’s vocabulary. These studies highlight the potential of training robot policies on large and diverse datasets as a promising strategy for enhancing their performance.

Vision-Language-Action Models. A number of recent works have explored fine-tuning large pretrained VLMs for predicting robot actions (Collaboration et al., 2023; Brohan et al., 2023a; Kim et al., 2024; Octo Model Team et al., 2024; Driess et al., 2023) Such models are often referred to as vision-language-action models (VLAs), as they fuse robot actions directly into VLM backbones and treating these actions as tokens within the language model vocabulary. This approach provides a simple yet scalable alternative, with models such as RT-2 (Brohan et al., 2023b), RT-2-X (Collaboration et al., 2023), and OpenVLA (Kim et al., 2024) demonstrating state-of-the-art performance and impressive generalization across diverse objects and environments. RT-2 integrates Internet-scale vision-language data with robotic trajectory data, while RT-2-X scales this further with a 55B-parameter policy trained on the Open X-Embodiment dataset (Collaboration et al., 2023). In contrast, OpenVLA integrates a robust open VLM backbone with an enriched robot pretraining dataset. Despite these advancements, current VLAs underutilize some of the most valuable features of their underlying language and vision-language models, specifically, their capacity to reason through the steps needed to solve complex tasks.

Reasoning for Robotics. Prompting large language models (LLMs) to "think step-by-step" (Kojima et al., 2022) when solving problems can significantly enhance their performance. Similar techniques have been explored in the context of high-level task planning for robotics (Liang et al., 2023; Ha et al., 2023). Expanding on this, Zawalski et al. (2024) introduced ECoT, a method that trains a VLA policy to autoregressively generate chain-of-thought (CoT) reasoning. ECoT combines high- and low-level reasoning with actionable steps, aligning these to an agent’s environment based on input instructions and observations. While this equips VLAs with a better understanding of the current situation and task, it falls short in two key areas: long-horizon spatial reasoning for robot movement and visual grounding when augmenting reasoning data with Gemini. Building on these limitations, we build EMMA-X designed to enhance spatial reasoning and task planning in robotic policy generation. EMMA-X integrates grounded chain-of-thought reasoning to predict future states and transitions, enabling robots to develop more informed and effective action policies.

#### B Prompts and Sample Outputs

Template for prompting Gemini to generate subtasks and reasoning.

The robot successfully completed a task specified by the instruction: '{instruction }'. Here is a list of segments of images for the robot hand to perform the task specified by the instruction.

Based on the sequence of segments provided in sequential order , pay attention to the robot hand and identify which subtask it is performing in each segment. Provide the justification for why the subtask should be done based on the environment. You can assign the same subtask to multiple segments.

You should output in dictionary format: {segment_number: [subtask , reason for justification], ...}. The segment_number starts from 1 and must be an integer. The output dictionary keys correspond to each segment , and the output dictionary

length should be the same as the number of segments. {segment_count}

- Segment 1: images in segment 1

- Segment 2: images in segment 2

... Segment n: images in segment n

We list the Gemini output of the example in Figure 1 as follows:

|Here\'s a breakdown of the robot\'s subtasks in each segment: ```json{<br><br>1: ["Positioning over pot cover", "The robot arm needs to move to the correct location above the pot cover to be able to grasp it. This involves adjusting its position and orientation."],<br><br>2: ["Grasping the pot cover", "The robot is closing its gripper around the pot cover\'s handle to securely pick it up. This requires precision to ensure a stable grip."],<br><br>3: ["Lifting and moving the pot cover", "The robot has grasped the pot cover and is now lifting it up and moving it towards the silver pot. It must avoid<br><br>collision with other objects in the environment."],<br><br>4: ["Positioning and placing the cover on the pot", "The robot carefully positions the cover above the silver pot and lowers it into place. This requires aligning the cover with the pot\'s opening and releasing the grip gently to avoid dropping or damaging the cover or pot."]<br><br><br>}<br><br>|
|---|

#### C Half Success Criteria

The half-success criteria to for the metric h_success are as follows. The prompts in magenta color are used to report the results in Table 2.

|Task<br><br>|Half Success Criteria|
|---|---|
|Put the banana in pot|Robot successfully grasps the banana.|
|Put the upper half of the carrot in the pot<br><br>|Robot moves towards or grasps the correct target object.|
|Pick up any object that is a kind of vegetable|Robot moves towards the correct vegetable.|
|Put the left half of the lemon in the pan|Robot moves towards and correctly selects the left half of the lemon.|
|Put the blue cube on the plate<br><br>|Robot grasps the blue cube.|
|Put the blue cube on the left plate|Robot grasps the blue cube and moves it towards the left plate.|
|Put the blue cube on the right plate<br><br>|Robot grasps the blue cube and moves it towards the right plate.|
|Put the inedible object on the towel|Robot moves towards the correct inedible object.|
|Put the edible object on the towel<br><br>|Robot moves towards the correct edible object.|
|Wipe the stove with a towel<br><br>|Robot touches the towel but does not wipe the stove.|
|Open the microwave|Robot partially opens the microwave door.|
|Close the microwave<br><br>|Robot partially closes the microwave door.|

#### D Segmentation Statistics

Average Number of frames per segment: 5.5 Average Number of segments per trajectory: 6.9 Average Number of frames per trajectory: 32.8

#### E Motion Plan Template

Translational Movements: move (left/right) x steps, move (forward/backward) y steps, move (upward/downward) z steps. Rotational Movements: pitch (upward/downward) α degrees, yaw (left/right) β degrees, roll (clockwise/counterclockwise) γ degrees. Gripper Action: (open/close) gripper.

#### F Pseudo Code for Training EMMA-X and Running Inference Notations defined in 2.2 Algorithm 1 Data Generation, Training, and Inference Process For each sample i in Embodied Dataset, we have: T: Number of time frames

- S = {st}Tt=1: Images at each time frame t, where st is the image at time frame t

G = {gt}Tt=1: Gripper poses at each time frame t (position, orientation, and open-or-close state)

- T : Task instruction in natural language format

##### Training Process:

- 1: while not converged do
- 2: for each sample i do
- 3: Mframes→segments ← dual_segmentation(G) ▷ Mapping from frame to segment
- 4: Msegments→subtasks, reasons ← Gemini(S,G,T ) ▷ Mapping from segment to subtasks
- 5: for each time frame t ∈ {1,2,...,T} do
- 6: σt ← Mframes→segments(t) ▷ Get segment for time t
- 7: GRt ← Msegments→subtasks, reasons(σt) ▷ Get grounded reasoning from Gemini
- 8: gt ← SAM(st) ▷ Get 2D gripper position Using SAM model
- 9: gend ← SAM(Send) ▷ Get 2D gripper position at end of current segment
- 10: mt ← Template(gt − gend) ▷ Translational change to movement plan in natural language
- 11: prediction ← Model(T ,st,gt,GRt,gend,mt,at) ▷ Perform supervised fine-tuning (SFT) with label: (T ,st,gt,GRt,gend,mt,at)
- 12: end for
- 13: end for
- 14: end while Inference Process:

- 1: while Task not completed do
- 2: gt ← SAM(st)
- 3: GRt,gend,mt,at ← EMMA-X(T ,st,gt)
- 4: Control the robot using at, to get new st,gt
- 5: end while

