## OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints

Mingjie Pan1,2∗, Jiyao Zhang1,2∗, Tianshu Wu1, Yinghao Zhao3, Wenlong Gao3, Hao Dong1,2† 1CFCS, School of CS, Peking University 2PKU-AgiBot Lab 3AgiBot https://omnimanip.github.io

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

# arXiv:2501.03841v1[cs.RO]7Jan2025

[Figure 9]

[Figure 10]

[Figure 11]

###### Pour tea into the cup.

“Open drawer”

“Pour tea into the cup” “Recycle battery”

“Press button with hammer”

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

“Insert flower into vase” “Insert pen into holder” “Close drawer” “Press green button”

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Active Passive

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

“Pick up cup on dish”

“Put the lid on the teapot” “Close lid of laptop” “Open bottle”

Figure 1. We proposed OmniManip, an open-vocabulary manipulation method that bridges the gap between the high-level reasoning of vision-language models (VLM) and the low-level precision, featuring closed-loop capabilities in both planning and execution.

### Abstract

loop for high-level planning through primitive resampling, interaction rendering and VLM checking, and another for low-level execution via 6D pose tracking. This design ensures robust, real-time control without requiring VLM finetuning. Extensive experiments demonstrate strong zero-shot generalization across diverse robotic manipulation tasks, highlighting the potential of this approach for automating large-scale simulation data generation.

The development of general robotic systems capable of manipulating in unstructured environments is a significant challenge. While Vision-Language Models(VLM) excel in high-level commonsense reasoning, they lack the finegrained 3D spatial understanding required for precise manipulation tasks. Fine-tuning VLM on robotic datasets to create Vision-Language-Action Models(VLA) is a potential solution, but it is hindered by high data collection costs and generalization issues. To address these challenges, we propose a novel object-centric representation that bridges the gap between VLM’s high-level reasoning and the low-level precision required for manipulation. Our key insight is that an object’s canonical space, defined by its functional affordances, provides a structured and semantically meaningful way to describe interaction primitives, such as points and directions. These primitives act as a bridge, translating VLM’s commonsense reasoning into actionable 3D spatial constraints. In this context, we introduce a dual closedloop, open-vocabulary robotic manipulation system: one

### 1. Introduction

Developing a general robotic manipulation system has long been a challenging task, primarily due to the complexity and variability of real-world [26, 47, 48]. Inspired by the rapid advancements in Large Language Models (LLM)[1, 42] and Vision-Language Models (VLM) [25, 28, 34, 54], which leverage vast amounts of internet data to acquire rich commonsense knowledge, researchers have recently turned attention to exploring their application in robotics[14, 53]. Most existing works focus on utilizing this knowledge for high-level task planning, such as semantic reasoning [4, 31, 37]. Despite these advances, current VLMs, primarily trained on extensive 2D visual data, lack the 3D spatial

*: Equal contributions. †: Corresponding author

understanding ability necessary for precise, low-level manipulation tasks. This limitation poses challenges in manipulations within unstructured environments.

One approach to overcoming this limitation is to finetune VLM on large-scale robotic datasets, transforming them into VLA [2, 3, 8, 19]. However, this faces two major challenges: 1) acquiring diverse, high-quality robotic data is costly and time-consuming, and 2) fine-tuning VLM into VLA results in agent-specific representations, which are tailored to specific robots, limiting their generalizability. A promising alternative is to abstract robotic actions into interaction primitives (e.g., points or vectors) and leverage VLM reasoning to define the spatial constraints of these primitives, while traditional planning algorithms handle execution [13, 15, 27]. However, existing methods for defining and using primitives have several limitations: The process of generating primitive proposals is task-agnostic, which poses the risk of lacking suitable proposals. Additionally, relying on manually designed rules for post-processing proposals also introduces instability. This naturally leads to an important question: How can we develop more efficient and generalizable representations that bridge VLM high-level reasoning with precise, low-level robotic manipulation?

To address this challenge, we propose a novel objectcentric intermediate representation incorporating interaction points and directions within an object’s canonical space. This representation bridges the gap between VLM’s high-level commonsense reasoning and precise 3D spatial understanding. Our key insight is that an object’s canonical space is typically defined based on its functional affordances. As a result, we can describe an object’s functionality in a more structured and semantically meaningful way within its canonical space. Meanwhile, recent advancements in universal object pose estimation [7, 55, 56] make it feasible to canonicalize a wide range of objects.

Specifically, we employ a universal 6D object pose estimation model [56] to canonicalize objects and describe their rigid transformations during interactions. In parallel, a single-view 3D generation network generates detailed object meshes [29, 40]. Within the canonical space, interaction directions are initially sampled along the object’s principal axes, providing a coarse set of interaction possibilities. Meanwhile, the VLM predicts interaction points. Subsequently, the VLM identifies task-relevant primitives and estimates the spatial constraints between them. To address the hallucination issue in VLM reasoning, we introduce a self-correction mechanism through interaction rendering and primitive resampling that enables closed-loop reasoning. Once the final strategy is determined, actions are computed through constrained optimization, with pose tracking ensuring robust, real-time control in a closed-loop execution phase. Our method offers several key advantages: 1) Efficient and Effective Interaction Primitive Sampling:

By leveraging the object’s canonical space, our approach enables efficient and effective sampling of interaction primitives, enhancing the system’s reasoning capabilities. 2) Dual Closed-Loop, Open-Vocabulary Robotic Manipulation System: Benefiting from the proposed object-centric intermediate representation, our method implements a dual closed-loop system. The rendering and resampling process drives a reasoning loop for decision-making, while pose tracking ensures a closed loop for action execution.

In summary, our contributions are threefold:

- • We propose a novel object-centric interaction representation that bridges the gap between VLM’s high-level commonsense reasoning and low-level robotic manipulation.
- • To the best of our knowledge, we are the first to present a planning and execution dual closed-loop open-vocabulary manipulation system without VLM fine-tuning.
- • Extensive experiments demonstrate our method’s strong zero-shot generalization across diverse manipulation tasks, and we also highlight its potential for automating robotic manipulation data generation.

### 2. Related Work

Foundation Models For Robotics The emergence of foundation models has significantly influenced the field of robotics[11, 18, 51], particularly in the application of vision-language models[1, 4, 12, 23, 28, 50], which excel in environment understanding and high-level commonsense reasoning. These models demonstrate the potential for controlling robots to perform general tasks in novel and unstructured environments. Some studies [2, 3, 19, 24] have fine-tuned VLM on robotics datasets to create VLA models that output robotic trajectories, but these efforts are limited by the high cost of data collection and issues with generalization. Other approaches attempt to extract operation primitives using visual foundation models [9, 13, 15, 21, 27, 33, 52], which are then used as visual or language prompts for VLM to perform high-level commonsense reasoning, combined with motion planners [38, 39, 41] for low-level control. However, these methods are constrained by the ambiguity of compressing 3D primitives into the 2D images or 1D text required by VLM and the hallucination tendencies of VLM themselves, making it difficult to ensure that the high-level plans generated by VLM are accurate. In this work, we demonstrate OmniManip’s unique advantages in addressing these challenges, particularly in fine-grained 3D understanding and mitigating large model hallucinations.

Representations for Manipulation Structural representations determine the capabilities and effectiveness of manipulation methods. Among various types of representations, keypoints are a popular choice due to their flexibility, generalization, and ability to model variability [32, 35, 36, 46]. However, these keypoints-based methods require

[Figure 46]

#### Pour tea into the cup. Task Relevant Object Grounding and Task Stage Partitioning

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

###### STAGES

Vision Language Model

###### 1. Grasp the handle of teapot with gripper.

[Figure 54]

[Figure 55]

[Figure 56]

- • Action: Grasp
- • Active: Gripper
- • Passive: Teapot

[Figure 57]

Cup

[Figure 58]

[Figure 59]

[Figure 60]

Teapot

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Vision Foundation Model

###### 2. Pour tea from teapot into cup.

- • Action: Pour
- • Active: Teapot
- • Passive: Cup

[Figure 67]

[Figure 68]

| | |
|---|---|
| | |

[Figure 69]

#### Object-Centric Interaction Primitives as Spatial Constraints

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Ⅱ. Task related interaction primitives

Ⅰ. Mesh generation and canonicalization

Ⅲ. Primitives as spatial constraints

[Figure 74]

| | |
|---|---|
| | |
| | |

[Figure 75]

[Figure 76]

[Figure 77]

CONSTRAINTS

[Figure 78]

Rendering

[Figure 79]

[Figure 80]

- 1. Active axis and passive axis are aligned.
- 2. Active point is <5cm> <negative> along passive axis to passive point.

Active Passive

3D AIGC 6D Pose Estimator

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Closed-Loop Planning

[Figure 88]

Multiple-Constrained Online Trajectory Planning

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Optimized trajectory

6D Pose Tracker

[Figure 101]

Closed-Loop Execution

Figure 2. Overview framework. Given instruction and RGB-D observation marked by VFM, VLM firstly filters task-related objects and partitions the task into stages. For each stage, VLM extracts object-centric canonical interaction primitives as spatial constraints in a closed-loop manner. For execution, the trajectory is optimized by constraints and updated in a closed loop using a 6D Pose Tracker.

### 3. Method

manual task-specific annotations to generate actions. To enable zero-shot open-world manipulation, studies such as [15, 27, 33] have transformed keypoints into visual prompts for VLM, facilitating the automatic generation of high-level planning results. Despite their advantages, keypoints can be unstable; they struggle under occlusion and pose challenges in the extraction and selection of specific keypoints. Another common representation is the 6D pose, which efficiently defines long-range dependencies between objects for manipulation and offers a degree of robustness to occlusion [16, 17, 44, 45]. However, these methods necessitate prior modeling of geometric relationships and, due to the sparse nature of poses, cannot provide fine-grained geometry. This limitation can lead to failures in manipulation strategies across different objects due to intra-class variations. To address these issues, OmniManip combines the fine-grained geometry of keypoints with the stability of the 6D pose. It automatically extracts detailed functional points and directions within the canonical coordinate system of objects using VLM, enabling precise manipulation.

Here we discuss: (1) How do we formulate robotic manipulation via interaction primitives as spatial constraints(Sec. 3.1)? (2) How to extract canonical interaction primitives in a generic and open vocabulary way (Sec. 3.2)? (3) Why can OmniManip achieve a dual closed-loop system (Sec. 3.3)?

##### 3.1. Manipulation with Interaction Primitives

In our formulation, complex robotic tasks are decomposed into stages, each defined by object interaction primitives with spatial constraints. This structured approach allows for the precise definition of task requirements and facilitates the execution of complex manipulation tasks. In this section, we detail how interaction primitives serve as the foundation for spatial constraints, enabling robust manipulation.

Task Decomposition. As shown in Figure 2, given a manipulation task T (e.g., pouring tea into a cup), we first utilize GroundingDINO[30] and SAM[20], two Visual Foundation Models (VFMs), to mark all foreground objects in the scene like [49] as visual prompt. Subsequently, a VLM [1] is employed to filter task-relevant ob-

Visible and Tangible

[Figure 102]

[Figure 103]

###### Task: Pour tea into the cup

(𝑥 , 𝑦 ) (𝑥 , 𝑦 )

| | |
|---|---|
| | |
|)| |

- Stage1: Grasp the handle of teapot with gripper
- Stage2: Pour tea from teapot into the cup

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Grasp (Passive) Pour (Active)

Cup

[Figure 108]

[Figure 109]

Teapot

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

• Action: Pour

|• Active: Teapot<br>• Passive: Cup<br>| | |
|---|---|---|
| | | |

[Figure 114]

(𝑥 , 𝑦 ) (𝑥 , 𝑦

)

[Figure 115]

The axis passes through the teapot's center, extending outward from the

1

Invisible or Intangible

side, perpendicular to the spout outlet.

anguageisionodelMVL

anguageargeodelMLL

[Figure 116]

[Figure 117]

(𝑥 , 𝑦 ) (𝑥 , 𝑦 )

(𝑥 , 𝑦 ) (𝑥 ,𝑦 )

[Figure 118]

| | |
|---|---|
| | |
| | |

[Figure 119]

The axis is aligned vertically through the teapot, passing from the base to the top,

[Figure 120]

Pour/GraspPlace (Passive)(Passive)

perpendicular to the spout outlet.

[Figure 121]

[Figure 122]

The axis is aligned horizontally, passing through the teapot's center, extending outward from the spout outlet.

0

(𝑥 ,𝑦 ) (𝑥 , 𝑦 )

(𝑥 ,𝑦 ) (𝑥 , 𝑦 )

Principal axis Sorted axis

Figure 3. Interaction points generation.

Figure 4. Interaction directions extraction.

jects and decompose the task into multiple stages S = {S1,S2,...,Sn}, where each stage Si can be formalized as Si = {Ai,Oiactive,Oipassive}, where Ai represents the action to be performed (e.g., grasp, pour), and Oiactive and Oipassive refer to the object initiating the interaction and the object being acted upon, respectively. For example, in Figure 2, the teapot is the passive object in the stage of grasping the teapot while the teapot is the active object and the cup is passive in the stage of pouring tea into the cup.

##### 3.2. Primitives and Constraints Extraction

In this section, we detail the process of extracting interaction primitives and their spatial constraints C for each stage. As illustrated in Figure 2, we first obtain 3D object meshes for both the task-relevant active and passive objects via single-view 3D generation [29, 40, 57], followed by pose estimation with Omni6DPose[56] for object canonicalization. Next, we extract task-relevant interaction primitives and their corresponding constraints.

Object-Centric Canonical Interaction Primitives. We propose a novel object-centric representation with canonical interaction primitives to describe how objects interact during manipulation tasks. Specifically, an object’s interaction primitives are characterized by its interaction point and direction in canonical space. The interaction point p ∈ R3 denotes a key location on the object where interaction occurs, while the interaction direction v ∈ R3 represents the primary axis relevant to the task. Together, these form the interaction primitive O = {p,v}, encapsulating the essential intrinsic geometric and functional properties required to meet task constraints. These canonical interaction primitives are defined relative to their canonical space, remaining consistent across different scenarios, enabling more generalized and reusable manipulation strategies.

Grounding Interaction Point. As shown in Figure 3, interaction points are categorized as Visible and Tangible (e.g., a teapot handle) or Invisible or Intangible (e.g., the center of its opening). To enhance VLM for interaction points grounding, SCAFFOLD [22] visual prompting mechanism is employed, which overlays a Cartesian grid onto the input image. Visible points are directly localized in the image plane, while invisible points are inferred through multi-view reasoning based on proposed canonical object representations, as illustrated in Figure 3. Reasoning begins from the primary viewpoint, with ambiguities resolved by switching to an orthogonal view. This approach enables more flexible and reliable interaction point grounding. For tasks like grasping, heatmaps are generated from multiple interaction points, improving the robustness of the grasping model.

Interaction Primitives with Spatial Constraints. At each stage Si, a set of spatial constraints Ci governs the spatial relationships between the active and passive objects. These constraints are divided into two categories: distance constraints di, which regulate the distance between interaction points, and angular constraints θi, which ensure proper alignment of interaction directions. Together, these constraints define the geometric rules necessary for precise spatial alignment and task execution. The overall spatial constraint for each stage Si is given by:

Sampling Interaction Direction. In the canonical space, the principal axes of an object are often functionally relevant. As illustrated in Figure 4, we treat the principal axes as candidate interaction directions. However, assessing the relevance of these directions to the task is challenging due to the limited spatial understanding of the current VLM. To address this, we propose a VLM caption and LLM scoring mechanism: first, we use the VLM to generate semantic descriptions for each candidate axis, and then employ a LLM to infer and score the relevance of these descriptions to the task. This process results in an ordered set of candidate directions that are most aligned with the task requirements.

Ci = Oiactive,Oipassive,di,θi (1) Once the constraints Ci have been defined, the task exe-

Ultimately, the interaction primitives with constraints are generated with VLM, yielding an ordered list of constrained

cution can be formulated as an optimization problem.

interaction primitives for each stage Si, denoted as Ki = {Ci(1),Ci(2),...,Ci(N)}.

##### 3.3. Dual Closed-Loop System

As outlined in Section 3.2, we obtain the interaction primitives of the active and passive objects, denoted as Oactive and Opassive, respectively, along with the spatial constraints C that define their spatial relationships. However, this is an open-loop inference, which inherently limits the robustness and adaptability of the system. These limitations arise primarily from two sources: 1) the hallucination effect in large models, and 2) the dynamic nature of real-world environments. To overcome these challenges, we propose a dual closed-loop system, as illustrated in Figure 2.

Algorithm 1 Self-Correction Algorithm via RRC Input: Task T , Stage Si, Initial List of Primitives with Constraints Ki = Ci(1),Ci(2),...,Ci(N) Output: Successful Constraints Cˆi or Task Failure

- 1: k ← 1, maxSteps ← N, refine ← False
- 2: while k ≤ maxSteps do
- 3: k ← k + 1
- 4: Render: Ii ← Render(Ci(k))
- 5: Check: state ← VLM(T ,Si,Ii,Ci(k),refine)
- 6: if state = ‘Refine’ and refine = False then
- 7: Resample: Update Ki ← Resample(Ci(k))
- 8: k ← 1, maxSteps ← M, refine ← True
- 9: else if state = ‘Success’ then
- 10: return Ci(k)
- 11: end if
- 12: end while
- 13: return Task Failed

Closed-loop Planning. To improve the accuracy of interaction primitives and mitigate hallucination issues in the VLM, we introduce a self-correction mechanism based on Resampling, Rendering, and Checking (RRC). This mechanism uses real-time feedback from a visual language model (VLM) to detect and correct interaction errors, ensuring precise task execution. The RRC process consists of two stages: the initial phase and the refinement phase. The overall RRC mechanism is outlined in Algorithm 1. In the initial phase, the system evaluates the interaction constraints Ki defined in Section 3.2, which specify the spatial relationships between active and passive objects. For each constraint Ci(k), the system renders an interaction image Ii based on the current configuration and submits it to the VLM for validation. The VLM returns one of three outcomes: success, failure, or refinement. If success, the constraint is accepted, and the task proceeds. If failure, the next constraint is evaluated. If refinement, the system enters the refinement phase for further optimization. In the refine-

ment phase, the system performs fine-grained resampling around the predicted interaction direction vi to correct misalignments between the functional and geometric axes of objects. The system uniformly samples six refined directions vi(j) around vi and evaluates them.

Closed-loop Execution. Once the interaction primitives and the corresponding spatial constraints C are defined for each stage, the task execution can be formulated as an optimization problem. The objective is to minimize the loss function to determine the target pose Pee∗ of the endeffector. The optimization problem can be expressed as:

 

 

N

Pee∗ = arg min

Lj(Pee)

, s.t.

(2)





Pee

j=1

L = {LC,Lcollision,Lpath},

where the constraint loss LC ensures that the action adheres to the task’s spatial constraints C, and is defined as

LC = ρ(C,Pactivet ,Ppassivet ), where Pactivet = Φ(Peet ) (3)

Here, ρ(·) measures the deviation between the current spatial relationship of the active object Pactivet and the passive object Ppassivet from the desired constraint C, while Φ(·) maps the end-effector pose to the active object’s pose. The collision loss Lcollision prevents the end-effector from colliding with obstacles in the environment and is defined as

N

max(0,dmin − d(Pee,Oj))2 , (4)

Lcollision =

j=1

where d(Pee,Oj) represents the distance between the end-effector and the obstacle Oj, and dmin is the minimum allowable safety distance. The path loss Lpath ensures smooth motion and is defined as

Lpath = λ1dtrans(Peet ,Pee) + λ2drot(Peet ,Pee), (5)

where dtrans(·) and drot(·) represent the translational and rotational displacements of the end-effector, respectively, and λ1 and λ2 are weighting factors that balance the influence of translation and rotation. By minimizing these loss functions, the system dynamically adjusts the end-effector pose Pee, ensuring successful task execution while avoiding collisions and maintaining smooth motion.

While Equation 3 outlines how interaction primitives and their corresponding spatial constraints can be leveraged to optimize the executable end-effector pose, real-world task execution often involves significant dynamic factors. For instance, deviations in the grasp pose may result in unintended object movement during a grasping task. Moreover, in certain dynamic environments, the target object may be displaced. These challenges highlight the critical importance of closed-loop execution in handling such uncertainties. To address these challenges, our system leverages the

###### Tasks VoxPoser CoPa ReKep OmniManip(Ours)

Auto Closed-loop Open-loop

Pour tea 0/10 1/10 3/10 7/10 6/10 Insert flower into vase 0/10 4/10 2/10 6/10 4/10 Insert the pen in holder 0/10 4/10 3/10 7/10 5/10 Recycle the battery 6/10 5/10 7/10 8/10 6/10 Pick up the cup on the dish 3/10 2/10 9/10 8/10 7/10 Fit the lid onto the teapot 0/10 2/10 3/10 5/10 3/10

- Total 15.0% 30.0% 45.0% 68.3% 51.7%

Open the drawer 1/10 4/10 - 6/10 4/10 Close the drawer 3/10 3/10 - 8/10 6/10 Hammer the button 0/10 3/10 - 4/10 2/10 Press the red button 0/10 3/10 - 7/10 6/10 Close the lid of the laptop 4/10 3/10 - 6/10 4/10 Open the jar 2/10 0/10 - 6/10 5/10

- Total 16.7% 26.7% - 61.7% 45.0%

- Table 1. Quantitative results across 12 real-world manipulation tasks. The first six tasks focus on rigid object manipulation, while the latter involves articulated object manipulation. ‘-’ indicates that the method can not handle this task due to its underlying principles.

proposed object-centric interaction primitives and directly employs an off-the-shelf 6D object pose tracking algorithm to continuously update the poses of both the active object Pactivet and the passive object Ppassivet in real-time, as required in Equation 4. This real-time feedback allows for dynamic adjustments to the target pose of the end-effector, enabling robust and accurate closed-loop execution.

### 4. Experiment

In this section, we aim to answer the following questions: (1) To what extent does OmniManip perform effectively in open-vocabulary manipulation tasks across diverse realworld scenarios (Section 4.2)? (2) What role do the system’s critical features play in enhancing its overall performance (Section 4.3)? (3) How promising is OmniManip for automating the collection of robot manipulation trajectories to enable scalable imitation learning (Section 4.4)?

##### 4.1. Experimental Setup

Hardware Configuration. Our experimental platform is built around a Franka Emika Panda robotic arm, with its parallel gripper’s fingers replaced by UMI fingers[6]. For perception, we employ two Intel RealSense D415 depth cameras. One camera is mounted at the gripper to provide a first-person view of the manipulation area, while the second camera is positioned opposite the robot to offer a thirdperson view of the workspace.

Tasks and Metrics. As shown in Figure 1, We designed 12 tasks to evaluate models’ manipulation capabilities in realworld scenarios. Six of these involve rigid object manipulation (e.g., pour tea), while the others focus on articulated manipulation (e.g., open the drawer). These tasks cover a

diverse set of objects and are intended to assess the models’ ability to generalize and adapt in complex environments. For each task, 10 trials were performed for each approach, and the success rate was recorded. After each trial, the object layout was reconfigured to ensure robust evaluation.

Baselines. We compare our approach with three baselines: 1) VoxPoser[14], which uses LLM and VLM to generate 3D value maps for synthesizing robot trajectories, excelling in zero-shot learning and closed-loop control; 2) CoPa[13], which introduces spatial constraints of object parts and combines with VLM to enable open-vocabulary manipulation; and 3) ReKep[15], which employs relational keypoint constraints and hierarchical optimization for realtime action generation from natural language instructions.

Implement Details We use GPT-4O from OpenAI API as the vision-language model, leveraging a small set of interaction examples as prompts to guide the model’s reasoning for manipulation tasks. The specific prompts used are detailed in the appendix. We employ off-the-shelf models [10, 43] for 6-DOF universal grasping and utilize GenPose++[56] for universal 6D pose estimation.

##### 4.2. Open-Vocabulary Manipulation

We conducted a comprehensive evaluation of OmniManip on 12 open-vocabulary manipulation tasks, ranging from straightforward actions such as pick-and-place to more complex tasks involving object-object interactions with directional constraints and articulated object manipulation. As shown in Table 1, our method exhibits robust zero-shot generalization and superior performance across the board without task-specific training. This generalization capability can be attributed to the commonsense knowledge em-

Method 0◦ 25◦ 45◦ 75◦ 90◦ ReKep 0/10 1/10 3/10 5/10 7/10 OmniManip 7/10 8/10 8/10 7/10 7/10

- Table 2. Quantitative analysis of the impact of viewpoints on the performance, using ‘Recycle the battery’ as a case study. bedded in VLM, while the proposed efficient object-centric interaction primitives facilitate precise 3D perception and execution. Additionally, we provide qualitative results in the appendix. OmniManip exhibits a substantial performance advantage over baseline methods, primarily due to two key factors: 1) the efficiency and stability of the proposed object-centric canonical interaction primitives, as further validated through extensive experiments in Section 4.3, and 2) the advanced dual closed-loop system for planning and execution. By incorporating a novel self-correction mechanism based on RRC, the system effectively mitigates hallucination issues of large models. As shown in Table 1, this closed-loop planning yields over a 15% improvement in performance for both rigid and articulated object manipulation tasks. A detailed qualitative analysis of the closed-loop reasoning and execution is provided in Section 4.3. 4.3. Core Attributes of OmniManip

Reliability of OmniManip. To effectively bridge VLM with low-level manipulation, reliable interaction primitives are crucial. We evaluate this across two key dimensions: stability and viewpoint consistency. Stability indicates the reliable extraction of task-relevant interaction primitives. As shown in Figure 5, ReKep extracts keypoint proposals through semantic clustering but lacks sensitivity to spatial geometry and task, making it challenging to generate sufficient task-relevant keypoints. CoPa extracts parts via explicit pixel segmentation, exhibiting high sensitivity to image texture and part shape. In contrast, OmniManip, an object-centric interaction primitive, samples interaction points in a canonical space aligned with the object’s functionality, ensuring both robustness and task-specific precision. Consistency of primitive extraction across varying viewpoints is critical to ensuring the stability of manipulation. Both ReKep and CoPa exhibit difficulties in this regard due to their reliance on sampling points directly from the object’s surface. Taking ReKep as an example, Figure 6 illustrates the planning results of ReKep and OmniManip for the ‘Recycle battery’ task across different viewpoints. As shown, ReKep successfully identifies interaction points from a 90◦ top-down view but fails under a 0◦ frontal view, where the ideal target point is floating in the air. In contrast, OmniManip utilizes an object-centric primitive representation in a canonical space, ensuring viewpoint invariance. Table 2 presents the quantitative comparison, demonstrating that OmniManip’s performance is nearly invariant across varying viewpoints, whereas ReKep’s performance

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

ExecutionPlanning

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Ours ReKep CoPa

- Figure 5. Stability analysis of interaction primitives. Visualization of planning and corresponding execution results across different methods, demonstrated using the ‘Pour tea’ as a case study.

OmniManip ReKep

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

90°

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

0°

45°

- Figure 6. Qualitative analysis of the impact of viewpoints on the performance, using ‘Recycle the battery’ as a case study. is significantly affected by changes in viewpoint.

Sampling Recycle Battery Pour Tea

Method Suc. Rate Iter. Suc. Rate Iter. Uniform 50% 1.8 30% 3.4

###### OmniManip 80% 1.7 70% 1.8

Table 3. Quantitative analysis of the primitive sampling efficiency.

Efficiency of OmniManip. Interaction direction proposals in OmniManip are driven by a targeted sampling strategy. Compared with uniform sampling in SO(3), OmniManip samples along the principal axes of the object’s canonical space. Since the canonical space is aligned with the object’s functionality, this ensures both efficient and effective sampling. To evaluate this efficiency, we compared OmniManip’s sampling strategy with uniform sampling in SO(3) using two key metrics: the number of iterations and the corresponding task success rate. As shown in Table 3 OmniManip not only requires fewer iterations but also achieves superior task performance, demonstrating that aligning the sampling process with the object’s functionality reduces sampling overhead while improving overall performance.

Closed-Loop Planning. In current methods, the planning component of VLM operates in an open-loop man-

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

Insert the pen in holder.

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

###### Self correction

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Execution

[Figure 178]

Initial Scene Final Scene

Figure 7. Closed-planning. Self-correction mechanism via RRC.

Figure 8. Two typical failure cases without closed-loop execution.

ner, meaning it cannot verify the correctness of the plan before execution. While ReKep achieves closed-loop control through point tracking, this only functions at the execution stage and does not provide feedback on the planning results generated by the VLM. In contrast, OmniManip introduces a unique self-correction mechanism via RRC, achieving closed-loop planning, which significantly reduces planning failures caused by VLM hallucinations, thereby offering more reliable planning. We report the results with closed-loop planning disabled in Table 1, where the task success rate decreases by over 15% in both rigid and articulated object manipulation tasks, demonstrating the effectiveness of the closed-loop planning approach. In Figure 7, we qualitatively illustrate the closed-loop planning results using the ”Insert the pen in a holder” task as an example. It is evident that OmniManip can effectively pre-render the planning outcomes and achieve self-correction through the RRC process, thereby enabling closed-loop planning.

object-centric pose tracking, enabling continued tracking of canonical space interaction primitives based on the object pose, even when the primitives are no longer visible.

##### 4.4. OmniManip for Demonstration Generation

We employed OmniManip to generate automatic demonstration data. Unlike prior methods reliant on task-specific privileged information, OmniManip collects demonstration trajectories for new tasks in a zero-shot manner, without needing task-specific details or prior object knowledge. To validate the effectiveness of OmniManip-generated data, we collected 150 trajectories per task to train behavior cloning policies [5]. These policies achieved high success rates, as shown in Table 4. Additional tasks and detailed results are provided in the appendix.

### 5. Conclusion

Task Success Rate

In this work, we presented a novel object-centric intermediate representation that effectively bridges the gap between VLM and the precise spatial reasoning required for robotic manipulation. We structured interaction primitives in object canonical space to translate high-level semantic reasoning into actionable 3D spatial constraints. The proposed dual closed-loop system ensures robust decision-making and execution, all without VLM fine-tuning. Our approach demonstrates strong zero-shot generalization across a variety of manipulation tasks, highlighting its potential for automating robotic data generation and improving the efficiency of robotic systems in unstructured environments. This work provides a promising foundation for future research into scalable, open-vocabulary robotic manipulation. Limitations. While advantageous, OmniManip also has limitations. It cannot model deformable objects due to pose representation. Its effectiveness also hinges on the mesh quality of 3D AIGC, which remains challenging despite progress. Additionally, multiple VLM calls present computational challenges, even with parallel processing.

Pick up the cup on the dish 95.24% Recycle the battery 91.30% Insert the pen in holder 86.36%

Table 4. Behavior cloning with demonstrations from OmniManip.

Closed-Loop Execution. Even with perfect planning, open-loop execution can still lead to task failure. Figure 8 illustrates two typical examples where planning succeeds, but open-loop execution causes failure. In the left image of Figure 8, the relative pose between the gripper and the object changes during the interaction, while the right image of Figure 8 shows a scenario where the target pose is dynamic, such as when the object moves during the task. To address these challenges, OmniManip employs pose tracking to enable real-time closed-loop execution. Recent work, ReKep, uses point tracking for closed-loop control but suffers from occlusions, leading to a 47% failure rate [15]. In contrast, OmniManip demonstrates greater robustness to occlusions caused by object movement. This is a benefit of

### Acknowledgments

We would like to thank Mingdong Wu and Tianhao Wu from PKU for their fruitful discussions, and Baifeng Xie from AgiBot for valuable technical support.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 2, 3

- [2] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2
- [3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 2
- [4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 1, 2

- [5] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 8
- [6] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024. 6
- [7] Qiyu Dai, Jiyao Zhang, Qiwei Li, Tianhao Wu, Hao Dong, Ziyuan Liu, Ping Tan, and He Wang. Domain randomizationenhanced depth simulation and restoration for perceiving and grasping specular and transparent objects. In European Conference on Computer Vision, pages 374–391. Springer, 2022. 2
- [8] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 2
- [9] Jiafei Duan, Wentao Yuan, Wilbert Pumacay, Yi Ru Wang, Kiana Ehsani, Dieter Fox, and Ranjay Krishna. Manipulateanything: Automating real-world robots using visionlanguage models. arXiv preprint arXiv:2406.18915, 2024. 2
- [10] Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, and Cewu

- Lu. Anygrasp: Robust and efficient grasp perception in spatial and temporal domains. IEEE Transactions on Robotics, 2023. 6
- [11] Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, page 02783649241281508, 2023. 2
- [12] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494,

2023. 2

- [13] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. arXiv preprint arXiv:2403.08248, 2024. 2, 6
- [14] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023. 1, 6
- [15] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024. 2, 3, 6, 8
- [16] Leslie Pack Kaelbling and Tom´as Lozano-P´erez. Hierarchical task and motion planning in the now. In 2011 IEEE International Conference on Robotics and Automation, pages 1470–1477. IEEE, 2011. 3
- [17] Leslie Pack Kaelbling and Tom´as Lozano-P´erez. Integrated task and motion planning in belief space. The International Journal of Robotics Research, 32(9-10):1194–1227, 2013. 3
- [18] Kento Kawaharazuka, Tatsuya Matsushima, Andrew Gambardella, Jiaxian Guo, Chris Paxton, and Andy Zeng. Realworld robot applications of foundation models: A review. Advanced Robotics, pages 1–23, 2024. 2
- [19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2
- [20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 3
- [21] Olivia Y Lee, Annie Xie, Kuan Fang, Karl Pertsch, and Chelsea Finn. Affordance-guided reinforcement learning via visual prompting. arXiv preprint arXiv:2407.10341, 2024. 2
- [22] Xuanyu Lei, Zonghan Yang, Xinrui Chen, Peng Li, and Yang Liu. Scaffolding coordinates to promote vision-language coordination in large multi-modal models. arXiv preprint arXiv:2402.12058, 2024. 4
- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In In-

ternational conference on machine learning, pages 19730–

19742. PMLR, 2023. 2

- [24] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378,

2023. 2

- [25] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023. 1
- [26] Chang Liu, Kejian Shi, Kaichen Zhou, Haoxiao Wang, Jiyao Zhang, and Hao Dong. Rgbgrasp: Image-based object grasping by capturing multiple views during robot arm movement with neural radiance fields. IEEE Robotics and Automation Letters, 2024. 1
- [27] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-vocabulary robotic manipulation through mark-based visual prompting. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024. 2, 3
- [28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 1, 2
- [29] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10072– 10083, 2024. 2, 4
- [30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3
- [31] Zeyi Liu, Arpit Bahety, and Shuran Song. Reflect: Summarizing robot experiences for failure explanation and correction. arXiv preprint arXiv:2306.15724, 2023. 1
- [32] Lucas Manuelli, Wei Gao, Peter Florence, and Russ Tedrake. kpam: Keypoint affordances for category-level robotic manipulation. In The International Symposium of Robotics Research, pages 132–157. Springer, 2019. 2
- [33] Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess, Ayzaan Wahid, Zhuo Xu, et al. Pivot: Iterative visual prompting elicits actionable knowledge for vlms. arXiv preprint arXiv:2402.07872, 2024. 2, 3
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1
- [35] Tanner Schmidt, Richard Newcombe, and Dieter Fox. Selfsupervised visual descriptor learning for dense correspon-

- dence. IEEE Robotics and Automation Letters, 2(2):420– 427, 2016. 2
- [36] Anthony Simeonov, Yilun Du, Andrea Tagliasacchi, Joshua B Tenenbaum, Alberto Rodriguez, Pulkit Agrawal, and Vincent Sitzmann. Neural descriptor fields: Se (3)equivariant object representations for manipulation. In 2022 International Conference on Robotics and Automation (ICRA), pages 6394–6400. IEEE, 2022. 2
- [37] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 11523–11530. IEEE, 2023. 1
- [38] Ioan A Sucan, Mark Moll, and Lydia E Kavraki. The open motion planning library. IEEE Robotics & Automation Magazine, 19(4):72–82, 2012. 2
- [39] Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa, Fabio Ramos, et al. Curobo: Parallelized collision-free robot motion generation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8112–8119. IEEE,

2023. 2

- [40] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151, 2024. 2, 4
- [41] Marc Toussaint, Jason Harris, Jung-Su Ha, Danny Driess, and Wolfgang H¨onig. Sequence-of-constraints mpc: Reactive timing-optimal control of sequential manipulation. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13753–13760. IEEE,

2022. 2

- [42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [43] Chenxi Wang, Hao-Shu Fang, Minghao Gou, Hongjie Fang, Jin Gao, and Cewu Lu. Graspness discovery in clutters for fast and accurate grasp detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15964–15973, 2021. 6
- [44] Bowen Wen, Wenzhao Lian, Kostas Bekris, and Stefan Schaal. You only demonstrate once: Category-level manipulation from single visual demonstration. arXiv preprint arXiv:2201.12716, 2022. 3
- [45] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868– 17879, 2024. 3
- [46] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. arXiv preprint arXiv:2401.00025, 2023. 2

- [47] Tianhao Wu, Jinzhou Li, Jiyao Zhang, Mingdong Wu, and Hao Dong. Canonical representation and force-based pretraining of 3d tactile for dexterous visuo-tactile policy learning. arXiv preprint arXiv:2409.17549, 2024. 1
- [48] Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, and Hao Dong. Learning score-based grasping primitive for human-assisting dexterous grasping. Advances in Neural Information Processing Systems, 36, 2024. 1
- [49] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 3
- [50] Senqiao Yang, Jiaming Liu, Ray Zhang, Mingjie Pan, Zoey Guo, Xiaoqi Li, Zehui Chen, Peng Gao, Yandong Guo, and Shanghang Zhang. Lidar-llm: Exploring the potential of large language models for 3d lidar understanding. arXiv preprint arXiv:2312.14074, 2023. 2
- [51] Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023. 2
- [52] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024. 2
- [53] Yiming Zeng, Mingdong Wu, Long Yang, Jiyao Zhang, Hao Ding, Hui Cheng, and Hao Dong. Lvdiffusor: Distilling functional rearrangement priors from large models into diffusor. IEEE Robotics and Automation Letters, 2024. 1
- [54] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 1
- [55] Jiyao Zhang, Mingdong Wu, and Hao Dong. Generative category-level object pose estimation via diffusion models. Advances in Neural Information Processing Systems, 36,

2024. 2

- [56] Jiyao Zhang, Weiyao Huang, Bo Peng, Mingdong Wu, Fei Hu, Zijian Chen, Bo Zhao, and Hao Dong. Omni6dpose: A benchmark and model for universal 6d object pose estimation and tracking. In European Conference on Computer Vision, pages 199–216. Springer, 2025. 2, 4, 6
- [57] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10324–10335, 2024. 4

