# arXiv:2412.04455v3[cs.RO]21Mar2025

## Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection

Enshen Zhou1*‡, Qi Su2,3,4*, Cheng Chi3*†, Zhizheng Zhang4, Zhongyuan Wang3, Tiejun Huang2,3, Lu Sheng1†, He Wang2,3,4†

1School of Software, Beihang University 2School of Computer Science, Peking University 3Beijing Academy of Artificial Intelligence 4Galbot

zhouenshen@buaa.edu.cn chicheng15@mails.ucas.ac.cn lsheng@buaa.edu.cn hewang@pku.edu.cn

[Figure 1]

Task: Move the pan with the lobster to the stove, and be careful not to let the lobster drop out.

[Figure 2]

[Figure 3]

[Figure 4]

Lobster

Stove

Pan

jump out

(a) Reactive Failure Detection (b) Proactive Failure Detection

(c) Real-world Test

Figure 1. For the task “Move the pan with lobster to the stove without losing the lobster”, (a) reactive failure detection identifies failures after they occur, and (b) proactive failure detection prevents foreseeable failures. In (a), at tR4 , the robot detects the failure after the lobster unpredictably jumps out due to the heat. In (b), pan tilting is detected at tP3 and corrected it at tP3 ′, requiring real-time precision. We formulate both tasks as spatio-temporal constraint satisfaction problems, leveraging our proposed constraint elements for precise, real-time checking. For example, in (a), a large relative distance between pan and lobster indicates failure; in (b), a large angle between the pan and the horizontal plane needs correction. (c) shows that our method combined with an open-loop policy forms a closed-loop system, enabling proactive (e.g., detecting moving glass during grasping) and reactive (e.g., removing toy after grasping) failure detection in cluttered scenes.

###### Abstract

Automatic detection and prevention of open-set failures are crucial in closed-loop robotic systems. Recent studies often struggle to simultaneously identify unexpected failures reactively after they occur and prevent foreseeable ones proactively. To this end, we propose Code-asMonitor (CaM), a novel paradigm leveraging the visionlanguage model (VLM) for both open-set reactive and proactive failure detection. The core of our method is to formulate both tasks as a unified set of spatio-temporal constraint satisfaction problems and use VLM-generated code to evaluate them for real-time monitoring. To enhance the accuracy and efficiency of monitoring, we fur-

∗ Equal contribution † Corresponding author ‡ Work done during internship at Galbot

ther introduce constraint elements that abstract constraintrelated entities or their parts into compact geometric elements. This approach offers greater generality, simplifies tracking, and facilitates constraint-aware visual programming by leveraging these elements as visual prompts. Experiments show that CaM achieves a 28.7% higher success rate and reduces execution time by 31.8% under severe disturbances compared to baselines across three simulators and a real-world setting. Moreover, CaM can be integrated with open-loop control policies to form closedloop systems, enabling long-horizon tasks in cluttered scenes with dynamic environments. See the project page at https://zhoues.github.io/Code-as-Monitor.

###### 1. Introduction

As expectations grow for robots to handle long-horizon tasks within intricate environments, failures are unavoidable. Therefore, automatically detecting and preventing those failures plays a vital role in ensuring the tasks can eventually be solved, especially for closed-loop robotic systems. There are two modes of failure detection [29], reactive and proactive. As depicted in Fig. 1, reactive failure detection aims to identify failures after they occur (e.g., recognizing that lobster lands on the table, indicating a delivery failure). In contrast, proactive failure detection aims to prevent foreseeable failures (e.g., recognizing that a tilted pan could cause the lobster to fall out, leading to a delivery failure). Both detection modes are even more challenging in open-set scenarios, where the failures are not predefined.

With the help of large language models (LLMs) [14, 60] and vision-language models (VLMs) [1, 38], recent studies can achieve open-set reactive failure detection [9, 12, 13, 16, 20, 39, 58, 59, 69, 75] as a special case of visual question answering (VQA) tasks. However, these methods often bear compromised execution speeds and coarse-grained detection accuracy, due to the high computational costs and inadequate 3D spatio-temporal perception capability of recent LLMs/VLMs. Moreover, open-set proactive failure detection, which has been less explored in the literature, presents even severer challenges as it is required to foresee potential causes of failure and monitor them in real-time with high precision to anticipate and prevent imminent failure. Simply adapting LLMs/VLMs cannot meet such expectations.

In this work, we aim to develop an open-set failure detection framework that achieves reactive and proactive detection simultaneously, not only benefiting from the generalization power of VLMs but also enjoying high precision in monitoring failure characteristics with real-time efficiency. We address this by formulating both tasks as a unified set of spatio-temporal constraint satisfaction problems, which can be precisely translated by VLMs into executable programs. Such visual programs can efficiently verify whether entities (e.g., robots, objects) or their parts in the captured environment maintain or achieve required states during or after execution (i.e., satisfying constraints), so as to immediately prevent or detect failures. To the best of our knowledge, this is the first attempt to integrate both detection modes within a single framework. We name this constraint-aware visual programming framework as Code-as-Monitor (CaM).

To be specific, the proposed spatio-temporal constraint satisfaction scheme abstracts the constraint-related entity or part segments from the observed images into compact geometric elements (e.g., points, lines, and surfaces), as shown in Fig. 1. It simplifies the monitoring of constraint satisfaction by tracking and evaluating the spatio-temporal combinational dynamics of these elements, eliminating the most irrelevant geometric and visual details of the raw enti-

ties/parts. The constraint element detection and tracking are grounded by our trained constraint-aware segmentation and off-the-shelf tracking models, ensuring speed, accuracy, and certain open-set adaptation capabilities. The evaluation protocol is in the form of VLM-generated code, i.e., monitor code, which is visually prompted by the starting frames of a sub-goal and its associated constraint elements, in addition to the textual constraints that must be fulfilled. Once generated, this monitor code could detect reactive or proactive failures just by being executed according to the tracked constraint elements, without needing to call the VLMs again. Therefore, this minimalist scheme is generalizable to openset failure detection for unseen entities and scenes (enabled by the potential diversity of the structured associations of constraint elements) as well as common skills (powered by the rich prior knowledge offered by VLMs), maintaining sufficient detection accuracy and real-time execution speed.

We conduct extensive experiments in three simulators (i.e., CLIPort [57] , Omnigibson [31], and RLBench [22]) and one real-world setting, spanning diverse manipulation tasks (e.g., pick & place, articulated objects, tool-use), robot platforms (e.g., UR5, Fetch, Franka) and end-effectors (e.g., suction cup, gripper, dexterous hand). The results show that CaM is generalizable and achieves both reactive and proactive failure detection in real-time, resulting in 28.7% higher success rates and reduced execution time by 31.8% under severe disturbances compared to the baselines. Moreover, in Sec. 4.3, CaM can be integrated with the existing open-loop control policy to form a closedloop system, enabling long-horizon tasks in cluttered scenes with environment dynamics and human disturbances. Our contributions are summarized as follows:

- • We introduce Code-as-Monitor (CaM), a novel paradigm that leverages VLMs for both reactive and proactive failure detection via constraint-aware visual programming.
- • We propose the constraint elements to enhance the accuracy and efficiency of constraint satisfaction monitoring.
- • Extensive experiments show that CaM is generalizable and achieves more real-time and precise failure detection than baselines across simulators and real-world settings.

###### 2. Related work

Robotic Failure Detection. Recent advances in LLMs [4, 14, 60, 61] and VLMs [1, 7, 23, 38, 46–50, 67, 77] greatly improve open-set reactive failure detection for robotic. Current LLM-based methods either convert visual inputs into text [40, 56, 66], potentially losing visual details, or rely on ground-truth feedback [20, 51, 58, 59], which is impractical in real-world scenarios. Recent studies employ VLMs as failure detectors, offering binary success indicators [12, 39, 69, 75] or textual explanations [9, 13] through visual question answering (VQA), such as DoReMi [16]. However, they often bear compromised execution speeds

|Next Subgoal ( )<br><br>Move the pan above the stove.<br><br>|
|---|

|Move the pan with the bread to the stove, and be careful not to let the bread fall out.<br><br>Task Instruction ( ):<br><br>|
|---|

|def constraint_monitor(end_effector, element_position, is_finished): points_yellow, point_green, surface_red, point_blue = element_position if not is_finished:<br><br>v1, v2 = surface_red[-1][1] - surface_red[-1][0], surface_red[-1][2] - surface_red[-1][0] normal_vector = np.cross(v1, v2) normal_vector = normal_vector / np.linalg.norm(normal_vector) angle = np.degrees(np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))) if angle > tol_angle:<br><br>return False, f"The pan surface is not horizontal, as the angle between the pan surface and the horizontal plane is {angle} degrees."<br><br>… return True, “No Failure Detected."<br><br>Monitor Code<br><br>|
|---|

- Step1
- Step2

Step3

[Figure 5]

|During execution:<br><br>1. The pan handle must be grasped.<br>2. The bread must remain in the pan.<br>3. The pan should remain horizontal. Upon completion:<br>4. The pan is directly above the stove.<br><br><br>Textual Constraints ( )<br><br>|
|---|

Constraint Generator

|Grasp the pan handle.<br><br>Previous Subgoal ( ):<br><br>|
|---|

Failure Feedback ( ):

No Failure Detected.

|[Figure 6]<br><br>[Figure 7]<br><br>|
|---|

|[Figure 8]<br><br>1<br><br>1<br><br>2 4 4<br><br>2<br><br>3<br><br>3<br><br><br><br><br>3<br><br>[Figure 9]<br><br>4<br><br>4<br><br><br>1<br><br>1<br><br>2<br><br>2<br><br>3<br><br>3<br><br><br><br><br>3<br><br>|
|---|

Computing

Computing

Computing

[Figure 10]

[Figure 11]

[Figure 12]

- Step1: Constraint Generation
- Step2: Element Painting
- Step3: Code Generation & Monitor

Tracking

Tracking Tracking

Constraint Monitor

Constraint

Painter

Off-the-shelf VLM (GPT-4o)

Off-the-shelf Tracker (Co-Tracker)

[Figure 13]

Failure Feedback ( )

Detect failure! The pan surface is not horizontal, as the angle between the pan surface and the horizontal plane is 25 degrees. Please use this feedback to re-plan.

Multi-view RGB-D Observations ( ) Multi-view RGB Observations with all constraint elements ( )

- Figure 2. Overview of Code-as-Monitor. Given task instructions and prior information, the Constraint Generator derives the next subgoal and corresponding textual constraints based on multi-view observations. The Painter maps these constraints onto images as constraint elements. The Monitor generates monitor code from these images and tracks them for real-time monitoring. If any constraint is violated, it outputs the reason for failure and triggers re-planning. This framework unifies reactive and proactive failure detection via constraints, more generally abstracts relevant entities/parts through constraint elements, and ensures precise and real-time monitoring via code evaluation.

and coarse-grained detection accuracy. Meanwhile, openset proactive failure detection is rarely explored, as previous methods [2, 10] are confined to predefined failures. This detection mode requires foreseeing potential failure causes and monitoring them in real-time with high precision. In this work, we formulate both reactive and proactive failure detection as constraint satisfaction problems and use VLMgenerated code to evaluate them, meeting the expectations of both modes above.

Visual Prompting. Visual prompts enhance the visual reasoning abilities of VLMs and are categorized into maskbased, point-based, and element-based methods. Maskbased approaches like SoM [65] apply numbered segmentation masks on images without considering instructions, whereas instruction-guided methods like IVM [74] and API [68] generate masks to block irrelevant regions. Pointbased prompting encodes functionalities via points, including affordances [28, 37, 70], motion [37, 44], and constraints like ReKep [21]. However, ReKep [21] extracts semantic keypoints using DINOv2 [45] and struggles to capture precise keypoints that accurately represent desired constraints. Element-based methods like CoPa [19], represent robotic control using basic elements (e.g., points, lines) via pre-trained models like SAM [26] and GPT-4V [1] simply. In contrast, this work explores constraint satisfaction monitoring and introduces constraint elements that more precisely represent relevant entities/parts. These elements are tracked and evaluated in real-time to simplify monitoring.

Visual Programming. Visual programming requires strong visual concept understanding and reasoning. Prior works show strong generalization (e.g., zero-shot) across various tasks, such as image editing [8, 17], 3D visual grounding [71], and robotics control [35, 62], by integrating LLMs with visual modules but lose fine-grained details. Recent studies [43, 63] explore visual programming using VLMs, but these methods take raw RGB images as inputs with predefined primitive libraries. In contrast, this work leverages constraint elements for visual programming and uses arithmetic operations in the code to encode their spatio-temporal combinational dynamics, presenting greater challenges.

###### 3. Method

We first give an overview of the proposed Code-as-Monitor (CaM) (Sec. 3.1). Then, we elaborate on the constraint element in Constraint Painter, especially constraint-aware segmentation (Sec. 3.2). Finally, we present Constraint Monitor for real-time detection (Sec. 3.3).

###### 3.1. Overview

The proposed CaM comprises three key modules: the Constraint Generator, Painter, and Monitor. We focus on longhorizon manipulation task instructions Lglobal (e.g., “Move the pan with the bread to the stove, and be careful not to let the bread fall out”), using RGB-D observations O from two camera views (front and top). As shown in Fig. 2, the RGB images O, along with instructions Lglobal, previous

Textual Constraint ( )

Project to

Annotate to

[Figure 14]

Voxelization Clustering Connecting

###### The pan should remain horizontal during

3D Space

multi views

transfer. Please segment the desired part.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

ConSeg

[Figure 22]

[Figure 23]

Multi-view RGB

Constraint-aware

Voxelized point cloud Clustered 3D Points

Constraint-aware point cloud Constraint element ( ) Multi-view RBG

observations ( )

part-level masks ( )

with one element ( )

- Figure 3. Constraint Element Pipeline. Given a constraint, our model ConSeg generates instance-level and part-level masks across multiple views, which are projected into 3D space. Through a series of heuristics, the desired elements are produced. Once all elements are obtained, they are annotated onto the original multi-view images. Here we display the annotation result of one element.

subgoal lpre, and failure feedback from the Constraint Monitor fpre (e.g., subgoal success or failure reason), are fed into the Constraint Generator FVLM (i.e., GPT-4o [1]) to generate the next subgoal lnext and associated textual constraints C. This process can be formulated as follows:

lnext, Cd, Cu = FVLM(O, Lglobal, lpre, fpre) (1)

where Cd = {c0d,c1d,...,cnd} denotes the constraints that must be maintained during subgoal execution (e.g., pan

handle must be grasped, bread must remain in the pan, pan should remain horizontal during transfer), and Cu = {c0u,c1u,...,cku} refers to the constraints that must be met upon subgoal completion (e.g., pan should be directly above the stove). We successfully unify reactive and proactive failure detection as these task-specific, situation-aware constraint satisfaction problems.

In Painter, for each textual constraint c from Cd or Cu, we generate corresponding constraint elements e (detailed in Sec. 3.2) from observations O. These elements, which are composed of 3D points, abstract the constraint-related entities or their parts to represent the desired textual constraint more easily (e.g., the constant distance between green points on bread and pan determines if bread remains in pan, as shown in Fig. 2.) These generated elements are then aggregated into the final set E = {e0,e1,...,en+k}, and numerically annotated across all views to produce the final visual prompted images OE.

In Monitor, we provide GPT-4o [1] with the next subgoal lnext, textual constraints C, and annotated observations OE for constraint-aware visual programming to generate the evaluation protocol, i.e., monitor code. This code inputs the elements’ 3D positions, calculates arithmetic operations within it, and returns a boolean to indicate potential or actual failure and a string to describe its reason. During subgoal execution, Monitor tracks the elements and evaluates the spatio-temporal combinational dynamics of these elements. If the code returns False, the policy execution halts immediately, and the accompanying string is used as feedback fpre for re-planning. Otherwise, the subgoal is considered completed. In either case, the cycle is repeated.

###### 3.2. Constraint Element

To simplify the monitoring of constraint satisfaction, we introduce constraint elements by abstracting constraintrelated entities/parts into compact geometric elements (e.g., points, lines) to get rid of the most irrelevant geometric and visual details, making them easier to track and generalize.

Pipeline. The constraint element generation pipeline is shown in Fig. 3. Our trained multi-granularity constraintaware model ConSeg performs two inference steps for each c and each RGB image o from the set of views O. First, instance-level masks Mi are generated to capture constraint-related entities. Then, fine-grained part-level masks Mp and corresponding element type descriptions le are produced, as shown in Fig. 4. Using corresponding depth data, we project Mp from all views into 3D space, fusing them into a point cloud. However, directly tracking and evaluating the spatio-temporal combinational dynamics of these constraint-related entities/parts is challenging. Therefore, we convert these entities into proposed constraint elements. We first apply voxelization to the point cloud with voxel size determined by element type le (e.g., the surface needs at least 3 points, we divide the occupied space into 2 × 2 voxels.). Then we cluster one representative point within each voxel and filter them to a specified number, also determined by le to extract the final desired 3D points. We connect points within each instance-level mask Mi to form the constraint element e associated with constraint c. Notably, points modeled as end-effectors (e.g., the points of the fingertips and hand’s center represent a dexterous hand) can be directly obtained from forward kinematics, bypassing the process above. Additionally, we perform parallel inference of ConSeg across all views to expedite the acquisition of the final set E and their annotations onto the corresponding views OE. Moreover, this minimalist approach, i.e., constraint elements, emphasizes the most relevant entities/parts, enabling generalization to unseen scenes and entities, which is critical for open-set failure detection. For more details, please check Supp. B.3.

###### Constraint-aware Segmentation. Since the textual con-

[Figure 24]

[Figure 25]

[Figure 26]

Trainable Frozen

Textual Constraint ( ):

The pan should remain horizontal during transfer. Please segment the desired part.

[Figure 27]

[Figure 28]

Vision Encoder

LoRA VLM

###### Decoder

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Response ( ): It needs a surface <SEG>.

- Figure 4. ConSeg architecture. Here we display the part-level segmentation, which will output the desired element type and mask.

straint does not explicitly specify relevant entities/parts, we require a model capable of logical reasoning and certain open-set adaptation, enabling precise constraint-aware instance and part-level segmentation. As depicted in Fig. 4, we propose ConSeg, which builds upon LISA [27] comprising a VLM FVLM (i.e., LLaVA [38]) with the vision encoder Fenc and decoder Fdec from SAM [26]. The textual constraint c and the image o are input into the VLM FVLM to generate an embedding for the <SEG> token. The VLM outputs a textual description le of the desired element type additionally for part-level segmentation. The decoder Fdec then generates the segmentation mask M by leveraging visual features from the vision encoder Fenc based on the current observation o, and features derived from the final-layer embedding h<SEG> of the <SEG> token, transformed via an MLP (FMLP). The process can be formulated as follows:

le, h<SEG> = FVLM(c, o), M = Fdec(FMLP(h<SEG>), Fenc(o)) (2)

The entire pipeline, including additional LoRA [18] parameters, is optimized end-to-end while keeping the parameters of the VLM FVLM and vision encoder Fenc frozen.

Dataset and Collection Pipeline. Our dataset utilizes images from BridgeData V2 [64]. While BridgeData V2 provides trajectory-level instructions, we require frame-level annotations for per-frame subgoals and constraints. To address this, we sample pick-and-place data, using external references (e.g., gripper open/close states) to generate 10,181 trajectories with 219,356 images. We employ GPT4o to decompose trajectory instructions into subgoals, constraints, and object-part associations to generate groundtruth annotations. Instance-level and part-level segmentations are performed using Grounded SAM [53] and Semantic SAM [32], respectively. These outputs are integrated and refined through manual inspection to produce the multigranularity dataset, which is combined with LISA’s training data to fine-tune our model. More details are in Supp. B.

###### 3.3. Real-time Monitor Module

After annotating constraint elements with unique colors and numerical labels across all views, these images serve as vi-

sual prompts to generate monitor code via GPT-4o [1] with constraints C and the next subgoal lnext. The generated code accepts the elements’ 3D positions as input and performs arithmetic operations (e.g., NumPy-supported calculations) to evaluate spatio-temporal constraints at various stages (i.e., during execution or upon completion), as shown in Fig. 2 code block. It returns a boolean flag indicating potential or actual failure if the computed result exceeds a predefined tolerance (e.g., accepting deviations of the pan’s surface normal from the z-axis within 15◦), along with a string explaining the cause. By tracking the elements using CoTracker [24], we achieve real-time detection via VLMs without frequent calls, reducing the computational cost.

Notably, including both current positions and historical trajectories of the elements as input allows us to monitor tasks needing high precision, such as the requirement of 2cm movement or 180◦ rotation, which previous reactive detection methods could not address through direct VLM queries using raw RGB images. Moreover, the potential diversity in the structured associations of constraint elements within the code (e.g., calculating the angle between the normal vector of the pan’s surface element and the z-axis determines if the pan is level, as shown in Fig. 2 code block) and rich prior knowledge offered by VLMs enable generalization to certain unseen tasks requiring common skills.

###### 4. Experiments

Our experiments aim to address the following questions: (1) Can CaM achieve open-set reactive and proactive failure detection across diverse robots, end-effectors, and objects, both in simulator (Sec. 4.2) and on real robots (Sec. 4.3)? (2) Can ConSeg infer multi-granularity constraint-aware masks for unseen scenes and objects (Sec. 4.4)? (3) Which design choices greatly enhance the performance (Sec. 4.5)?

###### 4.1. Experimental Setup

Environment Settings. We evaluate CaM across three simulators (i.e., CLIPort [57], Omnigibson [31], and RLBench [22]) and a real-world setting. In CLIPort, we employ a UR5 arm equipped with a suction cup for pick-andplace and a spatula for pushing, controlled by pre-trained CLIPort policy [57]. Omnigibson features a Fetch robot with a gripper, controlled by ReKep [21]. RLBench uses a Franka arm with a gripper, controlled by ARP [73]. We use a UR5 arm with a Leap Hand [55] for real-world experiments, controlled by an open-loop policy named DexGraspNet 2.0 [72]. More details are provided in Supp. C.

ConSeg Configuration. Given the significant gap between simulation and real-world data, we fine-tune the ConSeg model with 100 trajectories collected from each simulation environment before deploying it. To share the same data collection and auto-label pipeline, this fine-tuning

dataset is also limited to pick-and-place tasks. We manually annotate trajectory-level instructions, and image frames are captured at a frequency of 1 Hz. Using only pick-andplace task data helps mitigate the data distribution gap while allowing us to validate the model’s generalization to unseen tasks, including tool-use, articulated objects, and longhorizon complex tasks. Notably, the ConSeg model used in real-world experiments is not fine-tuned, ensuring a rigorous evaluation of ConSeg’s generalization capability.

Task Settings. We introduce disturbances across all environments to induce failures, classified by the affected constraints (e.g., points, lines, surfaces). We evaluate task success rate, execution time, and additional token usage in Omnigibson. Notably, we don’t assess failure judgment success rate because our code operates continuously in real-time, rendering this metric inapplicable. Moreover, there is no standard for evaluating proactive failure detection accuracy. Detailed evaluation settings are provided in each section.

Baselines. We compare DoReMi (DRM) [16] as the baseline for all settings, which uses VLMs to detect intermediate failures during execution via frequent VQA-based queries. In CLIPort, we include Inner Monologue [20] baseline, which detects failures only upon each subgoal completion.

###### 4.2. Main Results in Simulator 4.2.1 Results in CLIPort

We evaluate two tasks in CLIPort: (1) Stack in Order: The robot must stack blocks in a specified order, including two point-level disturbances: (a) with per-step probability p, the suction cup may release a block, causing it to drop; (b) placement positions are perturbed by uniform noise in [0,q] cm, potentially leading to tower collapse. Success is defined as correctly stacking the blocks within 70s. (2) Sweep Half the Blocks: To address the pre-trained policy’s tendency to sweep all blocks, the robot must determine when to halt, sweeping half of the blocks (±10%) into a specified colored area within 30s. Success is achieved by meeting these criteria. Tab. 1 shows the mean results over 5 different seeds, each with 12 episodes. For more details, please check Supp. D.1. The following paragraphs present our analyses.

Code better monitors 3D space relations. As shown in the Tab. 1, Under the most severe disturbances (p=0.3, q=3) in “Stack in Order”, CaM achieves a 17.5% higher success rate than DoReMi. Frequent VLM queries lead to increased incorrect failure judgments due to a limited 3D spatial understanding from single images, compared to code-based evaluation of block positions. For example, after placing the green block on the red and preparing to pick up the blue, DoReMi mistakenly thinks that the green block is no longer atop the red, causing it to re-execute the previous subgoal.

Code with elements achieve reactive and proactive failure detetction. As shown in Tab. 1, under severe distur-

Table 1. Performance in CLIPort. We report the success rate and execution time, compared to CLIPort (CP) [57], with Inner Monologue (IM) [20] and DoReMi (DRM) [16].

Tasks with Success Rate(%) ↑ Execution Time(s) ↓ disturbance CP +IM +DRM +Ours +IM +DRM +Ours

- Stack in p=0.0 100.0 100.0 100.0 100.0 13.4 13.4 13.4

- order with p=0.15 56.67 81.67 83.33 95.00 34.8 26.00 21.0 drop p p=0.3 21.67 75.00 76.67 88.33 42.8 34.20 25.4

Stack in q=1 90.00 90.00 96.67 98.33 24.8 24.6 24.2

- order with q=2 41.67 71.67 75.00 83.33 39.4 37.0 29.2 noise q q=3 15.00 40.00 40.00 63.33 58.2 54.2 36.8 Sweep Half the Blocks 0.00 18.33 16.67 75.00 22.0 16.6 16.4

bances in “Stack in Order”, CaM reduces execution time by 38.7% and 14.4% compared to Inner Monologue [20], which only detects failures upon subgoal completion, and DoReMi [16], which suffers from misjudgments due to repeated VLM queries, respectively. In contrast, CaM unifies both reactive and proactive failure detection with high precision in real-time, especially in preventing foreseeable failures. For example, if the green block is placed on the red block with heavy placement position disturbance, CaM anticipates that further stacking the blue may lead to collapse and then stabilizes the green block before proceeding.

Code with elements leads to more accurate counting. As shown in the Tab. 1, in “Sweep Half the Blocks”, CaM achieves average success rates 4.5× higher than DoReMi [16]. Calculating the block points in the designated surface region provides more accurate results than directly using the VLM to count, enabling a more precise halting of the policy to complete the task.

###### 4.2.2 Results in Omnigibson

We conduct experiments on three tasks in Omnigibson, each involving a distinct type of constraint-element disturbance: (1) Slot Pen: Insert a pen into a holder, facing point-level disturbances wherein (a) pen is moved during grasping; (b) pen is dropped during transport; (c) holder is moved during insertion. (2) Stow Book: Place a book to a bookshelf vertically, with line-level disturbances where (a) book is randomly rotated during grasping; (b) end-effector joint is randomly actuated to alter the book pose; (c) book is reoriented horizontally after placement. (3) Pour Tea: Pours from a teapot into a teacup, encountering surface-level disturbances wherein (a) teapot is tilted forward/backward during movement; (b) end-effector joint is actuated to induce a lateral tilt of the teapot during movement; (c) teapot is returned to a horizontal position during pouring. Tab. 2 reports the results across three tasks, each including one no-disturbance trial and three specific-disturbance trials, with 10 runs for each setting. More details are provided in Supp. D.2.

Code with elements detects richer failures. In Tab. 2, only CaM can detect failures caused by surface-level disturbances in “Pour Tea” compared to DoReMi [16]. The reason is that changes in the teapot’s pitch and roll angles

- Table 2. Performance in Omnigibson. We report the success rate (SR), execution time, and token usage, compared to DoReMi (DRM) [16].

Method

Slot Pen (Point-level Disturbances) Stow Book (Line-level Disturbances) Pour Tea (Surface-level Disturbances)

SR with Disturbance(%) ↑ Time Token SR with Disturbance(%) ↑ Time Token SR with Disturbance(%) ↑ Time Token

None Dist.(a) Dist.(b) Dist.(c) (s)↓ (k)↓ None Dist.(a) Dist.(b) Dist.(c) (s)↓ (k)↓ None Dist.(a) Dist.(b) Dist.(c) (s)↓ (k)↓ ReKep [21] 30 20 10 10 - - 40 30 30 20 - - 20 20 20 10 - -

+DRM 40 10 20 20 177.84 54.54 50 40 20 40 127.17 38.67 0 0 0 0 - -

+Ours 60 50 40 40 101.85 25.82 70 60 70 60 93.08 18.67 50 40 40 30 174.55 44.19

[Figure 35]

| |
|---|

[Figure 36]

| |
|---|

[Figure 37]

| |
|---|

[Figure 38]

| |
|---|

[Figure 39]

| |
|---|

Reasoning Pick &Place Task: Grasp the animals according to their distances to the fruits, from nearest to farthest.

[Figure 40]

| |
|---|

[Figure 41]

| |
|---|

[Figure 42]

| |
|---|

Horse is moved, target changed Pear is moved, target changed

Figure 5. Example of Real-world Evaluation. The red bounding box shows the current grasp target, which may shift due to environmental changes. CaM monitors and adapts to these changes in real-time, resulting in a closed-loop system with an open-loop policy.

- Table 3. Performance of Single Pick & Place. We report the success rate and execution time. DGN donates DexGraspNet 2.0 [72].

Tasks with Object Success Rate(%) ↑ Execution Time(s) ↓ disturbance types DGN +DRM +Ours +DRM +Ours Pick & Place Deformable 0.00 83.33 96.67 61.8 46.3 with the Transparent 0.00 66.67 93.33 72.6 48.1 objects being Small Rigid 0.00 80.0 96.67 65.7 45.4 moved Large Geometric 0.00 86.67 96.67 68.9 45.3 Pick & Place Deformable 0.00 76.67 93.33 68.7 62.5 with the Transparent 0.00 60.00 90.00 77.7 62.7 objects being Small Rigid 0.00 63.33 93.33 69.8 60.5 removed Large Geometric 0.00 76.67 96.67 72.3 60.3

- Table 4. Performance of Reasoning Pick & Place in cluttered scene. We report the success rate. The robot is controlled by an open-loop policy named DexGraspNet 2.0 (DGN) [72]. w/ CE with ✓ indicates using constraint elements; otherwise, constraintaware entities or parts are used for tracking and code computation.

Table 5. Performance of reasoning and constraint-aware segmentation. FMC denotes the foundation model combination baseline.

ReasonSeg ConstraintSeg

Method

Instance-level Instance-level Part-level

gIoU cIoU gIoU cIoU gIoU cIoU OVSeg [34] 28.5 18.6 32.9 31.4 20.2 21.5 GRES [36] 22.4 19.9 28.6 26.4 22.7 22.6 X-Decoder [78] 22.6 17.9 27.8 28.0 23.5 25.1 SEEM [79] 25.5 21.2 29.4 28.0 23.1 24.8 PixelLM [54] 56.0 61.4 44.4 43.2 24.1 22.6 LISA-13B [27] 56.6 65.1 42.1 38.9 23.4 24.3 FMC 51.2 53.3 49.3 49.6 40.8 39.3 ConSeg-13B 55.7 63.9 62.1 68.7 60.2 65.3

###### 4.2.3 Results in RLBench

We further evaluate CaM on RLbench and demonstrate its superior generalization across diverse manipulation tasks, including articulated objects, rotational manipulation, and tool use. Experimental details can be found in Supp. D.3.

Success Rate(%) ↑ DGN +DRM +Ours Clear all objects on table except for animals

Tasks w/ CE

###### 4.3. Main Results in Real World

✓ 0.00 10.00 60.00 ✗ 0.00 10.00 20.00

Grasp the animals according to their

We conduct real-world evaluations on two tasks: (1) Simple Pick & Place: The robot has 70s to pick up objects and place them at specified locations, facing two disturbances: (a) moving the object during grasping, and (b) removing the object from hand during movement. We test four object types (e.g., deformable, transparent), selecting 3 examples per type and conducting 10 trials for each (see Tab. 3). (2) Reasoning Pick & Place: The robot executes long-horizon tasks, involving ambiguous terms (e.g., “fruit”, “animal”), under the same disturbances. We evaluate 2 long-horizon tasks in cluttered scenes, performing 10 trials each (see Tab. 4). For more details, please check Supp. D.4. The following paragraphs present our analyses.

✓ 0.00 0.00 90.00 distances to fruits, from nearest to farthest

are hard to detect by querying VLM via VQA using one current image. The misjudgment leads to a 0% success rate of DoReMi in this task. Notably, DoReMi’s success rate is sometimes lower than that of ReKep alone due to VLM’s limited spatio-temporal understanding of single images.

Code with elements detects failure with lower computational cost. As shown in Tab. 2, we achieve a 34.8% reduction in execution time and a 52.2% decrease in token count compared to DoReMi [16]. This improvement stems from proactive failure detection, which prevents more severe failures ahead in real-time for timely re-planning while generating code only once per subgoal. For example, in “Pour Tea”, CaM detects failure by monitoring the teapot’s tilt angle in real-time, avoiding frequent VLM checks.

Elements generally abstract constraints and relevant entities. As shown in Tab. 3, using a different end-effector (i.e., Leap Hand [55]) in Simple Pick & Place, CaM also achieves success rates surpassing DoReMi [16] by 20.4% when handling different kinds of objects (e.g., deformable).

||Task: Open the oven. Constraint: Align the end effector<br><br>with the oven handle.|
|---|
<br><br>|Task: Put the ball into the frying pan. Constraint: Align the end effector<br><br>with the handle on the pot lid.|
|---|
<br><br>|Task: Fold the cloth from right to left. Constraint: Align the end effector<br><br>with the right side of the cloth.|
|---|
<br><br>|Task: Move the spatula onto the cloth. Constraint: Theend effectormust hold the spatula handle.<br><br>|
|---|
<br><br>Task and Constraint Observation<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]|
|---|

|GroundTruth<br><br>Instance-level part-level<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]|
|---|

|Instance-level<br><br>LISA ConSeg (Ours)<br><br>part-level Instance-level part-level<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]|
|---|

Val Set

Unseen

object

Unseen

Scene

Unseen Task

Figure 6. Visual comparison between our ConSeg and LISA [27] at instance and part level. The red masks are the segmentation results.

Table 6. Ablation studies in Omnigibson’s “Stow Book”, assessing the impact of Multi-View (MV), Constraint-aware Segmentation (CS) for elements, and Connect Points (CP) for element formation.

|MV CS CP|Success Rate with Disturbance(%) ↑ None Dist.(a) Dist.(b) Dist.(c) Avg<br><br>|
|---|---|
|✗ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✗ ✓ ✓ ✓|40.0 40.0 30.0 50.0 40.0 50.0 40.0 40.0 40.0 42.5 60.0 50.0 60.0 50.0 55.0 70.0 60.0 70.0 60.0 65.0<br><br>|

We find that abstracting constraint-related entities/parts removes the irrelevant visual details, enabling generalization to different kinds of entities in unseen scenes (as discussed in Sec. 3.2), leading to easier tracking and code evaluation. Reactive and Proactive failure detection combined with an open-loop policy forms a close-loop system. Tab. 4 shows that only CaM successfully handles long-horizon tasks in cluttered scenes, while all baselines fail. These tasks are challenging as the robot is controlled by an openloop policy that cannot handle environment dynamics and human disturbances in a closed-loop manner. By incorporating both reactive and proactive failure detection with the open-loop policy, as shown in Fig. 5, the robot can dynamically adjust its target object in real-time. For example, when a human moves the horse or pear during the task, the robot adapts by grasping the animal closest to the fruit, effectively forming a closed-loop system.

###### 4.4. Main Results of Segmentation

Tab. 5 presents segmentation results comparing our ConSeg, with SOTA models and a Foundation Model Combination (FMC) baseline. FMC integrates GPT-4o for reasoning over tasks and constraints to identify relevant instances and parts, along with Grounded SAM [53] and Semantic SAM [32] for instance and part-level segmentation, respectively, similar to our data collection pipeline. We evaluate performance on the ReasonSeg [27] benchmark and our proposed Constraint-Aware Segmentation (ConstraintSeg)

Table 7. Ablation study on training data. SemanticSeg includes ADE20K [76], COCO-Stuff [5], PACO-LVIS [52] and PASCALPart [6]. ReferSeg includes refCLEF, refCOCO, refCOCO+ [25] and refCOCOg [41]. VQA indicates LLaVAInstruct-150k [38].

|SemanticSeg ReferSeg VQA ReasonSeg<br><br>ContraintSeg Ins Part<br><br>|Ins-level Part-level gIoU cIoU gIoU cIoU|
|---|---|
|✓ ✓ ✓ ✓ ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|42.1 38.9 23.4 24.3 60.7 65.9 40.4 45.6 51.5 50.6 56.5 61.7 62.1 68.7 60.2 65.3|

benchmark. Our benchmark evaluates performance using gIoU and cIoU, following ReasonSeg’s evaluation setting.

ConSeg performs both reasoning and multi-granularity constraint-aware segmentation. As shown in Tab. 5, ConSeg performs comparably to LISA and PixelLM on the ReasonSeg but significantly surpasses them on ConstraintSeg, achieving nearly a 40% improvement at the part level. Visual comparisons in Fig. 6 further demonstrate ConSeg’s strong generalization to unseen objects, scenes, and tasks.

###### 4.5. Ablation Study

We conduct ablation studies following the corresponding environment settings and present analyses below.

Multi views are critical for visual programming. As shown in Tab. 6, using only front-view images with constraint elements reduces the average success rate from 65% to 40%. This is primarily due to (1) occlusion leading to suboptimal element generation and (2) errors in visual programming, where dimensional reduction causes surfaces to be misinterpreted as lines or lines as points.

Constraint-aware segmentation enhances element. As shown in Tab. 6, using DINOv2 [45] to generate semantic points to form elements, rather than through constraintaware segmentation, reduces the success rate from 65% to 42.5%, because these constructed elements fail to represent the desired constraints accurately. For example, capturing a

book’s vertical orientation requires two precise points on its edge, which DINOv2 can not provide.

Forming elements improves code generation. As shown in Tab. 6, using unconnected 3D points as final elements reduces the success rate from 65% to 55%. Pre-formed elements contain more prior constraint information, serving as visual cues better encoded in code, whereas standalone points require recombination during visual programming.

Elements simplify constraints computation. As shown in Tab. 4, the success rate decreases from 60% to 20% when entities or parts are not transformed into proposed elements in real-world evaluation. We find two key issues: (1) inaccurate 3D position and slow pose tracking of entities or parts; and (2) difficulties in performing arithmetic operations on their 3D positions or poses in code, hindering the encoding of their spatio-temporal constraints.

Both instance-level and part-level data improve performance. We conduct ablation studies to assess the impact of our proposed dataset on the ConstraintSeg benchmark. The results are shown in Tab. 7. The results indicate that both the instance-level and part-level subsets contribute to performance improvements and enhance each other’s effects.

###### 5. Conclusion

In this paper, we present a novel paradigm termed Codeas-Monitor, leveraging the VLMs for both open-set reactive and proactive failure detection. In detail, We formulate both detection modes as spatio-temporal constraint satisfaction problems and use VLM-generated code to evaluate them for real-time monitoring. We further propose constraint elements, which abstract constraints-related entities or their parts into compact geometric elements, to improve the precision and efficiency of monitoring. Extensive experiments demonstrate the superiority of the proposed approach and highlight its potential to advance closed-loop robot systems.

###### Acknowledgement

This work was supported by the National Natural Science Foundation of China (62132001), the National Science and Technology Major Project (2022ZD0116314), and the Fundamental Research Funds for the Central Universities.

We sincerely thank Xingqiang Yu for his valuable discussions and insightful feedback about the Leap Hand and UR5 in real-world experiments. We also sincerely appreciate Jingyi Yang’s excellent figure design (e.g., teaser figure, pipeline overview) and demo editing work.

###### References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,

Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv, 2023. 2, 3, 4, 5, 10, 23

- [2] Aneseh Alvanpour, Sumit Kumar Das, Christopher Kevin Robinson, Olfa Nasraoui, and Dan Popa. Robot failure mode prediction with explainable machine learning. CASE, 2020. 3
- [3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv, 2022. 10, 21
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 2
- [5] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. CVPR, 2018. 8
- [6] Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can: Detecting and representing objects using holistic models and body parts. CVPR, 2014. 8
- [7] Zeren Chen, Zhelun Shi, Xiaoya Lu, Lehan He, Sucheng Qian, Zhenfei Yin, Wanli Ouyang, Jing Shao, Yu Qiao, Cewu Lu, et al. Rh20t-p: A primitive-level robotic dataset towards composable generalization agents. arXiv preprint arXiv:2403.19622, 2024. 2
- [8] Jaemin Cho, Abhay Zala, and Mohit Bansal. Visual programming for step-by-step text-to-image generation and evaluation. NeurIPS, 2024. 3
- [9] Yinpei Dai, Jayjun Lee, Nima Fazeli, and Joyce Chai. Racer: Rich language-guided failure recovery policies for imitation learning. arXiv, 2024. 2
- [10] Maximilian Diehl and Karinne Ramirez-Amaro. A causalbased approach to explain, predict and prevent failures in robotic tasks. RAS, 2023. 3
- [11] Yufei Ding, Haoran Geng, Chaoyi Xu, Xiaomeng Fang, Jiazhao Zhang, Songlin Wei, Qiyu Dai, Zhizheng Zhang, and He Wang. Open6dor: Benchmarking open-instruction 6dof object rearrangement and a vlm-based approach. IROS,

2024. 10, 20

- [12] Yuqing Du, Ksenia Konyushkova, Misha Denil, Akhil Raju, Jessica Landon, Felix Hill, Nando de Freitas, and Serkan Cabi. Vision-language models as success detectors. arXiv,

2023. 2

- [13] Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, and Yijie Guo. Aha: A visionlanguage-model for detecting and reasoning over failures in robotic manipulation. arXiv, 2024. 2
- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv, 2024. 2
- [15] Ankit Goyal, Valts Blukis, Jie Xu, Yijie Guo, Yu-Wei Chao, and Dieter Fox. Rvt2: Learning precise manipulation from few demonstrations. RSS, 2024. 8

- [16] Yanjiang Guo, Yen-Jen Wang, Lihan Zha, Zheyuan Jiang, and Jianyu Chen. Doremi: Grounding language model by detecting and recovering from plan-execution misalignment. arXiv, 2023. 2, 6, 7, 5, 8, 9, 13
- [17] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. CVPR, 2023. 3
- [18] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 2021. 5
- [19] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. arXiv, 2024. 3
- [20] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv, 2022. 2, 6, 5, 7
- [21] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv,

2024. 3, 5, 7, 1, 4

- [22] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. RAL, 2020. 2, 5, 3, 4
- [23] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 2
- [24] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. ECCV, 2024. 5
- [25] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. EMNLP, 2014. 8
- [26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. ICCV, 2023. 3, 5
- [27] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. CVPR, 2024. 5, 7, 8, 2
- [28] Olivia Y Lee, Annie Xie, Kuan Fang, Karl Pertsch, and Chelsea Finn. Affordance-guided reinforcement learning via visual prompting. arXiv, 2024. 3
- [29] Gregory LeMasurier, Alvika Gautam, Zhao Han, Jacob W Crandall, and Holly A Yanco. Reactive or proactive? how robots should explain failures. HRI, 2024. 2
- [30] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. NeurIPS, 2020. 2
- [31] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang,

- Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. CoRL, 2022. 2, 5, 3, 4
- [32] Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. Semantic-sam: Segment and recognize anything at any granularity. arXiv, 2023. 5, 8, 2, 3
- [33] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ICML,

2023. 5

- [34] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. CVPR, 2023. 7
- [35] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. ICRA, 2023. 3
- [36] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. CVPR, 2023. 7
- [37] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-vocabulary robotic manipulation through mark-based visual prompting. arXiv, 2024. 3
- [38] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2024. 2, 5, 8
- [39] Jiaming Liu, Chenxuan Li, Guanqun Wang, Lily Lee, Kaichen Zhou, Sixiang Chen, Chuyan Xiong, Jiaxin Ge, Renrui Zhang, and Shanghang Zhang. Self-corrected multimodal large language model for end-to-end robot manipulation. arXiv, 2024. 2
- [40] Zeyi Liu, Arpit Bahety, and Shuran Song. Reflect: Summarizing robot experiences for failure explanation and correction. CoRL, 2023. 2, 10, 19
- [41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. CVPR,

2016. 8

- [42] Pasquale Minervini et al. awesome-hallucination-detection. https://github.com/EdinburghNLP/awesomehallucination-detection, 2024. 2
- [43] Yao Mu, Junting Chen, Qinglong Zhang, Shoufa Chen, Qiaojun Yu, Chongjian Ge, Runjian Chen, Zhixuan Liang, Mengkang Hu, Chaofan Tao, et al. Robocodex: Multimodal code generation for robotic behavior synthesis. arXiv, 2024. 3
- [44] Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess, Ayzaan Wahid, Zhuo Xu, et al. Pivot: Iterative visual prompting elicits actionable knowledge for vlms. arXiv, 2024. 3
- [45] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv, 2023. 3, 8, 4
- [46] Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, et al. Sofar: Language-grounded orientation bridges spatial reasoning and object manipulation. arXiv preprint arXiv:2502.13143, 2025. 2
- [47] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024.
- [48] Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, and Jing Shao. Mp5: A multi-modal open-ended embodied system in minecraft via active perception. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16307–

16316. IEEE, 2024.

- [49] Yiran Qin, Li Kang, Xiufeng Song, Zhenfei Yin, Xiaohong Liu, Xihui Liu, Ruimao Zhang, and Lei Bai. Robofactory: Exploring embodied agent collaboration with compositional constraints. arXiv preprint arXiv:2503.16408, 2025.
- [50] Yiran Qin, Ao Sun, Yuze Hong, Benyou Wang, and Ruimao Zhang. Navigatediff: Visual predictors are zero-shot navigation assistants. arXiv preprint arXiv:2502.13894, 2025. 2
- [51] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius, and Stefanie Tellex. Planning with large language models via corrective re-prompting. NeurIPS,

2022. 2

- [52] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. CVPR, 2023. 8
- [53] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv, 2024. 5, 8, 2, 3
- [54] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. CVPR, 2024. 7
- [55] Kenneth Shaw, Ananye Agarwal, and Deepak Pathak. Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning. arXiv, 2023. 5, 7, 4
- [56] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and selfreflection. arXiv, 2023. 2
- [57] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. CoRL,

2021. 2, 5, 6, 3, 4

- [58] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B Tenenbaum, Leslie Kaelbling, and Michael Katz. Generalized planning in pddl domains with pretrained large language models. AAAI, 2024. 2
- [59] Marta Skreta, Naruki Yoshikawa, Sebastian ArellanoRubach, Zhi Ji, Lasse Bjørn Kristensen, Kourosh Darvish,

- Al´an Aspuru-Guzik, Florian Shkurti, and Animesh Garg. Errors are useful prompts: Instruction guided task programming with verifier-assisted iterative prompting. arXiv, 2023. 2
- [60] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv, 2023. 2
- [61] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv, 2023. 2
- [62] Sai H Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. Chatgpt for robotics: Design principles and model abilities. IEEE Access, 2024. 3
- [63] David Venuto, Sami Nur Islam, Martin Klissarov, Doina Precup, Sherry Yang, and Ankit Anand. Code as reward: Empowering reinforcement learning with vlms. arXiv, 2024. 3
- [64] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. CoRL,

2023. 5, 2

- [65] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv, 2023. 3
- [66] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv, 2022. 2
- [67] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, Lei Bai, et al. Lamm: Language-assisted multimodal instruction-tuning dataset, framework, and benchmark. NeurIPS, 36:26650–26685, 2023. 2
- [68] Runpeng Yu, Weihao Yu, and Xinchao Wang. Attention prompting on image for large vision-language models. arXiv,

2024. 3

- [69] Zhouliang Yu, Jie Fu, Yao Mu, Chenguang Wang, Lin Shao, and Yaodong Yang. Multireact: Multimodal tools augmented reasoning-acting traces for embodied agent planning. ICLR,

2024. 2

- [70] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv, 2024. 3
- [71] Zhihao Yuan, Jinke Ren, Chun-Mei Feng, Hengshuang Zhao, Shuguang Cui, and Zhen Li. Visual programming for zero-shot open-vocabulary 3d visual grounding. CVPR,

2024. 3

- [72] Jialiang Zhang, Haoran Liu, Danshi Li, XinQiang Yu, Haoran Geng, Yufei Ding, Jiayi Chen, and He Wang. Dexgraspnet 2.0: Learning generative dexterous grasping in largescale synthetic cluttered scenes. CoRL, 2024. 5, 7, 9
- [73] Xinyu Zhang, Yuhan Liu, Haonan Chang, Liam Schramm, and Abdeslam Boularias. Autoregressive action sequence learning for robotic manipulation. arXiv, 2024. 5, 4, 8

- [74] Jinliang Zheng, Jianxiong Li, Sijie Cheng, Yinan Zheng, Jiaming Li, Jihao Liu, Yu Liu, Jingjing Liu, and Xianyuan Zhan. Instruction-guided visual masking. arXiv, 2024. 3
- [75] Zhi Zheng, Qian Feng, Hang Li, Alois Knoll, and Jianxiang Feng. Evaluating uncertainty-based failure detection for closed-loop llm planners. arXiv, 2024. 2
- [76] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. CVPR, 2017. 8
- [77] Enshen Zhou, Yiran Qin, Zhenfei Yin, Yuzhou Huang, Ruimao Zhang, Lu Sheng, Yu Qiao, and Jing Shao. Minedreamer: Learning to follow instructions via chain-ofimagination for simulated-world control. arXiv preprint arXiv:2403.12037, 2024. 2
- [78] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. CVPR, 2023. 7
- [79] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. NeurIPS,

2024. 7

## Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection

### Supplementary Material

The supplementary document is organized as follows:

- • Sec. A: More Discussions like Technical Details, Limitations, and Future Work.
- • Sec. B: Implementation Details of Painter, including data collection, training, and element pipeline details.
- • Sec. C: Environment Configuration, including environmental setups, control policies, and baseline details.
- • Sec. D: Evaluation Details, including detailed task definition, disturbances, and experimental results.
- • Sec. E: More Ablation Studies.
- • Sec. F: More Demonstrations of CaM.

###### A. More discussions

###### A.1. Methodology discussions

What contributes to generalization. VLM’s intrinsic knowledge and reasoning enable task-level generalization, while geometric abstraction ensures generalization across scenes and entities (e.g., objects, robots). The constraint-based formulation seamlessly integrates both aspects, achieving overall framework generalization.

Proactive detection for long-term dynamics. Our framework can potentially handle long-term dynamics by leveraging the Constraint Generator to decompose long-horizon tasks into subgoals and enables proactive detection within each subgoal to constrain the future state space rather than directly predict future states. Moreover, failed subgoals are re-planned with updated proactive constraints, ensuring adaptability to long-term dynamic changes.

Key methodology compared to Rekep [21]. Our key method differs from ReKep in two key aspects: (1) Element extraction: ReKep simply relies on DINOv2 for keypoint detection via 2D feature clustering, which offers a limited representation of constraints and geometric relationships crucial for failure detection. In contrast,

- our method achieves higher-order geometric abstraction by leveraging ConSeg to extract constraint-aware objects/parts in 2D space, then combining these with point clouds to integrate 3D geometric constraints. (2) Code Generation: Unlike ReKep, which lacks fine-grained guidance, our method leverages these geometry-rich elements as visual prompts

on multi-view images to generate code more accurately at each subgoal’s beginning, enabling seamless integration with various policies to form closed-loop systems.

Part segmentation for ambiguous or incomplete instructions. In our framework, we mainly leverage ConSeg’s abilities to handle cases where instructions with ambiguous or missing explicit part information. (1) We do not impose whether the textual instructions contain part information in our dataset collection pipeline. This allows the ConSeg to possess a certain ability to handle ambiguous instructions. (2) ConSeg architecture adopts a VLM-based design, leveraging VLM’s world knowledge to infer constraint-related parts for unseen objects. Based on our experiments (see Supp. F.4), ConSeg does demonstrate some generalization ability, but its performance on unseen objects is weaker than on objects within the training distribution.

###### A.2. Technical detail discussions

Executability and reliability of code. For executability, we validate the code using the path coverage of White-box Testing at the subgoal’s beginning, where the current state is used as input to verify the accuracy of each logic branch (i.e., every if-else block). If any error occurs during testing, the code is regenerated. For reliability, we prompt VLM to generate code that closely adheres to the given specifications and requirements. As frequent VLM calls increase hallucinations, we only invoke VLM once at each subgoal’s beginning, making it less affected by hallucination and more reliable.

Failure threshold Selection. We use two ways: (1) manually initializing thresholds for common tasks in an external knowledge base; (2) generating thresholds for unseen tasks using VLM’s internal knowledge. We find that the system remains robust to threshold selection in common settings, but the impact of threshold variations under diverse conditions remains unexplored and is left for future work.

Computational cost. The segmentation model is only invoked at the subgoal’s beginning to generate elements and excluded during execution, ensuring real-time code-based monitoring by tracking only the elements without high computational cost. We also perform parallel inference of segmentation models to accelerate at the subgoal’s beginning.

###### A.3. Limitations and Future Work

Despite promising results, our approach has several limitations. First, using Visual Language Models (VLMs)—such

as off-the-shelf GPT for code generation and constraintaware segmentation models—inevitably leads to hallucination issues. Even with minimized VLM usage, inaccuracies in code generation and biases in segmentation may persist. Integrating more relevant knowledge via methods like Retrieval-Augmented Generation (RAG) [30] or reducing multimodal large language model (MLLM) hallucinations [42] could alleviate these problems. Second, while we unify reactive and proactive failure detection as spatio-temporal constraint satisfaction problems and propose constraint elements for simplified real-time, highprecision monitoring, the constraint element representation has limitations: (1) It primarily focuses on failures detectable through explicit displacement and rotation, rendering it less effective for force-direction-related failures without noticeable displacement—for example, a robotic gripper failing to open a drawer due to incorrect force application. (2) The simplified representation abstracts objects and minimizes irrelevant visual details for efficient failure detection but may overlook critical visual cues and multimodal inputs, such as flowing water or audible sounds from a partially closed faucet, which are ignored in our current framework. Thus, exploring more robust representations that balance real-time precision with minimal information loss or integrate richer multimodal inputs is a promising direction for future research.

###### B. Constraint Painter

In this section, we first describe the data collection and annotation process of the Constraint-aware Segmentation Dataset. Next, we present the training details of the proposed ConSeg model. Finally, it is followed by a detailed explanation of the Element Pipeline.

###### B.1. ConSeg Data Collection

To ensure broader coverage of scenarios and objects in our dataset, along with text-based instructions, we utilized the BridgeData V2 dataset [64]. This large and diverse dataset of robotic manipulation behaviors comprises 60,096 trajectories collected across 24 environments, encompassing 13 distinct skills.

The dataset collection pipeline is shown in Fig. 7. The entire process is divided into three stages, i.e., trajectory decomposition, assigning textual information, and dataset collection. First, we decompose each trajectory’s instruction and initial observation from BridgeData V2 into subgoals for each stage, along with the constraints upon completion, constraints during execution, the corresponding object-part associations and element type for each constraint. Subfigure 1 in Fig. 7 illustrates a specific example, demonstrating the decomposition of the task “Take cup off plate”. Notably, the third subgoal, “Place the cup on the stove”, indicates that the initial observation ensures the decomposition process

fully understands the task’s contextual environment. During this process, we also obtain the constraint element type, which serves as the ground-truth text response for part-level constraint-aware segmentation. This decomposition is performed using the off-the-shelf GPT-4o API.

After obtaining the subgoals, constraints, and objectpart associations for each stage, we need to assign each frame to its corresponding stage. Since BridgeData V2 does not provide per-frame annotations, we addressed this issue by sampling pick-and-place data and leveraging the additional information (e.g., gripper open/close states) provided by BridgeData V2 for assignment. The pick-andplace task is typically divided into three stages: Approach, Grasp and Transfer, and Place, corresponding to the gripper states of open, closed, and open, respectively. Leveraging the characteristics of the pick-and-place task, we complete the frame-level assignment. Subfigure 2 in Fig. 7 illustrates a specific example of the assignment process.

Using the obtained frame-level constraint-aware object and part information, Instance-level and part-level segmentations are performed using Grounded SAM [53] and Semantic SAM [32], respectively. We conducted a sampled manual inspection of the final annotations to filter out errors and low-quality labels. Our final dataset is composed of 10,181 trajectories with 219,356 images.

###### B.2. ConSeg Training Details

We adopt LISA’s [27] loss function, including the nexttoken prediction loss for text output, and the combination of per-pixel BCE loss with DICE loss for mask output. Our ConSeg-13B model is trained on an 8 NVIDIA 80G H800 GPU for two days with a batch size of 4. Our training data comprises multiple components: SemanticSeg, ReferSeg, VQA, ReasonSeg, and ConstraintSeg, to ensure our model retains dialogue and reasoning segmentation capabilities while achieving constraint-aware segmentation. LISA inspires this training data setting. SemanticSeg includes ADE20K, COCO-Stuff, PACO-LVIS, and PASCALPart. ReferSeg includes refCLEF, refCOCO, refCOCO+, and refCOCOg. VQA includes LLaVAInstruct-150k. ConstraintSeg includes instance and part-level data.

The training setting described above is for the ConSeg-base model. Since the training data consists entirely of real-world scenarios, there is a significant gap between the simulation and real-world environments. To address this, the ConSeg model used in the simulation experiments is a fine-tuned version, called ConSeg-ft, finetuned on a small amount of data collected from the simulator. Specifically, we collect 100 trajectories from each simulator, sampled frames at 1 Hz, and utilize either ground truth masks from the simulator or annotations generated using Grounded SAM and Semantic SAM. Notably, we use the ConSeg-base model in real-world experiments, demon-

###### 1. Trajectory Decomposition

###### Step1: Decompose task instructions into subgoals, constraints, and object-part associations

Task: Take cup off plate Initial Observation

1.Subgoal: Grasp the cup

2.Subgoal: Move the cup off the plate

3.Subgoal: Place the cup on the stove

[Figure 71]

- - Constraints upon completion: Align the end-effector with the cup handle
- - Relevant objects: end-effector, cup
- - Relevant object parts: end-effector, cup handle

- Element Type: Point

- - Constraints during execution: None
- - Relevant objects: None
- - Relevant object parts: None

- - Constraints upon completion : The cup must be 10cm above the plate
- - Relevant objects: cup, plate
- - Relevant object parts: cup, plate

- Element Type: Point

- - Constraints during execution: The cup must remain within the gripper
- - Relevant objects: cup, Gripper
- - Relevant object parts: cup handle, Gripper

- - Constraints upon completion : The cup is on the stove
- - Relevant objects: cup , stove
- - Relevant object parts: cup , stove

- Element Type: Point

- - Constraints during execution : None
- - Relevant objects: None
- - Relevant object parts: None

[Figure 72]

- Element Type: Point

- Element Type: None

- Element Type: None

- 2. Assign Textual Information Step2: Assign each frame with subgoals, constraints, and object-part associations

Gripper Open Subgoal 1 Gripper Close Subgoal 2 Gripper Open Subgoal 3

- 3. Dataset Collection

|[Figure 73]<br><br>|
|---|

Step3: Leverage off-the-shelf models to auto-label and manually filter the results

Part-level Segmentation by Semantic SAM

[Figure 74]

[Figure 75]

[Figure 76]

Instance-level Segmentation by Grounded SAM

Manual Inspection

Final Dataset

Starting Frame of Subgoal 1 Instance-level Mask Part-level Mask

- Figure 7. Dataset Collection Pipeline. Our data is sourced from BridgeData V2 [31]. The data collection process consists of three steps:

(1) Using GPT-4o [1] to decompose the task instruction based on the initial observation from the first frame of the trajectory, generating subgoals along with two types of constraints for each subgoal (i.e., constraints during execution and upon completion) and object-part associations. (2) Utilizing external references (e.g., gripper open/close states) to assign subgoals, constraints, and object-part associations to each frame. (3) Leveraging off-the-shelf models (e.g., Grounded SAM [53], Semantic SAM [32]) to generate instance- and part-level masks (blue mask in this figure) automatically, followed by manual filtering to curate the final dataset.

strating the model’s generalization capability across different scenarios.

###### B.3. Element Pipeline Details

Here, we provide additional details about the Constraint Element Pipeline. We first filter out outliers after obtaining the constraint-aware object/part point clouds. Next, we calculate the 3D spatial bounds occupied by the remaining point cloud and determine the voxel size for voxelization based on the element type. We then perform point cloud clustering using the DBSCAN algorithm, which has advantages over other methods, including identifying clusters of arbitrary shapes, eliminating the need to predefine the number of clusters, and its effectiveness in high-density regions.

###### C. Environment Configuration

We first provide detailed descriptions of the simulators (CLIPort [57], Omnigibson [31], RLBench [22]) and realworld setups used in our study. We then discuss the lowlevel control policies implemented in these environments. Finally, we present the baselines and their implementation specific to each environment. Notably, our framework, CaM, is policy agnostic, meaning it can be adapted to any control policy without requiring any modification.

###### C.1. Environmental Setup

The CLIPort [57] simulator is a robotic manipulation benchmark to gather extensive data for imitation learning

https://github.com/cliport/cliport

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Leap Hand

RGB-D camera

Front View Left Shoulder View

Front View

Front View

UR5 Arm

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

RGB-D

camera

Wrist View Right Shoulder View

Top View

Top View

(a) CLIPort

(b) Omnigibson (c) RLBench (d) Real-world Setting

- Figure 8. Environmental Setup of three simulators and one real-world setting. For CLIPort [57] and OmniGibson [31], we provide thirdperson front and top views and the robot platforms are the UR5 arm and Fetch, respectively. RLBench [22] offers four camera views, including front left shoulder, right shoulder, and wrist views, with the robot platform being Franka equipped with a gripper. We provide a wrist and a third-person front view for the real-world setting, utilizing a UR5 robot equipped with a Leap Hand [55].

and train a language-conditioned multi-task low-level control policy. The environment features a UR5 robotic arm with a suction cup as the end effector for pick-and-place tasks and a spatula as the end effector for pushing tasks, both operating on a black tabletop. We use two cameras: a third-person front view and a top view to provide comprehensive perspectives of the tabletop, as shown in Fig. 8 (a).

The OmniGibson [31] simulator offers a realistic setting including a physics engine capable of supporting features such as lighting rendering, gravity effects, and temperature variations impacting objects within the environment. This platform also provides an extensive array of pre-configured scenes and objects, enabling researchers to customize setups and train mobile manipulation robots. We select version 1.1.0 for our study. This simulator involves a Fetch robot equipped with a gripper as the end effector, operating on a white tabletop. We utilize two cameras: a third-person front view and a top view to provide comprehensive perspectives of the tabletop, as shown in Fig. 8 (b).

RLBench [22] simulator is a widely used benchmark for robot manipulation, featuring tasks such as articulated objects and tool use. Researchers can gather data and train low-level control policies using imitation learning or reinforcement learning within this environment. RLBench features a Franka robotic arm with a gripper as its end effector, operating on a brown tabletop. Four cameras provide comprehensive tabletop coverage, as shown in Fig. 8 (c). Additionally, RLBench uses a sampling-based motion planner for motion planning given the next predicted action/pose.

In the real-world setup depicted in Fig. 8 (d), we utilize

https://github.com/StanfordVL/OmniGibson https://github.com/stepjam/RLBench

a fixed UR5 robotic arm with a Leap Hand [55] as the end effector. Two RealSense D415 RGB-D cameras capture the scene, one mounted on the wrist and the other positioned for a third-person front view.

###### C.2. Control Policy

In CLIPort, we use a pre-trained low-level policy the CLIPort [57] simulator provides to control the robotic arm and end effector. This policy can execute multi-task operations based on language instructions with RGB observations and its performance approaches perfection due to extensive imitation learning training. Notably, the policy is open-loop, meaning it does not adjust its actions in response to dynamic environmental changes (e.g., it will not immediately pick up a dropped block during movement but will continue with the previously planned actions).

In Omnigibson, We utilize ReKep [21] as our lowlevel control policy, transforming long-horizon tasks into a set of relationships between fixed keypoints at different stages. At each stage, an optimization algorithm computes these relationships to generate actions, enabling languageconditioned closed-loop control. Notably, ReKep employs a pre-trained large vision model (i.e., DINOv2 [45]) to process raw RGB data, extracting semantically relevant keypoints. This approach also serves as the compared method in our ablation study for extracting 3D points through constraint-aware segmentation, showing our superiority.

In RLBench, we employ the Autoregressive Policy (ARP) [73] as the control policy, which generates the next action based on historical observations and action sequences through an autoregressive process. This method achieves state-of-the-art performance in the RLBench.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Starting Frame Ending Frame Starting Frame Ending Frame Starting Frame Ending Frame

(a) Stack in Order

(b) Sweep Half the Blocks (c) Use Rope to Close the Opening Square

- Figure 9. CLIPort task demonstration. we present three types of tasks in our experiments, including the starting and ending frames.

In the real-world setting, We employ DexGraspNet 2.0 [72] as our low-level policy, which predicts the dexter-

- ous hand’s grasping pose based on the scene’s point cloud and facilitates the robotic arm’s action trajectory through motion planning to achieve robust generalized grasping. Notably, DexGraspNet 2.0 is an open-loop policy, which means it does not adjust to environmental changes during action execution. For example, if the target object’s position shifts while the arm moves, the system does not modify its motion plan. Therefore, it continues to execute toward the originally intended location to complete the grasp, failing.

###### C.3. Baseline Details

In CLIPort, we compare three baselines: CLIPort [57], CLIPort with Inner Monologue [20], and CLIPort with DoReMi [16]. We slightly modify the original implementations of these three baselines to suit our task requirements.

- (1) For CLIPort [57], the sole modification involves substituting the original oracle success detector with an offthe-shelf VLM (i.e., GPT-4o [1]) used as a failure detector. This change enables the robot system to determine whether to transition to the next subgoal using image-based vision question answering (VQA). Notably, CLIPort decomposes the instructions into a list of subgoals before the task begins and does not dynamically adjust or revert to previous subgoals upon detecting a failure. (2) For Inner Monologue [20], we replicate the implementation detailed in the original literature, by employing CLIPort for the low-level policy and an off-the-shelf VLM (i.e., GPT-4o [1]) as the planner. This pipeline determines the next subgoal based on the completed subgoals and current observations after each subgoal concludes. Notably, Inner Monologue queries the VLM only at the end of each subgoal, without considering events that occur during execution. (3) For DoReMi [16], we reproduce the implementation according to the original DoReMi paper and enhance it by replacing the VLM initially (i.e., BLIP2 [33]) used for repeated VQA-style queries during robotic execution with a more powerful VLM (i.e., GPT-4o [1]). Additionally, we substitute its LLM, which lacks environmental awareness, with the same GPT-4o to serve as the task planner.

In Omnigibson, we compare two baselines: ReKep [21]

and ReKep with DoReMi [16]. (1) For ReKep, we directly implement it using the official codebase. (2) For DoReMi, it is implemented as described above.

In RLBench, we compare two baselines: ReKep [21] and ReKep with DoReMi [16]. (1) For ReKep, we directly implement it using the official codebase. (2) For DoReMi, it is implemented as described above.

For the real-world evaluation, we compare two baselines: DexGraspNet 2.0 [72] and DexGraspNet 2.0 with DoReMi [16]. (1) For DexGraspNet 2.0, we directly implement it using the official codebase. (2) For DoReMi, it is implemented as described above.

###### D. Evaluation Details

In this section, we first detail the task specifications within the simulator and real-world evaluations. Then, we introduce the disturbances introduced in each task and the evaluation metrics used. Finally, we report the detailed experimental results and our analyses, with additional results not included in the main text due to space constraints.

###### D.1. CLIPort

###### D.1.1 Task, Disturbance and Metric Details

As shown in Fig. 9, we evaluate three tasks in CLIPort: (1) “Stack in Order”: Given blocks in red, green, and blue on a table, the robot must stack them with red at the bottom, green in the middle, and blue on top. (2) “Sweep Half the Blocks” With 40 blocks on the table, the robot must sweep approximately half of them (with a permissible error margin of ±10%, i.e., 16 to 24 blocks) to a designated area. (3) “Use Rope to Close the Opening Square”: The robot should use a rope to enclose an open square, to enclose the area sufficiently, rather than form a perfectly closed square.

We introduce two types of disturbances to the “Stack in Order” task: (1) after the suction cup grasps a block, there is a probability p at each step that the block will be released, causing it to drop; (2) The predicted placement position by the policy is perturbed by a uniform noise in the range [0,q] cm, potentially leading to block tower collapse.

For each task and disturbance type, we conduct 5 trials using different seeds, each comprising 12 episodes. We as-

Table 8. Detailed Performance in CLIPort. We report the success rate and execution time for three tasks, compared to baseline methods.

Tasks with Success Rate(%) ↑ Execution Time(s) ↓ disturbance CLIPort +Inner Monologue +DoReMi +Ours +Inner Monologue +DoReMi +Ours

- Stack in p=0.0 100.00 ± 0.00 100.00 ± 0.00 100.00 ± 0.00 100.00 ± 0.00 13.40 ± 1.82 13.40 ± 1.82 13.40 ± 1.82

- order with p=0.15 56.67 ± 6.11 81.67 ± 6.11 83.33 ± 5.17 95.00 ± 4.00 34.80 ± 3.12 26.00 ± 2.77 21.00 ± 1.75 drop p p=0.3 21.67 ± 8.33 75.00 ± 8.95 76.67 ± 9.52 88.33 ± 6.53 42.80 ± 3.18 34.20 ± 2.73 25.40 ± 2.95

Stack in q=1 90.00 ± 6.11 90.00 ± 6.11 96.67 ± 4.00 98.33 ± 3.27 24.80 ± 4.08 24.60 ± 4.66 24.20 ± 4.65

- order with q=2 41.67 ± 7.30 71.67 ± 8.33 75.00 ± 5.17 83.33 ± 5.17 39.40 ± 5.87 37.00 ± 6.29 29.20 ± 4.61 noise q q=3 15.00 ± 6.11 40.00 ± 8.00 40.00 ± 6.11 63.33 ± 8.33 58.20 ± 4.74 54.20 ± 6.02 36.80 ± 4.61 Sweep Half the Blocks 0.00 ± 0.00 18.33 ± 6.11 16.67 ± 8.95 75.00 ± 11.55 22.00 ± 2.91 16.60 ± 1.33 16.40 ± 1.00

Use Rope to Close

0.00 ± 0.00 68.33 ± 9.52 58.33 ± 18.62 76.67 ± 6.11 41.60 ± 6.34 65.80 ± 7.40 34.60 ± 2.81 the Opening Square

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Starting Frame Ending Frame Starting Frame Ending Frame

(a) Slot Pen

(b) Stow Book

[Figure 96]

[Figure 97]

Starting Frame Ending Frame

(c) Pour Tea

- Figure 10. Omnigibson task demonstration. we present three types of tasks in our experiments, including the starting and ending frames.

sess performance based on success rate and execution time, excluding the computational time for invoking the VLM. Results are reported as mean values with 95% confidence intervals. In the “Stack in Order” task, the robot must successfully stack the blocks in the specified order into a tower within 70 seconds, despite the two perturbations above. For the “Sweep Half the Blocks” task, the pre-trained policy aims to sweep blocks into a designated area. The robot must stop the policy once half of the blocks are in the target region. If, after 30 seconds, the number of blocks in the area falls within the required range (16–24), the task is considered successful. For the “Use Rope to Close the Opening Square” task, the pre-trained policy attempts to close an open rectangle into a perfect square using a rope. The robot must detect when the rectangle is sufficiently enclosed and immediately stop execution. Success is achieved if the robot halts within 70 seconds, and the enclosure is complete.

###### D.1.2 Detailed Experiment Results

In Tab. 8, we present detailed results in CLIPort, including those discussed in the main text and additional results.

In the “Stack in Order” task under severe interference conditions, our CaM shows an improvement of 18.33% and 17.5% in success rate over Inner Monologue [20] and DoReMi [16], respectively, while also reducing execution times by 38.7% and 14.4% compared to Inner Monologue and DoReMi, respectively. The failure detection and recovery processes are shown in Fig. 12 and Fig. 13.

In the “Sweep Half the Blocks” task, our CaM achieved success rates that are 4.1 and 4.5 times higher than those of Inner Monologue [20] and DoReMi [16], respectively. However, the success rate is not very high even in distraction-free scenarios. This is attributed to the high density of tracking points in the scene, which increases the likelihood of confusion and tracking errors, leading to inconsistent block counts within the target area. The completion

process of the task is also illustrated in Fig. 14.

In the “Use Rope to Close the Opening Square” task, our approach outperforms Inner Monologue [20] and DoReMi [16] by 8.34% and 18.34% in success rates, respectively, while also reducing execution times by 16.82% and 47.43% compared to Inner Monologue and DoReMi, respectively. We find that calculating the distance between the rope ends and the opening’s edges to determine closure is more accurate than directly querying a VLM with an image, allowing for earlier termination of the policy execution. The complete processes of the task are illustrated in Fig. 15.

###### D.2. OmniGibson

###### D.2.1 Task, Disturbance and Metric Details

As shown in Fig. 10, in the OmniGibson environment, we evaluated three distinct tasks: (1) Slot Pen: A pen placed on a desk is picked up, rotated to a near-vertical position, moved above a pen holder, and then inserted into the holder. (2) Stow Book: A book located on a desk is picked up and vertically positioned on a bookshelf. (3) Pour Tea: A teapot on the desk is lifted, horizontally moved above a teacup, and then tilted to pour tea into the cup.

We introduce three types of disturbances with varying constraint elements for each task: (1) Slot Pen Task: Pointlevel disturbances are applied as follows: (a) moving the pen while the robot is grasping it, (b) forcing the robot to release the pen mid-transfer, causing it to drop onto the table, and (c) moving the pen holder while the robot attempts to insert the pen. Despite these disturbances, the task is considered successful only if the robot can insert the pen into the holder. (2) Stow Book Task: Line-level disturbances include: (a) rotating the book during the robot’s grasping process, (b) altering the book’s pose during transfer to disrupt its vertical alignment, and (c) reorienting the book horizontally after it has been placed vertically on the shelf. Success requires the robot to place the book vertically on the shelf despite these disturbances. (3) Pour Tea Task: Surface-level disturbances involve: (a) tilting the container forward or backward during transfer, (b) inducing lateral tilts during movement, and (c) restoring the container to a level position during pouring. To succeed, the robot must prevent spillage and complete the pouring task under these disturbances.

We conduct experiments on three tasks, each consisting of one no-disturbance trial and three specific-disturbance trials. Each trial was repeated 10 times, and the performance was evaluated based on success rate, execution time (including the computation time for invoking the VLM), and the number of tokens used.

###### D.2.2 Detailed Experiment Results

Specific experimental results are detailed in the main text; here, we present additional demonstrations in Sec.F.2.

###### D.3. RLBench

- D.3.1 Task, Disturbance and Metric Details

As shown in Fig. 11, in RLBench, we evaluate six tasks across three categories of manipulation: (1) Articulated Object Interaction: (a) Open Drawer——Open the top drawer (b) Put in Drawer——Open the drawer and place an item into the open drawer. (2) Rotational Manipulation: (a) Screw Bulb——Screw in the red light bulb. (b) Turn Tap——Turn the left tap. (3) Tool Use: (a) Drag Stick——Use a stick to drag the cube onto the red target. (b) Sweep to Dustpan——Sweep dirt into the tall dustpan.

In RLBench, we avoid introducing additional disturbances, as its control policy naturally generates diverse failures to validate the effectiveness of our framework. The RLBench-trained policy lacks failure recovery mechanisms; thus, any episode flagged as a failure by the detection framework is deemed invalid, and a new episode is initiated.

For each task, we evaluate performance over 1,000 valid episodes (maximum 25 steps each), measured by the average success rate.

- D.3.2 Detailed Experiment Results

Code with elements can generalize better to monitor diverse tasks. Notably, despite our training data lacking information on articulated objects at both the instance and part levels, our method effectively handles them, accurately segmenting parts such as drawer handles. In the “Open Drawer” task, CaM achieves a 98.1% success rate, significantly outperforming DRM’s 90.6%. For “Put in Drawer”, CaM reaches 98.3%, surpassing DRM’s 87.7% by 10.6 percentage points. We attribute this generalization to two factors: (1) the inherent prior world knowledge of extensively pre-trained VLMs (e.g., SAM, LLaVA), which enables generalization to unseen tasks; and (2) our minimalist scheme that abstracts articulated objects into geometric components via constraint elements, ignoring irrelevant details, enhancing the generalizability.

We also demonstrate our method on additional unseen tasks, such as rotational manipulation and tool use, where it consistently outperforms baseline methods.

###### D.4. Real-world Evaluation D.4.1 Task, Disturbance and Metric Details

We conduct real-world evaluations on two tasks: (1) Simple Pick & Place: The robot should pick and place objects within 70 seconds. We include four kinds of objects: Deformable, Transparent, Small Rigid, and Large Geometric, with three objects in each category. The deformable objects are Loopy, Dog, and Rabbit toys, which undergo deformation when grasped by the dexterous hand. The transparent

Articulated Objects

Rotational Manipulation Tool Use

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Starting Frame

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Ending

Frame

(a) Open Drawer

(b) Put in Drawer (c) Screw Bulb (d) Turn Tap (e) Drag Stick (f) Sweep to Dustpan

Figure 11. RLBench task demonstration. we present six types of tasks in our experiments, including the starting and ending frames.

Table 9. Performance in RLBench. We report the success rate, compared to baseline methods.

|Method|Avg. Success Rate (%) ↑<br><br>|Articulated Object<br><br>|Tool-Use|Tool-Use|
|---|---|---|---|---|
| | |Open Drawer Put in Drawer<br><br>|Screw Bulb Turn Tap<br><br>|Drag Stick Sweep to Dustpan|
|RVT2 [15] ARP [73]<br><br>|89.83 91.27|90.3 97.6 93.9 91.0<br><br>|86.6 91.0 86.4 96.6<br><br>|93.8 79.7 88.1 91.6|
|+DRM [16]<br><br>+Ours|87.97 97.08<br><br>|90.6 87.7 98.1 98.3<br><br>|83.1 93.3 97.5 97.9|84.8 88.3 95.6 94.0<br><br>|

objects are a clear beverage bottle, a transparent glass, and a clear shampoo bottle. The small rigid objects consist of apple, pear, and peach models, while the large geometric objects include a large plate, ball, and pyramid. (2) Reasoning Pick & Place: In cluttered scenes, the robot must perform long-horizon tasks where the instructions contain ambiguous terms (e.g., “animals” or “fruits” without specifying particular types). Specifically, the tasks are: (a) “Clear all objects on the table except for animals”, and (b) ”Grasp the animals according to their distances to fruits, from nearest to farthest”, with ambiguous terms underlined.

For both tasks, we introduce two identical disturbances: (1) Moving the object during the robot’s grasping. (2) Removing the object from the robot’s dexterous hand during movement after grasping.

For each task, and each object involved in Simple Pick & Place, we conduct 10 trials. For each long-horizon task in Reasoning Pick & Place, we also conduct 10 trials. We evaluate performance based on success rate and execution time, including the computational time invoking the VLM. Results are reported as mean values with 95% confidence intervals. For the Simple Pick & Place, the robot has only one opportunity to autonomously release the object held in its gripper at a designated location. Any disturbance the robot encounters allows for a return and reattempt at grasping if the robot successfully detects it. Success is defined as

meeting these conditions within 90 seconds. For the Reasoning Pick & Place task, the robot must clear all objects on the table except for animals within 4 minutes for “Clear all objects on the table except for animals”. In the task “Grasp the animals according to their distances to fruits, from nearest to farthest”, the robot must sequentially grasp the animals in the correct order within 2 minutes, despite humaninduced distractions such as moving animals or fruits. Notably, this task is particularly challenging because the robot operates under an open-loop policy, preventing it from using closed-loop feedback to handle the dynamic distances between fruits and animals caused by external disturbances. Therefore, a failure detection framework is necessary to enable both reactive and proactive real-time detection with high precision, monitoring the distance changes and adjusting the grasping sequence accordingly.

D.4.2 Detailed Experiment Results

In Tab. 10, we present detailed results of Simple Pick & Place. CaM achieves success rates surpassing DoReMi by 20.4% when handling different kinds of objects. We show real-world demonstrations of Simple Pick & Place and Reasoning Pick & Place in Sec. F.3.

- Table 10. Detailed Performance of Single Pick & Place. We report the success rate and execution time. DGN is DexGraspNet 2.0 [72].

Tasks with Object Object Success Rate(%) ↑ Execution Time(s) ↓ disturbance types Name DGN +DoReMi +Ours +DoReMi +Ours Pick & Place with

Toy Loopy 0.00 80.00 100.00 64.91 ± 2.83 46.02± 3.11 the objects being Toy Dog 0.00 80.00 100.00 60.68 ± 4.00 47.06 ± 3.24 moved during Toy Rabbit 0.00 90.00 90.00 59.83 ± 1.82 45.77 ± 2.03 grasping

Deformable

Beverage Bottle 0.00 60.00 100.00 69.97 ± 7.89 47.61 ± 2.58 Glass Cup 0.00 70.00 90.00 76.99 ± 4.60 48.32 ± 3.22

Transparent

Shampoo Bottle 0.00 70.00 90.00 70.91 ± 5.68 48.31 ± 3.08 Small Rigid

Apple Model 0.00 80.00 100.00 64.65 ± 4.34 45.39 ± 0.71 Pear Model 0.00 90.00 90.00 67.11 ± 1.10 45.48 ± 1.01

Peach Model 0.00 70.00 90.00 65.48 ± 2.90 45.37 ± 0.64 Large Geometric

Plate 0.00 80.00 100.00 69.86 ± 2.64 45.18 ± 0.65 Ball 0.00 90.00 100.00 67.43 ± 2.63 45.37 ± 0.70

Pyramid 0.00 90.00 90.00 69.14 ± 3.32 45.42 ± 0.72 Pick & Place with

Toy Loopy 0.00 80.00 90.00 69.29 ± 4.87 60.86 ± 3.41 the objects being Toy Dog 0.00 70.00 100.00 66.09 ± 2.99 63.12 ± 3.75 removed during Toy Rabbit 0.00 80.00 90.00 70.86 ± 4.56 63.40 ± 3.88 movement

Deformable

Beverage Bottle 0.00 50.00 90.00 77.90 ± 2.89 61.97± 3.90 Glass Cup 0.00 70.00 90.00 70.00 ± 3.46 63.22 ± 4.35

Transparent

Shampoo Bottle 0.00 60.00 90.00 60 ± 4.28 63.00 ± 3.81 Small Rigid

Apple Model 0.00 70.00 90.00 70.21 ± 4.30 63.71 ± 3.91 Pear Model 0.00 60.00 100.00 72.70 ± 4.84 58.61± 2.41

Peach Model 0.00 60.00 90.00 66.48 ± 3.32 59.19 ± 2.59 Large Geometric

Plate 0.00 90.00 100.00 72.00 ± 2.77 59.21 ± 2.61 Ball 0.00 70.00 100.00 70.92 ± 3.37 61.57 ± 3.80

Pyramid 0.00 70.00 90.00 73.83 ± 2.82 60.25 ± 3.25

###### E. More ablation studies

Segmentation model ablations. Tab. 11 presents further ablation studies, replacing ConSeg with LISA and PixelLM. Our ConSeg shows superior overall framework performance (check Tab. 11 ID A & B & E), which represents a key technical contribution.

Failure detection mode ablations. (1) In Tab. 11 (ID C), proactive failure detection alone yields a lower success rate due to its inability to handle unforeseen failures; (2) In Tab. 11 (ID D), reactive detection alone achieves a slightly higher success rate but incurs longer execution times, as it only responds post-failure; (3) In Tab. 11 (ID E), the synergy of both modes achieves the best performance by addressing the limitations of each mode.

###### F. More Demonstrations and Prompts

This section presents additional demonstrations, including simulations and real-world scenarios of failure detection and recovery, as well as constraint-aware segmentation results and prompts.

###### F.1. CLIPort

Here, we present demonstrations of three tasks in CLIPort: “Stack in Order”, “Sweep Half the Blocks”, and “Use Rope

to Close the Opening Square”.

- Fig. 12 demonstrates how our framework detects failures

and assists in recovery when the placement positions predicted by the policy for the “Stack in Order” task are subject to a uniform [0,q] cm interference.

- Fig. 13 illustrates how our framework performs failure

detection and aids in recovery when, in the “Stack in Order” task, there is a probability p that blocks will fall due to being released by the robot’s suction cup at each step.

- Fig. 14 shows the “Sweep Half the Blocks” task, where

our framework precisely counts the blocks within a specified area and timely halts the policy execution to complete the task. In contrast, DoReMi [16] fails to stop the policy execution in time, leading to task failure.

- Fig. 15 depicts the “Use Rope to Close the Opening

Square” task. Our framework effectively detects when the rope closes the opening square and promptly stops the policy execution to complete the task successfully. Conversely, DoReMi fails to halt the policy execution on time; although it eventually succeeds in closing the opening, the excessive execution time results in task failure.

###### F.2. OmniGibson

As shown in Fig. 16, Fig. 17 and Fig. 18, we show how our framework detects failures and assists in recovery when

- Table 11. Following the Omnigibson evaluation protocol, we report the average success rate under disturbance (SR) and execution time to assess the impact of ConSeg and various failure detection modes on overall framework performance.

|ID<br><br>|Method<br><br>|Slot Pen SR(%) ↑ Time(s)↓|Stow Book SR(%) ↑ Time(s)↓<br><br>|Pour Tea SR(%) ↑ Time(s)↓|
|---|---|---|---|---|
|A<br><br>B<br><br><br>|LISA PixelLM|30.00 126.89 29.50 134.10<br><br>|42.50 118.93 41.00 124.26<br><br>|24.00 218.92 24.50 214.04|
|C D<br><br>|Only Proactive Only Reactive<br><br>|37.50 130.15 42.50 157.63|50.00 109.47 57.50 147.95<br><br>|32.50 192.23 35.50 284.15|
|E<br><br>|Ours|47.50 101.85<br><br>|65.00 93.08<br><br>|40.00 174.55|

facing point-, line- and surface -level disturbances.

###### F.3. Real-world Evaluation

Fig. 19 demonstrates the task “Clear all objects on the table except for animals”, where our framework achieves both reactive failure detection (e.g., detecting unexpected failures when humans remove objects from the robot’s grasp) and proactive failure detection (e.g., identifying target object movement during grasping to prevent foreseeable failures). This effectively enhances the task success rate and reduces the execution time.

###### F.4. Constraint-aware segmentation

As shown in Fig. 20, Fig. 21, Fig. 22, and Fig. 23 we present additional results on constraint-aware segmentation, including instance and part-level results. To demonstrate generalizability, we utilize out-of-distribution (OOD) data, including the RoboFail Dataset from REFLECT [40], datasets from the Open6DOF benchmark [11], and the RT-1 dataset [3]. Additionally, we showcase segmentation results from the OmniGibson.

###### F.5. Prompts

As illustrated in Fig. 24, we detail the prompt used to invoke an off-the-shelf VLM, i.e., GPT-4o [1], to generate Python code for monitoring.

###### Stack in Order —— The placement position is perturbed by a uniform noise

Initial Observation Step2: Move the

Step1: Pick the green block

Step3:Place the green block on the red block

Step4:Pick the green block

Step5: Move the green block over the red block

green block over the red block

|[Figure 110]|
|---|

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Detect Failure! The

green block is not on the red block

Step6:Place the green block on the red block

Step7: Pick the blue block

Step8: Move the blue block over the green block

Step9: Place the blue block on the green block

Step10: Pick the blue block

|[Figure 116]|
|---|

|[Figure 117]|
|---|

[Figure 118]

[Figure 119]

[Figure 120]

Detect Failure! The blue block is not on the green block

Detect Failure! The blue block is not on the green block

Step11: Move the blue block over the green block

Step12: Place the blue block on the green block

Step13: Pick the blue block

Step14: Move the blue block over the green block

Step15: Place the blue block on the green block

|[Figure 121]|
|---|

|[Figure 122]|
|---|

[Figure 123]

[Figure 124]

[Figure 125]

Success !

Detect Failure! The blue block is not on

the green block

- Figure 12. Demonstration of “Stack in Order”. We show how our framework detects failures and assists in recovery when the placement positions predicted by the policy for the “Stack in Order” task are subject to a uniform [0, q] cm interference. Red boxes indicate the occurrence of failures, while green boxes signify successful task execution.

Stack in Order —— The block will be released randomly, causing it to drop

Step6: Pick the blue block

Step5:Place the green block on the red block

Initial Observation Step2: Move the

Step1: Pick the green block

Step3: Pick the green block

Step4: Move the green block over the red block

green block over the red block

|[Figure 126]|
|---|

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Detect Failure! The green block is not held by robot

Step9: Move the blue block over the

Step10: Pick the blue block

Step15: Place the blue block on the

Step11: Move the blue block over the

Step7: Move the blue block over the green block

Step8: Pick the blue block

green block

green block

green block

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

[Figure 136]

[Figure 137]

[Figure 138]

Detect Failure! The

Success !

Detect Failure! The blue block is not held by robot

blue block is not held by robot

- Figure 13. Demonstration of “Stack in Order”. We show how our framework performs failure detection and aids in recovery when, in the “Stack in Order” task, there is a probability p that blocks will fall due to being released by the robot’s suction cup at each step. Red boxes indicate the occurrence of failures, while green boxes signify successful task execution.

Sweep Half the Blocks

Step1 Step2 Step3 Step4 Step5 Step6

|[Figure 139]|
|---|

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Ours

Success !

Step1 Step2 Step3 Step4 Step5 Step6

|[Figure 145]|
|---|

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Fail to stop !

DoReMi

Step8 Step9 Step10

[Figure 151]

|[Figure 152]|
|---|

[Figure 153]

Task Failed !

- Figure 14. Demonstration of “Sweep Half the Blocks” and comparison to baseline. We show our framework can precisely count the blocks within a specified area and timely halts the policy execution to complete the task. In contrast, DoReMi [16] fails to stop the policy execution in time, leading to task failure. Red boxes indicate the occurrence of failures, while green boxes signify successful task execution.

Use Rope to Close the Opening Square

Step1 Step2 Step3 Step4 Step5 Step6

|[Figure 154]|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Ours

Success !

Step1 Step2 Step3 Step4 Step5 Step6 Step7

|[Figure 160]|
|---|

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

DoReMi Fail to stop !

Step8 Step9 Step10 Step11 Step12 Step13 Step14

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

###### Fail to stop ! Fail to stop ! Time limit exceeded !

- Figure 15. Demonstration of “Use Rope to Close the Opening Square” and comparison to baseline. We show that our framework effectively detects when the rope closes the opening square and promptly stops the policy execution to complete the task successfully. Conversely, DoReMi fails to halt the policy execution on time; although it eventually succeeds in closing the opening, the excessive execution time results in task failure. Red boxes indicate the occurrence of failures, while green boxes signify successful task execution.

Slot Pen Task —— With Three Point-Level Disturbances

Proactive Failure Detect: The pen has moved

Moved Pen has been grasped

Initial Observation

Grasp the pen

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Lift the pen to reorient it upright

Reactive Failure Detect: The pen has dropped

Pen has been reoriented upright

Grasp the dropped pen

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Pen has been slotted into pen-holder successfully

Move the pen over penholder and drop it

Proactive Failure Detect: The pen-holder has moved

Move pen over the moved pen-holder

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

- Figure 16. Demonstration of “Slot Pen”. We show how our framework detects failures and assists in recovery when facing point-level disturbances. Red boxes indicate the occurrence of failures, light green indicates the recovery with subgoal success and dark green boxes signify successful task execution.

Stow Book Task —— With Three Line-Level Disturbances

Reactive Failure Detect: Book has been rotated down

Rotated Book has been grasped

Initial Observation

Grasp the book

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Leaned book has been reorient upright

Reorient the book upright

Lift the book to reorient it upright

Proactive Failure Detect: Book has been leaned

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Book has been stowed on the book-shelf

Reactive Failure Detect: Book is rotated horizontally

Grasp the horizontal book Book has been stowed onto book-shelf successfully

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

- Figure 17. Demonstration of “Stow Book”. We show how our framework detects failures and assists in recovery when facing line-level disturbances. Red boxes indicate the occurrence of failures, light green indicates the recovery with subgoal success and dark green boxes signify successful task execution.

Pour Tea Task —— With Three Surface-Level Disturbances

Lift the teapot with keeping upright to avoid spilling

Proactive Failure Detect: Teapot has tilted backward

Initial Observation

Grasp the teapot

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Teapot returns to horizontal pose

Proactive Failure Detect: Teapot has leaned to right

Teapot returns to horizontal pose

Teapot has aligned with cup opening

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Reactive Failure Detect: Teapot is forced to horizontal

Tea has been poured successfully

Tilt teapot to pour liquid

Tilt teapot to pour liquid

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

- Figure 18. Demonstration of “Pour Tea”. We show how our framework detects failures and assists in recovery when facing surface-level disturbances. Red boxes indicate the occurrence of failures, light green indicates the recovery with subgoal success and dark green boxes signify successful task execution.

###### Reasoning Pick &Place Task: Clear all objects on table except for animals.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Change the position of loopy and cup

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Remove the loopy from the robot hand Move the ball on the table

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Spin the transparent bottle

Success !

- Figure 19. Demonstration of “Clear all objects on the table except for animals”. We show that our framework achieves both reactive failure detection (e.g., detecting unexpected failures when humans remove objects from the robot’s grasp) and proactive failure detection (e.g., identifying target object movement during grasping to prevent foreseeable failures). This effectively enhances the task success rate and reduces the execution time.

###### RoboFail Dataset (Out of Distribution)

||Task: Boil water in a pot. Subgoal: Place the pot on the stove. Constraint: Thepot must be 10cm above the stove.<br><br>|
|---|
<br><br>|Task: Boil water in a pot. Subgoal: Pick up the pot. Constraint: Align the end effector with the handle on the pot.<br><br>|
|---|
<br><br>|Task: Sauté carrot slice in a saucepan. Subgoal: Slice carrot. Constraint: Theblade should be perpendicular to the carrot.<br><br>|
|---|
<br><br>|Task: Sauté carrot slice in a saucepan. Subgoal: Grasp the knife. Constraint: Align the end effector with the handle of the knife.<br><br>|
|---|
<br><br>Task,Subgoal, Constraint Observation<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>|Task: Secure pear and knife. Subgoal: Open the drawer. Constraint: Align the end effector with the handle of the drawer.<br><br>|
|---|
<br><br>|Task: Secure pear and knife. Subgoal: Put pear in drawer. Constraint: Thepears on the bottom<br><br>surface of the drawer.|
|---|
<br><br>[Figure 233]|
|---|

|ConSeg<br><br>Instance-level part-level<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]|
|---|

- Figure 20. Visualization of constraint-aware segmentation for the RoboFail Dataset [40]. This dataset is not included in the training data.

##### Open6DOR Benchmark (Out of Distribution)

|ConSeg<br><br>Instance-level part-level<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|
|---|

||Task: Put the ball into the upper drawer. Subgoal: Put the ball into the upper drawer. Constraint: The distance between the ball and the upper drawer should<br><br>be less than 10cm.|
|---|
<br><br>|Task: Put the ball into the upper drawer.<br><br>Subgoal: Grasp the ball. Constraint: Align the end effector with the ball.<br><br>|
|---|
<br><br>|Task: Take a piece of paper and lay it over the screwdriver.<br><br>Subgoal: Grasp a piece of paper.<br><br>Constraint: Align the end effector with the paper.<br><br>|
|---|
<br><br>Task,Subgoal, Constraint Observation<br><br>|Task: Place the mug on top of the green paper.<br><br>Subgoal: Grasp the mug. Constraint: Align the end effector with the handle of the mug.<br><br>|
|---|
<br><br>|Task: Place the mug on top of the<br><br>green paper. Subgoal: Move the mug on top of the green paper. Constraint: Move the mug 10cm above the green paper.<br><br>|
|---|
<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>|Task: Take a piece of paper and lay it over the screwdriver. Subgoal: Lay the paper over the screwdriver. Constraint: Move the paper 10cm above the screwdriver.<br><br>|
|---|
|
|---|

- Figure 21. Visualization of constraint-aware segmentation for the Open6DOF [11]. This dataset is not included in the training data.

#### RT-1 Dataset (Out of Distribution)

|ConSeg<br><br>Instance-level part-level<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]|
|---|

||Task: Place coke can upright. Subgoal: Place the can upright. Constraint: Let the line formed by the top and bottom of the can be parallel to the z-axis.<br><br>|
|---|
<br><br>|Task: Place coke can upright. Subgoal: Grasp the coke can. Constraint: Align theend effector<br><br>with the coke can.|
|---|
<br><br>|Task: Place red bull can in middle<br><br>drawer.<br><br>Subgoal: Grasp the red bull can. Constraint: Align the end effector with the red bull can.<br><br>|
|---|
<br><br>Task,Subgoal, Constraint Observation<br><br>|Task: Move sponge to green jalapeno chips.<br><br>Subgoal: Grasp the sponge. Constraint: Align theend effector with the sponge.<br><br>|
|---|
<br><br>|Task: Move sponge to green jalapeno chips. Subgoal: Move sponge to green jalapeno chips. Constraint: Make sure the distance between the sponge and the green jalapeno chips is less than 10cm.<br><br>|
|---|
<br><br>|Task: Place red bull can in middle drawer. Subgoal: Place the red bull can into the middle drawer. Constraint: The red bull can must be 10cm above the drawer.<br><br>|
|---|
<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]|
|---|

- Figure 22. Visualization of constraint-aware segmentation for the RT-1 dataset [3]. This dataset is not included in the training data.

#### Omnigibsom Dataset

|ConSeg<br><br>Instance-level part-level<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]|
|---|

||Task: Put the pen into the pen holder. Subgoal: Move the pen on the holder. Constraint: Keepthe penupright.<br><br>|
|---|
<br><br>|Task: Put the pen into the pen holder. Subgoal: Move the pen on the holder. Constraint: Keepthe penupright.<br><br>|
|---|
<br><br>|Task: Put the pen into the pen holder. Subgoal: Grasp the pen. Constraint: Align the end effector with the pen.<br><br>|
|---|
<br><br>Task,Subgoal, Constraint Observation<br><br>|Task: Pour the tea from the teapot into the teacup.<br><br>Subgoal: Grasp the teapot. Constraint: Align theend effector with the handle of teapot.<br><br>|
|---|
<br><br>|Task: Pour the tea from the teapot into the teacup.<br><br>Subgoal: Grasp the teapot. Constraint: Align the end effector with the handle of teapot.<br><br>|
|---|
<br><br>|Task: Put the pen into the pen holder.<br><br>Subgoal: Grasp the pen.<br><br>Constraint: Align the end effector with the pen.<br><br>|
|---|
<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]|
|---|

Figure 23. Visualization of constraint-aware segmentation for the Omnigibsom simulator.

Imagine you are monitoring a robot performing manipulation tasks by writing monitor code in Python. The monitors provides you with two images of the environment: one captured from the head camera showing a first-person perspective, and the other from a recorder camera showing a third-person perspective. Additionally, you receive a brief text instruction describing the next subgoal of the task that the robot needs to execute. These images are overlaid with our proposed elements, which consist of 3D points. Each element is associated with its own indices,

labeled on each point. For every given task, please follow these steps:

- For the given subgoal, specify two types of constraints to monitor: **"constraints during execution"** and **"constraints upon completion"**. Some examples are provided below:

- - Task: "Place the red block on top of the blue block":
- - Subgoal: "Grasp the red block":
- - Constraints upon completion: "The end-effector is aligned with the red block."
- - Constraints during execution: None.
- - Subgoal: "Move the red block over the blue block":
- - Constraints upon completion: "The red block is positioned higher than the blue block."
- - Constraints during execution: "The red block is held by the end-effector."
- - Subgoal: "Place the red block on the blue block":
- - Constraints upon completion: "The red block is on the blue block."
- - Constraints during execution: None. **Note:**
- - Each constraint function should take a dummy **end-effector position** and a set of **element positions** represented by 3D points, along with their past positions, as input. It should return two outputs:

- 1. A **boolean value** indicating whether the spatial positions satisfy the required constraints.
- 2. A **textual explanation** of the constraint being checked.

- - **Inputs to the constraints**:
- - `end_effector`: A NumPy array of shape `(T, 3)` representing the positions of the end-effector over the past `T` time steps, including the current position.
- - `element_position`: A NumPy array of shape `(T, E, K, 3)` representing the positions of `E` elements, each with `K` points, tracked over

the past `T` time steps.

- - `is_finished`: A boolean flag indicating whether this is a **"constraint upon completion"**.
- - **Indexing**:
- - Elements marked on the image correspond to indices starting from `0`, matching the indices in the `element_position` array.
- - **Allowed Libraries**:
- - You may only use **native Python functions** and **NumPy** functions.
- - **Function Return**:
- - The **last return statement** in the function must:
- - Return a **boolean value** (`True` or `False`), and
- - Be at the **outermost level** of the function.

- - **Guidelines for Writing Constraints**:
- - Avoid contradictory or overly detailed constraints.
- - Ensure the constraints are logically consistent and suitable for the specific task.

**Structure your output in a single python code block as follows:** ```python def constraint_monitor(end_effector, element_position, is_finished):

"""Put your explanation here."""

... return True, reason

```

## Query Query Task: "{task_instruction}" Query Current Subgoal: "{next_textual_subgoal}" Query Image:

Figure 24. Prompt of monitor code generation. We use this prompt, combined with additional task instructions, the current subgoal, and images from two perspectives, to enable an off-the-shelf VLM, i.e., GPT-4o [1], to generate Python code for monitoring.

