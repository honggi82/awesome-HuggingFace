## DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo

Hanwen Wang1,∗, Weizhi Zhao1,∗, Xiangyu Wang1,∗, Siyuan Huang2,∗, He Lin1, Boyuan Zheng1, Rongtao Xu3, Gang Wang4, Yao Mu2, He Wang5, Lue Fan1,†, , Hongsheng Li6, Zhaoxiang Zhang1, , Tieniu Tan1

1 NLPR & MAIS, CASIA 2 SJTU 3 MBZUAI 4 Beijing Institute of Basic Medical Sciences 5 PKU & Galbot 6 CUHK ∗ Equal contribution † Project lead Corresponding authors https://dexjoco.github.io

# arXiv:2605.16257v1[cs.RO]15May2026

[Figure 1]

[Figure 2]

###### Bimanual task

###### Tool-use task

Data collection system

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Low-cost User-friendly

[Figure 14]

Hardware Retarget software

|“Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.”|
|---|

|“Grasp the watering can and apply water to the plant.”|
|---|

[Figure 15]

Long-horizon task

Reasoning task

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

###### 11 dexterous manipulation tasks

Tool-use Bimanual setup Long-horizon Reasoning

[Figure 23]

[Figure 24]

###### 1.1K human demonstration trajectories

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Obs state:[third-person RGB, joint angle…] Action:[Absolute Pose, Absolute hand joint angle]

[Figure 29]

[Figure 30]

###### Domain randomizations

[Figure 31]

camera pose light conditions table texture table height

[Figure 32]

[Figure 33]

|“Open the microwave door, place the food inside the microwave, close the door and press the start button.”|
|---|

###### Supports multi-model evaluations

|“Execute the final two moves of the three-level Tower of Hanoi.”|
|---|

[Figure 34]

ACT Diffusion Policy 𝜋0.5 GR00T-N1.5

Figure 1: Overview of DexJoCo. DexJoCo is a dexterous manipulation benchmark with a toolkit for data collection and policy evaluation, covering tool-use, bimanual coordination, long-horizon execution, and reasoning. It includes 11 tasks, 1.1K human demonstration trajectories, and supports trajectory replay under domain randomization for robustness evaluation.

Abstract: Achieving human-level manipulation requires dexterous robotic hands capable of complex object interactions. Advancing such capabilities further demands standardized benchmarks for systematic evaluation. However, existing dexterous benchmarks lack tasks that reflect the unique manipulation capabilities of dexterous hands over parallel grippers, as well as comprehensive evaluation pipelines. In this paper, we present DexJoCo, a benchmark and toolkit for taskoriented dexterous manipulation, comprising 11 functionally grounded tasks that evaluate tool-use, bimanual coordination, long-horizon execution, and reasoning. We develop a low-cost data collection system and collect 1.1K trajectories across these tasks, with support for domain randomization to assess robustness. We benchmark modern models under diverse settings, including visual and dynamics randomization, multi-task training, and action-head adaptation. Through extensive empirical analysis, we identify several important insights and common limitations of current policies in dexterous manipulation, highlighting key challenges for future research in dexterous hand robot learning.

Keywords: Dexterous hand, Benchmark, Toolkit

### 1 Introduction

Learning from human demonstrations is an effective pathway toward generalist robot manipulation. In recent years, the robotics community has developed low-cost data collection pipelines [1, 2] and introduced a wide range of foundation models based on the VLA architecture [3, 4, 5, 6, 7]. However, most existing systems and datasets primarily focus on manipulator-gripper platforms. Human-level manipulation requires dexterous hands capable of fine-grained and contact-rich interactions, making dexterous manipulation learning increasingly important [8, 9, 10, 11, 12]. Advancing dexterous manipulation learning also requires standardized evaluation benchmarks to systematically measure model capabilities and guide future research.

Due to differences in environmental setups and robot configurations across laboratories, evaluating dexterous manipulation algorithms requires a benchmark. Although evaluation benchmarks for manipulator-gripper robotic systems have become relatively mature, and several benchmark efforts have also been introduced for dexterous hand manipulation, existing approaches still suffer from the following limitations: (1) Many existing works omit the manipulator and consider hand-only setups to enlarge the effective workspace, resulting in benchmark trajectories that are difficult to realize in real-world scenarios. (2) Current benchmarks evaluate in-hand manipulation or pick-and-place tasks; however, in-hand manipulation tasks are limited in functional diversity, while pick-and-place tasks fail to reveal the distinct capabilities of dexterous hands compared to simple grippers, restricting progress toward general manipulation. (3) Existing works lack reliable and user-friendly systems for collecting high-quality dexterous manipulation trajectories. Since complex dexterous hand behaviors are difficult to generate using conventional motion planning, most existing works rely on reinforcement learning or automated generation pipelines to obtain trajectories, which often produce behaviors that are inconsistent with natural human manipulation patterns. (4) Existing dexterous manipulation benchmarks lack standardized language instructions and unified data formats for modern VLA models, making systematic training and evaluation difficult.

Benchmark Hand Tool-Use Bimanual Reasoning Hand MoCap System Trajectory collection methods CALVIN [13] ✓ Motion Planning LIBERO [14] Human Demonstration RoboTwin 2.0 [15] ✓ Motion Planning DexMimicGen [16] ✓ ✓ Few Human + MimicGen Bi-DexHands [17] ✓ ✓ RL Policy DexJoCo (ours) ✓ ✓ ✓ ✓ Human Demonstration

- Table 1: Comparison with existing manipulation benchmarks. DexJoCo features more comprehensive evaluation task categories that highlight the unique capabilities of dexterous hands, together with an easy-to-use infrastructure for hand-motion-based data collection.

The robot learning community still lacks a standardized benchmark for dexterous hand manipulation, highlighting the need for an evaluation framework. Therefore, we present DexJoCo, a benchmark and toolkit for task-oriented dexterous manipulation, with comparisons to existing manipulation benchmarks summarized in Table 1. In designing the tasks, we emphasize functionally grounded interactions that highlight the unique capabilities of dexterous hands, particularly in tool-use scenarios that require fine-grained finger coordination and complex object interactions. Furthermore, we introduce long-horizon tasks, bimanual coordination tasks, and reasoning tasks to evaluate policy performance across multiple dimensions. A comprehensive evaluation framework requires not only diverse and functionally meaningful task definitions, but also an efficient system for collecting manipulation trajectories. To this end, we develop a low-cost teleoperation hardware setup together with a retargeting module that reduces the embodiment gap between human hand motions and dexterous hand control. Using this system, we collect demonstration data across our task suite and evaluate several modern manipulation policies, leading to several insights into the limitations and challenges of current dexterous manipulation policies, which may facilitate future progress in robot learning. Our contributions are summarized as follows:

- (1) DexJoCo benchmark: We introduce a dexterous manipulation benchmark featuring functionally grounded tasks that evaluate the unique capabilities of dexterous hands, including fine-grained manipulation, tool-use, bimanual coordination, long-horizon execution, and reasoning capabilities.
- (2) DexJoCo toolkit: We develop a low-cost teleoperation system with a retargeting module for efficient collection of dexterous manipulation demonstrations.
- (3) DexJoCo datasets: We collect 1.1K human demonstration trajectories in simulation and evaluate several modern policies, where dexterous hand trajectory data remains relatively limited in prior work.

### 2 Related Works

Dexterous Manipulation Benchmark When designing benchmarks for manipulator–gripper robotic systems, the relatively low degrees of freedom of these robots make it possible to collect large amounts of trajectory data at low cost or through automated procedures [18, 14, 19, 20, 13, 15, 21, 22, 23, 24, 25]. However, achieving human-level manipulation requires dedicated benchmarks for manipulator–hand robotic systems. Several existing dexterous hand benchmarks [17, 26, 27] are primarily designed for reinforcement learning and mainly focus on in-hand manipulation. While effective for evaluating low-level dexterous control, their task formulations often provide limited coverage of functional, task-oriented interactions with the environment. Moreover, without access to high-quality human demonstrations, reinforcement learning alone often struggles to generate reasonable and physically plausible manipulation trajectories. Some recent works have adopted human demonstrations or automatically generated trajectories to enable imitation learning for dexterous hand systems [28, 16, 29]. Nevertheless, the resulting task designs are often not sufficiently challenging or functionally rich to assess human-level dexterous manipulation, and therefore fail to highlight the fundamental differences between hand-based manipulation and gripper-based manipulation. Therefore, the tasks in the DexJoCo benchmark are designed to be more functional and closely aligned with real-world scenarios. By comparing dexterous hand systems with gripper-based systems, the DexJoCo benchmark explicitly reveals the advantages of dexterous hands in achieving human-level manipulation.

Dexterous Hand Trajectory Collection The technical pipeline for collecting trajectories on manipulator–gripper robotic systems has become increasingly mature [30, 31, 2, 1]. In practice, action recording only requires tracking the target 6D pose of the robot end-effector, while the gripper itself typically has only a single degree of freedom, eliminating the need for specialized hardware. Trajectory collection for dexterous hand systems is considerably more challenging due to their high degrees of freedom. In practice, specialized hardware is often required to capture the pose of each fingertip and retarget it to the robotic hand. Standard RGB camera-based solutions offer the lowest hardware cost [32, 33], but they frequently suffer from severe occlusion and inefficient hand pose estimation. VR headset-based systems can improve the efficiency of hand pose tracking [34, 35, 36], yet they are often uncomfortable for prolonged use and still remain susceptible to partial occlusion. In contrast, motion-capture gloves or exoskeleton devices can largely eliminate occlusion issues and avoid the need for dedicated vision-based hand pose estimation algorithms [37, 38, 39, 40, 41, 42, 43, 44], enabling the direct acquisition of high-frequency and high-precision hand motion data. Their main drawbacks, however, are the relatively high hardware cost, and in the case of exoskeletons, limited wearing comfort. Therefore, we aim to design a data collection system based on motion-capture gloves, together with an effective retargeting algorithm, to achieve both low cost and ease of use.

### 3 DexJoCo Benchmark and Toolkit

DexJoCo provides a benchmark and toolkit for dexterous manipulation, including task environments, human demonstration collection tools, policy training interfaces, and evaluation utilities. Fig. 2 illustrates the overall DexJoCo pipeline, from task construction and trajectory collection to policy training

##### and evaluation. In this section, we describe the Robot Setup and Observation State, teleoperation system, task design, domain randomization settings, and policy evaluation.

[Figure 35]

###### Structured scene construction in MuJoCo Human teleoperation

###### Data collection

[Figure 36]

|model state|
|---|

= mujoco.load_xml("scene.xml")

demo/ replay.zarr/

= init_state(model) task =

|Task(success_condition)|
|---|

Task(

|Retargeted action|
|---|

data: [action, state, timestamp] videos/

[Figure 37]

[Figure 38]

video: [ego, wrist_left, wrist_right]

Data format conversion and policy training

###### Domain randomization via replay(optional)

Policy evaluation

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

ACT

|Client| |Task environment|
|---|---|---|
| | | |

LeRobotDataset v3.0

𝜋0.5

Third-person camera pose

Table texture

| | | |
|---|---|---|
|Server| |Policy inference|

GR00T N1.5

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Zarr Diffusion Policy

Lighting direction and color

Table height

- Figure 2: DexJoCo pipeline. 3D assets are first imported into MuJoCo, where structured success conditions are defined based on object poses, articulated joint states, contact conditions, and temporal constraints. Human demonstrations are collected through the teleoperation system, with actions directly recorded as robot position control commands. Replay-based visual augmentation can optionally be applied to the collected trajectories. The data can then be converted into mainstream formats such as LeRobot and DP Zarr through the provided interface. After training, policies are evaluated in the constructed task environments using a server–client framework.

- 3.1 Robot Setup and Observation State

DexJoCo is developed on top of the MuJoCo physics simulator, enabling accurate and realistic physics modeling. The robotic system consists of three main components: a Rethink Robotics mount as the base, a Franka Panda manipulator, and an Allegro Hand for dexterous manipulation. These assets are mature, precisely modeled, and widely adopted in the robotics community. DexJoCo provides rich perceptual observations from the simulation environment, including third-person and wrist-mounted RGB and RGB-D images, object poses of the interactive entities in the scene, the robot’s motion states, the current end-effector pose, and the joint angles of the hand. The action space in the collected robot trajectories is defined as follows: manipulator actions are represented by the target absolute end-effector pose in the world coordinate frame, while hand actions are specified as target absolute joint angles.

- 3.2 Human Demonstration Data Collection System

[Figure 52]

Base station

Rokoko glove

Vive tracker

Retarget MLP

[Figure 53]

[Figure 54]

[Figure 55]

- Figure 3: Human demonstration data collection system. The left figure shows the overall teleoperation system. A Rokoko glove is used to capture hand poses, while an HTC Vive tracker is employed to track the wrist pose. The right figure shows that a retargeting mapping is trained to convert human fingertip poses into joint configurations of the Allegro hand.

Hardware Design The hardware system in DexJoCo is designed to balance low cost and usability. Hand motion capture is performed using Rokoko Smartgloves, avoiding the occlusion issues of camera-based methods, while two HTC Vive Trackers and two HTC Base Stations are used to track wrist motions and control the Franka end-effector pose. This setup enables accurate teleoperated

trajectory collection and remains low-cost at approximately $2,300 USD. A simple 3D-printed connector is further designed to integrate the trackers and gloves into a unified assembly.

Teleoperation Algorithm The teleoperation system consists of hand motion retargeting and wrist motion tracking. Due to the structural differences between human and robotic hands, direct linear mapping is infeasible. We adopt GeoRT [38], a lightweight self-supervised retargeting method without requiring paired human-robot annotations. The retargeting model f maps human fingertip keypoints xH to robot joint positions qR = f(xH) by minimizing:

L = Ldir + λ1Lcover + λ2Lflat + λ3Lpinch + λ4Lcol (1)

where Ldir preserves fingertip motion directions, Lcover enlarges workspace coverage, Lflat maintains uniform sensitivity, Lpinch preserves pinch behaviors, and Lcol avoids self-collisions. Only fingertip workspaces are recorded during data collection and used for training, enabling accurate real-time teleoperation. For wrist tracking, the tracker is fixed such that human wrist motions align with the Franka end-effector. The initial wrist pose is recorded as a reference, and subsequent actions are represented as relative pose changes. The robot then executes these delta actions to reproduce the desired motion.

#### 3.3 Task Design in the Benchmark

Task Design

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Water plant Fold glasses Click mouse Pinch tongs Pick bucket Hammer nail

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Bimanual photograph

Bimanual microwave cook

Bimanual

Bimanual

Bimanual

unlock ipad

hanoi

assembly

Interactive Asset Design

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Enter password to unlock Spray by pressing Click mouse to wake screen

- Figure 4: Task design in DexJoCo. The top panel illustrates the task environment design, showing the initial state of each task. The bottom panel presents the visual and interactive properties of the task assets.

Formulation Each task in DexJoCo is defined by a set of interactive objects and task goals: T = (O,G), where O = {o1,o2,...,om} denotes the set of interactive objects in the scene. The task goal is formulated as a set of functional success constraints G = {gseq,gpose,gjoint,gcontact}, where gseq denotes temporal or sequential execution constraints, gpose specifies target object pose conditions, gjoint represents articulated joint-state requirements, and gcontact defines collision. A task is considered successful only when all task-dependent goal constraints are satisfied simultaneously.

Task Design Principles DexJoCo tasks are systematically constructed to cover diverse dexterous manipulation capabilities, as shown in Fig. 4. We follow several core design principles. (1) Functional

Interaction: Tasks are designed with functional semantics that reflect everyday human activities rather than simple object relocation. Moreover, the involved objects provide explicit visual interaction feedback, enabling intuitive perception of task progress and completion. (2) Dexterity Dependency: Tasks are designed such that successful execution fundamentally depends on dexterous manipulation capabilities, including fine-grained finger coordination and articulated object interaction, which cannot be reliably achieved by parallel grippers. (3) Long-Horizon Compositionality: Tasks involve multi-stage execution with temporal dependencies between sub-goals. (4) Bimanual Coordination: A subset of tasks requires coordinated bimanual manipulation with asymmetric functional roles between the two hands. Based on these principles, tasks are organized into capability-oriented categories, including tool-use tasks, reasoning tasks, bimanual coordination tasks, and long-horizon tasks, ensuring broad and structured benchmark coverage. The construction cost of each individual task is relatively low, enabling efficient and scalable benchmark expansion.

Task Asset Construction The base scene design follows RoboSuite [45], and we adopt robot assets from MuJoCo Menagerie [46]. New tasks are constructed by instantiating task-specific objects within the base scene and defining corresponding success conditions. For each task, we curate high-quality assets from RoboCasa [22] and PartNet-Mobility from SAPIEN [47], which typically provide predefined physical and dynamic parameters. For assets without such annotations, we generate them using Hunyuan3D [48] and manually assign physically plausible properties. To enhance functional interaction realism, we additionally incorporate explicit visual state changes into task assets. For example, in the Water Plant task, water is displayed when the watering can handle reaches a predefined joint state threshold. In the iPad Unlock task, buttons are highlighted upon finger contact. In the Click Mouse task, pressing the mouse button activates the computer display, indicating successful interaction.

#### 3.4 Domain Randomizations

To evaluate the policy over a broader data distribution, we introduce a domain randomization option for all task scenarios. To generate more diverse trajectories, we not only randomize the placement of objects on the table plane but also vary the table height. To increase visual diversity, we randomize the third-person camera poses, the direction and color of scene illumination, and the tabletop textures. Notably, visual randomization can be efficiently applied by replaying the same trajectories under different rendering settings, enabling scalable augmentation without additional teleoperation effort. For camera pose randomization, we first densely sample camera poses uniformly on a spherical surface, and then select 50 poses with minimal occlusion. For lighting randomization, we follow a simple procedure inspired by our implementation. Each light in the scene is randomized in terms of its position, direction, and diffuse color to introduce diverse illumination conditions. For tabletop texture randomization, we sample textures from a pre-constructed texture library. Detailed visualization and task-specific settings are provided in App. C.

#### 3.5 Imitation Learning Policy Evaluation

Baseline Models We benchmark four policies on DexJoCo: ACT [49], Diffusion Policy [50] (DP-T and DP-C), π0.5 [6], and GR00T N1.5 [7]. ACT (via C-VAE) and DP (via diffusion) are trained from scratch using vision and proprioception. In contrast, π0.5 and GR00T N1.5 (fine-tuned via LoRA [51]) use flow-matching and additionally condition on language. Because their default 32-dimensional action heads are insufficient for bimanual tasks, we retain these pretrained weights but randomly initialize the extra dimensions (partial pretrain-AH). All baselines formulate action chunking as:

P(at:t+k−1) = πθ(at:t+k−1 | st−h+1:t,l) (2) In the formula, given h frames of historical observations s and an optional language instruction l, it models the conditional probability of a future k-step action chunk.

Model Deployment For evaluation, we use an asynchronous inference mechanism inspired by SmolVLA [52]: the next action chunk is generated while the current one executes, eliminating idle

waiting. Overlapping chunks are temporally ensembled for smoothness. This mirrors real-world deployment and highlights the impact of inference frequency: lighter policies run faster, utilizing more recent observations to reduce idle frames and improve reactivity.

### 4 Experiments

(a) Average Success Rate (b) 𝜋0.5 Failure Mode Analysis (c) DP-C Failure Mode Analysis

#### Figure 5: Performance evaluation and failure mode analysis. DP denotes Diffusion Policy, with

-T and -C representing Transformer and CNN-based architectures, respectively. (a) Comparison of average success rates across different baselines under the “rand-obj” (Table 2) condition. (b) and (c) provide a detailed breakdown of failure modes for π0.5 and DP-C. These statistics are aggregated from 550 evaluation trials (50 runs across 11 tasks) to identify main bottlenecks in dexterous manipulation.

DP-T DP-C ACT π0.5 GR00T N1.5

Task

rand-obj rand-full rand-obj rand-full rand-obj rand-full rand-obj rand-full rand-obj rand-full Hammer Nail 81.3±3.1 18.7±1.2 58.7±4.2 19.3±3.1 50.0±7.2 22.7±6.1 84.7±5.0 17.3±5.0 67.3±4.2 38.7±8.3

Click Mouse 62.0±2.0 25.3±8.1 74.0±5.3 34.7±4.2 61.3±3.1 48.7±5.0 64.7±8.1 54.7±7.0 85.3±3.1 74.0±2.0 Pick Bucket 83.3±3.1 58.7±15.0 70.0±2.0 68.0±3.5 64.0±4.0 36.0±5.3 84.0±7.2 78.7±6.1 72.0±6.0 69.3±6.1 Pinch Tongs 22.7±5.8 18.7±3.1 57.3±6.4 28.7±11.7 31.3±3.1 23.3±7.0 24.0±6.9 18.7±1.2 12.7±2.3 5.3±2.3 Fold Glasses 53.3±3.1 11.3±1.2 54.0±15.9 15.3±7.6 47.3±11.0 7.3±3.1 72.0±3.5 39.3±3.1 27.3±2.3 20.7±3.1

Water Plant 84.0±3.5 56.0±8.7 63.3±3.1 54.0±5.3 47.3±4.6 52.7±8.1 88.7±3.1 75.3±6.4 72.7±1.2 66.0±5.3 Unlock iPad /B 8.0±2.0 2.0±2.0 52.0±2.0 12.0±3.5 9.3±3.1 0.7±1.2 12.0±3.5 0.0±0.0 12.7±11.0 0.0±0.0 Hanoi /B 24.7±4.6 0.7±1.2 12.7±3.1 9.3±6.1 6.0±2.0 4.7±2.3 15.3±3.1 15.3±2.3 0.7±1.2 0.0±0.0

Assembly /B 4.7±3.1 0.0±0.0 3.3±1.2 0.0±0.0 0.0±0.0 0.0±0.0 5.3±1.2 0.0±0.0 0.7±1.2 1.3±1.2 Microwave /B 73.3±11.6 21.3±4.6 54.0±12.5 62.7±6.4 66.0±2.0 50.0±6.9 70.0±3.5 54.7±6.1 50.7±4.6 42.0±7.2 Photograph /B 56.7±4.6 7.3±1.2 24.0±8.7 8.7±4.2 7.3±1.2 3.3±1.2 56.7±5.0 21.3±2.3 40.7±7.0 18.7±7.0

Avg. 50.4±1.4 20.0±1.4 47.6±2.0 28.4±1.5 35.5±2.0 22.7±1.3 52.5±1.4 34.1±2.9 40.2±0.3 30.5±1.1

- Table 2: Performance comparison on benchmark tasks. Mean success rate (%) ± std over 11 tasks for five models. “/B”: bimanual tasks; “rand-obj”: only object placement and table height randomized; “rand-full”: additionally randomizes camera poses, illumination direction/color, and tabletop textures. Each task is trained under both “rand-obj” and “rand-full” data regimes.

Challenging DexJoCo Bench Exposes Trade-offs Among Pre-training, Scale, and Architecture. As shown in Table 2 and Fig. 5, the benchmark proves highly challenging: some policies never succeed on difficult bimanual tasks. For each task, policies are trained on in-domain data under both “rand-obj” and “rand-full” regimes. Under visual randomization (“rand-full” in Table 2), success rates drop sharply across nearly all policies, indicating limited robustness. π0.5 achieves the highest overall success rates, benefiting from large-scale pre-training, yet the much smaller DP-T (∼100M, trained from scratch) performs comparably: π0.5 dominates single-arm tasks while DP-T is competitive on bimanual ones, likely because training the extra action dimensions from scratch diminishes π0.5’s pre-training advantage. Surprisingly, DP-C substantially outperforms all other policies on Unlock iPad and Pinch Tongs. The right panel of Fig. 5 reveals that DP-C excels at precise operations (e.g., button pressing) and hinge interactions (e.g., squeezing tongs). We hypothesize that this advantage stems from being the only policy to use FiLM [53] for observation injection, rather than self or cross attention, which may provide stronger fine-grained visual perception and benefit precise manipulation.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

(a) missed button press (b) misaligned insertion (c) other failure cases

Figure 6: Visualization of failure cases in typical tasks.

Failures in Fine-grained Actions, Insertion, and Memory As Fig. 6 shows, in button-based tasks (Unlock iPad, Click Mouse, Photograph), the policies are able to pick up the tablet or camera, push the mouse onto the mousepad, yet often fail to click the intended buttons, suggesting they can perceive the object but overlook its interactive elements. Insertion steps pose a high probability of failure, as observed in Assembly and Hanoi. In Pinch Tongs, the policies often grasp but fail to squeeze and release the tongs, possibly due to insufficient temporal memory. In Microwave, the policies typically place the hot dog into the microwave but then withdraw it alongside the hand.

multi-task rand-dynamics rand-AH Task DP-T π0.5 DP-T π0.5 π0.5

Hammer Nail 58.7±5.0 86.7±3.1 77.3±6.4 82.0±12.5 76.7±1.2

Click Mouse 38.7±3.1 80.7±3.1 0.0±0.0 65.3±4.2 54.0±2.0 Pick Bucket 55.3±7.6 83.3±8.1 80.7±3.1 90.7±1.2 86.0±5.3 Pinch Tongs 6.0±5.3 45.3±6.1 15.3±4.2 17.3±7.6 26.0±2.0 Fold Glasses 11.3±5.0 42.0±6.0 40.7±4.6 60.7±7.0 64.0±3.5

Water Plant 60.0±6.9 84.0±4.0 76.0±6.0 88.0±4.0 90.7±5.0 Unlock iPad /B 0.0±0.0 0.7±1.2 0.7±1.2 0.7±1.2 2.7±3.1 Hanoi /B 8.0±2.0 6.0±0.0 29.3±2.3 11.3±3.1 20.7±1.2

Assembly /B 1.3±2.3 3.3±2.3 8.0±5.3 2.7±1.2 3.3±4.2 Microwave /B 42.7±6.4 39.3±13.0 70.0±9.2 41.3±12.2 61.3±3.1 Photograph /B 28.0±6.0 29.3±1.2 59.3±6.4 52.0±2.0 50.0±0.0

Avg. 33.2±2.4 45.5±1.5 41.6±0.3 46.5±2.6 48.7±0.9

- Table 3: Multi-task, dynamics, and action-head evaluations. “multi-task”: models trained jointly on all tasks; “rand-dynamics”: evaluation with randomized

dynamics parameters; “rand-AH”: π0.5 with randomly reinitialized action head.

Multi-task Training Degradation When jointly training on all tasks (Table 3, multitask) with the same number of steps as single-task training, DP-T degrades on every task, while π0.5 achieves a success rate increase on Click Mouse and Pinch Tongs, though its average success rate drops.

π0.5 Shows Stronger Robustness Under randomized joint friction, stiffness, and object mass (Table 3, rand-dynamics), π0.5 averages higher success than DP-T. This confirms our simulated benchmark captures performance trends under varying dynamics, serving as a proxy for real-world capabilities despite sim-to-real gaps.

Retaining Pretrained Action-Head Performs Better We compare partial pretrain-AH (Table 2) against fully random reinitialization (Table 3, rand-AH), and find that retaining pretrained weights yields higher success rates on most tasks and a better average.

VLA Model Fails to Exhibit Language Generalization We train π0.5 on Unlock iPad using single-digit passwords (1-5) and evaluate on seen digits (1,2,4), arithmetic expressions (1+1, 2+2), and English words (two, one plus one). The results show that the model defaults to a fixed action bias rather than true language conditioning, see App. A.

### 5 Discussion

Through our study, we identify several limitations in existing approaches: Lack of Dexterous Hand Centric Foundation Models. Current VLA models are largely pretrained on gripper-based data, resulting in an action space mismatch for dexterous hands. Their action heads fail to capture highdimensional joint coupling, limiting expressivity and transfer, and motivating embodiment-aware representations with hand-centric pretraining. Limitations of Vision-Only Policies in ContactRich Manipulation. Vision-only policies are insufficient for contact-rich manipulation. Even with proprioception, they miss critical cues such as contact forces; incorporating tactile sensing enables more complete interaction modeling, making multi-modal policies necessary for precision. We note that the following aspect is not addressed in this work and is left for future investigation: Sim-to-Real Transfer via More Realistic Modeling. Improving simulation fidelity across physical, visual, and sensing aspects (e.g., object properties, rendering, and sensor signals) can yield more consistent dynamics and perception, improving zero-shot transfer and motivating systematic sim–real alignment beyond domain randomization.

### References

- [1] Z. Fu, T. Z. Zhao, and C. Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. In Conference on Robot Learning (CoRL), 2024.
- [2] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems (RSS), 2024.
- [3] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.
- [4] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.
- [5] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning, pages 2679–2713. PMLR, 2025.
- [6] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fu-

sai, M. Y. Galliker, et al. π0.5: A vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025.

- [7] NVIDIA, J. Bjorck, N. C. Fernando Casta˜neda, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. GR00T N1: An open foundation model for generalist humanoid robots. In ArXiv Preprint, March 2025.
- [8] K. Shaw, A. Agarwal, and D. Pathak. Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning. Robotics: Science and Systems (RSS), 2023.
- [9] C. C. Christoph, M. Eberlein, F. Katsimalis, A. Roberti, A. Sympetheros, M. R. Vogt, D. Liconti, C. Yang, B. G. Cangan, R. J. Hinchet, et al. Orca: An open-source, reliable, cost-effective, anthropomorphic robotic hand for uninterrupted dexterous task learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8503–8510. IEEE, 2025.
- [10] B. Romero, H.-S. Fang, P. Agrawal, and E. Adelson. Eyesight hand: Design of a fully-actuated dexterous robot hand with integrated vision-based tactile sensors and compliant actuation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1853–1860. IEEE, 2024.
- [11] R. Zheng, D. Niu, Y. Xie, J. Wang, M. Xu, Y. Jiang, F. Casta˜neda, F. Hu, Y. L. Tan, L. Fu, et al. Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026.
- [12] L. Heng, H. Geng, K. Zhang, P. Abbeel, and J. Malik. Vitacformer: Learning cross-modal representation for visuo-tactile dexterous manipulation, 2025. URL https://arxiv.org/ abs/2506.15953.
- [13] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

- [14] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791, 2023.
- [15] Y. Mu, T. Chen, Z. Chen, S. Peng, Z. Lan, Z. Gao, Z. Liang, Q. Yu, Y. Zou, M. Xu, L. Lin, Z. Xie, M. Ding, and P. Luo. Robotwin: Dual-arm robot benchmark with generative digital twins. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 27649–27660, June 2025.
- [16] Z. Jiang, Y. Xie, K. Lin, Z. Xu, W. Wan, A. Mandlekar, L. J. Fan, and Y. Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16923–16930. IEEE, 2025.
- [17] Y. Chen, Y. Yang, T. Wu, S. Wang, X. Feng, J. Jiang, Z. Lu, S. M. McAleer, H. Dong, and S.-C. Zhu. Towards human-level bimanual dexterous manipulation with reinforcement learning. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/forum?id=D29JbExncTP.
- [18] J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. Huang, R. Chen, and H. Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023.
- [19] R. McLean, E. Chatzaroulas, L. McCutcheon, F. R¨oder, T. Yu, Z. He, K. Zentner, R. Julian, J. K. Terry, I. Woungang, N. Farsad, and P. S. Castro. Meta-world+: An improved, standardized, RL benchmark. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum? id=1de3azE606.
- [20] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. In 7th Annual Conference on Robot Learning, 2023.
- [21] T. Mu, Z. Ling, F. Xiang, D. C. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).
- [22] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024.
- [23] S. Nasiriany, S. Nasiriany, A. Maddukuri, and Y. Zhu. Robocasa365: A large-scale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations (ICLR), 2026.
- [24] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In arXiv preprint arXiv:2108.03298, 2021.
- [25] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T. kai Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y.-R. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.
- [26] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, et al. Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4737–4746, 2023.

- [27] W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang. Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalistspecialist learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3891–3902, 2023.
- [28] K. Zhu, F. Bai, Y. Xiang, Y. Cai, X. Chen, R. Li, X. Wang, H. Dong, Y. Yang, X. Fan, et al. Dexflywheel: A scalable and self-improving data generation framework for dexterous manipulation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.
- [29] S. Luo, Q. Peng, J. Lv, K. Hong, K. R. Driggs-Campbell, C. Lu, and Y.-L. Li. Human-agent joint learning for efficient robot manipulation skill acquisition. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1370–1377. IEEE, 2025.
- [30] P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel. Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12156–12163. IEEE, 2024.
- [31] J. J. Liu, Y. Li, K. Shaw, T. Tao, R. Salakhutdinov, and D. Pathak. Factr: Force-attending curriculum training for contact-rich policy learning. arXiv preprint arXiv:2502.17432, 2025.
- [32] Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox. Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system. In Robotics: Science and Systems, 2023.
- [33] K. Shaw, S. Bahl, A. Sivakumar, A. Kannan, and D. Pathak. Learning dexterity from human hand motion in internet videos. The International Journal of Robotics Research, 43(4):513–532, 2024.
- [34] R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang. Bunny-visionpro: Realtime bimanual dexterous teleoperation for imitation learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12248–12255. IEEE, 2025.
- [35] A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto. Open teach: A versatile teleoperation system for robotic manipulation. In CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data.
- [36] G. Zhang, Q. Xu, H. Zhang, J. Ma, L. He, Y. Bao, Z. Ping, Z. Yuan, C. Lu, C. Yuan, et al. Unidex: A robot foundation suite for universal dexterous hand control from egocentric human videos. arXiv preprint arXiv:2603.22264, 2026.
- [37] H. Zhang, S. Hu, Z. Yuan, and H. Xu. Doglove: Dexterous manipulation with a low-cost open-source haptic force feedback glove. arXiv preprint arXiv:2502.07730, 2025.
- [38] Z.-H. Yin, C. Wang, L. Pineda, K. Bodduluri, T. Wu, P. Abbeel, and M. Mukadam. Geometric retargeting: A principled, ultrafast neural hand retargeting algorithm. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 17376–17382. IEEE, 2025.
- [39] M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025.
- [40] R. Wen, G. Chen, Z. Cui, M. Du, Y. Gou, Z. Han, L. Huang, M. Lei, Y. Li, Z. Li, et al. Gr-dexter technical report. arXiv preprint arXiv:2512.24210, 2025.
- [41] C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and C. K. Liu. Dexcap: Scalable and portable mocap data collection system for dexterous manipulation. arXiv preprint arXiv:2403.07788, 2024.

- [42] Y. Gao, H. Ma, and P. Zheng. Glovity: Learning dexterous contact-rich manipulation via spatial wrench feedback teleoperation system. arXiv preprint arXiv:2510.09229, 2025.
- [43] H.-S. Fang, B. Romero, Y. Xie, A. Hu, B.-R. Huang, J. Alvarez, M. Kim, G. Margolis, K. Anbarasu, M. Tomizuka, E. Adelson, and P. Agrawal. Dexop: A device for robotic transfer of dexterous human manipulation. arXiv preprint arXiv:2509.04441, 2025.
- [44] Y. Feng, H. Fang, Y. He, J. Chen, C. Wang, Z. He, R. Liu, and C. Lu. Learning dexterous manipulation with quantized hand state. arXiv preprint arXiv:2509.17450, 2025.
- [45] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ın-Mart´ın, A. Joshi, K. Lin, S. Nasiriany, and Y. Zhu. robosuite: A modular simulation framework and benchmark for robot learning. In arXiv preprint arXiv:2009.12293, 2020.
- [46] K. Zakka, Y. Tassa, and MuJoCo Menagerie Contributors. MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo, 2022. URL http://github.com/ google-deepmind/mujoco_menagerie.
- [47] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su. SAPIEN: A simulated part-based interactive environment. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
- [48] T. H. Team. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation, 2024.
- [49] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.
- [50] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.
- [51] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.
- [52] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.
- [53] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Appendix

- A Statistical Analysis for Language Generalization Results

[Figure 81]

1 2 3 4 Other No input Actual input

- 1
- 2

4

- 1+1
- 2+2

two

one plus one

Instruction

0.153 0.327 0.213 0.073 0.053 0.180

0.067 0.307 0.267 0.053 0.033 0.273

0.113 0.273 0.307 0.040 0.093 0.173

0.053 0.247 0.313 0.027 0.027 0.333

0.067 0.267 0.380 0.013 0.027 0.247

0.133 0.300 0.260 0.053 0.060 0.193

0.080 0.207 0.340 0.020 0.040 0.313

[Figure 82]

0.00

0.05

0.10

0.15

0.20

0.25

0.30

0.35

0.40

Probability

- Figure 7: Output distribution of π0.5 (trained on single digits 1-5) across instructions on the Unlock iPad.

As shown in Fig. 7, the policy exhibits severe mode collapse. While certain unseen prompts like “two” (30.0%±5.3) and “1+1” (24.7%±10.3) appear to yield moderate precision, the heatmap reveals this to be a statistical illusion caused by the model’s prior bias. Specifically, the probability of outputting “2” remains nearly constant ( 30%) even when the correct answer is “1” or “4”. This lack of language conditioning is further evidenced by the model’s failure on the seen digit “4”, where precision drops to only 4.0%±2.0 because the model stubbornly outputs “2” (0.273) or “3” (0.307) instead of the requested digit. Quantitatively, although a chi-square test rejects the hypothesis of strict independence (p = 2.15 × 10−4), confirming that the VLA does react to varying language instructions, the Normalized Mutual Information between instruction and output is only 0.018, indicating a negligible relationship. The average JS divergence across all pairs of

instructions is 0.026, with a maximum of 0.057 (between “1” and “1+1”), further demonstrating that the policy’s action distribution remains nearly identical regardless of the prompt. We therefore conclude that the model fails to achieve true language generalization. The average precision (%) ± std across 3 seeds is “1”: 15.3%±5.8; “2”: 30.7%±12.7; “4”: 4.0%±2.0; “1+1”: 24.7%±10.3; “2+2”: 1.3%±1.2; “two”: 30.0%±5.3; “one plus one”: 20.7%±2.3.

Table 4: Detailed Language Instruction of Language Generalization Experiment

Password Language Instruction

- “1” “Grasp the iPad and enter the password 1 to unlock the device.”
- “2” “Grasp the iPad and enter the password 2 to unlock the device.”
- “3” “Grasp the iPad and enter the password 3 to unlock the device.”
- “4” “Grasp the iPad and enter the password 4 to unlock the device.”
- “5” “Grasp the iPad and enter the password 5 to unlock the device.”

- “1+1” “Grasp the iPad and enter the result of 1+1 as the password to unlock the device.”
- “2+2” “Grasp the iPad and enter the result of 2+2 as the password to unlock the device.” “two” “Grasp the iPad and enter the password two to unlock the device.”

“one plus one” “Grasp the iPad and enter the result of one plus one as the password to unlock the device.”

- B Visualization and Language Instruction of DexJoCo Tasks

Table 5: Visualization and language instruction of DexJoCo tasks.

Task Visualization Language Instruction

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Use the hammer to drive the nail into the wooden board.

Hammer Nail

Continued on next page

Move the mouse to the purple mouse pad and click the left mouse button.

Click Mouse

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Place the boxed food into the bucket and then lift the bucket.

Pick Bucket

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Grasp the tongs and perform three consecutive open-close motions.

Pinch Tongs

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Fold the glasses and place them into the case.

Fold Glasses

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Grasp the watering can and apply water to the plant.

Water Plant

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Grasp the iPad and enter the password 123 to unlock the device.

Unlock iPad /B

Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Hanoi /B

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.

Assembly /B

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Open the microwave door, place the food inside the microwave, close the door, and press the start button.

Microwave /B

Continued on next page

Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand.

Photograph /B

### C Randomization Settings of DexJoCo Tasks

Default Domain randomizations

|[Figure 127]|
|---|

|[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]|
|---|

Table height Lighting direction and color

Third-person camera pose

Table texture

- Figure 8: Domain randomization settings. The left panel shows the default scene configuration, while the right panel illustrates the effects of domain randomization, including variations in table height, third-person camera viewpoints, lighting conditions, and tabletop textures.

###### Randomized Camera Poses on Spherical Shells - Iso View

Single-arm (n=50)

Dual-arm (n=50)

Scene center

- 0

- 1

- 2

- 3

z

1

2

2

1

1

0

0

y

1

1

x

2

2

Figure 9: Preset third-person camera poses used for visual randomization.

The visual randomization protocol is shared by all 11 task environments. At reset time, each environment samples one preset third-person camera pose from the replay-camera pool, randomly selects a tabletop texture from the texture library, and perturbs scene lighting. Specifically, each light position is perturbed in the x and y axes by U(−0.3,0.3), each light direction is perturbed in the x and

y axes by U(−0.4,0.4), light diffuse RGB values are sampled from U(0.3,0.8), headlight ambient RGB values are sampled from U(0.3,0.7), and headlight diffuse RGB values are sampled from U(0.2,0.6). Fig. 9 visualizes the preset camera-pose pool used for third-person view randomization.

Table-height randomization is also shared across all task environments. At reset time, the table height offset is sampled as ∆h ∼ U(0,0.05) m, and task-relevant object heights are shifted consistently with this offset.

Table 6: Task-specific randomization settings for the 11 DexJoCo task environments. Object placement bounds are reported as planar (x,y) sampling ranges following the corresponding environment implementation; shared visual and table-height randomization are described above.

Task Object randomization Dynamics randomization Hammer Nail Hammer (x, y): low [−0.25, −0.35], high [−0.40, −0.50];

Hammer mass multiplier ∼ U(0.75, 1.25).

yaw ∼ U(−10◦, 10◦). Nail (x, y): low [−0.10, 0.00], high [0.00, 0.10].

Click Mouse Mouse (x, y): low [−0.20, 0.00], high [−0.25, 0.05]; yaw ∼ U(−10◦, 10◦). Monitor/mouse-pad target (x, y): fixed at [0.12, 0.30].

Mouse mass multiplier ∼ U(0.75, 1.25).

Pick Bucket Bucket (x, y): low [−0.20, −0.20], high [−0.15, −0.25]; yaw ∼ U(−10◦, 10◦). Boxed food (x, y): low [−0.35, 0.15], high [−0.30, 0.20]; yaw ∼ U(−10◦, 10◦).

Bucket joint friction multiplier ∼ U(0.75, 1.25). Bucket and boxed-food mass multipliers ∼ U(0.75, 1.25).

Pinch Tongs Tongs (x, y): low [−0.35, −0.25], high [−0.30, −0.20]. Tongs joint friction loss ∼ U(0, 0.05). Joint stiffness multiplier ∼ U(0.75, 1.25). Tongs mass multiplier ∼ U(0.75, 1.25).

Fold Glasses Glasses (x, y): low [−0.40, −0.225], high [−0.35, −0.175]; yaw ∼ U(−10◦, 10◦). Storage box (x, y): low [−0.275, 0.25], high [−0.225, 0.30]; yaw ∼ U(−10◦, 10◦).

Glasses joint friction loss ∼ U(0, 0.05). Joint stiffness multiplier ∼ U(1.0, 1.5). Glasses mass multiplier ∼ U(0.75, 1.25).

Water Plant Spray bottle (x, y): from [−0.35, −0.25] to [−0.30, −0.20]. Plant (x, y): from [−0.10, 0.15] to [−0.05, 0.20].

Spray joint friction loss ∼ U(0, 0.05). Joint stiffness multiplier ∼ U(0.75, 1.25). Spray body mass multiplier ∼ U(0.75, 1.25).

Unlock iPad /B iPad stand (x, y): low [−0.35, 0.05], high [−0.30, 0.10]. iPad and stand heights are shifted by the shared height offset.

iPad mass multiplier ∼ U(0.75, 1.25).

Each disk mass multiplier ∼ U(0.75, 1.25).

Hanoi /B Hanoi base (x, y): low [−0.25, 0.00], high [−0.20, 0.00]. All disks are translated consistently with the base and shared height offset.

Peg and socket/tray mass multipliers ∼ U(0.75, 1.25).

Assembly /B Peg (x, y): from [−0.30, −0.25] to [−0.25, −0.20]; yaw ∼ U(−10◦, 10◦). Socket/tray (x, y): from [−0.30, 0.15] to [−0.20, 0.25]; yaw ∼ U(−20◦, 20◦).

Microwave /B Hot dog (x, y): low [−0.35, −0.30], high [−0.25, −0.40]. Hot dog yaw ∼ U(−20◦, 20◦).

Microwave joint friction multiplier ∼ U(0.75, 1.25). Hot dog and plate mass multipliers ∼ U(0.75, 1.25).

Photograph /B Logo (y, z): from [−0.10, 1.22] to [0.10, 1.38]. Camera (x, y): from [−0.30, 0.10] to [−0.20, 0.20].

Camera mass multiplier ∼ U(0.75, 1.25).

