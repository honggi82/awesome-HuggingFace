[Figure 1]

#### ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks

##### Kaijun Wang1*, Liqin Lu2*, Mingyu Liu1, Jianuo Jiang3, Zeju Li1, Bolin Zhang1, Wancai Zheng2, Xinyi Yu2†, Hao Chen1†, Chunhua Shen1†

1Zhejiang University 2Zhejiang University of Technology 3The Chinese University of Hong Kong, Shenzhen

[Figure 2]

[Figure 3]

## arXiv:2508.08240v1[cs.RO]11Aug2025

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

###### Training

[Figure 18]

Generalize to Challenging Terrains

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Unified Policy

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Scene Graph Planner

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Hierarchical Task Planner

Whole-Body Control

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Benchmark

Sim2Real

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

58 OutDoor Tasks

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

100 + Assets

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Sim Real

246 InDoor Tasks

Figure 1. We present ODYSSEY, a unified mobile manipulation framework for agile quadruped robots equipped with manipulators, which seamlessly integrates high-level task planning with low-level whole-body control.

###### Abstract

derstudied in the literature.

In this work, we present ODYSSEY, a unified mobile manipulation framework for agile quadruped robots equipped with manipulators, which seamlessly integrates high-level task planning with low-level whole-body control. To address the challenge of egocentric perception in languageconditioned tasks, we introduce a hierarchical planner powered by a vision-language model, enabling long-horizon instruction decomposition and precise action execution. At the control level, our novel whole-body policy achieves robust coordination of locomotion and manipulation across challenging terrains. We further present the first comprehensive benchmark for long-horizon mobile manipulation, evaluating diverse indoor and outdoor scenarios. Through successful sim-to-real transfer, we demonstrate the system’s generalization and robustness in real-world deployments, underscoring the practicality of legged manipulators in unstructured environments. Our work advances the feasibility of generalized robotic assistants capable of complex, dynamic tasks. Our

Language-guided long-horizon mobile manipulation has long been a grand challenge in embodied semantic reasoning, generalizable manipulation, and adaptive locomotion. Three fundamental limitations hinder progress: First, although large language models have shown promise in enhancing spatial reasoning and task planning through learned semantic priors, existing implementations remain confined to tabletop scenarios, failing to address the constrained perception and limited actuation ranges characteristic of mobile platforms. Second, current manipulation strategies exhibit insufficient generalization when confronted with the diverse object configurations encountered in open-world environments. Third, while crucial for practical deployment, the dual requirement of maintaining high platform maneuverability alongside precise end-effector control in unstructured settings remains un-

*These authors contributed equally. †Corresponding authors.

project page: https://kaijwang.github.io/odyssey.github.io/

##### 1 Introduction

Open-world mobile manipulation allows robots to autonomously navigate and interact in dynamic, unstructured environments by tightly integrating mobility, manipulation, and real-time perception. Unlike traditional methods that separate navigation and manipulation, this unified approach unlocks emergent capabilities like active perception, which is essential for real-world tasks. For example, a robot might subtly adjust its position while reaching for an object to gain a better grasp position. This adaptive behavior emerges naturally in complex environments where static perception or sequential planning would fail.

Prior research has achieved robust solutions for navigation in dynamic environments (Grandia et al. 2023; Zhuang et al. 2023; Liu et al. 2024b) and manipulation in controlled settings (Kim et al. 2024; Brohan et al. 2022; Cheang et al. 2024). While recent works (Pan et al. 2025a; Fu, Cheng, and Pathak 2023; Liu et al. 2024a; Zhang et al. 2025; Wang et al. 2024b) have developed whole-body control frameworks, they face scalability limitations in open-world scenarios due to simplified environmental assumptions and evaluations restricted to short-horizon tasks. We present ODYSSEY, a reinforcement learning-based whole-body control system that unifies robust quadruped locomotion with precise manipulation through an integrated vision-language framework. Our approach achieves state-of-the-art control accuracy under out-of-distribution conditions while significantly expanding operational capabilities. Unlike prior work, we demonstrate generalization across diverse challenging terrains, enabling real-world deployment.

Recent works (Qi et al. 2025; Pan et al. 2025b) demonstrate that large language models can substantially enhance robotic task planning and generalizable manipulation through their spatial understanding abilities. We extend the capability of large language models to whole-body navigation and manipulation tasks. Specifically, our approach grounds task execution at two levels: task-level planning over a semantic instance map and fine-grained action guidance via geometry-constrained pose estimation.

Besides, to address the critical evaluation gap, we present the first comprehensive benchmark for evaluating longhorizon mobile manipulation, featuring eight diverse daily tasks across indoor/outdoor environments with hundreds of object configurations. The benchmark enables holistic assessment of embodied reasoning, task planning, navigation, and manipulation capabilities, while incorporating the standardized Arnold framework for precise manipulation evaluation.

Through extensive experiments, our system shows strong ability for sim2real transfer, showcasing exceptional generalization where both control and planning modules maintain consistent performance across diverse real-world scenarios. Our contributions are four folds:

(i) We introduce a hierarchical vision-language planner that bridges egocentric perception and language-conditioned

tasks, decomposing long-horizon instructions into executable actions.

- (ii) We propose the first whole-body control policy that

generalizes to challenging terrains while jointly coordinating locomobility and manipulation.

- (iii) We introduce the first long-horizon mobile manipu-

lation benchmark, covering a wide range of realistic indoor and outdoor scenarios.

- (iv) We further demonstrate successful sim-to-real trans-

fer of both high-level planners and low-level control policies, showing strong generalization and robustness in realworld deployments.

Our results highlight the feasibility and practicality of deploying a legged mobile manipulator in unstructured environments, paving the way toward generalized robotic assistants.

##### 2 Related Work

###### 2.1 Open-world mobile manipulation

Previous research has made significant advances in both navigation and manipulation as separate domains, with robust solutions developed for mobile robot path planning in dynamic environments (Grandia et al. 2023; Zhuang et al.

- 2023) and sophisticated manipulation techniques for object interaction in controlled settings (Kim et al. 2024; Brohan et al. 2022; Cheang et al. 2024). While pioneering works (Pan et al. 2025a; Fu, Cheng, and Pathak 2023; Liu et al. 2024a; Zhang et al. 2025; Wang et al. 2024b; Fu, Zhao, and Finn 2024; Jiang et al. 2025b) have developed initial whole-body control frameworks, they face scalability limitations in open-world scenarios due to oversimplified environmental assumptions and evaluations limited to short-horizon pick-and-place tasks. Some works (Ha et al. 2024; Qiu et al.
- 2024b) have attempted to attach more complex interactions by learning per-action policies from human demonstrations. However, these methods lack compositionality and scalability, needing task-specific data for each scenario. ODYSSEY overcomes these limitations by unifying terrain-aware locomotion with hierarchical planning, enabling robust mobile manipulation in unstructured environments.

###### 2.2 Foundation Models for Embodied Tasks

Vision-language models (VLMs) have shown promise in enhancing robotic reasoning (Qi et al. 2025; Pan et al. 2025b; Zhi et al. 2024; Qiu et al. 2024a; Wang et al. 2025), but their evaluation has been restricted to tabletop settings with fixed cameras. For navigation, foundation models improve spatial understanding (Gu et al. 2024; Jatavallabhula et al. 2023; Jiang et al. 2025a), yet they lack fine-grained manipulation support. ODYSSEY advances this by grounding hierarchical planning in egocentric perception, using VLMs to decompose tasks via scene graphs while generating precise endeffector trajectories. This contrasts with modular approaches (Zhang et al. 2025) that struggle with compositional reasoning under uncertainty.

###### 2.3 Benchmarks for Real-World Deployment

Existing benchmarks for mobile manipulation (Qiu et al. 2024a) focus narrowly on navigation or short-term interactions, lacking standardized metrics for long-horizon tasks. Simulation frameworks like IsaacSim have enabled progress in locomotion (Zhi et al. 2024), but manipulation-centric evaluations remain sparse. ODYSSEY introduces a comprehensive benchmark with diverse indoor/outdoor scenarios, addressing multi-stage reasoning, object-aware navigation, and precision manipulation. This complements prior work (Qiu et al. 2024b) by enabling scalable testing of sim-to-real transfer for integrated control and planning.

3 Method

In this section, we present ODYSSEY, a unified framework that spans long-horizon task planning, whole-body control, and standardized evaluation for mobile manipulation. It comprises three key components:

- 1. Coarse-to-Fine Task Planner (Section 3.1): a hierarchical planner that orchestrates top-down task execution under the guidance of foundation models.
- 2. Quadruped Whole-Body Policy (Section 3.2): a reinforcement learning-based whole-body controller that generalizes to diverse terrains and overcomes sim-to-real gap.
- 3. Mobile Manipulation Benchmark (Section 3.3): the first scalable evaluation suite for assessing long-term task performance across versatile real-world scenarios.

###### 3.1 Long-horizon Task Planner

To bridge the gap left by prior work in modeling the intricate dependencies between semantic reasoning-based navigation and fine-grained, generalizable manipulation, our hierarchical framework is explicitly designed to ensure the reliability of both components while reinforcing their mutual dependencies for coherent long-horizon task execution.

Map-Aware Task-Level Planning To support longhorizon task planning grounded in egocentric observations, we first build a global planner that integrates a lightweight multi-modal perception module as a plug-in component. Concretely, we fuse on-board RGB and LiDAR streams to form a unified spatial-semantic representation of the scene. Leveraging a suite of pre-trained foundation models, we map an instance graph that encodes object geometry and semantics for symbolic task reasoning. The pipeline of this module is detailed in Appendix A.1.

As illustrated in Fig. 2, given the instance-level semantic map, GPT-4.1 (Achiam et al. 2023) is used to break down template-free natural language instructions into a sequence of atomic actions from a predefined set: navigate, pick, place, and push/pull/drag. Each action is paired with a language description that tracks task progress and provides guidance for local planning.

For actions involving spatial displacement (navigate, drag), the model is further prompted to output a coarse target waypoint to guide planning. We project this target

onto a 2D occupancy map built via online SLAM from accumulated LiDAR scans. A local search is then performed around the projected waypoint to identify a collision-free goal pose, avoiding object bounding boxes and structural obstacles. This process yields a globally grounded task plan that aligns with the scene context and is feasible under physical constraints.

Geometry-Constrained Local Manipulation For atomic actions requiring close-range manipulation, we use wristmounted depth observations to guide a vision-language model for precise end-effector pose generation. Despite the diverse physical nature of different actions, we unify their execution through a single visuomotor interface, eliminating the need for per-action heuristics.

Specifically, given an RGB observation and the corresponding textual description of the current atomic action, we employ Qwen2.5-VL-72B-Instruct (Bai et al. 2025), a model enhanced with pixel-level grounding capabilities, to infer a task-relevant contact point p∗ ∈ R2 in the image space.

The contact point is projected onto the aligned depth image to recover its corresponding 3D position in the robot coordinate frame, denoted as Pee ∈ R3. We further prompt the model to generate the orientation Ree ∈ SO(3) of the endeffector by determining the gripper’s closing direction(xaxis) and approaching direction(z-axis), subject to the following geometric constraints:

- • Axis-alignment constraint: When the target object or contact region exhibits a dominant axis a ∈ R3, both the x-axis and z-axis of the end-effector should be orthogonal to it:

r⊤x a = 0, r⊤z a = 0. (1)

- • Surface-normal constraint: If the object is attached to a planar surface with normal vector n ∈ R3, then the z-axis of the end-effector should align with the surface normal without violating the first constraint:

rz ∥ n, s.t. r⊤z a = 0. (2)

By leveraging the expressive grounding capacity of Qwen-VL and constraining the output pose with interpretable geometric conditions, our system achieves reliable local guidance for interaction-intensive manipulation primitives. To the best of our knowledge, this constitutes the first fine-grained manipulation planning system without thirdperson observation or scripted policies, marking a significant step toward scalable deployment in mobile, in-the-wild environments.

###### 3.2 Policy for Whole-body Control

To effectively execute commands from a high-level planner and adapt to diverse terrains, a whole-body control policy is essential. This work proposes a two-stage, learning-based policy that utilizes a neural network to generate desired joint positions from a set of observations. To enhance the policy’s robustness, the training process incorporates a carefully designed, terrain-invariant end-effector sampling strategy and

Please clear out the toys in the bedroom.

Atomic Actions Nav

Map-aware Planning

[Figure 99]

➁

Push

[Figure 100]

InstanceGraph

Velocity Commands

➀

➃

Pull Drag

➂

[Figure 101]

[Figure 102]

Pick Place

Proprioception

[Figure 103]

[Figure 104]

[Figure 105]

π

Geometry-Constrained

...

Global Planner

[Figure 106]

[Figure 107]

[Figure 108]

Approach Closing

[Figure 109]

Arm Camera RGBD

[Figure 110]

EE Pose Commands

[Figure 111]

[Figure 112]

Local Planner

Longaxis

[Figure 113]

➀ Pick

[Figure 114]

➁ Place

[Figure 115]

➂ Drag

[Figure 116]

➃ Goal

- Figure 2: ODYSSEY pipeline spans the entire process of a long-horizon task, including multi-modal semantic perception, mapaware global planning, geometry-constrained action grounding, and step-wise execution by a learned low-level policy.

comprehensive domain randomization. The resulting controller is resilient to varied environmental interactions and can be directly deployed on a physical robot. This section first defines the policy and subsequently discusses the training methodology.

Mobile Manipulation Policy The mobile manipulation policy π is formulated as a single network that maps a comprehensive observation vector to a target action at ∈ R18 as shown in Eq.4. The observation includes the locomotive command ct = (ˆx,y,ˆ ωˆ), the 6-D end-effect target et, a local ground height map mt, the projected gravity vector gt, the previous timestep at−1 ∈ R18 and the proprioceptive state st ∈ R36 (joint positions qt and velocities ˙qt). All commands and targets are expressed in the robot’s base frame. To stabilize the policy output and reduce the simulation-toreality gap (Fu, Cheng, and Pathak 2023), the action at is formulated as the offsets to the default joint configuration qdefault ∈ R18. The final target, qtargett = qdefault + at, is then converted to torques by a Proportional-Derivative (PD) controller.

st = (qt, ˙q) (3) at = π(ct,et,st,gt,mt,at−1) (4)

To enhance training robustness and avoid local minima associated with searching a large action space, we employ a two-stage curriculum learning approach, as shown in Fig. 3. Stage 1 In this stage, the arm joints are fixed to focus training on locomotion under a static load, improving exploration efficiency. Inspired by (Mittal et al. 2023), we introduce a gait reward incorporated alongside the base tracking reward to structure the robot’s gait. Furthermore, a novel frequency reward is introduced to regulate the gait’s cadence. The gait reward rgait encourages specific synchronous (e.g., diagonal) and asynchronous (e.g., lateral) foot contact patterns, with reward functions rs and ra detailed in the Appendix B.1.

ra(k,l) (5)

###### rs(i,j) ·

rgait =

i,j∈sync pairs

k,l∈async pairs

The frequency reward rfre regulates the gait’s cadence based on the error from a target frequency ftarget. The gait frequency f(leg) is the inverse of the time between consecutive ground contacts (tcontk − tcontk−1). The reward is then:

err(leg) = (f(leg) − ftarget)2 (6) rf(leg) = exp(−0.5 · err(leg)) (7)

rf(leg) (8)

rfre =

leg∈FL,FR,RL,RR

Stage 2 Following 2k training iterations, the process transitions to the second stage. In this stage, the policy controls all 18 joints, encompassing both the manipulator and the four legs. Consequently, the reward function is expanded to include end-effector tracking terms, rarm, in addition to the previously described locomotion rewards, to guide the policy’s training.

Terrain-Invariant End-Effector Sampling To ensure robust performance across varied topographies, our method employs a terrain-invariant end-effector sampling strategy. The process begins by sampling a target position from a spherical volume defined in the world coordinate system, centered at the robot’s arm base. A key aspect of this strategy is that the target’s z-axis height is fixed within the world frame before the coordinates are transformed into a Cartesian target position relative to the robot’s moving base frame. This approach offers a significant advantage over sampling directly in the arm’s local frame, as it effectively decouples the end-effector target from disturbances caused by changes in the robot’s base pitch or the underlying terrain height. Consequently, this decoupling improves interaction accuracy during task execution.

Domain Randomizatoin To bridge the simulation-to-reality gap, domain randomization is employed throughout the training process, a strategy supported by recent research (Fu, Cheng, and Pathak 2023; Pan et al. 2025a). To ensure adaptability to different payloads, the end-effector’s mass is also randomized during training, improving the policy’s ability to handle objects of unknown weight. A detailed

Terrain-Invariant EE Sample Height Scan

|[Figure 117]<br><br>Rough|[Figure 118]<br><br>Bricks|
|---|---|
|[Figure 119]<br><br>Stairs|[Figure 120]<br><br>Pit|

[Figure 121]

38 cm

Joint States & Actions Gravity Vector

Velocity Sample

Commands Proprioception Privileged info

###### Unified Policy

[Figure 122]

[Figure 123]

Stage 1 Stage 2

2k Iterations

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- Figure 3: An overview of the mobile manipulator policy and its two-stage training framework.

breakdown of all randomization parameters and the key reward components is provided in the Appendix B.2.

###### 3.3 Simulation Benchmark

To evaluate navigation, manipulation, and whole-body control as a unified system, we present the first simulation benchmark tailored for long-horizon mobile manipulation in both indoor and outdoor environments.

Asset and Scene Library To support realistic and versatile evaluation environments, we curate a diverse set of assets encompassing both object instances and full-scale 3D scenes. The object assets are sourced from a combination of prior open-source datasets (Wang et al. 2024a; Team 2025; Nasiriany et al. 2024), publicly available object repositories, and manually created models.

Object Assets: We curate a diverse set of interactive objects categorized into four types: 50 rigid objects (e.g., common graspable items), 15 containers (e.g., bowls and bins with annotated containment region), 30 articulated structures (e.g., cabinets and doors), and 10 draggable items (e.g., carts and chairs).

Environments: Our benchmark includes 10 realistic

scenes, with 5 indoor homes, 2 supermarkets, 1 restaurant, and 2 outdoor courtyards featuring slopes and stairs. All environments are designed for full traversability by legged robots and support multiple initialization zones to allow sampling and spatial variation of large-scale tasks.

Rich Domain-style Variation To ensure generalization, we incorporate variability across four dimensions during simulation rollout: (1) Object layouts are varied within semantic constraints across episodes, promoting diversity in interaction. (2) Physical attributes, including mass, friction, and articulation limits, are resampled per episode to induce dynamic variability. (3) Environment conditions such as lighting, material textures, and clutter elements are randomized to simulate perceptual noise. (4) Terrain complexity is varied across outdoor scenes to assess locomotion robustness.

Multi-stage Task Suite Our benchmark includes two categories of tasks: short-horizon manipulation skills merged from ARNOLD (Gong et al. 2023), and long-horizon mobile manipulation tasks designed to reflect practical daily scenarios.

Short-Horizon Arnold Tasks. We integrate four singlestep manipulation tasks from the ARNOLD benchmark: PICKUPOBJECT, REORIENTOBJECT, OPENCABINET, and CLOSECABINET. While retaining their original goal state definitions and scene configurations, we adjust the spatial layout and object positioning to accommodate the kinematics and workspace of our quadruped robot platform, ensuring fair and consistent evaluation.

Long-Horizon Mobile Manipulation. To assess the system’s embodied reasoning, navigation, and sequential manipulation capabilities, we construct 8 multi-stage tasks spanning diverse indoor and outdoor scenarios. Each task consists of 2–3 subgoals, with a total of 246 indoor and 58 outdoor variations spanning object types, spatial layouts, and interaction modes.

Our task pool emphasizes a broad range of skills spanning grasping, reorientation, container placement, articulated manipulation, and long-term navigation over complex terrain. The combination of short and long-horizon tasks enables benchmarking at both low-level manipulation and high-level planning. Detailed task configurations are elaborated in Appendix C.

Modular Evaluation Protocol We evaluate both overall task success and per-action success rate. For instance, in the CARTDELIVERY task, we define subtasks such as nav to object, pick object, nav to cart, place object, drag cart, and nav to goal. We determine actions success by monitoring the robot and cart’s world poses, as well as the relative pose between the object and the cart. A subtask is considered complete if its corresponding goal condition is met during the task horizon. This protocol captures both execution precision and planning consistency.

##### 4 Experiment

- 4.1 High-level Planner Performance To evaluate the performance of our high-level planner in a modular and scalable manner, we conducted experiments based on the benchmark discussed in Section 3.3. Firstly, we tested our local planner on thousands of single-step test cases, focusing on precision and consistency. Secondly, we integrated the global planner and evaluated our proposed approach on several hundred long-horizon mobile manipulation tasks. Additionally, we performed a detailed analysis of the completion rates for the decomposed atomic actions within each task.

ARNOLD Short-horizon Tasks Before moving on to the long-horizon evaluation, we first conducted experiments in relatively confined spaces to demonstrate the fine-grained operation precision and generalization capabilities of our framework. We migrated four short-horizon tasks from ARNOLD and faithfully replicated their continuous monitoring system for goal states.

Their evaluation protocol divides five splits into two categories: 1) Seen includes shuffled seen data; 2) Novel features one of unseen components (objects, scenes, or goal states). We compare against their strongest baseline model PerAct (Shridhar, Manuelli, and Fox 2022), an end-to-end imitation learning paradigm trained on large-scale human trajectories, which leverages observations from five external cameras to achieve accurate spatial perception. As shown in Table 1, our method achieves substantial overall improvements, demonstrating superior fine-grained manipulation capabilities while relying solely on a single egocentric camera. Moreover, while their performance declined dramatically on novel splits, our method maintains stable performance across all datasets, showcasing generalized ability to handle O.O.D object configurations. Further experiment details are provided in Appendix D.1.

###### Seen Novel

PerAct Ours PerAct Ours

P.OBJECT 94.03 60.45 25.70 45.24 R.OBJECT 19.48 51.32 8.23 52.09

- O.CABINET 31.09 56.30 16.62 51.09 C.CABINET 60.81 74.32 41.32 79.50

Table 1: Performance comparison between our approach and PerAct on 4 ARNOLD tasks, evaluated on both seen and generalized splits by success rate (%).

ODYSSEY Long-horizon Tasks Table 2 summarizes the performance of our system across eight long-horizon mobile manipulation tasks, reporting both overall task success rates and success rates for decomposed atomic actions. Notably, ODYSSEY consistently achieves 40% or higher overall success across all tasks, and maintains over 60% success in each atomic skill category, demonstrating robust coordination in a generalized long-horizon task. Building on this, we highlight several key findings from different perspectives of system performance:

Low-level ability: The consistent success rates across indoor and outdoor settings, despite the presence of irregular terrains, validate the reliable locomotion and effective pose tracking enabled by our terrain-adaptive whole-body control policy. Most control-related failures are caused by interactions with objects positioned beyond the robot’s reachable range.

Fine-grained action: VLM-guided grounding enables high pick and place success rates across all tasks, demonstrating strong capability in identifying and localizing semantic targets. A large portion of failures stem from suboptimal gripper alignment, indicating limitations in the model’s spatial reasoning over object geometry. In addition, tasks involving more complex interactions such as drag and pull occasionally fail due to inaccurate localization, particularly with slim handles or partially occluded items.

Task-level planning: Our global task planner demonstrates strong symbolic reasoning over instance graphs, enabling reliable multi-stage task decomposition. In conjunction, our SLAM-based path planner ensures safe and consistent navigation. These components together lead to high navigate success rates across all tasks.

###### 4.2 Low-level Policy Performance

We evaluated the proposed whole-body control policy against RoboDuet (Pan et al. 2025a), a baseline that also uses a two-stage training process. In contrast to RoboDuet’s dual-policy (locomotion and manipulation) approach with base-centric sampling, our method employs a single, unified policy trained with a novel terrain-invariant end-effector sampling strategy. For the evaluation, 4096 parallel agents were instantiated with five data samples collected from each agent.

Metric To quantitatively evaluate performance in the simulator, the following metrics are defined:

- • Base Tracking Error: The error between commanded and actual base velocities, comprising linear (ex, ey) and angular eω components.
- • End-Effector Position Error: The Euclidean distance

(Dpos) between the current and commanded end-effector positions in the world frame.

- • End-Effector Orientation Error: The quaternion geodesic distance (Dori) between the current qcurr and target qtar orientations, calculated as Dori = 2 ∗ arcos(|qcurr · qtar|).

Simulation result Our method was compared with evaluations conducted under both static (standing) and dynamic (moving) conditions. Due to the robot’s structural constraints, sampling unreachable samples leads to selfcollisions, which negatively impact the training process. To mitigate this issue, we deliberately reduced the volume of the sampling space used during training. To ensure a fair comparison and test generalization, both methods were evaluated in the same, larger workspace, as detailed in Table 4, with the comparative results of this evaluation presented in Table 3.

I.COLLECT R.NAVIGATE C.DELIVERY C.STORAGE RESTOCKING SHOPPING O.COLLECT O.DELIVERY

Navigate 97.4 86.6 98.3 97.7 98.2 98.3 98.4 95.6 Pick 72.7 / 84.6 79.6 83.3 85.0 69.0 72.7 Place 96.8 / 72.7 83.8 79.2 76.5 95.0 80.0 Push/Pull / 94.1 / 71.0 / / / 85.7 Drag / / 69.2 / / 79.2 / /

Overall 66.7 69.8 41.0 44.9 56.7 47.5 63.3 46.4 Table 2: Overall success rates (%) of 8 ODYSSEY long-horizon tasks, along with per-action success rates for each task.

###### 4.3 Sim-to-real Performance

###### Stand still Move

Roboduet ours Roboduet Ours

We conducted real-world experiments to validate the simto-real performance of our framework, which integrates the high-level coarse-to-fine task planner with the low-level whole-body control policy.

- ex ↓ 0.32 0.08 9.70 0.36
- ey ↓ 0.34 2.69 15.42 2.31 eω ↓ 0.32 0.26 60.59 0.79

Dpos ↓ 11.08 11.48 10.75 10.57 Dori ↓ 47.14 46.93 47.53 47.15

Robot System Setup Our robot platform, as shown in Fig. 4, combines a 12-DoF Unitree Go2 quadruped with a 6-DoF Arx5 manipulator. The Go2 (15kg weight, 8kg payload) includes a built-in Unitree L1 LiDAR, and the 3.35kg Arx5 arm is mounted on its back, similar to (Ha et al. 2024). For high-level perception, the platform is equipped with a MID-360 LiDAR for localization and two RealSense cameras: a head-mounted D435i for RGB imagery and a grippermounted D405 for RGB-D data. The control policy operates at 50 Hz, with a PD controller issuing motor commands at 200 Hz.

- Table 3: The quantitative result under static (stand still) and dynamic (move) conditions.

Train Evaluation RoboDuet Ours

- xˆ [-1.00, 1.00] [-1.00, 1.00] [-1.50, 1.50]
- yˆ [0.00, 0.00] [-1.00, 1.00] [0.00, 0.00] wˆ [-0.60, 0.60] [-1.00, 1.00] [-1.50, 1.50]

ˆlee [0.30, 0.70] [0.30, 0.65] [0.20,0.80]

pˆee [-0.45π, 0.45π] [-0.17π, 0.33π] [-0.50π, 0.50π] yˆee [-0.50π, 0.50π] [-0.33π, 0.33π] [-0.50π, 0.50π] αˆee [-0.45π, 0.45π] [-0.50π, 0.50π] [-0.50π, 0.50π] βˆee [-0.33π, 0.33π] [-0.17π, 0.50π] [-0.50π, 0.50π] γˆee [-0.42π, 0.42π] [-0.50π, 0.50π] [-0.50π, 0.50π]

- Table 4: Sampling ranges for commands used during training and evaluation. The locomotion commands consist of linear velocities (xˆ, yˆ) and angular velocity (wˆ). The endeffector target is defined by a position in spherical coordinates (radius ˆlee, pitch pˆee and yaw yˆee) and an orientation in Euler angles (αˆee, βˆee, γˆee).

Real-world experiments The ODYSSEY framework was evaluated on two long-term tasks (”navigate to pick” and ”pick and place”) using five different objects. The entire system demonstrated successful sim-to-real transfer on task planning and execution as illusion in Fig. 4.

Despite this success, some sim-to-real gaps persist. For instance, the robot occasionally failed at grasping small objects due to the inaccuracies in end-effector tracking and visual perception. Through these experiments, our approach has demonstrated significant potential for solving longhorizon mobile exploration and manipulation tasks, while simultaneously identifying the primary challenges—robust perception and high-precision control—that must be addressed for seamless real-world deployment.

[Figure 128]

The evaluation results indicate that our policy achieves better performance in base velocity tracking (rows 1-3), an improvement we attribute to the inclusion of terrain data in the policy’s observation, which enhances the robot’s state estimation. End-effector pose tracking performance remains comparable to the baseline (rows 4-5). Notably, a key aspect of this evaluation is that our policy was trained in an endeffector workspace intentionally smaller than that used for the baseline, and our policy is adaptive to different topographies (e.g., steps). This demonstrates strong generalization capabilities from a more constrained training domain of our approach.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

1 2

[Figure 133]

Arm Cam: RealSense D405

[Figure 134]

[Figure 135]

3 4

ARX Arm

Back Lidar: Livox MID-360

[Figure 136]

[Figure 137]

[Figure 138]

5 6

Front Cam: RealSense D435i Camera

Front Lidar: UniTree L1

UniTree Go2

###### Figure 4: The robot system and real-world experiments

##### 5 Conclusions and Future Work

We present ODYSSEY, a unified framework for openworld mobile manipulation that integrates hierarchical task planning with terrain-adaptive whole-body control. Our approach demonstrates robust sim-to-real transfer and generalization across diverse environments and longhorizon tasks. Future work will extend our benchmark into a comprehensive evaluation paradigm for visionlanguage models (VLMs) and mobile manipulators, enabling cross-embodiment assessment of semantic reasoning and locomotion-manipulation coordination. Additionally, we aim to explore the emergent capabilities of active perception, where dynamic scene understanding and adaptive motion synergize for more efficient real-world interaction. This direction could unlock new behaviors in cluttered, unstructured environments, further bridging the gap between high-level planning and low-level control.

##### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Brohan, A.; Brown, N.; Carbajal, J.; Chebotar, Y.; Dabis, J.; Finn, C.; Gopalakrishnan, K.; Hausman, K.; Herzog, A.; Hsu, J.; et al. 2022. Rt-1: Robotics transformer for realworld control at scale. arXiv preprint arXiv:2212.06817.

Cheang, C.-L.; Chen, G.; Jing, Y.; Kong, T.; Li, H.; Li, Y.; Liu, Y.; Wu, H.; Xu, J.; Yang, Y.; et al. 2024. Gr2: A generative video-language-action model with webscale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158.

Fu, Z.; Cheng, X.; and Pathak, D. 2023. Deep whole-body control: learning a unified policy for manipulation and locomotion. In Conference on Robot Learning, 138–149. PMLR.

Fu, Z.; Zhao, T. Z.; and Finn, C. 2024. Mobile aloha: Learning bimanual mobile manipulation with low-cost wholebody teleoperation. arXiv preprint arXiv:2401.02117.

Gong, R.; Huang, J.; Zhao, Y.; Geng, H.; Gao, X.; Wu, Q.; Ai, W.; Zhou, Z.; Terzopoulos, D.; Zhu, S.-C.; et al. 2023. Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20483–20495.

Grandia, R.; Jenelten, F.; Yang, S.; Farshidian, F.; and Hutter, M. 2023. Perceptive locomotion through nonlinear model-predictive control. IEEE Transactions on Robotics, 39(5): 3402–3421.

Gu, Q.; Kuwajerwala, A.; Morin, S.; Jatavallabhula, K. M.; Sen, B.; Agarwal, A.; Rivera, C.; Paul, W.; Ellis, K.; Chellappa, R.; et al. 2024. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 5021–5028. IEEE.

Ha, H.; Gao, Y.; Fu, Z.; Tan, J.; and Song, S. 2024. Umi on legs: Making manipulation policies mobile with manipulation-centric whole-body controllers. arXiv preprint arXiv:2407.10353.

Huang, X.; Huang, Y.-J.; Zhang, Y.; Tian, W.; Feng, R.; Zhang, Y.; Xie, Y.; Li, Y.; and Zhang, L. 2023. Open-set image tagging with multi-grained text supervision. arXiv preprint arXiv:2310.15200.

Jatavallabhula, K. M.; Kuwajerwala, A.; Gu, Q.; Omama, M.; Chen, T.; Maalouf, A.; Li, S.; Iyer, G.; Saryazdi, S.; Keetha, N.; et al. 2023. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241.

Jiang, J.; Zhu, Y.; Wu, Z.; and Song, J. 2025a. DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes. arXiv

- preprint arXiv:2506.01950.

Jiang, Y.; Zhang, R.; Wong, J.; Wang, C.; Ze, Y.; Yin, H.; Gokmen, C.; Song, S.; Wu, J.; and Fei-Fei, L. 2025b. Behavior robot suite: Streamlining real-world whole-body manipulation for everyday household activities. arXiv preprint arXiv:2503.05652.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi,

- P.; et al. 2024. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246.

Liu, M.; Chen, Z.; Cheng, X.; Ji, Y.; Qiu, R.-Z.; Yang, R.; and Wang, X. 2024a. Visual whole-body control for legged loco-manipulation. arXiv preprint arXiv:2403.16967.

Liu, P.; Guo, Z.; Warke, M.; Chintala, S.; Paxton, C.; Shafiullah, N. M. M.; and Pinto, L. 2024b. Dynamem: Online dynamic spatio-semantic memory for open world mobile manipulation. arXiv preprint arXiv:2411.04999.

Mittal, M.; Yu, C.; Yu, Q.; Liu, J.; Rudin, N.; Hoeller, D.; Yuan, J. L.; Singh, R.; Guo, Y.; Mazhar, H.; Mandlekar, A.; Babich, B.; State, G.; Hutter, M.; and Garg, A. 2023. Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE Robotics and Automation Letters, 8(6): 3740–3747.

Nasiriany, S.; Maddukuri, A.; Zhang, L.; Parikh, A.; Lo, A.; Joshi, A.; Mandlekar, A.; and Zhu, Y. 2024. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523.

Pan, G.; Ben, Q.; Yuan, Z.; Jiang, G.; Ji, Y.; Li, S.; Pang, J.; Liu, H.; and Xu, H. 2025a. RoboDuet: Learning a Cooperative Policy for Whole-body Legged Loco-Manipulation. IEEE Robotics and Automation Letters.

Pan, M.; Zhang, J.; Wu, T.; Zhao, Y.; Gao, W.; and Dong, H. 2025b. Omnimanip: Towards general robotic manipulation via object-centric interaction primitives as spatial constraints. In Proceedings of the Computer Vision and Pattern Recognition Conference, 17359–17369.

Qi, Z.; Zhang, W.; Ding, Y.; Dong, R.; Yu, X.; Li, J.; Xu, L.; Li, B.; He, X.; Fan, G.; et al. 2025. Sofar: Languagegrounded orientation bridges spatial reasoning and object manipulation. arXiv preprint arXiv:2502.13143.

Qiu, R.-Z.; Hu, Y.; Song, Y.; Yang, G.; Fu, Y.; Ye, J.; Mu, J.; Yang, R.; Atanasov, N.; Scherer, S.; et al. 2024a. Learning generalizable feature fields for mobile manipulation. arXiv preprint arXiv:2403.07563.

Qiu, R.-Z.; Song, Y.; Peng, X.; Suryadevara, S. A.; Yang, G.; Liu, M.; Ji, M.; Jia, C.; Yang, R.; Zou, X.; et al. 2024b. Wildlma: Long horizon loco-manipulation in the wild. arXiv preprint arXiv:2411.15131.

Ren, T.; Liu, S.; Zeng, A.; Lin, J.; Li, K.; Cao, H.; Chen, J.; Huang, X.; Chen, Y.; Yan, F.; Zeng, Z.; Zhang, H.; Li, F.; Yang, J.; Li, H.; Jiang, Q.; and Zhang, L. 2024. Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks. arXiv:2401.14159.

Shridhar, M.; Manuelli, L.; and Fox, D. 2022. PerceiverActor: A Multi-Task Transformer for Robotic Manipulation. arXiv:2209.05451.

Team, G. 2025. GenieSim.

Wang, H.; Chen, J.; Huang, W.; Ben, Q.; Wang, T.; Mi, B.; Huang, T.; Zhao, S.; Chen, Y.; Yang, S.; et al. 2024a. Grutopia: Dream general robots in a city at scale. arXiv preprint arXiv:2407.10943.

Wang, J.; Rajabov, J.; Xu, C.; Zheng, Y.; and Wang, H.

- 2024b. Quadwbg: Generalizable quadrupedal whole-body grasping. arXiv preprint arXiv:2411.06782. Wang, Y.; Zhu, H.; Liu, M.; Yang, J.; Fang, H.-S.; and He, T.
- 2025. VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers. arXiv

- preprint arXiv:2507.01016.

Zhang, C.; Han, D.; Zheng, S.; Choi, J.; Kim, T.-H.; and Hong, C. S. 2023. MobileSAMv2: Faster Segment Anything to Everything. arXiv:2312.09579.

Zhang, H.; Yu, H.; Zhao, L.; Choi, A.; Bai, Q.; Yang, B.; and Xu, W. 2025. Slim: Sim-to-real legged instructive manipulation via long-horizon visuomotor learning. arXiv preprint arXiv:2501.09905.

Zhi, P.; Zhang, Z.; Zhao, Y.; Han, M.; Zhang, Z.; Li, Z.; Jiao, Z.; Jia, B.; and Huang, S. 2024. Closed-loop openvocabulary mobile manipulation with gpt-4v. arXiv preprint arXiv:2404.10220.

Zhou, C.; Loy, C. C.; and Dai, B. 2022. Extract free dense labels from clip. In European conference on computer vision, 696–712. Springer.

Zhuang, Z.; Fu, Z.; Wang, J.; Atkeson, C.; Schwertfeger, S.; Finn, C.; and Zhao, H. 2023. Robot parkour learning. arXiv preprint arXiv:2309.05665.

##### Appendix A Long-horizon Planner Pipeline

###### A.1 Instance-Level Semantic Graph Construction

LIDAR Point Cloud

###### Point Map Instance Map

[Figure 139]

Pose

[Figure 140]

project Front Camera RGB

[Figure 141]

[Figure 142]

2D Masks

[Figure 143]

[Figure 144]

- • Tags
- • 3D bbox center
- • 3D bbox extent

[Figure 145]

GroundingSAM

Tags [TV, cabinet, chair, …]

RAM++

Figure 5: Instance-level graph construct pipeline.

To enable egocentric task planning, we implement an online multi-modal perception pipeline that constructs an instance-level semantic graph from onboard sensors.

A downward-facing LiDAR provides real-time point clouds at 10Hz, which are aggregated into a global point map using the robot’s estimated pose. This map serves both as a geometric substrate for semantic abstraction and as a SLAM-based 2D occupancy grid for path planning.

In parallel, a front-facing RGB camera captures visual frames. We use RAM++ (Huang et al. 2023) to identify candidate object labels from each RGB image. These labels, along with the image, are fed into Grounding-SAM (Ren et al. 2024) to obtain instance-level segmentation masks mi. We replace the standard SAM model with MobileSAM(Zhang et al. 2023) to improve efficiency. The masks are projected onto the current LiDAR frame to obtain perobject point cloud segments.

Inspired by prior works(Gu et al. 2024; Jatavallabhula et al. 2023), to enable consistent object tracking across time, we fuse observations using both semantic and geometric similarity:

- • Semantic Similarity. For each instance mask, we ex-

tract a sparse visual-semantic feature map fI using MaskCLIP (Zhou, Loy, and Dai 2022). We pool features

within each mask mi to generate an instance-level descriptor fi ∈ Rd. Two instances i and j are considered semantically consistent if their cosine similarity satisfies:

simsem(fi,fj) =

fi⊤fj ∥fi∥∥fj∥

> τsem, τsem = 0.8. (9)

- • Geometric Similarity. Let Pi and Pj be the point sets of two object candidates. We consider them geometrically aligned if at least 80% of the points in Pi have a nearest neighbor in Pj within radius ϵ:

1 |Pi| p∈P

I min

∥p − q∥ < ϵ > τgeo, τgeo = 0.8.

q∈Pj

i

(10)

When both criteria are satisfied, the two instances are merged as the same object across time.

The final instance graph G = {Oi} contains nodes Oi with attributes including semantic tags and 3D bounding boxes. This symbolic representation serves as the input for downstream task decomposition and decision-making.

###### A.2 Hierarchical Planning Prompts

LLM-based Task-level Prompts This section illustrates how we leverage large language models (LLMs) to decompose tasks into structured symbolic action sequences (e.g., navigate, pick, place). The prompt input consists of the user instruction and a summarized semantic scene graph. Example prompts are shown in Fig. 10.

VLM-based Manipulation Prompts For the execution of atomic manipulation actions, we demonstrate how visionlanguage models (VLMs) are used to infer task-relevant contact points and end-effector orientations. The prompt input includes a first-person RGB observation and a natural language description of the current sub-task. Examples are shown in Fig. 11.

B Whole-body Policy Training

###### B.1 Reward design

This section details the key reward terms used during policy training, summarized in Table 5.

Weight Term Equation Stage 1 Stage 2

rxytrack exp(−((vˆxy − vxy))/γxy) 2.75 2.75 ryawtrack exp(−((ˆω − ω))/γω) 1.50 1.50

reetrack−pos (pˆee − pee)2 0.00 -1.20 reetrack−ori (ϕˆee − ϕee)2 0.00 -1.50

rgait Eq. 17 0.75 0.75 rfreq Eq. 19 1.25e1 1.25e1

rtorquebase |τbase|2 -2.0e-4 -2.0e-4

raccbase |q¨base|2 -2.5e-7 -2.0e-7 rpowerbase |τbase| ∗ |˙qbase| -2.0e-5 -2.0e-5 rtorquearm |τarm|2 0.00 -4.0e-4

raccarm |q¨arm|2 0.00 -2.5e-6 rpowerarm |τarm| ∗ |˙qarm| 0.00 -2.0e-4

rsmooth (at − at−1)2 -0.02 -0.02

Table 5: The key reward terms used in policy training

To encourage a stable gait, we use a gait reward rgait that promotes specific foot contact patterns. It encourages synchrony between diagonal leg pairs (FL/RR, FR/RL) using a reward term rs and asynchrony between all other pairs using ra. The final gait reward is a multiplicative combination of individual reward terms (rs,ra), which are functions of the squared error between the time two legs (A, B) have spent

in the air (Aair) or in contact (Acont).

tsair(A,B) = clip((Aair − Bair)2,0.0,0.04) (11) tscont(A,B) = clip((Acont − Bcont)2,0.0,0.04) (12)

s

air(FL,RR)+tscont(FL,RR)) (13) taair(A,B) = clip((Aair − Bcont)2,0.0,0.04) (14)

rs(FL,RR) = e−(t

tacont(A,B) = clip((Acont − Bair)2,0.0,0.04) (15) ra(FL,FR) = e−(t

air(FL,FR)+tacont(FL,FR)) (16) rgait =

a

###### rs(i,j) ·

###### ra(k,l)

i,j∈sync pairs

k,l∈async pairs

(17) A frequency reward rfre regulates the gait’s cadence by penalizing deviation from a target frequency ftarget. The frequency for each leg f(leg) is the inverse of the time between consecutive ground contacts (tcontk − tcontk−1), where k is the timestamp. The frequency reward rfre is defined as bellow:

rf(leg) = exp(−0.5 · (f(leg) − ftarget)2) (18) rfre =

rf(leg) (19)

leg∈FL,FR,RL,RR

In stage one, training focuses on locomotion. The reward function includes task-specific terms for base velocity tracking (rxytrack,ryawtrack) and gait shaping (rgait,rfreq). To improve stability and facilitate sim-to-real transfer, several regularization rewards are applied. These include penalties on torque (rtorque), joint acceleration (racc) and action-toaction changes (rsmooth) to encourage stable policy outputs. Additionally, a power consumption penalty (rpower) is used to discourage excessive mechanical work. In stage two, all Stage one rewards are retained, and new task-specific terms are introduced for end-effector pose tracking (position reetrack−pos and orientation reetrack

ori). Furthermore, the regularization penalties(torque, acceleration, and power) are extended to include the manipulator’s joints.

###### B.2 Domain randomization

###### Parameter Ranges Method

Friction [0.4, 2.0] Base Mass [-5.0, 5.0] add Base Pushing x: [-0.5, 0.5] interval

y: [-0.5, 0.5]

Actuator Gains [0.8, 1.2] scale Ee Link Mass [0.0, 0.2] add

Joint reset [0.5, 1.5] scale Base reset x: [-0.5, 0.5], y: [-0.5, 0.5] add

ω : [−π,π], vx: [-0.5, 0.5] vy: [-0.5, 0.5], vz: [-0.5, 0.5] α: [-0.5, 0.5], β: [-0.5, 0.5] γ: [-0.5, 0.5]

Table 6: The domain randomization ranges. The method means the method of randomization, where ”add” means add to the original, ”scale” means scale up the original, and ”interval” means continue for a while.

To enhance out-of-distribution generalization and facilitate sim-to-real transfer, our policy is trained with extensive domain randomization, as detailed in Table 6. This process includes randomizing the robot’s initial conditions, such as its base pose and joint configuration, at the start of each episode. Crucially, to ensure adaptability to tasks involving different payloads, the mass of the end-effector is also randomized.

##### C Benchmark Configurations

This section presents the task setup used in our benchmark evaluation, including both short-horizon and long-horizon mobile manipulation tasks.

###### C.1 ARNOLD Short-Horizon Tasks

To evaluate the fine-grained manipulation capabilities of our system, we migrate four short-horizon tasks from the ARNOLD benchmark into our custom simulation environment. These tasks are designed to assess step-wise visuomotor precision and generalization under diverse configurations.

Simulation Integration. To enable evaluation within our own simulation platform, we convert the raw ARNOLD data into standardized YAML configuration files. The conversion pipeline involves:

- 1. Parsing each .npz sample to extract task-specific parameters, including scene setup, object category, pose, and robot initial state.
- 2. Transforming all coordinate frames from ARNOLD’s convention to that of our simulator.
- 3. Automatically generating natural language task instructions and adjusting the robot’s initial pose to ensure reachability and feasibility.
- 4. Grouping tasks by scene and saving them as per-scene YAML files containing layout, object initialization, evaluation metrics, and interaction constraints.

This setup ensures consistent large-scale deployment and reproducible benchmarking of ODYSSEY’s short-horizon manipulation performance across multiple scenarios. Examples of multiple tasks sharing a single scene configuration are shown in Figure 6.

Action Sequence Structure. To facilitate scalable benchmarking, we adopt predefined action sequences in place of LLM-based task decomposition. This ensures consistent execution patterns across trials and enables focused evaluation of local manipulation accuracy independent of highlevel planning variance. For instance, the OPENCABINET task consists of the following four phases:

- • Observe & Plan: Capture the scene (o1) to identify the semantic target and compute the desired end-effector pose a∗ for contact.
- • Pre-Alignment: Move the gripper to a pre-contact pose located 10cm away from a∗ along the approach axis (opposite to gripper’s z-axis).

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

PickupObjectReorientObjectOpenCabinetCloseCabinet

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

- Figure 6: ARNOLD converted environments of four tasks.

- • Execute Grasp: Linearly advance the gripper along the approach direction to reach a∗ and close the fingers to secure the object.
- • Task Completion: Apply the task-specific motion (e.g., pulling, lifting) until the environment state satisfies the success condition.

This structured formulation allows for fine-grained progress monitoring at 0%, 33%, 66%, and 100% completion stages, as illustrated in Figure 7.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

- Figure 7: The continuous state monitoring of ARNOLD short-horizon tasks.

###### C.2 ODYSSEY Long-Horizon Tasks

As shown in Fig. 8. Our benchmark includes eight longhorizon tasks constructed to evaluate full-pipeline perfor-

mance, spanning perception, task planning, navigation, and manipulation. Each task consists of 2–3 subgoals and is designed with diverse object types, spatial distributions, and physical interaction modes.

Table 7 summarizes the tasks, listing the task name, number of variations, and number of unique asset types involved in each.

###### Task Names Variations Assets

INDOORCOLLECT 45 65 ROOMNAVIGATION 43 11 CARTDELIVERY 39 47 CABINETSTORAGE 49 69 RESTOCKING 30 34 SHOPPING 40 45

OUTDOORCOLLECT 30 47 OUTDOORDELIVERY 28 48

Table 7: Overview of 8 long-horizon tasks with names, variation counts, and asset usage per task.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

# ...

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Figure 8: ODYSSEY benchmark details. Including task type, scene, and object configuration details.

##### D Experiment Details

###### D.1 Short-Horizon Evaluation

Data Split To assess generalization under various task variations, the ARNOLD benchmark defines five splits:

- • Test: All components—objects, scenes, and states—are seen during training.
- • Novel Object: Involves unseen objects, with seen scenes and goal states.
- • Novel Scene: Uses unseen environments while keeping objects and goals seen.
- • Novel State: Includes novel goal configurations under seen objects and scenes.
- • Any State: Targets fine-grained generalization to arbitrary continuous goals.

In Section 4.1, we report the average success rate over the last four splits as a unified Novel setting for simplicity and comparison clarity. Here in the appendix, we present detailed breakdowns across all five splits.

Baselines and Evaluation Protocol In Section 4.1, we primarily compare ODYSSEY with PerAct, the strongest baseline from ARNOLD. For a more complete analysis, we extend the evaluation to include all three PerAct variants:

- • PerAct: The standard model based on voxelized RGB-D fusion and Perceiver Transformer.
- • PerAct†: Enhanced with value-function supervision to improve decision-making.
- • PerAct (MT)†: A multi-task trained variant leveraging more diverse training signals.

In addition, following the ARNOLD evaluation protocol, each model is evaluated in a two-phase setup:

- • Phase I (End-to-End): Includes both grasping and subsequent manipulation.
- • Phase II (Manipulation Only): Uses ground-truth grasp poses to isolate manipulation performance.

This extended evaluation reveals the specific strengths and failure modes of ODYSSEY compared to baselines, particularly under generalization stress tests.

Results and Failure Analysis Table 8 presents detailed performance across all tasks and data splits. ODYSSEY consistently outperforms all PerAct variants, particularly in generalization settings involving novel objects, scenes, or goal states. The two-phase evaluation further highlights the robustness of our manipulation module when decoupled from grasping errors.

Common failure cases across tasks are largely attributed to (1) inaccurate semantic reasoning by the vision-language model (VLM), such as object orientation confusion or suboptimal contact point prediction, and (2) challenges in physical interaction execution, including occlusion, narrow geometry, and unstable force transfer.

- • PICKUPOBJECT: Phase I failures mainly arise from misjudged object orientation or selecting hard-to-grasp regions (e.g., bottlenecks). These issues are largely resolved in phase II, validating the effectiveness of our pose-conditioned manipulation.
- • REORIENTOBJECT: Errors stem from inaccurate initial pose estimation or weak grasps on thin sections, leading to slippage during reorientation. ODYSSEY exhibits robust generalization even under such dynamic conditions.
- • OPENCABINET: Limited improvement in phase II suggests inherent difficulty in the manipulation itself. Failures often relate to misidentified handles, drawer confusion, or inaccurate grasp positioning.
- • CLOSECABINET: Performance is mainly affected by occlusion and poor camera framing. Handles are sometimes out of view during approach or are visually ambiguous due to lighting or textures.

In summary, ODYSSEY demonstrates strong semantic grounding, generalization, and precision control, particularly in high-complexity tasks. Remaining challenges highlight future directions in visual robustness and contact policy refinement.

###### D.2 Long-Horizon Evaluation

Failure Analysis To complement the long-horizon evaluation in Section 4.1, we present a failure mode analysis aggregated over all 6 indoor and 2 outdoor ODYSSEY tasks. Rather than reporting per-task breakdowns, we group failures by environment type and visualize them separately for indoor and outdoor scenarios.

Each failure is assigned to one of the following three categories:

- • Reasoning Failure: Errors arising from the LLM-based task planner or VLM-based local action advisor, including incorrect subgoal decomposition, ambiguous symbolic grounding, or mispredicted contact poses.
- • Control Failure: Low-level execution issues related to either locomotion or manipulation, such as unstable whole-body control on uneven terrain or failed grasp execution due to poor alignment or force instability.
- • Navigation Failure: Failures caused by inability to bring target objects into a feasible manipulation workspace, or collisions during path execution due to inaccurate waypoint planning or obstacle avoidance.

- Figure 9 shows the distribution of these failure types

across indoor and outdoor tasks. From the analysis, we draw several key observations:

• Indoor tasks are dominated by reasoning and control failures. Tasks such as CABINETSTORAGE and RESTOCKING, which involve complex semantic reasoning

Indoor

Navigation Success

Control Success

Processing Success

147 141

Total

190

2 2 2

Stop Too Far Navigation Collision

246

Navigation Failure

6

25 Stop Too Close Grasp Fail Push/Pull Fail Drag Fail Place Fail

Control Failure

43

- 7
- 8 3

Reasoning Failure

12

56 Task Break Failure

10

EE Orientation Error EE Position Error

30

4 Recognition Failure

Outdoor

Navigation Success

Control Success

Processing Success

31

34

Total

48

58

Navigation Failure

- 3

- 1
- 2

Stop Too Far Stop Too Close

7

- 1

4

- 2

Control Failure

Grasp Fail

14

Place Fail

1

Reasoning Failure

Push/Pull Fail Fall down

Task Break Failure

10

2

EE Position Error

EE Orientation Error

7

Figure 9: Failure breakdown by environment type. Failures are categorized into reasoning, control, and navigation errors. Indoor tasks suffer more from language and control failures, while navigation becomes more challenging in outdoor scenarios.

and precise manipulation, frequently fail due to incorrect subgoal decomposition or inaccurate contact prediction. Control-related issues, such as unstable grasps and force misalignment, are also prevalent due to tight spatial constraints and object diversity in indoor environments.

- • Control failures in outdoor scenarios are often tied to whole-body instability. Unlike indoors, where control issues center on gripper-object interaction, outdoor control failures are more frequently caused by the robot falling or losing balance, particularly during manipulation actions performed on slopes or uneven surfaces.
- • Reasoning errors span both global and local levels. Global planning failures include incorrect symbolic grounding or subgoal ordering, while local failures stem from misidentifying contact points or failing to detect objects due to occlusions or poor lighting.

Split Method PICKUPOBJECT REORIENTOBJECT OPENCABINET CLOSECABINET

PerAct 94.03 97.76 19.48 24.68 24.64 42.03 22.33 45.63 PerAct† 94.78 95.52 24.68 28.57 23.19 49.28 30.10 48.54 PerAct (MT)† 90.30 92.54 14.29 20.78 20.29 39.13 19.42 37.86 ODYSSEY 60.45 81.34 51.32 76.32 56.30 64.70 74.32 84.45

Test

PerAct 86.55 92.73 11.40 35.09 0.00 00.00 1.82 5.45 PerAct† 87.27 91.27 10.53 32.46 0.00 4.94 0.00 5.45 PerAct (MT)† 81.09 85.45 7.89 24.56 0.00 4.94 1.82 5.45 ODYSSEY 41.09 72.36 31.58 64.04 50.32 61.29 84.06 92.82

Novel Object

PerAct 72.85 84.62 17.07 31.71 0.00 4.97 5.10 19.11 PerAct† 69.68 84.16 13.41 37.80 0.55 4.97 6.37 19.75 PerAct (MT)† 67.87 79.64 7.32 29.27 1.10 4.42 7.01 14.01 ODYSSEY 43.43 63.80 41.98 54.32 52.94 64.31 62.96 70.37

Novel Scene

PerAct 2.38 0.68 0.00 0.95 0.00 5.81 1.39 1.39 PerAct† 0.68 2.38 0.48 00.00 0.00 6.22 0.00 8.33 PerAct (MT)† 2.04 3.06 0.95 2.38 0.00 3.73 2.78 11.11

Novel State

###### ODYSSEY 46.60 78.91 67.48 81.07 48.85 63.22 78.87 91.70

PerAct 47.01 50.75 7.79 19.48 5.80 21.74 3.88 15.53 PerAct† 48.51 47.76 14.29 14.29 4.35 24.64 6.80 13.59 PerAct (MT)† 46.27 47.01 12.99 12.99 4.35 5.80 4.85 8.74

Any State

###### ODYSSEY 47.22 80.59 51.95 81.82 54.78 67.83 83.33 94.93

Table 8: Evaluation results across short-horizon tasks and data splits. Each row corresponds to a specific split, and each column reports success rates (%) on a task. Gray entries denote phase-II performance using ground-truth grasp poses. ODYSSEY outperforms all PerAct variants, especially in generalization and manipulation settings.

### [System Prompt]

You are a quadruped robot with a robotic arm that needs to understand 3D scene graphs to complete task planning.

## Input Information

3D Scene Graph: JSON format describing scene objects and robot itself, where each object contains:

- - "object_tag": Brief tags categorizing the object (sometimes inaccurate)
- - "bbox_extent": Extents of the 3D bounding box for the object in x, y, z direction
- - "bbox_center": Centroid of the 3D bounding box for the object Task Requirements

When given a task description, you need to:

- 1. Task Decomposition: Break down complex sequential tasks into a series of atomic actions
- 2. Action Types: Only use these predefined atomic actions:

- - "nav": Navigate to a position
- - "pick": Pick up an object
- - "place": Place a held object
- - “pull": Pull an articulated object while the quadruped remains stationary.
- - “push": Push an articulated object while the quadruped remains stationary.
- - "drag": Drag an object along the ground to a target location while the quadruped walks with it. Output Format Respond with a JSON array (enclosed in a Markdown fenced code) of actions, where each action is a

dictionary containing: ```json {

"inferred_query": "Your interpretation of this specific action", "action_type": "One of [nav, pick, place, pull, push, drag]", "relevant_object": "the most relevant object name robot need to locate for this action (if applicable)", "query_achievable": "Boolean - whether this action is achievable using the objects in the 3D scene", "target_pose": [x, y], // Only for nav/drag actions: center of target object in 2D Map

} ```

### [User]

Task: {text} Scene JSON: {scene_graph} Robot current pose (x, y, yaw): {robot_pose}

- Figure 10: Prompt examples for task-level planning using LLM. The model is conditioned on a semantic instance graph and a goal instruction, and produces a structured list of atomic actions.

[Position System Prompt]

[Orientation System Prompt]

[Figure 239]

[User]

[Result]

- Figure 11: Prompt examples for local manipulation guidance using VLM. Given an egocentric image and sub-task description, the model predicts a contact point and end-effector orientation under geometric constraints.

