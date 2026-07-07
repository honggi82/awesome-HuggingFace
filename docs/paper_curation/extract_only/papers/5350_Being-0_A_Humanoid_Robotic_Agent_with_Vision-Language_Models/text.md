## Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills

Haoqi Yuan13 Yu Bai2 Yuhui Fu1 Bohan Zhou1 Yicheng Feng1 Xinrun Xu2 Yi Zhan1 B¨orje F. Karlsson2 Zongqing Lu123

# arXiv:2503.12533v2[cs.RO]11May2025

#### Foundation Model

Humanoid Robot

[Figure 1]

[Figure 2]

Make a cup of coffee.

Reasoning In the image, I see …

|[Figure 3]|
|---|

[Figure 4]

- • Whole-body Joint Commands

Observation:

- • Binocular RGB Images
- • Proprioception Active Camera

Detection

find_table success

[Figure 5]

[Figure 6]

| | |
|---|---|
| | |

GPT-4o

Planning

Grasp a coffee cup.

Action:

Dexterous Hands

[Figure 7]

Grasp a coffee cup.

Joystick Command

Navigation

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|go_straight|
|---|

|[Figure 13]|
|---|

[Figure 14]

Joystick Command

Adjustment

|turn_left|
|---|

Locomotion Manipulation

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Manipulation Policy

VLM

Manipulation

|grasp_cup|
|---|

Embodied Connector

Modular Skill Library

Figure 1. Overview of the Being-0 framework. The humanoid agent framework, Being-0, comprises three key components: (1) the Foundation Model (FM) for high-level task planning and reasoning, (2) the Connector, a vision-language model (VLM) that bridges the FM and low-level skills, and (3) the Modular Skill Library for robust locomotion and dexterous manipulation. Together, these components enable Being-0 to effectively control a full-sized humanoid robot equipped with multi-fingered hands and active vision, solving complex, long-horizon embodied tasks in real-world environments.

### Abstract

FM with a modular skill library. The FM handles high-level cognitive tasks such as instruction understanding, task planning, and reasoning, while the skill library provides stable locomotion and dexterous manipulation for low-level control. To bridge the gap between these levels, we propose a novel Connector module, powered by a lightweight vision-language model (VLM). The Connector enhances the FM’s embodied capabilities by translating language-based plans into actionable skill commands and dynamically coordinating locomotion and manipulation to improve task success. With all components, except the FM, deployable on low-cost onboard computation devices, Being-0 achieves efficient, real-time performance on a full-sized humanoid robot equipped

Building autonomous robotic agents capable of achieving human-level performance in real-world embodied tasks is an ultimate goal in humanoid robot research. Recent advances have made significant progress in high-level cognition with Foundation Models (FMs) and low-level skill development for humanoid robots. However, directly combining these components often results in poor robustness and efficiency due to compounding errors in long-horizon tasks and the varied latency of different modules. We introduce Being-0, a hierarchical agent framework that integrates an

1Peking University 2BAAI 3BeingBeyond. Correspondence to: Zongqing Lu <zongqing.lu@pku.edu.cn>.

with dexterous hands and active vision. Extensive experiments in large indoor environments demonstrate Being-0’s effectiveness in solving complex, long-horizon tasks that require challenging navigation and manipulation subtasks. For further details and videos, visit our project page.

### 1. Introduction

In the evolving field of embodied AI, humanoid robots represent an ideal platform for achieving human-level intelligence, enabling physical interactions with the real world in ways akin to humans. To realize the ultimate goal of allowing humanoid robots to autonomously perform tasks like humans, current research primarily focuses on improving individual skills, including locomotion (Radosavovic et al., 2024; Zhuang et al., 2024), bimanual manipulation (Ze et al., 2024a; Li et al., 2024a; Zhou et al., 2024), and whole-body control (He et al., 2024a; Fu et al., 2024a). However, building fully autonomous agents for humanoid robots remains a significant and largely unexplored challenge.

An autonomous robotic agent must solve diverse embodied tasks in the real world by grounding language instructions into feasible plans and reliably stitching skills to accomplish long-horizon tasks. Recent studies (Firoozi et al., 2023; Hu et al., 2023) in robotic agents have integrated Foundation Models (FMs) with learning-based robotic skills, leveraging FMs’ capabilities in general-purpose vision-language understanding for skill planning (Ahn et al., 2022; Chen et al., 2024), success detection (Huang et al., 2022), and reasoning. While these methods have achieved some success in building agents for robot arms (Liang et al., 2023), wheeled robots (Ahn et al., 2022), and quadruped robots (Chen et al., 2024), can the same success be replicated for humanoid robots? In this paper, we introduce Being-0, a hierarchical agent framework designed for humanoid robots.

We begin by equipping a universal FM-based agent framework (Tan et al., 2024) with a modular robotic skill library. This skill library includes a robust locomotion skill based on joystick commands and a set of manipulation skills with language descriptions, acquired through state-of-the-art teleoperation (Cheng et al., 2024b) and imitation learning (Zhao et al., 2023) methods. These skills enable the robot to walk and manipulate objects in response to language commands. In principle, the FM agent could call these skills based on image observations in a closed-loop manner to solve long-horizon tasks. However, we find that humanoid robots introduce unique challenges for such a system.

Unlike wheeled robots, which can precisely follow planned navigation trajectories and stop at specific positions for object manipulation, humanoid robots face inherent instability in bipedal locomotion. This instability necessitates

frequent adjustments to locomotion commands for error correction. However, existing FMs, such as GPT-4o, suffer from limitations in inference efficiency and embodied scene understanding, making humanoid agents less reactive and robust during the alternating phases of navigation and manipulation in long-horizon tasks.

To address these challenges, we propose a novel Connector module, which serves as an intermediate layer between the FM and skill library in Being-0. The Connector generates real-time commands for both locomotion and manipulation skills based on the FM’s language plan and visual observations. We model the Connector as a vision-language model (VLM) and train it using first-person images of indoor navigation annotated with language instructions, object labels, and bounding boxes. This training scheme distills embodied knowledge from vision-language navigation data into the lightweight VLM-based Connector, enabling accurate skill planning and efficient navigation at a higher control frequency. Furthermore, to seamlessly stitch navigation and manipulation skills, the Connector can send locomotion commands to adjust the humanoid’s pose, improving the initialization state for subsequent manipulation tasks.

We conduct extensive experiments on navigation, manipulation, and long-horizon tasks using a full-sized humanoid robot equipped with dexterous hands and an active camera. The results demonstrate that Being-0 achieves an average completion rate of 84.4% on challenging long-horizon tasks, highlighting the significant contribution of the Connector module and the use of active vision in the system. By deploying all modules – except for the FM on the cloud – on onboard computation devices, Being-0 achieves 4.2× efficiency in navigation compared to fully FM-based agents.

Our contributions can be summarized as follows:

- • We propose a hierarchical agent framework for humanoid robots, where each layer is optimally deployed on either the cloud or onboard devices, enabling efficient execution of long-horizon embodied tasks.
- • We introduce a VLM-based Connector module to bridge the gap between the FM’s language-based task plans and the execution of low-level skills. This module enhances embodied decision-making and effectively coordinates locomotion and manipulation skills for humanoid robots.
- • Our agent is capable of controlling humanoid robots with multi-fingered dexterous hands and active cameras, enhancing their dexterity in both navigation and manipulation tasks.

### 2. Humanoid Robot and Agent

As illustrated in Figure 1, we consider a humanoid robot with 41 degrees of freedom (DoFs), including a 13-DoF

|[Figure 20]<br><br>|[Figure 21]|[Figure 22]<br><br>|[Figure 23]|[Figure 24]|[Figure 25]<br><br>|[Figure 26]|[Figure 27]|
|---|---|---|---|---|---|---|---|

“Grasp a coffee cup.” “Find a coffee machine.”

###### FM

Connector

Navigation: table Adjustment Manipulation Navigation: Coffee Machine

Locomotion Grasp_Cup Locomotion

Skill

|[Figure 28]<br><br>|[Figure 29]|[Figure 30]<br><br>|[Figure 31]|[Figure 32]<br><br>|[Figure 33]|[Figure 34]<br><br>|[Figure 35]|
|---|---|---|---|---|---|---|---|

“Place the coffee cup.” “Select espresso.” Adjustment Manipulation Manipulation Manipulation Locomotion Place_Cup Grasp_Cup

“Select confirmation button.” “Grasp the coffee.”

###### FM

Manipulation Click_Button Click_Button

Connector

###### Skill

Figure 2. Workflow of Being-0 for the task “make a cup of coffee”. The figure illustrates the step-by-step execution of the task, with images arranged in two rows. The execution order proceeds left to right in the first row, then continues left to right in the second row. Images with yellow borders indicate decision-making points for the Foundation Model (FM). The yellow dialog boxes display the FM’s plans, the green boxes show decisions made by the Connector, and the blue boxes represent the skills called from the modular skill library.

lower body (two legs and a torso), two 7-DoF arms, two 6DoF dexterous hands, and a 2-DoF neck. The multi-fingered dexterous hands enable complex, human-like manipulation, while the actuated neck, equipped with a binocular RGB camera, provides active vision. This hardware configuration grants the robot human-level dexterity in visual perception, navigation, and object interaction.

this domain.

### 3. The Hierarchical Agent Framework

##### 3.1. Modular Skill Library

The first challenge we address is: how can we acquire diverse, robust low-level skills for a humanoid robot to support solving real-world, long-horizon tasks? In the literature on whole-body control (Fu et al., 2024a; He et al., 2024a), policies for individual skills typically map observations to whole-body target joint positions, simultaneously controlling leg motion and arm manipulation. However, these methods have not yet developed a wide range of manipulation skills due to the complexity of achieving precise manipulation, stable locomotion, and sim-to-real deployment with one policy.

An autonomous agent aims to complete real-world tasks described in natural language by controlling the robot’s whole-body joints. Formally, at any time, the agent has access to a task description l (e.g. “make a cup of coffee”) and can query the robot’s observations, including: (1) Proprioception: ql,qu,qh;q˙;ω , where ql,qu,qh denote joint positions of the lower body, upper body, and neck, respectively; q˙ represents joint velocity; and ω is the root velocity and angular velocity acquired from the IMU. (2) Visual input: binocular RGB images ol,or from the left and right cameras. The agent can take actions (al,au,ah), which specify target joint positions for the PD controller of the lower body, upper body, and neck, respectively.

For most tasks, we observe that the lower body and upper body serve distinct functionalities: the lower body is primarily used for navigation, while the upper body is used for manipulation (Cheng et al., 2024a). This observation motivates us to develop separate skills for stable lower-body locomotion and upper-body manipulation, building on recent advances (Kim et al., 2024; Cheng et al., 2024b) that have demonstrated the feasibility of acquiring abundant upper-body manipulation skills while keeping the lower body fixed.

Instead of directly mapping task descriptions and observations into muscle actuation, humans rely on a hierarchical system to solve real-world tasks. For example, the task “make coffee” is first decomposed into detailed plans such as “find a cup, grasp a cup, find a coffee machine...” based on prior experience. Then, practiced motor skills, such as walking and grasping, are reused to sequentially execute the task. Recent advances in robotic agents (Ahn et al.,

Stable Locomotion with Joystick Commands. The locomotion skill, which controls the lower-body joints, must enable navigation in various directions and maintain stable standing during manipulation tasks. We adopt reinforcement learning (RL) approaches (Ha et al., 2024) to train a goal-

- 2022; Huang et al., 2022) adopt this approach by integrating a high-level planner with a low-level skill library. In this paper, we aim to build such an agent for humanoid robots, addressing the unique and largely unexplored challenges in

conditioned proprioceptive policy πL(al|ql,qu,q,ω˙ ;vg) in simulation (Makoviychuk et al., 2021), followed by sim-to-real deployment at a control frequency of 50 Hz. Here, vg represents the joystick velocity command. By incorporating domain randomization and external forces during simulation, this skill enables the robot to walk according to joystick commands while maintaining balance. To integrate this into the skill library, we define a set of locomotion skills based on different joystick commands, along with skills for adjusting the head for active vision: {no action, go straight, walk backwards, turn left, turn right, sidestep left, sidestep right, tilt head, turn head}.

Acquiring Manipulation Skills. Teleoperation and imitation learning have emerged as promising approaches for acquiring diverse robotic manipulation skills at low cost. To collect high-quality, human-like manipulation data for the humanoid equipped with two dexterous hands and active vision, we use Apple VisionPro for teleoperation, following recent work (Cheng et al., 2024b). Binocular image observations ol,or are projected to the VisionPro, and the captured human motions of the head, wrists, and fingers are retargeted to robot actions at a control frequency of 10 Hz. For each skill, teleoperation trajectories τ = {(olt,ort,qtu,qth,aut ,aht )}Tt=1 are recorded, including robot observations and actions (excluding the lower body). We use ACT (Zhao et al., 2023), a behavior-cloning method with a Transformer architecture, to train the policy πM

[auj ,ahj ]tj+=Kt |olt,ort,qtu,qth for each manipulation skill Mi, associated with a language description such as “grasp bottle”. The length of the predicted action sequence, K, is set to 30 during training and 10 during deployment. This approach ensures scalability of the skill library, as a new skill can be acquired with 50 ∼ 150 trajectories, requiring less than 1 hour of teleoperation.

i

##### 3.2. Foundation Model

The high-level planner of the agent makes skill-level decisions across diverse tasks and environments, necessitating strong capabilities in general-purpose vision-language understanding and reasoning. Foundation Models (FMs) excel in these areas and have been widely adopted in recent research on AI agents (Wang et al., 2024; Tan et al., 2024). For example, Cradle (Tan et al., 2024), an agent framework built on GPT-4o, has been successfully applied to open-world games and software usage, operating keyboard and mouse skills based on image observations. Inspired by this work, we adapt the Cradle framework to build a generalist agent for humanoid robots, enabling the robot to operate skills from the skill library and solve real-world tasks.

Given an instruction l and an image observation ol, the FM (GPT-4o) performs three key functionalities for decision-

making: (1) Reasoning: The FM generates a description of the observed image and instruction, aiding in task understanding and identifying the current stage of execution. (2) Detection: The FM evaluates the success of recently executed skills, identifying failures and exceptions to inform task planning. (3) Planning: Based on the reasoning and detection results, the FM selects the next skill to execute from the skill library. Detailed prompt designs for the FM can be found in Appendix B.3.

However, when directly integrating the FM with the skill library, we encounter several challenges that severely hinder system performance. The inherent instability of bipedal locomotion makes the humanoid’s position unpredictable after short periods of walking, necessitating frequent adjustments to joystick commands rather than executing open-loop command sequences. Additionally, existing FMs, including GPT-4o, struggle with accurate 3D scene understanding, often failing to estimate the direction and depth of navigation targets correctly, which can lead to incorrect skill plans (see experimental results in Figure 3). Even when the agent successfully navigates to a target location (e.g., a table), its final standing position may not provide a suitable initial state for subsequent manipulation skills (e.g., “grasp cup”), resulting in task failure (see Figure 5). Furthermore, the low inference speed of large FMs significantly reduces system efficiency, causing the robot to move slowly and react less promptly to dynamic environments.

To address these challenges, we propose a novel Connector module in Being-0, which bridges the gap between the FM and the skill library, enhancing real-time, embodied decision-making.

### 4. Embodied Connector

The primary goal of the Connector is to translate high-level language-based plans generated by the FM into executable skill commands reliably and efficiently. At the core of the Connector is a lightweight vision-language model (VLM) trained on annotated navigation data, which enhances the agent’s embodied capabilities. This VLM enables several downstream functionalities, including grounded skill planning, closed-loop navigation, and improved transitions between navigation and manipulation during long-horizon task execution.

##### 4.1. Training the Vision-Language Model

To equip the VLM with spatial and object understanding, as well as the ability to anticipate future skills based on context, we train it on a dataset of first-person navigation images. These images are annotated with language descriptions, skills, object labels, and bounding boxes. We adopt VideoLLaMA2 (Cheng et al., 2024c) as the backbone archi-

tecture, using image observations and text instructions as inputs. The model is optimized through multi-task learning, encompassing image description, skill prediction, and object detection. The trained VLM achieves an average inference time of approximately 1 second on onboard devices across all tasks, significantly outperforming the latency of GPT-4o on cloud services. Further details on the dataset and training process are provided in Appendix B.2.

##### 4.2. Grounded Skill Planning

The main usage of the VLM is to convert the FM’s languagebased plans and real-time image observations into actionable skill plans, such as navigation targets or manipulation skills. By leveraging its enhanced understanding of relative 3D object locations, the VLM not only grounds the FM’s plans into executable skills but also corrects or refines them when necessary. For example: If the FM generates a plan to “grasp a cup” but the robot is still far from the table, the VLM interprets “grasp a cup” as a long-term goal and outputs the feasible skill (e.g., “move towards(table).”). Conversely, if the FM plans to “find a table” but the robot is already at the table, the VLM’s navigation functionality (Section 4.3) signals success to the FM, prompting it to proceed to the next skill through reasoning. This capability ensures that the planned skills remain grounded in the physical environment, reducing errors and improving task success rates.

##### 4.3. Visual Navigation with Locomotion Skills

To enable the robot to reach visual navigation goals (e.g., a table), the Connector leverages the VLM’s visual understanding and object detection capabilities. When the goal object is within the robot’s field of view, the Connector estimates its relative position using the detected bounding box and synthetic depth from binocular images. Based on this estimation, the VLM selects the most appropriate locomotion skill to move towards the object’s direction. If the object is not visible, the VLM triggers an exploration routine, combining locomotion skills with active camera movements to search for the goal. This approach significantly enhances the robot’s ability to locate objects compared to systems with fixed cameras. Implementation details are provided in Appendix B.2. By integrating the VLM’s efficient inference capabilities with modular locomotion skills, this method accelerates humanoid robot navigation while maintaining robustness in dynamic environments.

##### 4.4. Coordinating Navigation and Manipulation

To address the challenge that navigation processes may terminate in suboptimal poses for subsequent manipulation skills, we propose a pose adjustment method using the VLM. During navigation, the VLM predicts not only the object’s bounding box but also the optimal alignment direction for

the robot relative to the object. If the robot’s current facing direction deviates from this alignment, the VLM triggers a composite skill combining head rotation and forward movement to adjust the robot’s pose. This allows the robot to approach the target object along an arc-shaped path, ensuring it reaches an optimal position for manipulation. Further details are provided in Appendix B.2.

##### 4.5. Summary

Figure 2 illustrates the workflow of Being-0, highlighting the role of the Connector module. In summary, the embodied Connector provides several critical advantages for executing long-horizon tasks. By leveraging the lightweight VLM, the Connector ensures real-time responsiveness, enabling the robot to adapt dynamically to changes in its environment. This real-time capability is essential for efficient task execution, as the Connector dynamically selects and sequences modular skills, significantly reducing operational latency. Unlike the FM, the VLM’s enhanced spatial understanding allows the robot to accurately perceive and respond to its surroundings, grounding abstract language-based plans in real-time visual input. This spatial reasoning capability is particularly valuable in complex tasks, where the Connector’s robustness ensures adaptability to unexpected obstacles or environmental variations. Additionally, the Connector facilitates improved transitions between navigation and manipulation by adjusting the robot’s pose, ensuring that the robot reaches the proper positions for subsequent skills. Together, these features make the embodied Connector a cornerstone of Being-0, enabling it to tackle challenging, long-horizon tasks that require both navigation and manipulation in complex environments.

### 5. Experiments 5.1. Real-World Setup

We conduct experiments on a Unitree H1-2 humanoid robot equipped with two Inspire hands for manipulation, two Dynamixel motors for neck movement, and a ZED-mini camera mounted on the neck for active vision. The NVIDIA Jetson AGX onboard device is used to deploy the Connector and all modular skills.

Our experimental environment is a large office scene spanning a 20×20 (m) area, featuring multiple office cubicles, a wooden table, a coffee machine, and corridors connecting reception and meeting rooms. This complex and richly populated environment provides a challenging benchmark for evaluating navigation and long-horizon task execution capabilities.

To build the manipulation skill library, we collect data for a variety of daily manipulation tasks, including single-hand and bimanual tasks such as grasping and placing objects,

Table 1. Task completion rates for Being-0 with and without the Connector across various long-horizon tasks. The results demonstrate significant performance improvements when the Connector is used.

Task w/o Connector Being-0

Fetch-bottle 0.00 0.90 Deliver-basket 0.00 0.80 Prepare-coffee 0.00 0.75 Make-coffee 0.90 0.90 Deliver-coffee 0.33 0.87

Table 2. Ablation study on the proposed adjustment method in the Connector module. The results indicate the number of successful manipulations out of 5 navigation trials. (t) denotes “on the table” and (m) denotes “on the coffee machine”.

Task w/o Adjust. Being-0

Grasp-bottle 2 / 5 4 / 5 Place-basket 4 / 5 3 / 5 Grasp-coffee 1 / 5 4 / 5 Place-coffee (t) 4 / 5 5 / 5 Place-coffee (m) 0 / 5 3 / 5

operating a basket with items, using a coffee machine, and playing with toy bricks and chess games. The data collection and training details are presented in Appendix B.1.

We evaluate the agent on a diverse set of long-horizon tasks designed to test the system’s robustness in task planning and skill execution. These tasks include:

- • Fetch-bottle and Deliver-basket: These tasks require the robot to navigate to a distant wooden table and perform manipulation tasks.
- • Prepare-coffee, Make-coffee, and Deliver-coffee: These are particularly challenging tasks composed of multiple subtasks, including precise manipulation skills such as pressing buttons on the coffee machine and placing a cup in the correct position.

Further details on the task setups are provided in Appendix A.3.

##### 5.2. Solving Long-Horizon Embodied Tasks

We evaluate the performance of Being-0 on long-horizon embodied tasks, with the main results presented in Table 1. These tasks are designed to test the robot’s ability to execute complex sequences of skills in real-world environments, requiring precise coordination between high-level cognition and low-level skills.

The results demonstrate a significant performance improvement when the Connector module is utilized, particularly for tasks requiring multiple steps and integration of different skills. For example, in the Fetch-bottle task, the baseline system (w/o Connector) achieves a score of 0.00, whereas the system with the Connector attains a remarkable score of 0.90. Similarly, tasks such as Deliver-basket and Preparecoffee show substantial improvements, with performance increasing from 0.00 to 0.80 and 0.00 to 0.75, respectively.

These findings highlight the critical role of the Connector in enabling the robot to effectively complete long-horizon tasks. By bridging the gap between the FM and the skill library, the Connector enhances task success rates, particularly for scenarios requiring complex, sequential skills.

Overall, the results confirm that Being-0 is highly capable of executing long-horizon tasks with robust and reliable performance.

##### 5.3. Ablation Study

Adjustment in Navigation. We evaluate the proposed adjustment method in the Connector by testing the agent on two-stage tasks that involve navigation followed by manipulation. In this setup, the success rate of the manipulation task directly reflects the quality of the robot’s termination state after navigation. Table 2 presents the results comparing Being-0 with and without adjustment.

For grasping tasks, such as Grasp-bottle and Grasp-coffee, Being-0 with adjustment significantly outperforms the ablation baseline, achieving success rate gains of over 0.4. This improvement can be attributed to the robot’s ability to terminate navigation in positions that are favorable for grasping. Without adjustment, the robot may stop too far from the object or position the object behind the grasping hand, causing the subsequent grasping skill to fail (see Figure 5).

Placing tasks on the table, including Place-basket and Placecoffee (t), are less sensitive to adjustment. This is because, as long as the robot reaches the table, it can successfully place the object, regardless of its standing pose relative to the table. However, for Place-coffee (m), which requires placing the cup on a coffee machine with a very small available area, Being-0 with adjustment performs significantly better.

These results demonstrate that the proposed adjustment method enhances performance in sequential navigation and manipulation tasks, particularly for manipulation tasks where the success depends heavily on the robot’s initial state relative to the object.

Active Vision. The active camera is a core hardware component of our system, significantly enhancing the robot’s dexterity across various skills. We conduct an ablation study to evaluate the performance of Being-0 when using a fixed camera with different pitch angles, compared with the active

Table 3. Success rates of navigation and manipulation tasks with different active camera configurations. The number following Fixed Cam. denotes the pitch angle set for the camera in the absence of active neck movement.

Navigation Manipulation table coffee machine grasp coffee place coffee

Method

Fixed Cam. (0.3) 5 / 5 5 / 5 0 / 5 0 / 5 Fixed Cam. (0.6) 0 / 5 0 / 5 2 / 5 1 / 5 Fixed Cam. (0.9) 0 / 5 0 / 5 4 / 5 5 / 5 Being-0 (Active Cam.) 5 / 5 5 / 5 5 / 5 5 / 5

camera configuration. Given that the camera’s pitch angle impacts both navigation and manipulation performance, we test the agent with fixed camera setups at various angles. Table 3 presents the results across different tasks.

For navigation tasks, we observe that a small pitch angle (Fixed Cam. (0.3)) yields good performance, while larger pitch angles result in failure. This is because a camera with a large pitch primarily views the ground, causing the agent to lose sight of navigation targets. In contrast, for tabletop manipulation tasks, higher pitch angles improve success rates, as the robot needs to look downward to locate objects on the table.

However, no fixed camera configuration achieves high success rates for both navigation and manipulation tasks. In comparison, Being-0 with an active camera consistently achieves perfect success rates across all tasks. These results underscore the significant advantage of an active camera, enabling the robot to dynamically adapt its field of view to meet the requirements of diverse tasks.

Efficiency. Being-0 demonstrates notable advantages in efficiency, primarily due to the inclusion of the proposed Connector module. To evaluate this, we conduct an ablation study on the task “navigate to the wooden table,” with the results presented in Table 4.

The results indicate that Being-0 with the Connector achieves a 4.2× increase in moving speed compared to the configuration without the Connector, along with a perfect success rate of 5/5. In contrast, the agent without the Connector consistently fails to reach the distant target. This is because GPT-4o alone frequently makes errors in planning locomotion directions, leading to inefficient or incorrect navigation paths. These findings highlight the critical role of the Connector module in enhancing the efficiency of the Being-0 framework.

##### 5.4. Robustness and Scalability

Navigation. To assess the robustness of Being-0 in navigation, we test it across various scene configurations and tasks. The results, shown in Table 5, demonstrate that Being-0

- Table 4. Ablation study on the efficiency of Being-0 in navigation. The table reports the average moving speed (cm/s) and success rates for various agent configurations.

Method Avg. Speed Success

w/o Connector 2.3 0 / 5 Fixed Cam. (0.3) 8.5 5 / 5 Being-0 9.6 5 / 5

- Table 5. Navigation performance across various scene configurations and target locations.

Task Success

In-room 1.00 In-room with obstacles 0.80 Cross-room 0.83

consistently achieves high success rates across all settings. For navigation to targets within the same room, Being-0 achieves a perfect success rate of 1.0. When adapting to unseen layouts with obstacles, it maintains strong performance with a slight drop of 0.2 in success rate. Additionally, Being-0 successfully handles cross-room navigation tasks, achieving a high success rate of 0.83. These tasks require multi-step reasoning and planning by the FM. For example, to locate the reception table, the robot must first identify and navigate to the room’s exit before proceeding further.

Manipulation Skills. Table 6 presents the performance of manipulation skills across various settings. The success rate shows a slight decline when handling unseen objects or encountering visual perturbations, demonstrating the robustness and generalizability of the learned manipulation policies.

Furthermore, the same framework used for acquiring manipulation skills can be extended to dexterous hands equipped with tactile sensors (the Play-chess task in Table 6) and tasks requiring precise manipulation of small objects (see Figure 6), demonstrating the scalability of the skill library to

Table 6. Performance of manipulation skills across different scene configurations, including seen objects, unseen objects, and scenarios with visual perturbations. * denotes the use of dexterous hands equipped with tactile sensors.

Task Seen Obj.

Unseen Obj.

Perturb.

Grasp-bottle 0.86 0.63 0.77 Handout-snack 0.90 1.00 0.80 Place-pole 0.90 -- 0.80 Play-chess* 0.90 -- 0.90

support more complex and challenging manipulation tasks.

### 6. Related Work

Humanoid Locomotion and Manipulation. Humanoid robots (Goswami & Vadakkepat, 2018; Gu et al., 2025) are considered an ideal morphology for human-designed environments, where locomotion and manipulation are fundamental skills. Early approaches focused on locomotion using optimal control (Miura & Shimoyama, 1984; Dariush et al., 2008; Wensing et al., 2023), while recent advances have successfully trained locomotion policies with reinforcement learning (RL) and sim-to-real techniques (Ha et al., 2024), achieving robust walking on flat ground (Xie et al., 2020), complex terrains (Siekmann et al., 2021; Radosavovic et al., 2024; Li et al., 2024b), and advanced parkour skills (Zhuang et al., 2024). For manipulation, while RL-based methods (Yuan et al., 2024; Huang et al., 2024) suffer from the significant sim-to-real gap, imitation learning with teleoperation data has been a dominant approach due to its simplicity and effectiveness. Research has explored diverse teleoperation schemes, leveraging VR (Cheng et al., 2024b), exoskeletons (Fu et al., 2024b; Yang et al., 2024), or cameras (Fu et al., 2024a). Improved imitation learning methods, such as Diffusion Policies (Chi et al.,

- 2023; Ze et al., 2024b;a) and ACT (Zhao et al., 2023), have further advanced training performance. Recently, wholebody control (Fu et al., 2024a; He et al., 2024a;b; Ji et al.,

2024) has gained attention for integrating locomotion and manipulation within a single policy. However, this remains challenging due to the combined complexities of both fields.

Embodied Agents (Firoozi et al., 2023; Hu et al., 2023) for robotics require not only low-level skills but also strong capabilities in common-sense reasoning for highlevel decision-making. Recent research has explored two primary approaches to building embodied agents with Foundation Models (FMs). The first approach directly applies existing FMs, pre-trained on Internet-scale datasets, to robotic tasks without additional fine-tuning. These models leverage their strong general-purpose vision-language understand-

ing capabilities for embodied tasks such as planning (Ahn et al., 2022; Yuan et al., 2023; Chen et al., 2024; Kannan et al., 2024) and reasoning (Huang et al., 2022; Zhang et al., 2023a; Liu et al., 2024b). These methods typically rely on a predefined skill library for low-level execution. The second approach focuses on training robotic FMs using extensive robotic datasets. Notable efforts include Robotic Transformers (Brohan et al., 2022; 2023), vision-languageaction (VLA) models (Jiang et al., 2022; Kim et al., 2024; Team et al., 2024; Liu et al., 2024a; Black et al., 2024; Cheang et al., 2024), and video-language planning models (Yang et al., 2023; Du et al., 2023). While these methods have shown promise for robot arms with grippers, the lack of large-scale datasets for humanoid robots – particularly those with dexterous hands and active cameras – remains a significant barrier to developing FMs for humanoid robots.

Vision-Language Models (VLMs) build upon the remarkable success of Large Language Models (Achiam et al., 2023) to develop capabilities in multi-modal understanding and reasoning. Recent advancements include the development of text-image VLMs (Alayrac et al., 2022; Chen et al., 2023; Li et al., 2023; Bai et al., 2023; Liu et al., 2023) and text-video VLMs (Zhang et al., 2023b; Shu et al., 2023; Maaz et al., 2024; Jin et al., 2024). In this work, we utilize the open-source VideoLLaMA2 (Cheng et al., 2024c) to train the Connector module within the humanoid agent, enhancing efficiency and grounding decision-making for embodied tasks.

### 7. Conclusion and Limitations

In this work, we introduced Being-0, a hierarchical agent framework for humanoid robots, designed to control a humanoid equipped with dexterous hands and active vision to solve long-horizon embodied tasks. The novel VLM-based Connector module effectively bridges the gap between the high-level Foundation Model and low-level skills, significantly enhancing the performance and efficiency of the humanoid agent. Extensive real-world experiments demonstrate Being-0’s strong capabilities in navigation, manipulation, and long-horizon task-solving. The results highlight the effectiveness of the proposed Connector, the adjustment method for coordinating navigation and manipulation, and the use of active vision.

Despite these advancements, the current system does not incorporate complex locomotion skills such as crouching, sitting, or jumping. These skills could extend the humanoid’s functionality beyond flat-ground settings, enabling tasks like climbing stairs, working from seated positions, or manipulating objects at varying heights. Enhancing these capabilities will be an important direction for future work. Additionally, while the onboard system is efficient, Being-0 still relies on the slow Foundation Model for high-level decision-making.

Future research could explore lightweight Foundation Models tailored for robotics applications to further improve the system’s efficiency.

### Impact Statement

This work explores advancements in humanoid robotic agents, which come with specific safety concerns. The use of Foundation Models and skill libraries introduces the potential risks of predicting incorrect skills or executing actions in out-of-distribution scenarios. For full-sized humanoid robots, such errors could lead to damage to surroundings or harm to people. At present, these systems should be tested only in controlled, experimental environments to ensure safety. Future work should prioritize robust error handling, fail-safes, and ethical guidelines to mitigate these risks and enable safer deployment of humanoid agents.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Cheang, C.-L., Chen, G., Jing, Y., Kong, T., Li, H., Li, Y., Liu, Y., Wu, H., Xu, J., Yang, Y., et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

Chen, A. S., Lessing, A. M., Tang, A., Chada, G., Smith, L., Levine, S., and Finn, C. Commonsense reasoning for legged robot adaptation with vision-language models. arXiv preprint arXiv:2407.02666, 2024.

Chen, J., Zhu, D., Shen, X., Li, X., Liu, Z., Zhang, P., Krishnamoorthi, R., Chandra, V., Xiong, Y., and Elhoseiny, M. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.

Cheng, X., Ji, Y., Chen, J., Yang, R., Yang, G., and Wang, X. Expressive whole-body control for humanoid robots. arXiv preprint arXiv:2402.16796, 2024a.

Cheng, X., Li, J., Yang, S., Yang, G., and Wang, X. Opentelevision: Teleoperation with immersive active visual feedback. arXiv preprint arXiv:2407.01512, 2024b.

Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., and Bing, L. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv: 2406.07476, 2024c.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2023.

Dariush, B., Gienger, M., Jian, B., Goerick, C., and Fujimura, K. Whole body humanoid control from human motion descriptors. In 2008 IEEE International Conference on Robotics and Automation, pp. 2677–2684. IEEE, 2008.

Du, Y., Yang, M., Florence, P., Xia, F., Wahid, A., Ichter, B., Sermanet, P., Yu, T., Abbeel, P., Tenenbaum, J. B., et al. Video language planning. arXiv preprint arXiv:2310.10625, 2023.

Feng, Y., Li, Y., Zhang, W., Zheng, S., and Lu, Z. Videoorion: Tokenizing object dynamics in videos. arXiv preprint arXiv: 2411.16156, 2024.

Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, pp. 02783649241281508, 2023.

Fu, Z., Zhao, Q., Wu, Q., Wetzstein, G., and Finn, C. Humanplus: Humanoid shadowing and imitation from humans. arXiv preprint arXiv:2406.10454, 2024a.

Fu, Z., Zhao, T. Z., and Finn, C. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024b.

Goswami, A. and Vadakkepat, P. Humanoid robotics: a reference. Springer Publishing Company, Incorporated, 2018.

Gu, Z., Li, J., Shen, W., Yu, W., Xie, Z., McCrory, S., Cheng, X., Shamsah, A., Griffin, R., Liu, C. K., et al. Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning. arXiv preprint arXiv:2501.02116, 2025.

Ha, S., Lee, J., van de Panne, M., Xie, Z., Yu, W., and Khadiv, M. Learning-based legged locomotion: State of the art and future perspectives. The International Journal of Robotics Research, 2024.

He, T., Luo, Z., He, X., Xiao, W., Zhang, C., Zhang, W., Kitani, K., Liu, C., and Shi, G. Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. arXiv preprint arXiv:2406.08858, 2024a.

He, T., Xiao, W., Lin, T., Luo, Z., Xu, Z., Jiang, Z., Kautz, J., Liu, C., Shi, G., Wang, X., et al. Hover: Versatile neural whole-body controller for humanoid robots. arXiv preprint arXiv:2410.21229, 2024b.

Hu, Y., Xie, Q., Jain, V., Francis, J., Patrikar, J., Keetha, N., Kim, S., Xie, Y., Zhang, T., Fang, H.-S., et al. Toward general-purpose robots via foundation models: A survey and meta-analysis. arXiv preprint arXiv:2312.08782, 2023.

Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

Huang, Z., Yuan, H., Fu, Y., and Lu, Z. Efficient residual learning with mixture-of-experts for universal dexterous grasping. arXiv preprint arXiv:2410.02475, 2024.

Ji, M., Peng, X., Liu, F., Li, J., Yang, G., Cheng, X., and Wang, X. Exbody2: Advanced expressive humanoid whole-body control. arXiv preprint arXiv:2412.13196, 2024.

Jiang, Y., Gupta, A., Zhang, Z., Wang, G., Dou, Y., Chen, Y., Fei-Fei, L., Anandkumar, A., Zhu, Y., and Fan, L. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2(3):6, 2022.

Jin, Y., Sun, Z., Xu, K., Xu, K., Chen, L., Jiang, H., Huang, Q., Song, C., Liu, Y., Zhang, D., Song, Y., Gai, K., and Mu, Y. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. In ICML, 2024.

Kannan, S. S., Venkatesh, V. L., and Min, B.-C. Smartllm: Smart multi-agent robot task planning using large language models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., Bernstein, M. S., and Li, F.-F. Visual genome: Connecting language and vision using crowdsourced dense image annotations. arXiv preprint arXiv: 1602.07332, 2016.

Li, J., Li, D., Savarese, S., and Hoi, S. C. H. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

Li, J., Zhu, Y., Xie, Y., Jiang, Z., Seo, M., Pavlakos, G., and Zhu, Y. Okami: Teaching humanoid robots manipulation skills through single video imitation. In 8th Annual Conference on Robot Learning, 2024a.

Li, Z., Peng, X. B., Abbeel, P., Levine, S., Berseth, G., and Sreenath, K. Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control. The International Journal of Robotics Research, pp. 02783649241285161, 2024b.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, 2023.

Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024a.

Liu, S., Yuan, H., Hu, M., Li, Y., Chen, Y., Liu, S., Lu, Z., and Jia, J. Rl-gpt: Integrating reinforcement learning and code-as-policy. arXiv preprint arXiv:2402.19299, 2024b.

Maaz, M., Rasheed, H. A., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.

Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Rudin, N., Allshire, A., Handa, A., et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.

Miura, H. and Shimoyama, I. Dynamic walk of a biped. The International Journal of Robotics Research, 3(2):60–74, 1984.

Radosavovic, I., Xiao, T., Zhang, B., Darrell, T., Malik, J., and Sreenath, K. Real-world humanoid locomotion with reinforcement learning. Science Robotics, 9(89): eadi9579, 2024.

Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., and Li, H. Visual cot: Advancing multimodal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. arXiv preprint arXiv: 2403.16999, 2024.

Shu, F., Zhang, L., Jiang, H., and Xie, C. Audiovisual llm for video understanding. arXiv preprint arXiv:2312.06720, 2023.

Siekmann, J., Green, K., Warila, J., Fern, A., and Hurst, J. Blind bipedal stair traversal via sim-to-real reinforcement learning. arXiv preprint arXiv:2105.08328, 2021.

Tan, W., Zhang, W., Xu, X., Xia, H., Ding, G., Li, B., Zhou, B., Yue, J., Jiang, J., Li, Y., et al. Cradle: Empowering foundation agents towards general computer control. In NeurIPS 2024 Workshop on Open-World Agents, 2024.

Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Tian, Y., Ma, T., Xie, L., Qiu, J., Tang, X., Zhang, Y., Jiao, J., Tian, Q., and Ye, Q. Chatterbox: Multi-round multimodal referring and grounding. arXiv preprint arXiv: 2401.13307, 2024.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 2024.

Wensing, P. M., Posa, M., Hu, Y., Escande, A., Mansard, N., and Del Prete, A. Optimization-based control for dynamic legged robots. IEEE Transactions on Robotics, 2023.

Xie, Z., Clary, P., Dao, J., Morais, P., Hurst, J., and Panne, M. Learning locomotion skills for cassie: Iterative design and sim-to-real. In Conference on Robot Learning, pp. 317–329. PMLR, 2020.

Yang, M., Du, Y., Ghasemipour, K., Tompson, J., Schuurmans, D., and Abbeel, P. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.

Yang, S., Liu, M., Qin, Y., Ding, R., Li, J., Cheng, X., Yang, R., Yi, S., and Wang, X. Ace: A cross-platform visualexoskeletons system for low-cost dexterous teleoperation. arXiv preprint arXiv:2408.11805, 2024.

Yuan, H., Zhang, C., Wang, H., Xie, F., Cai, P., Dong, H., and Lu, Z. Skill reinforcement learning and planning for open-world long-horizon tasks. arXiv preprint arXiv:2303.16563, 2023.

Yuan, H., Zhou, B., Fu, Y., and Lu, Z. Cross-embodiment dexterous grasping with reinforcement learning. arXiv preprint arXiv:2410.02479, 2024.

Ze, Y., Chen, Z., Wang, W., Chen, T., He, X., Yuan, Y., Peng, X. B., and Wu, J. Generalizable humanoid manipulation with improved 3d diffusion policies. arXiv preprint arXiv:2410.10803, 2024a.

Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., and Xu, H. 3d diffusion policy. arXiv preprint arXiv:2403.03954, 2024b.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. IEEE International Conference on Computer Vision, 2023. doi: 10.1109/ICCV51070.2023.01100.

Zhang, C., Cai, P., Fu, Y., Yuan, H., and Lu, Z. Creative agents: Empowering agents with imagination for creative tasks. arXiv preprint arXiv:2312.02519, 2023a.

Zhang, H., Li, X., and Bing, L. Video-llama: An instructiontuned audio-visual language model for video understanding. In EMNLP, 2023b.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

Zhou, B., Yuan, H., Fu, Y., and Lu, Z. Learning diverse bimanual dexterous manipulation skills from human demonstrations. arXiv preprint arXiv:2410.02477, 2024.

Zhuang, Z., Yao, S., and Zhao, H. Humanoid parkour learning. arXiv preprint arXiv:2406.10759, 2024.

### A. Additional Results

In this section, we provide detailed additional results from our experiments. Video recordings are included in the supplementary material.

##### A.1. First-Person Video Records

|[Figure 36]<br><br>|[Figure 37]<br><br>|[Figure 38]<br><br>|[Figure 39]|
|---|---|---|---|

w/o Connector

“Head_down” “Grasp_cup” Locomotion Locomotion Grasp_cup

“Head_up”

FM

Skill

|[Figure 40]<br><br>|[Figure 41]|[Figure 42]<br><br>|[Figure 43]|[Figure 44]|[Figure 45]<br><br>|[Figure 46]|[Figure 47]|
|---|---|---|---|---|---|---|---|

Being-0

“Grasp a coffee cup.” “Find a coffee machine.” Navigation: table Adjustment Manipulation Navigation: Coffee Machine Locomotion Grasp_Cup Locomotion

###### FM

Connector

Skill

- Figure 3. A comparison of Being-0 w/o Connector and Being-0 in the long-horizon task “Prepare-coffee.” The first row shows recordings of Being-0 without the Connector, while the second row shows recordings of Being-0 with the Connector. Being-0 w/o Connector frequently queries the FM, which often fails to provide correct plans due to its limited embodied scene understanding. In contrast, Being-0 with the Connector completes the task, requiring only a few queries to the FM.

Navigation: coffee machine Manipulation: grasp coffee

Fixed Cam. (0.3)

Fixed Cam. (0.6)

Fixed Cam. (0.9)

Being-0 (Active Cam.)

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

- Figure 4. Recordings from the ablation study on the active camera. Each row represents a different camera configuration, with the left three images depicting the navigation task and the right three images depicting the manipulation task. Only Being-0 with an active camera achieves robust performance in both navigation and manipulation.

###### w/o Adjustment Being-0

navigation

manipulation end navigation manipulation end

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

Grasp-bottle

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

Place-coffee (t)

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

Place-coffee (m)

- Figure 5. A comparison of Being-0 with and without the adjustment method in two-stage tasks involving navigation and manipulation. Each row corresponds to a specific task, with the left three images showing results for Being-0 w/o Adjustment and the right three images showing results for Being-0. Without adjustment, the agent may terminate navigation in improper poses, leading to failed manipulations.

Play-chess

Toy-bricks

Open-beer

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

Handout-snack

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

Grasp-bottle

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

Place-pole

- Figure 6. First-person view recordings of the learned manipulation skills. Each row corresponds to a specific skill, with images from left to right depicting the progression of the manipulation process.

##### A.2. Foundation Model Planning Traces

Information Gathering Image_Description:

|[Figure 114]|
|---|

- 1. The image shows a room with a grey pattern carpet. There are chairs and a table visible in the upper part of the image. The floor has a distinct pattern of grey tones.
- 2. Based on the grey pattern carpet and the presence of chairs and a table, this room matches the description of the Robot Laboratory Room.
- 3. The target task is to get a cup from the table and get coffee from the machine. The table and chairs are visible, but the cup and coffee machine are not in the current frame.
- 4. The target task is not yet successful as the cup and coffee machine are not visible in the current frame. Target_name: null Reasoning_for_target: The current frame does not show the cup or the coffee machine. There is no new target to detect in this image. Area_location: Robot Laboratory Room

Self-reflection None

Task Inference

History_summary:

The task to get a cup from the table and get coffee from the machine has not been completed. The robot is currently in the Robot Laboratory Room, which has a grey pattern carpet. The previous subtask was to move towards the wooden table with a cup, but it has not been successfully completed as the cup is not yet in the robot's hand. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to locate the wooden table with the cup in the Robot Laboratory Room.
- 3. The previous subtask was to move towards the wooden table with a cup, but it was not successfully finished.
- 4. The robot was previously in the Robot Laboratory Room and is still in the same room.
- 5. The robot needs to find the wooden table with the cup in the current room.
- 6. The robot should search the current room to locate the wooden table with the cup.
- 7. If the wooden table with the cup is found, the robot should then move towards it.
- 8. If the wooden table with the cup is not found, the robot should continue searching the room. Subtask_description: The current subtask is to search the Robot Laboratory Room to locate the wooden table with the cup.

Action Planning Action: ```python search_and_move_towards(target_name="wooden table") ```

|[Figure 115]|
|---|

Information Gathering Image_Description:

- 1. The image shows a wooden table with a single cup placed on it. In the background, there are desks with computers, chairs, and a black divider. The floor has a grey pattern carpet.
- 2. Based on the semantic map and the details in the image, you are currently in the Robot Laboratory Room.
- 3. Yes, you have reached the target area, and the target task of locating the wooden table with the cup is already successful.
- 4. Yes, you have reached the target area, and the target task of locating the wooden table with the cup is already successful.
- 5. The cup is positioned on the wooden table, which is in the foreground of the image. The table is oriented horizontally across the image.
- 6. The target is in the current image frame, and the task of locating the wooden table with the cup is complete.
- 7. The target object (the cup) is in the current image frame, and its position is on the wooden table.
- 8. There are no new possible targets in the image frame that are more relevant than the current target. Target_name: cup Reasoning_for_target: The cup is the target object needed to complete the task of getting coffee from the machine. It is already located on the wooden table in the Robot Laboratory Room. Area_location: Robot Laboratory Room

###### Self-reflection

Self_Reflection_Reasoning:

- 1. The last executed action was `search_and_move_towards(target_name="wooden table")`.
- 2. Based on the images, the robot has successfully moved towards the wooden table, as the table with the cup is now visible in the frame.
- 3. The robot was previously in the Robot Laboratory Room, and it is still in the same room, now closer to the wooden table. Success_Detection: The overall task is not yet successful. The robot has successfully moved towards the table with the cup, but it still needs to grab the cup and get coffee from the machine.

Figure 7. Planning traces of the Foundation Model in Being-0 for the task “Prepare-coffee.”

|[Figure 116]|
|---|

Task Inference History_summary:

The task to get a cup from the table and get coffee from the machine has not been completed. The robot is currently in the

Robot Laboratory Room, which has a grey pattern carpet. The previous subtask was to move towards the wooden table with a cup, and it has been successfully completed as the table with the cup is now visible in the frame. The robot needs to grab the cup from the table next. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to grab the cup from the table.
- 3. The previous subtask was to move towards the wooden table with a cup, and it has been successfully completed.
- 4. The robot was previously in the Robot Laboratory Room and is still in the same room, now closer to the wooden table.
- 5. The robot needs to grab the cup from the table.
- 6. The robot should grab the cup from the table.
- 7. If the cup is successfully grabbed, the robot should then search for the coffee machine.
- 8. If the cup is not successfully grabbed, the robot should attempt to grab it again. Subtask_description: The current subtask is to grab the cup from the table.

###### Action Planning

Action:

```python grab(object_name="cup") ```

|[Figure 117]|
|---|

Information Gathering Image_Description:

- 1. The image shows a wooden table with a cup on it. The cup is located near the bottom right corner of the table. In the background, there are chairs, desks, and a black divider. The floor has a grey pattern carpet.
- 2. Based on the semantic map and the details in the image, you are currently in the Robot Laboratory Room.
- 3. The target area has been reached, and the target task of locating the cup is successful.
- 4. The cup is positioned near the bottom right corner of the table in the image. Target_name: cup Reasoning_for_target: The cup is the target object needed to complete the current subtask of grabbing the cup from the table. Area_location: Robot Laboratory Room

Self-reflection Self_Reflection_Reasoning:

- 1. The last executed action was `grab(object_name="cup")`.
- 2. Based on the images, the robot's hand is now holding the cup, indicating that the cup has been successfully grabbed from the table.
- 3. The robot is still in the Robot Laboratory Room, which is the correct location for the task. Success_Detection: The overall task is not yet successful. The robot has successfully grabbed the cup, but it still needs to get coffee from the machine.

Task Inference History_summary: The summary is that the robot is tasked with getting a cup from the table and getting coffee from the machine. The robot successfully moved towards the wooden table with the cup and grabbed the cup from the table. The robot is currently in the Robot Laboratory Room, holding the cup, and needs to find the coffee machine to complete the task. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to find the coffee machine.
- 3. The previous subtask was to grab the cup from the table, which has been successfully completed.
- 4. The robot was previously in the Robot Laboratory Room and is still in the same room.
- 5. The robot needs to search for the coffee machine in the Robot Laboratory Room.
- 6. The robot should search for the coffee machine in the current room.
- 7. If the coffee machine is found, the robot should move towards it.
- 8. If the coffee machine is not found, the robot should continue searching in the current room. Subtask_description: The current subtask is to search for the coffee machine in the Robot Laboratory Room.

Action Planning Action: ```python search_and_move_towards(target_name="coffee machine") ```

Figure 8. (Continued) Planning traces of the Foundation Model in Being-0 for the task “Prepare-coffee.”

Information Gathering Image_Description:

|[Figure 118]|
|---|

- 1. The image shows a room with grey pattern carpet flooring. There is a black divider, a wooden table with a cup, some machines, and a workspace with computers and desks. There is also a water dispenser and some cables on the floor.
- 2. Based on the semantic map and the details in the image, you are currently in the Robot Laboratory Room.
- 3. The target task of finding the coffee machine has not yet been successful.
- 4. The coffee machine is not visible in the current image frame.
- 5. The cup is visible in the lower right corner of the image, indicating that you are holding it. Target_name: coffee machine Reasoning_for_target: The current task is to search for the coffee machine in the Robot Laboratory Room. The coffee machine is the most relevant object to complete this task. Area_location: Robot Laboratory Room

Self-reflection Self_Reflection_Reasoning:

- 1. The last executed action was `search_and_move_towards(target_name="coffee machine")`.
- 2. Based on the images, the robot has successfully grabbed the cup from the table. The current image shows the robot holding the cup and moving towards the coffee machine.
- 3. The robot is still in the Robot Laboratory Room, as indicated by the grey pattern carpet and the presence of the coffee machine in the background.
- 4. The last executed action was successful. The robot has moved towards the coffee machine while holding the cup, as evidenced by the sequential frames showing the robot's movement and the cup in its grasp. Success_Detection: The overall task is not yet successful. The robot has successfully grabbed the cup and moved towards the coffee machine, but it still needs to get coffee from the machine. The next step should be to approach the coffee machine and get coffee.

Task Inference History_summary: The summary is that the robot is tasked with getting a cup from the table and getting coffee from the machine. The robot successfully moved towards the wooden table with the cup and grabbed the cup from the table. The robot is currently in the Robot Laboratory Room, holding the cup, and needs to find the coffee machine to complete the task. The robot has moved towards the coffee machine while holding the cup, as evidenced by the sequential frames showing the robot's movement and the cup in its grasp.

Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to get coffee from the machine.
- 3. The previous subtask was to search for the coffee machine in the Robot Laboratory Room.
- 4. The robot is still in the Robot Laboratory Room, as indicated by the grey pattern carpet and the presence of the coffee machine in the background.
- 5. The previous subtask was successfully finished as the robot has moved towards the coffee machine while holding the cup.
- 6. The robot is very close to the coffee machine.
- 7. The next subtask should be to put the cup on the coffee machine to get coffee. Subtask_description: The current subtask is to put the cup on the coffee machine.

Action Planning Action: ```python search_and_move_towards(target_name="coffee machine")

```

|[Figure 119]|
|---|

Information Gathering Image_Description:

- 1. The image shows a coffee machine with a digital display screen. The screen displays various coffee options. Below the screen, there is a dispensing area with a cup holder. A cup is already placed on the cup holder.
- 2. Based on the presence of the coffee machine and the description of the Robot Laboratory Room, it is likely that this image was taken in the Robot Laboratory Room.
- 3. The target task of putting the cup on the coffee machine is already successful as the cup is placed on the cup holder.
- 4. The cup is positioned correctly on the coffee machine, ready to receive coffee. Target_name: null Reasoning_for_target: The cup is already placed on the coffee machine, so there is no need to detect or use a new target. Area_location: Robot Laboratory Room

###### Self-reflection

Self_Reflection_Reasoning:

- 1. The last executed action was `search_and_move_towards(target_name="coffee machine")`.
- 2. Based on the images, the robot has successfully moved towards the coffee machine. The coffee machine is now visible in the second image frame, and the cup is still in the robot's hand.
- 3. The robot has reached the target location, which is the coffee machine, as indicated by the presence of the coffee machine in the second image frame. Success_Detection: The overall task is not yet successful. The robot has successfully grabbed the cup and moved towards the coffee machine, but it still needs to get coffee from the machine. The next step should be to approach the coffee machine and get coffee.

Task Inference History_summary:

|[Figure 120]|
|---|

The summary is that the robot is tasked with getting a cup from the table and getting coffee from the machine. The robot

successfully moved towards the wooden table with the cup and grabbed the cup from the table. The robot is currently in the Robot Laboratory Room, holding the cup, and has successfully moved towards the coffee machine. The coffee machine is now visible in the robot's view, and the cup is still in the robot's hand. The next step is to put the cup on the coffee machine to get coffee. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to get coffee from the machine.
- 3. The previous subtask was to put the cup on the coffee machine.
- 4. The robot is currently in the Robot Laboratory Room, very close to the coffee machine

.5. The previous subtask was successfully finished as the robot has moved towards the coffee machine while holding the cup.

- 6. The next subtask should be to put the cup on the coffee machine to get coffee.
- 7. The robot is very close to the coffee machine, and the cup is visible in the robot's hand.
- 8. The next subtask should be to put the cup on the coffee machine to get coffee. Subtask_description: The current subtask is to put the cup on the coffee machine.

###### Action Planning

Action: ```python put(object_name="cup") ```

|[Figure 121]|
|---|

Information Gathering Image_Description:

- 1. The image shows a coffee machine on a table. The machine has a digital display at the top showing various coffee options. Below the display, there is a dispenser area with a cup placed on the drip tray. The table is white, and there are some cables visible on the floor to the left. A robotic arm is visible in the lower right corner of the image, holding a cup.
- 2. Based on the semantic map and the presence of the coffee machine, the current location is the Robot Laboratory Room.
- 3. The target area has been reached, and the task of putting the cup on the coffee machine is already successful as the cup is placed on the drip tray.
- 4. The cup is positioned correctly on the coffee machine's drip tray, ready to receive coffee. Target_name: null Reasoning_for_target: The current subtask of placing the cup on the coffee machine is already completed successfully, as evidenced by the cup being correctly positioned on the drip tray. Area_location: Robot Laboratory Room

###### Self-reflection

Self_Reflection_Reasoning:

- 1. The last executed action was `put(object_name="cup")`.
- 2. Based on the images, the cup is correctly placed on the coffee machine's cup holder, indicating that the current subtask of placing the cup on the coffee machine is complete.
- 3. The robot is already in the Robot Laboratory Room, which is the correct location for the task.
- 4. The last executed action was successful because the cup is now positioned correctly on the coffee machine, ready to receive coffee. Success_Detection: The overall task is not yet successful. The robot has successfully placed the cup on the coffee machine, but it still needs to get coffee from the machine. The next step should be to interact with the coffee machine to get coffee.

Task Inference History_summary: The summary is that the robot is tasked with getting a cup from the table and getting coffee from the machine. The robot successfully moved towards the wooden table with the cup and grabbed the cup from the table. The robot is currently in the

Robot Laboratory Room, holding the cup, and has successfully moved towards the coffee machine. The coffee machine is now

visible in the robot's view, and the cup is still in the robot's hand. The next step is to put the cup on the coffee machine to get coffee. The robot has successfully placed the cup on the coffee machine, and the cup is positioned correctly on the coffee machine, ready to receive coffee. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to get coffee from the machine.
- 3. The previous subtask was to put the cup on the coffee machine.
- 4. The previous subtask was successfully finished as the cup is now positioned correctly on the coffee machine, ready to receive coffee.
- 5. The robot is currently in the Robot Laboratory Room, which is the correct location for the task.
- 6. The next subtask should be to interact with the coffee machine to get coffee.
- 7. The robot is very close to the coffee machine, and the cup is already placed on the cup holder.
- 8. The next subtask should be to select a coffee option on the coffee machine's digital display to get coffee. Subtask_description: The current subtask is to select a coffee option on the coffee machine's digital display to get coffee.

###### Action Planning

Action: ```python click(target="espresso button") ```

Information Gathering Image_Description:

|[Figure 122]|
|---|

- 1. The image shows a room with grey pattern carpet flooring. There are several desks with computers and office supplies on them. There are also some open cardboard boxes on the floor, a chair, and some scattered items. The room appears to be a workspace with various equipment and supplies.
- 2. Based on the semantic map and the details in the image, this room is the Robot Laboratory Room.
- 3. The target task of getting a cup from the table has not been completed yet.
- 4. The target task of getting a cup from the table has not been completed yet.
- 5. There is no visible cup on the table in the current image frame.
- 6. The target location is the table in the Robot Laboratory Room where a cup might be found.
- 7. The target object (cup) is not visible in the current image frame. The table in the Robot Laboratory Room should be checked for the cup.
- 8. No new possible targets are observed in the current image frame. Target_name: cup Reasoning_for_target: The task is to get a cup from the table. The cup is the relevant object needed to complete the current task. Area_location: Robot Laboratory Room

###### Self-reflection

None Task Inference History_summary: The task of getting a cup from the table and getting coffee from the machine has not been completed. The robot is currently in the Robot Laboratory Room, where the coffee machine is located. The previous subtask was to search for the cup on the wooden table, but it has not been completed yet. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to locate the cup on the wooden table.
- 3. The previous subtask was to search for the cup on the wooden table, but it has not been completed.
- 4. The robot is still in the Robot Laboratory Room, where the wooden table and coffee machine are located.
- 5. The robot needs to search for the cup on the wooden table.
- 6. The robot should search the area around the wooden table to find the cup.
- 7. The robot should continue with the previous subtask as it is still relevant.
- 8. The robot should search the area around the wooden table to find the cup. Subtask_description: The current subtask is to search the area around the wooden table to find the cup.

Action Planning Decision_Making_Reasoning:

- 1. ""False"" means the overall task was not successful.
- 2. The target location is the table in the Robot Laboratory Room.
- 3. The target (cup) is not visible in the current image.
- 4. I am already in the Robot Laboratory Room.
- 5. The previous action was not a search.
- 6. The current image is not the same as the one from the previous iteration.
- 7. The task of getting a cup from the table and getting coffee from the machine has not been completed. The robot is currently in the Robot Laboratory Room, where the coffee machine is located. The previous subtask was to search for the cup on the wooden table, but it has not been completed yet.
- 8. The last executed action was not a search, so I need to search the area for the cup.
- 9. I am already in the location required for the task.
- 10. The target (cup) is not visible in the current frame, so I need to search for it.
- 11. I should search the place after any movement forward towards a target that was a hallway.
- 12. I need to search the area around the wooden table to find the cup.
- 13. I have the skill to search the area, so I do not need to request help.
- 14. The most related skill to conduct the next step of the current subtask is to search the area.
- 15. Based on the action rules and self-reflection, the most suitable action in the skill library for the next step is to turn left or right to search the area around the wooden table. Actions: ```python turn_left(""small"") ``` Key_reason_of_last_action: The key reason for choosing this action is to search the area around the wooden table to find the cup, as the target (cup) is not visible in the current frame.

Figure 11. Planning traces of the Foundation Model in Being-0 w/o Connector for the task “Prepare-coffee.”

|[Figure 123]|
|---|

Information Gathering Image_Description:

- 1. The image shows a room with grey pattern carpet flooring. There are several desks with computers and monitors on the left side of the image. There is a white drawer unit next to the desks. In the center of the image, there is a black office chair and some scattered boxes and items on the floor. On the right side, there are more boxes and a black object that appears to be a helmet or similar item. The room has dark walls and ceiling.
- 2. Based on the semantic map and the description of the room, you are currently in the Robot Laboratory Room.
- 3. The target area, which is the wooden table, is not visible in the current image frame.
- 4. The task of finding the cup on the wooden table has not been completed yet. Target_name :null Reasoning_for_target: The wooden table, which is the target area to find the cup, is not visible in the current image frame. Therefore, there is no need to detect a new target at this moment. Area_location: Robot Laboratory Room

Self-reflection Self_Reflection_Reasoning:

- 1. The last executed action was `turn_left(""small"")`.
- 2. Based on the images, the robot has turned left but has not yet located the cup on the wooden table. The target object, a cup, is not visible in the current image frame.
- 3. The robot is still in the Robot Laboratory Room, as indicated by the grey pattern carpet and the surrounding environment.The robot has not yet completed the subtask of finding the cup on the wooden table.
- 4. The last executed action was partially successful. The robot did turn left, but it did not achieve the desired outcome of locating the cup on the table.
- 5. The most probable cause for the partial success is that the turn range was too small to bring the cup into view. The robot needs to adjust its position or turn further to locate the cup. Success_Detection: The overall task was unsuccessful because the subtask of getting a cup from the table has not been completed. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room.

Task Inference History_summary: The summary is the task of getting a cup from the table and getting coffee from the machine has not been completed. The robot is currently in the Robot Laboratory Room, where the coffee machine is located. The previous subtask was to search for the cup on the wooden table, but it has not been completed yet. The last executed action was `turn_left(""small"")`, which resulted in an error due to an invalid parameter. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room. The robot has turned left but has not yet located the cup on the wooden table. The target object, a cup, is not visible in the current image frame. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to locate the cup on the wooden table.
- 3. The previous subtask was to search for the cup on the wooden table, but it has not been completed.
- 4. The robot is still in the Robot Laboratory Room, where the wooden table and coffee machine are located.
- 5. The robot needs to search for the cup on the wooden table.
- 6. The robot should correct the turn action parameter and continue searching for the cup on the table.
- 7. The robot should search the area around the wooden table to find the cup.
- 8. The robot should continue with the previous subtask as it is still relevant. Subtask_description: The current subtask is to turn left with the correct parameter and search the area around the wooden table to find the cup.

Action Planning

Decision_Making_Reasoning:

- 1. The overall task was unsuccessful because the subtask of getting a cup from the table has not been completed. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room.
- 2. The Robot Laboratory Room is located in Hallway Section A.
- 3. The target, which is the cup on the wooden table, is not visible in the current image.
- 4. The robot is already in the Robot Laboratory Room, as indicated by the grey pattern carpet and the surrounding environment.
- 5. The current image is different from the previous image, indicating that the robot has turned left.
- 6. The robot needs to continue searching for the cup on the wooden table.
- 7. The recent history indicates that the robot has turned left but has not yet located the cup on the wooden table.8. The selfreflection for the last executed action indicates that the turn range was too small to bring the cup into view.

- 9. The robot is still in the Robot Laboratory Room and needs to adjust its position or turn further to locate the cup.
- 10. The robot should search the place after any movement forward towards a target that was a hallway.
- 11. The robot should move towards some target that it can see in front of it.
- 12. The most suitable action in the skill library for the next step is to turn left with a larger turn range to bring the cup into view. Actions: ```python turn_left(turn_range=""LARGE"") ``` Key_reason_of_last_action: The key reason for choosing this action is to adjust the robot's position by turning left with a larger turn range to bring the cup on the wooden table into view and continue searching for the cup in the Robot Laboratory Room.

- Figure 12. (Continued) Planning traces of the Foundation Model in Being-0 w/o Connector for the task “Prepare-coffee.”

Information Gathering Image_Description:

|[Figure 124]|
|---|

- 1. The image shows a room with grey pattern carpet flooring. There is a black office chair in front of a desk with a computer monitor. To the right, there is a wooden table with a cup on it. The background has a black divider with the label “xxx” in blue.
- 2. Based on the semantic map and the details in the image, this is the Robot Laboratory Room.
- 3. The target area has been reached, and the target task of locating the cup on the wooden table is successful.
- 4. The cup is located on the wooden table to the right side of the image. Target_name: cup Reasoning_for_target: The cup is the target object needed to complete the current subtask of getting a cup from the table. Area_location: Robot Laboratory Room

Self-reflection Self_Reflection_Reasoning:

- 1. The last executed action was `turn_left(turn_range=""LARGE"")`.
- 2. Based on the images, the current subtask of finding the cup on the wooden table has not been completed. The wooden table is not visible in the current frame, indicating that the turn did not bring the table into view.
- 3. The robot is still in the Robot Laboratory Room, as indicated by the grey pattern carpet and the presence of desks and computers. The target location (wooden table) is not yet in view.
- 4. The last executed action was not successful. The turn did not bring the wooden table into view, which means the robot's position did not change sufficiently to achieve the desired outcome.
- 5. The most probable cause for the unsuccessful action is that the turn range was too large, causing the robot to overshoot the target area. Success_Detection: The overall task was unsuccessful because the subtask of getting a cup from the table has not been completed. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room.

Task Inference History_summary:

The summary is the task of getting a cup from the table and getting coffee from the machine has not been completed. The

robot is currently in the Robot Laboratory Room, where the coffee machine is located. The previous subtask was to search for the cup on the wooden table, but it has not been completed yet. The last executed action was `turn_left(""LARGE"")`, which resulted in the wooden table not being visible in the current frame. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room. The robot has turned left but has not yet located the cup on the wooden table. The target object, a cup, is not visible in the current image frame. Subtask_reasoning:

- 1. The overall task is to get a cup from the table and get coffee from the machine.
- 2. The next step is to locate the cup on the wooden table.
- 3. The previous subtask was to search for the cup on the wooden table, but it has not been completed.
- 4. The robot is still in the Robot Laboratory Room, where the wooden table and coffee machine are located.
- 5. The robot needs to search for the cup on the wooden table.
- 6. The robot should correct the turn action parameter and continue searching for the cup on the table.
- 7. The robot should search the area around the wooden table to find the cup.
- 8. The robot should continue with the previous subtask as it is still relevant. Subtask_description: The current subtask is to turn right with a smaller turn range and search the area around the wooden table to find the cup.

Action Planning Decision_Making_Reasoning:

1. The overall task was unsuccessful because the subtask of getting a cup from the table has not been completed. The robot needs to correct the turn action parameter and continue searching for the cup on the table in the Robot Laboratory Room.

- 1.1. The target area is the Robot Laboratory Room, and the robot is already in this room.
- 1.2. The previous action did not fail with ""Move failed""; it just did not achieve the desired outcome.
- 3. The target (cup on the wooden table) is visible in the current image.
- 4. The robot is already in the target area (Robot Laboratory Room).

- 6. The current image is not the same as the one from the previous iteration.
- 7. Recent history shows that the robot turned left with a large turn range but did not bring the wooden table into view. The robot is still in the Robot Laboratory Room.
- 8. The self-reflection indicates that the last executed action was not successful because the turn range was too large, causing the robot to overshoot the target area.
- 9. The robot has reached the correct location (Robot Laboratory Room) and the target object (cup) is now visible on the wooden table.
- 10. The position of the target (cup) is on the wooden table to the right side of the image.
- 11. No further search is needed as the target is already visible.
- 12. The robot should move towards the cup on the wooden table.
- 13. The robot has the skill to grab the cup.

- 15. The most related skill to conduct the next step of the current subtask is ""grab()"".
- 16. Based on the action rules and self-reflection, the most suitable action is to grab the cup from the wooden table. Actions: ```python grab(object_name=""cup"") ``` Key_reason_of_last_action: The key reason for choosing this action is that the cup is now visible on the wooden table, and the robot needs to grab it to complete the current subtask.

- Figure 13. (Continued) Planning traces of the Foundation Model in Being-0 w/o Connector for the task “Prepare-coffee.”

##### A.3. Details of Experimental Setup and Results

- Table 7. Detailed sub-processes required to complete each long-horizon task, along with the success rates of Being-0 and the baseline. Task Sub-Process w/o Connector Being-0

Fetch-bottle

Navigate to table. 0 / 5 5 / 5 Grasp cup. 0 / 5 4 / 5

Deliver-basket

Navigate to table. 0 / 5 5 / 5 Place basket. 0 / 5 3 / 5

Prepare-coffee

Navigate to table. 0 / 5 5 / 5 Grasp cup. 0 / 5 4 / 5 Navigate to coffee machine. 0 / 5 3 / 5 Place cup. 0 / 5 3 / 5

Make-coffee

Place cup. 5 / 5 5 / 5 Select coffee. 5 / 5 5 / 5 Select confirmation. 4 / 5 4 / 5 Grasp cup. 4 / 5 4 / 5

Deliver-coffee

Grasp-cup. 5 / 5 5 / 5 Navigate to table. 0 / 5 4 / 5 Place cup. 0 / 5 4 / 5

B. Implementation Details

- B.1. Acquiring Manipulation Skills

Table 8 presents the number of successful trajectories collected for each skill via teleoperation. In the ACT policy for each skill, we utilize a ResNet-50 backbone pre-trained on ImageNet to process binocular images. To enhance robustness against visual perturbations, data augmentation techniques such as random cropping, rotation, and color jittering are applied. The entire model, including the pre-trained encoder, is updated during training. Table 9 lists the hyperparameters used for training ACT.

Table 8. Number of trajectories collected for each manipulation skill.

Skill Num. Trajectories Carry Basket 25 Handout Snack 50 Grasp Bottle 150 Grasp Cup 200 Open Beer 50 Place Basket 25 Place Cup 200 Place Pole 50 Play Chess 70 Play Toy Bricks 50

- B.2. The Embodied Connector

- B.2.1. DATASET STATISTICS

Our dataset consists of two major types of tasks: Visual Understanding (VLU) tasks and Action Planning (AP) tasks. The VLU tasks include bounding box detection, yes/no questions, and image description tasks. We collected a total of 3,177

Table 9. Hyperparameters used for training the ACT policy.

Hyperparameter Value Training steps 500,000 Batch size 90 Learning rate 1e-5 Gradient clip norm 10 Chunk size (train) 30 Chunk size (test) 10

images, with 2,483 images dedicated to visual understanding tasks and 694 images for action planning tasks. Specifically, the image description tasks were initially labeled by GPT-4o and then refined by human annotators to ensure accuracy. The examples of the visual understanding labeling tasks are presented in Table 10. Table 11 summarizes the number of samples available for each type of task. Table 12 and 13 shows the data statistics of the bounding box tasks and the yes/no tasks, respectively.

To enhance visual grounding capabilities, we also include a general visual grounding dataset. We filtered 300K data samples from three open-source visual grounding datasets (Krishna et al., 2016; Tian et al., 2024; Shao et al., 2024). These data are fed into the model prior to training on our VLU and AP datasets, which will be further mentioned in Appendix B.2.2.

Table 10. Examples of the training data for training the vision-language model to acquire the visual understanding ability.

Task Name Samples Bounding Boxes Text Prompt: Detect the bounding box of target object: Reception desk.

Answer: [0.471, 0.082, 0.765, 0.951]

Identify items Text Prompt: Is there an exit in the image? Do not generate anything else except ’Yes’ or ’No’.

###### Answer: No.

Describe image Text Prompt: Describe the current image. List all the items you could see.

Answer: The image shows a corridor or office area with the following items visible:

- 1. A person wearing a light-colored outfit.

- 2. A green-framed glass door.

- 3. A yellow sofa with cushions.

- 4. A wall-mounted tablet or screen.

Describe the ground and identify the transition Text Prompt: Describe the ground and identify the transition between different grounds if there are any.

Answer: grey pattern carpet

Table 11. Task Categories and Sample Numbers

Task categories Tasks Sample number

Bounding boxes 14,784 Identify items 20,536 Describe image 1,530 Describe the ground and identify the transition 1,530

Visual Understanding

Action Planning - 771

- B.2.2. TRAINING DETAILS

We fine-tuned our vision-language model using the VideoLLaMA2 framework with a multi-node distributed training strategy. The training was conducted with a global batch size of 128 and a local batch size of 2 per device, with gradient accumulation steps dynamically computed based on the number of nodes and processing units per node. The model was trained for three epochs using a learning rate of 2 × 10−5, a cosine learning rate scheduler, and a warmup ratio of 0.03. We employed AdamW as the optimizer with zero weight decay. To enhance computational efficiency, we enabled mixed precision training with bfloat16 (BF16) and TensorFloat32 (TF32). Gradient checkpointing was applied to reduce memory consumption, and the maximum sequence length was set to 4096 tokens. The vision encoder was based on SigLIP1 (Zhai et al., 2023),

1https://huggingface.co/google/siglip-so400m-patch14-384

Table 12. Category data overview for the bounding box task.

Category Value Kitchen Area 792 Hallway 3,834 Robot Laboratory Room 896 Reception Area 247 Coffee machine 1,323 Workspace 118 Meeting Room 1,590 Wooden Table 414 Closed door 727 Workspace Area 697 Reception Desk 1,026 Door label 513 Doorway 706 Reception 114 Digital screen 428 Cup area 338 Espresso coffee button 496 Confirm button 286 Cancel button 239 Total 14,784 Number of Categories 19

Table 13. Category data overview for the yes/no task.

Category Value Coffee machine 1,530 Reception desk 1,530 Closed door 1,530 Door label 1,530 Water fountain 1,530 Glass door 1,530 Hallway 1,530 Reception area 1,530 Exit 1,530 Workspace 1,530 Passage 1,530 Doorway 1,530 Digital screen 428 Espresso coffee button 428 Confirm button 446 Preparing screen 428 Coffee ready screen 428 Cancel button 18 Total 20,536 Number of Categories 18

while the projection module was implemented using a multi-layer perceptron (MLP). We grouped multimodal samples by modality length, selected vision features from the second-to-last layer, and applied image padding to maintain aspect ratios. Each sample contained 16 frames.

The overall training process is divided into two stages. First, we finetune the model with the filtered 300K general visual grounding dataset based on the checkpoint provided by VideoOrion+(Feng et al., 2024), which shares the same architecture with VideoLLaMA2 but offers better object-centric understanding capabilities. Considering the training efficiency, we modify VideoOrion by removing the object-centric branch. Then we finetune the resulting model with our collected dataset including VLU tasks and AP tasks together.

This training setup ensures efficient vision-language modeling, leveraging optimized data handling, memory-efficient techniques, and distributed computation for improved performance.

- B.2.3. USAGE OF THE CONNECTOR

Visual Understanding. For visual understanding , the trained model predicts the bounding boxes of target objects in an image by generating the coordinates of the box, or it outputs “None” if no target object is present. The visual understanding capability of the VLM provides the robot with concrete information about its environment, enabling effective navigation and laying the foundation for informed skill planning decisions.

Skill Planning. For skill planning, given an overall task and a subtask, the model predicts the appropriate skill code for the robot to execute from either the modular skill library or the composite navigation skills. Skill planning facilitates the acquisition of spatial and embodied knowledge by enabling the model to make decisions based on the presence, relative distances and positioning of objects within its environment. Through training, the model learns to assess these spatial relationships and select the appropriate skill such as navigation and manipulation according to the proximity and orientation of the target objects. This embodied understanding allows the robot to adapt its actions in real-time, ensuring the translation of high-level task instructions into precise, contextually relevant actionable skills.

Below, we present details of the composite locomotion skills for navigation.

Move towards. We define the move towards skill as a skill that help the robot navigate to an target object in its view. The bounding box generated by the VLM is leveraged in this skill to determine the angle and the existence of the target object. The pseudo code for this skill is shown in Algorithm 1.

- Algorithm 1 Move Towards Target

- 1: Input: target name

- 2: Output: Status of movement towards the target
- 3: Initialize: max iterations, angle threshold, max iterations

- 4: for each iteration from 1 to max iterations do

- 5: Get camera image and depth data
- 6: Detect the target object in the image using VLM
- 7: if No target detected then
- 8: Stop moving
- 9: Break the loop
- 10: end if
- 11: if Target is within threshold distance then
- 12: Stop moving
- 13: Break the loop
- 14: end if
- 15: if Obstacles detected then
- 16: Avoid obstacle using sidestep
- 17: else
- 18: Move forward or turn depending on angle
- 19: end if
- 20: end for
- 21: Return: status: True/False

Search for. We define the search for skill as constantly turning to one direction until the target object is found in the view. The bounding box generated by the VLM is leveraged in this skill to determine the existence of the target object. The pseudo code for this skill is shown in Algorithm 2.

- Algorithm 2 Search for Target

- 1: Input: target name

- 2: Output: Status of target search
- 3: Initialize: max iterations, direction, head angle, tilt angle

- 4: for each iteration from 1 to max iterations do

- 5: Get camera image
- 6: Check if target is detected in the image
- 7: if Target detected then
- 8: Stop moving
- 9: Break the loop
- 10: end if
- 11: if Direction is ”right” then
- 12: Turn right
- 13: else
- 14: Turn left
- 15: end if
- 16: end for
- 17: Return: status: True/False

Adjustment. To perform adjustment during navigation, we modify the move-forward skill so that the robot first adjusts its head to look aside to the direction of the item, then decides whether to turn or move forward based on the adjusted view. The direction to look aside is predicted by the VLM. This approach allows the robot to gradually approach the target object in an arc-shaped path, ultimately reaching the optimal position. The pseudo code for this skill is shown in Algorithm 3.

- Algorithm 3 Adjustment

- 1: Input: target name, direction

- 2: Output: Status of final approach
- 3: Initialize: head angle, tilt angle, max iterations

- 4: for each iteration from 1 to max iterations do

- 5: Set head position and tilt
- 6: Get camera image and detect target direction
- 7: if Target detected and within threshold distance then
- 8: Stop moving, adjust to face target
- 9: Break the loop
- 10: end if
- 11: If target angle is small/large, adjust direction (left/right)
- 12: If target angle is 0, check for obstacles and avoid if necessary
- 13: Move towards the target if no obstacles detected
- 14: end for
- 15: Return: status: True/False

##### B.3. Prompt Design

We present prompts designed to enable the foundation model to perform various levels of agent abilities. Specifically, we include prompts for information gathering, self-reflection, and subtask inference in Tables 14, 16, and 15, respectively. Additionally, we provide the action planning prompt for scenarios where the embodied Connector is not used in Table 17. For cases involving the embodied Connector, we utilize a more concise version of the prompt for action planning, which is shown in Table 18.

Table 14. The prompt we used for information gathering process.

Information Gathering

You are a helpful AI assistant integrated with a humanoid robot body equipped to handle diverse tasks in the real world. Your advanced capabilities enable you to process textual and visual information, including computer application screenshots, and to control the robot body.

<image introduction> Overall task: <task description> Subtask description: <subtask description> Semantic map: <semantic map> Current Location: <robot location> Holding Cup Status: <robot holding cup status> Image Description:

- 1. Using the latest image, please describe it in detail. Pay attention to the details in the image, if any, especially critical objects or icons.

- 2. Identify in which area of the map you are currently in, based on the semantic map provided above and the past action history.

- 3. Pay attention if you have reached a target area and if the target task is already successful.

- 4. Pay attention if you have reached a target area and if the target task is already successful.

- 5. Keep in mind the target object or area and describe its position and orientation in the image, if any.

- 6. If you are trying to navigate to a location, use its place name from the semantic map as a target, if possible.

- 7. If the target is not in the current image frame, but it has been found previously, use the recent actions and the previous frames to reason about its position, location, and orientation.

- 8. In the latest image, if you observe new possible targets, compare the new targets with the current target and decide which one is more likely to be the correct target.

Target Name: Assume you can use a detection model to detect the most relevant object, image area, or UI item to complete the current task, if any is needed. What target should be detected to complete the task based on the latest image and the current task? You should obey the following rules:

- 1. Identify an item or area that is relevant to the current or intermediate target of the task.

- 2. For a target object, consider its possible forms and list as many as possible.

- 3. If there is a new possible target object, compare it with the current one and choose the one that is the most promising.

- 4. If there is no need to detect a target, only output “null”.

###### Reasoning for Target:

Why was this target chosen, or why is there no need to detect or use a new target? Why is this target more promising than other possible targets? You should only respond in the format described below and not output comments or other information. DO NOT change the title of each item. Image Description:

- 1. ...

- 2. ...

- 3. ...

- 4. ... Target Name: name Reasoning for Target:

###### ... Area Location: area name

Table 15. The prompt for summarizing task progress and proposing a new subtask.

Task Reflection and Subtask Proposal

Overall task description: <task description>

Previous proposed subtask for the task: <subtask description>

Previous reasoning for proposing the subtask: <subtask reasoning>

<image introduction> Description of current image frame: <image description> Last executed action: <previous action> Error report for the last executed action: <executing action error> Key decision-making reasoning for the last executed action: <previous reasoning> Self-reflection for the last executed action: <self reflection reasoning> Success Detection for the overall task: <success detection> The following is the summary of history that happened before the last screenshot: <previous summarization> Semantic map: <semantic map> Current Location: <robot location> Holding Cup Status: <robot holding cup status> History summary: Summarize what happened previously, especially the last step according to the decision-making reasoning and self-reflection reasoning for the last executed action. The summary needs to be precise, concrete, highly related to the task, and follow the rules below.

- 1. Determine if the task has been completed successfully. If it is successful, ignore question 2 to 5.

- 2. Summarize the tasks from the history and the current task. What is the current progress of the task? For example, to open a file, you first need to select the file, then open it by clicking somewhere or using the keyboard. Subtasks may have other pre-requisites.

- 3. Record the successful actions and organize them into events, step by step.

- 4. What is the current area you are in? What is the target area? What is the next area going to be if you move forward?

- 5. Which subtask has been completed? Which subtasks have not been completed?

- 6. Do not forget the information and key events in the previous steps of the overall task. Subtask reasoning:

- 1. Based on the unfinished part of overall task and the current visual information, identify the way to complete the task without making any assumptions beyond the provided information.

- 2. Analyze the target task step by step to determine how to complete it.

- 3. What was the previous subtask? Was the previous subtask successfully finished according to self-reflection? Is it improper for the current situation? If finished or improper, please select a new subtask, otherwise you must reuse the last subtask.

- 4. What was the previous location you were in? Have you reached a new place based on the current observation? Pay attention if you have already reached the target location of the previous subtask or task.

- 5. If you are already in a new location, PLEASE propose a new subtask and skip question 14.

- 6. If the target of the action is not visible in the current image, DO NOT try to move towards it. Instead, if you are in the target location, the new subtask should be to search around to find it.

- 7. You should search the place for the target after any movement towards a previous target.

- 8. If the search does not find the target, propose a new task to move towards a new target in the current image, and then search again for the previous target.

- 9. If the next area in front of you is not the target area, DO NOT move towards it. Instead, if you are in the target location, propose a new subtask to search to find it.

- 10. If you want to propose a new subtask, give reasons why it is more feasible for the current situation. Please strictly follow the description and requirements in the current task.

- 11. The proposed subtask needs to be precise and concrete within one sentence.

- 12. If a given task or subtask is already very simple, like ”wave your hand”, no need to decompose it, the next subtask to perform is just the simple task. You should only respond in the format described below, and you should not output comments or other information.

History summary: The summary is...

Subtask reasoning:

- 1. ...

- 2. ...

- 3. ...

- 4. ...

- 5. ...

- 6. ...

- 7. ...

- 8. ...

... Subtask description: The current subtask is...

Table 16. The prompt for reflecting on the task and evaluating success.

Task Reflection and Success Evaluation

You are a helpful AI assistant integrated with a humanoid robot body equipped to handle diverse tasks in the real world. Your advanced capabilities enable you to process textual and visual information, including computer application screenshots, and to control the robot body. Your task is to examine any inputs, interpret the context, and determine whether the last executed action has succeeded and caused the correct effect.

Overall task description: <task description>

<image introduction> Description of current image frame: <image description>

Key reason for the last action: <key reason of last action>

Last executed action with parameters used: <previous action call>

Error report for the last executed action: <executing action error>

Success Detection flag for the overall task: <success detection>

Valid action set in Python format to select the next action: <skill library>

Current and previous image are the same: <image same flag>

Semantic map: <semantic map>

Self Reflection Reasoning:

- 1. What is the last executed action based on the text information above?

- 2. Make use of the information gathered from the images to decide if you have completed the current subtask. Pay special attention to the error report for the last executed action.

- 3. Think about your previous location and whether you have reached a new location (for example, moved from one hallway to another), based on the current observation and the semantic map. Consider if you are already in the target location for the last task.

- 5. Was the last executed action successful? Give reasons to this conclusion. You must refer to the following rules:

- - If the last action executed was empty, then the previous action is deemed successful.

- - If the action seemed to have no effect, pay attention to whether the robot position changed or if any of its hands move during the action execution process.

- 6. If the last action is not executed successfully, what was the most probable cause for it? You should give only one cause and refer to the following rules:

- - The reasoning to chose the last action was wrong.

- - If it is an interaction action, the most probable cause was that the action was unavailable or not activated in the current state.

- - If there is any errors, analyze the cause based on them.

- 7. Pay attention to targets like ”hallway”, ”exit”, ”doorway”, ”corridor”, ”passage”, ”open door”, ”hole in the wall”, ”opening”, etc. They usually refers to the same target. Always use the word ”hallway” for these.

Success Detection: Based on the last action, the current images, and the Success Detection flag, determine whether the overall task <task description> was successful. This assessment should consider the overall task’s success, not just individual actions.

- - If the last action executed was an empty list and <success detection> indicates the task is successful, then the overall task has a high chance of being considered a success.

- - If the overall task was unsuccessful, specify the reason of failure and which steps are missing.

- - If the overall task was successful, ONLY output “SUCCESSFUL”.

You should only respond in the format as described below. Self Reflection Reasoning:

- 1. ...

- 2. ...

- 3. ...

Success Detection:

...

Table 17. The prompt for decision-making and action execution in humanoid robot tasks.

Action Planning

You are a helpful AI assistant integrated with a humanoid robot body equipped to handle diverse tasks in the real world. Your advanced capabilities enable you to process textual and visual information, including computer application screenshots, and to control the robot body. By analyzing these inputs, you can understand the current context and situation of the robot. Use these insights to select the most suitable action for the robot to take next, given the current task.

Overall task description: <task description>

Subtask description: <subtask description>

Few shots: <few shots>

<image introduction> Description of current image: <image description> Current and previous image are the same:

<image same flag> Last executed action: <previous action>

Key reason for the last action: <key reason of last action>

Self-reflection for the last executed action: <previous self reflection reasoning>

Summarization of recent history: <previous summarization>

Valid action set in Python format to select the next action: <skill library>

Success detection for overall task: <success detection>

Semantic map: <semantic map>

Decision Making Reasoning:

- 1. Does <success detection> mean the overall task was successful? If successful, ignore questions 2 to 16. Otherwise, do not make conclusion before answering the other questions.

- 2. When going to a location target, in which hallway is it located? If you are already in the right hallway, DO NOT move to the wrong hallway. Make sure where you are first. Just because you can see another area, it does not mean you are in it.

- 3. If the previous action failed with ”Move failed” because the target was not in the latest image, NEVER try the same action with the same target a second time!

- 4. If the target is not visible in the current image, you MUST SEARCH for the target (unless the previous action was already a search).

- 5. If you are not in the target area, and the previous action was already a search for the target, then try to find a way to move towards the target area first. For example, follow down a hallway connected to the target area.

- 6. If the current image is the same as the one from the previous iteration, DO NOT output the same action as the previous step.

- 7. Summarize the contents of recent history, mainly focusing on historical tasks and behaviors.

- 8. Summarize the contents of self-reflection for the last executed action, and do not be distracted by other information.

- 9. Think about the previous location you were in and if you have reached a new place based on the current observation. Think if you are already in the location required for the task. DO NOT rely on door labels.

- 10. Keep in mind the position of the target, even if it is no longer in the current frame.

- 11. You should search the place after any movement forward towards a target that was a hallway.

- 12. If you have to move, try to move towards some target that you can see in front of you (like the ground near your destination).

- 13. If you do not have a skill to finish the desired task, use speak("request text") to request help.

- 14. Which skill in the skill library above is the most related to how to conduct the next step of the current subtask?

- 15. This is the most critical question. Based on the action rules and self-reflection, what should be the most suitable action in the skill library for the next step? You should analyze the possible effects of the action step by step. Actions: The best action, or first action step in a short sequence of actions, to execute next towards achieving the goal. Pay attention to the names of the available skills and the previous skills already executed, if any. Pay special attention to the coordinates or direction of any action that needs them. Do not make assumptions about the location of objects or UI elements, analyze in detail any provided images. You should also pay attention to the following action rules:

- 1. If <success detection> means the overall task was successful or equal to ”True”, then output action MUST be empty like ”. Be careful to first check that the task was really successful.

- 2. You should output actions in Python code format and specify any necessary parameters to execute that action. Only use function names and argument names exactly as shown in the skill library of valid actions. If a function has parameters, you should also include their names and decide their values, like turn right("small"). If an action does not have a parameter, just output the action function call, like go back().

- 3. You cannot open doors, so NEVER go to doors to open them. To move between rooms or areas, ALWAYS use open doorways or passages (openings) toward hallways. You should only move towards a closed door to stand in front of it if that is the final location target for the task.

- 4. Given the current situation and task, you should only choose the most suitable action from the skill library. You cannot use actions that are not in the skill library.

- 5. If you are walking down a hallway to try to find a target, you MUST perform a search for the target after any movement towards a hallway target. Key reason for the last action: Summarize the key reasons for choosing this action to execute.

You should only respond in the format described below. In your reasoning for the chosen actions, also describe which object or area you decided to interact with and why. DO NOT change the title of each item in the response. You should not output other comments or information besides the format below.

Decision Making Reasoning:

- 1. ...

- 2. ...

- 3. ...

- 4. ...

- 5. ...

- 6. ...

- 7. ...

- 8. ...

- 9. ...

- 10. ...

- 11. ...

- 12. ...

...

Actions: “‘python action(args1=x,args2=y) “‘

Key reason of last action:

...

Table 18. The shorter version of the prompt we used for decision making and action execution of humanoid robot

Decision Making and Action Execution for Humanoid Robot <image introduction> Overall task description: <task description> Subtask description: <subtask description> Map: <semantic map> Valid actions to select the next action: <skill library> You should only respond in the format described below. Action: ‘‘‘python action(args1=x,args2=y) ‘‘‘

