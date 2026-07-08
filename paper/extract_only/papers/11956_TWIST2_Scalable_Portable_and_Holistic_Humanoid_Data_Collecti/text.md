# arXiv:2511.02832v1[cs.RO]4Nov2025

TWIST2 Scalable, Portable, and Holistic Humanoid Data Collection System

Yanjie Ze12 Siheng Zhao13 Weizhuo Wang12 Angjoo Kanazawa14† Rocky Duan1† Pieter Abbeel14† Guanya Shi15† Jiajun Wu2† C. Karen Liu12† 1Amazon FAR 2Stanford University 3USC 4UC Berkeley 5CMU †Equal Advising

Holistic Whole-Body Teleop (long-horizon, dexterous, mobile manipulation)

Portable MoCap-Free Device (from walk in to finish setup: 1 min)

TWIST2

[Figure 1]

[Figure 2]

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Scalable Data Collection (100 successful pick & place in 15~20 mins)

Visuomotor Humanoid Policy Learning (fully whole-body control via a hierarchical framework)

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Portable, Scalable, Holistic Data Collection System

[Figure 40]

[Figure 41]

Fig. 1: We introduce TWIST2, a holistic humanoid data collection system designed with scalability and portability. TWIST2 enables scalable data collection, fast setup, and enjoyable user experience compared to MoCap solutions such as TWIST [1], while maintaining the full whole-body control. We build a 2-DoF Neck (TWIST2 Neck) to enable egocentric teleoperation, which costs $250. With TWIST2, our robots are able to perform long-horizon, dexterous, mobile whole-body manipulation and legged manipulation. All tasks are achieved with streamed robot egocentric vision, full whole-body control, and a single operator. We further train visuomotor policies upon data collected via TWIST2. Our entire system is open-sourced at https://yanjieze.com/TWIST2 and ensure full reproducibility.

Abstract— Large-scale data has driven breakthroughs in robotics, from language models to vision-language-action models in bimanual manipulation. However, humanoid robotics lacks equally effective data collection frameworks. Existing humanoid teleoperation systems either use decoupled control or depend on expensive motion capture setups. We introduce TWIST2, a portable, mocap-free humanoid teleoperation and data collection system that preserves full whole-body control while advancing scalability. Our system leverages PICO4U VR for obtaining real-time whole-body human motions, with a custom 2-DoF robot neck (cost around $250) for egocentric vision, enabling holistic human-to-humanoid control. We demonstrate long-horizon dexterous and mobile humanoid skills and we can collect 100 demonstrations in 15 minutes with an almost 100% success rate. Building on this pipeline, we propose a hierarchical visuomotor policy framework that autonomously controls the full humanoid body based on egocentric vision. Our visuomotor policy successfully demonstrates whole-body dexterous manipulation and dynamic kicking tasks. The entire system is fully reproducible and open-sourced at https://yanjieze.com/TWIST2. Our collected dataset is also open-sourced at https://twistdata.github.io.

1Work done during the internship of Yanjie Ze, Siheng Zhao, and Weizhuo Wang at Amazon Frontier AI & Robotics (FAR).

I. INTRODUCTION

The transformative power of large-scale data has fundamentally reshaped machine learning, driving breakthrough achievements from large language models like GPT-4 [5] to the recent success of vision-language-action (VLA) models in robotics. In the realm of bimanual manipulation, models such as π0 [6] and π0.5 [7] have demonstrated unprecedented capabilities, directly enabled by the robust and scalable data collection infrastructure [8]–[10]. However, this datadriven revolution has yet to reach humanoid robots, where the absence of equally effective data collection frameworks continues to limit progress toward human-level versatile manipulation and locomotion.

As summarized in Table I, existing humanoid teleoperation systems fall into three broad categories: a) Decoupled control of lower and upper body (e.g., MobileTV [11], HOMIE [2]); b) Partial whole-body control that coordinates selected body segments such as arms and torso while legs track base velocity commands (e.g., AMO [3], CLONE [4]); c) Full wholebody control that directly tracks human body pose across

TABLE I: Comparison of recent humanoid data collection systems. We compare existing humanoid teleoperation systems across key dimensions essential for effective data collection. TWIST2 is the first system to combine full whole-body control with portability, achieving comprehensive capabilities including egocentric teleoperation, accurate tracking, and single-operator efficiency. Unlike previous works that either sacrifice portability for full whole-body control (TWIST) or sacrifice full whole-body control for portability (AMO, CLONE), our system achieves all critical requirements for scalable humanoid data collection.

Humanoid Data Portability & Scalability Holistic Control Collection System Category Source Portable No Calibration Single Operator Whole-Body Tracking Egocentric Teleop Foot Control Wrist Control

HOMIE [2] Decoupled Exoskeleton ✗ ✓ ✓ ✗ ✗ ✗ ✓ AMO [3] Partial VR ✓ ✓ ✗ ✗ ✓ ✗ ✓ CLONE [4] Partial VR ✓ ✓ ✓ ✗ ✗ ✗ ✓ TWIST [1] Full MoCap ✗ ✗ ✓ ✓ ✗ ✓ ✗ TWIST2 (ours) Full VR ✓ ✓ ✓ ✓ ✓ ✓ ✓

all joints including arms, torso, and legs in a unified manner (e.g., HumanPlus [12], TWIST [1]). Among these, VR-based solutions such as AMO and CLONE offer practicality but are limited to mobile skills with simple locomotion, falling short of capturing dynamic whole-body coordination skills that humans naturally exhibit. In contrast, full whole-body control holds the greatest promise for unleashing the versatility of humanoid robots, as evidenced by TWIST [1]. However, such systems typically depend on expensive, non-portable motion capture setups, restricting deployment to lab environments.

In this work, we introduce TWIST2, a humanoid teleoperation and data collection system that preserves the power of full whole-body control while advancing portability and scalability. Our design leverages PICO4U [13], a lightweight VR device that provides whole-body motion streaming using a head goggle, handheld controllers, and two motion trackers on the ankles, without requiring expensive motion capture systems. Recognizing that egocentric vision is crucial for human-like task execution, we design a low-cost and noninvasive neck that seamlessly integrates with Unitree G1 and our VR teleoperation ecosystem. With these portable components, we build a comprehensive retargeting pipeline from full human body poses of PICO to corresponding humanoid motor joint positions. To execute the retargeted motions on the robot, we train a robust motion tracking controller using reinforcement learning and large-scale simulation interaction on carefully curated motion data.

These elements together enable efficient, long-horizon, inthe-wild teleoperation and data collection without reliance on motion capture systems, and only requiring a single operator. We showcase that 1) we can teleoperate robots to perform very long-horizon and fine-grained whole-body dexterous skills such as folding towels and mobile skills such as transporting objects through the door, and 2) we can collect human demonstrations efficiently, e.g., collecting around 100 successful demonstrations in 20 minutes without failure. We also find that egocentric active stereo vision is essential for the long-horizon mobile and dexterous teleoperation.

Building on this scalable data collection pipeline, we further propose a hierarchical visuomotor policy learning framework consisting of two components. The first component is the same motion tracking controller used during teleoperation, which serves as a low-level controller. The

second component is a Diffusion Policy that directly predicts whole-body joint positions based on visual observations that feeds into the low-level controller. To our knowledge, this is the first policy learning framework that enables vision-based autonomous control of the full humanoid body, moving beyond simplified commands such as root velocity. Importantly, this capability is made possible by our data collection system, which provides the high-quality demonstrations needed for training.

We showcase a few representative results where our humanoid robot autonomously performs a) consecutive wholebody dexterous pick & place and b) continuous kicking of a T-shaped box to target regions (Kick-T), illustrating the potential of this new framework.

To summarize, our main contributions are:

- 1) A portable, mocap-free humanoid teleoperation and data collection system with full whole-body control, enhanced with an attachable neck for egocentric active vision.
- 2) A hierarchical whole-body visuomotor policy learning framework that achieves full whole-body control.
- 3) Demonstration of long-horizon teleoperation skills such as towel folding/unfolding and object transporting through the door, effective data collection, and new autonomous humanoid skills including whole-body dexterous pick & place and Kick-T.

Our system, data, and model are fully open-sourced at https://yanjieze.com/TWIST2 to ensure full reproducibility.

II. RELATED WORK A. Whole-Body Humanoid Teleoperation

Teleoperation is crucial for enabling humanoid robots to interact with complex real-world environments and perform sophisticated loco-manipulation tasks. Unlike wheel-based robots or tabletop arms, the anthropomorphic nature of humanoids makes whole-body control the most natural and effective teleoperation approach [1], [3], [4], [12], [14]– [16]. As shown in Table I, we categorize recent works into three categories: a) decoupled control, b) partial whole-body control, and c) full whole-body control. Full whole-body control, as demonstrated by TWIST [1], shows promising results in coordinated whole-body dexterity, which is the

primary focus of this work. As detailed in Table I, we identify several critical aspects in scalable & holistic teleoperation and data collection that remain lacking in previous works for real-world deployment. which we address comprehensively in this work.

- B. Visual Humanoid Control

Previous works on visual humanoid control predominantly rely on LiDAR for perceptive locomotion [17]–[19], typically employing task-specific sim-to-real reinforcement learning (RL) approaches. Recent works like HEAD [20] propose keypoint-based hierarchical frameworks with humanoid egocentric vision, while limiting in simple navigation tasks. VideoMimic [18] introduced a real2sim2real pipeline that enables real robots to perform environment interactions such as sitting, though their interactions remain limited to static settings like the ground or stone chairs. Some works such as PDC [21] are conducted only in simulation and face significant sim-to-real transfer challenges. In contrast, our work focuses on developing general visuomotor humanoid policies that can interact with complex environments and perform long-horizon whole-body loco-manipulation and legged manipulation tasks—capabilities not demonstrated in previous works.

III. OUR SYSTEM

We introduce TWIST2, a scalable, portable, and holistic humanoid teleoperation and data collection system (see Figure 1 for capabilities). As illustrated in Figure 2, our system consists of four main components: a humanoid robot equipped with active vision (Section III-B), portable motion capture using VR devices (Section III-C), holistic human-to-robot motion retargeting (Section III-D), a general motion tracker for low-level control (Section III-E). These components work together to enable scalable data collection

- (Section III-F) and autonomous visuomotor policy execution
- (Section III-G).

A. Problem Formulation

We focus on enabling humanoid robots to perform diverse whole-body dexterous tasks with their own egocentric vision and proprioception within a single unified framework. To this end, we propose a two-level hierarchical control framework, consisting of a low-level controller πlow and a high-level controller πhigh.

Low-level control. We formulate the low-level controller πlow as a general motion tracking problem, so that our lowlevel control is task-agnostic. At each timestep, the low-level controller receives a reference command vector composed of root translational velocity in the x and y axes, root z position, root roll/pitch angles, root yaw angular velocity, and wholebody joint positions:

pcmd = x ˙ref, y˙ref, zref, ϕref, θref, ψ˙ref, qref . (1)

In addition, it has access to robot proprioception, including root orientation and angular velocity from IMU readings, as

well as joint positions and velocities from encoders:

s = ω, ω,˙ q, q˙ . (2) The controller outputs desired joint positions,

qtgt = πlow(s,pcmd), (3)

at 50Hz, which are then tracked by a PD controller to generate the final torque:

#### τ = KP (qtgt − q) − KD q˙. (4)

High-level control. The high-level controller πhigh focuses on generating task-specific motion commands pcmd conditioned on egocentric vision. We have two variants in this work: (1) a teleoperation policy πhighteleop, and (2) a visuomotor policy πhighauto. Both map visual observations o and proprioceptive states s into commands:

pcmd = πhigh(o,s). (5)

In this work, we employ πhighteleop, i.e., the human teleoperator plus the motion retargeter, to collect observation–action pairs

(o,s,pcmd), which are then used to train πhighauto, e.g., a Diffusion Policy.

Interface design. There are two key aspects of our command interface pcmd: (1) We use relative root translations/rotations rather than absolute poses, so that our system does not rely on accurate global state estimation [22], and remains stable during very long-horizon operation; (2) We include whole-body joint positions instead of simplifying lower-body control as root velocity only [3], [4], [11], which enables more precise control of lower-body movements and unlocks tasks such as legged manipulation and dancing.

B. Humanoid Robot with Active Vision

We use Unitree G1 with 29 DoF (3 DoF waist + two 6 DoF legs + two 7 DoF arms), equipped with two 7 DoF Dex31 hands. We find that neck DoFs are essential for effective and long-horizon teleoperation, so we build a portable robot neck with yaw and pitch DoFs.

Add-on low-cost neck (TWIST2 Neck). Unlike recent works [3], [23] that build built-in necks, we design an addon neck module that can be seamlessly attached to the Unitree G1 without disassembling its original head (see Figure 3). Our design is inspired by ToddlerBot [24]. We use two Dynamixel XC330-T288 motors to control the yaw and pitch angles, connected via a U2D2 and powered by the onboard 12V/5A supply. All structural parts are 3D printed. The cost of the neck is $250. We use Zed Mini as our stereo camera attached to the neck (the ZED Mini stereo camera will cost extra $400). Since human roll DoF is rarely used in everyday interaction, we find that the two-DoF design already enables smooth and human-like neck motions (Figure 5). To further standardize the TWIST2 neck usage, we build the corresponding simulation model in MuJoCo as shown in Figure 4.

## TWIST2

[Figure 42]

[Figure 43]

[Figure 44]

##### Portable Device

##### Holistic Whole-Body Teleoperation

##### Visuomotor Policy Learning

[Figure 45]

Human Motion

Robot Motion π

High DoF Control

Egocentric Active Vision

o

[Figure 46]

45 DoF=2 (Neck) + 29 (Body) + 14 (Hand)

[Figure 47]

q

[Figure 48]

[Figure 49]

###### π

###### PICO4U VR

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

p

+2x PICO Motion Tracker

p

p

General Motion Retargeter

General Motion Tracker

$1000

| | | |
|---|---|---|
| | | |

|a|
|---|

|a| |
|---|---|
| | |

[Figure 54]

p

[Figure 55]

General Motion Tracker

[Figure 56]

[Figure 57]

Diffusion Policy

| |a| |
|---|---|---|
| | | |

Scalable Data Collection

Add-On Low-Cost Neck $250

enabled by Intuitive and Holistic Teleoperation

π π

[Figure 58]

Example:

- 1) Bimanual Dex Manip: 20 min, 100 episodes
- 2) Mobile Dex Manip: 20 min, 50 episodes

p : {base vel, base ang vel, full body pos, neck pos, hand pos}

- Fig. 2: System overview of TWIST2. We build a holistic humanoid teleoperation system with portable devices and egocentric active vision, enabling scalable imitation data collection. With data collected, we build a hierarchical visuomotor policy learning framework that directly predicts whole-body joint positions.

[Figure 59]

- Fig. 3: TWIST2 Neck. We design a simple yet effective 2DoF neck that can be easily assembled for a non-expert user and can be attached/detached to/from a Unitree G1 without removing the original LiDAR.

[Figure 60]

- Fig. 4: TWIST2 Neck in MuJoCo. To facilitate the research in simulation and standardize our data, we build MuJoCo XML files for our TWIST2 neck.

cheaper and practical compared to an optical MoCap system. We use XRoboToolkit [26] for access to motion streaming from PICO (Figure 6). The motion can be streamed at 100Hz. Notably, PICO does not require heavy calibration compared to the MoCap system. As shown in Figure 1, it takes around only 1 minute to finish the setup of PICO.

Compared to HTC Vive Tracker [27] that is used in recent demos of Boston Dynamics [28], PICO’s whole-body estimation does not require extra third-person view camera setup, thus more flexible.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Fig. 5: Mimic the human neck with the robot neck. We found that a 2 DoFs neck (yaw and pitch) is sufficient to mimic major human neck movements.

D. Holistic Human-to-Humanoid Retargeting

In this section, we describe how human motion data is holistically leveraged to control the humanoid robot’s body, hands, and neck.

Body retargeting. We adapt GMR [1], [29], a real-time motion retargeting method, to the PICO human motion format (Figure 6). The original GMR employs a two-stage optimization: (1) solving for link rotation consistency, and (2) refining global pose alignment. Since PICO motion capture often yields inaccurate global pose estimation, we modify the second optimization stage as follows: 1) for the lower body, optimize for position and rotation constraints; 2) for the upper body, only optimize for rotation constraints. This ensures 1) less feet sliding and 2) better upper-body teleportation experience.

- C. Portable MoCap-Free Whole-Body Human Data Source

To obtain real-time full human body poses in a portable manner, we utilize PICO 4U [13] combined with two PICO Motion Trackers [25] that are bound on the humans’ calves to obtain global translations and rotations for each human body parts. Though PICO supports more than 2 motion trackers, we find the 2-tracker mode provides a more stable pose estimation. The cost for such a setup is around $1000. much

We partition the retargeted links into lower-body Llow (e.g., pelvis, hips, knees, ankles, feet) and upper-body Lup (e.g., spine, shoulders, elbows, wrists, head). Let Rihuman and Rirobot(q) be the link orientations, and phumank and probotk (q) the link positions for a selected set of lower-body points Plow (typically feet/ankles and optionally pelvis). To reduce sensitivity to noisy global pose estimation (and to support user teleportation), we measure all human positions in a pelviscentric frame. The stage-2 optimization is then formulated as:

wiR Rihuman − Rirobot(q) 2F

q∗ = arg min

q

i∈Lup∪Llow

wkp phuman,pelvisk − probotk (q) 22. (6)

+ λpos

k∈Plow

Here wiR and wkp are per-link weights, λpos balances the rotation and position terms, and phuman,pelvisk denotes human keypoints expressed in the human pelvis frame. This formulation enforces accurate foot and ankle placement to mitigate foot sliding, while keeping the upper body free of positional terms so that global-pose jumps (e.g., teleportation) do not introduce artifacts—upper-body retargeting depends only on local rotations.

###### VR Human Bodies Unitree G1 Links

[Figure 65]

[Figure 66]

Shoulder shoulder_yaw_link wrist_yaw_link Elbow elbow_link Wrist

Retargeting

Pelvis pelvis Hip hip_yaw_link

Knee knee_link

Foot ankle_roll_link

Fig. 6: Mapping VR human bodies to robot links.

Hand retargeting. Directly mapping a human five-finger hand to the Unitree Dex31 hand is not intuitive for teleoperation, since the Dex31 only provides three fingers with limited degrees of freedom. In practice, the functionality of the Dex31 hand is much closer to a parallel-jaw gripper than to a dexterous multi-fingered hand. Therefore, we simplify hand retargeting by treating the Dex31 as a gripper and not using hand pose estimation but controlling it by pressing buttons with PICO handheld controllers. We define two canonical configurations: an open pose qopen and a close pose qclose.

- A scalar grasp command α ∈ [0,1] is computed from the human hand signals, where α = 0 denotes fully open and α = 1 denotes fully closed. The commanded Dex31 hand joint configuration is then interpolated as

qhand = (1 − α)qopen + α qclose. (7)

For tasks that require power grasp (e.g., grasp a cup) and tasks that require fine-grained pinching (e.g., folding cloths), we define two sets of qopen and qclose.

Neck retargeting. Let Rhead,Rspine ∈ SO(3) be the global rotations of the human head and spine in the world frame, respectively. The relative rotation is

Rrel = Rspine⊤ Rhead. (8) From Rrel = [rij], the robot neck joint targets are defined as qneckyaw = ψ = arctan2(r21, r11), qneckpitch = θ = arcsin(−r31).

(9)

- E. Training General Motion Trackers for Low-Level Control To bring the retargeted kinematics motions onto a physical

robot, we need a whole-body controller πlow that takes into reference motions and outputs the desired PD target. Different from previous works that adopt a complex teacher-student pipeline to train a reasonable whole-body controller [1], [14], [30], we design a simple one-stage training framework for general motion tracking.

More specifically, we first curate a humanoid motion dataset consisting of around 20k motion clips. The motion dataset includes data retargeted via GMR [1], [31] (7k clips) and the original motion dataset from TWIST [1] (13k clips). The motion data source includes AMASS [32], OMOMO [33], and our in-house MoCap data. This mixture of the dataset ensures our policy learns omnidirectional walking. Similarly as found in TWIST [1], we find that curating a small set of motions from the teleoperation device is essential to bridge the domain gap. We only collect 73 motions via PICO, as these motions already cover most daily movements like walking, crouching, and manipulation. We then generate reward supervision from the motion datasets. The rewards are defined as r = rtrack + rreg, where rtrack is defined as:

rtrack = e−α∥p

cmd−pcur∥ (10)

where pcur denotes the actual state the robot achieved. rreg consists of the regularization terms, such as the penalty on the action change.

The actor πlow is trained via PPO and mainly consists of two parts: the convolutional history encoder and the MLP backbone. We find that compressing history robot proprioceptions and history reference motions into a compact latent vector boosts learning efficiency.

- F. Scalable Humanoid Data Collection

We now describe our humanoid teleoperation and data collection system built with the aforementioned modules.

Egocentric whole-body teleoperation. During teleoperation, we obtain real-time streamed human motions from PICO (Section III-C) and map human motions into robot motion commands pcmd, and then send pcmd to πlow (Section III-E) through Redis [34]. Additionally, our teleoperation system is equipped with stereoscopic vision via the custom shader implemented in [26] that adjusts the interpupillary distance and sets the focal point at approximately 3.3 feet, providing teleoperators with depth perception (see Figure 8) The stereo images are streamed from ZED Mini to PICO

SYSTEM 2 SYSTEM 1

History Action Motion Target

p : Base Vel Base Ang Vel Body Pos Neck Pos Hand Pos

p p

[Figure 67]

p

p

Diffusion Policy

Robot Image

…

General Motion Tracker

[Figure 68]

PD Target

a

20Hz

p

a

50Hz

s : IMU Body Pos Body Vel

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

Robot State

s

s

- Fig. 7: Hierarchical whole-body visuomotor policy learning framework built upon data collected via TWIST2. Unlike previous works that focus on upper-body manipulation or lower-body locomotion separately, our visuomotor policy controls the entire body, enabling complex tasks such as Kick-T that require coordinated whole-body movements.

[Figure 79]

- Fig. 8: The view of the teleoperator in PICO. The robot vision is floating in the center.

control center, as shown in Figure 9.

[Figure 80]

Data Record Start/End Teleop Start/End

Hand Close

Hand Open

Fig. 9: Illustrations on using the PICO joystick controller as the control center to make TWIST2 a single-operator system. Safe control. Humanoid robots are brittle; and this problem becomes more critical when designing a system that can fully control the robot. In TWIST2, we use motion interpolation for smooth state transition. For example, our system supports pause via the origin joystick from PICO; and when pause mode ends, we interpolate from the last robot pose to current target pose, to avoid sudden jump. This guarantees our system can operate in a quite long time safely and stop anytime when human operators are tired.

System delay. All modules in our system stream at a speed above 50Hz, ensuring the overall delay to be lower than 0.1s, significantly improved upon prior work [1] (0.5s delay).

Data filtering. During data collection, we consecutively record episodes. To process these trajectories, we developed a demonstration post-processing GUI that segments long sequences into multiple episodes, each corresponding to a completed task. We also reduce idle actions and remove failure episodes through filtering.

via GStreamer in the h265 format and to the data collection process via ZMQ in the JPEG format.

Single operator. A practical teleoperation/data collection system should only require a single operator. Recent wholebody humanoid teleoperation systems focus on showing their capabilities [1], [3], [4], [11], but most of them do not explicitly show how the teleoperation sessions start, pause, and terminate. AMO [3] and MobileTV [11] require two operators: one for the upper body and one for the lower body. TWIST [1] and CLONE [4] require only one operator for teleoperating the robot, but need another one to control the start/end of the entire process. We program the PICO’s handheld controllers to allow the demonstrator to safely and smoothly operate the entire system without the need for any assistance. The handheld controllers play the role of the

G. Whole-Body Visuomotor Policy Learning

Using the high-quality demonstration data collected through our teleoperation system, we develop a hierarchical visuomotor policy framework, as illustrated in Figure 7. This section details the design and training of the high-level visuomotor policy πhighauto.

Observation and action space. The visuomotor policy operates on visual observations and proprioceptive information to generate motion commands. Visual input consists of

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Teleop] Folding Towels

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Teleop] Transporting Baskets through Door

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Fig. 10: Long-horizon humanoid teleoperation powered by TWIST2. All tasks are achieved with streamed robot egocentric vision, full whole-body control, and a single operator.

360 × 640 RGB images captured by the ZED Mini camera, which are downsampled to 224 × 224 for computational efficiency. For robot proprioception, we use the historical command sequence pcmd rather than raw robot states s. This choice of proprioception serves two purposes: 1) it decouples the high-level policy from the low-level controller, enabling modular training and deployment, and 2) it mitigates error accumulation in this high-dimensional system by avoiding direct dependence on noisy raw robot states s. The action space consists of the same command vector pcmd used during teleoperation, ensuring consistency between data collection and policy execution. All proprioceptive inputs are normalized to improve training stability.

Network architecture. We employ Diffusion Policy [35] as our policy learning framework, utilizing 1D convolutional blocks for temporal modeling of action sequences. The policy predicts 64 action chunks using sample-based prediction [16], [36], corresponding to 2 seconds of future motion commands at the policy execution frequency. For visual encoding, we use a ResNet-18 backbone pre-trained with R3M [37], which provides robust visual representations learned from diverse robotic datasets.

Data augmentation and regularization. To enhance the robustness and generalization of the learned policy, we apply both state-space and visual augmentations. We inject 10% Gaussian noise into the proprioceptive inputs, encouraging the policy to rely more heavily on visual observations rather than overfitting to precise state information. For visual augmentation, we employ a comprehensive set of techniques including random cropping, random rotation, and color jittering. These augmentations improve the policy’s ability

to generalize across different lighting conditions, camera viewpoints, and visual variations that may occur during deployment.

Deployment and inference. For efficient real-time execution, the trained Diffusion Policy is converted to ONNX format, achieving a 20Hz inference rate on a single NVIDIA RTX 4090. We execute 48 out of the predicted 64-step action chunks at 30Hz, maintaining consistency with the data collection frequency.

IV. EXPERIMENT RESULTS

In this section, we show that powered by TWIST2, we can 1) teleoperate Unitree G1 to perform long-horizon challenging whole-body dexterous tasks, 2) collect imitation learning data effectively, and 3) make Unitree G1 autonomously perform whole-body tasks via its egocentric vision.

A. Long-Horizon Teleoperation

TWIST2 enables very long-horizon teleoperation. We showcase two representative tasks that cannot be achieved by previous systems (see Figure 10). We observe that 1) egocentric active perception and 2) smooth whole-body tracking instead of decoupled control are keys that enable such natural & smooth, long-horizon, whole-body, and mobile tasks.

Folding towels. The robot uses its egocentric vision to locate the towel, move the towel to its front, grasp it, and shakes it to spread. Then it will pinch the corner to fold the towel in half with two hands. It repeats the motion to fold into thirds (or quarters) to the target size, presses along the crease to set it, and neatly places the finished towel to its left-hand side. The entire process requires fine-grained control of the

[Figure 102]

[Figure 103]

[Autonomous] Kick-T

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

### [Autonomous] Whole-Body Dex Pick & Place

[Figure 121]

[Figure 122]

[Figure 123]

6x Consecutive Successful Grasp

- Fig. 11: Closed-loop whole-body visuomotor policy execution in the real world. TWIST2 enables effective and holistic whole-body humanoid data collection, which further enables versatile autonomous whole-body humanoid loco-manipulation & legged manipulation skills.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

wrists and hands, active vision, and whole-body reaching. Our robot can continuously fold 3 towels that are randomly placed on the table for now; and this is only bottlenecked by the underlying motor robustness, such as motor overheating. Transporting baskets through the door. The robot first adjusts its position via changing foot placements and bends down to pick up the baskets on its left side and on its right side, respectively. We casually put the basket so the teleoperator seeks the basket first via robot active perception. Then the robot moves close to the door, pushes the door open with the arm, walks across the door, and places the basket gently onto the shelf. Note that all the base movements of the robot are achieved via a single teleoperator by tracking the lower-body movements.

w/o Stereo w/o Neck w/o Egocentric View

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Human cannot grasp object beyond FOV.

Human cannot estimate actual depth.

Robot camera does not capture object.

Fig. 12: Comparison of different teleoperation settings.

successful bimanual pick&place or 2) around 50 successful mobile pick&place.

TABLE II: Scalable data collection. We show that we can easily collect several demonstrations via our system.

- B. Efficient Data Collection

We show that 1) how effective TWIST2 is in collecting imitation learning data and 2) how some key designs in our system improves data collection.

Task Time #Collected Episodes Success Rate Avg Time Per Episode

Bimanul Manip 1 18.5 min 98 100% 11 s Mobile Manip 19.5 min 46 100% 25 s

Second, we conduct a user study to quantify the effectiveness of our data collection system. We evaluate two users:

First, we show in Table II that within 20 minutes, the expert teleoperator can consecutively collect 1) around 100

1) an expert who has extensive experience using this system for data collection, and 2) a novice who is using the system for the first time during the test. Since the novice user gains proficiency through practice, we have them start with our complete system and then progressively remove features to isolate the impact of each component. As shown in Table III, TWIST2 achieves the shortest completion times and highest success rates across all configurations.

As illustrated in Figure 12, we observe several key findings: 1) without stereo vision for teleoperation, users tend to grasp higher than the actual object location, significantly increasing grasp failure rates; 2) without the neck module, users cannot perceive objects beyond the fixed field of view, making teleoperation extremely challenging; 3) when using third-person view with VR pass-through (i.e., w/o Egocentric View), the expert can collect data remarkably fast (10 episodes in 43 seconds), but this is only possible because the expert stands directly beside the robot, which is infeasible for long-horizon mobile manipulation tasks which require remote control via egocentric vision.

[Figure 134]

[Figure 135]

[Figure 136]

WB-Dex Robot View

[Figure 137]

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

[Figure 148]

[Figure 149]

Kick-T Robot View

[Figure 150]

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

Fig. 13: Visualization of training demonstrations (egocentric robot view and whole-body joint positions) for WB-Dex and Kick-T tasks.

TABLE III: Data collection efficiency of TWIST2 on different users and setups. The results show the necessity of using active egocentric stereo vision.

Collect 10 Demos Succes/Total Trials Time Cost (s)

Novice Expert Avg (Sum) Novice Expert Avg (Mean)

TWIST2 10/12 10/11 20/23 75.6 59.9 67.8 w/o Stereo 10/12 10/15 20/27 90.6 105.9 98.3 w/o Neck 7/17 9/12 16/29 144.0 80.5 112.3 w/o Egocentric View 10/13 10/10 20/23 94.3 43.0 68.7

- C. Whole-Body Policy Learning Results

We design two tasks to showcase autonomous results with our hierarchical visuomotor policy framework. We visualize the training data in Figure 13.

Whole-body dexterous pick & place (WB-Dex). In this task, the robot bends down to pick up a cup from the shelf using its dexterous hand and places it into a box on the ground. We train the policy with 170 human demonstrations and report the success and failure rates in Figure 14. We observe that the policy can reliably reach the cup in most cases. However, because the cup is very light, grasping it requires highly precise control; even a slight drift often results in grasp failure.

[Figure 162]

Fig. 14: All the success and failure cases in our WB-Dex task.

Kick T-shaped box to target (Kick-T). In this task, the robot uses its foot to kick a T-shaped green box toward a fixed T-shaped target position on the ground. The policy is trained with 50 demonstrations. In our data, the action pattern is consistent: the robot kicks with its left foot, and then takes a step forward with the right foot to maintain balance. This design ensures that the learned policy exhibits robust kicking behavior. We visualize policy rollouts in Figure 11. The policy successfully transports the T-shaped box to the target in 6 out of 7 trials. At present, the policy can only kick the box forward, without more flexible strategies such as walking around the box to adjust the kicking angle; we leave such capabilities to future work.

V. CONCLUSIONS AND LIMITATIONS

We introduce TWIST2, a portable and holistic mocapfree data collection system for humanoid robots with full whole-body control. By combining lightweight VR devices with an attachable neck for egocentric vision, our framework enables scalable data collection. On top of this, we designed a hierarchical visuomotor policy that allows a real humanoid robot to autonomously perform versatile whole-body skills including whole-body dexterous manipulation and Kick-T.

Limitations. 1) The general motion tracker struggles with highly dynamic movements such as sprinting due to challenges in tracking fast, complex motions. 2) PICO’s wholebody pose estimation is less accurate than high-cost motion capture systems, particularly for elbows and knees where no trackers are placed, resulting in reduced motion quality.

VI. DISCUSSIONS ON SCALING UP HUMANOID DATA There are several key challenges that must be addressed

before we can realistically scale up high-quality humanoid robot data.

Standardizing humanoid hardware. There is currently a wide range of humanoid platforms available

[Figure 163]

[Figure 164]

[Figure 165]

###### TWIST2 Neck 001 TWIST2 Neck 002 TWIST2 Neck 003

Fig. 15: We have manufactured 3 TWIST2 Necks, indicating that TWIST2 Neck is easy to assemble and can be democratized for research purposes.

for research—such as the Unitree G1/R1 and Booster T1/K1—which makes collected data difficult to reuse due to cross-embodiment discrepancies. Recently, the Unitree G1 has emerged as a popular choice because it offers a strong balance between performance and cost. At this stage, we advocate for standardizing humanoid hardware usage before attempting to scale data collection. In our work, we adopt the Unitree G1 platform and highlight that egocentric vision is crucial for capturing human-level manipulation data. We therefore introduce a low-cost neck add-on for the G1 that enables an egocentric camera setup and delivers roughly 80% of core human functionality.

Democratizing humanoid data collection. High-quality humanoid datasets have traditionally relied on motion-capture systems, limiting data collection to MoCap studios and specialized equipment. We demonstrate a portable and costefficient alternative that preserves rich human manipulation capabilities rather than only enabling whole-body movement. Sharing humanoid data. We believe that open humanoid datasets should serve as a foundation for future research. To that end, we publicly release all collected humanoid data on HuggingFace, and provide visualizations at https://twistdata.github.io , with the goal of making our dataset directly reusable and easily extendable by the community.

ACKNOWLEDGMENTS

We want to thank Charlie Cheng, Shaofeng Yin, Yuanhang Zhang, Yunchu Zhang, and Raven Huang for their help in real-world experiments. We also thank Wenhao Wang, Ke Jing, Ning Yang, Liuchuan Yu, and Zhigen Zhao for the helpful discussion in the PICO usage. The human motion datasets used in this work, including AMASS [32] and OMOMO [33], are solely for research purposes.

REFERENCES

- [1] Y. Ze, Z. Chen, J. P. Ara´ujo, Z. ang Cao, X. B. Peng, J. Wu, and C. K. Liu, “Twist: Teleoperated whole-body imitation system,” arXiv preprint arXiv:2505.02833, 2025.
- [2] Q. Ben, F. Jia, J. Zeng, J. Dong, D. Lin, and J. Pang, “Homie: Humanoid loco-manipulation with isomorphic exoskeleton cockpit,” arXiv preprint arXiv:2502.13013, 2025.
- [3] J. Li, X. Cheng, T. Huang, S. Yang, R. Qiu, and X. Wang, “Amo: Adaptive motion optimization for hyper-dexterous humanoid wholebody control,” Robotics: Science and Systems 2025, 2025.
- [4] Y. Li, Y. Lin, J. Cui, T. Liu, W. Liang, Y. Zhu, and S. Huang, “Clone: Closed-loop whole-body humanoid teleoperation for longhorizon tasks,” arXiv preprint arXiv:2506.08931, 2025.

- [5] OpenAI, “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774,

2023. [Online]. Available: https://arxiv.org/abs/2303.08774

- [6] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn,

N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π0: A visionlanguage-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

- [7] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg,

- K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang,
- L. Yu, and U. Zhilinsky, “π0.5: a vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025, published: April 22, 2025.

- [8] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” arXiv preprint arXiv:2304.13705, 2023.
- [9] J. Aldaco, T. Armstrong, R. Baruch, J. Bingham, S. Chan, K. Draper, D. Dwibedi, C. Finn, P. Florence, S. Goodrich, W. Gramlich, T. Hage, A. Herzog, J. Hoech, T. Nguyen, I. Storz, B. Tabanpour, L. Takayama, J. Tompson, A. Wahid, T. Wahrburg, S. Xu, S. Yaroshenko, K. Zakka, and T. Z. Zhao, “Aloha 2: An enhanced low-cost hardware for bimanual teleoperation,” arXiv preprint arXiv:2405.02292, 2024.
- [10] P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel, “Gello: A general, lowcost, and intuitive teleoperation framework for robot manipulators,” 2023.
- [11] C. Lu, X. Cheng, J. Li, S. Yang, M. Ji, C. Yuan, G. Yang, S. Yi, and X. Wang, “Mobile-television: Predictive motion priors for humanoid whole-body control,” arXiv preprint arXiv:2412.07773, 2024.
- [12] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “Humanplus: Humanoid shadowing and imitation from humans,” in Conference on Robot Learning (CoRL), 2024.
- [13] PICO Immersive Pte. Ltd., “PICO 4 Ultra: An All-New Mixed Reality Experience,” https://www.picoxr.com/global/products/pico4-ultra, 2023.
- [14] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi, “Omnih2o: Universal and dexterous humanto-humanoid whole-body teleoperation and learning,” arXiv preprint

- arXiv:2406.08858, 2024.

[15] X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang, “Open-television: Teleoperation with immersive active visual feedback,” arXiv preprint

- arXiv:2407.01512, 2024.

- [16] Y. Ze, Z. Chen, W. Wang, T. Chen, X. He, Y. Yuan, X. B. Peng, and J. Wu, “Generalizable humanoid manipulation with improved 3d diffusion policies,” arXiv preprint arXiv:2410.10803, 2024.
- [17] H. Wang, Z. Wang, J. Ren, Q. Ben, T. Huang, W. Zhang, and J. Pang, “Beamdojo: Learning agile humanoid locomotion on sparse footholds,” in Robotics: Science and Systems (RSS), 2025.
- [18] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa, “Visual imitation enables contextual humanoid control,” arXiv preprint arXiv:2505.03729, 2025.
- [19] J. Long, J. Ren, M. Shi, Z. Wang, T. Huang, P. Luo, and J. Pang, “Learning humanoid locomotion with perceptive internal model,”

2024. [Online]. Available: https://arxiv.org/abs/2411.14386

- [20] S. Chen, Y. Ye, Z.-A. Cao, J. Lew, P. Xu, and C. K. Liu, “Hand-eye autonomous delivery: Learning humanoid navigation, locomotion and reaching,” arXiv preprint arXiv:2508.03068, 2025.
- [21] Z. Luo, C. Tessler, T. Lin, Y. Yuan, T. He, W. Xiao, Y. Guo, G. Chechik, K. Kitani, L. Fan et al., “Emergent active perception and dexterity of simulated humanoids from visual reinforcement learning,” arXiv preprint arXiv:2505.12278, 2025.
- [22] T. E. Truong, Q. Liao, X. Huang, G. Tevet, C. K. Liu, and K. Sreenath, “Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion,” arXiv preprint arXiv:2508.08241, 2025.
- [23] H. Xiong, X. Xu, J. Wu, Y. Hou, J. Bohg, and S. Song, “Vision in action: Learning active perception from human demonstrations,” arXiv preprint arXiv:2506.15666, 2025.
- [24] H. Shi, W. Wang, S. Song, and C. K. Liu, “Toddlerbot: Opensource ml-compatible humanoid platform for loco-manipulation,” arXiv preprint arXiv:2502.00893, 2025.
- [25] PICO Immersive Pte.Ltd., “PICO Motion Tracker,” https://www. picoxr.com/global/products/pico-motion-tracker, 2023.

- [26] Z. Zhao, L. Yu, K. Jing, and N. Yang, “Xrobotoolkit: A cross-platform framework for robot teleoperation,” arXiv preprint arXiv:2508.00097, 2025.
- [27] “Vive tracker (3.0),” https://www.vive.com/us/accessory/tracker3/, 2025.
- [28] B. Dynamics and T. R. Team, “Large behavior models and atlas find new footing,” https://bostondynamics.com/blog/ large-behavior-models-atlas-find-new-footing/, Aug. 2025, accessed: 2025-09-08.
- [29] J. P. Araujo, Y. Ze, P. Xu, J. Wu, and C. K. Liu, “Retargeting matters: General motion retargeting for humanoid motion tracking,” arXiv preprint arXiv:2510.02252, 2025.
- [30] Z. Chen, M. Ji, X. Cheng, X. Peng, X. B. Peng, and X. Wang, “Gmt: General motion tracking for humanoid whole-body control,” arXiv:2506.14770, 2025.
- [31] Y. Ze, J. P. Ara´ujo, J. Wu, and C. K. Liu, “Gmr: General motion retargeting,” https://github.com/YanjieZe/GMR, 2025, gitHub repository.
- [32] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “Amass: Archive of motion capture as surface shapes,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5442–5451.
- [33] J. Li, J. Wu, and C. K. Liu, “Object motion guided human motion synthesis,” ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1–11, 2023.
- [34] Redis contributors, “redis/redis: The redis in-memory data structure store,” 2025, accessed: 2025-09-09. [Online]. Available: https: //github.com/redis/redis
- [35] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.
- [36] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” arXiv preprint arXiv:2403.03954, 2024.
- [37] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” arXiv preprint arXiv:2203.12601, 2022.

