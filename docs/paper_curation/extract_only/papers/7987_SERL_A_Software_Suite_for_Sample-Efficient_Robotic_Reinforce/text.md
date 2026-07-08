[Figure 1]

January 2024

# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

###### Jianlan Luo1*, Zheyuan Hu1*, Charles Xu1, You Liang Tan1, Jacob Berg2, Archit Sharma3, Stefan Schaal4, Chelsea Finn3, Abhishek Gupta2 and Sergey Levine1

*Equal Contribution, 1Department of EECS, University of California, Berkeley, 2Department of Computer Science and Engineering, University of Washington, 3Department of Computer Science, Stanford University, 4Intrinsic Innovation LLC

[Figure 2]

[Figure 3]

[Figure 4]

## arXiv:2401.16013v4[cs.RO]20Mar2025

Figure 1: Depiction of various tasks solved using SERL in the real world. These include PCB board insertion (left), cable routing (middle), and object relocation (right). SERL provides an out-of-the-box package for real-world reinforcement learning, with support for sample-efficient learning, learned rewards, and automation of resets.

In recent years, significant progress has been made in the field of robotic reinforcement learning (RL), enabling methods that handle complex image observations, train in the real world, and incorporate auxiliary data, such as demonstrations and prior experience. However, despite these advances, robotic RL remains hard to use. It is acknowledged among practitioners that the particular implementation details of these algorithms are often just as important (if not more so) for performance as the choice of algorithm. We posit that a significant challenge to widespread adoption of robotic RL, as well as further development of robotic RL methods, is the comparative inaccessibility of such methods. To address this challenge, we developed a carefully implemented library containing a sample efficient off-policy deep RL method, together with methods for computing rewards and resetting the environment, a high-quality controller for a widely-adopted robot, and a number of challenging example tasks. We provide this library as a resource for the community, describe its design choices, and present experimental results. Perhaps surprisingly, we find that our implementation can achieve very efficient learning, acquiring policies for PCB board assembly, cable routing, and object relocation between 25 to 50 minutes of training per policy on average, improving over state-of-the-art results reported for similar tasks in the literature. These policies achieve perfect or near-perfect success rates, extreme robustness even under perturbations, and exhibit emergent recovery and correction behaviors. We hope that these promising results and our high-quality open-source implementation will provide a tool for the robotics community to facilitate further developments in robotic RL. Our code, documentation, and videos can be found at https://serl-robot.github.io/

### 1. Introduction

Considerable progress on robotic reinforcement learning (RL) over the recent years has produced impressive results, with robots playing table tennis (Büchler et al., 2022), manipulating objects from raw images (Gupta et al., 2021; Kalashnikov et al., 2021; Levine et al., 2016b), grasping diverse objects (Levine et al., 2018; Mahler

et al., 2017), and performing a wide range of other skills. However, despite the significant progress on the underlying algorithms, RL remains challenging to use for real-world robotic learning problems, and practical adoption has been more limited. We argue that part of the reason for this is that the implementation of RL algorithms, particularly for real-world robotic systems, presents a very large design space, and it is the

Corresponding author(s): Jianlan Luo and Zheyuan Hu, Berkeley AI Research Lab (BAIR), University of California, Berkeley: jianlanluo@eecs.berkeley.edu, huzheyuan@berkeley.edu.

challenge of navigating this design space, rather than limitations of algorithms per se, that limit adoption. It is often acknowledged by practitioners in the field that details in the implementation of an RL algorithm might be as important (if not more important) as the particular choice of algorithm. Furthermore, real-world learning presents additional challenges with reward specification, implementation of environment resets, sample efficiency, compliant and safe control, and other difficulties that put even more stress on this issue. Thus, adoption and further research progress on real-world robotic RL may well be bottlenecked on implementation rather than novel algorithmic innovations.

To address this challenge, our aim in this paper is to provide an open-source software framework, which we call Sample-Efficient Robotic reinforcement Learning (SERL), that aims to facilitate wider adoption of RL in real-world robotics. SERL consists of the following components: (1) a high-quality RL implementation that is geared towards real-world robotic learning and supports image observations and demonstrations; (2) implementations of several reward specification methods that are compatible with image observations, including classifiers and adversarial training; (3) support for learning “forward-backward” controllers that can automatically reset the task between trials (Eysenbach et al., 2018); (4) a software package that can in principle connect the aforementioned RL component to any robotic manipulator; and (5) an impedance controller design principle that is particularly effective for dealing with contact-rich manipulation tasks. Our aim in this paper is not to propose novel algorithms or methodology, but rather to offer a resource for the community to provide roboticists with a well-designed foundation both for future research on robotic RL, and other methods that might employ robotic RL as a subroutine. However, in the process of evaluating our framework, we also make a scientifically interesting empirical observation: when implemented properly in a carefully engineered software package, current sample-efficient robotic RL methods can attain very high success rates with relatively modest training times. The tasks in our evaluation are illustrated in Fig. 1: precise insertion tasks involving dynamic contact, deformable object manipulation with complex dynamics, and object relocation where the robot must learn without manually designed resets. For each of these tasks, SERL is able to learn effectively within 15 - 60 min of training per policy (in terms of total wall-clock time), achieving near-perfect success rates, despite learning policies that operate

on image observations. This result is significant because RL, particularly with deep networks and image inputs, is often considered to be highly inefficient. Our results challenge this assumption, suggesting careful implementations of existing techniques, combined with well-designed controllers and carefully selected components for reward specification and resets, can provide an overall system that is efficient enough for real-world use.

### 2. Related Work

While our framework combines existing RL methods into a complete robotic learning system, the particular combination of parts is carefully designed to provide for efficient and out-of-the-box reinforcement learning directly in the real world and, as shown in our experiments, achieves excellent results on a wide range of tasks. Here, we summarize both related prior methods and systems.

Algorithms for real-world RL: Real-world robotic RL demands algorithms that are sample-efficient, can utilize onboard perception, and support easily specified rewards and resets. A number of algorithms have shown the ability to learn very efficiently directly in the real world (Riedmiller et al., 2009; Westenbroek et al., 2022; Yang et al., 2020; Zhan et al., 2021; Hou

- et al., 2020; Tebbe et al., 2021; Popov et al., 2017; Luo et al., 2019; Zhao et al., 2022; Hu et al., 2024; Johannink et al., 2018; Schoettler et al., 2020), using variants of off-policy RL (Kostrikov et al., 2023; Hu et al., 2024; Luo et al., 2023), model-based RL (Hester and Stone, 2013; Wu et al., 2022; Nagabandi et al., 2019; Rafailov
- et al., 2021; Luo et al., 2018), and on-policy RL (Zhu et al., 2019). These advances have been paired with advances in inferring rewards from raw visual observation through success classifiers (Fu et al., 2018; Li et al., 2021), foundation-model-based rewards (Du et al., 2023; Mahmoudieh et al., 2022; Fan et al., 2022), and rewards from videos (Ma et al., 2023b;a). Additionally, to enable autonomous training, there have been a number of algorithmic advances in reset-free learning (Gupta et al., 2021; Sharma et al., 2021; Zhu et al., 2020; Xie et al., 2022; Sharma et al., 2023) that enable autonomous training with minimal human interventions. While these algorithmic advances are important, the contribution we make in this work is to provide a framework and software package to enable sample efficient reinforcement learning in the real world with a ready-made choice of methods that can work well for a variety of tasks. In doing so, we hope to lower the barrier of entry for

dynamics, and 𝑟∶  ×  →ℝ is the reward function, which encodes the task. An optimal policy 𝜋 is one that maximizes the cumulative expected value of the reward, i.e., 𝐸[∑∞𝑡=0 𝛾𝑡𝑟(𝐬𝑡,𝐚𝑡)], where the expectation is taken with respect to the initial state distribution, transition probabilities, and policy 𝜋.

new researchers to build better algorithms and training methodologies for robot learning in the real world.

Software packages for RL: There are a number of packages (Seno and Imai, 2022; Nair and Pong; Hill et al., 2018; Guadarrama et al., 2018) for RL, though to our knowledge, none aim to directly address real-world robotic RL specifically. SERL builds on the recently proposed RLPD algorithm, which is an off-policy RL algorithm with a high update-to-data ratio. SERL is not a library of RL algorithms for training agents in simulation, although it could be adapted to be so. Rather, SERL offers a full stack pipeline for robot control, going from low-level controllers to the interface for asynchronous and efficient training with an RL algorithm to additional machinery for inferring rewards and training without resets. In doing so, SERL provides an off-the-shelf package to help non-experts start using RL to train their physical robots in the real world, unlike prior libraries that aim to provide implementations of many methods – that is, SERL offers a full “vertical” integration of components, whereas prior libraries focus on the “horizontal.” SERL is also not an RL benchmark package such as (Yu et al., 2019; James et al., 2020; Mittal et al., 2023). SERL allows users to define their own tasks and success metrics directly in the real world, providing the software infrastructure for actually controlling and training robotic manipulators in these tasks.

While the specification of the RL task is concise and simple, turning real-world robotic learning problems into RL problems requires care. First, the sample efficiency of the algorithm for learning 𝜋 is paramount: when the learning must take place in the real world, every minute and hour of training comes at a cost. Sample efficiency can be improved by using effective off-policy RL algorithms (Konda and Tsitsiklis, 1999; Haarnoja et al., 2018; Fujimoto et al., 2018), but it can also be accelerated by incorporating prior data and demonstrations (Rajeswaran et al., 2018; Ball et al., 2023; Nair et al., 2020), which is important to achieve the fastest training times.

Additionally, many of the challenges with robotic RL lie beyond just the core algorithm for optimizing 𝜋. For example, the reward function 𝑟 might depend on image observations, and difficult for the user to specify manually. Additionally, for episodic tasks where the robot resets to an initial state 𝐬0 ∼𝜌(𝐬0) between trials, actually resetting the robot (and its environment) into one of these initial states is a mechanical operation that must somehow be automated.

Software for real-world RL: There have been several previous packages that have proposed infrastructure for real world RL: for dexterous manipulation (Ahn et al., 2019), tabletop furniture assembly (Heo et al., 2023), legged locomotion (Kostrikov et al., 2023), and peg insertion (Levine et al., 2016a). These packages are effective in narrow situations, either using privileged information or training setups such as explicit tracking (Levine et al., 2016a; Ahn et al., 2019) or pure proprioception (Kostrikov et al., 2023), or limited to imitation learning. In SERL, we show a full stack system that can be used for a wide variety of robotic manipulation tasks without requiring privileging of the training setups as in prior work.

Furthermore, the controller layer, which interfaces the MDP actions 𝐚 (e.g., end-effector poses) to the actual low-level robot controls, also requires great care, particularly for contact-rich tasks where the robot physically interacts with objects in the environment. Not only does this controller need to be accurate, but it must also be safe enough that the RL algorithm can explore with random actions during training.

SERL will aim to provide ready-made solutions to each of these challenges, with a high-quality implementation of a sample-efficient off-policy RL method that can incorporate prior data, several choices for reward function specification, a forward-backward algorithm for learning resets, and a controller suitable for learning contact-rich tasks without damaging either the robot or objects in the environment.

### 3. Preliminaries and Problem Statement

Robotic reinforcement learning tasks can be defined via an MDP  = {,,𝜌,,𝑟,𝛾}, where 𝐬∈  is the state observation (e.g., an image in combination with the current end-effector position), 𝐚∈  is the action (e.g., the desired end-effector pose), 𝜌(𝐬0) is a distribution over initial states,  is the unknown and potentially stochastic transition probabilities that depend on the system

### 4. Sample Efficient Robotic Reinforcement Learning in the Real-World

Our software package, which we call Sample-Efficient Robotic reinforcement Learning (SERL), aims to make robotic RL in the real world accessible by providing

ready-made solutions to the problems detailed in the previous section. This involves providing efficient vision-based reinforcement learning algorithms and the infrastructure needed to support these learning algorithms for autonomous learning. We note that the purpose of such an endeavor is not to propose novel algorithms or tools, but rather to develop a software package that anyone can use easily for robotic learning, without complex setup procedures and painful integration across libraries.

The core reinforcement learning algorithm is derived from RLPD (Ball et al., 2023), which itself is a variant of soft actor-critic (Haarnoja et al., 2018): an off-policy Q-function actor-critic method that can readily incorporate prior data (either suboptimal data or demonstrations) into the replay buffer for efficient learning. The reward functions can be specified either with a binary classifier or VICE (Fu et al., 2018), which provides a method to update the classifier during RL training with additional negatives from the policy. The reward function can also be specified by hand in cases where the robot state is sufficient to evaluate success (e.g., as in our PCB board assembly task). The resets can be provided via a forward-backward architecture (Sharma et al., 2021), where the algorithm simultaneously trains two policies: a forward policy that performs the task, and a backward policy that resets the environment back to the initial state. On the robot system side, we also provide a universal adapter for interfacing our method to arbitrary robots, as well as an impedance controller that is particularly well-suited for contact-rich manipulation tasks.

- 4.1. Core RL Algorithm: RLPD There are several desiderata for reinforcement learning algorithm to be deployed in this setting: (1) it must be efficient and able to make multiple gradient updates per time step, (2) it must be able to incorporate prior data easily and then continue improving with further experience, (3) it must be simple to debug and build on for new users. To this end, we build on the recently proposed RLPD (Ball et al., 2023) algorithm, which has shown compelling results on sample-efficient robotic learning. RLPD is an off-policy actor-critic reinforcement learning algorithm that builds on the success of temporal difference algorithms such as soft-actor critic (Haarnoja et al., 2018), but makes some key modifications to satisfy the desiderata above. RLPD makes three key changes: (i) high update-to-data ratio training (UTD), (ii) symmetric sampling between prior data and

on-policy data, such that half of each batch comes from prior data and half from the online replay buffer, and (iii) layer-norm regularization during training. This method can train from scratch, or use prior data (e.g., demonstrations) to bootstrap learning. Each step of the algorithm updates the parameters of a parametric Q-function 𝑄𝜙(𝐬,𝐚) and actor 𝜋𝜃(𝐚|𝐬) according to the gradient of their respective loss functions:

𝑄(𝜙)=𝐸𝐬,𝐚,𝐬′[(𝑄𝜙(𝐬,𝐚)−(𝑟(𝐬,𝐚)+𝛾𝐸𝐚′∼𝜋𝜃[𝑄𝜙̄(𝐬′,𝐚′)]))

2

𝜋(𝜃)=−𝐸𝐬 [𝐸𝐚∼𝜋𝜃(𝐚)[𝑄𝜙(𝐬,𝐚)] +𝛼(𝜋𝜃(⋅|𝐬)], ]

where 𝑄𝜙̄ is a target network (Mnih et al., 2013), and the actor loss uses entropy regularization with an adap-

tively adjusted weight 𝛼 (Haarnoja et al., 2018). Each update step uses a sample-based approximation of each expectation, with half of the samples drawn from the prior data (e.g., demonstrations), and half drawn from the replay buffer (Mnih et al., 2013). For efficient learning, multiple update steps are performed per time step in the environment, which is referred to as the updateto-date (UTD) ratio, and regularizing the critic with layer normalization allows for higher UTD ratios and thus more efficient training (Ball et al., 2023).

#### 4.2. Reward Specification with Classifiers

Reward functions are difficult to specify by hand when learning with image observations, as the robot typically requires some sort of perception system just to determine if the task was performed successfully. While some tasks, such as the PCB board assembly task in Fig. 1, can accommodate hand-specified rewards based on the location of the end effector (under the assumption that the object is held rigidly in the gripper), most tasks require rewards to be deduced from images. In this case, the reward function can be provided by a

binaryand outputsclassifierthe probabilitythat takesofinathebinarystate“event”observation𝑒, corre-𝐬 sponding to successful completion. The reward is then given by 𝑟(𝐬) = log𝑝(𝑒|𝐬).

This classifier can be trained either using handspecified positive and negative examples, or via an adversarial method called VICE (Fu et al., 2018). The latter addresses a reward exploitation problem that can arise when learning with classifier based rewards, and removes the need for negative examples in the classifier training set: when the RL algorithm optimizes the reward 𝑟(𝐬) = log𝑝(𝑒|𝐬), it can potentially discover “adversarial” states that fool the classifier 𝑝(𝑒|𝐬) to erroneously output high probabilities. VICE addresses this

issue by adding all states visited by the policy into the training set for the classifier with negative labels, and updating the classifier after each iteration. In this way, the RL process is analogous to a generative adversarial network (GAN) (Goodfellow et al., 2014), with the policy acting as the generator and the reward classifier acting as the discriminator. Our framework thus supports all three types of rewards.

#### 4.3. Reset-Free Training with ForwardBackward Controllers

When learning episodic tasks, the robot must reset the environment between task attempts. For example, when learning the object relocation task in Figure 1, each time the robot successfully moves the object to the target bin, it must then take it out and place it back into the initial bin. To remove the need for human effort in “resets”, SERL supports “reset-free” training by using forward and backward controllers (Han et al., 2015; Gupta et al., 2021). In this setup, two policies are trained simultaneously using two independent RL agents, each with its own policy, Q-function, and reward function (specified via the methods in the previous section). The forward agent learns to perform the task, and the backward agent learns to return to the initial state(s). While more complex reset-free training procedures can also be possible (Gupta et al., 2021), we find that this simple recipe is sufficient for learning object manipulation tasks like the repositioning skill in Figure 1.

##### 4.4. Software Components Environment adapters: SERL aims to be easily usable for many robot environments. Although we provide a set of Gym environment wrappers and robot environments for the Franka arm as starter guides, users can also use their own existing environments or develop new environments as they see fit. Thus, the library does not impose additional constraints on the robot environment as long as it is Gym-like (Brockman et al.,

2016) as shown in Fig. 2. We welcome contributions from the community to extend the support for readily deployable environment wrappers for other robots and tasks.

Actor and learner nodes: SERL includes options to train and act in parallel to decouple inferring actions and updating policies with a few lines of code as illustrated in Fig. 2. We found this to be beneficial in sampleefficient real-world learning problems with high UTD ratios. By separating actor and learner on two different threads, SERL not only preserves the control frequency

[Figure 5]

Figure 2: Software architecture and real-world robot training example code. SERL runs three parallel processes, consisting of the actor, which chooses actions, and the learner node, which actually runs the training code, and the robot environment, which executes the actions from the actor and contributes data back to the learner.

at a fixed rate, which is crucial for tasks that require immediate feedback and reactions, such as deformable objects and contact-rich manipulations, but also reduces the total wall-clock time spend training in the real world.

#### 4.5. Impedance Controller for Contact-Rich Tasks

Although our package should be compatible with any OEM robot controller, as described in Sec. 4, we found that the choice of controllers can heavily affect the final performance. This is more pronounced for contactrich manipulation. For example, in the PCB insertion task in Fig. 1, an overly stiff controller might bend the fragile pins and make insertion difficult, while an overly compliant controller might struggle to move the object into position quickly.

A typical setup for robotic RL employs a two-layered control hierarchy, where an RL policy produces setpoint actions at a much lower frequency than the downstream real-time controller. The RL controller can set targets for the low-level controller to cause physically undesirable consequences. To illustrate this, let’s consider the hierarchical controller structure presented in Fig. 4, where a high-level RL controller 𝜋(𝐚|𝐬) sends control targets at 10HZ for the low-level impedance controller to track at 1K HZ, so one timestep from RL will block 100 timesteps of the low-level controller to execute. A typical impedance control objective for this controller is

𝐹=𝑘𝑝 ⋅𝑒+𝑘𝑑 ⋅𝑒̇+𝐹𝑓 𝑓 +𝐹𝑐𝑜𝑟,

where 𝑒=𝑝−𝑝𝑟𝑒𝑓 , 𝑝 is the measured pose, and 𝑝𝑟𝑒𝑓 is the target pose computed by the upstream controller, 𝐹𝑓 𝑓 is the feed-forward force, 𝐹𝑐𝑜𝑟 is the Coriolis force,

[Figure 6]

- Figure 3: Visualization of controller logs from the robot when commanded with different movements, for the z-axis of the endeffector. The orange line is the commanded target (the output of RL), red is the smoothed target sent to the real-time controller, blue is the clipped target, and green is the robot position after executing this controller. Left: The robot end-effector was commanded to move into contact with a hard surface and continue the movement despite the contact. The reference limiting mechanism clipped the target to avoid a hard collision. Right: The command is a fast free-space movement, which our reference limiting mechanism does not block, allowing fast motion to the target.

[Figure 7]

- Figure 4: A typical controller hierarchy for robotics RL. The output from the RL policy is tracked within a block of time by the downstream controller.

this objective will then be converted into joint space torques by multiplying Jacobian transpose and offset by nullspace torques. It acts like a spring-damper system around the equilibrium set by 𝑝𝑟𝑒𝑓 with the stiffness coefficient being 𝑘𝑝 and the damping coefficient being 𝑘𝑑. As described above, this system will yield large forces if 𝑝𝑟𝑒𝑓 is far away from the current pose, which can lead to a hard collision or damage when the arm is in contact with something. Therefore it’s crucial to constrain the interaction force generated by it. However, directly reducing gains will hurt the controller’s accuracy. Thus, we should bound 𝑒 so that |𝑒|≤Δ, and then the generated force from the spring-damper system will be bounded to 𝑘𝑝 ⋅|Δ|+ 2𝑘𝑑 ⋅|Δ|⋅𝑓 , 𝑓 is the control frequency.

One might wonder if we should directly clip the action output by the RL policy. This might seem reasonable, but can be impractical in some scenarios: some objects such as the PCB board may require a very small interaction force, implying a very small Δ, usually on the order of micrometers; if the RL policy is only allowed to move at increments of micrometers, it would result in an extremely long learning process or very un-

stable training, because the episode would need enough time steps to allow the arm to move over long distances (e.g., approaching the insertion point). However, if we directly clip at the real-time layer, this will largely mitigate the issue without the need to constrain the RL policy to small actions. It will not block the free space movement of the RL policy as long as 𝑀⋅|Δ|≥|𝑎|𝑚𝑎𝑥, where 𝑀 is the number of control time-steps inside a block, as in Fig. 4. This value is usually large (e.g., 𝑀= 100). At the same time, we strictly enforce the reference constraint at the real-time level whenever in contact. One might also wonder if it’s possible to achieve the same result by using an external force/torque sensor. This may be undesirable for several reasons: (1) force/torque sensors can have significant noise, and obtaining the right hardware and calibration can be difficult; (2) even if we get such a threshold value, it’s nontrivial to design robot motions to accommodate policy learning as well as obeying the force constraint. In practice, we found that clipping the reference in this way is simple but very effective, and is crucial to enable RL-based contact-rich manipulation tasks. We tested our controller on a Franka Panda robot and included the Franka Panda implementation with our package. However, this principle can be easily implemented on any torque-controlled robot. To verify the actual performance of the proposed controller, we report the actual tracking performance of moving the robot in free space and in contact with a table surface as in Fig. 3, where we can see the controller indeed clamps the reference whenever in contact, while permitting fast movement in free space.

#### 4.6. Relative Observation and Action Frame

The choice of action space is particularly important both for the ease of the RL training process and the ability of the learned policy to generalize to perturbations at test time. While SERL can operate on various action representations via a standard RL environment interface, we found that a convenient mechanism of representing observations and actions in a relative coordinate system.

To develop an agent capable of adapting to a dynamic target, we propose a training procedure that simulates a moving target without the need for physical movement. The target, for instance, the PCB insertion socket holes, is fixed relative to the robot base frame, and the reward can be specified using any of the standard methods provided in Sec. 4.2. At the beginning of each training episode, the pose of the robot’s end-effector was

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- Figure 5: Illustration of the robot performing each task with our method: PCB Insertion (top left), Cable Routing (top right), Object Relocation - Forward (bottom left), and Object Relocation - Backward (bottom right). The green box indicates a state where the robot receives a high reward for completing the task.

Package Task Training time Success rate Demos Shaping? Vision? Open-sourced? Guided Policy Search(Levine et al., 2016b) Peg insertion 3 hours 70% 0 Yes Yes Yes

DDPGfD (Vecerik et al., 2018) Peg/clip insertion 1.5-2.5 hours 97% / 77% 30 No Yes No Visual Residual RL (Schoettler et al., 2020) Connector insertion Not mentioned 52% ∼ 100% 0 Yes Yes No

SHIELD (Luo et al., 2021) Connector insertion 1.5 hours 99.8% 25 No Yes No InsertionNet (Spector and Castro, 2021) Connector insertion 40 mins 78.5% - 100% 0 Yes Yes No

SERL (Ours) PCB Insertion 20 mins 100% 20 No Yes Yes

- Table 1: Comparison to results reported on similar tasks in prior work. The overall success rates for our method are generally higher, and the training times are generally lower, as compared to prior results. Note also that the PCB board assembly task, shown in Figure 1, has very tight tolerances, likely significantly tighter than the coarser peg and connector insertion tasks studied in the prior works.

randomized uniformly within a pre-defined area in the workspace. The robot’s proprioceptive information is expressed with respect to frame of the end-effector’s initial pose; the action output from the policy (6D twist) is relative to the current end-effector frame. This is equivalent to physically moving the target when viewed relatively from the frame attached to the end-effector. More details on are described in the appendix 7. As a result, the policy can succeed even if the object moves or, as in some of our experiments, is perturbed in the middle of the episode.

### 5. Experiments

Our experimental evaluation aims to study how efficiently our system can learn a variety of robotic manipulation tasks, including contact-rich tasks, deformable object manipulation, and free-floating object manipulation. These experiments demonstrate the breadth of applicability and efficiency of SERL. We use a Franka Panda arm and two wrist cameras attached to the endeffector to get close-in views. Further details can be found at https://serl-robot.github.io/. We use an ImageNet pre-trained ResNet-10 (He et al., 2015) as a vision backbone for the policy network and connect it to a 2-layer MLP. Observations include camera images and robot proprioceptive information such as end-effector pose, twist, force, and torque. The policy outputs a 6D end-effector delta pose from the current pose, which

is tracked by the low-level controller. The evaluation tasks are illustrated in Fig. 5 and described below:

PCB insertion: Inserting connectors into a PCB board demands fine-grained, contact-rich manipulation with sub-millimeter precision. This task is ideal for realworld training, as simulating and transferring such contact-rich interactions can be challenging. At the beginning of each training and evaluation episode, the initial end effector pose is sampled uniformly from a starting region, as described in Table 2.

Cable routing: This task involves routing a deformable cable into a clip’s tight-fitting slot. This task requires the robot to perceive the cable and carefully manipulate it so that it fits into the clip while holding it at another location. This is particularly difficult for any method that relies on model-based control, or makes rigid-object assumptions, since both visual perception and handling of deformable objects is essential for access. Tasks of this sort often arise in manufacturing and maintenance scenarios. Similarly to the PCB task, the initial end effector pose is sampled uniformly within a starting region, as described in Table 2.

Object relocation: This task requires moving a freefloating object between bins, requiring grasping and relocation. The intricacies of reward inference and reset-free training become especially pronounced in the manipulation of such free-floating objects. We define the forward task as picking up the object from the bin

Task # of Demos Image Input Random Reset Reward Specification Bin Size Training Time PCB Component Insertion 20 2 wrist camera True Ground Truth 10cm 10cm 20 mins

Cable Routing 20 2 wrist camera True Binary Classifier 20cm 20cm 31 mins Object Relocation (Forward-Backward) 20 1 wrist, 1 side camera False Binary Classifier 20cm × 30cm 105 mins

- Table 2: Task parameters: During demo collection for both BC and RL, as well as online training, each episode’s initial end-effector pose resets uniformly at random within a fixed region for the PCB and Cable task, while the free-floating object relocation task resets above the center of each bin.

[Figure 12]

- Figure 6: Success rate comparisons: When evaluated for 100 trials per task, learned RL policies outperformed BC policies by a large margin, by 1.7x for Object Relocation, by 5x for Cable Routing, and by 10x for PCB Insertion.

on the right side and placing it on the left, while the backward task moves the object back to the starting bin, undoing the forward task.

For each task, we initialize RL training from 20 teleoperated demonstrations using a Space Mouse. To confirm that demonstrations alone are insufficient to solve the task, we include a behavioral cloning (BC) baseline using 100 high-quality expert teleoperated demonstrations, roughly matching the total amount of data in the RL replay buffer when RL converges. Note that this is 5 times more demonstrations than the amount provided by our method. Both RL and BC demonstrations are collected using the initial end effector randomization scheme described in Table 2. All training was done on a single Nvidia RTX 4090 GPU.

Results: We report the results in Table 2, and show example executions in Fig. 5. We evaluated both BC and RL policies under the same conditions and protocols as detailed in Section 5. Our RL policies achieve perfect success rates on all three tasks over all 100 trials. For the PCB insertion and cable routing task, our RL policies converge in under 30 minutes of real-world training, which includes all computation, resets, and intended stops. The free-floating object relocation task learns two policies (forward and backward), and total

[Figure 13]

Figure 7: Cycle time comparison: We recorded the average time taken for the robot to succeed in each task. RL policies are at least 2x faster than BC policies trained with 100 high-quality human teleoperated demonstrations for all three tasks.

time amounts to less than an hour per policy. For the cable routing task and PCB insertion task, our policies outperform BC baselines by a large margin, despite training with 5x fewer demonstrations than BC, suggesting that demos alone are insufficient. We report the results in terms of success rate and cycle time in Fig. 6 and Fig. 7. The learned RL policies not only outperformed their BC counterparts by as much as 10x in terms of success rate but also improved on the cycle time of the initial human demonstrations by up to 3x.

Comparison to prior systems: While it’s difficult to directly compare our results to those of prior systems due to numerous differences in the setup, lack of consistently open-sourced code, and other discrepancies, we provide a summary of training times and success rates reported for tasks that are most similar to our PCB board insertion task in Table 2. We chose this task because similar insertion or assembly tasks have been studied in prior work, and such tasks often present challenges with precision, compliant control, and sample efficiency. Compared to these prior works, our experiments do not use shaped rewards, which might require extensive engineering, though we do utilize a small amount of demonstration data (which some prior works eschew). The results reported in

[Figure 14]

Figure 8: Peg Insertion Task at University of Washington

these prior works generally have either lower success rates or longer training times, or both, suggesting our implementation of sample-efficient RL matches or exceeds the performance of state-of-the-art methods in the literature, at least on this type of task. The closest performance to ours in the work of Spector et al. (Spector and Castro, 2021) includes a number of design decisions and inductive biases that are specific to insertion, whereas our method is generic and makes minimal taskspecific assumptions. Although the components of our system are all based on (recent) prior work, the stateof-the-art performance of this combination illustrates our main thesis: the details of how deep RL methods are implemented can make a big difference.

Reproducibility: A core mission of SERL is to lower the setup barriers and encourage reproducible robotic RL on different robot systems. To this end, we demonstrate the successful integration of SERL software suite on a robot arm operated at a different institution.

Researchers at the University of Washington set up a Peg Insertion task using 3D printed parts from the Functional Manipulation Benchmark (Luo et al., 2024) and used SERL to solve this challenging task. The overall preparation time including setting up the relevant hardware and software is less than 3 hours. The policy converged in 19 minutes and achieved a 100/100 success rate with 20 initial human demonstrations, successfully reproducing our results.

### 6. Discussion

We described a system and software package for robotic reinforcement learning that is aimed at making realworld RL more accessible both to researchers and practitioners. Our software package provides a carefully designed combination of ingredients for sample-efficient RL, automating reward design, automating environment resets with forward-backward controllers, and a controller framework that is particularly well-suited for contact-rich manipulation tasks. Furthermore, our experimental evaluation of our framework demonstrates

that it can learn a range of diverse manipulation tasks very efficiently, with under an hour of training per policy when provided with a small number of demonstrations. These results qualitatively compare well to state-of-the-art results in RL for manipulation in the literature, indicating that the particular choices in our framework are well-suited for obtaining very good realworld results even from image observations. Our framework does have a number of limitations. First, we do not aim to provide a comprehensive library with every possible RL method, and some tasks and settings might be outside of our framework (e.g., non-manipulation tasks). Second, the full range of reward specifications and reset-free learning challenges still constitute an open problem in robotic RL research. Our classifierbased rewards and forward-backward controller might not be appropriate in every setting. Further research on these topics is needed to make robotic RL more broadly applicable. However, we hope that our software package will provide a reasonable “default” starting point for both researchers and practitioners wanting to experiment with real-world RL methods.

### Acknowledgments

This research was partially supported by Intrinsic Innovation LLC, the National Science Foundation under IIS-2150826, and ARO W911NF-21-1-0097. We would like to thank Rehaan Ahmad and Siri Gadipudi for participating in the discussion of this project.

### 7. Appendix

#### 7.1. Details on Relative Observation and Action Frame

Let the robot’s base frame be {𝑠}; for the 𝑖-th episode of rolling out the policy, we denote {𝑏𝑡(𝑖)} as the endeffector frame expressed w.r.t. {𝑠} at a particular time step 𝑡; where 1≤𝑖≤𝑀, 0≤𝑡≤𝑁. For each episode, {𝑏0(𝑖)} is sampled from a uniform distribution specifying the area of randomization. We want to express such proprioceptive information with respect to {𝑏0(𝑖)}. Thus, the policy will be applicable to a new location provided that the relative spatial distance between the robot’s end-effector and the target remains consistent. This approach prevents overfitting to specific global locations within the reference frame {𝑠}. We achieve this by applying the following homogeneous transformation:

𝑇𝑏(𝑖)

0 𝑏𝑡(𝑖) =𝑇𝑏−1(𝑖)

⋅𝑇𝑏(𝑖)

𝑡

0

where we use 𝑇𝑎𝑏 to denote the homogeneous transformation matrix between frame {𝑎} and {𝑏}. We feed the position and rotation information extracted from

muscular robots. IEEE Trans. Robotics, 38(6):3850– 3860, 2022. doi: 10.1109/TRO.2022.3176207. URL https://doi.org/10.1109/TRO.2022.3176207.

𝑇𝑏(𝑖)

0 𝑏𝑡(𝑖) to the policy. Here we use 𝑇𝑎𝑏 to denote the homogeneousand {𝑏}, definedtransformationas: matrix between frame {𝑎}

Yuqing Du, Ksenia Konyushkova, Misha Denil, Akhil Raju, Jessica Landon, Felix Hill, Nando de Freitas, and Serkan Cabi. Vision-language models as success detectors. arXiv preprint arXiv:2303.07280, 2023.

𝑅𝑎𝑏 𝑝𝑎𝑏 01×3 1]

𝑇𝑎𝑏 =

.

[

Benjamin Eysenbach, Shixiang Gu, Julian Ibarz, and Sergey Levine. Leave no trace: Learning to reset for safe and autonomous reinforcement learning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id= S1vuO-bCW.

The policy generates a six degrees of freedom (6 DoF) twist action, which is expressed in the reference frame from which it currently receives observations, i.e., {𝑏𝑡(𝑖)}. Mathematically, the 6 DoF twist action 𝑡(𝑖) expressed in frame {𝑏𝑡(𝑖)} at timestep 𝑡. To interface with the robot’s control software, which expects actions 𝑡(𝑖)′ expressed in the base frame {𝑠}, we apply the Adjoint mapping:

𝑡(𝑖)′ = [Ad(𝑖)𝑡 ]𝑡(𝑖)

Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_ and_Benchmarks.html.

where [Ad(𝑖)𝑡 ] is a function of the homogeneous transformation 𝑇𝑏(𝑖)

defined as:

0

𝑅𝑏(𝑖)

03×3 [𝑝𝑏(𝑖)

[Ad(𝑖)𝑡 ] =

.

] ×𝑅𝑏(𝑖)

𝑅𝑏(𝑡)

[

]

𝑡

𝑡

𝑡

𝑡

Justin Fu, Avi Singh, Dibya Ghosh, Larry Yang, and Sergey Levine. Variational inverse control with events: A general framework for data-driven reward definition. Advances in neural information processing systems, 31, 2018.

### References

Michael Ahn, Henry Zhu, Kristian Hartikainen, Hugo Ponte, Abhishek Gupta, Sergey Levine, and Vikash Kumar. ROBEL: robotics benchmarks for learning with low-cost robots. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura, editors, 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, volume 100 of Proceedings of Machine Learning Research, pages 1300–1313. PMLR, 2019. URL http://proceedings.mlr.press/v100/ahn20a.html.

Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actorcritic methods. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1582–1591. PMLR, 2018. URL http: //proceedings.mlr.press/v80/fujimoto18a.html.

Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. arXiv preprint arXiv:2302.02948, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Sergio Guadarrama, Anoop Korattikara, Oscar Ramirez, Pablo Castro, Ethan Holly, Sam Fishman, Ke Wang, Ekaterina Gonina, Neal Wu, Efi Kokiopoulou, Luciano Sbaiz, Jamie Smith, Gábor Bartók, Jesse

Dieter Büchler, Simon Guist, Roberto Calandra, Vincent Berenz, Bernhard Schölkopf, and Jan Peters. Learning to play table tennis from scratch using

Berent, Chris Harris, Vincent Vanhoucke, and Eugene Brevdo. TF-Agents: A library for reinforcement learning in tensorflow. https://github.com/tensorflow/ agents, 2018. URL https://github.com/tensorflow/agents. [Online; accessed 25-June-2019].

Abhishek Gupta, Justin Yu, Tony Z. Zhao, Vikash Kumar, Aaron Rovinsky, Kelvin Xu, Thomas Devlin, and Sergey Levine. Reset-free reinforcement learning via multi-task learning: Learning dexterous manipulation behaviors without human intervention. In IEEE International Conference on Robotics and Automation, ICRA 2021, Xi’an, China, May 30 - June 5, 2021, pages 6664–6671. IEEE, 2021. doi: 10.1109/ICRA48506.2021.9561384. URL https://doi. org/10.1109/ICRA48506.2021.9561384.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.

Weiqiao Han, Sergey Levine, and Pieter Abbeel. Learning compound multi-step controllers under unknown dynamics. In 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6435–6442. IEEE, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.

Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J. Lim. Furniturebench: Reproducible realworld benchmark for long-horizon complex manipulation. In Kostas E. Bekris, Kris Hauser, Sylvia L. Herbert, and Jingjin Yu, editors, Robotics: Science and Systems XIX, Daegu, Republic of Korea, July 10-14, 2023, 2023. doi: 10.15607/RSS.2023.XIX.041. URL https://doi.org/10.15607/RSS.2023.XIX.041.

Todd Hester and Peter Stone. Texplore: real-time sample-efficient reinforcement learning for robots. Machine learning, 90:385–429, 2013.

Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Stable baselines. https://github.com/hill-a/stable-baselines, 2018.

Zhimin Hou, Jiajun Fei, Yuelin Deng, and Jing Xu. Data-efficient hierarchical reinforcement learning for robotic assembly control applications. IEEE Transactions on Industrial Electronics, 68(11):11565–11575, 2020.

Zheyuan Hu, Aaron Rovinsky, Jianlan Luo, Vikash Kumar, Abhishek Gupta, and Sergey Levine. REBOOT: reuse data for bootstrapping efficient real-world dexterous manipulation. arXiv preprint arXiv:2309.03322, 2024.

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics Autom. Lett., 5(2):3019–3026, 2020. doi: 10.1109/LRA. 2020.2974707. URL https://doi.org/10.1109/LRA.2020. 2974707.

Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine. Residual reinforcement learning for robot control. CoRR, abs/1812.03201, 2018. URL http://arxiv.org/abs/1812. 03201.

Dmitry Kalashnikov, Jacob Varley, Yevgen Chebotar, Benjamin Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman. Mt-opt: Continuous multi-task robotic reinforcement learning at scale. CoRR, abs/2104.08212, 2021. URL https://arxiv.org/abs/2104.08212.

Vijay R. Konda and John N. Tsitsiklis. Actor-critic algorithms. In Sara A. Solla, Todd K. Leen, and KlausRobert Müller, editors, Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999], pages 1008–1014. The MIT Press, 1999. URL http: //papers.nips.cc/paper/1786-actor-critic-algorithms.

Ilya Kostrikov, Laura M. Smith, and Sergey Levine. Demonstrating A walk in the park: Learning to walk in 20 minutes with model-free reinforcement learning. In Kostas E. Bekris, Kris Hauser, Sylvia L. Herbert, and Jingjin Yu, editors, Robotics: Science and Systems XIX, Daegu, Republic of Korea, July 10-14, 2023, 2023. doi: 10.15607/RSS.2023.XIX.056. URL https://doi.org/10.15607/RSS.2023.XIX.056.

Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor

policies. J. Mach. Learn. Res., 17:39:1–39:40, 2016a. URL http://jmlr.org/papers/v17/15-522.html.

Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334–1373, 2016b.

Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. Int. J. Robotics Res., 37 (4-5):421–436, 2018. doi: 10.1177/0278364917710318. URL https://doi.org/10.1177/0278364917710318.

Kevin Li, Abhishek Gupta, Ashwin Reddy, Vitchyr H. Pong, Aurick Zhou, Justin Yu, and Sergey Levine. MURAL: meta-learning uncertainty-aware rewards for outcome-driven reinforcement learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 6346–6356. PMLR, 2021. URL http: //proceedings.mlr.press/v139/li21g.html.

Jianlan Luo, Eugen Solowjow, Chengtao Wen, Juan Aparicio Ojea, and Alice M Agogino. Deep reinforcement learning for robotic assembly of mixed deformable and rigid objects. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2062–2069. IEEE, 2018.

Jianlan Luo, Eugen Solowjow, Chengtao Wen, Juan Aparicio Ojea, Alice M Agogino, Aviv Tamar, and Pieter Abbeel. Reinforcement learning on variable impedance controller for high-precision robotic assembly. In 2019 International Conference on Robotics and Automation (ICRA), pages 3080–3087. IEEE, 2019.

Jianlan Luo, Oleg Sushkov, Rugile Pevceviciute, Wenzhao Lian, Chang Su, Mel Vecerik, Ning Ye, Stefan Schaal, and Jonathan Scholz. Robust Multi-Modal Policies for Industrial Assembly via Reinforcement Learning and Demonstrations: A Large-Scale Study. In Proceedings of Robotics: Science and Systems, Virtual, July 2021. doi: 10.15607/RSS.2021.XVII.088.

Jianlan Luo, Perry Dong, Yuexiang Zhai, Yi Ma, and Sergey Levine. Rlif: Interactive imitation learning as reinforcement learning. arXiv preprint arXiv:2311.12996, 2023.

Jianlan Luo, Charles Xu, Fangchen Liu, Liam Tan, Zipeng Lin, Jeffrey Wu, Pieter Abbeel, and Sergey Levine. Fmb: a functional manipulation benchmark for generalizable robotic learning. arXiv preprint arXiv:2401.08553, 2024.

Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. LIV: languageimage representations and rewards for robotic control. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 23301–23320. PMLR, 2023a. URL https://proceedings.mlr.press/v202/ ma23b.html.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. VIP: towards universal visual reward and representation via value-implicit pre-training. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023b. URL https://openreview.net/pdf?id= YJ7o2wetJ2.

Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and Ken Goldberg. Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. In Nancy M. Amato, Siddhartha S. Srinivasa, Nora Ayanian, and Scott Kuindersma, editors, Robotics: Science and Systems XIII, Massachusetts Institute of Technology, Cambridge, Massachusetts, USA, July 12-16, 2017, 2017. doi: 10.15607/RSS.2017.XIII.058. URL http://www. roboticsproceedings.org/rss13/p58.html.

Parsa Mahmoudieh, Deepak Pathak, and Trevor Darrell. Zero-shot reward specification via grounded natural language. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 14743–14752. PMLR, 2022. URL https://proceedings.mlr.press/v162/ mahmoudieh22a.html.

Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin Yuan, Poo-

ria Poorsarvi Tehrani, Ritvik Singh, Yunrong Guo, Hammad Mazhar, Ajay Mandlekar, Buck Babich, Gavriel State, Marco Hutter, and Animesh Garg. Orbit: A unified simulation framework for interactive robot learning environments, 2023.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Anusha Nagabandi, Kurt Konolige, Sergey Levine, and Vikash Kumar. Deep dynamics models for learning dexterous manipulation. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura, editors, 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, volume 100 of Proceedings of Machine Learning Research, pages 1101–1112. PMLR, 2019. URL http://proceedings.mlr.press/v100/nagabandi20a.html.

Ashvin Nair and Vitchyr Pong. rlkit. Github. URL https://github.com/rail-berkeley/rlkit.

Ashvin Nair, Murtaza Dalal, Abhishek Gupta, and Sergey Levine. Accelerating online reinforcement learning with offline datasets, 2020.

Ivaylo Popov, Nicolas Heess, Timothy Lillicrap, Roland Hafner, Gabriel Barth-Maron, Matej Vecerik, Thomas Lampe, Yuval Tassa, Tom Erez, and Martin Riedmiller. Data-efficient deep reinforcement learning for dexterous manipulation. arXiv preprint arXiv:1704.03073, 2017.

Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, and Chelsea Finn. Offline reinforcement learning from images with latent space models. In Ali Jadbabaie, John Lygeros, George J. Pappas, Pablo A. Parrilo, Benjamin Recht, Claire J. Tomlin, and Melanie N. Zeilinger, editors, Proceedings of the 3rd Annual Conference on Learning for Dynamics and Control, L4DC 2021, 7-8 June 2021, Virtual Event, Switzerland, volume 144 of Proceedings of Machine Learning Research, pages 1154–1168. PMLR, 2021. URL http://proceedings.mlr.press/v144/rafailov21a.html.

Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. In Proceedings of Robotics: Science

and Systems, Pittsburgh, Pennsylvania, June 2018. doi: 10.15607/RSS.2018.XIV.049.

Martin Riedmiller, Thomas Gabel, Roland Hafner, and Sascha Lange. Reinforcement learning for robot soccer. Autonomous Robots, 27:55–73, 2009.

Gerrit Schoettler, Ashvin Nair, Jianlan Luo, Shikhar Bahl, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine. Deep reinforcement learning for industrial insertion tasks with visual inputs and natural rewards. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5548– 5555, 2020. doi: 10.1109/IROS45743.2020.9341714.

Takuma Seno and Michita Imai. d3rlpy: An offline deep reinforcement learning library. Journal of Machine Learning Research, 23(315):1–20, 2022. URL http:// jmlr.org/papers/v23/22-0017.html.

Archit Sharma, Kelvin Xu, Nikhil Sardana, Abhishek Gupta, Karol Hausman, Sergey Levine, and Chelsea Finn. Autonomous reinforcement learning: Benchmarking and formalism. arXiv preprint arXiv:2112.09605, 2021.

Archit Sharma, Ahmed M. Ahmed, Rehaan Ahmad, and Chelsea Finn. Self-improving robots: End-toend autonomous visuomotor reinforcement learning. CoRR, abs/2303.01488, 2023. doi: 10.48550/arXiv.2303. 01488. URL https://doi.org/10.48550/arXiv.2303.01488.

Oren Spector and Dotan Di Castro. Insertionnet – a scalable solution for insertion, 2021.

Jonas Tebbe, Lukas Krauch, Yapeng Gao, and Andreas Zell. Sample-efficient reinforcement learning in robotic table tennis. In 2021 IEEE international conference on robotics and automation (ICRA), pages 4171– 4178. IEEE, 2021.

Mel Vecerik, Oleg Sushkov, David Barker, Thomas Rothörl, Todd Hester, and Jon Scholz. A practical approach to insertion with variable socket position using deep reinforcement learning, 2018.

Tyler Westenbroek, Fernando Castaneda, Ayush Agrawal, Shankar Sastry, and Koushil Sreenath. Lyapunov design for robust and efficient robotic reinforcement learning. arXiv preprint arXiv:2208.06721, 2022.

Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer:

World models for physical robot learning. In Karen Liu, Dana Kulic, and Jeffrey Ichnowski, editors, Conference on Robot Learning, CoRL 2022, 14-18 December 2022, Auckland, New Zealand, volume 205 of Proceedings of Machine Learning Research, pages 2226–2240. PMLR, 2022. URL https://proceedings.mlr.press/v205/ wu23c.html.

Annie Xie, Fahim Tajwar, Archit Sharma, and Chelsea Finn. When to ask for help: Proactive interventions in autonomous reinforcement learning. In NeurIPS, 2022. URL

http://papers.nips.cc/paper_files/paper/2022/hash/ 6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference. html.

Yuxiang Yang, Ken Caluwaerts, Atil Iscen, Tingnan Zhang, Jie Tan, and Vikas Sindhwani. Data efficient reinforcement learning for legged robots. In Conference on Robot Learning, pages 1–10. PMLR, 2020.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multitask and meta reinforcement learning. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura, editors, 3rd Annual Conference on Robot Learning, CoRL 2019, Osaka, Japan, October 30 - November 1, 2019, Proceedings, volume 100 of Proceedings of Machine Learning Research, pages 1094–1100. PMLR, 2019. URL http://proceedings.mlr.press/v100/yu20a.html.

Albert Zhan, Ruihan Zhao, Lerrel Pinto, Pieter Abbeel, and Michael Laskin. A framework for efficient robotic manipulation. In Deep RL Workshop NeurIPS 2021, 2021.

Tony Z. Zhao, Jianlan Luo, Oleg Sushkov, Rugile Pevceviciute, Nicolas Heess, Jon Scholz, Stefan Schaal, and Sergey Levine. Offline meta-reinforcement learning for industrial insertion. In 2022 International Conference on Robotics and Automation (ICRA), pages 6386– 6393, 2022. doi: 10.1109/ICRA46639.2022.9812312.

Henry Zhu, Abhishek Gupta, Aravind Rajeswaran, Sergey Levine, and Vikash Kumar. Dexterous manipulation with deep reinforcement learning: Efficient, general, and low-cost. In International Conference on Robotics and Automation, ICRA 2019, Montreal, QC, Canada, May 20-24, 2019, pages 3651–3657. IEEE, 2019. doi: 10.1109/ICRA.2019.8794102. URL https://doi.org/10.1109/ICRA.2019.8794102.

Henry Zhu, Justin Yu, Abhishek Gupta, Dhruv Shah, Kristian Hartikainen, Avi Singh, Vikash Kumar, and Sergey Levine. The ingredients of real world robotic reinforcement learning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=rJe2syrtvS.

