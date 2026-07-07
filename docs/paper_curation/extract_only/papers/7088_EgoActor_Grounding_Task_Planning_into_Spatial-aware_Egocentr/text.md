[Figure 1]

# EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models

## arXiv:2602.04515v1[cs.RO]4Feb2026

Yu Bai, MingMing Yu, Chaojie Li, Ziyi Bai, Xinlong Wang†, B¨orje F. Karlsson†

Beijing Academy of Artificial Intelligence

[Figure 2]

Fig. 1: Overview of EgoActor, which can control a humanoid robot by jointly predicting movement, active perception, manipulation, and human interaction actions to achieve coordinated and precise execution, enabling humanoid robots to conduct long-horizon multi-step task instructions described in natural language.

Abstract—Deploying humanoid robots in real-world settings is fundamentally challenging, as it demands tight integration of perception, locomotion, and manipulation under partial-information observations and dynamically changing environments. As well as transitioning robustly between sub-tasks of different types. Towards addressing these challenges, we propose a novel task EgoActing, which requires directly grounding high-level instructions into various, precise, spatially aware humanoid actions. We further instantiate this task by introducing EgoActor, a unified and scalable vision-language model (VLM) that can predict locomotion primitives (e.g., walk, turn, move sideways, change height), head movements, manipulation commands, and humanrobot interactions to coordinate perception and execution in real-time. We leverage broad supervision over egocentric RGBonly data from real-world demonstrations, spatial reasoning question–answering, and simulated environment demonstrations, enabling EgoActor to make robust, context-aware decisions and perform fluent action inference (under 1s) with both 8B and 4B parameter models. Extensive evaluations in both simulated and real-world environments demonstrate that EgoActor effectively bridges abstract task planning and concrete motor execution, while generalizing across diverse tasks and unseen environments1.

I. INTRODUCTION

Deploying humanoid robots in real-world environments and tasks is fundamentally challenging [34, 62, 58]. These challenges mainly arise from the inherent instability of humanoid

1We open-source our code, models, datasets, and benchmarks to facilitate future research: https://baai-agents.github.io/EgoActor/.

† Corresponding authors. {borje, xlwang 1}@baai.ac.cn

platforms and the complexity of partial-information real-world tasks. Humanoid robots are inherently unstable compared to wheeled platforms and are highly sensitive to issues in timing, precision, and obstacle handling. Even minor control inaccuracies can disrupt balance or lead to unsafe behaviors, especially in dynamic, cluttered, and previously unseen environments. While recent advances in low-level control—such as wholebody locomotion for balance [9, 30, 32], and dexterous handbased manipulation [21, 23, 19]—have substantially improved motor execution, approaches for these capabilities usually focus only on that task type only, and still require precise coordination based on spatial and visual understanding, thus remaining fragile under real-world uncertainty.

Beyond basic locomotion and manipulation, embodied robot systems also require fluid transitions between different actions and action types, often executing them in combination. Realworld tasks rarely involve actions in isolation [58]; instead, they require the coordinated use of movement, head orientation, hand manipulation, human-robot interactions (e.g., gestures and utterances, etc.), and full-body control in contextually appropriate sequences. For example, a robot may need to halt its walking, tilt its head to perceive a target, extend its arm to grasp an object, and then resume locomotion—all as part of a single coherent task [50]. Achieving such coordination demands not only reliable motor control, but also advanced spatial understanding and reasoning, enabling a model to infer how actions interact with the environment and to determine

when and how to effectively coordinate multiple skills.

To embody the above challenges, we introduce EgoActing, representing scenarios in which a humanoid robot must transform an actionable instruction into appropriate situated action sequences, based on egocentric observations, action history, and its available skills. Moreover, to instantiate this task, while allowing naturally scalable data collection and future expansion, we develop EgoActor, a vision–language model (VLM) [3] that grounds high-level instructions into low-level, executable humanoid actions. We leverage the reasoning capabilities of VLMs, while enhancing their spatial understanding to directly predict a wide range of low-level actions formulated in language, including movement, active perception, manipulation, and human-interaction (as shown in Fig. 1). For movement, EgoActor outputs precise locomotion primitives such as moving forward or strafing by specified distances, turning by specified angles, and performing postural adjustments like standing or crouching to support manipulation at different heights. For active perception, it predicts head orientation actions to facilitate exploration, target localization, and dynamic obstacle handling. For manipulation, it determines when and where to initiate hand or arm actions for coordinated object interaction. Additionally, it can produce human interaction actions that enable information seeking, communication, and collaboration with humans or other robots through gestures or spoken language.

To equip the model with these capabilities, we train EgoActor on a diverse mixture of real-world video demonstrations, spatial reasoning trajectories, action-timing annotations, and virtual environment examples; minimizing human annotation requirements. This broad supervision enables fully functional 8B and 4B models with sub-second inference latency, supporting real-time interaction and control on humanoid platforms. We evaluate the proposed framework across a wide range of settings, including human–robot interaction2, mobile manipulation, and traversability—defined as the robot’s ability to safely move through narrow spaces commonly encountered in daily environments without colliding with surrounding obstacles—in real-world and simulated environments. These experiments demonstrate task- and environment-level generalization, highlighting EgoActor’s ability to operate under diverse and unseen conditions. In addition, qualitative case studies showcase behaviors such as active perception and human-like movement patterns.

We summarize our contributions as follows:

- • We introduce EgoActing, a new task formulation that requires models with strong spatial understanding to directly transform language instructions into executable action sequences from egocentric observations, emphasizing the challenges of real-world humanoid deployment.
- • We propose EgoActor, a vision-centric model that fully leverages humanoid capabilities by unifying movement, perception, manipulation, and human interaction.

2When referring to the task setting, we use the term human–robot interaction. When referring to a candidate skill or action from the robot’s perspective, we term it human-interaction.

• We validate EgoActor through extensive real-world and simulated experiments, and release deployable opensource code and models, along with datasets and evaluation protocols to facilitate reproducibility and future research.

II. RELATED WORK

- A. VLM-based Embodied Agents

Vision-and-Language-Model (VLM) based embodied agents aim to ground natural language instructions into executable actions [48, 52, 42]. A representative example is SayCan [1], which decomposes language instructions into executable skills using learned affordances; however, such approaches typically rely on predefined skill libraries and are therefore less suitable for humanoid robots with complex and diverse embodiments. Recent surveys [45, 13] review progress in multimodal embodied agents and identify key challenges in grounding, long-horizon reasoning, and real-world deployment. Building on this line of research, several works explore LLM-based embodied agents in simulated or structured environments, including cooperative multi-agent systems [16], standardized benchmarking interfaces [29], offline reinforcement learning with LLM-generated rewards [26], language-supervised policy learning [54], and LLM-driven environment generation [59]. Other studies target specific capabilities, such as zero-shot object navigation [11] and visual perception in open-world [64].

In contrast, EgoActor focuses on humanoid robots and directly predicts egocentric, low-level executable actions such as locomotion and head movement, bridging textual task description and low-level motor control.

- B. Mobile-Manipulation

Mobile manipulation has been extensively explored across simulated and real-world frameworks. Early systems such as SayCan [1] combine language models with affordance-based planners to ground high-level instructions into sequential robot skills. Simulation platforms [12, 10, 44, 55, 28] provide embodied environments for evaluating navigation and manipulation jointly. Recent efforts [53, 33, 14] further expand largescale benchmarks, emphasizing diverse object interactions, long-horizon tasks, and more realistic robot embodiments.

In contrast to previous work, EgoActing and EgoActor unify decision-making at an egocentric level, jointly reasoning over locomotion, posture, spatial perception, manipulation, and human interaction within a single VLM-based action predictor. This integrated approach enables smooth movement–manipulation transitions and robust adaptation to unseen layouts and diverse natural-language instructions.

- C. Visual-Language Navigation

a) Vision-and-Language Navigation: Vision-andLanguage Navigation (VLN) has been extensively studied as a core embodied AI problem [15, 63]. Classical VLN methods predominantly focus on mapping language instructions to navigational trajectories without requiring physical interaction with the environment. Representative works include R2R [2],

Touchdown [6], and sub-instruction-aware navigation [17], all of which aim to follow language-guided routes in static or semi-structured environments. More recent approaches extend VLN with stronger visual–language models and improved spatial grounding. VLN-R1 [36], NaVid [60], Uni-NaVid [61], and Navila [8] leverage large multimodal encoders and unified navigation architectures to improve generalization, robustness, and long-horizon reasoning.

b) Object-goal Navigation: Object-goal navigation requires an agent to navigate to a specified object category using only its onboard perception [43, 57]. Chaplot et al. [5] propose a modular semantic-mapping system that leverages object-arrangement priors for efficient navigation in unseen environments. Qi et al. [35] introduce OAAM, which separately encodes object and action descriptions to improve languagevision alignment in VLN. Cao et al. [4] use cognitive-state modeling with a dynamic map and an LLM to guide navigation via map state reasoning, enhancing success in both simulation and real-world settings.

In contrast, EgoActor extends far beyond navigation: it coordinates whole-body humanoid behaviors by jointly reasoning over locomotion, posture control, active perception, manipulation, and human-interaction behaviors. This enables the model to ground high-level instructions into actionable, egocentric motor sequences suitable for real-world humanoid control in dynamic environments.

III. FRAMEWORK DESIGN A. Task Definition

In this work, we introduce EgoActing, a task that receives a direct and actionable instruction and predicts the next concrete actions a humanoid robot should perform. We assume the robot is equipped with a set of whole-body control and manipulation policies. The task could be represented as follows:

P(a | I, O1:t, a1:t−1, Π) (1)

at = arg max a∈A

where I is the (natural-language) task instruction, O1:t denotes the history of egocentric observations (i.e., RGB images in our setup), a1:t−1 is the past action history, and Π denotes the set of available low-level whole-body and manipulation policies. An example of the proposed EgoActor conducting an EgoActing-type task is shown in Fig. 2.

Instruction. We define an instruction as a high-level yet explicit specification of the intended task, describing the required movements and goals without prescribing low-level motor details. The instruction captures general motion primitives—such as passing through a corridor, turning at specific locations, or stopping to manipulate objects and interact at designated positions—thereby providing clear spatial and temporal guidance. When a target is involved, the target object is specified in an unambiguous and identifiable manner. This design discourages the model from relying on guesswork or producing hallucinated predictions, and instead promotes grounded reasoning and reliable action execution.

Especially, we include the following basic capability categories as source of candidate actions in the current task (specific examples shown in Appendix XV):

- a) Active perception: We introduce active perception

skills that allow the humanoid robot to better explore its surroundings, localize target objects, and dynamically respond to newly appearing obstacles along its path [39, 51].

- b) Manipulation: We include manipulation as an es-

sential action type as it plays a crucial role in humanoid robotics by allowing coordinated control of the hands and arms to perform precise interactions with objects and tools in unstructured environments [31].

- c) Human-interaction: We also include human-

interaction actions, allowing the robot to seek new information, engage in communication, and cooperate with humans or other humanoid robots by requiring items from others.

- d) Movement: For movement actions, we design a set of

basic locomotion skills (summarized in Appendix X). Beyond conventional navigation tasks, EgoActing also incorporates lateral movement actions. This capability enhances obstacle avoidance and enables more precise alignment with target objects. Meanwhile, to support manipulation tasks across varying heights and spatial constraints, we include postural adjustment actions such as standing up and crouching down. These skills help the robot adapt to tasks that require interacting with objects at different elevations or avoiding overhead obstacles.

B. Language-based Actions

We design the framework to represent robot behaviors as textual actions, combining structured language actions for precise movement and perception with natural language actions for manipulation and human interaction.

a) Structured language actions (SLAs): For movement and active perception, we adopt a set of structured language actions that describe spatial motion in a well structured and interpretable format. Each action is expressed in a concise natural-language-like template specifying the action type, direction, and magnitude, such as: Turn left 30.5 degrees or Look up 10.2 degrees, details shown in Appendix X.

These structured language actions encompass horizontal and vertical rotations (yaw and pitch), linear translations along the forward–backward and lateral axes, and vertical adjustments along the z-axis. Thresholds are applied to filter out negligible movements and reduce noise. The purpose of these actions is to enable the model to interpret spatial relationships from RGB observations and to position the robot appropriately for executing subsequent task-specific (natural language) actions.

b) Natural language actions (NLAs): For manipulation and human-interaction actions, we do not restrict the system to a fixed set of skills. Instead, we employ natural language to represent these actions, with examples shown in Fig. 3. This design provides the following key advantages:

• Generalization beyond predefined skill or code primitives, enabling the interpretation of novel instructions and the production of previously unseen actions.

[Figure 3]

- Fig. 2: Visualization of EgoActor’s working procedure for a given task: “Approach and pick up the orange on the desk”. The grey blocks represent structured language actions (SLAs) and the green blocks represent natural language actions (NLAs).

[Figure 4]

- Fig. 3: Example natural language actions (NLA) in EgoActing. EgoActor is trained to predict the corresponding actions based on obtained RGB observations.

when it appears at the end of the action sequence; otherwise, and for the interleaved-style data, we explicitly append a “Stop and no action” at the end of the sequence.

d) Action Parsing: During real-world experiments, we use a simple parser to extract parameters from SLAs, converting them into velocity/angle commands for the robot to conduct. For NLAs, execution is routed by keyword triggers. Actions with speech-related keywords (e.g., “Speak”, “Ask”) are converted to audio via text-to-speech models, predefined interaction keywords (e.g., “Say Hi”, “Shake Hands”) invoke preset motions, and all remaining actions are treated as manipulation commands, forwarded to pre-trained VLA models.

IV. TRAINING RECIPE

- • Effective reuse of low-level vision–language–action models for manipulation by precise pre-positioning and providing context-aware language commands for complex, open-ended interactions.
- • Transformation of task intentions into natural language actions. For example, for the task “Search and approach the woman and ask her to show you the way to a meeting room”, the model can output “Ask Could you please guide me to a meeting room?”. More details are provided in Appendix XIV-C.

We also include “Stop and no action” as an NLA, to mean the task is done and the robot should wait for a new instruction.

c) Discussion: We note that the primary role of structured language actions is to navigate and position the robot to enable subsequent natural language actions, which requires the model to possess strong spatial understanding ability. In most cases, each data sample consists of one sequence of structured language actions and one subsequent natural language action. We also include interleaved-style data from the EgoTaskQA dataset [22], which features multiple alternating sequences of structured and natural language actions. In the former case, we treat the natural language action as an implicit stop signal

In this section, we describe the model design for addressing the EgoActing-type tasks defined in this work. We deliberately adopt a general vision–language model (VLM) architecture to demonstrate that (1) our approach does not rely on specialized architectural modifications, and (2) the model can naturally benefit from additional easily-acquirable video training data, enabling straightforward scaling in future work.

- A. Model Structure and Training Setup

We use Qwen3-VL [47], a vision–language model built on a transformer-based architecture [49] with dynamic resolution support, as our base model. Following Schulman and Thinking Machines Lab [38], we apply LoRA [18] to finetune all linear layers of the model with a learning rate of 3e-4. The model is trained with one epoch of randomly mixed data from all sources (see Section IV-C) on 16 A100 40GB GPUs. We train both 4B and 8B variants of the model to accommodate different use cases, as we observe a trade-off between inference speed and performance across model sizes.

- B. Data Format

Generally, we format our data as illustrated in Fig. 1, with the detailed template provided in Appendix VIII. For each task, we uniformly sample 10 historical observations from

all previous images and use the most recent 3 observationaction pairs as key anchors for the model to predict the next action. To save computational resources while increasing the model’s adaptability to different hardware conditions, the recent observation images are processed at 480p resolution, while all sampled historical observation images are processed at 240p resolution. We note that this setup can be extended if practitioners have access to more computational resources and higher-quality cameras.

We manually annotate demonstrations collected from both virtual and real-world environments (see Section IV-C). For each example, we first annotate a concise textual description of the pre-shot video trajectory, which can be captured using standard RGB cameras (e.g., mobile phones). As mentioned in Section III-A, these descriptions are intentionally kept minimal yet precise, ensuring that the target object and the movement route are clearly specified and unambiguous to discourage the model from relying on guessing or producing hallucinated predictions. In addition, we append a final natural language action to each trajectory, enabling the model to explicitly associate movement sequences with subsequent manipulation or human-interaction actions. Owing to this lightweight and easily scalable annotation pipeline, our approach naturally supports large-scale data collection and is well-suited to further performance improvements through increased training data. Note that we apply additional pre-processing to the EgoTaskQA dataset, as each video trace contains multiple natural language actions (see Appendix IX-A).

Our approach suggests that using multiple observationaction pairs offers several advantages that previous work lacks. First, it enables more efficient training by allowing the model to learn to predict multiple actions within a single training sample. Second, it provides a richer context for decisionmaking—for example, the model can learn to have the robot turn left after previously turning right to avoid obstacles.

C. Data Acquisition

We utilize a diverse collection of multimodal datasets to train our model. For all movement data, we follow Cheng et al. [8] and extract step-by-step movement actions by estimating camera poses using MASt3R [27], identifying actions at 1.5second intervals. For all EgoActing sub-datasets, we augment them by oversampling samples with turning actions and natural language actions to balance the distribution across different action types. We summarize all the data sources as follows:

- a) Internet video data: We adopt the EgoTaskQA

dataset [22] as our primary source of internet-scale egocentric videos, and supplement it with 130 additional internetcollected egocentric videos. After processing, we produce 160,000 EgoActing training samples from EgoTaskQA and 7,111 additional samples from the additional collected videos. Additional details on dataset preprocessing and sample construction are provided in Appendix IX-A.

- b) Local environment data (EgoActing): We recorded

398 egocentric videos in local environments, yielding a total of 150,214 EgoActing training samples. These recordings capture

environmental variability due to frequent layout changes in the data collection areas.

c) Virtual environment data (Navigation): To incorporate controlled spatial navigation supervision, we sampled approximately 3% of the VLN-CE (Room-to-Room) training set [25], resulting in 60,000 training samples. This subset provides diverse indoor layouts and structured navigation instructions. Detailed processing is provided in Appendix IX-B.

- d) Virtual environment data (EgoActing): We manually

collected and annotated 714 EgoActing-style trajectories from the Habitat-Sim simulator [44] using scenes from the Roomto-Room dataset [2]. Following the VLN-CE scene-split protocol, we partition the data into 509 training trajectories and 205 validation trajectories from unseen environments. This training set resulted in 76,821 EgoActing samples.

- e) Spatial reasoning data (MindCube): To strengthen the

model’s spatial reasoning capabilities, we incorporate samples from the MindCube dataset [56]. We randomly sample 50% of its training set, leading to 44,160 spatial reasoning samples.

- f) Visual-language understanding data: To maintain ro-

bust visual-language understanding, we sample 300,000 instances from the GQA dataset [20]. We further augmented the dataset with 35,652 of GPT-4o–annotated description samples collected from our local environment.

- g) Visual-language planning data: We also include high-

level planning data from RoboVQA [40], EgoPlan [7], and ALFRED [41], which provide explicit step-by-step task decomposition and environment-aware planning supervision. The processed subset contains 241,603 data samples.

- h) Unsupervised movement prediction data: To enhance

spatial understanding and low-level motion grounding, we construct a small dataset of 10,575 samples in which we predict the movement transition between pairs of egocentric images. This unsupervised supervision allows the model learn spatial information without requiring manual annotation.

- i) DAgger experience data: Finally, we incorporate on-

policy trajectories collected through real-world executions with the DAgger algorithm [37]. We collected 70 successful traces with 3,629 EgoActing training samples that span navigation in local environments, object-approach behaviors, and simple human-interaction tasks.

D. Skill Setup

For downstream skills, we first finetune a GROOT-N 1.5 model [34] to perform manipulation tasks, with details shown in Appendix XI. For locomotion, we adopt the official Unitree walking policy3 as the movement controller. We manually calibrate the robot’s motion to achieve a positional precision of approximately 5 cm for forward/backward and lateral movements, and a turning precision of about 5 degrees. For speaking and querying behaviors, we currently assess task success by directly inspecting the model’s predicted natural-language outputs. The stand-up and crouch-down skills are implemented only in simulation, as the current Unitree locomotion policy does not support these actions in real-world deployment.

3https://github.com/unitreerobotics/unitree sdk2

Actions for moving forward and turning left/right are merged, i.e., they are not treated as discrete steps, to enhance motion speed and perform more human-like movement. To enable a faster movement of the robot, we amplify the forward distance predicted by the model by a factor of 1.2.

V. EXPERIMENTS

- A. Experimental Setup

- a) Inference: For EgoActor inference, we use stochastic

sampling with a temperature of 0.2. For all baseline models, we follow their original settings and apply greedy decoding. The instruction prompts we used for all different tasks are shown in Appendix XVI.

- b) Robot Setup: We deploy our model and all baseline

methods on the same Unitree G14 humanoid robot for realworld experiments. The robot is equipped with a pair of Unitree Dex3-1 hands5 and a custom 2-DoF head to support active perception6. A RealSense D455 camera7, as a typical camera model used in embodied projects, is mounted on the custom head for RGB-only capture. We acquire 480p monocular RGB images from the camera and no depth data is leveraged in our model or experiments.

- B. Baselines

To investigate the hypothesis of leveraging existing navigation models for the movement-focused component of our proposed EgoActing, we evaluate the navigation success rates of several representative navigation models based on vision–language foundation models on some of our benchmark tasks (including the movement part of Human Interaction, Traversability, and our virtual environment EgoActing benchmark, all of which are introduced in Sections V-C and V-D). The baseline models considered are as follows:

- 1) NaVid [60] is a video-based large vision–language model for vision-and-language navigation that operates solely on monocular RGB video streams, achieving state-of-the-art map-free navigation and strong Sim2Real generalization.
- 2) Uni-NaVid [61] is a video-based model that unifies multiple embodied navigation tasks within a single framework, enabling general-purpose, long-horizon navigation in unseen real-world environments.
- 3) NaVILA [8] is a two-level vision–language–action framework for legged robot navigation that converts language instructions into spatially grounded mid-level actions executed by a locomotion policy. We use its VLM component as a baseline.

- C. Real-world Benchmarking

a) Human-robot Interaction: In the human-robot interaction benchmark, the robot is required to navigate toward a specified person and perform the corresponding interaction,

- 4https://www.unitree.com/g1
- 5https://www.unitree.com/Dex3-1
- 6https://github.com/BAAI-Agents/PAK/
- 7https://www.realsenseai.com/products/real-sense-depth-camera-d455f/

- TABLE I: Single person human-robot interaction results comparing different models across three tasks.

Model

Single Person Tasks Approach Say hi Ask for location Request items

NaVILA-7B 2/12 - - NaVid-7B 8/12 - - UniNaVid-7B 8/12 - - -

EgoActor-4B 12/12 12/12 12/12 11/12 EgoActor-8B 12/12 12/12 12/12 12/12

- TABLE II: Multi-person human-robot interaction results for the “Say Hi” task using different model sizes.

Multi-person Attributes (Out-of-distribution) Clothing Accessories Posture Direction Gender

Model

EgoActor-4B 8/12 7/12 8/12 11/12 10/12 EgoActor-8B 11/12 10/12 10/12 12/12 11/12

such as greeting, requesting information (e.g., locations, etc.), or asking for help with an item.

Setup. All experiments are conducted in real-world environments and with people whose appearances and clothing are entirely different from those seen in the training data. For navigation-only baseline models, we evaluate whether the robot can stop at an acceptable location in front of a visible person (within approximately one meter and facing the person). For EgoActor, successful execution additionally requires generating an appropriate interaction action, such as requesting information. To further assess person-disambiguation capabilities, we design unseen scenarios involving multiple individuals that differ in attributes such as clothing color, accessories, posture, facing direction, and gender. In each trial, the spatial arrangement of the individuals is randomized.

Results. As shown in Table I and II, both the 4B and 8B variants of EgoActor are generally able to guide the robot to approach a person and perform basic interactions. However, the 4B model shows weaker performance in scenarios with multiple people, particularly when fine-grained attribute-based identification is required. In contrast, the 8B model is able to identify the target person and carry out the instructed interaction in most tested cases. We emphasize that human interaction in our setting requires not only accurate navigation, but also the ability to translate task intent into appropriate body postures or dialogue, which we further examine through qualitative case studies in Appendix XIV-C.

b) Mobile Manipulation: In the mobile manipulation benchmark, the robot is required to navigate to approach a target object and execute the corresponding manipulation action, such as picking or placing the object. An example of EgoActor controlling the robot to conduct a mobile manipulation task is shown in Appendix XIV-A.

Setup. We evaluate the model in an unseen layout of the experimental environment, where desk, layout, and surrounding objects are arranged differently from those used during training. Target objects are placed at three distinct positions

- TABLE III: Unseen layout environment results on the EgoActing Mobile Manipulation benchmark. Best results are bold.

Seen Objects

Models

Approach and Pick Approach and Place

Apple Bottle Apple Bottle

EgoActor-4B 5/6 5/6 3/6 4/6 EgoActor-8B 5/6 6/6 6/6 6/6

Unseen Objects

Models

Approach and Pick Approach and Place Pen Holder Pink Cup Pen Holder Pink Cup

EgoActor-4B 3/6 2/6 4/6 4/6 EgoActor-8B 5/6 6/6 4/6 5/6

- TABLE IV: Results on the proposed EgoActing Traversability benchmark. Best results are bold.

Seen Environments Unseen Environments

Models

Enter rooms Leave rooms Enter rooms Leave rooms Left Right Left Right Left Right Left Right

NaVILA-7B 5/12 4/12 3/12 3/12 2/8 1/8 3/8 1/8 NaVid-7B 3/12 5/12 9/12 8/12 1/8 0/8 8/8 4/8 UniNaVid-7B 4/12 3/12 6/12 4/12 2/8 0/8 5/8 5/8 EgoActor-4B 11/12 11/12 12/12 10/12 7/8 7/8 7/8 7/8 EgoActor-8B 11/12 12/12 10/12 10/12 7/8 7/8 8/8 7/8

on the desk (left, center, and right), and each position is tested twice to account for stochasticity. For the EgoActor model, the evaluation includes both in-distribution and outof-distribution object categories, whereas all object categories are in-distribution for the manipulation model. Pick and place tasks are evaluated separately. Variations in object height are currently not considered in the real-world setting and are only analyzed in the virtual environment through additional case studies in Section V-E. In certain few cases, manipulation may fail due to lower-level execution policy issues, even when the target object is fully within the robot’s reachability; such cases are still counted as successful if manipulation was triggered at the correct moment and position.

Results. We observe that the 8B EgoActor model is generally able to navigate to the correct objects and successfully support manipulation for both in-distribution and outof-distribution objects under the unseen layout, indicating reasonable robustness to object and scene variations. For the 4B model, failures mainly occur when it predicts a manipulation action while still too far away from the target.

c) Traversability: Traversability evaluates whether the robot can safely navigate through narrow spaces commonly encountered in daily environments without colliding with surrounding obstacles. As mentioned by Team [46], most of the current VLM-based navigation models would suffer from hitting obstacles in the real-world environment. We evaluate these baseline models together with our EgoActor.

Setup. We focus on room entry and exit scenarios, as doorways are typically narrow and have been observed in preliminary tests to be particularly challenging and collisionprone for humanoid robots. The evaluation includes five realworld rooms, including three seen environments and two unseen ones. The evaluated rooms consist of three meeting rooms (seen during training), a private office, and a storage room,

[Figure 5]

- Fig. 4: Multi-step illustration of obstacle avoidance generalization of our model, when faced with an unseen string obstacle.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]<br><br>[Figure 11]<br><br>Find two persons in front, look up to distinguish the correct person.|Avoid the wrong person on the path and head up to the correct person to say hi.|
|---|---|
| |[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|

Avoiding obstacles to get closer to the persons

|Task Instruction: Get out of this area and say hi to the person with a brown shirt|
|---|

[Figure 15]

- Fig. 5: First-person view of an EgoActor’s active perception trace. Color description blocks highlight model’s behaviors.

which differ in layout, visual style, and object arrangement. All room layouts are provided in Appendix XIII. For each room, we assess both entry and exit behaviors. To assess robustness to initial conditions, the robot is placed at 2 different starting positions for each trial—on the left and right sides of the doorway—and each position is tested 4 times.

Results. Quantitative results are reported in Table IV. Our results show that EgoActor is generally able to traverse narrow passages and avoid collisions more reliably than baseline VLM-based navigation models. In contrast, existing VLN models often collide with door frames or nearby obstacles. We also observe that some baselines, such as NaVid, are generally effective at exiting rooms, but occasionally perform unnecessary rotations before door traversal, even when a straight path is available. These observations suggest that EgoActor demonstrates improved robustness in narrow-space movements. In addition, we conduct qualitative obstacle-avoidance experiments under unseen layouts and with unseen obstacles. These case studies further demonstrate the robustness of EgoActor in navigating narrow spaces and are discussed in detail in Section V-E and Appendix XIV-B.

D. Virtual Environment Benchmarking

Setup. As described in Section IV-C, we evaluate our EgoActor models using 205 labeled EgoActing samples col-

TABLE V: Virtual environment results on our virtual benchmark. Best results are bold. “m” represents meters in the table.

Distance to the Goal Position Natural Language Action F1

Final View

Models

Similarity < 0.5 m < 0.8 m < 1.0 m < 1.2 m < 1.5 m < 2.0 m < 2.5 m < 3.0 m

NaVILA-7B 8.3% 21.0% 26.3% 28.8% 33.7% 41.5% 46.3% 52.2% - 0.35 NaVid-7B 8.8% 15.1% 20.5% 23.9% 31.7% 42.0% 52.2% 60.0% - 0.37 UniNaVid-7B 6.3% 15.6% 20.5% 23.9% 28.3% 35.1% 43.9% 51.7% - 0.36 EgoActor-4B 50.7% 63.7% 70.6% 74.1% 78.9% 84.4% 86.5% 87.8% 0.60 0.41 EgoActor-8B 51.4% 66.5% 69.9% 74.1% 78.5% 84.1% 87.8% 89.9% 0.62 0.41

|Task Instruction: Approach the small grey table on the right side of the bed and wipe it.|
|---|

|Task Instruction: Enter the meeting room on your right|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Avoid obstacles and get closer to the target of the small grey table

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Fig. 6: First-person view of an EgoActor’s traversability trace, showing the robot walking through a doorway.

###### Adjust the view to focus on the target table

Adjust the position and height to better conduct the task of wiping

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

lected in unseen virtual environments. Following the scene split protocol of the VLNCE dataset [24], all evaluation samples are drawn from environments unseen during training. Because LLM-based inference is stochastic, we evaluate each EgoActor model three times on the test set and report the average performance as the final result, with per-run results provided in Appendix XII. The evaluation results are summarized in Table V. For the evaluation metrics, we consider the precision of the ending position and the view similarity with the reference ending image. Additionally, we calculate the difference between the predicted natural language action and the reference natural language action by examining the F1 score of the 1-gram overlapping of them.

Fig. 7: First-person view of an EgoActor’s height change ability trace in virtual environments. Color description blocks highlight model’s behaviors.

instances, minor side collisions occur when the robot focuses on avoiding a large obstacle ahead and temporarily loses sight of small obstacles that have previously passed its view. Moreover, as illustrated in Fig. 4, the model can generalize and handle unseen obstructions, like the rope/stripe, and find a way to walk around them.

- b) Active Perception: We observe that the robot looks

downward to verify obstacle positions while passing them, improving obstacle-avoidance success. Toward the end of a trajectory, the model also actively keeps its gaze fixed on the target object to enable a smoother transition to manipulation or interaction actions. The case shown in Fig. 5 illustrates that the model actively moves backwards and looks upward to identify the color of upper-body clothing when only the lower bodies of two people are initially visible, as the instruction requires greeting the person wearing a specific shirt.

- c) Spatial Understanding: Unlike VLMs trained primar-

ily in simulation, our model learns from human videos and thus develops stronger spatial understanding. For example, it predicts different forward-movement distances when encountering a clear, wide path versus a partially obstructed path, or negotiating a doorway corner. Its predicted turning angles also adapt appropriately across different spatial configurations. An example of this in a mobile manipulation scenario is shown in Appendix XIV-A.

- d) Human-like Behaviors: Trained on real-world videos,

Results. As shown in Table V, EgoActor generalizes well to unseen environments and target objects, with the 8B model performing slightly better under smaller distance thresholds, while the 4B and 8B models achieve overall comparable performance. We check the failure cases and find that most errors arise from ambiguous labelled instructions or visually degraded or blurry virtual environments, with additional failures occurring in unfamiliar scene types such as churches or historical sites that differ substantially from the training data.

For baseline models, we observe that under the standard VLN success criterion (<3.0 m), performance remains comparable to their reported VLNCE Room-to-Room results (around 50%). However, under stricter criteria requiring precise positioning for interaction, these models frequently fail to stop at appropriate locations, often hallucinating continued navigation instead of stopping to execute the intended interaction.

E. Case Studies

In this section, we discuss representative case studies that demonstrate EgoActor’s capabilities in obstacle avoidance, active perception, spatial understanding, and human-like movement. A detailed example of human interaction is provided in Appendix XIV-C.

the model naturally exhibits human-like movement behaviors. It may move backward when too close to obstacles or after completing a manipulation to reorient toward the next target. During corner turning, the model often combines forward action, turning, and strafing (e.g., turning left while moving

a) Obstacle Avoidance: Through all real-world experiments, the G1 robot rarely collides with obstacles. In a couple

forward and strafing right), mirroring how humans execute smooth turns while maintaining visual alignment with the path (see Fig. 5 and Fig. 6). Height adjustments are also observed in the virtual environment experiments (see Fig. 7).

VI. CONCLUSION

We propose EgoActor, a unified vision–language model that addresses overlooked challenges in grounding high-level task intentions into egocentric, executable multi-step actions for humanoid robots in real-world settings, without requiring extra sensing modalities, multiple cameras, nor extensive teleoperation. By jointly predicting locomotion, manipulation, human interaction, and head movements, EgoActor tightly integrates perception and execution in dynamic environments. Trained on easily scalable diverse real-world, spatial reasoning, and simulated data, the model demonstrates strong generalization and timely inference. Extensive evaluations in both simulation and physical robots show EgoActor effectively bridges abstract task planning and low-level action execution, offering a practical step toward scalable humanoid autonomy. EgoActor is to be released as an open testbed to support future research (including code, models, dataset, and benchmark). Current limitations are further discussed in Appendix VII.

REFERENCES

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.
- [2] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674–3683, 2018.
- [3] Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Ma˜nas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al-Tahan, Karthik Padthe, Vasu Sharma, Hu Xu, Xiaoqing Ellen Tan, Megan Richards, Samuel Lavoie, Pietro Astolfi, Reyhane Askari Hemmat, Jun Chen, Kushal Tirumala, Rim Assouel, Mazda Moayeri, Arjang Talattof, Kamalika Chaudhuri, Zechun Liu, Xilun Chen, Quentin Garrido, Karen Ullrich, Aishwarya Agrawal, Kate Saenko, Asli Celikyilmaz, and Vikas Chandra. An introduction to vision-language modeling. arXiv preprint arXiv: 2405.17247, 2024.
- [4] Yihan Cao, Jiazhao Zhang, Zhinan Yu, Shuzhen Liu, Zheng Qin, Qin Zou, Bo Du, and Kai Xu. Cognav:

- Cognitive process modeling for object goal navigation with llms. arXiv preprint arXiv: 2412.10439, 2024.
- [5] Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Russ R Salakhutdinov. Object goal navigation using goal-oriented semantic exploration. Advances in Neural Information Processing Systems, 33: 4247–4258, 2020.
- [6] Howard Chen, Alane Suhr, Dipendra Misra, Noah Snavely, and Yoav Artzi. Touchdown: Natural language navigation and spatial reasoning in visual street environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12538– 12547, 2019.
- [7] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. Egoplan-bench: Benchmarking multimodal large language models for human-level planning. arXiv preprint arXiv: 2312.06722, 2023.
- [8] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv: 2412.04453, 2024.
- [9] Xuxin Cheng, Yandong Ji, Junming Chen, Ruihan Yang, Ge Yang, and Xiaolong Wang. Expressive wholebody control for humanoid robots. arXiv preprint arXiv:2402.16796, 2024.
- [10] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural generation. Advances in Neural Information Processing Systems, 35:5982–5994, 2022.
- [11] Vishnu Sashank Dorbala, James F Mullen, and Dinesh Manocha. Can an embodied agent find your “cat-shaped mug”? LLM-based zero-shot object navigation. IEEE Robotics and Automation Letters, 9(5):4083–4090, 2023.
- [12] Kiana Ehsani, Winson Han, Alvaro Herrasti, Eli VanderBilt, Luca Weihs, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Manipulathor: A framework for visual object manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4497–4506, 2021.
- [13] Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong Chen, Willy Chung, Emmanuel Dupoux, Hongyu Gong, Herv´e J´egou, Alessandro Lazaric, Arjun Majumdar, Andrea Madotto, Franziska Meier, Florian Metze, Louis-Philippe Morency, Th´eo Moutakanni, Juan Pino, Basile Terver, Joseph Tighe, Paden Tomasello, and Jitendra Malik. Embodied ai agents: Modeling the world, 2025. URL https://arxiv. org/abs/2506.22355.
- [14] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, et al. Maniskill2: A unified benchmark for generalizable manipulation skills. In The

- Eleventh International Conference on Learning Representations.
- [15] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7606–7623, 2022.
- [16] Xudong Guo, Kaixuan Huang, Jiale Liu, Wenhui Fan, Natalia V´elez, Qingyun Wu, Huazheng Wang, Thomas L. Griffiths, and Mengdi Wang. Embodied llm agents learn to cooperate in organized teams. arXiv preprint arXiv: 2403.12482, 2024.
- [17] Yicong Hong, Cristian Rodriguez, Qi Wu, and Stephen Gould. Sub-instruction aware vision-and-language navigation. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 3360–3376, 2020.
- [18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv: 2106.09685, 2021.
- [19] Yayu Huang, Dongxuan Fan, Haonan Duan, Dashun Yan, Wen Qi, Jia Sun, Qian Liu, and Peng Wang. Humanlike dexterous manipulation for anthropomorphic fivefingered hands: A review. Biomimetic Intelligence and Robotics, page 100212, 2025.
- [20] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang,

Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv: 2504.16054, 2025.

- [22] Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang. Egotaskqa: Understanding human tasks in egocentric videos. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper files/paper/2022/hash/ 161c94a58ca25bafcaf47893e8233deb-Abstract-Datasets and Benchmarks.html.

- [23] Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Jim Fan, and Yuke Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16923–16930. IEEE, 2025.
- [24] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Visionand-language navigation in continuous environments. In European Conference on Computer Vision, pages 104–

120. Springer, 2020.

- [25] Jacob Krantz, Erik Wijmans, Arjun Majundar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision and language navigation in continuous environments. In European Conference on Computer Vision (ECCV), 2020.
- [26] Yujeong Lee, Sangwoo Shin, Wei-Jin Park, and Honguk Woo. LLM-based offline learning for embodied agents via consistency-guided reward ensemble. In Yaser AlOnaizan, Mohit Bansal, and Yun-Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3006–3029, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.170. URL https://aclanthology.org/2024.findings-emnlp.170/.
- [27] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024.
- [28] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang, Gabrael Levine, Wensi Ai, Benjamin Martinez, Hang Yin, Michael Lingelbach, Minjune Hwang, Ayano Hiranaka, Sujay Garlanka, Arman Aydin, Sharon Lee, Jiankai Sun, Mona Anvari, Manasi Sharma, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Yunzhu Li, Silvio Savarese, Hyowon Gweon, C. Karen Liu, Jiajun Wu, and Li FeiFei. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation. arXiv preprint arXiv: 2403.09227, 2024.
- [29] Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, Weiyu Liu, Percy Liang, Li Fei-Fei, Jiayuan Mao, and Jiajun Wu. Embodied agent interface: Benchmarking llms for embodied decision making. arXiv preprint arXiv: 2410.07166, 2024.
- [30] Qiayuan Liao, Takara E Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, and C Karen Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025.
- [31] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pre-

- training from large-scale human videos. arXiv preprint arXiv: 2507.15597, 2025.
- [32] Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Casta˜neda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, et al. Sonic: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.
- [33] Andrew Melnik, Michael B¨uttner, Leon Harz, Lyon Brown, Gora Chand Nandi, Arjun PS, Gaurav Kumar Yadav, Rahul Kala, and Robert Haschke. Uniteam: Open vocabulary mobile manipulation challenge. arXiv preprint arXiv:2312.08611, 2023.
- [34] NVIDIA, Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi ”Jim” Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An open foundation model for generalist humanoid robots. In ArXiv Preprint, March 2025.
- [35] Yuankai Qi, Zizheng Pan, Shengping Zhang, Anton van den Hengel, and Qi Wu. Object-and-action aware model for visual language navigation. In European conference on computer vision, pages 303–317. Springer, 2020.
- [36] Zhangyang Qi, Zhixiong Zhang, Yizhou Yu, Jiaqi Wang, and Hengshuang Zhao. Vln-r1: Vision-language navigation via reinforcement fine-tuning. arXiv preprint arXiv: 2506.17221, 2025.
- [37] St´ephane Ross, Geoffrey J. Gordon, and J. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. International Conference on Artificial Intelligence and Statistics, 2010.
- [38] John Schulman and Thinking Machines Lab. Lora without regret. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20250929. https://thinkingmachines.ai/blog/lora/.
- [39] Bipasha Sen, Michelle Wang, Nandini Thakur, Aditya Agarwal, and Pulkit Agrawal. Learning to look around: Enhancing teleoperation and learning with a human-like actuated neck. In CoRL 2024 Workshop on Wholebody Control and Bimanual Manipulation: Applications in Humanoids and Beyond.
- [40] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, Pete Florence, Wei Han, Robert Baruch, Yao Lu, Suvir Mirchandani, Peng Xu, Pannag Sanketi, Karol Hausman, Izhak Shafran, Brian Ichter, and Yuan Cao. Robovqa: Multimodal long-horizon reasoning for robotics. arXiv preprint arXiv: 2311.00899, 2023.

- [41] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. arXiv preprint arXiv: 1912.01734, 2019.
- [42] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llmplanner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2998–3009, 2023.
- [43] Jingwen Sun, Jing Wu, Ze Ji, and Yu-Kun Lai. A survey of object goal navigation. IEEE Transactions on Automation Science and Engineering, 22:2292–2308, 2024.
- [44] Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. Advances in neural information processing systems, 34:251–266, 2021.
- [45] Andrew Szot, Bogdan Mazoure, Omar Attia, Aleksei Timofeev, Harsh Agrawal, Devon Hjelm, Zhe Gan, Zsolt Kira, and Alexander Toshev. From multimodal llms to generalist embodied agents: Methods and lessons, 2024. URL https://arxiv.org/abs/2412.08442.
- [46] InternVLA-N1 Team. InternVLA-N1: An open dualsystem navigation foundation model with learned latent plans, 2025.
- [47] Qwen Team. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.
- [48] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models - a critical investigation. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 75993–

76005. Curran Associates, Inc., 2023. URL https: //proceedings.neurips.cc/paper files/paper/2023/file/ efb2072a358cefb75886a315a6fcf880-Paper-Conference. pdf.

- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv preprint arXiv: 1706.03762, 2017.
- [50] Fangyuan Wang, Shipeng Lyu, Peng Zhou, Anqing Duan, Guodong Guo, and D. Navarro-Alarc´on. Instructionaugmented long-horizon planning: Embedding grounding mechanisms in embodied mobile manipulation. AAAI Conference on Artificial Intelligence, 2025. doi: 10. 48550/arXiv.2503.08084.
- [51] Hanqing Wang, Wenguan Wang, Wei Liang, Steven CH Hoi, Jianbing Shen, and Luc Van Gool. Active perception for visual-language navigation. International Journal of Computer Vision, 131(3):607–625, 2023.

- [52] Zhenyu Wu, Ziwei Wang, Xiuwei Xu, Jiwen Lu, and Haibin Yan. Embodied task planning with large language models. arXiv preprint arXiv:2307.01848, 2023.
- [53] Ruihan Yang, Yejin Kim, Rose Hendrix, Aniruddha Kembhavi, Xiaolong Wang, and Kiana Ehsani. Harmonic mobile manipulation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3658–3665. IEEE, 2024.
- [54] Yijun Yang, Tianyi Zhou, Kanxue Li, Dapeng Tao, Lusong Li, Li Shen, Xiaodong He, Jing Jiang, and Yuhui Shi. Embodied multi-modal agent trained by an llm from a parallel textworld. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26275–26285, 2024.
- [55] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin S Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander Clegg, John M Turner, et al. Homerobot: Open-vocabulary mobile manipulation. In 7th Annual Conference on Robot Learning.
- [56] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, and Li Fei-Fei. Spatial mental modeling from limited views. arXiv preprint arXiv: 2506.21458, 2025.
- [57] Mingming Yu, Fei Zhu, Wenzhuo Liu, Yirong Yang, Qunbo Wang, Wenjun Wu, and Jing Liu. C-NAV: Towards self-evolving continual object navigation in open world. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=SbfdxWibDn.
- [58] Haoqi Yuan, Yu Bai, Yuhui Fu, Bohan Zhou, Yicheng Feng, Xinrun Xu, Yi Zhan, B¨orje F. Karlsson, and Zongqing Lu. Being-0: A humanoid robotic agent with vision-language models and modular skills. arXiv preprint arXiv: 2503.12533, 2025.
- [59] Abhay Zala, Jaemin Cho, Han Lin, Jaehong Yoon, and Mohit Bansal. Envgen: Generating and adapting environments via llms for training embodied agents. arXiv preprint arXiv:2403.12014, 2024.
- [60] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and Wang He. Navid: Video-based vlm plans the next step for vision-and-language navigation. Robotics: Science and Systems, 2024. doi: 10.48550/arXiv.2402. 15852.
- [61] Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang. Uni-navid: A video-based visionlanguage-action model for unifying embodied navigation tasks. ROBOTICS, 2025. doi: 10.15607/rss.2025.xxi.013.
- [62] Xianqi Zhang, Hongliang Wei, Wenrui Wang, Xingtao Wang, Xiaopeng Fan, and Debin Zhao. Flam: Foundation model-based body stabilization for humanoid locomotion and manipulation. arXiv preprint arXiv: 2503.22249, 2025.

- [63] Yue Zhang, Ziqiao Ma, Jialu Li, Yanyuan Qiao, Zun Wang, Joyce Chai, Qi Wu, Mohit Bansal, and Parisa Kordjamshidi. Vision-and-language navigation today and tomorrow: A survey in the era of foundation models. Transactions on Machine Learning Research.
- [64] Sipeng Zheng, Jiazheng Liu, Yicheng Feng, and Zongqing Lu. Steve-eye: Equipping llm-based embodied agents with visual perception in open worlds. arXiv preprint arXiv:2310.13255, 2023.

CONTENTS

- I Introduction 1
- II Related Work 2

- II-A VLM-based Embodied Agents . . . . . 2
- II-B Mobile-Manipulation . . . . . . . . . . 2
- II-C Visual-Language Navigation . . . . . . 2

- III Framework Design 3

- III-A Task Definition . . . . . . . . . . . . . 3
- III-B Language-based Actions . . . . . . . . 3

- IV Training Recipe 4

- IV-A Model Structure and Training Setup . . 4
- IV-B Data Format . . . . . . . . . . . . . . . 4
- IV-C Data Acquisition . . . . . . . . . . . . . 5
- IV-D Skill Setup . . . . . . . . . . . . . . . . 5

- V Experiments 6

- V-A Experimental Setup . . . . . . . . . . . 6
- V-B Baselines . . . . . . . . . . . . . . . . . 6
- V-C Real-world Benchmarking . . . . . . . . 6
- V-D Virtual Environment Benchmarking . . 7
- V-E Case Studies . . . . . . . . . . . . . . . 8

- VI Conclusion 9
- VII Limitations 13
- VIII Data Format 13
- IX Data Processing Details 14

- IX-A EgoTaskQA Data Processing . . . . . . 14
- IX-B Virtual Environment Data Processing . . 14

- X Supported Skills 15
- XI Training Details of the Manipulation Model 15
- XII Detailed Results for All the Three Runs 15
- XIII Traversability Scenes 15
- XIV Additional Case Study 16

- XIV-A Mobile Manipulation Case Study . . . . 16
- XIV-B Obstacle Avoidance . . . . . . . . . . . 16
- XIV-C Human-robot Interaction . . . . . . . . 17

- XV Data Samples for the EgoActing Task 17
- XVI Prompts for Different Tasks 17
- XVII Difference between our work and existing work 18

VII. LIMITATIONS

The effectiveness of the proposed EgoActor relies heavily on the reliability of external components, including high-level planners (e.g., large language models) and downstream skills such as locomotion policies and visual-language-action models for manipulation. On its own, EgoActor does not function as a fully end-to-end system, as it depends on these supporting models. Future work could explore integrating these capabilities into a single, unified framework to facilitate more seamless deployment on humanoid robots. Another limitation lies in its handling of long-term context: the model may occasionally fall into locally optimal but incorrect decision patterns when navigating extended or multi-stage tasks.

VIII. DATA FORMAT

To enable robust grounding from high-level instructions to executable humanoid actions, we design a structured EgoActing prompt that explicitly exposes the model to egocentric visual context, temporal history, and recent action–observation pairs. The prompt frames the model as a vision–language agent operating from a first-person perspective and requires it to reason over both long-term historical observations and shortterm recent frames. By constraining the output to a predefined set of low-level locomotion, perception, manipulation, and interaction primitives, the prompt encourages spatially grounded, temporally coherent decision-making while preventing unconstrained or hallucinated responses. This design allows the model to infer the next usable action step conditioned on instruction intent, environmental state, and execution history, closely aligning the inference process with real-world humanoid control requirements.

### EgoActing Prompt Design

You are a Vision Language Model specialized in processing the first person view images of embodied robots. Your task is to analyze the provided image and respond to queries with answers. Focus on the spatial relations in the image and make the right decisions.

Given the following instruction, a series of sampled historical observation and recent observation image frames, predict a usable action sequence that you should perform next. Output format: ’Turn [direction] [degrees] degrees; Look [direction] [degrees] degrees; Move [direction] [distance] meters; [direction] sidewalk [distance] meters; [manipulation action text]; [interaction action text]; Stop and no action’. Your task is:

[instruction]

Sampled Historical Observations: [Sampled Historical Observation #1]

- [Sampled Historical Observation #2]
- [Sampled Historical Observation #3]
- [Sampled Historical Observation #4]
- [Sampled Historical Observation #5]
- [Sampled Historical Observation #6]
- [Sampled Historical Observation #7]
- [Sampled Historical Observation #8]
- [Sampled Historical Observation #9]
- [Sampled Historical Observation #10]

Recent Observations:

- [Recent Observation #1] Next action:

- [Recent Action #1]

[Recent Observation #2] Next action:

- [Recent Action #2]

- [Recent Observation #3] Next action:

IX. DATA PROCESSING DETAILS A. EgoTaskQA Data Processing

EgoTaskQA is a benchmark that evaluates models’ event understanding capabilities with goal-oriented questions. The dataset provides fine-grained temporal annotations, including the start and end frames of atomic actions, which makes it suitable for constructing sequential decision-making data. To adapt EgoTaskQA into the EgoActing format for training vision–language models, we reorganize each annotated sequence into structured action-conditioned observation samples.

a) Overview: Each processed training sample consists of three components:

- 1) a global natural language instruction
- 2) a set of historical observations
- 3) three interleaved recent observation-action pairs This structure is designed to accommodate EgoActing,

where models must predict the next feasible action based on the historical and recent observations.

b) Instruction Construction: EgoTaskQA does not provide explicit instruction annotations; instead, it includes detailed annotations of distinct action steps within each video. We first concatenate three adjacent actions (or fewer if insufficient actions are available) into a single instruction using a predefined set of connective patterns (e.g., “and then,” “and next,” “continue to”). For example, an instruction such as “Get the tank from the table, and then open the tank” contains two

sequential manipulation actions. We define the start frame as 60 frames before the start frame of the first action, unless there are preceding actions or this offset exceeds the beginning of the video.

c) Recent Observation-Action Pairs: We start with a target frame that is supposedly used to train the model to predict the next action. For each target frame, we construct three temporally adjacent observation–action pairs by traversing the sequence backward with a fixed stride of five frames. The corresponding action is determined according to the frame’s temporal role:

- 1) if the frame corresponds to the start of a manipulation phase, the action is the extracted manipulation action;
- 2) if the frame lies within a movement segment, the action is constructed by aggregating the calculated camera pose difference between frames;
- 3) if the frame follows the completion of the final manipulation step, a special Stop and no action token is used.

- d) Navigation Action Aggregation and Filtering: Nav-

igation actions are derived by accumulating camera pose changes (e.g., rotation, translation, vertical motion) between consecutive frames. The changes along opposite directions are algebraically combined to account for cancellation effects. To ensure action saliency, we apply magnitude-based thresholds (5 degrees for angular motion and 0.1 meters for translational motion) to determine whether a navigation action is considered valid.

- e) Historical Observations: Historical observations pro-

vide long-term visual context. For each sample, we collect all frames between the instruction start frame and the first recent observation frame, regardless of phase boundaries. From this interval, ten frames are uniformly sampled to form the historical observation set. This strategy avoids reliance on phase-specific image lists and ensures consistent temporal coverage.

- f) Sample Variants: For manipulation actions, we con-

struct two types of samples: one where the target action is a manipulation instruction, and another where the target action is the Stop and no action token, indicating task completion.

- g) Final Dataset: Following the above procedure, Ego-

TaskQA sequences are converted into structured EgoActing samples with aligned visual context, recent action history, and explicit action supervision. This processed dataset enables training models to jointly reason over instruction context, recent egocentric observations, and temporally grounded actions. The raw dataset contains 6,599,590 samples, of which 531,090 include natural language action predictions. Due to computational constraints, we randomly sample 40,000 instances from the full set and 120,000 instances from the subset containing natural language action predictions, resulting in 160,000 training samples for EgoActing. B. Virtual Environment Data Processing

This section describes the processing pipeline used to convert labeled virtual-environment trajectories into training and

evaluation samples compatible with EgoActing-style action prediction.

- a) Trajectory Parsing: Each episode is stored as a trajec-

tory consisting of RGB observations and discrete low-level actions. We first normalize action names (e.g., MOVE_FORWARD, TURN_LEFT, LOOK_UP) and append a terminal STOP action to mark episode completion. For each step, the corresponding RGB image path is recorded, forming an aligned image–action sequence.

- b) Action Merging: To better reflect continuous robot

motion and reduce action sparsity, we merge pairs of consecutive low-level actions into a single executable instruction. The merging process aggregates rotations, forward motion, lateral strafing, and camera pitch adjustments, while introducing small randomized perturbations to movement distances and angles to improve robustness. Terminal actions are mapped to a unified Stop and no action command.

- c) Sliding-Window Sample Construction: From each

episode, we construct multiple training samples using a sliding-window strategy. For a given window position, three recent observation–action pairs are selected at fixed temporal intervals, along with all preceding images treated as historical context. This design enables the model to jointly reason over long-term visual history and short-horizon action prediction.

- d) Final Sample Format: Each sample consists of (1)

a natural-language instruction, (2) a sequence of recent image–action pairs, and (3) a set of historical observation images. All samples are serialized into a unified JSON format for downstream training and evaluation.

This pipeline yields temporally coherent, instructionconditioned samples that closely match the inference-time setting of EgoActor in virtual environments.

X. SUPPORTED SKILLS

Table VI summarizes the set of skills supported by EgoActor, covering movement, perception, manipulation, and human–robot interaction. We represent these skills using two complementary forms of language-based actions: structured language actions for spatial control and natural language actions for open-ended interaction.

- a) Structured language actions: Movement and active

perception are modeled using structured, interpretable action templates that explicitly specify the action type, direction, and magnitude (e.g., Turn left 30.0 degrees, Move forward 0.26 meters). These actions support egocentric locomotion, body-height adjustment, lateral motion, and head orientation, enabling precise spatial positioning from visual observations. Small-magnitude motions are filtered to reduce noise and instability.

- b) Natural language actions: Manipulation and human-

interaction skills are represented using natural language actions rather than a fixed action inventory. This includes object manipulation (e.g., picking, placing, opening), communicative behaviors (e.g., speaking, asking), and gesture-based interactions. This representation allows flexible composition, gener-

alization to unseen actions, and direct grounding of task intent into executable language commands.

We suggest that the natural language actions could be easily extended to facilitate more varied actions in the future.

XI. TRAINING DETAILS OF THE MANIPULATION MODEL

We conducted full fine-tuning of the GROOT-N 1.5 model [34] for 40,000 training steps using an 80GB A800 GPU with a batch size of 50 with the official implementation8. For each task, we expanded the task descriptions to multiple different versions and randomly sampled one description at each training iteration to increase linguistic diversity. Data were collected using a monocular RGB camera. During acquisition, all objects were placed on a white table with a height of 70 cm. The grasped objects included apples, water bottles, plastic cups, oranges, tissue boxes, pen holders, and bowls. Containers consisted of both square and round plates, as well as shallow baskets. In total, the dataset comprises approximately 700 samples.

XII. DETAILED RESULTS FOR ALL THE THREE RUNS

We report the per-run evaluation results corresponding to the averaged performance presented in the main paper. To account for stochasticity introduced by LLM-based inference, we conduct three independent evaluation runs under identical settings. Tables VII and VIII summarize the success rates at different distance thresholds for each run and different models, enabling a finer-grained comparison of performance stability and variance across runs and model scales.

XIII. TRAVERSABILITY SCENES

Fig. 8 illustrates the real-world environments used in the traversability evaluation. Traversability focuses on assessing whether a humanoid robot can safely navigate through narrow spaces—particularly doorways—without colliding with surrounding obstacles, which is a common failure mode for vision–language–model-based navigation systems in realworld settings [46].

We evaluate traversability using room entry and exit scenarios, as doorways are typically constrained in width and require precise control and obstacle awareness. The evaluation includes five real-world rooms: three meeting rooms that are seen during training, and two unseen environments consisting of a private office and a storage room. These rooms differ in layout, visual appearance, carpet, and object arrangement, providing diverse traversal conditions.

For each room, both entering and exiting behaviors are tested. To assess robustness to initial positioning, the robot starts from two different locations relative to the doorway (left and right), with four repeated trials per starting position. This setup allows systematic evaluation of collision avoidance and narrow-passage traversal across varying spatial configurations.

8https://github.com/NVIDIA/Isaac-GR00T

TABLE VI: Supported skills in our training datasets.

Skill Category Skill Description Action Example Structured Language Skills

Move forward/backward Move forward/backward 0.26 meters Turn left/right Turn left/right 30.0 degrees Strafe left/right Left/right sidewalk 0.40 meters Stand up Rise up 0.12 meters Crouch down Lower down 0.08 meters

Movement Skills

Active Perception Skills Look up/down Look up/down 10.0 degrees

##### Natural Language Skills

Confirm/denial gesture Confirm with the woman in front of you Say hi Say hi to the boy Speak Speak ‘‘How you doing?’’ Ask Ask ‘‘Where is the bathroom?’’

Human Interaction Skills

Grab/Grasp/Pick up — Pick up the water bottle Pull — Pull the drawer Place — on — Place the plate on the desk Open — Open the door Close — Close the door Wash Wash hands Pour from — into — Pour from the bottle into the cup Turn on — Turn on the washing machine Turn off — Turn off the lamp Point to — Point to the painting Drop — Drop the garbage

Manipulation Skills

TABLE VII: Multi-threshold success rates across three evaluation runs for the 4B EgoActor model.

Run 1 Distance Success Count

Run 2 Distance Success Count

Run 3 Distance Success Count

- 0.5 m 52.20% 107/205

- 0.8 m 62.93% 129/205
- 1.0 m 70.24% 144/205

- 1.2 m 73.17% 150/205

- 1.5 m 78.05% 160/205
- 2.0 m 84.39% 173/205

- 2.5 m 86.34% 177/205
- 3.0 m 87.32% 179/205

- 0.5 m 48.29% 99/205

- 0.8 m 62.93% 129/205
- 1.0 m 68.78% 141/205

- 1.2 m 74.15% 152/205

- 1.5 m 80.00% 164/205
- 2.0 m 84.39% 173/205

- 2.5 m 87.80% 180/205
- 3.0 m 88.78% 182/205

- 0.5 m 51.71% 106/205

- 0.8 m 65.37% 134/205
- 1.0 m 72.68% 149/205

- 1.2 m 75.12% 154/205

- 1.5 m 78.54% 161/205
- 2.0 m 84.39% 173/205

- 2.5 m 85.37% 175/205
- 3.0 m 87.32% 179/205

XIV. ADDITIONAL CASE STUDY

We provide additional videos in the supplemental materials to illustrate EgoActor’s behavior across diverse scenarios. Occasional latency in the videos is primarily due to network instability rather than the model’s inference speed. The model itself operates with sub-second action prediction, while the observed end-to-end delay is mostly affected by transmission and streaming conditions.

A. Mobile Manipulation Case Study

Fig. 9 illustrates EgoActor performing a mobile manipulation task. We also provide videos of the illustrations in the supplemental materials. Starting from a distant position, the robot first takes larger locomotion steps to efficiently approach the workspace and progressively adjusts its trajectory to align

with the target. As it nears the object, the robot transitions to smaller, fine-grained movements to precisely refine its position for manipulation. The example further demonstrates EgoActor’s ability to adapt its motion to the spatial configuration of the scene: the target object—a previously unseen pink cup—is picked up successfully despite the presence of another object (a pen holder) on the desk, highlighting robust spatial reasoning and fine positioning under cluttered conditions.

B. Obstacle Avoidance

We provide additional qualitative examples of obstacle avoidance behaviors in the supplemental videos. These cases illustrate the model’s ability to navigate around static and dynamic obstacles under unseen layouts, particularly in narrow spaces. The videos highlight how EgoActor adjusts its loco-

TABLE VIII: Multi-threshold success rates across three evaluation runs for the 8B EgoActor model.

Run 1 Distance Success Count

Run 2 Distance Success Count

Run 3 Distance Success Count

- 0.5 m 48.78% 100/205 0.8 m 66.34% 136/205 1.0 m 68.78% 141/205
- 1.2 m 73.17% 150/205

- 1.5 m 78.05% 160/205
- 2.0 m 83.41% 171/205

- 2.5 m 86.83% 178/205
- 3.0 m 88.78% 182/205

- 0.5 m 51.71% 106/205

- 0.8 m 66.34% 136/205
- 1.0 m 68.78% 141/205

- 1.2 m 73.66% 151/205

- 1.5 m 77.07% 158/205
- 2.0 m 84.39% 173/205

- 2.5 m 88.29% 181/205
- 3.0 m 90.73% 186/205

- 0.5 m 53.66% 110/205

- 0.8 m 66.83% 137/205
- 1.0 m 72.20% 148/205

- 1.2 m 75.61% 155/205

- 1.5 m 80.49% 165/205
- 2.0 m 84.39% 173/205

- 2.5 m 88.29% 181/205
- 3.0 m 90.24% 185/205

[Figure 35]

Fig. 8: An illustration of the different scenes we used in our traversability experiments.

motion primitives to maintain safe clearance while preserving task progress.

C. Human-robot Interaction

- Table IX presents representative examples of natural lan-

guage actions predicted by EgoActor in human-interaction scenarios. Given high-level instructions with varying intents, the model generates context-appropriate verbal actions, such as asking for directions, requesting information, or initiating polite inquiries. Note that this task is trained with fewer than 20 samples in the training data. These examples suggest that EgoActor can flexibly map diverse instructional intents to corresponding linguistic actions, enabling basic informationseeking and communicative behaviors during human–robot interaction.

XV. DATA SAMPLES FOR THE EGOACTING TASK

- Table X illustrates representative instruction–action pairs

from both real-world and virtual environments for the EgoActing task. Each instruction specifies a high-level goal, while the

corresponding natural language action describes the concrete behavior executed by the agent. The examples highlight the diversity of tasks, including navigation, object interaction, and human interaction, and demonstrate how the dataset captures both real-world complexities and controlled virtual scenarios. This paired structure supports training and evaluation of models capable of grounding language instructions into egocentric, low-level action sequences.

XVI. PROMPTS FOR DIFFERENT TASKS

To evaluate EgoActor across a diverse set of embodied capabilities, we design task-specific instruction prompts that are high-level yet explicit, ensuring that the intended goal and required actions are clearly specified without prescribing lowlevel motor details. Specifically,

Table XI presents examples of human-interaction prompts, including greeting, asking for information, and requesting objects. These prompts cover both single-person and multiperson settings and require the model to resolve referential

[Figure 36]

Fig. 9: An illustration of our model conducting the mobile manipulation task: “Approach and grab the pink cup”.

TABLE IX: An example of the predicted natural language actions that show the EgoActor could transform the intentions in the instructions into actual words.

Instructions Predicted Natural Language Actions

Approach the person and ask him the way to cook Kung Pao Chicken Ask “How do you cook Kung Pao Chicken?” Approach the person and ask him to show you the way to the reception Ask “Could you please show me the way to the reception?” Approach the person and ask the status of the air conditioner Ask “How is the working of the air conditioner?” Approach the person and ask him to check the news politely Ask “Could you please check the news?” Approach and ask the person to check the news Ask “Do you know the news?” Approach and ask the person to check the news Ask “check the news” Approach the person and ask him the status of the data labeling Ask “How is the progress of the data labeling?” Approach the person and ask him to hand you the flowers Ask “Could you please give me the flowers?”

ambiguity using visual attributes (e.g., clothing, gestures, relative position) or relational descriptions.

Table XII summarizes the instruction prompts used for mobile manipulation tasks. These prompts focus on pickand-place behaviors with varying objects and appearances, requiring the robot to approach the workspace, localize the target, and execute manipulation actions under egocentric observations.

Finally, Table XIII lists prompts for room-level tasks, such as entering or exiting rooms and performing simple interactions after navigation. These prompts are designed to assess the model’s ability to handle narrow passages, spatial transitions, and action sequencing in real-world indoor environments.

XVII. DIFFERENCE BETWEEN OUR WORK AND EXISTING WORK

Our work differs from prior research across three major axes:

a) Scope of embodiment.: Most VLM-based embodied agents [48, 52, 42] and LLM-driven systems [1, 16, 26] focus

on manipulators or simulated agents with simplified embodiments, often relying on predefined skill libraries. In contrast, EgoActor targets full humanoid robots, directly predicting egocentric, low-level actions—including locomotion, posture adjustment, head orientation, manipulation, and humaninteraction—bridging abstract instruction reasoning with realworld motor control.

- b) Unified action reasoning.: Existing mobile-

manipulation and navigation frameworks [12, 44, 53] typically decompose tasks into modular subgoals or stagewise controllers for perception, locomotion, and manipulation. EgoActing and EgoActor unify these components, allowing the model to jointly reason over heterogeneous action types and generate temporally coherent, context-aware sequences without explicit intermediate planning.

- c) Task generalization and real-world deployment.:

Vision-and-Language Navigation (VLN) [2, 63, 36] and object-goal navigation methods [5, 4] primarily address static navigation or object-localization tasks. Our approach extends beyond navigation, supporting dynamic, long-horizon tasks

TABLE X: Examples of instruction–action pairs in real-world and virtual environments.

Instructions Natural Language Actions Real-World Environments

Turn a large right to get out of this area and say hi to the robot in a blue shirt. Say hi to the robot Get out of this area and turn a large left to the leftmost hallway. Stop and no action Turn left and go straight to a kitchen, stop in front of the people. Stop and no action Search and approach the girl and ask her information about humanoid robots. Ask “do you know anything about humanoid robots” Approach and grab the toy bear on the white table. Grab the toy bear

#### Virtual Environments

Turn right until you see a window, then point to it. Point to the window Walk forward to the chair and pull out the red chair in the middle. Pull out the red chair in the middle Approach and put the towels on the bathtub. Put the towels Approach the lamp on the left side of the bed and turn on the lamp. Turn on the lamp Approach and clean the mirror on the wooden cabinet. Clean the mirror

TABLE XI: Examples of human-interaction task prompts used in our evaluation.

Task Category Setting Prompt

Single Person Approach the person with the grey sweater Single Person Approach the person with a brown shirt and say hi to him

Approach the person with a grey shirt and say hi to him Approach the person with a brown shirt and say hi to him Approach the person with a black coat and say hi to him

Multi Person

Approach the person with a hat and say hi to him Approach the person with a hat and say hi to him (swap) Approach the person with a mask and say hi to him

Say Hi

Multi Person

Approach the person touching their own head and say hi to him Approach the person with open palm hand and say hi to him Approach the person squatting and say hi to him

Multi Person

Approach the man and say hi to him Approach the woman and say hi to her

Multi Person

Approach the person on the left and say hi to him Approach the person on the right and say hi to him

Multi Person

Approach the person with a black sweater and ask him the location of the classroom Approach the person with a black sweater and ask him the location of the kitchen Approach the person with a black sweater and ask him the location of the restroom

Ask the Location Single Person

Approach the person with a grey sweater and ask him to give you some flowers Approach the person with a grey sweater and ask him to give you a cup Approach the person with a grey sweater and ask him to hand you a controller

Hand Me Objects Single Person

that combine movement, manipulation, active perception, and human interaction in cluttered, unseen real-world environments. This enables robust instruction grounding into actionable motor sequences suitable for humanoid robots in practical deployment scenarios.

TABLE XII: Task prompts for mobile manipulation (pick-and-place) tasks.

Task Object Instruction Prompt

Pick Apple Approach and grab the red apple on the desk Pick Bottle Approach and grab the green bottle on the desk

Place Apple Approach and place the apple on the desk Place Bottle Approach and place the bottle on the desk

Pick Pen holder Approach and grab the black pen holder on the desk Pick Pink cup Approach and grab the pink cup on the desk

Place Pen holder Approach and place the black pen holder on the desk Place Pink cup Approach and place the pink cup on the desk

TABLE XIII: Room navigation and interaction task prompts.

Room Instruction Prompt Room #1 / #2 / #3

Go into the meeting room Get out of the meeting room

- Room #4

Go into the room in front of you Get out of the room

- Room #5

Go into the office and say hi to the person Get out of the room

