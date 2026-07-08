## GRUtopia: Dream General Robots in a City at Scale

Hanqing Wang1∗, Jiahe Chen1,2∗, Wensi Huang1,3∗, Qingwei Ben1,4∗, Tai Wang1∗, Boyu Mi1,2∗ Tao Huang1, Siheng Zhao1,5, Yilun Chen1, Sizhe Yang1,6, Peizhou Cao1,7, Wenye Yu1,3 Zichao Ye1, Jialun Li1, Junfeng Long1, Zirui Wang1,2, Huiling Wang1, Ying Zhao1 Zhongying Tu1, Yu Qiao1, Dahua Lin1,6, Jiangmiao Pang1†

1OpenRobotLab, Shanghai AI Laboratory 2Zhejiang University, 3Shanghai Jiao Tong University, 4Tsinghua University 5Nanjing University, 6The Chinese University of Hong Kong, 7Xidian University ∗equal contribution †corresponding author

# arXiv:2407.10943v1[cs.RO]15Jul2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Scenes, 89 Scene Categories Generated Tasks

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Navigation

###### Hospital Canteen

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Dialogue

Sure.

[Figure 20]

Office Library

[Figure 21]

[Figure 22]

Could you please bring me a spoon?

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Supermarket

School Manipulation

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Interactive and Finely Annotated Social Interaction with NPC

Figure 1: Key features of GRUtopia.

### Abstract

Recent works have been exploring the scaling laws in the field of Embodied AI. Given the prohibitive costs of collecting real-world data, we believe the Simulationto-Real (Sim2Real) paradigm is a crucial step for scaling the learning of embodied models. This paper introduces project GRUtopia, the first simulated interactive 3D society designed for various robots. It features several advancements: (a) The scene dataset, GRScenes, includes 100k interactive, finely annotated scenes, which can be freely combined into city-scale environments. In contrast to previous works mainly focusing on home, GRScenes covers 89 diverse scene categories, bridging the gap of service-oriented environments where general robots would be initially deployed. (b) GRResidents, a Large Language Model (LLM) driven Non-Player Character (NPC) system that is responsible for social interaction, task generation, and task assignment, thus simulating social scenarios for embodied AI applications. (c) The benchmark, GRBench, supports various robots but focuses on legged robots as primary agents and poses moderately challenging tasks involving Object Loco-Navigation, Social Loco-Navigation, and Loco-Manipulation. We hope that this work can alleviate the scarcity of high-quality data in this field and provide a more comprehensive assessment of Embodied AI research. The project is available at https://github.com/OpenRobotLab/GRUtopia.

© 2024 OpenRobotLab, Shanghai Artificial Intelligence Laboratory. All rights reserved.

### 1 Introduction

Scaling law has seen significant success in NLP and CV, inspiring the robotics community to explore its form in robot learning. A straightforward approach is collecting real-world robot action trajectories, such as recent efforts in Open X-Embodiment [51] and DROID [26]. However, such attempts pose persistent challenges regarding the data collection costs and generalization problems across different hardware platforms. We believe simulation is a crucial step in addressing these issues. Previous works have demonstrated the possibility of learning specific low-level policies, such as Legged Gym [23, 45], ManiSkill [22], and Orbit [38]. Many works have explored Embodied AI in simulation [31, 21, 48, 1]. However, existing platforms exhibit limited diversity and complexity regarding two critical aspects: scenarios and tasks, thereby struggling to meet the demand for achieving policy generalization.

To address the above limitations, this paper presents GRUtopia, the first simulated interactive 3D society designed for various robots that serve humans. GRUtopia distinguishes itself from previous platforms in three key aspects: (a) GRScenes, a large-scale scene dataset capable of constructing cityscale landscapes. It includes 100K fully interactive, finely annotated scenes covering 89 functional categories1; (b) GRResidents, an NPC system leveraging LLMs to generate diverse social characters for interaction, task creation, and assignment; and (c) GRBench, a benchmark featuring “moderately challenging” tasks aligned with current algorithm capabilities.

Firstly, GRScenes significantly expands the scope of environments in which robots can operate. Previous works have predominantly focused on developing general agents for home environments [42, 56, 48]. Beyond housework, we aim to extend the capabilities of general robots to service-oriented scenarios such as supermarkets and hospitals, where they can be initially deployed. GRScenes encompasses both indoor and outdoor environments, including restaurants, supermarkets, offices, libraries, museums, hospitals, exhibition halls, amusement parks, homes, etc. These scenes feature physically realistic materials, detailed external appearances and accessible internal structures, complete with furniture arrangements. The dataset includes numerous high-quality, part-level modeled objects, ensuring that the scenes are fully dynamic and interactive. We provide fine-grained, hierarchical, multi-modal annotations for both scenes and objects, covering levels from the overall scene, indoor regions, objects, down to individual parts.

Secondly, GRResidents, our NPC system, introduces a new dimension to human-robot interaction within simulations. The inclusion of NPCs is motivated by the goal that robots are ultimately meant to serve humans, and interaction with users is often helpful or necessary for task completion, such as resolving ambiguities according to user preferences. GRResidents integrates an LLM agent framework with a hierarchical scene perception module. It possesses comprehensive knowledge about the environment, including attributes, appearance, and structural information of objects provided in our dataset. NPCs can infer spatial relationships between objects, understand scene semantics, observe the activities of other agents in real-time, and engage in dynamic dialogues and task assignments based on this information. Resorting to the powerful NPCs, GRUtopia can generate an infinite number of scene-aware embodied tasks.

Lastly, GRBench serves as a comprehensive evaluation tool for assessing robot agents’ capabilities. To benchmark the robot agents’ ability to handle daily tasks, GRBench comprises three benchmarks: Object Loco-Navigation, Social Loco-Navigation, and Loco-Manipulation. These benchmarks are designed to progressively increase in difficulty, demanding enhanced robotic skills. We prioritize legged robots as primary agents due to their superior cross-terrain capability. However, in large-scale scenarios, it is challenging for current algorithms to simultaneously perform high-level perception, planning, and low-level control while achieving satisfactory results. Inspired by recent progress, which demonstrates the feasibility of training highly accurate policies for individual skills in simulation, the initial version of GRBench focuses on high-level tasks and officially provides learning-based control policies as APIs, such as walking and pick-and-place. Consequently, our benchmark offers a more physically genuine setting that narrows the gap between simulation and the real world.

We conduct extensive experiments to analyze the performance of our NPCs and control APIs, proposing LLM and VLM-driven baselines to validate the soundness of our benchmark design and study the ability of existing LLM or VLM-based agents to tackle embodied tasks. The experimental results reveal that integrating realistic actions with high-level planning increases task difficulty, presenting a key challenge for previous embodied algorithms when applied to real-world scenarios.

1Due to license issues, we will initially release 100 scenes across 7 building types. We are actively working on cleaning the remaining data and plan to release it progressively over time.

Table 1: Comparison of GRUtopia with other platforms in terms of Scene, Object, Platform, and Benchmark.

Vis.RoomRearr.[54]ManipulaTHOR[16]ProcTHOR-10k[15] VLN-CE[29]HomeRobot[56] SocialNavigation[43] Maniskill2[22] VLN[4]CVDN[49]

Behavior-1K[31] Arnold[21]ALFRED[48]

GRUtopia

| |Simulator<br><br>|Isaac Sim|AI2-THOR<br><br>|Habitat<br><br>|SAPIEN|Matterport|
|---|---|---|---|---|---|---|
|Scene<br><br>|Scene Types City Scale Region Label|89 8 1 1 1 1 1 1 1 1 1 1 1<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]| | | | |
|Object|Interactive Part Label Material Label Language Caption<br><br>|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]| | | | |
|Platform<br><br>|Learning-based Controller LLM NPC Kinematics Continuous|[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]| | | | |
|Benchmark|Language Instruction Task Generation Navigation Social Interaction Manipulation|[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>| | | | |

ObjectBenchmarkPlatformScene

It also shows that our constructed benchmark and evaluation metrics exhibit well-defined difficulty granularity and are capable of meeting the research needs across various fields and hierarchies. We hope this platform, with our continuous efforts to scale up the diversity of scenes and tasks on top, can benefit the community.

### 2 Related Work

Embodied AI Benchmarks. VirtualHome [42] and Alfred [48] abstract physical interactions to focus on symbolic reasoning but lack physical realism and action scope. Habitat [46] employs 3D scans of real houses for navigation tasks [8] but lacks physics-based interaction. To enhance physical realism, Habitat 2.0 [56] and iGibson [47, 30] offer realistic actions, environment interactions, and object state simulations. Recent simulation platforms such as ManiSkills [22], TDW [20], SoftGym [32], and RFUniverse [18] focus on physical realism yet lack task diversity. To enhance task diversity, some works explore language-conditioned tasks [57, 24, 36]. GenSim [52] explores the potential of LLMs in generating table-top tasks using code scripts. RoboGen [53] and MimicGen [35] leverage generative models and LLMs to generate tasks. In comparison, our platform supports automatically AI-generated tasks ranging from navigation to mobile manipulation across diverse and higher-quality room layouts. Moreover, our benchmark designs include social interactions [43, 42] between agents and human-like NPCs with access to simulator knowledge. In terms of scene diversity, previous benchmarks are mainly established in the home environments. Compared with Behavior-1K [31] offering 8 different scene types, GRUtopia provides a more diverse dataset encompassing 89 scene categories and builds a city-scale 3D scene with those assets. A detailed comparison between GRUtopia and other benchmarks is shown in Tab. 1.

Low-level control policy. Low-level control for legged robots [23, 45] is typically trained in simulators [50, 44, 14]. Isaac Orbit [38] and Gym [45] enable massive parallel training but are limited to specific terrains. In terms of low-level control policy for manipulation, the Aloha series [59, 19] demonstrates impressive imitation learning in bimanual and mobile manipulation. Recent works like RT-1 [9] scale robotic learning across tasks, while BC-Z [25] generalizes to unseen skills from human videos. Our simulator provides more diverse environments with physical simulations, on which policy training can be performed. Besides, it also offers low-level locomotion and manipulation policy APIs, to foster the study of high-level task planning.

NPC in Simulator. Recent advancements emphasize social interaction as pivotal in human-robot interaction. Habitat 3.0 [43] explores collaborations between humanoid and robot agents in home settings, similar to generative agents [41] that use LLMs to simulate authentic human behaviors. Not limited to task assignment, the NPC design in our research can provide crucial information during task execution, making it transcend traditional human-in-the-loop paradigm. Specifically, our platform employs LLM-driven NPCs to assist and interact with agents to complete domestic tasks, harnessing environmental data and utilizing platform APIs similar to MineDoJo [17].

[Figure 230]

- (a) Scene Category Statistics. (b) Objects Statistics.

[Figure 231]

- Figure 2: The richness and diversity of scenes and objects in GRScenes. (a) The distribution of scenes is shown in a fine granularity. GRScenes covers a vast range of functional scene categories.

- (b) The spectrum of annotated objects and some examples present in GRScenes.

### 3 Dataset & Platform

In this section, we first introduce the GRUtopia platform, which is built on a large-scale 3D scene dataset—GRScenes—featuring diverse objects (Sec.3.1). We then describe GRResidents, the NPC system driven by LLMs for social interactions (Sec.3.2), and the World Knowledge Manager (WKM), which provides platform APIs for NPCs. In Sec. 3.3, we provide the control APIs for various types of robots and evaluate the stability of locomotion in GRScenes.

##### 3.1 GRScenes: Fully Interactive Environment at Scale

To build a platform for training and benchmark embodied agents, fully interactive environments with diverse scene and object assets are indispensable. Therefore, we collect a large-scale 3D synthetic scene dataset with diverse object assets, which serves as the foundation of our GRUtopia platform. Next, we will elaborate on its details from three aspects: scenes, objects, and the corresponding hierarchical multi-modal annotations.

Diverse and Realistic Scenes. Due to the limited amount and categories of open-source 3D scene data, we first collect about 100k high-quality synthetic scenes from designer websites to obtain diverse scene prototypes. These prototypes are then cleaned, annotated with semantics in the region and object level, and finally combined together to form towns as the fundamental playground for robots. As shown in Fig. 2-(a), except for the common home scenes, our dataset consists of ∼30% scenes from other diverse categories, such as restaurants, offices, public, hotels, entertainment, etc. From the large-scale dataset, we preliminarily curate 100 scenes with fine-grained annotations for the open-source benchmark. These 100 scenes contain 70 home and 30 commercial scenes, among which the home scenes are composed of comprehensive common regions with diverse areas (see statistics in the appendix), and the commercial scenes cover common types, including hospitals, supermarkets, restaurants, schools, libraries, and offices. Furthermore, we collaborate with several professional designers to distribute objects following human habits to make these scenes more realistic, as shown in Fig. 1, which are usually ignored in previous works [42, 47, 28]. In this way, we establish a solid foundation to enable general robots to train and test in realistic, large-scale environments.

Interactive Objects with Part-Level Annotations. These scenes contain several 3D objects originally, but some of them do not have internal modeling, making it infeasible to train robots to interact with these objects. To address this problem, we work with a professional team to modify these assets and create complete objects to make them interactable in a physically plausible manner. In addition, to provide more comprehensive information to enable agents to interact with these assets, we annotate all the objects with fine-grained part labels attached to the interactive parts in X-form in NVIDIA Omniverse. Finally, the 100 scenes contain 2,956 interactive objects and 22,001 non-interactive objects from 96 categories, whose distributions are shown in Fig. 2-(b)2.

2For the remaining scenes, we plan to develop a method to automatically construct the interior structure of objects in the future

[Figure 232]

[Figure 233]

###### Annotation Simulator

[Figure 234]

[Figure 235]

[Figure 236]

Sofa

|study room|
|---|

{"Category": "Sofa Chair", "Room": "study room", "Parts": ["back", "cushion"], "Appearance": [ "It has a thick

toilet

lamp

dinning room

kitchen

book

cushioned seat."]}

table

laptop

living room

{"Mass": 20, Backend "Velocity": [0.0, 0.0, 0.0], "Position": [1.9, 3.6, 0.4], "Orientation": [1.0, 0.0, 0.0, 0.0], "Mesh": [...,], ...,}

###### Backend

bedroom bedroom

notebook

pen

balcony

notebook

| | |
|---|---|
| |NPC<br><br>[Figure 237]|

Message

###### World Knowledge Manager

Scene Graph

APIs LLM-based Planner

[Figure 238]

[Figure 239]

|Chat History|
|---|

near on/under

get_diff(target, objects)

near close to

|Response|
|---|

Response

[Figure 240]

get_info(object, type)

LLM-based Speaker

[Figure 241]

[Figure 242]

on/under

[Figure 243]

|LLM-based Programmer| |
|---|---|
| | |

on/under

filter(objects, condition)

close to

close to

[Figure 244]

[Figure 245]

…

- Figure 3: Overview of GRResidents (Sec. 3.2). It comprises two modules: (a) A world knowledge

- manager that organizes scene knowledge from dataset annotations and simulator backend and provides APIs for knowledge retrieving. (b) An LLM planner that is able to retrieve global knowledge from the world knowledge manager and generate responses according to dialogue context.

Hierarchical Multi-modal Annotations. Finally, to enable multi-modal interaction of embodied agents and the environment as well as NPCs, we also need language annotations for these scenes and objects. In contrast to previous multi-modal 3D scene datasets [12, 2, 5] only focusing on the object level or inter-object relationships, we also take different granularities of scene elements into consideration, such as object-region relationships. Given the lack of region labels, we first design a UI to annotate regions on a bird’s eye view of the scene with polygons and then can involve object-region relationships in language annotations. For each object, we prompt powerful VLMs (e.g., GPT-4v) with rendered multi-view images to initialize the annotation followed by human manual checks. The resulting language annotations provide the foundation to generate embodied tasks for subsequent benchmarks in Sec. 4. See more details in the supplementary material.

3.2 GRResidents: Generative NPCs in 3D Environments

In GRUtopia, we endowed the world with social capabilities by embedding some “residents”, i.e., generative NPCs driven by LLMs, thereby simulating social interactions within an urban environment. This NPC system is named as GRResidents. One of the main challenges of building authentic virtual characters in 3D scenarios is integrating the ability of 3D perception. However, the virtual characters can easily access scene annotations and internal states of the simulated world, making robust perception achievable. In this way, we devise a World Knowledge Manager (WKM) that

- manages the dynamic knowledge of the real-time world state and provides access through a series of data interfaces. Resorting to WKM, the NPCs can retrieve desired knowledge and perform fine-grained object grounding through parameterized function calls, which form the core of their perception capabilities.

World Knowledge Manager (WKM). The main duty of WKM is managing knowledge of the virtual environment persistently and offering high-level knowledge of the scene to NPCs. Concretely, the WKM takes hierarchical annotations and scene knowledge from our dataset and simulator backend respectively to build the scene graph as the scene representation, where each node represents an object instance and the edges indicate the spatial relationship between the objects. We adopt the spatial relationships defined in Sr3D [2] as the relation space. This scene graph is preserved by WKM at each simulation step. Moreover, WKM exposes three core data interfaces to extract knowledge from the scene graph: 1) find_diff(target, objects): compare the difference between an target object and a set of other objects, 2) get_info(object, type): get knowledge of an object in terms of the desired attribute type, and 3) filter(objects, condition): filtering the objects according to the condition. Please refer to supplementary materials for more details.

LLM Planner. The decision-making module of NPCs is an LLM-based planner that comprises three components (Fig. 3): a memory module that stores the chat history between the NPC and other agents, an LLM programmer that uses interfaces from WKM to query scene knowledge, and an LLM speaker that consumes the chat history and queried knowledge to produce responses. When an NPC receives a message, it first stores the message in memory and forwards the updated history to the LLM programmer. The programmer then iteratively calls the data interfaces to query the necessary scene knowledge. Finally, the knowledge and history are sent to the LLM speaker, which generates the response.

Experiments. We conduct experiments on object referring, language grounding, and objectcentric QA to demonstrate that our NPCs can: (1) generate object captions, (2) locate objects by description, and (3) provide object information for agents. The backend LLMs of NPC in these experiments include GPT-4o [39], InternLM2-Chat-20B [10], and Llama-3-70BInstruct [37].

Table 2: Cross-verification (Referring & Grounding) accuracy (%) and QA score of our NPCs driven by different LLMs.

|LLM<br><br>|Referring|Grounding|QA|
|---|---|---|---|
|GPT-4o[39] InternLM-2-Chat[10] Llama-3-70B[37]|100.0 93.2 95.8 95.9 83.3 88.7 100.0 88.6 92.5<br><br>| | |

- As shown in Fig. 4, in the referring experiment, we employ a human-in-the-loop evaluation. The NPC randomly selects an object and describes it, then the human annotator selects an object according to the description. Referring is successful if the human annotator can locate the correct object corresponding to the description. In the grounding experiment, the role of the human annotator is played by GPT-4o[39], which provides an object description, and the NPC locates it. Grounding is successful if the NPC can locate the corresponding object. The success rate in Tab. 2 (Referring and Grounding) shows that different LLMs achieve 95.9%-100% and 83.3%-93.2% accuracy, validating the accuracy of our NPC framework in object referring and grounding across LLMs.

[Figure 246]

Figure 4: Evaluation of the NPC’s capabilities in description and grounding through cross-verification between NPC and human.

In the object-centric QA experiment, we evaluate the NPC’s ability to provide object-level information to agents through question-answering in navigation tasks. We design a pipeline that generates object-centric navigation episodes simulating real-world scenarios where the agent asks the NPC questions for information and takes actions based on the answers. Given the agents’ questions, we assess the NPC based on the semantic similarity between its answers and the ground-truth answers. The overall score, as shown in Table 2 (QA), demonstrates that our NPC can provide precise and helpful navigation assistance. Please refer to supplementary materials for more details.

##### 3.3 Robot Control APIs

As physical simulation requires collision handling, low-level control APIs are essential for managing robot agents within the simulator. Unlike prior works that use animation and set positions for pseudoaction execution, we present RL-based controllers as APIs for driving robots. This methodology facilitates the deployment of agent algorithms in executing high-level tasks by leveraging pre-trained, robust low-level control systems. Specifically, we have developed and provided locomotion policies as APIs following state-of-the-art policy learning practices [34, 33] tailored to various robots, including humanoid robots (Unitree H1, Unitree G1, Fourier GR-1) and quadrupeds (Unitree Aliengo with Unitree Z1 Arm). These locomotion APIs have been meticulously designed to be user-friendly, enabling researchers and developers to integrate sophisticated control mechanisms without delving into the intricacies of RL training.

We conduct extensive evaluations of the locomotion controllers on both Unitree H1 and Unitree Aliengo (with Z1 arm) in two distinct environments: an open flat terrain and a cluttered, furniture-rich

- Table 3: Performance comparison of different controllers. (TE: Trajectory Error (m), SR: Success Rate (%), AS: Average Speed (m/s), AT: Average Time (s), PE: Position Error of the end effector (m).)

|Embodiments|Locomotion (Flat) TE SR AS AT|Locomotion (House) TE SR AS AT<br><br>|Manip. (Flat) PE SR|Manip. (House)<br><br>PE SR|
|---|---|---|---|---|
|Unitree H1 Unitree Aliengo+Z1|0.01 100 0.19 43.51 0.01 100 0.25 32.88<br><br>|0.07 58 0.17 36.46 0.13 14 0.16 32.35|- 0.00 100<br><br>|- 0.56 8<br><br>|

environment. We task the robots with point-to-point navigation using a simple shortest path search for path planning. A significant performance drop in success rates is observed, from 100% to 58% for H1 and 14% for Aliengo (See Tab. 3). Similarly, we test the manipulation capability and noted a reduction in success rate from 100% to 8%. We found that this disparity is mainly caused by the failure of path or motion planning, despite the robust performance of low-level control. This highlights the challenges of deploying policies trained in simple environments to real-world complex scenarios. Consequently, we advocate for a more integrated research approach that combines low-level control studies with high-level task execution in these realistic settings. As a temporary workaround, we introduce a reset function that restores the agent’s stable kinematic state and uses the number of resets to evaluate the robustness of the low-level policies. In Sec. 4.4, we study the applications and limitations of these control APIs in handling high-level tasks.

### 4 GRBench: A Benchmark for Assessing Embodied Agents

An embodied agent is expected to actively perceive its environment, engage in dialogue to clarify ambiguous human instructions, and interact with its surroundings to complete tasks. In this section, we introduce our benchmarks in GRUtopia, including the three benchmark setups in Sec.4.1 and the respective evaluation metrics for navigation and manipulation tasks in Sec.4.3.

dinning room

INSTRUCTION : Can you pick the bowl from the dining room, which is on a table, and place it on the table near the sofa?

INSTRUCTION: Can you find the bowl in the dining room that is on the table?

INSTRUCTION : Can you find the bowl on the table?

Dining Room

Dining Room

Dining Room

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

|Agent: Is this the bowl? NPC: Yes.|
|---|

[Figure 254]

[Figure 255]

[Figure 256]

Pick

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

|Agent: Is this the bowl? NPC: No.|
|---|
|Agent: Can you describe it? NPC: It is in the dining room.|

[Figure 261]

[Figure 262]

Place

Living Room

Object Loco-Navigation Social Loco-Navigation Loco-Manipulation

- Figure 5: GRBench task examples. Three benchmarks are established for evaluating embodied agents: Object Loco-Navigation, Social Loco-Navigation, and Loco-Manipulation. The target object in the instruction are subject to some constraints generated by the world knowledge manager. Navigation paths, dialogues, and actions are depicted in the figure; please zoom in for better visualization.

##### 4.1 Benchmark Setups

- As shown in Fig. 5, which displays some example cases, we set up three benchmarks for the comprehensive evaluation of an embodied agent: 1) Object Loco-Navigation, which assesses active perception and navigation; 2) Social Loco-Navigation, which evaluates effective communication with NPCs for instruction clarification; and 3) Loco-Manipulation, which measures mobile manipulation. We generate 300 episodes for each benchmark (100 for validation and 200 for the testing set for embodied agents). Refer to the supplementary files for more implementation details of task definition and task generation.

- Benchmark 1: Object Loco-Navigation. The object loco-navigation task requires the agent to navigate to the target object based on a given language goal. The world knowledge manager (Sec. 3.2) ensures the target is uniquely identified using non-ambiguous natural language. An episode is successful if the target object appears in the agent’s field of view.
- Benchmark 2: Social Loco-Navigation. Recognizing that human intention may not always be clearly provided, the social loco-navigation task evaluates the agent’s ability to actively interact with

#### Agent

###### Environment

|(d) Action Module<br><br>Point2Point Navigation<br><br>Mobile Manipulation<br><br>[Figure 263]<br><br>[Figure 264]|
|---|

Simulator Sensory Data

GRUtopia Codebase

|(a) Grounding Module<br><br>Semantic Point Cloud<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>Cabinet<br><br>Pillow<br><br>Pillow<br><br>2&3D BBox|
|---|

[Figure 268]

[Figure 269]

Isaac Sim

[Figure 270]

| | |
|---|---|
|(c) Decision Module<br><br>{”Action“:”Navigate“, ”Goal“:”Pillow on the bed”,}<br><br>Reasoning<br><br>Speaking<br><br>{”Question“: ”Which pillow should I Pick?”}| |

RGBD Image

[Figure 271]

###### Chat NPC

|(b) Memory Module<br><br>Act.-Obs. History<br><br>Dialogue BEV Map History<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>Act. Obs.| |
|---|---|
| | |

[Figure 280]

[Figure 281]

{”Answer“: ” Find the white pillow on the bed.

Robot State

Instruction

- Figure 6: Overview of the baseline agent. The grounding module (a) processes raw sensory data into semantic-rich information, and the memory module (b) stores historical information like actionobservation history. The decision module (c), which consists of a VLM or LLM, makes action decisions based on information from (a) and (b), while the action module (d) executes the output action. The environment simulates the physical changes resulting from the action and produces sensory data. The agent can also choose to ask the oracle NPC for further instructions about the task.

NPCs and identify the target through dialogue. The agent must ask questions to clarify the features of the target objects (with a maximum of three dialogue rounds). As with Object Loco-Navigation, an episode is considered successful if the target object appears in the agent’s field of view.

- Benchmark 3: Loco-Manipulation. The loco-manipulation task builds on loco-navigation by testing a robot’s ability to pick up and place objects using its arm. This task involves arm manipulations to pick up the target handheld object from initial positions and place it in the target receptacle in the correct position. Agents are required to understand both the appearance and relationships between objects and the receptacles. This task is defined by at most two conditions that describe the target location of the handheld object, including its appearance and relationship with the target receptacles; thus, multiple solutions are allowed since both the handheld objects and target receptacles are not guaranteed to be unique. An episode is considered successful if the handheld object is finally placed in a position that satisfies all conditions in the task specifications. This makes this task more challenging as it requires accurately transporting the correct object to the target position.

Robot Setups. All episodes begin with the agent at a predetermined location and orientation in the task specification. For navigation (Benchmark 1 & 2), we use a Unitree H1 humanoid robot equipped with an RGB-D camera for perception. For mobile manipulation (Benchmark 3), since the Unitree H1 currently lacks sufficient manipulation capabilities, the setup uses a mobile manipulator combining an AlienGo as the mobile base and a Unitree Z1 as the manipulator. An RGB-D camera is mounted on a pole located 0.8 meters above the AlienGo to ensure the camera’s FoV for environmental perception.

##### 4.2 Baselines

Zero-Shot VLM Baselines. We adopt state-of-the-art VLMs from open-source representatives such as InternVL-chat-1.5 [13], as well as closed-source ones from GPT-4o[39] and Qwen-VL[7] families. For Object Loco-Navigation, we directly input the current image observation of the robot and the language prompt into the VLM. The VLM then chooses one action from the following 12 actions as output: (1) Move forward 2/4/6 meters; (2) Advance 2/4/6 meters diagonally to the left/right; (3) Turn left/right 90 degrees; (4) Stop. After the robot executes the chosen action, new observation and language prompt will be given to the VLM to choose the next action. This process will continue until the action output by the VLM is Stop. For Social Loco-Navigation, a new action, Ask, is introduced. This allows the robot to ask NPC for more information about the target object. For Loco-Manipulation, new action types Pick and Place are introduced to support the mobile manipulation ability of the robot agent. Please refer to the appendix for more details.

LLM Agent Baselines. As shown in Fig. 6, the proposed LLM agent consists of a grounding module, a memory module, a decision module, and an action module. The environmental inputs to the agent are egocentric observations of the agent and the current state of the robot. Through the collaborative interactions among these modules, the agent can effectively analyze and utilize environmental inputs,

- Table 4: Quantitative results in GRBench using VLM baselines and LLM agent baselines. Metrics include: PL (Path Length), SR (Success Rate), SPL (Success rate weighted by normalized inverse Path Length), ECR (Excluded candidate rate), SCR (Satisfied condition rate), and RT (Reset times).

|Split|Method<br><br>| |Object Loco-Navigation|Social Loco-Navigation<br><br>|Loco-Manipulation|
|---|---|---|---|---|---|
| | | |PL SR SPL RT|PL SR SPL ECR RT<br><br>|PL SR SCR RT|
|validation|Random<br><br>| |19.78 2.50 1.42 -|20.21 3.00 1.56 - -<br><br>|0.30 0 0 -|
| |VLM|Qwen-VL [7] InternVL-Chat-1.5 [13] GPT-4o [39]|11.43 8.00 4.30 0.18<br><br>12.77 8.00 5.45 0.67<br><br><br>11.93 7.00 3.28 0.17<br><br>|0.60 1.00 0.78 0.92 0.28 15.19 0.00 0.00 0.00 6.67<br><br>6.55 6.00 5.13 1.00 0.96<br><br>|0.11 0 0 0.10 0 0 0.10 0 0 -|
| |LLM<br><br>|Qwen [6] Llama-3 8B [37] InternLM-2-Chat [10] ChatGLM3 [58] GPT-4o [39]<br><br>|25.27 19.00 12.86 0.67 25.97 10.00 7.21 0.84 22.29 18.00 10.97 0.42<br><br>22.75 22.00 12.97 0.60<br>23.36 10.00 5.87 0.87<br>|20.72 14.00 9.33 17.79 1.66<br><br>20.77 5.00 3.49 17.69 1.54<br><br>22.25 11.00 6.37 4.93 1.79<br><br>21.13 8.00 4.61 7.1 2.06<br><br>22.27 7.00 4.49 1.02 2.82<br><br><br>|0.23 0 0 0.21 0 0 0.23 0 0 0.26 0 0 0.25 0 0 -|
|test<br><br>|VLM|Qwen-VL [7] InternVL-Chat-1.5 [13] GPT-4o [39]|12.76 8.00 6.07 0.27 10.85 5.50 3.88 0.23 12.18 14.00 9.13 0.27<br><br>|1.30 0.00 0.00 0.69 0.38 16.80 0.00 0.00 0.00 6.35 11.10 2.00 1.69 0.98 3.10|0.15 0 0 0.13 0 0 0.15 0 0 -<br><br>|
| |LLM<br><br>|Qwen [6] Llama-3 8B [37] InternLM-2-Chat [10] ChatGLM3 [58] GPT-4o [39]|23.68 16.00 10.03 0.61 23.32 15.50 10.96 0.57<br><br>22.62 21.50 12.45 0.62<br><br>23.78 22.00 11.68 0.74<br><br><br>23.16 16.00 5.33 0.80<br><br>|21.14 12.50 9.18 5.21 1.76 21.42 11.00 6.26 7.21 1.89 20.97 7.50 4.40 6.32 2.22 20.36 11.00 6.97 0.13 2.25 20.88 7.00 5.10 5.39 2.16<br><br>|0.30 0 0 0.29 0 0 0.33 0 0 -<br><br>0.26 0 0 -<br>0.27 0 0 -<br>|

VLMLLM

validation

VLMLLM

test

enabling it to engage in both physical and linguistic interactions with the environment. Please refer to the appendix for more details.

##### 4.3 Evaluation

In this subsection, we outline the criteria and metrics used to evaluate the performance of the agent across different benchmarks.

Success Criteria. For Benchmarks 1 and 2, an episode is considered successful if the target object is within the agent’s field of view and less than 3 meters away. For Benchmark 3, success is achieved when the target handheld object is accurately placed at the target position, as evaluated using APIs provided by the world knowledge manager. In all benchmarks, the agent must execute the STOP action within a life horizon of 14,400 physical simulation steps; failure to do so results in the episode being deemed unsuccessful.

Object Loco-Navigation Metrics. Following prior navigation benchmarks [46, 4, 11], we employ four metrics: 1) SR: success rate, the primary metric; 2) PL: path length, indicating the distance traveled by the agent; 3) SPL: success rate weighted by normalized inverse path length [3], balancing success rate against trajectory length; and 4) RT: reset times, indicating the number of times the agent has fallen and been reset to the standing pose at the current position.3

Social Loco-Navigation Metrics. In addition to SR, PL, SPL, and RT, we introduce ECR (excluded candidate rate) for Benchmark 2. Initially, the candidates are filtered by category name in the initial candidate list. After each query from the agent, the object candidates in the temporary list are filtered, and the ratio between the remaining candidates and the initial objects is computed to obtain ECR.

Loco-Manipulation Metrics. In addition to SR, PL, and RT, we introduce SCR (satisfied condition rate) for Benchmark 3, which measures the ratio of satisfied conditions within all conditions.

##### 4.4 Quantitative Results

In this section, we present a comparative analysis of our large model-driven agent framework under different large model backends across three benchmarks. As shown in Tab. 4, we observe that a random strategy yielded a performance close to 0, indicating that our task is non-trivial. When utilizing a relatively superior large model as the backend, we achieve significantly better overall performance, consistently across all three benchmarks. Specifically, we observed that Qwen outperformed GPT-4o in dialogue.

3Because the current locomotion policy of H1 cannot enable it to crawl up by itself, we set the robot to a standing pose when it falls down, allowing it to continue the task.

Table 5: Diagnostic study on the components of our agent in 50 episodes from the validation split.

|#|Perception<br><br>|Object Loco-Navigation<br><br>|Social Loco-Navigation|Loco-Manipulation|
|---|---|---|---|---|
| | |PL SR SPL RT<br><br>|PL SR SPL EC RT|PL SR SC RT|
|1<br><br>2<br><br><br>|GPT-4o [39] Qwen-VL [7]|23.82 8.00 4.26 0.89<br><br>24.67 6.00 4.17 0.87<br><br><br>|21.70 12.00 9.88 1.57 1.80 21.41 12.00 10.36 0.56 2.30|0.23 0 0 0.23 0 0 -<br><br>|

Moreover, our agent framework demonstrates markedly superior performance compared to directly employing a multimodal large model for decision-making. This suggests that even the most advanced multimodal large models currently lack strong generalization capabilities in real-world embodied tasks. However, our approach also has considerable room for improvement. This indicates that when closer-to-real-world task settings are introduced, even tasks like navigation, which have been studied for many years, remain far from being fully solved.

For the loco-manipulation task, we observe a success rate (SR) of 0. Upon analysis, we identified two primary failure causes. First, failures during the locomotion process resulted in a significantly higher number of resets compared to object loco-navigation and social loco-navigation, where the reset counts were similar. We hypothesize that the main reason for this is the larger turning radius and base size of Aliengo compared to H1, making collisions more likely, thus highlighting the flexibility advantage of humanoid robots. Second, during manipulation, the arm frequently collides with other objects in the environment, indicating that current multimodal large models struggle with such complex motion planning.

Finally, the result shows that the difficulty of these three tasks is incremental, providing a good gradation of challenge for evaluation. It is consistent with our intuitions behind task designs: (a) task-oriented scene-aware conversation is more challenging compared to instruction understanding, and (b) the loco-manipulation task is much more difficult than both navigation and static manipulation tasks due to the larger action space, longer planning horizon, and the requirement of more precise navigation and whole body collaboration.

##### 4.5 Diagnostic Study

In this section, we present the results of ablation studies conducted to verify the impact of various factors on model performance and experimental outcomes. These studies were aimed at understanding the influence of task settings and model components on the performance of our agent framework.

First, we compare the effects of the perception module on the experimental results. As shown in Tab. 5, we replace the GPT-4o, responsible for the perception part of the agent framework, with QwenVL and conducted an evaluation. The results indicate that perception performance has a significant impact on overall performance. Specifically, the use of QwenVL leads to superior results compared to GPT-4o. We tested the use of an RL controller versus an oracle action setting where positions are directly set. The oracle actions significantly enhanced performance, validating our hypothesis that there is a substantial gap between this oracle action assumption and real-world application scenarios. This gap indicates that while oracle actions provide an idealized performance, real-world implementations must contend with more complex and less predictable movement dynamics.

##### 4.6 Qualitative Results

We showcase an episode performed by our LLM agent on the Social Loco-Navigation task in Fig. 7 to illustrate how the agent interacts with the NPC. The agent is allowed to talk to the NPC up to three times to query additional task information. At t = 240, the agent navigates to a chair and asks the NPC whether it is the target chair. The NPC then provides surrounding information about the goal to mitigate ambiguity. With the NPC’s assistance, the agent successfully identifies the target chair through an interactive process akin to human behavior. This demonstrates that our NPC is capable of providing natural social interactions for studying human-robot interaction and collaboration.

### 5 Conclusions

In this paper, we present GRUtopia, a novel platform designed to stimulate and benchmark advanced embodied AI research. GRUtopia offers a city-scale environment with diverse functional scenarios (GRScenes), learning-based control APIs for various embodiments, and multi-modal NPCs capable of social interaction, task generation, and task assignment. Utilizing our scene dataset and utilities,

[Figure 282]

NPC

Can you find a chair in the living room?

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

###### t=240

[Figure 288]

Agent

Is what I am facing the goal chair?

[Figure 289]

NPC

No. The target chair is next to a sofa.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

t=720

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

t=960

[Figure 300]

Agent

Is this the target chair in front of me?

[Figure 301]

Yes, that is the target chair.

NPC

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

target

t=1200

Follower View RGB Depth Semantic Segmentation

Figure 7: The agent successfully finds the target object with the help of the NPC.

[Figure 307]

we propose GRBench, which currently includes three benchmarks to assess robots’ capabilities in navigation, social interaction, manipulation. We conducted extensive experiments to thoroughly analyze our benchmark and the performance of state-of-the-art large vision-language models (VLMs). GRUtopia is in active development. In the initial version, we partially release 100 annotated, readyto-use indoor scenes and a city block. The current NPC system supports social interaction without physically realistic contact. We will continuously enhance the platform’s features, including 3D scene assets, control policies, task generation, NPC systems, and benchmarks, to facilitate the scaling up of embodied learning.

[Figure 308]

[Figure 309]

[Figure 310]

Acknowledgement. We clarify author contributions as follows: Jiangmiao Pang initiated and led the project. Hanqing Wang directed the project development. Tai Wang, Jiahe Chen, and Peizhou Cao contributed to GRScenes. Boyu Mi focused on GRResidents. Jiahe Chen, Wensi Huang, Siheng Zhao, and Yilun Chen contributed to GRBench. Qingwei Ben, Zirui Wang, and Junfeng Long trained locomotion policies. Tao Huang and Wenye Yu were responsible for manipulation policies. Sizhe Yang provided various utilities. Zichao Ye, Jialun Li, Ying Zhao, and Zhongying Tu were responsible for platform development. Huiling Wang served as the project manager. Yu Qiao and Dahua Lin provided project advisement.

### References

- [1] Abhishek Kadian*, Joanne Truong*, Aaron Gokaslan, Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis Savva, Sonia Chernova, and Dhruv Batra. Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance? In IEEE Robotics and Automation Letters, 2020.
- [2] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In European Conference on Computer Vision, 2020.
- [3] Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, et al. On evaluation of embodied navigation agents. arXiv preprint arXiv:1807.06757, 2018.
- [4] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- [5] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- [6] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.
- [8] Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi, Oleksandr Maksymets, Roozbeh Mottaghi, Manolis Savva, Alexander Toshev, and Erik Wijmans. ObjectNav Revisited: On Evaluation of Embodied Agents Navigating to Objects. In arXiv:2006.13171, 2020.
- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale. In arXiv preprint arXiv:2212.06817, 2022.
- [10] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.
- [11] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. International Conference on 3D Vision, 2017.
- [12] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, 2020.
- [13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [14] cyberbotics. webots. https://github.com/cyberbotics/webots, 2022.
- [15] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural generation. Advances in Neural Information Processing Systems, 2022.
- [16] Kiana Ehsani, Winson Han, Alvaro Herrasti, Eli VanderBilt, Luca Weihs, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Manipulathor: A framework for visual object manipulation. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- [17] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In "Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2022.
- [18] Haoyuan Fu, Wenqiang Xu, Han Xue, Huinan Yang, Ruolin Ye, Yongxi Huang, Zhendong Xue, Yanfeng Wang, and Cewu Lu. Rfuniverse: A physics-based action-centric interactive environment for everyday household tasks. arXiv preprint arXiv:2202.00199, 2022.
- [19] Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024.
- [20] Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel LK Yamins, James J DiCarlo, Josh McDermott, Antonio Torralba, et al. The threedworld transport challenge: A visually guided task-and-motion planning benchmark towards physically realistic embodied ai. In International Conference on Robotics and Automation, 2022.

- [21] Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, et al. Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes. In IEEE International Conference on Computer Vision, 2023.
- [22] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023.
- [23] Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 2019.
- [24] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 2020.
- [25] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, 2022.
- [26] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.
- [27] Ue-Hwan Kim, Jin-Man Park, Taek-Jin Song, and Jong-Hwan Kim. 3-d scene graph: A sparse and semantic representation of physical environments for intelligent agents. IEEE transactions on cybernetics, 2019.
- [28] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017.
- [29] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In European Conference on Computer Vision, 2020.
- [30] Chengshu Li, Fei Xia, Roberto Martín-Martín, Michael Lingelbach, Sanjana Srivastava, Bokui Shen, Kent Vainio, Cem Gokmen, Gokul Dharan, Tanish Jain, et al. igibson 2.0: Object-centric simulation for robot learning of everyday household tasks. arXiv preprint arXiv:2108.03272, 2021.
- [31] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, 2023.
- [32] Xingyu Lin, Yufei Wang, Jake Olkin, and David Held. Softgym: Benchmarking deep reinforcement learning for deformable object manipulation. In Conference on Robot Learning, 2021.
- [33] Minghuan Liu, Zixuan Chen, Xuxin Cheng, Yandong Ji, Ruihan Yang, and Xiaolong Wang. Visual whole-body control for legged loco-manipulation. arXiv preprint arXiv:2403.16967, 2024.
- [34] Junfeng Long, ZiRui Wang, Quanyi Li, Liu Cao, Jiawei Gao, and Jiangmiao Pang. Hybrid internal model: Learning agile legged locomotion with simulated robot response. In International Conference on Learning Representations, 2024.
- [35] Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. arXiv preprint arXiv:2310.17596, 2023.
- [36] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 2022.
- [37] Meta LLaMA Team. Introducing meta llama 3: The most capable openly available llm to date. https: //ai.meta.com/blog/meta-llama-3/, 2024.
- [38] Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin Yuan, Ritvik Singh, Yunrong Guo, Hammad Mazhar, Ajay Mandlekar, Buck Babich, Gavriel State, Marco Hutter, and Animesh Garg. Orbit: A unified simulation framework for interactive robot learning environments. IEEE Robotics and Automation Letters, 2023.
- [39] OpenAI. Hello GPT-4o . https://openai.com/index/hello-gpt-4o/, 2024.
- [40] OpenAI. New embedding models and API updates . https://openai.com/index/ new-embedding-models-and-api-updates/, 2024.
- [41] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Annual ACM Symposium on User Interface Software and Technology, 2023.
- [42] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.

- [43] Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, et al. Habitat 3.0: A co-habitat for humans, avatars and robots. arXiv preprint arXiv:2310.13724, 2023.
- [44] raisimTech. raisimlib. https://github.com/raisimTech/raisimLib, 2022.
- [45] Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using massively parallel deep reinforcement learning. In Conference on Robot Learning, 2022.
- [46] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In IEEE International Conference on Computer Vision, 2019.
- [47] Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan, Guanzhi Wang, Claudia PérezD’Arpino, Shyamal Buch, Sanjana Srivastava, Lyne Tchapmi, et al. igibson 1.0: a simulation environment for interactive tasks in large realistic scenes. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2021.
- [48] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In IEEE Conference on Computer Vision and Pattern Recognition, 2020.
- [49] Jesse Thomason, Michael Murray, Maya Cakmak, and Luke Zettlemoyer. Vision-and-dialog navigation. In Conference on Robot Learning, 2019.
- [50] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.
- [51] Quan Vuong, Sergey Levine, Homer Rich Walke, Karl Pertsch, Anikait Singh, Ria Doshi, Charles Xu, Jianlan Luo, Liam Tan, Dhruv Shah, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In Conference on Robot Learning Workshop, 2023.
- [52] Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, and Xiaolong Wang. Gensim: Generating robotic simulation tasks via large language models. arXiv preprint arXiv:2310.01361, 2023.
- [53] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455, 2023.
- [54] Luca Weihs, Matt Deitke, Aniruddha Kembhavi, and Roozbeh Mottaghi. Visual room rearrangement. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- [55] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 2021.
- [56] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton. Homerobot: Open vocabulary mobile manipulation, 2023.
- [57] Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong, Ivan Krasin, Dan Duong, Vikas Sindhwani, and Johnny Lee. Transporter networks: Rearranging the visual world for robotic manipulation. In Conference on Robot Learning, 2021.
- [58] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022.
- [59] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

### A More Details of GRScenes

In this section, we provide more details of GRScenes with respect to data annotations (Fig. H) and scene graph extraction.

Scene Graph Extraction. We employ scene graphs [27] to structurally represent scenes, with each node describing an object with attributes such as position, size, and appearance information in textual form obtained by utilizing a visual language model on its multi-view images. Each edge within the scene graph denotes the relationships between objects. We follow [2] for relationship annotation. Notably, unlike scanned scenes that contain surface information only, the detailed internal structures in our scenes enable scene graphs to incorporate additional spatial relationships, namely in and out of. We extract color and shape information from object captions in the scene graph and identify the corresponding objects through exact text match, ensuring precise appearance-to-object alignment.

### B More Details of GRResidents

##### B.1 Implementation Details of World Knowledge Manager

Our world knowledge manager possesses three core functions (Sec. 3.2) to support knowledge extraction from the environment. In this section, we provide Python-style pseudo-code in Algorithms A–C to showcase their implementations.

##### B.2 More Details of NPC Experiments

Object Grounding Experiment. We use GPT-4o[39] to play the role of human annotator in object grounding experiment, see (Sec. 3.2).

For the grounding utterance generation in the experiment, we randomly selected 10 home scenes and their corresponding scene graphs. Based on the appearance information from the object descriptions, we used the template <target object info> <relation> <anchor object info> to automatically generate 500 object descriptions. Subsequently, we utilized GPT-4o to introduce a series of rule-based modifications, making the automatically generated template descriptions more challenging and natural. These modifications include:

Hiding the target object’s category. In real scenarios, common object descriptions might omit the category of the target object, such as "find the object on the table." Hiding the target object’s category can make the descriptions more realistic and naturally increase the difficulty of the grounding task.

Relationship replacement. As the relationships in the template are derived directly from a limited set of scene graph relationships, replacing them with synonyms, such as substituting "near" with "beside" or "next to", enhances the naturalness.

Sentence adjustment. Modifying the original sentences by adding phrases such as "find the ..." or "I want to get ..." at the beginning of the utterances. This type of modification makes the utterances more similar to natural instructions, enhancing the realism and complexity of the descriptions.

For object grounding, we introduce a new API compatible with our framework: grounding(target_object_info, anchor_object_info, relation_name)

Parameter anchor_object_info contains text format description on category, color or shape of the anchor object and target object.

This API first finds out the most suitable object in the scene based on the anchor object information as the anchor object for grounding. Then among all other objects nearby the anchor object annotated in the scene graph, it can find out the target object by matching the spatial relationship between the anchor object and target object and parameter relation_name. Finally, it finds out the target object by matching target_object_info in the same process as finding the anchor object.

In our grounding experiment, with API definitions and task guidance, the NPC reasons through and composes proper arguments for the API call to get the grounding result.

Object-Centric QA Experiment. We use a data generation pipeline combining GPT-4o [39] generation with manual verification. Through the data-generation pipeline, we obtained 489 episodes

###### Figure H: Two examples of meta annotations in GRScenes (Sec. 3.1). The example objects are outlined in orange. Notably, the part labels and material labels are embedded within the object assets.

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

View 1 View 2 View 3 View 4 View 5 View 6

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

View 1 View 2 View 3 View 4 View 5 View 6

{

}

"couch/SM_01_6D7YPDMTDD522XTVKQ888888": { "instance_id": "couch/SM_01_6D7YPDMTDD522XTVKQ888888", "category": "couch", "scope": "Furnitures", "room": "1/living room", "position": [

2.008520397463428, 1.5190480830478437,

- 0.408894387374766

], "min_points": [

- 1.256558941300776, 0.09561449997743492,

-7.080078034960025e-09

], "max_points": [

2.7604818536260796, 2.9424816661182525, 0.8177887818296101

], "description": [

"The object is a white L-shaped couch.", "It has orange cushions.", "The couch has a rectangular base.", "There is a chaise lounge on one end.", "A backrest extends along the length of the couch.", "The couch has visible seams.", "It is made of a textured fabric.", "The cushions are square.", "The cushions have a smooth texture."

], "nearby_objects": {

"window/SM_05_628FFE90_19FA_4477_9708_F431EF1D3347": [ "near", 0.5333050583827639

], "light/SM_01_6IMBRKRVAVAACPTULY888888": [

"near", 2.0458988819787214

], "sofachair/SM_01_6PRLMUJVAJTTWPTUKM888888": [

"near", 0.8054710046610357

],

- "curtain/SM_09_6DJUMTMTD5TGAGWBJQ888888": [ "near", 0.14992408896487763

],

- "curtain/SM_10_6DJUMTMTD5TGAGWBJQ888888": [ "near", 0.245047940131119

], "blanket/SM_04_6MOOWWZVAUTV4PTUKI888888": [

"under", 0.02822649117670297

], "picture/SM_01_6PVLJ4JVAV6CAPTUJU888888": [

"above", 0.18700055798224122

], "trashcan/SM_04_ZFEFG2JVALHVSPTUJU888888": [

"below", 0.1733666386620497

], "teatable/SM_01_ZFEFO5ZVAIJG6PTULY888888": [

"near", 0.23229627379782494

] }

###### S16

{

}

"basket/SM_00_6PY52XBVAV6OSPTUKQ888888": { "instance_id": "basket/SM_00_6PY52XBVAV6OSPTUKQ888888", "category": "basket", "scope": "Furnitures", "room": "7/balcony", "position": [

-1.533837262909261, 4.720350097208155, 0.21521032797343898

], "min_points": [

-1.7399600061728762, 4.530086438764703, 0.000541858890546278

], "max_points": [

-1.3277145196456457, 4.910613755651608, 0.4298787970563317

], "description": [

"The object is a basket with a cylindrical shape.", "It is made of woven material.", "The basket has a light beige color.", "The basket has two handles, one on each side.", "The handles are made of the same woven material.", "The weaving pattern is consistent throughout the basket.", "Vertical and horizontal strands create a grid-like structure.", "The top edge of the basket is reinforced with a thicker woven band."

], "nearby_objects": {

"cabinet/SM_02_cabinet_1": [ "near", 0.3340383792697092

], "plant/SM_04_ZB2KC6ZVAJQQIPTUKU888888": [

"near", 0.9220090184554028

],

- "blanket/SM_04_6PY52XBVAV6OSPTUKQ888888": [ "in", 0.003548709679052394

],

- "blanket/SM_05_6PY52XBVAV6OSPTUKQ888888": [ "in", 0.004331730411871013

], "table/SM_04_ZB2KC6ZVAJQQIPTUKU888888": [

"near", 0.3451297420046326

], "sink/SM_01_6CK2X7MTDD52ANSSLE888888": [

"near", 0.6873307360030145

], "stool/SM_01_6GGTBW4TDD52APTUKA888888": [

"near", 0.07238074949219277

], "bottle/SM_00_ZB2KC6ZVAJQQIPTUKU888888": [

"near", 1.006464318234342

], "faucet/SM_04_ZEMERMZVAJSQIPTUKE888888": [

"near", 1.0355706964183513

], "washingmachine/SM_01_ZELJUSBVAJ5NEPTULM888888": [

"near", 0.0666639528382904

] }

of object-centric navigation. Each episode includes a target object, multiple rounds of agent actions, and natural language interactions with the NPC, covering a total of 1,669 interaction turns.

For evaluation, we use the cosine similarity of sentence embeddings from text-embedding-3-large [40] as the similarity metric. Based on a manual review of a subset of the data, we empirically set 0.6 as the similarity threshold. Answers with a similarity score above this threshold are marked as correct and receive a score of 100, while those below the threshold receive a score of 0. The final result is the average score across all QA pairs.

- Algorithm A This function finds one difference that can shrink the candidate set. We design a hierarchical searching priority that has “category” > “room” > “spatial relationship” > “appearance” to fit human habits in describing objects.

def find_diff(self, target: str, candidates: list[str]): """

return: (diff_type, difference) Tuple[str, list]

""" candidates = [(obj_id, self.spatial_relations[obj_id]) # scene graph

for obj_id in candidates]

categories = set() rooms = set() relations = defaultdict(list)

current_relation = {} for (obj_id, relation) in candidates:

cate = relation["category"] room = relation["room"] categories.add(cate) rooms.add(room) relation_sets = {"near": defaultdict(int)}

for obj, (rel_type, dist) in relation['nearby_objs'].items(): obj_cate = obj.category if not rel_type in relation_sets:

relation_sets[rel_type] = defaultdict(int) relation_sets[rel_type][obj_cate] += 1

for rel_type, item in relation_sets.items(): relations[rel_type].append(item) if obj_id == target:

current_relation[rel_type] = item

# category if len(categories) > 1:

return "category", categories

# rooms if len(rooms) > 1:

return "room", rooms

# relations for rel_type, obj_list in relations.items():

n = len(candidates) - len(obj_list) for i in range(n):

obj_list.append(defaultdict(int)) if not rel_type in current_relation:

return "relation", (rel_type, "nothing") obj_dict_current = current_relation[rel_type] for i, obj_dict_a in enumerate(obj_list):

for cate in obj_dict_current.keys():

if cate not in obj_dict_a: # one of the return "relation", (rel_type, cate)

# appearance return "appearance", None

### C More Details of GRBench

In this section, we provide more details about our three benchmarks in terms of episode generation (Sec. C.1), instruction generation (Sec. C.2), and additional setup details for GRBench, including metrics implementation and simulation setups (Sec. C.3).

- Algorithm B This function returns the request information of the target object.

def get_info(self, object_id: str, info: tuple[str, list]): """

return: info dict[str, any]

""" item_rel = self.spatial_relations[object_id] # scene graph item_attr = self.attribute_set[object_id] # object annotations info_type, info_content = info if info_type == 'category':

return { "cate": item_rel["category"] } if info_type == 'room':

return { "room": item_rel["room"] }

if info_type == 'relation': rel_type_target, target_cate = info_content relations = item_rel["nearby_objects"] flag = False rel_type_set = set() for obj, (rel_type, dist) in relations.items():

rel_type_set.add(rel_type) if rel_type_target == rel_type and target_cate == obj.category:

flag = True break

if target_cate == 'nothing': if rel_type_target in rel_type_set:

return {"relation": [(False, rel_type_target, "nothing")]} else:

return {"relation": [(True, rel_type_target, "nothing")]} return {"relation": [(flag, rel_type_target, target_cate)]}

len_item_attr = len(item_attr) while len(self.sampled) < len_item_attr:

attr_idx = np.random.randint(len_item_attr) _, attr = item_attr[attr_idx] if not attr in self.sampled:

self.sampled.add(attr) break

return {"appearance": [attr]}

[Figure 325]

[Figure 326]

[Figure 327]

(a) Bird’s-Eye View (BEV) map. (b) Occupancy map. (c) Collision-free paths.

- Figure I: Two steps of generating collision-free paths: 1) Generating the occupancy map of the scene. (The black/dark gray/light gray grids indicate the undefined/passable/obstacle regions respectively.)

2) Sampled collision-free paths on the occupancy map.

- Algorithm C This function returns the subset that meets the condition given the original object set.

def filter(self, candidates: set[str], condition: dict[str, list]): '''

return: res_candidates set[str]

''' res_candidates = set() candidates = [(obj_id, self.spatial_relations[obj_id]) # scene graph

for obj_id in candidates]

for obj_id, relation in candidates: attrs = self.attribute_set[obj_id] # object annotations if 'category' in condition:

cate_condition = condition['category'] cate_obj = relation['category'] if not cate_condition == cate_obj:

continue

if 'room' in condition: room_condition = condition['room'] room_obj = relation['room'] if not room_condition == room_obj:

continue

if 'relation' in condition: relation_info = condition['relation'] relation_item = relation['nearby_objects'] flag = True for rel_info in relation_info:

if flag == False:

break has_or_not, rel_a, cate_a = rel_info flag_has = False rel_b_set = set() for cate_b, (rel_b, dist) in relation_item.items():

rel_b_set.add(rel_b) if rel_a == rel_b and cate_a == cate_b:

flag_has = True flag = False break

if has_or_not: if (cate_a == 'nothing' and rel_a in rel_b_set) or flag_has == False:

flag = False else:

if cate_a == 'nothing': if not rel_a in rel_b_set: flag = False if not flag: # does not meet the condition continue

if 'appearance' in condition: app_info = condition['appearance'] app_item = attrs flag = False eb_a = self.get_embedding(app_info) for score, app_b in app_item:

eb_b = self.get_embedding[app_b] if calc_similarity(eb_a, eb_b) > 0.9:

flag = True break

if not flag: continue

res_candidates.add(obj_id) return res_candidates

##### C.1 Episode Generation

Sampling Valid Targets. The starting points of the agent and the target object in the navigation episode are sampled, requiring the guarantee of solvability, i.e., a collision-free path must exist from the starting location to the target location. To achieve this, we first generate the occupancy map (Fig. Ib) for each scene by projecting all objects with a height between [0.1,2.1] m in the scene onto the ground plane as occupied grids; grids outside the floor are marked as an undefined region, both of which are impassable. The remaining non-occupied grids represent passable areas. The resolution of the occupancy map is 1440 × 1440, with each pixel representing a unit length of 1.4 cm. Using the occupancy map, we then generate collision-free paths (Fig. Ic) from randomly sampled points to the locations of objects. The radius for collision detection is 34 cm4. The sampled paths should have a length between [7,20] m to maintain a moderate task horizon length. These paths are used as ground-truth paths for navigation episodes, and object-centric task specifications are generated with our NPC. For objects that do not have a path meeting the criteria, we exclude them from the current version of our benchmarks.

4The radius of Unitree H1 is about 30.5 cm. https://www.unitree.com/h1/

# Object Loco-Navigation def obj_nav(self, cands, target):

infos = [] while len(cands)>1:

diff = self.find_diff(target, cands) info = self.get_info(target, diff) cands = self.filter(cands, info) infos.append(info)

instruction = self.speaker(infos) return instruction

# Social Loco-Navigation def social_nav(self, cands, t):

rounds = np.random.randint(1, n) cnt = 0 while len(cands) > 1 and cnt < rounds:

diff = self.find_diff(t, cands) info = self.get_info(t, diff) cands = self.filter(cands, info) cnt += 1

instruction = self.speaker([info]) return instruction

# Loco-Manipulation def loco_mani(self, cands, src, target_and_conds):

"""target_and_conds: a list of \\ (possible_target, conditions)

""" # obtain the instruction for handheld object nav_instruction = self.obj_nav(cands, src) # obtain the instruction for target receptacle target_and_conds = target_and_conds.copy() all_infos = [] for target, cond in target_and_conds:

infos = [] temp_cands = cands.copy() while len(temp_cands)>1:

diff = self.find_diff(target, temp_cands) info = self.get_info(target, diff) temp_cands = self.filter(temp_cands, info) # randomly drop room or relation attributes if 'room' in info or 'relation' in info:

if np.random.rand() > 0.5: infos.append(info)

all_infos.append((infos, cond)) instruction = self.speaker(all_infos, nav_instruction) return instruction

Figure J: Pseudo code of generating instructions for three tasks.

Sampling Conditions for Loco-Manipulation. For Loco-Manipulation, we generate the “pick-andplace” episodes by sampling the handheld object and the target placement conditions. The source handheld object is sampled as the valid target sampling. The target placement locations are defined as conditions (target object and one spatial relation). We first sample the placement conditions from four potential spatial relations from {on,nearby,nearby × nearby,on × nearby}. Notably, for each condition, we randomly sample the target objects from all candidates. If there are two conditions, we ensure the sampled two nearby objects (within 1.5 meters). For the “on” condition, we ensure the sampled object belongs to receptacle types. To simulate the daily life situation of multiple solutions, during instruction generation, we randomly drop some attributes of the target object description (“room” or “relation”). For evaluation, the satisfaction of conditions is assessed using the same spatial relation program as the generation process.

##### C.2 Instruction Generation

Given the generated episodes for three benchmarks, we detail the instruction generation process by GPT-4 in this section. The pseudo code for generating instructions is shown in Fig. J.

Object Loco-Naivgation. In Object Loco-Navigation, the goal is specified by a language instruction that describes the target object. To generate an instruction that can exclusively describe the target object, we need to find a set of attributes that can identify the target object without ambiguity. We achieve this goal by calling find_diff and filter iteratively until the candidate set has only the target object, and collecting the searched information during the loop5. The collected information is then fed to the LLM-based speaker to generate the unique description of the target object. Due to the priority of difference searching, the generated instruction is concise and consistent with human speech habits.

Social Loco-Navigation. In Social Loco-Navigation, the given language instruction about the target object is often coarse and ambiguous, which is common in human daily dialogues. We aim to reproduce this uncertainty and thus require the agent to actively interact with the NPC efficiently to gather more concrete clues about the target. Since the difference searching has a coarse-to-fine priority, we can generate a coarse instruction by simply calling find_diff once to obtain a crucial attribute that makes the target more distinct. We also have a probability to accept the last attribute searched through multi-round find_diff and filter calling to enhance the diversity of instructions. This attribute is then sent to the speaker to generate the instruction.

Loco-Manipulation. In Loco-Manipulation, the goal is specified by several conditions, with each task defined by up to two placement conditions describing the target location of handheld objects. Each condition is paired with a specified spatial relation to target objects. The instruction generation process

5For the sake of high generation quality and stability, we replaced the LLM-based programmer of our NPC with a static procedure to extract world knowledge. We manually programmed different procedures for the generation of different tasks.

for these target locations, given the sampled conditions, proceeds as follows: 1) The instruction for initial handheld object navigation is generated as Object Loco-Navigation. 2) For each placement condition, information for each sampled target object is collected by calling find_diff and filter, following the prior benchmark. 3) For each collected piece of information, some attributes, including “room” and “relation,” are randomly dropped to simulate multiple solutions while ensuring the existence of a solution through the presence of target objects. The collected information, paired with the sampled relation condition and the initial handheld object navigation, is then fed into the LLM-based speaker to generate the final instruction.

##### C.3 More Details of Benchmark Setups

Metrics. In addition to commonly used navigation metrics such as PL, SR, and SPL, we design two novel metrics (Sec.4.2). ECR assesses the efficiency that the dialogue helps to alleviate ambiguous candidates in Social Loco-Navigation:

n i=1 |objectsi−1| − |objectsi|

ECR =

, objectsi = filter(objectsi−1, conditioni),

|objects0| − 1

(A)

where n is the number of dialogue rounds, conditioni is the new constraint obtained from the i-th round dialogue, objectsi is the subset of objectsi−1 that meets conditioni, and objects0 is the set of all objects belonging to the category of the target object. SCR evaluates the fine-grained task progress of an episode in the Loco-Manipulation task:

n i=1 1(conditioni)

, (B)

SCR =

n

where n is the number of conditions, conditioni indicates whether the i-th condition is satisfied, 1(·) is the indicator function that returns 1 when the input condition is satisfied, otherwise it returns 0.

Simulation Setups. The dt of the physical simulation is 1/240 second, which is aligned with the training setting of control policies. Since the rendering process is the main bottleneck of simulation efficiency, we adopt different working frequencies for the low-level controller and the high-level planner. The dt of rendering is assigned as 1 second, which means the agent can perform visiondependent high-level planning at a frequency of 1 Hz. The control policy works at a frequency of 60 Hz.

### D Baseline Implementation

##### D.1 Zero-Shot VLM Baseline

In this section, we demonstrate the prompts we used for building zero-shot VLM agents.

Prompt Detail of the Object Loco-Navigation VLM Baseline

You are a robot to find an object in the environment given the

→ question: {question} Your task is to select the most appropriate action based on the

→ robot's current view. Here is the action list:

- 1. Move forward 2 meters
- 2. Move forward 4 meters
- 3. Move forward 5 meters
- 4. Advance 2 meters to the left
- 5. Advance 4 meters to the left
- 6. Advance 6 meters to the left
- 7. Advance 2 meters to the right
- 8. Advance 4 meters to the right
- 9. Advance 6 meters to the right
- 10. Turn left 90 degrees
- 11. Turn right 90 degrees
- 12. Stop

You must follow the following action selection conditions: 1-9: If you think this action will help you to have a better view or

→ get closer to the target object. 10-11: Only choose these two actions if it's necessary. 12: If you think you find the target object and you are close enough

→ to the target object. You must follow the following rules:

- 1. Select the action by only output the number with the action. For

→ example: If you want to choose "Stop", just output "12".

- 2. Your output must only be a single integer from 1 to 12.
- 3. Never explain your choice.
- 4. Never include information in your answer that is not relevant to

→ the question.

- 5. Robot's current view is the given image.
- 6. You can't continuously choose to turn right or turn left for more than 2 times. Now you have continuously chosen turn right or turn left for {turning_time} times.

→ →

Prompt Detail of the Social Loco-Navigation VLM Baseline Task Introduction: You are a robot tasked with finding an object in an environment. Your task is to select the most appropriate action from 13 possible actions based on the robot's current view and known information about the target object in order to complete the task of finding the target object.

→ → → →

Action List:

- 1. Move forward 2 meter
- 2. Move forward 4 meters
- 3. Move forward 5 meters
- 4. Advance 2 meter to the left
- 5. Advance 4 meters to the left
- 6. Advance 6 meters to the left
- 7. Advance 2 meter to the right
- 8. Advance 4 meters to the right
- 9. Advance 6 meters to the right
- 10. Turn left 90 degrees
- 11. Turn right 90 degrees
- 12. Stop
- 13. Ask

Action Selection Conditions: 1-9: If you think this action will help you to have a better view or

→ get closer to the target object. 10-11: Only choose these two actions if it's necessary.

- 12: If you think you find the target object and the target object is in the center of your field of view and you are close enough to the target object.

→ →

- 13: If you want more information about the target object. Task: {task} Target Object Information: {goal_info}

Request for Action Selection: Based on the above information and

→ robot's current view, please select the most appropriate action. Follow these rules:

- 1. Select the action by only output the number with the

→ action.(Example: If you want to choose Stop, just output '13')

- 2. If your choice is Ask, you should also output the question you want to ask. Ensure to use a colon as the delimiter.(Example output: 13:Could you please tell me more information about the goal object?)

→ → →

- 3. If your choice is not Ask, your output must only be a single

→ integer from 1 to 12.

- 4. Never explain your choice.
- 5. Never include information in your answer that is not relevant to

→ the question.

- 6. Robot's current view is the given image.
- 7. You can't continuously choose to turn right or turn left for more than 2 times. Now you have continuously chosen turn right or turn left for {turning_time} times.

→ →

##### D.2 LLM Agent Baseline

As shown in Fig. 6, the proposed LLM agent consists of a grounding module, a memory module, a decision module, and an action module. The environmental inputs to the agent are egocentric observations of the agent and the current state of the robot. Through the collaborative interactions among these modules, the agent can effectively analyze and utilize environmental inputs, enabling it to engage in both physical and linguistic interactions with the environment.

Grounding Module is responsible for processing raw environmental inputs into semantically rich information. It takes egocentric RGB-D images captured by the agent and the robot state as inputs and outputs corresponding semantic segmentation results, 3D point cloud, and 2&3D bounding boxes for candidates. We employ SegFormer [55] to perform initial segmentation on the RGB images. Then, combined with the segmentation result, we can use RGB-D images and the robot state to compute the point cloud in the current view and the point cloud for candidates if they exist. By bounding the point cloud of candidates, we can obtain 3D bounding boxes and, with projection to the 2D image plane, 2D bounding boxes can also be obtained.

Memory Module is responsible for maintaining the BEV map, action-observation history, and information about the target object obtained from dialogues between the agent and NPC. It provides directly usable information for decision and action modules. The BEV map is a 2D occupancy map containing candidate positions and descriptions, where the candidate positions are provided by the grounding module and descriptions are generated by a large model. The 2D occupancy map is updated in real-time using the 3D point cloud produced by the grounding module.

Decision Module is responsible for selecting the next action of the robot based on the information provided by the memory module. This function is primarily realized through a large model. This module has two main abilities: reasoning and speaking. When reasoning, the large model uses the prompt illustrated in D.2 to choose the next navigation goal from the current candidates. If speaking, the model is in charge of generating a question to ask the NPC, as illustrated in the prompt in D.2.

Action Module consists of 2 capabilities, navigation and manipulation. 1) The navigation part makes real-time planning of a navigable path to the chosen navigation target. This is primarily achieved by using the RRT* algorithm. Since the 2D occupancy map used for path planning is updated in real-time, the navigation part replans a new path whenever the original path collides with the updated map, continuing until the robot successfully reaches the target or no viable path remains. 2) The manipulation part executes actions like pick and place. Given the target 3D position, the manipulator uses an inverse-kinematics (IK) solver to plan the motion trajectory in joint space. To ease the difficulties of physically realistic picking, the target object can be directly attached to the gripper when the distance between them is within a certain threshold.

D.2: Prompt Detail of Reasoning in Decision Module USER: Here are the descriptions of the current candidates for the goal

→ object {goal}: {description} Here are the known information about the goal object {goal}: {goal_info}

- 1. Each line of candidate description corresponds to a candidate.
- 2. The number in the description is the candidate's index, and the

→ text after ':' is the candidate's description.

- 3. Now, based on the provided information about the goal object,

→ please select the candidate most likely to be the goal object.

- 4. You only need to output the candidate's index. Please do not

→ output anything other than the candidate's index. ASSISTANT:

D.2: Prompt Detail of Speaking in Decision Module USER: Here are the descriptions of the current candidates for the goal

→ object {goal}: {description} Here are the known information about the goal object {goal}: {goal_info}

- 1. Now you can ask a question about the goal object.
- 2. Based on the information described above, what question do you

→ think will help to minimize the scope of the possible candidates?

- 3. Just output the question, don't include the reason or explanation. ASSISTANT:

### E Real-world Demo

We also demonstrate our LLM agent baseline on Object Loco-Navigation in the real world. In the demo, the locomotion capability of H1 is powered by the same control policy used in the simulation. It shows that the agents designed in our simulation platform can be smoothly transferred to a real-world robot driven by our sim-to-real-capable controller. We will release the real-world demo after further polishing in the near future.

