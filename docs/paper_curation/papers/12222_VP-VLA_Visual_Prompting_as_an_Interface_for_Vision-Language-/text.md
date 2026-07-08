## VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models

Zixuan Wang1∗ Yuxin Chen1∗ Yuqi Liu2∗ Jinhui Ye1 Pengguang Chen3 Changsheng Lu1 Shu Liu3 Bei Yu2 Jiaya Jia1,3

# arXiv:2603.22003v3[cs.RO]9May2026

HKUST1 CUHK2 SmartMore3

https://github.com/JIA-Lab-research/VP-VLA

###### Input Subtask Instruct Visual Prompt Execution

[Figure 1]

[Figure 2]

[Figure 3]

- Step1:

Pick up the bottle.

- Step2:

User Query: Recylcle the bottle.

[Figure 4]

###### VP-VLA

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Place it in green box.

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

close carbinet set spoon on towel recycle towel organize egg collect green egg

Figure 1: VP-VLA leverages a dual-system architecture to bridge high-level reasoning and low-level control, maintaining competitive performance across a wide variety of tasks on in-distribution and out-of-distribution settings.

### Abstract

Vision-Language-Action (VLA) models typically map visual observations and linguistic instructions directly to control signals. This “black-box” mapping forces a single forward pass to simultaneously handle instruction interpretation, spatial grounding, and low-level control, often leading to poor spatial precision and limited robustness in out-of-distribution scenarios. To address these limitations, we propose VP-VLA, a dual-system framework that decouples high-level reasoning and lowlevel execution via a structured visual prompting interface. Specifically, a “System 2 Planner” decomposes complex instructions into sub-tasks and identifies relevant target objects and goal locations. These spatial anchors are rendered directly within the native RGB observation space as modality-consistent visual prompts, such as crosshairs and bounding boxes. This avoids the modality mismatch introduced by dense masks, affordance maps, or additional control-specific representations. Guided by these prompts and enhanced by a novel auxiliary visual grounding objective during training, a “System 1 Controller” reliably generates precise lowlevel execution motions. Extensive experiments in simulation and real world demonstrate that VP-VLA surpasses state-of-the-art end-to-end baselines including QwenOFT and GR00T-N1.6.

Preprint.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Place egg at line 3 column 3

hard locate

vague location

Pick up the pear

[Figure 24]

[Figure 25]

Text Prompt

1 column 4

Text Prompt

1

Incorrect pick

Random place

e

lin

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

4

[Figure 30]

[Figure 31]

easy locate

clear location

(a) novel object

(b) OOD position

Visual Prompt

Visual Prompt

Accurate pick

Precise place

- Figure 2: When facing (a) novel objects and (b) unseen spatial configurations, existing VLA models often fail to achieve precise localization (Upper part), whereas our VP-VLA leverages visual prompts to enhance generalization, ensuring accurate placement for target objects (Lower part).

### 1 Introduction

Recent advances in vision language models (VLMs) have revolutionized robotic manipulation. Visionlanguage-action (VLA) models, in particular, aim to bridge semantic understanding and low-level control by fine-tuning pretrained VLMs on large-scale robotic datasets. By doing so, these models inherit strong real-world priors while acquiring embodied skills, offering a promising path toward generalist manipulation policies [1–3].

Despite these successes, existing VLA frameworks often overfit to specific training scene distributions rather than truly grounding instructions in the environment. This is evidenced by recent findings [4, 5] showing that substituting meaningful language with gibberish barely affects performance. Consequently, these policies often fail when encountering novel object categories or unseen spatial positions, as illustrated in Fig. 2. To mitigate these issues, several approaches introduce intermediate interfaces, such as goal images [6] or dense geometric supervision [7, 8], to provide fine-grained guidance. However, these methods typically focus on static, single-task scenarios and rely on rigid interface representations. They often fail to account for the dynamic nature of multi-stage tasks, where the required visual focus and affordance should evolve as the task progresses. Furthermore, curating dense geometric data for these models is prohibitively expensive, and the quality of predicted affordances remains inconsistent. More critically, current VLA systems struggle to effectively integrate high-level reasoning [9] with low-level execution within end-to-end models [10].

To address these challenges, we propose VP-VLA, a decoupled dual-system VLA framework. VPVLA utilizes visual prompts as an explicit, structured interface between high-level reasoning (the “System 2 Planner”) and low-level execution (the “System 1 Controller”). Unlike end-to-end models that attempt to implicitly solve instruction interpretation, spatial relation inference, and execution simultaneously, our approach employs a pretrained VLM as a high-level planner. This planner decomposes complex instructions into sub-tasks and identifies relevant target objects and goal locations. These spatial references are then translated into structured visual prompts, including crosshair markers for targets and bounding boxes for placement regions, which are overlaid onto the visual observations for the low-level controller. By integrating visual prompts directly in the image space, we transform complex linguistic instructions into precise spatial anchors. To ensure the policy effectively utilizes these cues, we introduce an auxiliary grounding objective. This objective encourages explicit spatial awareness within the VLA controller during training.

We evaluate VP-VLA on diverse simulation benchmarks and real-world scenarios, where it consistently outperforms state-of-the-art methods: On Robocasa-GR1-Tabletop benchmark, VP-VLA improves the average success rate by 5% over the baseline, surpassing competitive models like GR00T-N1.6 [1] without requiring additional large-scale robotic pretraining. On SimplerEnv benchmark, our method achieves substantial absolute improvement of +8.3% over baseline, surpassing prior VLA models including π0.5 [11]; In real-world cluttered scenario, our method consistently yield superior performance on both in-distribution and out-of-distribution evaluations. Our contributions are summarized as follows:

- • We propose VP-VLA, a novel framework that decouples high-level reasoning from low-level control through a structured visual prompting interface.

- • We introduce a visual grounding objective during training that enhances the spatial precision and robustness of VLA models.
- • Experiments on Robocasa-GR1-Tabletop, SimplerEnv and real-world scenario demonstrate VP-VLA achieves consistent gains over strong baselines.

### 2 Related Work

Vision-Language-Action Models. Vision-language-action (VLA) models have become a practical paradigm for general-purpose robotic manipulation, translating open-ended semantic instructions into visuomotor policies [2, 12, 13]. Leveraging large-scale robot demonstration datasets [14–20], recent VLAs generalize across diverse tasks and objects by integrating large-scale vision-language models [21–24], multi-modal inputs, and heterogeneous data sources, including real-robot trajectories, human videos, and synthetic simulations. However, most methods adopt a monolithic architecture that tightly couples reasoning, spatial grounding, and action generation, hindering task decomposition and intermediate representation [12, 2, 25]. Under distribution shifts or personalized scenarios [26], VLAs remain brittle, particularly for precise instance-level identification or fine-grained spatial reasoning [27–29]. These challenges highlight a fundamental gap between high-dimensional sensory observations and sparse, low-dimensional action outputs.

Visual Intermediates for Reasoning-Decomposed VLAs. Prior work has explored intermediate visual representations to improve spatial reasoning and action grounding in robotic manipulation. Affordance-based methods predict future image frames [6], key poses [30], trajectories [31, 32], or hierarchical spatiotemporal traces [25, 33] to inform downstream policies. However, because these intermediate modalities are not natively understood by standard Vision-Language-Action (VLA) models, they typically require task-specific supervision and complex end-to-end training that can compromise the model’s inherent reasoning capabilities without guaranteeing executable actions. Seeking a balance between simplicity and information density, recent approaches leverage VisionLanguage Models (VLMs) for subtask reasoning and target localization. Training-free pipelines [34, 35] often struggle with low precision due to the VLM’s imperfect spatial grounding compared to expert segmentation models. Conversely, methods integrating dense segmentation masks from expert models (e.g., DexGraspVLA [36], RoboGround [37]) introduce severe modality mismatches against the original RGB inputs. This necessitates cumbersome architectural modifications—such as training additional vision encoders, designing complex modality fusion layers, or relying on traditional transformer policies like GR-1 [38], which significantly increase computational overhead. Other approaches like Point-VLA [39] ground objects using static first-frame coordinates, which disconnects the visual prompt from the dynamic manipulation process and relies heavily on explicit image augmentation to prevent severe performance degradation. In contrast, our framework establishes an a native, computationally efficient grounded plan that seamlessly bridges high-level reasoning and low-level execution. By leveraging a pretrained VLM [21] for instruction decomposition and SAM3 [40] for precise target localization, we project interpretable spatial cues directly onto the RGB inputs as lightweight visual overlays. This modality-consistent augmentation provides precise spatial guidance while preserving the VLA’s native visual understanding, generalization, and flexibility, effectively eliminating the need for complex modality fusion or additional training overhead.

### 3 Method

We present VP-VLA, a decoupled dual-system framework for robotic manipulation. Following the problem formulation (Sec. 3.1), we propose two core components: (i) the System 2 planner PS2 (Sec. 3.2), an event-driven reasoning module that decomposes tasks into sub tasks and generates visual interface images; and (ii) the System 1 controller πθ (Sec. 3.3), a high-frequency controller that performs visuomotor tracking conditioned on these visual prompts.

#### 3.1 Preliminary

A standard VLA policy πθ typically maps a language instruction l and a sequence of visual observations ot to a sequence of action at at each time step t. The number of visual observation, ot = {o1t,o2t,...,omt }, which came from a series of overhead or wrist-mounted camera, may vary depending on the embodiment. The sequence of action at = {a1t,a2t,...,ant }, called action chunk,

[Figure 32]

[Figure 33]

User Query: “Please put these away into their proper places.”

Subtasks: pick the beige cup and place it in the green box pick the corn and place it in the red box pick the light bulb and place it in the black box subsequent tasks to be executed ...

###### VLM

###### VLA

target object target location

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

###### ... a

at at+1 t+H-1

(x,y) + (x1,y1,x2,y2) Grounding

###### SAM

Action Chunk

Time

Original Image Visual Prompt Original Image

or user defined

Grounding Loss L1 Loss

###### System 2 Planner System 1 Controller

- Figure 3: Overview of VP-VLA pipeline. Our VP-VLA leverages a dual-system architecture to bridge high-level reasoning and low-level control. The System 2 planner first decomposes a language instruction into subtasks and generates visual prompts as interaction anchors and spatial constraints. The System 1 controller then utilizes these grounded visual cues to generate precise sensorimotor trajectories for complex, multi-stage manipulation tasks, while a grounding loss aligns the policy with the highlighted regions during training.

will be executed in sequence to compensate the inference delay and keep the execution smooth. at will be predicted as follows:

at = πθ(l,ot), (1)

where πθ is comprised of a pretrained VLM and an action decoder typically implemented as an MLP or a diffusion model.

Existing VLA models often suffer from a monolithic bottleneck, where a single network must concurrently manage instruction parsing, spatial reasoning, and motor execution. To address this, we propose VP-VLA, a decoupled dual-system architecture that bridges high-level reasoning and low-level control through an explicit visual interface.

#### 3.2 System 2 Planner

The high-level System 2 planner, PS2, performs deliberative reasoning to obtain the visual interface Ivpt . This module operates through two interconnected stages: (i) Event-Driven Task Decomposition, and (ii) Visual Prompt Generation.

Event-Driven Task Decomposition. Instead of performing computationally expensive high-level reasoning regardless of current progression of tasks [41], PS2 utilizes an event-driven execution loop. We hypothesize that manipulation tasks are composed of discrete semantic phases (e.g., grasp, putting down), and the transitions between these phases are marked by transition events. We define the transition event E as a change in the robot’s physical interaction state St. Formally, the high-level planner is invoked only when:

Et = 1(|ϕ(St) − ϕ(St−1)| > ϵ), (2)

where ϕ is a state-mapping function. In our tabletop manipulation setting, we instantiate ϕ as the gripper status. A change in the gripper state (open to closed or vice-versa) serves as a physical proxy for a semantic phase shift, triggering a re-evaluation of the visual prompt to reflect the next sub-goal (e.g., shifting from the target object to the placement destination).

Visual Prompt Generation. Once an event is triggered, a pretrained VLM planner processes the language instruction l and observation ot, then reason about the subtask that needs to further operate, together with the corresponding target object and target location names from the scene. These names are then passed into a pretrained segmentation model to generate a visual interface image Ivpt . This image serves as a spatial bridge, translating abstract language instructions into action affordances. The whole process can be decomposed into semantic reasoning and spatial grounding: In semantic reasoning stage, the planner identifies the current subtask sk and the associated entities e ∈ {eobj,eloc}:

sk,eobj,eloc = VLMplanner(l,ot,St). (3) In spatial grounding stage, a segmentation model G maps these entities to visual prompts ψt:

ψt = G(ot,eobj,eloc), (4)

where ψt consists of an interaction anchor C ∈ R2 (denoted as a crosshair) and a spatial constraint B ∈ R4 represented as a bounding box. These visual prompts are then overlaid on the

overhead camera observation to obtain Ivpt . Unlike raw images, Ivpt provides explicit geometric priors: for manipulation primitives (e.g., “pick”), the system generates a crosshair C at the object’s centroid as an anchor for interaction. This reduces the policy’s search space from the entire image to a localized region of interaction. For placement primitives, a bounding box B defines the spatial constraint for target placement. By representing these as explicit visual overlays, we transform the VLA’s task from “interpreting intent” to “visuomotor tracking” of the provided prompts. After obtaining the visual interface image Ivpt , we feed it together with the original observation ot into the System 1 controller πθ.

#### 3.3 System 1 Controller

We extend the standard VLA formulation by introducing the visual prompt image Ivpt at each step, serves as a spatial bridge between the high-level reasoning and grounding and the low-level robot’s execution. Our policy is defined as:

at = πθ(l,ot,Ivpt ). (5) The VLA policy πθ consists of a VLM backbone fω, which processes multimodal inputs into highlevel embeddings, and an action decoder hψ, which maps these embeddings to continuous control signals. The policy is thus defined as:

at = πθ(l,ot,Ivpt ) = hψ fω(l,ot,Ivpt ) , (6) where ω and ψ are the parameters of the VLM and the action decoder, respectively, and θ = {ω,ψ}. Training Objective. A key challenge in visual prompting is ensuring the model treats the overlays as semantic anchors rather than extraneous image noise. To address this, we introduce a visual grounding objective that forces the model to internalize the spatial coordinates of the prompts. Our framework can be naturally extended with the auxiliary grounding task. During training, we add an auxillary grounding task on only key frames (first frame and the frame where Et = 1). We formulate grounding as a classification task over discretized spatial bins. Following the design of Qwen-3-VL, we divide the image dimensions into N uniform bins, where N = 1000. For target object crosshair with its center located at (x,y), we query the VLM inside the VLA πθ to predict the 2D location. For target location bounding box, we query the VLM to predict the location [x1,y1,x2,y2]. During training, the VLM fω is queried to predict these discretized locations in a structured JSON format. We optimize this using a Cross-Entropy (CE) loss for grounding, which provides a sharper and more structured training signal than traditional MSE. We use L1 loss for action prediction. Critically, the grounding loss is backpropagated only through the VLM parameters ω:

Ltotal = Laction(θ) + λ1eventLgrounding(ω), (7) where λ is the coefficient to balance action prediction and visual prompt grounding. This auxiliary grounding loss ensures that the policy’s internal representations are explicitly aligned with the visual prompts rather than treating them as external noise, leading to more precise and robust manipulation.

Data Preparation. For better consistency and efficiency, we use rule-based approach to first decompose the original task into a subtask list. At key frames, a VLM predicts the current subtask from the list, along with the target object and (if applicable) target location. Using the predicted object and location names, we perform text-conditioned segmentation on all frames to obtain masks and bounding boxes before the next key frame. These annotations are then converted into visual prompts ψ: a crosshair placed at the centroid of the target object mask and a bounding box over the target placement region. Each processed episode is stored with per-frame masks, boxes, and VLM subtask records. On Open X-Embodiment (OXE) dataset, we follow the original starVLA [42] experiment setting to discard episodes with empty prompt for both our methods and the baseline.

### 4 Experiment

We conduct extensive experiments to validate our method, both in simulation and real-world settings. First, we elaborate the implementation details in Section 4.1. Then, we assess performance on the simulation benchmark in Section 4.2 & 4.3. Next, we examine real-robot performance on cluttered and under-specified manipulation tasks to study instruction-following and OOD generalization in real-world deployment in Section 4.4.

- Table 1: Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. We highlight the best result in bold and the second-best results with underline.

Task Isaac-GR00TN1.5 Isaac-GR00TN1.6 QwenGR00T+Qwen3VL +Qwen3VLQwenPI +Qwen3VLQwenOFT QwenFAST+Qwen3VL +Qwen3VLOurs

PnP * to * Close (Avg) 45.3 24.2 50.3 42.3 43.7 35.0 54.3 PnP Novel From Cuttingboard To * (Avg) 46.4 56.9 52.8 46.0 50.4 50.4 60.8 PnP Novel From Placemat To * (Avg) 45.5 51.9 38.0 43.5 41.5 33.5 54.5 PnP Novel From Tray To * (Avg) 48.8 55.1 39.2 44.0 49.2 32.0 46.0 PnP Novel From Plate To * (Avg) 56.5 57.6 58.5 44.0 61.0 45.0 53.5 Average 48.2 47.6 47.8 43.9 48.8 39.0 53.8

#### 4.1 Implementation Details

We use SAM3 [40] to obtain the visual prompt. We use the default segmentation threshold where detection threshold and mask threshold are 0.5. We keep the visual prompt with highest score for target object and target location respectively. Our codebase is based on starVLA [42] framework, trained on 8 GPUs, and strictly follows the training and evaluation procedure to ensure reproducibility. We adopt QwenOFT architecture, which replace the Prismatic VLM [43] in OpenVLA-OFT [44] with Qwen3-VL-4B-Instruct. We use Qwen3-VL-4B-Instruct for System 2 Planner as well. We employ the AdamW optimizer with learning rate as 1e-5 for VLM and 1e-4 for the action model. We set the λ to be 0.1 when calculating loss. The visual prompts are all automatically predicted except for the egg carton placement task, where the target object is predicted by the framework and the target location is specified by the user. Baseline performance metrics are sourced from original papers or other peer-reviewed publications. To ensure a fair comparison, the training datasets for these baselines include the data utilized in our own experiments.

#### 4.2 Experiment on Robocasa Benchmark

We applied our pipeline to the Robocasa-GR1-Tabletop benchmark [45], a simulation framework with tabletop kitchen environment, consisting of 24 diverse tasks and in total 24,000 videos. These tasks involves multi-step complex pick and place interactions with varied attributes and geometries. We utilize the Humanoid Robot Tabletop Manipulation subset from the PhysicalAIRobotics-GR00TX-Embodiment-Sim [1] dataset, following [42, 46]. To guarantee reproductibility and statistical significance, we evaluate each task using 50 independent trials and report the average success rate.

The quantitative results on the RoboCasa Tabletop simulation benchmark are summarized in Table 1. Taking QwenOFT as the primary baseline, our method achieves a new state-of-the-art average success rate of 53.8%, outperforming QwenOFT (48.8%) by a clear margin of +5.0%. Our approach also surpasses other strong baselines, including Isaac-GR00T N1.5 (48.2%), Isaac-GR00T N1.6 (47.6%), QwenGR00T (47.8%), and QwenPI (43.9%). Notably, the improvement is particularly evident in the “PnP * to * Close” setting, where our method reaches 54.3%, significantly exceeding QwenOFT (43.7%) and all other competitors. We observed that for long complex instructions involving multiple steps and nonprehensile grasping, such as “pick up the wine, place it into the cabinet and close the cabinet”, the VLM reasoner successfully decompose the task into subtask list [“pick up the wine”, “place the wine into the cabinet”, “close the cabinet”]. In addition, it identifies the target object and the specific affordance required for the final action (the cabinet door). Furthermore, the reasoner accurately detects subtask transitions, ensuring the target object shifts from the “wine” to the “door” only after the wine has been successfully placed. We also observe consistent gains in several challenging novel generalization splits, such as “PnP Novel From Placemat To Plate” (70.0% vs. 52.0% for QwenOFT) and “PnP Novel From Tray To Plate” (66.0% vs. 56.0%), where the evaluation includes random initialized position, novel appearance, and distracting object and container. Our method not only improves overall task success rate but also enhances generalization for varing background, object attribute and position.

- Table 2: Results of evaluating the VLA models with the WidowX robot in the SimplerEnv simulation environment. We highlight the best results in bold and the second-best results with underline.

Put Spoon on Towel

Put Carrot on Plate

Stack Green Block on Yellow Block

Put Eggplant in Yellow Basket

Method

Average

RT-1-X [15] 0.0 4.2 0.0 0.0 1.1 Octo-Base [48] 15.8 12.5 0.0 41.7 17.5 Octo-Small [48] 41.7 8.2 0.0 56.7 26.7 OpenVLA-OFT [44] 34.2 30.0 30.0 72.5 41.8 RoboVLM [49] 50.0 37.5 0.0 83.3 42.7 Magma [50] 37.5 29.2 20.8 91.7 44.8 CogACT [51] 71.7 50.8 15.0 67.5 51.3 SpatialVLA [52] 20.8 20.8 25.0 70.8 34.4 TraceVLA [25] 12.5 16.6 16.6 65.0 27.7 VideoVLA [53] 75.0 20.8 45.8 70.8 53.1

π0 [2] 29.2 62.5 29.2 91.6 53.1 π0.5 [11] 49.3 64.7 44.7 69.7 57.1 Isaac-GR00T-N1.6-Bridge [1] 64.5 65.5 5.5 93.0 57.1

QwenOFT + Qwen3VL 58.3 50.0 20.8 70.8 50.0 Ours + Qwen3VL 66.7 50.0 20.8 95.8 58.3

#### 4.3 Experiment on SimplerEnv Benchmark

We utilize two large-scale subsets from the Open X-Embodiment (OXE) dataset: BridgeDataV2 [20] and Fractal [14]. The model is fine-tuned for 70k steps on 8 GPUs (batch size 32 per device). This benchmark includes four manipulation tasks: “Put spoon on towel”, “Put carrot on plate”, “Stack green cube on yellow cube”, and “Put eggplant in yellow basket”. We evaluate the policies using the official evaluation scripts provided by the SimplerEnv repository [47].

The quantitative results on the SimplerEnv simulation benchmark are summarized in Table 2. Using QwenOFT as the primary baseline (50.0% average), our method achieves a new state-of-the-art performance of 58.3%, yielding a substantial improvement of +8.3%. Compared with other strong competitors, our approach also surpasses π0.5 (57.1%) and Isaac-GR00T-N1.6-Bridge (57.1%), and outperforms prior VLA systems such as CogACT (51.3%) and VideoVLA (53.1%). At the task level, we observe notable improvements over QwenOFT in tasks requiring precise object identification, manipulation and target location grounding, including “Put Spoon on Towel” (66.7% vs. 58.3%) and a substantial gain in “Put Eggplant in Yellow Basket” (95.8% vs. 70.8%). These findings suggest that our approach more effectively leverages language-conditioned signals to guide action selection, establishing a new performance ceiling on this benchmark.

#### 4.4 Experiment on Real-world Scenario

We comprehensively evaluate VP-VLA across multiple real-world manipulation tasks to validate its core capabilities in several dimensions. Specifically, we focus on: (i) the reasoning and grounding ability within cluttered scenes reflected on overall success rates; (ii) the robustness and generalization ability in out-of-distribution (OOD) settings; and (iii) the effectiveness of visual prompting against textual prompting in scenarios requiring complex spatial reasoning.

For each of the following section, we introduce our experimental setup and evaluation results. We use a stationary, table-mounted Franka Research 3 7-DoF robot arm. Environment observation includes two RGB images: one from a fixed third-person perspective and the other from a first-person camera mounted on the end-effector. The setup is shown in 4. Both images are resized to 224 × 224 before feeding into the model. We fine-tune both the baseline model and our method using 8 GPUs, with total batch size of 256 and action chunk size of 16.

Robotic Waste-Sorting Categorization. To evaluate visual grounding and generalization in cluttered environments, we design a robotic pick-and-place task inspired by daily waste-sorting. Models must follow instructions to categorize randomly posed objects into specific containers, which requires both visual grounding ability and precise action prediction. We collect 50 trajectories per training object and evaluate both our method and the QwenOFT baseline at 10K training steps. We evaluate performance across In-Domain (ID) setting and out-of-distribution setting (OOD) with novel object.

Experimental Environment Real world Training Data In-Domain references challenge

Cluttered pick Color-based pick Precise location place

Select external camera based on the task Each task using one external+wrist camera

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Franka Research 3

External-camera A/B

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Wrist-camera

Out-of-Domain generalization challenge

OOD objects pick OOD position&color pick OOD location place

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

(a) Robotic Arm Embodiment (b) Training dataset (c) Task Overview

- Figure 4: Overview of our real-world task and robot setting. (a) Robot setting for the experiments. We use an external camera A for categorization task and pick colored egg task, while using the external camera B for egg carton placement task. (b) We collect real-world robot demonstration for three task suites. (c) We present examples of each task with the OOD setting illustration.

[Figure 60]

[Figure 61]

69%

91%

75%

75%

77%

55%

71%

54%

29%

58%

25% 45% 65% 85%

Egg Placement(OOD)

Egg placement(ID)

Colored Egg(OOD/P)

Colored Egg(OOD/C)

Colored Egg(ID)

Egg-related task results

QwenOFT Ours(Full)

[Figure 62]

[Figure 63]

85%

88%

85%

85%

85%

93%

85%

85%

63%

80%

40%

78%

85%

85%

75%

78%

35% 45% 55% 65% 75% 85% 95%

Overall(OOD)

Overall(ID)

Black box(OOD)

Black box(ID)

Red box(OOD)

Red box(ID)

Green box(OOD)

Green box(ID)

Categorization results

QwenOFT Ours(Full)

(a) (b)

- Figure 5: Results of real-world robot manipulation. (a) Categorization result; (b) Pick colored egg result and egg carton placement result.

As shown in Fig. 5, our method achieves an 87.5% ID success rate and 85% OOD success rate, significantly outperforming QwenOFT (80% ID, 63.3% OOD). Notably, our method exhibits a minimal generalization gap (2.5%), whereas QwenOFT suffers a 16.7% performance drop. This suggests that while the baseline overfits to the training distribution, our approach maintains robust object-level grounding across categories in cluttered scenes.

Object Reference by Attribute. To evaluate fine-grained attribute grounding, we design a colorbased reference task: “pick up the <color> egg.” Unlike category-level reasoning in the previous experiment, this task focuses on attribute-level grounding, requiring the model to bind specific linguistic color tokens to visual instances under spatial variation. Using a 4 × 4 grid, we collect 200 demonstrations (50 per color for blue, pink, red, and yellow). We evaluate performance across three settings: In-Domain (ID), OOD Color (novel purple and green eggs), and OOD Position (eggs placed in 12 unseen grid locations).

Fig. 5 summarizes the performance across all settings. Our method significantly outperforms the baseline, particularly under distribution shifts. For in-domain scenario, our model achieves 77.1% accuracy, outperforming the baseline by a large margin of 18.8%. For OOD Color scenario, our method maintains 75.0% success on novel colors, while the baseline drops to 29.2%. The minimal gap between our ID and OOD color performance (2.1%) suggests the learned representation captures attribute semantics rather than memorizing instances. In contrast, the baseline exhibits severe overfitting to seen color distributions. Finally, for OOD position with unseen grid locations, our model reaches 75.0% compared to 54.2% for the baseline, demonstrating spatial generalization beyond training data, while the baseline shows weaker position invariance.

Location reference with spatial grounding. In extreme scenario where it is hard to specify the exact location, we present an option to let the user provide the visual prompt. To evaluate whether our proposed method can still follow user’s intent in such scenario, we design an egg carton placement task. The robot is tested to pick up an egg and place it at a language-specified coordinate (e.g., “line 2, column 4”) on a 4×4 grid. We manually drawn visual prompt (bounding box) at the target location for our method to resolve potential linguistic ambiguity, whereas the baseline must ground the instruction

- Table 3: Ablation results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment

w/o grounding

w/ all frame grounding

w/ point

w/ direct overlay

Ours (Full)

Task

PnP * to * Close (Avg) 49.7 46.3 38.7 47.7 54.3 PnP Novel From Cuttingboard To * (Avg) 54.4 53.6 48.8 54.0 60.8 PnP Novel From Placemat To * (Avg) 46.0 45.0 46.0 46.5 54.5 PnP Novel From Tray To * (Avg) 43.2 46.0 49.6 50.0 46.0 PnP Novel From Plate To * (Avg) 54.0 58.0 56.5 56.5 53.5 Average 49.4 49.5 47.3 50.8 53.8

purely from text. We evaluate on In-Domain (ID) and Out-of-Distribution (OOD) coordinates using a partial credit system: 1.0 for the target cell, 0.5 for adjacent, and 0.25 for diagonal cells.

As shown in Fig. 5, our method consistently outperforms QwenOFT across all settings: For InDomain setting, our model achieves 91.25% accuracy (36.5/40), a more than 20% improvement over QwenOFT (70.63%). This reflects stable spatial grounding when coordinates are seen during training. For out-of-distribution scenario with novel row–column combinations, our method reaches 68.75% , compared to 55% for the baseline. While OOD compositional generalization is challenging for both, our model maintains a significant performance lead, suggesting superior geometric grounding over memorization of frequently seen coordinates.

#### 4.5 Ablation Study

We conduct ablation experiments on the RoboCasa Tabletop simulation to analyze the contribution of each design component. Table 3 reports the results.

Effect of the Grounding Objective. Removing the grounding loss (“w/o grounding”) reduces the overall success rate from 53.8% to 49.4%, demonstrating that explicitly aligning policy representations with the prompted regions is crucial. Without this constraint, the model may perceive the visual prompts but fail to consistently associate them with action generation. We further evaluate grounding applied to every frame (“w/ all frame grounding”). This variant achieves 49.5%, which is lower than our key-frame grounding strategy. Applying grounding supervision densely across all frames may introduce redundant or noisy constraints, leading to unstable training and suboptimal optimization. This result suggests that selective grounding at key decision frames provides a better balance between supervision strength and training stability.

Effect of Visual Prompt Design. Changing the target object prompt from a crosshair to a point (“w/ point”) degrades performance to 47.3% on average. A single point provides weaker spatial extent information compared to a structured crosshair, making it harder for the policy to infer object location but rather treating it as visual perturbation. This indicates that prompt geometry influences how effectively spatial cues are interpreted. We also test directly overlaying prompts on the primary RGB image (“w/ direct overlay”), which yields 50.8%. Separating the visual prompt avoids excessive interference with raw visual features and prevent overlapping.

Overall Discussion. Our full method ranked the best among other variants. The results demonstrate that (i) grounding supervision is necessary for reliable prompt utilization, (ii) supervision should be applied selectively rather than densely, and (iii) prompt representation design significantly affects perception. Together, these findings validate the current design.

### 5 Conclusion

This paper presents VP-VLA, a novel dual-system vision-language-action framework. By decoupling high-level reasoning from low-level execution, our approach leverages a “System 2 Planner” to translate complex linguistic instructions into explicit and structured visual prompts. These spatial anchors, combined with an auxiliary visual grounding objective during training, effectively guide the “System 1 Controller” to perform precise manipulation tasks. Extensive evaluations across simulated benchmarks, including Robocasa-GR1-Tabletop and SimplerEnv, as well as real-world cluttered scenarios, demonstrate the superior performance and generalization of VP-VLA.

### References

- [1] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo

Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

- [3] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.
- [4] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-languageaction models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.
- [5] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of vision-languageaction models. arXiv preprint arXiv:2510.13626, 2025.
- [6] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.
- [7] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025.
- [8] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Tianran Zhang, Wenxuan Song, Jiayi Chen, Xinhu Zheng, Hesheng Wang, et al. Flowvla: Visual chain of thought-based motion reasoning for vision-language-action models. arXiv preprint arXiv:2508.18269, 2025.
- [9] Daniel Kahneman. Thinking, fast and slow. macmillan, 2011.
- [10] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.
- [11] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny

Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [12] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model, 2024. URL https://arxiv. org/abs/2406.09246.
- [13] Jiajun Xi, Yinong He, Jianing Yang, Yinpei Dai, and Joyce Chai. Teaching embodied reinforcement learning agents: Informativeness and diversity of language use, 2024. URL https://arxiv.org/abs/2410.24218.
- [14] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar,

- Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023. URL https://arxiv.org/abs/2212.06817.
- [15] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open xembodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.
- [16] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. URL https://arxiv.org/abs/2307.15818.
- [17] Joshua Jones, Oier Mees, Carmelo Sferrazza, Kyle Stachowicz, Pieter Abbeel, and Sergey Levine. Beyond sight: Finetuning generalist robot policies with heterogeneous sensors via language grounding, 2025. URL https://arxiv.org/abs/2501.04693.
- [18] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset, 2025. URL https://arxiv.org/abs/2403.12945.
- [19] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https: //arxiv.org/abs/2306.03310.
- [20] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe HansenEstruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.
- [21] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [22] Wanwei He, Yinpei Dai, Yinhe Zheng, Yuchuan Wu, Zheng Cao, Dermot Liu, Peng Jiang, Min Yang, Fei Huang, Luo Si, Jian Sun, and Yongbin Li. Galaxy: A generative pre-trained model

- for task-oriented dialog with semi-supervised learning and explicit policy injection, 2022. URL https://arxiv.org/abs/2111.14592.
- [23] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.
- [24] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila

- Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.
- [25] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024.
- [26] Sangoh Lee, Sangwoo Mo, and Wook-Shin Han. Bring my cup! personalizing vision-languageaction models with visual attentive prompting. arXiv preprint arXiv:2512.20014, 2025.
- [27] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D’Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jingwei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020.
- [28] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning, 2025. URL https://arxiv.org/ abs/2407.08693.
- [29] William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning, 2025. URL https: //arxiv.org/abs/2505.08243.
- [30] Soroush Nasiriany, Sean Kirmani, Tianli Ding, Laura Smith, Yuke Zhu, Danny Driess, Dorsa Sadigh, and Ted Xiao. Rt-affordance: Affordances are versatile intermediate representations for robot manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8249–8257. IEEE, 2025.
- [31] Jiayuan Gu, Sean Kirmani, Paul Wohlhart, Yao Lu, Montserrat Gonzalez Arenas, Kanishka Rao, Wenhao Yu, Chuyuan Fu, Keerthana Gopalakrishnan, Zhuo Xu, et al. Rt-trajectory: Robotic task generalization via hindsight trajectory sketches. arXiv preprint arXiv:2311.01977, 2023.
- [32] Jinming Li, Yichen Zhu, Zhibin Tang, Junjie Wen, Minjie Zhu, Xiaoyu Liu, Chengmeng Li, Ran Cheng, Yaxin Peng, Yan Peng, et al. Coa-vla: Improving vision-language-action models via visual-textual chain-of-affordance. arXiv preprint arXiv:2412.20451, 2024.
- [33] Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memmel, Raymond Yu, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, et al. Hamster: Hierarchical action models for open-world robot manipulation. arXiv preprint arXiv:2502.05485, 2025.

- [34] Yingbo Tang, Shuaike Zhang, Xiaoshuai Hao, Pengwei Wang, Jianlong Wu, Zhongyuan Wang, and Shanghang Zhang. Affordgrasp: In-context affordance reasoning for open-vocabulary task-oriented grasping in clutter, 2025. URL https://arxiv.org/abs/2503.00778.
- [35] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. URL https://arxiv.org/abs/2204.01691.
- [36] Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Zhang Chen, Tianrui Guan, Fanlian Zeng, Ka Nam Lui, Yuyao Ye, Yitao Liang, et al. Dexgraspvla: A vision-language-action framework towards general dexterous grasping. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 18836–18844, 2026.
- [37] Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan Wang, Tai Wang, Jiangmiao Pang, and Zhou Zhao. Roboground: Robotic manipulation with grounded visionlanguage priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22540–22550, 2025.
- [38] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023.
- [39] Hang Yu, Juntu Zhao, Yufeng Liu, Kaiyu Li, Cheng Ma, Di Zhang, Yingdong Hu, Guang Chen, Junyuan Xie, Junliang Guo, et al. Point what you mean: Visually grounded instruction policy. arXiv preprint arXiv:2512.18933, 2025.
- [40] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025. URL https://arxiv.org/abs/2511.16719.
- [41] Yifan Zhong, Xuchuan Huang, Ruochong Li, Ceyao Zhang, Zhang Chen, Tianrui Guan, Fanlian Zeng, Ka Num Lui, Yuyao Ye, Yitao Liang, et al. Dexgraspvla: A vision-language-action framework towards general dexterous grasping. arXiv preprint arXiv:2502.20900, 2025.
- [42] StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.
- [43] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024.
- [44] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.
- [45] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.
- [46] Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, and Kai Chen. Bayesianvla: Bayesian decomposition of vision language action models via latent action queries. arXiv preprint arXiv:2601.15197, 2026.

- [47] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.
- [48] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.
- [49] Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. 2025.
- [50] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the computer vision and pattern recognition conference, pages 14203–14214, 2025.
- [51] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.
- [52] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.
- [53] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. arXiv preprint arXiv:2512.06963, 2025.

### A Limitations

Our interface currently utilizes 2D bounding boxes and crosshair coordinates as spatial anchors. While these primitives are sufficient for a wide range of robotic tasks, more complex manipulations, such as threading a needle or handling deformable objects, might benefit from richer visual prompts, such as 3D Gaussian splats or contact-point heatmaps. Exploring the optimal balance between prompt simplicity and information density is a promising direction for the community.

### B Compute Resources

All experiments were conducted on a server equipped with eight NVIDIA H200 GPUs, each with 140 GB of memory. Real world inference is conducted on a single NVIDIA 4090 with 24 GB memory. Training on the full dataset took approximately 46 hours.

### C Inference Latency

We further conduct experiments on system latency. We tested with Qwen3-VL-8B-Instruct as the VLM planner with OpenRouter service and locally host SAM3 server. We achieves a mean latency of 0.78s per reasoning call, while the action policy infers each action chunk in 0.108s on average. Meanwhile, SAM3 segmentation runs at 0.36s on average, remaining substantially faster than semantic VLM planning and compatible with asynchronous prompt updates.

### D Additional Demonstration to Tool-Use Manipulation Tasks

Tool-use tasks are inherently long-horizon and stage-dependent. To support the claim that our method can be applied beyound pick-and-place tasks, we conduct a study on tool-use tasks. We design a scoop bean task that requires the robot arm to first pick up the spoon, scoop on one bowl and then pour it on another bowl. Given each episode, we first detect subtask boundaries from end-effector motion. Unlike gripper-state heuristics, we use z-axis rise events as transition cues, which is crucial in tasks where the gripper remains mostly closed (e.g., spoon manipulation). We collected in total 106 episodes for training and trained both the baseline model and our methods for 50k steps with batch size 32.

Same as previous setting, at the first frame and each detected boundary, a VLM receives a reference frame and the current frame, then predicts whether to continue or proceed to the next subtask, together with target object and (if needed) target location. These semantic targets are subsequently grounded by a segmentation model (SAM), producing frame-level masks/boxes across the episode. The resulting annotations are saved as compact per-episode files and injected into training via a visual-prompt dataset: the policy conditions on the current observation augmented with object/location prompts. This creates a closed loop from high-level task progress reasoning to pixel-level action-relevant grounding. During experiment, we noticed the baseline struggles for moving towards the right bowl for pouring, while ours explicitly clue the model to proceed with visual prompts. Result are shown in Table 4. An illustration on the visual prompt are shown in Fig. 6.

Table 4: Comparison of Scoop Beans Performance Models Success Rate (%)

QwenOFT 30 Ours 50

### E Extended Analyze on Experiments

Full Per Task Results on RoboCasa. Table 12 presents the per-task success rates for the RoboCasa benchmark. This detailed view expands upon the main study, highlighting how our method performs across diverse tabletop manipulation tasks.

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

pick up the spoon, scoop a small bean from the left bowl with the spoon, and pour the bean into the right bowl

Figure 6: Inference visualization on real-world tasks.

Robotic Waste-Sorting Categorization. In Table 9, we further analyze the performance for each boxes: (i) In the Green Box (Recyclable), QwenOFT struggled with a red shoe (7/10), suggesting a reliance on color heuristics. Our model remained robust (9/10), demonstrating category grounding independent of surface appearance. (ii) For the Red Box (Kitchen Waste), our model maintained 85% accuracy on novel OOD objects (pear, carrot) compared to 75% for the baseline, reflecting superior semantic abstraction. (iii) The Black Box (Other Waste) highlighted the largest gap. On scrambled Rubik’s cubes, which share semantic identity with the training set but differ visually, QwenOFT’s performance collapsed (3/10) due to imprecise grasping and pattern overfitting. Our method achieved 9/10 by strictly following visual prompts and grounding the object to the correct placement location, even for challenging OOD items like sponge (8/10 vs. 5/10), where the baseline appears not knowing which box to place the object.

Object Reference by Attribute. In Table 10, we further analyze the performance for each configuration: (i) ID Colors: our method consistently surpasses baseline across all training colors. Failures in baseline often involve imprecise grasp positioning or failing to move above the target-colored egg, indicating weaker visuomotor alignment for attribute-conditioned manipulations. (ii) OOD Color: The baseline frequently confuses novel colors with visually similar training colors or defaults to the egg closest to the gripper while ignoring the color condition. In contrast, our model consistently moves accurately to the target, achieving 9/12 for both colors. This suggests better disentanglement between linguistic attributes and spatial proximity bias. (iii) OOD Position: For eggs placed at unseen positions, the baseline shows a bias toward previously demonstrated columns, indicating spatial memorization. Our model achieves 9/12 for both colored eggs, demonstrating spatial generalization and more stable grasping under positional shift.

Location reference with spatial grounding. In Table 11, we further analyze the performance for each configuration: (i) In-Domain Positions: Our model achieves near-perfect scores on most ID coordinates, indicating accurate row–column parsing. Conversely, QwenOFT shows high variance; for instance, it drops to 1/5 at L3C3 (vs. our 4.5/5), struggling to resolve spatial references when

vertical and horizontal axes must be composed jointly. (ii) Out-of-Distribution Positions: OOD tasks require the model to follow target masks at coordinates never jointly observed during training. Our method maintains strong performance at L3C2 (5/5) and L4C3 (4/5), successfully extending the visual-prompt-following mechanism to new locations. QwenOFT remains inconsistent (1.75/5 to 3.5/5), often failing to generalize the underlying grid geometry to novel index combinations.

### F Extended Ablation Results

In this section, we provide a comprehensive breakdown of our experimental evaluations to further validate the design choices of VP-VLA.

Full Ablation Results on RoboCasa. Table 5 presents the per-task success rates for the RoboCasa benchmark. This detailed view expands upon the main ablation study, highlighting how our design choice performs across diverse tabletop manipulation tasks.

Impact of Reasoning Decomposition. We conducted additional ablation studies in the SimplerEnv environment to assess the necessity of temporal decomposition. Our results in Table 6 indicate that models lacking decomposition, where both the interaction anchor (crosshair) and spatial constraint (bounding box) are rendered simultaneously, suffer a degradation in success rate, especially the “Put Eggplant in Yellow Basket” task. We hypothesize that concurrent prompts introduce visual noise that confuses the policy’s attention. Furthermore, prompting without decomposition would fail to generalize to complex, multi-step sequences, such as the “PnP * to * Close” task suite, where the system must distinguish between sequential objectives.

### G System 2 Planner Prompt Details

The prompt templates for the System 2 planner are provided in Table 7 and Table 8. The planner first decomposes a high-level language instruction into an ordered list of atomic subtasks. During execution, the planner is selectively invoked by transition events to re-evaluate the current stage and determine whether to proceed to the subsequent subtask. Upon triggering, the planner identifies the corresponding target object and destination names within the scene. Empirically, we find that incorporating the current gripper state into the planner’s input context enhances its decision-making accuracy regarding task completion.

### H VLA Experimental Demonstrations

To demonstrate the practical capabilities of VP-VLA, we visualize execution trajectories across simulation benchmarks and real-world deployments.

SimplerEnv Demonstrations. Fig. 7 illustrates the model’s performance across four distinct tasks. Given that SimplerEnv evaluates models in out-of-distribution (OOD) settings, success requires both high-precision localization and robust control. As shown, VP-VLA accurately decompose the manipulation process into discrete stages and effectively completes each phase.

RoboCasa Demonstrations. Fig. 8 showcases the model’s performance on the realistic RoboCasa benchmark, which involves complex, multi-step interactions in cluttered kitchen environments. These visualizations highlight VP-VLA’s ability to generalize to humanoid embodiments using egocentric observations while managing long-horizon activities.

Real-World Demonstrations. Fig. 9 illustrates the VP-VLA inference process in a physical environment. To achieve a successful rollout, the model must maintain precise reasoning and grounding in novel objects and unseen placement setting. The demonstration confirms that our method facilitates smooth object manipulation and placement in both in-distribution and OOD scenarios.

Table 5: Per-task result for the ablation on Robocasa.

Task groundingw/o w/groundingall frame pointw/ w/overlaydirect (Full)Ours PnP Bottle To Cabinet Close 52.0 34.0 40.0 46.0 54.0 PnP Can To Drawer Close 70.0 66.0 42.0 78.0 72.0 PnP Cup To Drawer Close 48.0 54.0 32.0 48.0 44.0 PnP Milk To Microwave Close 52.0 40.0 48.0 48.0 74.0 PnP Potato To Microwave Close 42.0 46.0 30.0 26.0 34.0 PnP Wine To Cabinet Close 34.0 38.0 40.0 40.0 48.0 PnP * to * Close (Avg) 49.7 46.3 38.7 47.7 54.3 PnP Novel From Cuttingboard To Basket 50.0 60.0 58.0 52.0 66.0 PnP Novel From Cuttingboard To Cardboardbox 58.0 50.0 44.0 58.0 54.0 PnP Novel From Cuttingboard To Pan 64.0 62.0 52.0 60.0 74.0 PnP Novel From Cuttingboard To Pot 64.0 56.0 48.0 54.0 54.0 PnP Novel From Cuttingboard To Tieredbasket 36.0 40.0 42.0 46.0 56.0 PnP Novel From Cuttingboard To * (Avg) 54.4 53.6 48.8 54.0 60.8 PnP Novel From Placemat To Basket 56.0 36.0 44.0 48.0 48.0 PnP Novel From Placemat To Bowl 48.0 46.0 46.0 60.0 74.0 PnP Novel From Placemat To Plate 56.0 66.0 70.0 60.0 70.0 PnP Novel From Placemat To Tieredshelf 24.0 32.0 24.0 18.0 26.0 PnP Novel From Placemat To * (Avg) 46.0 45.0 46.0 46.5 54.5 PnP Novel From Tray To Cardboardbox 40.0 46.0 58.0 46.0 44.0 PnP Novel From Tray To Plate 52.0 56.0 56.0 66.0 66.0 PnP Novel From Tray To Pot 56.0 56.0 62.0 58.0 38.0 PnP Novel From Tray To Tieredbasket 46.0 46.0 42.0 58.0 58.0 PnP Novel From Tray To Tieredshelf 22.0 26.0 30.0 22.0 24.0 PnP Novel From Tray To * (Avg) 43.2 46.0 49.6 50.0 46.0 PnP Novel From Plate To Bowl 58.0 70.0 60.0 54.0 52.0 PnP Novel From Plate To Cardboardbox 38.0 38.0 46.0 38.0 44.0 PnP Novel From Plate To Pan 54.0 50.0 58.0 60.0 56.0 PnP Novel From Plate To Plate 66.0 74.0 62.0 74.0 62.0 PnP Novel From Plate To * (Avg) 54.0 58.0 56.5 56.5 53.5 Average 49.4 49.5 47.3 50.8 53.8

Table 6: Ablation result on applying task decomposition on SimplerEnv.

Put Spoon on Towel

Put Carrot on Plate

Stack Green Block on Yellow Block

Put Eggplant in Yellow Basket

Method

Average

Ours w/o decomposition 70.8 62.5 16.7 79.2 57.3 Ours + Qwen3VL 66.7 50.0 20.8 95.8 58.3

- Table 7: Prompt structure used for task decomposition. The VLM planner decomposes a high-level task description into atomic robotic manipulation subtasks.

Section Prompt Content

Task Description {task_description} Instruction Decompose this robotic manipulation task into sequential atomic subtasks. Subtask Types

- • Pick action (e.g., “pick up the cup”)
- • Place action (e.g., “place the cup on the table”)
- • Manipulation action (e.g., “close the drawer”, “open the microwave”)

Rules

- • Keep subtask descriptions short and clear.
- • Preserve the object and location names from the original task description.

Output Format {

"subtasks": ["subtask 1", "subtask 2", ...] }

- Table 8: Prompt structure used for subtask completion detection. The VLM planner determines whether the current subtask has been completed using visual evidence and gripper state information.

Section Prompt Content

Task {task_description} Current Subtask {current_subtask} Next Subtask {next_subtask} or “None (this is the last subtask)” Visual Context • Image A: frame when the current subtask started

• Image B: current frame Use visual differences between Image A and Image B to judge whether the current subtask has been completed.

Gripper State LEFT_GRIPPER = {state} RIGHT_GRIPPER = {state} Optional notification if gripper state changes between frames: GRIPPER STATE CHANGE DETECTED: LEFT_GRIPPER: {prev_left} → {curr_left} RIGHT_GRIPPER: {prev_right} → {curr_right} This change may indicate that the current subtask has been completed. Consider both the visual evidence and this gripper state change when deciding whether to proceed to the next subtask.

Decision Requirement Decide whether to:

- • CONTINUE the current subtask "{current_subtask}" (if it’s still in progress)
- • PROCEED to the next subtask (if the current subtask is completed) Proceed only if the current subtask appears completed based on visual evidence and gripper state.

Output Format Rules • target_object: noun only (e.g., bottle, drawer, cabinet)

- • target_location: noun phrase if applicable (e.g., cabinet shelf, countertop surface)
- • PICK: specify only target_object
- • PLACE: specify target_object and target_location
- • OTHER actions (open/close/push): specify target_object

Output Format {

"reasoning": "...", "decision": "continue" or "proceed", "target_object": "<noun>", "target_location": "<location or null>" }

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Put the carrot on the plate

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

[Figure 101]

[Figure 102]

Put the eggplant into the sink

[Figure 103]

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

Put the spoon on the tablecloth

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Stack the green cube on the yellow cube

Figure 7: Inference visualization on the SimplerEnv simulation environment.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

pick up the bottled water, place it into the cabinet and close the cabinet

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

pick up the cup, place it into the drawer and close the drawer

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

pick the bell pepper from the cutting board and place it in the basket

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

pick the croissant from the cutting board and place it in the pan

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

pick the tomato from the cutting board and place it in the pot

Figure 8: Inference visualization on the RoboCasa Tabletop simulation environment.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

pick the toilet paper and place it in the black box

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

pick the potato and place it in the red box

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

pick the beige cup and place it in the green box

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

pick up the egg and place it at line 3 column 2

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

pick up the pink egg

Figure 9: Inference visualization on real-world tasks.

- Table 9: Categorization results (10k steps). Successes out of 10 trials. Bold indicates best performance.

Object Ours QwenOFT Object Ours QwenOFT Green Box Red Box

Bottle (ID) 7 7 Banana (ID) 8 6 Beige cup (ID) 9 8 Orange (ID) 9 9 Black shoe (ID) 10 9 Corn (ID) 10 9 Toy chicken (ID) 8 7 Potato (ID) 10 10 Blue cup (OOD) 8 8 Pear (OOD) 9 8 Red shoe (OOD) 9 7 Carrot (OOD) 8 7

Black Box Summary Statistics

Summary ID/OOD ID/OOD

Towel (ID) 10 10 Green Avg 34/17 31/15 Light bulb (ID) 6 4 Red Avg 37/17 34/15 Rubik’s cube (ID) 10 9 Black Avg 34/17 31/8

Toilet paper (ID) 8 8 Total ID 87.5% 80.0% Scrambled cube (OOD) 9 3 Total OOD 85.0% 63.3% Sponge (OOD) 8 5

- Table 10: Pick colored egg results. Performance (raw score/12) of our method vs. QwenOFT. We evaluate across In-Domain (ID) and two OOD scenarios (Color and Position change).

In-Domain OOD (Color) OOD (Pos.) Color Ours QwenOFT Ours QwenOFT Ours QwenOFT

Blue 10 7 Pink 8 6 Purple 9 4 Blue 9 6 Red 10 8 Green 9 3 Red 9 7 Yellow 9 7

Avg (%) 77.1 58.3 75.0 29.2 75.0 54.2

- Table 11: Egg carton placement results. Successes are reported out of 5 trials. LX CY denotes the position at Line X, Column Y . We report both In-Domain (ID) and Out-of-Distribution (OOD) performance.

In-Domain (ID) Out-of-Distribution (OOD) Pos. Ours QwenOFT Pos. Ours QwenOFT

L1 C2 5.00 4.25 L1 C3 2.50 1.75 L1 C4 5.00 4.50 L2 C1 2.25 2.50 L2 C4 5.00 2.25 L3 C2 5.00 3.25 L2 C2 4.00 4.25 L4 C3 4.00 3.50

- L3 C1 4.50 4.50 Summary Statistics L3 C3 4.50 1.00 Group Ours QwenOFT

- L4 C2 4.50 3.00 ID Avg. 91.3% 70.6% L4 C4 4.00 4.50 OOD Avg. 68.8% 55.0%

- Table 12: Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. We highlight the best result in bold and the second-best results with underline.

Task Isaac-GR00TN1.5 Isaac-GR00TN1.6 QwenGR00T+Qwen3VL +Qwen3VLQwenPI +Qwen3VLQwenOFT QwenFAST+Qwen3VL +Qwen3VLOurs

PnP Bottle To Cabinet Close 54.0 51.5 46.0 26.0 30.0 38.0 54.0 PnP Can To Drawer Close 50.0 13.0 80.0 62.0 76.0 44.0 72.0 PnP Cup To Drawer Close 38.0 8.5 54.0 42.0 44.0 56.0 44.0 PnP Milk To Microwave Close 60.0 14.0 48.0 50.0 44.0 44.0 74.0 PnP Potato To Microwave Close 32.0 41.5 28.0 42.0 32.0 14.0 34.0 PnP Wine To Cabinet Close 38.0 16.5 46.0 32.0 36.0 14.0 48.0

PnP * to * Close (Avg) 45.3 24.2 50.3 42.3 43.7 35.0 54.3 PnP Novel From Cuttingboard To Basket 38.0 58.0 48.0 40.0 50.0 54.0 66.0 PnP Novel From Cuttingboard To Cardboardbox 46.0 46.5 40.0 46.0 40.0 42.0 54.0 PnP Novel From Cuttingboard To Pan 58.0 68.5 68.0 60.0 70.0 58.0 74.0 PnP Novel From Cuttingboard To Pot 62.0 65.0 52.0 40.0 54.0 58.0 54.0 PnP Novel From Cuttingboard To Tieredbasket 28.0 46.5 56.0 44.0 38.0 40.0 56.0 PnP Novel From Cuttingboard To * (Avg) 46.4 56.9 52.8 46.0 50.4 50.4 60.8 PnP Novel From Placemat To Basket 30.0 58.5 42.0 44.0 32.0 36.0 48.0 PnP Novel From Placemat To Bowl 60.0 57.5 44.0 52.0 58.0 38.0 74.0 PnP Novel From Placemat To Plate 56.0 63.0 48.0 50.0 52.0 42.0 70.0 PnP Novel From Placemat To Tieredshelf 36.0 28.5 18.0 28.0 24.0 18.0 26.0 PnP Novel From Placemat To * (Avg) 45.5 51.9 38.0 43.5 41.5 33.5 54.5 PnP Novel From Tray To Cardboardbox 52.0 51.5 38.0 34.0 44.0 28.0 44.0 PnP Novel From Tray To Plate 48.0 71.0 56.0 64.0 56.0 34.0 66.0 PnP Novel From Tray To Pot 60.0 64.5 50.0 44.0 62.0 46.0 38.0 PnP Novel From Tray To Tieredbasket 52.0 57.0 36.0 50.0 54.0 36.0 58.0 PnP Novel From Tray To Tieredshelf 32.0 31.5 16.0 28.0 30.0 16.0 24.0 PnP Novel From Tray To * (Avg) 48.8 55.1 39.2 44.0 49.2 32.0 46.0 PnP Novel From Plate To Bowl 58.0 57.0 60.0 52.0 60.0 52.0 52.0 PnP Novel From Plate To Cardboardbox 44.0 43.5 50.0 40.0 50.0 30.0 44.0 PnP Novel From Plate To Pan 60.0 51.0 54.0 36.0 66.0 48.0 56.0 PnP Novel From Plate To Plate 64.0 78.7 70.0 48.0 68.0 50.0 62.0 PnP Novel From Plate To * (Avg) 56.5 57.6 58.5 44.0 61.0 45.0 53.5 Average 48.2 47.6 47.8 43.9 48.8 39.0 53.8

