# arXiv:2512.23162v4[cs.RO]11Mar2026

[Figure 1]

2026-3-12

## Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling

###### Yufan He1,‡ Pengfei Guo1,‡ Mengya Xu2,‡ Zhaoshuo Li1 Andriy Myronenko1 Dillan Imans3 Bingjie Liu4 Dongren Yang4 Mingxue Gu1 Yongnan Ji1 Yueming Jin5 Ren Zhao6 Baiyong Shen6 Daguang Xu 1

1NVIDIA 2The Chinese University of Hong Kong 3Sung Kyun Kwan University 4 Wenzhou Medical University 5 National University of Singapore 6 Ruijin Hospital ‡Equal First Author

Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While largescale vision–language–action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video-action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from Cosmos-H-Surgical, a world model designed for surgical physical AI. We curated the Surgical Action-Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built Cosmos-H-Surgical based on the most advanced physical AI world model and SATA. It’s able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse-dynamics model to infer pseudo-kinematics from synthetic surgical videos, producing synthetic paired video–action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data-efficient surgical robot policies.

Code: https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical Model: https://huggingface.co/nvidia/Cosmos-H-Surgical

[Figure 2]

Figure 1 | We curate SATA dataset with surgical videos and detailed text annotations for physical AI. A powerful world model (Cosmos-H-Surgical) is built using Cosmos2.5 [1] and SATA, which is able to generate high quality, generalizable videos for surgical robots. We are also the first to illustrate the efficacy of surgical world modeling for autonomous surgical robots.

### 1. Introduction

Autonomous robotic surgery promises to enhance precision, reduce surgeon fatigue, and scale complex procedures. Yet, despite extensive research, training robust robotic policies for surgical manipulation remains exceptionally challenging. A major bottleneck

is the lack of large, diverse datasets that include both high-fidelity visual observations (e.g., endoscopic video) and synchronized robot kinematics or control commands. Collecting such paired demonstrations is prohibitively expensive, constrained by operating room access, patient safety, and regulatory hurdles.

In parallel, the robotics community has seen

© 2026 NVIDIA. All rights reserved.

[Figure 3]

- Figure 2 | The overall workflow. The Cosmos-H-Surgical model is first pretrained with large scale surgical videos with text annotations, based on Cosmos 2.5 [1]. For downstream task with specific robot type and task, we finetune Cosmos-H-Surgical and train the inverse dynamic model (IDM) for the specific embodiment. In step 3 we generate synthetic video rollouts from Cosmos-H-Surgical and get pseudo kinematics from the IDM. We use both real data and synthetic data to train the surgical VLA model.

tremendous progress in large vision–language–action (VLA) models. Models such as RT-2 [2], OpenVLA [3], and GR00T [4] has demonstrated the potential of foundation models to generalize across diverse robotic manipulation tasks. These models leverage large-scale multimodal datasets that couple visual observations, language descriptions, and robot actions, enabling rich world understanding and robust policy learning, yielding significantly improved generalization to novel objects and commands in non-surgical manipulation domains. Similarly, recent work on synthetic data generation such as DreamGen [5] shows that video world-models combined with inverse dynamics can produce synthetic paired video–action datasets that boost policy learning in general manipulation tasks.

However, extending such capabilities to surgical robotics remains a significant challenge. Unlike household or industrial domains, surgical robotics suffers from severe data scarcity, particularly the lack of large-scale datasets with synchronized visual and kinematic information. Surgical data collection is constrained by privacy regulations, ethical considerations, and the cost of robotic surgical systems, limiting the ability to train data-hungry models such as VLAs and imitation learning (IL) policies. Synthetic physics-based simulators [6, 7] attempt to fill the gap, but often suffer from a large visual and dynamic domain shift to real surgical systems and lacking soft body simulation, limiting policy transfer.

To address these limitations, we propose CosmosH-Surgical, a unified framework that leverages a surgical world model to enable scalable policy learning. Specifically, we curate and annotate the Surgical Action–Text Alignment (SATA) dataset with expert-labeled surgical video clips covering four core

surgery actions with detailed spatial, interaction, and anatomical context. Using SATA, we train a diffusion-based world model [1] capable of generating photorealistic, task-consistent surgical scenes. We then employ an inverse dynamics model (IDM) to infer pseudo-kinematics, producing synthetic paired video–action data that can be directly used for surgical VLA policy learning. We validate our approach on a surgical robotic platform performing needle pick-up and hand-over tasks—a fundamental and dexterous actions representative of real surgical manipulation. We use both real surgical demonstrations with kinematic supervision and synthetic videos labeled with pseudoactions to train the GR00T N1.5 VLA model [4]. Our results demonstrate that incorporating synthetic world model data leads to substantial improvements in policy performance with lower trajectory prediction error.

While prior works such as DreamGen [5] explored world-model-based learning from synthetic videos, these domains lack the unique visual and physical complexity of surgery, where specular tissue surfaces, endoscopic occlusion, and constrained tool motion present distinct modeling challenges. Recent efforts have begun addressing this gap within surgical contexts: GAS [8] applies world-model-based reinforcement learning for grasping, SurgWM [9] generates controllable surgical videos with dynamic prediction, and the Suturing World Model [10] forecasts tool trajectories for automated suturing. However, these approaches remain limited to narrow tasks or visual prediction without explicit integration of text grounding and kinematics. Cosmos-H-Surgical is the first to integrate large-scale text-aligned surgical video modeling with pseudo-kinematics generation for embodied policy learning and bridging the gap between unlabeled surgical videos and robot actions. Scalable data generation

without collecting in-vivo trajectories can dramatically acceleratesurgicalautonomywhile maintaining patient safety. Our framework addresses the core bottleneck of data scarcity in surgical robotics and opens a scalable path toward autonomous surgical skill acquisition.

In summary, our key contributions are:

- 1. We curate the Surgical Action–Text Alignment (SATA) dataset, a large-scale surgical video–text corpus comprising 2,447 expert-annotated video clips (over 300k frames) that capture fine-grained spatial relationships and tool–tissue interactions across 8 procedures, designed specifically to support the development of physical AI models.
- 2. We develop the first surgical world model that’s based upon state-of-the-art physical AI world models, and finetuned with SATA, demonstrating strong generalizability, high video quality, and realistic dynamics.
- 3. We are the first to connect surgical world models with robot learning by synthesizing video–action data using inverse dynamics models, achieving substantial performance improvements in surgical robot learning.

Our results highlight the potential of generative world modeling to complement real surgical data and pave the way for foundation models that enable scalable, autonomous, and safe surgical policy learning.

### 2. Related Work

VLA Models and Imitation Learning (IL). Largescale vision–language–action (VLA) models have recently emerged as a powerful paradigm for generalpurpose robotic policy learning. Recent open-source efforts such as OpenVLA [3], 𝜋0 [11], and GR00T N1 [4] have been trained on massive and diverse data sources, leveraging diverse robot embodiments and multi-task datasets and can robustly handle real-world variability. To train those VLAs, imitation learning (IL) remains the most direct and scalable approach, typically via behavior cloning (BC) [12]. However, BC suffers from covariate shift, producing rapidly worsening performance when the dataset is limited [13]. In robotics, several works aim to improve data efficiency using offline datasets [14] or synthetic augmentation [15]. However, despite these advances, current VLA models still rely heavily on imitation learning on large scale paired image–action datasets, which are scarce in specialized domains such as surgical robotics.

#### Surgical World Models and Video generation.

Learning compact models of the world has long been a goal of model-based reinforcement learning (MBRL).

World Models [16] demonstrated that compact latent dynamics models can simulate plausible trajectories for policy learning. Subsequent works such as PlaNet [17], Dreamer [18], and DreamerV3 [19] extended these ideas to high-dimensional visual environments, enabling agents to imagine future rollouts and train in latent space. In the surgical domain, video generation and world modeling are only beginning to emerge. Endora [20] integrates a spatiotemporal transformer with a latent diffusion backbone for endoscopic video synthesis. SurGen [21] proposes a text-guided diffusion model tailored to laparoscopic cholecystectomy. VISAGE [22] formulates future surgical video prediction conditioned on a single frame and an action scene graph, modeling tool–tissue interactions for temporally coherent generation. GAS [8] applies world-model-based reinforcement learning for robust grasping across diverse objects. SurgWM [9] introduces controllable surgical video generation and interactive dynamics prediction from visual inputs alone. Suturing World Model [10] learns taskspecific dynamics to anticipate tool trajectories during automated suturing. Cosmos-Surg-dVRK [23] focuses on policy evaluation using action conditioned video generation model. While these studies mark important progress, they are limited to single-task or objectspecific scenarios, and rely on narrowly scoped datasets lacking high-quality text–action alignment or procedural diversity, and many are not open-sourced, limiting reproducibility and broader impact. In contrast, our approach leverages the curated SATA dataset to train a surgical world model capable of generating photorealistic, task-consistent videos explicitly designed for physical AI and downstream robot policy learning.

Learning Policy Models from Videos. Learning policy models directly from video has become increasingly important due to the lack of datasets with kinematics. Ye [24] and Jang [5] propose a latent-action pretraining scheme using internet-scale videos without robot action labels to bootstrap visionlanguage-action models. Hu [25] train predictive visual representations via video diffusion models and embed an implicit inverse-dynamics policy conditioned on these representations. Bharadhwaj [26] use humanvideo generation to condition robot policies on novel scenarios, reducing reliance on robot-collected data. Li [27] present a unified video-action latent model that jointly handles forward/inverse dynamics, video generation, and action inference within one framework. Tian [28] close the loop by using inverse-dynamics models conditioned on predicted visual states for large-scale manipulation training. These developments directly inspire our approach, which uses IDM to connect world models with surgical robot learning.

Automated Surgical Robotics. Automation in surgical robotics have increasingly leveraged learning-

based approaches. Long [29] introduced a vision-based embodied intelligence framework that enables zeroshot sim-to-real transfer for a variety of laparoscopic assistive tasks using imitation learning, Kim [30] proposed the Surgical Robot Transformer (SRT), which addresses the da Vinci robot’s inaccurate kinematics by formulating actions in a hybrid-relative space, achieving high success rates on fundamental tasks. The same group later developed SRT-H [31] that enables step-level autonomy in complex procedures such as cholecystectomy. While these works demonstrate impressive progress in task autonomy, they also reveal a critical need for large-scale, diverse surgical demonstration datasets, which in surgery is uniquely challenging: obtaining large-scale paired visual–kinematic datasets requires specialized hardware and surgeon supervision.

### 3. Method

The overall workflow is shown in Fig. 2. The surgical world model is obtained by finetuning Cosmos-Predict2.5 [1] model on surgical videos with detailed annotations. Then we demonstrate its efficacy in downstream surgical robot tasks. Since the world model has not seen specific surgical robot embodiments, we finetune the world model on robots and task specific data. Meanwhile we build an inverse dynamic model (IDM) for this specific robot. The world model generates video rollouts and IDM model label those videos with pseudo action kinematics.

#### 3.1. Dataset Curation

Surgical Action–Text Alignment (SATA) Dataset. We introduce SATA, a large-scale surgical action–text alignment dataset comprising 2,447 expert-annotated video clips (over 300k frames) collected across 8 different surgery types. Each clip captures one of four fundamental actions: needle grasping (689), needle puncture (989), suture pulling (475), and knotting (294), which provides diverse visual and procedural coverage. SATA is curated by aggregating and re-annotating videos from credentialed YouTube surgical channels [32] and publicly available datasets, including GraSP [33], SAR-RARP50 [34], Multiypass140 [35], SurgicalActions160 [36], AutoLaparo [37], and HeiCo [38]. Unlike surgical VLM datasets, such as SurgVLM-DB [39], which focus primarily on semantic reasoning and instruction following, SATA is specifically designed for physical AI: its fine-grained action labels and detailed text descriptions capture precise tool–tissue interactions and spatial relationships needed for training world models.

The four action categories are defined by decomposing the suturing procedure into its fundamental steps

and annotated according to the following criteria:

- • Needle grasping: Approaching and securing the needle with a smooth, controlled trajectory, emphasizing the dynamic “go-to-grasp” motion rather than the subsequent static hold.
- • Needle puncture: Inserting the needle into tissue with precise control over its entry angle and depth.
- • Suture pulling: Drawing the suture thread through tissue after puncture completion, typically by pulling on the needle or thread.
- • Knotting: Looping and tightening the suture material to secure the tissue layers together.

Each clip is paired with a rich textual description detailing (i) spatial relationships between surgical instruments, (ii) the anatomical structure being manipulated, and (iii) the description of instrument–tissue interaction. For example: “The left needle driver punctures the right side of the patient’s dorsal venous complex.” Additional statistics and dataset breakdowns for SATA are provided in the supplementary materials.

Real World Trajectories. For real-world validation, we aim to demonstrate that synthetic videos generated by the world model can enhance autonomous surgical robot policy learning. Due to the high cost and regulatory constraints of in-vivo experiments, we evaluate our method using the “Needle Pickup and Hand-Over” task on rubber pad. The experiments are conducted on a commercial endoscopic surgical system (robot and manufacturer anonymized), which consists of a stereo endoscope and two articulated robotic forceps (left and right arms). During the task, the left arm grasps the needle tip and hand it over to the right arm.

We collected a total of 60 successful humanteleoperated demonstrations for training and testing. Each episode includes synchronized endoscopic video (average length: 217 frames) and corresponding action kinematics. In addition to these task-specific demonstrations, we also utilized 66 out-of-domain episodes (around 60k action frames pairs) depicting general robot movements unrelated to the needle pickup task. These data are used to pretrain a foundational inverse dynamics model (IDM) for the surgical robot, providing transferable motion understanding across tasks.

The robot states are the same as the action kinematics with the same dimension and values. The action kinematics at each timestep is represented as a 20-dimensional continuous vector:

a𝑡=[p𝐿,r𝐿,𝑔𝐿,p𝑅,r𝑅,𝑔𝑅],

where each term encodes the motion of the left (L) and right (R) instruments relative to the endoscope frame.

Specifically,

representations, we can efficiently transfer general video dynamics knowledge to the endoscopic surgical domain, reducing the amount of domain-specific data required while preserving temporal coherence and realism. To adapt the pretrained model to the surgical domain, we fine-tune it on the curated SATA dataset and real-world surgical trajectories described in previous section, which enables the model to capture the unique visual dynamics of robotic endoscopic scenes, such as instrument–tissue interaction, limited fieldof-view motion, and constrained articulation patterns. Unlike policy models, our model conditions only on the first observed frame 𝐼0 and predicts future trajectories, capturing the temporal evolution of the surgical scene.

IDM Actions

Inverse Dynamic Models

Action decoder

Diffusion Transformer

SigLIP-2

Action encoder

[Figure 4]

[Figure 5]

Frame i Frame i + T

Noised actions

K step

predicted actions

GR00T Vision Language Action Model

Diffusion Transformer

Vision language Model State Encoder Action Encoder

“The Ieft forcep picks up a needle and hands over to the right forcep”

[Figure 6]

Robot State Noised actions

K step

We adopt Low-Rank Adaptation (LoRA) [41] to efficiently specialize Cosmos-Predict2.5 [1] for the surgical domain while preserving its general video modeling capabilities. LoRA modules are inserted into the transformer’s attention and feed-forward layers, enabling parameter-efficient finetuning with minimal forgetting. During adaptation, the model learns to predict future latent video frames conditioned on an initial observation and text prompt. Given an initial frame 𝐼0, the world model produces a rollout 𝐼^1:𝑇 =𝒲𝜃(𝐼0), where 𝒲𝜃 denotes Cosmos-Predict2.5 [1] augmented with LoRA adapters. A spatiotemporal encoder extracts features from 𝐼0, a transformer-based latent dynamics module models temporal evolution, and a decoder reconstructs the predicted frames. We adopt the Flow Matching (FM) formulation [42] to train the surgical video world model due to its conceptual simplicity and practical effectiveness. More training detail of CosmosH-Surgical can be found in the supplementary material.

- Figure 3 | The model architecture for inverse dynamic models (IDM) and the vision language action foundation model (GR00T N1.5). They share similar architectures but IDM does not use text prompt nor robot state.

- • p𝐿=[𝑥𝐿,𝑦𝐿,𝑧𝐿]∈R3 represents the translational offset (cartesian coordinates) of the left forcep tip with respect to the endoscope frame (represented by meters).
- • r𝐿 = [𝑟𝐿1,...,𝑟𝐿6] ∈ R6 denotes the 6D rotation representation of the left instrument’s end-effector orientation. The 6D rotation formulation [40] is used to avoid discontinuities and ensure smooth interpolation in SO(3) by dropping the last column of the rotation matrix.
- • 𝑔𝐿 ∈ R indicates the gripper jaw opening in radians.

- 3.3. Inverse Dynamic Models and Policy Models

We follow the DreamGen IDM design [5, 43] and use GR00T N1.5 [4] as the policy model. The model architectures are shown in Fig. 3. These two models predict robotic actions with DIT [44] and flow matching heads [42]. The major difference is that IDM’s inputs are two frames from the same video (T = 16 frames apart), and the model predicts the robot actions for every frame between these two input frames, while the GR00T policy model takes the current frame and text prompt, together with the robot state to predict the actions for future 16 frames.

- 4. Experiments

The same definitions apply for the right forcep with p𝑅, r𝑅, and 𝑔𝑅. This yields a total of 20 control dimensions. All translation and rotation components are expressed relative to the endoscope’s coordinate frame to ensure view-consistent control.

#### 3.2. Surgical World Model

In this work, we adopt Cosmos-Predict2.5 [1], a large-scale video world model pretrained on diverse robotic and embodied datasets, as our base model and adapt it to the surgical domain. Cosmos-Predict2.5 [1] leverages diffusion-based latent video prediction with transformer backbones to simulate high-fidelity spatiotemporal dynamics. Its large-scale pretraining on heterogeneous robotic and human teleoperation videos provides strong priors for object interactions, tool motion, and scene dynamics, making it particularly well-suited for domains with limited labeled data, such as surgical robotics. By leveraging these pretrained

4.1. Surgical World Model Evaluation

We first evaluate the proposed Comos-H-Surgical world model on (i) video generation quality using the

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>Same conditioning frame<br><br>Zero-shot<br><br>Prompt: left needle driver passes needle to right needle driver.<br><br>Action-category Cosmos-H-Surgical<br><br>|
|---|

- Figure 4 | Qualitative comparison of three variants of Cosmos-Predict2.5 [1] on the SATA dataset. Red arrows highlight incorrect surgical tools or actions in the generated frames.

curated SATA dataset of internet surgical videos and (ii) few-shot adaptation to collected real trajectories. The goal is to assess both perceptual fidelity and transferability to downstream policy learning.

Video Generation on SATA. We evaluate three variants of Cosmos-Predict2.5 [1]: (1) Zero-Shot, using the base model evaluated without any domain adaptation; (2) Action-Category, finetuned using coarse, category-level captions where all videos within the same action category share an identical prompt; and (3) Cosmos-H-Surgical, finetuned on SATA’s finegrained, expert-curated textual descriptions capturing detailed tool–tissue interactions and spatial relations. We report Fréchet Video Distance (FVD) [45] and follow [1] to report the three representative VBench [46] metrics. As shown in Table 1, Cosmos-H-Surgical trained with curated prompts achieves the lowest FVD and highest alignment scores, indicating substantial gains in perceptual realism and semantic coherence over zero-shot and coarse category-level prompt baselines. Qualitative results in Figure 4 further highlight these differences in a challenging scenario where the initial frame contains no visible surgical tools. The Zero-Shot model hallucinates an incorrect instrument due to limited domain priors, and the Action-Category model initiates a wrong action (tissue puncture). In contrast, the Cosmos-H-Surgical correctly follows the textual instruction and completes the intended needle grasping motion with consistent behavior.

New Behavior Generalization. To evaluate the model’s capacity to generalize to diverse and semantically consistent actions, we test its response to varying textual prompts describing distinct surgical behaviors.

Table 1|Quantitative evaluation of surgical video generation on the curated SATA dataset. We report FVD and VBench metrics: dynamic degree (DD), imaging quality (IQ), and overall consistency (OC).

|Method|FVD ↓ DD ↑ IQ ↑ OC ↑<br><br>|
|---|---|
|Zero-shot Action-category Cosmos-H-Surgical|175.4 26.9 48.7 18.0 143.0 26.5 49.0 18.1 106.5 62.4 49.3 21.5|

Figure 5 presents representative video rollouts generated from the same conditioning frame under four distinct textual prompts: one-time/two-time/threetime needle handover, and needle puncture. The multi-step handover cases are particularly noteworthy, because during data curation all multi-time handovers are decomposed into single handover segments. Therefore, the two- and three-time handover sequences represent novel compositions not explicitly observed during training. The model accurately follows each instruction, producing coherent sequences that reflect increasing motion complexity while preserving visual realism. These results demonstrate strong text–video alignment and show that the Cosmos-H-Surgical can recombine learned primitives to generate anatomically plausible and temporally consistent surgical behaviors from prompt-level conditioning.

Human Expert Evaluation. While quantitative metrics such as FVD and VBench capture perceptual and temporal quality, they fail to fully reflect the clinical realism required in surgical video generation. To bridge this gap, we conducted a human evaluation study to assess the anatomical plausibility, instrument behavior, and semantic faithfulness of generated videos from a surgical perspective. Three surgical experts (one surgeon supervising two residents) independently evaluated 50 video samples generated by different world model variants using a three-level clinical quality rubric (scores from 1 to 3, higher is better) across the following criteria:

- • Text–Video Alignment: Assesses whether the generated scenes match the textual prompt.
- • Tool Consistency: Evaluates whether surgical instruments correspond to the prompt and remain physically consistent throughout the video sequence.
- • Anatomical Structure: Measures the plausibility of tissue and organ appearance and reactions to interaction.

A detailed description of the rating rubric for each score level is in the supplementary materials. Figure 6 summarizes the averaged scores using a radar plot. The

###### Case1 (one-time needle handover): left needle driver passes needle to right needle driver. (left → right)

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Case2 (two-time needle handover): left needle driver passes needle to right needle driver, then right needle driver passes needle to left needle driver. (left → right → left)

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Same conditioning frame

[Figure 24]

- Case3 (three-time needle handover): left needle driver passes needle to right needle driver, then right needle driver passes needle to left needle driver, then left needle driver passes needle to right needle driver. (left → right → left → right)
- Case4 (needle puncture): left needle driver punctures tissue.

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

- Figure 5 | New behavior generalization via strong text–video alignment. Given the same conditioning frame, our surgical world model generates distinct video rollouts corresponding to four task prompts: (1) one-time needle handover, (2) two-time needle handover, (3) three-time needle handover, and (4) needle puncture.

[Figure 35]

Text-video Alignment

Average Tool Consistency

Anatomy Structure

Zero-Shot Action-category Cosmos-H-Surgical

2.79

2.81

2.73

2.82

2.18

1.86

2.23

2.45

1.38

1.14

1.26

1.73

- Figure 6 | Human expert evaluation of generated surgical videos. Radar plot summarizing expert ratings across three criteria using a 1–3 quality scale.

Table 2 | Few-shot finetuning results on real surgical trajectories. We report task success rate (SR) and image quality metrics: FVD and VBench metrics. FT and PT indicate whether the model is finetuned on the 5 real trajectories or pretrained on the SATA dataset, respectively.

|Method FT PT|SR ↑ FVD ↓ DD ↑ IQ ↑ OC ↑<br><br>|
|---|---|
|Zero-shot ✗ ✗ Finetuned-Orig ✓ ✗ Cosmos-H-Surgical ✓ ✓|0.0 235.2 53.6 70.3 20.1 51.8 212.5 85.7 72.0 21.1 73.2 207.1 89.3 73.3 22.4|

Few-Shot Adaptation. We further evaluate the few-shot finetuning capability of the proposed surgical world model using only 5 real-world trajectories from the Needle Pick-Up and Hand-Over task (Section 3.1). Three configurations are compared: (1) Zero-Shot, using the original model directly; (2) Finetuned-Orig, where the model is finetuned from the original Cosmos-Predict2.5 [1] checkpoint; and (3) Cosmos-H-Surgical, where the model is first pretrained on the curated SATA dataset and then finetuned on the 5 trajectories. For evaluation, we generate videos from 56 hold-out initial frames selected from the 66 out-of-domain episodes described in Section 3.1. These frames are chosen based on proper initial needle and forceps configurations to ensure physically meaningful task initialization and comparable scene context. We report success rate (SR) and the same suite of video quality metrics as in Table 1. To assess the success rate, surgical experts evaluate the completeness of the

Cosmos-H-Surgical achieves the highest ratings across all dimensions, reflecting superior text grounding, consistent instrument manipulation, and anatomically realistic tissue behavior. In contrast, the Zero-shot and Action-category variants exhibit weaker temporal coherence and less stable tool–tissue interaction, underscoring the importance of prompt-level curation for clinically faithful video generation.

[Figure 36]

- Figure 7 | An example of left arm cartesian trajectory (first 3-dim of action space). Comparing Real only (Blue), Real + Syn 10x (Green), and groundtruth (Red).

trajectories in the generated videos.

As shown in Table 2, Cosmos-H-Surgical achieves the best overall performance, with a 73.2% success rate and the lowest FVD among all variants. Compared to direct finetuning from the original Cosmos-2.5 checkpoint, SATA pretraining yields consistently better video quality metrics, indicating improved temporal stability and perceptual fidelity. These results demonstrate that large-scale surgical video pretraining substantially enhances the model’s ability to adapt from limited real-world data, enabling robust and data-efficient generation for downstream surgical policy learning.

#### 4.2. Robotic Policy Experiments

For the 60 human teleoperated data from surgical robots, we split the last 40 as held out test set. For the rest of the data we perform experiments by gradually increasing the training data number. We use 5, 10, and 20 data for finetuning the world model, IDM, and the GR00T policy. All videos are resized to 224×224 and augmented with color jitter, and the jaw opening and cartesian actions are normalized with min-max normalization.

World Model Rollouts. We finetune CosmosH-Surgical using 5, 10, and 20 real-world videos and each finetuned model is used to generate synthetic videos conditioned on the initial frames from the out-of-domain episodes. In total, two sets of 56 single-rollout (1×) and 560 multi-rollout (10×, generated with 10 random seeds) synthetic videos are generated for each data regime. These two sets are used to show if number of synthetic data matters.

IDM Training and Pseudo Action Labeling. For

IDM model training, we use the out-of-the-domain episodes together with varying 5, 10, 20 (thus three separate IDMs) training episodes. We started from the pretrained Franka IDM checkpoint from DreamGen [5] and finetuned the model for 10k steps with a learning rate of 1e-4. The IDM is then used to generate pseudo-labeled kinematics for the synthetic videos.

Robot Policy Results. For all the experiments, we start from the pretrained GR00T N1.5 checkpoint which is a strong starting point. Our base policy (Real Only) is the model finetuned only with 5, 10, 20 real training data with learning rate 1e-4, 200 steps. For 56 and 560 synthetic data (Real+Synthetic, Real+Synthetic 10x), we first finetune with learning rate 1e-4 for 400 steps, then we further finetune with 5, 10, 20 real training data with learning rate 1e-4 and 200 steps. We test these 6 policy models on the 40 held out test episodes. An example of left arm cartesian trajectory (first 3 digits of the model’s 20 dimension output) is shown in Fig. 7. The trajectory (left arm movement) of Real + Syn 10x showed closer similarity to the groundtruth. We calculate the mean square error (MSE) for all 20 dimension action prediction compared with groundtruth and averaged across all test data. The result is shown in Fig. 8. We separate the cartesian, rotation and jaw since they have different physical meaning. We can see that the average MSE is the largest for the VLA finetuned only with real data, while with synthetic videos the average MSE gets lower. The trend holds for both varying real trajectory number and varying synthetic video number. Similar trend can be observed for varying data finetuning hyperparameters and for VLAs other then GR00T model (e.g. 𝜋0.5 [47]) , as shown in the supplementary.

### 5. Conclusion

We present Cosmos-H-Surgical, the first surgical world model that connects high-quality synthetic surgical videos generation with robot action learning. By curating the SATA dataset and integrating inverse dynamics modeling, Cosmos-H-Surgical generates synthetic data that can improve downstream policy training, enabling scalable and safe surgical robot data curation. However, Cosmos-H-Surgical still has major limitations. It requires finetuning datasets from unseen robot embodiments for both world model and IDM, which require additional data curation efforts. Meanwhile, pseudo-kinematics from IDM still lack ground-truth precision and may introduce residual noise. Lastly, the current SATA dataset, though diverse and finely annotated for physical AI, is not covering all the publicly available datasets. Future work will focus on extending SATA with more complex and broader procedures and improving IDM for better pseudo-kinematic.

[Figure 37]

- Figure 8 | Trajectory MSE (Standard deviation) on 40 test data. We use 5, 10, 20 real data to finetune the policy, starting from GR00TN1.5 pretrained checkpoint (Real), and checkpoints pretrained from 56 (Real + Synthetic) and 560 (Real + Synthetic 10x) synthetic data.

### References

- [1] Arslan Ali, Bai Junjie, Bala Maciej, Balaji Yogesh, and et al. World simulation with video foundation models for physical ai, 2025.
- [2] Anthony Brohan, Noah Brown, Julio Carbajal, Yevgen Chebotar, Danny Driess, Chelsea Finn, Chen Fu, Brian Ichter, Alex Irpan, Ryan Julian, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.
- [3] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [4] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [5] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, KaushilKundalia, Yen-ChenLin, etal. Dreamgen: Unlocking generalization in robot learning through video world models. arXiv preprint arXiv:2505.12705, 2025.
- [6] Samuel Schmidgall, Axel Krieger, and Jason Eshraghian. Surgical gym: A high-performance gpu-based platform for reinforcement learning with surgical robots. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 13354–13361. IEEE, 2024.
- [7] Jiaqi Xu, Bin Li, Bo Lu, Yun-Hui Liu, Qi Dou, and Pheng-Ann Heng. Surrol: An open-source reinforce-

- ment learning centered and dvrk compatible platform for surgical robot learning. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1821–1828. IEEE, 2021.
- [8] Hongbin Lin, Bin Li, Chun Wai Wong, Juan Rojas, Xiangyu Chu, and Kwok Wai Samuel Au. World models for general surgical grasping. arXiv preprint arXiv:2405.17940, 2024.
- [9] Saurabh Koju, Saurav Bastola, Prashant Shrestha, Sanskar Amgain, Yash Raj Shrestha, Rudra PK Poudel, and Binod Bhattarai. Surgical vision world model. In MICCAI Workshop on Data Engineering in Medical Imaging, pages 1–10. Springer, 2025.
- [10] Mehmet Kerem Turkcan, Mattia Ballo, Filippo Filicori, and Zoran Kostic. Towards suturing world models: Learning predictive models for robotic surgical tasks. arXiv preprint arXiv:2503.12531, 2025.
- [11] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. 𝜋0: A vision-language-action flow model for general robot control. arXiv preprint arXiv.2410.24164, 2024.
- [12] Dean A. Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems (NeurIPS), 1989.
- [13] Stéphane Ross, Geoff Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.
- [14] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems, 33:1179–1191, 2020.

- [15] Stephen James, Paul Wohlhart, Mrinal Kalakrishnan, Dmitry Kalashnikov, Alex Irpan, Julian Ibarz, Sergey Levine, Raia Hadsell, and Konstantinos Bousmalis. Sim-to-real via sim-to-sim: Data-efficient robotic grasping via randomized-to-canonical adaptation networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12627–12637, 2019.
- [16] David Ha and Jürgen Schmidhuber. World models. In Advances in Neural Information Processing Systems (NeurIPS), 2018.
- [17] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 2555–2565. PMLR, 2019.
- [18] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.
- [19] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.
- [20] Chenxin Li, Hengyu Liu, Yifan Liu, Brandon Y Feng, Wuyang Li, Xinyu Liu, Zhen Chen, Jing Shao, and Yixuan Yuan. Endora: Video generation models as endoscopy simulators. In International conference on medical image computing and computer-assisted intervention, pages 230–240. Springer, 2024.
- [21] Joseph Cho, Samuel Schmidgall, Cyril Zakka, Mrudang Mathur, Dhamanpreet Kaur, Rohan Shad, and William Hiesinger. Surgen: Text-guided diffusion model for surgical video generation. arXiv preprint arXiv:2408.14028, 2024.
- [22] Yousef Yeganeh, Rachmadio Lazuardi, Amir Shamseddin, Emine Dari, Yash Thirani, Nassir Navab, and Azade Farshad. Visage: Video synthesis using action graphs for surgery. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 146–156. Springer, 2024.
- [23] Lukas Zbinden, Nigel Nelson, Juo-Tung Chen, Xinhao Chen, Ji Woong, Mahdi Azizian, Axel Krieger, Sean Huver, et al. Cosmos-surg-dvrk: World foundation model-based automated online evaluation of surgical robot policy learning. arXiv preprint arXiv:2510.16240, 2025.
- [24] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.
- [25] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video

- prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.
- [26] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024.
- [27] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025.
- [28] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2024.
- [29] Yonghao Long, Anran Lin, Derek Hang Chun Kwok, Lin Zhang, Zhenya Yang, Kejian Shi, Lei Song, Jiawei Fu, Hongbin Lin, Wang Wei, et al. Surgical embodied intelligence for generalized task autonomy in laparoscopic robot-assisted surgery. Science Robotics, 10(104):eadt3093, 2025.
- [30] Ji Woong Kim, Tony Z Zhao, Samuel Schmidgall, Anton Deguet, Marin Kobilarov, Chelsea Finn, and Axel Krieger. Surgical robot transformer (srt): Imitation learning for surgical tasks. arXiv preprint arXiv:2407.12998, 2024.
- [31] Ji Woong Kim, Juo-Tung Chen, Pascal Hansen, Lucy Xiaoyang Shi, Antony Goldenberg, Samuel Schmidgall, Paul Maria Scheikl, Anton Deguet, Brandon M White, De Ru Tsai, et al. Srt-h: A hierarchical framework for autonomous surgery via language-conditioned imitation learning. Science robotics, 10(104):eadt5254, 2025.
- [32] Samuel Schmidgall, Ji Woong Kim, Alan Kuntz, Ahmed Ezzat Ghazi, and Axel Krieger. Generalpurpose foundation models for increased autonomy in robot-assisted surgery. Nature Machine Intelligence, 6(11):1275–1283, 2024.
- [33] Nicolás Ayobi, Santiago Rodríguez, Alejandra Pérez, Isabela Hernández, Nicolás Aparicio, Eugénie Dessevres, Sebastián Peña, Jessica Santander, Juan Ignacio Caicedo, Nicolás Fernández, et al. Pixel-wise recognition for holistic surgical scene understanding. arXiv preprint arXiv:2401.11174, 2024.
- [34] Dimitrios Psychogyios, Emanuele Colleoni, Beatrice Van Amsterdam, Chih-Yang Li, Shu-Yu Huang, Yuchong Li, Fucang Jia, Baosheng Zou, Guotai Wang, Yang Liu, et al. Sar-rarp50: Segmentation of ssurgical instrumentation and action recognition on robot-assisted radical prostatectomy challenge. arXiv preprint arXiv:2401.00496, 2023.
- [35] Sanat Ramesh, Diego Dall’Alba, Cristians Gonzalez, Tong Yu, Pietro Mascagni, Didier Mutter, Jacques

- Marescaux, Paolo Fiorini, and Nicolas Padoy. Weakly supervised temporal convolutional networks for fine-grained surgical activity recognition. IEEE Transactions on Medical Imaging, 42(9):2592–2602, 2023.
- [36] Klaus Schoeffmann, Heinrich Husslein, Sabrina Kletz, Stefan Petscharnig, Bernd Muenzer, and Christian Beecks. Video retrieval in laparoscopic video recordings with dynamic content descriptors. Multimedia Tools and Applications, 77(13):16813–16832, 2018.
- [37] Ziyi Wang, Bo Lu, Yonghao Long, Fangxun Zhong, Tak-Hong Cheung, Qi Dou, and Yunhui Liu. Autolaparo: A new dataset of integrated multi-tasks for image-guided surgical automation in laparoscopic hysterectomy. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 486–496. Springer, 2022.
- [38] Lena Maier-Hein, Martin Wagner, Tobias Ross, Annika Reinke, Sebastian Bodenstedt, Peter M Full, Hellena Hempe, Diana Mindroc-Filimon, Patrick Scholz, Thuy Nuong Tran, et al. Heidelberg colorectal data set for surgical data science in the sensor operating room. Scientific data, 8(1):101, 2021.
- [39] Zhitao Zeng, Zhu Zhuo, Xiaojun Jia, Erli Zhang, Junde Wu, Jiaan Zhang, Yuxuan Wang, Chang Han Low, Jian Jiang, Zilong Zheng, et al. Surgvlm: A large vision-language model and systematic evaluation benchmark for surgical intelligence. arXiv preprint arXiv:2506.02555, 2025.
- [40] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.
- [41] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [42] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [43] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022.
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [45] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. arXiv preprint arXiv:1812.01717, 2019.

- [46] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [47] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael

Equi, Chelsea Finn, Niccolo Fusai, et al. 𝜋0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Supplementary Material

6. Human Expert Evaluation Criteria

Human expert evaluation criteria. To assess perceptual realism and clinical fidelity, we conducted a human expert study with 3 surgical experts (one surgeon guides two residents to rate generated videos). Each expert independently evaluated 50 videos generated by different world model variants using a three-level clinical quality rubric (scores from 1 to 3, higher is better) across the following criteria:

- • Text–Video Alignment: Assesses whether the generated scenes match the textual prompt and surgical viewing perspective.

- – 1: Scene roughly aligns with the prompt but exhibits camera mismatch or unrealistic perspective.
- – 2: Scene alignment is correct and transitions are mostly natural with minor motion jumps or deformations.
- – 3: Scene fully aligns with prompt and surgical viewpoint; transitions are smooth and visually coherent.

- • Tool Consistency: Evaluates whether surgical instruments correspond to the prompt and remain physically consistent throughout the sequence.

- – 1: Tool type matches the text but exhibits frequent deformation or discontinuity.
- – 2: Tools remain largely consistent with reduced deformation; grasping and manipulation appear mostly realistic.
- – 3: Tools behave continuously and naturally, accurately performing realistic grasping and needle handling without visual artifacts.

- • Anatomical Structure: Measures the plausibility of tissue and organ appearance and their reactions to tool interaction.

[Figure 38]

- Figure 9 | The illustration of the needle pick up and hand over task for the specific surgical robot platform.

- – 1: Basic anatomical layout correct but unrealistic structure or misaligned tissue response.
- – 2: Improved detail reproduction with occasional deformation inconsistencies.
- – 3: Anatomically accurate structures with realistic responses (e.g., traction, deformation, bleeding) closely resembling real surgery.

### 7. Cosmos-H-Surgical Training

We adopt the Flow Matching (FM) formulation [42] to train the surgical video world model due to its conceptual simplicity and practical effectiveness. FM defines a velocity-based target in latent space, providing a direct training signal that improves optimization stability and sample quality. Formally, given a data sample 𝐼 (e.glet@tokeneonedot, video frames), a noise vector 𝜖 ∼ 𝒩(0,𝐼), and a timestep 𝑡∈[0,1] sampled from a logit-normal distribution, the interpolated latent is defined as:

𝐼𝑡=(1−𝑡)𝐼+𝑡𝜖, (1) where the corresponding ground-truth velocity is

𝑣𝑡=𝜖−𝐼. (2)

The model predicts this velocity via a network 𝑢𝜃(𝐼𝑡,𝑡,𝑐), where 𝑐 represents the conditioning frame 𝐼0 and corresponding text prompt, and parameters 𝜃 correspond to the trainable LoRA adapters. The flow-matching loss is then the mean squared error (MSE) between predicted and ground-truth velocities:

ℒ(𝜃)=E𝐼,𝜖,𝑐,𝑡⃦𝑢𝜃(𝐼𝑡,𝑡,𝑐)−𝑣𝑡⃦2

2. (3)

### 8. Additional Results

#### 8.1. More Visual Comparisons

To further illustrate the effect of few-shot adaptation on real surgical scenes, Figure 10 presents qualitative comparisons across three model configurations, including Zero-Shot, Finetuned-Orig, and Cosmos-HSurgical, evaluated under two representative initial conditions. Each rollout begins from a distinct endoscopic view with different needle and forceps arrangements, enabling a controlled examination of how well each model handles realistic surgical complexity.

The Zero-Shot model, which has no domain adaptation, frequently produces implausible motions and incorrect surgical tools, leading to early task failure. TheFinetuned-Origmodelreducessomehallucinations but still struggles with consistent grasping behavior and coherent tool interaction. In contrast, the CosmosH-Surgical model (SATA-pretrained then finetuned on

| |[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>Case1:<br><br>Zero-Shot<br><br>Finetuned-Orig<br><br>Cosmos-H-Surgical|
|---|---|
| |[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>Case2:<br><br>Zero-Shot<br><br>Finetuned-Orig<br><br>Cosmos-H-Surgical|

- Figure 10 | Qualitative comparison of few-shot finetuning variants in real-world surgical scenes. We visualize predicted video rollouts from three model configurations—Zero-Shot, Finetuned-Orig, and Cosmos-HSurgical—under two representative initial states (Case 1 and Case 2). Red arrows highlight incorrect surgical tools or hallucinations in the generated frames.

five real trajectories) generates smooth and accurate motions that correctly execute the intended manipulation sequence. Red arrows mark common failure modes, including tool hallucination and incorrect action execution, emphasizing the improvement brought by domain-specific pretraining. These visual results underscore the importance of the proposed SATApretrained Cosmos-H-Surgical model for reliable fewshot generalization in real-world surgical environments.

pairs can still improve multi-view VLA policy performance. We use the exact same training scheme and the same synthetic data as in the main paper, but we train on the 5, 10, and 20 real demonstrations while including the two additional wrist cameras, resulting in data with three views. Note that GR00T N1.5 VLA can process varying numbers of input views with the same weights. The results are shown in Fig. . 11.

Varying Hyperparameters We also performed experiments varying the training steps. We repeated the GR00T policy training with varying 1k and 10k finetuning steps on the real data as shown in Fig. 12 and Fig. 13.

#### 8.2. Additional Policy Results

In this section we provide additional policy evaluation results. The task of “Needle pick up and handover" is illustrated in Fig. 9.

Other VLA 𝜋0.5 To validate if Cosmos-H-Surgical and IDM can improve other VLAs, we applied the same strategy to another recent foundational VLA model 𝜋0.5 [47], which showed strong open world generalization. We simply change the GR00T policy to 𝜋0.5 and repeated the experiments as shown in Fig. 14.

Multi-view Robotic Policy Internet video datasets typically contain only single-view endoscopic videos, and obtaining multi-view surgical videos for training a multi-view world model is currently not feasible. As a result, SurgeWorldgeneratesonlysingle-viewsynthetic videos. However, downstream surgical robots may use multiple cameras for multi-view policy learning. Here, we show that even when real surgical data includes multiple views—such as additional left and right wrist cameras—the single-view synthetic video–kinematic

#### Table 3 | Source distribution of SATA video clips across four fundamental surgical actions.

Dataset Knotting Needle Grasping Needle Puncture Suture Pulling AutoLaparo [37] 47 9 42 30 GraSP [33] 37 – 1 28 HeiCo [38] 16 3 6 4 Multiypass140 [35] 1 – – – SAR-RARP50 [34] 118 677 940 413 SurgicalActions160 [36] 7 – – – YouTube [32] 68 – – –

[Figure 65]

- Figure 11 | Multi-View real data finetuning. Trajectory MSE (Standard deviation) on 40 test data. We use 5, 10, 20 real data with 3-View camera input to finetune the policy, starting from GR00T N1.5 pretrained checkpoint (Real), and checkpoints pretrained from 56 (Real + Synthetic) and 560 (Real + Synthetic 10x) synthetic data Single View.

[Figure 66]

- Figure 12 | 1k step finetuning: Trajectory MSE (Standard deviation) on 40 test data. We use 5, 10, 20 real data to finetune the policy for 1k steps, starting from GR00T N1.5 pretrained checkpoint (Real), and checkpoints pretrained from 56 (Real + Synthetic) and 560 (Real + Synthetic 10x) synthetic data.

[Figure 67]

##### Figure 13 | 10k step finetuning. Trajectory MSE (Standard deviation) on 40 test data. We use 5, 10, 20 real data to finetune the policy for 10k steps, starting from GR00T N1.5 pretrained checkpoint (Real), and checkpoints pretrained from 56 (Real + Synthetic) and 560 (Real + Synthetic 10x) synthetic data.

[Figure 68]

##### Figure 14 | 𝜋0.5 results. Trajectory MSE (Standard deviation) on 40 test data. We use 5, 10, 20 real data to finetune the policy, starting from GR00T N1.5 pretrained checkpoint (Real), and checkpoints pretrained from 56 (Real + Synthetic) and 560 (Real + Synthetic 10x) synthetic data.

