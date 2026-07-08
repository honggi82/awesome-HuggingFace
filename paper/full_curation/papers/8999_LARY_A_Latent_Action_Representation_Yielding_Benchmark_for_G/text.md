# arXiv:2604.11689v1[cs.CV]13Apr2026

## LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment

#### Dujun Nie Fengjiao Chen∗ Qi Lv Jun Kuang Xiaoyu Li Xuezhi Cao Xunliang Cai Meituan, Beijing, China

### ABSTRACT

While the shortage of explicit action data limits Vision-Language-Action (VLA) models, human action videos offer a scalable yet unlabeled data source. A critical challenge in utilizing large-scale human video datasets lies in transforming visual signals into ontology-independent representations, known as latent actions. However, the capacity of latent action representation to derive robust control from visual observations has yet to be rigorously evaluated. We introduce the Latent Action Representation Yielding (LARY) Benchmark, a unified framework for evaluating latent action representations on both high-level semantic actions (what to do) and low-level robotic control (how to do). The comprehensively curated dataset encompasses over one million videos (1,000 hours) spanning 151 action categories, alongside 620K image pairs and 595K motion trajectories across diverse embodiments and environments. Our experiments reveal two crucial insights: (i) General visual foundation models, trained without any action supervision, consistently outperform specialized embodied latent action models. (ii) Latent-based visual space is fundamentally better aligned to physical action space than pixel-based space. These results suggest that general visual representations inherently encode action-relevant knowledge for physical control, and that semantic-level abstraction serves as a fundamentally more effective pathway from vision to action than pixel-level reconstruction. GitHub: https://github.com/meituan-longcat/LARYBench HomePage: https://meituan-longcat.github.io/LARYBench Hugging Face: https://huggingface.co/datasets/meituan-longcat/LARYBench

[Figure 1]

DATA SCALE

DIVERSE EMBODIMENTS

[Figure 2]

[Figure 3]

[Figure 4]

###### 1.2M+ 620K+ 595K+

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

11 RoboticPlatforms+Human

Video Clips Image Pairs Trajectories

###### ACTION TAXONOMY

###### SCENARIO DIVERSITY

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

###### 151

Classes

Home Kitchen Factory Store

High-Level Semantic Understanding Action Classification

###### Low-Level Control Mapping Action Regression

- • Kinematic-Level Atomic Primitives

LIBERO

25K+ Image Pairs, 28 CLASSES

- • Task-Level Composite Behaviors

###### • Single-Arm Platform

|[Figure 21]| |
|---|---|
| | |

|[Figure 22]|
|---|

|[Figure 23]| |
|---|---|
| | |

|[Figure 24]|
|---|

Left

Forward Right Close Gripper

|[Figure 25]<br><br>|
|---|

|[Figure 26]|
|---|

|[Figure 27]<br><br>|
|---|

|[Figure 28]|
|---|

|[Figure 29]<br><br>|
|---|

|[Figure 30]|
|---|

…

CALVIN

…

LIBERO

7 DoF

|[Figure 31]| |
|---|---|
| | |

|[Figure 32]|
|---|

|[Figure 33]| |
|---|---|
| | |

|[Figure 34]|
|---|

…

VLABench

Open Cut Drop

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

|[Figure 50]|[Figure 51]<br><br>|
|---|---|

|[Figure 52]|[Figure 53]<br><br>|
|---|---|

|[Figure 54]<br><br>|[Figure 55]|
|---|---|

321K+ Image Pairs & Trajectories

…

Human

###### • Bimanual-Arm Platform

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

EgoDex + SSv2 + Ego4D + HoloAssist + EPIC-KITCHENS + TACO

|[Figure 71]| |
|---|---|
| | |

|[Figure 72]|
|---|

|[Figure 73]| |
|---|---|
| | |

|[Figure 74]|
|---|

692K+ Video Clips, 123 CLASSES

…

12 DoF

RoboCOIN

Pour Fold Pick

|[Figure 75]|[Figure 76]<br><br>|
|---|---|

|[Figure 77]<br><br>|[Figure 78]|
|---|---|

|[Figure 79]|[Figure 80]<br><br>|
|---|---|

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

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

|[Figure 96]| |
|---|---|
| | |

|[Figure 97]|
|---|

|[Figure 98]| |
|---|---|
| | |

|[Figure 99]|
|---|

…

Robot

…

16 DoF

AgiBotWorld

[Figure 100]

[Figure 101]

[Figure 102]

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

AgiBotWorld

538K+ Video Clips, 54 CLASSES

273K+ Image Pairs & Trajectories

Figure 1: LARYBench evaluates vision-to-action transformation on both action generalization and robotic control.

∗Project Lead.

### Contents

- 1 Introduction 3
- 2 Related Work 4
- 3 The LARY Benchmark 4

- 3.1 Hierarchical Semantic Probing Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.2 Physical Execution Mapping Assessment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 4 Experiments 6

- 4.1 Taxonomy of Action Representation Paradigms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.2 Evaluation Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.3 Benchmark Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 5 Error Analysis 9

- 5.1 The Long-Tail Dilemma and Mid-Frequency Semantic Aliasing . . . . . . . . . . . . . . . . . . . . 9
- 5.2 Spatiotemporal Grounding and Action-Centric Attention . . . . . . . . . . . . . . . . . . . . . . . . 10
- 5.3 Stride Ablation: Latent Action Spaces Encode Robust Dynamic Trajectories . . . . . . . . . . . . . . 11

- 6 Conclusion 12

- A Overview of the Appendix 15
- B Details of LARYBench 15

- B.1 Composition of LARYBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- B.2 Additional Data Curation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- C Experimental Details of General LAMs 21

- C.1 Models and Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.2 Training Configurations and Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Additional Visualizations and Case Studies 22

- D.1 Cross-Embodiment Gap Across Different Model Paradigms . . . . . . . . . . . . . . . . . . . . . . . 22
- D.2 Spatiotemporal Grounding via Attention Visualization . . . . . . . . . . . . . . . . . . . . . . . . . 22

### 1 Introduction

The paradigm of learning from large-scale, unlabeled human video data has emerged as a promising solution to the “data island” problem in robotics, where diverse action-labeled datasets remain too scarce to train generalist foundation models. The pivotal challenge in leveraging human video data lies in the transformation of raw visual signals into ontology-agnostic action representations, commonly referred to as latent actions [Ye et al., 2024, Bu et al., 2025, Chen et al., 2025, 2024a, Yang et al., 2025, Chen et al., 2024b, NVIDIA et al., 2025, Gao et al., 2026]. Pioneering Latent Action Models (LAM) such as Latent Action Pretraining from Videos (LAPA) [Ye et al., 2024], Moto [Chen

- et al., 2024b], and LAPO [Schmidt and Jiang, 2023] introduced unsupervised frameworks that tokenize visual changes between video frames into discrete latent action tokens, analogous to word pieces in natural language processing. Similarly, IGOR [Chen et al., 2024a] treats the compression of visual changes between initial and goal images as atomic control units, facilitating semantic consistency across human and robot embodiments.

Despite recent progress in transforming vision to latent action, it lacks a thorough and rigorous framework for assessing the quality and effectiveness of latent action representations. Existing evaluations primarily rely on downstream manipulation task performance or qualitative methods such as cluster visualization [Ye et al., 2024, Bu et al., 2025, Gao et al., 2025, Schmidt and Jiang, 2023, Chen et al., 2024a,b, Ren et al., 2025, NVIDIA et al., 2025]. These approaches fail to decouple the assessment of the VLA components from the latent action quality itself. Furthermore, there is a notable absence of evaluation methods that span different entities, tasks, and granularities, making it difficult to assess the generalization capabilities of action representations. Crucially, the impact of different training architectures, strategies, and usage paradigms on action representation remains underexplored.

To bridge this gap, we introduce the Latent Action Representation Yielding (LARY) Benchmark, a quantitative framework designed to rigorously evaluate latent action representations. Our objective is to establish a standardized metric that assesses both embodied capabilities spanning cross-agent and cross-scenario applications and video understanding ability.

Actions in embodied intelligence inherently span two complementary levels: high-level semantic intent that specifies what to do, and low-level physical control that determines how to do it. LARYBench evaluates latent action representations along both dimensions. For High-Level Semantic Understanding, we assess whether representations can distinguish atomic primitives (e.g., “move up”, “close gripper”) and composite behaviors (e.g., “pick up”, “place”, “twist”), drawing on diverse human and robot manipulation videos that cover a wide range of scenarios. For Low-Level Control Mapping, we examine whether representations preserve sufficient physical detail to reconstruct end-effector trajectories, using embodied datasets with fine-grained robot motion annotations.

Since current embodied video datasets [Wang et al., 2023, Grauman et al., 2022, Goyal et al., 2017, Liu et al.,

- 2024, Damen et al., 2020, Hoque et al., 2025] often suffer from imprecise temporal boundaries and inconsistent action annotations, we develop an automated data engine to re-segment and re-annotate a large-scale corpus. The resulting benchmark comprises over 1.2M short videos (totaling more than 1,000 hours) spanning 151 unique action categories, alongside 620K image pairs and 595K motion trajectories. It covers both human and robotic agents, captured from egocentric and exocentric perspectives across simulated and real-world environments. Building on this diverse foundation, LARYBench instantiates the two evaluation dimensions as concrete tasks: Semantic Action Classification and Low-level Control Regression.

To systematically assess latent action quality, we benchmark three families of models: (i) Embodied LAMs (e.g., LAPA [Ye et al., 2024], UniVLA [Bu et al., 2025], villa-X [Chen et al., 2025]), which are purpose-built for robot manipulation; (ii) General Vision Encoders, including both semantic-level (DINOv3 [Siméoni et al., 2025], V-JEPA

##### 2 [Assran et al., 2025]) and pixel-level (FLUX.2-dev [Labs, 2024], Wan2.2 [Wan et al., 2025]) backbones, evaluated for their inherent capacity to encode action-relevant features without explicit action supervision; and (iii) General LAMs, a new class of models we propose by grafting the LAM training paradigm onto frozen general vision backbones (e.g., LAPA-DINOv2, LAPA-DINOv3, LAPA-SigLIP2, LAPA-MAGVIT2). Across 11 models, our evaluation reveals a consistent hierarchy: off-the-shelf General Vision Encoders outperform General LAMs, which in turn surpass Embodied LAMs, suggesting that current embodied-specific training not only fails to leverage powerful visual priors but may actually constrain representation quality. Our contributions are as follows:

- • We introduce LARYBench, a comprehensive benchmark that first decouples the evaluation of latent action representations from downstream policy performance. LARYBench probes representations along two complementary dimensions, high-level semantic action (what to do) encoding and the low-level physical dynamics required for robotic control (how to do it), enabling direct, standardized measurement of representation quality itself.

- • To support rigorous evaluation, we develop an automated data engine to re-segment and re-annotate a large-scale corpus, yielding 1.2M videos, 620K image pairs, and 595K trajectories across 151 action categories and 11 robotic embodiments, covering both human and robotic agents from egocentric and exocentric perspectives in simulated and real-world environments.
- • Through systematic evaluation of 11 models, we reveal two consistent findings: (i) action-relevant features can emerge from large-scale visual pre-training without explicit action supervision, and (ii) latent-based feature spaces tend to align with robotic control better than pixel-based ones. These results suggest that future VLA systems may benefit more from leveraging general visual representations than from learning action spaces solely on scarce robotic data.

### 2 Related Work

Latent Action Representation from Videos. To alleviate reliance on teleoperated data, recent research extracts latent control signals from unlabeled videos via Inverse Dynamics Models (IDMs). Existing approaches diverge into discrete and continuous paradigms. Discrete methods, such as LAPA [Ye et al., 2024] and Moto [Chen et al., 2024b], employ Vector Quantization (VQ) to facilitate autoregressive behavior cloning, though often at the cost of fine-grained information loss. Conversely, continuous approaches like CoMo [Yang et al., 2025] preserve motion fidelity but risk shortcut learning from background cues. To improve physical grounding and mitigate such artifacts, recent works incorporate semantic constraints via language or saliency (UniVLA [Bu et al., 2025], IGOR [Chen et al., 2024a]) and integrate physical priors, like robot trajectories (LatBot [Li et al., 2025]).

Latent Actions in World Models and VLAs. Latent actions function as a unification interface in generalist systems. In Vision-Language-Action (VLA) architectures like GR00T [NVIDIA et al., 2025], they decouple high-frequency control from low-frequency reasoning. Simultaneously, World Foundation Models (WFMs)—including Cosmos [et. al., 2025], VideoWorld [Ren et al., 2025], AdaWorld [Gao et al., 2025], and V-JEPA 2 [Assran et al., 2025]—utilize latent actions to condition future-frame prediction, enabling agents to internalize physical rules from passive observation. Expanding on the data sources for these models, DreamDojo [Gao et al., 2026] leverages large-scale human videos to construct a generalist robot world model, demonstrating the efficacy of extracting physical dynamics and actionable representations directly from human demonstrations. Advanced frameworks such as villa-X [Chen et al., 2025] refine this by jointly modeling action plans and video generation to ensure semantic alignment between intent and execution.

Evaluation of Latent Action Representations. Despite the proliferation of LAMs, quantitative evaluation remains challenging. Standard reconstruction metrics often fail to distinguish action dynamics from environmental noise. While benchmarks like EWMBENCH [Yue et al., 2025] and LAWM [Tharwat et al., 2025] utilize trajectory consistency or Canonical Correlation Analysis (CCA) for alignment, recent diagnostic studies [Zhang et al., 2025a] reveal that many models struggle with distractor robustness. Our work extends these inquiries by employing attentive probing and regression to rigorously test the semantic separability and embodied ability of latent action representations.

### 3 The LARY Benchmark

To enable a comprehensive assessment of latent action representations, we propose the Latent Action Representation Yielding Benchmark (LARY), which unifies the evaluation of both high-level semantic action encoding and the low-level physical dynamics required for robotic control. Formally, given a sequence of visual observations o1:T, the motion information is extracted by a latent action model (LAM) as the latents z ∈ Z. LARYBench evaluates the efficacy of Z through two tasks: fsem : Z → C for semantic action decoding via classification task, and fdyn : Z → A for robotic control construction via regression task.

As shown in Figure 1, LARYBench is curated from a massive amount of multi-embodiment datasets and human datasets, encompassing 151 meticulously defined actions (including 28 atomic actions and 145 composite actions), and corresponding 1.2M annotated samples. The dataset covers a large range of human activities from the frequent “pick” and “place”, to the long-tail “shovel” (snow) and “float” (balloon). To ensure morphological diversity, the dataset spans 11 distinct robotic embodiments, ranging from widely used single-arm manipulators such as Franka to complex bimanual and semi-humanoid platforms such as the AgiBot G1, Agilex Cobot, and Realman series, and includes extensive human-ego-centric interaction data. To ensure environmental diversity, the dataset captures thousands of unique object manipulations across a broad spectrum of unstructured environments, including simulated tabletops, authentic residential kitchens, commercial spaces, and industrial scenes.

With the diverse dataset spanning actions, embodiments, and objects, the evaluation framework of LARYBench is depicted in Figure 2. Next, we introduce the construction pipeline for the semantic action classification task and the robotic control regression task, respectively.

||[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>Action Classification Task<br><br>Robot (Exocentric)<br><br>Human (Egocentric)<br><br>Atomic Composite<br><br>Robot (Egocentric)<br><br>[Figure 125]<br><br>54<br><br>Classes Classes<br><br>[Figure 126]<br><br>28 123<br><br>Classes|
|---|
<br><br>|[Figure 127]<br><br>[Figure 128]<br><br>Action Regression Task<br><br>(Egocentric)<br><br>(Exocentric)<br><br>Interval<br><br>[Figure 129]<br><br>[Figure 130]|
|---|
<br><br>(a) Benchmark construction|
|---|

|General LAM Training<br><br>(b) Latent action extraction<br><br>E IDM FDM<br><br>Codebook<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>Vision Encoder<br><br>|[Figure 135]<br><br>E Frames Features<br><br>[Figure 136]|
|---|
<br><br>Embodied LAM<br><br>|IDM<br><br>[Figure 137]<br><br>|
|---|
<br><br>IDM<br><br>General LAM<br><br>E<br><br>[Figure 138]<br><br>[Figure 139]|
|---|

|(c) Evaluation<br><br>Latent-conditioned Action Regression<br><br>Action Expert<br><br>MSE<br><br>[Figure 140]<br><br>Probe-based Action Classification Predicted Class<br><br>|takeputgraspstirwipeothers<br><br>...|
|---|
<br><br>Attentive Probe<br><br>Accuracy<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>Evaluation Results<br><br>General Encoder<br><br>General LAMs<br><br>Embodied LAMs<br><br>>><br><br>[Figure 146]|
|---|

###### >>

E IDM FDM

Pre-trained Vision encoder Observation Image Feature Inverse dynamics model Forward dynamics model Continuous latent action Reconstruction Loss Ground Truth Trajectory Predicted Trajectory

- Figure 2: Overall pipeline of our benchmark LARY. First, (a) we construct a comprehensive dataset featuring both atomic and composite actions across varied domains to support classification and regression tasks. Next, (b) we extract the continuous latent action z using various representation paradigms, also highlighting the integration of pre-trained general vision encoders within the VQ-VAE training architecture. Finally, (c) we evaluate these representations utilizing probe-based classification and latent-conditioned regression to quantify their semantic separability (Accuracy) and physical dynamics modeling capabilities (MSE), respectively.

#### 3.1 Hierarchical Semantic Probing Protocol

Evaluating the semantic richness of latent action representations requires disentangling spatio-temporal complexities. We formalize this through a multi-granularity semantic probing protocol.

Kinematic-Level Atomic Primitives At the most granular semantic level, representations must capture instantaneous state variations. We define the Atomic Robot task by decomposing the robot’s end-effector actions into 28 discrete kinematic primitives, comprising finely resolved directional translations and binary gripper states. Leveraging exocentric demonstrations from the LIBERO suite [Liu et al., 2023], we apply detailed data preprocessing, such as thresholding motion along the z-axis relative to static orthogonal directions, to extract 25,940 high-quality image pairs with trajectories.

Task-Level Composite Behaviors To assess the capacity for abstracting complex behavioral semantics across diverse embodiments and scenarios, we introduce the Composite Human and Composite Robot tasks. To handle diverse data quality across existing datasets [Goyal et al., 2017, Grauman et al., 2022, Wang et al., 2023, Damen et al., 2020, Hoque et al., 2025], we develop an automated and scalable data engine to perform precise temporal segmentation and semantic alignment (see Figure 3). Comprehensive details of this engine’s architecture are provided in the Appendix. By deploying this system across a diverse corpus of ego-centric human datasets (HoloAssist [Wang

- et al., 2023], Ego4D [Grauman et al., 2022], Something-Somethingv2 [Goyal et al., 2017], TACO [Liu et al., 2024], EPIC-KITCHENS [Damen et al., 2020], EgoDex [Hoque et al., 2025]) and realistic bimanual robot demonstrations (AgiBotWorld-Beta [AgiBot-World-Contributors et al., 2025]), we extract 692,297 human clips and 538,423 robot clips under a unified taxonomy of 145 composite behavior classes. Beyond this initial curation, our automated engine holds the potential to continuously process future data streams, ensuring the ever-growing nature of the dataset.

#### 3.2 Physical Execution Mapping Assessment

While semantic features encode what to do, robotic manipulation ultimately demands how to do. Our trajectory regression task evaluates the latent space’s physical grounding by directly decoding continuous end-effector actions.

This protocol spans diverse hardware morphologies and action spaces. For single-arm exo-centric scenarios, we utilize CALVIN [Mees et al., 2022] and VLABench [Zhang et al., 2025b] featuring Franka arms. For bimanual ego-centric scenarios, we incorporate RoboCOIN [Wu et al., 2025] across 10 diverse platforms and AgiBotWorld-Beta featuring the AgiBot G1. The action space heterogeneity is meticulously preserved: AgiBotWorld-Beta targets a 16-DoF space containing absolute position, quaternions, and gripper state, whereas RoboCOIN maps to a 12-DoF space of position and Euler angles. We deliberately mask the dexterous hand joint data in RoboCOIN to focus the evaluation on macroscopic arm displacements, as fine-grained finger articulation remains an ill-posed inverse problem for current visual encoders.

Composite Classification

[Figure 147]

###### Doubao-1.5-pro-vision API

[Figure 148]

[Figure 149]

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

put take ...

Action Segmentation

Video-Description Matching

Video-Verb Consistency Check

Manual Sampling Inspection Video-Verb Pairs

Raw Video

Atomic Classification Regression

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

###### Trajectory Detection

Temporal Subsampling

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

+

+

Frame Pairs

[Figure 170]

Frame Pairs

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 171]

Raw Video + Trajectory

Robotic Trajectory Analysis Motion Labels

Raw Video + Trajectory

Fixed Interval Action Chunks

- Figure 3: Data curation process for LARY-Bench. To efficiently transform diverse raw videos into standardized evaluation tasks, we integrate a Vision-Language Model (VLM) with robust spatio-temporal video understanding capabilities. The VLM serves as the core reasoning agent for temporal video segmentation and semantic action alignment.
- 4 Experiments In this section, we conduct comprehensive experiments and try to answer following questions.

- 1. Do Latent Actions Capture Diverse Actions?
- 2. Do Latent Actions Encode Enough for Control?
- 3. What Constitutes an Effective Latent Action Model?
- 4.1 Taxonomy of Action Representation Paradigms To establish a comprehensive baseline, we categorize latent action models into four distinct learning paradigms:

- • Embodied Latent Action Models: Architectures natively designed for VLAs, such as LAPA [Ye et al., 2024], UniVLA [Bu et al., 2025], and villa-X [Chen et al., 2025], which explicitly entangle temporal dynamics with robotic control via forward/inverse dynamics modeling.
- • General Semantic Encoders: High-capacity visual backbones like DINOv3 [Siméoni et al., 2025] and V-JEPA2 [Assran et al., 2025] optimized for contrastive alignment or latent-level reconstruction, evaluated here for their zero-shot temporal modeling capacity.
- • Generative Pixel Encoders: Video synthesis models such as Wan2.2 VAE [Wan et al., 2025] and FLUX.2-dev VAE [Labs, 2024] that leverage spatial-temporal compression and pixel-level reconstruction to implicitly capture motion priors.
- • General Latent Action Models: To leverage the extensive knowledge embedded in pre-trained visual backbones, we explore modeling visual motion at the feature level. Specifically, we substitute the default encoder in the LAPA framework with various pre-trained backbones—ranging from semantic encoders like DINOv2 [Oquab et al., 2023], DINOv3 [Siméoni et al., 2025], and SigLIP 2 [Tschannen et al., 2025] to pixel ones such as MAGVIT2 [Luo et al.,

- 2024]. These hybrid models, trained on internet data containing human motions, robot motions, and environment motions, are included to assess how pre-trained visual priors enhance latent action learning.

#### 4.2 Evaluation Settings

Data Split and Evaluation Metrics To maintain consistent class distributions across classification tasks, we apply randomized stratified sampling, partitioning atomic and composite tasks at 75:25 and 70:30 ratios, respectively.

Regression tasks on VLABench [Zhang et al., 2025b] adhere to a standard 75:25 train-validation split. For crossenvironment evaluation on CALVIN [Mees et al., 2022], environments A, B, and C are allocated for training, with D designated for validation. RoboCOIN [Wu et al., 2025] and AgiBotWorld-Beta [AgiBot-World-Contributors et al.,

- 2025] are also divided at a 75:25 ratio for training and validation. For semantic action classification, we report the Top-1 Accuracy averaged across all action categories. For low-level control regression, we utilize the Mean Squared Error (MSE) to measure physical fidelity.

Sampling and Latent Action Extraction For the Atomic Robot and regression tasks, we mainly rely on curated image pairs, which allow us to obtain latent action representations from the LAM directly. To retain as much information as possible, we work with the continuous latent embeddings instead of discretized codebook indices, thereby avoiding quantization-related information loss. In contrast, extracting representations from video clips in the composite classification tasks is more difficult because the source datasets differ in FPS and exhibit non-uniform motion speeds. Simple uniform sampling generally fails to capture the true temporal dynamics and often yields latent actions that are effectively insensitive to the underlying motion. To address this, we employ the Motion-Guided Sampler (MGSampler) [Zhi et al., 2021] to select an effective sequence of frames that exhibits sufficient dynamic variation. Latent action features are subsequently extracted across all adjacent sampled frames to ensure the representation effectively encapsulates the transition dynamics.

Probing Protocol for Semantic Action Classification We uniformly sample 9 frames from each video clip (typically lasting under 5 seconds) and resize every frame to 224 × 224. To assess the semantic expressiveness of the learned latent actions, we employ a probe-based classification task. Following the architecture in V-JEPA [Assran et al., 2025], we use a 4-layer attentive probe as the classifier. Given the varying latent dimensions between general vision encoders and LAMs, we incorporate a projector to align the continuous latent actions into a uniform dimensional space prior to classification, thereby ensuring a fair comparison. The probe is trained for 20 epochs using bfloat16 precision. We apply a multi-head optimization strategy with learning rates from 3 × 10−4 to 5 × 10−3 and weight decay values from

- 0.01 to 0.8, to ensure a robust evaluation across different representation scales.

Action Experts for Low-level Control Regression The latent actions are derived from image pairs separated by a frame interval of s = 5. To assess the physical grounding of latent actions, we implement an easy MLP-based Action Experts to regress absolute end-effector trajectories. It adopts a standard MLP with residual connections architecture featuring 2 residual blocks and a hidden dimension of 4096. This model directly maps latent actions to an action chunk of size s (7-DoF/12-DoF/16-DoF per chunk).

Feature-Level Latent Action Model Training To leverage the pretrained vision encoders, we develop a family of Feature-Level Latent Action Models by substituting the visual representation from pixels to features generated by vision encoders in the LAPA framework, while retaining its VQ-VAE structure. During training, the encoder weights remain frozen, and the latent action representation is learned from scratch on an internal video dataset. Detailed training configurations are provided in the Appendix.

#### 4.3 Benchmark Results

Do Latent Actions Capture Diverse Actions? As shown in Table 1, general vision foundation models (e.g., V-JEPA 2 [Assran et al., 2025] and DINOv3 [Siméoni et al., 2025]) deliver surprisingly strong performance, even though they do not appear to perform any explicit motion extraction. This phenomenon demonstrates that the visual self-supervised training with large-scale data inherently yields general semantic action representations covering both robot and human data. In particular, V-JEPA 2 [Assran et al., 2025] learns directly from visual latent features instead of pixel features (the representations commonly used in world models), and achieves the best performance with a substantial margin. This supports the hypothesis that actions can be derived from latent features without needing to be explicitly represented in the pixel space. In contrast, existing Embodied LAMs (e.g., LAPA [Ye et al., 2024], UniVLA [Bu et al., 2025] and villa-X [Chen et al., 2025]) exhibit restricted generalization on diverse actions (averaging 17.99%–20.90%), due to the limited amount of training data or the early constraints to low-level actions [Bu et al., 2025, Chen et al., 2025].

We try to combine these two kinds of techniques, which extract motion using the self-supervised method in LAPA [Ye

- et al., 2024] on general vision foundation models, named as General LAMs. The training dataset is curated from the internet data, containing both human motions and non-human motions (e.g., car driving), which is very different from the test dataset. As shown in Table 1, the General LAMs perform better than existing Embodied LAMs, even on capturing the robot action. Although sharing the same vision foundation model, LAPA-DINOv2 significantly outperforms UniVLA (43.67% vs. 17.99%) by leveraging more diverse training data and imposing fewer constraints.

Classification Accuracy↑

Model Paradigm Params(M)

Avg. Atomic Robot Composite Human Composite Robot V-JEPA2 [Assran et al., 2025]

303.89 76.62 79.09 80.35 70.43 DINOv3 [Siméoni et al., 2025] 303.13 68.68 60.79 76.19 69.06 Wan2.2 [Wan et al., 2025]

Semantic Encoder

704.69 49.36 14.91 67.77 65.39 FLUX.2-dev [Labs, 2024] 84.05 47.48 51.95 46.12 44.36 LAPA [Ye et al., 2024]

Pixel Encoder

343.80 20.17 22.27 14.61 23.64 UniVLA [Bu et al., 2025] 287.75 17.99 18.62 19.08 18.56 villa-X [Chen et al., 2025] 238.71 20.90 15.00 17.80 29.90

Embodied LAM

116.40 40.78 33.12 59.70 29.53 LAPA-SigLIP2 200.83 43.67 46.83 54.74 29.44 LAPA-DINOv2 473.69 49.36 58.04 55.86 34.19 LAPA-DINOv3 472.45 49.17 56.27 64.19 27.04

LAPA-MAGVIT2

General LAM

- Table 1: Action Classification results of latent action representations. Avg. denotes the mean accuracy across Atomic Robot, Composite Human, and Composite Robot. Best results are in bold and second best are underlined.

Model Paradigm Params(M)

Regression MSE↓

Avg. CALVIN VLABench RoboCOIN AgiBotWorld-Beta V-JEPA 2 [Assran et al., 2025]

Semantic Encoder

303.89 0.25 0.27 0.07 0.32 0.33 DINOv3 [Siméoni et al., 2025] 303.13 0.19 0.22 0.06 0.22 0.24 Wan2.2 [Wan et al., 2025]

Pixel Encoder

704.69 0.30 0.39 0.09 0.34 0.39 FLUX.2-dev [Labs, 2024] 84.05 0.35 0.25 0.04 0.47 0.62 LAPA [Ye et al., 2024]

Embodied LAM

343.80 0.97 0.96 0.95 0.96 1.00 UniVLA [Bu et al., 2025] 287.75 0.87 0.82 0.74 0.94 0.97 villa-X [Chen et al., 2025] 238.71 0.87 0.86 0.72 0.94 0.97

LAPA-MAGVIT2

General LAM

116.40 0.65 0.59 0.36 0.80 0.83 LAPA-SigLIP2 200.83 0.65 0.57 0.30 0.86 0.88 LAPA-DINOv2 473.69 0.63 0.55 0.26 0.85 0.86 LAPA-DINOv3 472.45 0.60 0.50 0.25 0.82 0.84

- Table 2: Action Regression results of latent action representations. Avg. is the mean MSE across all datasets. Best results are in bold and second best are underlined.

Although the General LAMs achieve relatively high accuracy on human actions, their performance declines on robot actions due to limited data scale and diversity. We leave addressing this issue to future work.

In conclusion, vision foundation models can capture semantic action with self-supervised learning from large-scale of internet visual data. Learning based on latent features performances better than pixel features. More data diversity and fewer task-specific constrains may be the key to generalization.

Do Latent Actions Encode Enough for Control? As to the prediction of robotic control, the action regression results show similar performances among candidate models in Table 2, where general vision foundation models achieve significantly better performance among 4 kinds of datasets. Especially, latent-based vision encoders (DINOv3 [Oquab et al., 2023] and V-JEPA 2 [Assran et al., 2025]) capture better robotic control than pixel-based vision encoders (Wan2.2 [Wan et al., 2025] and FLUX.2-dev [Labs, 2024]). In other words, latent-based visual space is better aligned to robotic action space. Considering the good performance of the recent video world model based VLA, which generates pixels first and then decode to robotic control, we suggest the better and more agile way for general robotic control comes from learning latent features. We compared DINOv3 with V-JEPA 2 [Assran et al., 2025] and found that DINOv3 [Siméoni et al., 2025], thanks to its roots in visual contrastive learning, retains fine-grained recognition capabilities, which in turn improves the precision of fine-grained regression in robotic control tasks.

What Constitutes an Effective Latent Action Model? To systematically build a robust latent action model, we conduct ablations under the LAPA [Ye et al., 2024] framework. Figure 4 maps a performance evolution path bridging the gap between the worst baseline (LAPA) and the continuous upper bound (V-JEPA2). First, self-supervised visual encoders (e.g., DINOv3 [Siméoni et al., 2025]) consistently construct superior LAMs compared to reconstruction-

[Figure 172]

(sl=16, dim=32)

(sl=16, dim=32)

(cs=64, dim=32)

(cs=64, dim=32)

(cs=64, sl=49)

(cs=64, sl=49)

- Figure 4: Performance Evolution of Latent Action Models. The (*) denotes the default quantization settings for LAPA-DINOv3 (cs = 8,sl = 16,dim = 32). cs, sl, and dim represent Codebook Size, Sequence Length and Latent Dimension respectively. Composite Human classification on the left and AgiBotWorld-Beta AgiBot-World-Contributors et al. [2025] regression on the right.

Table 3: Ablation on Codebook Size. Default settings are Sequence Length = 49, Latent Dimension = 256. Best results are in bold and second best are underlined.

Codebook Size

Recon Loss

Codebook Utilization (%)

Accuracy (Composite task)↑ MSE Human Robot (AgiBotWorld-Beta)↓

8 0.00808 100.0 71.31 64.89 0.88 64 0.00751 100.0 70.15 64.04 0.83 256 0.00758 89.5 69.84 63.82 0.85

based and vision-language contrastive models like MAGVIT2 [Luo et al., 2024] and SigLIP2 [Tschannen et al., 2025](Tables 1-2), indicating that contrastive priors better capture fine-grained spatial-temporal correspondences. Setting LAPA-DINOv3 (cs = 8,sl = 16,dim = 32) as our foundation, we sequentially optimize its quantization bottleneck. Second, expanding codebook capacity improves downstream regression, but overly large sizes (e.g., 256) cause codebook utilization drops without further gains (Table 3), making a moderate size (cs = 64) optimal for dense representations. Third, sequence length critically dictates temporal diversity; short sequences (sl = 16) trigger catastrophic codebook collapse (1.6% utilization), whereas a moderate length (sl = 49) ensures 100% utilization and robust generalization (Table 4). Finally, scaling the latent dimension improves theoretical capacity but introduces quantization instability, evidenced by sudden utilization collapses at intermediate sizes (Table 5). Thus, dim = 256 strikes the best capacity-stability balance. Ultimately, beyond dataset quality, an effective latent action space requires two key components: robust self-supervised visual priors to capture precise spatial-temporal dynamics, and a strictly regularized quantization bottleneck to maintain stability and dense utilization.

5 Error Analysis

To provide deeper insights into the limitations of latent action representations, we conduct a fine-grained error decomposition.

5.1 The Long-Tail Dilemma and Mid-Frequency Semantic Aliasing

- Figure 5 illustrates action classification performance across class frequencies on the Composite Human dataset. While the baseline LAPA [Ye et al., 2024] model exhibits uniformly poor performance, LAPA-DINOv3 closely mirrors the continuous DINOv3 [Siméoni et al., 2025] encoder, demonstrating a direct inheritance of both its representational strengths and specific vulnerabilities. Stronger models generally outperform weaker ones across most actions, suggesting that the LARYBench provides a stable and reliable evaluation of model capabilities. As the frequency decreases, the performance gap between strong and weak models widens, indicating that strong models exhibit better generalization capabilities in long-tail scenarios.

- Table 4: Ablation on Sequence Length. Default settings are Codebook Size = 64, Latent Dimension = 256. Best results are in bold and second best are underlined.

Sequence Length

Recon Loss

Codebook Utilization (%)

Accuracy (Composite task)↑ MSE Human Robot (AgiBotWorld-Beta)↓

16 0.00908 1.6 69.23 63.33 0.79 49 0.00751 100.0 70.15 64.04 0.83 64 0.00773 79.7 71.37 64.70 0.72

- Table 5: Ablation on Latent Dimension. Default settings are Codebook Size = 64, Sequence Length = 49. Best results are in bold and second best are underlined.

Latent Dimension

Recon Loss

Codebook Utilization (%)

Accuracy (Composite task)↑ MSE Human Robot (AgiBotWorld-Beta)↓

32 0.01141 3.1 60.94 57.69 0.87 64 0.00967 100.0 66.94 63.07 0.83 256 0.00751 100.0 70.15 64.04 0.83 512 0.00732 1.6 71.25 64.97 0.80 1024 0.00670 84.4 72.55 65.78 0.81

[Figure 173]

- Figure 5: Action classification performance across the long-tail distribution of the Composite Human dataset. The background histogram indicates the number of samples per action class, sorted by descending frequency. Solid lines represent the moving average F1 scores, while transparent scatters denote exact class-wise scores.

- 5.2 Spatiotemporal Grounding and Action-Centric Attention

- Figure 6 visualizes the cross-attention maps (additional heatmap visualizations are provided in the Appendix), revealing a stark contrast in how different models ground physical interactions. First, among general visual encoders, selfsupervised models demonstrate superior spatiotemporal grounding. Notably, V-JEPA 2 [Assran et al., 2025] exhibits the most accurate attention distribution, sharply localizing on the exact interaction points between both the left and right hands and the manipulated object (bowl). DINOv3 [Siméoni et al., 2025] similarly maintains a precise, geometry-aware focus on the active end-effector. Conversely, generative priors Flux2-dev [Labs, 2024] and Wan2.2 [Wan et al., 2025] exhibit highly dispersed, unfocused attention distributions, indicating an inherent bias toward global scene understanding rather than localized physical interactions. Second, regarding LAMs versus general encoders, standard Embodied LAMs, like LAPA [Ye et al., 2024], completely fail to localize meaningful features, producing broad, uninformative attention blobs. However, our LAPA-DINOv2 effectively inherits the strong localization capabilities of its backbone. Despite the severe spatial quantization bottleneck (e.g., SL = 16, producing coarse 4 × 4 patches), it successfully anchors its attention to the active object. Finally, within the LAM architectures, our General LAMs consistently exhibit sharper object-centric grounding compared to other Embodied LAMs (e.g., UniVLA [Bu et al., 2025], villa-X [Chen

- et al., 2025]), which suffer from severe attention diffusion. Ultimately, this demonstrates that predictive failure

[Figure 174]

- Figure 6: Cross-attention heatmaps of the temporal pooler across various models for a 9-frame pour sequence. The spatial sequence length (SL) dictates visual granularity: SL = 14 ∗ 14 yield fine-grained 14 × 14 heatmaps, whereas LAMs (SL ∈ {4,16}) produce blocky 2 × 2 or 4 × 4 attention regions. Temporally, T = 9 models compute frame-by-frame attention; T = 8 models extract pairwise latent actions (leaving the 9th frame unmodified); and models with temporal compression (T ∈ {3,4}) duplicate attention maps across their corresponding frame groups.

Regression MSE↓ stride=5 stride=15 stride=30 Avg. V-JEPA 2 [Assran et al., 2025]

Model Paradigm Params(M)

303.89 0.07 0.13 0.16 0.12 DINOv3 [Siméoni et al., 2025] 303.13 0.06 0.20 0.25 0.17 Wan2.2 [Wan et al., 2025]

Semantic Encoder

704.69 0.09 0.16 0.24 0.16 FLUX.2-dev [Labs, 2024] 84.05 0.04 0.57 0.62 0.41 LAPA [Ye et al., 2024]

Pixel Encoder

343.80 0.95 0.87 0.77 0.86 UniVLA [Bu et al., 2025] 287.75 0.74 0.68 0.69 0.70 villa-X [Chen et al., 2025] 238.71 0.72 0.70 0.66 0.69

Embodied LAM

116.40 0.36 0.32 0.33 0.34 LAPA-SigLIP2 200.83 0.30 0.25 0.24 0.26 LAPA-DINOv2 473.69 0.26 0.23 0.37 0.29 LAPA-DINOv3 472.45 0.25 0.20 0.26 0.24

LAPA-MAGVIT2

General LAM

- Table 6: Ablation study on VLABench [Zhang et al., 2025b] sampling stride. We report the regression MSE with sampling strides of 5, 15, and 30, and their average. Best results are in bold and second best are underlined.

fundamentally stems from an inability to concentrate attention on the corresponding dynamically interacting objects.

- 5.3 Stride Ablation: Latent Action Spaces Encode Robust Dynamic Trajectories

To investigate the temporal robustness of various representations, we ablate the sampling stride (stride=5, 15, 30) between input frames on VLABench, as shown in Table 6. Crucially, as the stride increases, the number of actions to regress scales linearly, significantly amplifying the dimensionality and complexity of the task. Pure spatial generative encoders, such as FLUX.2-dev [Labs, 2024], excel at extremely short horizons (achieving 0.04 MSE at stride=5) but fail catastrophically under this increased temporal dimensionality (MSE spiking to 0.62 at stride=30). In contrast, Latent Action Models (LAMs)—encompassing both Embodied LAMs and our General LAMs—exhibit consistent stability across all strides. This shared characteristic proves that the latent action paradigm does not merely encode static spatial alignments; instead, it successfully captures and preserves underlying dynamic action trajectories. While traditional Embodied LAMs suffer from a high error (about 0.70 average MSE), introducing general visual priors as in our General LAMs effectively lowers this error margin while fully inheriting the paradigm’s temporal stability. In conclusion, although a performance gap remains when compared to the absolute regression accuracy of uncompressed general vision encoders, the fundamental mechanism of mapping visual observations into a latent action space provides a uniquely robust representation for long-horizon intents, proving its inherent necessity for continuous control tasks.

### 6 Conclusion

We introduce LARYBench to evaluate latent action representations across kinematic and semantic granularities. Our empirical results reveal that general visual foundation models consistently outperform specialized Embodied Latent Action Models (LAMs). Specifically, visual understanding models dominate in semantic tasks, while general visual encoders surprisingly achieve superior regression MSE without domain-specific training. This indicates that effective latent actions naturally emerge from large-scale visual pre-training, whereas specialized Embodied LAMs often suffer from representation collapse due to data scarcity or premature constraints to domain-specific low-level control. Consequently, we advocate for a paradigm shift in Vision-Language-Action (VLA) design: instead of learning action spaces from limited robotic data, future research should focus on aligning control policies with the robust feature spaces of general vision models. Addressing architectural challenges such as continuous signal decoding and feature alignment will be pivotal to unlocking these universal priors for data-efficient embodied agents.

### Acknowledgement

We hereby express our appreciation to the LongCat Team EVA Committee for their valuable assistance, guidance, and suggestions throughout the course of this work.

### References

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. arXiv preprint arXiv: 2410.11758, 2024.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions, 2025. URL https://arxiv.org/abs/2505. 06111.

Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, et al. Villa-x: enhancing latent action modeling in vision-language-action models. arXiv preprint arXiv:2507.23682, 2025.

Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, and Jiang Bian. Igor: Image-goal representations are the atomic control units for foundation models in embodied ai. arXiv preprint arXiv:2411.00785, 2024a.

Jiange Yang, Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Como: Learning continuous latent motion from internet videos for scalable robot learning, 2025. URL https://arxiv.org/abs/2505.17006.

Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. arXiv preprint arXiv: 2412.04445, 2024b.

NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv: 2503.14734, 2025.

Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

Dominik Schmidt and Minqi Jiang. Learning to act without actions. arXiv preprint arXiv:2312.10812, 2023. Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models

with latent actions. arXiv preprint arXiv:2503.18938, 2025.

Zhongwei Ren, Yunchao Wei, Xun Guo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld: Exploring knowledge learning from unlabeled videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29029–29039, 2025.

Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, Neel Joshi, and Marc Pollefeys. Holoassist: an egocentric human interaction dataset for

interactive ai assistants in the real world. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20270–20281, October 2023.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzy´nska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The "something something" video database for learning and evaluating visual common sense, 2017. URL https://arxiv.org/abs/1706.04261.

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. Taco: Benchmarking generalizable bimanual tool-action-object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21740–21751, 2024.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The epic-kitchens dataset: Collection, challenges and baselines. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):4125–4141, 2020.

Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Black Forest Labs. Flux.2. https://blackforestlabs.ai/, 2024. Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao

Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. Zuolei Li, Xingyu Gao, Xiaofan Wang, and Jianlong Fu. Latbot: Distilling universal latent actions for vision-language-

action models. arXiv preprint arXiv:2511.23034, 2025. NVIDIA et. al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, and Guanghui

Ren. Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. arXiv preprint arXiv:2505.09694, 2025.

Bahey Tharwat, Yara Nasser, Ali Abouzeid, and Ian Reid. Latent action pretraining through world modeling. arXiv preprint arXiv:2509.18428, 2025.

Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, and Jiang Bian. What do latent action models actually learn? arXiv preprint arXiv:2506.15691, 2025a.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, Yuxin Jiang, Cheng Jing, Hongyang Li, Jialu Li, Chiming Liu, Yi Liu, Yuxiang Lu, Jianlan Luo, Ping Luo, Yao Mu, Yuehan Niu, Yixuan Pan, Jiangmiao Pang, Yu Qiao, Guanghui Ren, Cheng Ruan, Jiaqi Shan, Yongjian Shen, Chengshi Shi, Mingkang Shi, Modi Shi, Chonghao Sima, Jianheng Song, Huijie Wang, Wenhao Wang, Dafeng Wei, Chengen Xie, Guo Xu, Junchi Yan, Cunbiao Yang, Lei Yang, Shukai Yang, Maoqing Yao, Jia Zeng, Chi Zhang, Qinglin Zhang, Bin Zhao, Chengyue Zhao, Jiaqi Zhao, and Jianchao Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv: 2503.06669, 2025.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334,

2022. doi:10.1109/LRA.2022.3180108.

Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11142–11152, 2025b.

Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

Yuan Zhi, Zhan Tong, Limin Wang, and Gangshan Wu. Mgsampler: An explainable sampling strategy for video action recognition. In Proceedings of the IEEE/CVF International conference on Computer Vision, pages 1513–1522, 2021.

- A Overview of the Appendix This Appendix is organized as follows:

- • Section B contains details about the composition of LARYBench and the automated data curation process;
- • Section C contains experimental details, including the compute environment, model training configurations, and hyperparameter settings;
- • Section D contains extended cross-attention heatmaps and additional case studies for different tasks.

- B Details of LARYBench

#### B.1 Composition of LARYBench

LARYBench comprises over 1.2 million short videos (totaling >1,000 hours), 620K image pairs, and 595K motion trajectories. Figure 1 illustrates the overall proportion of these diverse data sources and duration of the video clips. The benchmark evaluates representations across 151 meticulously curated unique action categories to assess both high-level semantic intent and low-level physical execution. We now provide a detailed breakdown of LARYBench by task track:

[Figure 175]

[Figure 176]

Figure 1: Left: Overall composition of the LARYBench dataset. Reg. and Classi. represent regression and classification. Ego and Exo represent Egocentric and Exocentric. AgiBot-B, EPIC, and SSv2 represent AgiBotWorld-Beta [Gao et al., 2025], EPIC-KITCHENS [Damen et al., 2020], and Something-Somethingv2 [Goyal et al., 2017]. Right: Duration distribution of all videos (Composite tasks).

Classification Track This track evaluates the classification capabilities of latent representations, encompassing both abstract behavioral semantics and fine-grained kinematic changes. While the subsets detailed below sum to more than 151 classes individually (28 Atomic Robot classes, 54 Composite Robot classes, and 123 Composite Human classes), substantial semantic overlap exists across embodiments (e.g., both human and robotic domains contain shared actions like pick and place). Consequently, the unified taxonomy encompasses exactly 151 unique action categories. To illustrate the scale and diversity of the dataset, Figure 2 presents the sample frequency across all categories. Furthermore, to highlight the rich semantic landscape of the physical interactions and manipulated objects, Figure 3 visualizes the distribution of verbs and nouns across the dataset annotations.

This classification track consists of two primary subsets:

• Kinematic Atomic Primitives (Atomic Robot, 28 classes): This subset contains 25,940 high-quality exocentric image pairs with trajectories derived from the LIBERO suite. Tasks evaluate 28 discrete kinematic primitives, including directional movements (e.g., move_top_left, move_forward) and binary end-effector states (gripper_open, gripper_close). Figure 4 illustrates representative examples of these fine-grained spatial displacements and state transitions.

|Atomic Robot|Composite Robot|
|---|---|
| | |
| | |
|Backward_DownOpenCloseForward_Right_UpForward_UpBackward_Left_DownBackward_Right_DownForward_Left_UpBackward_Right_UpForward_Left_DownBackward_Left_UpForward_Right_DownDownLeft_DownUpRight_UpRight_DownForward_DownLeft_UpBackward_UpBackward_LeftForward_RightBackward_RightForward_LeftLeftRightForwardBackward placepickgrabwithdrawreachfoldcloseopenpullpourwipehangscoo<br><br>Action C<br><br>Composit|pmixscanhammertiecarryhand_overdrop insertironhitclaptwistcutrollstampsweepwateruntiemopdipdunkflattenascendcoverpeelturnoffdumpplasterwaveturnoneraseunhangdescendgrateswirlkneadpinchfeedbouncetoastring<br><br>ategory<br><br>e Human|
| | |
| | |
| | |
| | |
|placepickreachfoldwithdrawopenpourplugunplugtwistrolldropinsertflipunfoldsqueezeshootpushwipecoverpulldrawthrowpointmeasureplayscoopcutgrabpanspraycatchapproachlower fillkickkneadteargatherflattenrevealhammerrubtopplepeelscrubstickhangshufflebrushdribblefitknotsprinkleclenchzoomoutloadzoominsearchcrackmixbreakpap<br><br>Action C<br><br>2: Distribution of the action categories of the Atomic R<br><br>[Figure 177]<br><br>(a) Verb Word Cloud<br><br>3: Word clouds illustrating the semantic diversity of<br><br><br>variety of actionable verbs indicating the interaction t representing the manipulated objects across both human and<br><br>16|ckinchclapdiscardbouncelockuprightfiddlechalkascendgratebootdigcompressclickstrainignitewringrakescatterraisebowbrewscrapeinvertpluckblotpumpstuffburypetpateraselubricatesawcombinegrindcookprintdumpshovelfloatswabwaterunboxstraightenmoptastecollidefeelswapsplashturnpagemassageexpandtapdrinkdistributescrollpedalwear<br><br>ategory<br><br>obot, Composite Human, and Composite Robot datasets.<br><br>[Figure 178]<br><br>(b) Noun Word Cloud<br><br>the LARYBench dataset annotations. (a) Highlights the ypes, while (b) showcases the extensive range of nouns robotic domains.|

- 101
- 102
- 103
- 104
- 105

Count

- 101
- 102
- 103
- 104
- 105

Count

Figure

Figure wide repr

• Composite Behaviors (Composite Human, 123 classes & Composite Robot, 54 classes): This subset evaluates abstract behavioral semantics using 1,190,820 video clips. As detailed in Table 1, within this composite video subset, the data is inherently balanced across domains, with human videos comprising 54.8% and robotic manipulations comprising 45.2%. Representative visual examples encompassing diverse human and robotic embodiments are shown in Figure 5 and Figure 6.

Regression Track This track focuses entirely on the continuous low-level execution required for robotic control. As detailed in Table 1, it comprises 595,237 image pairs and corresponding motion trajectories derived from diverse simulated and real-world robotic environments.

Table 1: Detailed composition and statistics of the LARY Benchmark dataset. The dataset is divided into two primary tracks: Classification (including both composite videos and atomic image pairs) and Regression. Proportions are calculated independently for each track.

Source Dataset

Task Allocation

Data Format

Domain Viewpoint

Count Proportion

Classification Track LIBERO [Liu et al., 2023] Robot Exo Atomic Image Pairs 25,940 2.1% EgoDex [Hoque et al., 2025] Human Ego Composite Video Clips 502,752 41.3% SSv2 [Goyal et al., 2017] Human Ego Composite Video Clips 106,911 8.8% Ego4D [Grauman et al., 2022] Human Ego Composite Video Clips 22,744 1.9% HoloAssist [Wang et al., 2023] Human Ego Composite Video Clips 10,825 0.9% EPIC-KITCHENS [Damen et al., 2020] Human Ego Composite Video Clips 5,879 0.5% TACO [Liu et al., 2024] Human Ego Composite Video Clips 3,286 0.3% AgiBotWorld-Beta [AgiBot-World-Contributors et al., 2025] Robot Ego Composite Video Clips 538,423 44.2%

Total Classification Samples: 1,216,760 Physical Regression Track

CALVIN [Mees et al., 2022] Robot Exo Regression Trajectories 209,921 35.3% VLABench [Zhang et al., 2025b] Robot Exo Regression Trajectories 112,030 18.8% RoboCOIN [Wu et al., 2025] Robot Ego Regression Trajectories 148,767 25.0% AgiBotWorld-Beta [AgiBot-World-Contributors et al., 2025] Robot Ego Regression Trajectories 124,519 20.9%

Total Regression Trajectories: 595,237

#### B.2 Additional Data Curation Details

Existing embodied datasets exhibit vast disparities in temporal boundaries and semantic annotations. To unify these heterogeneous sources, we designed an automated, multi-stage data processing engine.

Atomic Classification We identify discrete robotic arm movements within the LIBERO dataset by tracking the cumulative positional offset of the end-effector. We focus primarily on fundamental translations and binary gripper states. Specifically, we extract the precise start and end frames corresponding to the exact moments when the cumulative displacement surpasses a predefined threshold. Ground-truth action labels are subsequently assigned based on the directional shifts along the axes. This deterministic protocol yields high-quality data samples, each comprising a temporally aligned image pair paired with a kinematic action annotation.

Composite Classification The data curation pipeline operates in four primary stages. To ensure absolute annotation consistency across the entire dataset, we uniformly employ the doubao-1.5-pro-vision API for all processing steps:

- 1. Action Segmentation: Initially, we perform temporal action segmentation on the raw video sequences using the API. This process yields a massive corpus of short video clips, each paired with a brief, sentence-level action annotation describing the ongoing interaction.
- 2. Video-Description Matching: Given the inevitable noise from automated cropping and preliminary captioning, we implement a rigorous API-driven filtering protocol to sift the initial clips. Clips are retained strictly based on three criteria: (i) Temporal validity: Clip durations must fall within the [0.5s, 20s] interval. This discards clips that are too short to capture complete action semantics, or excessively long clips containing multiple entangled actions. (ii) Semantic alignment: The visual content must perfectly match the descriptive sentence, ensuring the clip captures a single, complete action rather than fragmented or disjointed sequences. (iii) Perspective consistency: We strictly retain only egocentric videos, explicitly filtering out exocentric viewpoints from mixed datasets (e.g.,

[Figure 179]

##### Figure 4: Representative examples of Kinematic Atomic Primitives from the Atomic Robot subset. The visualizations demonstrate precise, discrete end-effector movements and gripper state changes in structured exocentric environments.

[Figure 180]

##### Figure 5: Representative examples of Composite Human Behaviors.

[Figure 181]

##### Figure 6: Representative examples of Composite Robot Behaviors sourced from AgiBotWorld-Beta [AgiBot-WorldContributors et al., 2025].

Something-Somethingv2 [Goyal et al., 2017]). Concurrently, the model extracts the core action verb from the descriptive sentence. This stage outputs a refined set of video clips paired with extracted verb labels.

- 3. Video-Verb Consistency Check: To eliminate semantic ambiguity and ensure that the isolated verb accurately encapsulates the visual content, we conduct a secondary API-based verification. This step explicitly evaluates whether the single verb token alone maintains strict visual-semantic alignment with the corresponding video clip, discarding any poorly correlated pairs.
- 4. Manual Sampling Inspection: Finally, we perform a manual quality assurance review to exclude verbs that lack explicit or well-defined kinematic meanings. Action categories representing overly abstract or kinematically ambiguous operations (e.g., apply, arrange, clean) are systematically purged from the taxonomy to maintain the physical and dynamic rigor of the benchmark.

We provide the exact prompts used with doubao-1-5-thinking-vision-pro-250428 to trim and check the videos. Sampling FPS is set to 5, and the minimum number of sampling frames is 16 to ensure it can capture the precise duration of the action.

|[System Prompt] You will be provided with {n_frames} separate frames uniformly sampled from a video with their sampled time.<br><br>[Action Segmentation Prompt] <video_1>Please watch this video and find out all key actions in this video. Ensure each action is processed independently of the others. List every single action separately with its start/end timestamps. Keep descriptions concise, objective, and in English.<br><br>[Video Description Matching Prompt] <video_1>Video Analysis Task (Strict Matching):<br><br>1. Strict Verification: Does the video content EXCLUSIVELY represent the description {old_desc}?<br><br>- Match if: The video contains the described action and NOTHING else.<br><br>- Mismatch if: The video is missing parts of the action, OR contains any additional, unrelated actions before, during, or after.<br><br><br>2. Identify perspective: ’1st’ (ego) or ’3rd’ (non-ego).<br><br>3. Final Action: If and only if it’s a STRICT match, provide ONE most precise English verb defining the movement, not limited to the exact word in the description (e.g., ’take’); Otherwise, return ’None’. Return ONLY JSON: {{"perspective": "1st/3rd", "action": "verb/None"}}<br><br><br>[Video-Verb Consistency Check Prompt] <video_1>Please watch this video and determine if the action {action} is performed. Output Yes only if the action is clearly identifiable and the video content does not contradict the label. Output No if the action is missing, incorrect, or represents a different verb entirely. Return ONLY JSON: {{"output": "Yes/No"}}<br><br>|
|---|

Regression We obtain the requisite image pairs and action trajectories by sampling observations at a fixed temporal stride. The start and end frames are paired with the continuous sequence of absolute kinematic actions occurring strictly between them.

### C Experimental Details of General LAMs

#### C.1 Models and Settings

To rigorously evaluate the impact of different visual priors on latent action learning, we design four variants of General Latent Action Models (General LAMs): LAPA-DINOv2, LAPA-DINOv3, LAPA-SigLIP2, and LAPA-MAGVIT2. Across all variants, the weights of the pre-trained visual backbones are kept frozen.

For the first three understanding-oriented models (DINOv2 [Oquab et al., 2023], DINOv3 [Siméoni et al., 2025], and SigLIP2 [Tschannen et al., 2025]), we substitute the pixel-level inputs of the original LAPA VQ-VAE architecture with continuous feature embeddings extracted from the penultimate layers of their respective vision encoders. Consequently, their training objective is formulated to reconstruct these high-level latent features rather than raw pixels. Conversely,

LAPA-MAGVIT2 operates as a generative-based variant, where the reconstruction target remains in the visual pixel space.

#### C.2 Training Configurations and Hyperparameters

All General LAM variants are trained on an expansive internal video dataset comprising over 2 million frames. This corpus is meticulously curated to encompass a diverse mixture of human demonstrations, robotic manipulations, and general environmental dynamics. During the training phase, we extract the start and end frames by sampling with a randomized temporal stride within a predefined range. This stochastic sampling strategy prevents overfitting to specific frame rates and forces the latent representations to capture robust temporal dynamics across varying motion speeds. All training workloads were executed utilizing 8 NVIDIA H800 (141GB) GPUs.

The input image resolution is standardized to 224 × 224. The latent action quantization modules consistently employ a symmetric spatial-temporal transformer architecture. Both the spatial and temporal encoders are configured with a depth of 4 layers, utilizing 16 attention heads with a dimension of 64 per head. All General LAM variants are trained for exactly 100,000 steps and share a foundational quantization configuration, which includes a quantization dimension of 32, a codebook size of 8, and a sequence length of 16. Regarding the encoder-specific spatial parameters, both LAPA-DINOv3 and LAPA-DINOv2 process features with an input dimension of 1024; however, they utilize patch sizes of 16 and 14, respectively. LAPA-SigLIP2 operates on an input feature dimension of 768 with a patch size of 16. LAPA-MAGVIT2 is consistently configured with a patch size of 16, processing an input feature dimension of 18.

### D Additional Visualizations and Case Studies

#### D.1 Cross-Embodiment Gap Across Different Model Paradigms

To further investigate the morphological domain gap between human and robotic embodiments, we analyze how different representation paradigms handle shared semantic actions (e.g., pick, place, push, pull) present in both the Composite Human and Composite Robot subsets. Although these actions share identical high-level intentions, their visual appearances diverge significantly due to differences in end-effector morphologies.

As illustrated in Figure 7, the cross-domain gaps reveal four critical insights:

- • General LAMs Exhibit a Human-Centric Preference: While our General LAMs (e.g., LAPA-DINOv3, LAPASigLIP2) successfully elevate the overall F1 bounds compared to standard Embodied LAMs, they demonstrate a distinct preference for human embodiments. For the majority of actions, such as twist, drop, and roll, the human F1 scores significantly outstrip their robotic counterparts (e.g., for roll, LAPA-DINOv3 achieves H: 0.90 vs. R: 0.04). This phenomenon suggests the General LAMs effectively learn and capture a much richer set of human behaviors, likely benefiting from the extensive human-centric priors embedded within the frozen visual backbones.
- • Understanding Encoders Demonstrate Balanced Cross-Domain Robustness: In stark contrast to the LAMs, continuous foundational vision encoders, particularly DINOv3, emerge as the most morphologically balanced paradigms. DINOv3 maintains highly comparable and competitive F1 scores across both domains for a wide array of categories (e.g., place, fold, scoop). This demonstrates that robust visual priors possess a powerful, domain-agnostic capability to understand interactions, irrespective of the specific end-effector morphology.
- • Embodied LAMs Show a Marginal Edge on Robotic Domains: While specialized Embodied LAMs (e.g., LAPA, UniVLA) generally struggle with low overall precision across both domains, they exhibit a slight performance edge on robotic actions compared to human ones. This marginal robotic preference is fundamentally because these architectures are primarily trained on robotic manipulation datasets, thereby lacking the diverse human-centric exposure required for broader cross-domain generalization.
- • Data Imbalance Drives Robotic Bias in Specific Actions: Interestingly, a few specific actions, namely grab and mix, consistently show a distinct bias towards the robotic domain across all models. This anomalous robotic preference is primarily attributed to the underlying dataset distribution. As previously illustrated in Figure 2), these specific interactions are substantially overrepresented in the robotic training sets compared to the human datasets, structurally skewing the learned representations even for general vision models.

#### D.2 Spatiotemporal Grounding via Attention Visualization

Complementing the analysis in Section 5.2 of the main text, we provide an extended gallery of cross-attention visualizations.

###### Cross-Domain Performance Gap (F1)

Semantic Encoder Pixel Encoder

Embodied LAMs General LAMs

LAPA-MAGVIT2LAPA-SigLIP2 LAPA-DINOv2 LAPA-DINOv3

FLUX.2-dev

2 Wan2.2

DINOv3 V-JEPA

LAPA UniVLA villa-X

- H:.79 R:.76

H:.84 R:.78

H:.71 R:.71

H:.47 R:.49

H:.18 R:.35

H:.18 R:.26

H:.17 R:.41

H:.63 R:.38

H:.59 R:.39

H:.59 R:.43

H:.68 R:.38

H:.74

- R:.60

H:.80 R:.63

H:.65 R:.57

H:.42 R:.39

H:.21 R:.27

H:.23 R:.22

H:.20 R:.31

H:.58 R:.31

H:.52 R:.31

H:.55 R:.31

H:.62 R:.28

H:.18 R:.58

H:.19 R:.59

H:.11 R:.54

H:.06 R:.37

H:.00 R:.27

H:.04 R:.21

H:.03 R:.29

H:.09 R:.29

H:.05 R:.28

H:.06 R:.30

H:.07 R:.27

H:.80 R:.55

H:.82 R:.56

H:.70 R:.52

H:.38 R:.29

H:.25 R:.12

H:.34 R:.16

H:.34 R:.21

H:.62 R:.19

H:.61 R:.17

H:.62 R:.25

H:.69 R:.14

H:.88 R:.83

H:.90 R:.83

H:.85 R:.80

H:.76 R:.60

H:.22 R:.20

H:.33 R:.22

H:.29 R:.33

H:.76 R:.34

H:.67 R:.35

H:.64 R:.41

H:.77 R:.34

H:.69

- R:.68

H:.72 R:.70

H:.59 R:.63

H:.28 R:.29

H:.11 R:.16

H:.22 R:.20

H:.23 R:.28

H:.52 R:.21

H:.49 R:.23

H:.51 R:.32

H:.57 R:.19

H:.68 R:.83

H:.72 R:.83

H:.54 R:.79

H:.35 R:.54

H:.07 R:.08

H:.11 R:.16

H:.11 R:.26

H:.45 R:.24

H:.36 R:.22

H:.37 R:.33

H:.47 R:.20

- H:.80 R:.73

H:.85 R:.75

H:.70 R:.68

H:.53 R:.43

H:.07 R:.02

H:.24 R:.09

H:.19 R:.17

H:.62 R:.11

H:.60 R:.10

H:.62 R:.20

H:.67 R:.10

H:.49 R:.63

H:.61 R:.63

H:.34 R:.63

H:.09 R:.49

H:.06 R:.07

H:.08 R:.14

H:.12 R:.19

H:.33 R:.20

H:.33 R:.21

H:.38 R:.28

H:.39 R:.15

H:.79 R:.54

H:.81 R:.54

H:.72 R:.55

H:.53 R:.33

H:.05 R:.00

H:.11 R:.02

H:.11 R:.05

H:.60 R:.03

H:.51 R:.01

H:.51 R:.08

H:.64 R:.01

- H:.67

R:.61

H:.72 R:.63

H:.57 R:.61

H:.44 R:.43

H:.07 R:.01

H:.16 R:.07

H:.11 R:.07

H:.53 R:.19

H:.51 R:.03

H:.52 R:.16

H:.60 R:.02

H:.94 R:.89

H:.94 R:.91

H:.93 R:.85

H:.89 R:.69

H:.18 R:.02

H:.37 R:.06

H:.44 R:.25

H:.88 R:.15

H:.86 R:.07

H:.85 R:.28

H:.90 R:.04

H:.81 R:.88

H:.85 R:.92

H:.75 R:.87

H:.62 R:.66

H:.04 R:.04

H:.09 R:.13

H:.13 R:.24

H:.69 R:.26

H:.66 R:.27

H:.68 R:.35

H:.73 R:.15

- H:.51 R:.31

H:.53 R:.28

H:.45 R:.26

H:.34 R:.20

H:.02 R:.00

H:.05 R:.02

H:.05 R:.03

H:.33 R:.02

H:.25 R:.01

H:.20 R:.01

H:.33 R:.01

H:.61 R:.06

H:.64 R:.07

H:.51 R:.15

H:.36 R:.03

H:.02 R:.00

H:.09 R:.00

H:.13 R:.01

H:.45 R:.00

H:.40 R:.00

H:.44 R:.00

H:.50 R:.00

H:.68 R:.69

H:.72 R:.71

H:.66 R:.69

H:.58 R:.66

H:.02 R:.01

H:.23 R:.07

H:.25 R:.23

H:.63 R:.16

H:.60 R:.18

H:.63 R:.18

H:.63 R:.17

H:.71 R:.81

H:.77 R:.81

H:.62 R:.79

H:.58 R:.64

H:.01 R:.00

H:.00 R:.07

H:.12 R:.09

H:.65 R:.11

H:.51 R:.13

H:.59 R:.25

H:.61 R:.10

H:.79

- R:.91

H:.85 R:.92

H:.64 R:.90

H:.59 R:.80

H:.00 R:.02

H:.06 R:.04

H:.09 R:.13

H:.57 R:.06

H:.55 R:.08

H:.56 R:.14

H:.63 R:.03

H:.49

- R:.72

H:.64 R:.74

H:.03 R:.71

H:.00 R:.54

H:.00 R:.01

H:.00 R:.06

H:.00 R:.14

H:.03 R:.15

H:.00 R:.03

H:.09 R:.11

H:.00 R:.02

H:.96 R:.92

H:.94 R:.89

H:.90 R:.93

H:.90 R:.93

H:.01 R:.05

H:.11 R:.13

H:.04 R:.35

H:.79 R:.56

H:.74 R:.38

H:.75 R:.55

H:.82 R:.45

H:.52

- R:.95

H:.61 R:.96

H:.08 R:.94

H:.04 R:.85

H:.00 R:.07

H:.00 R:.21

H:.00 R:.20

H:.08 R:.26

H:.04 R:.22

H:.00 R:.45

H:.17 R:.31

H:.43 R:.13

H:.48 R:.18

H:.40 R:.22

H:.27 R:.05

H:.01 R:.00

H:.05 R:.00

H:.03 R:.00

H:.40 R:.00

H:.30 R:.00

H:.34 R:.00

H:.37 R:.00

H:.29 R:.67

H:.33 R:.86

H:.29 R:.86

H:.26 R:.40

H:.01 R:.00

H:.01 R:.00

H:.00 R:.00

H:.17 R:.00

H:.12 R:.00

H:.07 R:.00

H:.17 R:.00

H:.55

- R:.96

H:.50 R:.96

H:.00 R:.95

H:.00 R:.64

H:.00 R:.00

H:.00 R:.05

H:.00 R:.02

H:.00 R:.01

H:.00 R:.08

H:.00 R:.37

H:.00 R:.07

H:.75 R:.94

H:.80 R:.94

H:.00 R:.94

H:.00 R:.78

H:.00 R:.00

H:.00 R:.04

H:.00 R:.08

H:.00 R:.04

H:.00 R:.03

H:.00 R:.20

H:.00 R:.02

H:.59

- R:.73

place

pick

grab

[Figure 182]

reach

fold

0.75

withdraw

open

pour

0.50

pull

twist

drop

DomainGap(ΔF1=Robot-Human)

0.25

roll

wipe

insert

0.00

cover

scoop

hang

−0.25

cut

mix

hammer

−0.50

clap

flatten

knead

−0.75

water

mop

H:.25 R:.74

H:.06 R:.37

H:.00 R:.00

H:.01 R:.07

H:.01 R:.05

H:.19 R:.02

H:.13 R:.02

H:.09 R:.09

H:.25 R:.04

H:.71 R:.78

peel

H:.27 R:.30

H:.31 R:.29

H:.26 R:.34

H:.12 R:.11

H:.00 R:.01

H:.16 R:.06

H:.19 R:.03

H:.33 R:.01

H:.32 R:.01

H:.43 R:.13

H:.30 R:.01

ascend

Figure 7: Cross-domain performance gap analysis across shared action categories. The heatmap visualizes the F1 score discrepancy (∆F1 = Robot − Human) between human and robot domains. Red regions highlight superior performance on human actions, whereas blue regions indicate better performance on the corresponding robot actions. Absolute F1 scores for Human (H) and Robot (R) are explicitly reported within each cell.

[Figure 183]

- Figure 8: A case for the action catch.

[Figure 184]

- Figure 9: A case for the action compress.

[Figure 185]

Figure 10: A case for the action plug.All models failed.

[Figure 186]

Figure 11: A case for the action brew.

