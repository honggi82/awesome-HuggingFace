# arXiv:2512.16760v2[cs.RO]4Jan2026

[Figure 1]

## Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future

Tianshuai Hu1, , Xiaolu Liu2, , Song Wang2, , Yiyao Zhu1, , Ao Liang3, , Lingdong Kong3, , Guoyang Zhao4, Zeying Gong4, Jun Cen5, Zhiyu Huang6, Xiaoshuai Hao7, Linfeng Li3, Hang Song8, Xiangtai Li9, Jun Ma1,4, Shaojie Shen1, Jianke Zhu2, Dacheng Tao9, Ziwei Liu9, , Junwei Liang1,4,

[Figure 2]

1HKUST 2Zhejiang University 3National University of Singapore 4HKUST(GZ) 5DAMO Academy, Alibaba 6University of California, Los Angeles 7Xiaomi EV 8Xi’an Jiaotong University 9Nanyang Technological University, Singapore

[Figure 3]

WorldBench Team

[Figure 4]

[Figure 5]

[Figure 6]

Equal Contributions Project Lead Corresponding Authors

Autonomous driving has long relied on modular “Perception-Decision-Action” pipelines, where handcrafted interfaces and rule-based components often break down in complex or long-tailed scenarios. Their cascaded design further propagates perception errors, degrading downstream planning and control. Vision-Action (VA) models address some limitations by learning direct mappings from visual inputs to actions, but they remain opaque, sensitive to distribution shifts, and lack structured reasoning or instruction-following capabilities. Recent progress in Large Language Models (LLMs) and multimodal learning has motivated the emergence of Vision-Language-Action (VLA) frameworks, which integrate perception with language-grounded decision making. By unifying visual understanding, linguistic reasoning, and actionable outputs, VLAs offer a more interpretable, generalizable, and human-aligned paradigm for driving policies. This work provides a structured characterization of the emerging VLA landscape for autonomous driving. We trace the evolution from early VA approaches to modern VLA frameworks and organize existing methods into two principal paradigms: End-to-End VLA, which integrates perception, reasoning, and planning within a single model, and Dual-System VLA, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners). Within these paradigms, we further distinguish subclasses such as textual vs. numerical action generators and explicit vs. implicit guidance mechanisms. We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity. Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems.

[Figure 7]

Project Page: https://worldbench.github.io/vla4ad GitHub Repo: https://github.com/worldbench/awesome-vla-for-ad HuggingFace Leaderboard: https://huggingface.co/spaces/worldbench/vla4ad

[Figure 8]

[Figure 9]

1 Introduction

The pursuit of fully autonomous driving (AD) has long been a central goal in AI and robotics [37, 72, 122]. Conventional AD systems typically adopt a modular “Perception-Decision-Action” pipeline, where mapping [82, 83], object detection [121, 148, 152, 199], motion prediction [55, 116, 307], and trajectory planning [150, 308] are developed and optimized as separate components. While this design has achieved strong performance in structured environments, its reliance on hand-crafted interfaces and rules limits

Datasets and Benchmarks (§5)

Vision-Action Models (§3)

Conclusion (§7)

Introduction (§1)

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Trajectory Action

[Figure 14]

Input Modalities

[Figure 15]

[Figure 16]

|Action Prediction Model| |
|---|---|
| | |

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|Generative Model|
|---|

##### Preliminary

[Figure 21]

[Figure 22]

- (Section 2)

[Figure 23]

[Figure 24]

Vision-Action Models

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- (Section 3)

[Figure 29]

Vision-LanguageAction Models

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

- (Section 4)

[Figure 36]

[Figure 37]

Control Language

[Figure 38]

[Figure 39]

Preliminary (§2)

Vision-Language-Action Models (§4)

Challenges and Future Directions (§6)

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Action-Only Models Perception-Action Models

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

End-to-End VA World Models

[Figure 51]

[Figure 52]

[Figure 53]

Textual Action Models Numerical Action Models

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

End-to-End VLA

[Figure 58]

[Figure 59]

Explicit Action Guidance Implicit Representations Transfer

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Dual System VLA

[Figure 64]

Datasets and Benchmarks

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

- (Section 5) Open-Loop & Closed-Loop

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Challenges and Future Directions

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- (Section 6)

Guidance Sensors

Latent Feature Language

Vehicle State

[Figure 84]

[Figure 85]

Image-Based World Models Occupancy-Based World Models Latent-Based World Models

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

nuScenes NAVSIM navtest Bench2Drive

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Open-Loop Closed-Loop

[Figure 99]

###### Challenges:

[Figure 100]

[Figure 101]

Foundation Model Construction

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Model Interpretability

Computation Efficiency Generalization High-Quality

Long-Horizon Coherence

Training Data

Future Directions:

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Injecting World Knowledge

Continual Learning

Standardized Evaluations

Human-AI Interaction

Combined with World Model

- Figure 1 Outline. This work aims to provide a structured roadmap of the VLA paradigm for autonomous driving. We begin with Preliminary Foundations (Section 2), which formalize the general formulation of VLA models and detail their three core components: the multi-modal input modalities, the VLM backbone, and the action prediction head. It then traces the evolution from VA Models (Section 3), which directly map perception to control, towards VLA Models (Section 4), which incorporate language-grounded reasoning and interpretable decision-making. We further categorize VLA architectures into two major paradigms – End-to-End VLA (Section 4.1) and Dual-System VLA (Section 4.2) – that differ in their integration of vision, language, and action modules. Next, we review Datasets & Benchmarks (Section 5) that enable both open-loop and closed-loop evaluation of driving intelligence. Finally, we discuss Challenges & Future Directions (Section 6), highlighting interpretability, reasoning, and human-AI interaction as central themes driving the next generation of VLA-based autonomous driving research.

adaptability in complex [84, 85, 220], dynamic [120, 274, 275], and long-tailed scenarios [73, 197, 302]. Moreover, the sequential cascade is prone to cross-stage error propagation, where perception noise is amplified by downstream reasoning and control, compromising stability and safety.

To mitigate these issues, research has increasingly moved toward end-to-end autonomous driving, where Vision-Action (VA) models directly map raw sensory inputs to control commands or trajectory waypoints using imitation [42, 198, 204] and reinforcement learning [67, 150, 281]. Early systems such as ALVINN [203] and ChauffeurNet [9] demonstrated the viability of behavior cloning at scale. Subsequent advances introduced more expressive architectures: TransFuser [39] exploited transformer-based multimodal fusion, UniAD [93] unified perception and planning, VAD [107] leveraged vectorized scene representations, DriveTransformer [104] explored scalable transformer backbones, and DiffusionDrive [155] applied generative modeling to multi-modal trajectory prediction. Collectively, these VA models show that complex driving policies can be learned directly from data, laying the foundation for modern end-to-end AD systems.

Despite these successes, VA models exhibit fundamental limitations. They largely behave as “black boxes”, offering limited interpretability in safety-critical settings [27, 112, 123, 124, 189, 309]. Their generalization remains fragile under rare or long-tail scenarios that are underrepresented in training [2, 27, 68, 110, 145, 309]. By directly mapping perception to low-level actions, they lack chain-of-thought (CoT) reasoning and contextual

deliberation [27, 69, 110, 286], limiting their ability to resolve ambiguous or multi-stage interactions. Moreover, their focus on visual inputs prevents them from incorporating high-level plans or human instructions in natural language, leaving a gap in human-vehicle interaction [110, 197, 222, 320].

The emergence of Large Language Models (LLMs) and Large Multimodal Models (LMMs) has catalyzed a new paradigm: Vision-Language-Action (VLA) models [180, 217, 336]. VLA models couple a Vision-Language Model (VLM) backbone with an action-prediction head, enabling direct mapping from multimodal inputs (vision + language) to executable driving actions. By jointly modeling perception, language understanding, and decision-making, VLA frameworks aspire to provide human-like reasoning, interpretability, and instructionfollowing [66, 71, 340]. Initial explorations such as DriveMLM [46] and GPT-Driver [184] introduced language modules into driving pipelines for high-level decision understanding, paving the way for more integrated designs. Later systems advanced toward closed-loop and reasoning-centric VLA models: LMDrive [222] achieved language-guided closed-loop driving, DriveLM [225] enabled structured reasoning via visual question answering, and DriveGPT4 [287] provided natural-language rationales for decisions. Recent works further investigate tightly coupled reasoning and control, including AutoVLA [340] with fast/slow thinking and GRPO-based optimization [224], and SimLingo [213], which explicitly studies language-action alignment.

End-to-End VLA models, however, must simultaneously reason and act in real time, creating challenges for latency and safety. This has led to Dual-System VLA designs, where high-level decision making is separated from low-level trajectory execution. DriveVLM [249] generates textual rationales or decisions with a VLM while relying on a classical planner for trajectories. VLP [197] tokenizes waypoints and value maps to produce planning-aware latent actions, and Diff-VLA [106] synthesizes language-guided trajectories refined by rule- or optimization-based controllers. InsightDrive [227] integrates causal language reasoning with MPC, assigning why to the VLM and how to the planner.

Together, these developments signal a paradigm shift from perception-driven pipelines toward systems that jointly reason, understand, and act. Given the rapid evolution of this field, there is a need to consolidate its conceptual foundations, clarify architectural trends, and provide a structured analysis of emerging directions.

Contributions. This work provides a comprehensive characterization of VLA models for autonomous driving. Specifically:

- • We chart the evolution from precursor VA models (Section 3) to modern VLA frameworks (Section 4), providing historical context and clarifying the motivations behind this paradigm shift.
- • We propose a taxonomy that categorizes VLA architectures into End-to-End (Section 4.1) and DualSystem (Section 4.2) designs, and compare their principles, advantages, and limitations.
- • We present an organized synthesis of datasets and evaluation benchmarks relevant to VLA-based driving (Section 5), facilitating consistent and meaningful comparisons.
- • We identify key challenges in real-world VLA deployment and outline future research directions (Section 6) to guide progress toward safer and more reliable autonomous systems.

Scope. This work differs from prior studies on VLA models [110, 180, 217, 336] through its domain-specific focus, historical framing, and architectural depth. 1Domain-specific focus. Unlike broader analyses that span robotics or embodied AI [179, 223], our work focuses exclusively on autonomous driving, allowing a finegrained analysis of driving-specific challenges, dataset characteristics, and safety requirements. 2Historical and conceptual continuity. We adopt a “Past-Present-Future” narrative that traces the path from early VA models to modern VLA frameworks, emphasizing the motivations and technical lineage underlying the integration of language grounding into perception and control. 3Fine-grained architectural taxonomy. Unlike prior highlevel overviews [44, 110, 301, 338], we propose a hierarchical taxonomy that differentiates End-to-End and Dual-System VLA models and analyzes how they organize perception, reasoning, and control.

Through this combination of domain specificity, historical depth, and structured architectural analysis, we aim to provide a comprehensive and insightful reference for VLA research in autonomous driving.

Organization. The remainder of this paper is organized as follows. Section 2 introduces the preliminary foundations of VLA frameworks. Section 3 outlines the evolution of VA models. Section 4 presents our

- §3.1.1

- §3.1.2

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

[Figure 121]

[Figure 122]

Action-Only Models

###### Roach

###### TransFuser

###### GRI

###### BEVPlanner

###### RAD

###### TCP

###### Latent-DRL

LBC

[Figure 123]

[arXiv’25]

[ICCV’21]

[NeurIPS’22]

[TPAMI’23]

[Robotics’23]

[CVPR’24]

[CVPR’20]

[CoRL’20]

WorldModelEnd-to-EndVLADual-systemVLAEnd-to-EndModel

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

PerceptionAction Models

###### PARA-Drive

###### SparseAD

###### DriveTransformer

###### DiffusionDrive

###### VAD

###### OccNet

###### UniAD

[Figure 133]

[Figure 134]

[Figure 135]

[CVPR’24]

[arXiv’24]

[ICLR’25]

[CVPR’25]

[ICCV’23]

[ICCV’23]

[CVPR’23]

[Figure 136]

#### VA

[Figure 137]

- §3.2.1

- §3.2.2

- §3.2.3

- §4.1.1

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

Image-Based World Models

###### Imagine-2-Drive

###### Drive-WM

###### DrivingGPT

###### DriveDreamer

###### GenAD

###### DrivingWorld

###### Epona

Vision-Action Models (Sec 3)

[Figure 148]

[Figure 149]

[Figure 150]

[IROS’25]

[CVPR’24]

[ICCV’25]

[ECCV’24]

[arXiv’24]

[ICCV’25]

[CVPR’24]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

###### DFIT-OccWorld

###### OccWorld

###### RenderWorld

###### DynamicCity

###### Drive-OccWorld

Occupancy World Models

###### NeMo

[Figure 159]

[Figure 160]

[Figure 161]

[arXiv’24]

[arXiv’24]

[ECCV’24]

[ICLR’25]

[AAAI’25]

[ECCV’24]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

###### Covariate-Shift

###### WoTE

###### LAW

###### SSR

###### World4Drive

Latent World Models

###### Echo-Planning

###### SeerDrive

[Figure 170]

[Figure 171]

[Figure 172]

[arXiv’24]

[ICCV’25]

[ICLR’25]

[ICLR’25]

[ICCV’25]

[NeurIPS’25]

[arXiv’25]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Textual Action Generator

[Figure 181]

[Figure 182]

###### EMMA

###### DriveLM

###### LLaDA

###### OmniDrive

###### DriveGPT-4

###### DriveMLM

###### GPT-Driver

[Figure 183]

[TMLR’25]

[CVPR’24]

[CVPR’25]

[ECCV’24]

[RA-L’24]

[arXiv’23]

[NeurIPSW’23]

[Figure 184]

g

#### VLA

[Figure 185]

- §4.1.2

- §4.2.1

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Numerical Action Generator

[Figure 193]

###### LMDrive

###### LINGO-2

###### ORION

###### AutoVLA

###### OpenDriveVLA

###### DriveMoE

###### SimLingo

Vision-Language -Action Models (Sec 4)

[Figure 194]

[CVPR’24]

[arXiv’24]

[ICCV’25]

[NeurIPS’25]

[arXiv’25]

[arXiv’25]

[CVPR’25]

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

Explicit Action Guidance

###### DriveVLM

###### Senna

###### LeapAD

###### CoReVLA

###### DualAD

###### DME-Driver

###### DiffVLA

###### SOLVE

[Figure 206]

[CoRL’24]

[arXiv’24]

[arXiv’24]

[arXiv’25]

[arXiv’24]

[AAAI’25]

[arXiv’25]

[arXiv’25]

[Figure 207]

[Figure 208]

§4.2.2

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

###### VLP

Implicit Transfer

###### DiMA

[CVPR’24] VLM-AD [CoRL’25]

###### ReCoDrive

###### InsightDrive

###### NetRoller

###### VERDI

###### VLM-E2E

[Figure 216]

[CVPR’25]

[arXiv’25]

[arXiv’25]

[arXiv’25]

[arXiv’25]

[arXiv’25]

- Figure 2 Summary of representative VA and VLA models from existing literature, spanning End-to-End Models, World Models, Dual-Systems, etc. For the complete list of related approaches and the discussions on their specifications, configurations, and technical details, kindly refer to Section 3 and Section 4, respectively.

taxonomy and analysis of VLA architectures. Section 5 summarizes datasets and benchmarks. Section 6 discusses remaining challenges and future directions. Section 7 concludes this work.

### 2 Preliminary Foundations

Vision-Language-Action (VLA) frameworks [2, 179, 339, 340] leverage large Vision-Language Models (VLMs) [34, 40, 129, 258] to interpret complex driving scenes and produce executable actions. A typical formulation can be expressed as:

at = H(F(x|θ)) , (1)

where x denotes multimodal inputs at timestamp t, F(·) is a VLM backbone parameterized by θ, and H(·) is an action-generation head. This section introduces these three components: the input modalities (x), the VLM backbone (F), and the action prediction head (H).

- 2.1 Input Modalities

The input x aggregates heterogeneous signals that describe the external environment and the ego-vehicle state [16, 234, 269]. These inputs can be grouped into four categories: sensor observations, latent scene representations, language instructions, and proprioceptive states.

- 2.1.1 Sensor Inputs Sensor inputs include raw or preprocessed data directly obtained from vehicle-mounted sensors [23, 24, 140].

- • Visual Images. Surround-view RGB images that offer dense semantic information: ximg ∈ RN

c×H×W×3, where Nc is the number of cameras (e.g., 6 to 8), and H, W are the height and width of each image.

- • LiDAR Point Clouds. A sparse or dense set of 3D points representing the environment geometry:

p×D, D ≥ 4, where Np is the number of points, and D includes dimensions such as x,y,z, velocity, and intensity.

xlidar ∈ RN

- 2.1.2 Latent Representations Multiple VLA systems operate on intermediate spatial representations that fuse multimodal sensor inputs.

- • Bird’s-Eye View (BEV) Features. Top-down view representation, often generated by fusing camera or

LiDAR data [148, 154, 167]: xbev ∈ RC×H

bev×Wbev, where C is the number of feature channels, and Hbev,Wbev are the spatial dimensions of the BEV grid.

- • Occupancy Grids. 3D volumetric representation predicting occupancy and semantics for each spatial

location [166, 260, 267, 284]: xocc ∈ RC

occ×X×Y ×Z, where X,Y,Z are the spatial resolution of the 3D grid, and Cocc denotes the number of occupancy feature channels (e.g., occupancy, flow and semantics).

- 2.1.3 Language Inputs

To enable VLA capabilities, the model also receives high-level textual instructions or task descriptions [65, 76, 287]. It is composed of a sequence of tokens representing the driving task or goal (e.g., “turn left at the next intersection”): xlang ∈ ZT or xlang ∈ RT×D

emb, where T is the sequence length, and Demb is the embedding dimension (if token embeddings are used).

- 2.1.4 Vehicle State Information

There is also proprioceptive information describing the current dynamic state of the ego-vehicle [16, 149]: xstate ∈ RD

state, where Dstate is the dimension of the state vector, including speed, acceleration, steering angle, yaw rate, turn indicator status, etc.

- 2.2 VLM Backbone (F)

The VLM backbone F(·) is the core reasoning engine of the system. It is typically a large vision language model [5, 8, 34, 159, 258, 343]. Its primary role is to fuse the diverse input modalities into a single, powerful latent representation. It consists of a vision encoder (e.g., a Vision Transformer, ViT) [210] to process visual inputs and an LLM decoder that conditions its generation on the visual features. A bridge network [34, 159] or unified multimodal token modelling mechanism [5, 258] is used to align the vision features with the language embeddings. VLM can directly generate the actions or provide the guidance for another action expert to develop more robust results.

- 2.2.1 VLM for Direct Action Generation (Single-System)

In this paradigm, the VLM directly emits actions through its language head [46, 98, 184, 225] or a small attached head [213, 222, 288]. This fully end-to-end design exploits the VLM’s reasoning capabilities to map from visual/language inputs to executable controls.

- 2.2.2 VLM for Guidance Generation (Dual-System)

Alternatively, the VLM functions as a high-level reasoning module that produces intermediate guidance—textual rationales [80, 207, 249] or structured latent intents [145, 197], which a downstream planner converts into low-level actions [257]. This “slow thinking + fast execution” architecture improves interpretability and enables planners to enforce physical feasibility and safety constraints.

- 2.3 Action Prediction Head (H)

###### The head H(·) converts the VLM latent representation into action outputs. Consistent with the taxonomy used in existing literature, we categorize action heads into four types based on their output formulation and generation mechanism: Language Head (LH), Regression (REG), Trajectory Selection (SEL), and Trajectory Generation (GEN).

- • Language Head (LH). This design directly utilizes the VLM’s inherent text-generation capabilities to produce actions in the language space. The head is typically the language modeling head of the VLM, trained to output either free-form textual commands (e.g., “turn left”) [287] or a sequence of discretized action tokens [340]. The model autoregressively predicts these tokens, which are subsequently parsed into executable signals. This approach is widely adopted in textual action generators like DriveMLM [46] and DriveGPT4 [287].
- • Regression (REG). This formulation employs a decoder structure followed by a regressor (typically a Multi-Layer Perceptron) to directly predict continuous values. Unlike language heads, it avoids discretization by mapping the latent features aggregated via Transformers or GRUs to specific numerical outputs such as steering angles, throttle/brake values, or trajectory waypoints. Representative methods using this deterministic approach include LMDrive [222] and DriveGPT4-V2 [288].
- • Trajectory Selection (SEL). Instead of directly regressing a single path, this head evaluates a set of candidate trajectories and selects the optimal one based on a learned cost function or scoring mechanism. The model typically generates or samples a diverse set of dynamically feasible trajectories and uses the latent representation to predict the cost or probability for each candidate. This approach, utilized by methods like WoTE [144] and SeerDrive [315], ensures that the final output adheres to kinematic constraints by selecting from pre-defined candidates.
- • Trajectory Generation (GEN). This generative formulation synthesizes actions through probabilistic modeling, most notably using diffusion models or variational autoencoder [106, 155, 332]. Starting from noise, the head iteratively refines the trajectory sample conditioned on the VLM latent state and optionally language instructions. This allows the model to capture the multi-modality and uncertainty of future distributions. Prominent examples include ORION [66] and DiffVLA [106].

- 2.4 Action in Driving

In the context of autonomous driving, particularly for models like VLAs, the action space defines the set of possible outputs the model can generate to control the vehicle. The choice of action representation is a fundamental design decision that dictates how the model’s reasoning is translated into physical motion. We outline three primary paradigms for action space representation below.

- 2.4.1 Discrete Trajectory Representations

This paradigm represents the vehicle’s intended future path as a finite sequence of spatial waypoints [167]. Each waypoint is a spatial coordinate that the vehicle is expected to reach at a specific future time step. This representation allows for explicit geometric path planning and trajectory optimization. The action, at, formulated at the current time t, is a set of Φ future waypoints:

at = {(xi,yi)}Φi=1, where (xi,yi) ∈ R2 . (2)

Here, Φ is the prediction horizon (the total number of future steps), and each (xi,yi) is a coordinate in a 2D Cartesian plane representing the target position at step i.

- 2.4.2 Continuous Trajectory Representations

Instead of discrete points, this approach parameterizes the vehicle’s motion as a continuous function over a future time horizon [163]. The trajectory is typically defined by functions that govern the vehicle’s longitudinal and lateral motion, such as speed and turning radius. The action, at, is defined by these continuous functions over a time interval [0,T]:

at = (v(t),κ(t)), for t ∈ [0,T] . (3)

In this formulation, v(t) represents the vehicle’s speed profile, and κ(t) represents its curvature profile over the future time horizon T. This inherently captures the continuous nature of vehicle dynamics.

- 2.4.3 Direct Control Representations

This paradigm involves the direct output of low-level vehicle control commands that are immediately sent to the vehicle’s actuators [288]. These outputs typically consist of continuous signals for steering, acceleration, and braking control. The values are often normalized and constrained to lie within the vehicle’s physical operational ranges. The action vector, at, represents control signals for a specific time step t:

at = (δt,τt,βt) , (4)

where δt is the steering angle, τt is the throttle input, and βt is the brake input at time step t. Each component is bound by the vehicle’s hardware limits, e.g., δt ∈ [δmin,δmax].

- 2.4.4 Language Representations

This paradigm leverages the natural language capabilities of VLMs to express driving actions through textual descriptions [18]. The action is represented as a sequence of discrete tokens from a predefined vocabulary:

at = {w1,w2,...,wT}, where wi ∈ V . (5)

Here, V represents the model’s vocabulary, T is the sequence length, and each token wi corresponds to an element in the vocabulary. The language-based action can range from high-level commands (e.g., “turn left at the intersection”) to specific numerical trajectory representations encoded as text tokens.

### 3 Vision-Action Models

Vision-Action (VA) models represent one of the earliest and most influential lines of research in autonomous driving. Their core idea is to directly map sensory observations – typically camera inputs – to driving actions, thereby avoiding explicit modular decomposition into perception, prediction, and planning. Enabled by deep neural networks, VA models have been explored through two major training paradigms: 1imitation learning, which distills policies from expert demonstrations, and 2reinforcement learning, which optimizes behavior through trial-and-error interaction. More recently, world models have expanded this paradigm by enabling agents to simulate scene dynamics and reason about action consequences, improving robustness and scalability.

- Table 1 provides an overview of representative efforts. From an architectural perspective, VA methodologies for autonomous driving can be broadly grouped into:

- • End-to-End Models, which directly predict control commands or planned trajectories from sensory inputs.
- • World Models, which explicitly model action-conditioned future dynamics to support policy learning and decision-making.

- 3.1 End-to-End Models for Autonomous Driving

End-to-end (E2E) models learn a single neural network that maps raw or intermediate sensor observations to actions or planned trajectories [15, 27, 37, 144]. Unlike modular stacks, which isolate perception, prediction, and planning, E2E approaches implicitly couple these tasks within a unified representation [91, 108, 236]. Depending on whether perception supervision is employed, existing methods fall into two main categories: action-only models and perception-action models, as illustrated in Figure 3.

- 3.1.1 Action-Only Model

Action-only models adopt a streamlined one-stage formulation: sensory inputs are fed directly into a network that outputs low-level actions. These methods primarily differ in whether policies are learned from demonstrations or through exploration.

Imitation Learning (IL), especially behavior cloning [6], learns a policy by matching expert actions, as visualized in the IL branch of Figure 3. Early works [15, 42, 198, 203] demonstrated that actions can be predicted directly from monocular or multi-view inputs, and subsequent designs refined backbone architectures [23, 39, 206].

NEAT [38] highlights behaviorally relevant image regions via intermediate attention maps, while TCP [272] fuses a trajectory branch and control branch for complementary supervision. To better leverage scene geometry, BEV-Planner [149] predicts trajectories from BEV features enriched with ego states. Urban-Driver [218] moves beyond open-loop evaluation by training policies in a differentiable, datadriven simulator.

Table 1 Summary of Vision-Action models in autonomous driving.

- • Inputs: Camera, : LiDAR, and : Ego-Status.
- • Action Types: RL: Policy w/ Reinforcement Learning, REG: Decoder + MLP, SEL: Traj. Selection w/ Cost, and GEN: Traj. Generation w/ Generative Model.

- • Outputs: Ctrl.: Control Signal, Traj.: Numerical Trajectory.

- • Datasets:

###### Lyft [89], nuScenes [16],

###### CARLA [56],

###### NoCrash [43],

###### ProcGen [41],

|C|
|---|

|N|
|---|

|P|
|---|

|L|
|---|

###### NAVSIM [47], OpenOcc [250],

###### Bench2Drive [103],

|N|
|---|

|B|
|---|

|N|
|---|

###### Occ3D [248], Cam4DOcc [177], and

###### OpenDV [296],

###### nuPlan [17],

|O|
|---|

|O|
|---|

|N|
|---|

|O|
|---|

###### Private Data.

|C|
|---|

|P|
|---|

# Model Venue Input Dataset Vision Action Output

- • Sec. 3.1.1 Action-only Models

- 1 LBC [24] CoRL’20

[Figure 217]

|C|
|---|

|N|
|---|

ResNet [87] RL Ctrl.+Traj.

- 2 Latent-DRL [251] CVPR’20

[Figure 218]

|C|
|---|

ResNet [87] RL Ctrl.

- 3 NEAT [38] ICCV’21

[Figure 219]

|C|
|---|

ResNet [87] REG Traj.

- 4 Roach [326] ICCV’21

[Figure 220]

|C|
|---|

|N|
|---|

ResNet [87] RL Ctrl.

- 5 WoR [25] ICCV’21

[Figure 221]

|C|
|---|

|N|
|---|

|P|
|---|

ResNet [87] REG Ctrl.

- 6 TCP [272] NeurIPS’22

[Figure 222]

|C|
|---|

ResNet [87] REG Ctrl.+Traj.

- 7 Urban-Driver [218] CoRL’22

[Figure 223]

|L|
|---|

ResNet [87] REG Traj.

- 8 LAV [23] CVPR’22

[Figure 224]

[Figure 225]

|C|
|---|

ResNet [87] REG Ctrl.+Traj.

- 9 TransFuser [39] TPAMI’23

[Figure 226]

[Figure 227]

|C|
|---|

ResNet [87] REG Traj.

- 10 GRI [22] Robotics’23

[Figure 228]

|C|
|---|

EfficientNet [238] RL Ctrl.

- 11 BEVPlanner [149] CVPR’24

[Figure 229]

|N|
|---|

ResNet [87] REG Traj.

- 12 Raw2Drive [300] NeurIPS’25

[Figure 230]

|C|
|---|

|B|
|---|

ResNet [87] RL Ctrl.

- 13 RAD [67] NeurIPS’25

[Figure 231]

|P|
|---|

ResNet [87] RL Traj.

- 14 TrajDiff [75] arXiv’25

[Figure 232]

[Figure 233]

[Figure 234]

|N|
|---|

ResNet [87] GEN Traj.

• Sec. 3.1.2 Perception-Action Models

- 15 ST-P3 [91] ECCV’22

[Figure 235]

|N|
|---|

|C|
|---|

EfficientNet [238] SEL Traj.

- 16 UniAD [93] CVPR’23

[Figure 236]

|N|
|---|

ResNet [87] REG Traj.

- 17 VAD [107] ICCV’23

[Figure 237]

|N|
|---|

ResNet [87] REG Traj.

- 18 OccNet [250] ICCV’23

[Figure 238]

|N|
|---|

|O|
|---|

ResNet [87] SEL Traj.

- 19 GenAD [332] ECCV’24

[Figure 239]

|N|
|---|

ResNet [87] GEN Traj.

- 20 PARA-Drive [268] CVPR’24

[Figure 240]

|N|
|---|

ResNet [87] REG Traj.

- 21 Hydra-MDP [147] CVPRW’24

[Figure 241]

[Figure 242]

|N|
|---|

ResNet [87] SEL Traj.

- 22 SparseAD [317] arXiv’24

[Figure 243]

|N|
|---|

ResNet [87] REG Traj.

- 23 GaussianAD [333] arXiv’24

[Figure 244]

|N|
|---|

ResNet [87] REG Traj.

- 24 DiFSD [231] arXiv’24

[Figure 245]

|N|
|---|

ResNet [87] GEN Traj.

- 25 DriveTransformer [104] ICLR’25

[Figure 246]

|N|
|---|

|B|
|---|

ResNet [87] REG Traj.

- 26 SparseDrive [236] ICRA’25

[Figure 247]

|N|
|---|

ResNet [87] REG Traj.

- 27 DiffusionDrive [155] CVPR’25

[Figure 248]

[Figure 249]

|N|
|---|

|N|
|---|

ResNet [87] GEN Traj.

- 28 GoalFlow [279] CVPR’25

[Figure 250]

[Figure 251]

[Figure 252]

|N|
|---|

VoVNet [128] GEN Traj.

- 29 GuideFlow [161] arXiv’25

[Figure 253]

|N|
|---|

|N|
|---|

|B|
|---|

ResNet [87] GEN Traj.

- 30 ETA [78] arXiv’25

[Figure 254]

|B|
|---|

CLIP-ViT [210] REG Traj.

- 31 Geo [105] arXiv’25

[Figure 255]

|N|
|---|

ResNet [87] REG Traj.

- 32 DiffusionDriveV2 [345] arXiv’25

[Figure 256]

[Figure 257]

|N|
|---|

ResNet [87] GEN Traj.

- 33 NaviHydra [271] arXiv’25

[Figure 258]

[Figure 259]

|N|
|---|

ResNet [87] SEL Traj.

- 34 Mimir [280] arXiv’25

[Figure 260]

[Figure 261]

|N|
|---|

ResNet [87] GEN Traj.

- • Sec. 3.2.1 Image-Based World Models

35 DriveDreamer [264] ECCV’24

[Figure 262]

[Figure 263]

|N|
|---|

SD [214] REG Traj. 36 GenAD [296] CVPR’24

[Figure 264]

|O|
|---|

SDXL [202] REG Traj. 37 Drive-WM [266] CVPR’24

[Figure 265]

[Figure 266]

|N|
|---|

ConvNeXt [172] SEL Traj. 38 DrivingWorld [92] arXiv’24

[Figure 267]

[Figure 268]

|N|
|---|

VQ-VAE [254] REG Traj. 39 Imagine-2-Drive [70] IROS’25

[Figure 269]

|C|
|---|

SVD [13] SEL Traj. 40 DrivingGPT [33] ICCV’25

[Figure 270]

[Figure 271]

|N|
|---|

|N|
|---|

VQ-VAE [254] REG Traj. 41 Epona [321] ICCV’25

[Figure 272]

[Figure 273]

|N|
|---|

|N|
|---|

|N|
|---|

DC-AE [26] REG Traj. 42 VaViM [10] arXiv’25

[Figure 274]

|O|
|---|

|N|
|---|

|N|
|---|

LLaMAGen [235] GEN Traj.

- • Sec. 3.2.2 Occupancy-Based World Models

43 OccWorld [331] ECCV’24

[Figure 275]

[Figure 276]

|N|
|---|

|O|
|---|

ResNet [87] REG Traj. 44 NeMo [96] ECCV’24

[Figure 277]

|N|
|---|

ResNet [87] REG Traj. 45 OccVAR [111] -

[Figure 278]

[Figure 279]

|N|
|---|

|O|
|---|

ResNet [87] REG Traj. 46 RenderWorld [292] arXiv’24

[Figure 280]

|N|
|---|

|O|
|---|

Swin-T [171] REG Traj. 47 DFIT-OccWorld [318] arXiv’24

[Figure 281]

[Figure 282]

|N|
|---|

|O|
|---|

ResNet [87] REG Traj. 48 Drive-OccWorld [297] AAAI’25

[Figure 283]

|N|
|---|

|C|
|---|

ResNet [87] REG Traj. 49 T3Former [282] arXiv’25

[Figure 284]

[Figure 285]

|N|
|---|

|O|
|---|

ResNet [87] REG Traj. 50 AD-R1 [289] arXiv’25

[Figure 286]

[Figure 287]

[Figure 288]

|N|
|---|

|N|
|---|

- RL Traj.

- • Sec. 3.2.3 Latent-Based World Models

- 51 Covariate-Shift [204] arXiv’24

[Figure 289]

[Figure 290]

|C|
|---|

DINOv2 [196] REG Traj.

- 52 World4Drive [335] ICCV’25

[Figure 291]

|N|
|---|

|N|
|---|

ResNet [87] REG Traj.

- 53 WoTE [144] ICCV’25

[Figure 292]

[Figure 293]

|N|
|---|

|B|
|---|

ResNet [87] SEL Traj.

- 54 LAW [142] ICLR’25

[Figure 294]

|N|
|---|

|N|
|---|

|C|
|---|

Swin-T [171] REG Traj.

- 55 SSR [133] ICLR’25

[Figure 295]

|N|
|---|

|C|
|---|

ResNet [87] REG Traj.

- 56 Echo-Planning [233] arXiv’25

[Figure 296]

|N|
|---|

ResNet [87] REG Traj.

- 57 SeerDrive [315] NeurIPS’25

IL-based methods are simple, efficient, and require no reward engineering; however, they remain sensitive to distribution shift [56, 205, 215] and causal confusion [49, 190, 200], which can impair reliability in long-tailed or rare-event scenarios.

Reinforcement Learning (RL) optimizes actions through interaction, offering greater flexibility than imitation-based approaches [118, 237]. Several works address the sample inefficiency of RL by combining it with supervised pretraining: Latent-DRL [251] and Gri [22] pre-train visual encoders using semantic segmentation, while LSD [194] initializes policies via IL before performing RL fine-tuning. Privileged-information distillation has also proven effective: LBC [24], WoR [25], and Roach [326] use simulator-only states to guide sensor-based agents.

Combined with the world model, Think2Drive [136] trains the agent with the Model-Based RL (MBRL) method, paired with a compact latent world model learning the transitions of the environment. Raw2Drive [300] is a dual-stream MBRL approach, where the raw sensor world model is aligned with the privileged world model for camerabased action prediction. In contrast to studies in non-photorealistic CARLA [56], recent efforts have shifted toward photorealistic world modeling. RAD [67] establishes a 3DGS-based [115] closed-loop RL training paradigm regulated by IL in a realistic 3DGS environment. The key challenges in RL-based models include sample inefficiency [150], reward function design [119], and sim-to-real transfer [37].

VoVNet [128] SEL Traj.

[Figure 297]

[Figure 298]

|N|
|---|

|N|
|---|

- 3.1.2 Perception-Action Model

###### Perception-action models follow a two-stage paradigm in which perception tasks (e.g., mapping, tracking) supervise and constrain trajectory prediction. These methods generally adopt either dense BEV-based representations or sparse query-based representations, as shown in Figure 3.

Dense BEV-Based Models construct unified top-down features from multi-view cameras. ST-P3 [91] jointly learns spatial-temporal features for perception and planning; UniAD [93] integrates sequential task dependencies to support goal-directed planning. VAD [107] employs a vectorized scene representation to improve both planning safety and efficiency. OccNet [250] incorporates occupancy embeddings to capture 3D scene geometry. Para-Drive [268] proposes a fully parallel E2E architecture for real-time deployment.

[Figure 299]

Generative and sampling-based approaches have recently emerged: GenAD [332] frames planning as sampling from learned distributions; DiffusionDrive [155] introduces a truncated diffusion policy guided by multi-modal anchors; GuideFlow [161] incorporates explicit physical constraints into the generation process. While BEV representations naturally align with 2D trajectory planning, they require substantial computation due to their dense spatial structure.

Figure 3 The categorization of End-to-End VA models based on model structures and outputs, including Action-Only Models (Sec. 3.1.1), and Perception-Action Models (Sec. 3.1.2).

Sparse Query-Based Models avoid explicit BEV grids by using latent queries to aggregate image features. SparseAD [317] and SparseDrive [236] represent the entire scene using sparse perception queries and a parallel planner, achieving strong efficiency-accuracy trade-offs. DiFSD [231] introduces an ego-centric sparse formulation and models uncertainty through trajectory denoising. DriveTransformer [104] incorporates task parallelism, sparse attention, and streaming updates for improved stability. GaussianAD [333] adopts 3D semantic Gaussians for fine-grained yet compact scene representation.

Sparse query methods significantly reduce inference latency, but the absence of a dense future-world representation can restrict long-horizon reasoning and planning safety.

- 3.2 World Models for Autonomous Driving

World models aim to predict how driving scenes evolve under different ego actions [77, 125, 291]. By jointly modeling scene dynamics and ego motion, they provide a powerful mechanism for learning safe, long-horizon driving policies [53]. Their applications span immersive simulation [12, 90, 174, 212, 290, 342], end-to-end planning [68, 92, 264, 266], and feature learning for downstream tasks [28, 29, 137, 169, 188, 298, 319]. Here, we focus on world models designed for trajectory planning and categorize them by prediction modality and representation granularity into three groups: image-based, occupancy-based, and latent-based models (Figure 4).

- 3.2.1 Image-Based World Model

###### Image-based world models generate future frames conditioned on ego actions, enabling agents to “dream” scene evolution and evaluate the consequences of different trajectories. These methods leverage modern generative models to synthesize realistic, temporally coherent videos and are typically classified into diffusion-based and autoregressive architectures.

Diffusion-Based World Models use latent video diffusion [14, 214] to produce multi-step rollouts. For front-view forecasting, GenAD [296] and Vista [68] incorporate temporal reasoning modules to handle complex motion patterns. Imagine-2-Drive [70] integrates diffusion generation into a reinforcement-learning framework, training a policy actor inside the world model. To support multi-view predictions, DriveDreamer [264] employs a twostage pipeline for video synthesis and policy learning. Drive-WM [266] factors views within a spatiotemporal model and generates multiple plausible futures, selecting trajectories using image-based rewards.

Autoregressive (AR) Models tokenize images using VQ-VAE [254] and model scene evolution via next-token prediction [58, 127, 305]. DrivingWorld [92] builds a GPT-style architecture for highfidelity long-horizon video generation. DrivingGPT [33] interleaves image and action tokens, unifying simulation and trajectory planning. Epona [321] combines AR modeling with diffusion to produce high-resolution, long-duration rollouts.

[Figure 300]

Image-based world models provide photorealistic simulations crucial for training and evaluation. However, their reliance on 2D appearance limits explicit

- 3D reasoning, which can hinder safetycritical long-horizon planning.

- 3.2.2 Occupancy-Based World Models

Occupancy-based world models represent the driving scene as spatiotemporal occupancy grids and predict their evolution under different actions. Instead of synthesizing raw pixels, these models focus on the geometry and semantics of free space, obstacles, and agents [153, 170, 342]. As shown in the middle of Figure 4, AR prediction is commonly used in occupancy world models.

OccWorld [331] first introduces occupancy forecasting for planning, using a scene tokenizer to discretize

- 3D occupancy before applying a GPT-style transformer to synthesize future scenes and ego trajectories. RenderWorld [292] produces 3D occupancy through a self-supervised Gaussian module, while OccVAR [111] performs coarse-to-fine 4D occupancy forecasting. T3Former [282] encodes occupancy using compact triplanes and predicts future triplane updates from multi-scale history.

Figure 4 The categorization of World Models based on prediction modalities, including Image-Based Models (Sec. 3.2.1), Occupancy-Based Models (Sec. 3.2.2), and Latent-Based Models (Sec. 3.2.3).

An alternative line employs single-stage feedforward prediction. Drive-OccWorld [297] uses predicted future BEV features for action-conditioned 3D forecasting. DFIT-OccWorld [318] introduces a decoupled dynamic flow strategy to support efficient non-autoregressive prediction. NeMo [96] improves vision-based occupancy forecasting by combining it with self-supervised image reconstruction signals.

Occupancy-based models offer strong geometric fidelity and explicit free-space reasoning but rely on costly 3D annotations, which can limit scalability across diverse environments.

- 3.2.3 Latent-Based World Models

###### Latent-based world models bypass explicit image or occupancy forecasting and instead predict future dynamics directly in a compressed latent space. By operating on high-level features, these models capture behavioral patterns and long-horizon dependencies while avoiding the computational overhead of pixel-level generation. Regarding the forecasting strategy, the latent world model utilizes single-frame or AR prediction presented at the bottom of Figure 4.

###### Table 2 Categories of natural language prompts for Vision-Language-Action (VLA) models in autonomous driving.

Prompt Type Explanations

System Prompt Text templates or query formulations designed to interact with LLMs, guiding them to perform specific driving-related reasoning or trajectory prediction tasks. System prompts often define the task structure, provide role definitions, and shape the model’s reasoning behavior.

[Figure 301]

Instructions Commands or instructions provided by humans or systems, typically describing the driving goal or required maneuver (e.g., “turn left at the next intersection”).

[Figure 302]

Scene Description Textual descriptions of surroundings, including perceived objects, road layout, and contextual factors (e.g., “a pedestrian is crossing on the right”, or “a vehicle is 5 meters ahead on the left”).

[Figure 303]

Traffic Rules Prompts encoding regulatory constraints or domain knowledge, such as traffic laws, traffic light status, right-of-way rules, or safety guidelines.

[Figure 304]

Ego Status Information about the ego vehicle’s internal state, including speed,

[Figure 305]

position, heading, or navigation intent.

Context Information Demonstrations presented as paired examples of driving scenarios and corresponding actions, used to guide the model via in-context learning.

[Figure 306]

Early latent models [7, 337] learn feature-level dynamics for planning without generating visual frames. LAW [142] leverages self-supervised learning to predict future scene features from current features and planned ego trajectories, enabling end-to-end driving without perception labels. World4Drive [335] employs vision foundation models to create latent representations from which diverse planning trajectories can be generated and evaluated. Echo-Planning [233] introduces a bidirectional Current→Future→Current (CFC) cycle to enforce temporal consistency in latent BEV features. For robustness in imitation learning, Covariate-Shift [204] addresses distribution mismatch using latent rollouts. By injecting predicted BEV features, SeerDrive [315] refines both latent prediction and trajectory generation in a closed-loop manner.

Latent world models offer efficient and semantically informed forecasting. However, achieving high-quality planning still requires auxiliary supervision from 2D/3D annotations, such as bounding boxes or HD maps.

- 3.3 Limitations of VA Compared to VLA

While VA models remain widely deployed, they face structural limitations that hinder performance in complex, ambiguous, or long-tailed scenarios: areas where VLA models excel.

- • Limited Interpretability. VA models provide little insight into their decision-making process. In contrast, VLA models can articulate reasoning steps or explanations through language.
- • Weak Generalization. VA policies lack broad world knowledge and often require environment-specific retraining. VLA models leverage large-scale pretraining to generalize better under distribution shifts and long-tailed events.
- • No Chain-of-Thought Reasoning. VA models directly map pixels to actions, making it difficult to perform explicit reasoning or contextual analysis. VLAs natively support step-wise reasoning.
- • No Language Understanding. VA systems cannot incorporate human instructions or high-level goals expressed in texts. VLA models naturally integrate such inputs to guide planning and decision-making.

- 4 Vision-Language-Action Models

VLA models extend the Vision-Action paradigm by coupling visual perception with the multimodal reasoning capabilities of large vision-language models. Equipped with chain-of-thought style inference and broad world knowledge, these models are particularly promising for rare, ambiguous, and long-tailed driving scenarios.

- Table 2 summarizes typical prompting strategies, and Table 3 overviews representative VLA-based approaches.

- Table 3 Summary of Vision-Language-Action models in autonomous driving.

- • Input: :Camera, : Sys. Prompt, : Instruct., : Scene Descrip., : Status, : Traffic Rule, : Context Info.
- • Action: LH: Lang. Head, REG: Decoder+MLP, GEN: Traj. GEN with Generative Model.

- • Output: Desc.: Linguistic Descriptions, Traj.: Numerical Trajectory, Ctrl.: Control Signal. Meta.: Meta Action.

- • Datasets:

###### : Bench2Drive [103],

###### : nuScenes [16],

###### : BDD-X [117],

###### : DriveLM [225],

###### : SDN [183],

###### : VLAAD [201],

|N|
|---|

|B|
|---|

|D|
|---|

|S|
|---|

|V|
|---|

|B|
|---|

###### : OpenDV [296],

###### : Waymo [59],

###### : MetaAD [109],

###### : Carla [56],

###### : ImpromptuVLA [35],

###### : NAVSIM [47],

|W|
|---|

|M|
|---|

|C|
|---|

|I|
|---|

|N|
|---|

|O|
|---|

###### : nuPlan [17],

###### : Talk2Car [50],

###### : CoVLA [2],

###### : PhysicalAI-AV [193] and

###### : Private Data.

|N|
|---|

|T|
|---|

|C|
|---|

|P|
|---|

|P|
|---|

# Model Venue Input Modality Dataset Vision Language Action Output

- • Sec. 4.1.1: Textural Action Generator

- 1 DriveMLM [46] arXiv’23

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

|C|
|---|

EVA-CLIP [61] LLaMA [252] LH Desc.+Meta.

[Figure 311]

- 2 RAG-Driver [309] RSS’24

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

|B|
|---|

CLIP [210] Vicuna-1.5 [36] LH Desc.+Ctrl.

- 3 RDA-Driver [97] ECCV’24

[Figure 317]

[Figure 318]

[Figure 319]

|D|
|---|

|N|
|---|

BEVFusion [154] LLaMA [252] LH Desc.+Traj.

[Figure 320]

- 4 DriveLM [225] ECCV’24

[Figure 321]

[Figure 322]

|D|
|---|

BLIP-2 [132] LLaMA [252] LH Meta.+Traj.

[Figure 323]

- 5 DriveGPT4 [287] RA-L’24

[Figure 324]

[Figure 325]

|B|
|---|

CLIP [210] LLaMA-2 [252] LH Desc.+Ctrl.

[Figure 326]

- 6 DriVLMe [95] IROS’24

[Figure 327]

[Figure 328]

[Figure 329]

|B|
|---|

|S|
|---|

|C|
|---|

CLIP [210] Vicuna [36] LH Desc.+Ctrl.

- 7 LLaDA [130] CVPR’24

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

|N|
|---|

|N|
|---|

- GPT-4 [1] LH Ctrl.+Traj.

[Figure 335]

- 8 VLAAD [201] WACVW’24

[Figure 336]

[Figure 337]

[Figure 338]

|V|
|---|

BLIP-2 [132] LLaMA-2 [252] LH Ctrl.

[Figure 339]

- 9 OccLLaMA [267] arXiv’24

[Figure 340]

[Figure 341]

|N|
|---|

Swin-T [171], PointPillar [126] LLaMA-3 [74] LH Ctrl.+Traj.

[Figure 342]

- 10 Doe-1 [334] arXiv’24

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

|N|
|---|

Lumina-mGPT [157] BPE tokenizer [241] LH Ctrl.+Traj.

- 11 LINGO-2 [245] -

[Figure 347]

[Figure 348]

[Figure 349]

|P|
|---|

Wayve Vision Model [246] Wayve VLA Model [246] LH Desc.+Traj.

- 12 SafeAuto [320] ICML’25

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

|B|
|---|

|D|
|---|

LanguageBind [341] Video-LLaVA [156] LH Desc.+Ctrl.

[Figure 355]

- 13 OpenEMMA [278] WACV’25

[Figure 356]

[Figure 357]

[Figure 358]

|N|
|---|

[Figure 359]

Qwen2-VL [258] Qwen2-VL [258] LH Desc.+Traj.

[Figure 360]

- 14 ReasonPlan [168] CoRL’25

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

|B|
|---|

SigLIP [314] Qwen [4] LH Desc.+Traj.+Meta.

[Figure 365]

- 15 FutureSightDrive [312] NeurIPS’25

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

|N|
|---|

|D|
|---|

ViT [57] Qwen2-VL [258] LH Desc.+Traj.

[Figure 370]

- 16 ImpromptuVLA [35] NeurIPS’25

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

|I|
|---|

[Figure 376]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Traj.

[Figure 377]

- 17 WKER [313] AAAI’25

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

|N|
|---|

EVA-02 [62] LLaMA3 [74] LH Desc.+Traj.

[Figure 382]

- 18 OmniDrive [259] CVPR’25

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

|N|
|---|

|D|
|---|

EVA-02 [62] LLaMA2 [252] LH Desc.+Traj

[Figure 389]

- 19 S4-Driver [276] CVPR’25

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

|W|
|---|

|N|
|---|

PaLI3 [31] PaLI3 [31] LH Meta.+Traj.

- 20 EMMA [98] TMLR’25

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

|W|
|---|

|N|
|---|

[Figure 398]

Gemini-VLM [243] Gemini [242] LH Desc.+Traj.

[Figure 399]

- 21 Occ-LLM [284] ICRA’25

[Figure 400]

[Figure 401]

[Figure 402]

|N|
|---|

- LLaMA2 [252] LH Traj.

[Figure 403]

- 22 Sce2DriveX [327] RA-L’25

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

|N|
|---|

|B|
|---|

OpenCLIP [99] Vicuna-1.5 [36] LH Ctrl.+Traj.

- 23 DriveAgent-R1 [330] arXiv’25

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

|N|
|---|

|P|
|---|

[Figure 412]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Desc.+Meta.

[Figure 413]

- 24 Drive-R1 [146] arXiv’25

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

|N|
|---|

|D|
|---|

[Figure 418]

InternVL2 [263] InternVL2 [263] LH Desc.+Traj.+Meta.

[Figure 419]

- 25 FastDriveVLA [18] arXiv’25

[Figure 420]

[Figure 421]

|N|
|---|

[Figure 422]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Desc.+Traj.

[Figure 423]

- 26 WiseAD [323] arXiv’25

[Figure 424]

[Figure 425]

|B|
|---|

|C|
|---|

|D|
|---|

CLIP [210] MobileVLM [40] LH Traj.+Ctrl.

- 27 AutoDrive-R² [310] arXiv’25

[Figure 426]

[Figure 427]

[Figure 428]

|N|
|---|

|W|
|---|

[Figure 429]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Traj.

[Figure 430]

- 28 OmniReason [165] arXiv’25

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

|N|
|---|

|B|
|---|

EVA-02 [62] LLaVA-1.5 [160] LH Meta.+Traj.

[Figure 438]

- 29 OpenREAD [324] arXiv’25

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

|N|
|---|

[Figure 444]

Qwen3 [293] Qwen3 [293] LH Desc.+Meta.+Traj.

[Figure 445]

- 30 dVLM-AD [178] arxiv’25

[Figure 446]

[Figure 447]

[Figure 448]

|W|
|---|

|N|
|---|

SigLIP2 [253] LLaDA-V [303] LH Desc.+Traj.

- 31 PLA [325] arXiv’25

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

|N|
|---|

Sensor Encoder GPT-4.1 [1] LH Desc.+Traj.

[Figure 455]

- 32 AlphaDrive [109] arXiv’25

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

|M|
|---|

[Figure 460]

Qwen2-VL [258] Qwen2-VL [258] LH Desc.+Meta.

[Figure 461]

- 33 CoReVLA [60] arXiv’25

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

|B|
|---|

[Figure 466]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Ctrl.+Traj.

[Figure 467]

• Sec. 4.1.2: Numerical Action Generator

- 34 LMDrive [222] CVPR’24

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

|C|
|---|

ResNet [87] LLaMA [252] Vicuna [36] REG Ctrl.

[Figure 472]

- 35 BEVDriver [270] IROS’25

[Figure 473]

[Figure 474]

[Figure 475]

|C|
|---|

InterFuser [221] LLaMA-3.1 [74] REG Ctrl.+Traj.

[Figure 476]

- 36 CoVLA-Agent [2] WACV’25

[Figure 477]

[Figure 478]

[Figure 479]

|C|
|---|

CLIP [210] LLaMA2 [252] REG Desc.+Traj.

[Figure 480]

- 37 ORION [66] ICCV’25

[Figure 481]

[Figure 482]

[Figure 483]

|N|
|---|

|B|
|---|

EVA-02 [62] Vicuna-1.5 [36] GEN Desc.+Traj.

- 38 SimLingo [213] CVPR’25

[Figure 484]

[Figure 485]

[Figure 486]

|C|
|---|

|B|
|---|

[Figure 487]

InternViT [34] Qwen2 [258] REG Ctrl.+Traj.

[Figure 488]

- 39 DriveGPT4-V2 [288] CVPR’25

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

|C|
|---|

CLIP [210] SigLIP [314] Qwen [4] Tinyllama [322] REG Ctrl.+Traj.

[Figure 494]

- 40 AutoVLA [340] NeurIPS’25

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

|N|
|---|

|B|
|---|

|N|
|---|

|W|
|---|

[Figure 499]

Qwen2.5-VL [5] Qwen2.5-VL [5] LH Traj.

[Figure 500]

- 41 DriveMoE [299] arXiv’25

[Figure 501]

[Figure 502]

[Figure 503]

|B|
|---|

PaliGemma [11] PaliGemma [11] GEN Ctrl.

- 42 DSDrive [167] arXiv’25

[Figure 504]

[Figure 505]

[Figure 506]

|C|
|---|

ResNet [87] LLaMA [252] REG Desc.+Traj.

[Figure 507]

- 43 OccVLA [166] arXiv’25

[Figure 508]

[Figure 509]

[Figure 510]

|N|
|---|

VQ-VAE [254] PaliGemma-2 [230] REG Traj.

- 44 VDRive [76] arXiv’25

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

|N|
|---|

|B|
|---|

[Figure 515]

Qwen2.5-VL [5], CVQ [329] InternVL3 [343] GEN Desc.+Traj.

[Figure 516]

- 45 ReflectDrive [135] arXiv’25

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

|N|
|---|

LLaDA-V [303] LLaDA-V [303] GEN Traj.

- 46 E3AD [240] arXiv’25

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

|T|
|---|

[Figure 526]

Qwen2.5-VL [5] Qwen2.5-VL [5] REG Traj.

[Figure 527]

- 47 LCDrive [239] arXiv’25

[Figure 528]

[Figure 529]

[Figure 530]

|P|
|---|

DINOv2 [196] Qwen3 [293] LH Traj.

- 48 Alpamayo-R1 [265] arXiv’25

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

|P|
|---|

[Figure 536]

Cosmos-Reason1 [3] Cosmos-Reason1 [3] REG Desc.+Ctrl.+Traj.

[Figure 537]

- 49 UniUGP [173] arXiv’25

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

|W|
|---|

|N|
|---|

|N|
|---|

[Figure 542]

Qwen2.5-VL [5] Qwen2.5-VL [5] GEN Desc.+Traj.

[Figure 543]

- 50 MindDrive [232] arXiv’25

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

|N|
|---|

ResNet-34 [87] LLaVA-1B [129] GEN Traj.

[Figure 548]

- 51 AdaThinkDrive [176] arXiv’25

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

|N|
|---|

[Figure 553]

InternVL3 [343] InternVL3 [343] REG Desc.+Traj.+Meta.

[Figure 554]

- 52 Percept-WAM [79] arXiv’25

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

|N|N|
|---|---|

[Figure 559]

InternViT [34] InternVL2 [263] REG Traj.

[Figure 560]

- 53 Reasoning-VLA [316] arXiv’25

[Figure 561]

[Figure 562]

[Figure 563]

|N|
|---|

|N|
|---|

|W|
|---|

|A|
|---|

[Figure 564]

Qwen2.5-VL [5] Qwen2.5-VL [5] REG Traj.

[Figure 565]

- 54 SpaceDrive [134] arXiv’25

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

|N|
|---|

|B|
|---|

[Figure 570]

Qwen2.5-VL [5] Qwen2.5-VL [5] REG Desc.+Traj.

[Figure 571]

- 55 OpenDriveVLA [339] AAAI’26

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

|N|
|---|

BEVFormer [148] Qwen2.5 [294] LH Ctrl.+Traj.

[Figure 576]

- • Sec. 4.2.1: Dual-System: Explicit Action Guidance

56 DriveVLM [249] CoRL’24

[Figure 577]

[Figure 578]

[Figure 579]

|N|
|---|

|S|
|---|

- QwenVL [4] REG Desc.+Traj. 57 LeapAD [187] NeurIPS’24

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

|C|
|---|

[Figure 585]

QwenVL [4] GPT-4 [1], Qwen1.5 [244] LH Ctrl.+Traj. 58 FasionAD [207] arXiv’24

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

|N|
|---|

|C|
|---|

GenAD [332] CLIP [210], QwenVL [4] GEN Ctrl.+Traj. 59 Senna [108] arXiv’24

[Figure 591]

[Figure 592]

[Figure 593]

|N|
|---|

VADv2 [30], ViT [57] Vicuna [36] REG Desc.+Traj.

- 61 DME-Driver [80] AAAI’25

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

|H|
|---|

UniAD [93] LLaVA [159] REG Desc.+Traj.

[Figure 599]

- 62 SOLVE [32] CVPR’25

[Figure 600]

[Figure 601]

[Figure 602]

|N|
|---|

EVA-02 [62] LLaVA-1.5 [160] REG Ctrl.+Traj.

[Figure 603]

- 63 ReAL-AD [175] ICCV’25

[Figure 604]

[Figure 605]

[Figure 606]

|N|
|---|

|B|
|---|

UniAD [93], VAD [107] MiniCPM-2.5 [306] REG Ctrl.+Traj.

- 64 LeapVAD [181] TNNLS’25

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

|D|
|---|

|C|
|---|

[Figure 611]

QwenVL [4] InternVL2 [263] GPT-4o [195] LH Ctrl.+Traj.

[Figure 612]

- 65 DiffVLA [106] arXiv’25

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

|N|
|---|

CLIP [210] Vicuna-1.5 [36] GEN Traj.

- 66 FasionAD++ [208] arXiv’25

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

|N|
|---|

|C|
|---|

BEVFormer [148] Vicuna-1.5 [36], QwenVL [4] GEN Ctrl.+Traj.

• Sec. 4.2.2: Dual-System: Implicit Representations Transfer

- 67 VLP [197] CVPR’24

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

|N|
|---|

UniAD [93], VAD [107] CLIP [210] REG Traj.

- 68 VLM-AD [286] CoRL’25

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

|N|
|---|

UniAD [93], VAD [107] CLIP [210] REG Traj.

- 69 DiMA [88] CVPR’25

[Figure 631]

[Figure 632]

[Figure 633]

|N|
|---|

UniAD [93], VAD [107] LLaVA-1.5 [160] REG Ctrl.+Traj.

[Figure 634]

- 70 ALN-P3 [182] arXiv’25

[Figure 635]

[Figure 636]

[Figure 637]

|N|
|---|

VAD [107] LLaMA-2 [252] REG Desc.+Traj.

[Figure 638]

- 71 VERDI [63] arXiv’25

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

|N|
|---|

VAD [107] Qwen2.5-VL [5] REG Ctrl.+Traj.

[Figure 643]

- 72 VLM-E2E [163] arXiv’25

[Figure 644]

[Figure 645]

[Figure 646]

|N|
|---|

EfficientNet [238] CLIP [210] REG Traj.

- 73 ReCogDrive [145] arXiv’25

[Figure 647]

[Figure 648]

[Figure 649]

|N|
|---|

[Figure 650]

InternViT [34] Qwen2.5 [294] GEN Desc.+Traj.

[Figure 651]

- 74 InsightDrive [227] arXiv’25

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

|N|
|---|

ResNet [87] BERT [52] REG Traj.

- 75 NetRoller [277] arXiv’25

[Figure 656]

[Figure 657]

|N|
|---|

|D|
|---|

CLIP [210] LLaMA-2 [252] REG Traj.

[Figure 658]

- 76 ViLaD [45] arXiv’25

[Figure 659]

[Figure 660]

[Figure 661]

|N|
|---|

SigLIP-2 [253] LLaDA-V [303] GEN Ctrl.+Traj.

- 77 OmniScene [164] arXiv’25

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

|N|
|---|

ResNet [87] Qwen2.5-VL [5] REG Traj.

[Figure 666]

- 78 LMAD [226] arXiv’25

###### VAD [107] LLaVA-1.5 [160] LH Desc.+Meta.

|N|
|---|

|D|
|---|

[Figure 667]

[Figure 668]

[Figure 669]

From an architectural standpoint, current VLA methodologies for autonomous driving can be grouped into:

- • End-to-End VLA: a single model directly maps multimodal sensory inputs and language to actions.
- • Dual-System VLA: a VLM provides high-level reasoning or guidance, while a specialized driving module executes fast, low-level action.

- 4.1 End-to-End VLA for Autonomous Driving

End-to-end VLA frameworks aim to unify perception, reasoning, and planning within a single architecture. By leveraging the generalization ability of multimodal large language models (MLLMs), they directly transform multimodal observations into actions, reducing reliance on hand-crafted modules and task-specific heuristics. According to the form of their outputs, existing approaches can be broadly divided into two families, as illustrated in Figure 5: textual action generators, which operate primarily in the language space, and numerical action generators, which predict trajectories or controls in a continuous or discretized numeric space.

- 4.1.1 Textual Action Generator

Textual action generators formulate driving as a reasoning problem in the language space. The model produces human-readable symbolic decisions, allowing it to “think” and justify its outputs in words. Depending on the abstraction level of these outputs, existing methods can be grouped into meta-actions and trajectory waypoints.

Meta-Actions are discrete, semantic driving decisions, such as “accelerate”, “stop”, or “change lane”. They form an interpretable interface between high-level reasoning in VLMs and downstream controllers. Early works mainly used language models to output free-form text or conceptual descriptions, which are not directly executable. DriveMLM [46] narrows this gap by aligning LLM outputs with behavioral planning states in a modular stack, enabling language models to act as intermediate planners whose symbolic decisions can be converted into control commands.

[Figure 670]

Figure 5 The categorization of End-to-End VLA models based on the form of model outputs, including Textual Action Models (Sec. 4.1.1), and Numerical Action Models (Sec. 4.1.2).

Subsequent methods strengthen robustness and reasoning-planning alignment with reinforcement learning and chain-of-thought supervision [109, 146, 224]. AlphaDrive [109] introduces Group Relative Policy Optimization (GRPO) [224] to refine meta-actions using rewards that jointly consider trajectory quality, decision correctness, and format consistency. DriveAgent-R1 [330] first fine-tunes on a curated CoT dataset to encourage step-wise visual reasoning, then applies RL with trajectory- and meta-action-based rewards to bias reasoning paths toward decisions that are practically useful for driving. Recognizing that single-frame front-view inputs limit temporal and spatial understanding, Sce2DriveX [327] further incorporates multi-view video streams and BEV representations, enabling context-aware meta-decisions that are consistent with road topology and spatiotemporal dynamics.

Trajectory Waypoints–based textual generators frame motion planning as the prediction of future coordinates expressed in natural language, thereby unifying reasoning and trajectory forecasting within a single linguistic sequence. DriveLM [225] is an early representative of this paradigm, modeling autonomous driving as graph-structured visual question answering and generating textualized trajectory waypoints conditioned on multi-stage perception, prediction, and planning. Building on this idea, subsequent works adopt end-to-end multimodal formulations. EMMA [98] integrates camera observations and navigation commands into a unified language-driven pipeline for joint perception, road-graph understanding, and trajectory prediction. To enhance robustness in challenging scenarios, ImpromptuVLA [35] introduces an 80K-clip corner-case dataset,

demonstrating that pretraining on diverse edge cases significantly improves trajectory accuracy and closed-loop stability. LightEMMA [209] further benchmarks 12 vision–language models, revealing clear trade-offs between interpretability and numerical precision.

A complementary research direction focuses on better aligning reasoning with decision-making. RDADriver [97] enforces consistency between chain-of-thought explanations and trajectory outputs through tailored constraints, while Drive-R1 [146] leverages reinforcement learning to improve alignment between textual reasoning and waypoint prediction. Beyond alignment, efficiency and knowledge integration are explored by FastDriveVLA [18] via token pruning, WiseAD [323] through explicit driving priors, and OmniDrive [259] using counterfactual reasoning. WKER [313] further enhances robustness under occlusion by combining instruction-guided token selection with external knowledge sources.

Overall, textual action generators offer strong interpretability and rich reasoning but must bridge a fundamental gap between discrete language tokens and continuous control spaces. This mismatch can introduce precision limits and, in extreme cases, unstable or collapsed trajectories.

- 4.1.2 Numerical Action Generator

Numerical action generators augment VLM backbones with mechanisms that produce directly usable numeric outputs. The model still leverages language-driven reasoning internally, but its final predictions are expressed as trajectories, waypoints, or control values that can be consumed by classical planners or low-level controllers. Two main realizations exist: additional action heads attached to the backbone, and additional action tokens that discretize continuous actions into a token space.

Additional Action Head. A common strategy is to attach specialized prediction heads to vision-language models. BEVDriver [270] couples a multimodal encoder with a GRU-based head over BEV features, linking language-grounded reasoning with spatial waypoint prediction. CoVLA-Agent [2] uses a lightweight MLP head trained on the CoVLA dataset, demonstrating that joint supervision from trajectories and captions can simultaneously improve interpretability and numeric accuracy. DriveGPT4-V2 [288] augments token-based planning with an MLP that maps multimodal embeddings to continuous trajectories, enhancing sample efficiency while retaining GPT-style reasoning.

To specialize behaviors, DriveMoE [299] employs a Mixture-of-Experts design whose action head dynamically activates experts for skills such as lane following or overtaking. DSDrive [167] proposes a dual-head coordination module, with one head predicting waypoints and another generating reasoning outputs; distillation from larger VLMs keeps the model compact yet interpretable. LMDrive [222] integrates multimodal encoders with an MLP that directly outputs control signals in a closed loop, marking one of the first instruction-following, language-guided end-to-end systems.

Beyond simple MLPs, ORION [66] replaces deterministic heads with a diffusion-based predictor, modeling multi-modal trajectory distributions under uncertainty. SimLingo [213] decouples temporal speed waypoints from geometric path waypoints via a disentangled MLP head, enabling finer-grained control.

Additional Action Tokens. Instead of explicit heads, some works reuse the language token space to represent actions. AutoVLA [340] discretizes continuous trajectories into a codebook of action tokens, which are autoregressively generated alongside reasoning tokens, thereby unifying semantic reasoning and planning within a single sequence. Reinforcement fine-tuning penalizes redundant reasoning and improves token efficiency. OpenDriveVLA [339] follows a similar token-based paradigm but grounds token generation in a hierarchical alignment between 2D/3D perception and the language model. Structured features are embedded into a unified semantic space, and interaction tokens for the ego vehicle, environment, and other agents are autoregressively decoded into driving actions.

Numerical action generators are well-suited for downstream control, as their outputs are natively compatible with planners and actuators. However, they typically sacrifice some interpretability and often require substantial supervised data for stable training. When discretized action tokens are used, quantization artifacts can further limit fine-grained control accuracy.

- 4.2 Dual-System VLA

Dual-system VLA frameworks draw inspiration from the dual-process theory popularized by Thinking, Fast and Slow [113]. In this paradigm, a VLM serves as the slow, deliberative system that performs high-level reasoning, situational assessment, and linguistic inference, while a specialized autonomous driving module acts as the fast system responsible for real-time, low-latency trajectory generation and control. By combining these complementary strengths, dual-system frameworks aim to achieve both interpretability and safety-critical reactivity.

Depending on how VLM outputs interact with the specialized planner, existing methods can be categorized into two families: explicit action guidance and implicit representation transfer, as illustrated in Figure 6.

[Figure 671]

- 4.2.1 Explicit Action Guidance

Explicit action guidance frameworks use VLMs as structured action generators, whose high-level outputs are subsequently transformed or refined by the fast driving module. These approaches differ in their abstraction level and are mainly grouped into meta-action guidance and waypoint supervision.

Figure 6 The categorization of Dual-System VLA Models based on how VLM interacts with E2E module, including explicit action guidance (Sec. 4.2.1), and implicit representations transfer models (Sec. 4.2.2).

Meta-Action Guidance resorts to VLMs to output symbolic driving intentions, such as “slow down”, “change lane”, or “turn left”, which act as semantic priors for downstream planners. This design leverages the interpretability of linguistic actions while avoiding the precision challenges of directly generating continuous trajectories. Early work such as FasionAD [207] embodies the dual-process design by pairing a fast, data-driven planner with a slow VLM that issues meta-actions; a learned switching mechanism selects the appropriate pathway based on confidence and scene context. LeapVAD [181] refines this structure by combining an analytic branch that builds a memory bank with a heuristic branch that retrieves prior meta-actions for familiar situations.

More recent systems integrate high-level reasoning more tightly with planning. Senna [108] couples a commonsense VLM with an end-to-end planner: Senna-VLM produces natural-language decisions, which Senna-E2E converts into executable trajectories. DiffVLA [106] injects VLM-generated lateral and longitudinal decisions as one-hot priors into a diffusion-based planner, guiding multi-modal trajectory denoising.

Hierarchical frameworks such as DME-Driver [80] further separate decision and execution: a VLM-based Decision-Maker supplies meta-decisions or visual attention priors, and a dedicated Executor translates them into fine-grained control. ReAL-AD [175] extends this to a full three-layer hierarchy: strategy, decision, and operation, where VLM-derived situational insights shape progressively refined planning commands.

Waypoint Supervision is an explicit guidance that uses VLMs to generate coarse trajectory waypoints, which the fast planning module refines into dense, executable trajectories. DriveVLM [249] adopts a hierarchical reasoning-to-planning pipeline: the VLM produces meta-actions and coarse waypoints through chain-of-thought reasoning, and conventional planners transform them into detailed trajectories.

SOLVE [32] strengthens VLM-planner coordination through a shared vision encoder and a Trajectory Chainof-Thought module that iteratively refines candidate waypoints before final selection by the E2E planner. These designs provide a tighter numerical interface between reasoning and control, enabling VLMs to influence planning while retaining stability through classical refinement.

###### Overall, explicit guidance approaches maintain strong interpretability and grant VLM a direct role in decisionmaking. However, they remain sensitive to the accuracy and consistency of VLM outputs; misaligned or ambiguous commands can propagate downstream and degrade planning safety.

- 4.2.2 Implicit Representations Transfer

Implicit feature constraint refers to methods where the VLM acts as a teacher or auxiliary module during training, transferring reasoning ability or cognitive priors as latent features to the compact E2E network. These approaches fall into two main groups: knowledge distillation and multimodal feature fusion.

Knowledge Distillation-based approaches transfer VLM-generated explanations, reasoning traces, or structured action semantics into the latent space of the E2E driving model. VLP [197] aligns BEV features and planning queries with pretrained language embeddings using contrastive and supervisory objectives, enabling planners to inherit commonsense scene understanding. VLM-AD [286] generates free-form textual justifications and structured behavior labels using a VLM, distilling them into the planner through an alignment head and an action classification head. This dual-supervision design helps the E2E module acquire richer semantic representations while remaining computationally light during deployment.

More comprehensive alignment is seen in VERDI [63], which aligns perception, prediction, and planning outputs with VLM-generated chain-of-thought explanations, injecting structured reasoning across all stages of the pipeline. ALN-P3 [182] extends this principle with full-stack co-distillation: perception tokens, predicted motions, and planned trajectories are jointly aligned with VLM reasoning to unify cognition and execution.

Multimodal Feature Fusion-based approaches directly integrate VLM-derived features into the fast planner. InsightDrive [227] introduces language-guided scene representations, where VLM-generated descriptions highlight critical regions and modulate BEV features via cross-attention. VLM-E2E [163] explicitly models driver attention by fusing textual attention cues with BEV features through a learnable gating mechanism. Beyond attention cues, NetRoller [277] extracts latent reasoning variables from VLMs and adapts them into compact features suitable for real-time planners. ReCogDrive [145] aligns linguistic priors with a diffusionbased planner, refining trajectories through reinforcement learning to promote safety and human-like behavior. ETA [78] focuses on efficiency: VLM reasoning is computed asynchronously in earlier frames and fused into current features using an action-mask mechanism, ensuring guidance without incurring high real-time costs.

Implicit transfer methods reduce inference cost and avoid dependence on large VLMs at runtime, but they may sacrifice interpretability, and excessive distillation can oversimplify reasoning signals. Their effectiveness also depends strongly on how well the distilled or fused features align with the capacity of the fast driving module.

### 5 Datasets & Benchmark

Standardized datasets and benchmarks form the empirical foundation of VLA research, supporting model development, training, and evaluation. Since VLA driving systems integrate perception, language, and action, VLA datasets exhibit substantial diversity in modality composition, annotation granularity, and task definitions. Accordingly, evaluation protocols vary substantially, encompassing conventional trajectory-based metrics, language-centric assessments, and interactive closed-loop evaluations.

- 5.1 Datasets for VLA in Autonomous Driving

Traditionally, VA datasets provide rich sensory observations (cameras, LiDAR, RADAR) paired with control actions, enabling end-to-end mapping from images to trajectories [16, 17, 304]. These datasets underpin the development of early IL/RL-based VA models.

As language becomes an increasingly important modality for reasoning, instruction following, and explainability, VLA datasets have emerged [46, 225, 249, 262]. These datasets extend traditional driving logs with textual instructions, question-answer pairs, or rationales aligned with visual observations and expert actions [2, 35]. In general, a dataset is considered VLA-compatible when it provides temporally or semantically aligned language

- Table 4 Summary of existing Datasets & Benchmarks for training and evaluating the VA and VLA models.

- • Vision Sensor Inputs: Camera, LiDAR point cloud, RADAR point cloud, and Map.
- • Vision Types: Real: Data collected from real driving scenes, and Sim: Data collected from simulator.

- • Language Annotation Types: A: Automatic labeling process, and M: Manual labeling process.

- • Action Types: Traj.: Numerical trajectory output, and Ctrl: Control signal output.

- • Action Metrics: Open: Open-loop Evaluation, CL: Closed-loop Evaluation, and Lang.: Language-based Evaluation.

Dataset Year

Vision Language Action

[Figure 672]

Other Tasks Sensor Type Scale Category Label Scale Type Metric

- • Vision-Action Datasets BDD100K [304] 2020 Real 120M - - - Traj. - Percept. nuScenes [16] 2020 Real 1.4M - - - Traj. Open Percept. Waymo[234] 2020 Real 200M - - - Traj. Open Percept., Forecast. nuPlan [17] 2021 Real 4.6M - - - Ctrl, Traj. CL Forecast. Argoverse 2 [269] 2021 Real 300K - - - Traj. Open Percept., Forecast. Bench2Drive [103] 2024 Sim 2M - - - Traj. CL RoboBEV [275] 2025 Real 866K - - - Traj. Open Percept. WOD-E2E [283] 2025 Real 800K - - - Traj. Open -

|B|
|---|

[Figure 673]

|N|
|---|

[Figure 674]

[Figure 675]

[Figure 676]

|W|
|---|

[Figure 677]

[Figure 678]

[Figure 679]

|N|
|---|

[Figure 680]

[Figure 681]

[Figure 682]

|A|
|---|

[Figure 683]

[Figure 684]

[Figure 685]

|B|
|---|

[Figure 686]

[Figure 687]

[Figure 688]

|R|
|---|

[Figure 689]

[Figure 690]

[Figure 691]

|W|
|---|

[Figure 692]

- • Vision-Language-Action Datasets

BDD-X [117] 2018 Real 8.4M Caption M 26K Ctrl Open Reason. Talk2Car [51] 2022 Real 400K Caption M 12K Ctrl, Traj. Open Ground. SDN [183] 2022 Sim - Instruction, QA A+M 8.4K Ctrl, Traj. CL QA DriveMLM [46] 2023 Sim - Reason., Deci. A+M - Ctrl, Traj. CL, Lang. Reason., QA LMDrive [222] 2024 Sim 3M Instruction A+M 64K Traj. CL DriveLM-N [225] 2024 Real 4.8K QA M 445K Ctrl, Traj. Open Reason., QA DriveLM-C [225] 2024 Sim 64K QA A 3.76M Ctrl, Traj. Open Reason., QA HBD [80] 2024 Real,Sim - Deci., Descrip., QA A+M - Traj. Open Descrip., QA VLAAD [201] 2024 Real - Reason., QA A+M 64K Ctrl. Lang. Caption, QA SUP-AD [249] 2024 Real - Action, Reason., QA A+M - Ctrl, Traj. Open, Lang. Reason, QA NuInstruct [54] 2024 Real 11.8K Instruction A 91K Ctrl Lang. Reason. WOMD-Reason [141] 2024 Real 63K QA A 2940K Plan. Lang. Reason., QA DriveCoT [262] 2024 Sim - CoT, Deci. A 36K Ctrl Open Reason. Reason2Drive [192] 2024 Real - Reason., QA A 632K Ctrl, Traj. Open Reason., QA DriveBench [274] 2025 Real 19.2K QA A+M 20.5K Ctrl Lang. QA MetaAD [109] 2025 Real 120K Reason., Plan, QA - 30K Ctrl Lang. Reason. OmniDrive [259] 2025 Real - Reason., QA A - Ctrl, Traj. Open Reason. NuInteract [328] 2025 Real 34K Caption, QA A 1.5M Ctrl Lang. Percept., Ground. DriveAction [86] 2025 Real 2.6K QA A 16.18K Ctrl Lang. ImpromptuVLA [35] 2025 Real,Sim 2M Instruction, QA A+M 80K Ctrl, Traj. Open, CL QA CoVLA [2] 2025 Real 6M Caption A 6M Traj. Open OmniReason-N [165] 2025 Real - QA A - Ctrl, Traj. Open Reason., QA OmniReason-B2D [165] 2025 Sim - QA A - Ctrl, Traj. Open Reason., QA

|B|
|---|

[Figure 693]

|T|
|---|

[Figure 694]

[Figure 695]

[Figure 696]

|S|
|---|

[Figure 697]

|D|
|---|

[Figure 698]

[Figure 699]

|L|
|---|

[Figure 700]

[Figure 701]

|B|
|---|

[Figure 702]

[Figure 703]

[Figure 704]

|B|
|---|

[Figure 705]

[Figure 706]

[Figure 707]

|H|
|---|

[Figure 708]

|V|
|---|

[Figure 709]

|S|
|---|

[Figure 710]

|N|
|---|

[Figure 711]

[Figure 712]

[Figure 713]

|W|
|---|

[Figure 714]

|D|
|---|

[Figure 715]

[Figure 716]

|R|
|---|

[Figure 717]

[Figure 718]

[Figure 719]

|D|
|---|

[Figure 720]

[Figure 721]

[Figure 722]

|M|
|---|

[Figure 723]

|O|
|---|

[Figure 724]

[Figure 725]

[Figure 726]

|N|
|---|

[Figure 727]

[Figure 728]

[Figure 729]

|D|
|---|

[Figure 730]

|I|
|---|

[Figure 731]

|C|
|---|

[Figure 732]

|O|
|---|

[Figure 733]

[Figure 734]

[Figure 735]

|O|
|---|

[Figure 736]

###### annotations that connect visual observations with actions or trajectories, enabling tri-modal learning. The summarized collections are provided in Table 4.

- 5.1.1 Vision-Action Datasets

Originally, BDD100K [304] provides 100K diverse driving videos from across the United States, covering a wide spectrum of weather, lighting, and traffic conditions, making it a foundational dataset for behavioral cloning and end-to-end driving. Later, nuScenes [16] offers 1,000 multi-sensor driving scenes with synchronized 6camera surround views, LiDAR sweeps, radar, 3D boxes, and motion trajectories, supporting both perception tasks and multi-agent motion forecasting. Larger-scale datasets such as the Waymo Open Dataset [59] and Argoverse 2 [269] further extend this paradigm with higher-resolution sensors, longer trajectories, and detailed HD maps, enabling robust training of perception-to-prediction pipelines in diverse urban settings. Complementing these efforts, nuPlan [17] incorporates long-horizon ego trajectories, dense map context, and simulation interfaces for closed-loop testing, providing comprehensive supervision for evaluating decisionmaking and planning under complex, real-world conditions. While lacking explicit language supervision, these VA datasets establish the visual–action foundation for VLA development by providing structured supervision that links visual perception, temporal dynamics, and expert decision-making.

- 5.1.2 Vision-Language-Action Datasets

###### Building upon the visual–action foundation established by VA datasets, VLA datasets enrich driving logs with structured or free-form natural language to support joint perception-language-action learning.

- Table 5 Comparisons of state-of-the-art models for Open-Loop Planning on the nuScenes [16] benchmark.

- • Input: :Camera, :LiDAR, : Prompt, : Instruct., : Scene Descrip., : Status, : Rule, : Context.
- • Action: LH: Language Head, RL: Policy w/ Reinforcement Learning, REG: Decoder + MLP, SEL: Trajectory Selection w/ Cost, and GEN: Trajectory Generation w/ Generative Model.

- • Evaluation Metrics: L2 (↓): L2 Error in meters, and CR (↓): Collision Rate.

Model Year Input Vision Language Action

[Figure 737]

L2 ↓ CR ↓ 1s 2s 3s Avg. 1s 2s 3s Avg.

- • Vision-Action Models

[Figure 738]

ST-P3 [91] 2022 EfficientNet - REG 1.33 2.11 2.90 2.11 0.23 0.62 1.27 0.71 UniAD [93] 2022 ResNet - REG 0.44 0.67 0.96 0.69 0.04 0.08 0.23 0.12 VAD [107] 2023 ResNet - REG 0.17 0.34 0.60 0.37 0.07 0.10 0.24 0.14

[Figure 739]

[Figure 740]

[Figure 741]

OccNet [250] 2023 ResNet - SEL 1.29 2.13 2.99 2.14 0.21 0.59 1.37 0.72 BEV-Planner [149] 2024 ResNet - REG 0.30 0.52 0.83 0.55 0.10 0.37 1.30 0.59

[Figure 742]

[Figure 743]

Drive-WM [266] 2024 ConvNeXt - SEL 0.43 0.77 1.20 0.80 0.10 0.21 0.48 0.26 GenAD [332] 2024 ResNet - GEN 0.36 0.83 1.55 0.91 0.06 0.23 1.00 0.43 OccWorld [331] 2024 ResNet - REG 0.43 1.08 1.99 1.17 0.07 0.38 1.35 0.60 DriveDreamer [264] 2024 SD - REG - - - 0.29 - - - 0.15

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

SparseAD [317] 2024 ResNet - REG 0.15 0.31 0.56 0.34 0.00 0.04 0.15 0.06 GaussianAD [333] 2024 ResNet - REG 0.40 0.64 0.88 0.64 0.09 0.38 0.81 0.42

[Figure 750]

[Figure 751]

LAW [142] 2024 Swin-T - REG 0.24 0.46 0.76 0.49 0.08 0.10 0.39 0.19 SSR [133] 2024 ResNet - REG 0.18 0.36 0.63 0.39 0.01 0.04 0.12 0.06

[Figure 752]

[Figure 753]

Drive-OccWorld [297] 2025 ResNet - REG 0.25 0.44 0.72 0.47 0.03 0.08 0.22 0.11 DriveTransformer [104] 2025 ResNet - REG 0.19 0.34 0.66 0.40 0.03 0.10 0.21 0.11

[Figure 754]

[Figure 755]

[Figure 756]

DiffusionDrive [155] 2025 ResNet - GEN 0.27 0.54 0.90 0.57 0.03 0.05 0.16 0.08 World4Drive [335] 2025 ResNet - REG 0.23 0.47 0.81 0.50 0.02 0.12 0.33 0.16

[Figure 757]

[Figure 758]

Epona [321] 2025 DC-AE - REG 0.61 1.17 1.98 1.25 0.01 0.22 0.85 0.36 SeerDrive [315] 2025 ResNet - SEL 0.20 0.39 0.69 0.43 0.00 0.05 0.14 0.06

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

GuideFlow [161] 2025 ResNet - GEN - - - - 0.00 0.02 0.18 0.07

• Vision-Language-Action Models

[Figure 763]

Agent-Driver [185] 2023 - GPT-3.5 LH 0.16 0.34 0.61 0.37 0.02 0.07 0.18 0.09 DriveVLM [249] 2024 ViT QwenVL GEN 0.18 0.34 0.68 0.40 0.10 0.22 0.45 0.27 DriveVLM-Dual [249] 2024 ViT QwenVL REG 0.15 0.29 0.48 0.31 0.05 0.08 0.17 0.10

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

RAG-Driver [309] 2024 CLIP Vicuna-1.5 LH 0.34 0.37 0.69 0.40 0.01 0.05 0.26 0.10 Senna [108] 2024 ViT Vicuna-1.5 REG 0.37 0.54 0.86 0.59 0.09 0.12 0.33 0.18 Doe-1 [334] 2024 Lumina-mGPT BPE Tokenizer LH 0.37 0.67 1.07 0.70 0.02 0.14 0.47 0.21

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

VLP [197] 2024 UniAD, VAD CLIP REG 0.30 0.53 0.84 0.55 0.01 0.07 0.38 0.15 VLM-AD [286] 2024 UniAD, VAD CLIP-ViT REG 0.24 0.46 0.75 0.48 0.12 0.17 0.41 0.23

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

OpenDriveVLA [339] 2025 ResNet Qwen2.5-VL LH 0.14 0.30 0.55 0.33 0.02 0.07 0.22 0.10

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

OmniDrive [259] 2025 EVA-02 LLaMA2 LH 0.40 0.80 1.32 0.84 0.04 0.46 2.32 0.94 ORION [66] 2025 EVA-02 Vicuna-1.5 GEN 0.17 0.31 0.55 0.34 0.05 0.25 0.80 0.37 EMMA [98] 2025 Gemini-VLM Gemini LH 0.14 0.29 0.54 0.32 - - - WKER [313] 2025 EVA-02 LLaMA3 LH 0.14 0.30 0.55 0.33 0.07 0.14 0.32 0.18

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

Drive-R1 [146] 2025 InternVL2 InternVL2 LH 0.14 0.28 0.50 0.31 0.02 0.06 0.19 0.09 ReAL-AD [175] 2025 ResNet MiniCPM REG 0.30 0.48 0.67 0.48 0.07 0.10 0.28 0.15 ReAL-AD [175] 2025 ResNet QwenVL REG 0.35 0.53 0.71 0.53 0.09 0.12 0.31 0.17

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

DiMA [88] 2025 VAD LLaVA-1.5 REG 0.18 0.50 1.03 0.57 0.00 0.05 0.16 0.08 FasionAD [207] 2025 BEVFormer VLM+Thinking GEN 0.19 0.62 1.25 0.69 0.02 0.09 0.44 0.18

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

InsightDrive [227] 2025 ResNet VLMs REG 0.23 0.41 0.68 0.44 0.09 0.10 0.27 0.15

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

S4-Driver [276] 2025 ViT-G PaLI3 LH 0.13 0.28 0.51 0.31 - - - SOLVE [32] 2025 EVA-02 LLaVA-1.5 REG 0.13 0.25 0.47 0.28 0.00 0.16 0.43 0.20 VERDI [63] 2025 VAD Qwen2.5-VL REG 0.36 0.62 0.96 0.65 - - - -

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

OmniReason [165] 2025 EVA-02 LLaVA-1.5 LH 0.15 0.31 0.57 0.34 0.04 0.18 0.98 0.40 FutureSightDrive [312] 2025 ViT Qwen2-VL LH 0.14 0.25 0.46 0.28 0.03 0.06 0.21 0.10

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

Occ-LLM [284] 2025 - LLaMA2 LH 0.12 0.24 0.49 0.28 - - - -

[Figure 875]

[Figure 876]

[Figure 877]

FastDriveVLA [18] 2025 Qwen2.5-VL Qwen2.5-VL LH 0.14 0.29 0.54 0.33 0.00 0.18 0.70 0.29 AutoDrive-R² [310] 2025 Qwen2.5-VL Qwen2.5-VL LH 0.13 0.19 0.25 0.19 - - - -

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

VDrive [76] 2025 Qwen2.5-VL, CVQ InternVL3 GEN 0.12 0.26 0.50 0.29 0.03 0.16 0.36 0.18 OccVLA [166] 2025 VQ-VAE PaliGemma-2 REG 0.18 0.26 0.40 0.28 - - - -

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

FasionAD++ [207] 2024 GenAD CLIP, QwenVL GEN 0.13 0.26 0.45 0.28 0.05 0.08 0.15 0.09

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

ALN-P3 [182] 2025 VAD CLIP, LLaMA-2 REG - - - - 0.05 0.09 0.35 0.16 VLM-E2E [163] 2025 EfficientNet CLIP REG 0.28 0.50 0.80 0.53 0.01 0.06 0.20 0.09

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

NetRoller[277] 2025 CLIP LLaMA-2 REG 0.38 0.66 1.01 0.68 0.06 0.13 0.30 0.16 OmniScene [164] 2025 ResNet Qwen2.5-VL REG 0.28 0.53 0.91 0.57 0.00 0.04 0.19 0.08 Sce2DriveX [327] 2025 OpenCLIP Vicuna-1.5 LH 0.15 0.33 0.59 0.36 - - - -

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

dVLM-AD [178] 2025 SigLIP2 LLaDA-V LH 0.15 0.40 0.68 0.41 - - - Percept-WAM [79] 2025 InternViT InternVL2 REG 0.16 0.33 0.60 0.36 - - - -

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

Reasoning-VLA [316] 2025 Qwen2.5-VL Qwen2.5-VL REG 0.05 0.19 0.41 0.22 0.02 0.06 0.13 0.07

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

Representative examples include BDD-X [117], which extends BDD100K with time-aligned human rationales, where annotators describe why the driver performed a specific action. This dataset provides early grounding for language-based explanations. DriveLM [225] constructs graph-structured question-answer pairs based on nuScenes and CARLA scenarios. These QA pairs target conditional reasoning, enabling models to infer high-level intent, spatial relations, and driving decisions. Impromptu VLA [35] aggregates data from eight public driving datasets and supplements them with captions, instructions, and QA pairs aligned with expert trajectories. The focus is on corner cases and long-tailed events. Other datasets, such as LingoQA [186] and CoVLA [2], collect real-world driving videos paired with natural language QA or behavior descriptions, emphasizing spatiotemporal reasoning and human-understandable driving motivations.

Notably, QA-style annotations have emerged as a dominant paradigm for extending driving datasets, serving as a common foundation for training and evaluating reasoning and planning capabilities [86, 225]. However, the scope and assumptions embedded in such annotations naturally influence model behavior, motivating further exploration of more diverse perspectives, planning horizons, and evaluation protocols for real-world deployment.

- 5.2 Evaluation Metrics

Evaluation metrics differ according to the model’s output modality: trajectory-based metrics for continuous action prediction and text-based metrics for models producing linguistic commands or rationales.

5.2.1 Trajectory-Based Action Evaluation

Trajectory-based outputs are typically evaluated in open-loop and closed-loop settings.

Open-Loop Evaluation. The predicted trajectory is directly compared against expert trajectories without executing in a simulator. Metrics such as L2 error and collision rate [91], along with trajectory-based indicators including Average Displacement Error (ADE), Final Displacement Error (FDE), and Miss Rate (MR) [93], are widely used. These metrics regard human driving demonstrations as the ground truth and formulate planning essentially as an imitation learning task. By measuring the deviation between predicted and expert trajectories, they provide a straightforward way to assess the accuracy of motion prediction.

Closed-Loop Evaluation. Instead, it measures the model’s performance when interacting with a simulation environment (e.g., CARLA [56]). Representative metrics include route completion (RC), driving score (DS), and infraction distance (ID). Bench2Drive [103] further considers success rate, efficiency, and comfort. NAVSIM [47], built on nuPlan [17], introduces the Predictive Driver Model Score (PDMS), which aggregates subscores for ego progress (EP), time-to-collision (TTC), and comfort (C), while applying penalties on collisions (NC) and driving admissibility (DAC). These metrics provide a holistic view of the safety, feasibility, and deployability of planning actions.

- 5.2.2 Text-Based Action Evaluation

For low-level vehicle control expressed in natural language, evaluation covers both linguistic quality and control effectiveness. Standard text metrics, such as BLEU, ROUGE, and CIDEr, are commonly used to assess the quality of generated language [225, 328], which measures n-gram overlap with human-annotated reference commands. Beyond command accuracy, reasoning quality is assessed through rationale consistency [192] and human preference ratings of language explanations, particularly in datasets following the BDD-X [117] format. To assess driving applications, execution-based metrics are introduced for behavior assessment. SimLingo [213] introduces an action-dreaming benchmark. The corresponding actions are mapped from the input instruction, which is open-loop evaluated using the success rate.

Regardless of output modality, these benchmarks emphasize key aspects of action quality, including accuracy, executability, safety, and intention alignment.

- 5.3 Quantitative Experiments & Analyses

This section reviews quantitative benchmarks for evaluating VLA models across action prediction, planning accuracy, and closed-loop driving performance. Among them, nuScenes [16], NAVSIM [47], and Bench2Drive [103]

- Table 6 Comparisons of state-of-the-art models on the WOD-E2E [283] test split.

- • Input: :Camera, : Sys. Prompt, : Instruct., : Scene Descrip., : Status, : Traffic Rule, : Context Info.
- • Action: LH: Language Head, RL: Policy with Reinforcement Learning, REG: Decoder + MLP, and GEN: Trajectory Generation w/ Generative Model.

- • Evaluation Metrics: RFS (Overall/Spotlight) (↓): Rater Feedback Score, ADE 5s/3s (↓): Average Displacement Error. Model Year Input Vision Language Action RFS(Overall)(↑) RFS(Spotlight)(↑) ADE 5s(↓) ADE 3s(↓)

[Figure 931]

- • Vision-Action Models

Waymo Baseline 2025 - - - 7.53 6.60 3.02 1.32 Swin-Trajectory [283] 2025 SwinT - REG 7.54 6.68 2.81 1.21

[Figure 932]

[Figure 933]

DiffusionDrive [155] 2025 ResNet - GEN 7.69 6.65 2.99 1.31 RAP-DINO [64] 2025 DINO - REG 8.04 7.20 2.65 1.17

[Figure 934]

[Figure 935]

• Vision-Language-Action Models

OpenEMMA [278] 2025 Qwen2-VL Qwen2-VL LH 5.16 4.71 12.74 6.68 HMVLM [256] 2025 ViT Qwen2.5-VL LH 7.74 6.73 3.07 1.33 AutoVLA [340] 2025 Qwen2.5-VL Qwen2.5-VL LH 7.56 6.94 2.96 1.35 Poutine [216] 2025 ViT Qwen2.5-VL LH 7.99 6.89 2.74 1.21 LightEMMA [209] 2025 Qwen2.5-VL Qwen2.5-VL LH 6.52 5.71 3.73 1.71 dVLM-AD [178] 2025 SigLIP2 LLaDA-V LH 7.63 - 3.02 1.29

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

###### are the most widely used. More recently, WOD-E2E [283] introduces long-tail, safety-critical scenes with human-preference annotations, enabling more robust assessment of modern E2E and VLA systems.

- 5.3.1 nuScenes Benchmark

The nuScenes open-loop benchmark evaluates planning quality using trajectory-based metrics, including L2 displacement error and Collision Rate, as summarized in Table 5. Basically, vision-action models such as UniAD [93] reports 0.69m L2 and 0.12 collision rate. Incorporating language generally improves performance by providing semantic cues for safer planning. For instance, Drive-R1 [146] combines supervised CoT alignment with RL finetuning to reach 0.31m L2 and 0.09 collision rate.

Beyond accuracy, recent studies explore the role of language in handling complex and long-tailed driving scenarios. While improvements are most evident in common cases, rare and highly complex situations remain an active area of investigation, motivating the integration of richer reasoning signals and data sources.

From a systems perspective, computational efficiency is an important consideration for practical deployment. Lightweight and efficiency-oriented designs, such as InsightDrive [227] (16.3 FPS) and token-pruned architectures like FastDriveVLA [18], illustrate ongoing efforts to balance model capacity with real-time feasibility. For cross-domain evaluation, nuScenes highlights generalization to unseen cities and distribution shifts as a key benchmark dimension. Works such as VLP [197] and DiMA [88] examine this setting and motivate complementary strategies including domain adaptation, distillation, and data augmentation.

- 5.3.2 WOD-E2E Benchmark

The Waymo Open Dataset for End-to-End Driving (WOD-E2E) [283] is a large-scale benchmark designed to evaluate end-to-end driving systems under long-tail, safety-critical scenarios that rarely appear in conventional datasets. It contains 4K segments with high-level routing commands, ego-status signals, and multi-camera views, enabling rigorous assessment of perception-planning coupling. A key contribution of WOD-E2E is the Rater Feedback Score (RFS), which measures trajectory quality based on alignment with human preference annotations rather than logged expert trajectories. As shown in Table 6, RFS (Overall and Spotlight) complements conventional ADE metrics by providing a more human-aligned assessment of driving behavior.

Overall results indicate that while vision–action models achieve stable displacement accuracy, VLA models exhibit more diverse performance. Approaches such as Poutine [216], and dVLM-AD [178] achieve balanced RFS and ADE performance, highlighting the importance of effectively aligning language reasoning with trajectory generation. Figure 7 presents the visualized performance of AutoVLA [340] in the WOD-E2E dataset.

- Table 7 Comparisons of state-of-the-art models for Open-Loop Planning on the NAVSIM [47] navtest benchmark.

- • Input: :Camera, : LiDAR, : Prompt, : Instruct., : Scene Descrip., : Status, : Rule, : Context.
- • Action: LH: Language Head, RL: Policy with Reinforcement Learning, REG: Decoder + MLP, SEL: Trajectory Selection w/ Cost, and GEN: Trajectory Generation w/ Generative Model.

- • Evaluation Metrics: NC: Navigation Completion, DAC (↑): Driving Accuracy, TTC (↑): Time-To-Collision, Comf. (↑): Comfort, EP (↑): Ego Progress, and PDMS (↑): Perception Driving Metric Score.

Model Year Input Vision Language Action NC(↑) DAC(↑) TTC(↑) Comf(↑) EP(↑) PDMS(↑)

[Figure 964]

• Vision-Action Models

TransFuser [39] 2022 ResNet - REG 97.7 92.8 92.8 100 79.2 84.0 UniAD [93] 2023 ResNet - REG 97.8 91.9 92.9 100 78.8 83.4 VADv2 [30] 2024 ResNet - REG 97.2 89.1 91.6 100 76.0 80.9

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

PARA-Drive [268] 2024 ResNet - REG 97.9 92.4 93.0 99.8 79.3 84.0 LAW [142] 2024 Swin-T - REG 96.4 95.4 88.7 99.9 81.7 84.6 DRAMA [308] 2024 ResNet - REG 98.0 93.1 94.8 100 80.1 85.5 DiffusionDrive [155] 2024 ResNet - GEN 98.2 96.2 94.7 100 82.2 88.1 WoTE [144] 2025 ResNet - SEL 98.5 96.8 94.9 99.9 81.9 88.3 World4Drive [335] 2025 ResNet - REG 97.4 94.3 92.8 100 79.9 85.1 DrivingGPT [33] 2025 VQ-VAE - REG 98.9 90.7 94.9 95.6 79.7 82.4

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

AD-R1 [289] 2025 - - RL 98.7 97.8 94.8 100 87.5 91.9 SeerDrive [315] 2025 VoVNet - SEL 98.8 98.6 95.8 100 84.2 90.7

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

Epona [321] 2025 DC-AE - REG 97.9 95.1 93.8 99.9 80.4 86.2 GoalFlow [279] 2025 VoVNet - GEN 98.4 98.3 94.6 100 85.0 90.3 TrajDiff [75] 2025 ResNet - GEN 98.1 97.0 94.3 100.0 82.7 88.5 DiffusionDriveV2 [345] 2025 ResNet - GEN 98.3 97.9 94.8 99.9 87.5 91.2 NaviHydra [271] 2025 ResNet - SEL 98.7 98.6 88.7 96.2 100.0 92.7 Mimir [280] 2025 ResNet - GEN 98.2 97.5 94.6 100 83.6 89.3

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

• Vision-Language-Action Models ReCogDrive [145] 2025 InternViT Qwen2.5-VL GEN 98.2 97.8 95.2 99.8 83.5 89.6

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

AutoVLA [340] 2025 Qwen2.5-VL Qwen2.5-VL LH 99.1 97.1 97.1 99.9 87.6 92.1 ReflectDrive [135] 2025 LLaDA-V LLaDA-V GEN 99.7 99.5 99.1 99.9 88.9 94.7

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

AdaThinkDrive [176] 2025 InternVL3 InternVL3 REG 99.1 98.8 97.2 100.0 87.9 93.0 Percept-WAM [79] 2025 InternViT InternVL2 REG 98.8 98.6 94.4 99.5 84.8 90.2 Reasoning-VLA [316] 2025 Qwen2.5-VL Qwen2.5-VL REG 97.8 93.2 98.1 99.8 80.7 91.7

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

- 5.3.3 NAVSIM Benchmark

NAVSIM [47] is built on OpenScene (a redistribution of nuPlan [17]), provides a closed-loop simulation environment designed to evaluate planning quality under realistic urban driving conditions. It adopts the PDMS metric, which aggregates multiple driving aspects, including No-Collision (NC), Driving Admissibility (DAC), Time-to-Collision (TTC), Ego Progress (EP), and Comfort (C), offering a holistic assessment of safety, efficiency, and driving smoothness. As shown in Table 7, most methods achieve strong performance on safety-related metrics such as NC and DAC, while TTC and EP serve as more discriminative indicators of planning foresight and long-horizon decision quality. These metrics highlight differences in how models balance safety and progress when interacting with dynamic environments.

Representative vision–action approaches, such as WoTE [144], achieve 88.3 PDMS by integrating a BEV-based world model with reward-guided trajectory selection, demonstrating the effectiveness of structured world modeling for closed-loop planning. Building upon this foundation, language-conditioned methods further enhance decision-making. For instance, AutoVLA [340] improves performance to 99.1 NC and 87.6 EP by leveraging language-driven decision priors and a Best-of-N oracle scoring strategy, illustrating how language supervision can guide trajectory selection and improve long-horizon planning behavior.

- 5.3.4 Bench2Drive Benchmark

Bench2Drive [103] provides a closed-loop evaluation protocol built on CARLA V2, focusing on success rate and composite driving scores to assess goal-directed driving behavior under interactive settings. Unlike open-loop benchmarks, Bench2Drive explicitly evaluates an agent’s ability to execute long-horizon tasks and respond to dynamic environmental feedback.

Recent VLA approaches demonstrate clear benefits from language grounding in this benchmark. For example, SimLingo [213] introduces an action dreaming mechanism that aligns natural language instructions with control sequences, achieving a leading driving score of 85.94, as reported in Table 8. These results indicate

- Table 8 Closed-loop and Open-loop performance comparison of E2E-AD Methods on the Bench2Drive benchmark.

- • Input: : Camera, : LiDAR, : Prompt, : Instruct., : Scene Descrip., : Status, : Rule, : Context.
- • Action: LH: Language Head, RL: Policy with Reinforcement Learning, REG: Decoder + MLP, SEL: Trajectory Selection w/ Cost, and GEN: Trajectory Generation w/ Generative Model.

- • Evaluation Metrics: DS (↑): Driving Score, SR (↑): Success Rate. Avg. L2 (↓): Averaged L2 distance of trajectory.

Method Year Input Vision Language Action

[Figure 1030]

Closed-Loop Open-Loop

DS↑ SR(%)↑ Efficiency↑ Comfort↑ Avg. L2 ↓

- • Vision-Action Models

TCP [272] 2022 ResNet - REG 40.70 15.00 54.26 47.80 1.70 ThinkTwice [102] 2023 ResNet - REG 62.44 31.23 69.33 16.22 0.95

[Figure 1031]

[Figure 1032]

DriveAdapter [101] 2023 ResNet - REG 64.22 33.08 70.22 16.01 1.01 UniAD-Base [93] 2023 ResNet - REG 45.81 16.36 129.21 43.58 0.73

[Figure 1033]

[Figure 1034]

[Figure 1035]

VAD [107] 2023 ResNet - REG 42.35 15.00 157.94 46.01 0.91 GenAD [332] 2024 ResNet - GEN 44.81 15.90 - - -

[Figure 1036]

[Figure 1037]

DriveTransformer [104] 2025 ResNet - REG 63.46 35.01 100.64 20.78 0.62 ETA [78] 2025 CLIP - REG 69.53 38.64 184.51 28.43 -

[Figure 1038]

[Figure 1039]

WoTE [144] 2025 ResNet - SEL 61.71 31.36 - - GuideFlow [161] 2025 ResNet - GEN 75.21 51.36 - - Raw2Drive [300] 2025 ResNet - RL 71.36 50.24 214.17 22.42 -

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

• Vision-Language-Action Models

ORION [66] 2025 EVA-02 Vicuna-1.5 GEN 77.74 54.62 151.48 17.38 0.68 AutoVLA [340] 2025 SigLIP Qwen2.5-VL LH 78.84 57.73 146.93 39.33 -

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

SimLingo-Base [213] 2025 InternViT Qwen2 REG 85.94 66.82 244.18 25.49 SimLingo [213] 2025 InternViT Qwen2 REG 85.07 67.27 259.23 33.67 -

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

ReAL-AD [175] 2025 ResNet QwenVL REG 40.76 10.93 - - 0.87 ReasonPlan [168] 2025 SigLIP Qwen LH 64.01 34.55 180.64 25.63 0.61

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

DriveMoE [299] 2025 BEV Encoder LLaMA REG 74.22 48.64 175.96 15.31 0.31

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

VDRive[76] 2025 Qwen2.5-VL, CVQ InternVL3 GEN 66.15 50.51 110.23 22.90 0.55 CoReVLA [60] 2025 Qwen2.5-VL Qwen2.5-VL LH 72.18 50.00 145.41 34.35 -

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

that language-guided reasoning can effectively influence closed-loop decision-making and improve planning performance in interactive driving scenarios.

Taken together with open-loop benchmarks, Bench2Drive highlights the growing importance of language–action alignment in VLA systems, particularly for interpreting high-level goals, guiding long-horizon behavior, and adapting actions under complex, human-centered instructions.

### 6 Challenges & Future Directions

VLA models mark a shift from modular stacks toward holistic, reasoning-driven driving agents. By leveraging large multimodal backbones, they promise richer environmental understanding, stronger generalization, and more interpretable decision-making. Yet, realizing their full potential in safety-critical autonomous driving requires addressing several fundamental challenges. In parallel, emerging research directions point toward next-generation systems that are more efficient, trustworthy, and capable of long-horizon reasoning.

- 6.1 Current Challenges

- 6.1.1 Model Architecture and System Efficiency

Real-Time Processing and Latency. VLA models inherit the substantial computational footprint of modern vision-language backbones. High-resolution, high-frame-rate camera inputs generate long visual-token sequences, and multi-view fusion amplifies memory and latency costs. Meeting the strict real-time constraints of autonomous vehicles, therefore, remains difficult [268, 273]. Recent advances in streaming token compression and adaptive visual encoders [18, 138] offer promising directions, but achieving sub-50ms inference remains an unmet requirement for safety-critical deployment.

Lack of Domain-Specific Foundation Models. General-purpose VLMs [5, 34, 343] provide strong priors but are not optimized for driving-specific perception, physics, or multi-sensor fusion. Autonomous driving requires precise spatial reasoning, adherence to traffic rules, and an understanding of rare, high-stakes edge cases – abilities not fully captured by generic models. As highlighted in Section 3, dedicated driving foundation models [151] remain a missing cornerstone for scalable and dependable VLA systems.

|[Figure 1088]<br><br>Ground Truth Planning|
|---|

###### <think> This is a complex scenario requiring additional reasoning. Scene Description:

###### Slow Thinking

The scene is set during the daytime in an urban area with clear weather. The road is undergoing construction, as indicated by orange barriers and traffic cones placed along both sides of the road. The front view shows a straight path ahead with no immediate obstacles. The front left and front right views reveal buildings and parked cars on either side of the road. The back view shows a clear path behind the vehicle with no following traffic. The traffic light ahead is green, allowing vehicles to proceed.

###### Critical Object Description:

- 1. Traffic Cones: Located on both sides of the road, these cones mark the boundaries of the construction zone. They are stationary and serve as visual guides for navigating the narrowed lane. - What they are doing: Marking the construction area. - What they will do: Remain stationary. - Why they are important: They help the ego vehicle stay within the designated lane and avoid entering the construction zone.
- 2. Green Traffic Light: Positioned ahead, it indicates that vehicles can proceed. - What it is doing: Allowing vehicles to move forward. - What it will do: Stay green unless the traffic signal changes. - Why it is important: It dictates the flow of traffic and confirms that the ego vehicle can continue moving forward. Reasoning on Intent:

- - Firstly, the traffic cones indicate that the ego vehicle must stay within the marked lane to navigate the construction zone safely. This requires the vehicle to maintain a steady course and possibly reduce speed to ensure precise navigation.
- - Secondly, the green traffic light confirms that the ego can proceed without stopping. However, given the construction and narrowed lane, it is prudent to approach cautiously.
- - The ego vehicle has been moving forward at a constant speed in the past 4 seconds, and the current driving command is to go straight. Considering the construction zone and the presence of cones, it would be prudent to decelerate slightly to maintain control and safety.

###### Best Driving Action:

Move forward with a deceleration. This action allows the ego to navigate the construction zone safely while maintaining control and avoiding any potential hazards posed by the narrowed lane and construction equipment. Decelerating slightly ensures that the vehicle can respond promptly to any unforeseen obstacles or changes in the road conditions. </think>

<answer> The final output action is: <action_119><action_590>… </answer>

|[Figure 1089]<br><br>Ground Truth Planning|
|---|

Slow Thinking

###### <think> This is a complex scenario requiring additional reasoning. Scene Description:

The scene is a residential area during the daytime with clear weather. The ego vehicle is approaching a T-intersection with a visible "STOP" marking on the road. There are several parked cars along both sides of the street. A few vehicles are visible in the distance, traveling on the intersecting road. No pedestrians are present near the intersection. The traffic appears calm, with no immediate obstacles or hazards.

###### Critical Object Description:

- 1. STOP Marking: Located directly in front of the ego vehicle on the road. This indicates that the vehicle must come to a complete stop before proceeding.
- 2. Vehicles on the Intersecting Road: Positioned further ahead, beyond the intersection. Reasoning on Intent:

- - Firstly, the presence of the "STOP" marking requires the ego vehicle to come to a complete stop. This is a legal obligation and ensures safety by allowing the vehicle to check for any oncoming traffic or pedestrians before proceeding.
- - Secondly, observing the vehicles on the intersecting road helps determine when it is safe to proceed. Since they appear to be either stopped or moving slowly, the ego vehicle can likely proceed once it has come to a full stop and confirmed that the way is clear.
- - The ego vehicle has been decelerating in the past 4 seconds, indicating an intention to slow down and prepare for the stop. Given the current velocity and acceleration, a quick deceleration is necessary to bring the vehicle to a halt promptly.
- - The driving command is to go straight, which aligns with the ego vehicle's current trajectory and the absence of any immediate obstacles or hazards.

###### Best Driving Action:

Move forward with a quick deceleration. This action ensures that the ego vehicle comes to a complete stop at the “STOP” marking while maintaining control and safety. After stopping, the vehicle can then assess the situation and proceed if it is safe to do so. </think>

<answer> The final output action is: <action_990><action_932>… </answer>

|[Figure 1090]<br><br>From Test Set, No Ground Truth Planning|
|---|

Slow Thinking

###### <think> This is a complex scenario requiring additional reasoning. Scene Description:

The scene is a residential street during the daytime with clear weather. The road is lined with parked cars on both sides, and there are trees along the sidewalks. Two children are crossing the road in front of the ego vehicle, one on a scooter and the other walking. The front left and front right cameras show parked cars and houses on either side of the street. There are no visible traffic lights or stop signs in the immediate vicinity.

###### Critical Object Description:

- 1. Two children crossing the road: The Located in the center of the front view, the child on the scooter is crossing the road while the other child is walking. They appear to be mid-crossing and are likely to finish crossing soon. Their presence is crucial as the ego vehicle must ensure they have safely crossed before proceeding.
- 2. Parked Cars: Located on both sides of the street, these cars are stationary and do not pose an immediate threat. However, they restrict the lateral movement of the ego vehicle and require caution to avoid any potential collisions. Reasoning on Intent:

- 1. Two children crossing the road: The two children are currently crossing the road, and the ego vehicle must wait until they have safely crossed. Once they have finished crossing, the ego vehicle can proceed.
- 2. Parked Cars: The parked cars limit the lateral movement options. The ego vehicle cannot change lanes or make wide turns without risking collision with the parked cars.
- 3. Driving Command Instruction: The driving command indicates "go straight”, which aligns with the current situation as the ego vehicle is positioned in the center of the road. Given that the two children are expected to finish crossing soon and the road ahead is clear, the ego should prepare to move forward once the children have cleared the path.

###### Best Driving Action:

Move forward with a quick acceleration. This action is appropriate because once the two children have safely crossed the road, the ego vehicle can quickly accelerate forward to maintain smooth traffic flow while ensuring safety. The quick acceleration helps in efficiently utilizing the road space and maintaining a safe distance from other vehicles. </think>

<answer> The final output action is: <action_0><action_0>… </answer>

|[Figure 1091]<br><br>From Test Set, No Ground Truth Planning|
|---|

Slow Thinking

###### <think> This is a complex scenario requiring additional reasoning. Scene Description:

The scene is a residential area during a snow-covered day. The road appears to be a two-lane street with houses on both sides. There are no visible vehicles directly in front of the ego vehicle, but there are parked cars along the sides of the road. The front camera shows a clear path ahead with no immediate obstacles. The front left and front right cameras reveal open spaces on either side of the road, with no significant traffic or pedestrians present. The back camera confirms that the road behind is clear as well.

###### Critical Object Description:

- 1. Parked Cars: Located on both sides of the road, these cars are stationary and pose no immediate threat to the ego vehicle's movement. However, they should be monitored for any potential opening doors or pedestrians emerging.
- 2. Residential House: These structures line both sides of the road and do not directly impact the ego vehicle's path. However, they may obscure the view of pedestrians or other vehicles entering the road. Reasoning on Intent:

- 1. Clear Path Ahead: The front camera indicates a clear path with no immediate obstacles, suggesting that the ego vehicle can proceed forward without interruption.
- 2. No Vehicles or Pedestrians: The absence of any vehicles or pedestrians in the immediate vicinity allows the ego vehicle to maintain its current trajectory without needing to adjust speed or direction.
- 3. Driving Command Instruction: The current driving command is "go straight," which aligns with the observed conditions and the vehicle's recent behavior of deceleration followed by stopping. Given the clear path ahead, lack of nearby traffic, and the driving command to go straight, the ego vehicle should aim to resume a constant speed.

###### Best Driving Action:

Move forward with a constant speed. This action ensures the vehicle maintains a steady pace while adhering to the driving command and the current road conditions, which are safe and free of immediate hazards. </think>

<answer> The final output action is: <action_337><action_1041>… </answer>

###### Figure 7 Visualization examples of the AutoVLA [340] reasoning/planning results on WOD-E2E [283] dataset.

- 6.1.2 Data and Generalization

Generalizing to Rare and Novel Scenarios. One motivation for VLAs is their ability to leverage strong visual-language priors to interpret complex scenes. However, while VLM components may generalize well perceptually, aligning this understanding to the action space introduces new uncertainties. As noted in Section 4, reasoning-rich representations do not automatically translate to robust action generation. Longtailed scenarios – misbehaving traffic agents, unusual road layouts, unpredictable weather – remain failure points [48, 73, 247, 302, 311].

Cost of High-Quality Data. VLAs rely on diverse, high-quality multimodal datasets [131, 162], yet collecting paired vision-action-language triplets at scale is expensive. Synthetic environments [21, 191, 228] help, but face substantial sim-to-real gaps, with discrepancies in noise characteristics, lighting, and behavior of other agents [121, 275]. Improving data efficiency and mitigating distribution shifts remain long-standing challenges.

- 6.1.3 Core Capabilities and Trustworthiness

Interpretability & Hallucination. While VLA models produce natural-language rationales via chain-of-thought prompting [66, 261, 340], these explanations are generated artifacts – not faithful reflections of the underlying causal reasoning. Language hallucination [94, 158, 211] presents new risks: the model may justify an incorrect decision with a confident but spurious narrative. Ensuring consistent grounding between perceptions, actions, and explanations is an open challenge.

Long-Horizon Temporal Coherence. Driving depends on anticipating multi-stage interactions and maintaining situational awareness across extended time horizons [199, 229, 285]. Current transformer-based VLA architectures remain constrained by limited context windows and short-term conditioning, inherited from standard VLM designs. Temporal fragmentation leads to inconsistent decisions, especially in multi-agent or highly dynamic traffic scenes.

6.2 Future Directions

- 6.2.1 Next-Generation Model Paradigms

Unified Vision-Language-World Models. A promising evolution integrates VLA with predictive world models [8, 20, 142, 143], extending the VA-based models in Section 3.2. Rather than reacting frame by frame, such systems simulate future scene evolution conditioned on candidate actions, enabling proactive planning and more reliable behavior under uncertainty. Building unified, end-to-end world models that jointly reason about perception, language, and dynamics may form the backbone of next-generation autonomous agents.

Richer Multimodal Fusion. As sensor suites diversify, future architectures will incorporate early and tight fusion of LiDAR, Radar, event cameras, and high-definition maps [121, 222, 275]. Language enhances semantic grounding, but robust 3D geometry is indispensable for safe decision-making [166, 248]. Holistic multimodal fusion can combine the interpretability of VLMs with the spatial precision of geometric sensors.

- 6.2.2 Advancing Intelligence and Adaptation

Socially Aware, Knowledge-Grounded Driving. VLA models must acquire deeper commonsense reasoning – understanding intent, conventions, and causal relationships beyond explicit annotations [139, 313]. Future efforts will draw from large-scale video-language corpora, leveraging external knowledge bases and structured reasoning modules to support socially compliant and anticipatory driving.

Continual & Onboard Learning. Static, offline-trained models cannot capture evolving road infrastructures or regional driving customs [19, 255, 344]. Enabling safe, incremental learning from everyday driving, while avoiding catastrophic forgetting and ensuring safety guarantees, is essential for long-term deployment. This relates closely to addressing long-tail generalization gaps.

###### Table 9 Summary of the evaluation metrics used for evaluating the trajectory-based and text-based action outputs.

|Abbr.|-|Full Name|Description|Ref.|
|---|---|---|---|---|

|Action-Planning Open-Loop Evaluation| | | | |
|---|---|---|---|---|
|L2|↓<br><br>|L2 Error|L2 distance error between the planned trajectory and the human driving trajectory in 3 seconds.<br><br>|[91]|
|CR<br><br>|↓<br><br>|Collision Rate|How often the self-driving vehicle would collide with other agents on the road.<br><br>|[91]|
|ADE<br><br>|↓|Average Displacement Error<br><br>|Mean displacement error between predicted trajectories and expert waypoints across the horizon, reflecting overall trajectory accuracy.<br><br>|[93]|
|FDE|↓<br><br>|Final Displacement Error|Displacement error at the final predicted waypoint compared with expert trajectories, emphasizing long-term accuracy.|[93]|
|MR|↓|Miss Rate<br><br>|Fraction of prediction time steps where displacement error exceeds horizon-specific thresholds, reflecting failure in trajectory coverage.|[93]|
|AHE<br><br>|↓|Average Heading Error<br><br>|Mean absolute angular deviation between predicted and expert heading over the trajectory horizon, measuring orientation accuracy.|[114]|
|FHE<br><br>|↓|Final Heading Error<br><br>|Absolute angular deviation of predicted heading from expert at the final timestep, reflecting terminal orientation accuracy.|[114]|
|SLE|↓<br><br>|Speed L1 Error|Mean absolute error of predicted speed control signals.|[100]|
|SALE|↓<br><br>|Steer Angle L1 Error|Mean absolute error of predicted steering angle control signals.<br><br>|[100]|
|RFS<br><br>|↑|Rater Feedback Score<br><br>|Measure how well the predicted trajectory aligns with human driving preferences by checking whether it falls within trust regions.|[283]|

|Trajectory-Based Closed-Loop Evaluation| | | | |
|---|---|---|---|---|
|RC<br><br>|↑|Route Completion<br><br>|The percentage of route distance completed.|[206]|
|DS|↑|Driving Score|RC weighted by a penalty factor that accounts for collisions with pedestrians, vehicles, etc.<br><br>|[206]|
|NC|↑<br><br>|No Collision|Fraction of scenarios without ego-fault collisions, focusing exclusively on responsibility-aware collision evaluation.<br><br>|[47]|
|DAC<br><br>|↑|Driving Admissibility Check<br><br>|Boolean evaluation that checks whether the ego vehicle remains inside drivable polygons throughout the rollout.|[47]|
|TTC|↑<br><br>|Time To Collision|Boolean verification that the time-to-collision value exceeds safety thresholds, preventing imminent crashes.<br><br>|[47]|
|C|↑|Driving Comfort<br><br>|The comfort of driving.|[47]|
|EP|↑<br><br>|Ego Progress|Penalization of excessive jerk, acceleration, or yaw-rate, reflecting ride quality and passenger comfort.<br><br>|[47]|
|PDMS<br><br>|↑<br><br>|Predictive Driver Model Score|A flexible weighted evaluation score in autonomous driving that aggregates multiple safety, progress, and comfort subscores into a single metric.<br><br>|[47]|
|SR|↑|Success Rate<br><br>|Percentage of navigation episodes that successfully reach the goal within a fixed time budget, indicating overall task completion.|[56]|
|ID<br><br>|↑|Infraction Distance<br><br>|Average driving distance between two infractions, with longer distances reflecting safer and more reliable policy behavior.|[56]|

|Text-Based Action Evaluation| | | | |
|---|---|---|---|---|
|CIDEr<br><br>|↑|Consensus-based Image Description Evaluation<br><br>|Measures similarity of generated captions to multiple human references using TF-IDF weighted n-grams.|[182]|
|BLEU<br><br>|↑|Bilingual Evaluation Understudy<br><br>|Precision-based metric that compares n-grams of the generated text against reference texts.|[182]|
|METEOR<br><br>|↑<br><br>|Metric for Evaluation of Translation with Explicit Ordering|Considers unigram precision and recall with stemming, synonym matching, and fragmentation penalty.<br><br>|[182]|
|Rouge|↑|Recall-Oriented Understudy for Gisting Evaluation<br><br>|Recall-focused metric using overlapping n-grams, word sequences, or word pairs between generated and reference texts.|[182]|
|Top-1 Acc<br><br>|↑<br><br>|Visual Question Answering Top-1 Accuracy|Percentage of predictions where the most confident output matches the ground truth label.<br><br>|[182]|

- 6.2.3 Ecosystem for Safe Deployment

Standardized Evaluation & Safety Guarantees. Evaluation metrics from current benchmarks, e.g., NAVSIM [47] and Bench2Drive [103], assess safety and comfort but do not capture key VLA-specific risks such as reasoning failures, instruction-following errors, or cross-modal inconsistencies [158, 211]. Future benchmarks should evaluate multi-step instruction execution, robustness to ambiguous language, and resistance to hallucination. Beyond empirical testing, formal verification tools are needed to provide theoretical guarantees for safety-critical behaviors.

Human-Centric Interaction & Personalization. VLA systems open the door to richer in-car interaction [219, 295]. Natural language enables drivers to specify goals, constraints, and preferences (“drive cautiously”, “avoid unprotected left turns”). Personalization modules [81] can adapt driving styles to different users, enhancing comfort and trust. The challenge lies in balancing personalization with strict safety and regulatory requirements.

- 7 Conclusion

Vision-Language-Action models are reshaping autonomous driving by coupling perception with high-level reasoning and natural language understanding. This work formalizes the VLA problem setting, outlines the progression from traditional VA pipelines, and organizes existing methods into coherent architectural families together with the datasets and benchmarks that support their development. VLA systems offer clear advantages in interpretability, generalization, and human interaction, but core challenges remain: aligning symbolic reasoning with continuous control, ensuring robustness in long-tail scenarios, and establishing evaluation protocols that faithfully measure instruction following and safety. Progress will depend on advances in efficient architectures, deeper multimodal fusion, world-model-driven planning, and more rigorous humancentered testing. Overall, VLA represents a promising direction for building autonomous agents that are not only competent drivers but also communicative, transparent, and responsive to human intent.

References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Hidehisa Arai, Keita Miwa, Kento Sasaki, Kohei Watanabe, Yu Yamaguchi, Shunsuke Aoki, and Issei Yamamoto. CoVLA: Comprehensive vision-language-action dataset for autonomous driving. In IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 1933–1943, 2025.
- [3] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Liang Feng, Francesco Ferroni, Rama Govindaraju, Jinwei Gu, Siddharth Gururani, Imad El Hanafi, Zekun Hao, Jacob Huffman, Jingyi Jin, Brendan Johnson, Rizwan Khan, George Kurian, Elena Lantz, Nayeon Lee, Zhaoshuo Li, Xuan Li, Maosheng Liao, Tsung-Yi Lin, Yen-Chen Lin, Ming-Yu Liu, Xiangyu Lu, Alice Luo, Andrew Mathau, Yun Ni, Lindsey Pavao, Wei Ping, David W. Romero, Misha Smelyanskiy, Shuran Song, Lyne Tchapmi, Andrew Z. Wang, Boxin Wang, Haoxiang Wang, Fangyin Wei, Jiashu

- Xu, Yao Xu, Dinghao Yang, Xiaodong Yang, Zhuolin Yang, Jingxu Zhang, Xiaohui Zeng, and Zhe Zhang. Cosmos-Reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.
- [6] Michael Bain and Claude Sammut. A framework for behavioural cloning. In Mach. intell., volume 15, pages 103–129, 1995.
- [7] Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann LeCun, Patrick Labatut, Maximilian Seitzer, and Piotr Bojanowski. Back to the features: DINO as a foundation for video world models. arXiv preprint arXiv:2507.19468, 2025.
- [8] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 3: A new frontier for world models, 2025. URL https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/.
- [9] Mayank Bansal, Alex Krizhevsky, and Abhijit Ogale. ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst. arXiv preprint arXiv:1812.03079, 2018.
- [10] Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, Renaud Marlet, Alexandre Boulch, Mickael Chen, Éloi Zablocki, Andrei Bursuc, Eduardo Valle, and Matthieu Cord. VaViM and VaVAM: Autonomous driving through video generative modeling. arXiv preprint arXiv:2502.15672, 2025.
- [11] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3B VLM for transfer. arXiv preprint arXiv:2407.07726, 2024.
- [12] Hengwei Bian, Lingdong Kong, Haozhe Xie, Liang Pan, Yu Qiao, and Ziwei Liu. DynamicCity: Large-scale 4D occupancy generation from dynamic scenes. In Int. Conf. Learn. Represent., 2025.
- [13] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [14] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 22563–22575, 2023.

- [15] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and Karol Zieba. End-to-end learning for self-driving cars. arXiv preprint arXiv:1604.07316, 2016.
- [16] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11621–11631, 2020.
- [17] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom, and Sammy Omari. nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021.
- [18] Jiajun Cao, Qizhe Zhang, Peidong Jia, Xuhui Zhao, Bo Lan, Xiaoan Zhang, Zhuo Li, Xiaobao Wei, Sixiang Chen, Liyun Li, Xianming Liu, Ming Lu, Yang Wang, and Shanghang Zhang. FastDriveVLA: Efficient end-to-end driving via plug-and-play reconstruction-based token pruning. arXiv preprint arXiv:2507.23318, 2025.
- [19] Zhong Cao, Xiang Li, Kun Jiang, Weitao Zhou, Xiaoyu Liu, Nanshan Deng, and Diange Yang. Autonomous driving policy continual learning with one-shot disengagement case. IEEE Trans. Intell. Veh., 8(2):1380–1391, 2022.
- [20] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.
- [21] Qianwen Chao, Huikun Bi, Weizi Li, Tianlu Mao, Zhaoqi Wang, Ming C. Lin, and Zhigang Deng. A survey on visual traffic simulation: Models, evaluations, and applications in autonomous driving. In Computer Graphics Forum, volume 39, pages 287–308. Wiley Online Library, 2020.
- [22] Raphael Chekroun, Marin Toromanoff, Sascha Hornauer, and Fabien Moutarde. GRI: General reinforced imitation and its application to vision-based autonomous driving. Robotics, 12(5):127, 2023.
- [23] Dian Chen and Philipp Krähenbühl. Learning from all vehicles. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 17222–17231, 2022.
- [24] Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by cheating. In Conf. Robot Learn., pages 66–75. PMLR, 2020.
- [25] Dian Chen, Vladlen Koltun, and Philipp Krähenbühl. Learning to drive from a world on rails. In IEEE/CVF Int. Conf. Comput. Vis., pages 15590–15599, 2021.
- [26] Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, and Han Cai. DC-AE 1.5: Accelerating diffusion model convergence with structured latent space. In IEEE/CVF Int. Conf. Comput. Vis., pages 19628–19637, 2025.
- [27] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. IEEE Trans. Pattern Anal. Mach. Intell., 46(12):10164–10183, 2024.
- [28] Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, and Wenping Wang. Towards label-free scene understanding by vision foundation models. In Adv. Neural Inf. Process. Syst., volume 36, pages 75896–75910, 2023.
- [29] Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, and Wenping Wang. CLIP2Scene: Towards label-efficient 3D scene understanding by CLIP. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 7020–7030, 2023.
- [30] Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. VADv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024.
- [31] Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, and Radu Soricut. PaLI-3 vision language models: Smaller, faster, stronger. arXiv preprint arXiv:2310.09199, 2023.

- [32] Xuesong Chen, Linjiang Huang, Tao Ma, Rongyao Fang, Shaoshuai Shi, and Hongsheng Li. SOLVE: Synergy of language-vision and end-to-end networks for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 12068–12077, 2025.
- [33] Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. DrivingGPT: Unifying driving world modeling and planning with multi-modal autoregressive transformers. arXiv preprint arXiv:2412.18607, 2024.
- [34] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 24185–24198, 2024.
- [35] Haohan Chi, Huan ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, Leichen Wang, Xingtao Hu, Hao Sun, Hang Zhao, and Hao Zhao. Impromptu VLA: Open weights and open data for driving vision-language-action models. arXiv preprint arXiv:2505.23757, 2025.
- [36] Wei-Lin Chiang et al. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality, 2023. URL https://vicuna.lmsys.org.
- [37] Pranav Singh Chib and Pravendra Singh. Recent advancements in end-to-end autonomous driving using deep learning: A survey. IEEE Trans. Intell. Veh., 9(1):103–118, 2023.
- [38] Kashyap Chitta, Aditya Prakash, and Andreas Geiger. NEAT: Neural attention fields for end-to-end autonomous driving. In IEEE/CVF Int. Conf. Comput. Vis., pages 15793–15803, 2021.
- [39] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. TransFuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 45(11):12878–12895, 2022.
- [40] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, and Chunhua Shen. MobileVLM v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.
- [41] Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In Int. Conf. Mach. Learn., pages 2048–2056. PMLR, 2020.
- [42] Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. End-to-end driving via conditional imitation learning. In IEEE Int. Conf. Robot. Autom., pages 4693–4700, 2018.
- [43] Felipe Codevilla, Eder Santana, Antonio M. Lopez, and Adrien Gaidon. Exploring the limitations of behavior cloning for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 9329–9338, 2019.
- [44] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, Tianren Gao, Erlong Li, Kun Tang, Zhipeng Cao, Tong Zhou, Ao Liu, Xinrui Yan, Shuqi Mei, Jianguo Cao, Ziran Wang, and Chao Zheng. A survey on multimodal large language models for autonomous driving. In IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 958–979, 2024.
- [45] Can Cui, Yupeng Zhou, Juntong Peng, Sung-Yeon Park, Zichong Yang, Prashanth Sankaranarayanan, Jiaru Zhang, Ruqi Zhang, and Ziran Wang. ViLaD: A large vision language diffusion framework for end-to-end autonomous driving. arXiv preprint arXiv:2508.12603, 2025.
- [46] Erfei Cui, Wenhai Wang, Zhiqi Li, Jiangwei Xie, Haoming Zou, Hanming Deng, Gen Luo, Lewei Lu, Xizhou Zhu, and Jifeng Dai. DriveMLM: Aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245, 2023.
- [47] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In Adv. Neural Inf. Process. Syst., volume 37, pages 28706–28719, 2024.
- [48] Cainan Davidson, Deva Ramanan, and Neehar Peri. RefAV: Towards planning-centric scenario mining. arXiv preprint arXiv:2505.20981, 2025.
- [49] Pim De Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning. In Adv. Neural Inf. Process. Syst., volume 32, pages 11698–11709, 2019.

- [50] Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, and Marie-Francine Moens. Talk2Car: Taking control of your self-driving car. In Conf. Empirical Methods Natural Lang. Process., pages 2088–2098, 2019.
- [51] Thierry Deruyttere, Dusan Grujicic, Matthew B. Blaschko, and Marie-Francine Moens. Talk2Car: Predicting physical trajectories for natural language commands. IEEE Access, 10:123809–123834, 2022.
- [52] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [53] Jingtao Ding, Yunke Zhang, Yu Shang, Jie Feng, Yuheng Zhang, Zefang Zong, Yuan Yuan, Hongyuan Su, Nian Li, Jinghua Piao, Yucheng Deng, Nicholas Sukiennik, Chen Gao, Fengli Xu, and Yong Li. Understanding world or predicting future? a comprehensive survey of world models. ACM Comput. Surveys, 2024.
- [54] Xinpeng Ding, Jianhua Han, Hang Xu, Xiaodan Liang, Wei Zhang, and Xiaomeng Li. Holistic autonomous driving understanding by bird’s-eye-view injected multi-modal large models. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 13668–13677, 2024.
- [55] Nemanja Djuric, Vladan Radosavljevic, Henggang Cui, Thi Nguyen, Fang-Chieh Chou, Tsung-Han Lin, NITIN SINGH, and Jeff Schneider. Uncertainty-aware short-term motion prediction of traffic actors for autonomous driving. In IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 2095–2104, 2020.
- [56] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Conf. Robot Learn., pages 1–16. PMLR, 2017.
- [57] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Int. Conf. Learn. Represent., 2021.
- [58] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 12873–12883, 2021.
- [59] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R. Qi, Yin Zhou, Zoey Yang, Aurélien Chouard, Pei Sun, Jiquan Ngiam, Vijay Vasudevan, Alexander McCauley, Jonathon Shlens, and Dragomir Anguelov. Large scale interactive motion forecasting for autonomous driving: The Waymo Open Motion dataset. In IEEE/CVF Int. Conf. Comput. Vis., pages 9710–9719, 2021.
- [60] Shiyu Fang, Yiming Cui, Haoyang Liang, Chen Lv, Peng Hang, and Jian Sun. CoReVLA: A dual-stage end-to-end autonomous driving framework for long-tail scenarios via collect-and-refine. arXiv preprint arXiv:2509.15968, 2025.
- [61] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA: Exploring the limits of masked visual representation learning at scale. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 19358–19369, 2023.
- [62] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171, 2024.
- [63] Bowen Feng, Zhiting Mei, Baiang Li, Julian Ost, Filippo Ghilotti, Roger Girgis, Anirudha Majumdar, and Felix Heide. VERDI: VLM-embedded reasoning for autonomous driving. arXiv preprint arXiv:2505.15925, 2025.
- [64] Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, and Alexandre Alahi. RAP: 3D rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333, 2025.
- [65] Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, and Yu Qiao. Drive like a human: Rethinking autonomous driving with large language models. In IEEE/CVF Winter Conf. Appl. Comput. Vis. Worksh., pages 910–919, 2024.
- [66] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. ORION: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. arXiv preprint arXiv:2503.19755, 2025.
- [67] Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, and Xinggang Wang. RAD: Training an end-to-end driving policy via large-scale 3DGS-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025.

- [68] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Adv. Neural Inf. Process. Syst., 37:91560–91596, 2024.
- [69] Xiangbo Gao, Yuheng Wu, Rujia Wang, Chenxi Liu, Yang Zhou, and Zhengzhong Tu. LangCoop: Collaborative driving with language. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 4226–4237, 2025.
- [70] Anant Garg and K Madhava Krishna. Imagine-2-Drive: Leveraging high-fidelity world models via multi-modal diffusion policies. arXiv preprint arXiv:2411.10171, 2024.
- [71] Maoning Ge, Kento Ohtani, Yingjie Niu, Yuxiao Zhang, and Kazuya Takeda. VLA-MP: A vision-language-action framework for multimodal perception and physics-constrained action generation in autonomous driving. Sensors, 25(19):6163, 2025.
- [72] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 3354–3361, 2012.
- [73] Anurag Ghosh, Shen Zheng, Robert Tamburo, Khiem Vuong, Juan Alvarez-Padilla, Hailiang Zhu, Michael Cardei, Nicholas Dunn, Christoph Mertz, and Srinivasa G. Narasimhan. ROADWork dataset: Learning to recognize, observe, analyze and drive through work zones. arXiv preprint arXiv:2406.07661, 2024.
- [74] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, et al. The LLaMA 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [75] Xingtai Gui, Jianbo Zhao, Wencheng Han, Jikai Wang, Jiahao Gong, Feiyang Tan, Cheng zhong Xu, and Jianbing Shen. TrajDiff: End-to-end autonomous driving without perception annotation. arXiv preprint arXiv:2512.00723, 2025.
- [76] Ziang Guo and Zufeng Zhang. VDRive: Leveraging reinforced VLA and diffusion policy for end-to-end autonomous driving. arXiv preprint arXiv:2510.15446, 2025.
- [77] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [78] Shadi Hamdan, Chonghao Sima, Zetong Yang, Hongyang Li, and Fatma Guney. ETA: Efficiency through thinking ahead, a dual approach to self-driving with large models. In IEEE/CVF Int. Conf. Comput. Vis., pages 26529–26538, 2025.
- [79] Jianhua Han, Meng Tian, Jiangtong Zhu, Fan He, Huixin Zhang, Sitong Guo, Dechang Zhu, Hao Tang, Pei Xu, Yuze Guo, Minzhe Niu, Haojie Zhu, Qichao Dong, Xuechao Yan, Siyuan Dong, Lu Hou, Qingqiu Huang, Xiaosong Jia, and Hang Xu. Percept-WAM: Perception-enhanced world-awareness-action model for robust end-to-end autonomous driving. arXiv preprint arXiv:2511.19221, 2025.
- [80] Wencheng Han, Dongqian Guo, Cheng-Zhong Xu, and Jianbing Shen. DME-Driver: Integrating human decision logic and 3D scene perception in autonomous driving. In AAAI Conf. Artifi. Intell., volume 39, pages 3347–3355, 2025.
- [81] Ruiyang Hao, Bowen Jing, Haibao Yu, and Zaiqing Nie. StyleDrive: Towards driving-style aware benchmarking of end-to-end autonomous driving. arXiv preprint arXiv:2506.23982, 2025.
- [82] Xiaoshuai Hao, Mengchuan Wei, Yifan Yang, Haimei Zhao, Hui Zhang, Yi Zhou, Qiang Wang, Weiming Li, Lingdong Kong, and Jing Zhang. Is your HD map constructor reliable under sensor corruptions? Adv. Neural Inf. Process. Syst., 37:22441–22482, 2024.

- [83] Xiaoshuai Hao, Yunfeng Diao, Mengchuan Wei, Yifan Yang, Peng Hao, Rong Yin, Hui Zhang, Weiming Li, Shu Zhao, and Yu Liu. MapFusion: A novel BEV feature fusion network for multi-modal map construction. Information Fusion, 119:103018, 2025.
- [84] Xiaoshuai Hao, Lingdong Kong, Rong Yin, Pengwei Wang, Jing Zhang, Yunfeng Diao, and Shu Zhao. SafeMap: Robust HD map construction from incomplete observations. In Int. Conf. Mach. Learn., pages 22091–22102. PMLR, 2025.
- [85] Xiaoshuai Hao, Guanqun Liu, Yuting Zhao, Yuheng Ji, Mengchuan Wei, Haimei Zhao, Lingdong Kong, Rong Yin, and Yu Liu. MSC-Bench: Benchmarking and analyzing multi-sensor corruption for driving perception. arXiv preprint arXiv:2501.01037, 2025.
- [86] Yuhan Hao, Zhengning Li, Lei Sun, Weilong Wang, Naixin Yi, Sheng Song, Caihong Qin, Mofan Zhou, Yifei Zhan, and Xianpeng Lang. DriveAction: A benchmark for exploring human-like driving decisions in VLA models. arXiv preprint arXiv:2506.05667, 2025.
- [87] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 770–778, 2016.
- [88] Deepti Hegde, Rajeev Yasarla, Hong Cai, Shizhong Han, Apratim Bhattacharyya, Shweta Mahajan, Litian Liu, Risheek Garrepalli, Vishal M. Patel, and Fatih Porikli. Distilling multi-modal large language models for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 27575–27585, 2025.
- [89] John Houston, Guido Zuidhof, Luca Bergamini, Yawei Ye, Long Chen, Ashesh Jain, Sammy Omari, Vladimir Iglovikov, and Peter Ondruska. One thousand and one hours: Self-driving motion prediction dataset. In Conf. Robot Learn., pages 409–418. PMLR, 2021.
- [90] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
- [91] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. ST-P3: End-to-end vision-based autonomous driving via spatial-temporal feature learning. In Eur. Conf. Comput. Vis., pages 533–549. Springer, 2022.
- [92] Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. DrivingWorld: Constructing world model for autonomous driving via video GPT. arXiv preprint arXiv:2412.19505, 2024.
- [93] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 17853–17862, 2023.
- [94] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Info. Syst., 43(2):1–55, 2025.
- [95] Yidong Huang, Jacob Sansom, Ziqiao Ma, Felix Gervits, and Joyce Chai. DriVLMe: Enhancing llm-based autonomous driving agents with embodied and social experiences. In IEEE/RSJ Int. Conf. Intell. Robots Syst., pages 3153–3160, 2024.
- [96] Zanming Huang, Jimuyang Zhang, and Eshed Ohn-Bar. Neural volumetric world models for autonomous driving. In Eur. Conf. Comput. Vis., pages 195–213. Springer, 2024.
- [97] Zhijian Huang, Tao Tang, Shaoxiang Chen, Sihao Lin, Zequn Jie, Lin Ma, Guangrun Wang, , and Xiaodan Liang. Making large language models better planners with reasoning-decision alignment. In Eur. Conf. Comput. Vis., pages 73–90. Springer, 2024.
- [98] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, Yin Zhou, James Guo, Dragomir Anguelov, and Mingxing Tan. EMMA: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262, 2024.
- [99] Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, et al. OpenCLIP. Zenodo, 2021.
- [100] Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao, Yuqing Wen, Chi Zhang, Xiangyu Zhang, and Tiancai Wang. ADriver-I: A general world model for autonomous driving. arXiv preprint arXiv:2311.13549, 2023.

- [101] Xiaosong Jia, Yulu Gao, Li Chen, Junchi Yan, Patrick Langechuan Liu, and Hongyang Li. DriveAdapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving. In IEEE/CVF Int. Conf. Comput. Vis., pages 7953–7963, 2023.
- [102] Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.
- [103] Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2Drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In Adv. Neural Inf. Process. Syst., volume 37, pages 819–844, 2024.
- [104] Xiaosong Jia, Junqi You, Zhiyuan Zhang, and Junchi Yan. DriveTransformer: Unified transformer for scalable end-to-end autonomous driving. In Int. Conf. Learn. Represent., 2025.
- [105] Xiaosong Jia, Chenhe Zhang, Yule Jiang, Songbur Wong, Zhiyuan Zhang, Chen Chen, Shaofeng Zhang, Xuanhe Zhou, Xue Yang, Junchi Yan, and Yu-Gang Jiang. Spatial retrieval augmented autonomous driving. arXiv preprint arXiv:2512.06865, 2025.
- [106] Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Yunda Dong, Zongzheng Zhang, Xianda Guo, Hao Sun, and Hao Zhao. DiffVLA: Vision-language guided diffusion planning for autonomous driving. arXiv preprint arXiv:2505.19381, 2025.
- [107] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. VAD: Vectorized scene representation for efficient autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 8340–8350, 2023.
- [108] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024.
- [109] Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. AlphaDrive: Unleashing the power of VLMs in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608, 2025.
- [110] Sicong Jiang, Zilin Huang, Kangan Qian, Ziang Luo, Tianze Zhu, Yang Zhong, Yihong Tang, Menglin Kong, Yunlong Wang, Siwen Jiao, Hao Ye, Zihao Sheng, Xin Zhao, Tuopu Wen, Zheng Fu, Sikai Chen, Kun Jiang, Diange Yang, Seongjin Choi, and Lijun Sun. A survey on vision-language-action models for autonomous driving. arXiv preprint arXiv:2506.24044, 2025.
- [111] Bu Jin, Xiaotao Hu, Songen Gu, Yupeng Zheng, Xiaoyang Guo, et al. OccVAR: Scalable 4D occupancy prediction via next-scale prediction. https://openreview.net/forum?id=X2HnTFsFm8, 2025.
- [112] Taotao Jing, Haifeng Xia, Renran Tian, Haoran Ding, Xiao Luo, Joshua Domeyer, Rini Sherony, and Zhengming Ding. Inaction: Interpretable action decision making for autonomous driving. In European Conference on Computer Vision, pages 370–387. Springer, 2022.
- [113] Daniel Kahneman. Thinking, fast and slow. Farrar, Straus and Giroux, 2011.
- [114] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, and Holger Caesar. Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving. In IEEE Int. Conf. Robot. Autom., pages 629–636, 2024.
- [115] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [116] Siddhesh Khandelwal, William Qi, Jagjeet Singh, Andrew Hartnett, and Deva Ramanan. What-if motion prediction for autonomous driving. arXiv preprint arXiv:2008.10587, 2020.
- [117] Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. Textual explanations for self-driving vehicles. In Eur. Conf. Comput. Vis., pages 563–578. Springer, 2018.
- [118] B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A. Al Sallab, Senthil Yogamani, and Patrick Pérez. Deep reinforcement learning for autonomous driving: A survey. IEEE Int. Conf. Intell. Transport. Syst., 23(6):4909–4926, 2021.
- [119] W Bradley Knox, Alessandro Allievi, Holger Banzhaf, Felix Schmitt, and Peter Stone. Reward (mis)design for autonomous driving. Artifi. Intell., 316:103829, 2023.

- [120] Lingdong Kong, Youquan Liu, Runnan Chen, Yuexin Ma, Xinge Zhu, Yikang Li, Yuenan Hou, Yu Qiao, and Ziwei Liu. Rethinking range view representation for LiDAR segmentation. In IEEE/CVF Int. Conf. Comput. Vis., pages 228–240, 2023.
- [121] Lingdong Kong, Youquan Liu, Xin Li, Runnan Chen, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Robo3D: Towards robust and reliable 3D perception against corruptions. In IEEE/CVF Int. Conf. Comput. Vis., pages 19994–20006, 2023.
- [122] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Yaru Niu, Wei Tsang Ooi, Benoit R. Cottereau, Lai Xing Ng, Yuexin Ma, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, Weichao Qiu, Wei Zhang, Xu Cao, Hao Lu, Ying-Cong Chen, Caixin Kang, Xinning Zhou, Chengyang Ying, Wentao Shang, Xingxing Wei, Yinpeng Dong, Bo Yang, Shengyin Jiang, Zeliang Ma, Dengyi Ji, Haiwen Li, Xingliang Huang, Yu Tian, Genghua Kou, Fan Jia, Yingfei Liu, Tiancai Wang, Ying Li, Xiaoshuai Hao, Yifan Yang, Hui Zhang, Mengchuan Wei, Yi Zhou, Haimei Zhao, Jing Zhang, Jinke Li, Xiao He, Xiaoqiang Cheng, Bingyang Zhang, Lirong Zhao, Dianlei Ding, Fangsheng Liu, Yixiang Yan, Hongming Wang, Nanfei Ye, Lun Luo, Yubo Tian, Yiwei Zuo, Zhe Cao, Yi Ren, Yunfan Li, Wenjie Liu, Xun Wu, Yifan Mao, Ming Li, Jian Liu, Jiayang Liu, Zihan Qin, Cunxi Chu, Jialei Xu, Wenbo Zhao, Junjun Jiang, Xianming Liu, Ziyan Wang, Chiwei Li, Shilong Li, Chendong Yuan, Songyue Yang, Wentao Liu, Peng Chen, Bin Zhou, Yubo Wang, Chi Zhang, Jianhang Sun, Hai Chen, Xiao Yang, Lizhong Wang, Dongyi Fu, Yongchun Lin, Huitong Yang, Haoang Li, Yadan Luo, Xianjing Cheng, and Yong Xu. The RoboDrive challenge: Drive anytime anywhere in any condition. arXiv preprint arXiv:2405.08816, 2024.
- [123] Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. LargeAD: Large-scale cross-sensor data pretraining for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 2025.
- [124] Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi-modal data-efficient 3D scene understanding for autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3748–3765, 2025.
- [125] Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C. H. Hoi, and Ziwei Liu. 3D and 4D world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025.
- [126] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. PointPillars: Fast encoders for object detection from point clouds. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 12697–12705, 2019.
- [127] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11523–11532, 2022.
- [128] Youngwan Lee, Joong-won Hwang, Sangrok Lee, Yuseok Bae, and Jongyoul Park. An energy and GPUcomputation efficient backbone network for real-time object detection. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. Worksh., pages 1–9, 2019.
- [129] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LlaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [130] Boyi Li, Yue Wang, Jiageng Mao, Boris Ivanovic, Sushant Veer, Karen Leung, and Marco Pavone. Driving everywhere with large language model policy adaptation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14948–14957, 2024.
- [131] Hongyang Li, Yang Li, Huijie Wang, Jia Zeng, Huilin Xu, Pinlong Cai, Li Chen, Junchi Yan, Feng Xu, Lu Xiong, Jingdong Wang, Futang Zhu, Chunjing Xu, Tiancai Wang, Fei Xia, Beipeng Mu, Zhihui Peng, Dahua Lin, and Yu Qiao. Open-sourced data ecosystem in autonomous driving: The present and future. arXiv preprint arXiv:2312.03408, 2023.
- [132] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Int. Conf. Mach. Learn., pages 19730–19742. PMLR, 2023.
- [133] Peidong Li and Dixiao Cui. Navigation-guided sparse scene representation for end-to-end autonomous driving. arXiv preprint arXiv:2409.18341, 2024.

- [134] Peizheng Li, Zhenghao Zhang, David Holtz, Hang Yu, Yutong Yang, Yuzhi Lai, Rui Song, Andreas Geiger, and Andreas Zell. SpaceDrive: Infusing spatial awareness into VLM-based autonomous driving. arXiv preprint arXiv:2512.10719, 2025.
- [135] Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, and Xianpeng Lang. Discrete diffusion for reflective vision-language-action models in autonomous driving. arXiv preprint arXiv:2509.20109, 2025.
- [136] Qifeng Li, Xiaosong Jia, Shaobo Wang, and Junchi Yan. Think2Drive: Efficient reinforcement learning by thinking with latent world model for autonomous driving (in CARLA-V2). In Eur. Conf. Comput. Vis., pages 142–158. Springer, 2024.
- [137] Rong Li, Yuhao Dong, Tianshuai Hu, Ao Liang, Youquan Liu, Dongyue Lu, Liang Pan, Lingdong Kong, Junwei Liang, and Ziwei Liu. 3EED: Ground everything everywhere in 3D. In Adv. Neural Inf. Process. Syst., volume 38, 2025.
- [138] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. TokenPacker: Efficient visual projector for multimodal LLM. Int. J. Comput. Vis., 133:6794–6812, 2025.
- [139] Xin Li, Yeqi Bai, Pinlong Cai, Licheng Wen, Daocheng Fu, Bo Zhang, Xuemeng Yang, Xinyu Cai, Tao Ma, Jianfei Guo, Xing Gao, Min Dou, Yikang Li, Botian Shi, Yong Liu, Liang He, and Yu Qiao. Towards knowledge-driven autonomous driving. arXiv preprint arXiv:2312.04316, 2023.
- [140] Ye Li, Lingdong Kong, Hanjiang Hu, Xiaohao Xu, and Xiaonan Huang. Is your LiDAR placement optimized for 3D scene understanding? In Adv. Neural Inf. Process. Syst., volume 37, pages 34980–35017, 2024.
- [141] Yiheng Li, Cunxin Fan, Chongjian Ge, Zhihao Zhao, Chenran Li, Chenfeng Xu, Huaxiu Yao, Masayoshi Tomizuka, Bolei Zhou, Chen Tang, Mingyu Ding, and Wei Zhan. WOMD-Reasoning: A large-scale dataset for interaction reasoning in driving. arXiv preprint arXiv:2407.04281, 2024.
- [142] Yingyan Li, Lue Fan, Jiawei He, Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang, and Tieniu Tan. Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481, 2024.
- [143] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, and Zhaoxiang Zhang. DriveVLA-W0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025.
- [144] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941, 2025.
- [145] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, Wenyu Liu, and Xinggang Wang. ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025.
- [146] Yue Li, Meng Tian, Dechang Zhu, Jiangtong Zhu, Zhenyu Lin, Zhiwei Xiong, and Xinhai Zhao. Drive-R1: Bridging reasoning and planning in VLMs for autonomous driving with reinforcement learning. arXiv preprint arXiv:2506.18234, 2025.
- [147] Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, Yu-Gang Jiang, and Jose M. Alvarez. Hydra-MDP: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978, 2024.
- [148] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. BEVFormer: learning bird’s-eye-view representation from LiDAR-camera via spatiotemporal transformers. IEEE Trans. Pattern Anal. Mach. Intell., 47(3):2020–2036, 2024.
- [149] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14864–14873, 2024.
- [150] Zhuoren Li, Guizhe Jin, Ran Yu, Zhiwen Chen, Nan Li, Wei Han, Lu Xiong, Bo Leng, Jia Hu, Ilya Kolmanovsky, and Dimitar Filev. A survey of reinforcement learning-based motion planning for autonomous driving: Lessons learned from a driving task perspective. arXiv preprint arXiv:2503.23650, 2025.
- [151] Li Auto Inc. MindVLA, 2025. URL https://ir.lixiang.com/news-releases/news-release-details/ li-auto-inc-march-2025-delivery-update.

- [152] Ao Liang, Lingdong Kong, Dongyue Lu, Youquan Liu, Jian Fang, Huaici Zhao, and Wei Tsang Ooi. Perspectiveinvariant 3D object detection. In IEEE/CVF Int. Conf. Comput. Vis., pages 27725–27738, 2025.
- [153] Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, and Wei Tsang Ooi. LiDARCrafter: Dynamic 4D world modeling from LiDAR sequences. In AAAI Conf. Artifi. Intell., volume 40, 2026.
- [154] Tingting Liang, Hongwei Xie, Kaicheng Yu, Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Tao Tang, Bing Wang, and Zhi Tang. BEVFusion: A simple and robust LiDAR-camera fusion framework. Adv. Neural Inf. Process. Syst., 35:10421–10434, 2022.
- [155] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, and Xinggang Wang. DiffusionDrive: Truncated diffusion model for end-to-end autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 12037–12047, 2025.
- [156] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-LLaVA: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [157] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yi Xin, Xinyue Li, Qi Qin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-MGPT: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024.
- [158] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024.
- [159] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Adv. Neural Inf. Process. Syst., volume 36, pages 34892–34916, 2023.
- [160] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 26296–26306, 2024.
- [161] Lin Liu, Caiyan Jia, Guanyi Yu, Ziying Song, JunQiao Li, Feiyang Jia, Peiliang Wu, Xiaoshuai Hao, and Yandan Luo. GuideFlow: Constraint-guided flow matching for planning in end-to-end autonomous driving. arXiv preprint arXiv:2511.18729, 2025.
- [162] Mingyu Liu, Ekim Yurtsever, Jonathan Fossaert, Xingcheng Zhou, Walter Zimmer, Yuning Cui, Bare Luka Zagar, and Alois C. Knoll. A survey on autonomous driving datasets: Statistics, annotation quality, and a future outlook. IEEE Trans. Intell. Veh., 9(11):7138–7164, 2024.
- [163] Pei Liu, Haipeng Liu, Haichao Liu, Xin Liu, Jinxin Ni, and Jun Ma. VLM-E2E: Enhancing end-to-end autonomous driving with multimodal driver attention fusion. arXiv preprint arXiv:2502.18042, 2025.
- [164] Pei Liu, Hongliang Lu, Haichao Liu, Haipeng Liu, Xin Liu, Ruoyu Yao, Shengbo Eben Li, and Jun Ma. OmniScene: Attention-augmented multimodal 4D scene understanding for autonomous driving. arXiv preprint arXiv:2509.19973, 2025.
- [165] Pei Liu, Qingtian Ning, Xinyan Lu, Haipeng Liu, Weiliang Ma, Dangen She, Peng Jia, Xianpeng Lang, and Jun Ma. OmniReason: A temporal-guided vision-language-action framework for autonomous driving. arXiv preprint arXiv:2509.00789, 2025.
- [166] Ruixun Liu, Lingyu Kong, Derun Li, and Hang Zhao. OccVLA: Vision-language-action model with implicit 3D occupancy supervision. arXiv preprint arXiv:2509.05578, 2025.
- [167] Wenru Liu, Pei Liu, and Jun Ma. DSDrive: Distilling large language model for lightweight end-to-end autonomous driving with unified reasoning and planning. arXiv preprint arXiv:2505.05360, 2025.
- [168] Xueyi Liu, Zuodong Zhong, Yuxin Guo, Yun-Fu Liu, Zhiguo Su, Qichao Zhang, Junli Wang, Yinfeng Gao, Yupeng Zheng, Qiao Lin, Huiyong Chen, and Dongbin Zhao. ReasonPlan: Unified scene prediction and decision reasoning for closed-loop autonomous driving. arXiv preprint arXiv:2505.20024, 2025.
- [169] Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. Segment any point cloud sequences by distilling vision foundation models. In Adv. Neural Inf. Process. Syst., volume 36, pages 37193–37229, 2023.
- [170] Youquan Liu, Lingdong Kong, Weidong Yang, Xin Li, Ao Liang, Runnan Chen, Ben Fei, and Tongliang Liu. La La LiDAR: Large-scale layout generation from LiDAR data. In AAAI Conf. Artifi. Intell., volume 40, 2026.

- [171] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In IEEE/CVF Int. Conf. Comput. Vis., pages 10012–10022, 2021.
- [172] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11976–11986, 2022.
- [173] Hao Lu, Ziyang Liu, Guangfeng Jiang, Yuanfei Luo, Sheng Chen, Yangang Zhang, and Ying-Cong Chen. UniUGP: Unifying understanding, generation, and planing for end-to-end autonomous driving. arXiv preprint arXiv:2512.09864, 2025.
- [174] Jiachen Lu, Ze Huang, Zeyu Yang, Jiahui Zhang, and Li Zhang. WoVoGen: World volume-aware diffusion for controllable multi-camera driving scene generation. In Eur. Conf. Comput. Vis., pages 329–345. Springer, 2024.
- [175] Yuhang Lu, Jiadong Tu, Yuexin Ma, and Xinge Zhu. ReAL-AD: Towards human-like reasoning in end-to-end autonomous driving. In IEEE/CVF Int. Conf. Comput. Vis., pages 27783–27793, 2025.
- [176] Yuechen Luo, Fang Li, Shaoqing Xu, Zhiyi Lai, Lei Yang, Qimao Chen, Ziang Luo, Zixun Xie, Shengyin Jiang, Jiaxin Liu, Long Chen, Bing Wang, and Zhi xin Yang. AdaThinkDrive: Adaptive thinking via reinforcement learning for autonomous driving. arXiv preprint arXiv:2509.13769, 2025.
- [177] Junyi Ma, Xieyuanli Chen, Jiawei Huang, Jingyi Xu, Zhen Luo, Jintao Xu, Weihao Gu, Rui Ai, and Hesheng Wang. Cam4DOcc: Benchmark for camera-only 4D occupancy forecasting in autonomous driving applications. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 21486–21495, 2024.
- [178] Yingzi Ma, Yulong Cao, Wenhao Ding, Shuibai Zhang, Yan Wang, Boris Ivanovic, Ming Jiang, Marco Pavone, and Chaowei Xiao. dVLM-AD: Enhance diffusion vision-language-model for driving via controllable reasoning. arXiv preprint arXiv:2512.04459, 2025.
- [179] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied AI. arXiv preprint arXiv:2405.14093, 2024.
- [180] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. Survey on vision-language-action models. arXiv preprint arXiv:2502.06851, 2025.
- [181] Yukai Ma, Tiantian Wei, Naiting Zhong, Jianbiao Mei, Tao Hu, Licheng Wen, Xuemeng Yang, Botian Shi, and Yong Liu. LeapVAD: A leap in autonomous driving via cognitive perception and dual-process thinking. arXiv preprint arXiv:2501.08168, 2025.
- [182] Yunsheng Ma, Burhaneddin Yaman, Xin Ye, Mahmut Yurt, Jingru Luo, Abhirup Mallik, Ziran Wang, and Liu Ren. ALN-P3: Unified language alignment for perception, prediction, and planning in autonomous driving. arXiv preprint arXiv:2505.15158, 2025.
- [183] Ziqiao Ma, Benjamin VanDerPloeg, Cristian-Paul Bara, Yidong Huang, Eui-In Kim, Felix Gervits, Matthew Marge, and Joyce Chai. DOROTHIE: Spoken dialogue for handling unexpected situations in interactive autonomous driving agents. arXiv preprint arXiv:2210.12511, 2022.
- [184] Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. GPT-Driver: Learning to drive with GPT. arXiv preprint arXiv:2310.01415, 2023.
- [185] Jiageng Mao, Junjie Ye, Yuxi Qian, Marco Pavone, and Yue Wang. A language agent for autonomous driving. In Conf. Lang. Model., 2025.
- [186] Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, Elahe Arani, and Oleg Sinavski. LingoQA: Visual question answering for autonomous driving. In Eur. Conf. Comput. Vis., pages 252–269. Springer, 2024.
- [187] Jianbiao Mei, Yukai Ma, Xuemeng Yang, Licheng Wen, Xinyu Cai, Xin Li, Daocheng Fu, Bo Zhang, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yong Liu, and Yu Qiao. Continuously learning, adapting, and improving: A dual-process approach to autonomous driving. In Adv. Neural Inf. Process. Syst., volume 37, pages 123261–123290, 2024.
- [188] Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, Liping Jing, Yiming Nie, and Bin Dai. DriveWorld: 4D pre-trained scene understanding via world models for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 15522–15533, 2024.
- [189] Mona Mirzaie and Bodo Rosenhahn. Interpretable decision-making for end-to-end autonomous driving. arXiv preprint arXiv:2508.18898, 2025.

- [190] Urs Muller, Jan Ben, Eric Cosatto, Beat Flepp, and Yann Cun. Off-road obstacle avoidance through end-to-end learning. In Adv. Neural Inf. Process. Syst., volume 18, pages 739–746, 2005.
- [191] Ferdinand Mütsch, Helen Gremmelmaier, Nicolas Becker, Daniel Bogdoll, Marc René Zofka, and J. Marius Zöllner. From model-based to data-driven simulation: Challenges and trends in autonomous driving. arXiv preprint arXiv:2305.13960, 2023.
- [192] Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, and Li Zhang. Reason2Drive: Towards interpretable and chain-based reasoning for autonomous driving. In Eur. Conf. Comput. Vis., pages 292–308. Springer, 2024.
- [193] NVIDIA. Physical AI autonomous vehicles dataset. https://huggingface.co/datasets/nvidia/ PhysicalAI-Autonomous-Vehicles, October 2025.
- [194] Eshed Ohn-Bar, Aditya Prakash, Aseem Behl, Kashyap Chitta, and Andreas Geiger. Learning situational driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11296–11305, 2020.
- [195] OpenAI. Hello GPT4-o. https://openai.com/index/hello-gpt-4o, 2024.
- [196] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [197] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G. Allievi, Senem Velipasalar, and Liu Ren. VLP: Vision language planning for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14760–14769, 2024.
- [198] Yunpeng Pan, Ching-An Cheng, Kamil Saigol, Keuntaek Lee, Xinyan Yan, Evangelos Theodorou, and Byron Boots. Agile autonomous driving using end-to-end deep imitation learning. In Robot. Sci. Syst., 2018.
- [199] Jinhyung Park, Chenfeng Xu, Shijia Yang, Kurt Keutzer, Kris Kitani, Masayoshi Tomizuka, and Wei Zhan. Time Will Tell: New outlooks and a baseline for temporal multi-view 3D object detection. In Int. Conf. Learn. Represent., 2023.
- [200] Jongjin Park, Younggyo Seo, Chang Liu, Li Zhao, Tao Qin, Jinwoo Shin, and Tie-Yan Liu. Object-aware regularization for addressing causal confusion in imitation learning. In Adv. Neural Inf. Process. Syst., volume 34, pages 3029–3042, 2021.
- [201] SungYeon Park, MinJae Lee, JiHyuk Kang, Hahyeon Choi, Yoonah Park, Juhwan Cho, Adam Lee, and DongKyu Kim. VLAAD: Vision and language assistant for autonomous driving. In IEEE/CVF Winter Conf. Appl. Comput. Vis. Worksh., pages 980–987, 2024.
- [202] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In Int. Conf. Learn. Represent., 2023.
- [203] Dean A Pomerleau. ALVINN: An autonomous land vehicle in a neural network. In Adv. Neural Inf. Process. Syst., volume 1, pages 305–313, 1988.
- [204] Alexander Popov, Alperen Degirmenci, David Wehr, Shashank Hegde, Ryan Oldja, Alexey Kamenev, Bertrand Douillard, David Nistér, Urs Muller, Ruchi Bhargava, Stan Birchfield, and Nikolai Smolyanskiy. Mitigating covariate shift in imitation learning for autonomous vehicles using latent space generative world models. arXiv preprint arXiv:2409.16663, 2024.
- [205] Aditya Prakash, Aseem Behl, Eshed Ohn-Bar, Kashyap Chitta, and Andreas Geiger. Exploring data aggregation in policy learning for vision-based urban autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11763–11773, 2020.
- [206] Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multi-modal fusion transformer for end-to-end autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 7077–7087, 2021.
- [207] Kangan Qian, Zhikun Ma, Yangfan He, Ziang Luo, Tianyu Shi, Tianze Zhu, Jiayin Li, Jianhui Wang, Ziyu Chen, Xiao He, Yining Shi, Zheng Fu, Xinyu Jiao, Kun Jiang, Diange Yang, and Takafumi Matsumaru. FasionAD: Fast and slow fusion thinking systems for human-like autonomous driving with adaptive feedback. arXiv preprint arXiv:2411.18013, 2024.

- [208] Kangan Qian, Ziang Luo, Sicong Jiang, Zilin Huang, Jinyu Miao, Zhikun Ma, Tianze Zhu, Jiayin Li, Yangfan He, Zheng Fu, Yining Shi, Boyue Wang, Hezhe Lin, Ziyu Chen, Jiangbo Yu, Xinyu Jiao, Mengmeng Yang, Kun Jiang, and Diange Yang. FasionAD++: Integrating high-level instruction and information bottleneck in fat-slow fusion systems for enhanced safety in autonomous driving with adaptive feedback. arXiv preprint arXiv:2503.08162, 2025.
- [209] Zhijie Qiao, Haowei Li, Zhong Cao, and Henry X. Liu. LightEMMA: Lightweight end-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2505.00284, 2025.
- [210] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Int. Conf. Mach. Learn., pages 8748–8763. PMLR, 2021.
- [211] Vipula Rawte, Amit Sheth, and Amitava Das. A survey of hallucination in large foundation models. arXiv preprint arXiv:2309.05922, 2023.
- [212] Xuanchi Ren, Yifan Lu, Tianshi Cao, Ruiyuan Gao, Shengyu Huang, Amirmojtaba Sabour, Tianchang Shen, Tobias Pfaff, Jay Zhangjie Wu, Runjian Chen, Seung Wook Kim, Jun Gao, Laura Leal-Taixe, Mike Chen, Sanja Fidler, and Huan Ling. Cosmos-Drive-Dreams: Scalable synthetic driving data generation with world foundation models. arXiv preprint arXiv:2506.09042, 2025.
- [213] Katrin Renz, Long Chen, Elahe Arani, and Oleg Sinavski. SimLingo: Vision-only closed-loop autonomous driving with language-action alignment. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 11993–12003, 2025.
- [214] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 10684–10695, 2022.
- [215] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Int. Conf. Artifi. Intell. Stat., pages 627–635, 2011.
- [216] Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, and Liam Paull. Poutine: Vision-languagetrajectory pre-training and reinforcement learning post-training enable robust end-to-end autonomous driving. arXiv preprint arXiv:2506.11234, 2025.
- [217] Ranjan Sapkota, Yang Cao, Konstantinos I. Roumeliotis, and Manoj Karkee. Vision-language-action models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769, 2025.
- [218] Oliver Scheel, Luca Bergamini, Maciej Wołczyk, Błażej Osiński, and Peter Ondruska. Urban driver: Learning to drive from real-world demonstrations using policy gradients. In Conf. Robot Learn., pages 718–728. PMLR, 2022.
- [219] Mariah L Schrum, Emily Sumner, Matthew C Gombolay, and Andrew Best. Maveric: A data-driven approach to personalized autonomous driving. IEEE Trans. Robot., 40:1952–1965, 2024.
- [220] Hao Shan, Ruikai Li, Han Jiang, Yizhe Fan, Ziyang Yan, Bohan Li, Xiaoshuai Hao, Hao Zhao, Zhiyong Cui, Yilong Ren, and Haiyang Yu. Stability under scrutiny: Benchmarking representation paradigms for online HD mapping. arXiv preprint arXiv:2510.10660, 2025.
- [221] Hao Shao, Letian Wang, Ruobing Chen, Hongsheng Li, and Yu Liu. Safety-enhanced autonomous driving using interpretable sensor fusion transformer. In Conf. Robot Learn., pages 726–737. PMLR, 2023.
- [222] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L. Waslander, Yu Liu, and Hongsheng Li. LMDrive: Closed-loop end-to-end driving with large language models. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 15120–15130, 2024.
- [223] Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, and Liqiang Nie. Large VLM-based vision-language-action models for robotic manipulation: A survey. arXiv preprint arXiv:2508.13073, 2025.
- [224] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [225] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, , and Hongyang Li. DriveLM: Driving with graph visual question answering. In Eur. Conf. Comput. Vis., pages 256–274. Springer, 2024.
- [226] Nan Song, Bozhou Zhang, Xiatian Zhu, Jiankang Deng, and Li Zhang. LMAD: Integrated end-to-end visionlanguage model for explainable autonomous driving. arXiv preprint arXiv:2508.12404, 2025.

- [227] Ruiqi Song, Xianda Guo, Yanlun Peng, Qinggong Wei, Hangbin Wu, and Long Chen. InsightDrive: Insight scene representation for end-to-end autonomous driving. arXiv preprint arXiv:2503.13047, 2025.
- [228] Zhihang Song, Zimin He, Xingyu Li, Qiming Ma, Ruibo Ming, Zhiqi Mao, Huaxin Pei, Lihui Peng, Jianming Hu, Danya Yao, and Yi Zhang. Synthetic datasets for autonomous driving: A survey. IEEE Trans. Intell. Veh., 9(1): 1847–1864, 2023.
- [229] Ziying Song, Caiyan Jia, Lin Liu, Hongyu Pan, Yongchang Zhang, Junming Wang, Xingyu Zhang, Shaoqing Xu, Lei Yang, and Yadan Luo. Don’t shake the wheel: Momentum-aware planning in end-to-end autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 22432–22441, 2025.
- [230] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. PaliGemma 2: A family of versatile VLMs for transfer. arXiv preprint arXiv:2412.03555, 2024.
- [231] Haisheng Su, Wei Wu, and Junchi Yan. DiFSD: Ego-centric fully sparse paradigm with uncertainty denoising and iterative refinement for efficient end-to-end self-driving. arXiv preprint arXiv:2409.09777, 2024.
- [232] Bin Sun, Yaoguang Cao, Yan Wang, Rui Wang, Jiachen Shang, Xiejie Feng, Jiayi Lu, Jia Shi, Shichun Yang, Xiaoyu Yan, and Ziying Song. MindDrive: An all-in-one framework bridging world models and vision-language model for end-to-end autonomous driving. arXiv preprint arXiv:2512.04441, 2025.
- [233] Jintao Sun, Hu Zhang, Gangyi Ding, and Zhedong Zheng. Echo planning for autonomous driving: From current observations to future trajectories and back. arXiv preprint arXiv:2505.18945, 2025.
- [234] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 2446–2454, 2020.
- [235] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [236] Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. SparseDrive: End-to-end autonomous driving via sparse scene representation. In IEEE Int. Conf. Robot. Autom., pages 8795–8801, 2025.
- [237] Richard S Sutton, Andrew G Barto, et al. Reinforcement Learning: An Introduction, volume 1. MIT Press, Cambridge, 1998.
- [238] Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Int. Conf. Mach. Learn., pages 6105–6114. PMLR, 2019.
- [239] Shuhan Tan, Kashyap Chitta, Yuxiao Chen, Ran Tian, Yurong You, Yan Wang, Wenjie Luo, Yulong Cao, Philipp Krahenbuhl, Marco Pavone, and Boris Ivanovic. Latent chain-of-thought world modeling for end-to-end driving. arXiv preprint arXiv:2512.10226, 2025.
- [240] Yihong Tang, Haicheng Liao, Tong Nie, Junlin He, Ao Qu, Kehua Chen, Wei Ma, Zhenning Li, Lijun Sun, and Chengzhong Xu. E3AD: An emotion-aware vision-language-action model for human-centric end-to-end autonomous driving. arXiv preprint arXiv:2512.04733, 2025.
- [241] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [242] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, Heng-Tze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian Güra, Hao Zhou, Xinying Song, Aurelien Boffy, Harish Ganapathy, Steven Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Laurent El Shafey, Yujing Zhang,

- Olcan Sercinoglu, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [243] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bohez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D’Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, TsangWei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddineni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Carolina Parada, Emilio Parisotto, Peter Pastor, Acorn Pooley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Stefano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, et al. Gemini robotics: Bringing AI into the physical world. arXiv preprint arXiv:2503.20020, 2025.
- [244] Qwen Team. Introducing Qwen1.5, February 2024. URL https://qwenlm.github.io/blog/qwen1.5.
- [245] Wayve Team. LINGO-2: Driving with natural language, 2024. URL https://wayve.ai/thinking/ lingo-2-driving-with-language.
- [246] Waywe Research Team et al. LINGO-2: Driving with natural language, 2024.
- [247] Ran Tian, Boyi Li, Xinshuo Weng, Yuxiao Chen, Edward Schmerling, Yue Wang, Boris Ivanovic, and Marco Pavone. Tokenize the world into object-level knowledge to address long-tail events in autonomous driving. In Conf. Robot Learn., pages 3656–3673. PMLR, 2025.
- [248] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3D: A large-scale 3D occupancy prediction benchmark for autonomous driving. In Adv. Neural Inf. Process. Syst., volume 36, pages 64318–64330, 2023.
- [249] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. DriveVLM: The convergence of autonomous driving and large vision-language models. In Conf. Robot Learn., pages 4698–4726. PMLR, 2025.
- [250] Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, and Hongyang Li. Scene as occupancy. In IEEE/CVF Int. Conf. Comput. Vis., pages 8406–8415, 2023.
- [251] Marin Toromanoff, Emilie Wirbel, and Fabien Moutarde. End-to-end model-free reinforcement learning for urban driving using implicit affordances. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 7153–7162, 2020.
- [252] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. LLaMA 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [253] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

- [254] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In Adv. Neural Inf. Process. Syst., volume 30, pages 6309–6318, 2017.
- [255] Eli Verwimp, Kuo Yang, Sarah Parisot, Lanqing Hong, Steven McDonagh, Eduardo Pérez-Pellitero, Matthias De Lange, and Tinne Tuytelaars. CLAD: A realistic continual learning benchmark for autonomous driving. Neural Net., 161:659–669, 2023.
- [256] Daming Wang, Yuhao Song, Zijian He, Kangliang Chen, Xing Pan, Lu Deng, and Weihao Gu. HMVLM: Multistage reasoning-enhanced vision-language model for long-tailed driving scenarios. arXiv preprint arXiv:2506.05883, 2025.
- [257] Dingrui Wang, Marc Kaufeld, and Johannes Betz. DualAD: Dual-layer planning for reasoning in autonomous driving. In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2025.
- [258] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [259] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M. Alvarez. OmniDrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 22442–22452, 2025.
- [260] Song Wang, Jiawei Yu, Wentong Li, Wenyu Liu, Xiaolu Liu, Junbo Chen, and Jianke Zhu. Not all voxels are equal: Hardness-aware semantic scene completion with self-distillation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14792–14801, 2024.
- [261] Song Wang, Gongfan Fang, Lingdong Kong, Xiangtai Li, Jianyun Xu, Sheng Yang, Qiang Li, Jianke Zhu, and Xinchao Wang. PixelThink: Towards efficient chain-of-pixel reasoning. arXiv preprint arXiv:2505.23727, 2025.
- [262] Tianqi Wang, Enze Xie, Ruihang Chu, Zhenguo Li, and Ping Luo. DriveCoT: Integrating chain-of-thought reasoning with end-to-end driving. arXiv preprint arXiv:2403.16996, 2024.
- [263] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, and Jifeng Dai. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.
- [264] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. DriveDreamer: Towards real-world-drive world models for autonomous driving. In Eur. Conf. Comput. Vis., pages 55–72. Springer, 2024.
- [265] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, Liang Feng, Greg Heinrich, Jack Huang, Peter Karkus, Boyi Li, Pinyi Li, Tsung-Yi Lin, Dongran Liu, Ming-Yu Liu, Langechuan Liu, Zhijian Liu, Jason Lu, Yunxiang Mao, Pavlo Molchanov, Lindsey Pavao, Zhenghao Peng, Mike Ranzinger, Ed Schmerling, Shida Shen, Yunfei Shi, Sarah Tariq, Ran Tian, Tilman Wekel, Xinshuo Weng, Tianjun Xiao, Eric Yang, Xiaodong Yang, Yurong You, Xiaohui Zeng, Wenyuan Zhang, Boris Ivanovic, and Marco Pavone. Alpamayo-R1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.
- [266] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14749–14759, 2024.
- [267] Julong Wei, Shanshuai Yuan, Pengfei Li, et al. OccLLaMA: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272, 2024.
- [268] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. PARA-Drive: Parallelized architecture for real-time autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 15449–15458, 2024.
- [269] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.
- [270] Katharina Winter, Mark Azer, and Fabian B. Flohr. BEVDriver: Leveraging BEV maps in LLMs for robust closed-loop driving. arXiv preprint arXiv:2503.03074, 2025.

- [271] Hanfeng Wu, Marlon Steiner, Michael Schmidt, Alvaro Marcos-Ramiro, and Christoph Stiller. NaviHydra: Controllable navigation-guided end-to-end autonomous driving with hydra-distillation. arXiv preprint arXiv:2512.10660, 2025.
- [272] Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. In Adv. Neural Inf. Process. Syst., volume 35, pages 6119–6132, 2022.
- [273] Jihong Xie, Xiang Zhou, and Lu Cheng. Edge computing for real-time decision making in autonomous driving: Review of challenges, solutions, and future trends. Int. J. Adv. Comput. Sci. & Appl., 15(7), 2024.
- [274] Shaoyuan Xie, Lingdong Kong, Yuhao Dong, Chonghao Sima, Wenwei Zhang, Qi Alfred Chen, Ziwei Liu, and Liang Pan. Are VLMs ready for autonomous driving? an empirical study from the reliability, data, and metric perspectives. In IEEE/CVF Int. Conf. Comput. Vis., pages 6585–6597, 2025.
- [275] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3878–3894, 2025.
- [276] Yichen Xie, Runsheng Xu, Tong He, Jyh-Jing Hwang, Katie Luo, Jingwei Ji, Hubert Lin, Letian Chen, Yiren Lu, Zhaoqi Leng, Dragomir Anguelov, and Mingxing Tan. S4-Driver: Scalable self-supervised driving multimodal large language model with spatio-temporal visual representation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 1622–1632, 2025.
- [277] Ren Xin, Hongji Liu, Xiaodong Mei, Wenru Liu, Maosheng Ye, Zhili Chen, and Jun Ma. NetRoller: Interfacing general and specialized models for end-to-end autonomous driving. arXiv preprint arXiv:2506.14589, 2025.
- [278] Shuo Xing, Chengyuan Qian, Yuping Wang, Hongyuan Hua, Kexin Tian, Yang Zhou, and Zhengzhong Tu. OpenEMMA: Open-source multimodal model for end-to-end autonomous driving. In IEEE/CVF Winter Conf. Appl. Comput. Vis., pages 1001–1009, 2025.
- [279] Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. GoalFlow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 1602–1611, 2025.
- [280] Zebin Xing, Yupeng Zheng, Qichao Zhang, Zhixing Ding, Pengxuan Yang, Songen Gu, Zhongpu Xia, and Dongbin Zhao. Mimir: Hierarchical goal-driven diffusion with uncertainty propagation for end-to-end autonomous driving. arXiv preprint arXiv:2512.07130, 2025.
- [281] Chengkai Xu, Jiaqi Liu, Yicheng Guo, Yuhang Zhang, Peng Hang, and Jian Sun. Towards human-centric autonomous driving: A fast-slow architecture integrating large language model guidance with reinforcement learning. arXiv preprint arXiv:2505.06875, 2025.
- [282] Haoran Xu, Peixi Peng, Guang Tan, Yiqian Chang, Yisen Zhao, and Yonghong Tian. Temporal triplane transformers as occupancy world models. arXiv preprint arXiv:2503.07338, 2025.
- [283] Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Ekaterina Tolstaya, Sarah Tang, Brandyn White, Ben Sapp, Mingxing Tan, Jyh-Jing Hwang, and Dragomir Anguelov. WOD-E2E: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. arXiv preprint arXiv:2510.26125, 2025.
- [284] Tianshuo Xu, Hao Lu, Xu Yan, Yingjie Cai, Bingbing Liu, and Yingcong Chen. Occ-LLM: Enhancing autonomous driving with occupancy-based large language models. In IEEE Int. Conf. Robot. Autom., 2025.
- [285] Xiang Xu, Lingdong Kong, Song Wang, Chuanwei Zhou, and Qingshan Liu. Beyond one shot, beyond one perspective: Cross-view and long-horizon distillation for better LiDAR representations. In IEEE/CVF Int. Conf. Comput. Vis., pages 25506–25518, 2025.
- [286] Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P. Meyer, Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M. Wolff, and Xin Huang. VLM-AD: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024.
- [287] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K. Wong, Zhenguo Li, and Hengshuang Zhao. DriveGPT4: Interpretable end-to-end autonomous driving via large language model. IEEE Robot. Autom. Lett., 9(10):8186–8193, 2024.

- [288] Zhenhua Xu, Yan Bai, Yujia Zhang, Zhuoling Li, Fei Xia, Kwan-Yee K. Wong, Jianqiang Wang, and Hengshuang Zhao. DriveGPT4-V2: Harnessing large language model capabilities for enhanced closed-loop autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 17261–17270, 2025.
- [289] Tianyi Yan, Tao Tang, Xingtai Gui, Yongkang Li, Jiasen Zhesng, Weiyao Huang, Lingdong Kong, Wencheng Han, Xia Zhou, Xueyang Zhang, Yifei Zhan, Kun Zhan, Cheng zhong Xu, and Jianbing Shen. AD-R1: Closedloop reinforcement learning for end-to-end autonomous driving with impartial world models. arXiv preprint arXiv:2511.20325, 2025.
- [290] Tianyi Yan, Dongming Wu, Wencheng Han, Junpeng Jiang, Xia Zhou, Kun Zhan, Cheng-zhong Xu, and Jianbing Shen. DrivingSphere: Building a high-fidelity 4D world for closed-loop simulation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 27531–27541, 2025.
- [291] Xu Yan, Haiming Zhang, Yingjie Cai, Jingming Guo, Weichao Qiu, Bin Gao, Kaiqiang Zhou, Yue Zhao, Huan Jin, Jiantao Gao, Zhen Li, Lihui Jiang, Wei Zhang, Hongbo Zhang, Dengxin Dai, and Bingbing Liu. Forging vision foundation models for autonomous driving: Challenges, methodologies, and opportunities. arXiv preprint arXiv:2401.08045, 2024.
- [292] Ziyang Yan, Wenzhen Dong, Yihua Shao, Yuhang Lu, Haiyang Liu, Jingwen Liu, Haozhe Wang, Zhe Wang, Yan Wang, Fabio Remondino, and Yuexin Ma. RenderWorld: World model with self-supervised 3D label. In IEEE Int. Conf. Robot. Autom., pages 6063–6070, 2025.
- [293] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [294] An Yang et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [295] Haohan Yang, Yanxin Zhou, Jingda Wu, Haochen Liu, Lie Yang, and Chen Lv. Human-guided continual learning for personalized decision-making of autonomous driving. IEEE Int. Conf. Intell. Transport. Syst., 26(4): 5435–5447, 2025.
- [296] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, Jun Zhang, Andreas Geiger, Yu Qiao, and Hongyang Li. Generalized predictive model for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14662–14672, 2024.
- [297] Yu Yang, Jianbiao Mei, Yukai Ma, Siliang Du, Wenqing Chen, Yijie Qian, Yuxiang Feng, and Yong Liu. Driving in the occupancy world: Vision-centric 4D occupancy forecasting and planning via world models for autonomous driving. In AAAI Conf. Artifi. Intell., volume 39, pages 9327–9335, 2025.
- [298] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 14673–14684, 2024.
- [299] Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan. DriveMoE: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving. arXiv preprint arXiv:2505.16278, 2025.
- [300] Zhenjie Yang, Xiaosong Jia, Qifeng Li, Xue Yang, Maoqing Yao, and Junchi Yan. Raw2Drive: Reinforcement learning with aligned world models for end-to-end autonomous driving (in CARLA-V2). arXiv preprint arXiv:2505.16394, 2025.
- [301] Zhenjie Yang et al. LLM4Drive: A survey of large language models for autonomous driving. In Adv. Neural Inf. Process. Syst. Worksh., 2024.
- [302] Huaiyuan Yao, Pengfei Li, Bu Jin, Yupeng Zheng, An Liu, Lisen Mu, Qing Su, Qian Zhang, Yilun Chen, and Peng Li. LiloDriver: A lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. arXiv preprint arXiv:2505.17209, 2025.
- [303] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. LLaDA-V: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

- [304] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 2636–2645, 2020.
- [305] Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion-tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [306] Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, Xiaocheng Feng, Jun Song, Bo Zheng, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. RLAIF-V: Open-source AI feedback leads to super GPT-4V trustworthiness. arXiv preprint arXiv:2405.17220, 2024.
- [307] Ze Yu, Jun Li, Yuzhen Wei, Yuandong Lyu, and Xiaojun Tan. Combining camera-LiDAR fusion and motion planning using bird’s-eye view representation for end-to-end autonomous driving. Drones, 9(4):281, 2025.
- [308] Chengran Yuan, Zhanqi Zhang, Jiawei Sun, Shuo Sun, Zefan Huang, Christina Dao Wen Lee, Dongen Li, Yuhang Han, Anthony Wong, Keng Peng Tee, and Marcelo H. Ang Jr. DRAMA: An efficient end-to-end motion planner for autonomous driving with mamba. arXiv preprint arXiv:2408.03601, 2024.
- [309] Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, and Matthew Gadd. RAGDriver: Generalisable driving explanations with retrieval-augmented in-context learning in multi-modal large language model. arXiv preprint arXiv:2402.10828, 2024.
- [310] Zhenlong Yuan, Chengxuan Qian, Jing Tang, Rui Chen, Zijian Song, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. AutoDrive-R2: Incentivizing reasoning and self-reflection capacity for VLA model in autonomous driving. arXiv preprint arXiv:2509.01944, 2025.
- [311] Mahmut Yurt, Xin Ye, Yunsheng Ma, Jingru Luo, Abhirup Mallik, John Pauly, Burhaneddin Yaman, and Liu Ren. LTDA-Drive: LLMs-guided generative models based long-tail data augmentation for autonomous driving. arXiv preprint arXiv:2505.18198, 2025.
- [312] Shuang Zeng, Xinyuan Chang, Mengwei Xie, Xinran Liu, Yifan Bai, Zheng Pan, Mu Xu, Xing Wei, and Ning Guo. FutureSightDrive: Thinking visually with spatio-temporal CoT for autonomous driving. arXiv preprint arXiv:2505.17685, 2025.
- [313] Mingliang Zhai, Cheng Li, Zengyuan Guo, Ningrui Yang, Xiameng Qin, Sanyuan Zhao, Junyu Han, Ji Tao, Yuwei Wu, , and Yunde Jia. World knowledge-enhanced reasoning using instruction-guided interactor in autonomous driving. In AAAI Conf. Artifi. Intell., volume 39, pages 9842–9850, 2025.
- [314] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF Int. Conf. Comput. Vis., pages 11975–11986, 2023.
- [315] Bozhou Zhang, Nan Song, Jingyu Li, Xiatian Zhu, Jiankang Deng, and Li Zhang. Future-aware end-to-end driving: Bidirectional modeling of trajectory planning and scene evolution. arXiv preprint arXiv:2510.11092, 2025.
- [316] Dapeng Zhang, Zhenlong Yuan, Zhangquan Chen, Chih-Ting Liao, Yinda Chen, Fei Shen, Qingguo Zhou, and Tat-Seng Chua. Reasoning-VLA: A fast and general vision-language-action reasoning model for autonomous driving. arXiv preprint arXiv:2511.19912, 2025.
- [317] Diankun Zhang, Guoan Wang, Runwen Zhu, Jianbo Zhao, Xiwu Chen, Siyu Zhang, Jiahao Gong, Qibin Zhou, Wenyuan Zhang, Ningzi Wang, Feiyang Tan, Hangning Zhou, Ziyao Xu, Haotian Yao, Chi Zhang, Xiaojun Liu, Xiaoguang Di, and Bin Li. SparseAD: Sparse query-centric paradigm for efficient end-to-end autonomous driving. arXiv preprint arXiv:2404.06892, 2024.
- [318] Haiming Zhang, Ying Xue, Xu Yan, Jiacheng Zhang, Weichao Qiu, Dongfeng Bai, Bingbing Liu, Shuguang Cui, and Zhen Li. An efficient occupancy world model via decoupled dynamic flow and image-assisted training. arXiv preprint arXiv:2412.13772, 2024.
- [319] Haiming Zhang, Wending Zhou, Yiyao Zhu, Xu Yan, Jiantao Gao, Dongfeng Bai, Yingjie Cai, Bingbing Liu, Shuguang Cui, and Zhen Li. VisionPAD: A vision-centric pre-training paradigm for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog., pages 17165–17175, 2025.
- [320] Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, and Bo Li. SafeAuto: Knowledge-enhanced safe autonomous driving with multimodal foundation models. arXiv preprint arXiv:2503.00211, 2025.

- [321] Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving. arXiv preprint arXiv:2506.24113, 2025.
- [322] Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. TinyLLaMA: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
- [323] Songyan Zhang, Wenhui Huang, Zihui Gao, Hao Chen, and Chen Lv. WiseAD: Knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951, 2024.
- [324] Songyan Zhang, Wenhui Huang, Zhan Chen, Chua Jiahao Collister, Qihang Huang, and Chen Lv. OpenREAD: Reinforced open-ended reasoing for end-to-end autonomous driving with LLM-as-Critic. arXiv preprint arXiv:2512.01830, 2025.
- [325] Yi Zhang, Erik Leo Haß, Kuo-Yi Chao, Nenad Petrovic, Yinglei Song, Chengdong Wu, and Alois Knoll. A unified perception-language-action framework for adaptive autonomous driving. arXiv preprint arXiv:2507.23540, 2025.
- [326] Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In IEEE/CVF Int. Conf. Comput. Vis., pages 15222–15232, 2021.
- [327] Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Zhenhai Gao, and Fei Gao. Sce2DriveX: A generalized MLLM framework for scene-to-drive learning. IEEE Robot. Autom. Lett., 10(12):12580–12587, 2025.
- [328] Zongchuang Zhao, Haoyu Fu, Dingkang Liang, Xin Zhou, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Extending large vision-language model for diverse interactive tasks in autonomous driving. arXiv preprint arXiv:2505.08725, 2025.
- [329] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In IEEE/CVF Int. Conf. Comput. Vis., pages 22798–22807, 2023.
- [330] Weicheng Zheng, Xiaofei Mao, Nanfei Ye, Pengxiang Li, Kun Zhan, Xianpeng Lang, and Hang Zhao. DriveAgentR1: Advancing VLM-based autonomous driving with hybrid thinking and active perception. arXiv preprint arXiv:2507.20879, 2025.
- [331] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. OccWorld: Learning a 3D occupancy world model for autonomous driving. In Eur. Conf. Comput. Vis., pages 55–72. Springer, 2024.
- [332] Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. GenAD: Generative end-to-end autonomous driving. In Eur. Conf. Comput. Vis., pages 87–104. Springer, 2024.
- [333] Wenzhao Zheng, Junjie Wu, Yao Zheng, Sicheng Zuo, Zixun Xie, Longchao Yang, Yong Pan, Zhihui Hao, Peng Jia, Xianpeng Lang, and Shanghang Zhang. GaussianAD: Gaussian-centric end-to-end autonomous driving. arXiv preprint arXiv:2412.10371, 2024.
- [334] Wenzhao Zheng, Zetian Xia, Yuanhui Huang, Sicheng Zuo, Jie Zhou, and Jiwen Lu. Doe-1: Closed-loop autonomous driving with large world model. arXiv preprint arXiv:2412.09627, 2024.
- [335] Yupeng Zheng, Pengxuan Yang, Zebin Xing, Qichao Zhang, Yuhang Zheng, Yinfeng Gao, Pengfei Li, Teng Zhang, Zhongpu Xia, Peng Jia, XianPeng Lang, and Dongbin Zhao. World4Drive: End-to-end autonomous driving via intention-aware physical latent world model. In IEEE/CVF Int. Conf. Comput. Vis., pages 28632–28642, 2025.
- [336] Yifan Zhong et al. Yifan zhong and fengshuo bai and shaofei cai and xuchuan huang and zhang chen and xiaowei zhang and yuanfei wang and shaoyang guo and tianrui guan and ka nam lui and zhiquan qi and yitao liang and yuanpei chen and yaodong yang. arXiv preprint arXiv:2507.01925, 2025.
- [337] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In Int. Conf. Mach. Learn. PMLR, 2025.
- [338] Xingcheng Zhou, Mingyu Liu, Ekim Yurtsever, Bare Luka Zagar, Walter Zimmer, Hu Cao, and Alois C. Knoll. Vision language models in autonomous driving: A survey and outlook. IEEE Trans. Intell. Veh., pages 1–20, 2024.
- [339] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, Volker Tresp, and Alois Knoll. OpenDriveVLA: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463, 2025.
- [340] Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. AutoVLA: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757, 2025.

- [341] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. LanguageBind: Extending video-language pretraining to N-modality by language-based semantic alignment. In Int. Conf. Learn. Represent., 2024.
- [342] Dekai Zhu, Yixuan Hu, Youquan Liu, Dongyue Lu, Lingdong Kong, and Slobodan Ilic. SPIRAL: Semantic-aware progressive LiDAR scene generation and understanding. In Adv. Neural Inf. Process. Syst., volume 38, 2025.
- [343] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [344] Huiping Zhuang, Di Fang, Kai Tong, Yuchen Liu, Ziqian Zeng, Xu Zhou, and Cen Chen. Online analytic exemplar-free continual learning with large models for imbalanced autonomous driving task. IEEE Trans. Veh. Tech., 74(2):1949–1958, 2024.
- [345] Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. DiffusionDriveV2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745, 2025.

