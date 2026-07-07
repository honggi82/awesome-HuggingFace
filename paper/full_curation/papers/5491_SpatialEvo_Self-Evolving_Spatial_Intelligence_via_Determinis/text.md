# arXiv:2604.14144v1[cs.CV]15Apr2026

[Figure 1]

## SpatialEvo: Self-Evolving Spatial Intelligence via Deterministic Geometric Environments

Dingming Li1,∗, Yingxiu Zhao2,∗, Xinrui Cheng1, Kangheng Lin2, Hongbo Peng2, Hongxing Li1, Zixuan Wang1, Yuhong Dai2, Haodong Li2, Jia Wang2, Yukang Shi2, Liang Zhao2, Jianjian Sun2, Zheng Ge2,

Xiangyu Zhang2, Weiming Lu1, Jun Xiao1, Yueting Zhuang1, Yongliang Shen1,† 1 Zhejiang University 2 StepFun

GitHub Hugging Face

### Abstract

Spatial reasoning over three-dimensional scenes is a core capability for embodied intelligence, yet continuous model improvement remains bottlenecked by the cost of geometric annotation. The self-evolving paradigm offers a promising path, but its reliance on model consensus to construct pseudo-labels causes training to reinforce rather than correct the model’s own geometric errors. We identify a property unique to 3D spatial reasoning that circumvents this limitation: ground truth is a deterministic consequence of the underlying geometry, computable exactly from point clouds and camera poses without any model involvement. Building on this insight, we present SpatialEvo, a self-evolving framework for 3D spatial reasoning, centered on the Deterministic Geometric Environment (DGE). The DGE formalizes 16 spatial reasoning task categories under explicit geometric validation rules and converts unannotated 3D scenes into zero-noise interactive oracles, replacing model consensus with objective physical feedback. A single shared-parameter policy co-evolves across questioner and solver roles under DGE constraints: the questioner generates physically valid spatial questions grounded in scene observations, while the solver derives precise answers against DGE-verified ground truth. A task-adaptive scheduler endogenously concentrates training on the model’s weakest categories, producing a dynamic curriculum without manual design. Experiments across nine benchmarks demonstrate that SpatialEvo achieves the highest average score at both 3B and 7B scales, with consistent gains on spatial reasoning benchmarks and no degradation on general visual understanding.

##### 1. Introduction

Spatial reasoning requires models to form persistent perception and understanding of complex three-dimensional scenes, serving as a core capability for embodied intelligence, robot navigation, and scene question answering (Chen et al., 2024a; Cheng et al., 2024; Li et al., 2025d; Ma et al., 2024b). Yet how to continuously drive model iteration at scale and overcome the bottleneck of data acquisition remains an open challenge, one that existing approaches address primarily through large-scale annotated datasets pairing multi-view images with geometric question-answer pairs (Chen et al., 2024a; Song et al., 2025).

These efforts have yielded meaningful progress, yet they share a common structural weakness: the training distribution is fixed at dataset creation time. A static corpus cannot respond to where a model is weak today, cannot generate harder examples as the model grows stronger (Tian et al., 2025; Wang et al., 2025c), and cannot scale without proportional investment in human

∗ Equal Contribution † Corresponding Author

[Figure 2]

[Figure 3]

[Figure 4]

#### Static Data Tuning Consensus Self-Evolve SpatialEvo (Ours)

[Figure 5]

[Figure 6]

3D Scene Parsing

[Figure 7]

Consensus-based Answer Anchoring

[Figure 8]

[Figure 9]

Manually labeled data

Extracted Labeled Data

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

DGE

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Provide GT

Visual Visual Observation Observation Visual Observation

Deliver Question

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

😡 😡 😃 😃

Manually Annotated, Fixed Dataset Incorrect or Biased GT Dynamic Learning Accurate GT

[Figure 40]

[Figure 41]

[Figure 42]

- Figure 1 | Comparison of three training paradigms. Static Data Tuning relies on manually annotated fixed datasets. Consensus Self-Evolve eliminates manual annotation but introduces biased ground truth via model voting. SpatialEvo (Ours) leverages the DGE to compute exact ground truth from 3D scene assets, enabling zero-noise dynamic curriculum learning.

annotation (Lyu et al., 2024; Zheng et al., 2026). This raises a fundamental question: how can a model continuously improve its spatial reasoning without depending on an ever-growing supply of labeled data?

The self-evolving paradigm offers a principled answer. By actively distilling training signals from raw visual inputs through iterative self-play, self-evolving models can progressively internalize reasoning paradigms (Deng et al., 2024; He et al., 2025; Li et al., 2025e) and continuously update their training distribution alongside their own capability, providing a natural dynamic curriculum without manual curation (Liu et al., 2024; Zhao et al., 2025). Yet existing self-evolving methods share a critical limitation: because ground truth cannot be read off the environment directly, the training signal must be constructed by aggregating the model’s own predictions through majority voting or self-consistency(He et al., 2025; Shi et al., 2025a; Tao et al., 2024; Wang et al., 2025a; Wu et al., 2024). This introduces a systematic bias: the pseudo-labels inherit the model’s own prediction errors, and gradient updates anchored to such labels risk reinforcing rather than correcting the model’s existing errors rather than correcting them.

We observe that 3D spatial reasoning possesses a distinctive property that circumvents this limitation entirely. Unlike natural language or general vision tasks, the ground truth for a spatial question is a deterministic consequence of the underlying geometry (Cheng et al., 2024; Luo et al., 2025; Ma et al., 2025; Xu et al., 2025). Given a dense point cloud, calibrated camera poses, and a well-formed geometric question, the correct answer can be computed exactly and programmatically, with no appeal to any model’s judgment.

A question about absolute distance reduces to a nearest-point computation on object bounding boxes; a question about relative camera orientation reduces to an arithmetic operation on rotation matrices (Mayer et al., 2025; Zhou et al., 2025). The model consensus that other domains rely on as a noisy proxy is therefore unnecessary (Cai et al., 2025a; Song et al., 2025): the physical world itself serves as an exact and impartial judge. Every unannotated 3D indoor scene is, in principle, an inexhaustible source of noise-free supervision (Cheng et al., 2024; Liu et al., 2025a), waiting to be converted into a training signal.

To this end, we propose SpatialEvo, the first framework to introduce the self-evolving paradigm into 3D spatial reasoning. Its core contribution is the Deterministic Geometric Environment (DGE), which designs an atomic geometric verification rule set covering 16 spatial reasoning task categories, programmatically computes exact ground truth from 3D point clouds

and camera pose sequences, and transforms 3D scene assets into zero-noise online reward judges (Cheng et al., 2024). Building on this, the GRPO (Shao et al., 2024) framework drives a single VLM to co-evolve between questioner and solver roles (He et al., 2025; Li et al., 2026): the questioner perceives the overall scene layout and generates physically valid spatial questions, while the solver internalizes spatial reasoning paradigms through explicit geometric derivation under DGE constraints. This division naturally aligns with the two cognitive levels of spatial reasoning, namely holistic environment perception and local geometric solving. A task scheduler dynamically regulates the sampling distribution based on historical accuracy, yielding adaptive curriculum learning (Shi et al., 2025b).

Extensive experiments across nine benchmarks show that SpatialEvo achieves the highest average score at both 3B and 7B scales, with consistent gains on spatial reasoning benchmarks and no degradation on general visual understanding. Ablation studies confirm that replacing DGE ground truth with majority-vote pseudo-labels produces the single largest performance drop, directly validating the role of deterministic physical feedback.

In summary, our contributions are:

- • We propose SpatialEvo, a self-evolving framework for 3D spatial reasoning centered on a deterministic geometric environment, integrating VLM self-play with programmatic physical verification to replace model consensus with objective environmental feedback for noise-free continuous evolution.
- • We design an automated geometric verification pipeline that decouples 3D spatial reasoning into executable atomic verification rules spanning 16 task categories, transforming unannotated 3D scene datasets into an online-interactive, zero-noise ground truth judge engine.
- • We introduce a spatially grounded policy co-evolution mechanism where a single model simultaneously occupies questioner and solver roles under DGE constraints, with adaptive task scheduling driving curriculum self-emergence, validated through extensive experiments.

##### 2. Related Works

Spatial Reasoning in Vision-Language Models Early work sought to enhance the spatial perception of VLMs through multi-view image fusion or depth information injection (Chen et al., 2024a; Cheng et al., 2024; Ma et al., 2024a). More recent approaches have proceeded along two parallel tracks: data construction and reasoning-oriented training. Methods such as SpatialVLM (Chen et al., 2024a), SpatialBot (Cai et al., 2025a), and SpatialLadder (Li et al., 2025b) achieve notable gains by fine-tuning on large-scale spatial annotation datasets (Cai et al., 2025b; Ouyang et al., 2025), while other efforts explore programmatic synthesis or bird’s-eye-view augmentation to reduce annotation costs (Yang et al., 2025b). However, the datasets produced by these approaches are static by nature, fixed at generation time and thus insensitive to the model’s dynamically shifting weaknesses during training, which fundamentally limits training efficiency.

Self-Evolution of Vision-Language Models Self-evolution has emerged as a prominent research direction in the LLM community (Deng et al., 2025; Li et al., 2025f; Tao et al., 2024; Zhai

- et al., 2024) and has progressively extended to VLMs (Deng et al., 2024; Liu et al., 2024). Notably, the spatial reasoning domain is particularly well-suited for self-evolution, as its visual inputs inherently carry physical information enabling deterministic ground truth computation, naturally

###### 3D Scene Context

Deterministic Geometric Environment (DGE) – Geometric Oracle

Spatial-Grounded GRPO

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Entity Extraction Logic Check GT Computation

[Figure 48]

[Figure 49]

Questioner

[Figure 50]

Sampled Questions

[Figure 51]

(Observation & Valid)

𝑪𝒉𝒆𝒄𝒌 𝑽𝒂𝒍𝒊𝒅𝒊𝒕𝒚(𝒇𝒗𝒂𝒍𝒊𝒅) 𝑮(𝒊) Zero-noise of Ground Truth (𝒇𝒂𝒄𝒄)

[Figure 52]

{𝑸(𝒊)}

###### Solver

|Floating RGB Images<br><br>[Figure 53]<br><br>[Figure 54]<br><br>z<br><br>y<br><br>x<br><br>x<br><br>[Figure 55]<br><br>Camera Pose Sequence<br><br>[Figure 56]<br><br>[Figure 57]|
|---|

(Accurate || Explanation)

Q(𝒊) 𝑹𝒆𝒘𝒂𝒓𝒅 𝒓𝑸(𝒊)

###### Invaild Reason (𝒇𝒆𝒙𝒑𝒍𝒂𝒊𝒏)

###### …

Lightweight Task Scheduler

…

𝑹𝒆𝒘𝒂𝒓𝒅 𝒓𝑨(𝒊,𝒋)

[Figure 58]

[Figure 59]

Sampled Answers

###### Q

###### A

Sampling Probability

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

{𝑨(𝒊,𝒋)}

𝝅𝜽

𝝅𝜽

Solver

Questioner

𝓪𝒌

…

Raw Rewards in Groups

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

𝛁𝜽 Joint

[Figure 74]

Arrows of Dual Knowledge Flow

[Figure 75]

###### Visual Observation Spatial Reasoning

[Figure 76]

[Figure 77]

[Figure 78]

Optimization

[Figure 79]

[Figure 80]

16 Spatial Tasks

[Figure 81]

Questioner(VisualObservation)

[Figure 82]

[Figure 83]

[Figure 84]

r = 𝛼 𝑓 +(1 − 𝛼) 𝑓 𝑓

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

###### Adaptive Curriculum

[Figure 90]

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

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

𝑖𝑓 𝑓 = 1

Solver

(Spatial Reasoning)

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

(1 − 𝛼) 𝑓 (1 − 𝛼) 𝑓

Prompt: Carefully examine the images and answer the following question step by step. Write your reasoning in plain text, and keep each step grounded in the visual evidence. Only the…

Prompt: Performs holistic 3D scene reasoning by linking global spatial layout to a localized target, enabling the formulation of a single, well-grounded and task-consistent question…

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

r = 𝛼 𝑓 +

[Figure 118]

[Figure 119]

[Figure 120]

###### DGE

𝑖𝑓 𝑓 = 0

[Figure 121]

[Figure 122]

[Figure 123]

GRPO — Advantage Normalization

𝑂𝑛𝑒 𝐺𝑟𝑜𝑢𝑝

(n ∗ 𝑄𝑢𝑒𝑠𝑡𝑖𝑜𝑛)

Observe and Geometric Solution Raise Questions

𝑟( ) − 𝑚𝑒𝑎𝑛(𝑟( )) 𝑠𝑡𝑑(𝑟( )) + 𝜖

[Figure 124]

𝐴 ( ) =

[Figure 125]

𝑂𝑛𝑒 𝐺𝑟𝑜𝑢𝑝

𝑚 ∗ (𝑛 ∗ 𝐴𝑛𝑠𝑤𝑒𝑟)

3D Point Cloud

Spatial Self-Evolution Loop

- Figure 2 | Overview of SpatialEvo. The DGE computes zero-noise ground truth directly from
- 3D point clouds and camera pose sequences, enabling the Spatial Self-Evolution Loop to jointly optimize a single VLM as both Questioner and Solver via Spatial-Grounded GRPO.

circumventing reliance on model consensus. VisPlay (He et al., 2025), EvolMM (Thawakar et al., 2025), and V-Zero (Wang et al., 2026) advance multimodal self-play but remain constrained by static image corpora; Vision-Zero (Wang et al., 2025b) and MM-Zero (Li et al., 2026) achieve continual improvement via verifiable rewards or zero-data rendering. Critically, all rely on model consensus as a reward proxy, introducing systematic bias where precise physical grounding is required. SpatialEvo is the first to bring self-evolution to 3D spatial reasoning, replacing model-derived judgment with deterministic geometric computation.

##### 3. Methodology

###### 3.1. Preliminaries

Problem Formulation. We formulate spatial reasoning as a visual question-answering process grounded in multi-view RGB imagery. Given a 3D indoor scene S whose underlying assets comprise dense point clouds, semantic annotations, and camera pose sequences, the observable input to the policy model is restricted to the corresponding set of RGB image frames I = {𝐼1, 𝐼2, . . . , 𝐼𝑇}. Our objective is to learn a single vision-language policy model 𝜋𝜃 endowed with two complementary capabilities: as a questioner, generating spatially meaningful reasoning questions 𝑄 from visual observations; and as a solver, predicting precise answers 𝐴 given an image context and a question description.

The task space T encompasses 16 core spatial reasoning tasks, organized into three categories by observational granularity. Multi-image scene-level tasks (6 tasks)—object counting, object size, absolute distance, relative distance, relative direction, and room size estimation—require the model to integrate global 3D layout information across multiple frames (Yang et al., 2025a). Single-image tasks (3 tasks)—single-view relative direction, camera-to-object distance (Distance Cam-Obj), and inter-object depth ordering (Depth Order Obj-Obj)—assess the model’s understanding of single-frame perspective geometry. Dual-image tasks (7 tasks)—inter-camera relative position (Position Cam-Cam), camera-to-object position (Position Cam-Obj), camera-to-region

position (Position Cam-Reg), camera motion estimation (Yang et al., 2025c), visibility comparison, inter-camera elevation (Elevation Cam-Cam), and attribute measurement—focus on geometrically consistent inference across viewpoints. The complete definitions and verification rules for all 16 task categories are provided in Appendix B.1.

Collectively, these tasks span three semantic dimensions: metric measurement, topological spatial relations, and camera pose reasoning. For instance, computing the relative orientation between two cameras requires resolving the rotational component between their extrinsic matrices, while inferring inter-object depth ordering requires aligning perspective projection with the scene point cloud. A defining characteristic shared by all tasks is that their answers can be computed programmatically and exactly from the scene’s underlying geometric assets, providing a natural foundation for automatic ground-truth generation and, in turn, for noise-free reinforcement learning signals.

Framework Overview. As illustrated in Figure 2, the SpatialEvo framework consists of two core components: the Deterministic Geometric Environment (DGE) and Spatial-Grounded Policy Co-Evolution. The DGE takes 3D point clouds and camera pose sequences as underlying scene assets, applies task-specific geometric verification rules to validate candidate questions, and programmatically computes exact ground-truth for verified questions, serving as a noise-free physical feedback judge. Policy co-evolution is built on the GRPO algorithm, where a single policy model 𝜋𝜃 switches between Questioner and Solver via role-conditioned prompting: the Questioner perceives the global 3D scene layout to generate spatially valid and reasoning-worthy questions, while the Solver focuses on precise geometric inference under the hard constraints of DGE-verified ground truth, with rewards anchored directly to physical computation results. A lightweight task scheduler dynamically adjusts the task sampling distribution based on the Solver’s historical performance, driving the emergence of an adaptive curriculum without any human intervention.

###### 3.2. Deterministic Geometric Environment

The Deterministic Geometric Environment (DGE) constitutes the physical feedback core of the SpatialEvo framework, serving as the Geometric Oracle within the self-evolution loop. Its fundamental function is to map natural language questions generated by the policy model onto the underlying 3D scene assets for objective verification, replacing unreliable model judgment with programmatic geometric computation to supply reinforcement learning with zero-noise supervisory signals. The DGE operates through two tightly coupled submodules: a task-specific geometric validation rule set and an automated verification pipeline. Owing to a rich library of spatial numerical computation tools, the DGE is designed for extensibility: supporting a new task requires only the specification of the corresponding rule set and validation logic.

###### 3.2.1. Task-Specific Geometric Validation Rule Sets

For each of the 16 spatial reasoning task categories, the DGE pre-defines a geometric verification rule set that decouples complex spatial intuition into executable atomic criteria, rendering abstract geometric reasoning problems mathematically decidable. Each rule set constrains question validity along three dimensions: premise consistency requires that all scene entities referenced in a question (e.g., frame indices, object categories, or spatial regions) exist in the underlying assets and can be uniquely localized; Inferential solvability requires that the geometric premises of a question are unambiguously computable: for instance, metric tasks must ensure that the reference object’s point cloud density meets a minimum precision threshold,

while orientation tasks must ensure sufficient viewpoint disparity between the two frames to constitute a meaningful spatial relationship; geometric degeneracy filtering discards physically unstable, highly ambiguous, or low-training-value edge cases. Note that numerical tolerances for continuous-valued tasks are not part of the validity rules but are defined separately at the reward stage; see Appendix B.1 for this distinction.

Taking camera orientation tasks as an example, the rule set requires the Questioner to specify valid frame index pairs; the DGE then computes the relative rotation matrix and translation vector between the two poses to determine whether they constitute an answerable spatial relationship, mapping the result to a discrete directional label. For object size or room metric tasks, the DGE directly performs bounding box fitting or plane normal estimation on the scene point cloud to obtain precise metric values. For depth ordering tasks, the DGE projects the scene point cloud onto the corresponding camera image plane and derives the groundtruth ordering by comparing the median depth of target objects. This rule-based structured decomposition strategy ensures that the verification logic across all task categories carries unambiguous mathematical semantics. Complete rule definitions and verification logic for all task categories are provided in Appendix B.1.

###### 3.2.2. Automated Verification Pipeline

Driven by the validation rule sets, the DGE instantiates a fully automated, end-to-end pipeline spanning question parsing to ground-truth synthesis, converting existing 3D scene datasets such as ScanNet (Dai et al., 2017) and ScanNet++ (Yeshwanth et al., 2023) into interactive, deterministic ground-truth judging engines. The pipeline executes in three stages.

- Stage 1: Entity Parsing. Since questions generated by the Questioner are in free-form natural language, the DGE invokes a lightweight LLM to perform structured entity extraction over the question text, identifying core semantic elements such as frame indices, object categories, and spatial relationship descriptions, and normalizing them into structured representations directly consumable by the backend geometric computation module. The prompt design is detailed in Appendix B.2.
- Stage 2: Legality Verification. The DGE validates the extracted entities against the rule set corresponding to the current task category, checking premise consistency and reasoning chain validity in sequence. If a question references a non-existent scene entity or its geometric premises violate physical constraints, such as an invalid frame index or a reference object whose point cloud is too sparse, the DGE triggers a truncation mechanism and returns a negative reward to the questioner, suppressing the propagation of hallucinated questions at the data generation stage and thereby preserving the overall signal-to-noise ratio of the training signal.
- Stage 3: Ground-Truth Synthesis. For questions that pass legality verification, the DGE invokes its geometric toolkit to perform precise numerical computation in the global coordinate frame. Core operators include rigid-body coordinate transformation, point cloud bounding-box fitting and topological analysis, depth-map perspective projection, and planar normal estimation, collectively covering the full computational requirements of all 16 task categories across the metric, topological, and orientation semantic dimensions. The pipeline outputs not only groundtruth answers with guaranteed precision but also intermediate geometric states recorded during inference, such as keypoint coordinates and camera trajectory segments, providing interpretable intermediate representations for subsequent qualitative analysis and error attribution.

This paradigm of replacing black-box model prediction with programmatic physical computation eliminates reward noise from the self-evolution process, ensuring that every gradient

update of the policy model is anchored to objective physical laws.

###### 3.3. Spatial-Grounded Policy Co-Evolution

Within the SpatialEvo framework, the co-evolution of the questioner and the solver serves as the primary driver of spatial intelligence advancement. This section details the spatial-grounded policy co-evolution mechanism, covering the self-play architecture, task scheduling design, reward function formulation, and the GRPO training procedure.

###### 3.3.1. Spatial Self-Play Mechanism

SpatialEvo employs a single policy model 𝜋𝜃 that alternates between the questioner and solver roles via role-conditioned prompting. Parameter sharing confers a dual advantage: gradients acquired by the solver through geometric derivation directly improve the questioner’s visualspatial perception, while the geometric intuitions developed by the questioner during boundary exploration in turn deepen the solver’s reasoning capacity, establishing a virtuous cycle of mutual knowledge reinforcement between the two roles (He et al., 2025; Li et al., 2026).

In this mechanism, the task scheduler first infers the feasible task set T𝑠feasible from the asset characteristics of the current scene, then samples a task type according to the solver’s historical effective accuracy and assigns it to the questioner. The questioner perceives the holistic 3D layout of the scene from multi-view RGB images and generates a physically valid spatial reasoning question for the designated task type. The DGE immediately performs legality verification on the generated question and computes the geometric ground truth. The model then switches to the solver role and independently derives an answer under the hard constraint of the deterministic ground truth. This design encourages the model to actively probe the boundaries of its own spatial cognition during the question-generation phase, while correcting geometric hallucinations against the absolute ground truth supplied by the environment during the answering phase, forming a continuous self-reinforcement loop within the policy space.

###### 3.3.2. Task-Adaptive Scheduling

To realize curriculum learning without manual intervention, SpatialEvo introduces a lightweight task scheduler that dynamically modulates the task sampling distribution. The scheduler first infers the physically realizable feasible task set T𝑠feasible ⊆ T for the current scene. It then maintains, for each task category 𝑘, a cumulative score 𝑆𝑘 and a sample count 𝑁𝑘, and estimates the historical effective accuracy 𝑎¯𝑘 via pseudo-observation smoothing to mitigate estimation instability caused by sparse samples in the early stages of training. The sampling weight 𝑤𝑘 is negatively correlated with 𝑎¯𝑘, and a minimum exploration weight 𝛿 is introduced to prevent task categories that have been thoroughly mastered from being entirely excluded from the sampling pool. This mechanism automatically concentrates training resources on the solver’s current cognitive weak spots; as accuracy levels shift across task categories, the training focus migrates accordingly, giving rise to a fully adaptive curriculum driven endogenously by model performance, without any manually prescribed difficulty sequence. The detailed implementation and derivation of the scheduler are provided in Appendix C.1.

###### 3.3.3. Questioner Reward Design

The Questioner’s optimization objective focuses on generating a physically valid, visually grounded question of high quality for a given task category. The Questioner reward comprises

two components: format compliance and substantive quality, with 𝛼 = 0.1: 𝑟𝑄 = 𝛼 𝑓fmt + (1 − 𝛼) 𝑓valid · 𝑓obs.

Here 𝑓fmt measures whether the output conforms to predefined structural constraints; 𝑓valid is the geometric validity score from DGE verification; and 𝑓obs is a visual observation quality score that assesses whether the Questioner has sufficiently leveraged the global spatial information present in the multi-view images. 𝑓obs is evaluated by a lightweight LLM judge without relying on a powerful VLM, which scores only the Questioner’s textual observation description, examining whether it exhibits a natural perceptual hierarchy from global scene to local target and provides adequate visual grounding for the generated question; scoring criteria are detailed in Appendix C.2. The coupled term 𝑓valid · 𝑓obs carries a critical gating semantics: a positive signal is contributed only when a question satisfies both geometric validity and sufficient visual observation simultaneously, preventing the model from generating “superficially valid” questions that conform to format but lack genuine spatial understanding. Severe structural errors additionally trigger hard penalties in the implementation, which are omitted here for brevity.

###### 3.3.4. Solver Reward Design

The Solver receives reward signals from all candidate questions passed by the Questioner, with both valid and invalid questions serving as effective learning signal. For valid questions, 𝑓acc is anchored directly to the geometric ground truth produced by the DGE. For invalid questions, the DGE outputs a specific invalidation reason alongside its rejection, and the Solver is required to explicitly analyze and explain this reason in light of the task type, question generation rules, and visual observations; explanation quality is scored by a lightweight LLM judge that evaluates whether the model accurately identifies the core failure cause and correctly attributes it to task rules or scene constraints. Detailed evaluation functions for each task category and explanation scoring criteria are provided in Appendix C.3. Format compliance is incorporated into the Solver reward via 𝑓fmt to enforce adherence to predefined structural requirements. The unified Solver reward is thus written as, with 𝛼 = 0.1:

𝑟𝐴 =    

𝛼 𝑓fmt + (1 − 𝛼) 𝑓acc, if 𝑄 is valid, 𝛼 𝑓fmt + (1 − 𝛼) 𝑓explain, if 𝑄 is invalid,

where 𝑓fmt measures adherence to the expected output format, 𝑓acc measures agreement with DGE ground truth, and 𝑓explain measures the quality of the explanation for the invalidation reason. This design ensures that invalid questions also serve as meaningful learning signal, further deepening the model’s understanding of spatial task rules and geometric constraints. As with the Questioner, severe formatting errors trigger additional hard penalties in the implementation, which are omitted in favor of the weighted main form presented here.

###### 3.3.5. GRPO Training Procedure

We adopt the GRPO algorithm to drive the co-evolution of the questioner and solver. For each training scene, the scheduler samples a task type and directs the questioner to generate 𝑛

candidate questions {𝑄(𝑖)}𝑛𝑖=1. The DGE performs legality verification on each 𝑄(𝑖); for questions that pass, it computes the geometric ground truth 𝐺(𝑖) and assigns questioner reward 𝑟𝑄(𝑖); for questions that fail, it returns a specific invalidity reason. The 𝑛 candidate questions form a single GRPO group for the questioner, within which advantage values are computed independently based on {𝑟𝑄(𝑖)}𝑛𝑖=1.

The 𝑛 candidate questions are then deduplicated semantically to yield 𝑚 (𝑚 ≤ 𝑛) unique questions {𝑄(𝑖)}𝑚𝑖=1, which are passed to the solver. For each unique question 𝑄(𝑖), the solver independently samples 𝑛 candidate answers {𝐴(𝑖,𝑗)}𝑛𝑗=1 and computes the solver reward 𝑟𝐴(𝑖,𝑗) against 𝐺(𝑖); for invalid questions, the solver generates an explanation based on the invalidity reason returned by the DGE, with the reward assigned by the LLM judge. The solver forms 𝑚 independent GRPO groups, one per unique question, with advantage values computed independently within each group from the nn n candidate answers.

Advantage values are computed independently within their respective groups to eliminate reward-scale bias introduced by inherent difficulty variation across scenes:

𝑟(𝑖) − mean {𝑟(𝑖)} std {𝑟(𝑖)} + 𝜖

𝐴ˆ(𝑖) =

(1)

The questioner and solver share a single set of model parameters, and gradients from both sides are applied jointly at each training step.

##### 4. Experiments

###### 4.1. Experimental Setup

Environment and Data Configuration. We construct the DGE from the training splits of ScanNet (Dai et al., 2017), ScanNet++ (Yeshwanth et al., 2023), and ARKitScenes (Baruch et al., 2021), comprising approximately 4K source scenes in total. The DGE derives noise-free geometric supervision from dense reconstructions and multi-view streams, while requiring only standard RGB images as model input. Dataset and scene statistics are reported in Appendix D. We apply SpatialEvo to both Qwen2.5-VL-3B-Instruct (Bai et al., 2025) and Qwen2.5-VL-7B-Instruct (Bai

- et al., 2025) as backbone models.

Training Protocol. SpatialEvo trains entirely via online reinforcement learning using the GRPO framework, without any supervised fine-tuning stage. The Questioner and Solver share a single policy model and continuously interact with the DGE, enabling self-evolving spatial intelligence through deterministic geometric feedback. Full training details are provided in Appendix C.

Baselines. We compare against methods spanning the key training paradigms discussed in Section 1. SpatialLadder (Li et al., 2025b), SpaceR-SFT (Ouyang et al., 2025), and ViLaSR (Wu et al., 2025) represent static data tuning or RL-based training with either curated annotations or fixed reward functions, without access to deterministic geometric feedback. Spatial-SSRL (Liu et al., 2025b) adopts a self-supervised RL paradigm that derives training signals directly from RGB images, probing whether annotation-free generalization can match geometry-grounded self-evolution.

Evaluation Benchmarks. We evaluate across nine benchmarks covering spatial reasoning and general visual understanding. VSI-Bench (Yang et al., 2025a) is the primary metric, assessing quantitative multi-view spatial reasoning including object size, distance, and relative orientation, evaluated at 32 frames. EmbSpatial (Du et al., 2024) and ViewSpatial (Li et al., 2025a) target embodied spatial understanding and perspective-dependent reasoning. RealWorldQA (Zhang et al., 2024) tests spatial comprehension in open real-world scenes. V-STAR (Cheng et al., 2025) benchmarks video spatio-temporal reasoning. SpatialViz (Wang et al.) evaluates spatial visualization and object transformation reasoning. STARE (Li et al., 2025c) assesses multi-step visual simulation across geometric transformations and real-world spatial tasks. CoreCognition

Qwen2.5-VL-3B Qwen2.5-VL-7B

Benchmark

Baseline SpatialLadder SpaceR SpatialSSRL SpatialEvo Baseline VILASR SpaceR SpatialSSRL SpatialEvo

VSI-Bench (Yang et al., 2025a) 28.1 45.7 36.0 28.0 39.2 31.1 45.4 36.8 33.7 46.1 RealWorldQA (Zhang et al., 2024) 63.4 57.1 61.4 65.4 66.5 69.5 57.9 64.7 69.9 66.7 EmbSpatial (Du et al., 2024) 55.9 57.6 55.6 59.8 61.2 63.6 47.8 60.3 69.3 66.0 SpatialViz (Wang et al.) 24.2 28.6 31.9 25.9 25.4 27.0 29.8 30.9 28.4 28.6 STARE (Li et al., 2025c) 33.1 26.4 36.8 36.8 36.9 41.8 21.4 36.2 43.3 41.3 CoreCognition (Li et al., 2024) 56.8 58.3 29.1 57.6 57.4 59.6 56.4 56.4 60.2 60.2 ViewSpatial (Li et al., 2025a) 36.2 43.0 35.9 38.4 42.3 36.4 32.3 35.1 37.5 43.2 V-STAR (Cheng et al., 2025) 74.9 36.7 75.4 77.0 75.4 78.5 35.6 73.8 79.1 78.0 MMStar (Chen et al., 2024b) 54.6 45.8 44.9 56.5 55.2 61.6 60.8 54.9 63.5 62.5

AVG 47.5 44.4 45.2 49.5 51.1 52.1 43.0 49.9 53.9 54.7

Table 1 | Main Results. Nine benchmarks spanning spatial reasoning and general visual understanding. Shaded columns indicate SpatialEvo (ours). Bold denotes the best result per benchmark under each backbone.

probes (Li et al., 2024) core cognitive knowledge spanning both spatial and general perceptual reasoning. MMStar (Chen et al., 2024b) verifies that spatial training does not degrade general visual capabilities. All benchmarks follow their original evaluation protocols unless otherwise noted.

###### 4.2. Main Results

- Table 1 reports results across nine benchmarks for both Qwen2.5-VL-3B and Qwen2.5-VL-7B backbones. SpatialEvo achieves the highest average score under both settings, reaching 51.1 (3B) and 54.7 (7B), outperforming all baselines by a consistent margin across both model scales.

Core Spatial Reasoning. SpatialEvo yields substantial improvements on VSI-Bench, EmbSpatial, and ViewSpatial, which directly correspond to the multi-view scene-level and perspectivedependent task categories covered by our DGE. On VSI-Bench, SpatialEvo achieves 39.2 (3B) and 46.1 (7B), surpassing all baselines including SpatialLadder (45.4) and SpaceR (36.8) under the 7B setting. Consistent gains are similarly observed on EmbSpatial and ViewSpatial, confirming that DGE-driven self-evolution effectively internalizes both embodied and perspective-dependent spatial reasoning and transfers well to held-out benchmarks.

General Capability Retention. SpatialEvo maintains competitive performance on MMStar and RealWorldQA, demonstrating that spatial specialization does not degrade general visual understanding. On MMStar, SpatialEvo scores 55.2 (3B) and 62.5 (7B), remaining close to the untuned baseline while annotation-dependent methods such as SpatialLadder (45.8) and ViLaSR (60.8) fall noticeably below. On RealWorldQA, SpatialEvo achieves 66.5 (3B) and 66.7 (7B), outperforming both the baseline and all spatial-specialized competitors under the 3B setting. Spatial-SSRL also retains general capabilities reasonably well; however, its overall average remains below SpatialEvo in both settings, indicating that annotation-free generalization alone is insufficient to match geometry-grounded self-evolution.

Baseline Degradation Analysis. The degradation patterns among baselines further contextualize SpatialEvo’s advantage. SpatialLadder and ViLaSR collapse on V-STAR to approximately 36, far below the baseline of 74.9 (3B) and 78.5 (7B), indicating that their training pipelines introduce distributional shifts that impair spatio-temporal reasoning. SpaceR similarly drops to 29.1 on CoreCognition (3B, baseline 56.8), suggesting that fixed reward functions tailored to specific spatial tasks can suppress broader cognitive capabilities. SpatialEvo achieves the most favorable balance between spatial specialization and general capability preservation across all evaluated configurations.

Variant VSI-Bench RealWorldQA EmbSpatial SpatialViz STARE CoreCognition ViewSpatial V-STAR MMStar Avg ΔAvg SpatialEvo (Ours) 46.1 66.7 66.0 28.6 41.3 60.2 43.2 78.0 62.5 54.7 – Architecture Design

w/o Questioner 40.2 67.1 65.7 27.7 39.0 60.5 40.4 77.0 59.9 53.1 ↓ 1.6 w/o Solver 36.6 70.2 61.8 26.1 39.8 58.5 34.5 75.4 60.5 51.5 ↓ 3.2 w/o Physical Grounding 18.8 68.5 63.5 23.4 39.7 59.7 35.4 77.0 60.6 49.6 ↓ 5.1 w/o Adaptive Scheduler 43.4 68.5 68.0 27.5 39.5 60.3 43.2 77.0 62.4 54.4 ↓ 0.3

Questioner Reward

w/o Validity Reward 41.2 69.7 64.3 29.3 39.2 60.7 41.3 76.4 62.6 53.9 ↓ 0.8 w/o Observation Reward 43.6 69.2 68.1 28.0 41.1 59.1 41.8 78.5 61.0 54.5 ↓ 0.2

Solver Reward w/o Explanation Reward 42.9 65.1 65.9 30.8 39.5 59.9 40.9 81.7 62.3 54.3 ↓ 0.4

- Table 2 | Ablation study on SpatialEvo (Qwen2.5-VL-7B). ΔAvg denotes performance drop relative to the full model. Bold denotes the best Avg.

- 4.3. Ablation Studies

To systematically validate the contribution of each core component in SpatialEvo, we conduct ablation experiments across three design dimensions; results are summarized in Table 2.

Architectural Design. Removing the Questioner (w/o Questioner) degrades average performance to 53.1, with notable drops on VSI-Bench (40.2) and ViewSpatial (40.4), as online self-play is replaced by offline data. Removing the Solver (w/o Solver) produces a larger drop to 51.5, with VSI-Bench declining sharply to 36.6, confirming that online geometric derivation is essential for internalizing spatial reasoning. Replacing DGE ground truth with majority-voting pseudo-GT (w/o Physical Grounding) yields the largest degradation, with the average collapsing to 49.6 and VSI-Bench plummeting to 18.8; the majority-voting mechanism consolidates systematic prediction bias into pseudo-ground-truth, fundamentally corrupting the training signal for geometry-intensive tasks. Additional details are in Appendix D.3. Removing the adaptive task scheduler (w/o Adaptive Scheduler) results in a marginal drop to 54.4, confirming that accuracydriven dynamic allocation outperforms uniform random sampling.

Questioner Reward. Removing the geometric validity reward (w/o Validity Reward) degrades performance to 53.9, with VSI-Bench declining to 41.2, as the proportion of physically invalid questions generated by the Questioner increases substantially, reducing the volume of valid questions passed to the Solver and degrading overall training signal quality. Removing the visual observation reward (w/o Observation Reward) leads to a smaller but consistent drop to 54.5, as the Questioner bypasses holistic scene perception and generates questions without sufficient visual grounding, resulting in shallower reasoning depth in the Solver.

Solver Reward. Removing the explanation reward for invalid questions (w/o Explanation Reward) causes performance to decline to 54.3. The effect is most visible on spatially complex benchmarks such as VSI-Bench (42.9 vs. 46.1) and ViewSpatial (40.9 vs. 43.2), as the geometric constraint information carried by invalid questions is entirely discarded, depriving the Solver of the opportunity to learn task rules from the Questioner’s errors and validating the design choice of converting invalid questions into effective learning signal.

- 5. Analysis and Discussion

- 5.1. Online Evolution vs. Static Learning

To evaluate the practical gains of online self-evolution over static data training, we conduct a controlled comparison along two dimensions: reinforcement learning and supervised fine-

Numerical Question Multiple-Choice Question

Paradigm Method

Avg. Obj. Count Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Qwen2.5-VL-3B (Base) 33.5 21.1 17.9 22.6 32.8 42.7 29.9 21.0 28.0

w/ SpatialLadder RL 62.8 29.8 59.2 27.8 38.9 44.2 33.5 24.9 40.1 SpatialEvo (Online RL) 65.2 35.1 61.3 51.4 46.2 44.3 29.9 26.5 46.3

RL

w/ SpatialLadder Data 63.0 31.9 61.2 43.0 43.0 43.6 32.5 31.7 43.7 w/ SpaceR Data 28.3 25.9 36.3 34.9 35.1 46.8 35.1 43.4 36.3 w/ SpatialSSRL Data 35.6 24.6 15.4 21.8 34.1 39.2 28.9 23.5 28.1 w/ SpatialEvo Offline Data 62.6 28.9 60.0 49.1 40.3 45.7 31.4 25.9 43.9

SFT

- Table 3 | Comparison of online self-evolution and static learning paradigms on VSIBench. SpatialEvo outperforms static RL and SFT baselines across both training paradigms.

tuning. We adopt Qwen2.5-VL-3B as the backbone throughout. To align with the data sources of SpatialLadder (26K) and SpaceR (151K), SpatialEvo is restricted to ScanNet scene assets with six core task categories: object counting, object size, room size, absolute distance, relative distance, and relative direction. It is worth noting that SpatialLadder and SpaceR additionally include the Appearance Order task, giving them broader task coverage than the SpatialEvo configuration used here. SpatialSSRL (81K) is included in the SFT comparison not for task coverage alignment, but to assess whether its self-supervised training signal generalizes sufficiently to VSI-Bench’s spatial task categories.

For the RL comparison, we select SpatialLadder as the reference given its closest task coverage to our configuration, using identical GRPO training settings. For the SFT comparison, samples produced offline during SpatialEvo’s online training are curated for supervised finetuning, compared against SFT on the three static datasets. SpatialEvo’s online training is continued until the number of recorded unique QA pairs reaches approximately 20K. Full training and data details are provided in Appendix C.

Results are shown in Table 3. Despite operating under a narrower task scope, SpatialEvo surpasses the SpatialLadder RL baseline and outperforms all static dataset SFT counterparts, achieving the highest average of 46.3 and 43.9 respectively. This outcome reveals the core advantage of online self-evolution: the training distribution of a static dataset is frozen at generation time, whereas SpatialEvo continuously aligns the distribution of training samples to the solver’s current cognitive frontier through real-time interaction between the Questioner and the DGE, realizing adaptive hard-sample mining that static datasets cannot replicate.

###### 5.2. Curriculum Emergence Analysis

Effect of the Adaptive Scheduler across iterations. Table 4 compares SpatialEvo with and without the Adaptive Scheduler across four iterative self-evolution rounds on VSI-Bench. Without the scheduler, the model performs comparably in early iterations (44.2 at Iter 1, 44.5 at Iter 2) but subsequently stagnates and declines, reaching only 43.4 at Iter 4, as uniform task sampling fails to concentrate training resources on categories where the model remains weak. In contrast, the full SpatialEvo with the Adaptive Scheduler exhibits monotonically increasing average performance across all four iterations (44.2 → 45.0 → 45.1 → 46.1), with particularly strong late-stage gains on Abs. Dist. (32.8), Rel. Dist. (45.1), and Appr. Order (40.1) at Iter 4, categories that receive increasing sampling weight as the scheduler identifies them as persistent weak spots.

Reward trajectories and curriculum emergence. As shown in Figure 3, both Questioner and Solver rewards improve steadily throughout training. The Validity Reward rises rapidly and

Questioner Reward

Scene-Task Curriculum Dynamics

Solver Reward

0.24

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | |uniform share|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

1.0

1.0

0.22

Rel. Dir. 21.8%

0.8

0.8

Within-SceneSamplingRate

0.20

Rel. Dist. 18.7%

Counting 18.3%

Score/Ratio

0.6

0.6

0.18

Score

16.7%

0.16

0.4

0.4

Abs. Dist. 15.4%

Round2 Reward

Format Score

0.14

Obj. Size 13.4%

Accuracy Reward

Invalid Ratio

0.2

0.2

Room Area 12.5%

0.12

Round1 Reward

Validity Reward

Observation Reward

Format Reward

0.0

0.0

0.10

0 500 1000 1500 2000

0 500 1000 1500 2000

0 500 1000 1500 2000

Training Step

Training Step

Training Step

Figure 3 | Training dynamics of SpatialEvo. Questioner rewards stabilize as valid question generation is learned (Left); Solver accuracy improves while the invalid ratio declines (Middle); the Adaptive Scheduler up-weights harder categories (Rel. Dir., Rel. Dist.) and down-weights easier ones (Right).

Numerical Question Multiple-Choice Question

Method

Avg.

Obj. Count Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order Qwen2.5-VL-7B (Base) 30.7 10.2 37.14 43.40 37.2 37.1 32.0 27.8 31.1 SpatialEvo

- Iter 1 60.6 28.9 60.3 42.5 42.7 44.4 32.0 30.7 44.2
- Iter 2 63.3 28.9 62.5 33.3 43.4 43.8 30.9 36.7 45.0
- Iter 3 65.0 26.9 63.4 47.3 43.0 43.4 28.9 32.9 45.1
- Iter 4 62.5 32.8 61.6 35.7 45.1 43.9 27.3 40.1 46.1

SpatialEvo w/o Adaptive Scheduler

- Iter 1 55.9 22.4 61.6 51.8 44.5 43.8 33.0 36.3 44.2
- Iter 2 62.1 28.6 62.0 44.0 41.0 43.3 28.9 34.3 44.5
- Iter 3 57.8 29.3 61.6 35.8 42.3 43.4 32.5 31.9 43.7
- Iter 4 60.2 30.6 58.7 39.1 40.1 43.5 28.4 32.2 43.4

- Table 4 | Effect of the Adaptive Scheduler across iterative self-evolution on VSI-Bench subtasks. Both groups are initialized from the same base model and trained for four iterations; the only difference is whether the Adaptive Scheduler is enabled.

stabilizes near 1.0, indicating that the Questioner quickly learns to generate physically valid questions; the Accuracy Reward of the Solver increases consistently while the Invalid Ratio declines, confirming progressive internalization of geometric reasoning. The Scene-Task Curriculum Dynamics plot reveals how this improvement is shaped: sampling rates diverge from the uniform share (16.7%) as training progresses, with harder categories such as Rel. Dir. (21.8%) and Rel. Dist. (18.7%) being up-weighted and easier categories such as Room Area (12.5%) and Obj. Size (13.4%) being down-weighted. This endogenously emergent curriculum, driven entirely by historical accuracy without manual design, explains the sustained performance improvement observed in Table 4.

##### 6. Conclusions

We present SpatialEvo, the first framework to introduce the self-evolving paradigm into 3D spatial reasoning. Unlike conventional self-evolving methods that rely on model voting to construct pseudo-labels, SpatialEvo exploits the unique property of spatial reasoning that visual inputs inherently carry physical information such as 3D point clouds and camera poses, enabling programmatic computation of exact ground truth. This transforms unannotated 3D scene assets into zero-noise reward judges, replacing model consensus with deterministic physical feedback. A single policy model co-evolves as Questioner and Solver under DGE constraints, with a task

scheduler enabling adaptive curriculum self-emergence. Experiments validate significant gains across multiple spatial reasoning benchmarks.

We hope the physically grounded self-evolution paradigm explored by SpatialEvo can serve as a reference for broader embodied intelligence research. When self-exploration is rooted in a verifiable physical environment, the continual emergence of spatial reasoning capability is driven endogenously by the model’s interaction with the objective world, rather than costly human annotation.

##### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse realworld dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.

Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 9490–9498. IEEE, 2025a.

Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, et al. Scaling spatial intelligence with multimodal foundation models. arXiv preprint arXiv:2511.13719, 2025b.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024b.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.

Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495,

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

Shijian Deng, Kai Wang, Tianyu Yang, Harsh Singh, and Yapeng Tian. Self-improvement in multimodal large language models: A survey. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1987–2006, 2025.

Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension. Advances in Neural Information Processing Systems, 37:131369–131397, 2024.

Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 346–355, 2024.

Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. Visplay: Selfevolving vision-language models from images. arXiv preprint arXiv:2511.15661, 2025.

Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multiperspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025a.

Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025b.

Linjie Li, Mahtab Bigverdi, Jiawei Gu, Zixian Ma, Yinuo Yang, Ziang Li, Yejin Choi, and Ranjay Krishna. Unfolding spatial cognition: Evaluating multimodal models on visual simulations. arXiv preprint arXiv:2506.04633, 2025c.

Yijiang Li, Qingying Gao, Tianwei Zhao, Bingyang Wang, Haoran Sun, Haiyun Lyu, Robert D Hawkins, Nuno Vasconcelos, Tal Golan, Dezhi Luo, et al. Core knowledge deficits in multimodal language models. arXiv preprint arXiv:2410.10855, 2024.

Zechuan Li, Hongshan Yu, Yihao Ding, Yan Li, Yong He, and Naveed Akhtar. Embodied intelligence for 3d understanding: A survey on 3d scene question answering. Information Fusion, page 103624, 2025d.

Zejun Li, Yingxiu Zhao, Jiwen Zhang, Siyuan Wang, Yang Yao, Runzhou Zhao, Jun Song, Bo Zheng, and Zhongyu Wei. Mixture-of-visual-thoughts: Exploring context-adaptive reasoning mode selection for general visual reasoning. arXiv preprint arXiv:2509.22746, 2025e.

Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding vision-language model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025f.

Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang, et al. Mm-zero: Self-evolving multi-model vision language models from zero data. arXiv preprint arXiv:2603.09206, 2026.

Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, and Junxian He. Diving into selfevolving training for multimodal reasoning. arXiv preprint arXiv:2412.17451, 2024.

Yang Liu, Ming Ma, Xiaomin Yu, Pengxiang Ding, Han Zhao, Mingyang Sun, Siteng Huang, and Donglin Wang. Ssr: Enhancing depth perception in vision-language models via rationaleguided spatial reasoning. arXiv preprint arXiv:2505.12448, 2025a.

Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-ssrl: Enhancing spatial understanding via selfsupervised reinforcement learning. arXiv preprint arXiv:2510.27606, 2025b.

Shixian Luo, Zezhou Zhu, Yu Yuan, Yuncheng Yang, Lianlei Shan, and Yong Wu. Geogrambench: Benchmarking the geometric program reasoning in modern llms. arXiv preprint arXiv:2505.17653, 2025.

Ruiyuan Lyu, Jingli Lin, Tai Wang, Shuai Yang, Xiaohan Mao, Yilun Chen, Runsen Xu, Haifeng Huang, Chenming Zhu, Dahua Lin, et al. Mmscan: A multi-modal 3d scene dataset with hierarchical grounded language annotations. Advances in Neural Information Processing Systems, 37:50898–50924, 2024.

Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. Advances in neural information processing systems, 37:68803–68832, 2024a.

Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025.

Xianzheng Ma, Brandon Smart, Yash Bhalgat, Shuai Chen, Xinghui Li, Jian Ding, Jindong Gu, Dave Zhenyu Chen, Songyou Peng, Jia-Wang Bian, et al. When llms step into the 3d world: A survey and meta-analysis of 3d tasks via multi-modal large language models. arXiv preprint arXiv:2405.10255, 2024b.

Julius Mayer, Mohamad Ballout, Serwan Jassim, Farbod Nosrat Nezami, and Elia Bruni. ivispar—an interactive visual-spatial reasoning benchmark for vlms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26745–26769, 2025.

Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. Continual learning of large language models: A comprehensive survey. ACM Computing Surveys, 58(5):1–42, 2025a.

Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025b.

Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780,

Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.

Omkar Thawakar, Shravan Venkatraman, Ritesh Thawkar, Abdelrahman Shaker, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, and Fahad Khan. Evolmm: Self-evolving large multimodal models with continuous rewards. arXiv preprint arXiv:2511.16672, 2025.

Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, and Zhengzhong Tu. Nuscenes-spatialqa: A spatial understanding and reasoning benchmark for vision-language models in autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4567–4576, 2025.

Han Wang, Yi Yang, Jingyuan Hu, Minfeng Zhu, and Wei Chen. V-zero: Self-improving multimodal reasoning with zero annotation. arXiv preprint arXiv:2601.10094, 2026.

Peidong Wang, Ming Wang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang, and Kaisong Song. Language models as continuous self-evolving data engineers. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18108–18127, 2025a.

Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, and Wentian Zhao. Vision-zero: Scalable vlm self-improvement via strategic gamified self-play. arXiv preprint arXiv:2509.25541, 2025b.

Siting Wang, Minnan Pei, Luoyang Sun, Cheng Deng, Yuchen Li, Kun Shao, Zheng Tian, Haifeng Zhang, and Jun Wang. Spatialviz-bench: A cognitively-grounded benchmark for diagnosing spatial visualization in mllms. In The Fourteenth International Conference on Learning Representations.

Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24669–24679, 2025c.

Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025.

Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari.

Continual learning for large language models: A survey. arXiv preprint arXiv:2402.01364, 2024. Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, and Yunjian Zhang. Spatialbench: Benchmarking

multimodal large language models for spatial cognition. arXiv preprint arXiv:2511.21471, 2025.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025a.

Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025b.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025c.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935– 110971, 2024.

Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.

Yuanhong Zheng, Ruichuan An, Xiaopeng Lin, Yuxing Liu, Sihan Yang, Huanyu Zhang, Haodong Li, Qintong Zhang, Renrui Zhang, Guopeng Li, et al. Pearl: Personalized streaming video understanding model. arXiv preprint arXiv:2603.20422, 2026.

Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8600–8612, 2025.

##### A. Limitations

Although SpatialEvo demonstrates substantial improvements in spatial reasoning, several limitations merit explicit discussion.

Dependency on high-fidelity 3D assets. The framework’s core reliance on the Deterministic Geometric Environment (DGE) confines its applicability to scenes equipped with complete 3D assets. Specifically, SpatialEvo requires high-quality indoor point cloud reconstructions, calibrated camera pose parameters, and comprehensive scene coverage, which currently restricts its use to static indoor environments such as those in the ScanNet dataset family. In outdoor or dynamic settings, geometric consistency is difficult to guarantee due to sparse point clouds, complex scale variation, or moving objects, thereby undermining the reliability of ground-truth computation.

Sensitivity to entity parsing quality. The question parsing stage of the DGE pipeline relies on a language model to extract structured entities from free-form natural language. When questions contain ambiguous references or underspecified targets, parsing errors may arise and propagate into subsequent verification and computation stages, introducing a source of noise that deterministic geometric reasoning alone cannot fully mitigate.

Sensitivity to point cloud quality. The DGE’s geometric ground-truth computation is inherently sensitive to the fidelity of the underlying point clouds. Reconstruction artifacts, point sparsity, and occlusions can degrade the precision of geometric operators such as bounding box fitting and depth estimation, leading to approximation errors in continuous-valued tasks (e.g., absolute distance and object size estimation). Although relative-error tolerance bands are introduced at the reward stage to partially absorb reconstruction noise, their effectiveness remains fundamentally bounded by the quality of the underlying data.

These limitations largely stem from the reliance on explicit 3D representations. Future work may explore reducing dependence on point clouds via alternative or implicit spatial representations, as well as on-demand geometry construction, to improve scalability and generalization.

##### B. Implementation Details of the DGE

This appendix provides a more granular formalization of the methodology introduced in Section 3.2, with the aim of enabling full reproducibility of the DGE’s decision logic, task scheduling mechanism, and reward definitions. We proceed as follows: complete task-specific validity rules (Appendix B.1), and the automated verification pipeline and entity extraction prompts (Appendix B.2).

- B.1. Complete Task-Specific Validation Rules

- B.1.1. Unified Validation Decomposition

For any question 𝑄, task type 𝑡, and scene context 𝑥, the DGE acceptance condition is expressed as

Valid(𝑄,𝑡, 𝑥) = I Cmode ∧ Cextract ∧ Cpool ∧ Cschema ∧ Csolver , (2)

where the five conditions are defined as follows. Cmode requires that the input modality be consistent with the task type, i.e., the input contract for scene-level, single-image, or image-pair

inputs is satisfied. Cextract requires that structured entity extraction succeed with all task-required

fields populated. Cpool requires that all extracted labels fall within the grounded candidate pool that is both observable for the current sample and permitted by the task definition. Cschema requires that task-level structural constraints be satisfied, including non-empty candidate lists, no repetition across distinct roles, and the query target not appearing within its own candidate set. Csolver requires that the downstream deterministic geometry operator be executable, meaning all required metadata, camera parameters, object anchors, or detection results are available.

For conciseness, we define the following grounded label pools:

Uscene = {ℓ | 𝑛scene(ℓ) = 1}, Cscene = {ℓ | 𝑛scene(ℓ) ≥ 2}, (3)

where 𝑛scene(ℓ) is the instance count of scene-level object label ℓ. For single-image and image-pair tasks, we define over the per-frame visible object set:

U𝑓 (𝑣min) = {ℓ | 𝑛𝑓 (ℓ; 𝑣min) = 1}, (4)

where 𝑛𝑓 (ℓ; 𝑣min) denotes the number of visible instances of label ℓ in frame 𝑓 under visibility threshold 𝑣min. All frame-level visibility checks satisfy 𝑣min ≥ 0.1; when a specific tool supplies a stricter threshold, the larger value is applied.

In addition to the task-specific rules tabulated below, the DGE uniformly enforces three categories of structural constraints: (i) same-label constraints, wherein measurement pairs or reference-target pairs must not refer to the same object; (ii) distinct-field-group constraints, wherein entities assigned to different roles in multi-role orientation tasks must be mutually distinct; and (iii) list constraints, wherein candidate lists must be non-empty, deduplicated, and must not contain the query target itself.

###### B.1.2. Comprehensive Task Introduction and Validation Rules

Validation rules and detailed descriptions for scene-level spatial reasoning tasks are provided in a unified manner. Each task is characterized by its semantic objective, representative question form, grounding scope, and core validity constraints to ensure well-posedness, while the deterministic solver specifies how ground-truth answers are computed within the DGE framework for consistent and verifiable evaluation. Detailed definitions and examples are summarized in Tables 5, 6, and 7.

###### B.1.3. Remarks on Degeneracy Filtering

It is important to distinguish between the DGE’s question validity judgment and the numerical tolerance applied during the Questioner reward stage. The former determines whether a question is deterministically solvable; the latter determines whether the model’s response is sufficiently close to the DGE-computed ground truth. Accordingly, for continuous-valued tasks such as absolute distance, object size, and room size estimation, the DGE always outputs an exact geometric ground truth, while the relative error thresholds used at evaluation are defined separately in Appendix C.2.2.

Several degenerate question types are additionally suppressed by design. In object counting, categories with zero instances are excluded from the candidate pool; categories with exactly one instance yield questions of diminished discriminative value and are assigned a reduced validity weight rather than being fully rejected. For inter-camera elevation, degenerate same-level frame pairs are not actively sampled during training to avoid allocating budget to examples with negligible vertical discrimination. For depth ordering and visibility comparison, same outcomes are retained but down-weighted in the Questioner reward, preventing the curriculum from over-concentrating on geometrically trivial edge cases.

- Table 5 | Validation rules and detailed descriptions for scene-level spatial reasoning tasks.

Task Description Example Question Grounding Pool

Core Constraints Deterministic

Solver Object Counting Count the number

“How many windows are there in this room?”

Cscene Target category must be countable and scene-wide

Count instances of the label

of instances of an object category in the scene

Object Size Estimate the longest geometric dimension of a unique target

“What is the longest edge of this television in centimeters?”

Uscene Target must be

Read the longest bounding box edge

unique

Uscene Both targets unique

Compute nearest-point distance

Absolute Distance

Measure the metric distance between the nearest points of two objects

“What is the straight-line distance between the fireplace and the TV at their nearest points, in meters?”

and distinct

Relative Distance

Identify the candidate closest to the anchor among a candidate set

“Among the chair, coat rack, and table, which is closest to the bed at nearest points?”

Uscene Anchor unique;

Return nearest object label

candidate set valid

Relative Direction

Standing at one object and facing another, determine the direction of a third

“If I stand near the toilet and face the sink, in which direction is the bathtub relative to me?”

Uscene All three roles unique and mutually distinct

Output quadrant direction label

Room Size Estimation

Estimate the overall floor area of the room

N/A Scene-level area metadata required

Read room area from metadata

“Approximately how many square meters is this room?”

- B.2. Automated Verification Pipeline and Extraction Prompts

- B.2.1. End-to-End Verification Pipeline

Given input (𝑄,𝑡, 𝑥), the automated verification pipeline executes the following sequential stages.

###### Stage 1: Task normalization and input contract checking. The noisy task name is parsed

to a canonical task type, and the input is verified to satisfy the modality contract—scene-level, single-image, or image-pair—required by the current task.

###### Stage 2: Context injection. Scene metadata, frame metadata, camera poses, and visible

object statistics are automatically populated from the scene identifier, image paths, and frame identifiers associated with the current sample.

###### Stage 3: Structured extraction. For tasks requiring language parsing, a lightweight language

model extracts objects, candidate sets, region phrases, or image indices according to a taskspecific schema. The extracted output is a structured parameter vector indexed to the task’s expected fields. If certain fields remain null after extraction, a heuristic fallback attempts to resolve them from the grounded candidate set before proceeding.

###### Stage 4: Canonicalization and sanitization. The extracted parameter vector undergoes

lowercasing, label normalization against the scene’s grounded object pool, candidate pool constraint enforcement, and heuristic completion. Region phrases additionally pass through an ontology-based anchor resolution step.

###### Stage 5: Rule-based rejection. If any extracted field is empty, any label falls outside the

- Table 6 | Validation rules and detailed descriptions for single-image tasks.

Task Description Example Question Grounding Pool

Core Constraints Deterministic Solver

Single-View Relative Direction

Determine the direction of a target relative to a reference object within a single frame

U𝑓 (𝑣min) Both targets uniquely visible and distinct

“In this image, in which direction is the sink relative to the washing machine?”

Output relative direction

Camera-Object Distance

Estimate the distance from the camera to the target object

“In this image, what is the distance from the camera to the bed in meters?”

U𝑓 (𝑣min) Target uniquely

Compute camera-to-object distance

visible

“Which is closer to the camera, the chair or the trash can?”

Depth Ordering Compare which of two objects is closer to the camera

U𝑓 (𝑣min) Both targets uniquely visible with valid depth

Return nearer object or same

permitted grounded pool, roles conflict, or candidate lists are structurally invalid, the DGE immediately returns a rejection decision together with a task-specific invalidity reason, without invoking any downstream geometric operator.

Stage 6: Deterministic rubric execution. For questions passing all prior checks, the DGE invokes the geometry operators prescribed by the task’s rubric in sequence and synthesizes the final ground-truth answer. The pipeline additionally records intermediate geometric statesincluding key point coordinates, relative pose matrices, and depth projections—to support downstream qualitative analysis and error attribution.

###### B.2.2. Prompted Entity Extraction

Rather than relying on a single free-form parser, the DGE specifies an explicit extraction schema for each task and instructs the extraction model to output only the structured fields. Scene-level and single-image tasks share the following common prompt prefix. And image-pair tasks use a more camera-relation-oriented prefix.

###### Entity Extraction Prompt – Scene and Single-Image Tasks

Role: Expert Entity Extractor for 3D Scenes. Instruction: Extract objects from the question exactly as they appear. Rules:

- - Use lowercase and singular forms.
- - Keep original spelling; do not paraphrase or replace with synonyms.
- - If missing or uncertain, output null.
- - Output only the extraction result in the required format.

Entity Extraction Prompt – Image-Pair Tasks

Role: Expert Entity Extractor for spatial reasoning. Instruction: Extract entities exactly as asked in the question. Rules:

- - Output lowercase.
- - Keep nouns only.
- - If uncertain, output null.
- - Output only the required format.

- Table 7 | Validation rules and detailed descriptions for image-pair tasks.

Task Description Example Question Grounding Pool

Core Constraints Deterministic Solver

Inter-Camera Relative Position

Determine the relative position of Image 2’s camera in Image 1’s camera coordinate frame

N/A Both frame poses parseable

Output camera relative direction

“When you took Image 1, where is Image 2’s camera relative to you?”

Inter-Camera Elevation

Determine the vertical relationship between two camera viewpoints

“Compared to Image 2’s camera, is Image 1’s camera higher or lower?”

N/A Both frame poses parseable

Output higher/lower

Visibility Comparison

Compare target visibility across two images

“For the cabinet, which image shows it more clearly, Image 1 or Image 2?”

Pair-visible object pool

Target grounded without ambiguity

Return image1/image2/same/neither

Camera-Object Position

Determine the direction of a target object relative to the reference camera

“When I took Image 1, in which direction is the table relative to me?”

Ref-frame unique object pool

Target uniquely grounded in reference image

Output camera-centric direction

Output camera-centric direction

Ref-frame region anchor pool

Region phrase resolvable to an anchor

Camera-Region Position

Determine the direction of a region anchor relative to the reference camera

“When you took Image 1, in which direction is the dining area relative to you?”

“Based on this image sequence, in which direction is the camera turning?”

N/A Both frame poses and trajectory parseable

Output dominant motion direction

Camera Motion Estimation

Determine the dominant motion direction of the camera between two frames

Both targets distinct and uniquely bindable

“Which has the larger longest edge, the laptop or the oven?”

Pair nonambiguous object pool

Attribute Measurement

Compare the longest dimension of two objects

Return the label of the larger object

Each task appends its own schema specification on top of this prefix. Some representative templates are provided below.

###### Extraction Schema – Object Counting

Task: Extract the object category being counted. Important:

- - Extract only the queried category.
- - Use lowercase.
- - If missing or uncertain, output null. Format: category_name Question: {question} Extraction:

###### Extraction Schema – Single-View Relative Direction

Task: Extract two objects for relative direction in one image. Important:

- - The answer means the target object relative to the reference object.
- - Output the reference object first and the target object second.
- - If either object is missing or uncertain, output null for that slot. Format: reference_object, target_object Question: {question} Extraction:

For image-pair spatial questions, we separate camera reference parsing from semantic entity extraction. The observer image index is first inferred from the question (e.g., “when taking Image 1”) and stored as a camera-reference variable. Conditioned on this, the extraction module only identifies the queried object or region, reducing ambiguity and yielding simpler, more robust prompts across diverse camera-centered questions.

###### Extraction Schema – Camera-Object Position

Task: Extract the target object phrase relative to the camera. Important:

- - The observer image index (Image 1 or Image 2) is resolved separately and should NOT be extracted here.
- - Extract only the queried object phrase.
- - Keep the object phrase concise and noun-based.
- - If the target object cannot be determined from the question, output null. Format: target_object Question: {question} Extraction:

Extraction Schema – Camera-Region Position

Task: Extract the target region phrase relative to the camera. Important:

- - The observer image index (Image 1 or Image 2) is resolved separately and should NOT be extracted here.
- - If the question refers to an abstract region such as a sleeping area, bathroom area, kitchen area, or living area, keep that phrase verbatim.
- - Do NOT replace the region phrase with an object label.
- - Extract only the queried region phrase.
- - If the target region cannot be determined from the question, output null. Format: region_name Question: {question} Extraction:

Not all tasks require LLM-based extraction. Room size estimation, inter-camera relative position, inter-camera elevation, and camera motion estimation are fully determined by scene or camera pose parameters and therefore skip the extraction stage entirely.

###### B.2.3. Questioner Prompt Templates Used in Round 1

The Questioner in Round 1 employs three task-conditioned prompt templates corresponding to scene-level, single-image, and image-pair input modes. Curly-brace placeholders denote runtime-populated content. The three templates share the same structural organization but differ in their grounding scope: the scene-level template mandates whole-scene evidence; the singleimage template restricts all grounding to the current frame’s visible content; and the image-pair

template emphasizes cross-frame comparison and requires explicit Image 1 or Image 2 references when the task focuses on a specific viewpoint.

###### Questioner Prompt – Scene-Level Template

# SYSTEM ROLE: You are an expert 3D scene analyst and question designer. Assigned task type: {task_display_name}.

# TASK GOAL: {task_goal}

# AVAILABLE CONTEXT: {task_context_sections}

# TASK-SPECIFIC VALIDITY GUIDE: {task_candidate_guidance}

# YOUR JOB:

- 1. Write one grounded whole-scene observation, then exactly one scene-level question for the assigned task.
- 2. Make the observation flow from global scene layout to the local target so it naturally supports the question.
- 3. Keep the observation detailed and spatially grounded rather than short or list-like.
- 4. Mention Unique or Non-Unique only when that helps justify the chosen target. # HARD CONSTRAINTS:

- - Use whole-scene evidence, not a single sampled frame.
- - Copy object labels exactly from the provided label list.
- - For tasks requiring unique objects, use labels marked (Unique).
- - Avoid generic, list-only, or weakly grounded observations.

# OUTPUT FORMAT: <observation>...</observation> <question>...</question>

###### Questioner Prompt – Single-Image Template

# SYSTEM ROLE: You are an expert single-image spatial reasoning question designer. Assigned task type: {task_display_name}.

# TASK GOAL: {task_goal}

# AVAILABLE CONTEXT: {task_context_sections}

# TASK-SPECIFIC VALIDITY GUIDE: {task_candidate_guidance}

# YOUR JOB:

- 1. Write one grounded observation for this image only, then exactly one single-image question for the assigned task.
- 2. Make the observation flow from overall image layout to the local target.
- 3. Use only relations visible in this image; keep the observation detailed rather than list-like.
- 4. Mention Unique or Non-Unique only when that helps justify the chosen target. # HARD CONSTRAINTS:

- - Use only evidence from the current image.
- - Copy object labels exactly from the provided label list.
- - Prefer (Unique) labels when ambiguity would otherwise be high.
- - Avoid generic, list-only, or cross-image reasoning.

# OUTPUT FORMAT: <observation>...</observation> <question>...</question>

###### Questioner Prompt – Image-Pair Template

# SYSTEM ROLE: You are an expert two-image spatial reasoning question designer. Assigned task type: {task_display_name}.

# TASK GOAL: {task_goal}

# AVAILABLE CONTEXT: {task_context_sections}

# TASK-SPECIFIC VALIDITY GUIDE: {task_candidate_guidance}

# YOUR JOB:

- 1. Write one grounded comparative observation for the image pair, then exactly one image-pair question for the assigned task.
- 2. Make the observation flow from pair-level relation to the local target.
- 3. If the task focuses on one image, explicitly say Image 1 or Image 2.
- 4. Mention Unique, Non-Unique, or shared visibility only when that helps justify the chosen target. # HARD CONSTRAINTS:

- - Use only evidence grounded in the provided image pair.
- - Copy object labels exactly from the provided label list.
- - Explicitly state Image 1 or Image 2 when the task requires a concrete image reference.
- - Avoid generic, weakly grounded, or repetitive observations. # OUTPUT FORMAT: <observation>...</observation> <question>...</question>

###### B.2.4. Candidate-Pool Sanitization and Heuristic Fallback

Before passing extracted results to the solver, two normalization layers are applied. The first is lexical sanitization, which includes case normalization, compact matching for compound labels (e.g., mapping nightstand to night stand), phrase-level matching, and word-level mapping after stopword removal. The second is pool-aware sanitization: if an extracted label does not appear in the grounded pool for the current sample, the system first attempts to remap it to the nearest canonical label; if this fails, the field is set to null within the heuristic candidate set, triggering downstream rejection.

For camera-region position tasks, the DGE maintains a region-object ontology. When a question supplies an abstract region phrase such as sleeping area, the system does not immediately reject the question; it instead retains the phrase and resolves it to a spatially grounded anchor object via the ontology, the available anchor candidate list, and, when necessary, an auxiliary language model call. The question is rejected only if this resolution ultimately fails.

###### B.2.5. Unified Response Object

Once a question passes the full pipeline, the DGE returns a unified structured response object containing at minimum: a validity flag, a deterministic supervision signal, structured parsing results, and a failure summary including the error code, failure stage, and validation result.

For invalid questions, the system additionally constructs a diagnostic summary used by the explanation judge in Round 2. This unified interface enables the DGE to simultaneously fulfill two responsibilities: providing noise-free geometric ground truth for valid questions, and providing task-semantically interpretable rejection evidence for invalid ones.

##### C. Algorithmic Details of Spatial-Grounded Policy Co-Evolution

This appendix provides a detailed algorithmic exposition of the Spatial-Grounded Policy CoEvolution mechanism introduced in Section 3.3, with the aim of ensuring full reproducibility of the co-evolution training procedure. We proceed as follows: the task-adaptive scheduling formulation (Appendix C.1), and detailed reward specifications for both the Questioner and the Solver (Appendices C.2 and C.3).

- C.1. Task-Adaptive Scheduling

- C.1.1. Feasible-Task Inference

The feasible task set T𝑠feasible is not manually specified but is inferred online from the DGE environment summary Σ𝑠 of the current sample:

T𝑠feasible = {𝑘 ∈ T | 𝜙𝑘(Σ𝑠) = 1}, (5)

where 𝜙𝑘 is the feasibility predicate for task 𝑘. For scene-level tasks: object counting is feasible if at least one non-unique label exists; absolute distance requires at least two unique labels; object size requires at least one; room size estimation is always feasible; relative distance requires at least four unique labels; and relative direction requires at least three. For single-image tasks: single-view relative direction and depth ordering each require at least two uniquely visible objects; camera-object distance requires at least one. For image-pair tasks: inter-camera relative position and camera motion estimation are feasible whenever both frame poses are available; inter-camera elevation is feasible only when the height difference is non-degenerate; visibility comparison is feasible if the pair visibility pool is non-empty; camera-object position requires at least one uniquely visible label in the reference frame; camera-region position requires at least one resolvable region anchor; and attribute measurement requires at least two labels in the pair’s non-ambiguous object pool.

- C.1.2. Scheduler-Adjusted Historical Accuracy

The scheduler maintains three statistics for each task 𝑘: historical sample count 𝑁𝑘, cumulative raw accuracy 𝑆𝑘, and a calibrated accuracy sum 𝑆𝑘sched used specifically for scheduling (Chen et al., 2025). The calibrated accuracy applies task-specific difficulty normalization:

𝑎sched𝑘 = clip

𝑎𝑘 𝜏𝑘

, 0, 1 , 𝜏𝑘 =

- 0.50, 𝑘 ∈ Tnum,
- 1.00, otherwise,

(6)

where Tnum denotes the set of all numeric-output tasks. Setting 𝜏𝑘 = 0.5 for numeric tasks reflects the fact that their accuracy is a smooth continuous metric rather than a hard exact-match score (Chen et al., 2024a); a calibrated value exceeding 0.5 thus serves as a natural indicator of adequate mastery.

###### C.1.3. Task Sampling from Smoothed Accuracy

The smoothed accuracy 𝑎¯𝑘 is computed with a pseudo-count prior to mitigate estimation instability during early training:

𝑆𝑘sched + 𝑎0𝑛0 𝑁𝑘 + 𝑛0

, 𝑎0 = 0.35, 𝑛0 = 2.0. (7)

𝑎¯𝑘 =

Task sampling weights are then computed as 𝑤𝑘 = max(𝛿, 1 − 𝑎¯𝑘) with 𝛿 = 0.05, assigning higher weight to tasks with lower historical accuracy while ensuring well-mastered tasks are not entirely excluded. The final sampling probability within the feasible task set is

∑︁

𝑤𝑘

, 𝑘 ∈ T𝑠feasible. (8)

𝑝𝑘 =

𝑤𝑗

𝑗∈T𝑠feasible

###### C.1.4. Deduplication-Aware Statistics

As noted in Section 3.3.5 Round 1 samples multiple candidate questions per visual context. Before entering Round 2, an intra-context deduplication step is applied; deduplication operates strictly within each source context and does not merge questions across distinct scene samples.

Each candidate question is assigned a semantic signature determined jointly by its canonical task type and the structured parameters extracted by the DGE, rather than by the raw question string. Fields such as object pairs, candidate sets, and image references are normalized before signature construction, including lowercasing, canonical image index mapping, and sorting of unordered object pairs. For tasks with explicit image roles such as inter-camera relative position and camera-object position, reference and target image roles are encoded in the signature. Exactly one representative question per unique semantic signature is retained for Round 2. To preserve curriculum consistency despite deduplication, each retained representative carries a duplicate weight equal to its signature’s frequency in the original rollout; the scheduler statistics (𝑁𝑘,𝑆𝑘,𝑆𝑘sched) are updated using this weight rather than the deduplicated question count. If a retained question is later judged invalid before answer evaluation, its scheduler-facing accuracy is set to zero, while its duplicate weight is still counted in the effective statistics.

###### C.2. Questioner-Side Reward

- C.2.1. Final Reward Form The Questioner reward in the full implementation takes the form

𝑟𝑄 = −1, if a severe structural failure occurs, 0.1 𝑓fmt + 0.9 𝑓valid 𝑓obs, otherwise,

(9)

where 𝑓fmt measures adherence to the expected observation-question template, 𝑓valid reflects the DGE’s geometric validity and non-degeneracy assessment, and 𝑓obs is assigned by a lightweight text judge assessing the quality of reasoning in the observation description.

- C.2.2. Validity Factor

𝑓valid functions first as a binary gate: the DGE rejection yields 𝑓valid = 0; a standard valid question yields 𝑓valid = 1. Task-specific downward adjustments are applied to questions that are valid yet

carry limited discriminative value. In object counting, a true count of zero renders the question invalid; a count of exactly one is assigned 𝑓valid = 0.5. In depth ordering, a same outcome yields 𝑓valid = 0.5. In inter-camera elevation, a same-level outcome is assigned 𝑓valid = 0. In visibility comparison, outcomes of same or neither are assigned 𝑓valid = 0.5. This design suppresses the Questioner from repeatedly sampling edge-case questions that, while formally valid, offer limited geometric learning signal.

###### C.2.3. Observation-Based Judgment

Observation quality is scored by a GPT-OSS-120B (Agarwal et al., 2025) text judge that assesses whether the observation constitutes a high-quality visual-to-question transition rather than re-evaluating question correctness. The verbatim prompt is provided below.

Observation Quality Judge Prompt

You are evaluating the quality of a task-conditioned observation written by a 3D spatial reasoning model. ### OBSERVATION: {observation} ### QUESTION TYPE CHOSEN: {q_type_str} ### QUESTION GENERATED: {raw_question} ### CONTEXT: The ideal Observation has a global-to-local flow: (1) summarize the overall scene or image-pair layout; (2) narrow to the local objects, region, or relation most relevant to the task; (3) lead naturally into the exact target of the question; (4) explain Unique/Non-Unique status when useful. The Observation need not describe every visible object. You are NOT grading question correctness—you are grading whether the Observation is a good visualspatial lead-in for the question. Judge holistically. Do NOT apply fixed word-count or clause-count rules. ### YOUR TASK: Evaluate on: (1) Groundedness, (2) Global caption quality, (3) Local focus quality, (4) Transition quality,

(5) Support quality, (6) Sample-specificity. ### SCORING RULES: 1.0 – Clearly grounded, meaningful overall caption, narrows to the task-relevant target, makes the question feel natural, sample-specific. 0.6 – Grounded and useful, but one major component is weaker. 0.3 – Shows real visual effort but is fragmentary, generic, or only loosely connected to the question. 0.0 – Empty, template-like, contradictory, hallucinated, or lacks visual-spatial content. Think step by step. Output ONLY: <score>[1.0 / 0.6 / 0.3 / 0.0]</score>

The judge is invoked only when the question is valid, the format is correct, and the observation is non-empty without inner-tag contamination. To prevent the coupled reward from collapsing due to an excessively conservative judge score, a minimal lower bound is imposed:

𝑓obs = max(0.1, 𝑓obsjudge). For cases with no observation, an invalid question, or a format violation, 𝑓obs is set to 0 directly.

###### C.2.4. Representative High- and Low-Score Observations

We provide two illustrative examples consistent with the scoring criteria, using abstract object names to avoid overlap with main-paper case studies.

###### High-Score Observation Example (score 1.0 or 0.6)

Across the two images, the overall room layout remains stable while the camera shifts slightly toward the right side of the scene. Large furniture still anchors the perimeter, and the open floor in the middle makes the camera change easy to track. Narrowing to Image 2, the chair is isolated near the left side of the current viewpoint with clear free space between the camera and the chair, making it the most natural grounded target for a camera-relative direction question.

###### Low-Score Observation Example (score 0.3 or 0.0)

There is a chair in the room. The question is about the chair.

The high-score example first describes the pair-level global structure, then narrows to the task-relevant local target, and transitions naturally to a camera-relative direction question. The low-score example provides neither a global description nor a justification for target selection, and is essentially a restatement of the target label.

###### C.3. Solver-Side Reward

- C.3.1. Round 2 Format and Reward Decomposition For valid questions, the Solver receives the following prompt:

###### Solver Prompt – Valid Question

Carefully examine the images and answer the following question step by step. Write your reasoning in plain text, keeping each step grounded in the visual evidence. Avoid redundant or repetitive descriptions. Only the final line should contain the answer in <answer>...</answer> format.

Question: {question} Response:

For invalid questions, the Solver generates a structured explanation of the invalidity reason:

###### Solver Prompt – Invalid Question

You are an expert spatial reasoning analyst. The following question has already been judged invalid for the assigned task. Your job is to explain why it is invalid based on the task validity rules below.

Assigned task type: {task_display_name} Question: {question} {task_reference_text}

Explain step by step why this question is invalid. Ground each step in the task validity rules and the simulator-backed diagnostic context. Focus on the main issue, such as: using object labels outside the grounded pool; using an ambiguous or non-unique object; using the wrong number of candidates or roles; reusing the same object in conflicting roles; or asking in a way that does not match the assigned task schema.

Provide the final conclusion inside <answer> tags. Example: <answer>The target object is not uniquely grounded.</answer>

Response:

The Solver reward in Round 2 is

 

−1, if 𝑄 is valid and the response has a hard format failure, 0.1 𝑓fmt + 0.9 𝑓acc, if 𝑄 is valid, 0.1 𝑓fmt + 0.9 𝑓explain, if 𝑄 is invalid,

(10)

𝑟𝐴 =

 

where 𝑓fmt ∈ {−1,0,1} equals 1 if exactly one <answer> tag is present with no other XML/HTMLstyle tags, 0 if the tag is absent, and −1 if multiple tags or other XML/HTML-style tags appear. A hard format failure on a valid question triggers an overall penalty of −1; for invalid-question explanations, the format score enters the reward linearly without this hard penalty.

###### C.3.2. Accuracy Scoring for All 16 Tasks

Rather than approximating ground truth via majority-vote consensus, the implementation scores each sample directly against the deterministic ground truth produced by the DGE. For continuous-valued tasks, accuracy is computed via a relative-error threshold grid (Li et al., 2025b):

∑︁

𝑑rel(𝑦ˆ, 𝑦) = |𝑦ˆ − 𝑦| max(𝑦,𝜀)

1 |Crel|

, 𝐴rel(𝑦ˆ, 𝑦) =

I[𝑑rel(𝑦ˆ, 𝑦) ≤ 1 − 𝑐] , Crel = LinSpace(0.50, 0.95, 11),

𝑐∈Crel

(11) with 𝜀 = 10−9. This constructs 11 equally-spaced confidence points mapped into tolerance thresholds ranging from coarse to strict. Object size, absolute distance, room size estimation, and camera-object distance all use 𝐴rel as 𝑓acc.

Object counting uses a piecewise absolute-error reward: exact match scores 1.0; error ≤ 1 scores 0.3; error ≤ 2 scores 0.1; all other cases score 0. Relative distance is evaluated by exact label match. Relative direction first normalizes direction aliases and decomposes compound directions into set form, then applies set consistency comparison. Single-view relative direction uses direction-set matching with partial-credit rules for subset matches. Depth ordering is a three-class task (object 1, object 2, or same) evaluated by exact match after label normalization. Inter-camera relative position, camera-object position, and camera-region position apply

camera-centric direction-set matching; camera motion estimation uses the same rule, mapping clockwise/counterclockwise to right/left in the normalization step. Inter-camera elevation, visibility comparison, and attribute measurement are all evaluated by exact match after label normalization.

###### C.3.3. Invalid-Question Explanation Scoring

When the DGE rejects a question, the system converts the structured rejection evidence into an explanation-based training signal rather than discarding the sample. Explanation quality is scored with 𝑓explain ∈ {1.0, 0.6, 0.3, 0.0}, where a score of 1.0 requires correctly identifying the DGE’s primary invalidity reason and grounding it in task rules, the object pool, extraction failure, non-unique reference, or a validation constraint; 0.6 indicates the main failure is mostly captured but an important detail is missing; 0.3 applies when only part of the problem is identified; and 0.0 penalizes contradictions or complete failure to explain the invalidity.

The judge receives the model’s explanation alongside authoritative DGE diagnostic signals including the final invalid reason, error code, failure stage, extraction field resolution status, and validation issues, assessing whether the explanation faithfully reflects the DGE’s structured conclusions. The verbatim prompt is as follows.

###### Invalid-Question Explanation Judge Prompt

You are judging whether a model correctly explained why a spatial reasoning question is invalid. The simulator diagnostics below are authoritative deterministic ground truth from the real validation pipeline. Your job: check whether the model explanation captures the simulator’s actual failure reason. Prefer the simulator’s final conclusion over fluent but unsupported explanations.

### ASSIGNED TASK TYPE: {task_type} ### ORIGINAL QUESTION: {question} ### TASK REFERENCE: {task_reference} ### SIMULATOR INVALID SIGNALS: {simulator_reference} ### MODEL EXPLANATION: {explanation}

### SCORING RUBRIC: 1.0 – Correctly identifies the simulator’s main invalid reason and grounds it in task rules, grounded object pool, extraction failure, non-unique reference, or validation issue. 0.6 – Mostly correct on the main failure, but misses an important supporting detail or states it too vaguely. 0.3 – Catches only part of the problem, or gives a generic reason that weakly overlaps with the diagnostics. 0.0 – Wrong, contradicts the simulator diagnostics, or fails to explain the invalidity.

Return ONLY one score: 0 / 0.3 / 0.6 / 1.0

The key property of this mechanism is that the training signal for invalid questions is anchored to the DGE’s structured rejection evidence rather than to inter-model consensus. Even geometrically unsolvable samples therefore provide valuable supervision: they teach the model which questions should not be asked, why, and how to correctly attribute the failure to the appropriate geometric or task constraint.

##### D. Implementation and Experimental Details

This appendix supplements Section 4 with complete implementation details covering the twostage training configuration, the composition of SFT and RL data, and the experimental scope of

the ablation and analysis studies.

- D.1. Training Configuration

- D.1.1. Backbone Model and Training Configuration

Unless otherwise specified, the main experiments evaluate both Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-3B-Instruct as backbone models, while all other experiments default to Qwen2.5VL-7B-Instruct.

The default training paradigm of SpatialEvo is online GRPO reinforcement learning, applied directly without an SFT warm-start. The paradigm comparison in Table 3 additionally includes an SFT baseline for reference; its hyperparameters are reported in Table 8 and remain unchanged from the configuration described therein. The complete hyperparameter configurations for both stages are reported in Tables 8 and 9.

Table 8 | Hyperparameters for the offline SFT warm-start stage.

Hyperparameter Value

global_batch_size 96 gradient_accumulation_steps 6 micro_batch_size 1 bf16 true data_seed 42 gradient_checkpointing true attn_implementation flash_attn lr_scheduler_type cosine warmup_ratio 0.1 learning_rate 2 × 10−5 num_train_epochs 1 max_pixels 802,816 min_pixels 3,136 sequence_parallel true tensor_parallel_size 2

Table 9 | Hyperparameters for the online GRPO reinforcement learning stage.

Hyperparameter Value

num_generations 4 per_device_train_batch_size 1 gradient_accumulation_steps 4 bf16 true data_seed 42 gradient_checkpointing true attn_implementation flash_attn num_train_epochs 4 actor_learning_rate 1 × 10−6 KL coefficient (𝛽) 10−2 temperature 1.0 top-𝑝 1.0 max_pixels 150,528 min_pixels 100,352 tensor_parallel_size 2

The RL stage is implemented using the EasyR1 framework and runs on 1 node × 8 H800 (80GB) GPUs with a two-round architecture: Round 1 handles Questioner-side question generation and Round 2 handles Solver-side answer generation. With 𝑛rollout = 4 maintained across both rounds, each source context yields at most 4 × 4 = 16 candidate question-answer rollout chains in the ideal upper bound without deduplication. The SFT baseline is implemented using the MS-Swift framework and trained on 4 nodes × 8 H800 (80GB) GPUs with sequence parallelism and tensor model parallelism (size = 2) under a full-shard training configuration.

- D.1.2. Auxiliary Language Models

All auxiliary language model calls within the framework—DGE entity and region extraction (Appendix B.2), Round 1 observation quality judging (Appendix C.2), and Round 2 invalidquestion explanation judging (Appendix C.3)—are unified to a single GPT-OSS-120B (Agarwal et al., 2025) text backend. All auxiliary calls are pure text-only operations and introduce no additional vision backbone. This unification controls system complexity, avoids uncertainty from additional visual modules, and ensures all judges and parsers share the same language

prior, reducing system-level heterogeneity.

###### D.2. Online RL Data Composition

Online RL operates over a pre-filtered multi-source visual context pool rather than sampling directly from raw video streams. The pool contains scene-level multi-frame, image-pair, and single-image contexts drawn from the training splits of ScanNet, ScanNet++, and ARKitScenes. Crucially, the policy model receives raw image contexts as input; both question and answer generation are performed online during training.

Quality filtering ensures that scene-level contexts have sufficiently high grounded visible object counts and low zero-visibility ratios; image-pair contexts contain at least three shared visible objects across frames and at least five per frame; and single-image contexts include at least six visible objects. During training, for each video we sample a limited number of contexts per modality (no more than three), preventing excessive data redundancy while maintaining diverse and informative visual inputs. After this process, the final context pool contains 4,365 contexts, with the composition shown in Table 10. The online RL context pool is balanced by modality in proportion to the number of supported task types, yielding an approximate 6:7:3 ratio for scene-level, image-pair, and single-image inputs.

Table 10 | Composition of the online RL context pool by modality and data source.

Input Modality ScanNet ScanNet++ ARKitScenes Total Scene-level 846 171 620 1,637 Image-pair 884 298 728 1,910 Single-image 281 97 440 818 Total 2,011 566 1,788 4,365

###### D.3. Ablation and Analysis Experiment Details

Majority-Vote Pseudo-Label Ablation Unless otherwise specified, all ablation experiments share exactly the same training configuration as the main method; the sole variable is the source of Round 2 supervision.

In the w/o Physical Grounding ablation, we replace DGE-derived ground truth with pseudo labels obtained by voting over the model’s own sampled responses. Following Round 1 deduplication, all Round 2 responses per question are aggregated using task-appropriate strategies: majority vote for discrete and direction-based tasks, integer-rounded plurality for counting, and the median of the largest tolerance-clustered group for continuous values. For invalid-question cases, free-form explanations are mapped to canonical failure labels and the plurality is used. If no consensus is reachable, the pseudo label is left empty and the content reward is set to zero.

The Round 2 reward retains the same format–content decomposition as the main method, with the content term computed against these vote-based pseudo labels instead of geometric ground truth. This ablation thereby isolates the contribution of physically grounded supervision.

Reproduced baselines. To ensure a fair comparison with static-dataset methods in Section 5.1, we reproduce SpatialLadder under a unified training framework, fine-tuned from the same Qwen2.5-VL-3B backbone on the official 26K corpus under identical SFT settings, with all remaining hyperparameters following our configuration. For SpaceR and Spatial-SSRL, we directly evaluate their publicly released checkpoints based on Qwen2.5-VL-3B without further

###### Questioner Case (Step 1116):

###### Reward-0.1 Reward-0.67 Reward-1.0

###### Questioner Case (Step 20):

###### Questioner Case (Step 1937):

[Figure 126]

[Figure 127]

[Figure 128]

<observation> The water bottle is prominently visible in the foreground, positioned roughly in the center-right of the image on the curved portion of the desk......image, placed on the back panel of the curved desk closer to the wall, partially obscured and less visible compared to the water bottle. The desk itself is a light-colored curved workstation with multiple black monitors, keyboards, and various items scattered across its surface. Additional office furniture and equipment are visible in the background on the left side. </observation> <question> In this image, where is the water bottle relative to the computer tower? </question>

<observation>The image shows a room with a table at the center and a backpack placed to the left of the table. Both objects sit in a well-lit area with windows, red carpeting, and a few other desks and chairs in the background. The backpack is relatively smaller in size and positioned closer to the foreground, while the table is larger and situated further back in the same foreground region. The backpack is clearly left of the table as seen from the front.</observation><question>In this image, where is the backpack relative to the table? </question>

<observation>In this image, the chair and the computer tower occupy clearly distinct positions in the frame. The chair is prominent in the foreground, roughly splitting the frame symmetrically, while the computer tower is positioned further back and to the left, separated by the desk and other equipment. Their clear spatial separation makes them an ideal directional pair for this question.</observation> <question>Where is the chair relative to the computer tower?</question>

- Figure 4 | Evolution of the Questioner across training steps on the single-view relative direction task. Observations become increasingly detailed and spatially grounded, with generated questions more tightly coupled to visual evidence.

reproduction. For the GRPO comparison, we train on the SpatialLadder dataset with a rollout multiplicity of 8 to match the original paper, while all remaining hyperparameters follow our method as reported in Table 9.

SpatialEvo controlled configuration. In Section 5.1, SpatialEvo is restricted to the training split of the ScanNet dataset, comprising 992 scenes, and six task categories: object counting, object size, room size estimation, absolute distance, relative distance, and relative direction. All RL hyperparameters (except for the number of training epochs) follow Table 9, with the number of epochs determined dynamically based on data generation. Online RL training runs until approximately 20K non-redundant question-answer pairs have been generated—corresponding to roughly 12 training epochs—which can be tracked using the method described in Section C.1.4. The SFT comparison uses an offline corpus collected from the same online training trajectory, organized into multiple-choice (MCQ) or numerical (NQ) format, consistent with other datasets.

##### E. Supplementary Qualitative Cases

We present qualitative cases illustrating the evolution of the Questioner and Solver throughout training, using the single-view relative direction task as a representative example.

Questioner Evolution. As shown in Figure 4, the Questioner’s output quality improves markedly across training steps. At Step 20 (Reward 0.1), the observation is brief and superficial, providing only coarse positional descriptions without meaningful spatial grounding; the generated question is thus weakly supported by the visual evidence. By Step 1116 (Reward 0.67), the observation becomes more structured, describing object sizes, foreground-background relationships, and relative positions with greater precision. At Step 1937 (Reward 1.0), the observation is detailed and spatially grounded, accurately localizing target objects within the scene and explicitly establishing the geometric relationship that motivates the question. This progression demonstrates that the Questioner progressively internalizes holistic scene perception and learns to generate questions that are tightly coupled to visual evidence.

Solver Evolution. As shown in Figure 5, the Solver’s reasoning process undergoes a parallel improvement. At Step 136 (Reward 0.1), the Solver produces a brief, unstructured description

###### Reward-0.1 Reward-1.0 Reward-1.0

###### Solver Case (Step 136):

###### Solver Case (Step 1116):

###### Solver Case (Step 1860):

[Figure 129]

[Figure 130]

[Figure 131]

In this image, where is the trash can relative to the backpack?

In this image, where is the telephone relative to the chair?

In this image, where is the table relative to the window?

###### GT: Left GT: Right GT: Front-left

To determine the relative position of the telephone to the chair, let's analyze the scene ...... 1. The picture shows an office setting with a desk, a chair, a telephone, and a computer setup. 2. The chair is located on then...... 3. The telephone is positioned on the desk and towards the right side of the chair. 4. From the perspective of the observer, the telephone is closer to the chair and slightly to the right of it. Considering these observations: <answer>right</answer>

To determine the relative position of the trash can to the backpack in the image: 1. **Identify the p...... the left edge of the frame, in the middle of the image, next to a table.......osition of the trash can relative to the position of the backpack on the table, we can conclu......trash can is in a frontal left position relative to the backpack. In the common convention of specifying the vague p.......he trash can relative to the backpack is front-left. <answer>front-left</answer>

To determine the relative position of the table to the window in the image, let's proceed with the following observations: The table is a long rectangular surface in the foreground, equipped with multiple monitors and keyboards. In the background, there is a window on the back wall allowing natural light into the room. Since the table occupies the foreground while the window is positioned on the far back wall, the table is clearly in front of the window. <answer>up-front</answer>

- Figure 5 | Evolution of the Questioner across training steps on the single-view relative direction task. Observations become increasingly detailed and spatially grounded, with generated questions more tightly coupled to visual evidence.

and arrives at an incorrect answer, failing to establish a coherent geometric reasoning chain. At Step 1116 (Reward 1.0), the reasoning becomes multi-step and logically structured, explicitly identifying object positions, inferring spatial relationships, and deriving the correct answer through sequential analysis. At Step 1860 (Reward 1.0), the Solver adopts a clearly enumerated reasoning format, decomposing the spatial inference problem into explicit sub-steps and arriving at a precise answer with consistent geometric justification. These cases collectively demonstrate that DGE-grounded self-evolution drives the internalization of structured spatial reasoning paradigms in both roles.

