## Exploring Spatial Intelligence from a Generative Perspective

Muzhi Zhu1,2∗ Shunyao Jiang1∗ Huanyi Zheng1 Zekai Luo1 Hao Zhong1 Anzhou Li1,2 Kaijun Wang1 Jintao Rong4 Yang Liu1 Hao Chen1† Tao Lin3,2 Chunhua Shen1,2†

1Zhejiang University, State Key Laboratory of CAD & CG 2Ant Group 3Westlake University 4Zhejiang University of Technology

# arXiv:2604.20570v1[cs.CV]22Apr2026

### Abstract

Spatial intelligence is essential for multimodal large language models, yet current benchmarks largely assess it only from an understanding perspective. We ask whether modern generative or unified multimodal models also possess generative spatial intelligence (GSI)—the ability to respect and manipulate 3D spatial constraints during image generation—and whether such capability can be measured or improved. We introduce GSI-Bench, the first benchmark designed to quantify GSI through spatially grounded image editing. It consists of two complementary components: GSI-Real, a high-quality real-world dataset built via a 3Dprior-guided generation and filtering pipeline, and GSISyn, a large-scale synthetic benchmark with controllable spatial operations and fully automated labeling. Together with a unified evaluation protocol, GSI-Bench enables scalable, model-agnostic assessment of spatial compliance and editing fidelity. Experiments show that fine-tuning unified multimodal models on GSI-Syn yields substantial gains on both synthetic and real tasks and, strikingly, also improves downstream spatial understanding. This provides the first clear evidence that generative training can tangibly strengthen spatial reasoning—establishing a new pathway for advancing spatial intelligence in multimodal models.

### 1. Introduction

Spatial intelligence [2, 4, 16, 38, 41]—the capacity to reason about objects, scenes, and their geometric relationships in the real 3D physical world—is foundational for multimodal large language models (MLLMs) [1, 4, 5, 7, 14, 28, 31, 34, 37]. It governs how models ground language in space and interact with the physical world, and is indispensable for embodied navigation [44, 47, 49], robotic manipulation [13, 15, 42], and 3D scene understanding [8, 9, 25, 41] under partial observability and domain shift. Despite this centrality, the prevailing ecosystem of datasets [16, 40], benchmarks [38, 41], and modeling

∗ Equal contribution. † Corresponding authors.

choices [1, 29] has developed spatial intelligence predominantly from an understanding perspective: recognition- or QA-style supervision, 2D/3D perception pipelines, and offline diagnostics on curated test suites.

Meanwhile, a parallel trend has emerged: unified multimodal models [5, 6, 8, 10, 19, 22, 29, 33, 35, 36] that jointly perform understanding and generation, aiming to demonstrate the mutual benefits between the two. Existing evidence largely confirms that stronger visual understanding can enhance image generation quality [5, 16, 28, 31]. Yet the reverse direction remains underexplored—can generation itself help models acquire a deeper grasp of spatial concepts and thereby strengthen their understanding? We argue that spatial intelligence offers a principled lens through which to investigate this question.

This paper takes a generative perspective on spatial intelligence. We ask: (1) Do modern generative or unified multimodal models exhibit generative spatial intelligence (GSI)—the capacity to respect and manipulate spatial constraints during image generation? (2) Can GSI be measured in a reliable, scalable, and model-agnostic way? (3) Can we enhance GSI via targeted interventions, and does such enhancement transfer to downstream spatial understanding tasks?

Such generative spatial intelligence is not only crucial for generating and editing images [7, 17, 27, 46] that faithfully preserve real-world spatial relationships, but also serves as a bridge connecting unified understanding–generation models with emerging paradigms such as “thinking with images” [26, 38, 41] and world models [11, 19, 21, 39]. This connection provides a foundational step toward deploying these models in embodied and interactive real-world tasks such as navigation and manipulation. To this end, we operationalize generative spatial intelligence through a spatially grounded image editing task, where a unified multimodal model receives an input image and an unambiguous, spatially-related editing instruction, and is required to generate an output image that satisfies the specified spatial constraints. Constructing datasets and automated evaluation pipelines that accurately reflect such precise spatial concepts is highly non-trivial. We address this

[Figure 1]

[Figure 2]

##### Object Rotation

[Figure 3]

Model Performance on GSI Bench

[Figure 4]

[Figure 5]

Rotate the faucet clockwise by 45 degrees.

[Figure 6]

[Figure 7]

##### Object Movement

Move the Tissue Box backward by 18 cm.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Object Scaling

[Figure 12]

[Figure 13]

Reduce the size of the notebook by 50%.

GSI-Syn fine-tuning boosts spatial intelligence and understanding

[Figure 14]

Object Removal

[Figure 15]

[Figure 16]

Remove the Dresser at the leftmost position.

[Figure 17]

[Figure 18]

Perspective Control

Evaluation Protocol

[Figure 19]

[Figure 20]

Rotate the camera 30 degrees to the left.

[Figure 21]

|Edit Locality (EL)<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>Appearance Consistency (AC)|
|---|

|Spatial Accuracy (SA)<br><br>[Figure 24]|
|---|

|[Figure 25]<br><br>Instruction Compliance (IC)|
|---|

[Figure 26]

- Figure 1. We introduce GSI Bench, a benchmark for grounded spatial intelligence that spans both real-world and synthetic scenes. GSI Bench evaluates a diverse set of spatial editing skills across multiple domains. By incorporating fine-grained evaluation protocols covering instruction compliance, spatial accuracy, edit locality, and appearance consistency, GSI Bench enables rigorous assessment of spatial reasoning in image-editing models. We further show that fine-tuning with GSI-Syn significantly boosts models’ spatial understanding and generalization across all subsets of the benchmark.

challenge from both real-world and synthetic perspectives.

simulation-based training, and such improvements consistently transfer to spatial understanding tasks.

For real scenes, the key advantage lies in the small domain gap to downstream applications, making them naturally aligned with embodied and perception-based tasks. However, they also pose inherent challenges: existing datasets [16, 24, 40] rarely contain precise annotations for spatial manipulations, and it is often difficult to express the spatial operations between image pairs using clear, unambiguous natural language descriptions that humans can easily understand. To overcome this, we design a complete data generation and filtering pipeline that leverages 3D grounding priors and rule-based spatial operation generation, uses MLLMs as captioners and validators, and incorporates human verification. This process results in GSI-Real, the first high-quality real-world benchmark for spatially grounded image editing. Nevertheless, real-world data remains limited in scale and diversity. To complement it, we construct a large-scale synthetic benchmark, GSI-Syn, based on simulation environments [12, 18, 48] with controllable rendering. GSI-Syn provides abundant, precisely labeled image pairs with diverse types and difficulty levels of spatial operations, and also offers an automated data generation pipeline for potential training use.

In summary, our main contributions are as follows: 1) We introduce GSI-Bench, a comprehensive benchmark that operationalizes generative spatial intelligence through spatially grounded image editing, enabling unified models to reason about and manipulate spatial relations during generation. 2) We construct two complementary components of GSI-Bench: GSI-Real, the first high-quality real-world dataset for spatially grounded editing, and GSI-Syn, a large-scale synthetic dataset and benchmark with controllable spatial operations and difficulty levels. 3) We establish an automated pipeline for dataset generation and evaluation that leverages 3D grounding priors, rule-based operation generation, multimodal captioning, and human verification. 4) We empirically demonstrate that simulationbased fine-tuning on GSI-Syn enhances generative spatial intelligence and further improves downstream spatial understanding tasks.

### 2. Related Work

#### 2.1. Spatial Intelligence in MLLMs

Spatial intelligence serves as a critical bridge connecting multimodal large language models to the physical 3D world. However, existing research has primarily focused on the understanding aspect of spatial reasoning, with lim-

Finally, we fine-tune existing unified multimodal models on GSI-Syn and evaluate them on both synthetic and real benchmarks. Our experiments demonstrate that generative spatial intelligence can be effectively enhanced through

ited exploration of generative spatial capabilities. On the benchmark side, recent efforts have probed MLLMs’ spatial understanding from various perspectives. VSI-Bench [38] evaluates video-based spatial reasoning over temporal sequences. MindCube [41] examines 3D spatial modeling from sparse multi-view observations. OmniSpatial [16] provides a systematic assessment across multiple spatial reasoning dimensions, including dynamic reasoning, spatial interaction, and perspective taking. On the methodology side, several works [15, 41, 42] aim to enhance spatial understanding in MLLMs. Spatial-MLLM [34] introduces an auxiliary spatial encoder to explicitly inject 3D geometric information into the model. SAT [24] leverages simulation environments to generate large-scale rule-based spatial reasoning data for training (real-world evaluation: SAT-Real). REVISION [3] demonstrates that data from simulated rendering engines (e.g., Blender) can benefit both image generation and spatial understanding when used as additional guidance. Despite these advances, prior work has not explored spatial intelligence from a unified understandinggeneration perspective. This work pioneers the evaluation of generative spatial intelligence in unified MLLMs, showing that fine-tuning on spatial editing tasks improves spatial reasoning in both modalities.

#### 2.2. Unified Multimodal Models

Recently, unified multimodal models for both image understanding and image generation have made rapid progress. Among closed-source systems, GPT-Image [14] integrates image generation directly into autoregressive language modeling, enabling attribute binding, text rendering, and iterative controlled editing within a unified token space. NanoBanana [7] further emphasizes spatially controllable generation, supporting multi-image conditioning, localized editing, and pose/object manipulation while preserving structural and geometric consistency. Meanwhile, the open-source community [5, 6, 10, 22, 35, 36] is actively advancing the paradigm of a single model that unifies understanding and generation. BAGEL [10] employs a Mixtureof-Transformers structure and achieves competitive performance on both vision understanding and generation. Emu3 [29] introduces native multimodal next-token prediction, and Emu3.5 [8] further extends this to interleaved image–text input/output, demonstrating capabilities in longhorizon scene modeling. Despite these advances, existing unified models still lack systematic evaluation of spatial understanding and controllable editing capabilities. To address this gap, we systematically benchmark multimodal models for their Generative Spatial Intelligence capability, providing the first comprehensive evaluation framework that connects generative and understanding aspects of spatial reasoning.

### 3. Generative Spatial Intelligence

#### 3.1. What is Generative Spatial Intelligence?

We define Generative Spatial Intelligence (GSI) as the capability of a unified multimodal model to respect, reason about, and manipulate spatial constraints during image generation. In contrast to traditional spatial understanding—which focuses on perceiving or describing spatial configurations—GSI reflects whether a model can actively enforce spatial relationships when generating new visual content.

Ideally, text-to-image generation could also manifest certain aspects of GSI, since generating a scene from a spatially descriptive prompt inherently requires reasoning about object layouts and relations. However, such setups typically lack sufficient constraints for precise assessment: the open-ended nature of text prompts introduces ambiguity, and there is no unique ground-truth target against which spatial consistency can be objectively measured. To more faithfully and quantitatively capture GSI, we therefore adopt an image-to-image editing formulation. In this setting, the model receives both a reference image and a spatially grounded instruction, and must produce an edited image that satisfies the specified spatial transformation. This task demands not only understanding the spatial structure of the input image but also manipulating it coherently according to the instruction—thus directly revealing the model’s generative spatial reasoning capability.

#### 3.2. Task Formulation

We operationalize GSI through a spatially grounded image editing task that emphasizes quantitative, controllable, and physically grounded spatial transformations. Formally, given an input image I and a textual instruction T specifying a spatial manipulation, the model is required to generate an output image I′ = f(I,T) that accurately satisfies the intended transformation while maintaining realism and semantic consistency.

Different from prior qualitative editing tasks that focus on semantic or stylistic changes (e.g., “make it look sunny”), our formulation introduces a suite of quantitative spatial operations that explicitly modify the underlying scene geometry rather than only pixel appearance. To formalize these operations, we first model each visual scene through its latent 3D structure, which defines object layouts, camera parameters, and their geometric relationships. This abstraction enables us to describe spatial manipulations as structured 3D transformations that can be consistently reflected in the generated image.

3D Scene Representation. We represent each scene as S = {Oi}Ni=1 ∪ {C}, where Oi = (ci,si,Ri) denotes the i-th object with center ci ∈ R3, size si ∈ R3, and rotation Ri ∈ SO(3), and C = (Rc,tc,K) denotes the camera.

Table 1. Spatial operation taxonomy.

ID Operation Description & Capability Tested

CM Camera-Relative Move Move along camera axes; egocentric reasoning OP Object-Relative Place Position relative to reference; pairwise relations

- OR Object Rotation Rotate objects; 3D orientation control RP Receptacle Placement Place in containers; hierarchical reasoning PC Perspective Control Change viewpoint; view-aware adaptation SR Spatial Removal Delete by criteria; global spatial understanding
- OS Object Scaling Scale uniformly; metric reasoning

Any 3D point pi can be projected to the image plane as p˜i = π(K(Rcpi + tc)), establishing the geometric foundation for spatial manipulations and evaluation.

Spatial Operation Representation. Each spatial instruction is structured as T = ⟨R,A,Φ3D⟩, where R identifies target objects, A specifies the action, and Φ3D : Ssrc → Sdst defines the geometric transformation by updating object poses or camera parameters: (ci,Ri,Rc,tc)src  → (c′i,R′i,R′c,t′c)dst. For instance, ”Move the apple 15 cm left” induces a camera-relative translation, while ”Place the cup left of the plate” defines a relational constraint c′cup = cplate + ∆left. This formulation explicitly links linguistic spatial instructions with 3D geometric transformations, providing a unified interface for data synthesis, model training, and quantitative evaluation in GSI-Bench.

- 3.3. Categories of Spatial Operations We define seven quantitatively grounded spatial operations spanning object-, camera-, and scene-level transformations, enabling comprehensive GSI evaluation. Formal mathematical definitions are provided in the appendix material.
- 4. GSI-Bench Construction 4.1. Synthetic Benchmark: GSI-Syn

To facilitate scalable and controllable evaluation, we construct GSI-Syn, a large-scale synthetic benchmark for generative spatial intelligence. GSI-Syn is built upon opensource simulators including AI2-THOR [18] and MesaTask [12], covering varied scenarios like indoor navigation and tabletop manipulation. The primary advantage of this simulation-based approach is two-fold. First, it provides perfect ground-truth data, including the initial 3D scene representation (Ssrc), the precise geometric transformation (Φ3D), and the resulting target scene (Sdst), allowing for unambiguous, automated validation. Second, we can render the ground-truth edited image (I′) directly from Sdst, yielding high-quality (I,T,I′) triplets for both evaluation and training. Our automated synthesis pipeline consists of the following stages.

Scene Initialization and Viewpoint Curation. A key aspect of our data generation is sampling diverse and meaningful camera viewpoints. For each indoor scene, we employ DBSCAN clustering [23] on the floor plan to partition the space into distinct rooms. Within each room, we per-

form maximally dispersed viewpoint sampling. To ensure these viewpoints are ”actionable,” we prioritize those containing more manipulable objects, guaranteeing each viewpoint can support a rich set of potential spatial operations.

Action Candidate Generation and Geometric Grounding. For each viewpoint, we generate valid action candidates through object selection and multi-level geometric validation. We randomly select a target object, ensuring it is not occluded and rests on a stable surface. For relational operations (e.g., ”place the apple to the left of the bowl”), a reference or container object is also selected. We then perform rigorous 3D geometric checks to verify physical plausibility: camera-relative translations are validated by ensuring the target remains visible and does not fall off its supporting surface; object-relative placements are checked for spatial sufficiency and collision avoidance. A templatebased module generates the corresponding textual instruction T.

Simulated Execution and Success Validation. With a valid instruction T and transformation Φ3D, we execute the action in the physics-enabled simulator. We first analytically compute the ideal destination state Sidealdst , then the physics engine executes the action to produce the actual outcome Sactualdst . An operation succeeds only if the actual state matches the ideal state, confirmed by checking the final position and visibility of the target object. Failed executions (e.g., due to unforeseen collisions) are rolled back and resampled.

Post-Generation Filtering and Quality Assurance. To ensure benchmark quality, we apply two-stage filtering. First, using instance segmentation masks, we filter out samples where pixel-level change is negligible, ensuring every edit is visually significant. Second, we leverage an MLLM (Qwen3-VL-235B) as a quality gate to identify and discard samples with subtle anomalies difficult to capture with hard-coded rules, such as simulation artifacts (e.g., object clipping), physically implausible outcomes, or severe occlusions that render instructions ambiguous.

Through this automated pipeline, GSI-Syn generates diverse, physically valid, and geometrically precise editing pairs at scale, offering a reproducible and extensible platform for probing spatial reasoning in generative models under fully controlled conditions.

#### 4.2. Real-world Benchmark: GSI-Real

To complement the synthetic GSI-Syn, we curate GSIReal, a real-world benchmark for evaluating generative spatial intelligence in natural images. Unlike simulationbased GSI-Syn, constructing GSI-Real presents unique challenges: we cannot obtain perfect 3D scene representations nor directly execute physical transformations to acquire ground-truth edited images (I′). Consequently, we develop an alternative evaluation protocol that bypasses the

Synthetic Benchmark: GSI-Syn

Action Generation and Grounding

Simulated Execution and Success Validation

Scene Initialization and Viewpoint Curation

Post-Generation Filtering and Quality Assurance

[Figure 27]

Randomly select target object and its corresponding container

[Figure 28]

###### Instance Segmentation Mask Filtering

1. Compute Destination Scene State

2. Simulator Executes The Action

Maximally dispersed viewpoint sampling

[Figure 29]

[Figure 30]

[Figure 31]

3D Geometric Checks

[Figure 32]

Layout Map

Check Whether Actual State Matches The Ideal State

[Figure 33]

X Pixel-level change is negligible

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Multimodal Large Language Model

[Figure 38]

X Object breaks

Won’t Fall

Avoid collisions

Not Obscured

Discard Anomalies

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

No: Roll Back

[Figure 44]

Match?

[Figure 45]

[Figure 46]

[Figure 47]

Yes

Template Based Module

Output

X Severe Occlusion

Select views that includes more operable objects

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Move the HousePlant's center to the left of the Vase's center by 20 centimeters, relative to the camera view.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

X Object Clipping

Generated Instruction

Real-world Benchmark: GSI-Real

Human Review and Refinement

3D Scene Reconstruction and Operation Generation

Simulated Execution and Success Validation

Select Candidate Frames

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Before-and-after States

DetAny3D

Manual review

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

3D Bbox, Pose Semantic Label

[Figure 65]

Frequency-domain Analysis

[Figure 66]

Correct ambiguous instructions

Identify residual annotation errors

Original Bbox

Modified Bbox

[Figure 67]

MLLM

[Figure 68]

[Figure 69]

1. Check Physical Errors

2. Rewrite Or Enrich Captions

[Figure 70]

Remove the rightest bottle from the scene, while keeping other objects unchanged.

[Figure 71]

Move the rightest red chair 10 centimeters to the left.

X Motion Blur ✔ Clear Frame

Generate Spatial Operations

Concrete Caption

- Figure 2. Benchmark curation pipeline.The pipeline builds both synthetic (GSI-Syn) and real-world (GSI-Real) benchmarks through unified scene processing, action generation, and validation. For GSI-Syn, scenes are sampled from diverse viewpoints, feasible actions are generated via 3D geometric checks, and a simulator validates outcomes before filtering failures and anomalies. For GSI-Real, clear frames are selected, 3D scene structure is reconstructed, and spatial operations are generated and validated on bounding boxes. Human review then refines captions and corrects residual annotation errors, ensuring high-quality spatial-editing supervision.

need for I′. Each sample in GSI-Real is represented as (I,T,Ssrc,Φ3D,Sdst), where the edited image is generated by the model under test, and success is evaluated by analyzing spatial consistency between the predicted edit and the specified 3D transformation.

Image Source and Frame Selection. We source real-world images from ScanNet++ [40], a large-scale indoor RGB-D dataset. To ensure diversity and visual quality, we sample one frame from every 20 frames and apply multi-criteria filtering. We perform frequency-domain analysis to prioritize frames with high sharpness and minimal motion blur, and employ a 3D object grounding model to detect manipula-

ble objects in each candidate frame. Frames exhibiting both high visual clarity and rich object content are retained for subsequent processing.

3D Scene Reconstruction and Operation Generation. For each selected image I, we leverage DetAny3D [43], an open-vocabulary 3D grounding model, to reconstruct the source scene Ssrc = g(I). This extracts object-level 3D bounding boxes, poses, and semantic labels in the camera coordinate system, with camera intrinsics obtained from dataset metadata. With Ssrc established, we generate candidate spatial operations (move, rotate, remove) through a rule-based procedure similar to GSI-Syn: randomly select-

ing a target object and proposing a plausible transformation Φ3D to compute Sdst. However, due to positional uncertainty in 3D grounding and the absence of physics simulation, additional quality control is essential.

Visualization-based Verification and MLLM Gating. To filter invalid operations, we employ a visualization-driven validation approach. For each candidate operation, we project both the original bounding box Oi and transformed bounding box O′

i onto the image plane, generating side-byside before-and-after visualizations. An MLLM then serves three critical functions: (1) identifying and discarding physically implausible operations (e.g., collisions, floating objects, out-of-frame placements, severe occlusions), (2) correcting annotation errors such as label-object mismatches, and (3) generating diverse natural language instructions by rewriting template-based captions based on visual context and operation metadata.

Human Review and Refinement. As a final quality assurance step, we conduct comprehensive manual review of the entire GSI-Real dataset to identify and correct residual annotation inaccuracies or ambiguous instructions, ensuring high annotation quality and a genuinely challenging testbed for real-world spatial reasoning.

#### 4.3. Evaluation Protocol

To comprehensively assess generative spatial intelligence, we design a multi-faceted evaluation protocol with four core metrics.

Instruction Compliance (IC). This binary metric evaluates whether the edited scene satisfies the spatial semantics specified in the instruction (e.g., directional relations, containment). We allow reasonable tolerance rather than strict numerical precision: an operation succeeds if the final object pose falls within a plausible range of the ideal target.

Spatial Accuracy (SA). For edits passing compliance, we measure fine-grained geometric precision by computing normalized translation error, relative pose error for multiobject operations, and geodesic rotation error on SO(3). Errors are aggregated into a single continuous accuracy score per sample.

Edit Locality (EL). We assess localized editing by computing LPIPS [45] on non-target regions between original and edited images, using the projected 3D bounding box as a mask to exclude the edited object. Lower LPIPS scores indicate better consistency of unaffected regions. To ensure higher score indicates better performance, we take 100(1 − LPIPS) as EL score. Before scoring IC and SA, we apply a dataset-specific locality gate using masked SSIM [30] and LPIPS [45] (stricter on synthetic data than on GSI-Real); full thresholds are in the appendix.

Appearance Consistency (AC). We leverage an MLLM (Qwen3-VL-235B) to verify appearance quality. For transformation operations (move, rotate, scale), it checks

whether the edited object retains its original visual attributes (category, texture, color). For removal operations, it assesses background inpainting quality, identifying residual artifacts or visual discontinuities.

Detailed definitions and thresholds are in the appendix.

### 5. Fine-tuning Unified MLLMs for GSI

Beyond evaluation, GSI-Syn’s automated synthesis pipeline enables us to construct large-scale editing training data for fine-tuning unified multimodal large language models. This allows us to explore two key questions: (1) whether generative training can directly enhance spatial understanding capabilities, and (2) whether unified models can effectively bridge the sim-to-real gap through joint perceptiongeneration learning. We choose BAGEL [10] as our base model, which natively supports image editing and employs self-attention for deep interaction between perception and generation modules, potentially enabling mutual reinforcement between understanding and generation. We construct a training set from GSI-Syn comprising diverse spatial operations (move, rotate, resize, remove, scaling, view change) Further training details are provided in the appendix.

### 6. Experiments

#### 6.1. Experimental Setup

Benchmarks and Dataset Statistics. Our evaluation suite consists of two complementary benchmarks. GSI-Real contains 441 samples from 211 diverse indoor scenes in ScanNet++ [40], spanning three operation types. GSISyn comprises two subsets: GSI-Syn-Room (593 samples, six operations) built on AI2-THOR [18], and GSISyn-Tabletop (600 samples, three operations) using MesaTask [12]. Dataset statistics are in Figure 1. To evaluate cross-view generalization, we construct GSI-Syn-Bathroom with 200 samples featuring randomized viewpoints. For fine-tuning, GSI-Syn-Train contains 1,500 training samples per operation type per environment, totaling 10,500 samples with strict scene separation from test sets.

Baseline Models. We evaluate nine state-of-the-art models: seven open-source models (BAGEL [10], Anyedit [17], Uniworld [19], Ultra [46], Qwen-Image-Edit [32], Omnigen2 [33], Emu3.5 [8]) and two proprietary models (NanoBanana [7], GPT-image [14]). These models span diverse architectures including unified MLLMs and instruction-based editors, evaluated using publicly available checkpoints or API endpoints with default settings.

#### 6.2. Benchmarking Generative Spatial Intelligence

Across GSI-Bench, we observe clear performance disparities reflecting different levels of spatial reasoning capability. Closed-Source Models. On GSI-Syn-Table, Nano Banana and GPT-img reach 37.03 and 33.97 average respectively,

Table 2. Performance comparison on the proposed GSI-Bench across three datasets and four spatial reasoning dimensions: Instruction Compliance (IC), Spatial Accuracy (SA), Appearance Consistency (AC), and Edit Locality (EL). Higher is better.

|Evaluation Dimension|Closed-Source Models|Open-Source Models| |
|---|---|---|---|
| |Nano Banana GPT img<br><br>|Anyedit Uniworld Ultra Qwen Omnigen2 Emu3.5 BAGEL BAGEL+GSI-Syn<br><br>|∆ ↑|
|GSI-real<br><br>IC SA AC EL Avg<br><br>|38.78 41.72 21.60 28.04 38.78 41.52 34.92 27.52 33.52 34.70|10.20 28.80 10.66 51.02 33.56 51.70 31.97 40.14<br><br>8.37 18.36 5.70 31.22 19.62 29.51 22.07 27.76<br><br>9.68 28.75 9.48 50.95 33.20 51.70 31.88 40.14<br><br><br>8.75 18.51 8.97 40.55 29.82 41.17 27.89 37.11<br><br>9.25 23.61 8.70 43.44 29.05 43.52 28.46 36.28<br><br><br>|+8.16 +5.68<br><br>+8.25<br><br>+9.22<br><br><br>+7.83<br><br>|
|GSI-syn-table<br><br>IC SA AC EL Avg|36.62 39.33 38.96 26.16<br><br>36.62 38.40<br><br>35.91 31.98<br><br>37.03 33.97<br><br><br>|10.33 15.83 2.17 27.33 0.00 39.17 27.17 50.67 22.84 30.33 3.09 25.52 0.00 24.09 26.52 44.10 10.33 15.58 1.33 27.27 0.00 38.82 26.52 50.67<br><br>9.52 14.43 1.93 25.51 0.00 34.91 26.17 49.52 13.26 19.04 2.13 26.41 0.00 34.25 26.59 48.74<br><br>|+23.50<br><br>+17.58<br><br>+24.15<br><br><br>+23.36 +22.15|
|GSI-syn-room<br><br>IC SA AC EL Avg|20.65 8.05<br><br>16.85 8.05<br><br>28.01 16.69<br><br>19.65 7.34<br><br>21.29 10.03<br><br><br>|7.00 12.69 2.20 20.40 18.71 20.70 16.11 24.01<br><br>6.46 11.55 2.21 17.73 15.03 16.56 14.53 19.41 11.85 20.40 3.46 28.67 25.94 26.98 24.00 31.64<br><br>5.50 11.03 1.86 18.48 17.13 17.56 14.82 22.61<br><br>7.70 13.92 2.43 21.32 19.20 20.45 17.37 24.42<br><br><br>|+7.90 +4.88 +7.64 +7.79 +7.05<br><br>|

-roomGSI-realGSI-syn-table

with strength in IC and AC. On GSI-Real, however, their averages (33.52 and 34.70) are only comparable to opensource systems like Qwen (43.44) and Emu3.5 (43.52), indicating that closed-source models, despite strong general visual generation, struggle with fine-grained spatial manipulations requiring explicit geometric understanding.

Open-Source Baselines. Emu3.5 is the strongest opensource performer, achieving the best results on GSI-Real (43.52 average) and high scores across all dimensions. In contrast, general-purpose models like Uniworld, Ultra, and Omnigen2 show substantially lower scores, with extremely low AC or IC values revealing difficulty following structured spatial instructions. These results suggest most opensource models lack 3D-aware inductive biases for precise spatial reasoning, whereas Emu3.5 benefits from stronger spatial priors through its video-centric training.

Qualitative results (Fig 3 and more in appendix) reveal consistent trends: most models perform better on removal than other operations, indicating deletion is easier than precise geometric manipulation. Emu3.5 produces the cleanest removals with strongest spatial consistency. However, Ultra and AnyEdit often fail to preserve object identity; AnyEdit, BAGEL, and Omnigen2 introduce artifacts; AnyEdit frequently leaves targets unchanged; BAGEL sometimes misinterprets translation as camera motion. While BAGEL, Emu3.5, and Qwen reliably follow referential cues, they occasionally remove additional content, indicating finegrained localization remains challenging.

#### 6.3. Impact of Fine-tuning on GSI-Syn

Effective Sim-to-Real Transfer. Fine-tuning on GSI-Syn yields consistent improvements across both domains. On GSI-Real, the model achieves a 7.83-point average gain over BAGEL (28.46→36.28). The largest gains are in Edit Locality (+9.22), Appearance Consistency (+8.25), and In-

- Table 3. Evaluation on OmniSpatial benchmark. We report accuracy (%) across four core reasoning dimensions. Fine-tuning on GSI-Syn improves spatial understanding, particularly in Spatial Interaction and Perspective Taking. Best results among opensource 7B models are bolded. †Proprietary models.

Model Size Overall Dynamic Spatial Logic Persp.

GPT-4-turbo† [1] – 34.06 38.39 36.49 24.80 33.69 Gemini-2.5† [27] – 52.12 63.59 67.46 35.67 43.10

LLaVA-1.5 [20] 7B 34.97 35.38 35.13 25.99 38.82 Qwen-VL-2.5 [28] 7B 39.25 46.30 30.06 35.65 39.68 BAGEL [10] 7B 41.55 47.38 45.67 32.14 39.22

BAGEL + GSI-Syn 7B 42.07 48.33 47.67 28.97 40.29

- Table 4. Evaluation on SAT-Real benchmark [24]. Accuracy (%) across five spatial reasoning dimensions. Fine-tuning with GSI-Syn notably improves goal-directed and egocentric understanding. Best results among open-source 7B models are bolded.

Model Overall Pers GoalAim EgoAct ObjM EgoM Qwen-VL-2.5 [28] 56.33 43.94 67.65 56.76 56.52 56.52 BAGEL [10] 65.33 46.97 75.00 75.68 65.22 60.87 BAGEL + GSI-Syn 69.33 48.48 85.29 72.97 65.22 73.91

struction Compliance (+8.16), indicating better preservation of object identity and more precise, spatially constrained edits despite training exclusively on synthetic images; Spatial Accuracy also improves (+5.68). On synthetic benchmarks, improvements are even larger: +22.15 on GSISyn-Table and +7.05 on GSI-Syn-Room. The model benefits particularly from the structured geometric variations in GSI-Syn-Table targeting localized edits. Gains on GSISyn-Room are more modest due to increased scene complexity and spatial ambiguities, highlighting remaining limitations in global spatial reasoning. These results demonstrate that geometrically grounded synthetic supervision

Task Instruction Input Image Emu3.5 BAGEL BAGEL+ GT

[Figure 72]

- (a) CameraRelative

Movement

Move the trash bin 20 centimeters to the left.

[Figure 73]

(c) Object Rotation

Turn the toy (Plush teddy bear, soft body, light brown and white mottled design, rounded shape, short limbs.) 45 degrees to the left.

- (b) Spatial Removal

[Figure 74]

Remove the frontest table lamp from the scene, while keeping other objects unchanged.

[Figure 75]

- (d) Object Scaling

Scale down the bouquet (Dried flowers in a round ceramic vase with elongated stems and rounded flower heads.) by 25%.

[Figure 76]

- (e) Perspective Control

Look up by 30 degrees.

- Figure 3. Qualitative comparison of spatial editing results across five instruction types. Rows 1–2 use GSI-Real samples, Rows 3–4 use GSI-Table, and the last row uses GSI-Room. Columns show the input image, outputs from Emu3.5, BAGEL, BAGEL+(fine-tuned with GSI-Syn), and the ground-truth target. BAGEL+ demonstrates stronger spatial fidelity and better preservation of unaffected content. Further examples and corresponding metrics are provided in the appendix.

significantly enhances spatial editing capabilities and transfers robustly to real images without requiring real-world annotations.

and reasoning-based objectives. Results on SAT-Real [24] (Table 4) further validate this finding: fine-tuning on GSISyn yields notable improvements in Allocentric Perspective, Goal Aiming, and Egocentric Movement, achieving an overall gain of +4.00%.

Enhanced Spatial Understanding through Generative Training. As shown in Table 3, fine-tuning BAGEL solely on spatially-related generative editing data (GSI-Syn)without any understanding or reasoning data—improves performance on the OmniSpatial benchmark. BAGEL shows consistent gains in the most relevant dimensions: Dynamic Reasoning (+0.95%), Spatial Interaction (+2.00%), and Perspective Taking (+1.07%). We observe a moderate decrease in Complex Logic, attributable to the absence of explicit reasoning supervision in the fine-tuning corpus. Nevertheless, the overall improvement provides strong evidence that generative spatial training alone substantially enhances spatial understanding, highlighting a promising direction for unified MLLMs that jointly leverage generative

### 7. Conclusion

This paper studies Generative Spatial Intelligence. We introduce GSI-Bench, a benchmark spanning seven spatial operation categories, with a real-world set, a large-scale synthetic set, and automated pipelines based on 3D grounding priors. Experiments show that current state-of-theart models still struggle with spatially accurate generation. Fine-tuning on GSI-Syn improves spatial compliance and transfers to real-world and spatial understanding tasks, suggesting that generative training enhances spatial reasoning.

### Acknowledgments

This work was supported in part by The Pioneer R&D Program of Zhejiang (Grant No. 2025C01011), by the Ant Group Research Intern Program, and by the National Natural Science Foundation of China (Grant No. 62576315).

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1, 7

- [2] Zhongang Cai, Yubo Wang, Qingping Sun, Ruisi Wang, Chenyang Gu, Wanqi Yin, Zhiqian Lin, Zhitao Yang, Chen Wei, Xuanke Shi, et al. Has gpt-5 achieved spatial intelligence? an empirical study. arXiv preprint arXiv:2508.13142, 2025. 1
- [3] Agneet Chatterjee, Yiran Luo, Tejas Gokhale, Yezhou Yang, and Chitta Baral. Revision: Rendering tools enable spatial fidelity in vision-language models. In European Conference on Computer Vision, pages 339–357. Springer, 2024. 3
- [4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168, 2024. 1
- [5] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models—architecture, training, and dataset. arXiv preprint arXiv:2505.09568, 2025. 1, 3
- [6] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1, 3

- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 3, 6
- [8] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583,

2025. 1, 3, 6

- [9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 1
- [10] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal

- pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3, 6, 7
- [11] David Ha and J¨urgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. 1
- [12] Jinkun Hao, Naifu Liang, Zhen Luo, Xudong Xu, Weipeng Zhong, Ran Yi, Yichen Jin, Zhaoyang Lyu, Feng Zheng, Lizhuang Ma, et al. Mesatask: Towards task-driven tabletop scene generation via 3d spatial reasoning. arXiv preprint

- arXiv:2509.22281, 2025. 2, 4, 6

[13] Zheng Huang, Mingyu Liu, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Xiaoman Li, Yiduo Jia, Hao Zhong, Hao Chen, et al. Notvla: Narrowing of dense action trajectories for generalizable robot manipulation. arXiv preprint

- arXiv:2510.03895, 2025. 1

- [14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 3, 6
- [15] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 1, 3
- [16] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 1, 2, 3

- [17] Houcheng Jiang, Junfeng Fang, Ningyu Zhang, Guojun Ma, Mingyang Wan, Xiang Wang, Xiangnan He, and Tat-seng Chua. Anyedit: Edit any knowledge encoded in language models. arXiv preprint arXiv:2502.05628, 2025. 1, 6
- [18] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474,

2017. 2, 4, 6

- [19] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 1, 6
- [20] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 7
- [21] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268,

- 2024. 1

[22] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751,

- 2025. 1, 3

- [23] Mudasir Raja, Priya Hasan, Md Mahmudunnobe, Md Saifuddin, and SN Hasan. Membership determination in open clusters using the dbscan clustering algorithm. Astronomy and Computing, 47:100826, 2024. 4
- [24] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. Sat: Dynamic spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024. 2, 3, 7, 8
- [25] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021. 1
- [26] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 1
- [27] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 7
- [28] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191, 2024. 1, 7
- [29] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 3
- [30] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [31] Zehan Wang, Jiayang Xu, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Genspace: Benchmarking spatially-aware image generation. arXiv preprint arXiv:2505.24870, 2025. 1
- [32] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 6
- [33] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 6
- [34] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025. 1, 3

- [35] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1, 3
- [36] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 1, 3
- [37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1
- [38] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 3
- [39] Sherry Yang, Yilun Du, Seyed Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Representation Learning, pages 45210–45234, 2024. 1
- [40] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 1, 2, 5, 6
- [41] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, and Li Fei-Fei. Mindcube: Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025. 1, 3
- [42] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024. 1, 3
- [43] Hanxue Zhang, Haoran Jiang, Qingsong Yao, Yanan Sun, Renrui Zhang, Hao Zhao, Hongyang Li, Hongzi Zhu, and Zetong Yang. Detect anything 3d in the wild. arXiv preprint arXiv:2504.07958, 2025. 5
- [44] Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, et al. Embodied navigation foundation model. arXiv preprint arXiv:2509.12129, 2025. 1
- [45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [46] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024. 1, 6

- [47] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei Wang. Towards learning a generalist model for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13624–13634, 2024. 1
- [48] Weipeng Zhong, Peizhou Cao, Yichen Jin, Li Luo, Wenzhe Cai, Jingli Lin, Hanqing Wang, Zhaoyang Lyu, Tai Wang, Bo Dai, et al. Internscenes: A large-scale simulatable indoor scene dataset with realistic layouts. arXiv preprint arXiv:2509.10813, 2025. 2
- [49] Yufeng Zhong, Chengjian Feng, Feng Yan, Fanfan Liu, Liming Zheng, and Lin Ma. Robotrom-nav: A unified framework for embodied navigation integrating perception, planning, and prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6416–6425, 2025. 1

