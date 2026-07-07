# arXiv:2511.22659v1[cs.AI]27Nov2025

## Geometrically-Constrained Agent for Spatial Reasoning

Zeren Chen1,2∗, Xiaoya Lu2,3*, Zhijie Zheng1,2, Pengrui Li1, Lehan He1,4, Yijin Zhou2,3,4 Jing Shao2, Bohan Zhuang5†, Lu Sheng1† 1School of Software, Beihang University 2Shanghai AI Laboratory 3Shanghai Jiao Tong University 4Shanghai Innovation Institute 5ZIP Lab, Zhejiang University

{czr1604,lsheng}@buaa.edu.cn, {luxiaoya, shaojing}@pjlab.org.cn

Homepage: https://gca-spatial-reasoning.github.io

### Abstract

Vision Language Models (VLMs) exhibit a fundamental semantic-to-geometric gap in spatial reasoning: they excel at qualitative semantic inference but their reasoning operates within a lossy semantic space, misaligned with highfidelity geometry. Current paradigms fail to bridge this gap. Training-based methods suffer from an “oracle paradox,” learning flawed spatial logic from imperfect oracles. Toolintegrated methods constrain the final computation but critically leave the VLM’s planning process unconstrained, resulting in geometrically flawed plans. In this work, we propose Geometrically-Constrained Agent (GCA), a trainingfree agentic paradigm that resolves this gap by introducing a formal task constraint. Specifically, we strategically decouples the VLM’s role into two stages. First, acting as a semantic analyst, the VLM translates the user’s ambiguous query into the formal, verifiable task constraint, which defines the reference frame and objective. Second, acting as a task solver, the VLM generates and executes tool calls strictly within the deterministic bounds defined by the constraint. This geometrically-constrained reasoning strategy successfully resolve the semantic-to-geometric gap, yielding a robust and verifiable reasoning pathway for spatial reasoning. Comprehensive experiments demonstrate that GCA achieves SOTA performance on multiple spatial reasoning benchmarks, surpassing existing training-based and tool-integrated methods by ∼27%.

### 1. Introduction

Intelligent agents operating in real-world applications, such as robotics [41, 54, 62], AR/VR [2, 11, 31], and autonomous driving [9, 42, 51], demand a perceptual understanding of the world akin to humans. Humans intuitively comprehend their surroundings as a cohesive 3D environment, effort-

*Equal contribution. †Corresponding author.

lessly discerning object orientations and complex spatial relationships. However, equipping Vision Language Models (VLMs) into agents with this holistic spatial reasoning capability remains a critical challenge [18, 53, 55, 59, 60].

As shown in Figure 1 (a), current VLMs lossily translate rich visual information into a textual semantic space, leading fine-grained geometric details to be omitted or distorted [24, 50]. This creates a fundamental semantic-togeometric gap: VLMs excel at probabilistic, qualitative semantic inference, but their lossy semantic space required for spatial reasoning fails to ground high-fidelity geometry. For example, a VLM may possess the spatial commonsense (e.g., intuitively knowing that “sitting on a sofa” implies a viewpoint aligned with the sofa’s orientation), yet critically fail at high-precision geometric computation (e.g., determining the sofa’s orientation) and robust spatial imagination (e.g., imagining the user’s egocentric perspective). To reconcile this gap, robust constraints must be imposed, guiding the VLM’s reasoning onto a geometrically sound and verifiable pathway.

However, effectively applying these constraints remains a formidable challenge. Recent approaches that apply implicit constraints via end-to-end training [8, 22, 30, 33, 41, 48, 50, 52] attempt to embed geometric logic by fine-tuning on massive datasets. These methods, however, face an “oracle paradox”: their data generation relies on oracles like GPT-4o [17] which themselves struggle with spatial reasoning [18, 53, 55, 59, 60]. Consequently, the VLM is often trained on flawed spatial logic rather than sound geometric principles. An alternative paradigm, tool integration [12, 49, 62], attempts to bridge this gap by adopting an iterative plan-then-execute strategy, which offloads highprecision geometric computation to deterministic external tools. While this constrains the final computation process, the VLM’s planning process remains unconstrained. To plan next step, the VLM must still perform spatial imagination and further decision-making within its lossy semantic space, inevitably producing geometrically flawed plans.

[Figure 1]

<think/> Sitting on sofa means that the viewpoint is aligned with sofa’s orientation. … </think>

[Figure 2]

Q: Sitting on the sofa in Figure 2, where is the round dinner table in Figure 1 relative to you?

VLM as Semantic Analyst

[Figure 3]

[Figure 4]

Formal Task Constraint !!"#$

Reference Frame Constraint !!: viewpoint is aligned with sofa’s orientation, +#ℛ = +##$%& = front Objective Constraint !': the positional relationship of table relative to sofa , *#$%&→)&*+,

[Figure 5]

Robust Semantic Inference

[Figure 6]

1

Lossy TranslationTextual Space

Vison Language Models

Gap

[Figure 7]

###### Lossy Geometric Details

[Figure 8]

| |
|---|

Semantic-to-Geometric Bridge

Let me imagine where the table is from sofa

[Figure 9]

table

[Figure 10]

[Figure 11]

VLM as Task Solver Constrained by +)&#-

[Figure 12]

!

[Figure 13]

[Figure 14]

###### …

[Figure 15]

[Figure 16]

Feedback

2

|table|
|---|

Call

[Figure 17]

|sofa|
|---|

[Figure 18]

!' Compute

[Figure 19]

|[Figure 20]<br><br>sofa|
|---|

& Visualize

[Figure 21]

Maybe, in back-left?

|+X| |
|---|---|

|+Z|
|---|

[Figure 22]

Tool Box

[Figure 23]

[Figure 24]

!!

[Figure 25]

|+Y|
|---|

front-right

(b) Geometrically-Constrained Spatial Reasoning

Spatial Reasoning Query

(a) Semantic-Geometric Gap

- Figure 1. Overview. (a) Semantic-Geometric Gap. The geometric details required for spatial reasoning are lost when translating visual information into textual space, leading to VLM’s flawed reasoning or unconstrained planning. (b) Geometrically-Constrained Spatial Reasoning. We propose a formal task constraint that serves as a deterministic bridge between semantics and geometry in spatial reasoning.

For instance, when asked to reason from the perspective of a user “sitting on the sofa” (see Figure 1), its unconstrained plan may default to the camera’s viewpoint, compromising the problem definition before any tool is even called.

ing strictly within the deterministic bounds defined by Ctask. This two-stage decoupling directly bridges the semantic-togeometric gap. Through formulating a geometrically sound constraint, we force the VLM to solve deterministic mathematical problems, thereby avoiding the demands for directly computing or imagining about high-fidelity geometric details that are lost in its semantic space. Extensive experiments demonstrate the effectiveness and generalizability of GCA paradigm. GCA yields substantial performance gains when applied to several foundation VLMs (by an average of ∼37%), establishing a new state-of-the-art across a diverse suite of challenging spatial reasoning benchmarks.

These challenges reveal the critical research question: How do we bridge the VLM’s semantic-to-geometric gap? We argue the solution is not to force the VLM to reason about lossy geometric details directly, but to reframe the problem into a task that leverages its inherent strengths: using its spatial commonsense to define a formal task constraint Ctask for subsequent computation. Specifically, this Ctask must be (1) grammatically rich enough to define complex spatial concepts, such as viewpoints, which elude traditional state-based formalisms (2) semantically clear enough for a VLM to generate using its qualitative strengths, and (3) geometrically sound enough to provide a deterministic, verifiable constraint for subsequent computation.

### 2. Related Work

Spatial Reasoning in Vision Language Models. Spatial reasoning, including comprehension and mental manipulation of 3D spatial relationships [18, 53, 55, 59, 60], remains a foundational challenge for Vision Language Models (VLMs) [17, 21, 36, 43]. To address this deficit, recent research [4, 8, 22, 30, 33, 39, 48, 50, 52] focuses on large-scale, end-to-end training on specialized spatial datasets. These methods attempt to bridge the 2D-3D cognitive gap by incorporating geometric priors, such as explicit 3D structural features [48], or depth maps [4], directly into the VLM’s architecture, but they are often hindered by the reliance on high-quality datasets generated by flawed oracle. Another line of research introduce tool-integrated reasoning [12, 25, 29, 49, 62] to offloads deterministic geometric computation to external modules. For example, Spa-

To this end, we introduce Geometrically-Constrained Agent (GCA), a training-free agentic paradigm for geometrically-constrained spatial reasoning. As shown in Figure 1 (b), this strategy leverages a formal task constraint, Ctask, to decouple the reasoning process into two stages: (1) Task Formalization. The VLM, acting as a semantic analyst, translates the ambiguous query and visual data into the formal, verifiable task constraint Ctask. This stage defines what to solve, establishing immutable sub-constraints: a reference frame constraint and an objective constraint. (2) Constrained Geometric Computation. The VLM then, acting as a task solver, generates and executes tool calls to compute the final answer, operat-

Stage 1: Task Formalization

Q: The fireplace faces north. In which direction is the painting on the wall in the fitness area facing?

[Figure 26]

VLM as Analyst Formal Task Constraint !!"#$

[Figure 27]

[Figure 28]

Reference Frame Constraint “Fireplace faces north” defines the absolute direction. The reference frame's Z-axis must align with the fireplace’s Z-axis: +#!"# = +#$%!"&'()" = North.

Prompt: Please find element that provides nonnegotiable definition for relative perspectives or absolute directions.

1 2

Objective Constraint The orientation of the painting on the wall in the fitness area.

[Figure 29]

Stage 2: Constrained Geometric Computation Toolbox

VLM as Constrained Task Solver

Solve Reference Frame Constraint !%

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

3D Reconstruction Detect “Fireplace” Segment “Fireplace” Fireplace Orientation

Object Detection

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Object Orientation

[Figure 40]

Segmentation

[Figure 41]

3D Reconstruction

[Figure 42]

Scene Alignment

###### Solve Objective Constraint !&

[Figure 43]

###### Code Generation

###### Detect “Painting”

Segment “Fireplace”

# world-to-reference transformation R_ref = fireplace_ori.T

Painting Orientation

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

OCR Tool

[Figure 48]

!

...

Ambiguity 2 paintings

[Figure 49]

[Figure 50]

+Y

BEV

[Figure 51]

# paint frame => world frame => ref frame paint_z_w = paint_ori.T @ tensor([0,0,1]) paint_z_ref = R_ref @ paint_z_w

[Figure 52]

Optical Flow

+X

+Z

... # Judge component if paint_z_ref[2] > 0: return “north” if paint_z_ref[0] > 0: return “east”

[Figure 53]

###### Utility Tool

Both is OK

Close-loop

[Figure 54]

[Figure 55]

Python Tool

[Figure 56]

[Figure 57]

Execution east

- Figure 2. Overall Paradigm of GCA. Given a spatial reasoning query, our GCA leverages a geometrically-constrained reasoning strategy centered on the formal task constraint (Ctask). The VLM first translates the ambiguous query into this explicit Ctask, establishing a non-negotiable reference frame (CR) and objective (CO). Strictly constrained by Ctask, the VLM then orchestrates a toolbox to perform deterministic geometric computation and derive the final answer.

tialAgent [49] and TIGeR [12] focus on translating the input query directly into an iterative sequence of tool executions. However, unconstrained planning process could lead to geometrically-flawed results, causing the agent to conflate “what to solve” with “how to solve it”.

free-form language into relational keypoint constraints and solves the constraints to generate final robot actions.

### 3. Methodology

As illustrated in Figure 2, we propose GeometricallyConstrained Agent (GCA), a training-free agentic paradigm designed for geometrically-constrained spatial reasoning. The core of GCA is the introduction of a formal task constraint Ctask that serves as a deterministic bridge between semantics and geometry. Section 3.1 defines this geometrically-constrained paradigm. Section 3.2 details the formal task constraint Ctask and its automated generation. Section 3.3 describes the subsequent constrained computation stage, which is strictly governed by this constraint. Finally, Section 3.4 discusses how GCA resolves the VLM’s semantic-to-geometric gap in spatial reasoning.

Constrained-Guided Reasoning. Constraint-guided reasoning, which involves restricting a search space by defining variables and the constraints governing them [38], has been adapted to manage the probabilistic nature of LLMs and VLMs. A primary application is in neuro-symbolic reasoning [1, 13, 14, 34, 40, 58], where the LLM is constrained to act as a translator, converting ambiguous natural language into a formal, verifiable representation. For example, LogicLM [34] leverage LLMs to translate NL problems into task-specific formalisms like formal logic. This constraint-guided reasoning can also be extended to planning [3, 16, 26, 35, 56]. Frameworks like LLM+P [26] uses an LLM to translate an NL problem into a formal PDDL format and then applies an optimal planner to generate the plan. Similarly, ReKep [16] employs a VLM to translate a

#### 3.1. Geometrically-Constrained Spatial Reasoning

Contemporary agentic frameworks often model reasoning as a generic, iterative policy. Those based on the ReAct

[Figure 58]

|lamp|
|---|

[Figure 59]

|sink|
|---|

+X

|glass|
|---|

+#. = ,-./ 0 1*2, − 0 45,6

+Z

= ,-.7ℎ

|book|
|---|

+Y

This gap necessitates a new formalism. We thus propose a novel formal task constraint Ctask specifically designed to capture the geometric nature of spatial reasoning. We define Ctask as a tuple containing two key sub-constraints: a single, non-negotiable Reference Frame Constraint (CR) that defines the coordinate system for answering the query, and an Objective Constraint (CO) that specify the objective to be measured within that frame.

[Figure 60]

[Figure 61]

[Figure 62]

Direction-based Frame

###### Object-based Frame

###### Camera-based Frame

“From viewpoint of Fig.1”

“When washing hands…”

“Oven is north of sink…” +#ℛ = %+,-.→012- = north

+#ℛ = +#3456

+#ℛ = −#+,-. = front

[Figure 63]

[Figure 64]

[Figure 65]

|oven|
|---|

|sink|
|---|

|sink|
|---|

|+Z|
|---|

|cam1|
|---|

+Z

|+X|
|---|

|+X|
|---|

|+X|
|---|

|+Z|
|---|

|+Y|
|---|

|+Y|
|---|

|+Y|
|---|

Reference Frame Constraint. Humans intuitively understand spatial language (e.g., “north of”) by grounding it within a specific coordinate system, namely a reference frame (R). In contrast, VLM failures often stem from ambiguity in this crucial grounding step, causing them to adopt flawed geometrically flawed plans [55] (e.g., defaulting to the camera’s viewpoint). The Fformalize stage addresses this ambiguity by requiring the VLM to first formally anchor R to the scene’s geometry.

- Figure 3. Reference Frame. Here, vsink→owen denotes a vector calculated by “normalize (Centroid(owen) − Centroid(sink))”.

framework [57], for example, can be defined by:

rt = A(q,v,T ,rt−1). (1)

In this framework, an agent A produces a response rt based on a query q, visual information v, a set of tools T , and its past history rt−1. This generic policy A is unconstrained, making it unreliable for high-stakes, deterministic domains like spatial reasoning. Recent work [12, 49] attempts to mitigate this by using external tools to constrain the final computation. However, they fail to constrain the VLM’s planning process. The VLM may still rely on its flawed spatial imagination in lossy semantic space to formulate plan, conflating “what to solve” with “how to solve it”.

We model all spatial queries as requiring a 3D cartesian coordinate system defined by an origin OR and three orthogonal basic vectors (xR,yR,zR). This system follows the OpenCV convention, where +zR points forward, +yR points down and +xR follows the right-hand rule. The agent’s task is to anchor R to one of three geometric primitives (see Figure 3) derived from the visual information:

- • Object-based Frame. R is defined by an object’s intrinsic coordinate system. For example, the query “when the user is washing hand” implies a reference frame defined by +zR = −zsink (one must face a sink to wash hand).
- • Camera-based Frame. R is defined by a specific camera’s viewpoint. For “from the viewpoint of Figure 1”, the reference frame is defined by +zR = +zcam1.
- • Direction-based Frame. R is defined by a vector connecting two locations. For “Owen is north

We solve this by replacing the generic policy A with a two-stage process. This paradigm is built on the formal task constraint Ctask, which functions as the architectural scaffolding to align the VLM’s asymmetric capabilities:

Ctask ← Fformalize(q,v), rt = Fcompute(Ctask,T ,rt−1).

(2)

In the Fformalize stage, the VLM acts as a semantic analyst, translating the ambiguous query q and visual information v into a formal, verifiable task constraint Ctask. This stage defines what to solve by establishing the necessary geometric scaffolding (e.g., the reference frame and target subjects). In the Fcompute stage, the VLM’s role shifts to a task solver. Governed by the constraint Ctask established in the Fformalize stage, the VLM iteratively executes tool calls to acquire necessary geometric data and perform final computations.

of sink”, the reference frame is defined by +zR = normalize(Centroid(owen) − Centroid(sink)) = north.

The output of this step is a human-readable and machineparsable definition of R, which becomes a non-negotiable constraint CR for all subsequent computation.

Objective Constraint. Concurrently, the agent identifies the objective O from the query. This constraint CO defines what must be measured relative to the established R. For the query, “Is chair to the west of toaster?”, the toaster defines CR, while the positional relationship between toaster and chair is the objective constraint CO.

#### 3.2. Task Constraint Formalization

###### 3.2.1. Constraint for Spatial Reasoning

###### 3.2.2. Automated Formalization via VLM

While existing constraint-guided reasoning paradigms leverage formalisms such as PDDL [26] or relational keypoint constraints [16], these constraints are insufficient for spatial reasoning. PDDL, for instance, excels at describing discrete, symbolic object states (e.g., “is on(A, B)”) but fundamentally lacks the geometric grammar to express the continuous, relative, and perspective-dependent nature of spatial queries (e.g., egocentric vs. allocentric viewpoints).

We exploit the VLM’s innate strength in semantic interpretation to generate Ctask automatically. Acting as a semantic analyst, the VLM performs qualitative interpretation, guided by the formal definitions of Ctask, to generate the Ctask = (CR,CO). This formal task constraint, generated by the VLM but grounded in geometry, serves as the geometrically sound contract for the Fcompute stage. In our

implementation, we enforce this architectural decoupling procedurally. The VLM executes the Fformalize stage and formalizes the Ctask before any computation begins.

#### 3.3. Constrained Geometric Computation

###### 3.3.1. Tool Integration and Code Generation

Once the formal task constraint Ctask = (CR,CO) is established, the VLM’s role shifts to a constrained task solver. This Fcompute stage then operates as a ReAct-style framework, consuming the Ctask as an immutable constraint. This execution is not a one-shot generation but an iterative, closed-loop process involving data acquisition, ambiguity resolution, and augmented computation.

Data Acquisition. Ctask dictates a set of geometric ingredients that the agent must acquire. For instance, as shown in Figure 2, to instantiate an object-based frame R defined by a sink, the agent must acquire the orientation of that sink. The Fcompute stage begins by generating a sequence of tool calls to parameterize the geometry, and acquire all variables necessary to instantiate Ctask.

Tool Orchestration and Ambiguity Resolution. The VLM is responsible for managing tool feedback and resolving ambiguity, ensuring the data acquired from tools correctly binds to the symbols in Ctask. For example, considering CO involves an object like “leftmost chair”, the perception tool returns several “chair” detections. The VLM analyzes this feedback (e.g., visualizing bounding boxes) and resolves the ambiguity by determining which object index correctly corresponds to the context (“leftmost”) specified. This closed-loop mechanism allows the agent to handle noisy tool outputs while ensuring the final computation remains strictly grounded in the intent of Ctask.

Knowledge-Augmented Code Generation. Once all variables in Ctask are bound to concrete geometric data, the agent invokes a code generator for the final computation. To prevent the coder from hallucinating incorrect formulas, we leverage a knowledge-augmented strategy, which functions analogously to a static Retrieval-Augmented Generation (RAG) [10, 20] system. Specifically, when invoking the code generator, the VLM specifies a high-level requirement and the necessary bound variables (e.g., object’s orientation). Instead of expecting the coder to generate complex geometric formulas from memory, our framework maintains a pre-prepared, fixed library of basic, verified geometric formulas. Based on the data types of the bound variables, the system automatically retrieves the relevant, fixed set of formulas (e.g., object’s local-to-world transformation formula) and injects them directly into the code generator’s context. This ensures the computation steps do not produce black-box guesses, but rather deterministic results, derived from a formally structured task and sound geometric principles. More details are provided in Section E.

###### 3.3.2. Toolbox

We equips the agent with perceptual and computation capabilities required to execute its geometrically-constrained reasoning flow in Fcompute, as shown in Figure 2. Detailed APIs for all tools are provided in Section C.2.

Geometry and Perception Tools. These tools are responsible for parameterizing the visual world. “3D Reconstruction” tool leverages foundational models like VGGT [44] to build a unified, high-fidelity 3D representation of the scene. This provides the geometric context required for complex scenarios. This category also contains a suite of 2D perception tools, such as “Object Detection” for open-vocabulary object detection, “Segmentation” for instance segmentation. Computation and Utility Tools. These tools operate on the data extracted by the perception tools and executes the final deterministic geometric computation. “Python Tool” is the core computational engine, which prompts the VLM to generate and execute Python code in a sandbox environment, using the knowledge-augmented strategy. This category also includes essential utilities (“Utility Tool”). For example, “project box to points” bridges 2D perception to 3D computation by converting 2D bounding boxes into corresponding 3D point clouds.

#### 3.4. Discussion

Our GCA decouples VLM’s spatial reasoning through the formal constraint Ctask, jointly addressing two core deficiencies in spatial reasoning.

Fformalize Solves Flawed Planning and Imagination. Directly solving an ambiguous query forces the VLM to plan and perform spatial imagination within its native lossy semantic space. This is a primary failure mode, as unconstrained planning can lead to geometrically flawed assumptions before any computation even begins. Our paradigm resolves this by reframing the problem. Leveraging VLM’s strength in qualitative semantic interpretation, the Fformalize stage transform the original spatial query into a deterministic mathematical problem with constraint, preventing the VLM to solve the query in its lossy semantic space directly. Fcompute Solves Flawed Execution and Computation. In this stage, the VLM acting as the task solver, orchestrating external tools to execute the plan. Crucially, its entire reasoning and execution process is bound by the formal task constraint Ctask generated in Fformalize. This ensures that all subsequent high-precision computations are executed strictly within the deterministic, geometrically sound constraint, effectively bridging the semantic-to-geometric gap.

### 4. Experiments

#### 4.1. Experimental Setup

Implementation Details. GCA is implemented as a training-free agentic paradigm, requiring no model fine-

- Table 1. Experimental Results on Several Spatial Reasoning Benchmarks. The best and second best results are shown in bold and underlined, respectively. “Avg.” denotes the average of overall accuracy across all benchmarks. More details about these benchmarks’ subcategory (e.g., “PR.”) are provided in Appendix.

MMSI-Bench MindCube-tiny OmniSpatial SPBench CV-Bench

Avg. PR. Attr. Mot. MSR All Rot. Ard. Amg. All Dyn. Pers. All SI MV All 2D 3D All

###### Baseline Foundation VLMs

Qwen3-VL-Thinking [36] 33.7 40.0 23.3 31.8 32.6 87.0 47.3 35.0 47.3 60.5 43.9 51.0 51.9 61.2 54.1 81.9 92.6 86.8 54.4 GLM-4.5V [15] 35.6 36.9 29.3 30.3 33.8 60.0 25.5 42.2 39.6 58.6 47.2 52.1 50.0 55.1 51.3 80.7 91.6 85.6 52.5 GPT-4o [17] 28.0 32.3 36.0 30.8 30.3 33.5 35.0 37.2 35.8 58.7 46.2 51.5 42.4 48.3 43.8 69.4 84.9 76.5 47.6 Gemini-2.5-Pro [6] 39.0 36.2 33.3 34.3 36.9 89.5 54.5 48.8 57.5 70.7 44.6 55.8 55.6 58.3 56.3 81.2 92.5 86.3 58.5

###### Training-based Spatial VLMs

SpatialLLM [30] 24.5 23.1 22.7 30.8 25.3 34.0 26.8 33.0 31.1 59.6 42.9 49.5 32.2 26.4 30.7 51.3 78.6 64.5 40.2 Spatial-MLLM [48] 28.5 25.4 18.0 26.3 26.1 33.8 34.5 28.3 32.1 37.2 42.1 40.0 52.0 52.0 52.0 59.5 63.3 61.2 42.3 SpatialLadder [22] 30.3 23.3 16.0 21.2 25.4 30.5 39.8 47.8 42.3 46.5 43.1 44.5 70.2 70.9 70.3 72.4 74.9 73.7 51.2 SpaceR [33] 29.1 29.4 21.9 22.5 26.9 29.8 30.0 26.8 28.3 53.5 40.5 46.0 48.6 59.4 51.1 74.1 77.4 75.6 45.7 Video-R1 [8] 30.5 25.4 22.0 26.8 27.8 30.0 30.5 41.3 35.8 50.0 44.2 46.7 44.8 40.7 43.8 73.5 74.7 74.0 45.6 RoboBrain-2.0 [41] 28.9 28.8 22.5 28.0 28.9 29.7 35.8 45.2 39.6 49.4 42.2 45.2 49.1 46.8 48.5 77.1 90.7 83.4 49.1 VILASR [50] 35.9 26.0 21.0 23.2 29.8 34.4 25.7 29.4 29.1 37.5 42.2 40.2 50.2 57.6 51.9 75.7 77.7 76.6 45.5 VLaser [52] 29.8 26.9 26.0 18.9 27.3 31.5 24.8 38.2 32.6 39.1 42.6 41.1 53.2 69.2 56.9 79.9 87.8 83.6 48.3

###### Tool-Integrated Spatial Agents

TIGeR [12] 29.1 27.7 26.0 25.8 27.8 33.0 28.3 26.7 28.3 52.9 45.7 49.8 48.7 38.8 46.3 75.2 95.7 84.5 47.3 GCA (ours) 52.8 45.0 44.7 38.0 47.6 82.0 61.8 59.8 64.2 73.6 58.6 65.1 61.7 61.9 61.8 83.6 90.8 86.9 65.1

tuning. It centers on a VLM responsible for both stages of our paradigm: acting as the semantic analyst to generate the Ctask in the Fformalize stage, and as the task solver to manage a suite of off-the-shelf foundation models for perception and computation [27, 37, 44, 45, 47]. For our primary experiments, we utilize Qwen3-VL-Thinking [36] as the central VLM. To assess the paradigm’s generalizability, we also evaluate other leading VLMs in our ablation studies, including GLM-4.5V [15], GPT-4o [17], and etc. All open-source VLMs are deployed using the vLLM inference engine [19] for efficiency. The agent’s architecture is built using Ray [32] for concurrent tool execution and LangGraph for robust state management.

Evaluation Benchmarks and Counterparts. We conduct comprehensive experiments on several spatial reasoning benchmarks. As our current toolbox is primarily designed for image-based inputs, we focus on evaluations that test complex spatial logic from single and multiple images, including MMSI-Bench [55], MindCube-tiny [59], OmniSpatial (Perspective Taking + Dynamic Reasoning) [18], SPBench [22] and CV-Bench [43]. For all benchmarks, we report both overall accuracy (%) and subcategory accuracy (%). We compare our paradigm against several counterparts, including baseline foundation VLMs [6, 15, 17, 36], training-based methods [8, 22, 30, 33, 41, 48, 50, 52] and tool-integrated agents [12].

#### 4.2. Main Results

SOTA Performance. As shown in Table 1, GCA establishes a new state-of-the-art across a wide range of spatial reasoning benchmarks, achieving an average accuracy of 64.8%. Our geometrically-constrained paradigm sur-

passes the strongest foundation VLM baseline (Gemini-2.5Pro [6] by 12%) and demonstrates a massive lead over other training-based (e.g., SpatialLadder [22] by 27%) or agentic approaches (e.g., TIGeR [12] by 38%). These results strongly validate that our strategy, centered on the Ctask, successfully bridges the VLM’s semantic-to-geometric gap.

Effectiveness on Challenging Benchmarks. The advantage of our constrained paradigm is most pronounced on complex, multi-step spatial reasoning benchmarks. For example, on MMSI-Bench, the performance of even SOTA foundation VLMs remain severely limited. Considering its 4-choice questions, most counterparts perform near the 25% random-guess threshold. In contrast, GCA achieves an overall accuracy of 47.6%, surpassing the strongest VLM baseline (Gemini-2.5-Pro) by a 28% relative improvement. A similar trend is evident on other challenging benchmarks like MindCube-tiny, where GCA (64.2%) also significantly outperforms the top baselines. This superior performance stems directly from our paradigm. The introduction of Ctask prevents the VLM from defaulting to flawed semantic shortcuts or falling into a lossy spatial imagination.

Generalizability Across Benchmarks. Our training-free paradigm also demonstrates superior generalizability compared to training-based specialists, which often suffer from biases inherent to their training data. For example, SpatialLadder [22] is fine-tuned on data originating from the same source as the SPBench, leading to a high in-domain score of 70.3%. However, its performance on out-of-domain benchmarks is suboptimal, where GCA consistently outperforms it, often by a margin of ∼20 points. A similar bias affects TIGeR [12]. While its tools theoretically support multiview processing, the model is primarily trained on single-

Impact of Task Constraint

OverallAccuracyonMMSI-Bench

49.5

50

47.6

+49%

+46%

45

+34%

41.9

+19%

40.1

40

35

32.6

30

25

Baseline (CoT-Only)

Tool (Uncon.)

Tool (Prompt)

Ours Oracle (Anno.)

Figure 5. Ablation Study on Generalizability across Different VLMs. Our GCA achieves an average of 37% relative performance improvement across all tested foundation VLMs.

- Figure 4. Ablation Study on Formalization. We compare our method in against several baselines: (1) no tool integration (“Baseline (CoT-Only)”), (2) unconstrained tool integration with (“Tool (Prompt)”) or without (“Tool (Uncon.)”) hints, (3) using a humanannotated Ctask (“Oracle (Anno.)”).

to first establish what to solve before determining how to solve it. Furthermore, we explore the theoretical upper bound using a human-annotated oracle formalization (“Oracle (Anno.)”). The gap between our method (47.6%) and oracle (49.5%) is relative small. As revealed in Section 4.4, the Fformalize stage achieves ∼70% accuracy, confirming the formalization task is well within the VLM’s capabilities.

image tasks. Consequently, it performs well on singleimage benchmarks like CV-Bench but fails on multi-view benchmarks such as MMSI-Bench and MindCube. GCA, in contrast, is not compromised by these training priors and leverages its multi-view tools as dictated by the problem. This demonstrates that our GCA, which forces the VLM to derive a geometrically sound task constraint for each new problem, provides a more robust and generalizable pathway to spatial reasoning.

###### 4.3.2. Generalizability Across VLMs

We assess the generalizability of our GCA paradigm by applying it to several leading foundation VLMs, including GLM-4.5V [15], GPT-4o [17], and Gemini-2.5-Pro [6]. As shown in Figure 5, GCA proves to be a highly generalizable architectural solution, substantially enhancing the spatial reasoning capabilities of every VLM tested compared to their CoT-only baselines. We observe that the magnitude of this enhancement appears to correlate strongly with the VLM’s inherent agentic proficiency and their baseline spatial reasoning capability. It is most evident that Gemini-2.5-Pro, which holds the strongest CoT-only baseline on MMSI-Bench (36.9%), also achieves the most dramatic gain (+49%), rising to 55.0%. On the other hand, the improvement on GPT-4o, while significant, is more modest (+19%). We attribute it to its suboptimal agentic reasoning capability and coding skills. Through introduction of formal task constraint Ctask, our paradigm serves as a catalyst, successfully unlocking and guiding the VLM’s powerful execution engine towards the robust spatial reasoning across a diverse set of SOTA models.

#### 4.3. Ablation Study

In this section, we conduct extensive ablation studies to dissect the GCA paradigm and validate its core design. Our analysis aims to answer four critical questions. (1) How essential is the formal task constraint Ctask? (2) How generalizable is the GCA paradigm across different VLMs? (3) What is the contribution of each system component?

- 4.3.1. Formalization Analysis We first investigate the necessity and impact of our core

contribution, the Ctask constraint, by comparing our method against different reasoning strategies in Figure 4. The results strongly confirm our central hypothesis. Simply prompting the VLM to “pay attention to the reference frame and objective in the query” (“Tool (Prompt)”) only yields a negligible improvement on unconstrained tool integration. This empirically suggests that the VLM’s unconstrained planning process remains fundamentally flawed and unreliable, even when weakly guided by hints. In comparison, the introduction of our formal Ctask constraint (“Ours”) delivers a substantial performance boost, far surpassing all unconstrained methods. This demonstrates that a deterministic and verifiable constraint is essential for bridging the VLM’s semantic-to-geometric gap, as it forces the VLM

###### 4.3.3. Component Contribution

We quantify the importance of each component in the GCA, as presented in Table 2. This analysis reveals improvements in two distinct parts. First, building a standard tool-integrated agent by adding tool integration (+4.2 points), knowledge-augmented code generation (KACG, +1.9 points), and visual feedback (+1.4 points) provides

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

###### （a) Failure Case Distribution (b) Task Formalization

###### (e) Reconstruction

###### (c) Detection

###### (d) Orientation

Dim Lighting

No Textual Input

Implication Multiple Images Complex Semantics

Distracting Objects Complex Referring Severe Occlusion

Small Objects

Rot.-trans. Ambiguity

Others 21%

Task Formalization 30%

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

“each shot after rotating 60 degrees”

In top-down view, “down” imply gravity down (+Z), not camera down (+Y)

|chair|
|---|

[Figure 78]

[Figure 79]

|mural|
|---|

Reconstruction 8%

|+Z|
|---|

[Figure 80]

Detection 5% Orientation 11%

|+X|
|---|

[Figure 81]

PythonTool 25%

|+X|
|---|

|+Z|
|---|

wrong order

|+Y|
|---|

|+Y|
|---|

Figure 6. Error Attribution and Failure Cases. We provide a detailed error attribution analysis to identify the main failure modes within the VLM’s reasoning trajectory.

- Table 2. Ablation Study on Each Component in GCA. Here, “KACG” denotes applying knowledge-augmented code generation, and “Feedback” denotes applying the VLM to manage tool feedback and resolve ambiguity.

Errors in Fcompute. The remaining 70% of errors occur during Fcompute stage. Perception failures (∼24%) are a major bottleneck, particularly in “Reconstruction” and “Orientation”. A typical reconstruction failure, shown in Figure 6 (e), is caused by the inability of the underlying VGGT [44] to accept textual input. The query’s textual input, “each shot after rotating 60 degrees” provides a deterministic rotational sequence. However, the VGGT model, which cannot accept this textual input, parameterize the scene incorrectly, resulting in the “wrong order” of cameras and a flawed geometric foundation. Errors from “Python Tool” (25%) are also significant, often stemming from forgotten coordinate transformations or lacking nuanced problem-solving logic, such as identifying a principal direction. Besides, “Other” (21%) errors capture issues like incorrect parameter passing between tools, exhausting the predefined budget (e.g., a maximum of 15 turns), and etc.

Tool Integration KACG Feedback Ctask MMSI-Bench

32.6 ✓ 36.8 ✓ ✓ 38.7 ✓ ✓ ✓ 40.1

✓ ✓ ✓ ✓ 47.6

a cumulative +7.5 points gain over the CoT-only baseline. The second part, the introduction of Fformalize, brings an additional massive improvement, increasing the overall accuracy by +7.5 points. This result strongly validates that constraining the VLM’s planning via a formal Ctask is essential to prevent flawed reasoning within its lossy semantic space.

### 5. Conclusion

#### 4.4. Error Attribution and Failure Cases

In this work, we introduce GCA, a training-free agentic paradigm designed to bridge the VLM’s semantic-togeometric gap in spatial reasoning. We address it through leveraging a formal task constraint, transforming the ambiguous spatial query into a deterministic mathematic problem with constraints, preventing the VLM reasoning about the geometric details within its lossy semantic space. As demonstrated experimentally, GCA establishes a new stateof-the-art on multiple challenging spatial reasoning benchmarks, showcasing a effective and generalizable pathway for robust spatial reasoning.

A key advantage of GCA paradigm is its verifiable and interpretable nature, which allows us to trace the reasoning pathway and perform detailed error attribution. As shown in Figure 6 (a), this analysis pinpoints the current bottlenecks, attributing failures to either the VLM’s initial formalization or the subsequent tool orchestration.

Errors in Fformalize. Failures in the initial Fformalize stage account for 30% of all errors. Given this is the first step of the paradigm, it indicates the VLM achieves approximately 70% accuracy in correctly formalizing the task constraint Ctask. A deeper analysis reveals these failures primarily lie in challenging cases involving complex semantics, ambiguity in multiple images, or ignored implications. For instance, as shown in Figure 6 (b), when asked about a topdown view, the VLM fails to grasp the query’s implication that “down” referred to the direction of gravity, defaulting instead to “camera down” and establishing an incorrect reference frame.

Limitations and Future Prospects. The GCA paradigm, involving iterative tool calls and VLM interactions, is computationally more costly than simple end-to-end CoT reasoning. However, this trade off yields a more robust and verifiable reasoning pathway. Furthermore, we believe the structured outputs from our Fformalize and Fcompute stages can serve as a valuable source of supervision, such as a process reward in reinforcement learning, for training more ef-

ficient end-to-end spatial VLMs in the future. Besides, our current toolbox is primarily designed for image-based spatial reasoning. A key direction for future work is to extend this geometrically-constrained framework by incorporating tools for temporal reasoning and motion tracking, thereby addressing a broader range of spatial intelligence tasks.

### References

- [1] Bikram Pratim Bhuyan, Amar Ramdane-Cherif, Ravi Tomar, and TP Singh. Neuro-symbolic artificial intelligence: a survey. Neural Computing and Applications, 36(21):12809– 12844, 2024. 3
- [2] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Crist´obal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Li Fei-Fei. Hourvideo: 1-hour video-language understanding. Advances in Neural Information Processing Systems, 37:53168–53197, 2024. 1
- [3] Zeren Chen, Zhelun Shi, Xiaoya Lu, Lehan He, Sucheng Qian, Zhenfei Yin, Wanli Ouyang, Jing Shao, Yu Qiao, Cewu Lu, et al. Rh20t-p: A primitive-level robotic dataset towards composable generalization agents. arXiv preprint arXiv:2403.19622, 2024. 3
- [4] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024. 2
- [5] Jae-Woo Choi, Youngwoo Yoon, Hyobin Ong, Jaehong Kim, and Minsu Jang. Lota-bench: Benchmarking languageoriented task planners for embodied agents. In The Twelfth International Conference on Learning Representations, 2024. 12
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6, 7, 14
- [7] Gunnar Farneb¨ack. Two-frame motion estimation based on polynomial expansion. In Scandinavian conference on Image analysis, pages 363–370. Springer, 2003. 14
- [8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. Advances in Neural Information Processing Systems, 2025. 1, 2, 6
- [9] Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, and Yu Qiao. Drive like a human: Rethinking autonomous driving with large language models. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW), pages 910–919. IEEE, 2024. 1
- [10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1), 2023. 5

- [11] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 1
- [12] Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, and Shanghang Zhang. Tiger: Tool-integrated geometric reasoning in vision-language models for robotics. arXiv preprint arXiv:2510.07181, 2025. 1, 2, 3, 4, 6
- [13] Yilun Hao, Yang Zhang, and Chuchu Fan. Planning anything with rigor: General-purpose zero-shot planning with llm-based formalized programming. In The Thirteenth International Conference on Learning Representations, 2024. 3
- [14] Pascal Hitzler and Md Kamruzzaman Sarker. Neurosymbolic artificial intelligence: The state of the art. IOS press, 2022. 3
- [15] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 6, 7
- [16] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. In Conference on Robot Learning, pages 4573–4602. PMLR, 2025. 3, 4, 12
- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 2, 6, 7
- [18] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 1, 2, 6, 11, 12, 15

- [19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023. 6, 14
- [20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020. 5
- [21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025. 2
- [22] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive train-

- ing for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025. 1, 2, 6, 11, 12, 15
- [23] Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Erran Li Li, Ruohan Zhang, et al. Embodied agent interface: Benchmarking llms for embodied decision making. Advances in Neural Information Processing Systems, 37: 100428–100534, 2024. 12
- [24] Songtao Li and Hao Tang. Multimodal alignment and fusion: A survey. arXiv preprint arXiv:2411.17040, 2024. 1
- [25] Zefu Lin, Rongxu Cui, Chen Hanning, Xiangyu Wang, Junjia Xu, Xiaojuan Jin, Chen Wenbo, Hui Zhou, Lue Fan, Wenling Li, et al. Embodiedcoder: Parameterized embodied mobile manipulation via modern coding model. arXiv preprint arXiv:2510.06207, 2025. 2, 12
- [26] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. Llm+p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023. 3, 4
- [27] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer,

- 2024. 6, 13

[28] Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, and Jing Shao. Is-bench: Evaluating interactive safety of vlm-driven embodied agents in daily household tasks. arXiv preprint arXiv:2506.16402,

- 2025. 12

- [29] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. Advances in neural information processing systems, 37:68803–68832, 2024. 2
- [30] Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17249–17260, 2025. 1, 2, 6
- [31] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 1
- [32] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I Jordan, et al. Ray: A distributed framework for emerging {AI} applications. In 13th USENIX symposium on operating systems design and implementation (OSDI 18), pages 561–577, 2018. 6, 14
- [33] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 1, 2, 6
- [34] Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. Logic-lm: Empowering large language models with symbolic solvers for faithful logical

- reasoning. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 3
- [35] Mingjie Pan, Jiyao Zhang, Tianshu Wu, Yinghao Zhao, Wenlong Gao, and Hao Dong. Omnimanip: Towards general robotic manipulation via object-centric interaction primitives as spatial constraints. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17359–17369,

2025. 3

- [36] QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. https://qwen.ai/blog?id= 99f0335c4ad9ff6153e517418d48535ab6d8afef& from=research.latest- advancements- list,

2025. 2, 6, 13, 14

- [37] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2024. 6, 13
- [38] Stuart Russell, Peter Norvig, and Artificial Intelligence. A modern approach. Artificial Intelligence. Prentice-Hall, Egnlewood Cliffs, 25(27):79–80, 1995. 3
- [39] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025. 2
- [40] Oren Sultan, Eitan Stern, and Dafna Shahaf. Towards reliable proof generation with llms: A neuro-symbolic approach. arXiv preprint arXiv:2505.14479, 2025. 3
- [41] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, et al. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025. 1, 6
- [42] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, XianPeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. In Conference on Robot Learning, pages 4698–4726. PMLR, 2025. 1
- [43] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 2, 6
- [44] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 5, 6, 8, 13
- [45] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025. 6
- [46] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong

- Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546, 2025. 13
- [47] Zehan Wang, Ziang Zhang, Tianyu Pang, Chao Du, Hengshuang Zhao, and Zhou Zhao. Orient anything: Learning robust object orientation estimation from rendering 3d models. In Forty-second International Conference on Machine Learning, 2024. 6, 13
- [48] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. Advances in Neural Information Processing Systems, 2025. 1, 2, 6
- [49] Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, and Weidi Xie. Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv preprint arXiv:2505.17012, 2025. 1, 2, 3, 4
- [50] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. Advances in Neural Information Processing Systems, 2025. 1, 2, 6
- [51] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024. 1
- [52] Ganlin Yang, Tianyi Zhang, Haoran Hao, Weiyun Wang, Yibin Liu, Dehui Wang, Guanzhou Chen, Zijian Cai, Junting Chen, Weijie Su, et al. Vlaser: Vision-language-action model with synergistic embodied reasoning. arXiv preprint arXiv:2510.11027, 2025. 1, 2, 6
- [53] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 2, 11, 12
- [54] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. In Forty-second International Conference on Machine Learning, 2025. 1, 12
- [55] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multiimage spatial intelligence. arXiv preprint arXiv:2505.23764,

2025. 1, 2, 4, 6, 11, 12, 14

- [56] Zhutian Yang, Caelan Garrett, Dieter Fox, Tom´as LozanoP´erez, and Leslie Pack Kaelbling. Guiding long-horizon task and motion planning with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16847–16853. IEEE, 2025. 3
- [57] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022. 4

- [58] Xi Ye, Qiaochu Chen, Isil Dillig, and Greg Durrett. Satlm: Satisfiability-aided language models using declarative prompting. Advances in Neural Information Processing Systems, 36:45548–45580, 2023. 3
- [59] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025. 1, 2, 6, 11, 12, 14
- [60] Songsong Yu, Yuxin Chen, Hao Ju, Lianjie Jia, Fuxi Zhang, Shaofei Huang, Yuhan Wu, Rundi Cui, Binghao Ran, Zaibin Zhang, et al. How far are vlms from visual spatial intelligence? a benchmark-driven perspective. arXiv preprint arXiv:2509.18905, 2025. 1, 2
- [61] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, YuGang Jiang, et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with longhorizon reasoning tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11142– 11152, 2025. 12
- [62] Filippo Ziliotto, Tommaso Campari, Luciano Serafini, and Lamberto Ballan. Tango: Training-free embodied ai agents for open-world tasks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24603–24613,

2025. 1, 2

### A. Spatial Task Constraint

As introduced in the main paper, the core of GeometricallyConstrained Agent (GCA) paradigm is the formal task constraint Ctask. It serves as a deterministic bridge to resolve the fundamental semantic-to-geometric gap, effectively decoupling the VLM’s role into a semantic analyst and a constrained task solver. Recall that Ctask is formally defined as a tuple containing two key sub-constraints: the Reference Frame Constraint (CR), which defines the coordinate system, and the Objective Constraint (CO), which specifies what to measure within that frame. We guide the Visual Language Model to automatically generate Ctask (for specific prompts used in GCA, refer to Section E). In this section, we first elaborate on the universality of this constraint across different spatial task domains.

#### A.1. Universality of Ctask in Spatial Tasks

The Ctask is not a rigid definition fixed to a single problem type, but rather a general principle for a wide range of spatial tasks with semantic-to-geometric gap. The core idea is to leverage the VLM’s semantic advantage to formalize the most significant geometric ambiguity of the task. The nature of this ambiguity shifts depending on the task domain. Spatial Understanding and Reasoning. For spatial reasoning tasks, such as those evaluated in the main paper [18, 22, 53, 55, 59], the objective is typically simple

and explicitly stated in the query (e.g., “what is the relative position?” or “which object is wider?”). Therefore, the objective constraint CO is often trivial to formalize. The primary geometric ambiguity lies in the reference frame constraint CR. A query like “where is the table relative to you?” is unsolvable until the reference frame (“you”) is geometrically grounded. GCA’s formalization of CR (e.g., objectbased, camera-based, direction-based frames) is specifically designed to resolve this ambiguity.

Robotic Manipulation and Interaction. Conversely, for robotic manipulation and interaction tasks [5, 23, 28, 54, 61], the reference frame constraint is often simple. The frame is typically the robot’s egocentric perspective or a fixed world frame aligned with the primary camera. The geometric ambiguity shifts entirely to the objective constraint CO. A command like “pour tea into the cup” has a trivial CR but a highly complex CO. The objective is not a simple measurement but a complex, multi-stage procedure involving affordances, contact points, and trajectories. For example, ReKep [16] tackles the manipulation by instructing the VLM to formalize the objective CO as a set of Relational Keypoint Constraints. These constraints CO are literally generated as cost functions written by VLM in Python that map 3D keypoints to a numerical cost. The cost functions are then passed to an inverse kinematics solver (Fcompute) to find the optimal robot action. Similarly, EmbodiedCoder [25] formalizes the CO as an executable program. The VLM is prompted to first generate code for geometric parameterization, fitting point clouds to functional primitives like a rectangle and a hinge axis for a door. Through generating a precise, parametric motion that conforms to the geometric shape just defined, the Python interpreter (Fcompute) simply executes the code to produce the final waypoints.

These works are not in conflict with our proposed task constraint Ctask. Instead, they can serve as complementary examples of its core principle. They demonstrate how the CO for complex manipulation and interaction can be formalized as code, cost functions, or geometric constraints. These constraints are then passed to a solver, just as GCA proposes. This confirms the universality of the Ctask: whether the ambiguity lies in the reference frame or the objective, the first and most critical step is to use the VLM’s semantic strength to formalize a deterministic, geometrically-sound constraint to bridge the semantic-to-geometric gap.

#### A.2. Generalizability of R Definintion

We define three types of reference frame R in GCA: objectbased, camera-based and direction-based reference frame, providing a robust and flexible framework. We find that these categories are sufficient to cover the vast majority of static spatial reasoning queries encountered in existing benchmarks [18, 22, 55, 59]. As spatial reasoning advances

Table 3. Ablation Study on Task Constraint.

Tool Integration Ref. CR Obj. CO MMSI-Bench

✓ ✓ ✓ 47.6 ✓ ✓ 46.4 ✓ ✓ 41.0 ✓ 40.1

✓ ✓ 33.5

toward more complex, dynamic, and abstract scenarios, we identify key limitations and challenges for the current implementation of CR. These represent important avenues for future research.

Dynamic and Time-Varying Reference Frame. A significant challenge arises in video-based spatial reasoning [53], particularly in tasks involving continuous navigation or long-horizon agent actions. For example, an instruction like, “Move forward to the right first, then move backward to the right, and finally turn left,” involves a CR that is continuously changing at each step, contingent on the agent’s previous state. Solving this requires extending the CR formalization to become a time-dependent function CR(t), capable of tracking and updating the agent’s pose and orientation throughout a sequence.

Frames from Abstract Concepts. This challenge emerges when the reference entity is not a rigid, easily-definable object. A query like “the living room is south of the kitchen” poses a significant problem. It is often impossible to compute a meaningful direction vector between the geometric centroids of two abstract areas or regions. Currently, GCA relies on proxies, for example, if such direction (from kitchen to living room) aligns with the camera’s view from the background to the foreground, we might substitute −Zcam as the direction vector. This workaround, however, can introduce cumulative errors and lacks generalizability. Future work could explore novel methods to address this. One promising direction is to empower the VLM to directly annotate the reference frame, i.e., outputting two points in the image whose corresponding 3D vector defines the abstract direction.

### B. More Ablation Studies

To further explore the proposed formal task constraint Ctask and the stability of GCA, we present three additional ablation studies. These experiments are designed to (1) precisely quantify the relative importance of the reference frame constraint (CR) versus the objective constraint (CO) for spatial reasoning tasks, (2) validate the constraint’s effectiveness in improving the VLM’s internal, tool-free reasoning, and (3) evaluate the stability and robustness of the GCA paradigm.

Importance of CR and CO in Spatial Reasoning. To validate our claim in Section A.1 that CR represents the primary geometric ambiguity in spatial reasoning tasks, we conduct a detailed ablation on the sub-components of Ctask. As shown in Table 3, removing the objective constraint (CO) results in a minor 1.2 point performance drop. This demonstrates that for spatial reasoning queries, the objective is often simple and clearly stated, allowing the task solver to infer it from the query during the Fcompute stage. In contrast, removing the reference frame constraint (CR) causes a 6.6 point performance drop. This suggests that CR is the most critical component in spatial reasoning, as it resolves the core geometric ambiguity of “from where” that VLMs cannot solve in their lossy semantic space.

Ctask without Tool Integration. The Ctask is not a prompt that can fix the VLM’s internal spatial reasoning. Instead, it unlocks the VLM’s agentic reasoning capability and coding skills by constraining the subsequent computation stage (Fcompute). This is confirmed by the results in Table 3, where we compare GCA with a VLM that receives Ctask as the prompt but relies solely on chain-of-thought (CoT) reasoning. With the constraint as a textual hint, it only yields a negligible 0.9 point improvement. Even when told the reference frame and objective, the VLM still cannot bypass the internal flawed spatial imagination and high-precision computation in its lossy semantic space.

Stability and Robustness Analysis. A key consideration for any agentic framework, especially one involving multiple VLM calls and tool interactions, is the stability of its results. The probabilistic nature of VLMs could potentially lead to high variance in final performance. To assess the robustness of GCA, we conduct N = 10 independent evaluation runs on the complete MMSI-Bench dataset, using the same Qwen3-VL-Thinking [36] for each run. All settings are kept identical as in the Table 1 (main paper). The mean accuracy and the standard deviation across all 10 runs is 47.6 ± 0.3. The results demonstrate a very low standard deviation, indicating that the GCA framework is highly stable. This stability is a direct benefit of our core design: by forcing the VLM to first generate a deterministic formal task constraint Ctask, we significantly reduce the stochasticity and ambiguity in the subsequent Fcompute stage. The task solver operates within the non-negotiable geometric bounds defined by Ctask, leading to a consistent and verifiable reasoning pathway.

### C. More Implementation Details

#### C.1. Visual Foundation Models

We deploy several Visual Foundation Models (VFMs) for agent to parameterize the visual world, facilitating the deterministic Fcompute stage constrained by Ctask.

- • VGGT [44]. Visual Geometry Grounded Transformer

(VGGT) is a large feed-forward model that infers key 3D attributes from one or multiple images. It predicts camera parameters, point maps, and depth maps for all input views. Within GCA, VGGT serves as the primary geometry parameterization engine for 3D reconstruction.

- • MoGe-2 [46]. The Monocular Geometry (MoGe) estimation model is designed to recover 3D point maps with metric scale from a single image. It achieves this by decoupling the problem, predicting both an affine-invariant point map and a separate global scale factor. GCA leverages this unique capability to derive the correct real-world scale for the scene.
- • GroundingDINO [27]. GroundingDINO is an open-set object detector that combines a transformer-based detector with grounded language pre-training. This architecture enables it to detect arbitrary objects specified by natural language, such as category names or referring expressions. It serves as a specialized detection tool in GCA.
- • SAM-2 [37]. SAM-2 is a foundation model for promptable visual segmentation in both images and videos. It generalizes the original SAM by incorporating a streaming memory architecture to handle temporal data. GCA uses SAM-2 as a bridge connecting pixels and boxes, allowing us to extract object point clouds from the VG-GT output based on the boxes.
- • Orient Anything [47]. Orient Anything is trained on rendered 3D models to estimate the 3D orientation of an object from a single, free-view image. It predicts the object’s azimuth, polar, and rotation angles relative to the camera. This capability is crucial for GCA to construct a object-based reference frame.

Note that these foundation models are not provided directly to the agent. Instead, we wrap them and offer some abstract tool interfaces as APIs for invocation (see Section C.2).

#### C.2. Tool Interfaces

The GCA agent’s Fcompute stage is driven by a discrete set of 8 exposed tool APIs. These APIs form the agent’s action space, encapsulating the underlying VFMs.

- • reconstruct. It ingests one or more images and produces a comprehensive 3D reconstruction. Internally, it leverages VGGT and automatically selects the optimal reconstruction strategy. If multiple, non-static images are provided, it first consults the VLM to identify common static objects for alignment. The output includes the 3D world points, camera extrinsics, and intrinsics.
- • detect. It detects target objects in a single image based on a text prompt. For capable VLMs like Qwen3-VLThinking [36], we directly instruct the VLM itself to locate the target object through prompts. Otherwise, we use GroundingDINO as the detector. It returns the bounding boxes and corresponding labels.

- • project box to 3d points. It takes a 2D bounding box and projects it into the 3D world coordinate system defined by the VGGT model output. Internally, it first uses SAM-2 to convert the bounding box into a precise pixel mask, then filters the points using this mask.

- • predict obj pose. It computes the 6-DoF semantic pose of an object, which is essential for establishing an object-based reference frame. This tool first calls project box to 3d points to find the object’s 3D centroid, and then calls Orient Anything to determine its 3D orientation. It then combines these to return the final object-to-world transformation matrix.

- • estimate scale. This tool is called when metric measurements (e.g., “meters”, “feet”) are required. It aligns MoGe-2’s metric depth with the VGGT model’s relative depth prediction to compute a single scale factor that converts the entire reconstruction into meters.

- • ocr. It performs optical character recognition (OCR) on an image using the EasyOCR library. It returns a list of recognized texts and their bounding boxes.
- • analyze motion. It analyzes pixel-level motion between two sequential images using a Farneback optical flow algorithm [7]. It is used to infer subtle camera movements that may be too small for full 3D reconstruction.

- • code. This is the agent’s primary computation engine. It generates and executes Python code within a sandbox environment. It tasks a set of context variables (e.g., poses, points) and an natural language description (e.g., request and description of the variables) as input. The code is generated using a knowledge-augmented strategy, where relevant geometric formulas are injected into the prompt, ensuring the computation is both deterministic and sound.

#### C.3. Agentic Framework

The GCA paradigm is implemented as a high-throughput, modular system. The core VLM deployment and the perception tool suite are physically decoupled to ensure scalability and robustness.

System Backend and State Management. The entire system is built using Ray [32] and LangGraph. LangGraph is used to define and manage the agent’s state and orchestrate the two-stage, graph-based reasoning flow (i.e., Fformalize followed by the Fcompute loop).

Tool Suite and VFMs Deployment. The perceptual and computational tools (listed in Section C.2) are encapsulated as independent Ray Serve actors. This microservice architecture allows GCA to make concurrent perception requests (e.g., running reconstruct and detect in parallel), enabling high parallelism and automatic scaling. This entire tool suite is deployed on 2 NVIDIA A100 GPUs.

VLM Roles and Deployment. In GCA, a single VLM fulfills the three distinct roles within the GCA paradigm:

- • Semantic Analyst. In the Fformalize stage, it interprets the query and visual context to generate the formal Ctask.
- • Tool Orchestrator. In the Fcompute stage, it manages the ReAct-style tool call loop, resolves ambiguities, and generates natural language descriptions for the coder.
- • Coder. It generates Python code for the code tool. The VLM deployment is separate from the tool suite. For open-source models (e.g., Qwen3-VL-Thinking [36]), we use vLLM [19] for efficient, high-throughput inference on 8 NVIDIA A100 GPUs. For closed-source models (e.g., Gemini-2.5-Pro [6]), we access them via their standard commercial APIs. We employ different sampling parameters based on the VLM’s role. For the semantic analyst and tool orchestrator roles, which require reasoning and flexibility, we use TEMPERATURE=0.6 and TOP P=0.95. For the coder role, which demands deterministic and reliable output, we set TEMPERATURE=0.0. All roles use MAX TOKENS=32768.

D. Evaluation Benchmark Details

We evaluate GCA on multiple challenging spatial reasoning benchmarks. This section provides a detailed description of each benchmark and its subcategories, corresponding to the results presented in Table 1 of the main paper.

- D.1. MMSI-Bench

MMSI-Bench [55] is a comprehensive benchmark designed to evaluate a VLM’s ability to perform spatial reasoning by integrating information from multiple, distinct images. It is organized into four distinct subcategories:

• PR. (Positional Relationship). This subcategory evaluates the model’s ability to understand the relative positions between different objects, cameras and semantic regions (e.g., a kitchen) across multiple views.

- • Attr. (Attribute). This subcategory evaluates the model’s ability to identify object attributes related to spatial properties, such as geometric properties (e.g., size, length) or visual characteristics (e.g., shape).
- • Mot. (Motion). This subcategory evaluates the model’s ability to understand object or camera’s movement.
- • MSR (Multi-Step Reasoning). This subcategory evaluates the model’s ability to perform complex reasoning by chaining multiple spatial understandings described above together to arrive at a final answer.

- D.2. MindCube-tiny

MindCube-tiny is a subset of the MindCube benchmark [59], which is designed to test Spatial Mental Modeling (SMM). The core task evaluates a VLM’s ability to construct and manipulate a 3D mental model of a scene using only a limited set of 2D images as input. The “tiny”

version is a smaller-scale version of the full benchmark. We evaluate on its three primary sub-tasks:

[Figure 82]

- Example 1 Q: The three pictures are hanging on the north wall. What is the position of ...?

reasoning: The absolute direction 'North' is tied to 'the wall'. Let’s use a physical object as a proxy. The pictures are on the wall, and their 'front' faces away from it. Therefore, the picture's orientation is used to define the reference frame, where the back of the pictures (-Z_picture) points North.

formalization: +#1,% = −#523)61, = North

[Figure 83]

- Example 2 Q: Given that the knife is north of the green trash can, what is …?

[Figure 84]

reasoning: The absolute direction 'North' is explicitly defined by the relative position of two physical and detectable objects: the knife and the trash can. The vector pointing from the trash can to the knife establishes the primary direction.

formalization: +#1,% = Centroid knife − Centroid trash

= North

[Figure 85]

- Example 3 Q: Walking from the dining room into the room with the extinguisher, facing north. What is ...?"

[Figure 86]

- • Rot. (Rotation). This task requires the model to infer the complete environment based on partial visual information, testing its understanding of sequential views and consistent spatial cues between images (e.g., lighting).
- • Ard. (Around). This task requires the model to infer the scene from a novel viewpoint, testing its ability to interpolate and extrapolate its mental 3D model.
- • Amg. (Among). This task requires the model to infer the 3D spatial arrangement based on four orthogonal views characterized by significant occlusion, testing its ability to establish consistency relationships across perspectives and reason about the relative positions of unseen objects.

#### D.3. OmniSpatial

[Figure 87]

reasoning: The anchor is a motion between two abstract regions. Let’s translate such abstract motion into a camerarelative direction. Given the movement from the background to the foreground, this direction can be formalized as the camera's negative Z-axis.

OmniSpatial [18] is a comprehensive benchmark designed to evaluate a broad spectrum of visual-based spatial intelligence capabilities in VLMs. The full benchmark consists of four categories. We primarily focus on the two subcategories most closely related to geometric perception:

formalization: +#1,% = −#3&0 2 = North

Figure 7. In Context Examples Used in Formalizing Reference Frame. The output format follows the prompt in Table 4.

- • Pers. (Perspective Taking). This subcategory assesses the model’s understanding of 3D spatial relationships by adopting varied viewpoints, e.g., egocentric, allocentric, and hypothetical perspective.
- • Dyn. (Dynamic Reasoning). This subcategory assesses the model’s understanding of object motion and judgments in uncertain or rapidly changing environments.

- • 2D (2D Relationship). This subcategory tests fundamental 2D spatial understanding, including 2D positional relationships and object counting.
- • 3D (3D Relationship). This subcategory assesses the model’s grasp of 3D concepts, such as depth analysis and 3D distance comparisons between objects in the scene.

We exclude the other two subcategories: “Spatial Interaction” (which focuses on diagrams and user-interface, e.g., terrain map) and “Complex Logic” (which involves abstract spatial reasoning, e.g., puzzles), as they are more centered on abstract or symbolic reasoning rather than the highfidelity geometric perception that GCA is designed to solve.

### E. Prompts Used in GCA

We provide detailed prompts used in GCA, including task formalization (Table 4 and 5), tool orchestration (Table 6) and knowledge-augmented code generation (Table 7 and 8). Besides, we also provide the in context examples used in the reference frame formalization (see Figure 7 and Table 4).

#### D.4. SPBench

SPBench [22] is a benchmark designed to evaluate both VLM’s spatial reasoning in single view and multiple views:

### F. Qualitative Case Study

- • SI (Single Image). This subset contains questions that test the model’s understanding and reasoning capabilities within a single image, including absolute distance, object size, relative distance, and relative direction.
- • MV (Multiple Views). This subset requires the model to integrate information from multiple viewpoints to answer questions about relative position and object counts within a overlapping scene.

We provide several qualitative case studies on how GCA effectively tackles spatial reasoning queries. These challenging cases includes unique object counting across multiple views (Figure 8), direction-based reference frame (Figure 9), object-based reference frame (Figure 10), camera rotation analysis (Figure 11), object movement analysis (Figure 12), and metric-scale estimation (Figure 13).

#### D.5. CV-Bench

### G. Broader Impacts

CV-Bench is a visual-centric benchmark that evaluates the spatial understanding capabilities of VLMs. It is broadly composed of 2D and 3D reasoning tasks:

Advancing Embodied AI Systems. Applications in robotics and AR/VR depend on an agent’s ability to translate ambiguous human instructions (e.g., “sit on the sofa”)

into precise geometric actions. GCA provides a robust framework for this translation, potentially facilitating the development of robots and AR/VR interfaces that can interact with the physical world with high fidelity.

Trust, Interpretability, and Verification. Unlike opaque end-to-end spatial VLMs, GCA’s reasoning process is highly traceable. Ctask serves as an explicit, human-readable artifact that can be verified before a high-stakes action is executed. This verify-then-execute capability is critical for safety in real-world applications. Furthermore, the structured outputs from both the Fformalize and Fcompute stages can serve as a valuable source of process-level supervision, acting as a reliable validator to guide the training of more efficient end-to-end spatial VLM.

Inheritance and Amplification of Bias. The reliance on VFMs for perception and geometry (e.g., 3D reconstruction, object orientation) creates a new dependency chain for bias and failure. If these perceptual tools perform poorly on objects, scenes, or lighting conditions, GCA will not only inherit these biases but may also amplify them, leading to incorrect outcomes in real-world interactions.

Table 4. Prompts Used for Formalizing Reference Frame Constraint. Here, {example} is the in-context examples (see Figure 7), and {question} is the placeholder that will be replaced.

###### [CORE MISSION]

You are an expert spatial reasoning analyst. Your sole mission is to analyze a user’s question and define the final Reference Frame. Your goal is to find the single element (an object, a camera, or a vector) that provides the ultimate, non-negotiable definition for absolute directions (e.g., North, South) or relative perspectives (e.g., “front”, “left”) in the final answer. Ask yourself: “What element holds the final authority on what ‘north’, ‘front’, or ‘left’ means in the question?”

[OUTPUT FORMAT] Your response MUST be a single, valid JSON object: ‘‘‘json {

"reasoning": "A brief, step-by-step logical deduction, explaining WHY this anchor is the arbiter of direction.",

"formalization": "The precise mathematical mapping of a semantic direction to one of the Solvable Geometric Primitive listed below, e.g., $-Z_cam0, +Z_toaster$."

} ‘‘‘

[FORMALIZATION]

- 1. Identify the Final Arbiter of Direction

- - Priority 1: Absolute Direction. If an absolute direction (North, South, etc.) is explicitly tied to an element, that element is the arbiter, overriding everything else.
- - Priority 2: Relative Query. If no absolute direction is given, the arbiter is the object of the relative question.

- 2. Formalize Reference Frame Using Solvable Geometric Primitives: o construct a mathematical formalization of reference frame, you MUST use following three types of Solvable Geometric Primitives:

- - A. Camera Axes: A vector from a specific camera’s coordinate system. Format: ±Xcam[i], ±Ycam[i], ±Zcam[i]. The camera coordinate system follows OpenCV convention: +Z points forward, +Y points down, and +X follow right-hand rules.
- - B. Object Axes: A vector from a specific object’s semantic coordinate system. Format: ±X[obj], ±Y[obj], ±Z[obj]. The object’s local coordinate system is defined by: +Z points its semantic “front”, +Y points its semantic “down”, +X follows right hand rules, and origin at centroid.
- - C. Inter-Object Vector (Direction): A vector connecting the centroids of two concrete, detectable objects. Format: Centroid(B) − Centroid(A).

- 3. Semantic Formalization

- - A. Object-based Reference Frame: Usually can be defined by corresponding object’s axes. Examples:
- - “when using the toaster” suggests user’s “forward” is opposite the toaster’s semantic “front”, i.e., +Zref = −Ztoaster.
- - You must choose a Physical and Detectable object as the object anchor. Don’t use abstract concepts like room/region/area.
- - B. Camera-based Reference Frame: Usually can be defined by corresponding camera’s axes. Examples:
- - “from the perspective of Figure 1” suggests reference frame is identical to camera 0’s, i.e., +Zref = +Zcam0.
- - C. Direction-based Reference Frame
- - For spatial relationship between two Physical and Detectable Objects, it can be defined by inter-object vector. Examples: “object A is north of object B” suggests the direction from object B to A is north, i.e., +Zref = BA⃗ = Centroid(A) − Centroid(B) = North
- - For spatial relationship between two Abstract Concepts, you must use a physic object’s axes or a camera’s axes as the proxy to tie this direction. Examples: “moves from room A to room B, facing north”.Assume this motion aligns with moving from background towards the foreground, formalized as +Zref = −Zcam[i] = North.

[EXAMPLES] {examples}

[QUESTION] {question}

Now, please analyze the above question and provide your response in the specified JSON format.

Table 5. Prompts Used for Formalizing Objective Constraint. Here, {question} is the placeholder that will be replaced.

###### [CORE MISSION]

You are an expert spatial reasoning analyst. Your sole mission is to analyze a user’s question and define the final Objective. Your goal is to rephrase the user’s natural-language question into a single and precise sentence. This sentence describes the specific value or piece of information that definitively answer the question. Ask yourself: “What is the single, final piece of information (e.g., a scalar value, a 3D vector, a sequence of rotations) that the user is finding?”

[OUTPUT FORMAT] Your response MUST be a single, valid JSON object: ‘‘‘json {

"reasoning": "A brief, step-by-step logical deduction that breaks down the user’s question into its final objective.", "formalization": "A single, concise sentence that defines the ultimate goal of the question, stated in technical terms."

} ‘‘‘

###### [OBJECTIVE]

- - Identify the target variable: What type of answer is being sought? Is it a distance, a speed, an orientation, a direction, a count, a relationship, or a sequence of actions?
- - Identify the Key Entities: What are the specific, concrete objects, cameras, or locations involved in the question?
- - Synthesize the Objective: Combine the target variable and entities into a single, unambiguous sentence. This sentence must be a statement or a noun phrase, not a question. Example:
- - Bad: Which way the object are going?
- - Good: The 3D direction vector of the object’ movement.

[QUESTION] {question}

Now, please analyze the above question and provide your response in the specified JSON format.

- Table 6. Prompts Used for Tool Orchestration. Here, {api documents} is the detailed documentation of provided APIs. {history} includes the initial user question, task formalization, previous planning and corresponding execution results.

[CORE MISSION] You are an expert spatial intelligence agent. Your mission is to generate a sequence of tool calls that rigorously computes the answer. It follows an iterative Plan → Update cycle, using a workspace as your computational memory.

- - Plan: Decide the next tool calls based on the goal and current workspace.
- - Update: Tool results are saved as new variables in the workspace.
- - Repeat: Continue until the workspace contains enough information to conclude the final answer.

[AVAILABLE APIS] {api documents}

[A TYPICAL WORKFLOW]

- 1. Strictly Follow the Task Formalization

- - Task Formalization: The input question is pre-formalized and consists of two parts: Reference Frame Constraint and Objective Constraint. You MUST strictly follow this formalization to solve the question.
- - Reference Frame: Reference frame is the only one coordinate system that matters for interpreting the final answer (left/right, north/south, etc.). The “formalization” is the equation you must solve with the specified geometric tools. E.g., +Zref = −Ztoaster indicates reference frame is defined by object toaster’s local frame, so we MUST perform all calculations in toaster’s frame.
- - Objective: Objective is the ultimate goal of the question. You must calculate this objective within the reference frame.

- 2. Acquire Geometric Data: Based on user’s question and pre-defined task formalization, plan the necessary tool calls to gather all data required for the final calculation. This involves two parallel goals:

- A. Solve for the Reference Frame: The formalization mathematically defines the World-to-Reference Transformation. To solve this formalization, your plan MUST gather all the geometric ingredients in the world frame.

- - “reconstruct” tool provides the 3D reconstruction context in a unified world frame, bridging the gap between input 2D images and geometric perception.
- - If formalization involves an object’s axes (e.g., +Zref = −Ztoaster), call “predict obj pose” to solve that object’s local frame. The resulting “T obj2world” is required to establish the reference frame.

- - If formalization involves a camera’s axes (e.g., +Zref = −Zcam0), you must acquire reconstruction context (include extrinsic matrix) to implement the formalization.
- - If formalization involves a vector between two objects (e.g., Centroid(B)−Centroid(A) = North), your plan MUST include

calls to “project box to 3d points” for both object A and B.

- - B. Solve for the Objective: Follow the objective to identify the target data that need to be analyzed within the reference frame.

- 3. Perform Final Calculation in Reference Frame: Once all required variables are available in the workspace, call “code”.
- 4. Conclusion: Conclude the final answer using “generate final answer”.

[OUTPUT FORMAT] Your response MUST be a single, valid JSON object: ‘‘‘json {

"analysis": "Briefly analyze how you will implement the formalization and what

target data is need. State the immediate next tool(s) you will call", "tool_calls": [

{

"api": "API name", "args": {...}, "output_variable": "A unique name for output, stored in the workspace"

}, ... ]

} ‘‘‘

[HISTORY] Here is the history so far: {history}

Please analyze current situation and history messages, and then generate your response. Your plan MUST only includes the immediate next one step.

- Table 7. Prompts Used for Coder. Here, {question}, {formalization}, {objective} and {var docs} are the placeholder that will be replaced. {knowledge} includes a set of releveant, fixed formulas based on the type of input variables.

[CORE MISSION] You are an expert Python programmer. Your goal is to write a single Python function that correctly implements the computational objective based on the provided context and documentation.

[User’s Question] This provides the high-level context for your task. {question}

[Reference Frame] All geometric data are defined in the world frame (defined by camera 0) unless specified. All final interpretations MUST be expressed in the reference frame. The reference frame is defined: {formalization}

[Objective] The ultimate goal from the high-level question. You MUST write code to calculate this objective to answer the question. {objective}

[Documentation of Available Variables] {var docs}

[Additional Knowledge] This information is always true for the environment your code runs in. {knowledge}

[Available Libraries] You can use “numpy”, “torch”, “scipy”, “math”, and other standard Python libraries. [Critical Rules and Output Format]

- 1. Synthesize and Self-Correct: Your primary duty is to write correct code. Use the objective as your goal, but critically verify and implement the logic using the provided documentation.
- 2. Handle Multiple-Choice Questions: If the user’s question is multiple-choice, your code MUST systematically evaluate the conditions for every option (e.g., A, B, C, D). Besides, the logic of your code should focus ONLY on the given options.
- 3. DO NOT add any explanation in your final output. Your output MUST follow this format: ‘‘‘python def execute(func_signature):

# Your code here

... return serializable_value # return value MUST be a serializable type

‘‘‘

- Table 8. Knowledge and Formulas Used in Knowledge-Augmented Code Generation. We will inject the relevant knowledge based on the type of input variable.

[Output of “reconstruct”] Extrinsic Transformation (World ↔ Camera):

- - World → Camera: To transform a world point “P world” into camera s’s frame, use its extrinsic matrix E s = extrinsic[s]. The formula is: P cam homo = P world homo @ E s.T.

- - Camera → World: The pose of camera “s” in the world, “Pose s”, is the inverse of its extrinsic matrix: Pose s = np.linalg.inv(extrinsic[s]). To transform a point from camera s’s local frame to the world frame, use the formula: P world homo = P cam homo @ Pose s.T.

- - Relative Rotation Analysis (Camera → Camera): To describe the rotation of camera j’s pose relative to camera i’s pose, use the camera poses in the world frame (“Pose i”, “Pose j”). The relative rotation from i to j is: R rel = R j pose @ R i pose.T, which simplifies to R rel = (R j.T) @ (R i.T).T = R j.T @ R i.

[Output of “predict obj pose”] Object Pose Transformation (World ↔ Object):

- - Object → World: The “T obj2world” matrix (aliased as “Pose obj”) transforms points from the object’s local frame to the world frame using the formula: P world homo = P local homo @ Pose obj.T.

- - World → Object: To transform “P world” into the object’s local frame, use the inverse matrix: T world to obj = np.linalg.inv(Pose obj). The formula is: P local homo = P world homo @ T world to obj.T.

[Output of “reconstruct”, “predict obj pose”] Interpreting Rotation:

- - General Rotation Direction: For most questions about rotation direction, convert the relative rotation matrix to a rotation vector [rx, ry, rz] via scipy.spatial.transform.Rotation.from matrix(R).as rotvec().
- - The component with the largest absolute value indicates the primary axis of rotation.
- - Based on the OpenCV coordinate system (+X right, +Y down, +Z forward) and the right-hand rule: ry > 0 corresponds

to a pan to the right, rx > 0 corresponds to a tilt upward, rz > 0 corresponds to a clockwise roll.

- Sequential Rotations: Use scipy.spatial.transform.Rotation.from matrix(R).as euler(order) to compute sequential rotation. Verify the signs of the resulting angles. It must match the option’s description.

[If formalization includes cardinal direction]

- 1. Identify the Cardinal Anchor Axis from the formalization.

- - The formalization string (e.g., +Zref = −Zobj = South) links one of your reference axes to a cardinal direction.
- - Parse this string to find the anchor. In the example, the anchor is South, and it corresponds to the Zref axis. This defines your

first cardinal vector: South axis = Zref axis.

- 2. Derive the Complete Set of Cardinal Axes: You must find the remaining cardinal axes via applying cross product:

- - If South axis is known, starting with West axis = np.cross(Y ref axis, South axis).

- - If West axis is known, starting with North axis = np.cross(Y ref axis, West axis).

- - If North axis is known, starting with East axis = np.cross(Y ref axis, North axis).

- - If East axis is known, starting with South axis = np.cross(Y ref axis, East axis).

- 3. Project and Determine the Final Quadrant: Project the target vector onto the primary horizontal cardinal axes (North and East).

- - projection north = np.dot(disp vec world, cardinal map["N"])

- - projection east = np.dot(disp vec world, cardinal map["E"])

- - Use the signs of these projections to determine the final answer.

[Output of “detect”] Bounding Box Format: All bounding box in input variable is provided in the [x1, y1, x2, y2] format.

Spatial Reasoning Query 1 2

[Figure 88]

[Figure 89]

Q: These images show the 3D model of the same building. How many different trees appear in total in the two images?

[Figure 90]

###### Stage 1: Task Formalization

VLM as Analyst

###### Objective Constraint

Reference Frame Constraint

The count of trees in a unified reference

The question doesn't specify a reference frame, it would be simpler to choose the camera 0 as reference: +#'() = +#*+,-, +%'() = +%*+,-.

frame (determine whether the trees shown in images are the same one).

[Figure 91]

Stage 2: Geometric Computation

VLM as Constrained Task Solver

[Figure 92]

[Figure 93]

###### 3D Reconstruction Detect “Tree” in Fig.1 and Fig.2

|[Figure 94]<br><br>1|
|---|

[Figure 95]

|[Figure 96]<br><br>2|
|---|

[Figure 97]

1

[Figure 98]

[Figure 99]

[Figure 100]

###### Segment “Tree” and Extract 3D Point Cloud

[Figure 101]

[Figure 102]

[Figure 103]

- tree0_img1
- tree1_img1

- tree0_img0
- tree1_img0

|[Figure 104]<br><br>[Figure 105]<br><br>1<br><br>|
|---|

|[Figure 106]<br><br>[Figure 107]<br><br>2<br><br>|
|---|

- box 0
- box 1

- box 0
- box 1

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

Code Generation

def execute(tree0_img0, tree1_img0, tree0_img1, tree1_img1): centroids = [np.mean(pc, axis=0)

for pc in [tree0_img0, tree1_img0, tree0_img1, tree1_img1]] unique_centroids = [] for c in centroids:

[Figure 118]

Execution

is_unique = True for uc in unique_centroids:

if np.sum((c - uc) ** 2) < 0.01: is_unique = False break

[Figure 119]

3

if is_unique:

unique_centroids.append(c) return len(unique_centroids)

Figure 8. Case Study #1. Unique object counting across multiple views.

- Stage 1: Task Formalization VLM as Analyst

[Figure 120]

The positional relationship of table relative to the book , &.//0→2+.3(.

Objective Constraint

The reference Frame is defined by an inter object vector: + #'() = Centroid glass − Centroid(book) = North.

Reference Frame Constraint

[Figure 121]

1 2

[Figure 122]

Spatial Reasoning Query

Q: What is the direction of the glass coffee table relative to the book on the bar counter (wine glass is north of the book)?

A: Northwest; B: Northeast; C: Southwest; D: Southeast

- Stage 2: Geometric Computation Reconstruction

[Figure 123]

[Figure 124]

VLM as Constrained Task Solver

[Figure 125]

[Figure 126]

###### Detect “glass”, “book” and “table”

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Choose 1: wine glass

[Figure 135]

[Figure 136]

###### Segment “glass”, “book” and “table”, and Extract 3D Point Cloud

[Figure 137]

[Figure 138]

book_points

[Figure 139]

[Figure 140]

[Figure 141]

coffee_table _points

box 2

[Figure 142]

[Figure 143]

box 0

[Figure 144]

wine_glass

box 1

[Figure 145]

[Figure 146]

Code Generation

def execute(book_points, wine_glass_points, coffee_table_points): book_centroid = torch.mean(book_points, dim=0) wine_centroid = torch.mean(wine_glass_points, dim=0) coffee_centroid = torch.mean(coffee_table_points, dim=0) Z_ref_vec = wine_centroid - book_centroid Z_ref = Z_ref_vec / torch.norm(Z_ref_vec) Y_ref = torch.tensor([0.0, 1.0, 0.0], dtype=Z_ref.dtype) X_ref = torch.cross(Y_ref, Z_ref) X_ref = X_ref / torch.norm(X_ref) disp_vec = coffee_centroid - book_centroid proj_north = torch.dot(disp_vec, Z_ref) proj_east = torch.dot(disp_vec, X_ref) if proj_north >= 0 and proj_east < 0: return 'A' elif proj_north >= 0 and proj_east >= 0: return 'B' elif proj_north < 0 and proj_east < 0: return 'C' else: return 'D'

[Figure 147]

Execution

C

[Figure 148]

Figure 9. Case Study #2. Direction-based reference frame.

Spatial Reasoning Query

[Figure 149]

[Figure 150]

1 2

Q: When you are sitting in the sofa, where is the well-lit area in Figure 1 relative to you?

A: front right; B: back right; C: front left; D: back left

[Figure 151]

Stage 1: Task Formalization VLM as Analyst

###### Reference Frame Constraint Objective Constraint

The well-lit area in Figure 1 is the kitchen island, so the objective is the positional relationship of the round dinner table relative to sofa, &4/)+→56789:

The viewpoint is aligned with sofa’s orientation. The reference frame is +#ℛ = +#4/)+ = front

[Figure 152]

[Figure 153]

[Figure 154]

Stage 2: Geometric Computation 3D Reconstruction

VLM as Constrained Task Solver

[Figure 155]

[Figure 156]

##### Detect “sofa” and “kitchen island”

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Choose 0: kitchen island

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Segment “Tree” and Extract 3D Point Cloud

“Sofa” Orientation

|+Y|
|---|

island_points sofa_points sofa_pose

[Figure 169]

Code Generation

def execute(sofa_pose, island_points): sofa_origin = sofa_pose.T_obj2world[:3, 3] island_centroid = torch.mean(island_points, dim=0) disp_vec = island_centroid - sofa_origin R = sofa_pose.T_obj2world[:3, :3] X_obj, Y_obj, Z_obj = torch.unbind(R[:, :3], dim=1) Z_ref, Y_ref = Z_obj, Y_obj X_ref = torch.cross(Y_ref, Z_ref) dx = torch.dot(disp_vec, X_ref) dz = torch.dot(disp_vec, Z_ref) if dx >= 0 and dz >= 0: return 'A' elif dx >= 0 and dz < 0: return 'B' elif dx < 0 and dz >= 0: return 'C' else: return 'D'

[Figure 170]

Execution

A

[Figure 171]

Figure 10. Case Study #3. Object-based reference frame

###### Spatial Reasoning Query

[Figure 172]

[Figure 173]

1 2

Q: The camera coordinate system is defined as +Y up, -Z forward, and is a righthanded system. How can the viewpoint in the second image be obtained from the viewpoint in the first image in its corresponding camera coordinate system?

- A: Rotate a positive angle around the Z axis, then a positive angle around the Y axis
- B: Rotate a positive angle around the Y axis, then a negative angle around the X axis
- C: Rotate a positive angle around the Y axis, then a positive angle around the X axis
- D: Rotate a positive angle around the Z axis, then a negative angle around the Y axis

[Figure 174]

###### Stage 1: Task Formalization

VLM as Analyst

Reference Frame Constraint

Objective Constraint

The question asks for the viewpoint transformation, establishing cam_0 as the anchor. The custom coordinate system (+Y up, -Z forward) must be mapped to cam_0: +#'() = −#*+,-, +<'() = −<*+,-.

The sequence of axis rotations required to transform the camera viewpoint from the first image to the second image.

[Figure 175]

Stage 2: Geometric Computation

VLM as Constrained Task Solver

- 1 From Camera 0 BEV Perspective Reconstruction

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- 2 extrinsic

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Code Generation

def execute(extrinsic): R_0, R_1 = extrinsic[0][:3, :3], extrinsic[1][:3, :3] T = np.diag([1, -1, -1]) # y_user = -y_cam0, z_user = -z_cam0 R_user = T @ R_1.T @ R_0 @ T

options = { 'A': {'order': 'zyx', 'signs': [1, 1]}, 'B': {'order': 'yxz', 'signs': [1, -1]}, 'C': {'order': 'yxz', 'signs': [1, 1]}, 'D': {'order': 'zyx', 'signs': [1, -1]}

} for opt, params in options.items():

angles = Rotation.from_matrix(R_user).as_euler(params['order'], degrees=False)

- sign1_match = (params['signs'][0] == 1 and angles[0] > 1e-6 ) or \

- (params['signs'][0] == -1 and angles[0] < -1e-6 )

sign2_match = (params['signs'][1] == 1 and angles[1] > 1e-6 ) or \

- (params['signs'][1] == -1 and angles[1] < -1e-6 )

[Figure 186]

Execution

if sign1_match and sign2_match:

return opt return None

[Figure 187]

C

Figure 11. Case Study #4. Camera rotation analysis.

[Figure 188]

[Figure 189]

###### Spatial Reasoning Query

1

###### 2

Q: The pictures were taken in sequence. In which direction are the two grippers moving?

A: Left did not move, right moved to left; B: Left moved to right, right moved to right C: Left moved to right, right did not move; D: Left moved to left, right did not move

- Stage 1: Task Formalization

Objective Constraint

The direction of grippers’ movement from the first image to the second.

Reference Frame Constraint The pictures are stated to be “taken in sequence,” so, the reference frame is anchored by the first camera:

+ #'() = +#*+,-, +%'() = +%*+,-.

VLM as Analyst

[Figure 190]

- Stage 2: Geometric Computation

[Figure 191]

[Figure 192]

VLM as Constrained Task Solver

[Figure 193]

[Figure 194]

Reconstruction Detect “Gripper” in Fig.1 and Fig.2

[Figure 195]

[Figure 196]

[Figure 197]

1 2

[Figure 198]

###### Segment “Gripper” and Extract 3D Point Cloud

[Figure 199]

- right0

[Figure 200]

- right1

[Figure 201]

[Figure 202]

- left0

[Figure 203]

- left1

1 box 0

2

[Figure 204]

- box 0
- box 1

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

box 1

[Figure 209]

[Figure 210]

Code Generation

def execute(left0, left1, right0, right1): left_disp = torch.mean(left1[:, 0]) - torch.mean(left0[:, 0]) right_disp = torch.mean(right1[:, 0]) - torch.mean(right0[:, 0]) left_no_move = abs(left_disp) < 1e-5 left_move_right, left_move_left = left_disp > 1e-5, left_disp < -1e-5 right_no_move = abs(right_disp) < 1e-5 right_move_right, right_move_left = right_disp > 1e-5, right_disp < -1e-5

[Figure 211]

if left_no_move and right_move_left: return 'A' elif left_move_right and right_move_right: return 'B' elif left_move_right and right_no_move: return 'C' elif left_move_left and right_no_move: return 'D' return None

Execution

[Figure 212]

B

Figure 12. Case Study #5. Object movement analysis.

###### Spatial Reasoning Query

[Figure 213]

###### 1

Q: What speed must the photographer reach out at to touch the white teddy bear in 4s?

A. 35.7m/s; B. 24.7m/s; C. 0.33m/s; D. 8.33m/s

[Figure 214]

###### Stage 1: Task Formalization

VLM as Analyst

Reference Frame Constraint Objective Constraint Speed = Distance/Time. Time is given (4s). The primary objective is the Distance from camera origin to the white teddy bear.

To measure the speed of movement from the photographer to white teddy bear, cam_0 is defined as the origin of the reference frame: +#'() = +#*+,-.

[Figure 215]

Stage 2: Geometric Computation

VLM as Constrained Task Solver

[Figure 216]

[Figure 217]

[Figure 218]

Reconstruction

Detection Segmentation

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Compute “scale_factor” from Depth Map

###### VGGT (relative)

MoGe (metric) ∑!∈#$%&-#'()*+

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

∑!∈#$%&-)',$(*-'

[Figure 230]

scale_factor

[Figure 231]

Code Generation

def execute(teddy_points, scale_factor): centroid = torch.mean(teddy_points * scale_factor, dim=0) distance = torch.norm(centroid) speed = distance / 4 options = {'A': 35.7, 'B': 24.7, 'C': 0.33, 'D': 8.33} best_option = None min_diff = float('inf') for key, value in options.items():

[Figure 232]

Execution

diff = abs(speed.item() - value) if diff < min_diff:

min_diff = diff best_option = key

[Figure 233]

C

return best_option

Figure 13. Case Study #6. Metric-scale estimation.

