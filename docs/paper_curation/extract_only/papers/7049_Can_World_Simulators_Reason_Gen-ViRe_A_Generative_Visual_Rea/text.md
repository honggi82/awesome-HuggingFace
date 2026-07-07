## Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark

Xinxin Liu1*, Zhaopan Xu2*, Ming Li1, Kai Wang2, Yong Jae Lee3, Yuzhang Shang1† 1University of Central Florida, 2National University of Singapore, 3UW-Madison Project Code

# arXiv:2511.13853v3[cs.CV]12Feb2026

Abstract

|A red square smoothly slides along the white path from start to the green goal square, never crossing black walls.<br><br>[Figure 1]<br><br>[Figure 2]|
|---|

While Chain-of-Thought (CoT) prompting enables sophisticated symbolic reasoning in LLMs, it remains confined to discrete text and cannot simulate the continuous, physicsgoverned dynamics of the real world. Recent video generation models have emerged as potential world simulators through Chain-of-Frames (CoF) reasoning—materializing thought as frame-by-frame visual sequences, with each frame representing a physically-grounded reasoning step. Despite compelling demonstrations, a challenge persists: existing benchmarks, focusing on fidelity or alignment, do not assess CoF reasoning and thus cannot measure core cognitive abilities in multi-step planning, algorithmic logic, or abstract pattern extrapolation. This evaluation void prevents systematic understanding of model capabilities and principled guidance for improvement. We introduce GenViRe (Generative Visual Reasoning Benchmark), a framework grounded in cognitive science and real-world AI applications, which decomposes CoF reasoning into six cognitive dimensions –from perceptual logic to abstract planning –and 24 subtasks. Through multi-source data curation, minimal prompting protocols, and hybrid VLMassisted evaluation with detailed criteria, Gen-ViRe delivers the first quantitative assessment of video models as reasoners. Our experiments on SOTA systems reveal substantial discrepancies between impressive visual quality and actual reasoning depth, establishing baselines and diagnostic tools to advance genuine world simulators.

###### Human

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

human simulation picture

MLLM

[Figure 8]

[Figure 9]

[Figure 10]

symbolic reasoning

(CoT)

VGM

[Figure 11]

[Figure 12]

[Figure 13]

generative visual reasoning

(CoF)

Figure 1. A comparison of reasoning approaches for a mazesolving task. Humans visualize the path via mental simulation. A Multimodal Large Language Model (MLLM) uses symbolic reasoning (CoT) to describe the path, e.g., via coordinates. In contrast, a Video Generation Model (VGM) uses generative visual reasoning (CoF) to physically simulate the process, generating frames of the square moving from start to finish.

represent a significant milestone in AI’s ability to perform complex reasoning. However, CoT operates exclusively in the symbolic realm—reasoning over language tokens about abstract concepts. While this enables powerful logical inference, it fundamentally cannot capture the continuous, visual, and spatial nature of the real world. This limitation is evident in tasks ranging from abstract spatial navigation to complex physical interaction.

### 1. Introduction

Recent breakthroughs in Chain-of-Thought (CoT) prompting have demonstrated that large-scale language models (LLMs) can exhibit emergent reasoning capabilities across diverse symbolic domains, from mathematical problemsolving to code generation [18, 23, 36]. These advances

For instance, in the maze-solving task (shown in Figure 1), when an MLLM using CoT is asked to find the path, it describes the solution symbolically—perhaps by outputting a list of coordinates [e.g., "(3, 1), (2, 1)..."]. It cannot, however, simulate the actual, continuous, dynamic process of the "red square" moving through the path. Simi-

*Equal contribution. †Corresponding author

larly, consider a robotic manipulation task: planning how to open a nailed wooden box. CoT can describe the steps (‘use a crowbar, apply leverage’), but it cannot verify whether the plan is physically feasible—whether the robot’s gripper can grasp the crowbar, whether the force angle is sufficient, or whether nearby objects will interfere. This reveals a critical limitation: to build AI systems that truly understand and simulate spatial dynamics and physical interaction, we need models that can reason through continuous visual simulation, not just symbolic description.

Analogical

Abstract

Perceptual

Planning

#### Gen-ViRe

This gap—the need for dynamic, generative reasoning—is also precisely where conventional computer vision paradigms fall short. Conventional vision models function as specialized, passive perceivers—excelling at singular discriminative tasks like object detection or segmentation, yet lacking the unified generative reasoning capabilities needed to simulate dynamic processes. Recognizing this gap, recent pioneering works have proposed a revolutionary paradigm shift: video models as zero-shot learners and world simulators [2, 3, 16, 37]. As world simulators, these models are transitioning from merely rendering pixels to building an implicit spatiotemporal and physical engine by training on massive video data. This emergent “simulator” then enables them to act as zero-shot learners; they are no longer confined to the specific compositions seen during training but can, much like humans, leverage a fundamental understanding of the world to “zero-shot” imagine, reason about, and create entirely new, logical scenarios. At the core of this new paradigm lies the Chain-ofFrames (CoF) mechanism [37], where reasoning is actualized through frame-by-frame video generation. Unlike CoT’s discrete symbolic transitions (text → text), CoF materializes reasoning as a goal-directed, continuous visual state evolution (frame → frame). When a model solves a visual Sudoku puzzle, navigates a maze, or plans multi-step tool use through CoF, each generated frame represents an incremental reasoning step that is both physically grounded and temporally coherent. The generative process itself becomes an act of thinking, not just describing the symbolic coordinates for the maze but generating the dynamic trajectory of the red square moving. It transforms models from passive descriptors into executable world simulators.

Algorithmic &Logical

Spatial& Temporal

Benchmark Task Distribution

Figure 2. Our Gen-ViRe evaluates six core cognitive dimensions: (1) Perceptual, (2) Analogical, (3) Abstract, (4) Planning, (5) Spatial & Temporal, and (6) Algorithmic & Logical, with each dimension comprising four different sub-categories.

how well do these models actually reason? Qualitative showcases, while impressive, provide no systematic measure of reasoning depth. More critically, existing video generation benchmarks, which focus primarily on fidelity, alignment, or object-level consistency, do not assess CoF reasoning and therefore cannot measure a model’s core cognitive abilities in multi-step planning and algorithmic logic, or abstract pattern extrapolation [10, 11, 19, 21, 27, 30, 31, 34]. Without rigorous quantitative evaluation, we cannot distinguish genuine reasoning from sophisticated pattern matching, nor can we diagnose where models failperception, physics understanding, or planning—to guide systematic improvement.

To address this critical evaluation gap, we propose GenViRe (Generative VisualReasoning Benchmark), the first comprehensive framework purpose-built to systematically assess CoF reasoning by video generation models. GenViRe is grounded in dual foundations: cognitive science principles defining core pillars of human reasoning [4, 5, 15, 24, 25, 33] , and application-driven requirements from emerging domains like embodied AI and autonomous systems [12, 40]. We decompose Generative VisualReasoning into six fundamental dimensions— Perceptual, Analogical, Algorithmic & Logical, Spatio-Temporal & Dynamic, Procedural & Planning, and Abstract Reasoning—spanning the complete spectrum from foundational perception to highorder planning [1]. Our data curation employs a multisource strategy as shown in Figure 4: curating from web resources and academic papers, integrating existing datasets (e.g., ARC-AGI for abstract reasoning, KiVA for analogical

The potential of this paradigm is evidenced by rapid progress in video generation technology. Large-scale models—from commercial systems like Sora [2] and Kling [28] to open models like CogVideo [20] and HunyuanVideo [26]—have demonstrated impressive emergent capabilities in understanding physical interactions, temporal causality, and spatial relationships. Qualitative demonstrations from pioneering works are compelling: models generating coherent multi-step processes, respecting object permanence, and exhibiting intuitive physics. Yet despite this progress, a fundamental question remains unanswered:

tasks), and generating novel tasks using T2I models where suitable data is absent. We implement rigorous prompt design through minimal prompting principles—deliberately providing only high-level goals to assess autonomous reasoning rather than instruction-following—validated through iterative peer review. Our evaluation methodology features a hybrid VLM-assisted approach: Image VLMs judge final outputs for static reasoning tasks, while Video VLMs assess the complete generation process for dynamic tasks, with each judge provisioned with detailed, sub-categoryspecific criteria refined through multi-round human validation. Through extensive experiments evaluating 7 state-ofthe-art video generation models across 72 distinct reasoning prompts (with each model generating 5 instances per prompt, totaling 360 videos per model and over 2,500 overall), Gen-ViRe delivers the first systematic quantification of CoF reasoning capabilities, revealing critical gaps between qualitative potential and quantitative performance, and establishing scientific baselines to guide the development of genuine world simulators.

Our main contributions are as follows:

- • We propose Gen-ViRe, the first comprehensive benchmark specifically designed to systematically evaluate Chain-of-Frames reasoning across six fundamental dimensions grounded in cognitive science and practical AI applications, providing rigorous scientific assessment from foundational perception to high-order planning.
- • We establish a complete evaluation methodology combining minimal prompting design, multi-source data curation (web/academic resources, existing datasets, generative creation), and hybrid VLM-assisted assessment with detailed criteria, enabling quantitative measurement of generative reasoning and failure mode diagnosis.
- • Through extensive experiments on state-of-the-art video generation models, we provide the first systematic analysis of current CoF reasoning capabilities, revealing significant gaps between qualitative demonstrations and quantitative performance, and establishing baselines for future research toward genuine world simulators.

### 2. Related Work

- 2.1. Discrete Chain-of-Thought Reasoning. Chain-of-Thought (CoT) prompting has revolutionized how large language models approach complex reasoning tasks by explicitly generating intermediate reasoning steps [36]. GPT-o1 [23] demonstrated that LLMs can leverage CoT for test-time scaling—trading computation for reasoning depth—while DeepSeek-R1 [18] advanced this through Reinforcement Learning with Verifiable Rewards (RLVR), democratizing sophisticated reasoning via open-source release. Recent work has extended CoT to multimodal domains: GPT-4o [22] pioneered vision-language joint reasoning, and Bagel [9] leverages vision-text interleaved CoT

to enhance visual generation. However, CoT remains fundamentally symbolic and discrete—operating in the space of language tokens rather than continuous visual states. While effective for formal logic, CoT is inherently discriminative and passive: it cannot dynamically simulate the physical evolution of the visual world. A CoT chain can describe "a ball rolls down a slope," but cannot generate the actual frame-by-frame trajectory governed by gravity and collision. This limitation necessitates a new paradigm grounded in continuous, generative visual simulation.

##### 2.2. Continuous Chain-of-Frames Reasoning.

The emerging Chain-of-Frames (CoF) paradigm represents a revolutionary shift in how AI systems perform reasoning [37]. Unlike CoT, which operates on discrete symbolic transitions (text → text), CoF materializes reasoning as continuous visual state evolution (frame → frame). In this paradigm, reasoning is not merely described—it is executed and visualized through frame-by-frame video generation. When a model generates a sequence showing how to solve a visual puzzle, navigate a maze, or manipulate objects to achieve a goal, each generated frame represents an incremental reasoning step that is physically grounded and temporally coherent. Recent pioneering works have begun to explore this new frontier. The Genie series [3] demonstrates how interactive world generation can enable embodied reasoning, allowing agents to learn causal relationships through simulated physical interactions. Veo-3 pushes the boundaries of long-horizon reasoning, generating extended video sequences that maintain spatial-temporal consistency across complex dynamic scenes [16]. These works provide compelling qualitative evidence that video models have the potential as world simulators, capable of understanding physics, causality, and spatial relationships [37].

##### 2.3. Video Generation: Potential World Simulators.

Recent years have witnessed remarkable progress in video generation, predominantly driven by diffusion-based architectures. Models such as Sora [2], Kling [28], and SeeDance [14] have achieved breakthroughs in visual quality, generation length, and resolution through large-scale diffusion transformers. While these commercial models remain closed-source, the research community has also contributed open-sourced models like CogVideo [20] and HunyuanVideo [26] and WanVideo [35]. These models demonstrate impressive implicit learning of physical principles purely from observing massive video datasets. Crucially, these models are evolving beyond mere visual generators—they are becoming potential world simulators. For instance, when prompted to generate a video of “a basketball bouncing downstairs”, Veo-3 [16] can produce sequences that respect gravity, conserve momentum, and exhibit realistic deformation upon impact [37].

### 3. Gen-ViRe Benchmark

##### 3.1. Definition: Generative Visual Reasoning

In this paper, we aim to evaluate an emergent capability beyond traditional discriminative VQA, which we term Generative Visual Reasoning (GVR). We conceptually define GVR as an agent’s ability to solve a complex visual problem by simulating a spatio-temporal dynamic or executing a multi-step plan through a sequential, generative process. Unlike traditional benchmarks that aim to evaluate a single, deterministic final answer (e.g., a class label or a bounding box), the goal of a GVR task is to evaluate the generation process (i.e., a frame sequence F) itself for its logical coherence, physical plausibility, and goal-orientation. This Generative Visual Reasoning (GVR) capability is primarily actualized through the “Chain-of-Frames" (CoF) mechanism. We can formalize this generative process as an autoregressive sequence:

F = {f1,f2,...,fN},fi = g(Ii,q,F<i), (1) where q is the initial context or task query (e.g., “Solve this maze”), F is the complete reasoning sequence, and F<i = {f1,...,fi−1} is the generation history up to the previous step. fi is the i-th step in the sequence. As this work focuses on video generation models, fi represents the next generated video frame. The entire reasoning sequence F thus constitutes a complete and coherent video, which itself is the manifestation of the model’s “thinking" process.

##### 3.2. Taxonomy of Generative Visual Reasoning

Our proposed Gen-ViRe taxonomy is rooted in two complementary perspectives: (i) the theoretical foundations of cognitive science and (ii) the practical demands of key emerging GVR application domains. More specifically: (i) Cognitive Science Foundations: A robust GVR benchmark should be able to evaluate the core pillars of human cognitive ability [1, 5, 7, 15, 24, 25, 33],. This provides the theoretical basis for our categories of Perceptual Reasoning, Analogical Reasoning, and Abstract Reasoning. (ii) Application-Driven Requirements: Concurrently, as AI expands from the symbolic to the physical realm, emerging applications (e.g., Embodied AI, Autonomous Driving) demand dynamic, multi-step reasoning [17]. These capabilities are essential for the next frontier of AI but cannot be evaluated by existing benchmarks. These practical requirements provide the basis for our categories of Planning Reasoning, Spatial/Temporal Reasoning, and Algorithmic & Logical Reasoning.

Grounded on the above two pillars, we decompose GVR into six critical and complementary dimensions that form a full spectrum of capabilities, from foundational perception to high-order planning: (1) Perceptual Reasoning; (2) Analogical Reasoning; (3) Algorithmic & Logical Reasoning;

(4) Spatial&Temporal & Dynamic Reasoning; (5) Procedural & Planning Reasoning; (6) Abstract Reasoning. This framework is not only theoretically robust but also practically essential. It provides the first comprehensive evaluation framework for this new paradigm, allowing us to scientifically quantify and diagnose the nascent reasoning abilities driven by the Chain-of-Frames paradigm.

Perceptual Reasoning. This category probes a model’s foundational cognition: the ability to move beyond passive perception and actively reason about the logical relationships between visual attributes. We test four key logic types via worksheet-style puzzles: association (color), morphology (shape), quantification (quantity-to-numeral), and Gestalt (part-to-whole). Models must execute a precise, procedural, spatio-temporal action (e.g., “draw a connecting line") to demonstrate their conclusion, rather than just outputting a static answer.

Spatial & Temporal Reasoning. Tasks in this category assess a model’s ability to reason about motion, causality, and change within a continuous scenarios. These tasks require the model to generate a temporally coherent and physically plausible chain-of-frames (CoF). We probe faculties such as the ability to predict and model motion and to plan and execute navigation under constraints (e.g., Autonomous Driving, VLA Manipulation, Maze Traversal, Spatial Obstacle Navigation). It measures the model’s ability to build an internal world model.

Planning Reasoning. This category targets a model’s higher-order cognitive ability to perform multi-step, goaldirected planning. It requires models to decompose a complex goal into a discrete, logical, and correctly-ordered sequence of sub-actions. We probe four domains: (1) Causal Tool Reasoning (e.g., selecting the correct tool); (2) Sequential Task Decomposition (e.g., changing a lightbulb); (3) Hierarchical Digital Planning (e.g., GUI navigation); and (4) Physically-Constrained Assembly (e.g., brick-bybrick construction with physical stability).

Analogical Reasoning. This dimension focuses on the capability of relational abstraction. We adopt the classic visual analogy task (A → B :: C →?). The model must perform a two-stage inferential process: (1) Discover the latent transformation rule by comparing the source pair (A → B); and (2) Apply this inferred rule to the new target object C.

Algorithmic & Logical Reasoning. Here, we evaluate a model’s ability to follow and execute formal rules and constraints. It requires the model to apply symbolic reasoning to the visual domain to solve intellectual puzzles, including Visual Sudoku, Graph Traversal, Geometry, and Crosswords. In these tasks, the model must demonstrate its understanding of abstract rules and successfully apply them within the visual context to arrive at a correct solution.

Abstract Reasoning. This category measures a model’s highest-order cognitive ability: identifying and extrapolat-

||≈<br><br>[Figure 14]|
|---|
<br><br>Planning Reasoning (Multi-Step Procedural Planning)<br><br>(CoF) (CoF) (CoF)<br><br>|[Figure 15]|
|---|
<br><br>|[Figure 16]|
|---|
<br><br>|[Figure 17]|
|---|
<br><br>|[Figure 18]|
|---|
<br><br>A person changes the lightbulb in the lamp.<br><br>(CoF)|Planning Reasoning<br><br>(Assemble Reasoning)<br><br>|[Figure 19]|
|---|
<br><br>|[Figure 20]|
|---|
<br><br>|[Figure 21]|
|---|
<br><br>|[Figure 22]|
|---|
<br><br>LEGO bricks fall vertically from above one by one following physical laws, landing sequentially on the baseplate or existing bricks, gradually assembling into a complete and stable bookshelf.<br><br>(CoF) (CoF) (CoF)|
|---|---|

|Algorithmic & Logical Reasoning （Sudoku)<br><br>This is a 4x4 Sudoku puzzle-solving task.The goal is to fill all empty cells in the initial grid.The solution must follow Sudoku rules: each row, each column, and each 2x2 subgrid must contain the numbers 1-4 exactly once.<br><br>|[Figure 23]|
|---|
<br><br>|[Figure 24]|
|---|
<br><br>(CoF)|
|---|

|Abstract Reasoning<br><br>(CoF)<br><br>(3D Rule Extrapolation)<br><br>Analyze the logical rule established by the completed columns . Apply this same inferred logical 3D transformation rule to the object in the top-right cell.<br><br>|[Figure 25]|
|---|
<br><br>|[Figure 26]|
|---|
|
|---|

||[Figure 27]|
|---|
<br><br>|[Figure 28]<br><br>[Figure 29]|
|---|
<br><br>Create a smooth animation to generate the missing object in the lower right region and solve the visual analogy. The original three objects must remain still. Static shot, no zoom no pan no dolly.<br><br>Analogy Reasoning (Rotation)<br><br>(CoF)|
|---|

|(Raven’s Matrix)<br><br>(CoF)<br><br>Abstract Reasoning Analyze the logical rule established by the patterns in the first two rows and Apply this same logical rule to the third row to determine the missing item.<br><br>|[Figure 30]|
|---|
<br><br>|[Figure 31]|
|---|
|
|---|

|Perceptual Reasoning (NUMBER MATCHING)<br><br>|[Figure 32]|
|---|
<br><br>|[Figure 33]<br><br>|
|---|
<br><br>From top to bottom, draw solid lines connecting the five elements to their corresponding quantity numbers. Static shot, no zoom no pan no dolly.<br><br>(CoF)|
|---|

|Algorithmic & Logical Reasoning （Geometry)<br><br>Draw the line BF, starting from point B, so that it is perpendicular to AC. This line should intersect the extension of AC at point F. End the video once the line and point F are fully drawn.<br><br>(CoF)<br><br>|[Figure 34]|
|---|
<br><br>|[Figure 35]|
|---|
|
|---|

###### (CoF)

###### (CoF) (CoF)

Spatial & Temporal Reasoning (Autonomous Driving)

###### Spatial & Temporal Reasoning (Spatial Obstacle）

[Figure 36]

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

A dashcam perspective of a vehicle in a highway traffic. The vehicle smoothly following the white SUV. The lead SUV travels at a steady speed, then suddenly decelerates and comes to a complete stop.

This is a robot's first-person perspective. The task is to go to the kitchen sink and get the paper towels. The video must show the entire process of completing the task in a physically realistic and continuous sequence of actions.

|[Figure 41]|
|---|

|s<br><br>[Figure 42]|
|---|

[Figure 43]

(CoF) (CoF)

- Figure 3. Qualitative examples of Gen-ViRe tasks. It illustrates sample inputs and their expected Chain-of-Frames (CoF) visual reasoning outputs across the six cognitive dimensions, highlighting the benchmark’s breadth from foundational perception to high-order planning.

ing abstract patterns and rules. This is closely related to human “fluid intelligence." Tasks require the model to look beyond superficial features to discover the hidden generative principles in the data. Tasks we test include Symmetry, 2D/3D Rule Extrapolation, and Raven’s Progressive Matrices. Success in these tasks indicates the model is not just a pattern mimic but a rule discoverer.

fectly align with the logical and visual constraints of our benchmark tasks.

Integration of Existing Datasets. To evaluate model capabilities in specific domains, we extracted or adapted tasks from several public datasets. This includes GUI navigation datasets [29] for Planning Reasoning, Geometry datasets [13] for Algorithmic & Logical Reasoning, and tasks borrowed from KiVA [39] for Analogical Reasoning. Critically, to test high-order abstract reasoning, we also incorporated challenging tasks from the ARC-AGI benchmark [7], which is widely considered a gold standard for evaluating fluid intelligence.

### 4. Data Curation

##### 4.1. Data Collection

As described in Section 3.2, our taxonomy covers six key dimensions. To construct a diverse and information-rich benchmark capable of comprehensively evaluating GVR abilities, we designed a multi-source data collection strategy tailored to the unique requirements of each category. Our data sources are primarily divided into three categories: Web and Academic Sources, Integration of Existing Datasets, and Generative Data Creation.

Generative Data Creation. For many tasks in our benchmark—especially in the Planning Reasoning category (e.g., Tool Selection and Use)—no large-scale, logicallyconsistent datasets readily exist. Inspired by the qualitative examples in the pioneering work on Chain-of-Frames [37], we defined the generative rules and underlying logic for these advanced reasoning tasks and leveraged advanced Text-to-Image models to create entirely new visual puzzles. This approach allows us to systematically control task difficulty, compositional complexity, and generalization requirements, which is unachievable by passively collecting existing data.

Web and Academic Sources. The foundation of our data collection comes from public web resources and academic publications. We use targeted keywords to collect candidate images from search engines like Google. Concurrently, we extract high-quality figures, illustrations, and qualitative examples from relevant academic papers [17] [38]. Notably, for the Perceptual Reasoning category, we also sourced numerous children’s intelligence tests from the web (e.g., color, shape, and quantity matching puzzles). All collected materials underwent rigorous manual screening and secondary editing (e.g., cropping, annotation, content modification)—such as the modifications made to the aforementioned Spatial Obstacle image—to ensure they per-

##### 4.2. Prompt Design and Validation

Prompt Design and Validation. Our prompt design and validation process ensures high task fidelity through a minimal prompting principle and a rigorous, iterative peerreview workflow. It begins with a core design adherence to minimal prompting, aiming to assess the model’s autonomous reasoning capabilities rather than its proficiency

|(a) Gen-ViRe Data Curation|
|---|

|[Figure 44]<br><br>Perceptual Analogical Algorithmic & Logical<br><br>Spatial-Temporal & Dynamic<br><br>Procedural & Planning<br><br>Abstract<br><br>Taxonomy of GVR 6 Topics & 24 Tasks<br><br>[Figure 45]<br><br>💡 💡 💡<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>💡 💡 💡<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>- Web & Academic Sources<br><br>[Figure 52]<br><br>[Figure 53]<br><br>Data Collection<br><br>- Existing Datasets<br>- Generative Data Creation<br><br><br>[Figure 54]<br><br>[Figure 55]<br><br>|
|---|

|Peer Review<br><br>Prompt Design and Validation<br><br>[Figure 56]<br><br>[Figure 57]<br><br>❌<br><br>Goal<br><br>|A person opens the box.|
|---|
<br><br>How to achieve goal<br><br>[Figure 58]<br><br>✅<br><br>[Figure 59]<br><br>[Figure 60]<br><br>|A person opens the box with harmer.|
|---|
<br><br>[Figure 61]|
|---|

|（c) VLM-based Autorating Framework|
|---|

|(b) Formulation of Evaluation Criteria|
|---|

（

|[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>| |
|---|
<br><br>[Figure 65]<br><br>| |
|---|
<br><br>[Figure 66]<br><br>| |
|---|
<br><br>[Figure 67]<br><br>| |
|---|
<br><br>[Figure 68]<br><br>❌ ❌ ❌<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>❌ ❌<br><br>[Figure 72]<br><br>C1 C2 C3<br><br>C4 Autorater C5|
|---|

|[Figure 73]<br><br>Formulate Criteria for Each Subtask<br><br>C1:Perspective Adherence C2:Physical Continuity C3: Obstacle Interaction C4:Path Planning Navigation C5:Goal Completion Interaction<br><br>Spatial-Temporal & Dynamic Reasoning / Spatial Obstacle<br><br>[Figure 74]<br><br>This is a robot's firstperson perspective. The task is to go to the kitchen sink and get the paper towels<br><br>[Figure 75]<br><br>c3: Handle the tall wooden stool in a physically plausible manner, i.e., "must" move it aside (e.g., by pushing or grasping with a robotic arm). "Bypassing" the stool is not considered successful task completion.<br><br>...<br><br>...<br><br>[Figure 76]<br><br>[Figure 77]<br><br>✅<br><br>review<br><br>Criteria Refiner|
|---|

- Figure 4. The evaluation framework of Gen-ViRe. (a) Data Curation: Shows the benchmark development process, including defining the taxonomy, collecting data from multiple channels (web, existing datasets, AI generation), and designing & validating prompts through Peer Review. (b) Formulation of Evaluation Criteria: Demonstrates the process of formulating detailed, multi-dimensional evaluation criteria (as shown by C1-C5 in the figure) for each prompt of every subtask. (c) VLM-based Autorating Framework: Illustrates how the VLM (Autorater) conducts item-by-item analysis and automatic scoring of the generated videos based on the specific criteria defined in (b).

### 5. Evaluation Methodology

in adhering to complex instructional formats. For instance, in our Embodied Spatial Obstacle task, the model is provided with an initial state image (e.g., a first-person robot view of a kitchen, blocked by a large table and a high stool) and a high-level objective (e.g., “This is a robot’s first-person perspective. The task is to go to the coffee machine in the kitchen and get the paper towels.”). We deliberately abstain from specifying how to handle the obstacles. A successful output requires the model to autonomously reason about the implicit physical and spatial constraints. The prompt does not mention the obstacles, but the model must infer that (1) it must navigate around the large table, not pass through it, (2) it must perform the mandatory action of physically moving the stool aside, as bypassing or stepping over it is defined as a failure, and (3) the interaction must be performed by a robotic manipulator, not a human hand, to adhere to the “robot’s perspective” constraint.

Formulation of Evaluation Criteria. The core of our evaluation pipeline is the development of detailed criteria for each task subcategory. This formulation process is a unique, hybrid approach combining VLM assistance with multiround human refinement. First, our team drafts preliminary evaluation standards for each task. We then provide these preliminary standards, along with the corresponding input image, text prompt, and task objective, to Gemini 2.5 Pro. The model’s role is to refine these standards into a more detailed, rigorous, and operational evaluation rubric based on the full context of the specific task. Finally, the detailed criteria generated with VLM assistance undergo a final multiperson review and refinement by our team to ensure absolute accuracy and consistency.

VLM-Assisted Evaluation Methods. Our evaluation methodology employs powerful Vision Language Models (VLMs) as automated judges. We utilize Gemini 2.5 Pro [8] as our unified VLM judge, leveraging its respective modality capabilities based on the task’s requirements. For tasks where the evaluation is contingent upon the final visual output (such as analogical or geometric reasoning), we employ Gemini 2.5 Pro as an Image VLM judge. For more complex dynamic tasks (such as planning and spatial-temporal reasoning) that require assessing the entire generated process, we utilize Gemini 2.5 Pro as a Video VLM judge. Crucially, for every task, the designated VLM judge (Gemini 2.5 Pro) is provisioned with a detailed, sub-categoryspecific set of criteria. The judge then decomposes and assesses the model’s output against each criterion, providing an independent score for its decision. This criteria-

Iterative Peer Review Process. To ensure the clarity and robustness of all task prompts, we implemented a strict, iterative peer-review process. A task draft formulated by one annotator is submitted to at least one other independent annotator for review. This review scrutinizes the task for clarity, potential ambiguities, and whether the ground truth is an indisputable, sole answer. Any flagged issues are returned for team discussion and revision. A primary focus of this process is the resolution of “ambiguous references,” a common source of model error. Our annotation team is trained to identify and rectify such vague language (e.g., ambiguous pronouns like “it” or “that”), replacing them with precise descriptions to ensure high prompt fidelity.

centric approach ensures a consistent and rigorous evaluation across the entire benchmark.

### 6. Experiments

##### 6.1. Experimental Setup

Evaluated Models. We evaluated a comprehensive suite of state-of-the-art (SOTA) video models, including Kling-

v1, Seedance-1.0-Pro, Seedance-1.0-Lite, Veo-3.1, Sora-2, Wan-2.5, and Hailuo-2.3. Videos were generated in either 16:9 or 9:16 aspect ratio, determined by the native orientation of the input task image. For generation duration, we adopted the default 5-second for the Hailuo-2.3, 8-second for Veo-3.1 and Sora-2, and the default 10-second for Klingv1, Seedance-1.0-Pro, Seedance-1.0-Lite, and Wan-2.5.

Evaluation Metrics. The VLM-assisted evaluation process described in Section 5 yields an output score for each task. To aggregate these individual scores, we follow the aggregation strategy of MEGA-BENCH [6]. The score for each individual task is first normalized to a consistent [0, 1] range, where 1.0 signifies perfect adherence to all standards. Subsequently, to report comprehensive performance, we compute the macro-mean score of all normalized scores.

##### 6.2. Main Results 6.2.1. Qualitative Analysis and Case Studies

###### Case 1: Analogical Reasoning (Rotation and Color)

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

| |
|---|

[Figure 82]

last frame of Sora 2 last frame of Veo 3.1

input image ground truth image

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

| |
|---|

[Figure 87]

input image ground truth image last frame of Sora 2 last frame of Veo 3.1

Figure 6. Showcase of Analogical Reasoning by Sora-2 and Veo-3.1. On the complex Rotation task (top row), both Sora-2 and Veo-3.1 failed to recognize and apply the abstract rotational rule. In sharp contrast, both models easily solved the simpler Color analogy (bottom row), which only required matching an attribute.

I2V Prompt. Create a smooth animation to generate the missing object in the lower right region and solve the visual analogy. The original three objects must remain still. Static shot, no zoom no pan no dolly.

Takeaway. Current models’ analogical reasoning performance correlates directly with task abstraction complexity: they can easily solve simple attribute matching (e.g., color) but expose core deficits when handling abstract, rule-based transformations (e.g., rotation).

Case 2: Spatio-Temporal Reasoning (Spatial Obstacle)

[Figure 88]

[Figure 89]

[Figure 90]

input image

[Figure 91]

[Figure 92]

[Figure 93]

mid frame of Sora 2

- Figure 7. Showcase of Sora-2’s failures in Spatio-Temporal & Dynamic Reasoning. The top row shows input images; the bottom shows Sora-2’s mid-frames. These frames reveal fundamental failures in simulating basic physical laws: (Left) violating object permanence by showing a dog phasing through a closed glass door ; and (Middle) failing to simulate a continuous process, instead spawning paper towels into the scene ; (Right) depicting telekinesis, where an object is retrieved without contact.

Takeaway. Abstract logical reasoning (e.g., following algorithmic rules) and physical reality simulation (e.g., adhering to physical plausibility) are two distinct capabilities. The model’s excellence in the former does not equate to mastery in the latter.

Case 3: Algorithmic & Logical Reasoning (Geometry)

[Figure 94]

input image ground truth image last frame of Sora 2 last frame of Veo 3.1

| |
|---|

[Figure 95]

[Figure 96]

[Figure 97]

| |
|---|

- Figure 8. Showcase of Sora-2’s in geometry task. Both Sora2 and Veo-3.1 failed to identify the existing point D in the static image. This perceptual error caused them to ignore the instruction connect point C to point D’ and instead hallucinate a new point D, connecting C to this incorrect location.

Takeaway. Models expose a critical perceptual flaw in complex, symbol-reliant tasks (like geometry): they fail to parse in-context abstract symbols (like "D") as addressable logical components, instead misinterpreting them as incidental visual noise.

Case 4: Algorithmic & Logical Reasoning (Sudoku)

input image last frame of Sora 2

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

| |
|---|

mid frames of Sora 2

- Figure 9. Showcase of Sora-2’s Sudoku task. In the Sudoku task, Sora-2 exhibits an emergent, human-like thinking process. The model uses a question mark (?) as a placeholder for the unknown value in the third row. This suggests it can hold an internal state of the problem (“this cell is unsolved"). Following the placeholder, the model generates frames that simulate the “moving" of numbers

(2) into their correct, logically-deduced positions.

Takeaway. Sora-2’s Sudoku solving process (using “?" placeholders and “moving" numbers) indicates it is acquiring a genuine algorithmic capability, as it is simulating the problem-solving process of “following Sudoku rules" rather than just pattern-matching the final answer.

Table 1. Performance comparison across different reasoning categories for various video generation models. We highlight the top-three performing models in each column with varying shades of purple, where a darker shade indicates a higher rank.

SpatioTemporal

###### Methods #Videos Avg. Abstract Algorithmic

Analogy Perceptual Planning

& Logical

Kling-v1 [28] 360 0.198 0.071 0.057 0.117 0.140 0.443 0.359 Seedance-1.0-Lite [14] 360 0.279 0.087 0.256 0.083 0.146 0.572 0.532 Seedance-1.0-Pro [14] 360 0.301 0.154 0.164 0.083 0.171 0.609 0.621 Wan-2.5 [35] 360 0.490 0.412 0.411 0.500 0.378 0.702 0.536 Veo-3.1 [16] 360 0.486 0.440 0.451 0.367 0.386 0.722 0.550 Hailuo-2.3 [32] 360 0.493 0.494 0.355 0.383 0.425 0.778 0.524 Sora-2 [2] 360 0.560 0.604 0.472 0.483 0.496 0.768 0.537

Abstract Reasoning

| | |
|---|---|
| | |

Spatial Reasoning

Algorithmic Logical Reasoning

Analogy Reasoning

Planning Reasoning

Perceptual Reasoning

- Figure 5. Left: The main chart compares the overall performance of the 7 state-of-the-art models across the six core cognitive dimensions (Abstract, Algorithmic, Analogy, Perceptual, Planning, and Spatial Reasoning). Right: The six sub-charts provide a detailed performance breakdown for the individual subtasks within each dimension. The legend (bottom) links each colored line to its respective model.
- 6.2.2. Quantitative Results

### 7. Conclusion

Our evaluation of 7 state-of-the-art models across six reasoning dimensions reveals a clear performance hierarchy, as shown in Figure 5 and Table 1. Sora-2 achieves the highest overall score (0.560), establishing the top tier with particularly strong performance in the most cognitively demanding domains: “Abstract Reasoning” (0.604), “Algorithmic & Logical” (0.472), and “Perceptual” (0.496). The second tier comprises three highly competitive models—Hailuo-2.3 (0.493), Wan-2.5 (0.490), and Veo-3.1 (0.486)—each exhibiting distinct specialized strengths. Hailuo-2.3 achieves the highest score in “Planning” (0.778), showcasing exceptional sequential decision-making capabilities, while Wan-2.5 leads in “Analogy” (0.500), excelling at analogical reasoning. Veo-3.1 has balanced performance, ranking second in both “Algorithmic & Logical” (0.451) and “Planning” (0.722). In contrast, Kling-v1 (0.198) and Seedance1.0-Lite (0.279) form the lower tier with scores substantially below the leading models, indicating considerable room for improvement in reasoning capabilities.

Video generation models are transitioning from visual synthesizers to potential world simulators capable of physically-grounded reasoning. However, without rigorous evaluation, we cannot distinguish genuine understanding from sophisticated pattern matching. Gen-ViRe addresses this by providing systematic assessment across six cognitive dimensions, establishing the foundation for quantitative science in generative visual reasoning. Our experiments reveal a critical gap: while current models achieve impressive visual fidelity, they exhibit limitations in sustained logical coherence, physics compliance, and multi-step planning. By diagnosing these specific deficits—whether in perceptual grounding, spatial reasoning, or goal-directed planning—Gen-ViRe provides actionable insights to guide targeted improvements. As AI systems increasingly simulate and interact with physical reality, benchmarks like GenViRe are essential for measuring genuine progress toward intelligent world models that reason about the world, rather than merely rendering plausible pixels.

### References

- [1] Mikel Bober-Irizar and Soumya Banerjee. Neural networks for abstraction and reasoning. Scientific Reports, 14(1): 27823, 2024. 2, 4
- [2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. Technical report, OpenAI, 2024. 2, 3, 8
- [3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024. 2, 3
- [4] Henry R Burke. Raven’s progressive matrices: A review and critical evaluation. The Journal of genetic psychology, 93(2): 199–228, 1958. 2
- [5] Ruth M J Byrne. The Rational Imagination: How People Create Alternatives to Reality. MIT Press, 2007. 2, 4
- [6] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024. 7
- [7] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. 4, 5
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6
- [9] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 3
- [10] Carl Doersch and et al. Tap-vid: A benchmark for tracking any point in any video. In Advances in Neural Information Processing Systems, 2022. 2
- [11] Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. AIGCBench: Comprehensive evaluation of image-to-video content generated by AI. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, 3(4):100152,

2024. 2

- [12] Tongtong Feng, Xin Wang, Yu-Gang Jiang, and Wenwu Zhu. Embodied ai: From llms to world models. arXiv preprint arXiv:2509.20021, 2025. 2
- [13] Yumeng Fu, Jiayin Zhu, Lingling Zhang, Bo Zhao, Shaoxuan Ma, Yushun Zhang, Yanrui Wu, and Wenjun Wu. Geolaux: A benchmark for evaluating mllms’ geometry performance on long-step problems requiring auxiliary lines. arXiv preprint arXiv:2508.06226, 2025. 5
- [14] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 3, 8

- [15] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2):155–170, 1983. 2, 4
- [16] Google DeepMind. Veo 3 technical report. Technical report, Google DeepMind, 2025. 2, 3, 8
- [17] Ji Gu, Boyuan Chen, and Yuke Zhu. Controlvla: Few-shot object-centric adaptation for pre-trained vision-languageaction models, 2024. 4, 5
- [18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 3
- [19] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, Chi Zhang, Li Li-jia, and Yongxin Ni. Video-bench: Human-aligned video generation benchmark. arXiv preprint arXiv:2504.04907, 2025. 2
- [20] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2, 3
- [21] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [22] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3
- [23] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 1, 3
- [24] Philip N Johnson-Laird. Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness. Harvard University Press, 1983. 2, 4
- [25] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011. 2, 4
- [26] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3
- [27] Tengchuan Kou, Xiaohong Liu, Zicheng Zhang, Chunyi Li, Haoning Wu, Xiongkuo Min, Guangtao Zhai, and Ning Liu. Subjective-aligned dataset and metric for text-to-video quality assessment. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 7793–7802, 2024. 2
- [28] Kuaishou Technology. Kling ai, 2024. 2, 3, 8
- [29] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional highresolution computer use. In Proceedings of the 33rd ACM

International Conference on Multimedia, pages 8778–8786,

2025. 5

- [30] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. FETV: A benchmark for fine-grained evaluation of open-domain text-tovideo generation. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track,

2023. 2

- [31] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22139–22149, 2024. 2
- [32] MiniMax. Hailuo ai video generation, 2024. 8
- [33] Allen Newell and Herbert A Simon. Human Problem Solving. Prentice-Hall, 1972. 2, 4
- [34] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [35] Alibaba Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314,

2025. 3, 8

- [36] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022. 1, 3
- [37] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2, 3, 5
- [38] Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, and Chuang Gan. Mindjourney: Test-time scaling with world models for spatial reasoning. arXiv preprint arXiv:2507.12508, 2025. 5
- [39] Eunice Yiu, Maan Qraitem, Anisa Noor Majhi, Charlie Wong, Yutong Bai, Shiry Ginosar, Alison Gopnik, and Kate Saenko. Kiva: Kid-inspired visual analogies for testing large multimodal models. arXiv preprint arXiv:2407.17773, 2024. 5
- [40] H. Zhang, K. Liu, L. Zhao, et al. Embodiedvsr: Dynamic scene graph-guided chain-of-thought reasoning for visual spatial tasks. arXiv preprint arXiv:2503.11089, 2025. 2

