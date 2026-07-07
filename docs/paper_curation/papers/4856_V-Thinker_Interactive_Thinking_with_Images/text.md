# arXiv:2511.04460v2[cs.CV]18Dec2025

## V-Thinker: Interactive Thinking with Images

Runqi Qiao1*†, Qiuna Tan1*†, Minghan Yang1, Guanting Dong1, Peiqing Yang1, Shiqiang Lang1, Enhui Wan1, Xiaowan Wang1, Yida Xu1, Lan Yang1, Chong Sun2, Chen Li2‡, Jing LYU2, Honggang Zhang1‡ 1Beijing University of Posts and Telecommunications 2WeChat Vision, Tencent Inc. qrq@bupt.edu.cn qiunatan@bupt.edu.cn https://github.com/We-Math/V-Thinker

Abstract

Empowering Large Multimodal Models (LMMs) to deeply integrate image interaction with long-horizon reasoning capabilities remains a long-standing witness in this field. Recent advances in vision-centric reasoning explore a promising “Thinking with Images” paradigm for LMMs, profoundly shifting from image-assisted reasoning to image-interactive thinking. While this milestone enables models to focus on fine-grained image regions, progress remains constrained by narrow visual tool spaces and task-specific workflow designs. To bridge this gap, we present V-Thinker, a generalpurpose multimodal reasoning assistant that enables interactive, vision-centric thinking through end-to-end reinforcement learning. V-Thinker comprises two key components: (1) a Data Evolution Flywheel that automatically synthesizes, evolves, and verifies interactive reasoning datasets across three dimensions—diversity, quality, and difficulty; and (2) a Visual Progressive Training Curriculum that first aligns perception via point-level supervision, then integrates interactive reasoning through a two-stage reinforcement learning framework. Furthermore, we introduce VTBench, an expert-verified benchmark targeting vision-centric interactive reasoning tasks. Extensive experiments demonstrate that V-Thinker consistently outperforms strong LMM-based baselines in both general and interactive reasoning scenarios, providing valuable insights for advancing image-interactive reasoning applications.

#### 1. Introduction

"The soul never thinks without an image."

— Aristotle

*Equal contribution. †Work done as interns at WeChat, Tencent Inc. ‡Corresponding author. zhhg@bupt.edu.cn; chaselli@tencent.com

[Figure 1]

Figure 1. The three paradigms of vision-centric reasoning.

Humans often simplify complex problem-solving via heuristic multi-modal interactions, especially in vision-centric reasoning [54, 68]. In challenges like geometric proofs, people interactively add auxiliary lines or sketches [19, 55], modeling visual relations to solve geometric problems more intuitively. In recent years, visual language models (LMMs) demonstrate exceptional performance in stepwise reasoning tasks, fostering expectations for their development into interactive, image-oriented thinking [5, 7, 15, 21, 37, 52, 71, 74]. However, despite producing lengthy and coherent chain-of-thought (CoT), these models often detach from visual grounding, leading to hal-

lucinations [20, 28, 30]. This indicates that current visual reasoning depends more on linguistic priors than on visual perception.

To address these challenges, OpenAI’s o3 model first actively interacts with images during reasoning via visual tools (e.g., cropping, rotation), shifting the paradigm from vision-assisted reasoning to vision-centric thinking [39, 49]. Building on this milestone, recent efforts attempt to reproduce o3-like interactive thinking through end-to-end visual reinforcement learning [61, 82]. Moreover, Thyme [79] autonomously generates executable code to render diverse visual operations on images throughout reasoning. However, the available visual actions of them are still limited and heavily depend on precise spatial localization. To further expand the visual tool space and broaden interactive thinking patterns, foundational works such as DeepSketcher [64, 73] enable LMMs to autonomously add auxiliary lines within images, explicitly modeling logical relations between visual elements. While insightful for vision-centric reasoning, their tool designs are often tightly coupled to specific task types. Moreover, a heavy reliance on "Image2Code" pipelines for image editing struggles to accurately depict spatial relationships between visual elements and may introduce extra noise [17, 76]. This raises a fundamental research question: How can we effectively integrate image interaction into the visual chain-of-thought process?

In this paper, we introduce V-Thinker, a general-purpose multimodal reasoning assistant that fosters interactive vision-centric thinking via end-to-end reinforcement training. As shown in Figure 1, V-Thinker aims to simplify complex problems via autonomous interaction with images, thereby advancing the next generation of vision-centric reasoning paradigms. Specifically, V-Thinker introduces a comprehensive vision-centric post-training paradigm, comprising two key components: the "data evolution flywheel" and a "progressive interactive training curriculum" spanning perception to reasoning.

Data Evolution Flywheel. An ideal vision-centric interactive reasoning model should generalize across real-world diverse tasks. To this end, V-Thinker targets three fundamental dimensions of interactive reasoning data synthesis: (1) Diversity. We directly generate QA pairs using knowledge concepts and visual tool systems, shifting from data expansion to genuine data creation. Then, the flywheel iteratively enlarges both the concept and visual tool sets using newly synthesized data, thereby sustaining a continuous stream of diverse datasets. (2) Quality. To ensure strict quality control, we implement a coordinated calibration mechanism where a data checker rigorously screens textual, visual, and image-action dimensions while a repairer calibrates annotations across modalities, ensuring high fidelity. (3) Difficulty. Building on the preceding stages, we further introduce a pro-

gressive expansion stage that enriches the difficulty ladder by incorporating parallel and sequential extension strategies.

Through a three-stage data synthesis flywheel, we obtain a high-quality visual interactive reasoning dataset, namely V-Interaction-400K.

Visual Progressive Training Curriculum. Following the data evolution flywheel, we design a two-stage curriculum that progressively builds robust perception and then aligns it with interactive reasoning capabilities: (1) Perception Alignment. We model the visual space through three key dimensions and synthesize perception-specific datasets to enhance the LMM’s localization capabilities, named V-Perception40K. Furthermore, the LMM is fine-tuned on this dataset to align the localization and referencing of visual anchors. (2) Interactive Reasoning Alignment. Built upon the perceptual foundation, we implement cold-start supervised fine-tuning followed by RL within a sandboxed executor environment, enabling the LMM to autonomously generate executable code that interacts with visual elements and maintains coherent reasoning chains grounded in visual evidence.

To thoroughly evaluate LMMs’ visual-interactive reasoning capabilities, we introduce VTBench, a benchmark targeting tasks that inherently demand visual interaction. Each instance is verified by domain experts and sourced from diverse public datasets. In summary, the key contributions are as follows:

- • We formalize Interactive Thinking with Images and introduce V-Thinker, an end-to-end multimodal reasoner that bridges visual grounding and interactive thinking through code-driven visual tools.
- • We propose a Data Evolution Flywheel that automatically synthesizes, evolves, and verifies interactive reasoning datasets across three dimensions: Diversity, Quality, and Difficulty, and further release a large-scale, highquality visual interaction dataset, V-Interaction-400K, which can also be extended to image-to-code and other vision-language tasks.
- • We introduce a Visual Progressive Training Curriculum that first aligns perception via point-level supervision using a high-quality dataset V-Perception-40K, and then aligns interactive reasoning through a two-stage curriculum training framework.
- • We introduce VTBench, an expert-reviewed benchmark featuring standardized protocols. Extensive experiments show that V-Thinker consistently outperforms mainstream LMM-based baselines on both interactive and general reasoning scenarios, providing valuable insights for advancing image-interactive reasoning.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Math Physics Music DroneBiologyView Logic

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Chemistry

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

### …

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Write Python code to load and edit images.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

### …

[Figure 30]

…

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

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

[Figure 50]

Figure 2. Representative examples of V-Thinker’s knowledge-driven synthesis spanning diverse reasoning domains.

#### 2. Related Work

performing executable visual tools (e.g., cropping) as part of their reasoning chain. Meanwhile, DeepSketcher [73] explores implicit visual reasoning, where models leverage abstract visual cues instead of explicit pixel-level manipulation. Building on this progress, our work advances interactive thinking with images by enabling the model to autonomously generate, execute, and iteratively refine visual code during reasoning. This opens a new perspective on integrating visual interaction into reasoning, paving the way toward more intuitive and human-like multimodal cognition.

Multimodal Reasoning. Recent advances in large language models (LLMs) and multimodal large language models (MLLMs) have significantly enhanced their reasoning capabilities across diverse tasks [2, 3, 8–14, 22, 23, 29, 41, 42, 48, 50, 51, 53, 59, 66]. Recent efforts such as MathVista [34], MathVision [57], MathVerse [77], We-Math [40], Dynamath [88], LogicVista [67], and VisuLogic [69] have introduced comprehensive benchmarks that systematically evaluate model performance across mathematical, logical, and visual reasoning scenarios. Methodologically, prior studies have improved visual–textual alignment [46, 58, 78], incorporated step-wise reasoning [35, 87], and explored RL–based optimization to strengthen multimodal reasoning [1, 4, 25, 33, 36, 43, 56, 70, 75, 83]. In particular, RL-based frameworks such as MM-Eureka [37] and VisionR1 [25] have introduced reinforcement learning into visual reasoning, revealing new possibilities for enhancing model reasoning depth. These works have laid a foundation for the continued advancement of visual reasoning.

#### 3. Preliminary

##### 3.1. Problem Formulation

Interactive Thinking with Images. We envision an ideal vision-centric reasoning paradigm where reasoning unfolds through progressive interaction with the image—by perceiving, modifying, and reflecting on its visual state. Building upon this idea, V-Thinker treats reasoning as a code-driven visual interaction process. At each reasoning step, the model generates a textual thought rt and, when necessary, a code segment ct that operates on the current image It. The environment E executes ct, resulting in an updated image It+1. Formally,

Thinking with Images. Interactive visual reasoning is a long-term research vision. Early explorations, such as LLaVA-Plus [31] and Visual Sketchpad [24], pioneered this direction by enabling models to conduct visual operations during reasoning. These studies laid an important foundation for integrating interactive perception into multimodal reasoning systems. With the growing success of reinforcement learning (RL) in visual reasoning, OpenAI’s o3 [39] introduced the concept of thinking with images. Subsequent works [6, 26, 27, 47, 63, 65, 72, 73, 81, 84, 85] have further developed this paradigm, provides a fresh perspective for the field of agentic RL. In particular, DeepEyes [82] and Thyme [80] employ RL-based training to guide models in

F : (Q,I0)  → (R,A), R = {(rt,ct,It+1)}Tt=1, It+1 = E(It,ct)

where Q is the task query, R the reasoning trajectory, and A the final answer. Through this interactive process, the model reasons by generating code to modify the image and leveraging the resulting visual feedback to guide subsequent reasoning.

##### 3.2. Rethinking on Data Synthesis Paradigm

Traditional vision-centric reasoning datasets are built upon manually defined tasks, where models act as solvers gen-

erating chain-of-thought reasoning and answers. This data synthesis paradigm limits diversity and scalability, while interactive reasoning requiring precise spatial and logical alignment remains challenging.

Image Code Image

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

With the growth of modern model generation capabilities, we re-examine this paradigm from a new perspective:

In data synthesis, can models evolve from solving problems to creating problems?

To end this, we explore this paradigm from two directions. (1) Role Transformation: from "solver" to "creator"

Visual Tool Code Edited Visual States

[Figure 57]

[Figure 58]

[Figure 59]

Traditional data synthesis methods employ models as solvers to distill their reasoning trajectories, based on the core assumption that models cannot directly generate high-quality multimodal QA pairs, particularly those involving complex geometric shapes and spatial relationships. However, this approach constrains the diversity and solvability of synthesized samples.

[Figure 60]

[Figure 61]

[Figure 62]

In this work, we revisit the role of vision-language models in data synthesis, revealing that existing strong LMMs are capable of reaching the creator level in data synthesis. As shown in Figure 3, GPT-5 can directly generate Python code to render high-quality original images along with corresponding auxiliary line diagrams and reasoning trajectories.

Figure 3. The rendering process from code to image.

tive vision-centric reasoning through end-to-end reinforcement learning. As shown in Figure 4, V-Thinker comprises two core components:

Figure 2 shows representative rendered image comparisons. The synthesis quality far exceeds our initial expectations. The exceptional coding capabilities enable finegrained, outstanding image editing, such as highlighting chemical elements and annotating vertical guide lines, directly creating image annotations. Therefore, we shift from the distillation paradigm, enabling innovative generation of complex problems, diagrams, and reasoning trajectories rather than passively responding to predefined tasks.

- 1. Data Evolution Flywheel (§4.1): We automatically synthesize, evolve, and verify interactive reasoning datasets across three dimensions: Diversity, Quality, and Difficulty.
- 2. Visual Progressive Training Curriculum (§4.2): We introduce a two-stage training framework to achieve progressive alignment from perception understanding to visioncentric interactive reasoning patterns.

Below, we delve into the specifics of our approach.

(2) Knowledge-Driven Representation. Knowledge concepts serve as anchors for human knowledge, where an ideal comprehensive knowledge system can generate reasoning data covering diverse scenarios. We therefore move beyond using seed samples as references and instead employ structured knowledge systems, where knowledge concepts serve as condensed representations of reasoning semantics, capturing diverse real-world scenarios. Knowledge specifies what to reason about, while models determine how to unfold reasoning through interaction. As shown in Figure 2, by providing only knowledge concepts and their descriptions, knowledge-driven synthesis produces problems with precise spatial alignment, coherent reasoning, and diverse visual structures, broadening the construction and evolution of reasoning data.

##### 4.1. Data Evolution Flywheel

Building on the knowledge-driven paradigm outlined in § 3.2, we design the Data Evolution Flywheel, an automated, scalable, and verifiable framework for synthesizing interactive reasoning data. The framework is divided into three processes (as shown in Figure 4):

- • Knowledge-driven Evolution (§4.1.1) generates reasoning data through the co-evolution of knowledge and tool sets, iteratively expanding both to yield diverse problems and reasoning trajectories.
- • Coordinated Calibration (§4.1.2) verifies the correctness of generated questions, rendered images, and edited visual states, ensuring consistency across text, code execution, and visual outcomes.
- • Progressive Expansion (§4.1.3) enhances reasoning difficulty via iterative step extension and compositional knowledge integration, gradually constructing more complex and challenging reasoning chains.

#### 4. Methodology

Overview. This section introduces V-Thinker, a generalpurpose multimodal reasoning assistant that enables interac-

[Figure 63]

Figure 4. The Data Evolution Flywheel framework: Left: knowledge-driven evolution mechanism. Middle: coordinated calibration and progressive expansion stages. Right: representative synthetic QA instances generated through the flywheel.

###### 4.1.1. Diversity: Knowledge-driven Evolution

Prior data synthesis methods rely on curated seed images, inherently constraining diversity. We propose that foundational knowledge concepts and visual tool system serve as the most granular anchors for synthesizing interactive data, enabling orthogonal and diverse generation. Our pipeline avoids distillation and instead instructs a strong LMM to create datasets from scratch.

Following this guideline, given an initial knowledge system K0 derived from We-Math 2.0 [43] and a curated taskcentric tool set T0, knowledge and tools jointly drive iterative data synthesis through a co-evolutionary loop (Algorithm 1).

At each iteration n, sampled combinations Combos(K) from the current knowledge system are provided to the strong generator G (e.g. GPT5) to construct reasoning data:

(DK,Tˆ) = G(Combos(K)). (1)

Each generated instance contains a question Q, the original code c0 that renders the problem-specific image I0, the corresponding tool predictions Tˆ, and a reasoning trajectory R = {(rt,ct,It+1)}T

t=1, where each executable code ct renders the visual component It+1. Figure 3 illustrates the image rendering process from code to image.

R

In parallel, tool-driven generation applies the same procedure on Combos(T ), producing data DT and predicted knowledge Kˆ:

(DT,Kˆ) = G(Combos(T )). (2)

Together, these complementary processes constitute a coevolutionary loop in which K and T continuously generate new counterparts. The newly predicted elements are then

Algorithm 1 Constructive Evolution for Dataset Synthesis

Require: Initial knowledge set K0, tool set T0, generator G, ex-

pansion rules ΦK, ΦT, iterations N

Ensure: Evolved knowledge K∗, tools T ∗, and dataset Dinit

- 1: K = K0, T = T0, Dinit = ∅ ▷ Initialize from scratch
- 2: for n = 1, 2, . . . , N do
- 3: (DK, Tˆ) ← G(Combos(K)) ▷ Generate QA and predicted tools
- 4: (DT, Kˆ) ← G(Combos(T )) ▷ Generate QA and predicted knowledge concepts
- 5: Dinit ← Dinit ∪ DK ∪ DT ▷ Accumulate new samples
- 6: ∆K ← ΦK(Kˆ, K) ▷ Knowledge expansion
- 7: ∆T ← ΦT(Tˆ, T ) ▷ Tool expansion
- 8: K ← K ∪ ∆K, T ← T ∪ ∆T
- 9: end for
- 10: Output: K∗ = K, T ∗ = T , Dinit

integrated via expansion functions ΦK and ΦT:

K ← K ∪ ΦK(Kˆ,K), T ← T ∪ ΦT(Tˆ,T ), (3)

where ΦK and ΦT filter, merge, and normalize novel elements during incorporation via BGE-based hierarchical clustering. The co-evolution mechanism among synthetic data, knowledge concepts, and visual tool systems is illustrated in the Figure 4 (left).

Through repeated execution over N rounds, the system gradually enriches both K and T , ultimately yielding an initial dataset Dinit that serves as the foundation for subsequent calibration and expansion stages.

###### 4.1.2. Quality: Coordinated Calibration

After obtaining the synthesized diverse dataset Dinit, strict quality control is essential. Therefore, we introduce a regula-

tive calibration stage to ensure multi-level consistency across generated samples. This stage consists of two modules.

###### Question

[Figure 64]

[Figure 65]

Arectangularprismisshown.Thethreeedgelengths areindicatedonthefigureas8(width),6(height), and3(depth).LetAbethefront-bottom-leftvertex andBbetheback-top-rightvertex.AB=?

Checker. As shown in Figure 4 (Mid), each instance is examined by a data checking module V that verifies (1) answer correctness, (2) validity of the rendered original image, and (3) coherence of intermediate visual states produced during reasoning. Only samples satisfying all three criteria are retained as valid candidates.

CoT

[Figure 66]

TodeterminethelengthofthespacediagonalABinarectangularprismwithgivenedge lengths(8,6,and3),it'sessentialtovisualizetherighttrianglein3Dspacewhosehypotenuse isAB.Toolsareneededforgeometricclarity:drawingthebasediagonalvectorfromA(frontbottom-left)totheback-bottom-rightcorner(inbaseplane),addingaperpendicularsegment (theverticalheight)fromthatback-bottom-righttoB(theback-top-right),andshadingthe resultingtrianglewilldirectlydisplaytherighttriangleusedtoapplythePythagoreanTheorem. Thisisnecessarybecausein3D,thediagonalABisnotanedge;onlybyconstructingthe auxiliaryrighttriangledoweseewhichsegmentsareperpendicularandthuslegitimatelegsofa righttriangle.

|<think>|
|---|

Repairer. For cases where the textual answer is incorrect but the rendered image is valid and intermediate visual states remain coherent, we follow the principle that a reasoning chain is fundamentally guided by its question, reconstructing the question from original and edited visual states to realign textual and visual reasoning. The reconstructed instances are re-evaluated by V, and the loop repeats until inconsistencies are resolved.

|<code>|
|---|

import numpy as np … # Load the original figure image

[Figure 67]

[Figure 68]

###### Image Edited Visual States

|[Figure 69]<br><br>[Figure 70]|
|---|

image_path = "images/1.png" img = plt.imread(image_path) # Auxiliary tools overlay

FBL = np.array([160, 380]) W, H = 220, 160 d = np.array([90, -70]) ax.plot([FBL[0], BBR[0]], [FBL[1], BBR[1]]) … # Save the processed image … plt.savefig(f"{processed_path}") plt.close()

Through the iterative calibration of data checking and repair, we refine Dinit into a coherent and verified dataset Dverified, which serves as the foundation for subsequent progressive expansion.

|</code>|
|---|

<image>

|<sandbox_output>|
|---|

|</sandbox_output>|
|---|

Withtheauxiliarylinesandhighlightedtriangle,wecannowseethatthebasediagonalisaleg oflength√(8²+3²)=√73,andtheverticaledgeisorthogonaltoitwithlength6.Theright triangleformedby{basediagonal,height,AB}clearlyindicateswecanapplythePythagorean Theoremin3D:AB=√((basediagonal)²+(vertical)^2)=√(73+36)=√109.Thus,theexact spacediagonalis√109.

###### 4.1.3. Difficulty: Progressive Expansion

√109

<

|</think>|
|---|

|<answer>|
|---|

|</answer>|
|---|

To establish difficulty-stratified data partitions and deepen reasoning chain complexity, our intuition is to extend the CoT’s context length to achieve difficulty escalation, thereby inversely mapping more challenging QA pairs. This naturally motivates us to introduce two complementary strategies to progressively expand the difficulty: parallel and sequential extensions, constructing more challenging QA pairs:

Figure 5. A representative sample from the synthesized dataset V-Interaction-400K (D).

often struggle with fine-grained spatial perception, failing to precisely localize points, intersections, and other anchors.

To address this, we develop a visual progressive training curriculum that aligns perception and interaction, starting with perceptual grounding through point-level supervision (§ 4.2.1) and advancing to interactive reasoning via progressive alignment training (§ 4.3).

- (1) Parallel Extension. New auxiliary constructions are

introduced independently of the existing ones, providing additional key observations that complement the original reasoning to reach the final answer.

- (2) Sequential Extension. New auxiliary constructions

are closely linked to the existing reasoning, requiring prior results or original geometric entities to define subsequent operations. For example, if the original auxiliary line is DM, a new line perpendicular to DM can be introduced to support further deductions.

###### 4.2.1. Perception Alignment

Perception Data Synthesis. A critical aspect of perception training is synthesizing perception-specific data. Following the paradigm in Section 3.2, we model visual space via three dimensions: element relations, element count, and knowledge concepts. These aspects jointly define visual information, with style governed by knowledge and complexity by element count.

Under these two strategies, a subset of verified data Dverified is sampled and provided to the expansion model, which generates extended reasoning code segments and corresponding visual states. The generated extensions are revalidated by the verification module V and iteratively refined until convergence. We limit the maximum extension depth to three steps and merge all results to form the final dataset D. Figure 5 display a representative sample from the synthesized dataset D.

- • Element Relations (PE): We model visual elements such as points, lines, angles, and circles, based on foundational geometry principles [18], and extend this set with textual and symbolic elements for alignment with real-world scenarios. This modeling integrates both element types and their spatial relationships, such as "point on line" and "point outside circle," capturing the core geometric interactions in visual reasoning.
- • Element Count (PC): We sample the number of elements from a normal distribution N(µ = 8,σ2 = 4) to control

##### 4.2. Visual Progressive Training Curriculum

For reliable vision-centric interactive reasoning, accurate perception of visual elements is crucial. While recent multimodal models demonstrate strong reasoning ability, they

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

##### 4.3. Interactive Reasoning Alignment

###### Annotations

[Figure 76]

Element Relations

[Figure 77]

“point”:“{‘P4’:(280,80)}",…

……

[Figure 78]

”line“:"x1":100,"y1":260,"x2":160,"y2":200“

Data Selection. For comprehensive training, we select data from the following sources:

…

Point-on-line

[Figure 79]

Code

[Figure 80]

Point-inside-circle

[Figure 81]

[Figure 82]

SFT Dataset: In the Cold-Start Fine-tuning stage, we first construct the Dperception dataset to fine-tune V-Thinker with fine-grained perception capabilities, and then align D to equip it with basic interactive reasoning abilities.

[Figure 83]

[Figure 84]

[Figure 85]

Render Image

[Figure 86]

Engineering

[Figure 87]

……

RL Dataset: To progressively align V-Thinker for robust interactive reasoning, we consider the following two components of the dataset: (1) Open-Source Samples: We use the We-Math 2.0 [43], MMK12 [37] and ThinkLite [60], which provide a broad range of visual reasoning data, from basic to complex, ensuring comprehensive coverage of visual reasoning tasks. (2) Targeted Sampling from D: We select data from D where the base model gives incorrect answers for the original image but correct answers for the edited version. A total of 3,000 instances are sampled from D, ensuring a mix of simpler and more complex data, aimed at improving the model’s reasoning ability with visual states.

Chemistry

Circle

Art

Point Angle Lines

Text

Knowledge Concepts Element Count

###### C

……

###### Surface Level Semantic Level Integrated Reasoning

[Figure 88]

[Figure 89]

[Figure 90]

Detect and output the pixel coordinatesofpointsP1,P2,and P3intheimage.

Detect and output the pixel coordinates of the peak vertex of thepolylinewheretheslopeswitches fromincreasingtodecreasing.

Findallpointswherethepolyline intersects the horizontal line at the Y-value 20 and output their pixelcoordinatesasadictionary.

[Figure 91]

[Figure 92]

[Figure 93]

{'P3':(220,140),'P5':(340,140)}

{'P1’:(100,260),'P2’:(160,200), 'P3’:(220,140)}

{'P4':(280,80)}

Figure 6. The overview of the perception data synthesis.

task complexity, where µ represents the mean and σ2 the variance.

• Knowledge Concepts (PK): We sample from the knowledge system K to define reasoning objectives for each task.

Cold-Start Fine-tuning To enable the model to generate code and interact with visual elements, we apply standard supervised fine-tuning objective on the dataset D, consisting of input-output pairs (xi,yi), where xi includes both the problem statement and the corresponding visual information, and yi represents the reasoning chain with Python code and the edited visual elements. The objective is to minimize the loss function:

As shown in Figure 6, each combination of these dimensions generates corresponding coordinates, which serve as visual tags representing the spatial relationships between elements. Using these coordinates, we then structure the tasks into three levels of complexity: surface-level perception, semantic-level reasoning, and integrated reasoning:

i,yi)∼D [log Pθ(yi | xi)] (5)

LSFT(θ) = −E(x

- • Surface-level perception: Basic tasks, such as identifying the coordinates of a specific point, for example, point A.
- • Semantic-level reasoning: Tasks requiring geometric understanding, such as identifying the top-left vertex of a cube.
- • Integrated reasoning: Tasks combining perception and computation, such as finding the center of a cube based on its dimensions.

This phase initially enables the model to generate code that reads and interacts with images to perform reasoning tasks.

Reinforcement Learning for Interactive Reasoning We apply reinforcement learning (RL) to enhance the model’s interactive reasoning capabilities. Building on the Thyme [80], we use its sandbox environment to execute the model’s generated code. The model decodes reasoning tasks into executable code, which interacts with the visual elements, and the resulting outputs are fed back into the reasoning process. For optimization, we adopt Group Relative Policy Optimization (GRPO) [45], which has been shown to be effective for diverse tasks. Given an input question x and a policy model πθ, GRPO enables the reference policy πref to generate a group of G outputs {y1,y2,...,yG} and optimizes the policy by maximizing:

This hierarchical structure generates diverse questionanswer pairs, forming the dataset Dperception, which spans all complexity levels and enhances perceptual capabilities.

Perception Training. The model is trained using supervised fine-tuning, with the objective of minimizing the loss function:

Tj

G

1

min δj,tA˜(t),

LRL(θ) = Ex∼D

[−log Pθ(A | Q)] (4)

LSFT(θ) = E(Q,A)∼D

G j=1 Tj

perception

t=1

j=1

This process enhances the model’s ability to process visual information, particularly for tasks involving point-level localization.

clip(δj,t,1 − ϵl,1 + ϵh)A˜(t) , (6)

where δj,t = ππθ(yj,t|x,yj,<t)

ref(yj,t|x,yj,<t) represents the importance sampling ratio for the t-th token of the j-th output, and A˜(t) denotes the advantage at time step t.

Reward Design. We follow a reward function based on the Thyme framework [80], consisting of three components: accuracy (Racc), formatting (Rformat), and tool usage (Rtool). The total reward is defined as:

R(τ) = Racc(τ)+λ1 ·Rformat(τ)+λ2 ·IRacc(τ)>0 ·Rtool(τ) (7) where IR

acc(τ)>0 is an indicator function ensuring that the tool usage reward is applied only when the final answer is correct and involves at least one tool. We empirically set λ1 = 0.5 and λ2 = 0.3 to balance the contributions of formatting and tool usage.

#### 5. VTBench

To evaluate the model’s vision-centric interactive reasoning capabilities, we introduce VTBench, a benchmark designed for tasks that inherently require interaction with visual elements. The design of VTBench follows the following principles:

- • Specific Interactive Tasks: VTBench prioritizes tasks that inherently require interaction with visual elements, such as drawing auxiliary lines or labeling.
- • Expert Evaluation: Each sample is assessed by a team of five experts, who determine whether visual interaction is necessary for solving the problem. The sample is included if at least three experts agree on its necessity.
- • Diversity and Expansion: To ensure broad coverage, we extend the benchmark by collecting and annotating datasets from multiple sources, including both opensource benchmarks and additional tasks from public platforms, following the same annotation procedure.

- 5.1. Data Collection and Annotation As shown in Figure 7, VTBench is constructed in two stages:

- • Sample Selection: We collect samples from opensource benchmarks: MathVista [34], MathVision [57], MathVerse [77], Dynamath [88], We-Math [40], LogicVista [67], CMM-Math [32], CharXiv [62] and ZeroBench [44], as well as additional samples gathered from public platforms. Expert evaluation is then used to determine whether interaction with visual elements is required. Each sample is included if at least three out of five experts agree on its necessity.
- • Interaction Annotation: We design interaction instructions based on the problem-solving chain of thought for each sample. The expert team manually generates interaction graphs in the interactive interface, which simultaneously captures perceptual coordinates. These interaction

tags are then transformed into QA pairs with GPT-4.1, followed by expert validation to ensure consistency and accuracy.

##### 5.2. Evaluation Dimensions

VTBench evaluates vision-centric interactive reasoning capabilities across three hierarchical dimensions, modeling the problem-solving process from perception to adaptive interaction during reasoning (in Figure 7):

Perception → Instruction-Guided Interaction → Interactive Reasoning

- • Perception: Tasks that assess the model’s ability to recognize and interpret visual elements, such as identifying coordinates.
- • Instruction-Guided Interaction: Tasks where the model receives explicit instructions (e.g., drawing lines or labeling) and must interact with the visual elements to fulfill these instructions.
- • Interactive Reasoning: Tasks that require the model to solve reasoning tasks involving visual interaction, such as drawing auxiliary lines or modifying diagrams.

##### 5.3. Data Statistics and Evaluation Metrics

VTBench comprises 1,500 question-answer pairs across three task types, with 500 samples per task. It incorporates 9 open-source benchmarks across four domains (Logical Reasoning, Geometry, Algebra, Statistics), categorized into three key evaluation metrics:

- • For Perception Task: Considering the inconsistency in coordinate systems across different models, the model is instructed to generate Python code that draws the point at the perceived location. The generated image, compared with the annotated image, is judged by LMMs.
- • For Instruction-Guided Interaction Task: For tasks that require explicit instructions (e.g., drawing lines or labeling regions), the model is instructed to generate Python code to perform the required visual interaction. The result, compared with the annotated image, is judged by LMMs.
- • For Interactive Reasoning Task: For reasoning tasks, the model generates answers, which are then evaluated by Large Language Models (LLMs) based on correctness.

#### 6. Experiments

##### 6.1. Experimental Setup

Datasets. In this paper, we use two constructed datasets for supervised fine-tuning: V-Perception-40K for perceptual alignment and V-Interaction-400K for interactive alignment. In the reinforcement learning stage, 40K samples are sampled from We-Math 2.0 [43], MMK12 [37], ThinkLite [60], and V-Interaction-400K. All datasets are curated in compliance with copyright and licensing regulations. Experiments

[Figure 94]

VTBench Construction

###### Human Study

###### Data Collection

###### Annotation & Verify

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

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Data Validation Voting Data Validation VTBench

Data Collection

Annotation

###### Visualization

[Figure 114]

Task Example

[Figure 115]

[Figure 116]

[Figure 117]

###### Perception Task

###### Interactive Reasoning Task

Instruction-Guided Interaction Task

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

###### Question:

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

###### Question:

Question: Determine the position of point O, point A and point B.

[Figure 127]

As shown in the figure, the radius of ⊙O is 10, and the distance from the center O to chord AB is

Connect OA and OB, and draw OC⊥AB at point C.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

###### Answer:

[Figure 133]

6. What is the length of AB? Answer: 16

[Figure 134]

{‘O’: (93.5, 91), ‘A’: (33.5, 160), ‘B’: (153.5, 160)}

[Figure 135]

[Figure 136]

Answer: “./images/example.png"

[Figure 137]

[Figure 138]

Figure 7. The construction guideline of our VTBench.

VTBench General Reasoning Perception

Method

Instruct. Interaction

Interactive Reasoning

MathVision Acc.

We-Math Acc.

VisuLogic Acc.

Avg.

Avg.

GPT-4o 12.6 26.0 36.4 25.0 43.8 68.8 26.3 46.3 InternVL3-78B 13.8 19.0 43.4 25.4 43.1 64.2 27.7 45.0 InternVL3-8B 10.4 6.8 33.8 17.0 29.3 58.8 24.9 37.7 LLaVA-OV-1.5-8B 12.2 12.2 30.2 18.2 25.6 56.7 23.7 35.3 InternVL3-2B 3.0 3.4 22.0 9.5 23.3 41.7 24.3 29.8 Qwen2.5-VL-7B 12.6 8.8 31.8 17.7 23.0 61.7 26.0 36.9 V-Thinker-7B 18.6 31.6 40.4 30.2 29.3 62.8 26.6 39.6 ∆ (vs Qwen2.5-VL-7B) +6.0 +22.8 +8.6 +12.5 +6.3 +1.1 +0.6 +2.7

Table 1. Overall performance on VTBench (left) and general reasoning (right). (Instruct. Interaction denotes Instruction-Guided Interaction.)

are conducted on VTBench and three standard visual reasoning benchmarks: MathVision [57], We-Math [40], and VisuLogic [69].

The final SFT checkpoint initializes the RL stage, which is trained for one epoch with eight rollouts per iteration, a learning rate of 5 × 10−7, and a warm-up ratio of 0.05.

Baselines. We conduct our experiments based on the Qwen2.5-VL-7B [3] model and compare our method with two categories of baselines:

For VTBench evaluation, we employ two models as judges: Qwen3-235B-A22B and Qwen3-VL-235B-A22B. For Perception Tasks, the judges follow the prompts defined in Table 4, while for Instruction-Guided Interaction Tasks, they adopt the prompts specified in Table 5. Given the current limitations in model perception capabilities and the sensitivity of evaluation criteria to prompt variations, we fix the prompt version throughout all experiments to ensure consistent and fair judgments.

- • Closed-source models: For example, GPT-4o [38].
- • Open-source general models: Including models like InternVL3 series [59], LLaVA-OneVision-1.5 series [2] and Qwen2.5-VL series [3].

Our evaluation is performed using VLMEvalKit [16], with modifications made to the API-calling components due to network constraints.

For data construction, We employ GPT-5 as the generator G and adopt Qwen3-VL-232B-A22B as the data checking module V. Furthermore, GPT-4.1 is utilized as the repairer and perform progressive expansion.

Implementation Details. For training, we conduct all experiments on 64 × 8 H20 GPUs (i.e., 64 nodes with 8 GPUs per node). Both SFT stages use a learning rate of 1 × 10−5.

[Figure 139]

Perception & Instruction-Guided Interaction Tasks (from VTBench)

Determine the position of the upper-leftmost point connected to an odd number of lines, and return its absolute coordinates.

Determine the position of the y-coordinate 90.

Connect PR Draw an arrow pointing to the red circle.

[Figure 140]

[Figure 141]

[Figure 142]

Input

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

V-ThinkerInputV-ThinkerInputV-ThinkerInputV-Thinker

[Figure 147]

[Figure 148]

Visual Reasoning Tasks (from Open-Source Benchmark)

Are there the same number of big blue trucks and large purple metal double buss?

Is the food half eaten? How many people were born after the end of World War II?

How many tiles does the glass corner shelf in the top-left occupy?

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Theresa follows the line with her pencil, starting at the arrow. In what order does she pass the shapes?

In each move, you may change the color of three adjacent squares in a row or column. What is the minimum number of moves required to form the chessboard pattern shown?

The mouse can move only to neighbouring cells along the arrows. How many distinct paths lead to the cheese?

A 5×5 grid has some squares missing. How many are missing?

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Mathematical Reasoning Tasks (from Our Constructed Validation Set)

[Figure 165]

A Depth-First Search (DFS) starts at the root and traverses each edge twice (down and back). Each edge cost equals the vertical drop between levels: the drop from L_0 to L_1 is 45 units, and each subsequent level drop is scaled by r = \frac{1}{3}. Using the figure to identify the branching and number of edge layers, compute the total DFS traversal cost.

A container is a conical frustum (see figure) with top radius r=3cm and bottom radius R=7cm. For axial height h>0, V(h) = \frac{\pi h}{3}\big(R^2 + Rr + r^2\big). Using only the figure and its scale bar, find the axial height h and compute V(h).

A circular billboard has two identical horizontal bars, AB and CD. Bar CD is obtained by translating AB vertically downward by 12 cm. Each bar measures 16 cm along the glass. R = ( )?

Line L intersects the axes at A(12,0) and B(0,9). Marks on the x-axis are at x = 3, 6, 9. For each x_n, let P_n be the foot of the perpendicular from (x_n, 0) onto L, and define c_n = frac{y_n}{x_n}, where y_n is the y-coordinate of P_n. Which of the following holds? A. (c_2^2<c_1c_3) B. (c_2^2=c_1c_3) C. Insufficient information

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Figure 8. Qualitative analysis of V-Thinker-7B on vision-centric interactive reasoning tasks.

[Figure 174]

Figure 9. Visualization of a series of samples in rollout sampling.

##### 6.2. Main Results

To thoroughly analyze the effectiveness of our V-Thinker in the field of interactive reasoning, we conduct experimental analyses from both quantitative and qualitative perspectives:

Quantitative Analysis. As shown in Table 1, we compare V-Thinker with strong baseline LMMs on VTBench. VThinker demonstrates consistent improvements across tasks requiring vision-centric interactions, yielding the following key insights:

###### (1) Perception Challenges Across LMMs. Despite im-

pressive visual reasoning abilities, perceptual alignment remains a critical bottleneck for advanced models. On VTBench, existing LMMs struggle with fine-grained visual interaction tasks, particularly in identifying spatial relationships and localizing precise points on images. While models like GPT-4o and Qwen2.5-VL excel at visual problemsolving, they underperform on tasks demanding direct visual interaction (e.g., Qwen2.5-VL: 8.8%). This reveals a substantial gap between general visual reasoning and the specific perceptual grounding required for interactive visual reasoning.

###### (2) Effectiveness of V-Thinker. Under identical exper-

imental setups, V-Thinker consistently outperforms baseline LMMs across all three interactive reasoning domains, achieving an average accuracy improvement of 12.5% and maintaining over 6% gains on individual domains. Notably, it achieves over 22% performance improvement in the Instruction-Guided Interaction domain. These results underscore V-Thinker’s superior interactive thinking capability.

Qualitative Analysis. To further showcase V-Thinker’s beyond-expected performance in interactive thinking, we analyze three aspects: visual interaction editing, rollout sampling behavior, and fully complete cases.

(1) Visual Interaction Editing (Fig. 8): On tasks requiring visual interaction, V-Thinker-7B accurately draws

SFT (Per.)

SFT (Int.)

Method

RL MVs WM VS

M0 ✓ ✓ ✓ 29.3 62.8 26.6 M1 ✓ ✓ - 21.4 55.5 24.8 M2 - ✓ ✓ 28.3 62.6 26.1

Table 2. Results of the ablation study. MVs: MathVision; WM: We-Math; VS: VisuLogic.

squares in diagrams, fully identifying and labeling instances. For arithmetic tasks that do not strictly require interaction, the model not only produces correct answers but also proactively annotates images to clarify intermediate steps. In realworld scenarios (e.g., meal images, multi-person photos), V-Thinker likewise generates instructive visual annotations, demonstrating stable and fine-grained visual interaction editing.

###### (2) Rollout Sampling Analysis (Fig. 9): We quantify

sampling results within a single rollout step for the same image. The results show that V-Thinker develops multidimensional derivative reasoning and covers a broader solution space during RL stage, reflecting stronger exploration and more diverse reasoning paths.

###### (3) Complete Cases (Fig. 12): To illustrate the end-to-

end interactive thinking process, we present V-Thinker’s visual reasoning trajectory. During reasoning, the model autonomously generates image-editing code and immediately renders edited outputs, externalizing intermediate states and simplifying the reasoning chain—forming a think–edit loop.

Overall, these three lines of evidence jointly demonstrate that V-Thinker exhibits superior and transferable interactive reasoning capabilities.

##### 6.3. Results on Generalized Reasoning Domain

To further validate V-Thinker’s generalization capability on general reasoning tasks, Table 1 presents its performance across three widely-used benchmarks. V-Thinker-7B demon-

[Figure 175]

Figure 10. Visualization of the evolved knowledge system through the Data Evolution Flywheel.

500

500

Init-2 Init-5 Init-10

Init-2 Init-5 Init-10

NumberofKnowledgeConcepts

400

400

NumberofVisualTools

| |
|---|

| |
|---|

300

300

| |
|---|

| |
|---|

200

200

| |
|---|

| |
|---|

100

100

| |
|---|

| |
|---|

0

0

1 2 3 4 5 Iterations

1 2 3 4 5 Iterations

Figure 11. Scaling analysis of the iterations in the Data Evolution Flywheel.

strates consistent improvements across three benchmarks, with stable gains in vision-centric mathematics and logical reasoning. Notably, without specific in-domain data introducing, V-Thinker achieves substantial improvements on complex multi-step reasoning tasks such as MathVision (+6.3%). These results validate that V-Thinker’s interactive reasoning paradigm generalizes effectively across diverse reasoning domains.

##### 6.4. Ablation Study.

To explore the roles of different modules in V-Thinker, we perform an ablation study in general reasoning scenarios, as shown in Table 2. We have the following observations: (1)

The performance of V-Thinker declines when any training phrase is removed, indicating that all progressive curriculum training stages are effective. (2) Removing perception alignment results in notable performance drop, demonstrating that perception capability serves as a crucial foundation for subsequent interactive reasoning. (3) Ablating RL training brings substantial performance degradation, particularly showing over 6% performance drop in Mathvision and WeMath. This indicates that RL stage is essential for exploring interactive reasoning patterns and enabling effective generalization to general reasoning scenarios.

##### 6.5. Analysis of the Data Evolution Flywheel

Data Evolution Flywheel Effectively Expands Knowledge Systems. Figure 10 illustrates an evolved knowledge system derived from seed concepts through the Data Evolution Flywheel, encompassing 25 distinct domains. The resulting hierarchical structure reaches a maximum depth of 7 layers and 24,767 nodes, demonstrating the flywheel’s capability to construct diverse, large-scale, and multi-level knowledge representations. This scalable architecture establishes a systematic foundation for domain-specific and cross-domain reasoning, facilitating knowledge organization and synthesis across interactive reasoning tasks.

###### Question

[Figure 176]

[Figure 177]

[Figure 178]

How many lines of symmetry does this figure have?

CoT

[Figure 179]

< To determine how many lines of symmetry the figure has, we first need to analyze its structure visually. The figure is composed of two identical 'H' shapes, one above and one below, with their legs aligned vertically and their heads forming a horizontal line… To rigorously test for symmetry, it‘s necessary to overlay candidate axes of symmetry and check if both halves map onto each other under reflection. Drawing the vertical midline through the center of the figure allows us to test for vertical symmetry; drawing the horizontal midline through the center tests for horizontal symmetry. …We proceed by running the following code to overlay these axes on the figure.

|<think>|
|---|

|<code>|
|---|

|<code>|
|---|

import numpy as np … # Load original figure image image_path = "images/7.png" img = plt.imread(image_path) … ax.plot([x_mid, x_mid], [0, h], color='red', linewidth=3,

[Figure 180]

[Figure 181]

- image_path = "images/8.png" img = plt.imread(image_path) # Auxiliary lines: OA and OB … ax.plot([cx, Ax], [cy, Ay], color='red', linewidth=2,

linestyle='--') ax.plot([cx, Bx], [cy, By], color='red', linewidth=2, linestyle='--')

… # Save the processed image with a random suffix … plt.savefig(f"{processed_path}") plt.close()

|</code>|
|---|

|<code>|
|---|

|<think>|
|---|

First, let‘s analyze the problem: We are told that Mary walks from the northernmost point A of a circular flower bed to the easternmost point B…. We are to find the central angle ∠AOB in degrees…

To relate the arc length AB to the central angle, it's necessary to visualize the geometry: drawing radii OA and OB from the center O to points A and B, respectively, forms triangle OAB….. Without these auxiliary lines, it would not be clear how the arc relates to the central angle…Let's execute the auxiliary line code to overlay OA and OB on the image, confirming visually that the arc from A to B subtends a right angle at O.

CoT

[Figure 182]

As shown in the figure, there is a circular flower bed. Mary walks from the northernmost point of the flower bed along the edge to the easternmost point, taking a total of 80 steps. Given that Mary's average step length is 0.628m, what is the measure of ∠AOB in degrees?

[Figure 183]

Question

|<sandbox_output>|
|---|

< <image>

|</sandbox_output>|
|---|

<

Now, having visually confirmed that OA and OB form a right angle at O, we proceed:

- - The total arc length from A to B is 80 × 0.628 = 50.24 m.
- - The arc length formula is s = rθ (with θ in radians).
- - Since the arc is a quarter of the circle, θ = π/2 radians = 90°.
- - Therefore, the central angle ∠AOB is 90°.

|</think>|
|---|

|<answer>|
|---|

< A

|</answer>|
|---|

<

[Figure 184]

[Figure 185]

|<think>|
|---|

To determine the age gap between the two individuals in the image, we need their ages. However, since their ages are not directly labeled, we must rely on visual cues and known facts about the individuals. The woman ….

To visually support this, it's helpful to overlay their birth dates onto the image, as well as highlight the difference in their ages, making the calculation explicit and verifiable from the image itself.\n\nWe proceed by running the following code ….

CoT

[Figure 186]

What is the age gap between these two people in image? (Unit: years)

[Figure 187]

Question

import numpy as np … # Load original figure image

- image_path = "images/9.png" img = plt.imread(image_path) … Aung = (110, 170) Obama = (440, 150) … # Compute age gap and mark it near the center … ax.text(w//2, h//2, f'Age gap: {age_gap} years', fontsize=14, color='#cc3333', ha='center', va='center') … # Save the processed image with ... plt.savefig(f"{processed_path}") plt.close() <

linestyle='--’) ax.plot([0, w], [y_mid, y_mid], color='red', linewidth=3, linestyle='--’)

… # Save the processed image with a random suffix … plt.savefig(f"{processed_path}") plt.close()

|</code>|
|---|

|<sandbox_output>|
|---|

|</sandbox_output>|
|---|

<image>

<

|</code>|
|---|

Now, examining the overlaid image, we see that reflecting the figure across either the vertical or horizontal midline maps the entire figure onto itself. This confirms that there are two lines of symmetry: one vertical and one horizontal. No other axes (such as diagonals) yield such a mapping due to the arrangement of the ‘H’ shapes. Therefore, the figure has exactly 2 lines of symmetry.

|<sandbox_output>|
|---|

|</sandbox_output>|
|---|

<image>

<

Now, having read the annotated image, we see that Aung San Suu Kyi was born on June 19, 1945, and Barack Obama on August 4, 1961. Calculating the age gap: 1961 - 1945 = 16 years. Thus, the age gap between them is 16 years.

< 2

<

|<answer>|
|---|

|</answer>|
|---|

|</think>|
|---|

< 16

<

|<answer>|
|---|

|</think>|
|---|

|</answer>|
|---|

###### Question

[Figure 188]

[Figure 189]

In the rectangular coordinate plane, the graph of the function y = |x-2| + |x-6| is shown in the figure. When t > 4, the line y = t intersects the graph at two points A and B. Let M be the midpoint of segment AB. What is the x-coordinate of point M?

###### CoT

[Figure 190]

First, consider the function y = |x-2| + |x-6|. This function is piecewise linear with breakpoints at x = 2 and x = 6, forming a 'V' shape… For t > 4, the line y = t will intersect the middle flat segment 2 ≤ x ≤ 6, since the left and right branches only reach up to y=4. … To visually confirm this and support the symmetry argument, it is necessary to draw the horizontal line y = t (with t > 4), mark the intersection points A and B, and then draw the vertical line through their midpoint M. This vertical line will pass through x = 4, the axis of symmetry of the V-shaped graph. … Now, let's execute the auxiliary drawing code to visualize these relationships.

|<think>|
|---|

###### import numpy as np … # Load original figure image

|<code>|
|---|

###### import numpy as np … # Load original figure image

[Figure 191]

image_path = "images/6.png" img = plt.imread(image_path) # Draw horizontal line

… ax.plot([X(xmin), X(xmax)], [Y(t), Y(t)], color='#d62728', lw=2, linestyle='--’) … # Save the processed image with a random suffix … plt.savefig(f"{processed_path}") plt.close()

|</code>|
|---|

|</sandbox_output>|
|---|

|<sandbox_output>|
|---|

<image>

<

Having read the generated image, we see that the horizontal line y = t intersects the graph at A=(2,t) and B=(6,t). The midpoint M of segment AB has coordinates !"#! ,& = (4, &). The vertical line through M confirms this symmetry about x=4. Therefore, the x-coordinate of M is 4.

4

<

|</answer>|
|---|

|</think>|
|---|

|<answer>|
|---|

Figure 12. Complete interactive reasoning samples of V-Thinker on open-source benchmarks.

Scaling Analysis of Evolution Iterations. To investigate the relationship between evolution iterations and the expansion of knowledge systems and visual tools, Figure 11 quantifies the incremental growth of knowledge concepts and visual tools derived from synthesized data at each iteration.

We observe two key findings: (1) Non-linear Growth with Evolve knowledge Expansion. As evolution iterations increase, both knowledge concepts and visual tools exhibit non-linear growth without saturation. After five iterations, the system scales to approximately 50× the initial seed size. This validates the superiority of our Data Evolution Flywheel design, which fundamentally enhances data diversity by discovering orthogonal knowledge concepts and visual tools as anchors, thereby sustaining continuous flywheel momentum. (2) Impact of Initial Seed Diversity. Richer initial knowledge concepts or tool sets yield superior evolution trajectories, underscoring the critical importance of providing diverse seed knowledge concepts as foundational anchors.

#### 7. Conclusion

In this paper, we propose V-Thinker, a general-purpose multimodal reasoning assistant that enables interactive, visioncentric thinking through end-to-end reinforcement learning. Our main contributions include: (1) formalizing Interactive Thinking with Images and developing an end-to-end framework that bridges visual grounding with code-driven interactive reasoning; (2) proposing a Data Evolution Flywheel that automatically synthesizes, evolves, and verifies datasets across diversity, quality, and difficulty dimensions; (3) introducing a Visual Progressive Training Curriculum that aligns perception and interactive reasoning through a two-stage training framework; and (4) releasing VTBench, an expert-verified benchmark for comprehensive evaluation. Extensive experiments show that V-Thinker consistently outperforms mainstream vision-language baselines, advancing the field of interactive visual reasoning and providing practical insights for future multimodal system development.

#### References

- [1] Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, et al. M2-reasoning: Empowering mllms with unified general and spatial reasoning. arXiv preprint arXiv:2507.08306, 2025. 3
- [2] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 3, 9, 21
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 3, 9, 21

- [4] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large visionlanguage models. arXiv preprint arXiv:2504.11468, 2025. 3
- [5] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. SFT or rl? an early investigation into training r1-like reasoning large visionlanguage models. CoRR, abs/2504.11468, 2025. 1
- [6] Mengjie Deng, Guanting Dong, and Zhicheng Dou. Toolscope: An agentic framework for vision-guided and longhorizon tool use, 2025. 3
- [7] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. CoRR, abs/2503.17352, 2025. 1
- [8] Guanting Dong, Xiaoshuai Song, Yutao Zhu, Runqi Qiao, Zhicheng Dou, and Ji-Rong Wen. Toward general instructionfollowing alignment for retrieval-augmented generation. arXiv preprint arXiv:2410.09584, 2024. 3
- [9] Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic entropy-balanced policy optimization, 2025.
- [10] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning. CoRR, abs/2505.16410, 2025.
- [11] Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.
- [12] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization. CoRR, abs/2507.19849, 2025.
- [13] Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, and Ji-Rong Wen. Progressive multimodal reasoning via active retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 3579–3602. Association for Computational Linguistics, 2025.
- [14] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and Zhicheng Dou. Understand what LLM needs: Dual preference alignment for retrieval-augmented generation. In Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025, pages 4206–4225. ACM, 2025. 3
- [15] Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang

- Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu, Sihan Cao, Tao Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, Ziwei Chen, and Zongyu Lin. Kimi-vl technical report. CoRR, abs/2504.07491, 2025. 1
- [16] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 9
- [17] Michael Elad, Bahjat Kawar, and Gregory Vaksman. Image denoising: The deep learning revolution and beyond - A survey paper. SIAM J. Imaging Sci., 16(3):1594–1654, 2023. 2
- [18] Richard Fitzpatrick. Euclid’s elements of geometry. 2008. 6
- [19] Vinod Goel. Sketches of thought. MIT press, 1995. 1
- [20] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 14375–14385. IEEE, 2024. 2
- [21] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao

- Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, and Ke Shen. Seed1.5-vl technical report. CoRR, abs/2505.07062, 2025. 1
- [22] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 3
- [23] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 3
- [24] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024. 3
- [25] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 3

- [26] Xiaoyuan Li, Moxin Li, Wenjie Wang, Rui Men, Yichang Zhang, Fuli Feng, Dayiheng Liu, and Junyang Lin. Mathopeval: A fine-grained evaluation benchmark for visual operations of mllms in mathematical reasoning. arXiv preprint arXiv:2507.18140, 2025. 3, 19
- [27] Xuchen Li, Xuzhao Li, Jiahui Gao, Renjie Pi, Shiyu Hu, and Wentao Zhang. Look less, reason more: Rollout-guided adaptive pixel-space reasoning. arXiv preprint arXiv:2510.01681,

2025. 3, 19

- [28] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan L. Boyd-Graber, Haitao Mi, and Dong Yu. Self-rewarding vision-language model via reasoning decomposition. CoRR, abs/2508.19652, 2025. 2
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024. 3
- [30] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. CoRR, abs/2402.00253, 2024. 2
- [31] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In European conference on computer vision, pages 126–142. Springer, 2024. 3
- [32] Wentao Liu, Qianjun Pan, Yi Zhang, Zhuo Liu, Ji Wu, Jie Zhou, Aimin Zhou, Qin Chen, Bo Jiang, and Liang He. Cmmmath: A chinese multimodal math dataset to evaluate and enhance the mathematics reasoning of large multimodal models. arXiv preprint arXiv:2409.02834, 2024. 8, 21
- [33] Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. Noisy-

- rollout: Reinforcing visual reasoning with data augmentation. arXiv preprint arXiv:2504.13055, 2025. 3
- [34] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 3, 8, 21
- [35] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Ruihang Chu, Jin Zeng, et al. Ursa: Understanding and verifying chain-ofthought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025. 3
- [36] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 3
- [37] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. CoRR, abs/2503.07365, 2025. 1, 3, 7, 8
- [38] OpenAI. Hello gpt-4o, 2024. 9, 21
- [39] OpenAI. Thinking with images. Thinking with images, 2025. 2, 3
- [40] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 3, 8, 9, 21
- [41] Runqi Qiao, Lan Yang, Kaiyue Pang, and Honggang Zhang. Making visual sense of oracle bones for you and me. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12656–12665, 2024. 3
- [42] Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Jiapeng Wang, YiFan Zhang, Zhuoma GongQue, Chong Sun, Yida Xu, Yadong Xue, Ye Tian, Zhimin Bao, Lan Yang, Chen Li, and Honggang Zhang. V-oracle: Making progressive reasoning in deciphering oracle bones for you and me. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20124–20150, Vienna, Austria, 2025. Association for Computational Linguistics. 3
- [43] Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, et al. We-math 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. arXiv preprint arXiv:2508.10433, 2025. 3, 5, 7, 8, 19
- [44] Jonathan Roberts, Mohammad Reza Taesiri, Ansh Sharma, Akash Gupta, Samuel Roberts, Ioana Croitoru, Simion-Vlad Bogolin, Jialu Tang, Florian Langer, Vyas Raina, et al. Zerobench: An impossible visual benchmark for contemporary large multimodal models. arXiv preprint arXiv:2502.09696,

2025. 8, 21

- [45] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,

- Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 7
- [46] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Mathllava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294,

2024. 3

- [47] Weikang Shi, Aldrich Yu, Rongyao Fang, Houxing Ren, Ke Wang, Aojun Zhou, Changyao Tian, Xinyu Fu, Yuxuan Hu, Zimu Lu, et al. Mathcanvas: Intrinsic visual chainof-thought for multimodal mathematical reasoning. arXiv preprint arXiv:2510.14958, 2025. 3
- [48] Xiaoshuai Song, Muxi Diao, Guanting Dong, Zhengyang Wang, Yujia Fu, Runqi Qiao, Zhexu Wang, Dayuan Fu, Huangxuan Wu, Bin Liang, et al. Cs-bench: A comprehensive benchmark for large language models towards computer science mastery. arXiv preprint arXiv:2406.08587, 2024. 3
- [49] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 2
- [50] Qiuna Tan, Runqi Qiao, Guanting Dong, YiFan Zhang, Minhui Wu, Jiapeng Wang, Miaoxuan Zhang, Yida Xu, Chong Sun, Chen Li, and Honggang Zhang. Ocr-critic: Aligning multimodal large language models’ perception through critical feedback. In Proceedings of the 33rd ACM International Conference on Multimedia, page 5385–5393, New York, NY, USA, 2025. Association for Computing Machinery. 3
- [51] Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025. 3
- [52] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan,

- Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. 1
- [53] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. 3
- [54] Barbara Tversky and Masaki Suwa. Thinking with sketches.

2009. 1

- [55] Barbara Tversky, Masaki Suwa, Maneesh Agrawala, Julie Heiser, Chris Stolte, Pat Hanrahan, Doantam Phan, Jeff Klingner, Marie-Paule Daniel, Paul Lee, et al. Sketches for design and design of sketches. In Human Behaviour in Design: Individuals, Teams, Tools, pages 79–86. Springer, 2003. 1
- [56] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, et al. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning. arXiv preprint arXiv:2506.01713, 2025. 3
- [57] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024. 3, 8, 9, 21
- [58] Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, et al. Mathcoder-vl: Bridging vision and code for enhanced multimodal mathematical reasoning. arXiv preprint arXiv:2505.10557, 2025. 3
- [59] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 3, 9
- [60] Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025. 7, 8
- [61] Ye Wang, Qianglong Chen, Zejun Li, Siyuan Wang, Shijie Guo, Zhirui Zhang, and Zhongyu Wei. Simple o3: Towards interleaved vision-language reasoning. CoRR, abs/2508.12109,

2025. 2

- [62] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural In-

- formation Processing Systems, 37:113569–113697, 2024. 8, 21
- [63] Jingxuan Wei, Caijun Jia, Qi Chen, Honghao He, Linzhuang Sun, Conghui He, Lijun Wu, Bihui Yu, and Cheng Tan. Geoint-r1: Formalizing multimodal geometric reasoning with dynamic auxiliary constructions. arXiv preprint arXiv:2508.03173, 2025. 3, 19
- [64] Shichao Weng, Zhiqiang Wang, Yuhua Zhou, Rui Lu, Ting Liu, Zhiyang Teng, Xiaozhang Liu, and Hanmeng Liu. Geosketch: A neural-symbolic approach to geometric multimodal reasoning with auxiliary line construction and affine transformation. CoRR, abs/2509.22460, 2025. 2
- [65] Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255, 2025. 3, 19
- [66] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 3
- [67] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 3, 8, 21
- [68] Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, and Yixin Zhu. Interactive visual reasoning under uncertainty. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 16, 2023, 2023. 1
- [69] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. 3, 9
- [70] Jie Yang, Feipeng Ma, Zitian Wang, Dacheng Yin, Kang Rong, Fengyun Rao, and Ruimao Zhang. Wethink: Toward general-purpose vision-language reasoning via reinforcement learning. arXiv preprint arXiv:2506.07905, 2025. 3
- [71] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. CoRR, abs/2503.10615, 2025. 1
- [72] Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng Zhang, Yiwei Wang, Yujun Cai, et al. Video-star: Reinforcing openvocabulary action recognition with tools. arXiv preprint arXiv:2510.08480, 2025. 3
- [73] Chi Zhang, Haibo Qiu, Qiming Zhang, Zhixiong Zeng, Lin Ma, and Jing Zhang. Deepsketcher: Internalizing visual manipulation for multimodal reasoning. arXiv preprint arXiv:2509.25866, 2025. 2, 3, 19
- [74] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-VL: learning to reason with multimodal large language mod-

- els via step-wise group relative policy optimization. CoRR, abs/2503.12937, 2025. 1
- [75] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025. 3
- [76] Kejia Zhang, Keda Tao, Jiasheng Tang, and Huan Wang. Poison as cure: Visual noise for mitigating object hallucinations in lvms. CoRR, abs/2501.19164, 2025. 2
- [77] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer,

2024. 3, 8, 21

- [78] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv e-prints, pages arXiv–2407, 2024. 3
- [79] Yifan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. Thyme: Think beyond images. CoRR, abs/2508.11630, 2025. 2
- [80] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025. 3, 7, 8, 19
- [81] Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025. 3, 19
- [82] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing "thinking with images" via reinforcement learning. CoRR, abs/2505.14362, 2025. 2, 3
- [83] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 3
- [84] Chenyue Zhou, Mingxuan Wang, Yanbiao Ma, Chenxu Wu, Wanyi Chen, Zhe Qian, Xinyu Liu, Yiwei Zhang, Junhao Wang, Hengbo Xu, et al. From perception to cognition: A survey of vision-language interactive reasoning in multimodal large language models. arXiv preprint arXiv:2509.25373,

2025. 3, 19

- [85] Zetong Zhou, Dongping Chen, Zixian Ma, Zhihan Hu, Mingyang Fu, Sinan Wang, Yao Wan, Zhou Zhao, and Ranjay Krishna. Reinforced visual perception with tools. arXiv preprint arXiv:2509.01656, 2025. 3, 19
- [86] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 21

- [87] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 26183–26191,

2025. 3

- [88] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024. 3, 8, 21

## Appendix

#### Contents

- 1. Introduction 1
- 2. Related Work 3
- 3. Preliminary 3

- 3.1. Problem Formulation . . . . . . . . . . . . . 3
- 3.2. Rethinking on Data Synthesis Paradigm . . . 3

- 4. Methodology 4

- 4.1. Data Evolution Flywheel . . . . . . . . . . . 4

- 4.1.1 . Diversity: Knowledge-driven Evolution 5
- 4.1.2 . Quality: Coordinated Calibration . . 5
- 4.1.3 . Difficulty: Progressive Expansion . . 6

- 4.2. Visual Progressive Training Curriculum . . . 6 4.2.1 . Perception Alignment . . . . . . . . 6
- 4.3. Interactive Reasoning Alignment . . . . . . 7

- 5. VTBench 8

- 5.1. Data Collection and Annotation . . . . . . . 8
- 5.2. Evaluation Dimensions . . . . . . . . . . . . 8
- 5.3. Data Statistics and Evaluation Metrics . . . . 8

- 6. Experiments 8

- 6.1. Experimental Setup . . . . . . . . . . . . . . 8
- 6.2. Main Results . . . . . . . . . . . . . . . . . 11
- 6.3. Results on Generalized Reasoning Domain . 11
- 6.4. Ablation Study. . . . . . . . . . . . . . . . . 12
- 6.5. Analysis of the Data Evolution Flywheel . . 12

- 7. Conclusion 14

- A. Details of V-Thinker 19

- A.1. Prompts Used in V-Thinker . . . . . . . . . 19

B. Additional Results and Analyses 20

- B.1. Extended Quantitative Analysis . . . . . . . 20 B.2. Additional Analysis of the Data Evolution

- C. Details of VTBench 20

- C.1. Evaluation Dimensions . . . . . . . . . . . . 20
- C.2. Data Statistics and Evaluation Metrics . . . . 21

- D. Detailed Experimental Setup 21

- D.1. Implementation Details (Evaluation) . . . . . 21
- D.2. Details of Baselines. . . . . . . . . . . . . . 21

- E. Broaden Impact 21
- F. Limitation 22

Flywheel . . . . . . . . . . . . . . . . . . 20

#### A. Details of V-Thinker

##### A.1. Prompts Used in V-Thinker

Prompt Templates for Knowledge-Driven and ToolDriven Synthesis. To operationalize the two synthesis pathways in our co-evolution framework, we design two prompt specifications that explicitly encode how new instances are generated under knowledge-driven or tool-driven conditioning.

- As shown in Table 6, the knowledge-driven prompt formalizes how sampled knowledge combinations guide the construction of problems, the rendering of figures, and the generation of reasoning trajectories involving visual tools. The underlying knowledge inventory is initialized from the 1,819 fundamental principles in We-Math 2.0 [43], and we instantiate both single-concept inputs and compositional combinations, so that the synthesized data can cover a broad range of reasoning objectives.
- As shown in Table 7, the tool-driven prompt specifies the complementary mechanism in which visual tools act as the generative anchor. The initial visual tool system contains 61 visual tools, constructed by abstracting common operations from existing tool-augmented multimodal systems [26, 27, 63, 65, 73, 80, 81, 84, 85]. It defines how tool combinations constrain the image, the tool operations, and the resulting reasoning process, thereby enabling tool-conditioned data construction.

Together, these two prompt templates characterize how knowledge and tools drive generation along orthogonal dimensions. To ensure consistency across all visual tools, the rendering protocol enforces a unified set of conventions: all figures and tool overlays use absolute pixel coordinates, a common top-left origin, and full-bleed canvases without margins. These choices guarantee stable visual structure, precise localization, and reproducible execution across iterations. Further implementation details are provided in the corresponding prompt specifications.

Prompts for Coordinated Calibration To support the coordinated calibration stage, we provide three checker prompt templates that separately evaluate (1) answer correctness, (2) integrity of the rendered image, and (3) consistency of auxiliary visual states. These templates are shown in Tables 8, 9, and 10.

For cases where only the textual answer is inconsistent while the visual components remain valid, a dedicated repair prompt (Table 11) reconstructs the question from the visual states to realign the reasoning. Together, these templates form the basis of the calibration mechanism used to refine the initial dataset into a coherent and verified collection.

Prompts for Progressive Expansion Progressive expansion is operationalized through two coordinated prompt templates. The parallel extension prompt (Table 12) specifies how additional visual constructions are incorporated to introduce complementary observations beyond the original configuration. The sequential extension prompt (Table 13) prescribes constructions that are conditioned on existing entities or intermediate results, enabling reasoning to unfold through successive stages. These templates define the procedural rules for generating extended visual states and deeper reasoning trajectories, supporting the formation of higher-difficulty instances.

Prompts for Perception Data Synthesis Perceptionoriented data generation is supported by four prompt templates, corresponding to code construction and three levels of perceptual QA. The figure-construction prompt (Table 14) governs the generation of executable drawing code and the associated element-level annotations, ensuring that visual content is specified in absolute pixel coordinates and adheres to the structural constraints defined by element relations, element count, and sampled knowledge concepts.

Based on these annotations, three QA prompts convert visual structures into perception tasks along the hierarchical dimensions introduced above. The surface-level prompt (Table 15) produces localization questions grounded in explicit element coordinates. The semantic-level prompt (Table 16) builds questions around structurally defined feature points, requiring spatial interpretation beyond raw coordinates. The integrated reasoning prompt (Table 17) constructs tasks that combine perceptual identification with computational reasoning, such as determining derived geometric or structural quantities.

Together, these templates operationalize the construction of Dperception by transforming synthesized visual structures into systematically layered perceptual QA pairs.

#### B. Additional Results and Analyses

##### B.1. Extended Quantitative Analysis

Table 3 provides an extended evaluation on VTBench by incorporating additional variants from the Qwen2.5-VL, InternVL3, and LLaVA-OneVision-1.5 families. Across this expanded comparison, two observations remain consistent.

Fine-grained perceptual grounding is consistently weak across model scales. Large models such as GPT-4o, Qwen2.5-VL-72B, and InternVL3-78B exhibit limited accuracy on perception-oriented tasks, which reflects the difficulty posed by tasks that require local spatial grounding, such as identifying specific points, intersections, or geometric primitives.

V-Thinker shows consistent improvements in perception and interaction. Relative to 7B-scale open-source models, V-Thinker achieves higher accuracy (+6.0%) on perceptionoriented questions and a larger gain (+22.8%) on instructionguided interaction, together with stable improvements in visual reasoning (+8.6%). These trends are consistently observed across the expanded baseline set.

##### B.2. Additional Analysis of the Data Evolution Flywheel

Expanded Knowledge Structures. Figure 11 provides a comprehensive visualization of the knowledge system produced through the Data Evolution Flywheel. Beyond the summary shown in the main paper, the extended structure highlights the breadth and depth achieved through iterative evolution. The resulting hierarchy spans 25 domains and reaches up to 7 layers, forming a graph with 24,767 nodes. This expanded view further illustrates how the flywheel progressively enriches conceptual coverage, organizes related concepts into coherent clusters, and builds multi-level structures that support both fine-grained and cross-domain reasoning.

Tool System Analysis. Within the Data Evolution Flywheel, the visual tool system co-evolves alongside the knowledge concepts, and the resulting tool set undergoes an additional consolidation stage. Since visual tools are instantiated through executable Python drawing routines, functionally identical operations may differ only in minor parameter settings (for example, line styles or rendering options), which can artificially inflate the apparent diversity of tools without reflecting meaningful distinctions in visual operations. To obtain a more faithful representation of the underlying tool space, we apply an additional round of BGE-based hierarchical clustering to the co-evolved tool set, using a strict similarity threshold of 0.3. The resulting clusters are then normalized through LMM-assisted (GPT-4.1) unification of tool names, followed by manual adjustment to ensure semantic consistency. This refinement phase collapses redundant variants and yields a compact, semantically coherent library of 234 visual tools, providing a clearer and more accurate characterization of the evolved visual tool system.

#### C. Details of VTBench C.1. Evaluation Dimensions

VTBench assesses interactive reasoning across three hierarchical dimensions, modeling the progression from perception to adaptive interaction:

Perception → Instruction-Guided Interaction → Interactive Reasoning

• Perception: Evaluates fine-grained visual perception, such as identifying or locating specific coordinates.

VTBench Perception

Method

Instruct. Interaction

Interactive Reasoning

Avg.

GPT-4o 12.6 26.0 36.4 25.0 Qwen2.5-VL-72B 38.0 34.2 51.4 41.2 InternVL3-78B 13.8 19.0 43.4 25.4 InternVL3-8B 10.4 6.8 33.8 17.0 LLaVA-OV-1.5-8B 12.2 12.2 30.2 18.2 InternVL3-2B 3.0 3.4 22.0 9.5 Qwen2.5-VL-3B 2.6 2.2 28.6 11.1 LLaVA-OV-1.5-4B 8.2 10.8 30.2 16.4 Qwen2.5-VL-7B 12.6 8.8 31.8 17.7 V-Thinker-7B 18.6 31.6 40.4 30.2 ∆ (vs Qwen2.5-VL-7B) +6.0 +22.8 +8.6 +12.5

Table 3. Overall performance on VTBench. (Instruct. Interaction denotes Instruction-Guided Interaction.)

- • Instruction-Guided Interaction: Tests the ability to execute explicit visual instructions (e.g., drawing lines, labeling regions).
- • Interactive Reasoning: Evaluates reasoning tasks that require visual interaction.

##### C.2. Data Statistics and Evaluation Metrics

VTBench contains 1,500 QA pairs (500 per task type) across 9 open-source benchmarks (MathVista [34], MathVision [57], MathVerse [77], Dynamath [88], We-Math [40], LogicVista [67], CMM-Math [32], CharXiv [62] and ZeroBench [44]) covering four domains: Logical Reasoning, Geometry, Algebra, and Statistics. Evaluation is conducted under three task-specific metrics:

- • Perception Tasks: Models generate Python code to mark perceived coordinates. The resulting image is compared with annotations by large multimodal models (LMMs).
- • Instruction-Guided Interaction Tasks: Models generate Python code to perform instructed actions. The visual output is compared with expert annotations using LMM judgment.
- • Interactive Reasoning Tasks: Models output final answers, which are evaluated by LLMs for correctness.

#### D. Detailed Experimental Setup D.1. Implementation Details (Evaluation)

For VTBench evaluation, we employ two models as judges: Qwen3-235B-A22B and Qwen3-VL-235B-A22B. Their results differ by less than 2% from GPT-4.1, validating their reliability as open-source evaluators under a unified evaluation protocol. For all other benchmarks used in this work, including MathVision, VisuLogic, and We-Math, we follow the official evaluation procedures specified by each benchmark,

including the provided scoring scripts and answer-matching rules.

##### D.2. Details of Baselines.

We evaluate all models on MathVision, VisuLogic, We-Math and VTBench, following each benchmark’s official protocol and reporting accuracy as the primary metric. Our baseline comparison includes a broad spectrum of multimodal reasoning systems. GPT-4o is OpenAI’s flagship multimodal model designed for unified vision–language understanding with strong cross-domain reasoning capability [38]. The Qwen2.5-VL family (72B/7B/3B) is an open-source vision–language series that emphasizes visual–language alignment and multimodal reasoning [3]. The InternVL3 series (78B/8B/2B) represents an open-source vision–language architecture with enhanced multimodal perception and reasoning capability [86]. The LLaVA-OneVision-1.5 family (8B/4B) consists of open multimodal models incorporating a unified vision encoder and language backbone, optimized for efficient and competitive vision–language reasoning [2]. Together, these baselines span a wide range of architectures and parameter scales, enabling a comprehensive and balanced comparison against our model.

#### E. Broaden Impact

Advancing interactive, vision-centered reasoning in multimodal systems. Our work encourages a transition from passive visual perception to interactive reasoning within images. Instead of only understanding visual content, VThinker can actively engage with visual elements through pointing, annotating, and manipulating structured regions. This shift supports explicit intermediate reasoning steps and improves transparency in visual-language decision making. Such interactive capabilities may influence a wide range

of tasks including diagram understanding, scientific figure analysis, embodied perception, and human–AI collaborative interfaces, where traceable and verifiable reasoning is essential.

Enabling a creator-oriented synthetic data paradigm that expands future possibilities. V-Thinker revisits the longstanding paradigm in which models act purely as solvers when synthesizing data. This solver-centric view restricts the diversity and structural richness of generated samples, particularly for tasks requiring precise spatial or logical alignment. Our work reveals that modern multimodal models can instead function as creators capable of generating complex visual problems, programmatically rendered images, auxiliary diagrams, and coherent reasoning paths. When integrated with knowledge-driven representations, this creator-oriented paradigm significantly enlarges the design space for synthetic reasoning data. It reduces reliance on handcrafted seeds, enables scalable and controlled curriculum evolution, and offers a foundation for autonomous dataset construction and simulation-based training. This paradigm shift may inspire future research on model-driven data ecosystems, controllable synthetic corpora, and new forms of interactioncentric supervision for robust multimodal intelligence.

Bridging foundational models with practical, tooloriented applications. The structured visual interactions and interpretable reasoning processes supported by VThinker help narrow the gap between multimodal foundation models and their real-world deployment. The ability to generate, manipulate, and reason with structured visual information can benefit scientific analysis, education technologies, robotics perception, and interactive decision-support systems. The explicit reasoning traces also promote safer and more accountable AI behavior, which is crucial in settings that demand correctness and transparency. By supporting precise visual alignment and interpretable reasoning, our work contributes to building reliable multimodal systems that operate effectively in complex visual environments.

#### F. Limitation

V-Thinker explores a generalized paradigm for interactive visual reasoning, where models perform reasoning through visual interactions. However, it still has several limitations. Due to computational constraints, our current model iteration remains limited in scale and exhibits reduced capability in knowledge-intensive or domain-specific tasks. Our primary goal in this work is to establish a unified framework for interactive reasoning. In future work, we will further optimize the model and enhance its robustness and generalization across diverse reasoning domains. Moreover, through V-Thinker, we believe that as model capabilities continue to advance,

both the paradigm of data construction and the upper bound of model reasoning should be re-examined and redefined. We envision future models achieving increasingly natural and human-like forms of visual interaction and reasoning.

###### LMM Judge Prompt (Perception Task)

You are an expert visual evaluator. Your goal is to determine whether the point(s) marked in the [Generated Image] meaningfully implement the visual operation described in the [Instruction], using the [Ground Truth Image] as the semantic reference.

Your evaluation should prioritize conceptual correctness while allowing moderate spatial deviation.

###### Judging Criteria

Judgement = 1 (Consistent) Output 1 if ALL of the following are satisfied:

###### 1. Correct Visual Concept

- - The marked point(s) reflect the correct visual idea from the instruction.
- - The point(s) are placed on the correct target feature or structure (for example, the intended corner, midpoint, center, or intersection), even if the position is not exact.

###### 2. Minor Spatial Deviation Allowed

- - The modification may deviate moderately, but only if:

- • It remains clearly associated with the intended feature,
- • The positional error is small enough that the point still unambiguously indicates the correct target,
- • The structural relation remains recognizable (e.g., the point lies on or near the correct line segment, sits close to the correct vertex, or falls within a reasonable neighborhood of the correct intersection).

- - Examples of acceptable deviation:

- • A midpoint marker slightly off-center but still indicating the middle region of the correct segment,
- • A vertex marker somewhat offset but still within the local vicinity of the intended vertex.

###### 3. Stylistic Variations Ignored

- - Differences in point color, size, shape, or style (for example, dot vs small circle vs “x”) must be ignored.
- - Minor rendering artifacts that do not change the intended target should also be ignored.

- Judgement = 0 (Inconsistent)

- Output 0 if ANY of the following hold:

- 1. Wrong Concept

- - The point(s) are placed on the wrong feature or structure (for example, a different vertex, a different segment, a different circle, or an unrelated location).
- - The marking does not correspond to the operation described in the instruction.

- 2. Major Conceptual Misalignment

- The location of the point(s) is so far from the intended target that the underlying operation is no longer recognizable, even under generous tolerance.

- 3. Missing or Insufficient Marking

- The required point(s) are missing, or the markings are too incomplete or ambiguous to reflect the instruction.

- 4. No Effective Change

- - The [Generated Image] is effectively identical to the [Original Image].

Output Format

- - If consistent, output 1.
- - If inconsistent, output 0. Output ONLY 0 or 1. Do not provide any explanation.

###### Evaluation Inputs

[Original Image] <|vision_start|><|image_pad|><|vision_end|> [Generated Image] <|vision_start|><|image_pad|><|vision_end|> [Ground Truth Image] <|vision_start|><|image_pad|><|vision_end|> [Instruction]: {instruction} Provide your judgement. Judgement:

###### Table 4. Prompts for LMM Judge Prompt (Perception Task).

###### LMM Judge Prompt (Instruction-Guided Interaction Task)

You are an expert visual evaluator. Your goal is to determine whether the modifications in the [Generated Image] meaningfully implement the visual operation described in the [Instruction], using the [Ground Truth Image] as the semantic reference.

Your evaluation should prioritize conceptual correctness while allowing moderate spatial deviation.

###### Judging Criteria

- Judgement = 1 (Consistent)

- Output 1 if ALL of the following are satisfied:

###### 1. Correct Visual Concept

- - The modification reflects the correct visual idea from the instruction.
- - The modification targets the correct general location or structure, even if not precise.

###### 2. Broader Tolerance for Spatial Deviation

- The modification may deviate significantly, as long as:

- • it is within the correct overall area,
- • it connects or marks the correct conceptual components,
- • and the intended structural relation remains recognizable.

- Examples of acceptable deviation:

- • endpoints not exactly touching but pointing to correct vertices,
- • a region roughly outlined but not tightly aligned,
- • a line slightly tilted or offset but indicating the right relation.

###### 3. Stylistic Variations Ignored

- - Differences in color, stroke thickness, line style, opacity, and rendering artifacts must be ignored.

Judgement = 0 (Inconsistent) Output 0 if ANY of the following hold:

1. Wrong Concept

- - The modification represents the wrong type of operation (e.g., a line instead of a point, marking the wrong region).
- - The wrong endpoints, wrong angle, or wrong region are used.

###### 2. Major Conceptual Misalignment

- The modification is placed in a way that the intended structure is no longer recognizable, even with generous tolerance.

###### 3. Missing or Insufficient Modification

- The required visual change is absent or too incomplete to reflect the instruction.

###### 4. No Effective Change

- The [Generated Image] is effectively identical to the [Original Image].

###### Output Format

- - If consistent, output 1.
- - If inconsistent, output 0. Output ONLY 0 or 1. Do not provide any explanation.

###### Evaluation Inputs

[Original Image] <|vision_start|><|image_pad|><|vision_end|> [Generated Image] <|vision_start|><|image_pad|><|vision_end|> [Ground Truth Image] <|vision_start|><|image_pad|><|vision_end|> [Instruction]: {instruction} Provide your judgement. Judgement:

###### Table 5. Prompts for LMM Judge Prompt (Instruction-Guided Interaction Task).

###### Generator (Knowledge-driven)

You are an expert exam question designer and visualization engineer. I will provide you with a knowledge point and a category. Please generate three high-quality exam questions that meet the following requirements:

- • Quantity: output exactly three questions
- • Question type: multiple choice or fill-in-the-blank
- • Difficulty: one easy, one medium, one hard
- • Figure: each problem must require a figure to solve; the figure is generated using pure Python code
- • LaTeX: all math expressions must be written in LaTeX
- • Aesthetics: the figure must be clear and visually appealing, with proper labels if needed; it can omit a title
- • If the figure contains text or symbols, ensure the font and symbols are aesthetically pleasing and consistent
- • Ensure the correctness of the provided answer
- • Auxiliary lines: the problem must be solvable only after adding one or more auxiliary lines in the figure.
- • In the solution, explicitly explain which auxiliary lines are added and why they work.
- • The auxiliary lines must be those truly required for solving the problem (not optional or cosmetic).
- • A valid auxiliary line should:

- – Be necessary (without it, the problem cannot be solved efficiently or at all),
- – Create useful geometric structures (e.g., similar triangles, right triangles, symmetry, equal radii),
- – Connect meaningful points (such as vertices, midpoints, centers, or intersections),
- – Have a clear role in the reasoning (explainable in the solution).

• Do not include instructions in the problem statement that explicitly tell the student to add auxiliary lines. • If multiple knowledge points are provided, each generated question must require applying them together. • In addition to descriptions, also summarize the auxiliary tool construction type in concise English terms (e.g., “perpendicular line”, “angle bisector”, “midpoint connection”). This summary should not influence the figure or solution construction, only be added afterward.

Python Figure Rules (for ALL original figure code blocks):

- • Use only matplotlib and numpy
- • In the figure, text and symbols must not use LaTeX rendering in matplotlib (do not enable text.usetex; avoid $...$); use standard non-LaTeX fonts
- • Code must be self-contained and directly runnable (import, setup, plt.show() at the end)
- • Figure size should be appropriate, with readable labels and no unnecessary clutter
- • No saving files, printing text, or using external data
- • Output must explicitly set canvas width/height and DPI, then create a full-bleed axes with no margins and a top-left origin in pixel space:

- – Define: w, h = ... DPI = ...
- – Boilerplate (exact order): fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI) ax = fig.add_axes([0,0,1,1]) # full-bleed, no borders ax.set_xlim(0, w) ax.set_ylim(h, 0) # top-left origin (y downwards) ax.axis(’off’) ax.set_aspect(’equal’)
- – End every script with: plt.show()

- • Do NOT use plt.tight_layout()
- • Do NOT save files inside the code (no fig.savefig, etc.)

Auxiliary-lines Code Rules (for ALL auxiliary-line code blocks):

- • Load the original figure image ONLY via plt.imread(image_path).
- • Assume NO prior canvas info (do not reuse w, h, or DPI from the original figure).
- • Infer dimensions directly from the loaded image: img = plt.imread(image_path) h, w = img.shape[:2] fig = plt.figure() ax = fig.add_axes([0,0,1,1]) ax.set_xlim(0, w) ax.set_ylim(h, 0) # top-left origin ax.axis(’off’) ax.set_aspect(’equal’) ax.imshow(img)
- • Oracle-capable construction (preferred for exactness):

- – You may directly reuse or reconstruct the exact geometric parameters/coordinates used in the original figure’s construction (e.g., vertex coordinates, lengths, radii, centers, slopes), as long as auxiliary-line endpoints are computed and drawn in absolute pixel coordinates consistent with the displayed image.

- • Perception → Geometry → Overlay (fallback):

- 1. Perceive anchors (deterministically identify landmarks/edges/symmetry/tangency).
- 2. Compute exact pixel endpoints for the auxiliary lines from the perceived anchors.
- 3. Overlay with ax.plot() / ax.scatter() in absolute pixel coordinates.

– Special case: if perception directly yields absolute coordinates, step (2) can be omitted.

- • Do NOT use DPI anywhere in auxiliary-line code.
- • No printing, no saving, no external data or libraries.
- • End with plt.show()

Knowledge point: {knowledge_point} Note: Each of the three generated questions must be designed to simultaneously assess the provided knowledge points in combination, not individually.

—

Output Format: Output exactly three JSON objects inside a JSON array (no extra explanations, no backticks). Each object must follow this schema:

{{ "Idx": "{idx}", "Tag": "{subject_area}", "Knowledge point": "{knowledge_point}", "difficulty": "easy" \ "medium" \ "hard", "problem_type": "multiple choice or fill_in_blank", "python_code": "Pure Python code as a string, directly runnable (generates the original figure)", "python_code_auxiliary": "Pure Python code as a string, directly runnable (loads the saved figure image via plt.imread and overlays auxiliary lines in pixel coordinates)", "Question": "Problem statement in Markdown with LaTeX.", "answer": "string", "solution_markdown": "Step-by-step solution in Markdown with LaTeX, where the auxiliary lines are naturally introduced and utilized as part of the reasoning process (CoT).", "auxiliary_lines_description": "Brief description of the auxiliary lines to be added (what and where, in words)", "auxiliary_tools_summary": ["List of concise English action names of auxiliary tool types, e.g., [’Construct Symmetry Axis’, ’Construct Tangent Line’]] }, ...}

Table 6. Prompts for knowledge-driven evolution.

###### Generator (Tool-driven)

You are an expert exam question designer and visualization engineer. I will provide you with a task name and one or more tool names. Please generate three high-quality exam questions that meet the following requirements: (You may be given one or more task names. If you find that the task(s) and tool(s) are difficult to align naturally, you may creatively interpret or extend the task, but you must still follow the required output format exactly.)

- • Quantity: output exactly three questions
- • Question type: multiple choice or fill-in-the-blank
- • Difficulty: one easy, one medium, one hard
- • Figure: each problem must require a figure to solve; the figure is generated using pure Python code
- • LaTeX: all math expressions must be written in LaTeX
- • Aesthetics: the figure must be clear and visually appealing, with proper labels if needed; it can omit a title
- • If the figure contains text or symbols, ensure the font and symbols are aesthetically pleasing and consistent
- • Ensure the correctness of the provided answer
- • Auxiliary tools: the problem must be solvable only after adding one or more auxiliary tools in the figure.
- • The purpose of the exam question is not to test whether the student knows how to use the tool, but to design a problem where the tool is naturally and necessarily required to solve it.
- • In the solution, explicitly explain which auxiliary tools are added and why they work.
- • The auxiliary tools must be those truly required for solving the problem (not optional or cosmetic).
- • A valid auxiliary tool should:

- – Be necessary (without it, the problem cannot be solved efficiently or at all),
- – Create useful geometric or algebraic structures (e.g., similar triangles, symmetry, tangent, midpoints),
- – Connect meaningful points (such as vertices, midpoints, centers, or intersections),
- – Have a clear role in the reasoning (explainable in the solution).

- • Do not include instructions in the problem statement that explicitly tell the student to add auxiliary tools.
- • Additionally, provide a fine-grained hierarchical knowledge point list that captures the layered concepts involved in solving the problem.

- – Each element must be a full-path string from broad domain to specific property, e.g.: (Geometry)-(Plane Geometry)-(Basic Plane Figures)-(Triangle)-(Properties of Isosceles and Equilateral Triangles)-(Properties of Sides)
- – If multiple knowledge points are involved, return multiple path strings in a list.
- – This list should not influence the figure or solution construction, only be added afterward.

Python Figure Rules (for ALL original figure code blocks):

- • Use only matplotlib and numpy
- • In the figure, text and symbols must not use LaTeX rendering in matplotlib (do not enable text.usetex; avoid $...$); use standard non-LaTeX fonts
- • Code must be self-contained and directly runnable (import, setup, plt.show() at the end)
- • Figure size should be appropriate, with readable labels and no unnecessary clutter
- • No saving files, printing text, or using external data
- • Output must explicitly set canvas width/height and DPI, then create a full-bleed axes with no margins and a top-left origin in pixel space:

- – Define: w, h = DPI =
- – Boilerplate (exact order): fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI) ax = fig.add_axes([0,0,1,1]) # full-bleed, no borders ax.set_xlim(0, w) ax.set_ylim(h, 0) # top-left origin (y downwards) ax.axis(’off’) ax.set_aspect(’equal’)
- – End every script with: plt.show()

- • Do NOT use plt.tight_layout()
- • Do NOT save files inside the code (no fig.savefig, etc.)

Auxiliary-tools Code Rules (for ALL auxiliary-line code blocks):

- • Load the original figure image ONLY via plt.imread(image_path).
- • Assume NO prior canvas info (do not reuse w, h, or DPI from the original figure).
- • Infer dimensions directly from the loaded image: img = plt.imread(image_path) h, w = img.shape[:2] fig = plt.figure() ax = fig.add_axes([0,0,1,1]) ax.set_xlim(0, w) ax.set_ylim(h, 0) # top-left origin ax.axis(’off’) ax.set_aspect(’equal’) ax.imshow(img)
- • Oracle-capable construction (preferred for exactness):

- – You may directly reuse or reconstruct the exact geometric parameters/coordinates used in the original figure’s construction (e.g., vertex coordinates, lengths, radii, centers, slopes), as long as auxiliary-line endpoints are computed and drawn in absolute pixel coordinates consistent with the displayed image.

- • Perception → Geometry → Overlay (fallback):

- 1. Perceive anchors (deterministically identify landmarks/edges/symmetry/tangency).
- 2. Compute exact pixel endpoints for the auxiliary lines from the perceived anchors.
- 3. Overlay with ax.plot() / ax.scatter() in absolute pixel coordinates.

– Special case: if perception directly yields absolute coordinates, step (2) can be omitted.

- • Do NOT use DPI anywhere in auxiliary-line code.
- • No printing, no saving, no external data or libraries.
- • End with plt.show()

Task: {task_name} Tool: {tool_name} Output Format: Output exactly three JSON objects inside a JSON array (no extra explanations, no backticks). Each object must follow this schema:

{{ "Idx": "{idx}", "Task": "{task_name}", "Tool": "{tool_name}", "difficulty": "easy" \ "medium" \ "hard", "problem_type": "multiple choice or fill_in_blank", "python_code": "Pure Python code as a string, directly runnable (generates the original figure)", "python_code_auxiliary": "Pure Python code as a string, directly runnable (loads the saved figure image via plt.imread and overlays auxiliary tools in pixel coordinates)", "Question": "Problem statement in Markdown with LaTeX.", "answer": "string", "solution_markdown": "Step-by-step solution in Markdown with LaTeX, where the auxiliary lines are naturally introduced and utilized as part of the reasoning process (CoT).", "auxiliary_tools_description": "Brief description of the auxiliary tools to be added (what and where, in words)", "knowledge_point_hierarchy": [e.g. "(Geometry)-(Plane Geometry)-(Basic Plane Figures)-(Triangle)-(Properties of Isosceles and Equilateral Triangles)-(Properties of Sides)", "(Geometry)-(Plane Geometry)-(Basic Plane Figures)-(Triangle)-(Properties of Isosceles and Equilateral Triangles)-(Properties of Angles)"] }, ...}

Table 7. Prompts for tool-driven generation.

###### Checker (Q&A)

You are an extremely strict and skeptical problem auditor. Your core task is to identify “solution to be evaluated” errors.

###### Core Instructions:

- • Critically Compare: Review the “Provided Solution Process” and “Provided Answer”. Your goal is to find all logical fallacies, calculation errors, or missing steps. Do not be misled or influenced by its reasoning process.
- • Scoring: Based on your independent analysis and comparison, use the following rules to assign a score.

Inputs: Question: "{question}" Provided Answer: "{answer}" Primary Image Path: "{image}" Solution Process: "{solution}"

Scoring (conservative; deduct for ANY issue; ignore drawing issues here):

- • Start from 10 ONLY if EVERY step is logically valid, calculations are correct, and the final answer exactly matches (value/units/format).
- • By using the provided Question, recalculate the problem and check if the answer is completely consistent with Provided Answer. If not, deduct 5 points.
- • Check if there are any logical problems or reasoning errors in the problem-solving process. If so, deduct 5 points.
- • Deduct 3–5 for each logical gap, unstated assumption, undefined term, or missing critical step.
- • Deduct 2–4 for misapplied theorems or arithmetic/algebra mistakes (even minor).
- • Deduct 4–6 if the final value/units/format does not exactly match the provided answer.
- • Answer/question type mismatch (e.g., numeric answer for MCQ, option letter for fill-in): set score ≤ 3.
- • Clamp to [0,10]; prefer lower scores when uncertain.

Output Requirement: Output ONLY one integer 0–10.

Table 8. Prompt for Q&A correctness checking.

###### Checker (Original Image)

You are an IMAGE VALIDITY auditor. Your ONLY task is to evaluate if the shapes in the diagram are drawn completely and correctly.

###### Important:

- • IGNORE the text question and any labels in the image.
- • Your evaluation must NOT consider semantic accuracy (whether the diagram answers the question).
- • Focus purely on the drawing quality.

Inputs: Question: "{question}" Primary Image Path: "{image}"

Scoring Criteria (Based ONLY on Drawing Quality): 10: Excellent

- • All shapes are drawn correctly and are complete.
- • Lines are closed where they should be (e.g., in a triangle, square, circle).
- • The drawing is clear and well-formed.

5–9: Minor to Moderate Flaws

- • Shapes are mostly complete but may have small imperfections (e.g., slightly wavy lines, corners not perfectly joined but still closed).
- • The overall shape is still easily recognizable.

0–4: Severe Drawing Errors

- • A primary shape is fundamentally incomplete or malformed.
- • Examples: a triangle is missing a side, a square has a large gap, a circle is not a closed loop.
- • The shape is distorted to the point of being unrecognizable.

Output Requirement: Output ONLY one integer 0–10.

Table 9. Prompt for validity check of the rendered original image.

###### Checker (Visual Tool)

You are a STRICT visual tool auditor. You will be given two images: the original problem and the problem with an visual tool drawn. Your task is to compare them and evaluate ONLY the correctness of the visual tool. Output MUST be a single integer score from 0 to 10.

###### Images:

- • Image 1: Original Image
- • Image 2: Intermediate Visual States

Inputs: Question: "{question}" Original Image Path: "{image}" Auxiliary Image Path: "{aux_image}" Auxiliary Line Description: "{visual_tools_description}" Solution Process that uses the line: "{solution}"

Scoring (conservative; compare Image 2 to Image 1):

- • Start from 10 ONLY if the line in Image 2 is correctly drawn based on the context from Image 1, matches its description, and is properly used in the solution.
- • Deduct 4–6 for clear mismatches in type, location, or endpoints (e.g., line from wrong vertex, not tangent).
- • Deduct 3–5 if a claimed property is not visually/logically satisfied (e.g., line is clearly not perpendicular to the base in Image 1).
- • Deduct 2–4 if the solution process does not actually reference or use the drawn line as described.
- • Clamp to [0,10]; prefer lower scores when uncertain.

Output Requirement: Output ONLY one integer 0–10.

Table 10. Prompt for visual tool consistency checking.

###### Repairer

You will receive the following inputs: a problem, the corresponding solution steps and answer, the original diagram code, the code for the diagram with visual tools and the image with visual tools. Your task is to use the image, both sets of diagram codes and the context to design or rewrite a new problem, along with its solution and answer, that necessarily depends on the structures or properties introduced by the visual tools. Please follow the steps below to complete the task.

###### Input includes:

- • Problem: {question}
- • Solution steps: {solution}
- • Answer: {answer}
- • Original diagram code: {code_ori}
- • Visual tools diagram code: {code_aux}

- 1. Comprehensive Understanding:

- • Read and understand the original problem, solution steps, and answer.
- • Analyze the role and function of the visual tools, using the diagram codes to understand the geometrical structure.
- • Determine how the visual tools introduce new properties or conclusions that could be leveraged for a novel problem.

- 2. Create a New Problem:

- • Based on the original problem and the visual tools, rewrite or design a new problem that cannot be solved or proven without the visual tools or their structural impact.
- • The new problem should not be solvable directly from the original diagram code alone, and preferably should not mention the visual tools explicitly.
- • Ensure that the solution to the new problem crucially uses properties, conclusions, or new geometric configurations stemming from the visual tools.
- • The new problem may be a proof or computation, but must fundamentally depend on what the visual tools contribute.

- 3. Instructions for the problem statement:

- • Do not include any instructions or hints such as “draw”, “connect”, “extend”, “symmetrize”, “complete”, etc., that refer to constructing visual tools.
- • If such hints are present, please remove or revise the statement so it gives no guidance on drawing new elements.
- • The new question should not be answerable by text alone; one should need the accompanying auxiliary-line image.

###### Final Output Json Format (Output only the final JSON result. No extra explanations or comments):

{ "id": "{id}", "new_problem": "Newly Designed Problem", "solution_markdown": "Step-by-step solution in Markdown with LaTeX, where the visual tools are naturally introduced and utilized as part of the reasoning process (CoT)", "answer": "Final Answer" }

Table 11. Prompt for text–visual consistency repair via question reconstruction.

###### Progressive Expansion (Parallel Extension)

You are an expert exam question designer and visualization engineer. Based on the given original problem, standard answer, auxiliary line image, original drawing code, and auxiliary line drawing code, you need to design either a multiple choice or fill-in-the-blank problem that requires two rounds of auxiliary line code calls to be solved completely. Output the new problem, a clear multi-step auxiliary line drawing plan, the drawing code, and the solution process with the final answer.

###### Steps:

- • You need to read and understand all inputs. Use the provided auxiliary line image to interpret the problem, the purpose of the auxiliary line, and the solution process.
- • The existing auxiliary line provides one key conclusion for problem-solving; another key conclusion must come from the newly introduced auxiliary line. The new problem you create must depend on at least these two key conclusions to be solved.
- • Read the auxiliary line image code; based on this image, write executable Python code to draw the new auxiliary lines. You may refer to the original drawing code to help interpret the geometric structure and coordinates of the points and lines in the original and auxiliary images.
- • Ensure the new problem maintains the original point naming. Do not pose questions about nonexistent geometric elements.
- • The answer should not be directly obtainable from the original or auxiliary diagram, and each round of auxiliary lines must be necessary and enhance subsequent reasoning.
- • Generate the output: problem statement, reasoning for why multiple rounds of auxiliary lines are necessary, drawing code, detailed problem-solving process and final answer.

Rules for introducing new visual tools:

- • The new auxiliary line must be independent from the original auxiliary line; both contribute separate key conclusions that are jointly necessary for the solution.
- • Do not design multiple sub-problems; each step must serve the solution to the same final problem.
- • The new auxiliary line can leverage the structure of the original auxiliary line — it can be a simple connection, or a new point constructed using the original auxiliary line (e.g. midpoint, projection, external point), a new segment/intersection, or projection.
- • Load auxiliary line image and draw through the code below: image_path = {aux1} img = plt.imread(image_path) h, w = img.shape[:2] fig = plt.figure() ax = fig.add_axes([0,0,1,1]) ax.set_xlim(0, w) ax.set_ylim(h, 0) ax.axis(’off’) ax.set_aspect(’equal’) ax.imshow(img)
- • End the code with plt.show().

Instructions for the problem statement:

- • The new problem must be based on the original diagram, but must not reference the original problem, auxiliary line, or hint at the construction (e.g. do not provide the original auxiliary line or answer as given conditions). The original auxiliary line and answer must be included as steps necessary for solving the new problem.
- • All variable names used in the new problem statement, solution, and auxiliary line drawing code must strictly follow the naming conventions established in the original diagram and the provided drawing code.
- • The problem statement must not include any explicit or implicit hints for construction or drawing, such as “draw”, “connect”, “extend”, “symmetry”, “complete”, etc.

Available inputs:

- • Original problem text: “{question}”
- • Original solution path: “{solution}”
- • Original standard answer: “{answer}”
- • Original drawing code: “{code_ori}”
- • Original auxiliary tools description: “{auxiliary_tools_description}”
- • Auxiliary line code: “{code_aux}”
- • Auxiliary line image

###### Final Output Json Format (Output only the final JSON result. No extra explanations or comments):

{ "id": "{id}", "new_problem": "Newly Designed Problem", "problem_type": "multiple choice or fill_in_blank", "new_auxiliary_tools_description": "Brief description of the new auxiliary tools to be added (what, where and functions)", "python_code_auxiliary": "Pure Python code as a string, directly runnable (loads the saved figure image via plt.imread and overlays auxiliary tools in pixel coordinates)", "solution_markdown": "Step-by-step solution in Markdown with LaTeX, where the auxiliary lines are naturally introduced and utilized as part of the reasoning process (CoT)", "final_answer": "Final Answer" }

Table 12. Prompts for parallel extension.

###### Progressive Expansion (Sequential Extension)

You are an expert exam question designer and visualization engineer. Based on the given original problem, standard answer, auxiliary line image, original drawing code, and auxiliary line drawing code, you need to design either a multiple choice or fill-in-the-blank problem that requires two rounds of auxiliary line code calls to be solved completely. Output the new problem, a clear multi-step auxiliary line drawing plan, the drawing code, and the solution process with the final answer.

###### Steps:

- • You need to read and understand all inputs. Use the provided auxiliary line image to interpret the problem, the purpose of the auxiliary line, and the solution process.
- • The existing auxiliary line is one key conclusion for solving; the other key conclusion must rely on a newly introduced auxiliary line. The new question you provide must require at least two key conclusions or constructions to be solved.
- • Read the auxiliary line image and, based on it, write executable Python code to draw a new auxiliary line. You may use the original code to help understand the points, lines, coordinates, and geometric constructions in the original and auxiliary images.
- • Ensure the new problem maintains the original point naming. Do not pose questions about nonexistent geometric elements.
- • The answer should not be directly obtainable from the original or auxiliary diagram, and each round of auxiliary lines must be necessary and enhance subsequent reasoning.
- • Generate the output: problem statement, reasoning for why multiple rounds of auxiliary lines are necessary, drawing code, detailed problem-solving process and final answer.

Rules for introducing new visual tools:

- • The new auxiliary line must be closely related to the original auxiliary line or the original problem’s key conclusion. It must be constructed based on knowledge obtained from the original auxiliary line or key conclusion. For example, if the original auxiliary line is DM, a new auxiliary line could be the perpendicular to DM.
- • Do not design multiple small subproblems; every step must serve the solution to a single, unified problem.
- • The new auxiliary line can leverage the structure of the original auxiliary line — it can be a simple connection, or a new point constructed using the original auxiliary line (e.g. midpoint, projection, external point), a new segment/intersection, or projection.
- • Load auxiliary line image and draw through the code below: image_path = {aux1} img = plt.imread(image_path) h, w = img.shape[:2] fig = plt.figure() ax = fig.add_axes([0,0,1,1]) ax.set_xlim(0, w) ax.set_ylim(h, 0) ax.axis(’off’) ax.set_aspect(’equal’) ax.imshow(img)
- • End the code with plt.show().

Instructions for the problem statement:

- • The new problem must be based on the original diagram, but must not reference the original problem, auxiliary line, or hint at the construction (e.g. do not provide the original auxiliary line or answer as given conditions). The original auxiliary line and answer must be included as steps necessary for solving the new problem.
- • All variable names used in the new problem statement, solution, and auxiliary line drawing code must strictly follow the naming conventions established in the original diagram and the provided drawing code.
- • The problem statement must not include any explicit or implicit hints for construction or drawing, such as “draw”, “connect”, “extend”, “symmetry”, “complete”, etc.

Available inputs:

- • Original problem text: “{question}”
- • Original solution path: “{solution}”
- • Original standard answer: “{answer}”
- • Original drawing code: “{code_ori}”
- • Original auxiliary tools description: “{auxiliary_tools_description}”
- • Auxiliary line code: “{code_aux}”
- • Auxiliary line image

###### Final Output Json Format (Output only the final JSON result. No extra explanations or comments):

{ "id": "{id}", "new_problem": "Newly Designed Problem", "problem_type": "multiple choice or fill_in_blank", "new_auxiliary_tools_description": "Brief description of the new auxiliary tools to be added (what, where and functions)", "python_code_auxiliary": "Pure Python code as a string, directly runnable (loads the saved figure image via plt.imread and overlays auxiliary tools in pixel coordinates)", "solution_markdown": "Step-by-step solution in Markdown with LaTeX, where the auxiliary lines are naturally introduced and utilized as part of the reasoning process (CoT)", "final_answer": "Final Answer" }

Table 13. Prompts for sequential extension.

###### Perception Data Synthesis (Elements & Visual tag)

You are an expert in constructing visual representations across diverse domains. Based on the specified knowledge point, number of elements, and relationships between elements, please select elements that are consistent with the knowledge point to generate the figure. Output executable Python drawing code and precise data annotations.

###### 1. Element Generation Rules

- • The basic figure must completely reflect the specified knowledge point.
- • For special knowledge points (tables, stem–leaf plots, mazes, ...), use dedicated drawing tools (e.g., ax.table) rather than composing basic elements manually.
- • For statistical tables: the number of elements should equal the number of table cells plus required text/labels. No extra points/lines.
- • All basic elements must be clearly annotated in the element property list.
- • A line may be a diagonal, or a segment with existing endpoints.
- • Angles and extra lines must be derived from existing vertices.
- • Added complexity must not obscure the assessment of the knowledge point. Completely random/unrelated elements are forbidden.
- • No element may exceed the canvas boundary. 2. Element Definition & Attribute Rules
- • Point: "type":"point","x":...,"y":...,"label":"A","semantic":""
- • Line: "type":"line","x1":...,"y1":...,"x2":...,"y2":...,"semantic":""
- • Angle: "type":"angle","vertex":[x,y],"dir1":[dx1,dy1],"dir2":[dx2,dy2]
- • Circle: "type":"circle","cx":...,"cy":...,"r":...
- • Function Curve: "type":"function_curve","expression":"y=sin(x)","domain":[...],"style":"solid"
- • Text: "type":"text","content":"...", "position":[x,y]
- • Symbol: "type":"symbol","position":[x,y],"symbol_type":"perpendicular" All elements must include a semantic field (write “None” if unnecessary). 3. Coordinate & Size Requirements
- • All coordinates must be absolute pixel values (integers).
- • No out-of-bound elements; circles must stay within canvas.
- • Angle arcs must be computed from the given direction vectors. 4. Drawing & Coding Specifications
- • Must output fully executable Python code using matplotlib.
- • Required boilerplate: w, h = DPI = fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI) ax = fig.add_axes([0,0,1,1]) ax.set_xlim(0, w) ax.set_ylim(h, 0) ax.axis(’off’) ax.set_aspect(’equal’)
- • Do NOT use plt.tight_layout().
- • End with plt.show().
- • The diagram must not contain extra explanatory text.

###### 5. Output Format Output must follow this JSON format (no extra text):

{ "id": "new_id", "knowledge_point": "knowledge_point", "element_num": "element_num", "python_code": "...", "element_arrtibute": "..." }

###### Input Arguments:

- • Knowledge Concept: "knowledge_concept"
- • Number of elements: "greater than {element_num}"
- • Element relationships: "relation"

Table 14. Prompt for element and perception tag construction.

###### Perception Data Synthesis (Surface-level)

You are a point-level visual QA data generation expert. Based on the image description and the names of the points/elements to be detected, please generate question–answer data. The question should ask for the locations of these points or vertices in the image, and the answer should provide their coordinates in dictionary format.

Input: The element annotation information in the image (absolute coordinates) is:

{element_arrtibute}

Example: Question: Detect and output the pixel coordinates of points A, B, and C in the image (dictionary format is recommended, e.g., {’A’: (x1, y1), ’B’: (x2, y2), ’C’: (x3, y3)}). Answer: {’A’: (x1, y1), ’B’: (x2, y2), ’C’: (x3, y3)}

###### Specification notes:

- • If there are no explicit geometric shapes, ask perception-related questions, such as the coordinates of the points where a function achieves a specific value in the image, or comparing/calculating statistical data shown in image tables.
- • Always provide the answers as absolute coordinates (pixel positions) of relevant targets or data within the image, rather than abstract mathematical results.

Output Format: Final output must follow this JSON format (output only the JSON, no extra explanations or comments):

{ "id": "{id}", "question": "", "answer": "", "type": "Surface Level Perception QA" }

Table 15. Prompt for surface-level perception Q&A construction.

###### Perception Data Synthesis (Semantic-level)

You are a semantic-level visual QA data generation expert. Please create QA pairs for specified spatial–semantic points based on the structural semantic annotations of the input image. The question should ask for detection of specific structural feature points in the image (e.g., “the top-left vertex of the cube”), and the answer should return their coordinates.

Input: The element annotation information in the image (absolute coordinates) is:

{element_arrtibute}

Example: Question: Detect and output the pixel coordinates of the top-left front corner of the cube in the image (dictionary format is recommended, e.g.,

{’Top-left vertex’: (x, y)}). Answer: {’Top-left vertex’: (x, y)}

###### Specification notes:

- • Use spatial descriptions or mathematical definitions (e.g., “top-left front corner of the upper surface”, “vertex of a triangle”) in the question.
- • If the element annotation information provides explicit naming or numbering for the corresponding vertex (e.g., A, B, P1, V3), use these symbols as the standard expression in the answer. Do not use these explicit names in the question.
- • If there are no explicit geometric shapes, ask perception-related questions, such as the coordinates of the points where a function achieves a specific value in the image, or comparing/calculating statistical data shown in image tables.
- • Always provide the answers as absolute coordinates (pixel positions) of relevant targets or data within the image, rather than abstract mathematical results.

Output Format: Final output must follow this JSON format (output only the JSON, no extra explanations or comments):

{ "id": "{id}", "question": "", "answer": "", "type": "Semantic Level Reasoning QA" }

Table 16. Prompt for semantic-level visual Q&A construction.

###### Perception Data Synthesis (Integrated reasoning)

You are an expert in generating QA data for visual perception and computation tasks. Please combine the input image’s structural information and geometric reasoning to generate visual QA pairs such as finding the pixel coordinates of the center point of a square shown in the image.

Input: The element annotation information in the image (absolute coordinates) is:

{element_arrtibute}

Example: Question: Detect and output the pixel coordinates of the center point of the square in the image (dictionary format recommended, e.g.,

{’Center’: (x, y)}). Answer: {’Center’: (x, y)}

###### Specification notes:

- • Use spatial descriptions or mathematical definitions (e.g., “foot of the perpendicular from vertex A of the trapezoid to base CD”, “incenter of the triangle”) in the question.
- • If the element annotation information provides explicit naming or numbering for corresponding points (e.g., A, B, O, V3), use these annotation symbols in the answer, but never in the question.
- • If the image contains no explicit geometric shapes, ask perception-related tasks such as:

- – locating where a function achieves a specific value;
- – detecting coordinates corresponding to statistical data in charts or tables.

- • All answers must be absolute pixel coordinates of the target elements, not abstract math results.

Output Format: The final output must follow this JSON format (output only the JSON, no extra explanations or comments):

{ "id": "{id}", "question": "", "answer": "", "type": "Integrated Reasoning QA" }

Table 17. Prompt for integrated reasoning Q&A construction.

