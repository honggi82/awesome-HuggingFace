# arXiv:2511.16668v1[cs.CV]20Nov2025

## V-ReasonBench: Toward Unified Reasoning Benchmark Suite for Video Generation Models

Yang Luo1,∗, Xuanlei Zhao1,∗, Baijiong Lin2,∗, Lingting Zhu3, Liyao Tang4 Yuqi Liu5, Ying-Cong Chen2, Shengju Qian6,†, Xin Wang6, Yang You1 1NUS 2HKUST(GZ) 3HKU 4USYD 5CUHK 6LIGHTSPEED Project Page: https://oahzxl.github.io/VReasonBench/

Abstract

Recent progress in generative video models, such as Veo3, has shown surprising zero-shot reasoning abilities, creating a growing need for systematic and reliable evaluation. We introduce V-ReasonBench, a benchmark designed to assess video reasoning across four key dimensions: structured problem-solving, spatial cognition, pattern-based inference, and physical dynamics. The benchmark is built from both synthetic and real-world image sequences and provides a diverse set of answer-verifiable tasks that are reproducible, scalable, and unambiguous. Evaluations of six state-of-the-art video models reveal clear dimensionwise differences, with strong variation in structured, spatial, pattern-based, and physical reasoning. We further compare video models with strong image models, analyze common hallucination behaviors, and study how video duration affects Chain-of-Frames reasoning. Overall, V-ReasonBench offers a unified and reproducible framework for measuring video reasoning and aims to support the development of models with more reliable, human-aligned reasoning skills.

### 1. Introduction

Recent advancements in generative video models [2, 16, 24, 28, 44], such as Veo-3.1 [14], Sora-2 [27], and Kling-2.5 [19], have demonstrated remarkable zero-shot reasoning capabilities across diverse visual tasks. These models exhibit emergent abilities to perceive, model, and reason over dynamic visual data, reflecting a growing trajectory toward general-purpose vision foundation models. Such progress suggests that video models are beginning to move beyond mere frame synthesis, displaying cognitive-like behaviors

*Equal Contribution. †Corresponding Author.

Pattern-based Inference

Structured Problem-Solving

72.00

46.86 26.29

###### 23.64

###### 32.73

3.64

19.91

40.00

5.14

0

0.57

0.57

0

20

40

60

33.33 36.67

80

36.76

26.47

23.33 36.67

5.88 13.24

8.82

33.82

26.67

33.33

Seedance-1.0-Lite Vidu-Q2 Kling-2.5-Turbo-Pro Hailuo-02 Veo-3.1 Sora-2

| |
|---|

PhysicalDynamics

SpatialCognition

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 1. Evaluation of Video Generation Models on VReasonBench. The performance of six video generation models across the four core reasoning dimensions is illustrated. Detailed numerical results are provided in Tab. 2.

such as pattern inference, causal prediction, and strategic planning [41]. However, despite these promising signs, the capacity to quantitatively evaluate reasoning ability in these models remains underexplored and poorly standardized.

The “Chain-of-Frame” (CoF) [12, 41] paradigm treats video generation as a sequence of reasoning steps, in direct analogy to “Chain-of-Thought” in language models [23, 37, 38, 40, 47]. In this framework, a model receives an initial image and a prompt, then produces a series of frames where the intermediate frames embody its reasoning trajectory and the final frame represents its answer or outcome. Because the final frame encapsulates the model’s inferred solution, CoF enables a last-frame evaluation pipeline: we

judge the model on its concluding frame rather than requiring annotation of all intermediate steps. This provides an efficient and scalable approach for unambiguous evaluation of video reasoning, because evaluating every intermediate step requires substantial annotation effort, incurs prohibitive computational cost, and often yields noisy or ambiguous supervisory signals [6, 18, 33].

While last-frame evaluation offers an efficient way to assess reasoning within the CoF paradigm, vision-language models (VLMs) are not always reliable as sole automatic judges. In particular, VLMs may face difficulty interpreting visually dense or grid-structured layouts that require precise recognition of small cells, thin boundaries, or subtle geometric relationships [18, 21, 29]. To complement these limitations and ensure stable evaluation across diverse task types, V-ReasonBench adopts a hybrid strategy: maskbased evaluation for tasks with well-defined object regions, grid-based evaluation for tasks requiring fine-grained structural accuracy, and lightweight VLM-based evaluation only for visually simple outputs where consistency can be maintained. Each method produces a numerical score that is converted into a pass or fail decision using task-specific thresholds, enabling scalable and reproducible pass@k evaluation while preserving strong alignment with human judgment.

To support consistent evaluation under the Chain-ofFrame framework and our last-frame scoring strategy, VReasonBench is partitioned into four complementary reasoning classes, each targeting a specific dimension of reasoning: 1) Structured problem-solving encompasses tasks involving numerical manipulation, strategic planning in dynamic game states, and procedural logic derived from visualized program traces. 2) Spatial cognition evaluates understanding of spatial relations, geometric transformations, and symmetry-based patterns. 3) Pattern-based inference probes sequence completion, analogical mapping, and abstract rule induction beyond surface-level visual cues. 4) Physical dynamics examines comprehension of motion, force interactions, temperature effects, and pressure-driven behavior. Together, these four classes provide a comprehensive assessment of cognitive capabilities in video understanding and reasoning.

Built upon these methodological principles, VReasonBench provides a unified suite for evaluating reasoning in generative video models under the CoF framework. Each instance consists of an initial image, a task instruction, and a target final image, with both procedurally generated and curated scenarios enabling fine control over difficulty and perceptual variation. By applying deterministic, last-frame scoring across four complementary reasoning classes, the benchmark offers consistent measurements of a model’s ability to interpret structured inputs, follow task-specific rules, and produce coherent final outcomes.

Evaluating six cutting-edge video generation models (Sora-2 [27], Veo-3.1 [14], Kling-2.5-Turbo-Pro [19], Seedance-1.0-Lite [11], Vidu-Q2 [32], and Hailuo-02 [26]) reveals clear strengths and limitations across multiple reasoning dimensions in Fig. 1. Our analysis further examines each model’s characteristic reasoning patterns, the influence of video duration on CoF performance, and the contrast between video and image models, highlighting several recurring hallucination behaviors unique to video generation. As an early exploratory effort, V-ReasonBench provides an efficient and reproducible reasoning-centric framework that offers a unified foundation for benchmarking video reasoning and guiding future models toward more reliable, humanaligned visual reasoning.

### 2. Related Works

Video Generation. Recent advances in video generation have been strongly driven by diffusion–transformer models, which provide scalable architectures for producing highquality visual content [3, 10, 25, 46]. Commercial systems such as OpenAI’s Sora-2 [27], Minimax’s Hailuo-

- 02 [26], Runway’s Gen-3 [31], and Google DeepMind’s Veo-3.1 [14] achieve impressive visual quality and strong alignment with text prompts, but their training data, model designs, and evaluation procedures remain undisclosed. Alongside these developments, recent research efforts, including CogVideoX [42], HunyuanVideo [35], and the Wan series [36], have focused on enhancing temporal consistency, motion fidelity, and semantic coherence, reflecting a broader trend toward more controllable and robust video generation systems. Video Reasoning. Recent video generation models have shown emergent visual reasoning abilities that go beyond their original training goals [41]. The Chain-of-Frame (CoF) paradigm [41], analogous to chain-of-thought reasoning in language models [39], frames video generation as a stepwise spatial and temporal reasoning process. This idea has been explored through approaches such as VChain [17], which uses multimodal models to produce keyframes as intermediate reasoning signals, and visual chain-of-thought methods [4, 45], which combine visual perception with iterative reasoning for knowledge-intensive tasks and control. However, despite growing interest in evaluating these emergent capabilities, there remains a lack of a scalable and comprehensive benchmark that systematically measures the breadth and depth of video reasoning across diverse task settings.
- 3. V-ReasonBench Pipeline

##### 3.1. Design Principles

V-ReasonBench is designed to evaluate the reasoning capabilities of generative video models through a comprehensive

Data Preparation

Generated Videos

Evaluation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Vidu-Q2

Hailuo-02 Seedance-1.0-Lite

Last Frame GT (if any)

[Figure 7]

[Figure 8]

[Figure 9]

| | |
|---|---|
| | |

Sora-2 Veo-3.1 Kling-2.5-Turbo-Pro

Video Generation Models

[Figure 10]

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

Pass or Not

Sampled Video

- Figure 2. Overview of V-ReasonBench pipeline. The benchmark covers four reasoning dimensions, integrates both synthetic and realworld scenarios, and supports reproducible, large-scale evaluation of video reasoning capabilities.

suite of tasks that require genuine understanding of visual reasoning, as illustrated in Fig. 2. Our benchmark is built on three core principles:

- • Last-Frame Dependency: All tasks are designed such that the final answer can be determined exclusively from the last frame of generated videos, enabling unambiguous and scalable evaluation of reasoning outcomes.
- • Unified Evaluation Metric: We employ pass@k as our primary evaluation metric across all reasoning classes, providing a consistent measure of model performance and facilitating direct comparison between different approaches.
- • Multi-Faceted Reasoning: The benchmark spans 4 distinct reasoning dimensions to comprehensively assess models’ capabilities across mathematical, strategic, spatial, logical, physical, and programmatic reasoning.

##### 3.2. Reasoning Tasks and Formulations

V-ReasonBench formalizes a comprehensive suite of reasoning classes, each targeting specific cognitive capabilities essential for holistic video understanding. The following is an overview of the 4 core reasoning dimensions in our benchmark. More explanations and visualizations of all tasks are provided in Appendix 7.

Structured Problem-Solving. This dimension evaluates the model’s ability to perform systematic, rule-based reasoning across quantitative and procedural contexts. It consists of four sub-tasks:

- • Arithmetic Operation examines numerical reasoning from visual representations, requiring the model to perform basic arithmetic operations, including addition, subtraction, multiplication, and division.
- • Code Execution tests procedural and logical thinking through visualized programming tasks, where the model observes LeetCode-style code with inputs and must pre-

- dict the correct program output.
- • Sudoku evaluates constraint-based reasoning by asking the model to complete a partially filled grid while adhering to fixed numeric rules.
- • Tic-Tac-Toe assesses strategic planning in a dynamic, adversarial environment, requiring the model to infer or select the optimal next move given a game state.

Spatial Cognition. This dimension examines spatial intelligence across three sub-tasks:

- • Shape Fitting assesses mental rotation and spatial arrangement skills.
- • Visual Symmetry evaluates recognition of reflective and rotational symmetries.
- • Color Connection tests pathfinding by requiring the model to establish valid connections between elements of matching color.

Pattern-based Inference. This dimension probes abstract relational reasoning and inductive generalization across three sub-domains:

- • Sequence Completion evaluates the ability to infer the next element in a visual or symbolic progression by recognizing hidden temporal or spatial rules.
- • Analogy Solving tests the understanding of relational structure through problems of the form “A:B as C:?” requiring cross-domain correspondence beyond surface similarity.
- • Rule Following examines the capacity to infer governing principles from a few examples and apply them consistently to novel instances.

Physical Dynamics. This dimension evaluates the understanding of fundamental physical principles through four sub-tasks:

• Block Sliding examines the ability to predict motion under gravity and friction, determining whether an object placed on a slope will remain stationary or slide down.

[Figure 18]

[Figure 19]

[Figure 20]

#### Ground Truth

Model Prediction

Input

V-ReasonBench:

VLM: Since the provided results

Score = 0.71 Pass = False

perfectly and logically complete all three independent patterns, the reasoning is flawless.

#### Evaluation

[Figure 21]

[Figure 22]

- Figure 3. Example failure case from Sequence Completion task illustrating the limitations of VLM-based automatic evaluation. Although the underlying rule is simple, the VLM incorrectly assesses the model’s output due to difficulties in recognizing small grid cells and fine structural differences. More examples are given in Appendix 9.

Table 1. Overview of reasoning dimensions, tasks, and number of videos in V-ReasonBench. Each instance is an initial–final image pair, with each model generating five videos for pass@5 evaluation.

- • Communicating Vessels (CV) evaluates understanding of fluid pressure and equilibrium, asking the model to infer how liquid levels adjust across connected containers when pressure or volume changes.
- • Temperature-Induced Deformation assesses reasoning about material properties under thermal variation, such as predicting melting, shrinking, or deformation of ice blocks as temperature changes.

###### Dimension Task # Videos

Arithmetic Operation

Structured Problem-Solving

Code Execution Sudoku Tic-tac-toe

5,250

##### 3.3. Dataset Construction

Shape Fitting Visual Symmetry 2,040 Color Connection

Spatial Cognition

Image Pair Generation. We instantiate the benchmark with an image–pair framework rather than full videos. Approximately 90% of instances are programmatically synthesized in custom simulation environments, yielding an initial image that represents the starting state and a final image that serves as the ground truth outcome, which enables scalable data generation. Procedural generation provides broad coverage across reasoning types while preserving consistent state transitions, and every pair passes automated validation followed by targeted manual inspection to ensure that each transformation is unambiguous, solvable by reasoning, and free of annotation shortcuts. This design follows best practices in controlled datasets for spatiotemporal and causal reasoning [13, 15, 43].

Sequence Complete Analogy Solving 1,590 Rule Follow

Pattern-based Inference

Block Sliding CV 900 Temperature

Physical Dynamics

##### 3.4. Limitations of VLM-Based Evaluation

Although recent benchmarks increasingly rely on visionlanguage models (VLMs) for automatic judgment, this approach faces notable limitations. VLMs often struggle to interpret complex visual layouts, particularly grid-based or densely structured scenes that require precise detection of small cells, thin boundaries, or fine spatial relationships, as shown in Fig. 3. These difficulties lead to incorrect judgments even when the task logic is straightforward, as the model may miscount, overlook subtle structural cues, or misidentify local patterns. These challenges highlight the need for more reliable evaluation methods for tasks that de-

Dataset Statistics. V-ReasonBench includes 326 reasoning instances represented by 652 images paired with corresponding question annotations across four reasoning classes as summarized in Tab. 1. By generating 5 videos for each instance-model pair, this yields a total of 9,780 generated videos assessed in the benchmark. Each instance is specified by an initial–final image pair that encodes the state transition necessary for reasoning.

pend on fine-grained visual understanding.

- 3.5. Evaluation Methodology

To enable precise and scalable last-frame-based evaluation, we design three complementary assessment methods tailored to the characteristics of different task types in VReasonBench.

- • Mask-based Evaluation. Tasks with clear object boundaries and localized reasoning regions, including Sequence Completion, Analogy Solving, Block Sliding, Communicating Vessel, and Tic-tac-toe, are evaluated using a maskbased comparison strategy. Masks are generated either from annotated templates or through automated segmentation tools such as SAM-2 [30] to isolate target regions. Pixel-level mean squared error is then computed with higher weights inside the masked areas and lower weights in non-essential regions, reducing the influence of background changes or stylistic variations.
- • Grid-based Evaluation. For tasks that require structured layouts or fine-grained spatial precision, including Visual Symmetry, and Rule Following, we employ a grid-based evaluation. Each frame is divided into uniform cells, and cell-wise accuracy is computed by comparing the predicted and ground truth states in corresponding grid locations. This method captures discrete positional errors and ensures sensitivity to geometric and structural constraints.
- • VLM-based Evaluation. Tasks composed of simple items that VLMs can easily handle, including Arithmetic Operation, Sudoku, Code Execution, TemperatureInduced Deformation, Color Connection, and Shape Fitting, are scored using a lightweight VLM-based procedure. For mathematical and code tasks, the VLM extracts textual or symbolic outputs from designated regions. For perception-oriented tasks, the VLM assesses structural correctness and relational consistency based on carefully designed prompts. This approach provides flexible and standardized scoring where purely pixel-based metrics are insufficient.

Each of these evaluation strategies produces a numerical score for the final frame, which is then converted into a binary passed or unpassed decision using task-specific thresholds. This unified procedure ensures consistent, scalable, and interpretable pass@k evaluation across all reasoning categories, with strong alignment to human assessment.

- 4. Experiments

- 4.1. Experimental Setup

Models Evaluated. We evaluate six prevalent generative video models that represent the current frontier of commercial video generation capabilities. The models include: Sora-2 [27], Veo-3.1 [14], Hailuo-02 [26], Vidu-Q2 [32],

KlingAI-2.5-Turbo-Pro [19], and Seedance-1.0-Lite [11]. This diverse selection supports a comprehensive evaluation of reasoning behaviors across models with different architectures, training pipelines, and capability tiers.

Evaluation Protocol. We evaluate each model using a standardized protocol that generates five videos per prompt. All models are run with their default parameter settings and produce videos of approximately five seconds in duration. Resolutions of 720p or 768p are used when supported, and every video is generated in a 16:9 aspect ratio to ensure consistency. Video generation prompts are carefully designed and kept identical across models to minimize prompt-induced bias. Full prompts, score calculation methods and corresponding thresholds for each task are provided in Appendix 7. For all tasks, we use pass@5 as the primary evaluation metric. In VLM-based evaluation tasks, such as color connect and shape fit tasks, where VLM-based assessment is reliable, we employ Gemini-2.5-Pro [7] as the automatic evaluator.

##### 4.2. Per-dimension Evaluation

Across the four reasoning dimensions, the results in Tab. 2 suggest that current video models exhibit distinct areas of strength rather than uniformly high performance. Sora2 [27] continues to lead in Structured Problem-Solving (72.00), Spatial Cognition (36.76), and Pattern-based Inference (40.00), with Hailuo-02 [26] following as a strong performer in these categories. These trends indicate that some models have developed comparatively stronger capabilities in discrete decision-making, spatial understanding, and abstraction from visual patterns. In contrast, Seedance1.0-Lite [11], Vidu-Q2 [32], and Kling-2.5-Turbo-Pro [19] score lower in these dimensions, which may reflect differing design priorities, such as a stronger focus on visual generation fidelity over rule-driven or symbolic reasoning.

A different pattern emerges in Physical Dynamics. Here, Hailuo-02 and Vidu-Q2 achieve the highest scores (both 36.67), while Sora-2 reaches a moderate 26.67 despite its strengths in the other dimensions. This divergence suggests that the inductive biases supporting structured and pattern-based reasoning may not directly translate into robust physical understanding, and that some models may instead focus on producing visually coherent physical motions without fully capturing underlying physical principles. Overall, the dimension-wise results highlight that video reasoning is multifaceted: different systems capture different aspects of the task, and there remains substantial opportunity to develop models that more uniformly integrate abstract, spatial, and physically grounded reasoning.

##### 4.3. Human Preference Alignment

To verify the reliability of our scoring pipeline, we perform a human–alignment study that directly compares the

Table 2. Model-level overall and per-dimension performance on V-ReasonBench. A pass@5 score for each model is calculated within each dimension and presented accordingly.

Structured Problem-Solving

Spatial Cognition

Pattern-based Inference

Physical Dynamics

Average

Model

Seedance-1.0-Lite [11] 0.57 8.82 0.00 33.33 10.68 Vidu-Q2 [32] 0.57 5.88 23.64 36.67 16.69 Kling-2.5-Turbo-Pro [19] 5.14 13.24 3.64 23.33 11.34 Veo-3.1 [14] 26.29 26.47 10.91 33.33 24.25 Hailuo-02 [26] 46.86 33.82 32.73 36.67 37.52 Sora-2 [27] 72.00 36.76 40.00 26.67 43.86

###### Human Preference Alignment

[Figure 23]

[Figure 24]

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |

97.09

thinking

80

Accuracy(%)

60

40

20

Figure 5. Example from the Seedance-1.0-Lite model on the horizontal visual symmetry task. The model introduces additional decorative patterns across the mirrored axis, illustrating its tendency to enrich visual appearance rather than preserve original geometric form.

0

TicTacToe Math CodeSudokuRule FollowingVisual AnalogySequence CompletionShapeFittingVisual SymmetryTemperature-Induced DeformationColor ConnectionCommunicating VesselsBlockSliding

sessment framework that remains well aligned with human reasoning preferences.

- Figure 4. Human–alignment validation of our benchmark’s scoring pipeline. Each point compares binary Pass/Unpass decisions from the automatic evaluation with human judgments across four reasoning categories.

### 5. Discussions

##### 5.1. Reasoning Patterns in Video Generation

binary passed/unpassed decisions produced by our benchmark with human judgments. For each reasoning task, we randomly sample 120 videos spanning a range of difficulty levels. Six graduate-level examiners familiar with reasoning benchmarks independently assess whether each model’s final-frame prediction correctly solves the task, following the same execution-based criteria defined in our benchmark.

We then compute the accuracy of agreement between the benchmark’s passed/unpassed outputs and the human passed/unpassed labels. Across models and categories, we observe consistently high alignment (accuracy = 97.09% on average) as presented in Fig. 4, demonstrating that our automatic decisions closely track human evaluations. The remaining discrepancies typically arise in visually ambiguous cases, such as nearly symmetric configurations or physics scenarios with partial occlusion, where human evaluators show slightly higher tolerance to minor perceptual deviations.

Overall, these results indicate that our evaluation protocol produces human-consistent pass/fail judgments and that V-ReasonBench provides a scalable, reproducible as-

Across several reasoning dimensions, we observe a consistent phenomenon in which some video models emphasize visual enhancement over structural accuracy, which correlates with lower reasoning performance. Taking Seedance1.0-Lite as examples in Fig. 5, the tendency appears on tasks featuring minimal, clean backgrounds (often pure white or black) with only the required elements. In these minimalist settings, the models frequently add extra textures or objects or modify the scene layout, which alters the intended structure and reduces pass@5 scores. The same pattern appears beyond structured problem solving, including spatial and geometric tasks, pattern-based inference, and parts of physical reasoning that require a stable canvas. More visualizations of reason patterns can be found in Appendix 10.

A likely cause is a creative bias inherited from pretraining on large, open-domain video corpora where realism and visual richness are valued. When the input scene is visually sparse, the model may treat it as incomplete and attempt to improve it with additional detail rather than preserve the given structure. Training and decoding choices can strengthen this behavior, such as reconstruction objectives

[8, 22, 34, 48]. While longer clips can improve performance when additional frames contain relevant information and the model effectively integrates distant cues, excessive temporal expansion tends to dilute attention and accumulate noise. A deeper investigation into how frame budget, sampling stride, and context window interact with reasoning quality will be an important direction for future work.

[Figure 25]

[Figure 26]

4s thinking

Different CoF Time

[Figure 27]

[Figure 28]

8s thinking

##### 5.3. Video Models vs. Image Models

To isolate the contribution of temporal reasoning, we compare Veo-3.1 with NanoBanana [9] on V-ReasonBench, treating them as representative video-based and imagebased reasoning paradigms. Veo-3.1 is evaluated in full video-generation mode, while NanoBanana functions as a powerful image-only baseline that performs reasoning without temporal progression. Fig. 7 shows representative outcomes on physics-law and code-reasoning tasks, highlighting the performance gap attributable to temporal modeling.

Inputs Outputs

[Figure 29]

[Figure 30]

5s thinking

Different CoF Time

[Figure 31]

[Figure 32]

10s thinking

Image-based models operate on a single static frame and therefore rely heavily on structural priors, textual cues, and pattern recognition. This enables high reliability on codereasoning and symbolic tasks, where syntax, layout, and character-level precision drive performance. However, the lack of temporal information limits their ability to infer dynamics. When facing tasks involving momentum transfer, balance, collisions, spatial transformations, or chainstructured geometric manipulation, they often choose visually plausible outcomes that do not reflect the correct causal process.

Inputs Outputs

- Figure 6. Effect of video duration on reasoning outcomes of Sora2 in the Chain-of-Frame setting. Each row compares model generations with different “thinking” durations for tasks such as Sudoku and Rule Following. Although longer durations correspond to longer reasoning processes (4s vs. 8s, 5s vs. 10s), the resulting outputs do not consistently improve.

that reward fine texture, temporal smoothness terms that encourage motion even when the correct solution is static, and limited exposure to diagram-like data with small symbols and thin boundaries. Together, these factors push the generator toward aesthetic completion instead of structurepreserving rendering, which can conflict with tasks that rely on precise spatial or symbolic constraints.

Video models exhibit the opposite strength profile. By generating a CoF sequence, Veo-3.1 can explicitly model transitions, represent latent motion paths, and maintain spatial and causal continuity across time. This frame-wise evolution provides the model with an internal mechanism for simulating physical dynamics and multi-step spatial transformations, which directly improves accuracy on physicsoriented tasks. Importantly, the same CoF mechanism also benefits code-reasoning tasks: intermediate frames act as visual checkpoints that stabilize the symbolic generation process, reducing local inconsistencies and improving stepwise logical execution.

##### 5.2. Influence of Duration on Video Reasoning

We regard the Chain-of-Frame (CoF) as the thinking process of video reasoning and analyze how video duration influences the correctness of final answers. In CoF, a longer duration corresponds to a longer or more detailed reasoning process, which intuitively might be expected to enhance reasoning accuracy. However, as shown in Fig. 6, our observations reveal a counterintuitive pattern: extending the duration does not consistently lead to better reasoning or higher-quality outputs. Instead, longer sequences often introduce redundant or irrelevant content, and in some cases, cause the model to hallucinate unrelated objects in the final frame.

Temporal modeling through CoF gives video models clear advantages in both physical and procedural reasoning. Image models are strong at static structural tasks, while video models leverage process-aware temporal dynamics to handle multi-step, causal, and simulation-heavy problems. Combining precise static parsing with CoF-based temporal modeling offers a promising path toward stronger visual reasoning systems.

This phenomenon aligns with findings from prior studies on temporal reasoning, which indicate that increasing sequence length expands the available causal evidence but also magnifies attention drift and temporal mis-binding

##### 5.4. Hallucination in Video Reasoning

During the exploratory stage of benchmark design, we observed several hallucination phenomena emerging within

|[Figure 33]<br><br>Prompt: Given the initial frame of a<br><br>rigid rectangular block atop an incline and the annotated parameters, predict<br><br>the final frame showing where the<br><br>block comes to rest after sliding.|
|---|

[Figure 34]

NanoBanana

NanoBanana

|[Figure 35]<br><br>Prompt: Complete this code problem.<br><br>Enter the answer behind \"# Output:\".<br><br>Do not modify the existing code.|
|---|

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

Input

Input

Veo-3.1

Veo-3.1

- Figure 7. Comparison between Veo-3.1 (video model) and NanoBanana (image model) on Block Sliding (left) and Code Execution (right). Video models leverage the Chain-of-Frames process to simulate intermediate states, enabling stronger performance on tasks that require causal or temporal reasoning, although intermediate transitions may still appear physically inconsistent. Image models provide clean and stable outputs and excel at text-based tasks such as code execution, but their single-frame reasoning limits their ability to capture the underlying physical dynamics.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

…… …… ……

[Figure 53]

[Figure 54]

…… …… ……

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

…… …… ……

[Figure 59]

[Figure 60]

Inputs Outputs

Chain-of-Frames Thinking Process

Navigation

Maze

Solving

Newton

Cradle

- Figure 8. Examples of hallucinations in the Chain-of-Frame reasoning process. Each row shows a trajectory from input to output, where the final frame is correct but intermediate frames display unrealistic or physically inconsistent transitions.

the Chain-of-Frame process. Models may sometimes produce the correct final outcome (last frame) while following an incorrect reasoning process. As illustrated in Fig. 8, in the maze-solving task, a mouse successfully reaches the cheese in the final frame even though its intermediate trajectory passes through solid walls. A similar issue also appears in navigation tasks. In the Newton’s cradle task, the final configuration of moving and stationary balls aligns with the ground-truth label, yet the intermediate frames violate momentum conservation. For instance, when the leftmost ball is released, the entire system remains still instead of transferring motion immediately. These cases exemplify temporal hallucination, where invented or misordered actions and fabricated transitions preserve the correct endpoint but

break causal consistency. This phenomenon has been documented in recent evaluations of video language models [5, 20] and corroborated by multimodal hallucination surveys highlighting their vulnerability to dense or abstract visual patterns [1].

From a benchmarking viewpoint, such “right answer, wrong process” failures are hard to detect if we only check endpoints, and they are also difficult to adjudicate using VLMs as mid-frame judges because VLMs themselves can mis-bind temporal relations or hallucinate missing steps [5, 20]. Accordingly, we favor end-state–verifiable tasks where any process error necessarily yields an incorrect terminal state.

### 6. Conclusion

We present V-ReasonBench, a unified benchmark suite for evaluating video reasoning under the Chain-of-Frame paradigm. Using scalable last-frame scoring across four core reasoning dimensions, it provides a reliable and scalable assessment beyond visual fidelity. Experiments on six state-of-the-art models reveal distinct strengths and systematic failure modes, underscoring the gap between generation quality and true reasoning ability. As an early exploration, V-ReasonBench offers a reproducible foundation for advancing human-aligned video reasoning models.

### References

- [1] Zhuosheng Bai and et al. Hallucination of multimodal large language models. arXiv preprint, 2024. 8
- [2] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreelsv2: Infinite-length film generative model, 2025. 1
- [3] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Diffusion transformers for image and video generation, 2024. 2
- [4] Zhuowan Chen, Qihang Zhou, Yibo Shen, Yuzhe Hong, Zhiyuan Sun, Dan Gutfreund, and Chuang Gan. Visual chain-of-thought prompting for knowledge-based visual reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1244–1252, 2024. 2
- [5] W. Y. Choong and et al. Benchmarking temporal hallucinations in vision llms. arXiv preprint, 2024. 8
- [6] Z Chu and et al. Navigate through enigmatic labyrinth: A survey of chain-of-thought reasoning in large language models. ACL Long 2024, 2024. 2
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, and et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. 5, 13, 16
- [8] Xinyu Fang and et al. Mmbench-video: A long-form multi-shot benchmark for video understanding. In NeurIPS Datasets and Benchmarks, 2024. 7
- [9] Alisa Fortin, Guillaume Vernade, Kat Kampf, and Ammaar Reshi. Introducing Gemini 2.5 Flash Image (aka “nano banana”). Blog post, Google Developers, 2025. Available at: https://developers.googleblog.com/en/introducinggemini-2-5-flash-image/. 7
- [10] Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, Renrui Zhang, Junlin Xi, Wenqi Shao, Zhengkai Jiang, Tianshuo Yang, Weicai Ye, He Tong, Jingwen He, Yu Qiao, and Hongsheng Li. Lumina-t2x: Trans-

- forming text into any modality, resolution, and duration via flow-based large diffusion transformers, 2024. 2
- [11] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 2, 5, 6

- [12] S. Ghazanfari and et al. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318, 2025. 1
- [13] Rohit Girdhar and Deva Ramanan. Cater: A diagnostic dataset for compositional actions and temporal reasoning. arXiv preprint arXiv:1910.04744, 2019. 4
- [14] Google DeepMind. Veo 3.1: Google deepmind video generation model. https://deepmind.google/models/ veo/, 2025. 1, 2, 5, 6
- [15] Michaela Gr¨unde-McLaughlin, Ranjay Krishna, Juan Carlos Niebles, and Li Fei-Fei. Agqa: A benchmark for compositional spatio-temporal reasoning. In CVPR, 2021. 4
- [16] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 1
- [17] Ziqi Huang et al. Vchain: Chain-of-visual-thought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025. 2
- [18] Ryo Kamoi, Yusen Zhang, Sarkar Snigdha Sarathi Das, Ranran Haoran Zhang, and Rui Zhang. Visonlyqa: Large vision language models still struggle with visual perception of geometric information, 2025. 2
- [19] Kuaishou Technology. Kling ai: Next-generation ai creative studio. https://klingai.com/, 2024. 1, 2, 5, 6
- [20] Chong Li and et al. Vidhalluc: Evaluating temporal hallucinations in multimodal large language models. In CVPR,

2025. 8

- [21] Chengxi Li, Wenxuan Zhang, Xuwu Sun, Zhaoxiang Zhang, Yuntao Chen, and Kaipeng Zhang. Vision-language models as top-view spatial reasoners. In EMNLP, 2024. 2
- [22] D. Li, Z. Huang, H. Liu, K. Zou, Y. He, F. Zhang, Y. Zhang, J. He, W.-S. Zheng, Y. Qiao, and Z. Liu. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024. preprint. 7
- [23] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning, 2025. 1
- [24] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, Yu Zhou, Deshan Sun, Deyu Zhou, Jian Zhou, Kaijun Tan, Kang An, Mei Chen, Wei Ji, Qiling Wu, Wen Sun, Xin Han, Yanan Wei, Zheng Ge, Aojie Li, Bin Wang, Bizhu Huang, Bo Wang, Brian Li, Changxing Miao, Chen Xu, Chenfei Wu, Chenguang Yu, Dapeng Shi, Dingyuan Hu, Enle Liu, Gang Yu, Ge Yang, Guanzhe Huang, Gulin Yan, Haiyang Feng, Hao Nie, Haonan Jia, Hanpeng

- Hu, Hanqi Chen, Haolong Yan, Heng Wang, Hongcheng Guo, Huilin Xiong, Huixin Xiong, Jiahao Gong, Jianchang Wu, Jiaoren Wu, Jie Wu, Jie Yang, Jiashuai Liu, Jiashuo Li, Jingyang Zhang, Junjing Guo, Junzhe Lin, Kaixiang Li, Lei Liu, Lei Xia, Liang Zhao, Liguo Tan, Liwen Huang, Liying Shi, Ming Li, Mingliang Li, Muhua Cheng, Na Wang, Qiaohui Chen, Qinglin He, Qiuyan Liang, Quan Sun, Ran Sun, Rui Wang, Shaoliang Pang, Shiliang Yang, Sitong Liu, Siqi Liu, Shuli Gao, Tiancheng Cao, Tianyu Wang, Weipeng Ming, Wenqing He, Xu Zhao, Xuelin Zhang, Xianfang Zeng, Xiaojia Liu, Xuan Yang, Yaqi Dai, Yanbo Yu, Yang Li, Yineng Deng, Yingming Wang, Yilei Wang, Yuanwei Lu, Yu Chen, Yu Luo, Yuchu Luo, Yuhe Yin, Yuheng Feng, Yuxiang Yang, Zecheng Tang, Zekai Zhang, Zidong Yang, Binxing Jiao, Jiansheng Chen, Jing Li, Shuchang Zhou, Xiangyu Zhang, Xinhao Zhang, Yibo Zhu, Heung-Yeung Shum, and Daxin Jiang. Step-video-t2v technical report: The practice, challenges, and future of video foundation model, 2025. 1
- [25] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation, 2024. 2
- [26] MiniMax AI. Hailuo 02 model card. Technical report, MiniMax AI, 2025. 2, 5, 6
- [27] OpenAI. Sora 2 system card. Technical report, OpenAI,

2025. 1, 2, 5, 6

- [28] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, and et al. Bowen Shi. Movie gen: A cast of media foundation models, 2025. 1
- [29] Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. Capture: Evaluating spatial reasoning in vision language models via occluded object counting. In ICCV, 2025. 2
- [30] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos,

2024. 5

- [31] Runway Research. Introducing gen-3 alpha: A new generation of generative video models. https:// runwayml.com/research/introducing- gen3-alpha, 2024. 2
- [32] ShengShu Technology. Vidu q2 model release. Technical report, ShengShu Technology, 2025. 2, 5, 6
- [33] Gaurav Shinde, Anuradha Ravi, Emon Dey, Shadman Sakib, Milind Rampure, and Nirmalya Roy. A survey on efficient vision-language models, 2025. 2
- [34] Xichen Tan, Yuanjing Luo, Yunfan Ye, Fang Liu, and Zhiping Cai. Allvb: All-in-one long video understanding benchmark, 2025. 7
- [35] Tencent Hunyuan Video Team. Hunyuanvideo: A systematic framework for large video generation models. https:// github.com/Tencent-Hunyuan/HunyuanVideo,

2024. 2

- [36] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al.

- Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2
- [37] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning, 2025. 1
- [38] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain-of-thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 1

- [39] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022. 2
- [40] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903,

2022. 1

- [41] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 1, 2
- [42] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2025. 2
- [43] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Joshua B. Tenenbaum, and Antonio Torralba. Clevrer: Collision events for video representation and reasoning. In ICLR, 2020. 4
- [44] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation, 2025. 1
- [45] Xinyi Zhao et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. arXiv preprint arXiv:2503.22020, 2025. 2
- [46] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 2
- [47] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2022. 1
- [48] Jun Zhou and et al. Mlvu: Benchmarking multi-task long video understanding. In CVPR, 2025. 7

### 7. Additional Details for Each Task

##### 7.1. Structured Problem-Solving

This reasoning class targets systematic computational and logical reasoning abilities. The four tasks collectively assess models’ capacity to handle rule-driven problems that require step-by-step execution, numerical manipulation, and strategic thinking.

Arithmetic Operation. Models compute mathematical expressions presented visually, including addition, subtraction, multiplication, and division. Problems are categorized by difficulty: easy (addition/subtraction, range 1-15), medium (addition/subtraction/multiplication, range 1-50), and hard (all four operations, range 1-100). Each instance may contain 2 or 4 problems. All instances are programmatically generated with randomized numerical values and operator combinations, with visual representations using standard mathematical notation rendered in various fonts and sizes to ensure diversity. Fig. 9 illustrates an example of the generated video, showing the starting frame with the math problems (left), an intermediate frame (middle), and the final frame with all answers filled in (right). The evaluation prompt is: “Complete the math problems in the picture. Enter the final answer directly. Do not change anything else in the picture. Static camera, no zoom, no pan, no dolly.” We employ VLM-based evaluation where Gemini-2.5-Pro extracts all numerical results from the final frame. The prediction is marked as passed only if all problems and their answers are completely correct.

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

- Figure 9. An illustration of the arithmetic operation task, showing the starting frame (left), an intermediate frame (middle), and the final frame with completed answers (right).

Code Execution. Models simulate program execution by tracing through visualized Python code with given inputs to predict outputs. The task includes three difficulty levels: easy, medium, and hard, based on the complexity of the Python code problems sourced from LeetCode. Each instance contains only 1 problem. Code snippets are curated from classical algorithm problems, with inputs and expected outputs synthetically generated. Fig. 10 illustrates an example of the generated video, showing the starting frame with the code and input (left), an intermediate frame (middle), and the final frame with the output correctly filled in (right). The evaluation prompt is: “Complete this code problem. Enter the answer behind “# Output:”. Do not modify the existing code.” VLM-based evaluation extracts the code problem and output from the final frame. The prediction passes only if the extracted problem and result exactly match the ground truth.

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

- Figure 10. An illustration of the code execution task, showing the starting frame with code and input (left), an intermediate frame (middle), and the final frame with the output filled in (right).

Sudoku. Models complete partially filled 4 × 4 or 9 × 9 Sudoku grids, with difficulty levels determined by the number of pre-filled cells: easy (more givens), medium (moderate givens), hard (fewer givens requiring deeper logical deduction).

Puzzles are generated using constraint satisfaction algorithms that ensure unique solutions, and each puzzle is validated for solvability before inclusion with visual grids using clear cell boundaries and digit rendering. Fig. 11 illustrates an example of the generated video, showing the starting frame with the initial puzzle (left), an intermediate solving stage (middle), and the final frame with the complete solution (right). The evaluation prompt is: “Create a static, smooth, animation that solves the given [4×4 or 9×9] sudoku. Enter the missing numbers one by one. Do not change anything else in the picture. Only fill the numbers in the empty cells so the sudoku is solved properly. A cursor moves and fills the correct number in the empty boxes.” (The grid size in the prompt matches the actual puzzle size.) VLM-based evaluation extracts all cell values from the last frame, and a puzzle passes only if all cells are the same numbers as the ground truth.

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

- Figure 11. An illustration of the 4 × 4 Sudoku task, showing the starting frame with the initial puzzle (left), an intermediate solving frame (middle), and the final frame with the completed solution (right).

Tic-Tac-Toe. Models identify optimal moves in 3 × 3 Tic-Tac-Toe games across different scenarios: 2 ‘X’/‘O’, 3 ‘X’/‘O’, and 4 ‘X’/‘O’. Game states are programmatically generated to cover diverse tactical situations, with each state validated to have a clearly optimal move according to the minimax strategy. Fig. 12 illustrates an example of the generated video, showing the starting game state (left) and the final frame with the optimal move placed (right). The evaluation prompt is: “Given the current state of the Tic-Tac-Toe board in the image, predict the next move that results in a win for the player whose turn it is. Output the position of the winning move.” Grid-based evaluation partitions the board into individual cells and measures accuracy by comparing each predicted cell with its ground-truth counterpart; a prediction is marked as passed only when all cells match exactly.

[Figure 70]

[Figure 71]

- Figure 12. An illustration of the Tic-Tac-Toe task, showing the starting game state (left) and the final frame with the optimal move placed (right).

##### 7.2. Spatial Cognition

This category focuses on visual-spatial reasoning, testing how well models comprehend geometric properties, spatial transformations, and positional relationships in 2D environments.

Shape Fitting. Models receive a collection of geometric pieces and a target outline, then must determine the correct arrangement through rotation and positioning, with difficulty varying by shape complexity (simple triangles/rectangles vs. irregular polygons) and the number of pieces (2-5 pieces). The images were collected from the Internet or generated by image generation models. Fig. 13 illustrates an example of the generated video, showing the starting frame with separate pieces and empty holes (left), an intermediate fitting process (middle), and the final frame with all pieces correctly placed in their corresponding holes (right). The evaluation prompt is: “The scene shows three colored pieces, and a panel with three holes.

Each colored piece fits into one and only one hole. A hand grabs each colored piece and puts it into an empty hole that has the exact same shape - if it doesn’t fit, the hand tries another hole. All the objects must be placed in their respective holes.” VLM-based evaluation uses Gemini-2.5-Pro [7] to assess whether all pieces are correctly positioned. A trial is considered a pass only if the VLM confirms that all pieces are properly fitted without any shape alteration.

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

- Figure 13. An illustration of the shape fitting task, showing the starting frame with pieces and empty holes (left), an intermediate fitting process (middle), and the final frame with all pieces correctly fitted (right).

Visual Symmetry. Models complete incomplete symmetric patterns, with tasks categorized by symmetry type: reflective (mirror across vertical/horizontal/diagonal axes) and 180◦ rotational symmetry. For diagonal symmetry, patterns are generated on 8 × 8 grids by randomly coloring cells; for other symmetry types, 10 × 16 grids are used with the same random coloring process. Partial patterns are created by showing only half of the symmetric structure, requiring models to infer and complete the missing portion. Fig. 14 illustrates an example of the generated video, showing the starting frame with the incomplete pattern (left), an intermediate completion stage (middle), and the final frame with the fully symmetric pattern (right). The evaluation prompt is: “Instantly reflect this pattern [along the central, vertical axis or along the central, horizontal axis or along the main diagonal axis (top-left to bottom-right) or 180◦ along the central, vertical axis] while keeping the existing colored pattern without modification. Static camera perspective, no zoom or pan.” Grid-based evaluation divides the image into cells and compares cell-wise matching between predicted and ground truth patterns, and a prediction passes only if all cells match exactly.

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

- Figure 14. An illustration of the vertical symmetric in the visual symmetry task, showing the starting frame with an incomplete pattern (left), an intermediate completion stage (middle), and the final frame with the fully symmetric pattern (right).

Color Connection. Models link colored endpoints through non-intersecting paths on grids, with difficulty varying by the number of color pairs (2-4 pairs). Puzzles are generated with random positions for all circles. Fig. 15 illustrates an example of the generated video, showing the starting frame with colored endpoints (left), an intermediate path-drawing stage (middle), and the final frame with all color pairs correctly connected (right). The evaluation prompt is: “Draw three curves, one connecting each pair of circles of the same color. Do not change anything else in the picture. Static camera, no zoom, no pan, no dolly.” VLM-based evaluation assesses path connectivity and non-intersection, and the prediction passes if the VLM confirms all color pairs are correctly connected without crossings.

##### 7.3. Pattern-based Inference

This reasoning class examines inductive capabilities and abstract thinking, focusing on how models extract implicit rules from limited observations and generalize them to new situations.

[Figure 78]

[Figure 79]

[Figure 80]

- Figure 15. An illustration of the color connection task, showing the starting frame with colored endpoints (left), an intermediate drawing stage (middle), and the final frame with all paths correctly connected (right).

Sequence Completion. Models infer the next element in a visual sequence by recognizing how objects evolve across successive frames. Each sequence introduces variation in object count, shape, and color, requiring the model to detect consistent visual regularities and extend them to the next plausible state. Sequences are programmatically generated using rule-based transformations, with each sequence following a deterministic rule that governs element transitions, and 3-5 examples are provided before asking for a prediction. Fig. 16 shows one demonstration of this task. The evaluation prompt is: “Finish the incomplete frames of this shape-sequence puzzle. Do not change given patterns. Each row contains equalsized square cells outlined in black on white. Shapes are solid black arrows, dots, or stripes that obey a clear mathematical rule (rotation, translation, scaling, or toggling). Produce the missing panels so the sequence remains consistent and has a single logical solution.” Mask-based evaluation compares the predicted next element against ground truth using pixel-level accuracy within the target region, and a prediction passes if Acc > 0.90.

|[Figure 81]|
|---|

|[Figure 82]|
|---|

- Figure 16. An illustration of the sequence completion task, showing the starting frame with one blank cell (left) and the final frame with all completed cells (right).

Analogy Solving. Models solve visual analogies of the form “A:B::C:?”, testing different relationship types: spatial transformations (rotation, reflection, scaling), attribute changes (color, texture), and structural modifications (adding/removing components). Analogy problems are created by first defining a transformation, applying it to create the A→B pair, then providing a different C and asking models to apply the same transformation. Fig. 17 presents an example of this relationship, showing the A:B:C triplet and ground truth answer. The evaluation prompt is: “Create a smooth animation to generate the missing object in the lower right region and solve the visual analogy. The original three objects must remain still. Static shot, no zoom no pan no dolly.“ Mask-based evaluation measures pixel-wise similarity between predicted and ground truth answer images at the expected location for the transformed object, and a prediction passes if the mean MSE distance < 0.1.

Rule Following. Models infer transformation rules from example pairs and apply them to novel inputs, with rules varying in grid size and number of colored components in each grid. Rule sets are synthetically defined, with 2-4 demonstration pairs generated per rule, and test cases use different inputs but follow the same underlying transformation logic. Fig. 18 illustrates an example of the generated video, showing the starting frame with demonstration pairs and an empty test grid (left) and the final frame with the rule correctly applied to the test grid (right). The evaluation prompt is: “Modify the lower-right grid to adhere to the rule established by the other grids. You can fill cells, clear cells, or change a cell’s color in the lower-right

[Figure 83]

[Figure 84]

|[Figure 85]|
|---|

| |
|---|

- Figure 17. An illustration of the scaling relation in the analogy-solving task, showing the starting frame with the exemplar transformation in the top row (left), and the final frame with the corresponding transformation applied to the target object in the bottom row (right).

grid only. Don’t modify any of the other grids. Static scene, no zoom, no pan, no dolly.” Grid-based evaluation partitions the image into grids and verifies the cell-wise correctness of the bottom-right table to be predicted against the ground truth; a prediction passes only when all cells match perfectly (100% accuracy).

[Figure 86]

[Figure 87]

- Figure 18. An illustration of the rule following task, showing the starting frame with demonstration pairs and empty test grid (left) and the final frame with the rule correctly applied to the test grid (right).

##### 7.4. Physical Dynamics

This category assesses intuitive physics understanding, probing whether models can predict real-world physical behaviors based on visual observations of forces, materials, and environmental conditions.

Block Sliding. Models predict whether objects will slide down inclined planes, with scenarios varying by slope angle (15°60°), object shape (cube, sphere, cylinder), and surface properties (smooth, rough). Scenes are programmatically generated through controlled code specifications, and each scenario is verified to ensure a single, unambiguous outcome regarding whether sliding should occur under the defined conditions. Fig. 19 illustrates an example of the generated video, showing the starting frame with objects on the inclined surface (left), an intermediate sliding motion (middle), and the final frame showing the final positions of all objects (right). The evaluation prompt is: “Observe the objects on the inclined surface. Predict which objects will slide down and which will remain stationary. Show the final state after gravity acts.” Mask-based evaluation compares object positions in the final frame against ground truth, checking if each object’s position indicates sliding or staying, and a prediction passes if all object states are correctly predicted.

Communicating Vessels. Models predict liquid level redistribution in communicating vessels of diverse shapes and liquid colors. The simulations compute final states based on hydrostatic equilibrium to ensure physical accuracy. As shown in

- Fig. 20 illustrates an example of the generated video, showing the starting frame with initial liquid levels (left), an intermediate redistribution process (middle), and the final frame at equilibrium (right). The evaluation prompt, “A connected container (communicating vessel) with two vertical tubes filled with liquid. The liquid level in both tubes changes slowly and smoothly until it reaches equilibrium, where the levels remain still. The motion should appear smooth and continuous, with no splashing or turbulence”, guides the assessment. Performance is measured using a mask-based method that computes the height

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

###### Figure 19. An illustration of the block sliding task, showing the starting frame with objects on an inclined surface (left), an intermediate sliding motion (middle), and the final frame showing which objects slid down (right).

difference between predicted and ground truth levels in each vessel. A prediction is considered passed if the equilibrated liquid positions are level.

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

###### Figure 20. An illustration of the communicating vessels task, showing the starting frame with initial liquid levels (left), an intermediate redistribution stage (middle), and the final frame at equilibrium (right).

Temperature-Induced Deformation. Models predict material (ice) changes under temperature variations across scenarios, with temperature ranges varying from -20°C to 100°C. Material transformations are created using physics-based rendering and simulation, with each scenario specifying specific temperature and material properties to ensure deterministic outcomes.

- Fig. 21 illustrates an example of the generated video showing ice melting, with the starting frame showing solid ice at low temperature (left), an intermediate melting stage (middle), and the final frame showing the completely melted state (right). The evaluation prompt is: “Observe the material at the initial temperature. Predict how it will change (melt, expand, contract, deform) when the temperature changes to the target value. Show the final state.” VLM-based evaluation uses Gemini-2.5Pro [7] to assess whether the predicted material state matches expected physical behavior (e.g., ice should melt above 0°C, metal should expand when heated), and a prediction passes if the VLM confirms correct physical transformation.

[Figure 94]

[Figure 95]

[Figure 96]

###### Figure 21. An illustration of the temperature-induced deformation task using an ice melting example, showing the starting frame with solid ice (left), an intermediate melting stage (middle), and the final frame with completely melted ice/water (right).

Table 3. Complete pass@5 results for all models across all 13 tasks in V-ReasonBench. Tasks are grouped by their reasoning dimensions.

Dimension Task Seedance-1.0-Lite Vidu-Q2 Kling-2.5-Turbo-Pro Veo-3.1 Hailuo-02 Sora-2

Arithmetic Operation 0.00 56.00 0.00 42.00 90.00 100.00 Code Execution 0.00 0.00 2.22 11.11 31.11 53.33 Sudoku 0.00 0.00 0.00 0.00 18.00 50.00 Tic-Tac-Toe 3.33 3.33 26.67 63.33 46.67 90.00

Structured Problem-Solving

Shape Fitting 21.43 10.34 21.43 32.14 46.43 35.71 Visual Symmetry 0.00 3.33 3.33 26.67 10.00 26.67 Color Connection 0.00 0.00 20.00 10.00 60.00 60.00

Spatial Cognition

Sequence Completion 0.00 10.00 10.00 5.00 15.00 0.00 Analogy Solving 0.00 10.00 0.00 0.00 10.00 20.00 Rule Following 0.00 24.00 0.00 20.00 40.00 48.00

Pattern-based Inference

Block Sliding 20.00 20.00 10.00 20.00 10.00 20.00 Communicating Vessels 0.00 10.00 0.00 10.00 20.00 0.00 Temperature-Induced Deformation 80.00 90.00 60.00 90.00 100.00 60.00

Physical Dynamics

### 8. Complete Task-Level Results

Tab. 3 and Fig. 22 present the detailed pass@5 performance of all evaluated models across each of the 13 reasoning tasks in VReasonBench. This granular breakdown complements the dimension-level aggregation reported in the main paper, revealing task-specific strengths and weaknesses for each video generation model.

### 9. Limitation of VLM-Based Evaluation

Fig. 23, 24, 25 provide additional failure cases across multiple reasoning tasks. These examples highlight recurring issues in grid-structured and spatially dense scenes, where VLMs frequently misread fine-grained details such as cell boundaries, object adjacency, and subtle geometric relations.

### 10. More Reasoning Patterns Demonstrations

To provide a clearer view of the structural deviation patterns discussed in the paper, we include additional example predictions of Seedance-1.0-Lite and Vidu-Q2 from the Tic-Tac-Toe task. These cases further illustrate how models differ in their handling of minimalist scenes in the video reasoning process that requires strict structural fidelity.

Taken together, these two examples in Fig. 26 and 27 show a consistent pattern in how Seedance-1.0-Lite handles simple board-based reasoning tasks. In both cases, the model starts from a clean tic-tac-toe target board and generates a richer, more realistic scene: in the first example, the board appears in a cartoon setting with characters and logo-like icons; in the second, the tic-tac-toe layout is mapped onto a chessboard surrounded by chess pieces and tokens. The model roughly keeps the 3×3 grid region and the idea of a board game, but it replaces several cells with new symbols or removes parts of the board entirely.

These changes are harmless from a storytelling point of view, but they break the strict structural requirements of the task. The final frames no longer contain the exact X/O pattern specified by the ground truth, so the predictions fail under our grid-based evaluation. These two cases illustrate how the model’s preference for visually rich, realistic scenes can conflict with reasoning tasks that depend on precise symbolic and spatial accuracy.

Across these four tic-tac-toe examples, Seedance-1.0-Lite and Vidu-Q2 display the same tendency discussed in the main paper. When the input is a simple board on a plain background, the models transform it into a cartoon scene, a chessboard arrangement, or a heavily textured abstract image, instead of preserving the intended X/O pattern. Although they maintain a rough sense of a central board, the exact cell-level structure required for correct reasoning is lost.

This pattern is closely related to the models’ training data, which mostly consists of open-domain videos that emphasize rich visuals, motion, and diverse scenes, while containing very few clean, diagram-like examples. As a result, the models interpret minimal inputs as incomplete and replace them with visually complex content. However, video reasoning tasks require the opposite: models must treat simple scenes as complete and maintain precise spatial and symbolic structure. This indicates that training data and objectives for reasoning should place stronger weight on structure-preserving generation rather than visually complex outputs.

Arithmetic Operation

Temperature-Induced Deformation

Code Execution

Communicating Vessels

Sudoku

Tic-Tac-Toe

Block Sliding

Rule Following

Shape Fitting

Analogy Solving

Visual Symmetry

Sequence Completion Color Connection

Seedance-1.0-Lite

Kling-2.5-Turbo-Pro

Hailuo-02

Vidu-Q2

Veo-3.1

Sora-2

- Figure 22. Summary of V-ReasonBench performance across video models. The figure illustrates how six video generation models perform on 13 reasoning tasks, with scores rescaled within each dimension to enable direct comparison.

### 11. More Hallucination in Video Reasoning

[Figure 97]

[Figure 98]

[Figure 99]

Model Prediction

Ground Truth

Input

V-ReasonBench:

VLM: The prediction is incorrect because it is not a perfect reflection. For example, the blue square in the input at row 3, column 1 should be reflected to row 3, column 16, but this square is missing in the prediction

Score = 1.00 Pass = True

Evaluation

[Figure 100]

[Figure 101]

image.

- Figure 23. Example failure case from Visual Symmetry task. Evaluation prompt for VLM: “You are given three images: an input image, a ground-truth image, and a prediction image. The input image shows an original colored pattern. The correct transformation is: instantly reflect the entire pattern across its central vertical axis while preserving all existing colors and shapes exactly. No other modifications are allowed. Your task is to determine whether the prediction is correct. The prediction is correct only if two conditions are met: (1) The predicted pattern is a proper reflection of the input pattern across the central vertical axis, with no changes to colors, shapes, or structure other than mirroring. (2) The predicted pattern exactly matches the reflected pattern shown in the ground-truth image. This includes correct placement of every cell, color consistency, and no unintended alterations.”

Input

Model Prediction

Ground Truth

VLM: The prediction fails the second condition. While the predicted lower-right grid is correct, the prediction image incorrectly modifies the example grids in the top row, which should have remained identical to the input and ground-

truth images. The problem requires that only the lower-right grid changes.

[Figure 102]

[Figure 103]

V-ReasonBench:

Score = 1.00 Pass = True

Evaluation

[Figure 104]

[Figure 105]

[Figure 106]

- Figure 24. Example failure case from Rule Following task. Evaluation prompt for VLM: “You are given three images: an input image, a ground-truth image, and a prediction image. All images contain several example grids and a lower-right grid. Only the lower-right grid is supposed to change. Your task is to judge whether the prediction is correct. The prediction is correct only if two conditions are met:

(1) The predicted lower-right grid follows the same transformation rule demonstrated by the example grids in the input image. (2) The predicted lower-right grid visually and structurally matches the lower-right grid in the ground-truth image, including filled cells, empty cells, colors, and spatial arrangement. All other grids must remain identical across images.”

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

Model Prediction

Ground Truth

Input

VLM: The prediction is physically correct and visually matches the groundtruth image. The block correctly slides down the slope and comes to rest on the horizontal plane at the same final position shown in the ground-truth, which is consistent with the physical dynamics derived from the provided parameters.

V-ReasonBench:

Score = 0.83 Pass = False

Evaluation

[Figure 110]

[Figure 111]

- Figure 25. Example failure case from Block Sliding task. Evaluation prompt for VLM: “You are given three images: an input image, a ground-truth image, and a prediction image. The input image shows a rigid rectangular block resting at the top of an incline. The ground-truth image shows the final resting position of the block after sliding down the slope under ideal planar contact, given the annotated

parameters: slope length Lslope, slope angle θ, friction coefficients µslope and µplane, block length and height, and gravitational acceleration g = 9.81m/s. Air resistance, elastic collisions, and secondary impacts at the slope foot are ignored. Your task is to determine whether the prediction is physically correct. The prediction is correct only if the following conditions are satisfied: (1) The predicted final resting position of the block is consistent with the physical dynamics implied by the slope geometry and friction parameters. The block must appear at the location where it should come to rest after sliding, without penetrating surfaces or hovering. (2) The predicted final frame visually matches the ground-truth image in block position, orientation, and contact location on either the slope or the flat plane. Colors, shapes, and scene layout must remain unchanged except for the block’s final location. (3) No extraneous motion or unintended scene modifications are introduced.”

…… …… ……

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- Figure 26. Given an initial board (left), Seedance-1.0-Lite generates a stylized video in which the board is embedded in a cartoon scene with characters and additional logos overlaid on several cells (right). Although the overall 3×3 grid and most X/O placements are preserved, the added icons and altered symbols change the board configuration, causing the prediction to fail under strict grid-based evaluation.

…… …… ……

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- Figure 27. Starting from a simple target tic-tac-toe board (left), the model generates a chessboard scene in which the 3×3 grid is roughly aligned but many cells are filled with chess pieces and tokens instead of the required X and O marks. These changes alter the intended board configuration, so the prediction does not satisfy the 100% cell-wise match required by the grid-based evaluation.

…… …… ……

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- Figure 28. Starting from a simple initial board (left), the model generates a video in which the tic-tac-toe grid is placed on a colorful, highly textured background. Although the final frame roughly keeps the 3×3 grid and most X/O positions, some cells are partially overwritten by visual effects and altered marks.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

…… …… ……

- Figure 29. Starting from a defined initial board (left), the model generates frames in which strong abstract textures rapidly cover the scene. The tic-tac-toe grid and marks become barely visible and are eventually replaced by a large digit and background pattern.

