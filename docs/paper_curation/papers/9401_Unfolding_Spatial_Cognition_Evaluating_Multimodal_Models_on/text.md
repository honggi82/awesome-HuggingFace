arXiv:2506.04633v1[cs.CV]5Jun2025

# Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations

[Figure 1]

Linjie Li 1, Mahtab Bigverdi 1, Jiawei Gu 2, Zixian Ma 1, Yinuo Yang 1, Ziang Li 1, Yejin Choi 3, Ranjay Krishna 1

[Figure 2]

[Figure 3]

1 University of Washington 2 Sun Yat-sen University 3 Stanford University

[Figure 4]

[Figure 5]

[Figure 6]

Question Step-by-Step Human Mental Simulation

[Figure 7]

###### Yes!

Can the net be folded to form a cube, yes or no?

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

MLLM Textual Simulation

[Figure 13]

GPT-4o: Yes, this net can be folded into a cube. Here's how:

Visualization of Textual Reasoning:

- ① Face 1 becomes the top of the cube.
- ② Face 2 folds down to the left of Face 1.
- ③ Face 3 folds down to the right of Face 1.
- ④ Face 4 folds down below Face 2, becoming the front face.
- ⑤ Face 5 folds down below Face 3, becoming the back face.
- ⑥ Face 6 folds up to close the bottom of the cube.

[Figure 14]

[Figure 15]

[Figure 16]

① ② ③

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

……

Reasoning Error	Analysis: If Face 2 is on the left of Face 1 (②) and Face 3 is on the right (③), they would be opposite, but they are adjacent in the net. The correct third step should follow the net's layout to maintain their connection.

[Figure 23]

All faces will perfectly align , forming a six-faced cube with no overlaps or gaps.

- Figure 1: Visual simulations play a crucial role in real-world tasks, from assembling complex structures to interpreting mechanical diagrams and predicting spatial interactions. Different from how humans would approach a cube net folding problem, existing multimodal models rely heavily on textual simulation, which is not sufficient for reaching human-level spatial cognition. The above example shows how textual simulations of GPT-4o make obvious errors when we simulate the steps in 3D space.

## Abstract

Spatial cognition is essential for human intelligence, enabling problem-solving through visual simulations rather than solely relying on verbal reasoning. However, existing AI benchmarks primarily assess verbal reasoning, neglecting the complexities of non-verbal, multi-step visual simulation. We introduce STARE (Spatial Transformations and Reasoning Evaluation), a benchmark designed to rigorously evaluate multimodal large language models on tasks better solved through multi-step visual simulation. STARE features ∼4K tasks spanning foundational geometric transformations (2D and 3D), integrated spatial reasoning (cube net folding and tangram puzzles), and real-world spatial reasoning (perspective and temporal reasoning), reflecting practical cognitive challenges like object assembly, mechanical diagram interpretation, and everyday spatial navigation. Our evaluations show that models excel at reasoning over simpler 2D transformations, but perform close to random chance on more complex tasks like 3D cube net folding and tangram puzzles that require multi-step visual simulations. Humans achieve near-perfect accuracy but take considerable time (up to 28.9s) on complex tasks, significantly speeding up (down by 7.5 seconds on average) with intermediate visual simulations. In contrast, models exhibit inconsistent performance gains from visual simulations, improving on most tasks but declining in specific cases like tangram puzzles (GPT-4o, o1) and cube net folding (Claude3.5, Gemini-2.0 Flash), indicating that models may not know how to effectively leverage intermediate visual information.1

1STARE is available at https://github.com/STARE-bench/STARE

Preprint.

## 1 Introduction

Spatial reasoning is not merely a subset of human cognitive abilities but rather the fundamental underpinnings of intellectual processes [1]. Reasoning with space enables individuals to solve complex tasks through visually simulating transformations of objects in the mind, anticipating how their actions would physically manipulate other artifacts. Cognitive psychologists have found ample evidence that humans simulate 2D and 3D transformations to reason about spatial problems [2–5]. Shepard and Metzler [6] found that the time taken by a subject to recognize two perspective drawings as the same 3D shape increases linearly with their angular difference in orientation, suggesting an analog mental rotation process. Hegarty [7] found that humans employ mental animation, incrementally simulating the movement of parts to understand mechanical diagrams. Such abilities enable everyday tasks like assembling furniture, reading maps or instructional diagrams, navigating new environments, and are strongly correlated with success in STEM disciplines [8–10].

Despite their prevalence in real-world applications—from arranging furniture in a house to molecular docking for drug discovery—dynamic visual simulations are still under-represented when evaluating multimodal large language models (MLLMs). Existing datasets largely target static recognition or problems that can be re-phrased as linguistic reasoning [11–16]. In contrast, humans frequently solve spatial challenges—such as folding a 2D net into a 3D object, assembling a tangram, or taking another visual perspective—by running internal, step-wise visual simulations (Figure 1), which have a long pedigree in the cognitive science studying human spatial reasoning [17–21].

To bridge this gap, we introduce STARE (Spatial Transformations and Reasoning Evaluation), a benchmark focused on spatial reasoning tasks that can be better solved through multi-step visual simulations. STARE evaluates whether MLLMs can perform complex visual reasoning akin to the visual simulations humans perform. It spans a spectrum of spatial cognition challenges (Figure 2), structured in increasing complexity:

- • Foundational geometric transformations: Tasks involving basic planar (2D) and volumetric (3D) transformations, such as rotations, translations, and reflections.
- • Integrated spatial reasoning: Cube net folding, requiring understanding how 2D patterns fold into 3D objects, and tangram puzzles, assessing sequential assembly and spatial positioning.
- • Real-world spatial reasoning: Tasks demanding reasoning about perspective changes and temporal frame sequences, simulating realistic spatial cognition scenarios encountered in daily life.

In the first two categories, each transformation or operation (e.g., folding a face) can be explicitly visualized step by step, and indeed humans often draw or imagine intermediate states when solving them. The last category demands higher-level visual simulation skills without always having clear intermediate visual cues (e.g., perspective reasoning) [22, 23]. We carefully curate ∼4K total instances across these categories, controlling difficulty via distractor similarity and number of simulation steps, to push models beyond superficial pattern-matching.

Our experiments show that models find reasoning over simple 2D transformations relatively easy but struggle with 3D cube net folding and tangram puzzles, performing near random chance due to the need for multi-step simulations. Humans, though nearly perfect in accuracy, took significantly longer—up to 28.9 seconds—to solve some tasks but sped up considerably (down by 7.5 seconds on average) when given intermediate steps. Meanwhile, when models receive intermediate visual steps, their performance varies: e.g., GPT-4o, Gemini-2.0 Flash Thinking and o1 improve while Gemini-2.0 Flash and Claude worsen on cube net folding, suggesting that not all models effectively utilize visual guidance. In general, models lag behind human performance significantly. To better understand these gaps, we conduct detailed error analyses, pinpointing specific reasons for model failures, such as difficulties in accurately interpreting 3D spatial relationships, inadequate of “imagining in space”, and struggles with extended visual contexts even when providing explicit visual simulations. Fundamentally, models cannot effectively perform visual simulation.

Overall, STARE aims to comprehensively test MLLMs’ ability to perform sequential visual simulations as opposed to pure textual reasoning. By evaluating models on tasks grounded in cognitive phenomena like mental imagery, we aim to reveal whether current MLLMs can approach the flexible spatial problem-solving of humans.

“Implicit” Simulation Real-world Spatial Reasoning

[Figure 24]

[Figure 25]

###### Perspective Reasoning

Temporal Frame Reasoning

[Figure 26]

The image is from the top-down view of a scene with a red square indicating an agent and a red arrow indicating the agent's direction of view. Which option best represents the agent's view?

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Which image correctly fills the missing frame?

[Figure 31]

A B C D

A B C

[Figure 32]

Integrated Spatial Reasoning

[Figure 33]

[Figure 34]

[Figure 35]

Cube Net Folding

###### Tangram Puzzle

[Figure 36]

[Figure 37]

Check out a net with 6 faces ：

Can the Tangram puzzle be completed with the available pieces, yes or no?

A B C D

[Figure 38]

Can the net be folded to form a cube, yes or no?

Foundational Geometric Transformations

[Figure 39]

[Figure 40]

2D Transformation 3D Transformation

Make it smaller then move it left and slightly upwards. Move it up and then make it bigger.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A B C D

A B C D

“Explicit” Simulation

- Figure 2: Overview of STARE. STARE consists of 3 levels of tasks, 2D Transformation and 3D Transformation for foundational spatial reasoning skills, tangram puzzle and cube net folding for integrated spatial reasoning, temporal frame inference and perspective reasoning to mimic real-world scenarios. The intermediate steps for completing tasks in the first two levels can be explicitly simulated, while the more real-word spatial reasoning tasks requires more abstract and implict mental simulations.

## 2 The STARE benchmark

STARE is designed to evaluate multimodal models’ abilities in spatial cognition and visual reasoning, focusing specifically on tasks that humans solve non-linguistically, through visual simulation. Current perception-focused multimodal benchmarks still rely heavily on linguistic reasoning [24– 26] or static visual recognition [27–29], failing to measure models’ abilities in sequential visual problem-solving. Parallel work in spatial cognition [30, 12, 31, 16, 32] probes analogy making and pattern induction, yet simulation is optional and intermediate visual states are seldom provided because of annotation cost. VSI-Bench [33] underscores the role of mental imagery in spatial reasoning, but focuses on spatial memory and estimation from video rather than explicit step-by-step simulation. STARE closes the gap by testing multimodal models across diverse spatial tasks that require step-by-step visual simulations with or without explicit linguistic guidance. We describe the overall design of STARE (§2.1), highlighting key differences compared to existing benchmarks. We then provide detailed descriptions of each task, discussing how the data was curated (§2.2).

###### 2.1 Overview of STARE

STARE is structured to comprehensively cover spatial reasoning at multiple complexity levels, from basic geometric transformations (2D and 3D) to more integrated tasks (cube net folding and tangram puzzles) and real-world spatial reasoning scenarios (temporal frame and perspective reasoning). Each task is presented as a multiple-choice or yes/no question using carefully designed visual and textual prompts. In total, the dataset contains ∼4K instances across different evaluation setups (Figure 3). Detailed statistics of STARE are provided in Appendix Figure 7.

STARE separates tasks that can be visually simulated, i.e., where each transformation step is visually observable, from tasks demanding more abstract and implicit mental simulations, such as perspective reasoning. To support more fine-grained evaluation, we synthesize the tasks that humans can mentally picture or even explicitly draw the intermediate steps, including 2D transformations, 3D transformations, cube net folding and tangram puzzle. Additionally, STARE tasks are intentionally crafted to closely reflect real-world scenarios such as assembling objects (e.g., tangram puzzles), interpreting mechanical diagrams (e.g., cube net folding) and navigating environments (e.g., perspective reasoning). These scenarios can potentially shed lights on models’ abilities in practical, everyday spatial cognition, providing meaningful assessments aligned with common human chal-

Question

Evaluation Settings

[Figure 47]

Check out a Tangram puzzle below. The left panel is an empty Tangram puzzle, while the right panel shows available pieces to complete the puzzle.

###### Without Visual Simulations

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Question Question Steps

Steps

Below are the steps to complete the Tangram puzzle:

- Step 1: Rotate piece B by about 90 degrees clockwise, and place piece B with its upper-left corner at (x, y) = (1, 3).
- Step 2: Place piece A with its upper-left corner at (x, y) = (0, 3).
- Step 3: Rotate piece C by about 90 degrees clockwise, and place piece C with its upper-left corner at (x, y) = (0, 1). Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?

Intermediate Visual Simulations

With Visual Simulations

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Question Steps

Intermediate Visual Simulations

- Figure 3: The different variants in the Tangram Puzzle task. We provide visualizations of the complete interleaved inputs for all three types in Appendix F.2.

lenges. A detailed discussion about related works in human visual reasoning and MLLM benchmarks are provided in Appendix D.

###### 2.2 Data curation

- 2D transformations: We design two types of tasks assessing spatial reasoning through twodimensional shape transformations: visual analogy, and instruction-based tasks. In visual analogy tasks, a shape A is shown to transform visually into shape A′, after which a shape B is provided with candidate shapes for applying the same transformation sequence to B. Instruction-based tasks explicitly describe transformations (e.g., “Rotate 90 degrees clockwise, then make it bigger”) and require selecting the correctly transformed shape from 4 answer choices. Transformations include rotations, translations, uniform scaling, reflection and shearing, with clearly defined parameters. Each task is created with three difficulty levels: easy (with two distractors out of three clearly different in appearance), medium (one obvious distractor), and hard (all distractors visually similar, forcing the model to pay attention to the transformation itself). In addition, we synthesize samples with 1/2/3 transformation steps to facilitate evaluations in multi-turn visual transformations. We programmatically generate all shapes and their transformed version using Matplotlib [34]. Visualization of different variants of 2D transformation samples is shown in Figure 8 of the Appendix.

We develop two experimental setups: (1) question + transformation steps, where the transformation steps are shown either verbally (for instruction-based tasks) or visually (for visual analogy tasks); and (2) question + transformation steps + intermediate visual simulations, showing all intermediate visualizations of shape B, excluding the final step. We synthesize a total of ∼1000 instances, ∼600 of which are without intermediate visual simulations.

- 3D transformations: We extend the 2D transformation tasks to three dimensions, creating similar tasks using 3D shapes. Reflection is omitted in 3D because the mirror plane isn’t obviously recognizable to human evaluators. The transformations include rotations around arbitrary axes, translations in 3D space, scaling, and shearing. Tasks, difficulty levels, and experimental setups mirror those of the 2D tasks, with a total of ∼1000 instances. Following [35], we create abstract 3D shapes as detailed meshes and use Blender [36] to render realistic and consistent visuals.

Tangram puzzles: Tangram puzzles test spatial reasoning about how individual pieces fit together to form a complete shape. Each puzzle provides a target grid and pieces, and the task is to determine whether the pieces can exactly fill the grid. Valid puzzles were generated by randomly dividing small grids (3x3 or 4x4) into rectangular or square shapes, then randomly rotated. Irregular variants were also created by merging adjacent rectangles. Invalid puzzles were constructed by adding or removing pieces, altering piece sizes, or giving incorrect placement instructions.

We create three setups for evaluation: (1) question-only, which presents the initial puzzle configuration with a query about solvability; (2) question + assembly steps, adding descriptive instructions of each assembly step without visual aids; and (3) question + assembly steps + intermediate visual simulations, providing both descriptive annotations and intermediate visualizations of the assembly

process, excluding the final visualization indicating success or failure. This task comprises ∼800 puzzles, evenly divided into solvable and unsolvable instances.

Cube net folding: This task evaluates the model’s capacity to mentally fold flat 2D patterns into 3D cubes. We provide examples comprising both valid nets (correctly folding into a cube) and invalid nets (leading to overlapping or disconnected faces). Each cube net has explicitly labeled faces. To generate these examples, we implement a step-by-step algorithm that simulates the folding process by designating a stationary base face and sequentially folding the connected faces. During each folding step, we detect and annotate errors, such as overlaps or disconnected faces, and generate corresponding visualizations using Matplotlib, clearly delineating face boundaries. Similar to tangram puzzles, we evaluate models in three setups, including (1) question-only, (2) question + folding steps, and (3) question + folding steps + intermediate visual simulations. The final cube net folding task contains ∼320 samples, balanced between valid and invalid configurations.

Temporal frame reasoning: This task evaluates a model’s ability to infer missing sequential visual information. Each example consists of four consecutive frames from a video, with one frame hidden. The model must identify the missing frame from a set of three options, relying on temporal consistency and logical scene progression.

We construct 471 examples from the Objectron [37] dataset, which contains short, object-centric videos with camera pose annotations. To create meaningful sequences, we extract the longest continuous segment where the camera moves only in one direction (left or right), divide it into four equal intervals, and select a frame from the central portion of each interval. One of these frames is hidden, and the model must identify it from three choices: the correct missing frame and two distractor frames sampled from different, non-overlapping parts of the video.

Perspective reasoning: This task assesses a model’s ability to understand how scenes appear from different viewpoints. Each example consists of a top-down map that indicates an agent’s position and orientation, represented by an arrow showing the agent’s viewing direction. The model must then select the correct first-person view from four choices, emphasizing spatial perspective reasoning and spatial relationships in various indoor environments.

We construct 250 samples using the HM3D dataset [38], a large collection of 3D indoor spaces derived from real-world environments. To generate each example, we use the Habitat simulator [39– 41] to place an agent at a random position on the floor while ensuring the surrounding scene contains enough visual cues, such as objects and structures, rather than just walls. A top-down view of the agent’s position is then captured, and a random viewing direction is assigned (forward, right, left, or backward). The four answer choices correspond to these fixed 90-degree viewpoints, ensuring clear distinctions between them. To improve dataset quality, we conduct human filtering to remove ambiguous cases and low-resolution images.

## 3 Experiments

In this section, we describe our experimental setup in detail, present comprehensive results, and provide an in-depth analysis of common model errors and limitations.

###### 3.1 Experimental Setup

For synthetic tasks involving explicit simulations (2D transformations, 3D transformations, cube net folding, tangram puzzles), we explore two evaluation settings:

- • Without Visual Simulations: Models receive only an initial image with or without step-by-step textual instructions and had to mentally infer the subsequent transformations without visual guidance, thereby testing their internal mental simulation capabilities.
- • With Visual Simulations: Models were provided with step-by-step visualizations clearly illustrating each transformation step before the final result, enabling explicit visual reasoning. Instead of collating the complex step-by-step visualizations into a single image, we provide the model with interleaved image and text query for evaluation.

For real-world reasoning tasks, including temporal frame and perspective reasoning, we evaluate models under the standard single image setting without providing explicit intermediate visual steps.

2D Trans. 3D Trans. Cube Net Tangram Temp- Pers-

Model

Overall

✗VSim ✓VSim ✗VSim ✓VSim ✗VSim ✓VSim ✗VSim ✓VSim oral pective Random 25.0 25.0 25.0 25.0 50.5 50.5 50.5 49.1 33.3 25.0 34.8 Closed-source Models

GPT-4o 71.2 82.7 (↑ 11.5) 65.5 68.4 (↑ 2.9) 50.3 52.2 (↑ 1.9) 52.5 51.5 (↓ 1.0) 39.0 38.7 53.9 Claude-3.5 Sonnet 65.9 71.4 (↑ 5.5) 51.5 57.8 (↑ 6.3) 52.3 51.6 (↓ 0.7) 59.0 67.6 (↑ 8.6) 54.0 26.1 53.1 Gemini-2.0 Flash 69.5 75.2 (↑ 5.7) 56.1 59.3 (↑ 1.6) 37.7 35.6 (↓ 2.1) 65.0 65.5 (↑ 0.5) 38.6 37.2 51.3 Gemini-2.0 Flash Think 60.6 62.8 (↑ 2.2) 49.5 56.1 (↑ 6.6) 48.3 50.7 (↑ 2.4) 39.8 62.8 (↑ 23.0) 45.0 32.7 48.8 o1 81.8 87.7 (↑ 5.9) 67.9 71.6 (↑ 3.7) 51.3 53.4 (↑ 2.1) 55.3 53.2 (↓ 2.1) 45.0 36.8 57.2

Open-source Models

LLaVA-OneVision-72B 32.9 32.2 (↓ 0.7) 27.0 30.6 (↑ 3.6) 28.5 34.2 (↑ 3.7) 30.3 39.8 (↑ 9.5) 35.7 24.8 31.4 InternVL2.5-78B 47.5 50.1 (↑ 2.6) 38.1 36.5 (↓ 1.6) 37.1 37.3 (↑ 0.2) 60.7 48.2 (↓ 12.5) 31.4 26.0 39.2 Qwen2.5-VL-3B 16.6 20.0 (↑ 3.4) 29.1 31.4 (↑ 2.3) 43.5 41.0 (↓ 2.5) 50.1 42.7 (↓ 7.4) 33.3 23.3 32.3 Qwen2.5-VL-7B 35.4 32.4 (↓ 3.0) 28.8 31.7 (↑ 2.9) 40.7 44.9 (↑ 4.2) 54.5 52.9 (↓ 1.6) 36.5 23.2 36.7 Qwen2.5-VL-72B 45.2 48.5 (↑ 3.2) 43.0 49.1 (↑ 6.1) 35.2 53.4 (↑ 18.2) 61.2 56.9 (↓ 4.3) 31.4 26.0 42.3

Human Performance

Accuracy 95.0 97.0 (↑ 2.0) 96.0 97.5 (↑ 1.5) 99.0 99.0 ( - ) 87.5 94.0 (↑ 6.5) 98.1 98.4 96.5 Response Time (s) 17.1 11.1 (↓ 6.0) 14.4 11.8 (↓ 2.6) 15.4 6.0 (↓ 9.4) 28.9 17.1 (↓ 11.8) 10.8 23.9 15.9 ∆(Best Model, Human) -13.2 -9.3 -28.1 -25.9 -46.7 -45.6 -22.5 -26.4 -44.1 -59.7 -39.3

- Table 1: Model Performance With or Without Visual Simulation (VSim) Across Tasks in STARE. Even the top performer, o1, achieves just under 60% accuracy. Humans, in contrast, get near perfect scores. Green (Red) arrows indicate performance improvements (degradations) with visual simulation.

Evaluation Metrics. We report accuracy for multiple-choice questions in 2D/3D transformations, temporal frame, and perspective reasoning tasks. For cube net folding and tangram puzzles, which involve binary yes/no questions, we report the F1 score. We report macro-average performance across tasks as the overall evaluation metric.

Models. We consider the following models: (1) Closed-source models: GPT-4o [42], Claude-3.5 Sonnet [43], Gemini2.0 Flash [44], and the reasoning-focused Gemini2.0 Flash Thinking [45] and o1 [46]. (2) Open-source models: InternVL2.5-78B [47], LLaVA-OneVision-72B [48], Qwen2.5VL-3B, Qwen2.5-VL-7B, and Qwen2.5-VL-72B [49].

Additionally, we invite two undergraduate students to complete the same tasks as the models. The averaged performance and response time are recorded to benchmark model capabilities against human-level spatial reasoning.

###### 3.2 Main Results

The results present in Table 1 show notable variations in model performance across different spatial reasoning tasks in the STARE benchmark. Models achieve the highest accuracy (up to 87.7%) on simpler 2D transformation tasks, significantly surpassing random chance (25%). Accuracy decreases by roughly 5% on average for more complex 3D transformations. Tasks involving intricate multi-step reasoning, such as cube net folding and tangram puzzles, resulted in even worse model performance, closer to random chance (∼50%). Additionally, temporal frame reasoning and perspective reasoning, which require interpreting sequential visual contexts and viewpoint changes, posed considerable difficulties, with most models performing similarly to random chance.

The use of visual simulations (VisSim) enhances model performance in most cases, but not all. GPT-4o exhibits a notable improvement of 11.5% accuracy on 2D transformations with visual simulations, and Claude-3.5 Sonnet shows significant gains (+8.6%) on tangram puzzles. However, visual simulations did not uniformly benefit model performance; certain models like Gemini-2.0 Flash experienced slight performance declines (e.g., a 2.1% decrease on F1 for cube net tasks), indicating that models can not always effectively leverage intermediate visual information. The reasoning-focused o1 model outperforms all other models with visual simulations, except for the tangram puzzles. Overall, it improves over GPT-4o by 3.3% on average, but still lag behind human performance. Despite the large improvement observed for Gemini-2.0 Flash Thinking from adding visual simulation on tangram puzzles (+23.0%), it notably underperforms its non-reasoning counterpart (Gemini-2.0 Flash) across tasks like 2D/3D transformations, tangram puzzles, and perspective reasoning with or without visual simulations, suggesting that optimizing for linguistic reasoning does not always benefit spatial reasoning.

Open-source models generally exhibite lower accuracy compared to closed-source counterparts, highlighting a significant performance gap. Larger models like InternVL2.5-78B and Qwen2.5-VL72B performe relatively better, suggesting benefits from scale, but their results with visual simulations also varied. For instance, InternVL2.5-78B’s performance decreases significantly in tangram tasks (-12.5%), whereas Qwen2.5-VL-72B improves notably (18.2%) in cube net tasks.

Human performance consistently surpasses that of models, achieving high accuracy across all STARE tasks, and further improved by intermediate visual simulations. However, these tasks were cognitively demanding even for humans, reflected by relatively long response times (e.g., 28.9 seconds on tangram puzzles without visual simulations). Although intermediate visual simulations significantly reduces cognitive load and response time, humans still require more than 5 seconds to mentally manipulate and reason through these problems and complete the last step. Thus, STARE tasks clearly involve complex, multi-step spatial reasoning beyond simple recognition tasks solvable at a glance [29]. These findings underscore humans’ superior spatial reasoning capabilities, particularly when aided by visual simulations.

Moreover, to study whether gains on abstract, synthetic spatial tasks translate to real-world tasks, we computed model-level correlations between the two domains. Concretely, for each model, we average its performance across with or without visual simulation on the 4 synthetic tasks and contrast that with its mean accuracy on the two real-world tasks. This yields a strong overall Pearson correlation (r ≈ 0.88, p ≈ 5e−4) across all 11 models. Counting in human performance, further increase the correlation to (r ≈ 0.97, p ≈ 1e−7).

###### 3.3 Detailed Analysis

To gain deeper insights into model limitations and identify specific reasoning challenges, we structure our detailed analysis around several targeted questions. We focus our discussion below on the GPT-4o model, given that it achieves the best performance among the non-thinking models. Analysis on all the other models can be found in Appendix G.

- Q1: How well do models understand individual transformation types in 2D and 3D? We evaluate model accuracy on individual transformation operations—rotation, translation, scaling, reflection, and shearing—for both 2D and 3D tasks, comparing performance with and without visual simulation (Figure 4). For 2D tasks, scaling achieves the highest accuracy (approximately 90% without visual simulation), improving further with visual simulation. Shearing was the most challenging in

- 2D (around 54%), showing minimal improvement from visual aids. Reflection, rotation, and translation significantly benefits from visual simulation, improving roughly 10 percentage points each. In 3D tasks, translation had the highest accuracy (about 76% without visual simulation), although it slightly declines with visual simulation. However, shearing, scaling, and rotation notably improve with visual simulation by about 3–8 percentage points. Overall, visual simulation substantially enhances performance for complex transformations, especially in 2D, though the added complexity of
- 3D transformations continues to present significant challenges.

- Q2: How does model accuracy change as task complexity increases? (1) Performance vs. Difficulty-level: The left sub-figure in Figure 6 shows model accuracy decreased as tasks became harder. For 2D tasks, GPT-4o performed best on easy tasks (∼86% with visual simulation), with accuracy declining notably for medium and hard tasks, especially without visual simulation (dropping to ∼66% for hard tasks). For 3D tasks, overall accuracy was lower, decreasing from easy tasks (∼72% without visual simulation) to hard tasks (∼60% without). Visual simulation generally improved accuracy but was less effective or even slightly detrimental for the hardest 3D tasks (60.5% without, 57.4% with). (2) Performance vs. Number of Turns: The right sub-figure in Figure 6 shows that how model performance varies with the number of transformation steps (N = 1, 2, 3). Without visual simulation, accuracy for both 2D and 3D tasks initially increases from N = 1 to N

= 2, and then decreases at N = 3. The observed peak at N = 2 likely occurs because two-step transformations combine simpler transformations (e.g., scaling) with more challenging ones (e.g., shearing), allowing models to leverage the simpler transformations to determine the correct answer. In contrast, one-step transformations are evenly distributed across all transformation types, while at N = 3, the increased complexity from multiple transformations compounds cognitive demands, reducing overall model accuracy. With visual simulation, accuracy remains consistently high across 2 and 3 steps in 2D tasks and shows stable or slightly improved performance at N = 3 in 3D tasks. Performance at N = 1 with visual simulation is not shown because there is no intermediate step.

[Figure 58]

[Figure 59]

###### Perception Error

[Figure 60]

[Figure 61]

|Question: Check out a net with 6 square faces. Can the net be folded to form a cube, yes or no?|
|---|

Claude: Looking at the image, I see a net with 6 labeled squares arranged in a specific pattern:

Error	Analysis:

- • Square 1 (yellow) is on the left
- • Square 2 (purple) is at the top
- • Square 3 (pinkish) is in the middle right of Square 2
- • Square 4 (blue) is below Square 3 and to the right of Square 1
- • Square 5 (orange) is to the right of Square 4
- • Square 6 (green) is below Square 4 …… So the answer is no, this specific net cannot be folded to form a cube.

Square 6 (green) is below Square 5.

[Figure 62]

[Figure 63]

[Figure 64]

- Figure 4: GPT-4o performance on individual 2D/3D transformation types, with and without Visual Simulation (VisSim).

Figure 5: A perception error from Claude-3.5 Sonnet. Refer to Appendix F.5 for more case study.

[Figure 65]

Figure 6: GPT-4o performance vs. task complexity (left: difficulty levels and right: number of transformation steps) with or without Visual Simulation (VSim).

- Q3: Do model failures originate from basic visual perception errors? To determine if model failures originate from fundamental visual perception rather than higher-level reasoning limitations, we design a straightforward probing experiment. Specifically, we simplify the task by directly presenting the model with the final, fully simulated outcomes, reducing the problem to visually matching these outcomes to the correct candidate answers. Under these conditions, accuracy improves by 4.2% (from 82.7% to 86.9%) on 2D transformations and 2.8% (from 68.4% to 71.2%) on 3D transformations, indicating only a modest improvement when eliminating intermediate steps. However, for more structured tasks like cube net folding and tangram puzzles, providing the fully completed final form drastically raises accuracy to 100% and 91.6%, respectively, highlighting that models can solve these tasks when the perceptual complexity is minimized. To further isolate the nature of perceptual errors in cube net folding, we create targeted tasks to test both 2D perception (color recognition and face connectivity) and 3D perception (identifying if a face has been folded). Results from these tasks (Table 2) reveal perfect color recognition but a notable decrease in accuracy for face connectivity (94.1%) and particularly low accuracy in correctly identifying folded faces (57.4%). Figure 5 illustrates an example of perception error on connectivity misalignments from Claude-3.5 Sonnet. Moreover, these specific perceptual errors in folding explain the limited benefits from visual simulations observed in Table 1 for GPT-4o. Overall, while some errors indeed stem from basic visual perception deficits, particularly in more complex 3D scenarios, the results suggest higher-level reasoning likely plays a larger role in overall model failures.
- Q4: How well do models reason spatially in text? To evaluate how well models reason spatially from text alone, we translate each visual task into clear, concise descriptions. For 2D and 3D transformation tasks, each object is described by stating its shape, color, position, size and etc.—for instance, “a red square at position (3,4) with size 2”. In the cube-net folding task, the unfolded cube is represented by numbering each face and arranging these numbers in a grid matching the cube net’s visual layout. For example, “123456" represents all six faces in a single row. Lastly, for the tangram puzzle task, each piece is labeled (e.g., “Piece A”) and represented by a compact grid indicating occupied cells marked by “1”. For instance, a square piece might be shown as two rows of “11”. Examples of text representations of each task are provided in Appendix F.4.

###### Model 2D Perception 3D Perception

Color Connectivity Folded? GPT-4o 100.0 94.1 57.4

- Table 2: 2D and 3D perception accuracy in cube-net folding.

Input 2D Trans. 3D Trans. Cube Nets Tangram

Text-only 87.5 64.7 57.0 72.6 Image-only 75.1 67.7 56.0 62.5 Image+Text 90.8 70.0 62.1 –

Table 3: GPT-4o performance without visual simulation under different input representations.

Input Cube Nets Tangram

Question-only 50.2 62.4 Question+Steps 50.4 34.7

Table 4: GPT-4o performance with question-only vs. explicit reasoning steps.

Simulation State 2D Trans. 3D Trans. Cube Nets Tangram

Partial 86.8 72.1 51.3 43.5 All 82.7 68.4 52.2 51.5 Last 89.4 68.4 35.2 43.4

Table 5: GPT-4o performance with different intermediate visualsimulation states.

As shown in Table 3, providing the model with a text representation removes much of the perception challenge, yet accuracy remains well below human performance—about 57% on cube-net folding, 65% on 3D transformations, and roughly 73% on tangram puzzles, suggesting that the model still lacks the ability to mentally simulate the steps to solve each task. Text helps most on 2D spatial reasoning: accuracy on 2D transformations rises from 75% with images alone to 87% with text, and tangram performance climb from 63% to 73%. For tasks involving 3D spatial reasoning, however, text gives little benefit, partly because the simple text description about shape, color, material, center, and size, cannot capture all the depth and adjacency cues in 3D spatial reasoning.

- Q5: How well do models verbally simulate without visual simulation? We evaluate how effectively models verbally simulate spatial reasoning without intermediate visual simulations by comparing performance when provided only the question (Question-only) versus explicit verbal reasoning steps (Question+Steps). Table 4 shows minimal improvement in cube net folding (50.2% to 50.4%), indicating limited benefit from verbal reasoning alone. Conversely, tangram performance notably decreases (62.4% to 34.7%), suggesting models adopt shortcuts like summing piece areas rather than genuine spatial simulation. This result partially reflects a bias in our question-only set: models can achieve ∼75% accuracy by checking the total areas of available pieces.
- Q6: How well do models integrate textual context with isolated visual simulations? We compared accuracy when presenting models with complete visual sequences versus only the final or most relevant visual state (Table 5). Easier tasks like 2D and 3D transformations showed improved or comparable accuracy when presented only the final state (e.g., 82.7% for complete vs. 89.4% for last), suggesting that for these tasks, the final visual state closely resembles the initial state, reducing cognitive load. However, in complex tasks such as cube net folding (52.2% complete vs. 35.2% last) and tangram puzzles (51.5% complete vs. 43.4% last), the final state becomes more disconnected from the initial configuration, requiring deeper understanding of preceding verbal steps. This disconnection introduces significant challenges for models, aligning with earlier findings (Q4) and underscoring their difficulties in integrating complex visual sequences during multi-step reasoning.

## 4 Conclusion

In this paper, we introduced STARE, a novel benchmark specifically designed to evaluate multimodal models on diverse spatial cognition tasks involving complex visual reasoning and mental simulations. STARE uniquely assesses model capabilities across foundational geometric transformations, integrated spatial reasoning tasks, and real-world scenarios requiring temporal and perspective reasoning. Our extensive experiments reveal significant performance variations among multimodal models, highlighting substantial challenges, especially in complex, multi-step reasoning scenarios. Visual simulations notably enhance performance on simpler tasks but yield mixed results for more sophisticated tasks. The substantial gap in performance between closed-source and open-source models further emphasizes the necessity for advancements in multimodal reasoning. Overall, STARE sets a critical benchmark to guide future research towards human-level spatial reasoning capabilities in AI.

## References

- [1] Barbara Tversky and Masaki Suwa. Thinking with sketches. 2009. 1
- [2] Alex Mitko and Jason Fischer. When it all falls down: The relationship between intuitive physics and spatial cognition. Cognitive research: principles and implications, 5:1–13, 2020. 1
- [3] Jiafei Duan, Samson Yu, Soujanya Poria, Bihan Wen, and Cheston Tan. Pip: Physical interaction prediction via mental simulation with span selection. In European Conference on Computer Vision, pages 405–421. Springer, 2022.
- [4] Jonathan Wai, David Lubinski, and Camilla P Benbow. Spatial ability for stem domains: Aligning over 50 years of cumulative psychological knowledge solidifies its importance. Journal of Educational Psychology, 101(4):817, 2009.
- [5] Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the National Academy of Sciences, 110(45):18327–18332. 1
- [6] Roger N Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. Science, 171

(3972):701–703, 1971. 1, D

- [7] Mary Hegarty. Mental animation: Inferring motion from static displays of mechanical systems. Journal of Experimental Psychology: Learning, Memory, and Cognition, 18(5):1084–1102, 1992. 1
- [8] Nicholas Judd and Torkel Klingberg. Training spatial cognition enhances mathematical learning in a randomized study of 17,000 children. Nature Human Behaviour, 5(11):1548–1554, 2021. 1
- [9] Bo T Christensen and Christian D Schunn. The role and impact of mental simulation in design. Applied Cognitive Psychology: The Official Journal of the Society for Applied Research in Memory and Cognition, 23(3):327–344, 2009.
- [10] Mary Hegarty. Mechanical reasoning by mental simulation. Trends in Cognitive Sciences, 8(6):280– 285, 2004. ISSN 1364-6613. doi: https://doi.org/10.1016/j.tics.2004.04.001. URL https://www. sciencedirect.com/science/article/pii/S1364661304001007. 1
- [11] et al. Johnson, Justin. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2901–2910, 2017. 1, D
- [12] C. et al. Zhang. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5317–5327, 2019. 2, D
- [13] et al. Ji, W. Abstract visual reasoning with tangram shapes. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2350–2360, 2022.
- [14] Jiafei Duan, Samson Yu, and Cheston Tan. Space: A simulator for physical interactions and causal learning in 3d environments. In Proceedings of the ieee/cvf international conference on computer vision, pages 2058–2063, 2021.
- [15] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.
- [16] S. K. Ramakrishnan, E. Wijmans, P. Krahenbuhl, and V. Koltun. Does spatial cognition emerge in frontier models? arXiv preprint arXiv:2410.06468, 2024. 1, 2, D
- [17] Janellen Huttenlocher and Clark C Presson. Mental rotation and the perspective problem. Cognitive Psychology, 4(2):277–299, 1973. ISSN 0010-0285. doi: https://doi.org/10.1016/0010-0285(73)90015-7. URL https://www.sciencedirect.com/science/article/pii/0010028573900157. 1
- [18] Peri Gunalp, Tara Moossaian, and Mary Hegarty. Spatial perspective taking: Effects of social, directional, and interactive cues. Memory & cognition, 47:1031–1043, 2019.
- [19] Roger N. Shepard and Christine Feng. A chronometric study of mental paper folding. Cognitive Psychology, 3(2):228–243, 1972. ISSN 0010-0285. doi: https://doi.org/10.1016/0010-0285(72)90005-9. URL https://www.sciencedirect.com/science/article/pii/0010028572900059.
- [20] Kai Preuss, Christopher Hilton, Klaus Gramann, and Nele Russwinkel. Identifying cognitive processes and neural substrates of spatial transformation in a mental folding task with cognitive modeling. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 46, 2024.

- [21] Hasan Ayaz, Patricia A Shewokis, Meltem ˙Izzeto˘glu, Murat P Çakır, and Banu Onaral. Tangram solved? prefrontal cortex activation analysis during geometric problem solving. In 2012 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, pages 4724–4727. IEEE, 2012. 1
- [22] Ilona Bass, Kevin A. Smith, Elizabeth Bonawitz, and Tomer D. Ullman. Partial mental simulation explains fallacies in physical reasoning. Cognitive Neuropsychology, 2022. 1, D
- [23] Tony Chen, Kelsey R. Allen, Samuel J. Cheyette, Joshua B. Tenenbaum, and Kevin A. Smith. Just¨ in time¨representations for mental simulation in intuitive physics. In Proceedings of the 45th Annual Meeting of the Cognitive Science Society (CogSci), 2023. 1, D
- [24] Deqing Fu∗, Ghazal Khalighinejad∗, Ollie Liu∗, Bhuwan Dhingra, Dani Yogatama, Robin Jia, and Willie Neiswanger. IsoBench: Benchmarking multimodal foundation models on isomorphic representations,

2024. 2, D

- [25] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Intergps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The 59th Annual Meeting of the Association for Computational Linguistics (ACL), 2021.
- [26] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. European Conference on Computer Vision, 2024. 2
- [27] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209, 2024. 2
- [28] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. ArXiv, abs/2312.14135, 2023. URL https://api.semanticscholar.org/CorpusID:266436019. D
- [29] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 2, 3.2, D
- [30] Eunice Yiu, Maan Qraitem, Charlie Wong, Anisa Noor Majhi, Yutong Bai, Shiry Ginosar, Alison Gopnik, and Kate Saenko. Kiva: Kid-inspired visual analogies for testing large multimodal models. arXiv preprint arXiv:2407.17773, 2024. 2, D
- [31] Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1567–1574, 2021. 2, D
- [32] Sina Rismanchian, Yasaman Razeghi, Sameer Singh, and Shayan Doroudi. Turtlebench: A visual programming benchmark in turtle geometry. arXiv preprint arXiv:2411.00264, 2024. 2, D
- [33] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024. 2, D
- [34] Matplotlib. Matplotlib: Visualization with python. https://matplotlib.org/, 2012. 2.2
- [35] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017. 2.2
- [36] Blender. Blender is free software. https://www.blender.org/. 2.2
- [37] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021. 2.2
- [38] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021. URL https://arxiv.org/abs/2109.08238. 2.2
- [39] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A Platform for Embodied AI Research. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 2.2

- [40] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0: Training home assistants to rearrange their habitat. In Advances in Neural Information Processing Systems (NeurIPS), 2021.
- [41] Xavi Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Ruslan Partsey, Jimmy Yang, Ruta Desai, Alexander William Clegg, Michal Hlavac, Tiffany Min, Theo Gervet, Vladimír Vondruš, VincentPierre Berges, John Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A co-habitat for humans, avatars and robots, 2023. 2.2
- [42] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/. 3.1
- [43] Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet. 3.1
- [44] Google Deepmind. Introducing gemini 2.0: our new ai model for the agentic era. https://blog. google/technology/google-deepmind/google-gemini-ai-update-december-2024//, . 3.1
- [45] Google Deepmind. Gemini 2.0 flash thinking mode. https://ai.google.dev/gemini-api/docs/ thinking-mode, . 3.1
- [46] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720. 3.1
- [47] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 3.1

- [48] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326,

2024. 3.1

- [49] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/ abs/2502.13923. 3.1
- [50] Dedre Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2):155– 170, 1983. D
- [51] P.A. Carpenter, M.A. Just, and P. Shell. What one intelligence test measures: A theoretical account of the processing in the raven progressive matrices test. Psychological Review, 97(3):404–431, 1990. D
- [52] A. Lovett and K. Forbus. Modeling visual problem solving as analogical reasoning. Psychological Review, 124(1):60–90, 2017. D
- [53] Nicholas Ichien, Qing Liu, Shuhao Fu, Keith J. Holyoak, Alan Yuille, and Hongjing Lu. Visual analogy: Deep learning versus compositional models. In Proceedings of the 43rd Annual Meeting of the Cognitive Science Society (CogSci), 2021. D
- [54] Taylor W. Webb, Shuhao Fu, Trevor Bihl, Keith J. Holyoak, and Hongjing Lu. Zero-shot visual reasoning through probabilistic analogical mapping. arXiv preprint arXiv:2209.15087, 2022.
- [55] N. Ichien, Q. Liu, S. Fu, K.J. Holyoak, A. Yuille, and H. Lu. Two computational approaches to visual analogy: Task-specific models versus domain-general mapping. Cognitive Science, 47(4):e13347, 2023. D
- [56] M. Hegarty. Mechanical reasoning by mental simulation. Trends in Cognitive Sciences, 8(6):280–285,

2004. D

- [57] L.W. Barsalou. Grounded cognition. Annual Review of Psychology, 59:617–645, 2008. D
- [58] P.W. Battaglia, J.B. Hamrick, and J.B. Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the National Academy of Sciences, 110(45):18327–18332, 2013. D
- [59] J.B. Tenenbaum, T.L. Griffiths, and C. Kemp. Theory-based bayesian models of inductive learning and reasoning. Trends in Cognitive Sciences, 10(7):309–318, 2006.
- [60] T.D. Ullman, E.S. Spelke, P. Battaglia, and J.B. Tenenbaum. Mind games: Game engines as an architecture for intuitive physics. Trends in Cognitive Sciences, 21(9):649–665, 2017. D
- [61] Lingxiao Yang, Hongzhi You, Zonglei Zhen, Dahui Wang, Xiaohong Wan, Xiaohua Xie, and Ru-Yuan Zhang. Neural prediction errors enable analogical visual reasoning in human standard intelligence tests. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. D
- [62] Luis S. Piloto, Ari Weinstein, Peter Battaglia, and Matthew Botvinick. Intuitive physics learning in a deep-learning model inspired by developmental psychology. Nature Human Behaviour, 6(9):1257–1267,

2022. D

- [63] Daniel M. Bear, Elias Wang, Damian Mrowca, Felix J. Binder, Hsiao-Yu F. Tung, R. T. Pramod, Cameron Holdaway, Sirui Tao, Kevin A. Smith, Fan-Yun Sun, Li Fei-Fei, Nancy Kanwisher, Joshua B. Tenenbaum, Daniel L. K. Yamins, and Judith E. Fan. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2022. D
- [64] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In ICCV, 2015. D
- [65] Xiyao Yue, Yifan Ni, Kai Zhang, Tao Zheng, Ruixuan Liu, Wen Chen, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023. D
- [66] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Tung Nguyen, Daron Anderson, Imad Ali Shah, Mikhail Doroshenko, Alun Cennyth Stokes, Mobeen Mahmood, Jaeho Lee, Oleksandr Pokutnyi, Oleg Iskra, Jessica P. Wang, Robert Gerbicz, John-Clark Levin, Serguei Popov, Fiona

Feng, Steven Y. Feng, Haoran Zhao, Michael Yu, Varun Gangal, Chelsea Zou, Zihan Wang, Mstyslav Kazakov, Geoff Galgon, Johannes Schmitt, Alvaro Sanchez, Yongki Lee, Will Yeadon, Scott Sauers, Marc Roth, Chidozie Agu, Søren Riis, Fabian Giska, Saiteja Utpala, Antrell Cheatom, Zachary Giboney, Gashaw M. Goshu, Sarah-Jane Crowson, Mohinder Maheshbhai Naiya, Noah Burns, Lennart Finke, Zerui Cheng, Hyunwoo Park, Francesco Fournier-Facio, Jennifer Zampese, John Wydallis, John B. Wydallis, Ryan G. Hoerr, Mark Nandor, Tim Gehrunger, Jiaqi Cai, Ben McCarty, Jungbae Nam, Edwin Taylor, Jun Jin, Gautier Abou Loume, Hangrui Cao, Alexis C Garretson, Damien Sileo, Qiuyu Ren, Doru Cojoc, Pavel Arkhipov, Usman Qazi, Aras Bacho, Lianghui Li, Sumeet Motwani, Christian Schroeder de Witt, Alexei Kopylov, Johannes Veith, Eric Singer, Paolo Rissone, Jaehyeok Jin, Jack Wei Lun Shi, Chris G. Willcocks, Ameya Prabhu, Longke Tang, Kevin Zhou, Emily de Oliveira Santos, Andrey Pupasov Maksimov, Edward Vendrow, Kengo Zenitani, Joshua Robinson, Aleksandar Mikov, Julien Guillod, Yuqi Li, Ben Pageler, Joshua Vendrow, Vladyslav Kuchkin, Pierre Marion, Denis Efremov, Jayson Lynch, Kaiqu Liang, Andrew Gritsevskiy, Dakotah Martinez, Nick Crispino, Dimitri Zvonkine, Natanael Wildner Fraga, Saeed Soori, Ori Press, Henry Tang, Julian Salazar, Sean R. Green, Lina Brüssel, Moon Twayana, Aymeric Dieuleveut, T. Ryan Rogers, Wenjin Zhang, Ross Finocchio, Bikun Li, Jinzhou Yang, Arun Rao, Gabriel Loiseau, Mikhail Kalinin, Marco Lukas, Ciprian Manolescu, Nate Stambaugh, Subrata Mishra, Ariel Ghislain Kemogne Kamdoum, Tad Hogg, Alvin Jin, Carlo Bosio, Gongbo Sun, Brian P Coppola, Haline Heidinger, Rafael Sayous, Stefan Ivanov, Joseph M Cavanagh, Jiawei Shen, Joseph Marvin Imperial, Philippe Schwaller, Shaipranesh Senthilkuma, Andres M Bran, Andres Algaba, Brecht Verbeken, Kelsey Van den Houte, Lynn Van Der Sypt, David Noever, Lisa Schut, Ilia Sucholutsky, Evgenii Zheltonozhskii, Qiaochu Yuan, Derek Lim, Richard Stanley, Shankar Sivarajan, Tong Yang, John Maar, Julian Wykowski, Martí Oller, Jennifer Sandlin, Anmol Sahu, Cesare Giulio Ardito, Yuzheng Hu, Felipe Meneguitti Dias, Tobias Kreiman, Kaivalya Rawal, Tobias Garcia Vilchis, Yuexuan Zu, Martin Lackner, James Koppel, Jeremy Nguyen, Daniil S. Antonenko, Steffi Chern, Bingchen Zhao, Pierrot Arsene, Sergey Ivanov, Rafał Po´swiata, Chenguang Wang, Daofeng Li, Donato Crisostomi, Ali Dehghan, Andrea Achilleos, John Arnold Ambay, Benjamin Myklebust, Archan Sen, David Perrella, Nurdin Kaparov, Mark H Inlow, Allen Zang, Kalyan Ramakrishnan, Daniil Orel, Vladislav Poritski, Shalev BenDavid, Zachary Berger, Parker Whitfill, Michael Foster, Daniel Munro, Linh Ho, Dan Bar Hava, Aleksey Kuchkin, Robert Lauff, David Holmes, Frank Sommerhage, Anji Zhang, Richard Moat, Keith Schneider, Daniel Pyda, Zakayo Kazibwe, Mukhwinder Singh, Don Clarke, Dae Hyun Kim, Sara Fish, Veit Elser, Victor Efren Guadarrama Vilchis, Immo Klose, Christoph Demian, Ujjwala Anantheswaran, Adam Zweiger, Guglielmo Albani, Jeffery Li, Nicolas Daans, Maksim Radionov, Václav Rozhoˇn, Vincent Ginis, Ziqiao Ma, Christian Stump, Jacob Platnick, Volodymyr Nevirkovets, Luke Basler, Marco Piccardo, Niv Cohen, Virendra Singh, Josef Tkadlec, Paul Rosu, Alan Goldfarb, Piotr Padlewski, Stanislaw Barzowski, Kyle Montgomery, Aline Menezes, Arkil Patel, Zixuan Wang, Jamie Tucker-Foltz, Jack Stade, Declan Grabb, Tom Goertzen, Fereshteh Kazemi, Jeremiah Milbauer, Abhishek Shukla, Hossam Elgnainy, Yan Carlos Leyva Labrador, Hao He, Ling Zhang, Alan Givré, Hew Wolff, Gözdenur Demir, Muhammad Fayez Aziz, Younesse Kaddar, Ivar Ängquist, Yanxu Chen, Elliott Thornley, Robin Zhang, Jiayi Pan, Antonio Terpin, Niklas Muennighoff, Hailey Schoelkopf, Eric Zheng, Avishy Carmi, Jainam Shah, Ethan D. L. Brown, Kelin Zhu, Max Bartolo, Richard Wheeler, Andrew Ho, Shaul Barkan, Jiaqi Wang, Martin Stehberger, Egor Kretov, Peter Bradshaw, JP Heimonen, Kaustubh Sridhar, Zaki Hossain, Ido Akov, Yury Makarychev, Joanna Tam, Hieu Hoang, David M. Cunningham, Vladimir Goryachev, Demosthenes Patramanis, Michael Krause, Andrew Redenti, David Aldous, Jesyin Lai, Shannon Coleman, Jiangnan Xu, Sangwon Lee, Ilias Magoulas, Sandy Zhao, Ning Tang, Michael K. Cohen, Micah Carroll, Orr Paradise, Jan Hendrik Kirchner, Stefan Steinerberger, Maksym Ovchynnikov, Jason O. Matos, Adithya Shenoy, Michael Wang, Yuzhou Nie, Paolo Giordano, Philipp Petersen, Anna Sztyber-Betley, Paolo Faraboschi, Robin Riblet, Jonathan Crozier, Shiv Halasyamani, Antonella Pinto, Shreyas Verma, Prashant Joshi, Eli Meril, Zheng-Xin Yong, Allison Tee, Jérémy Andréoletti, Orion Weller, Raghav Singhal, Gang Zhang, Alexander Ivanov, Seri Khoury, Nils Gustafsson, Hamid Mostaghimi, Kunvar Thaman, Qijia Chen, Tran Quoc Khánh, Jacob Loader, Stefano Cavalleri, Hannah Szlyk, Zachary Brown, Himanshu Narayan, Jonathan Roberts, William Alley, Kunyang Sun, Ryan Stendall, Max Lamparth, Anka Reuel, Ting Wang, Hanmeng Xu, Pablo Hernández-Cámara, Freddie Martin, Thomas Preu, Tomek Korbak, Marcus Abramovitch, Dominic Williamson, Ida Bosio, Ziye Chen, Biró Bálint, Eve J. Y. Lo, Maria Inês S. Nunes, Yibo Jiang, M Saiful Bari, Peyman Kassani, Zihao Wang, Behzad Ansarinejad, Yewen Sun, Stephane Durand, Guillaume Douville, Daniel Tordera, George Balabanian, Earth Anderson, Lynna Kvistad, Alejandro José Moyano, Hsiaoyun Milliron, Ahmad Sakor, Murat Eron, Isaac C. McAlister, Andrew Favre D. O., Shailesh Shah, Xiaoxiang Zhou, Firuz Kamalov, Ronald Clark, Sherwin Abdoli, Tim Santens, Harrison K Wang, Evan Chen, Alessandro Tomasiello, G. Bruno De Luca, Shi-Zhuo Looi, VinhKha Le, Noam Kolt, Niels Mündler, Avi Semler, Emma Rodman, Jacob Drori, Carl J Fossum, Luk Gloor, Milind Jagota, Ronak Pradeep, Honglu Fan, Tej Shah, Jonathan Eicher, Michael Chen, Kushal Thaman, William Merrill, Moritz Firsching, Carter Harris, Stefan Ciobâc˘a, Jason Gross, Rohan Pandey, Ilya Gusev, Adam Jones, Shashank Agnihotri, Pavel Zhelnov, Siranut Usawasutsakorn, Mohammadreza Mofayezi, Alexander Piperski, Marc Carauleanu, David K. Zhang, Kostiantyn Dobarskyi, Dylan Ler, Roman Leventov, Ignat Soroko, Thorben Jansen, Scott Creighton, Pascal Lauer, Joshua Duersch, Vage Taamazyan, Dario Bezzi, Wiktor Morak, Wenjie Ma, William Held, Tran Ðuc Huy, Ruicheng Xian, Armel Randy Ze-

baze, Mohanad Mohamed, Julian Noah Leser, Michelle X Yuan, Laila Yacar, Johannes Lengler, Katarzyna Olszewska, Hossein Shahrtash, Edson Oliveira, Joseph W. Jackson, Daniel Espinosa Gonzalez, Andy Zou, Muthu Chidambaram, Timothy Manik, Hector Haffenden, Dashiell Stander, Ali Dasouqi, Alexander Shen, Emilien Duc, Bita Golshani, David Stap, Mikalai Uzhou, Alina Borisovna Zhidkovskaya, Lukas Lewark, Miguel Orbegozo Rodriguez, Mátyás Vincze, Dustin Wehr, Colin Tang, Shaun Phillips, Fortuna Samuele, Jiang Muzhen, Fredrik Ekström, Angela Hammon, Oam Patel, Faraz Farhidi, George Medley, Forough Mohammadzadeh, Madellene Peñaflor, Haile Kassahun, Alena Friedrich, Claire Sparrow, Rayner Hernandez Perez, Taom Sakal, Omkar Dhamane, Ali Khajegili Mirabadi, Eric Hallman, Kenchi Okutsu, Mike Battaglia, Mohammad Maghsoudimehrabani, Alon Amit, Dave Hulbert, Roberto Pereira, Simon Weber, Handoko, Anton Peristyy, Stephen Malina, Samuel Albanie, Will Cai, Mustafa Mehkary, Rami Aly, Frank Reidegeld, Anna-Katharina Dick, Cary Friday, Jasdeep Sidhu, Hassan Shapourian, Wanyoung Kim, Mariana Costa, Hubeyb Gurdogan, Brian Weber, Harsh Kumar, Tong Jiang, Arunim Agarwal, Chiara Ceconello, Warren S. Vaz, Chao Zhuang, Haon Park, Andrew R. Tawfeek, Daattavya Aggarwal, Michael Kirchhof, Linjie Dai, Evan Kim, Johan Ferret, Yuzhou Wang, Minghao Yan, Krzysztof Burdzy, Lixin Zhang, Antonio Franca, Diana T. Pham, Kang Yong Loh, Joshua Robinson, Abram Jackson, Shreen Gul, Gunjan Chhablani, Zhehang Du, Adrian Cosma, Jesus Colino, Colin White, Jacob Votava, Vladimir Vinnikov, Ethan Delaney, Petr Spelda, Vit Stritecky, Syed M. Shahid, Jean-Christophe Mourrat, Lavr Vetoshkin, Koen Sponselee, Renas Bacho, Florencia de la Rosa, Xiuyu Li, Guillaume Malod, Leon Lang, Julien Laurendeau, Dmitry Kazakov, Fatimah Adesanya, Julien Portier, Lawrence Hollom, Victor Souza, Yuchen Anna Zhou, Julien Degorre, Yi˘git Yalın, Gbenga Daniel Obikoya, Luca Arnaboldi, Rai, Filippo Bigi, M. C. Boscá, Oleg Shumar, Kaniuar Bacho, Pierre Clavier, Gabriel Recchia, Mara Popescu, Nikita Shulga, Ngefor Mildred Tanwie, Denis Peskoff, Thomas C. H. Lux, Ben Rank, Colin Ni, Matthew Brooks, Alesia Yakimchyk, Huanxu, Liu, Olle Häggström, Emil Verkama, Hans Gundlach, Leonor BritoSantana, Brian Amaro, Vivek Vajipey, Rynaa Grover, Yiyang Fan, Gabriel Poesia Reis e Silva, Linwei Xin, Yosi Kratish, Jakub Łucki, Wen-Ding Li, Sivakanth Gopi, Andrea Caciolai, Justin Xu, Kevin Joseph Scaria, Freddie Vargus, Farzad Habibi, Long, Lian, Emanuele Rodolà, Jules Robins, Vincent Cheng, Tony Fruhauff, Brad Raynor, Hao Qi, Xi Jiang, Ben Segev, Jingxuan Fan, Sarah Martinson, Erik Y. Wang, Kaylie Hausknecht, Michael P. Brenner, Mao Mao, Xinyu Zhang, David Avagian, Eshawn Jessica Scipio, Alon Ragoler, Justin Tan, Blake Sims, Rebeka Plecnik, Aaron Kirtland, Omer Faruk Bodur, D. P. Shinde, Zahra Adoul, Mohamed Zekry, Ali Karakoc, Tania C. B. Santos, Samir Shamseldeen, Loukmane Karim, Anna Liakhovitskaia, Nate Resman, Nicholas Farina, Juan Carlos Gonzalez, Gabe Maayan, Sarah Hoback, Rodrigo De Oliveira Pena, Glen Sherman, Elizabeth Kelley, Hodjat Mariji, Rasoul Pouriamanesh, Wentao Wu, Sandra Mendoza, Ismail Alarab, Joshua Cole, Danyelle Ferreira, Bryan Johnson, Mohammad Safdari, Liangti Dai, Siriphan Arthornthurasuk, Alexey Pronin, Jing Fan, Angel RamirezTrinidad, Ashley Cartwright, Daphiny Pottmaier, Omid Taheri, David Outevsky, Stanley Stepanic, Samuel Perry, Luke Askew, Raúl Adrián Huerta Rodríguez, Ali M. R. Minissi, Sam Ali, Ricardo Lorena, Krishnamurthy Iyer, Arshad Anil Fasiludeen, Sk Md Salauddin, Murat Islam, Juan Gonzalez, Josh Ducey, Maja Somrak, Vasilios Mavroudis, Eric Vergo, Juehang Qin, Benjámin Borbás, Eric Chu, Jack Lindsey, Anil Radhakrishnan, Antoine Jallon, I. M. J. McInnis, Pawan Kumar, Laxman Prasad Goswami, Daniel Bugas, Nasser Heydari, Ferenc Jeanplong, Archimedes Apronti, Abdallah Galal, Ng Ze-An, Ankit Singh, Joan of Arc Xavier, Kanu Priya Agarwal, Mohammed Berkani, Benedito Alves de Oliveira Junior, Dmitry Malishev, Nicolas Remy, Taylor D. Hartman, Tim Tarver, Stephen Mensah, Javier Gimenez, Roselynn Grace Montecillo, Russell Campbell, Asankhaya Sharma, Khalida Meer, Xavier Alapont, Deepakkumar Patil, Rajat Maheshwari, Abdelkader Dendane, Priti Shukla, Sergei Bogdanov, Sören Möller, Muhammad Rehan Siddiqi, Prajvi Saxena, Himanshu Gupta, Innocent Enyekwe, Ragavendran P V, Zienab EL-Wasif, Aleksandr Maksapetyan, Vivien Rossbach, Chris Harjadi, Mohsen Bahaloohoreh, Song Bian, John Lai, Justine Leon Uro, Greg Bateman, Mohamed Sayed, Ahmed Menshawy, Darling Duclosel, Yashaswini Jain, Ashley Aaron, Murat Tiryakioglu, Sheeshram Siddh, Keith Krenek, Alex Hoover, Joseph McGowan, Tejal Patwardhan, Summer Yue, Alexandr Wang, and Dan Hendrycks. Humanity’s last exam, 2025. URL https://arxiv.org/abs/2501.14249. D

- [67] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. D
- [68] Wenxuan Zhang, Sharifah M. Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks, 2023. D
- [69] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. In arXiv preprint arXiv:2306.13394, 2023. D
- [70] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In Proceedings of the European Conference on Computer Vision (ECCV), 2024. D

- [71] Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. arXiv preprint arXiv:2412.03548, 2024. D
- [72] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,

2023. F.1

- [73] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6. F.1

- A Overview of the Appendix This Appendix is organized as follows:

- • Section B and C discuss the limitations and broader impact of STARE.
- • Section D presents an extended discussion about related works.
- • Section E details the statistics of STARE and the design spaces for all synthetic tasks, including 2D transformations, 3D transformations, cube net folding, and tangram puzzles.
- • Section F describes the experimental setup, covering the prompoints used, model configurations, hyperparameter settings, and presents full visualizations of different experimental settings (e.g., evaluation settings with or without visual simulations, perception probing questions).
- • Section G provides experimental results on additional models for analysis conducted in Section 3.

- B Limitations

Although STARE provides valuable insights, it still has several limitations. First, it uses simplified synthetic images that do not fully represent real-world complexity; future versions could include realistic or dynamic scenes with clutter and occlusion. Second, it focuses only on rigid shape transformations; adding tasks involving flexible shapes, articulated objects, or additional sensory cues (such as audio or depth) would cover a wider range of spatial reasoning skills. Lastly, multiple-choice scoring hides intermediate reasoning steps; extending evaluations with explanations, step-by-step checks, or open-ended responses would give more detailed insights.

Still, STARE ’s current design has clear strengths. The simplified images isolate spatial reasoning from general object recognition tasks. Its structured variety of tasks helps pinpoint specific model difficulties. Automatic scoring ensures consistent and easy-to-scale evaluations. Modular task presentations (image-only, text-only, image+text prompts) let researchers analyze individual modality contributions. Additionally, synthetic data makes STARE easily reproducible, accessible, and extensible. Overall, STARE is a strong first step toward measuring multimodal spatial reasoning, with clear pathways toward more realistic and comprehensive future benchmarks.

- C Broader Impact

STARE provides a standardized way to measure AI capabilities in spatial reasoning tasks, potentially guiding research toward AI systems that can better support robotics, autonomous driving, augmented reality, and education. However, improved spatial reasoning could also lead to negative societal impacts if misused, such as enhanced surveillance or military applications. Additionally, the synthetic nature of STARE may introduce biases toward simplified or artificial scenarios, limiting direct applicability to real-world conditions. Future versions should aim to include more realistic, diverse datasets and consider ethical guidelines to minimize risks and ensure fair, positive societal outcomes.

- D Related work

Human visual reasoning. Human visual reasoning relies on two complementary faculties: relational analogy—mapping abstract structures across scenes—and mental simulation—predicting future states through incremental transformations. Structure–Mapping Theory [50] and analyses of Raven’s Progressive Matrices [51] first showed that success in visual problem-solving hinges on aligning relations rather than surface features. Computational accounts echo this claim: explicit relational models reproduce human-like performance [52], whereas modern deep networks still struggle with visual analogy tasks [53–55].

Mental simulation complements analogy-making. Classic work on mental rotation [6] and mechanical reasoning [56] demonstrates that people mentally “run” transformations, consistent with grounded-cognition theories [57]. Intuitive-physics studies cast the mind as a noisy physics engine

that combines object-centric structure with probabilistic dynamics [58–60]. Object-based predictivecoding models such as PLATO extend these ideas, achieving human-like physical prediction and developmental trajectories [61, 62]. Simulations are also selective: people allocate attention “just in time,” focusing on the most diagnostic elements instead of exhaustively modeling the entire scene [22, 63, 23].

Together, these findings suggest that effective problem-solving hinges on the ability to carry out step-by-step visual simulations; our benchmark therefore probes whether multimodal models can effectively leverage or even produce such simulations and exhibit human-like visual reasoning on sequential, compositional tasks.

Multimodal evaluation benchmarks. Recent advances in evaluating multimodal large language models have led to the development of benchmarks targeting diverse aspects of visual reasoning. Early benchmarks such as VQA [64] and CLEVR [11] focus on compositional reasoning and general visual question answering. However, more challenging benchmarks, such as MMMU [65] and Humanity’s Last Exam (HLE) [66], assess expert-level, domain-specific reasoning using complex multimodal inputs, where state-of-the-art models achieve only around 60% on MMMU-pro [67] and below 20% on HLE.

In response to the growing demand for robust evaluation, several new benchmarks [24–28] have been introduced. For example, M3Exam repurposes multilingual professional-license questions [68]. MME [69] and MMBench [70] separate low-level perception from higher-level cognition. BLINK [29] departs from pure linguistic reasoning tasks to include tasks grounded in core computer vision capabilities, including relative depth estimation, semantic correspondence, visual similarity assessment, inpainting, etc. Improvements on BLINK require the use of perception tokens [71], which generate latent intrinsic images to reason, demonstrating for the first time, that reasoning doesn’t have to be linguistic. In this work, we build upon this finding, targeting primarily visual reasoning that can be better solved with visual cues.

The most relevant benchmarks to ours are perhaps KiVA [30], RAVEN/I-RAVEN [12, 31], SPACE [16], and TurtleBench [32], which mostly evaluate static analogy or pattern induction, often step-wise visual simulations is optional and curating the intermediate visual simulations is either not feasible or requiring extensive human efforts. VSI-Bench [33] underscores the role of mental imagery in spatial reasoning, but focuses on spatial memory and estimation from video rather than explicit step-by-step simulation. STARE bridges this gap with programmatically generated puzzles—2D/3D transformations, cube-net folding, and tangram assembly—that isolate a model’s capacity to benefit from explicit visual simulations. We further extend the benchmark with perspective taking and temporal frame reasoning tasks that mirror real-world scenarios.

## E Data Curation Details

Figure 7 presents the overall composition of STARE. Table 6 details the number of instances for each task in STARE, further broken down by whether the input contains an explicit intermediate visual simulations.

Task category Without visual simulation With visual simulation Total Foundational Geometric Transformations

- 2D transformations 639 423 1,062
- 3D transformations 612 408 1,020 Integrated Spatial Reasoning

Cube net folding 193 120 313 Tangram puzzle 532 289 821

Real-world Spatial Reasoning

Perspective reasoning 250 – 250 Temporal frame reasoning 471 – 471 Total 2,697 1,240 3,937 Table 6: Dataset statistics grouped by task category and by the presence of full intermediate visual simulation.

[Figure 66]

Figure 7: Data Statistics of STARE.

Below, we summarize the design space of data curation for synthetic tasks, including (1) 2D Transformations (§E.1); (2) 3D Transformations (§E.2); (3) Cube Net Folding (§E.3); and (4) Tangram Puzzles (§E.4);

- E.1 2D Transformations Shape generation. Shapes are selected from a fixed set and assigned properties as follows:

- • Types: Circle, Square, Rectangle, Triangle, Ellipse, Hexagon, Pentagon.
- • Colors: Face color is a random RGB tuple (r,g,b ∈ [0,1]); edge color is fixed (black).
- • Center & Size: All shapes are centered at (0,0). For circles, squares, triangles, hexagons, and pentagons, size is a scalar drawn from [30,35]; for rectangles and ellipses, size is a tuple (width in [30,35], height in [20,25]).

Transformations. A sequence of randomly sampled operations is applied to the shapes:

- • Rotate:

- – Squares: ±30◦, ±60◦ (avoiding 90◦).
- – Hexagons: ±30◦, ±90◦.
- – Others: ±30◦, ±60◦, or ±90◦.

Rotation is applied w.r.t the shape’s center.

- • Flip: Horizontal (about y = 0) or vertical (about x = 0); not applied when the shape is centered at (0,0) for symmetric shapes such as square, circle and etc.
- • Translate: (dx,dy) with dx,dy ∈ {−30,−10,0,10,30} with constraints to ensure a nonzero translation.
- • Scale: Factors chosen from {0.5,2.0}, ensuring the resultant size is within roughly [10,40].
- • Shear: Parameters (shearx,sheary) are drawn from approximately [−1,1], with constraints to ensure a perceptible skew. Shear is excluded for 2D text instructed transformation tasks, as human participants find it hard to describe the degree of shear such that they can differentiate among the answer candidates.

Number of Transformation Steps. The final dataset contains instances with 1, 2, or 3 transformation steps.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

###### Figure 8: Design space of 2D Transformations (1).

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

- Figure 9: Design space of 2D Transformations (2).

###### E.2 3D Transformations

Shape generation. 3D objects are loaded from external blend files and instantiated with random properties defined in a JSON file. Their attributes include:

- • Types: Various 3D models such as cube, sphere, cone, cylinder, torus, pyramid, etc.
- • Colors & Materials: Colors are sampled from a predefined set, and materials are selected from external files.
- • Size & Location: Objects are assigned a size scalar (from the JSON-specified values) and an initial 3D location (typically near the origin), with adjustments to ensure they remain above the ground plane.

Transformations. A sequence of randomly sampled operations is applied to the objects in 3D space:

###### • Translate:

- – Axis selection: Randomly choose one or more axes from x, y, and z (e.g., “x”, “xy”, “xz”, “yz”).
- – Displacement: Translations are applied with discrete displacements: along x and y by ±2 units and along z by ±1 unit, with constraints to keep the object above the ground (z ≥ 0).

###### • Rotate:

- – Axis: A single rotation axis is chosen randomly from x, y, or z.
- – Angle: The rotation angle is drawn from a discrete set (typically ±30◦, ±60◦, or ±90◦), with the range sometimes adjusted for specific shapes (e.g., cubes or pyramids).
- – Rotation is applied about the object’s center.

###### • Shear:

- – Plane: The shear operation is applied along one of three directional pairs: xy, xz, or yz.
- – Factors: Two shear factors are sampled uniformly from the interval [0.2,1.0], with an enforced minimum difference (approximately 0.4) to ensure a perceptible skew.

###### • Scale:

– Factor: A uniform scaling factor is chosen from 0.5,2.0, either reducing or enlarging

the object while keeping its final size within acceptable bounds.

###### • Flip:

– Direction: The object is reflected along a principal axis—flipped horizontally (reflec-

tion across the x-axis) or vertically (reflection across the y-axis).

All transformation operations are applied sequentially, updating the object’s 3D coordinates (including its bounding box and center) to reflect the cumulative effects.

Number of Transformation Steps. Instances are generated with transformation sequences comprising 1, 2, or 3 steps, where each step randomly selects one of the available operations. This multi-step approach enables a diverse design space of 3D transformations, as the operations can compound in various orders and combinations.

###### E.3 Cube Net Folding

Net Representation. Cube nets are represented as collections of faces, where each face is defined by its vertices in 3D space. Additional attributes include:

- • Face Geometry: Each face is a polygon (typically a quadrilateral) with vertex coordinates stored as NumPy arrays.
- • Connectivity: A mapping of face connections identifies which faces share common edges, serving as potential hinges.
- • Visual Attributes: Faces are rendered with colors (sampled from a colormap) and labeled with their keys for easy identification.

Folding Operations. The folding process simulates converting a 2D cube net into a 3D cube via a sequence of rotation operations:

- • Shared Edge Detection: The algorithm locates the common edge between a candidate face and an already folded face. A tolerance is used to robustly identify two shared vertices.
- • Rotation Calculation: Using the shared edge as a hinge, a rotation is computed with a fixed magnitude of 90◦ (i.e. ±π/2 radians). The sign of the angle is chosen by comparing the candidate face’s center (projected onto the hinge’s perpendicular plane) with the desired direction toward the cube’s center, which is derived from the base face.

- • Recursive Propagation: The rotation is applied not only to the candidate face but also recursively to all connected faces that have not been folded yet, ensuring that the entire net adjusts consistently.

Folding Sequence and Visualization. The design space supports iterative, step-by-step folding, with each step comprising:

- • Candidate Selection: Among the faces not yet folded, the algorithm picks one that is connected to an already folded face.
- • Folding Parameters: It computes the rotation axis (the shared edge) and the appropriate 90◦ rotation (with correct sign) to fold the face into its 3D position.
- • Instruction Generation: Each fold is described in natural language (e.g., “Fold face 2 upwards towards face 3”) based on changes in the face’s center relative to the cube’s base.
- • 3D Rendering: After each step, the current state of the net is visualized using a 3D plot (with Poly3DCollection) and saved as an image.

Perturbation and Validity. To enrich the design space and introduce challenge:

- • Perturbations: Selected folding steps can be intentionally altered by inverting the rotation angle or modifying the rotation axis. This simulates errors or variations, yielding nets that might fold incorrectly.
- • Validity Checks: Functions are provided to verify that folded faces do not overlap, that shared edges are consistently maintained, and that face connections remain intact. These checks ensure that the final folded cube is geometrically valid.

Dataset Generation and Perception Tasks. Beyond simulating the folding process, the design space incorporates mechanisms to create annotated datasets:

- • Instructional Sequences: Detailed, step-by-step folding instructions (with corresponding images) are generated, supporting tasks that require understanding the folding procedure.
- • Perception Variants: Additional tasks query the observer’s perception—such as verifying if a particular face has been folded or determining the connectivity between faces—using intermediate folding images.

Randomness and Parameter Control. Stochastic elements pervade the folding simulation:

- • Random seeds govern the selection of candidate faces, the decision to perturb a folding step, and the choice of rotation adjustments.
- • This randomness ensures that a diverse range of cube nets and folding sequences are produced, which is crucial for generating robust datasets and for studying perception and reasoning in 3D folding tasks.

###### E.4 Tangram Puzzle

Segmentation. The puzzle begins with an iterative segmentation algorithm that splits a full rectangular board into smaller pieces. The process is governed by a minimum piece size and a maximum number of pieces. At each segmentation step, the algorithm:

- • Selects a splittable rectangle based on its area.
- • Chooses a split direction (horizontal if the height is greater or vertical otherwise) and a split line ensuring both resulting pieces exceed the minimum size.
- • Records each split as an action with details (original rectangle, split line, and direction) that form the basis for later textual instructions.

Piece Generation & Attributes. Each tangram piece is defined by its board coordinates (e.g., (r0, r1, c0, c1)) and derived properties such as area and dimensions. Additionally:

- • Colors: Pieces are assigned unique, randomly generated colors.

- • Visualization: Grid lines and labels are overlaid on each piece to indicate its boundaries and area, facilitating clear visualization during reassembly.

Scrambling and Transformation. Once segmented, pieces are scrambled to increase puzzle complexity. This involves applying a series of random transformation operations:

- • Rotation: Each piece is rotated by a discrete angle chosen from 0◦,30◦,60◦,90◦.
- • Translation: Pieces are repositioned into non-overlapping cells on a larger canvas.
- • Flip: In some reassembly variants, horizontal or vertical flips are applied to further randomize the piece orientations.

## F Experimental Details

###### F.1 Models and Settings

To expedite response generation, we use the vLLM [72] library, an open-source tool for fast LLM inference and serving. For all other cases, we load models directly using the Transformers [73] library. All model sources are official and listed in Table 7. When evaluating different models, we use default hyperparameter values unless otherwise specified, with detailed parameter settings provided in Table 7. For all models, we explicitly prompt it with Think step-by-step, and then put your final answer in \"\\boxed{}\". to encourage chain-of-thought reasoning and for easier answer parsing.

Model Parameter Setting Source URL

GPT-4o temperature = 0.0 chatgpt-4o-latest https://platform.openai.com

Claude 3.5 Sonnet temperature = 0.0 claude-3-5-sonnet https://www.anthropic.com/ Gemini 2.0 Flash temperature = 0.0 gemini-2.0-flash-exp https://ai.google.dev/ Gemini 2.0 Flash Thinking

gemini-2.0-flashthinking-exp-1219 https://ai.google.dev/

temperature = 0.0

OpenAI o1 temperature = 0.0 o1-2024-12-17 https://platform.openai.com Qwen2.5-VL-3B

do sample=True, temperature = 0.7

local checkpoint https://huggingface.co/Qwen/

Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B

do sample=True, temperature = 0.7

local checkpoint https://huggingface.co/Qwen/

Qwen2.5-VL-7B-Instruct Qwen2.5-VL-72B

do sample=True, temperature = 0.7

local checkpoint https://huggingface.co/Qwen/

Qwen2.5-VL-72B-Instruct LLaVA-Onevision-72B

do sample=True, temperature = 0.7

local checkpoint https://huggingface. co/llava-hf/ llava-onevision-qwen2-72b-ov-hf

do sample=True, temperature = 0.7

InternVL2.5-78B

local checkpoint https://huggingface.co/

OpenGVLab/InternVL2_5-78B

Table 7: The sources of models used in the experiments and the hyperparameters configuration.

###### F.2 Visualization of Evaluation Settings

Figures 10–11 provide full visualizations of evaluation settings illustrated in Figure 3. In addition, we show an example of how real-world spatial reasoning task – temporal frame reasoning is evaluated without visual simulation in Figure 12.

[Figure 77]

[Figure 78]

Figure 10: Examples of Tangram Puzzle under “without Visual Simulations" Evaluation Setting (top: questiononly, bottom: question+assembly steps).

[Figure 79]

[Figure 80]

Figure 11: Example of Tangram Puzzle under “with Visual Simulations" Evaluation Setting.

Question

You see 4 sequential frames of a video, but one is missing (marked with "?'). Choose which of the images in the second row correctly fills the missing frame. Remember, the camera only moves in one direction (left or right) in the video.

[Figure 81]

Without Visual Simulations

[Figure 82]

[Figure 83]

Question

Choose the correct missing frame.

[Figure 84]

###### A B C

Figure 12: Examples of Temporal Frame Reasoning under “without Visual Simulations" Evaluation Setting.

###### F.3 Visualizations of Perception Probing Questions

In Figure 5, Claude demonstrates a perceptual error: while it correctly identifies all face colors, it incorrectly perceives face 6 to be positioned beneath face 4, when it is actually located beneath face 5. Such errors prompt an important question regarding task performance: for challenging tasks like cube net folding, to what extent does the low performance stem from perceptual inaccuracies rather than deficiencies in simulation capabilities or an inability to correctly interpret simulation outcomes? We design probing questions to evaluate model performance 2D and 3D perception on cube nets (Figure 13), which reveals that model fail substantially on 3D perception (Table 2), which may be the main bottleneck in understanding intermediate visualizations in cube net folding (Table 1).

2D Perception (Color)

|Question: In the cube net shown in the image, is face 2 directly connected to face 3, yes or no?<br><br>2D Perception (Connectivity)|
|---|

|Question: What color is face 5 in the cube net shown in the image?<br><br>A. Light orange<br>B. Yellow<br>C. Light purple<br>D. Light red<br>|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|Question: In the cube net folding process shown below, has face 6 been folded towards 1 yet, yes or no?<br><br>3D Perception (Folded?)|
|---|

Figure 13: Exemplary questions on cube nets to probe model performance on 2D and 3D perception.

###### F.4 Visualizations of STARE Task in Different Representations

Figures 14–17 provide concrete examples of the input modalities evaluated in STARE. For every task family we visualize the image-only variant (the original format in STARE), the text-only variant (compact symbolic description that can be consumed without vision), and—where applicable—the combined image+text variant that concatenates the two.

- • 2D and 3D transformations. In the text-only panels, each object is serialized as <shape>, <color>, <x,y>, <size>, with attributes separated by commas (e.g., “square, red, (3, 4), 2”). The image+text panels place the same textual description beneath the image, so that language and vision can be attended to jointly.
- • Cube-net folding. We flatten the cube into a 2D grid and enumerate its faces from 1 to 6. The text-only representation thus becomes a short digit string (e.g., “123456”) or a block array that mirrors the spatial arrangement of the net.
- • Tangram puzzle. Because rotations in the image cannot be expressed succinctly in the image+text setting, we show only image-only and text-only variants. Each piece is labeled alphabetically and encoded by a binary occupancy grid—rows of “1” indicate filled cells, yielding a representation that is both human-readable and unambiguous for MLLMs.

Together, these examples clarify the correspondence between the natural visual stimuli and the stripped-down symbolic forms used in our text-only experiments, as introduced in Section 3.3.

Imagine transforming a triangle step by step. Follow these changes:

[Figure 90]

###### Image+Text

• Shift the triangle to the left by a significant amount and

[Figure 91]

slightly upward.

Imagine transforming a triangle step by step. Initially, you see a triangle located at (0.00, 0.00), 54.4 × 51.9 units, RGB(0.03, 0.32, 0.97).

[Figure 92]

After these transformations, which of the following shapes best represents the final result? For reference, the black dots in each panel of the figures indicate the origin.

[Figure 93]

Follow these changes:

• Shift the triangle to the left by a significant amount and

slightly upward.

After these transformations, which of the following shapes best represents the final result? For reference, the black dots in each panel of the figures indicate the origin.

[Figure 94]

###### Text-only

[Figure 95]

Imagine transforming a triangle step by step. Initially, you see a triangle located at (0.00, 0.00), 54.4 × 51.9 units, RGB(0.03, 0.32, 0.97). Follow these changes:

The answer choices are:

- A. a triangle located at (-3.29, -1.24), 23.3 × 30.6 units, RGB(0.03, 0.32, 0.97).
- B. a triangle located at (-30.00, 10.00), 54.4 × 51.9 units, RGB(0.03, 0.32, 0.97).
- C. a triangle located at (-6.33, 2.58), 45.9 × 57.8 units, RGB(0.2, 0.2, 0.18).
- D. a triangle located at (8.00, 2.57), 54.4 × 51.9 units, RGB(0.95, 0.22, 0.67).

• Shift the triangle to the left by a significant amount and slightly upward.

After these transformations, which of the following shapes best represents the final result?

The answer choices are:

- A. a triangle located at (-3.29, -1.24), 23.3 × 30.6 units, RGB(0.03, 0.32, 0.97).
- B. a triangle located at (-30.00, 10.00), 54.4 × 51.9 units, RGB(0.03, 0.32, 0.97).
- C. a triangle located at (-6.33, 2.58), 45.9 × 57.8 units, RGB(0.2, 0.2, 0.18).
- D. a triangle located at (8.00, 2.57), 54.4 × 51.9 units, RGB(0.95, 0.22, 0.67).

- Figure 14: Visualizations of 2D transformations (w/ text instructions) in different representations (upper left: image-only, lower left: text-only, right: image+text).

Image-only

[Figure 96]

Imagine transforming a pyramid step by step. Follow these changes: • Move the object up After these transformations, which of the following shapes best represents the final result? For reference, the black dots in each panel of the figures indicate the origin.

Text-only

[Figure 97]

Imagine transforming a pyramid step by step. Initially, you see a large pyramid located at (0.00, 0.00, 0.80), size large, red made of MyMetal. Follow these changes:

• Move the object up.

After these transformations, which of the following shapes best represents the final result?

The answer choices are:

- A. a large pyramid located at (0.00, 0.00, 0.80), size large, red made of Material_0.
- B. a large cylinder located at (0.00, 0.00, 1.09), size large, red made of Material_0.
- C. a large torus located at (0.00, 0.00, 0.20), size large, red made of Material_0.
- D. a large pyramid located at (0.00, 0.00, 1.80), size large, red made of Material_0.

Image+Text

[Figure 98]

Imagine transforming a pyramid step by step.

Initially, you see a large pyramid located at (0.00, 0.00, 0.80), size large, red made of MyMetal.

Follow these changes: • Move the object up

The answer choices are:

- A. a large pyramid located at (0.00, 0.00, 0.80), size large, red made of Material_0.
- B. a large cylinder located at (0.00, 0.00, 1.09), size large, red made of Material_0.
- C. a large torus located at (0.00, 0.00, 0.20), size large, red made of Material_0.
- D. a large pyramid located at (0.00, 0.00, 1.80), size large, red made of Material_0.

After these transformations, which of the following shapes best represents the final result? For reference, the black dots in each panel of the figures indicate the origin.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

- Figure 15: Visualizations of 3D transformations (w/ text instructions) in different representations (upper left: image-only, lower left: text-only, right: image+text).

Check out a net with 6 faces below:

[Figure 104]

###### Image+Text

Here are the steps to fold the net with face 4 as the base:

[Figure 105]

Check out a net with 6 faces below:

- Step 1: Fold face 2 upwards
- Step 2: Fold face 5 upwards
- Step 3: Fold face 3 inwards towards face 2
- Step 4: Fold face 6 downwards towards face 2
- Step 5: Fold face 1 inwards towards face 4 Based on the above steps, can the net be folded to form a cube, yes or no?

[Figure 106]

The net above can be represented the following in text:

[Figure 107]

###### Text-only

Here are the steps to fold the net with face 4 as the base:

[Figure 108]

Check out a net with 6 faces below:

- Step 1: Fold face 2 upwards
- Step 2: Fold face 5 upwards
- Step 3: Fold face 3 inwards towards face 2
- Step 4: Fold face 6 downwards towards face 2
- Step 5: Fold face 1 inwards towards face 4 Based on the above steps, can the net be folded to form a cube, yes or no?

[Figure 109]

- Step 1: Fold face 2 upwards
- Step 2: Fold face 5 upwards
- Step 3: Fold face 3 inwards towards face 2
- Step 4: Fold face 6 downwards towards face 2
- Step 5: Fold face 1 inwards towards face 4 Based on the above steps, can the net be folded to form a cube, yes or no?

- Figure 16: Visualizations of cube net folding in different representations (upper left: image-only, lower left: text-only, right: image+text).

Image-only Text-only

[Figure 110]

[Figure 111]

Check out a Tangram puzzle below.

Tangram-Style Text Puzzle Can you use all of the following pieces—rotating or flipping them as needed—to exactly fill a 3x3 square without overlaps or gaps?

[Figure 112]

Target board: a 3x3 grid Available pieces:

- Piece A: 11
- Piece B: 11 11
- Piece C: 111 For each piece, the grid cells it occupies are marked with 1s and the rest with 0s. Instructions:

The left panel is an empty Tangram puzzle, while the right panel shows available pieces to complete the puzzle. Keep in mind that you can rotate or flip the pieces. Can the Tangram puzzle be completed with the available pieces, yes or no?

- • You must place every cell of each piece somewhere in the grid.
- • Pieces may be rotated in 90° increments or flipped horizontally/vertically. Question: Is it possible to fill the 3x3 board exactly with these pieces? (Yes or No)

Figure 17: Visualizations of tangram in different representations (left: image-only, right: text-only).

###### F.5 Case Study

- Figure 18 presents error cases on 2D transformation tasks from o1 and GPT-4o. The o1 explanation correctly identifies that a regular hexagon rotated 30° counter-clockwise will have a vertex pointing straight up, so it chooses panel B, but its justification overlooks potential differences in size, color, or shape variants among the answer choices. GPT-4o, meanwhile, mistakenly claims the hexagon looks “exactly the same” after a 30° rotation (confusing 60° symmetry with 30°) and vacillates between panels A and C, revealing it doesn’t fully grasp the visual outcome of the specified rotation.

[Figure 113]

[Figure 114]

###### GPT-4o Error on 2D Transformation

###### O1 Error on 2D Transformation

[Figure 115]

GPT-4o: To solve this problem, we need to visualize a transformation of a hexagon through rotation:

|Question: Imagine transforming a hexagon step by step. Follow these changes:<br><br>- Rotate the hexagon by 30 degrees counter-clockwise around its center.<br><br>After these transformations, which of the following shapes best represents the final result?|
|---|

[Figure 116]

- 1. **Initial Hexagon**: Start with a regular hexagon centered at the origin (black dot).
- 2. **Rotation by 30 Degrees Counter-Clockwise**:

- - A regular hexagon has six sides and rotational symmetry through angles of \(60^\circ\) (each internal angle is \(120^\circ\)).
- - Rotating the hexagon by \(30^\circ\) counter-clockwise will result in the hexagon looking exactly the same due to its symmetry.
- - Essentially, each vertex is shifted to the position of its neighboring

vertex. After considering these steps, the shape and size of the hexagon remain unchanged, just its orientation. Since it’s a regular	hexagon, it will look the same after	a \(30^\circ\) rotation. Now referring to the images provided:

- - **(A)**: This looks identical to a regular hexagon, which could match the rotated state of the hexagon due to its symmetry.
- - **(B)**: This appears to be a distorted pentagon, not a suitable match.
- - **(C)**: This also looks like a regular hexagon but in purple.
- - **(D)**: This looks like a smaller version of (C). Assuming the only transformation was a rotation, neither the size nor shape should change beyond orientation.

[Figure 117]

[Figure 118]

###### A B C D

o1: Solution Sketch:

- 1. A regular hexagon in the “flat-top” orientation (as in panel A) has a horizontal edge along the top.
- 2. Rotating it \(30^\circ\) counterclockwise around its center puts one of its vertices directly upward, without changing its size.
- 3. Among the choices given, panel B shows precisely that same hexagon reoriented by \(30^\circ\), matching both the new orientation and the same overall size.

[Figure 119]

[Figure 120]

Based on these considerations, the best option that represents a regular	hexagon after	rotation is both (A) and (C). Given that the original hexagon is most likely the same size and color	as shown in (A) based on typical transformation descriptions, the most direct match without size or color changes mentioned would be (A).

Therefore, the final shape after the specified rotation is the one shown in panel B.

Error	Analysis:

Error	Analysis:

Model only focuses on the detail “one of its vertices directly upwards” while ignore the differences in size, color and shape.

Model does not know what the shape would look like after 30 degree rotation.

Figure 18: Error cases of o1 and GPT-4o on 2D transformation.

- Figure 19 presents error cases on 3D transformation tasks from o1 and GPT-4o. The o1 response accurately places the torus up and left of the origin and selects panel A, but it overlooks that material/appearance differences could also rule out that choice. GPT-4o correctly sees that both panels A and D share the required displacement, yet it arbitrarily favors D, giving a justification (“clearer displacement”) unrelated to the stated transformation criteria, so its final selection is wrong despite partially sound reasoning.

[Figure 121]

[Figure 122]

###### GPT-4o Error on 3D Transformation

###### O1 Error on 3D Transformation

[Figure 123]

[Figure 124]

|Question: Imagine transforming a torus step by step. Follow these changes:<br><br>- Move the object to the left along the y-axis and up<br><br>After these transformations, which of the following shapes best represents the final result?|
|---|

GPT-4o: To solve this problem, we need to perform the given transformations on the torus and determine the final position.

[Figure 125]

- 1. Initial State: The torus is at the origin, as indicated by the black dot.
- 2. Transformation:

- - Move the torus to the left along the y-axis: In standard coordinate systems, moving leftwards means decreasing the y-coordinate. So, the torus is moved negatively along the y-axis.
- - Move the torus up: This involves increasing the z-coordinate, moving the torus upward.

- 3. Analysis of Options:

[Figure 126]

- - A: The torus is positioned to the left and up, which matches the

transformation requirements.

- - B: The torus is directly at the origin, which does not match any

transformations.

- - C: The torus is at the origin, which does not match any

transformations.

- - D: The torus is positioned both to the left and up, which matches

###### A B C D

o1: Step by step:

the transformation requirements. Both A and D match the transformations described. However, option D shows a clearer	displacement compared to A for	both axes. Therefore, after considering these movements: D

- • “Left along the y-axis” (as shown in the panels) places the torus to the viewer’s left of the black dot.
- • “Up” raises it above that black dot.
- • Among the choices, only panel A shows the torus both leftward and above the origin.

[Figure 127]

[Figure 128]

Error	Analysis:

Error	Analysis:

Model correctly recognizes A and D are with the same transformation, but chooses D for the wrong reason.

Model only focuses on the detail “the torus both leftward and above the origin” while ignore the differences in material

Figure 19: Error cases of o1 and GPT-4o on 3D transformation.

On cube net folding task, besides the perception error from Claude in Figure 5 and the text simulation error from GPT-4o in Figure 1, when provided with intermediate visual simulation, models like GPT4o still struggles with understanding the intermediate visual cues. In Figure 20, GPT-4o asserts that face 1 becomes the cube’s top surface, whereas face 1 is actually enclosed between faces 6 and 3. This misrepresentation of face adjacency highlights GPT-4o’s difficulty in maintaining accurate 3D spactial cognition, which led to the wrong final result. This observation aligns with the conclusion from the perception probing test on cube nets in Section 3.3, where GPT-4o fails substantially on 3D perception than 2D perception.

[Figure 129]

###### Fail to Understand Visual Simulation

|Question: Here are the steps to fold the net with face 2 as the base:<br><br>Step 1: Fold face 6 upwards<br>Step 2: Fold face 3 downwards<br>Step 3: Fold face 1 inwards towards face 6<br>Step 4: Fold face 4 downwards towards face 1<br>Step 5: Fold face 5 downwards towards face 1 Based on the above steps, can the net be folded to form a cube, yes or no?<br>|
|---|

[Figure 130]

[Figure 131]

Intermediate Visual Simulation:

[Figure 132]

[Figure 133]

[Figure 134]

GPT-4o: To determine if the net can be folded into a cube, let's verify each step:

- 1. Initial Structure:

- - Faces touching face 2: 1, 3, 6.
- - Stand up 6 and 3 makes them upright walls connected to face 2.
- - Faces 4 and 5 are connected to face 6.

- 2. Step 1: Fold face 6 upwards:

- - Face 6 stands up next to face 2.
- - Faces 4 and 5 are now connected to face 6 above face 2.

- 3. Step 2: Fold face 3 upwards:

- Face 3 stands up opposite face 6 and next to face 2.

- 4. Step 3: Fold face 1 inwards towards face 6:

- Face 1 moves inward and is now covering part of the top of the cube-like structure.

- 5. Position at this step:

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

###### Error	Analysis:

Face 1 is not covering the top of the cube, rather connected to both face 6 and face 3.

- - Face 6 is vertical.
- - Face 3 is vertical opposite face 6.
- - Face 1 is starting to cover	the top.

[Figure 144]

… The final answer is: \(\boxed{\text{Yes}}\)

- Figure 20: A error case from GPT-4o on cube net folding, where the model fails to understand the intermediate visual simulation.

For tangram puzzles, the left example of Figure 21 shows how Claude takes a shortcut by conducting an area-based feasibility check, correctly noting that the 4 × 4 board contains 16 unit squares, but it erroneously counts the L-shaped piece C as 5 rather than its actual 7 squares. This miscalculation reduces the summed piece area to 14 instead of the correct 16, leading to conclude—incorrectly—that the puzzle cannot be completed. The right example of Figure 21 shows how Claude fails to reason about the piece positions when given clear step-by-step instructions. Claude correctly interprets the coordinate instructions and board dimensions, yet its spatial reasoning falters when projecting how the rotated pieces occupy the grid. It underestimates the extents of pieces B and D, inventing overlaps and gaps that never arise, and therefore erroneously concludes the puzzle cannot be completed. This error stems from an inaccurate mental simulation of spatial relationships.

[Figure 145]

[Figure 146]

[Figure 147]

###### Perception Error

###### Fail to Simulate in Text

|Question: Can the Tangram puzzle be completed with the available pieces, yes or no?|
|---|

|Question: Below are the steps to complete the Tangram puzzle:<br><br>Step 1: Rotate piece B by about 90 degrees clockwise, and place<br><br>piece B with its upper-left corner at (x, y) = (1, 3).<br><br>Step 2: Rotate piece C by about 90 degrees clockwise, and place<br><br>piece C with its upper-left corner at (x, y) = (0, 1).<br><br>Step 3: Rotate piece A by about 90 degrees clockwise, and place piece A with its upper-left corner at (x, y) = (0, 3). Step 4: Rotate piece D by about 90 degrees clockwise, and place<br><br>piece D with its upper-left corner at (x, y) = (1, 1). Based on the above steps, can the Tangram puzzle be completed with the available pieces, yes or no?<br><br><br>|
|---|

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

The4x4Tangram Board

The 3x3 Tangram Board

- 0

- 1

- 2

- 3

- 4

- 0 1 2 3

- 1

- 2

- 3

[Figure 153]

Claude: Let me solve this step by step.

- 1) First, let's observe the puzzle and pieces:

- * The board is 4×4
- * We have 3 pieces: A, B, and C
- * Piece A is a 1×3 rectangle (coral color)
- * Piece B is a 2×3 rectangle (pink)
- * Piece C is a 2×3 L-shaped piece (purple)

- 2) Let's calculate the total area:

- * The board is 4×4 = 16 squares
- * Piece A: 1×3 = 3 squares
- * Piece B: 2×3 = 6 squares
- * Piece C: 2×3 - 1 = 5 squares
- * Total area of pieces: 3 + 6 + 5 = 14 squares

- 3) Since the board has 16 squares but our pieces only cover 14 squares, it's impossible to complete the puzzle. We would have 2 squares left uncovered no matter how we arrange the pieces.
- 4) Even if we try different rotations and arrangements, we can never cover all 16 squares with pieces that sum to only 14 squares. \boxed{No}

- * The board is 4×

× ×

× -

- * The board is 4× × ×

0

[Figure 154]

0 1 2 3 4

Available Pieces

B: 2x2

Available Pieces

[Figure 155]

[Figure 156]

A: 2x1

B

[Figure 157]

[Figure 158]

A

Claude: Let me solve this step by step.

D: 1x2

C

C

- 1) First, let's understand the dimensions:

- * The board is 3×3
- * Total area needed to be filled = 9 square units

- 2) Let's calculate the total area of the available pieces:

- * Piece A (2×1) = 2 square units
- * Piece B (2×2) = 4 square units
- * Piece C (1×1) = 1 square unit
- * Piece D (1×2) = 2 square units
- * Total area of pieces = 9 square units
- * This matches the board area, which is promising

- 3) Let's follow the steps and check if the pieces overlap or leave gaps:

- * After	Step 1: Piece B takes up positions (1,1)-(2,2)
- * After Step 2: Piece C takes up position (0,1)
- * After Step 3: Piece A takes up positions (0,2)-(0,3)
- * After	Step 4: Piece D takes up positions (1,0)-(2,0)

- 4) Checking for overlaps and gaps:

- * There are overlaps between some pieces
- * For example, Piece B and Piece A would overlap
- * Also, some parts of the 3×3 grid would remain unfilled

- 5) Therefore, following these specific placement instructions:

C: 1x1

× -

[Figure 159]

[Figure 160]

Error	Analysis: After	step 1: Piece B takes up position (1,1)

###### Error	Analysis:

Piece C contain 2x3 +

– (3,3)

- 1 = 7 squares

- 0 1 2 3

- 1

- 2

- 3

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

C

C

[Figure 165]

[Figure 166]

[Figure 167]

0

[Figure 168]

After	step 4: Piece D takes up position (1,0)

– (3,1)

- 0 1 2 3

- 1

- 2

- 3

[Figure 169]

- * The pieces will overlap
- * Not all squares will be covered
- * The puzzle cannot be completed as instructed

[Figure 170]

0

\boxed{\text{No}}

- Figure 21: Left: A perception error case from Claude on tangram puzzle. Right: An error case from Claude on tangram puzzle, which failed to simulate the intermediate steps even when step-by-step instructions are given.
- Figure 22 presents two error cases from Claude on temporal frame reasoning. In the left example, Claude correctly inferred the camera’s left-to-right movement across the given frames, yet it mis-evaluated the viewpoints depicted in the answer choices and consequently selected the wrong completion frame. In the right example, the model erred even earlier, misconstruing the direction of camera motion itself; this foundational misinterpretation then led to an incorrect choice despite a seemingly systematic rationale.

[Figure 171]

[Figure 172]

##### Claude Error on Temporal Reasoning

##### Claude Error on Temporal Reasoning

[Figure 173]

[Figure 174]

|Question: Which image correctly fills the missing frame?|
|---|

|Question: Which image correctly fills the missing frame?|
|---|

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

###### A B C

A B C

Claude: Let me solve this step by step:

Claude: Let me solve this step by step:

- 1) Looking at <Image 1>, I can see a bottle on a wooden floor being photographed from different angles.
- 2) Analyzing the motion:

- - Frame 1: Bottle is viewed more from the left
- - Frame 2: Camera moves more to the right
- - Frame 3: Camera continues moving right
- - Frame 4: Missing frame (should show continued rightward motion)

- 3) Looking at the options in <Image 2>:

- - Option A: Shows a view from the left side
- - Option B: Shows a middle/centered view
- - Option C: Shows a view from the right side

- 4) Since the camera motion in the original sequence moves from left to right, and we need to continue this motion after frame 3, we need the view that shows the bottle from the furthest right angle.
- 5) Option C shows the bottle from the rightmost angle, which would maintain the consistent rightward camera movement of the sequence.

- 1) Looking at the first 3 frames in Image 1:

- - Frame 1: Shows a table and chair from one angle
- - Frame 2: Camera moves slightly to the right, showing similar	view
- - Frame 3: Camera continues moving right, showing more of the chair

- 2) The missing frame (marked with '?') should be the last frame in this sequence, continuing the right-moving motion.
- 3) Looking at options in Image 2:

- - Option A: Shows the initial view (same as first frame)
- - Option B: Shows a middle view of the chair
- - Option C: Shows the chair from furthest right angle

- 4) Since the camera is moving right consistently:

- - The missing frame should show the chair from the most rightward angle
- - This would complete the smooth motion from left to right
- - Option C shows exactly this view

- 5) Option C is the only one that would logically complete the sequence, showing the final rightmost view of the chair.

[Figure 183]

[Figure 184]

Error	Analysis:

Error	Analysis:

Model correctly recognizes the camera motion, however fail to interpret the camera view angle of the answer choices.

Model fail to recognize the camera motion of the video.

Figure 22: Error case on temporal frame reasoning.

- Figure 23 presents an error for perspective reasoning from GPT-4o. GPT-4o misprojects the agent’s top-down pose into egocentric space: it assumes the arrowed direction is pointing towards a curved seating area and therefore selects option A. A correct geometric mapping should conclude that the agent would instead stand next to the curved seating area. This error underscores the model’s difficulty in reason about perspective changing in 3D space.

[Figure 185]

#### GPT-4o Error on Perspective Reasoning

[Figure 186]

|Question: The image is from the topdown view of a scene with a red square indicating an agent and a red arrow indicating the agent's direction of view.<br><br>Which option best represents the agent's view?|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

###### A B C D

GPT-4o: Let me solve this step by step:

- 1) Looking at Image 1, I see a top-down view of what appears to be an interior space, with a red square and arrow indicating a viewpoint and direction.
- 2) The arrow appears to be pointing towards what looks like a curved or	semi-circular	seating area.
- 3) Looking at the four options in Image 2:

- - Image A shows a curved seating area with a view into another room
- - Image B shows a central area with a rug
- - Image C shows a curved seating area with columns
- - Image D shows a straight seating area near a wall

- 4) From the agent's position and direction in Image 1, they would be looking directly at a curved seating arrangement, with a view into another space beyond.
- 5) Comparing this expected view with the options, Image A most closely matches what we would expect to see from that position and direction, showing a curved seating area with a view through to another space.

[Figure 192]

Error	Analysis:

Model simply cannot understand the visual details from the top-down view image.

Figure 23: Error case on perspective reasoning.

## G Detailed Analysis Results

Correlation Analysis between Synthetic tasks and Real tasks. In Section 3.2, we briefly discussed the correlation between averaged model performance on synthetic tasks (including 2D transformation, 3D transformation, cube net folding and tangram puzzle) and that on real-world tasks (including temporal frame reasoning and perspective reasoning). Figure 24 shows the averaged model performance on synthetic and real-world tasks across 11 models and the fitted line with correlation coefficient r ≈ 0.88.

[Figure 193]

Claude 3.5 Sonnet o1

GPT-4o

Gemini-2.0 Flash

Gemini-2.0 Flash Think

LLaVA-OneVision-72B

Qwen-2.5-VL-7B

Qwen-2.5-VL-72B InternVL-2.5-78B

Qwen-2.5-VL-3B

Figure 24: Correlation between model performance on synthetic tasks and that on real-world tasks.

Note that for open-source models, the real-world task performance is close to random guessing (29%). Removing the open-source models, the correlation coefficient decreased to r ≈ 0.58, still showing a weak but positive correlation between sythetic task performance and real-world task performance.

Model Performance on 2D/3D Individual Transformation Types. Table 8 presents model accuracy across 2D visual analogy and text instruction tasks. Across the nine subtasks, adding visual simulation lifted accuracy for every model except in a few narrow cases, and the size of the gain correlates strongly with baseline capability. Closed-source leaders that were already solid on the raw pixel tasks—o1 (∼ +3 points overall) and GPT-4o (∼ +8 points)—were pushed into the mid-80 s and low-90 s, effectively reaching ceiling on the text-instruction variants, where gains were biggest (e.g., GPT-4o jumps +25 points on “Reflection” and +18 points on both “Rotation” and “Translation”). Mid-tier proprietary models such as Gemini 2.0 Flash (∼ +5 points) and its “Flash Thinking” mode (∼ +5.5 points) benefited even more on instructions than on analogies, narrowing the gap to GPT-4-class systems. Open-source vision-language models lag a full generation behind—the best of them (InternVL 2.5-78B) still sits below 55% on average after simulation—but they, too, record healthy boosts of 6–12 points, chiefly on the analogy side. The lone regression is GPT-4o’s –5 pt dip on “Reflection” analogies, suggesting that simulation may occasionally overwrite a correct latent heuristic. Overall, the pattern indicates that visual simulation chiefly helps models convert verbal transformation instructions into precise spatial operations, while stronger base perception/reasoning models harvest the largest absolute improvements and approach human-like proficiency.

- Table 9 presents model accuracy across 3D visual analogy and text instruction tasks. Visual simulation gives 3D spatial reasoning a measurable—but more uneven—boost than in 2D: averaged over all eight subtasks, every proprietary model gains between ∼+2 points (GPT-4o, o1) and +6 points (Claude-3.5 Sonnet, Gemini-Flash Thinking), while the open-source field improves by ∼

+4–7 points—except InternVL, which slips a point. Gains concentrate in the conceptually harder operations: across models, Shearing (both analogy +6.6 points and instruction +6.6 points) and Rotation-instruction ( +6.4 points) see the largest lifts, whereas Translation under visual analogy actually falls slightly (–0.9 points), echoing a smaller 2D reflection dip. Even after simulation, closed-source leaders plateau in the high-60s to mid-70s on most 3D subtasks—roughly 15 points below their 2D ceilings—indicating that depth-aware transformations remain a major bottleneck. Open-source VL models still trail a full generation (≤45% average), but their sharper relative gains suggest they, too, leverage synthetic roll-outs to bridge language and geometry.

2D Transformations w/ Visual Analogy 2D Transformations w/ Text Instruction

Model

Reflection Rotation Shearing Scaling Translation Reflection Rotation Scaling Translation Without Visual Simulation

GPT-4o 82.1 69.8 53.7 88.5 72.0 65.8 67.8 90.6 73.3 Claude-3.5 Sonnet 75.0 60.9 55.8 87.4 71.2 63.8 58.9 85.9 66.5 Gemini2.0 Flash 85.7 63.8 51.0 84.4 71.4 65.8 62.3 88.4 70.3 Gemini2.0 Flash Thinking 52.4 48.9 46.9 71.9 55.1 63.2 60.6 83.0 67.8 o1 92.9 70.7 59.2 83.3 84.0 89.5 78.1 92.2 92.2 LLaVA-OneVision 7.1 25.9 32.7 24.4 25.4 31.7 33.1 51.0 34.6 Qwen2.5-VL-72B 57.1 38.8 34.7 64.4 42.3 29.3 49.6 62.5 38.8 InternVL2.5-78B 35.7 41.4 34.7 45.6 36.6 41.5 51.1 75.0 51.9

With Visual Simulation

GPT-4o 76.9 72.8 54.8 91.9 80.0 91.2 86.0 93.2 91.5 Claude-3.5 Sonnet 73.1 70.9 50.0 85.5 73.9 55.9 72.9 83.8 73.9 Gemini2.0 Flash 73.1 70.9 59.5 85.5 74.5 79.4 74.8 90.5 78.2 Gemini2.0 Flash Thinking 61.5 68.2 40.5 71.0 56.4 70.6 68.2 89.2 73.9 o1 80.8 80.6 54.8 87.1 84.2 100 93.5 94.6 97.6 LLaVA-OneVision 15.4 30.1 31.0 30.6 24.8 20.6 41.1 48.6 33.9 Qwen2.5-VL-72B 65.4 56.3 35.7 71.0 57.0 41.2 40.2 60.8 39.4 InternVL2.5-78B 69.2 43.7 33.3 59.7 47.3 50.0 53.3 73.0 53.9

- Table 8: Model Performance With or Without Visual Simulation across 2D Transformation types in Visual Analogy and Text Instruction Tasks.

3D Transformations w/ Visual Analogy 3D Transformations w/ Text Instruction Rotation Shearing Scaling Translation Rotation Shearing Scaling Translation

Model

Without Visual Simulation GPT-4o 60.7 55.7 76.0 80.1 60.1 46.9 71.1 71.2

- Claude-3.5 Sonnet 50.0 46.2 63.3 62.6 45.9 40.4 55.6 53.4 Gemini2.0 Flash 54.2 53.9 63.3 73.0 55.86 44.44 61.90 51.63 Gemini2.0 Flash Thinking 42.4 43.6 61.5 63.8 37.8 32.7 52.5 55.7 o1 65.6 58.1 76.7 85.6 61.3 46.3 70.5 73.9

LLaVA-OneVision 18.8 29.1 28.9 25.3 27.0 19.4 41.0 30.7 Qwen2.5-VL-72B 36.5 40.2 61.1 46.6 36.9 33.3 47.6 45.1 InternVL2.5-78B 31.2 30.8 51.1 37.4 37.8 32.4 60.0 40.5

With Visual Simulation GPT-4o 64.3 64.3 78.2 76.0 62.6 54.7 75.3 68.5

- Claude-3.5 Sonnet 51.2 59.5 69.2 59.7 55.6 48.0 64.5 59.3 Gemini2.0 Flash 46.4 64.3 62.8 68.2 60.9 49.5 64.9 56.1 Gemini2.0 Flash Thinking 50.0 47.6 60.3 66.7 48.5 46.7 59.1 59.3 o1 63.1 63.1 76.9 79.8 69.7 50.7 79.6 74.1

LLaVA-OneVision 27.4 28.6 32.1 29.5 27.3 26.7 45.2 35.2 Qwen2.5-VL-72B 46.4 54.8 69.2 55.0 45.5 34.7 48.4 46.3 InternVL2.5-78B 31.0 28.6 43.6 32.6 43.4 37.3 57.0 40.7

- Table 9: Model Performance With or Without Visual Simulation across 3D Transformation types in Visual Analogy and Text Instruction Tasks.

Task complexity vs. performance. Table 11 presents model performance across different task difficulties for 2D and 3D transformations. Adding visual simulation helps most when tasks get tougher, but the effect differs by setting. For 2D text instructions tasks, we observe big boost – closed-source models jump about 10-20 points on medium and hard tasks, often hitting 90%+. For 2D visual analogy tasks, we observe smaller lift—several points on easy, up to 10˜ points on medium/hard. For 3D tasks, only a few-point gain, and some models slip on the hardest visual analogy tasks, showing 3D reasoning is still hard. Open-source MLLMs stay well behind; their scores move up and down unpredictably, meaning they haven’t yet learned to use the simulated views well.

- Table 10 presents model performance across different number of transformation steps for 2D and 3D transformations. Models struggle more as the number of transformation steps grows, and visual simulation mainly fixes that. Without simulation, accuracy often peaks at one or two steps and drops at three—especially in 3D visual-analogy, where GPT-4o falls from 73% (N = 2) to 49% (N = 3). When simulation is added, scores for the multi-step cases (N = 2–3) jump 10–15 points for the top proprietary systems and a few points for open-source ones, erasing most of the earlier decline in 2D tasks and cutting the 3D drop roughly in half. Single-step problems were already easy for the best models and see little change. Overall, simulation is most useful for longer, instruction-driven chains of transforms, while depth-heavy 3D sequences remain the hardest setting.

2D Visual Analogy 2D Text Instruction 3D Visual Analogy 3D Text Instruction N=1 N=2 N=3 N=1 N=2 N=3 N=1 N=2 N=3 N=1 N=2 N=3

Model

Without Visual Simulation GPT-4o 60.46 74.84 73.86 67.27 77.56 73.55 62.75 73.37 48.69 63.07 63.40 60.78 Claude-3.5 Sonnet 63.73 75.82 65.69 65.17 65.02 60.61 45.10 57.35 57.35 50.98 55.23 45.75 Gemini2.0 Flash 64.71 73.53 68.53 63.96 76.24 70.25 61.76 60.78 63.73 46.08 56.86 56.86 Gemini2.0 Flash Thinking 54.58 52.94 55.56 61.71 67.33 71.07 47.71 53.92 57.19 45.59 50.00 20.59 o1 66.7 81.4 82.4 82.0 89.1 89.3 66.67 72.55 77.45 61.76 66.67 62.75 LLaVA-OneVision 30.39 26.47 24.51 49.57 33.70 31.53 25.49 28.43 22.55 30.39 30.39 24.51 InternVL2.5-78B 43.14 34.31 42.16 61.74 52.17 50.45 40.2 29.41 36.27 34.31 48.04 40.2 Qwen2.5-VL-72B 50.00 45.10 41.18 55.65 36.96 40.54 48.04 42.16 45.10 38.24 43.14 41.18 With Visual Simulation

GPT-4o - 78.43 73.53 - 88.04 90.57 - 70.59 72.55 - 61.76 68.63 Claude-3.5 Sonnet - 70.59 70.59 - 71.74 72.64 - 56.86 57.84 - 65.69 50.98 Gemini2.0 Flash - 69.6 73.5 - 80.43 77.40 - 61.76 59.80 - 61.76 53.92 Gemini2.0 Flash Thinking - 46.08 58.82 - 79.35 67.92 - 55.88 60.78 - 53.92 53.92 o1 - 73.4 85.3 - 94.6 97.2 - 70.6 75.5 - 70.6 69.6

LLaVA-OneVision - 30.39 25.49 - 38.04 34.91 - 28.43 28.43 - 36.27 29.41 InternVL2.5-78B - 39.22 51.96 - 56.52 52.83 - 25.49 35.29 - 46.08 39.22 Qwen2.5-VL-72B - 51.96 58.82 - 43.48 41.51 - 49.02 58.82 - 47.06 43.14

- Table 10: Model Performance With or Without Visual Simulation across number of transformation steps (N) in 2D/3D Visual Analogy and Text Instruction Tasks.

Model

2D Visual Analogy 2D Text Instruction 3D Visual Analogy 3D Text Instruction easy medium hard easy medium hard easy medium hard easy medium hard

Without Visual Simulation GPT-4o 80.4 67.3 61.4 76.2 70.4 71.3 74.2 64.9 65.7 69.0 63.1 55.2 Claude-3.5 Sonnet 76.5 66.7 62.1 68.7 61.8 59.4 54.9 54.4 50.5 55.6 52.0 44.4

- Gemini 2.0 Flash 78.4 63.7 64.7 75.0 67.2 67.3 67.6 59.8 58.8 56.9 52.9 50.0 Gemini 2.0 Flash Think 66.3 52.3 44.4 65.5 69.4 65.4 54.6 53.9 50.3 48.0 44.7 46.1 o1 83.3 77.5 69.6 90.6 81.1 89.1 78.4 70.6 67.7 69.6 64.7 56.9

LLaVA-OneVision 22.6 32.4 26.5 39.5 46.3 29.2 25.5 20.6 30.4 31.4 28.4 25.5 InternVL 2.5-78B 45.1 40.2 34.3 63.2 50.9 50.0 32.4 34.3 39.2 48.0 37.3 37.3 Qwen 2.5-VL-72B 57.8 40.2 38.2 50.9 41.7 41.7 55.9 40.2 39.2 42.2 38.2 42.2 With Visual Simulation

GPT-4o 80.9 79.4 67.7 91.6 89.4 86.9 80.9 75.0 58.8 75.0 64.7 55.9 Claude-3.5 Sonnet 76.5 72.1 63.2 78.9 65.2 72.1 67.7 52.9 51.5 66.2 57.4 51.5

- Gemini 2.0 Flash 79.4 72.1 63.2 81.7 86.4 67.2 64.7 58.8 58.8 57.4 55.9 60.3 Gemini 2.0 Flash Think 54.4 55.9 47.1 76.1 74.2 68.9 72.1 54.4 48.5 63.2 48.5 50.0 o1 80.9 82.4 75.0 94.4 98.5 95.1 85.3 69.1 64.7 73.5 75.0 61.8

LLaVA-OneVision 36.8 19.1 27.9 39.4 34.9 34.4 26.5 20.6 38.2 45.6 25.0 27.9 InternVL 2.5-78B 57.4 44.1 35.3 64.8 48.5 49.2 23.5 27.9 39.7 55.9 27.9 44.1 Qwen 2.5-VL-72B 72.1 50.0 44.1 59.2 30.3 36.1 63.2 50.0 48.5 47.1 44.1 44.1

- Table 11: Model Performance With or Without Visual Simulation across different difficulty levels in 2D/3D Visual Analogy and Text Instruction Tasks.

- 2D and 3D Perception Probing with Cube Nets. Table 12 presents model performance on 2D and 3D perception probing questions about cube nets, in comparison to the success rate on cube net folding task. The results show that success on cube-net folding is driven by a model’s 3D perception, not its 2D eyesight. All closed-source systems (and several open-source ones) already read colors and 2D face connectivity at or near ceiling, yet their cube-net scores diverge sharply. When we compare cube accuracy ( ✗VSim column) with each perceptual measure, the strongest linear relationship is with the 3D “Folded?” test (Pearson r ≈ 0.89), while 2D connectivity (r ≈ 0.68) and color (r ≈ 0.72) are weaker. Gemini Flash illustrates the pattern: it pairs the top 3D perception score (69%) with the best cube-net performance (65%), whereas GPT-4o and InternVL match its 2D vision but lag 10-20 points on both 3D perception and cube folding. In short, being able to

judge how faces come together in depth—rather than recognizing colors or flat adjacencies—largely determines how well a model can reason about folded cubes.

Model 2D Perception 3D Perception Cube Net Performance

Color Connectivity Folded? ✗VSim ✓Vsim Random 25.0 50.0 50.0 50.5 50.5

Closed-source Models

GPT-4o 100.0 94.1 57.4 52.5 49.1 Gemini-2.0-Flash 100.0 84.9 68.8 65.0 65.5 Gemini-2.0-Flash-Thinking 99.0 49.4 54.3 39.8 62.8

Open-source Models

LLaVA-OneVision 88.0 10.0 22.0 28.5 34.2 InternVL 2.5-78B 92.0 86.0 40.2 43.5 41.0 Qwen 2.5-VL-72B 96.0 81.7 42.1 35.2 53.4

Table 12: 2D and 3D perception performance in cube net folding.

Question-only vs. Question+Steps As shown in Table 13, adding explicit reasoning steps (“Q + Steps") has opposite effects on cube-net tasks for the two model groups: open-source models gain, while closed-source ones do not. The three open-source VL models jump a mean + 20 points on cube nets (driven by LLaVA’s + 40 pts), whereas the five proprietary models average a small decline (-1 pt, with mixed signs). On tangram puzzles, however, the pattern converges: every model—open or closed—drops sharply once reasoning steps are included, with average losses of about -24 pts for closed-source and -19 pts for open-source models. Again, the trivial solution on tangram puzzles would be comparing the total areas of all available pieces and the grid area, which can easily lead to 75% performance. This result suggest that the models cannot leverage explicit text reasoning steps.

Model Cube Nets Tangram Puzzles

Q-only Q+Steps ∆ Q-only Q+Steps ∆

Closed-source Models GPT-4o 50.2 50.4 +0.2 62.4 34.7 -27.7 Claude-3.5 Sonnet 51.5 46.4 -5.1 71.1 41.9 -29.2 Gemini-2.0 Flash 47.4 51.5 +4.1 72.8 59.8 -13.0 Gemini-2.0 Flash Thinking 47.2 49.6 +2.4 42.9 35.3 -7.6 o1 56.0 47.0 -7.0 73.5 29.6 -43.9 Open-source Models

LLaVA-OneVision 0.0 40.5 +40.5 30.3 14.6 -15.7 InternVL 2.5-78B 33.2 41.4 +8.2 69.5 51.7 -17.8 Qwen 2.5-VL-72B 29.0 41.6 +12.6 72.3 47.7 -24.6

- Table 13: Model performance on question-only prompts versus prompts that include explicit reasoning steps (Q+Steps). ∆ values are Q+Steps performance - Q-only performance.

Intermediate Visual Simulation States vs. Performance

- Table 14 summarizes extended results on varying the slice of intermediate visual simulation presented to the model across different tasks. Across models, which slice of the simulation you show matters, and the “best slice” shifts with task type. For 2D transformations, most closed-source models and the stronger open-source one (InternVL) peak when they see only the last intermediate state, gaining 2–6 points over the full roll-out; showing every intermediate frame (“all”) often drags accuracy down a few points. For 3D transformations, the pattern flips—accuracy is usually highest with “all” states (≈ +2–4 points over “partial”), while the last-only view tends to erase that gain, especially for GPT-4o, Gemini Flash, and o1. For cube nets, no single view helps every model. Scores barely change with “all” frames, and last-only often hurts closed-source models (-8 points on average) yet uniquely rescues LLaVA (+11 points). For Tangram puzzles, seeing “all” steps is consistently best: every model but LLaVA jumps 7–24 points versus the partial view, whereas last-only falls back to—or below—the partial baseline. Overall, for more complex tasks, models struggle to leverage intermediate visual states effectively.

###### Model 2D Transformation 3D Transformation Cube Nets Tangram Puzzles

Partial All Last Partial All Last Partial All Last Partial All Last

Closed-source Models GPT-4o 86.8 82.8 89.4 72.1 68.4 68.4 51.3 52.2 35.2 43.5 51.5 43.4 Claude-3.5-Sonnet 67.8 71.4 70.7 54.9 57.8 55.9 58.7 51.6 46.8 43.5 67.6 43.3 Gemini-2.0-Flash 75.4 75.2 79.3 61.0 59.3 57.8 40.5 35.6 41.5 63.8 65.5 58.2 o1 89.3 87.7 93.4 70.1 71.6 65.2 54.4 53.4 45.4 34.8 53.2 46.0 Open-source Models

LLaVA-OneVision 28.3 32.2 31.8 25.5 30.6 29.4 40.2 34.2 45.6 44.9 40.2 39.8 InternVL 2.5-78B 48.3 54.5 56.6 32.3 36.5 40.2 34.7 37.3 37.8 54.3 48.2 41.8 Qwen 2.5-VL-72B 44.4 48.5 44.4 48.7 49.1 43.6 41.9 53.4 42.3 49.0 56.7 44.3

Table 14: Model performance with partial, all, and last intermediate visual simulations.

