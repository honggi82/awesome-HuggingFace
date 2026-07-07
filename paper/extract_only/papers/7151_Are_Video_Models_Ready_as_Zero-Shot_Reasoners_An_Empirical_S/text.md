# arXiv:2510.26802v1[cs.CV]30Oct2025

## Are Video Models Ready as Zero-Shot Reasoners?

### An Empirical Study with the MME-COF Benchmark

Ziyu Guo∗†1, Xinyan Chen∗2, Renrui Zhang∗‡2, Ruichuan An∗3, Yu Qi∗4, Dongzhi Jiang2 Xiangtai Li3, Manyuan Zhang2, Hongsheng Li2, Pheng-Ann Heng1

CUHK 1IMIXR & 2MMLab 3Peking University 4Northeastern University

∗Equal Contribution †Project Lead ‡Corresponding Author

Project Page: https://video-cof.github.io

#### Abstract

Recent video generation models can produce high-fidelity, temporally coherent videos, indicating that they may encode substantial world knowledge. Beyond realistic synthesis, they also exhibit emerging behaviors indicative of visual perception, modeling, and manipulation [70]. Yet, an important question still remains: Are video models ready to serve as zero-shot reasoners in challenging visual reasoning scenarios? In this work, we conduct an empirical study to comprehensively investigate this question, focusing on the leading and popular Veo-3 [21]. We evaluate its reasoning behavior across 12 dimensions, including spatial, geometric, physical, temporal, and embodied logic, systematically characterizing both its strengths and failure modes. To standardize this study, we curate the evaluation data into MME-COF, a compact benchmark that enables in-depth and thorough assessment of Chain-of-Frame (CoF) reasoning. Our findings reveal that while current video models demonstrate promising reasoning patterns on short-horizon spatial coherence, fine-grained grounding, and locally consistent dynamics, they remain limited in long-horizon causal reasoning, strict geometric constraints, and abstract logic. Overall, they are not yet reliable as standalone zero-shot reasoners, but exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models.

#### 1 Introduction

Video models [21, 63, 55, 81, 11], including text-to-video and video-to-text generation models, have made rapid progress in recent years. Thanks to advances in diffusion [75, 7, 84] and autoregressive [36, 76, 16] architectures, current video models can produce high-fidelity videos maintaining consistent object relations and realistic motion dynamics across frames. This suggests that the models may have internalized substantial visual and structural knowledge about the world. Recent research from Google [70] further hints that, such models are evolving beyond pure content generation: Veo-3 [21] has been shown to perform dozens of distinct vision tasks across perception, modeling, manipulation, and reasoning, without any task-specific training. These emergent capabilities have led researchers to posit that video models could serve as unified, generalist vision models, much like large language models (LLMs) [1, 13, 3, 30] have become foundation models for natural language.

Crucially, the sequential nature of video generation provides a new perspective on how such models might reason. Each generated frame builds upon the last, creating a temporal chain of information propagation. This has been dubbed “Chain-of-Frame” (CoF) reasoning [70], an analogy to the chainof-thought (CoT) process in LLMs [69, 35, 82, 23, 79] and their multi-modal variants (MLLMs) [12, 4, 40, 31, 10]. In essence, as a video model generates a sequence of frames, it can iteratively refine and update the scene, thereby working through a problem step-by-step in time and space. This CoF concept suggests that, beyond surface-level pattern generation, general-purpose visual reasoning may emerge from video generative models.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

3D Geometry

Real-world Spatial

Reasoning

Reasoning

Visual Detail

Visual Trace

[Figure 5]

Zero-shot Reasoning?

Reasoning

Reasoning

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Physics-based Reasoning

2D Geometry Reasoning

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Veo Sora Seedance

Kling

...

[Figure 18]

[Figure 19]

Video Models

Table and Chart

Object Counting

Reasoning

Reasoning

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

GUI

Medical

Reasoning

Reasoning

Rotation

Embodied

Reasoning

Reasoning

- Figure 1: Overview of Our Study on the Reasoning Potential of Video Models. We investigate whether state-of-the-art video models exhibit emergent reasoning potentials beyond content synthesis. The analysis spans 12 reasoning dimensions under a unified perspective, exploring whether large-scale video models can serve as zero-shot visual reasoners via CoF reasoning.

However, it remains unclear to what extent current video models truly exhibit reasoning about the content they create. Strong generative performance does not automatically imply robust reasoning potential. Emerging evidence [22, 47, 5, 78] shows that a model may produce coherent videos by learning surface-level patterns in the training data, rather than by internalizing general principles. For instance, a video model can maintain object continuity yet fail to grasp physical plausibility across a long sequence, or it may mimic observed visual sequences without understanding the underlying cause-and-effect relationships. This motivates our central question: Are video models, purely through large-scale visual learning, obtain the zero-shot reasoning potential?

To this end, we present the first empirical study to systematically probe the CoF reasoning capabilities of modern video models, spanning 12 dimensions such as spatial, geometric, physical, temporal, and embodied logic, as detailed in 1. We carry out our analysis on Veo-3, which has been systematically examined as a zero-shot learner in prior work [70]. Our preliminary observations suggest that current leading video models exhibit comparable reasoning patterns, making Veo-3 a representative choice. Our analysis builds on reasoning scenarios distilled from diverse reasoning-oriented benchmarks [25, 67, 45, 71, 29, 34], as well as those we design ourselves, providing a compact yet expressive foundation. The prompts for video models are meticulously crafted by transforming the underlying, textual reasoning process of problem-solving into a clear, video-presentation format. Each case receives a qualitative assessment across three performance levels, i.e., good, moderate, and bad, complemented by a quantitative success rate to measure robustness.

To standardize evaluation, we curate these tasks into the MME-COF benchmark, as illustrated in

- Figure 2 and Section 3.2. Leveraging this benchmark, we measure several state-of-the-art video models, i.e., Veo-3 [21], Sora-2 [56], Kling [38], and Seedance [19], to obtain directly comparable scores and qualitative behaviors across categories. Our investigation reveals that the models exhibit promising reasoning patterns in short-horizon spatial coherence, fine-grained grounding, and con-

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Visual Detail Reasoning

Medical Reasoning

Visual Trace Reasoning

Real-world Spatial

Embodied

Reasoning

Reasoning

Object Counting Reasoning

3D Geometry Reasoning

GUI Reasoning

2D Geometry Reasoning

Table & Chart

Physics-based

Reasoning Rotation

Reasoning

Reasoning

Sora-2-pro Sora-2

Veo-3-fast Veo-3-preview

Kling-v1 Seedance-1.0-pro

(a) Evaluation Radar Map. (b) Word Cloud.

- Figure 2: Illustration of the MME-COF Benchmark. It showcases that different models specialize in distinct reasoning aspects, but most models exhibit limited reasoning capability across all tasks.

sistent local dynamics; however, they struggle with complex reasoning conditions, particularly in long-horizon causal consistency, geometric constraint adherence, and abstract logic. Overall, current video models are not yet ready as standalone zero-shot reasoners. Still, they show encouraging signs of emergent reasoning, suggesting strong potential as complementary reasoning agents alongside specialized models.

Our main contributions are summarized as follows:

- • A Comprehensive Empirical Study. We provide the first investigation of video models (Veo-3) to analyze their visual reasoning potential, detailing representative successes, characteristic errors, and the conditions under which CoF reasoning emerges, holds, or breaks.
- • The MME-COF Benchmark. We curate MME-COF, a compact benchmark providing a standardized taxonomy and an evaluation protocol aligned with CoF reasoning, enabling consistent and category-wise assessment beyond surface-level visual fidelity.
- • Insights and Directions. We summarize common success patterns (e.g., short-horizon coherence and stable spatial layout) and failure patterns (e.g., long-horizon degradation, violations of basic geometry/physics, and temporal logic), making clear when the behavior reflects genuine reasoning versus pattern replay.

#### 2 Deep-Dive Analysis on Veo-3

##### 2.1 Overview

To ensure a rigorous empirical study, we detail our core methodology in this section, including the taxonomy of reasoning tasks, test case curation process, the standardized style for prompt design, and the analysis setup.

Task Taxonomy. To capture different dimensions of reasoning, our study starts from dozens of reasoning-oriented tasks, which can be organized into the following 12 categories:

- 7) Rotation Reasoning
- 8) Table and Chart Reasoning
- 9) Object Counting Reasoning
- 10) GUI Reasoning
- 11) Embodied Reasoning
- 12) Medical Reasoning

- 1) Visual Detail Reasoning
- 2) Visual Trace Reasoning
- 3) Real-world Spatial Reasoning
- 4) 3D Geometry Reasoning
- 5) 2D Geometry Reasoning
- 6) Physics-based Reasoning

Each category comprises several representative cases selected to test specific aspects of reasoning.

Test Case Curation. We recruit five PhD-level experts with deep expertise in text-image reasoning, who are tasked with selecting representative cases from benchmarks [25, 67, 45, 52, 74] corresponding to each task category. For each reasoning case, the experts manually constructed text prompts that explicitly or unambiguously define the target reasoning objective, aiming to evaluate the potential of video models for multi-modal reasoning.

Prompt Design Style. To ensure consistency and fairness, all prompts follow a unified style emphasizing explicit visual constraints, controlled motion, and minimal linguistic ambiguity. Prompts are encouraged to be written in imperative form and designed to reduce variance from language interpretation, focusing the model’s behavior on the intended visual reasoning objective. The overall design principles are as follows:

- 1) Static camera and fixed viewpoint, unless motion is explicitly required by the task.
- 2) Stable spatial composition, consistent framing, and unchanging scene layout across frames.
- 3) Clear specification of allowed and disallowed changes (e.g., “no zoom, no pan, no dolly”) to constrain camera dynamics.
- 4) Explicit temporal phrasing to control the pace of motion, using cues such as “instantly”, “smoothly”, or “step-by-step”.
- 5) Avoidance of direct textual hints toward the answer; instructions are purely visual and task-oriented.
- 6) Inclusion of realistic phrasing and scene context to align with the model’s natural video priors while minimizing artifacts.

The standardized prompt style ensures that differences in output primarily reflect the model’s internal reasoning potential rather than prompt variability.

Analysis Setup. For every reasoning case, we construct a text prompt that explicitly or implicitly specifies the target reasoning objective. Each prompt produces six video samples at a resolution of 1280×720, 24 FPS, and a duration of 8 seconds. All experiments are conducted in a unified zero-shot setup without fine-tuning, additional supervision, or auxiliary tools.

We evaluate model outputs through qualitative judgments along three levels of performance, i.e., Good , Moderate , and Bad, based on the clarity, correctness, and temporal stability of the visual reasoning process. Detailed definitions and examples of these evaluation criteria are provided in the corresponding task subsections. Note that, since we observe that most video models struggle to follow the requirement of ‘static shot’ reliably, we apply more permissive qualitative criteria for static-shot evaluations. We further define a success rate to measure robustness across generations for each case, computed as the proportion of successful samples among the six generated. For cases categorized as Bad, the success rate is always 0. Non-zero success rates only appear in cases evaluated as Good or Moderate , indicating that Veo-3 exhibits some potential to perform as a visual reasoner. A higher success rate reflects a more stable reasoning capability of the model.

- I. Question: Text-to-Video Prompt:

Q: What is the color of the Apple logo?

1st frame

Input Image: Reasoning Video:

- II. Question: Text-to-Video Prompt:

1st frame

- III. Question: Text-to-Video Prompt:

A: The color of the Apple logo is polychromatic.

- IV. Question:

Zoom in on the black bag with the Apple logo to focus on the logo's color. Static shot.

[Figure 29]

~ Moderate Success Rate: 17%

Input Image: Reasoning Video:

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Gradually zoom in on the group of people walking along the path, centering on the person carrying the handbag. Keep the surrounding park and benches softly blurred to emphasize the handbag’s color. Static shot.

Q: What is the color of the handbag? A: The color of the handbag is white.

[Figure 37]

✓ Good Success Rate: 33%

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Smoothly zoom in on the dog near the lower right corner of the scene, then highlight the motorcycle parked near it. Keep the surrounding jeeps and people slightly blurred to emphasize spatial relation. Static shot.

Q: Is the motorcycle on the left or right side of the dog?

A: The motorcycle is on the left side of the dog.

[Figure 45]

✓ Good Success Rate: 83%

Input Image: Reasoning Video:

1st frame

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Text-to-Video Prompt:

Q: Is the baby carriage on the left or right side of the cone? A: The baby carriage is on the right side of the cone.

Gradually zoom in on the area near the cone along the pathway, centering both the cone and the baby carriage in the frame. Keep the surrounding trees and grass softly blurred to emphasize these two objects. Static shot.

[Figure 53]

✗ Bad

Input Image: Reasoning Video:

1st frame

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Figure 3: Showcase of Visual Detail Reasoning by Veo-3. It illustrates Veo-3’s ability to localize targets and maintain fine-grained visual attributes across frames, together with common failure modes when targets are small, occluded, or embedded in clutter.

##### 2.2 Visual Detail Reasoning

Task Description and Evaluated Aspects. In the visual detail reasoning category, the objective is to assess a model’s ability to discern and maintain fine-grained visual attributes and spatial relations within generated video sequences. It covers attribute recognition, e.g., identifying color, texture or material of an object, and spatial relation identification, e.g., recognizing that one object is on the left of or behind another object. The model is evaluated on the capacity both to attend to the correct target region and to maintain visual consistency, across frames, of the attribute or relation in question.

Definition of Good / Moderate / Bad. We define the three-level evaluation criteria as follows:

✓ Good: The reasoning video accurately centers on the correct target region, clearly resolves the relevant attribute, such as color, texture or position, and maintains sharp, stable and natural rendering throughout the sequence. There are no visible frame drops, artifacts or unintended motion.

~ Moderate: The region of interest is approximately correct, and the attribute remains inferable, but the sequence suffers from minor blur, incomplete framing, slight instability mild unnatural motion, or sometimes deviates from the textual instruction and produces a plausible but unaligned or self-directed visual interpretation, limiting confident interpretation.

✗ Bad: The target region is incorrect or ambiguous, the attribute cannot be reliably inferred, or the video exhibits severe artifacts: abrupt frame jumps, major jitter, unintended zoom or crop, extraneous objects interfering, or conspicuous quality degradation that obstructs the reasoning task altogether.

Data Source. We sample data from the V∗Bench [71], which provides a comprehensive set of evaluation dimensions including spatial relationship and color/attribute consistency tasks.

Example and Analysis. We illustrate typical behaviors of Veo-3 in visual detail reasoning through four representative cases in Figure 3. In case I, the model performs well in localizing the target: although it does not strictly execute the “zoom in” instruction, it instead achieves an equivalent visual outcome through a semantically consistent motion with a person’s hand. This slight deviation suggests that the model may exhibit certain generation preferences in how it interprets and realizes spatial instructions, possibly reflecting stylistic tendencies learned from training data. In cases II and III, the model achieves better success rates when the targets are visually salient and contextually distinct. For the handbag and dog-motorcycle scenes, Veo-3 attends to the correct regions and maintains smooth temporal coherence. However, when the object (e.g., the motorcycle) is small or surrounded by distracting elements, the model occasionally fails to locate it accurately, indicating limited fine-grained spatial discrimination in cluttered scenes. In case IV, when the target object is tiny and visually indistinct, Veo-3 cannot identify it even with explicit positional hints, highlighting that the model’s perceptual grounding and reasoning weaken sharply when object size and salience are too low for reliable attention.

Takeaway 1

Veo-3 performs well in fine-grained attribute and spatial reasoning for salient, well-grounded targets, but fails when objects are small, occluded, or cluttered. It sometimes exhibits stylistic generation biases that lead to plausible yet instruction-divergent outcomes.

##### 2.3 Visual Trace Reasoning

Task Description and Evaluated Aspects. The visual trace reasoning category evaluates a model’s ability to represent and maintain causal continuity across sequential actions. Typical tasks include maze navigation, path following, and multi-step object manipulation, where the video must visually encode a coherent sequence of intermediate decisions that lead to the correct goal. Performance is assessed based on two major aspects: (i) temporal coherence, which is the smoothness and logical

- I. Question: Text-to-Video Prompt:

Q: Starting from the red dot, follow the given

movement instructions and determine the final position. Down 1, left 1, left 1, up 1, up 1. A: A

1st frame

Input Image: Reasoning Video:

- II. Question†: Text-to-Video Prompt:

1st frame

- III. Question†: Text-to-Video Prompt:
- IV. Question:

Starting at the red dot in the middle-right cell, animate step-by-

step moves: go down 1 cell, left 1, left 1, up 1, and up 1, drawing arrows for each step and finishing with a glow around

the final cell. Static shot.

[Figure 61]

Input Image: Reasoning Video:

- ✗ Bad

[Figure 62]

- ✗ Bad

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Animate the elf moving step by step toward the gift while carefully

Q: The character must avoid falling into the

avoiding the icy frozen lake. Highlight the successful path and end with the elf standing beside the gift. Static shot.

frozen lake and reach the gift pack safely.

[Figure 71]

✓ Good Success Rate: 17%

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

[Figure 79]

Animate the red triangle moving step by step toward the white printer,

Q: Move the character (red triangle) to pick up the white printer and

picking it up once it reaches it. Then have the triangle carry the printer

upward and place it on the brown area representing the table. End with a subtle highlight around the printer to show it is toggled on. Static shot.

place it anywhere on the desk.

Input Image: Reasoning Video:

1st frame

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Text-to-Video Prompt:

Q: The given picture is a maze, and the black lines

Animate a bright path tracing from the blue point at the

represent walls that cannot be walked. Now you want to walk from the blue point to the red point. Is there a

top through the maze’s open corridors toward the red point at the bottom, highlighting each green numbered

feasible path? If so, which of the green marks numbered

mark it passes. Keep the maze and all walls fixed while

1-5 In the picture must be passed in the path? A: Yes, 3.

the glowing path moves smoothly through the correct route. Static shot.

[Figure 88]

✗ Bad

Input Image: Reasoning Video:

1st frame

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

- Figure 4: Showcase of Visual Trace Reasoning by Veo-3 (Part I). It shows short-horizon pathfollowing successes, object-grounding failures, and a certain bias that causes step omissions/mistakes in multi-step traces. † The ground-truth answers of cases II and III are intuitive and non-unique, which are omitted to highlight the key reasoning behaviors.

- V. Question: Text-to-Video Prompt:

Q: In the diagram, the red arrow is the initial arrow, and the green arrow

is the final arrow. The arrow can

move in four directions (forward, backward, left, right), where

'forward' always refers to the current

direction the arrow is pointing. After

each movement, the arrow's

direction is updated to the direction of movement. Which of the

following paths can make the arrow

move from the starting position to the ending position?

A: (Forward, 1 unit) - (Left, 1 unit)

1st frame

Input Image: Reasoning Video:

- VI. Question: Text-to-Video Prompt:

Create a 2D animation based on the provided diagram. The red arrow is the initial arrow, and the green arrow is the final arrow. The arrow can

move in four directions (forward, backward, left, right), where 'forward'

always refers to the current direction the arrow is pointing. After each movement, the arrow's direction is updated to the direction of movement.

Movement commands:

- - The red arrow moves forward for 1 unit.
- - The red arrow moves left for 1 unit (relative to its new current direction after step 1). Then turns green. Scene:
- - No change in scene composition.
- - No change in the layout of the diagram. Camera: Static camera. No zoom. No pan. No glitches, noise, or artifacts.

- ✗ Bad

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

[Figure 114]

[Figure 115]

- ✗ Bad

Input Image: Reasoning Video:

Two small characters start from the same purple origin at the same

Q: What are the advantages of the green

time, and move along the red and green paths toward another purple destination at the same speed. Static camera, no zoom, no pan.

route and the red route respectively?

[Figure 116]

1st frame

- Figure 5: Showcase of Visual Trace Reasoning (Part II) by Veo-3. The examples highlight long-horizon planning breakdowns, inconsistent arrow/trajectory rendering, and failures to preserve comparative or sequential information across frames.

progression between consecutive steps; and (ii) goal consistency, which means whether the full sequence visually completes the intended reasoning trajectory without deviation or contradiction.

Definition of Good / Moderate / Bad. We rate the performance according to the following criteria:

✓ Good: Each movement step is depicted continuously and logically toward the correct goal. The motion is smooth, temporally consistent, and follows causal order with no skipping, stuttering, or direction reversal.

~ Moderate: The overall trajectory roughly aligns with the intended sequence, but small discontinuities, timing irregularities, or partial missteps occur. The reasoning path remains interpretable, and the goal can still be inferred.

✗ Bad: Key steps are missing, reversed, or illogical. The sequence shows abrupt jumps, inconsistent object trajectories, or goal confusion, breaking the temporal and causal coherence of the reasoning process.

Data Source. We select samples from MVoT [41], FrozenLake [8, 72], MiniBehavior [32], RBenchV [25], SpatialViz-Bench [66], and OmniSpatial [29], which provide controlled multi-step environments for evaluating temporal reasoning, sequential planning, and causal continuity in visual simulations.

Example and Analysis. In Figure 4 and Figure 5, we showcase six representative visual-trace examples. In case I, the model repeatedly fails to execute the exact step sequence and instead drifts toward a visually salient central cell. However, case II is one of the few successes: the model can produce a coherent step-by-step path in simple, low-branching settings, but this behavior is not robust across trials. Case III largely fails, where the model often does not ground the specified object (printer), sometimes hallucinating its appearance or placement rather than performing a consistent pickup-and-place. Case IV shows near-uniform failure on long-horizon, highly branched navigation: outputs contain wrong turns, discontinuities, and no faithful global plan. Case V reveals difficulty grounding abstract movement rules, producing inconsistent arrow trajectories. Case VI produces visually plausible motions along individual paths but fails to preserve or present the comparative information required for contrastive reasoning. Taken together, these examples indicate that the model can simulate locally coherent short traces but systematically fails at long-horizon planning, rule-grounded execution, and object-persistent manipulations.

##### Takeaway 2

Veo-3 can produce locally coherent, short-horizon trace animations in simple, low-branching scenarios, but it does not reliably execute long-horizon plans or rule-grounded sequences.

##### 2.4 Real-World Spatial Reasoning

Task Description and Evaluated Aspects. This task investigates Veo-3 [21]’s ability to perceive and maintain spatial relations within natural scenes, with a focus on reasoning about viewpoint change, orientation consistency, and reference-frame alignment. We assess whether the model preserves a stable global coordinate frame and coherent scene orientation under varying viewpoints, and whether objects retain correct relative positions and orientations with respect to each other across different views.

Definition of Good / Moderate / Bad. We define the evaluation criteria in three levels:

✓ Good: Scene orientation, reference frame, and viewpoint are consistent and correctly represent spatial relations. The camera remains steady and the motion is natural.

~ Moderate: Scene roughly matches the instruction but contains small perspective errors, unnatural transitions, or partial mirroring. Motion remains interpretable but not physically coherent.

✗ Bad: Reference frame or direction is wrong; viewpoint shifts abruptly or inconsistently. Video suffers from strong camera drift, disorienting motion, or spatial chaos.

Data Source. To evaluate on orientation and layout reasoning, we specifically sample data from MMSI-Bench [74]. Also, the tasks of perspective taking and spatial interaction are selected from the OmniSpatial dataset [29].

Example and Analysis. As shown in Figure 6, Veo-3 can correctly handle basic spatial layouts in case I, but struggles with complex viewpoints or orientation changes in case II. The perspective transformations are sometimes inaccurate or even incorrect, suggesting that the model tends to prioritize visual plausibility over precise spatial reasoning, which hinders further reasoning in case IV. Moreover, case III demonstrates that Veo-3 has difficulty understanding depth, further limiting its spatial reasoning capability.

##### Takeaway 3

While Veo-3 exhibits an emerging ability for simple real-world spatial reasoning, its capability remains insufficient for handling more complex spatial understanding tasks.

- I. Question: Text-to-Video Prompt:

1st frame

[Figure 117]

Input Image: Reasoning Video:

An arrow points from the player wearing jersey number 10 in purpleto the

basketball. Static camera view, no zoom or pan.

- II. Question: Text-to-Video Prompt:

Question: From the perspective of

the player wearing jersey number 10

in purple, where is the basketball? A: Left front.

1st frame

Input Image: Reasoning Video:

The camera slowly and smoothly elevates from its current isometric

view, gradually rising upwards while maintaining focus on the apartment layout. It continues to ascend until it reaches a complete

overhead, bird's-eye perspective, providing a full top-down view of the

entire floor plan, displaying all rooms and furniture clearly from above. The movement is fluid and controlled, ending with a static shot from the high vantage point.

IV. Question: Text-to-Video Prompt:

Q: If you are facing the washing machine,

how should you walk to the stove and face the stove?

A: Turn around and go straight, then turn

left and go straight, then turn right and go straight, finally turn left to face the stove.

1st frame

A red arrow point from the green chair toward the balcony. Another red arrow point from the door to the balcony. Static camera view, no zoom or pan.

Question: The balcony is

north relative to the door, in which direction on

the balcony is the chair?

A: Southwest.

Input Image: Reasoning Video:

- III. Question: Text-to-Video Prompt:

Input Image: Reasoning Video: ✓ Good Success Rate: 33%

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

~ Moderate Success Rate: 17%

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Q: From the dunker's

The image transitions to a depth-map of the scene: Darker colors represent pixels further

viewpoint, which whiteuniformed player is the

from the camera, lighter colors represent pixels closer to the camera. The exact color map to use is provided on the right side of the image. Static scene, no pan, no zoom, no dolly.

farthest from them?

A: Five.

[Figure 130]

✗ Bad

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

1st frame

[Figure 137]

~ Moderate Success Rate: 20%

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

- Figure 6: Showcase of Real-World Spatial Reasoning by Veo-3. Although Veo-3 can reason about simple spatial layouts, it still struggles to maintain consistency under complex perspective or orientation changes.

- I. Question: Text-to-Video Prompt:

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

1st frame

[Figure 147]

Input Image: Reasoning Video:

A hand moves the object to the left along the y-axis and then moves it up. Static camera

view, no zoom or pan, and the perspective of the object remains unchanged throughout.

- II. Question: Text-to-Video Prompt:

Q: Move the object to the left along the y-axis and up.

1st frame

Input Image: Reasoning Video: ✗ Bad

Smoothly zoom in to the "Initial State" figure. The yellow block, starting at (1,0,0), moves one

unit in the positive Y direction to position (1,1,0). Then, move back to (1,0,0). The cyan block,

starting at (2,0,0), moves one unit in the positive Y direction to position (2,1,0), exchange the position with the purple block. Static shot. No pan. No glitches, noise, or artifacts.

IV. Question: Text-to-Video Prompt:

Q: The sequence of moves that

turns the first cube stack into

the final one is _______. A: (1,0,0)y+ (1,1,0)y- (2,1,0)y+

1st frame

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

A: D.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

✗ Bad

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Move the object up. Static camera view, no zoom or pan, and the perspective of the

object remains unchanged throughout.

Q: Move the object up.

[Figure 165]

A: D.

Input Image: Reasoning Video:

- III. Question: Text-to-Video Prompt:

Input Image: Reasoning Video: ✓ Good Success Rate: 83%

[Figure 166]

[Figure 167]

Q: Check out a net with 6 faces below: Can the net be folded to form a cube, yes or no?

The net is folded to form a single cube, with folding edges clearly shown. Static camera

perspective, no zoom or pan.

A: Yes.

[Figure 168]

✗ Bad

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

1st frame

- Figure 7: Showcase of 3D Geometry Reasoning by Veo-3 (Part I). While Veo-3 shows certain potential in basic 3D geometry reasoning, its performance remains unstable for complex geometry transformations.

###### 4.Visual-Logical Reasoning

cropped_0.jpg

###### I. Question: Text-to-Video Prompt:

Q: From the given four options, select the most suitable one to fill in the question mark. A: C

Rotate the first figure to align with the second, then apply the same rotation to the third to reveal the expected fourth figure; hold on the resulting pattern.

[Figure 175]

###### ✗ Bad

Input Image: Reasoning Video:

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

1st frame

[Figure 180]

cropped_0.jpg

V. Question: Text-to-Video Prompt:

Q: From any angle, which one on the right is not a view of the three dimensional shape given on the left? A: C

Rotate only the left 3D shape in place with a small, constant-speed yaw at fixed scale; keep everything else static.

[Figure 181]

~ Moderate Success Rate: 17%

Input Image: Reasoning Video:

1st frame

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

- Figure 8: Showcase of 3D Geometry Reasoning by Veo-3 (Part II). The model often generates misaligned or self-intersecting structures, compromising geometric consistency.

##### 2.5 3D Geometry Reasoning

Task Description and Evaluated Aspects. We also evaluate Veo-3’s potential on 3D geometry reasoning tasks, such as geometric object motion and three-dimensional structural transformations like reconstructing a cube net. The assessment focuses on three key dimensions: geometric accuracy, structural completeness throughout the transformation, and visual continuity across frames.

Definition of Good / Moderate / Bad. We categorize the model’s performance into three levels:

✓ Good: Transformations like folding, rotation and assembly are geometrically correct, visually smooth, and continuous, maintaining structural integrity and realistic motion. No broken edges, jumps, or spatial artifacts.

~ Moderate: Transformations are partially correct but show local misalignment, unrealistic deformation, or discontinuous motion; geometry is roughly interpretable but imperfect.

✗ Bad: Transformation fails. For example, wrong fold, structure collapse, or impossible geometry. Motion is erratic, discontinuous, or visually implausible, breaking the sense of physical realism.

Data Source. To construct diverse and representative evaluation data, we adapt tasks from established geometric spatial reasoning datasets, including the 3D-Text-Instruct and Folding Nets subsets of the STARE benchmark [43], the BlockMoving subset from the SpatialViz-Bench [66], as well as VisuLogic [73] benchmark.

Example and Analysis. We showcase the results of Veo-3 on 3D geometry reasoning tasks in Figure 7 and Figure 8. Veo-3 demonstrates a degree of potential on 3D geometry reasoning, performing reasonably well on simple, single-step geometric transformations, as shown in case I. However, its performance degrades noticeably when facing multi-step or compositionally complex transformations in case II. As presented in cases III and V, the model frequently produces misaligned or self-intersecting structures, leading to a loss of geometric consistency. Further observations in case

- IV, show that while the model can partially understand the geometric shape of individual objects, it lacks a coherent understanding of coordinate systems and the spatial relationships among multiple objects.

##### Takeaway 4

Veo-3 exhibits emerging reasoning potential on basic 3D transformations but breaks down on complex or multi-step geometry, often yielding misaligned or self-intersecting structures. Its

##### Takeaway 5

3D geometric reasoning remains fragile, revealing substantial gaps in its ability to function as a reliable 3D geometry reasoner.

##### 2.6 2D Geometry Reasoning

Task Description and Evaluated Aspects. To assess a model’s competence in 2D geometric reasoning, we evaluate its zero-shot performance on planar geometric construction tasks. These tasks involve drawing geometric relations by connecting points, adding auxiliary lines, and moving geometric shapes. The evaluation focuses on whether the generated constructions or movements accurately reflect the described geometric relationships and adhere to the given instructions, while maintaining smooth, stable operations that ensure visual clarity and coherence throughout the process.

Definition of Good / Moderate / Bad. We rate the performance according to the following criteria:

✓ Good: Constructions and movements are geometrically accurate and visually smooth. Endpoints, intersections, angles, and motion trajectories align correctly with the instructions. Both drawing and movement processes are stable, fluid, and natural, resembling human sketching or manipulation.

~ Moderate: Constructions and movements roughly follow the intended geometry but exhibit minor inaccuracies in line placement, shape alignment, trajectory, or smoothness. Some local jitter or abrupt motion may appear, but the overall structure and motion remain interpretable.

✗ Bad: Constructions or movements deviate substantially from geometric correctness. Lines or shapes may be misplaced, disconnected, or moved in a chaotic or discontinuous manner (e.g., jittering, overlapping, or distorted paths), leading to visual instability and loss of interpretability.

Data Source. The evaluation data are drawn from multiple established sources, including the Geo170k dataset [18], the VarsityTutors subset of Math-PUMA [85] dataset, the line-connection subset of RBench-V [25], the MAVIS-Gen [80], Tangram Puzzle subsets of the STARE [43] benchmark, and data from VAT [46].

Example and Analysis. The representative examples of the 2D geometry reasoning task are presented in Figures 9 and 10. Veo-3 demonstrates a foundational capability for simple geometric connection tasks, correctly identifying and linking elements in straightforward scenarios like in case III. However, this basic competence is inconsistent. The model often prioritizes producing visually symmetric or semantically meaningful patterns rather than strictly adhering to geometric instructions (cases I and II). Furthermore, case II reveals instances where the model unintentionally modifies the original figures, indicating a limited awareness of geometric constraints and poor spatial consistency. When tackling more complex connection tasks, the model frequently fails to interpret the intended drawing order or point indices, resulting in incorrect connection sequences, as demonstrated in cases

- V, VI, and VII. This is often coupled with an inability to control task termination, as the model tends to continue drawing beyond the required constructions. Finally, for tasks involving the movement of geometric shapes in cases IV and VIII, the model struggles to maintain geometric structural consistency throughout the motion.

##### Takeaway 6

Veo-3 shows initial 2D geometric reasoning ability but still falls short of consistent, constraintaware geometric understanding, remaining far from a robust geometric reasoner.

- I. Question: Text-to-Video Prompt:

Q: In the figure shown, let 'n' represent the length of side AB of the inscribed

rectangle ABCD, where n is an undetermined value. With BC equal to 6.0 and the diameter of circle O equal to 10.0, what is the value of 'n’?

A: 8

1st frame

Input Image: Reasoning Video:

- II. Question: Text-to-Video Prompt:

1st frame

- III. Question: Text-to-Video Prompt:
- IV. Question:

A line connecting point A and point C. The

video ends once the connection process is

complete. Static view, no zoom or pan.

[Figure 187]

~ Moderate Success Rate: 83%

Input Image: Reasoning Video:

Q: The figure presented depicts a square designated as ABCD. Within this square, point M is identified as the midpoint of the side AB, while point N is the midpoint of the opposing side CD. Additionally, point O is located at the midpoint of segment CN. Your

Smoothly connecting point

M and point N. The video

ends once the connection process is complete. Static view, no zoom or pan.

task is to draw the segment MO. It is given that the length of segment AM is represented

by t. The objective is to determine which of the following expressions accurately represents the length of the segment MO in terms of t.

|A: 417𝑡<br><br>|
|---|

[Figure 188]

- ✗ Bad

[Figure 189]

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

- ✗ Bad

Q: AB equals to 8.0. What would the area of the entire shape ABCD be?

Smoothly connecting point C and point D with a line. The video

ends once the connection process is complete. Static view, no zoom or pan.

A: 62.87

[Figure 204]

~ Moderate Success Rate: 33%

Input Image: Reasoning Video:

1st frame

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Text-to-Video Prompt:

Q: Check out an Tangram puzzle below. The left panel is an empty Tangram puzzle,

Place piece A with its upper-

while the right panel shows available pieces to complete the puzzle. Keep in mind that you can rotate or flip the pieces. Can the Tangram puzzle be completed with the available pieces, yes or no? A: Yes.

left corner at (x, y) = (0, 3).

Input Image: Reasoning Video:

1st frame

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

- Figure 9: Showcase of 2D Geometry Reasoning by Veo-3 (Part I). While Veo-3 shows potential in recognizing simple patterns, it lacks the robust constraint awareness essential for accurate geometric manipulation.

[Figure 219]

✓ Good Success Rate: 83%

- V. Question: Text-to-Video Prompt:

Q: Connect the black dots in the image sequentially with straight

lines according to the edge numbers (i.e., connect dot 1 to dot 2,

dot 2 to dot 3, and so on). The final result will form a simple line

drawing. What does this drawing represent? A: Duck.

1st frame

Input Image: Reasoning Video:

- VI. Question: Text-to-Video Prompt:

1st frame

- VII. Question: Text-to-Video Prompt:

~ Moderate Success Rate: 83%

[Figure 220]

Input Image: Reasoning Video:

1st frame

✗ Bad

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Animate the numbered dots connecting

sequentially from 1 to 118, each straight line appearing smoothly as the outline gradually emerges. Keep all numbers and dots visible while the connecting lines form step by step. Static shot.

Q: Connect the black dots in the image sequentially with straight lines according to the edge numbers (i.e., connect dot 1 to dot 2,

dot 2 to dot 3, and so on). The final result will form a simple line

drawing. What does this drawing represent? A: Lion.

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Animate the numbered dots connecting sequentially from 1 to 118, each straight line

appearing smoothly as the outline gradually

emerges. Keep all numbers and dots visible while the connecting lines form step by step. Static shot.

Q: Connect the black dots in the image sequentially with straight lines according to the edge numbers (i.e., connect dot 1 to dot 2,

dot 2 to dot 3, and so on). The final result will form a simple line

drawing. What does this drawing represent? A: Bird.

✗ Bad

[Figure 246]

cropped_0.jpg

Input Image: Reasoning Video:

Pan vertically at a steady speed to center the top yellow line, then the bottom one, keeping both visible with identical scale and exposure. Ensure all elements stay unchanged and finish

with a full view showing both lines together.

- VIII. Question: Text-to-Video Prompt:

Animate the dots connecting sequentially from 1 to

25, each straight line appearing smoothly until the

full outline emerges. Keep the background with the

smiling sun and plants unchanged. Static shot.

Input Image: Reasoning Video:

Q: Is the top yellow line shorter than the bottom

yellow line?

A: No.

[Figure 247]

✗ Bad

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

1st frame

- Figure 10: Showcase of 2D Geometry Reasoning by Veo-3 (Part II). Veo-3’s reasoning abilities are further challenged by complex sequential instructions and the need to preserve structural integrity.

- I. Question: Text-to-Video Prompt:

Q: The figure shows a rough semicircular track whose ends are at a vertical height h. A block placed at point P at one end of the track is

released from rest and slides past the bottom of the track. Which of the

following is true of the height to which the block rises on the other side

of the track?

A: It is between zero and h; the exact height depends on how much energy is lost to friction.

1st frame

Input Image: Reasoning Video:

- II. Question: Text-to-Video Prompt:

1st frame

- III. Question: Text-to-Video Prompt:
- IV. Question:

Show the rough semicircular track with

height label h and a block at P; release it, add faint friction streaks as it slides down

and up the right side, stopping below the

rim. Show the move quickly and completely. Static shot.

[Figure 253]

Input Image: Reasoning Video:

- ✗ Bad

[Figure 254]

- ✗ Bad

[Figure 255]

- ✗ Bad

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Q: The red ball moves in the direction indicated by the blue arrow

Animate the red ball moving along the blue

and bounces off the black side walls upon collision; the component of its velocity perpendicular to the wall reverses in direction but

arrow's direction, bouncing off the black

walls according to reflection rules, keeping

maintains its magnitude, while the component parallel to the wall

speed consistent. Continue its path upward until it reaches and collides with one of the

remains unchanged. Based on this behavior, please estimate which

numbered brick (from 1 to 10) at the top the red ball will hit first.

numbered top bricks. Static shot.

A: 1.

[Figure 262]

~ Moderate Success Rate: 83%

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Q: Think about the magnetic force between the magnets in each pair.

Dynamically depict the attraction between magnets, paying

attention to speed and intensity. Static shot.

A: The magnetic force is stronger in Pair 2.

Input Image: Reasoning Video:

1st frame

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Text-to-Video Prompt:

Q: The orange gear is fixed on the

The orange gear rotates counterclockwise in the given view.

stationary green gear. If the orange

Animate the provided planetary gear system. The orange gear is

gear rotates counterclockwise in the given view, what is the motion of the yellow gear relative to the orange gear? A: Clockwise rotation.

fixed on the green gear. The central orange sun gear rotates counterclockwise, driving the yellow planet gear. All components must maintain their relative axial positions and proper gear meshing. The camera is static, with no zoom or pan.

Counterclockwise revolution.

Input Image: Reasoning Video:

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

1st frame

- Figure 11: Showcase of Physics-based Reasoning by Veo-3. The physics scenarios demonstrate locally plausible dynamics and reflections, alongside systematic quantitative and causal inconsistencies under frictional, force-driven, or constrained interactions.

##### 2.7 Physics-based Reasoning

Task Description and Evaluated Aspects. The physics-based reasoning category assesses a model’s capacity to depict and reason about motion dynamics, physical causality, and rule-based interactions between objects. Tasks in this group involve gravity, collisions, reflection, momentum, or energy conservation, requiring the model to generate physically plausible and temporally coherent motion. Evaluation focuses on two complementary aspects: (i) physical plausibility, which means whether the simulated motion obeys common physical principles; and (ii) causal correctness, which is whether object interactions are consistent with the underlying cause-and-effect relationships described in the prompt.

Definition of Good / Moderate / Bad. We rate the performance according to the following criteria:

✓ Good: The motion sequence adheres to physical laws such as gravity, momentum, and energy conservation. Object interactions are realistic and temporally smooth, and the visual outcome remains coherent and credible throughout.

~ Moderate: The physical relations are approximately correct but include minor inconsistencies, such as irregular acceleration, timing mismatch, or slight violation of conservation. The overall motion remains interpretable and visually plausible.

✗ Bad: The motion is physically implausible or visually chaotic—objects float, stop abruptly, or behave contrary to basic causal principles. Severe artifacts or temporal discontinuities disrupt the perception of a coherent physical process.

Data Source. We draw samples from MMMU [77], ScienceQA [49], and related physical reasoning subsets of RBench-V [26] and SpatialViz-Bench [66], covering scenarios such as object collisions, pendulum motion, frictional sliding, and optical or magnetic interactions.

Example and Analysis. Figure 11 presents four representative physics tasks and their outputs. Case I shows that the model can produce a visually coherent slide, but the behavior violates basic physical laws. Case II is the most reliable, where reflections and general trajectory shape are rendered plausibly and the task attains a high success rate, although small angular or timing offsets are common. In case III, the model conveys attraction through motion, yet the depicted dynamics do not reliably track the intended force magnitudes or causal ordering. Finally, case IV exposes structural failures, incorrect meshing, inconsistent relative rotations, and nonphysical contact behavior occur frequently, so the mechanical constraints are not respected. Overall, the model can synthesize locally plausible dynamics and handle simple reflection rules, but it fails to maintain quantitative physical constraints and causal fidelity in frictional, force-driven, or mechanically constrained scenarios.

Takeaway 7

Veo-3 often generates visually plausible short-term dynamics, but it systematically fails to preserve quantitative physical constraints (energy, momentum), causal ordering, and contact mechanics in frictional, force-driven, or mechanically constrained scenarios. Thus, its outputs are somewhat useful for qualitative illustration but are not reliable for quantitative physics inference or causal prediction.

##### 2.8 Rotation Reasoning

Task Description and Evaluated Aspects. The rotation reasoning task assesses the ability to reason about planar object rotation and maintain consistent spatial grounding under rotational transformations, thereby supporting subsequent reasoning processes. In each instance, the model is required to accurately rotate target objects within a fixed 2D plane while preserving the overall scene structure and structural consistency, followed by performing reasoning tasks like grounding and OCR. The evaluation focuses on both the accuracy of the rotation in terms of angle and direction, and the precision of the resulting reasoning tasks.

- I. Question: Text-to-Video Prompt:

1st frame

Input Image: Reasoning Video:

Rotate the scene 180 degrees clockwise. Then draw a bounding

box around the leftmost vending machine.

- II. Question: Text-to-Video Prompt:

Q: Looking up from the floor, how many rows

of drinks are in the leftmost vending machine?

A: 2

1st frame

Input Image: Reasoning Video: ✗ Bad

The entire 'Original' grid figure performs one smooth, continuous 360-degree

rotation clockwise within its own 2D plane. The camera stays static, with no pan.

IV. Question: Text-to-Video Prompt:

Q：Which grid can be obtained by rotating the grid only?

1st frame

[Figure 281]

✗ Bad

[Figure 282]

Rotate the scene 45 degrees clockwise. Then draw bounding boxes around the

frontmost skiing character.

Q: Is the frontmost skier

wearing a scarf?

A: No.

Input Image: Reasoning Video:

- III. Question: Text-to-Video Prompt:

[Figure 283]

~ Moderate Success Rate: 83%

Input Image: Reasoning Video:

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Q: On which floors are the 'IKEA' labels located? A: One on the top floor, one on the middle floor,

Rotate the video frame 90 degrees counterclockwise in the 2D

plane, then draw bounding boxes around each 'IKEA' label.

and one on the bottom floor.

[Figure 296]

✗ Bad

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

1st frame

[Figure 305]

A: A

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

- Figure 12: Showcase of Rotation Reasoning by Veo-3. Veo-3 struggles in complex scenes. However, its foundational grasp of simple rotations signals its potential to support rotation-based reasoning tasks.

Definition of Good / Moderate / Bad. Model outputs are categorized into three quality levels:

✓ Good: The rotation is accurate, complete, and strictly confined to the 2D plane, with no extraneous scene motion. The following reasoning tasks are completed correctly. Target objects remain precisely grounded after rotation.

~ Moderate: The rotation is largely correct but may be incomplete or slightly off-angle, though still confined to the 2D plane. The following reasoning tasks are mostly completed. Minor temporal or visual inconsistencies may appear, but do not alter the core 2D structure or object grounding.

✗ Bad: The model fails to perform the correct rotation, extends the transformation into 3D space, or introduces substantial scene distortion. Cannot complete the following reasoning task. The original 2D structure is altered, leading to inaccurate grounding of the target objects.

Data Source. To specifically assess the rotation reasoning task, we recruit some PhD-level experts with deep expertise in text-image reasoning to design the evaluation data manually, followed by the necessary review process, as mentioned in Section 3.2. Each question is designed following the principle that it must involve a 2D rotation to reach the correct solution, ensuring the task genuinely probes rotational understanding rather than simple visual matching. Moreover, we sample data from the 2DRotation subset from the SpatialViz-Bench [66], and reformulate the question into instructions for the video models.

Example and Analysis. The results are shown in Figure 12. In case I, we find that Veo-3 handles small-angle rotations and simple planar scenes reasonably well, demonstrating a basic grasp of rotational motion. However, in more complex scenarios like cases II, III, and IV, the model often ignores the 2D rotation constraint and inadvertently alters the 3D structure, resulting in incorrect rotations and degraded spatial grounding. Such errors frequently propagate to downstream tasks, such as OCR in case III, or object localization in case II, due to inconsistencies in post-rotation alignment. These observations suggest that the reasoning behavior of Veo-3 remains more pattern-driven rather than principle-driven. However, as it demonstrates a partial understanding of planar rotation, this can to some extent facilitate subsequent reasoning tasks.

##### Takeaway 8

Veo-3 exhibits only a superficial understanding of rotation reasoning. While it can approximate small planar rotations, it fails to preserve geometric consistency under larger or compound transformations.

##### 2.9 Table and Chart Reasoning

Task Description and Evaluated Aspects. The table and chart reasoning task requires the model to identify and focus on the key elements within visualizations or tabular data. For evaluation, we further consider how effectively the model identifies the regions relevant to the query and whether it can transition smoothly and visually coherently to these areas, preserving clarity, continuity, and proper scaling.

Definition of Good / Moderate / Bad. We rate the performance according to the following criteria:

✓ Good: Camera precisely focuses on the correct chart or table segment, smoothly highlighting or zooming into the queried data (e.g., correct year, category, or value). Motion is continuous, the chart and table remain clear, and no distortion or overexposure occurs.

~ Moderate: Camera approximately focuses on the right region but partially misses boundaries, introduces slight blur, or transitions abruptly. Data can still be inferred.

✗ Bad: Video fails to locate the correct region or changes the chart or table geometry unnaturally. Motion jitter, scaling errors, or artifacts make data unreadable or misleading.

- I. Question: Text-to-Video Prompt:

Q: What is the sum of footwear manufacturing establishments in

Nova Scotia and Mantioba as of

December 2020? A: 3

1st frame

Input Image: Reasoning Video:

- II. Question: Text-to-Video Prompt:

1st frame

- III. Question: Text-to-Video Prompt:
- IV. Question:

Start with smoothly zooming in to focus on the 'Nova Scotia' row. Then,

smoothly zoom out to the full view of the chart. End with smoothly zooming in to focus on the 'Manitoba' row. The chart itself, including all its data, lines,

and labels, must remain completely static and unchanged throughout the video.

- ✗ Bad

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

- ✗ Bad

Input Image: Reasoning Video:

[Figure 320]

Start with a static, full view of the chart. Then, smoothly zoom the camera in to focus on the vertical area corresponding to the year 2014. The chart itself, including all its data,

Q: In the year 2014, which

opinion is dominant? A: Unfavorable.

lines, and labels, must remain completely static and unchanged throughout the video.

[Figure 321]

~ Moderate Success Rate: 83%

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Q: What' the color of

Zoom in to focus on the smallest section in the chart. The chart itself, including all its data,

smallest section in the chart?

lines, and labels, must remain completely static and unchanged throughout the video.

A: Gray.

[Figure 329]

Input Image: Reasoning Video:

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

1st frame

Text-to-Video Prompt:

Q: What is the end market for the

Draw a bounding box around the end market for the Engineered

Engineered Systems segment? A: Printing & Identification, Industrials.

Systems segment. The table itself, including all its text, lines, and labels,

must remain completely static and unchanged throughout the video.

[Figure 337]

✗ Bad

Input Image: 1st frame Reasoning Video:

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

- Figure 13: Showcase of Table and Chart Reasoning by Veo-3. Veo-3 demonstrates an initial ability to focus on relevant data regions but lacks the precision and consistency required for reliable visual analysis.

Data Source. We use samples from the ChartQA [52] dataset and TableVQA-Bench [34].

Example and Analysis. For charts, as presented in cases I, II and III in Figure 13, Veo-3 can often zoom into an approximately correct region but lacks the precision needed to accurately locate the queried data. For tables, as shown in case IV, Veo-3 fails to correctly identify the required element and tends to select entries randomly. The model also frequently adds, modifies, or distorts existing chart and table elements, resulting in visual inconsistencies that undermine the accuracy of chart interpretation.

##### Takeaway 9

Veo-3 demonstrates emerging competence and potential in structured visual understanding, but still falls short of functioning as a precise and reliable chart-table reasoner.

##### 2.10 Object Counting Reasoning

Task Description and Evaluated Aspects. In this category, we focus on the ability to accurately enumerate objects within a 2D or 3D scene. In each instance, the model is required to identify, ground, and count target objects, typically by highlighting, drawing bounding boxes, applying numerical labels, or panning. The evaluation focuses on the accuracy of the count and the precision of the spatial grounding, performed within a scene that remains static or experiences only minimal motion, ensuring the counting process is not influenced.

Definition of Good / Moderate / Bad. Model outputs are categorized into three quality levels:

✓ Good: The model precisely highlights, draws bounding boxes around, or labels the objects with correct numbers, and performs smooth and controlled panning when necessary to cover all targets. Motion is continuous, and the scene remains static or experiences only slight changes that do not influence the counting process.

~ Moderate: The model approximately highlights or draws bounding boxes around the objects, or performs panning with minor instability or incomplete coverage. Objects or the scene may move or change slightly, but this does not strongly affect the counting process.

✗ Bad: The model fails to correctly highlight, label, or draw bounding boxes around the objects, or pans erratically such that parts of the scene are missed or revisited unnecessarily. Objects or the scene move or change substantially, severely affecting the counting process.

Data Source. The 2D object counting data are sampled from the counting subset of RBench-V [25]. The 3D object counting data are from the Super-CLEVER dataset [45] and VAT [46].

Example and Analysis. The results are shown in Figures 14 and 15. In the 2D counting tasks from cases I to III, objects frequently move or change during the process, negatively impacting counting stability and accuracy. In the 3D counting tasks, Veo-3 successfully handles simple grounding and counting scenarios, as demonstrated in case V, but struggles with scenes involving complex materials or geometric variations in cases VI and VII, leading to inaccurate counts. Additionally, in the panning process of case VII, the camera fails to precisely move to the regions containing all target objects, further hindering the counting process.

##### Takeaway 10

Veo-3 demonstrates basic counting capability but lacks the spatial control and robustness required for reliable object enumeration in dynamic or complex scenes.

cropped_0.jpg

###### I. Question: Text-to-Video Prompt:

Q: How many unit squares does the line

A scanner dot moves along the black line from bottom-left to top-right. As soon as this

dot enters a new grid square, that entire square is instantly filled with yellow color and stays yellow. A square only turns yellow if the scanner dot on the line has entered it.

segment pass through in

the given grid diagram? A: 16

Static camera, no zoom.

- ✗ Bad

[Figure 343]

- ✗ Bad

Input Image: Reasoning Video:

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

1st frame

cropped_0.jpg

###### II. Question: Text-to-Video Prompt:

Q: How many rectangles

Highlight only the rectangles in the figure with a bright yellow color. Not highlight any other shapes like squares, triangles, circles, or irregular polygons. Static camera,

are there in the figure?

A: 8

no zoom, no pan.

[Figure 350]

Input Image: Reasoning Video:

[Figure 351]

[Figure 352]

1st frame

[Figure 353]

[Figure 354]

[Figure 355]

cropped_0.jpg

###### III. Question: Text-to-Video Prompt:

Q: How many rectangles are there in the figure?

Label all the fish with increasing numbers (1, 2, 3, ...). The fish keep static. Static camera, no zoom, no pan.

A: 18

[Figure 356]

✗ Bad

Input Image: Reasoning Video:

1st frame

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

| |
|---|

- Figure 14: Showcase of 2D Object Counting Reasoning by Veo-3. Veo-3’s lack of spatial control often introduces object motion, undermining the stability and accuracy of the counting process.

##### 2.11 GUI Reasoning

Task Description and Evaluated Aspects. In the Graphical User Interface (GUI) reasoning task, we focus on the capability to understand and interact with graphical user interfaces across different operating systems, including Android, Linux, and Web environments. In each instance, the model is required to perform actions, such as clicking on specific UI elements. The evaluation focuses on the accuracy of the click and the temporal coherence of the interaction, ensuring the scene and irrelevant UI elements remain consistent.

Definition of Good / Moderate / Bad. We define the evaluation criteria in three levels:

✓ Good: The click is precise, with no extraneous actions. No superfluous icons appear, and the original data and icons remain unchanged.

~ Moderate: The click is precise but may be accompanied by minor extraneous actions. Superfluous icons might appear but do not obscure the click target, and original data or icons show only slight alterations.

cropped_0.jpg

IV. Question: Text-to-Video Prompt:

Q: How many tiny things have the same material as the green motorbike?

Draw bounding boxes around the tiny things that have the same material as the green motorbike. Static shot.

A: 1

[Figure 362]

✗ Bad

Input Image: Reasoning Video:

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

1st frame

cropped_0.jpg

- V. Question: Text-to-Video Prompt:

Q: There is a small yellow object that is to the left of the tiny metal motorbike;

how many brown metal mountain bikes

are to the right of it?

A: 1

1st frame

cropped_0.jpg

Input Image: Reasoning Video:

Draw bounding boxes around any matte tandem bikes and metal cruisers present in the

scene. Static shot.

- VI. Question: Text-to-Video Prompt:

Q: How many cyan things

are matte tandem bikes or metal cruisers?

A: 1

1st frame

| |
|---|

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

✓ Good Success Rate: 100%

[Figure 377]

[Figure 378]

cropped_0.jpg

Input Image: Reasoning Video:

Pan smoothly to include both the lid–body interface and the spout or cap in view at a fixed

scale, keeping exposure steady and avoiding any visual or geometric changes.

- VII. Question: Text-to-Video Prompt:

Draw bounding boxes around the brown metal mountain bikes to the right of the origami crane. Static shot.

Input Image: Reasoning Video:

[Figure 379]

✓ Good Success Rate: 33%

Q: How many burners are on the stove? A: 4

[Figure 380]

~ Moderate Success Rate: 17%

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

1st frame

- Figure 15: Showcase of 3D Object Counting Reasoning by Veo-3. Veo-3’s basic 3D counting abilities are challenged by complex materials, geometric variations, and imprecise camera control.

###### 5. GUI

cropped_0.jpg

###### I. Question: Text-to-Video Prompt:

Q: Collapse the pkgs folder. A:

Click the pkgs folder to collapse it. Static shot.

[Figure 387]

[Figure 388]

###### ✗ Bad

Input Image: Reasoning Video:

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

1st frame

cropped_0.jpg

###### II. Question: Text-to-Video Prompt:

Q: A calendar icon located to the right of the flight date options, next to the price display for June 6.

A:

[Figure 394]

Click the calendar icon located to the right of the flight date options, next to the price display for June 6. Static shot.

[Figure 395]

###### ✗ Bad

Input Image: Reasoning Video:

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

1st frame

cropped_0.jpg

III. Question: Text-to-Video Prompt:

Q: A navigation arrow located at the right edge of the browse by category carousel.

A:

[Figure 401]

Click the navigation arrow located at the right edge of the browse by category carousel. Static shot.

[Figure 402]

✗ Bad

Input Image: Reasoning Video:

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

1st frame

- Figure 16: Showcase of GUI Reasoning by Veo-3. Veo-3’s attempts at graphical interface interaction exhibit visual inconsistencies and logical inaccuracies, indicating only a shallow grasp of underlying GUI logic. Note that the answer to each question is a bounding box. For visual clarity, screenshots with the ground-truth bounding boxes are shown.

✗ Bad: The click is imprecise or erratic. Original data and icons are significantly altered, hindering judgment and assessment.

Data Source. The Linux data are selected from the Common Linux Screenshot subset of ScreenSpotPro [42], while the Android and Web data are drawn from the OS Android and OS Web subsections of MMBench-GUI [67], respectively.

Example and Analysis. Across the three cases in Figure 16, Veo-3 fails to accurately capture the correct click position and often exhibits inconsistencies between the click location and the resulting on-screen effect. In addition, it occasionally alters or generates new icons and text, which can interfere with judgment. In the Web system in case III, however, the model demonstrates partial GUI responsiveness and provides some degree of visual feedback.

cropped_0.jpg

###### I. Question†: Text-to-Video Prompt:

A:

Q: Which point corresponds to

Pan to the banana while keeping tray edges in view. Fix scale (banana ~two-thirds of the frame, axis horizontal). Sweep once along the inner concave edge from stem to tip

[Figure 409]

the affordance for

at constant speed, then stop and hold at its midpoint.

manipulating the banana?

[Figure 410]

~ Moderate Success Rate: 33%

Input Image: Reasoning Video:

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

1st frame

cropped_0.jpg

###### II. Question†: Text-to-Video Prompt:

A:

Q: Which set of 4

Keep the cucumber’s start and the pot opening in view. Sweep once from start to pot

points is a right

[Figure 416]

at fixed scale and speed, briefly dwelling at four evenly spaced waypoints (p1→p4) along the path, then hold on both endpoints.

trajectory when doing place a

cucumber into a pot?

[Figure 417]

✗ Bad

Input Image: Reasoning Video:

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

1st frame

cropped_0.jpg

###### III. Question†: Text-to-Video Prompt:

Q: Is the container sealed? A: No.

Pan smoothly to include both the lid–body interface and the spout or cap in view at a fixed scale, keeping exposure steady and avoiding any visual or geometric changes.

[Figure 423]

~ Moderate Success Rate: 17%

Input Image: Reasoning Video:

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

1st frame

- Figure 17: Showcase of Embodied Reasoning by Veo-3. It illustrates plausible static affordance detection in simple settings, common workaround/hallucination behaviors for dynamic manipulations, and failures to reliably localize or preserve manipulation-relevant context. † Green points in the answer image denote ground-truth points or trajectories.

##### Takeaway 11

Veo-3 demonstrates a limited awareness of GUI click actions, imitating interaction behaviors without fully grasping the underlying functional logic.

##### 2.12 Embodied Reasoning

Task Description and Evaluated Aspects. This category evaluates the model’s potential to perceive and reason about object affordances and manipulation dynamics. It involves recognizing both static and dynamic affordances, as well as identifying manipulation-relevant object and scene attributes. Evaluation focuses on two aspects: (i) the generation of stable and contextually relevant visual sequences, and (ii) the maintenance of reasoning fidelity without resorting to implausible planning shortcuts or hallucinated interactions.

Definition of Good / Moderate / Bad. We define the evaluation criteria in three levels:

✓ Good: The sweep/framing covers all candidates fairly (equal or near-equal dwell), centers the manipulation-relevant geometry (e.g., handle + frame/gap, lid-body interface, hinge side) with crisp focus and stable scale; no cropping of key context; no content alterations.

~ Moderate: The view roughly includes the right region(s) but with minor bias or coverage issues: slight off-center, brief under-exposure of one candidate, small motion jitter, or shallow context (still enough to infer).

✗ Bad: The camera misses or biases the evidence (e.g., lingers only on one point, crops away the hinge/rail, over-zooms a non-relevant patch), introduces distortion/content edits, or produces footage from which a fair decision cannot be made.

Data Source. We select samples from Robobench [51] for the analysis. In addition to a general understanding of static attributes, we also sample data to assess whether Veo-3 can perform direct reasoning on tasks involving the generation of static and dynamic affordances.

Example and Analysis. As shown in Figure 17, Veo-3 demonstrates the ability to comprehend objects within real-world scenes. However, its capacity for assisting visual reasoning in embodied scenarios remains constrained by insufficient stability. As illustrated in case I, when provided with a clearly defined object for manipulation, Veo-3 is capable of generating plausible manipulation affordances. When it comes to dynamic affordances, Veo-3 tends to employ workarounds to compensate for its planning deficiencies, as evidenced in case II, where it generated a new cucumber instead of the intended object. With respect to static attributes, Veo-3 struggles to accurately differentiate visual prompts and misidentifies the position of containers. As shown in case III, the green box, intended to specify the location of the container, inadvertently led Veo-3 to produce hallucinations.

##### Takeaway 12

Veo-3’s capabilities are currently limited to basic object recognition rather than true embodied reasoning. It lacks the necessary planning and stability to reliably interpret and act upon dynamic or spatially constrained instructions, indicating its limitations in understanding and reasoning of real-world interactions.

##### 2.13 Medical Reasoning

Task Description and Evaluated Aspects. This category assesses the model’s ability to localize lesions or structures, identify relevant attributes (e.g., side, lobe), recognize pathological patterns (e.g., “jump distribution”), and make binary decisions (e.g., presence or absence). The evaluation focuses on both the correctness of object manipulation and the visual stability of the surrounding regions.

Definition of Good / Moderate / Bad. We define the evaluation criteria in three levels:

✓ Good: The camera cleanly settles on the correct anatomical level/lesion, with clear margins and readable context; motion is reasonable; no geometric distortion or content alteration.

~ Moderate: The view roughly covers the right area but is slightly off (partial coverage, mild blur, small framing mistakes). The general shape of the tissue or organ can still be observed.

✗ Bad: The video misses the target region or introduces distortions/crops that hide key cues. Tissues or organs begin to distort. Misleading results due to confusion of medical terminology.

Data Source. We select samples representing different body parts from the ViTAR [9] dataset.

Example and Analysis. We showcase the evaluation results in Figure 18. Veo-3 retains the ability to manipulate images when dealing with medical images. However, due to its lack of medical

cropped_0.jpg

###### I. Question: Text-to-Video Prompt:

Show the full sagittal lumbar view, then sweep smoothly from L1 to S1 at constant speed

Q: What is the distribution pattern of stenotic segments? A: Jump distribution.

without pausing. End on a view showing adjacent disc spaces, including narrow and normal levels. Keep image content and geometry unchanged.

- ✗ Bad

[Figure 429]

- ✗ Bad

Input Image: Reasoning Video:

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

1st frame

cropped_0.jpg

###### II. Question: Text-to-Video Prompt:

Show the full PA chest view, then adjust framing to include both the heart silhouette and

Q: Is cardiomegaly present？ A: No.

the widest inner thoracic diameter at a fixed scale. Keep contrast and geometry unchanged, holding steady for visual CTR estimation.

[Figure 436]

✗ Bad

Input Image: Reasoning Video:

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

1st frame

cropped_0.jpg

###### III. Question: Text-to-Video Prompt:

| |
|---|

Q: Which lobe contains

Show the full axial CT, then pan and zoom smoothly to the right lung so the nodule and

the pulmonary nodule?

nearby fissure appear together. Keep windowing standard and geometry unchanged.

A: Left lobe.

[Figure 443]

Input Image: Reasoning Video:

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

1st frame

- Figure 18: Showcase of Medical Reasoning by Veo-3. As shown in cases I and III, Veo-3 fails to maintain the shape of the rest of medical organization. Veo-3 also can not understand and precisely locate the mentioned medical terminology in the prompt, as demonstrated in case II.

knowledge, Veo-3 struggles to accurately manipulate the correct objects when instructions include medical terminology. This phenomenon is evident across all cases. Furthermore, Veo-3 cannot model medical organs effectively. When performing operations such as zooming in, the medical images suffer from significant distortion, resulting in a substantial loss of detail.

##### Takeaway 13

Veo-3’s failure to handle the reasoning in the medical domain, causing distortion even on simple zoom-ins, highlights its limited grasp of specialized, non-general knowledge.

|MME-COF<br><br>Figure 19: Category Distribution.|
|---|

##### Table 1: Key Statistics of MME-COF.

Statistic Number

Total entries 59 Total categories 12 Max prompt length 124 Avg prompt length 36.7 Max entries per category 7 Avg entries per category 4.9

#### 3 MME-COF

##### 3.1 Benchmark Overview

To standardize the empirical study and systematically evaluate the reasoning potential of state-of-theart generative video models [21, 55, 56], we introduce MME-COF, which, to our knowledge, is the first benchmark specifically designed to reveal and quantify the reasoning potential of video models.

##### 3.2 Benchmark Composition

Data Curation and Distribution. Aligning with the task taxonomy in Section 2.1, the MME-COF benchmark is curated from the cases used in our empirical study. It comprises 59 curated entries and instruction prompts spanning 12 diverse reasoning categories. The key statistics of MME-COF and its overall composition are summarized in Table 1, Figure 2b and Figure 19.

Review Process. Following the prompt design protocol in Section 2.1, all prompts undergo a two-stage review process. In the cross-validation phase, each prompt was independently reviewed by another expert to ensure semantic clarity, alignment with the intended reasoning task, and the absence of linguistic bias. In the final adjudication phase, discrepancies were jointly discussed and resolved through consensus. This multi-step procedure ensured that every prompt was conceptually precise, visually grounded, and fully aligned with the evaluation objectives of MME-COF.

##### 3.3 Evaluation Protocol

Models and Generation Settings. We evaluate the leading video models in a zero-shot setting, including Kling-v1 [38], Seedance-1.0-pro [19], Veo-3.0-preview [70], Veo-3.0-fast [70], Sora-2 [56], Sora-2-pro [56]. Each model generates six video samples per prompt, and final scores were computed as the mean across all samples. All videos are generated at a 16:9 aspect ratio. We adopt the default 8-second duration for the Sora and Veo series, while retaining the default 5-second length for Kling and Seedance. Note that, since most video models apply automated safety filters and content moderation, which may block sensitive content, we exclude videos that are suppressed by such filters from our evaluation.

Evaluation Metrics. We employ Gemini-2.5-Pro [12] as an automatic verifier to evaluate each generated video. Gemini is prompted with the following evaluation criteria and returns structured scores between 0 and 4, where higher values indicate better performance:

- 1) Instruction Alignment (0-4): Measures how well the video follows the described structure and sequence in the prompt. A high score indicates that the visual steps faithfully reflect the textual instructions.
- 2) Temporal Consistency (0-4): Evaluates the smoothness and continuity between frames. Disjointed or abrupt transitions will lead to a lower score.

- Table 2: Model-level Overall and Per-dimension Performance on MME-COF. Mean scores and standard deviations are reported on a 0–4 scale, as graded by Gemini-2.5-Pro.

Model Overall

Instruction Alignment

Temporal Consistency

Visual Stability

Content Fidelity

Focus Relevance

Kling-v1 [38] 0.64 ± 0.91 0.01 ± 0.09 0.15 ± 0.75 2.43 ± 1.86 0.21 ± 0.79 0.43 ± 1.07 Seedance-1.0-pro [19] 1.41 ± 1.51 0.30 ± 0.86 1.65 ± 1.57 2.00 ± 1.72 1.13 ± 1.65 1.98 ± 1.75 Veo-3.0-fast [21] 1.44 ± 1.51 0.56 ± 1.09 1.37 ± 1.51 1.88 ± 1.73 1.10 ± 1.52 2.27 ± 1.69 Veo-3.0-preview [21] 1.45 ± 1.50 0.54 ± 1.06 1.43 ± 1.53 1.89 ± 1.71 1.12 ± 1.49 2.26 ± 1.73 Sora-2-pro [56] 1.66 ± 1.53 0.48 ± 0.96 1.36 ± 1.59 2.39 ± 1.65 1.64 ± 1.72 2.44 ± 1.73 Sora-2 [56] 1.72 ± 1.59 0.59 ± 1.12 1.52 ± 1.69 2.32 ± 1.68 1.62 ± 1.75 2.52 ± 1.71

- Table 3: Per-category Scores on MME-COF. Mean scores and standard deviations are reported on a 0–4 scale, as graded by Gemini-2.5-Pro.

Seedance-1.0 Pro [19]

Veo-3.0 Fast [21]

Veo-3.0 Preview [21]

Sora-2 Pro [56]

Category Kling-v1 [38]

Sora-2 [56]

Visual Detail 0.72 ± 0.69 1.37 ± 1.39 1.10 ± 1.24 1.59 ± 1.68 1.14 ± 1.32 1.08 ± 1.89 Visual Trace 0.49 ± 0.65 1.23 ± 1.13 1.43 ± 1.26 1.48 ± 1.24 1.51 ± 1.37 1.75 ± 1.31 Real-world Spatial 0.77 ± 0.76 1.79 ± 1.53 2.07 ± 1.54 2.10 ± 1.46 1.84 ± 1.43 1.77 ± 1.35

- 3D Geometry 0.61 ± 0.58 1.95 ± 1.64 1.71 ± 1.54 1.54 ± 1.43 1.37 ± 1.49 1.42 ± 1.45 2D Geometry 0.49 ± 0.67 0.96 ± 1.11 1.18 ± 1.15 1.27 ± 1.20 1.77 ± 1.45 1.77 ± 1.21 Physics-based 0.60 ± 0.62 1.27 ± 1.25 1.44 ± 1.39 1.44 ± 1.35 2.13 ± 1.32 2.10 ± 1.33 Rotation 0.22 ± 0.34 2.30 ± 1.46 1.83 ± 1.44 1.60 ± 1.29 1.62 ± 1.37 1.44 ± 1.28 Table & Chart 0.87 ± 0.72 0.71 ± 1.18 0.82 ± 1.30 0.96 ± 1.44 1.84 ± 1.61 1.48 ± 1.59 GUI 1.09 ± 0.51 0.70 ± 0.76 1.11 ± 1.09 1.18 ± 0.89 1.88 ± 1.64 1.52 ± 1.48 Object Counting 0.64 ± 0.58 1.15 ± 0.97 2.03 ± 1.42 1.84 ± 1.42 2.06 ± 1.48 1.86 ± 1.41 Embodied 0.80 ± 0.00 1.82 ± 1.67 1.33 ± 1.57 1.18 ± 1.46 1.30 ± 1.51 1.40 ± 1.42 Medical 1.15 ± 1.17 1.56 ± 1.41 0.27 ± 0.39 0.30 ± 0.58 2.08 ± 1.56 1.81 ± 1.42

- 3) Visual Stability (0-4): Assesses the stability of the video in terms of camera motion, object appearance, and scene composition. Shaky or glitchy outputs are penalized.
- 4) Content Fidelity (0-4): Determines how accurately the key elements described in the prompt are preserved. Hallucinated or missing objects/events will reduce the score.
- 5) Focus Relevance (0-4): Examines whether the video’s visual attention remains focused on the correct objects or regions throughout. Irrelevant distractions or poorly framed targets are penalized.

We adopt a direct prompting strategy, instructing Gemini with the prompt, videos, and evaluation criteria to produce numerical scores in JSON format directly.

##### 3.4 Quantitative Results and Analysis

We report the quantitative scores of the five evaluated models across the five reasoning dimensions in Table 2, and provide detailed per-category results in Table 3 and Figure 2a.

Overall, most models exhibit limited reasoning capability across all tasks in MME-COF, reflected by generally low scores. Among the five dimensions, Visual Stability achieves the highest average, indicating that current video models can generate smooth and coherent sequences. Yet, their behavior remains largely at the level of pattern replay rather than genuine reasoning.

The Sora-2 series [56] shows relative advantages in physics-based, embodied, and medical reasoning, while the Veo-3.0 series [21] performs comparatively better in real-world spatial reasoning. Seedance-1.0-pro [19] demonstrates relative strength in rotation and 3D geometry reasoning. These trends suggest that different models specialize in distinct reasoning aspects. However, their mean scores remain below 2.0 out of 4, highlighting substantial room for improvement and pointing to opportunities for more targeted enhancement in future development.

#### 4 Related Work

Video Models. Video models have been progressively evolving both in the fields of video understanding and generation. For video understanding methods, earlier approaches, such as MViT [14], Video Swin Transformer [48], and VideoMAE [62], aim to learn a robust representation that fosters downstream tasks. With the rise of LLMs, recent approaches encode videos as tokens and exploit the language backbone for captioning [61], event localization [59], and high-level reasoning [28, 83]. Video generation models have also attracted much attention. Closed system, including OpenAI’s Sora [55, 56], Runway’s Gen-3 [58], Pika Labs [57], Luma AI [50], and Google DeepMind’s Veo series [20, 21], have exhibited impressive results. However, they remain inaccessible due to their closed-source nature. Open-source alternatives have recently become available: Stable Video Diffusion [6] introduces efficient training strategies, Hunyan-Video [37] proposes systematic scaling, and Wan-2.1 [64] presents an efficient 3D VAE with expanded pipelines.

Reasoning with Video. The advent of large reasoning models [24, 60, 27, 69], such as OpenAI o1 [54] and DeepSeek-R1 [23], has spurred the development of video reasoning benchmarks. Most current methods [15, 44, 53] employ MLLMs specialized in video reasoning understanding. For example, Video-R1 [15] specifically targets temporal reasoning capabilities by introducing a temporal group relative policy optimization (GRPO) loss. VideoChat-R1 [44] focuses on spatio-temporal reasoning abilities by training with GRPO and rule-based rewards. A two-stage training strategy, combining SFT and RL, is used by VideoRFT [65]. When trained on vast collections of images and videos, this strategy boosts the model’s ability to handle QA tasks, whether in general contexts or reasoning-focused ones. These methods primarily focus on enhancing specific types of question-answering or captioning tasks. Concurrently, [70] demonstrates the large potential of video generative models in video reasoning. These models have implicitly acquired world knowledge throughdemonstrates impressive performance on various tasks, includinging and reasoning capability. Yet, this direction has rarely been explored and only experimented with in zero-shot settings.

Evaluation of Video Models as Zero Shot Learner. Recently, several works have been exploring the zero-shot capability of video generation models in various domains, including general-purpose vision understanding [70, 17], medical imaging [39], and world models [68]. [70] conducts experiments on Veo 3 with a variety of vision tasks that have not been explicitly included during training. The video model showcases surprising performance on multiple tasks like object segmentation, image editing, and even maze solving. [39] later adopts a similar paradigm to medical images understanding tasks and finds video generation models also show powerful capabilities, e.g., delineation of anatomical structures in CT scans, medical image segmentation, and even forecasting of future 3D CT phases. Besides, [68] shows that video generation models could also understand complex temporal causality and world knowledge in the real world, thereby serving as a world model [2, 33].

#### 5 Conclusions and Insights

Video models demonstrate an intuitive understanding of the simple visual world. Recent video models can generate high-fidelity videos with realistic motion dynamics, suggesting that they have internalized substantial visual and structural knowledge about the world. Through qualitative results from our empirical study and quantitative results from the MME-COF benchmark, our work confirms that these models do exhibit intuitive yet local reasoning potential. This emergent behavior, which aligns with the “Chain-of-Frame” (CoF) mechanism, is revealed across several common success patterns. (i) Fine-grained Grounding. Models demonstrate a capability for fine-grained attribute and spatial grounding, especially when targets are visually distinct, as presented in visual detail reasoning tasks. (ii) Short-horizon Trace Consistency. In Visual Trace Reasoning tasks, models can maintain short-term consistency in visual traces. (iii) Emergent Tool-Use Simulation. An emergent ability to follow CoF instructions that mimic tool-use is presented, such as drawing lines in 2D geometry, highlighting targets in object counting, or controlling the camera in table and chart reasoning. (iv) Foundational Spatial and Geometric Grasp. This includes single-step 3D geometry transformations, understanding basic real-world spatial layouts, finding coherent sequential paths, and handling small-angle Rotations. (v) Preliminary Real-world Interaction. Models display a preliminary comprehension of real-world interaction, generating coherent manipulation paths in embodied reasoning.

Complex visual reasoning reveals fundamental limitations. However, visual reasoning demands more than these foundational skills. It tests a model’s ability to maintain long-horizon logical consistency, adhere to abstract constraints, and understand functional principles. In these complex areas, our study reveals fundamental limitations and several common failure patterns. (i) Causal and Physical Logic. This is evident in physics-based reasoning, where the model generates implausible motion that violates basic causal principles, and in visual trace reasoning, where the generated sequences break causal order with illogical steps. (ii) Long-horizon and Rule-grounded Reasoning. In visual trace reasoning, models fail to maintain state and adhere to task-specific rules over extended sequences. (iii) Geometric and Spatial Logic. Models fail at multi-step or complex transformations in 3D/2D geometry and real-world spatial tasks, often breaking constraints or prioritizing visual plausibility over correctness. (iv) Functional and Interaction Logic. They merely imitate GUI actions without grasping their purpose and lack the necessary planning and stability for reliable Embodied tasks, often resorting to workarounds. (v) Perceptual Precision and Specialized Knowledge. This weakness appears when models fail to identify small or indistinct targets in visual detail reasoning, distort data in table and chart tasks, and fail to process specialized medical imagery due to a lack of domain understanding.

Current video models are not yet ready as standalone zero-shot reasoners. Overall, our findings show that current video models are not yet reliable as standalone zero-shot reasoners. Strong generative performance does not automatically imply robust reasoning during inference. The model’s behavior appears to be driven more by learning surface-level patterns and correlations rather than by internalizing general principles. It excels at short-term coherence rather than long-horizon causality. This is evident when the model prioritizes visual plausibility over precise spatial reasoning, or favors visually symmetric patterns over strictly adhering to geometric instructions. This tendency to produce plausible but instructionally flawed outputs reveals a reasoning process that is pattern-driven, not principle-driven, thereby undermining its ability to function as a standalone zero-shot reasoner.

The potential in advancing next-generation collaborative visual reasoning. Despite these limitations, the emergent behaviors observed in video models signal strong potential. The CoF concept suggests a novel modality for reasoning through visual problems step by step. While these models are not yet robust standalone reasoners, their foundational capabilities demonstrate that they can be guided through carefully designed prompts. This suggests a path where video models exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.
- [5] Zechen Bai, Hai Ci, and Mike Zheng Shou. Impossible videos. arXiv preprint arXiv:2503.14378, 2025.
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [7] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent

- diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.
- [8] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.
- [9] Kaitao Chen, Shaohao Rui, Yankai Jiang, Jiamin Wu, Qihao Zheng, Chunfeng Song, Xiaosong Wang, Mu Zhou, and Mianxin Liu. Think twice to see more: Iterative visual reasoning in medical vlms. arXiv preprint arXiv:2510.10052, 2025.
- [10] Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-of-thought reasoning. arXiv preprint arXiv:2506.05331, 2025.
- [11] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.
- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.
- [14] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6824–6835, 2021.
- [15] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [16] Zhanzhou Feng, Qingpei Guo, Xinyu Xiao, Ruihan Xu, Ming Yang, and Shiliang Zhang. Unified video generation via next-set prediction in continuous domain. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19427–19438, 2025.
- [17] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. CVPR 2025 Highlight, 2024.
- [18] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.
- [19] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [20] Google DeepMind. Veo 2, 12 2024. Accessed: 2024.
- [21] Google DeepMind. Veo-3 technical report. Technical report, Google DeepMind, May 2025.
- [22] Kaisi Guan, Zhengfeng Lai, Yuchong Sun, Peng Zhang, Wei Liu, Kieran Liu, Meng Cao, and Ruihua Song. Etva: Evaluation of text-to-video alignment via fine-grained question generation and answering. arXiv preprint arXiv:2503.16867, 2025.
- [23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [24] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [25] Meng-Hao Guo, Xuanyu Chu, Qianrui Yang, Zhe-Han Mo, Yiqing Shen, Pei-lin Li, Xinjie Lin, Jinnian Zhang, Xin-Sheng Chen, Yi Zhang, et al. Rbench-v: A primary assessment for visual reasoning models with multi-modal outputs. arXiv preprint arXiv:2505.16770, 2025.
- [26] Meng-Hao Guo, Xuanyu Chu, Qianrui Yang, Zhe-Han Mo, Yiqing Shen, Pei-Lin Li, Xinjie Lin, Jinnian Zhang, Xin-Sheng Chen, Yi Zhang, Kiyohiro Nakayama, Zhengyang Geng, Houwen Peng, Han Hu, and Shi-Min Hu. Rbench-v: A primary assessment for visual reasoning models with multi-modal outputs. 2025.
- [27] Ziyu Guo*, Renrui Zhang*, Chengzhuo Tong*, Zhizheng Zhao*, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Can we generate images with cot? let’s verify and reinforce image generation step by step. CVPR 2025, 2025.
- [28] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.
- [29] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025.
- [30] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.
- [31] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025.
- [32] Emily Jin, Jiaheng Hu, Zhuoyi Huang, Ruohan Zhang, Jiajun Wu, Li Fei-Fei, and Roberto Martín-Martín. Mini-behavior: A procedurally generated benchmark for long-horizon decisionmaking in embodied ai. arXiv preprint arXiv:2310.01824, 2023.
- [33] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Huang Gao, and Jiashi Feng. How far is video generation from world model? – a physical law perspective. arXiv preprint arXiv:2406.16860, 2024.
- [34] Yoonsik Kim, Moonbin Yim, and Ka Yeon Song. Tablevqa-bench: A visual question answering benchmark on multiple table domains. arXiv preprint arXiv:2404.19205, 2024.
- [35] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.
- [36] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [37] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [38] Kuaishou Technology. Kling ai: Next-generation ai creative studio. https://klingai.com/, June 2024.
- [39] Yuxiang Lai, Jike Zhong, Ming Li, Yuheng Li, and Xiaofeng Yang. Are video models emerging as zero-shot learners and reasoners in medical imaging? arXiv preprint arXiv:2510.10254, 2025.

- [40] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [41] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025.
- [42] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.
- [43] Linjie Li, Mahtab Bigverdi, Jiawei Gu, Zixian Ma, Yinuo Yang, Ziang Li, Yejin Choi, and Ranjay Krishna. Unfolding spatial cognition: Evaluating multimodal models on visual simulations. arXiv preprint arXiv:2506.04633, 2025.
- [44] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.
- [45] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14963–14973, 2023.
- [46] Dairu Liu, Ziyue Wang, Minyuan Ruan, Fuwen Luo, Chi Chen, Peng Li, and Yang Liu. Visual abstract thinking empowers multimodal reasoning. arXiv preprint arXiv:2505.20164, 2025.
- [47] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation. Advances in Neural Information Processing Systems, 36:62352–62387, 2023.
- [48] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3202–3211, 2022.
- [49] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [50] LumaLabs. Dream machine, 06 2024. Accessed: 2024.
- [51] Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Mengdi Zhao, Bo-Wen Zhang, Cheng Chi, Jiaming Liu, Gaole Dai, Rongyu Zhang, Ruichuan An, Kun Wu, Zhengping Che, Shaoxuan Xie, Guocai Yao, Zhongxia Zhao, Pengwei Wang, Guang Liu, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobench: A comprehensive evaluation benchmark for multimodal large language models as embodied brain, 2025.
- [52] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May

2022. Association for Computational Linguistics.

- [53] Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, and Zhuochen Wang. Open-o3 video: Grounded video reasoning with explicit spatio-temporal evidence. arXiv preprint arXiv:2510.20579, 2025.
- [54] OpenAI. Openai o1 system card. https://openai.com/index/ openai-o1-system-card/, December 2024. Accessed: 2024-12-05.
- [55] OpenAI. Video generation models as world simulators. Technical report, OpenAI, 2024.
- [56] OpenAI. Sora 2 system card. Technical report, OpenAI, September 2025.

- [57] PikaLabs. Pika 1.5, 10 2024. Accessed: 2024.
- [58] Runway. Introducing gen-3 alpha: A new frontier for video generation. https://runwayml. com/research/introducing-gen-3-alpha/, June 2024.
- [59] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event localization in unconstrained videos. In Proceedings of the European conference on computer vision (ECCV), pages 247–263, 2018.
- [60] Chengzhuo Tong*, Ziyu Guo*, Renrui Zhang*, Wenyu Shan*, Xinyu Wei, Zhenghao Xing, Hongsheng Li, and Pheng-Ann Heng. Delving into rl for image generation with cot: A study on dpo vs. grpo. arXiv preprint arXiv:2505.17017, 2025.
- [61] Tony Cheng Tong, Sirui He, Zhiwen Shao, and Dit-Yan Yeung. G-veval: A versatile metric for evaluating image and video captions using gpt-4o. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7419–7427, 2025.
- [62] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [63] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [64] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [65] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025.
- [66] Siting Wang, Minnan Pei, Luoyang Sun, Cheng Deng, Kun Shao, Zheng Tian, Haifeng Zhang, and Jun Wang. Spatialviz-bench: An mllm benchmark for spatial visualization. arXiv preprint arXiv:2507.07610, 2025.
- [67] Xuehui Wang, Zhenyu Wu, JingJing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, et al. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478, 2025.
- [68] Zeqing Wang, Xinyu Wei, Bairui Li, Zhen Guo, Jinrui Zhang, Hongyang Wei, Keze Wang, and Lei Zhang. Videoverse: How far is your t2v generator from a world model? arXiv preprint arXiv:2510.08398, 2025.
- [69] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [70] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [71] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024.
- [72] Qiucheng Wu, Handong Zhao, Michael Saxon, Trung Bui, William Yang Wang, Yang Zhang, and Shiyu Chang. Vsp: Assessing the dual challenges of perception and reasoning in spatial planning tasks for vlms. arXiv preprint arXiv:2407.01863, 2024.
- [73] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

- [74] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.
- [75] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [76] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023.
- [77] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [78] Chenyu Zhang, Daniil Cherniavskii, Andrii Zadaianchuk, Antonios Tragoudaras, Antonios Vozikis, Thijmen Nijdam, Derck WE Prinzhorn, Mark Bodracska, Nicu Sebe, and Efstratios Gavves. Morpheus: Benchmarking physical reasoning of video generative models with real physical experiments. arXiv preprint arXiv:2504.02918, 2025.
- [79] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? ECCV 2024, 2024.
- [80] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv e-prints, pages arXiv–2407, 2024.
- [81] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024.
- [82] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.
- [83] Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, et al. Mmvu: Measuring expert-level multi-discipline video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8475–8489, 2025.
- [84] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [85] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 26183–26191, 2025.

