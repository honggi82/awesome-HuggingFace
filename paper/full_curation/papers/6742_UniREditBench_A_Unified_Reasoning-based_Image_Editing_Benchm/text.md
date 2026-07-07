#### UniREditBench: A Unified Reasoning-based Image Editing Benchmark

##### Feng Han1,2*, Yibin Wang1,2*, Chenglin Li2,3, Zheming Liang2, Dianyi Wang1,2, Yang Jiao1, Zhipeng Wei4, Chao Gong1, Cheng Jin1,2, Jingjing Chen1†, Jiaqi Wang2† 1Fudan University, 2Shanghai Innovation Institute, 3Zhejiang University, 4UC Berkeley Project Page: maplebb.github.io/UniREditBench

### arXiv:2511.01295v2[cs.CV]22Nov2025

[Figure 1]

(a) Real World Multi-Object Interaction

(b) Game World Logic/Strategy Reasoning

UniREdit-Bagel (Ours)

UniREdit-Begal (Ours)

Input Image GPT-4o

Input Image GPT-4o

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Edit

Edit

[Figure 8]

[Figure 9]

Instruction: Place the paddle into the water beside the canoe and draw it backward through the water.

Instruction: Push the box onto the target. Boxes cannot be pulled or pushed through walls or other boxes.

UniREdit-Bagel (Ours)

UniREdit-Bagel (Ours)

Input Image Nano Banana

Nano Banana

Input Image

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Edit

Edit

[Figure 16]

[Figure 17]

Instruction: Drive the pointed end of the dart firmly into the surface of the beach ball and maintain contact.

Instruction: Swap the gem at (2,3) with (3,3).

Figure 1. UniREditBench covers both real-world and game-world reasoning scenarios across 8 primary dimensions and 18 sub-dimensions. We provide qualitative editing cases of (a) real-world multi-object interaction, and (b) game-world logical/strategy reasoning.

##### Abstract

ation, providing both textual and ground-truth image references for each sample assessment. Furthermore, we design an automated multi-scenario data synthesis pipeline and construct UniREdit-Data-100K, a large-scale synthetic dataset with high-quality chain-of-thought (CoT) reasoning annotations. We fine-tune Bagel on this dataset and develop UniREdit-Bagel, demonstrating substantial improvements in both in-domain and out-of-distribution settings. Through thorough benchmarking of both open-source and closedsource image editing models, we reveal their strengths and weaknesses across various aspects.

Recent advances in multi-modal generative models have driven substantial improvements in image editing. However, current generative models still struggle with handling diverse and complex image editing tasks that require implicit reasoning, underscoring the need for a comprehensive benchmark to systematically assess their performance across various reasoning scenarios. Existing benchmarks primarily focus on single-object attribute transformation in realistic scenarios, which, while effective, encounter two key challenges: (1) they largely overlook multi-object interactions as well as game-world scenarios that involve human-defined rules, which are common in real-life applications; (2) they only rely on textual references to evaluate the generated images, potentially leading to systematic misjudgments, especially in complex reasoning scenarios. To this end, this work proposes UniREditBench, a unified benchmark for reasoning-based image editing evaluation. It comprises 2,700 meticulously curated samples, covering both real- and game-world scenarios across 8 primary dimensions and 18 sub-dimensions. To improve evaluation reliability, we introduce multimodal dual-reference evalu-

##### 1. Introduction

Recent advances in multimodal generative models have led to remarkable improvements in instruction-conditioned image editing. Generative models [3, 33, 38, 40, 44, 47, 49], including Step1X-Edit [21], FLUX-Kontext [2], Bagel [6], Nano Banana [7], and GPT-4o [13], have demonstrated a powerful ability to understand diverse textual instructions and generate semantically consistent image edits. In parallel, reinforcement learning-based training strategies [19, 31, 36, 43] are continuously advancing, further enhancing the capabilities of image editing models. With

∗Equal contribution. †Corresponding author.

Table 1. Reasoning-based image editing benchmark comparison. Our UniREditBench excels in broader scenario and evaluation dimension coverage. “S-Obj” indicates single-object while “M-Obj” indicates multi-object.

Real World Scenario Game World Scenario Attribute Temporal Pose Spatial Motion Mechanic Medium

Benchmark Size Reference Images

Logical Long-planing Strategic Spatial (S-Obj) (S-Obj) (S-Obj) (M-Obj) (M-Obj) (M-Obj) (M-Obj)

SmartEdit [12] 219 219 ✔ ✔ RISE [51] 360 70 ✔ ✔ ✔ ✔ ✔ KRIS [41] 1,267 50 ✔ ✔ ✔ ✔ ✔ ✔ UniREditBench 2,700 2,700 ✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔

Input Image Edited Image

these rapid developments, the need for a more comprehensive benchmark to evaluate model editing capabilities across different aspects has become increasingly essential. Early benchmarks [21, 46] focus on local details or global stylistic changes, e.g., style transfer, color alteration, and object removal. However, they fail to cover editing tasks that require models to perform implicit reasoning [9, 48], which are commonly used in real-life applications. As illustrated in Fig. 1, when editing instructions involving realworld or human-defined game rules, current models often generate results that lack physical plausibility. To this end, recent efforts have introduced reasoning-aware evaluation across temporal, spatial, and logical dimensions [51], and proposed a knowledge-grounded taxonomy assessing factual, conceptual, and procedural knowledge types [41].

[Figure 18]

[Figure 19]

(a) Previous Evaluation w/o Reference Image

(b) Our Evaluation w/ Reference Image

Reference GT Image

[Figure 20]

###### Evaluation:

Evaluation:

[Figure 21]

The output image does have a continuous red path. However, the path is not the same as in the reference image. It includes an unnecessary detour into a dead-end, which is not present in the reference or implied by the description. Final Score: 3

The path does not cross any light-blue wall cells. It follows the correct sequence as described in the groundtruth description. The entire path is continuous, in red, and adheres to maze rules. Final Score: 5

Instruction: Draw one red path from the green start to the red end along walkable white cells.

Gpt-4.1

Textual Reference

[Figure 22]

[Figure 23]

The direction of the path is: down → right → up

→ right → down → right → down → left → down

→ right, and finally reaches the red end.

Figure 2. Image editing evaluation comparison. Current textreference-only evaluation potentially leads to misjudging, while our dual-reference evaluation results in more reliable assessments.

unified benchmark for reasoning-based image editing assessment with broader evaluation dimension coverage and robust evaluation pipeline. Specifically, (1) we adopt a scenario-to-category hierarchical dimension design, covering diverse reasoning types in both real-world and gameworld scenarios (shown in Fig. 1): it includes 2,700 carefully curated samples organized across 8 primary dimensions and 18 sub-categories, e.g., multi-object interaction in real world, and long-horizon game planning in game world. Meanwhile, (2) as illustrated in Fig. 2, in contrast to existing work that relies solely on textual references for evaluation, we introduce additional reference GT images to facilitate direct visual comparison with the generated image. By utilizing the visual cues provided by the reference image, the evaluator is able to more accurately and reliably assess the alignment of the generated image with the given instruction, as shown in Fig. 2 (b). Furthermore, to ensure the diversity and reliability of samples in this benchmark, we design a multi-scenario data synthesis pipeline. Specifically, as shown in Fig. 3, (a) For real-world scenarios, we first handcraft a few reference text prompts, including the original image description, the editing instruction, and the textual reference of edited effect. These prompts are then scaled up using the VLM. Finally, all resulted textual descriptions are directly used to generate pairs of original and edited image. (b) For game-world scenarios, we first design diverse game problems, and then use Python programs to generate image pairs, instructions, and textual reference of edited effects, ensuring both logical and visual correctness in these ruleintensive scenarios [17, 27]. Ultimately, all data samples in UniREditBench undergo VLM-based filtering and human inspection to ensure their reliability and accuracy.

Despite their effectiveness, these benchmarks still face two significant challenges: (1) they primarily focus on single-object attribute changes in realistic scenarios, neglecting multi-object interactions and game-world scenarios involving human-defined rules (see Tab. 1). This narrow scope restricts their ability to evaluate how effectively models generalize across a wider range of complex reasoning contexts; Additionally, (2) they mainly rely on textual reference to evaluate the generated images [41, 51], which may lead to systematic misjudgments, especially in complex reasoning-based editing scenarios (see Fig. 2).

In this work, we posit that: (1) While current models exhibit proficiency in perceptual instruction following and simple reasoning editing settings (e.g., Transform an intact apple to a bitten one), they still struggle with complex reasoning-based image editing that necessitates the comprehension of multi-object interaction characteristics (e.g., Draw the paddle backward through the water) as well as logical constraints of puzzle and game scenarios (e.g., Control the player and push the box to the target), as illustrated in Fig. 1. (2) Relying solely on textual references in evaluating complex reasoning-based image editing task often leads to unreliable judgments. As shown in Fig. 2 (a), the textreference-only evaluator assigns an inflated score even the edited image introduces an additional faulty path. Therefore, we intuitively believe that incorporating a ground-truth (GT) image as an additional visual reference can enable more precise evaluation.

Based on our data synthesis pipeline, we also propose UniREdit-Data-100K, a comprehensive reasoning-based image editing dataset with high-quality chain-of-thought

To this end, this work proposes UniREditBench, a

instructions with accurate visual manipulation. Traditional methods perform editing by altering the diffusion trajectory without requiring additional training, including partial denoising from intermediate SDE steps [22], cross-attention control [11, 30], mask-guided blending [1, 33, 38], CLIP- or diffusion-guided manipulation [15], and latent inversion for fidelity preservation [14, 32]. Besides, several studies employ visual-language models (VLMs) to provide prompts, spatial priors, or synthetic supervision to guide a generative editing model [3, 8, 10, 49, 50]. Recent unified frameworks aim to use a single model for both image understanding and editing in a complementary direction [29, 40, 44]. For instance, Bagel [6] features a think mode that produces reasoning text prior to editing to enhance instruction fidelity and consistency. While effective, current methods still face challenges with complex reasoning-based editing, underscoring the need for comprehensive benchmarks to assess their performance across various reasoning scenarios.

- (a) Real-World Data Synthesis

1. Text Prompt Scale Up 2. Image Synthesis 3. Quality Filter & CoT Generation

- (b) Game-World Data Synthesis

Few Ref.

Original & Edited Images

Gemini 2.5 Pro 1. Description of Original Image

Gemini 2.5 Pro High Quality

Hand-crafted

Text Prompts

[Figure 24]

Image Pair

[Figure 25]

[Figure 26]

2. Instruction 3. Detailed Textual Reference of Edited Effect

[Figure 27]

CoT

Reasoning Text

1. Game Program Generation 2. Image Synthesis 3. Quality Filter & CoT Transformation

Designed Game Problems

Python Programs 1. Original & Edited Images

Gemini 2.5 Pro High Quality

Image Pair

[Figure 28]

[Figure 29]

2. Instruction &

[Figure 30]

Textual Reference 3. Programmatic CoT Text

Natural CoT

Reasoning Text

(c) Case Study of Our Synthetic Data

CoT Reasoning Text (used for training): In the initial image, I observe a carefully constructed house of cards standing upright on a wooden floor, with no visible disturbances. The instruction is to ‘Allow an orange cat to walk past so that it moves against the lowest row of cards.’ From this, I infer that the cat will make contact with the base of the card structure. Given the inherent instability of a house of cards, even a gentle brush against the bottom layer is likely to dislodge or knock over some of the supporting cards, leading to a partial or progressive breakdown of the entire house of cards. all other aspects of the scene—including the background, lighting—will remain unchanged.

[Figure 31]

Edited Image Instruction

Input Image

(Reference GT Image)

[Figure 32]

[Figure 33]

Allow an orange cat to walk past so that it

moves against the lowest row of cards.

Textual Reference of Edited Effect: The bottom layer of the house of cards is disturbed by the motion of an orange cat, resulting in several cards at the base toppled and part of the house beginning to fall.

[Figure 34]

- Figure 3. Multi-scenario data synthesis pipeline. (a) Real-world data synthesis pipeline; (b) Game-world data synthesis pipeline; and (c) Case study of our synthesized data.

Reasoning-based Benchmarks for Image Generation and Editing. In T2I generation, several benchmarks [18, 23, 25, 34] have been developed to assess the reasoning capabilities of models in generating images. For example, WISE [23] focuses on assessing models’ world knowledge, such as cultural and physical understanding, while UniGenBench++ [34] unifies semantic generation evaluation, covering 10 primary dimensions and 27 sub-dimensions, such as logic reasoning, relational understanding, supporting multilingual and varying-length assessments. In image editing evaluation, recent reasoning-based benchmarks like RISEBench [51] aim to examine temporal, spatial, and logical editing capabilities of editing models. Besides, KRISBench [41] introduces a knowledge-grounded taxonomy covering factual, conceptual, and procedural types. However, these benchmarks primarily focus on single-object knowledge and attribute reasoning. We suppose that extending evaluation to multi-object interactions and scenarios governed by human-defined rules is a crucial next step. As for image quality evaluation [16, 45], recent works like UnifiedReward [35, 37] adopt the “VLM-as-a-judge” paradigm, leveraging the powerful capabilities of VLMs to score and provide explanatory judgments. In image editing tasks, evaluation is more challenging because the evaluator needs to assess not only image quality but also understand complex editing instructions and final edited effects. Most studies like RISEBench and KRISBench utilize the property model [13], to rate instruction following, temporal consistency, and image quality. Despite effectiveness, their evaluation relies solely on textual references, which may lead to systematic misjudgments in complex reasoning tasks.

(CoT) reasoning annotations, consisting of detailed, stepby-step reasoning traces generated using VLM, as shown in Fig. 3. To validate its reliability and effectiveness, we finetune the Bagel [6] on this dataset, resulting in UniREditBagel. Experimental results demonstrate that the fine-tuned model achieves substantial improvements on both UniREditBench and other out-of-distribution benchmarks [41, 51]. Additionally, through comprehensive evaluation of both open- and closed-source editing models on our UniREditBench, we reveal their strengths and weaknesses across diverse reasoning-based scenarios.

Contribution: (1) We introduce UniREditBench, a unified benchmark for reasoning-based image editing that covers both real-world and game-world scenarios across 8 primary dimensions and 18 sub-dimensions, augmented with reference GT images to enable robust evaluation; (2) We design a multi-scenario data synthesis pipeline and develop UniREdit-Data-100K, a large-scale synthetic reasoningbased image editing dataset that includes high-quality CoT reasoning annotations. By fine-tuning the Bagel on this dataset, we develop UniREdit-Bagel and achieve substantial improvements, validating the effectiveness and reliability of our dataset; (3) Through comprehensive benchmarking of both open- and closed-source models, we systematically identify their strengths and weaknesses across diverse reasoning-based editing scenarios, offering valuable insights for advancing future models.

To this end, this work proposes UniREditBench, a unified reasoning-based image editing benchmark that spans a broad range of evaluation dimensions across real-world and game-world scenarios with multimodal dual-reference eval-

##### 2. Related Work

Instruction-based Image Editing. Instruction-based image editing models aim to bridge semantic understanding of

[Figure 35]

Real World Scenario

(i)Single object (ii)Multiple objects

###### i.1. Viewpoint Transformation

ii.1. Structural Integrity Change

[Figure 36]

[Figure 37]

Adjust the camera to a lower position, aiming upwards towards the bonsai tree.

[Figure 38]

[Figure 39]

Swing the splitting axe downward so that its blade forcefully strikes the log and penetrate it.

Push the needle into the surface of the balloon and continue until the membrane gives way.

Adjust the camera’s angle to the front profile

ii.2. Motion State Change

###### i.2. Pose Adjustment

Extend the goalkeeper's arm and glove toward the incoming ball until the fingertips reach its surface."

[Figure 40]

[Figure 41]

Swing the cat’s tail toward the toy mouse with enough force to move it from its position.

Lower the person to the ground, and balance on one arm with their body held steady.

[Figure 42]

[Figure 43]

Rotating its body backward until its belly is facing upward and its paws elevated.

###### ii.3. Mechanical Reaction

Activate the pressure washer and guide the water jet steadily over the stone's surface until the entire area has been covered.

[Figure 44]

[Figure 45]

Rotate the handle to drive the spiral fully into the cork until the levers are raised.

###### i.3. Temporal Evolution

Leave the painting undisturbed for an extended period, allowing it to undergo its natural process.

[Figure 46]

[Figure 47]

Cease all maintenance and leave the pool

###### ii.4. Medium Interaction

untouched for an extended period.

[Figure 48]

[Figure 49]

Immerse the heated end of the poker into the water and keep it there until the reaction subsides.

Tip the glass so that its contents pour onto the surface below.

###### i.4. Material Modification

Cover the surface of the taxi with small, brightly colored and securely attached ceramic tiles.

[Figure 50]

[Figure 51]

Substitute the rubber duck with a jade carving of the same shape and size.

###### ii.5. Spatial Arrangement

Remove the bicycle from its wall mount and set it rest against the left wall; shift the lawnmower to the right side.

[Figure 52]

[Figure 53]

Slide both magnets across the surface until their edges are brought together at the center

###### Game World Scenario

[Figure 54]

###### 1. Spatial Intelligence 2. Strategic Reasoning

[Figure 55]

Game: Pacman. Choose exactly ONE direction(UP/DOWN/L EFT/RIGHT) that collects the most beans and move straight until hitting a wall.

[Figure 56]

Game: 3DReconstruction. You are shown a 3D voxel structure. Update BOTH projections so that they match the shown 3D structure.

[Figure 57]

[Figure 58]

Game: Jewel2.

Game: Space Invader. Move the ship to column 5, then shoot once.

Swap the gem at

(0,3) with (1,3).

###### 4. Logic Puzzle Solving

3. Long-Horizon Planning

[Figure 59]

Game: Maze. Using the green color to draw one continuous path from the green start to the red end

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Game: Word Search. Highlight the word 'KAYO' in the grid by coloring its border with purple.

Game: Sokoban. Move the player to push the box at (x=5, y=2) one cell to the Right. Use a shortest path.

Game: Tic-tac-toe. O=red block, X=blue block. O moves first. Play the optimal move.

Game: Sudoku. Solve the Sudoku puzzle.

along walkable white cells only

- Figure 4. Qualitative cases of evaluation dimensions in UniREditBench. We present qualitative examples for each dimension across both real-world and game-world scenarios. uation for more reliable and accurate assessments.

To this end, this work proposes UniREditBench, a unified reasoning-based image editing benchmark that covers a broad spectrum of reasoning dimensions in different scenarios. Compared with previous studies, this benchmark exhibits several key superiorities:

##### 3. UniREditBench

###### 3.1. Overview

- • Broader scenario and reasoning dimension coverage. It contains 2,700 high-quality samples organized into 8 primary reasoning dimensions and 18 sub-dimensions, spanning both real-world and game-world image editing tasks (Sec. 3.2).
- • Reliable dual-reference evaluation. For each sample assessment, we introduce both the textual reference and ground-truth (GT) image reference. This multi-modal reference enables vision-language model (VLM) evaluators to perform direct and fine-grained comparisons at both the textual and visual levels with the generated images, leading to more reliable evaluation (Sec. 3.3).
- • Scalable multi-scenario data synthesis. We propose an automatic data synthesis pipeline with distinct generation

With the rapid advancements in image editing models, existing benchmarks are gradually becoming less adequate to fully capture their comprehensive capabilities, particularly their reasoning-based editing abilities. Specifically, current benchmarks encounter two major challenges: (1) their evaluation primarily focuses on simple single-object attribute edits in real-world scenarios, neglecting complex multiobject interactions, as well as logical or strategic reasoning in game-world scenarios, where explicit human-defined rules govern the outcomes (Tab. 1); (2) their evaluation predominantly rely on clip-based metrics or VLM-based evaluators with text-only references, which may offer insufficient or inaccurate assessments, particularly in complex reasoning-intensive editing scenarios (Fig. 2).

strategies tailored for real-world and game-world scenarios (Sec. 3.4).

###### 3.2. Evaluation Dimensions

In real-life applications, image editing scenarios often involve diverse requirements spanning both real-world and game-world contexts, where complex contextual understanding and implicit reasoning capabilities are crucial for accurate image edits. Therefore, UniREditBench organizes reasoning-based image editing tasks into a scenarioto-category hierarchy framework. As illustrated in Fig. 1, it covers both real-world and game-world scenarios across 8 primary dimensions and 18 sub-categories, each representing a unique visual reasoning challenge with 150 humaninspected examples. We will elaborate on each dimension in the following.

###### 3.2.1. Real-World Scenarios

Real-world scenarios involve editing tasks that reflect the perceptual and interaction dynamics commonly observed in natural environments. These tasks may involve transformations of individual objects or complex interactions among multiple objects. To handle such tasks, models must capture the semantic, physical, and temporal characteristics of objects, as well as their relationships.

- 1. Single-Object Transformation targets variations intrinsic to an individual object, including viewpoint and attribute changes that do not disrupt spatial relationships within the scene:

- • Viewpoint Transformation: Altering the perspective or viewing angle to exhibit alternative views of the same object (e.g., side, top-down, close-up).
- • Pose Adjustment: Modifying the articulation or positioning of an object’s parts, such as limb configurations or postural shifts.
- • Temporal Evolution: Simulating natural progressions over time like aging, decay, or seasonal changes impacting the object’s appearance.
- • Material Modification: Changing inherent surface or material properties (e.g., color, texture) while preserving geometry and location.

- 2. Multi-Object Interaction involves mutual influences and state changes arising from the physical or spatial interactions among multiple objects:

- • Structural Integrity Change: Physical deformations resulting from forces or collisions.
- • Motion State Change: Dynamics induced by contact or force transmission leading to altered movement or posture.
- • Mechanical Reaction: State transitions caused by device operation or functional interactions.
- • Medium Interaction: Changes mediated by substances or environmental factors that affect appearance or state.

(a) Word Cloud of UniREdit-Data-100K (I) Real-World Scenario (II) Game-World Scenario

[Figure 64]

[Figure 65]

(b) Data Distribution of UniREdit-Data-100K (I) Real-World Scenario (II) Game-World Scenario

[Figure 66]

[Figure 67]

6343

5900

6001

5857

5698 5697 5695 5694

5700

5700 5700

5746

5606

5550

5355

5001

4700

4478

Mechanical Reaction

EvolutionMaterialModificationIntegritychangedStructuralMotionState

Reconstruction Jewel2 Maze PacmanSokobanSpaceInvader SudokuTictactoeWordSearch

TransformationViewpoint AdjustmentPose Temporal

Medium InteractionArrangementSpatial

3D

Change

Figure 5. Statistic visualization. We visualize (a) word clouds and (b) data distribition of our UniREdit-Data-100K.

• Spatial Arrangement: Reorganization or repositioning of multiple objects within the scene.

###### 3.2.2. Game-World Scenarios

Game-world scenarios consist of tasks within synthetic environments governed by human-defined rules, evaluating logical, strategic, spatial, and long-horizon reasoning capabilities. These tasks require models to plan, deduce, and act in accordance with the explicit rules that govern the environment.

- • Long-Horizon Planning requires multi-step sequential reasoning to accomplish distant goals, exemplified by navigation or puzzle games such as Maze-solving and Sokoban.
- • Logical Puzzle Solving involves constraint satisfaction and symbolic inference to produce valid solutions under formal rule sets, including Sudoku, Tic-Tac-Toe, and Word Search.
- • Strategic Reasoning requires resource management, adversarial planning over time, modeled after games like Pacman, Jewel2, and Space Invader.
- • Spatial Intelligence focuses on geometric and topological reasoning within 3D environments, such as reconstructing spatial layouts in gaming contexts.

Representative examples are provided in Figs. 1 and 4 to illustrate the scope and diversity of evaluation dimensions, and highlight the complexity and variety of tasks in our benchmark.

###### 3.3. Dual-Reference Evaluation

Evaluating reasoning-based image editing is intrinsically challenging due to the need for the evaluator to accurately understand the implicit reasoning intentions within the instruction. To achieve reliable and comprehensive assessments, we introduce a VLM-based multi-dimensional scor-

Input Image Wan2.5 Qwen-image-Edit Bagel-Think Seedream4.0 GPT-4o Nano Banana UniREdit-Bagel

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Instruction: Move the player to push the box at (x=5, y=2) one cell to the Right.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Instruction: Rotate the handle completely until it cannot turn further in its cycle.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Instruction: Using the red color, draw one continuous path from the green start to the red end along walkable white cells.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Instruction: Direct the stream of hot air from the heat gun onto the side of the plastic bottle and maintain the exposure for seconds.

Figure 6. Qualitative editing result comparison. Our UniREdit-Bagel demonstrates significant superiority in both instruction following and visual quality compared with state-of-the-art closed-sourced and open-sourced models.

ing schema, leveraging both textual and visual evaluation references. Specifically, for each sample, this pipeline evaluates three core dimensions:

- • Instruction Following measures how accurately the generated image reflects the input instruction, focusing on whether explicit effect of the edit is properly manifested. Here, the VLM compares the output image G against both the textual reference of edited effect Rt and the corresponding reference GT image Ri to verify compliance:

SIF = VLM(O,I,G,Ri,Rt) where O represents the original image, I denotes the editing instruction.

- • Visual Consistency assesses the preservation of image regions and attributes unrelated to the edit instruction, ensuring that changes are localized and do not inadvertently alter irrelevant scene elements. This criterion favors models capable of accurate, fine-grained editing rather than wholesale regeneration:

SVC = VLM(O,I,G)

- • Visual Quality evaluates the realism of the generated output, checking for artifacts, distortions, and physical or logical implausibility in the final image:

SVQ = VLM(G) We choose GPT-4.1 [13] as VLM evaluator. Each score S ranges from 1 to 5, following prior detailed scoring guidelines [41, 51]. Finally, the overall evaluation score aggregates these via weighted sum:

SOverall = α1SIF + α2SVC + α3SVQ,

where α1 = 0.5, α2 = 0.3, and α3 = 0.2.

This setting prioritizes instruction following to emphasize the importance of accurately adhering to the instruction’s intent, and also incorporates visual consistency and quality, ensuring that areas unrelated to the instruction are preserved and that the overall image quality is maintained.

###### 3.4. Multi-Scenario Data Synthesis

Given the distinct characteristics of real- and game-world contexts, we develop specialized data generation process for each scenario, as illustrated in Fig. 3. Detailed elaboration of each data synthesis process is provided below.

For Real-World Scenario, we employ a “text-thenimage” data generation strategy. Specifically, (1) this process begins with hand-crafted textual triples that describe the original image, the editing instruction, and the textual reference of edited effect (a reasoning-based narrative of the anticipated outcome). Next, we use the powerful VLM [5], to expand this initial set into a large corpus of text triples. Subsequently, (2) these curated textual triples are input to GPT-4o [13] to synthesize the original and edited images in alignment with the described textual reference of edited effect. (3) Finally, in the quality filtering stage, VLM [5] is used to assess the generated images based on visual fidelity, instruction alignment, and potential hallucination risks. Additionally, it generates reasoning chain-of-thought (CoT) text for each qualified instance, ensuring the production of high-quality, reasoning-based image editing training data.

In Game-World Scenario, game states are inherently well-suited to be represented as structured reasoning-based editing data, where instructions can naturally be solved us-

Table 2. In-domain quantitative comparisons on UniREditBench. GPT-4.1 is used as the evaluator. Best scores are in bold.

###### Real World Scenario Game World Scenario Attribute Structure Physical Property

Model

Spatial Strategic Long-Horizon Logic Puzzle

Avg. Overall

Avg.

Modification Transform Interaction Response Intelligence Reason Plan Solving

###### Closed-source Models

FLUX-Kontext-Pro 47.35 47.16 44.37 41.44 45.00 49.12 48.58 51.16 40.49 46.52 45.77 Seedream4.0 67.98 72.93 65.07 59.49 66.22 38.87 42.16 44.09 51.65 45.38 55.77 Wan2.5 74.66 69.74 63.51 62.87 67.23 63.73 47.13 55.00 54.93 52.67 61.36 Nano Banana 77.10 78.88 71.86 74.70 75.22 66.74 56.11 56.83 64.91 60.39 68.26 GPT-4o 82.67 84.75 79.81 77.40 81.01 77.73 51.44 67.61 63.79 62.07 73.39

Open-source Models

|MagicBrush Omnigen2 Lumina-DiMOO Step1X-Edit Bagel-Think DreamOmni2 UniWorld-V2 Qwen-Image-Edit|43.97 46.09 42.93 46.64 44.69 55.47 58.27 51.44 50.70 53.69 52.84 52.31 50.31 50.83 51.44 59.69 56.36 53.85 54.84 55.93 61.45 59.51 52.65 55.68 56.80 63.52 58.35 52.68 54.02 56.64 72.96 69.37 63.65 61.69 66.55 75.68 73.03 70.59 64.67 70.95<br><br>|63.58 33.59 30.72 35.31 36.85 70.28 27.50 36.25 24.31 33.14 61.23 36.96 39.57 53.09 45.61<br><br>65.68 34.89 47.01 43.89 44.00<br><br>66.29 43.23 43.00 41.30 45.10<br><br><br>72.42 42.17 48.80 48.09 48.98 49.27 40.00 53.41 37.53 43.19 56.73 36.63 48.80 37.68 41.92<br><br>|40.77 43.41 48.54 50.15 50.96 52.81 54.87 56.52<br><br>|
|---|---|---|---|
|UniREdit-Bagel (Ours)<br><br>|76.73 77.80 76.57 71.44 75.74<br><br>|84.90 72.83 84.88 83.72 80.48<br><br>|78.15|

ing Python code. Inspired by Game-RL [27], (1) we first design a diverse collection of game-based problems and develop corresponding Python programs tailored to each category. (2) Then, these programs automatically generate paired original and edited images, along with instructions, textual reference effects, and programmatic CoT reasoning traces. (3) To bridge the gap between programmatic and natural language CoT reasoning formats, we use VLMs to convert these reasoning traces into explanations that align with human inference patterns. Finally, quality filtering are applied to ensure the integrity and reliability of the data.

Overall, this multi-scenario data synthesis pipeline generates UniREditBench, a unified reasoning-based image editing benchmark, and UniREdit-Data-100K, a largescale dataset with high-quality CoT annotations. We detail the elaboration of this dataset in the next section.

##### 4. UniREdit-Data-100K

To enhance the capability of current generative models on reasoning-driven image editing, we propose UniREditData-100K, which contains 100,421 samples spanning 8 reasoning dimensions and 18 categories defined in Sec. 3.2.

###### 4.1. Statistical Analysis

UniREdit-Data-100K is designed with an emphasis on balance and diversity, ensuring that each reasoning category contains over 4,000 instances to effectively support model training across a wide range of editing tasks. It is divided into two primary scenarios: (i) Real-World Scenario, which captures natural object attributes and complex multiobject interactions, and (ii) Game-World Scenario, presenting structured, rule-based editing challenges, such as puz-

zles and strategic planning games. We visualize the word cloud for both real-world and game-world subsets in Fig. 5 (a) and the detailed distribution of samples across different categories in Fig. 5 (b). These visualizations highlight the extensive vocabulary as well as the broad coverage across various categories of our dataset.

###### 4.2. UniREdit-Bagel

To further validate the effectiveness of our dataset, we use it to fine-tune Bagel [6], a unified understanding and generative model. Specifically, each training sample consists of the input image O, an editing instruction I, a stepwise CoT text C that grounds the edit effects step by step, and the target edited image G. During training, the original image and instruction are first input into the model, which then generates a textual reasoning trace and synthesizes the edited image. We supervise both the textual reasoning trace and the visual edit. Formally, for reasoning text supervision, we minimize the negative log-likelihood:

Ltext = −

T

t

log pθ yt | y<t,O,I .

For image generation, we supervise the latent flowmatching loss [20] between the VAE latents of O and G, conditioned on (O,I,C):

Limg = Et∼U(0,1) uθ(zt,t; O,I,C) − u⋆(zt,t) 22,

where uθ is the learned time-conditioned velocity field on the latent path from zO to zG, and u⋆ is the target velocity. Finally, the overall objective is

###### L = λtextLtext + λimgLimg.

Under the influence of Ltext, the model enhances its reasoning ability through explicit CoT learning, which effectively guides the accurate image generation, while Limg improves both the correctness and fidelity of the edited image.

##### 5. Experiment

###### 5.1. Implementation Details

Baselines. We benchmark closed-source models including: GPT-4o [13], Nano Banana [7], Gemini-2.0 [5], Seedream4.0 [24], Wan 2.5 [28], and FLUX-Kontext-Pro [2], as well as open-source models including: Bagel [6], QwenImage-Edit [39], Step1X-Edit [21], FLUX.1-Kontextdev [2], Emu2 [26], Omnigen2 [40], Omnigen [42], HiDream-Edit [4], MagicBrush [49], and AnyEdit [47].

Training and Evaluation. We train all Bagel [6] components, except the VAE, for 5,000 iterations on UniREditData-100K using the Adam optimizer and a cosine learningrate schedule with 500 warm-up steps, a peak learning rate of 2×10−5, and a minimum learning rate of 10−6. The loss weights are set to λtext = 2 and λimg = 1. During inference, we use the official inference settings provided by Bagel. To ensure fair comparisons with other baselines, we adopt the original inference configurations of these models.

###### 5.2. Benchmarking Results on UniREditBench

As shown in Tab. 2, among closed-source models, GPT-4o achieves the highest average performance across all scenarios, with Nano Banana performing comparably. Wan2.5 delivers balanced results on real-world tasks but lags on game scenarios that require strategic reasoning. Besides, Seedream4.0 is competitive on structure transform yet encounters challenges in game scenarios. Among opensource baselines, Qwen-Image-Edit performs strongly on real-world tasks such as attribute modification and structure transform. However, most models remain comparatively weak on game scenarios like Strategic Reasoning. Overall, compared with open-source methods, closed-source models, particularly GPT-4o, maintain a clear advantage. While some open-source models are competitive on specific realworld tasks, they generally struggle with complex reasoning in game scenarios. Notably, only GPT-4o and Nano Banana achieve an average score greater than 60 on game scenarios, underscoring that this setting remains highly challenging and serves as a useful test for current models.

###### 5.3. Comparison Results of UniREdit-Bagel

Quantitative. UniREdit-Bagel achieves the best overall performance among all closed- and open-source models on UniREditBench, surpassing the second-place GPT-4o by a substantial margin. The largest gains occur in game-world scenarios (+17.08), indicating exceptional capability of understanding and processing complex reasoning image edit-

Table 3. Out-of-distribution quantitative performance comparison on RISEBench [51]. GPT-4.1 is used as the evaluator. Best scores are in bold.

Models Temporal Causal Spatial Logical Overall Closed-source Models

Gemini-2.0-Flash-pre 10.6% 13.3% 11% 2.3% 9.4% Seedream-4.0 12.9% 12.2% 11.0% 7.1% 10.8% Gemini-2.0-Flash-exp 8.2% 15.5% 23.0% 4.7% 13.3% GPT-4o 34.1% 32.2% 37.0% 10.6% 28.9% Nano Banana 25.9% 47.8% 37.0% 18.8% 32.8%

Open-source Models

|HiDream-Edit OmniGen Step1X-Edit Bagel FLUX.1-Kontext-Dev Qwen-Image-Edit Bagel-Think|0.0% 0.0% 0.0% 0.0%<br><br>1.2% 1.0% 0.0% 1.2%<br><br>0.0% 2.2% 2% 3.5%<br><br>3.5% 4.4% 9.0% 5.9%<br><br>2.3% 5.5% 13.0% 1.2%<br><br>4.7% 10.0% 17.0% 2.4%<br><br><br><br><br>4.7% 15.5% 14.0% 1.2%<br><br>|0.0%<br><br>0.8%<br>1.9% 5.8% 5.8%<br><br><br>8.9%<br>9.2%<br>|
|---|---|---|
|UniREdit-Bagel (Ours)<br><br>|22.4% 18.9% 21.0% 10.6%|18.3%|

ing tasks. In out-of-distribution performance comparison, UniREdit-Bagel achieves the strongest open-source results across all four categories on RISEBench, shown in Tab. 3, improving upon the Bagel-Think baseline by 9.1 points and surpassing the closed-source Gemini-2.0-Flash-exp by 5.0 points. It also remains competitive with top closed-source models like Nano Banana and GPT-4o, narrowing the gap between open- and closed-source models.

Qualitative. The qualitative results presented in Fig. 6 highlight the strengths of our UniREdit-Bagel across various tasks. Specifically, in Fig. 6 (Row 4), most models fail to reliably reproduce the physical heat effect. Although several baselines, such as Nano Banana and QwenImage-Edit, successfully capture the heat-induced warping of a plastic bottle under sustained heat gun exposure, they fail to preserve the heat trace. Notably, UniREdit-Bagel not only renders the deformation accurately but also preserves the heat trace, offering superior visual consistency. Besides, in the Sokoban and Maze game settings (rows 1 and 3), Seedream-4.0, Nano Banana, and Wan-2.5 generally preserve instruction-irrelevant content but struggle with instruction-specific objectives. In contrast, UniREdit-Bagel excels in both fulfilling the instruction and maintaining the coherence of unrelated content.

##### 6. Conclusion

This paper presents UniREditBench, a unified reasoningbased benchmark for image editing with broad dimension coverage and a robust dual-reference evaluation protocol. We further introduce a multi-scenario synthesis pipeline and release UniREdit-Data-100K, a large-scale dataset with high-quality CoT annotations. To demonstrate its effectiveness, we fine-tune Bagel on this dataset and yields UniREdit-Bagel, which achieves substantial quantitative and qualitative gains. Comprehensive benchmarking of open-source and closed-source image editing models reveals their strengths and weaknesses across various aspects.

##### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In CVPR, pages 18208–18218, 2022. 3
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 1, 8
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 18392–18402, 2023. 1, 3
- [4] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 8
- [5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6, 8
- [6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3, 7, 8
- [7] Google. Introducing gemini 2.5 flash image, our stateof-the-art image model. https://developers. googleblog.com/en/introducing-gemini-25-flash-image/, 2025. 1, 8
- [8] Feng Han, Yang Jiao, Shaoxiang Chen, Junhao Xu, Jingjing Chen, and Yu-Gang Jiang. Controlthinker: Unveiling latent semantics for controllable image generation through visual reasoning. arXiv preprint arXiv:2506.03596, 2025. 3
- [9] Qingdong He, Xueqin Chen, Chaoyi Wang, Yanjie Pan, Xiaobin Hu, Zhenye Gan, Yabiao Wang, Chengjie Wang, Xiangtai Li, and Jiangning Zhang. Reasoning to edit: Hypothetical instruction-based image editing with visual reasoning. arXiv preprint arXiv:2507.01908, 2025. 2
- [10] Runze He, Kai Ma, Linjiang Huang, Shaofei Huang, Jialin Gao, Xiaoming Wei, Jiao Dai, Jizhong Han, and Si Liu. Freeedit: Mask-free reference-based image editing with multi-modal instruction. arXiv preprint arXiv:2409.18071,

2024. 3

- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [12] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, pages 8362–8371, 2024. 2

- [13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 3, 6, 8
- [14] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 6007–6017, 2023. 3
- [15] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In CVPR, pages 2426–2435, 2022. 3
- [16] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. NeurIPS, 36:36652–36663, 2023. 3
- [17] Chenglin Li, Qianglong Chen, Zhi Li, Feng Tao, and Yin Zhang. Vcbench: A controllable benchmark for symbolic and abstract challenges in video cognition. arXiv preprint arXiv:2411.09105, 2024. 2
- [18] Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Xiaojuan Qi, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025. 3
- [19] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 1
- [20] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 7
- [21] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 2, 8
- [22] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3
- [23] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 3
- [24] ByteDance Seed. Seedream 4.0. https://seed. bytedance.com/en/seedream4_0, 2025. 8
- [25] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025. 3
- [26] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In CVPR, pages 14398–14409, 2024. 8

- [27] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, et al. Code2logic: Game-code-driven data synthesis for enhancing vlms general reasoning. arXiv preprint arXiv:2505.13886, 2025. 2, 7
- [28] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 8
- [29] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3
- [30] Yibin Wang, Changhai Zhou, and Honghui Xu. Enhancing object coherence in layout-to-image synthesis. arXiv preprint arXiv:2311.10522, 2023. 3
- [31] Yibin Wang, Zhiyu Tan, Junyan Wang, Xiaomeng Yang, Cheng Jin, and Hao Li. Lift: Leveraging human feedback for text-to-video model alignment. arXiv preprint arXiv:2412.04814, 2024. 1
- [32] Yibin Wang, Weizhong Zhang, and Cheng Jin. Magicface: Training-free universal-style human image customized synthesis. arXiv preprint arXiv:2408.07433, 2024. 3
- [33] Yibin Wang, Weizhong Zhang, Jianwei Zheng, and Cheng Jin. Primecomposer: Faster progressively combined diffusion for image composition with attention steering. In ACM MM, pages 10824–10832, 2024. 1, 3
- [34] Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, et al. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation. arXiv preprint arXiv:2510.18701, 2025. 3
- [35] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement finetuning. arXiv preprint arXiv:2505.03318, 2025. 3
- [36] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025. 1
- [37] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. 3
- [38] Yibin Wang, Weizhong Zhang, Honghui Xu, and Cheng Jin. Dreamtext: High fidelity scene text synthesis. In CVPR, pages 28555–28563, 2025. 1, 3
- [39] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 8
- [40] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 3, 8

- [41] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, MingHsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv:2505.16707, 2025. 2, 3, 6
- [42] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In CVPR, pages 13294–13304, 2025. 8
- [43] Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint arXiv:2505.13031, 2025. 1
- [44] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025. 1, 3
- [45] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. NeurIPS, 36:15903–15935, 2023. 3
- [46] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 2
- [47] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In CVPR, pages 26125–26135, 2025. 1, 8
- [48] Dong Zhang, Lingfeng He, Rui Yan, Fei Shen, and Jinhui Tang. R-genie: Reasoning-guided generative image editing. arXiv preprint arXiv:2505.17768, 2025. 2
- [49] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. NeurIPS, 36:31428–31449, 2023. 1, 3, 8
- [50] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In CVPR, pages 9026–9036,

2024. 3

- [51] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 2, 3, 6, 8

#### UniREditBench: A Unified Reasoning-based Image Editing Benchmark Supplementary Material

##### A. Data Filtering

We design a comprehensive, multi-stage pipeline that performs data filtering, i.e., instruction de-duplication, quality filtering, and human inspection, to remove redundancy and low-quality data. Detailed elaboration of the filtering pipeline is provided below.

###### A.1. Instruction De-duplication

During the first stage of the real-world scenario, text prompt for image editing are sampled from the Gemini-2.5-Pro, which may potentially introduce repeated or near-duplicate entries. We remove redundancy along two aspects: exact matches and semantic similarity.

- • Exact-Match Deduplication: We first normalize the original image description by converting it to lowercase and removing punctuation. Afterward, we extract the set of words from the normalized text. If two samples contain identical word sets, they are considered duplicates, as the descriptions are effectively the same. These duplicate samples are then filtered out to ensure data diversity.
- • Semantic-Similarity Deduplication: We use a sentencetransformers model to extract sentence embeddings for both the original image description and edit instruction. We then compute the pairwise similarity between these embeddings. If the similarity score exceeds a threshold of 0.7 for either description and instruction, the samples are deemed semantically redundant and are filtered out to enhance dataset diversity.

These complementary exact-match and semantic filters improve dataset diversity by eliminating both literal and paraphrastic duplicates.

###### A.2. Quality Filtering

To ensure the quality of both the generated text and images, we evaluate and filter them across six key dimensions: text hallucination, instruction adherence, content preservation, visual quality, image hallucination, and CoT quality. Scores for each dimension are assigned by the Gemini-2.5-Pro on a 1–5 scale. Only samples that achieve the maximum score across all dimensions are retained.

- • Text Hallucination: We evaluate the textual reference for hallucinated content, defined as entities or visual effects that are not mentioned in the Instruction or that cannot be plausibly induced by the given Instruction.
- • Instruction Following: We compare the edited image with the textual reference to assess whether the generated visual changes accurately reflect the specified ef-

(a) Closed-source Model Comparison (b) Open-source Model Comparison

Physical Interaction

Physical Interaction

[Figure 100]

[Figure 101]

GPT-4o

Ours

Structure Transform

Property Response

Structure Transform

Property Response

Omnigen2

Seedream4.0

Qwen-Image-Edit

Spatial Intelligence

Attribute Modification

Spatial Intelligence

Attribute Modification

Wan2.5

MagicBrush

Nano Banana

Bagel-Think

Strategic Reason.

Logic-Puzzle Solving

Strategic Reason.

Logic-Puzzle Solving

Flux-Kontext-Pro

Step-1X-Edit

Long-Horizon Plan

Long-Horizon Plan

Figure 7. Benchmarking result visualization. (a) Closed-source model comparison; (b) Open-source model comparison.

Table 4. Quantitative comparisons on KRISBench. GPT-4.1 is used as the evaluator. Best scores are in bold while second-best is underlined.

Attribute Spatial Temporal Social Natural Logical Instruction Overall

Model

Perception Perception Prediction Science Science Reasoning Decompose Score

###### Closed-source Models

Doubao 70.92 59.17 40.58 65.50 61.19 47.75 60.58 60.70 Step 3ϕ vision 69.67 61.08 63.25 66.88 60.88 49.06 54.92 61.43 Gemini-2.0 66.33 63.33 63.92 68.19 56.94 54.13 71.67 62.41 GPT-4o 83.17 79.08 68.25 85.50 80.06 71.56 85.08 80.09

Open-source Models

|MagicBrush AnyEdit Emu2 Step1X-Edit Bagel-Think<br><br>|53.92 39.58 - 42.94 38.06 30.00 23.08 47.67 45.17 - 38.56 42.94 36.56 26.92 51.50 48.83 22.17 34.69 38.44 24.81 45.00 55.50 51.75 - 44.69 49.06 40.88 22.75 69.27 67.58 - 65.00 62.11 47.33 49.22<br><br>|37.15 38.55 39.70 43.29 60.77|
|---|---|---|
|UniREdit-Bagel (Ours)|71.75 71.00 - 69.20 65.99 59.91 51.55|65.45|

fects. Samples that demonstrate poor adherence to the instructions and text reference are discarded.

- • Content Preservation: We assess whether regions unrelated to the edit instruction, such as the background, remain consistent between the original and edited images, ensuring stability in unaffected areas.
- • Visual Quality: We assess whether the generated images meet fundamental quality standards, specifically by ensuring they are free from artifacts or degradation.
- • Image Hallucination: We examine the edited images for any unintended additions or alterations beyond the specified textual reference, such as the appearance of additional objects.
- • CoT Quality: We evaluate the correctness of the chainof-thought (CoT) reasoning text, focusing on whether the analysis of the original image and instruction is logical and sound.

###### A.3. Human Inspection

In addition to our automated filtering pipeline, we perform a final manual check of each data instance. To facilitate this, we developed two web-based interfaces and enlisted eight expert annotators to carry out two-stage filtering process:

• Initial Filtering: Annotators remove samples with extremely erroneous textual references or substandard generated images.

###### Table 5. Detailed in-domain quantitative comparisons on UniREditBench. GPT-4.1 is used as the evaluator. Best scores are in bold.

###### Real World Scenario Game World Scenario

Attribute Modification

Structure Transform

Physical Interaction

Property Response

Spatial Intelligence

Strategic Reason

Logic Puzzle Solving

Long-Horizon Plan

Model

Overall

Motion State Change

Structural Integrity Change

Spatial Arrangement

Mechanical Reaction

Medium Interaction

Viewpoint Transformation

Material Modification

Pose Adjustment

Temporal Evolution

3D Reconstruction

Space Invader

Word Search

Jewel2 Pacman

Tictactoe Sudoku Maze Sokoban

###### Closed-source Models

FLUX-Kontext-Pro 37.55 57.15 56.79 37.52 41.46 44.62 47.03 40.56 42.32 49.12 53.18 38.12 54.43 32.45 61.23 27.80 48.45 53.87 45.77 Seedream4.0 64.18 71.78 79.98 65.87 56.10 61.60 77.52 59.15 59.83 38.87 45.47 40.87 40.13 48.05 38.17 68.72 55.27 32.90 55.77 Wan2.5 73.78 75.54 79.95 59.53 62.67 64.38 63.49 60.83 64.91 63.73 43.44 56.57 41.38 58.38 58.57 47.86 64.03 45.97 61.36 Nano Banana 75.82 78.39 86.12 71.63 71.37 71.07 73.14 77.58 71.82 66.74 59.97 54.33 54.04 64.13 39.80 90.81 62.50 51.15 68.26 GPT-4o 83.83 81.52 92.18 77.33 75.86 74.98 88.60 78.98 75.81 77.73 58.96 45.60 49.75 64.62 58.97 67.78 83.73 51.49 73.39

###### Open-source Models

|MagicBrush Omnigen2 Lumina-DiMOO Step-1X-Edit Bagel-Think DreamOmni2 UniWorld-V2 Qwen-Image-Edit|36.74 51.20 47.75 44.43 44.05 47.06 37.67 45.87 47.42<br><br>51.25 59.68 68.25 48.28 48.37 54.48 51.48 50.77 50.64<br><br>54.22 51.47 54.40 50.23 48.67 52.62 49.65 51.63 50.03<br><br>52.57 66.82 62.15 50.58 53.07 58.68 49.80 54.23 55.45<br><br><br>59.05 63.84 63.52 55.49 54.05 55.62 48.27 55.03 56.33 61.22 65.82 67.07 49.63 49.58 54.83 53.62 54.11 53.92 71.45 74.47 83.19 55.55 58.53 65.47 66.95 62.20 61.17 74.60 76.76 81.45 64.62 66.48 68.78 76.50 64.65 64.69<br><br>|63.58 45.33 25.60 29.83 33.92 31.45 40.55 33.20 28.23 70.28 51.33 1.38 29.80 22.87 44.97 5.10 39.57 32.93 61.23 48.23 36.29 26.35 41.73 65.07 52.47 43.57 35.57<br><br>65.68 35.05 36.38 33.25 45.49 46.97 39.20 55.94 38.08<br><br>66.29 45.48 42.89 41.33 48.42 42.20 33.27 48.62 37.38<br><br><br>72.42 38.78 42.33 45.40 45.86 46.98 51.42 56.55 41.05 49.27 34.44 42.90 42.65 49.67 34.73 28.20 71.59 35.22 56.73 38.42 41.02 30.45 48.52 33.60 30.93 61.38 36.21<br><br>|40.77 43.41 48.54 50.15 50.96 52.81 54.87 56.52|
|---|---|---|---|
|UniREdit-Bagel (Ours)<br><br>|81.08 72.39 84.78 70.81 73.85 72.73 83.13 71.27 71.62<br><br>|84.90 87.77 57.68 73.05 86.80 71.16 93.20 98.30 71.47|78.15|

Effect of Training Set Size on Overall Score

- 73

- 74

- 75

- 76

- 77

- 78

Overall

20 30 40 50 60 70 80 90 100 Training set size (k examples)

Figure 8. Overall performance on UniREditBench of models trained on different size of UniREdit-Data.

##### E. Effect of Training Set Size on Overall Score

We provide overall performance comparison on UniREditBench of models trained on difference size of UniREditData, which is visualized in Fig. 8.

##### F. Ethical statement

In this work, we affirm our commitment to ethical research practices and responsible innovation. To the best of our knowledge, this study does not involve any data, methodologies, or applications that raise ethical concerns. All experiments and analyses were conducted in compliance with established ethical guidelines, ensuring the integrity and transparency of our research process.

• Manual Correction: Annotators make refinements to the textual reference effect that are only slightly incorrect, ensuring alignment and accuracy.

Two web interfaces are shown in Figs. 12 and 13.

##### B. Detailed Benchmarking Results

We provide detailed benchmarking results on our UniREditBench for each category in Tab. 5.

##### C. More Quantitative Results

We provide more quantitative out-of-distribution performance comparisons on KRISBench in Tab. 4.

##### D. More Qualitative Comparison Results

We provide additional qualitative comparisons on UniREditBench in Fig. 9 and 10, and comparisons on RISEBench in Fig. 11.

Input Image Wan2.5 Qwen-image-Edit Bagel-Think Seedream4.0 GPT-4o Nano Banana UniREdit-Bagel

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Instruction: Launch a cannonball towards the tower at high velocity.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Instruction: Direct the vehicle so that it moves forward and makes forceful contact with the side barrier.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Instruction: Swing the cleated shoe forward and make contact with the center of the soccer ball.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Instruction: Gradually bring the handles of the nutcracker together while maintaining steady pressure on the almond.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Instruction: Tilt the cream pitcher and allow a small amount of cream to flow into the coffee, then set the pitcher down without agitating the mug.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Instruction: Move the mug to the spot where the plate is, and place the plate where the mug was.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Instruction: Erase the blue cloud and redraw it on the opposite side of the yellow sun.

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Instruction: Submerge the tip of the feather into the ink container and hold it there until the feather absorbs the ink.

Input Image Wan2.5 Qwen-image-Edit Bagel-Think Seedream4.0 GPT-4o Nano Banana UniREdit-Bagel

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Instruction: Swap the gem at (3,2) with (4,2).

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Instruction: Push the box onto a target using a shortest sequence. The final move to place the box on the target will be a push Up.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Instruction: Using the blue color, draw one continuous path from the green start to the red end along walkable white cells only

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Instruction: Do not move Pac-Man. Draw in blue the straight line (to the wall) from the current cell that collects the most beans.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Instruction: Move the ship to the column with the most enemies and shoot repeatedly until that column is empty.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Instruction: Solve the Sudoku puzzle.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Instruction: In row 7, highlight every 'S' by coloring its cell border with orange.

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Instruction: You are shown a 3D voxel structure AFTER removing exactly ONE voxel, but the two projections (Y-Z and X-Z) are still the ones BEFORE the removal. Update BOTH projections so that they match the shown 3D structure.

Input Image Seedream4.0 Qwen-Image-Edit Bagel-Think GPT-Image-1 Nano Banana UniREdit-Bagel

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Instruction: Draw what it looks like three seconds after being punctured.

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Instruction: Draw the shape after rotating it 180 degrees counterclockwise around the center point.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Instruction: Draw what they will look like one year later.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Instruction: Draw an image of a fully assembled lamp using the provided components.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Instruction: This picture is observed from the Northern Hemisphere. Draw what it will look like 7 days later.

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Instruction: Draw what it will look like after being rubbed against a rough surface.

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Instruction: Draw what it will look like after 30 minutes.

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Instruction: Draw an image showing a Christmas tree built by stacking three triangles from largest to smallest from bottom to top, and adding a rectangle at the bottom as the trunk.

Figure 11. Qualitative editing result comparison on RISEBench. Our UniREdit-Bagel demonstrates significant superiority in both instruction following and visual quality compared with state-of-the-art closed-sourced and open-sourced models.

## Initial Filtering

[Figure 286]

- Figure 12. Web interface of the initial filtering stage.

# Manual Correction

[Figure 287]

- Figure 13. Web interface of the manual correction stage.

