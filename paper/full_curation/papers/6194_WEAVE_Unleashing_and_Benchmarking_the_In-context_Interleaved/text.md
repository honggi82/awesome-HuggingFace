# arXiv:2511.11434v1[cs.CV]14Nov2025

[Figure 1]

## WEAVE: Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation

Wei Chow1* Jiachun Pan1* Yongyuan Liang3 Mingze Zhou4 Liyu Jia2 Saining Zhang2 Xue Song2 Siliang Tang4 Juncheng Li4 Fengda Zhang2† Weijia Wu1† Hanwang Zhang2 Tat-Seng Chua1

1National University of Singapore, 2Nanyang Technological University, 3University of Maryland, College Park, 4Zhejiang University

https://weichow23.github.io/weave

#### Abstract

Recent advances in unified multimodal models (UMMs) have enabled impressive progress in visual comprehension and generation. However, existing datasets and benchmarks focus primarily on single-turn interactions, failing to capture the multi-turn, context-dependent nature of realworld image creation and editing. To address this gap, we present WEAVE, the first suite for in-context interleaved cross-modality comprehension and generation. Our suite consists of two complementary parts. WEAVE-100k is a large-scale dataset of 100K interleaved samples spanning over 370K dialogue turns and 500K images, covering comprehension, editing, and generation tasks that require reasoning over historical context. WEAVEBench is a human-annotated benchmark with 100 tasks based on 480 images, featuring a hybrid VLM judger evaluation framework based on both the reference image and the combination of the original image with editing instructions that assesses models’ abilities in multi-turn generation, visual memory, and world-knowledge reasoning across diverse domains. Experiments demonstrate that training on WEAVE-100k enables vision comprehension, image editing, and comprehension-generation collaboration capabilities. Furthermore, it facilitates UMMs to develop emergent visual-memory capabilities, while extensive evaluations on WEAVEBench expose the persistent limitations and challenges of current approaches in multi-turn, context-aware image generation and editing. We believe WEAVE provides a view and foundation for studying in-context interleaved comprehension and generation for multi-modal community.

*Equal Contribution. †Corresponding Author.

#### 1. Introduction

Recent advances in unified multimodal models (UMMs) [22, 52, 81, 94] have significantly reshaped the landscape of visual understanding and generation. This unified formulation enables models to describe and edit visual content through language, integrate visual references, and perform iterative editing across multiple images. Recent works have shown its remarkable potential for image editing [49, 63, 69], and multi-image composition [62, 79].

However, real-world image creation is rarely a one-shot process. Human creators typically engage in reversible refinement, reusing or reverting to previous results as needed. Moreover, creating a comic or visual story inherently involves multiple rounds of progressive refinement to maintain visual consistency, where each frame must remain coherent with previous scenes in terms of character appearance, lighting, and narrative flow [31, 33].

While some closed-source models [21, 62] have recently demonstrated promising capabilities in multi-turn reasoning and editing, such as maintaining visual memory and context coherence, most open-source models [9, 37, 86] remain confined to single-turn editing. This gap stems from the absence of high-quality interleaved datasets capturing the temporal dependencies and iterative workflows of real-world multi-turn editing. Existing datasets [72, 84, 85, 90] are fundamentally single-turn, treating each edit as an isolated instruction and thus failing to represent the long-horizon reasoning required for authentic interactive image creation, as illustrated in Figure 1(a). This lack has hindered systematic exploration, and benchmarks for evaluating multi-turn editing with historical context remain absent.

To address this gap, we present WEAVE, the first suite for in-context interleaved cross-modality comprehension and generation. Our suite consists of two complementary parts. WEAVE-100k is a large-scale dataset containing 100K interleaved samples, spanning over 370K dialogue

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Let it be angry and hiss

Change the image style of the gin and tonic into neon art

Replace the lime with orange and remove the bowl of salted nuts

Let the cat have blue eyes

Add back the small bowl of salted nuts

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Replace the vase with a taller white ceramic one.

Add the flowers back into the new vase

Remove the flowers above the rim of the vase

Add a barn in the background

Add a bale of hay in field

(b) Ours

(a) Previous Works

Figure 1. Comparisons among and existing datasets. (a) Previous Works: Simple overlay of single-turn edits. (b) Ours: Multi-turn edits involving in-context visual memory recall.

turns and 500K images across comprehension, editing, and generation tasks that require reasoning over historical context. As shown in Figure 1, effective multi-turn editing tasks demand strong visual memory to retrieve and reuse objects, layouts, and styles from previous rounds, for instance by removing an item in one turn and accurately restoring it later. This interleaved design captures the iterative nature of realistic multi-turn image editing, in which each modification can rely on information from prior rounds.

WEAVEBench is a human-annotated benchmark of 100 tasks with 480 images, featuring a hybrid VLM judge evaluation framework with four metrics that evaluate alignment with reference images and fidelity to original images and correctness for editing instructions. The benchmark assesses models’ capabilities in multi-turn generation, visual memory, and world-knowledge reasoning across diverse domains, including science, creation, logic, and games. WEAVEBench reveals that current models struggle with in-context interleaved generation and exhibit performance degradation as content length increases, indicating substantial room for improvement.

Experiments demonstrate that training on WEAVE100k yields substantial improvements in vision comprehension (9.8% on MMMU), image editing (4.8% on GEditBench), and comprehension-generation collaboration (approximately 50% on RISE). Moreover, training facilitates the emergence of visual memory capabilities in UMMs, while evaluations on WEAVEBench reveal persistent limitations in multi-turn, context-aware image generation.

To summarize, our contributions are threefold:

- • We introduce WEAVE-100k, the first large-scale dataset for multi-turn, context-aware image understanding and generation, comprising over 100K samples, 370K dialogue turns, and 500K images across comprehension, editing, and generation tasks.
- • We present WEAVEBench, the first human-annotated benchmark for interleaved multimodal comprehension and generation, featuring 100 carefully designed cases

with 480 images and a hybrid VLM judge evaluation framework that assesses multi-turn generation, visual memory, and world-knowledge reasoning.

• Through extensive experiments, we demonstrate that training on WEAVE-100k significantly improves performance on established benchmarks and facilitates the emergence of visual memory capabilities, while evaluation on WEAVEBench reveals persistent limitations in multi-turn, context-aware generation.

#### 2. Related Works

Unified Multimodal Models represent a paradigm designed to seamlessly integrate multimodal comprehension and generation capabilities within a single framework. To achieve this unified objective, seminal works [16, 41, 55, 75] leverage image tokenization and autoregressive nexttoken prediction to generate visual tokens. Subsequent developments, driven by the pursuit of enhanced image synthesis quality, incorporate diffusion-based or flow-matching heads [47] integrated with shared transformer architectures [22, 52, 58, 94]. Recent works have demonstrated remarkable potential for instruction-based image editing [49, 63, 69] and multi-image composition [62, 79]. However, the capabilities of UMMs for in-context interleaved comprehension and generation remain largely unexplored.

Image Editing. Recent text-guided image editing has achieved substantial progress [5, 49, 85]. For instance, AnyEdit [86] provides general-purpose editing datasets that unify diverse edit types, such as insertion, replacement, and style modification. GPT-Image-Edit-1.5M [72] and Echo4o [84] expand data scale and instruction diversity by leveraging GPT-4o. However, these approaches remain limited to one-shot edits without historical context or iterative refinement. While MagicBrush [90] introduces multi-turn editing, each instruction is treated as an independent request without multi-turn dependencies. As shown in Table 1, WEAVE-100k introduces the first in-context interleaved

- Table 1. Summary of Multimodal Reasoning Benchmarks. We compare existing works from aspects including: 1interleave, 2multiturn, 3vision memory, 4multidimensional evaluation, 5hybrid evaluation, and 6whether manual annotations and filtering are applied.

means text to image, image edit and image comprehension.

Multi- Vision Multi- Hybrid

Benchmark Venue Inter.

# Domain # Num #Types Turn Mem. Dim. Eval

ReasonPix2Pix [40] arXiv’24 ✗ ✗ ✗ ✗ ✗ 40,212 1 ReasonEdit [36] CVPR’24 ✗ ✗ ✗ ✗ ✗ 219 1 Reason50K [30] arXiv’25 ✗ ✗ ✗ ✗ ✗ 51,039 4

Zebra-CoT [45] arXiv’25 ✓ ✓ ✗ ✗ ✗ 182,384 4 KRIS-Bench [78] NeurIPS’25 ✗ ✗ ✗ ✓ ✓ 1,267 7

RISEBench [93] NeurIPS’25 ✗ ✗ ✗ ✓ ✓ 360 4 CoMM [14] CVPR’25 ✓ ✗ ✗ ✗ ✗ 227,000 1

IRG-300k [34] arXiv’25 ✓ ✗ ✗ ✗ ✗ 1 1 Echo-4o [84] arXiv’25 ✗ ✗ ✗ ✗ ✗ 179,000 3 ROVER [46] arXiv’25 ✓ ✗ ✗ ✓ ✓ 1,312 23

WEAVE Ours ✓ ✓ ✓ ✓ ✓ 100,100 16

cross-modal dataset that explicitly captures multi-turn editing and context-dependent generation, enabling models to learn visual memory and consistent reasoning. More dicussion for the related works can be found in Appendix C.

#### 3. WEAVE

To assess in-context interleaved comprehension and generation, we first introduce the data collection pipelines WEAVE-100k and WEAVEBench in Section 3.1. We then detail the evaluation settings and metrics in Section 3.2, and present key statistics for WEAVE in Section 3.3.

###### 3.1. Data Collection

WEAVE-100k In order to generate rich and diverse data with visual memory capabilities, we constructed a data pipeline as illustrated in Figure 3. This pipeline incorporates four distinct generation pathways followed by multiple filtering and refinement stages to ensure accuracy and quality of the produced data. To generate multi-turn editing data with visual memory capabilities, we implemented four methodological approaches: (i) Multi-image fusion: We achieved reference to previous iterations by fusing edited or directly generated images. (ii) Remove-then-back: We employed a technique of first removing or replacing objects, then adding them back, enabling the system to recall previously deleted visual elements. (iii) Derivative imagination and comparison: We incorporated methods for deriving or imagining alternative solutions or new images before fusion. (iv) Sequential procedures: We implemented sequential edits following narrative progressions or structured editing operations. Further details regarding the data collection methodology are presented in Appendix A.1.

WEAVEBench is annotated by individuals with graduatelevel STEM degrees. It comprises 100 items across 16 task

categories, incorporating both multi-turn editing tasks requiring visual memory and challenges demanding world knowledge (cultural contexts, physical phenomena, and chemical processes). As illustrated in Figure 2, tasks included generating examples involving the Tokyo Tower and demonstrating comprehension of traffic signal reactions. The images used include web-sourced content and synthetically generated images from three models: Seedream 4.0 [62], Nano Banana [20] and SeedEdit 3.0 [69].

###### 3.2. Evaluation Settings and Metrics

We adopt the VLM-as-judge [49] automated evaluation framework, with detailed templates provided in Appendix B.1. To enable focused assessment, we employ a key-point-based scoring approach using structured evaluation criteria. Specifically, we leverage a hybrid strategy that instructs the VLM to evaluate based on both the reference image and the combination of the original image with editing instructions. As shown in Figure 5, the judge invokes different images as references and assigns scores according to predefined key points.

Our evaluation comprises 4 metrics (the first three apply to editing tasks; the last applies to comprehension tasks): Key Point Correctness (KP): Measures whether the edited image satisfies the specified editing requirements. Visual Consistency (VC): Ensures non-target elements remain unchanged, maintains consistency with the original image (unedited regions remain intact when the scene is preserved; edited regions maintain stylistic coherence when the scene is modified), and assesses identity preservation of edited objects. Image Quality (IQ): Evaluates the overall quality of the generated image. Accuracy (Acc): Measures the correctness of the reasoning result. Details regarding score calculation methodology can be found in Appendix B.3.

[Figure 19]

Optics

Geography

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Draw a landscape Add a castle at the highest point Edit the terrain map

The fringes of thin-film interference change over time

[Figure 28]

[Figure 29]

[Figure 30]

Physics Biology

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Natural Science

[Figure 38]

###### Astronomy Chemistry

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Changes in lunar phases over time Traffic light response

Story

[Figure 48]

The story of Little Red Riding Hood unfolds scene by scene, requiring the characters' traits to remain consistent.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

###### Recall

Edit 2

- Edit 1: Change the background in #1 to a plain black studio backdrop and remove the light.
- Edit 2: Replace shoes in Image #2 with white sneakers; add the light from #1.
- Edit 3: Change the image style in Image

- #3 into glitch art.

Edit 4: Combine glitch-style figure from

- #4 with the studio background from #1.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Edit 1

Edit 3 Edit 4

Creation Edit

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Edit 2: Generate the most famous tower in the women's country's capital shown in #1.

Edit 1: Change the background to the most famous mountain in the country of the person.

Edit 3: Place the woman in #1 in front of the tower of #4.

[Figure 66]

###### Fusion

[Figure 67]

[Figure 68]

- Edit 1: Person in #1 is wearing the flowers in #2 and the clothes in #3.
- Edit 2: Person in #4 is holding the bag in #6 with the background from #5.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Edit 1 Edit 2

Spatial Mathematics

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Logic

Generate a frontal view of the space capsule

Iterative conditional solution of mathematical problems

Chess Game

[Figure 83]

Maze

[Figure 84]

- Edit 1: have Black's rook capture the pawn on d7.
- Edit 2: move Black's queen to c7.
- Edit 3: advance Black's apawn to a6. Question: What is the best move for the three?

Edit 3

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Drawing and comparing routes in maze

Edit 1 Edit 2

Answer: #3 shows the best move.

Visual Jigsaw

[Figure 93]

Game

Minecraft

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

- Edit 1: Walk forward
- Edit 2: Stand at the second-floor window on the left side of the house and look directly outward

[Figure 98]

[Figure 99]

[Figure 100]

Fill in different puzzle pieces and select the best one

Edit 1 Edit 2

###### Figure 2. Overview for WEAVEBench. We have shown only a subset of the WEAVE; further details are in the Appendix D.2.

###### Multi-turn Data Creation Post-verification Extension

###### Remove-then-back Sequential Procedures Derivative Comparison

[Figure 101]

[Figure 102]

Multi-turn dataset

[Figure 103]

[Figure 104]

Instruction Creation

[Figure 105]

Instruction Creation

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Auto extension

[Figure 111]

[Figure 112]

Refine

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

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

CLIP Check

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

or

or

[Figure 144]

Self-verify

Images Creation Images Creation

VQA Re-edit

Captions

[Figure 145]

[Figure 146]

###### Multi-image Fusion

Check

Images Creation

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Instruction Creation

CLIP+ Qwen Check

[Figure 151]

[Figure 152]

[Figure 153]

| | |
|---|---|
| | |

Re-edit

[Figure 154]

Images Fusion

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Text Prompts Initial Images GT Images

[Figure 159]

[Figure 160]

Echo Dataset

[Figure 161]

More Edits

[Figure 162]

[Figure 163]

Refine and Repair

###### Quality Check

Refined Dataset

Pass

[Figure 164]

- Figure 3. Data Annotation Pipeline for WEAVE. Our methodology ensures data diversity and quality through a multi-round image generation process, supplemented by two rounds of validation and refinement. Additional details are provided in Appendix A.

Statistic Number

- • Total Chats 100,750

- - ≥ 4 Images Chats 100,584
- - ≥ 5 Images Chats 60,361
- - ≥ 6 Images Chats 31,571

• Average Chat Turns 3.79

- -Average Question Length 195.49

• Total Images 505,186

- - Maximum Image Per Chats 8
- - Average Image Per Chats 5.01

- Figure 4. Statistics for WEAVE-100k.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

|Train<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]|
|---|

|Test<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]|
|---|

[Figure 190]

Start GT #2 GT #4

KP VC IQ

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Edited

Images

Img #2 Img #3 Img #4

Start: ID Star: belt

Star: pose bg

Img #2: bg

Figure 5. Summary of domain distributions and evaluation methods for WEAVE.

###### 4.1. WEAVEBench

###### 3.3. Data Statics

[Figure 195]

Multi-image Fusion

Settings. We evaluated 4 LLMs, 7 Edit models, and 11 UMMs on WEAVEBench as presented in Table 2. Evaluations were conducted under three distinct in-context conditions: (1) no in-context (single-turn generation without contextual information), (2) partial in-context (using only selfgenerated images with explicitly mentioned visual context, excluding other historical interactions), and (3) complete incontext (with all previous interactions visible). For image placement, we employed two configurations: ”yes-first,” where images appear at their first mention position, and ”yes-front,” where all images are consolidated at the beginning of the input (results for this configuration are reported in Table 2). For models incapable of processing sequenceformat inputs, we implemented a concatenation approach following methodologies established in prior work [19, 89]. Based on the results presented in the table, we can derive the following conclusions:

[Figure 196]

For each instance in WEAVE, we provide a text prompt, one or more initial images, and ground-truth examples. The test set additionally includes key information that the correct output images must satisfy.

Edit

[Figure 197]

Recall

[Figure 198]

Visual Jigsaw

[Figure 199]

Chess Game

Representative dataset examples are provided in Appendix D. Table 4 presents key statistics of the training set. The majority of instances contain more than five images, with an average of 3.8 conversational turns per instance. Figure 5 illustrates the category distribution across both training and test sets, demonstrating a relatively balanced distribution across data types.

#### 4. Experiment

We first evaluate 22 models on WEAVEBench in Section 4.1, revealing that current models struggle with incontext interleaved generation and exhibit performance degradation as content length increases. Subsequently, in Section 4.2, we validate the high quality of WEAVE-100k through fine-tuning Bagel. Finally, we conduct quality analysis and assess judge effectiveness in Section 4.3 and 4.4.

In-context image generation remains challenging. Among the models tested, the best-performing Edit and UMM approaches achieved maximum scores of only 0.68 and 0.767, respectively. Furthermore, significant domain biases were observed, with performance in creative imagery consistently surpassing that in scientific and logical do-

Size In-context Modality Format Science Creation Logic Game Avg

Intern3.5-VL [70] 8B 0.114 0.500 0.667 0.292 0.231 Qwen3-VL [7] 8B 0.432 0.000 0.000 0.250 0.298 GPT-4o [1] - 0.591 0.500 0.167 0.083 0.381 GPT-4.1 [1] - 0.705 0.500 0.167 0.167 0.464

AnyEdit [86] 1B 0.445 0.514 0.351 0.419 0.472 UltraEdit(SD3) [91] 2B 0.493 0.561 0.491 0.440 0.522 VAREdit-8B [53] 8B 0.536 0.636 0.584 0.580 0.603

- Step1X-Edit v1.1 [48] 12B 0.574 0.714 0.700 0.625 0.669
- Step1X-Edit v1.2 [48] 12B 0.560 0.644 0.530 0.562 0.605 FLUX.1 Kontext [42] 12B 0.589 0.756 0.639 0.610 0.689 Qwen-Image-Edit [76] 20B 0.586 0.715 0.589 0.628 0.665

OminiGen [80] 4B 0.398 0.474 0.401 0.177 0.404 OminiGen2 [77] 7B * 0.511 0.682 0.551 0.511 0.609 Ovis-U1 [67] 3B * 0.402 0.557 0.364 0.357 0.422 UniPic [68] 1.5B 0.472 0.590 0.463 0.316 0.511 UniPic2-SD3.5M [73] 2B 0.477 0.625 0.543 0.497 0.568 UniPic2-Metaquery [73] 9B 0.493 0.666 0.507 0.444 0.582 NextStep-1-Large [66] 15B 0.519 0.620 0.437 0.309 0.534 Seedream 4.0 [62] - 0.683 0.847 0.679 0.635 0.765 Seedream 4.0 [62] - 0.667 0.830 0.646 0.599 0.746 Nano Banana [20] - 0.715 0.823 0.666 0.666 0.764 Nano Banana [20] - 0.710 0.843 0.730 0.613 0.767 Bagel [22] 14B * 0.378 0.475 0.406 0.365 0.446 Bagel-Zebra [45] 14B * 0.399 0.456 0.393 0.396 0.449 + WEAVE-100k 14B * 0.537 0.706 0.567 0.531 0.640

- Table 2. Main results on WEAVEBench. The symbols and * denote full and partial in-context, respectively. Icons , , and * indicate image-only, text-only, and combined evaluations, respectively. and represent sequential and concatenated image inputs, respectively. We use blue, orange, and green to represent the optimal results across three modalities.

mains. This suggests substantial room for improvement in generation ability to effectively integrate world knowledge. In-context usage matters (a) For comprehension tasks, we observed significant performance improvements when utilizing in-context information compared to baseline conditions without historical context. This effect was particularly pronounced in QwenVL, which demonstrated a remarkable 163% improvement as illustrated in Figure 6(a), indicating that WEAVEBench successfully incorporated historical information into the model evaluation. (b) For generation tasks, increasing in-context content produced divergent effects across model types. Open-source models exhibited progressive performance degradation with additional historical context—Qwen-Edit showed decremental performance of 5.3% and 8.6% respectively. This suggests that open-source models, constrained by single-round editing capabilities, experience diminished localization accuracy when processing expanded contextual information, thereby failing to effectively utilize in-context data. Conversely, proprietary models such as Nano demonstrated incremental improvement, indicating successful utilization of contextual information. (c) WEAVEBench exhibits superior image quality. As illustrated in Figure 6(b), incorpo-

rating WEAVEBench’s ground truth images as in-context examples resulted in performance improvements across all models. Notably, Qwen-Image-Edit demonstrated a significant improvement of 7.1%, potentially attributable to Qwen-Image-Edit’s inherently weaker generative capabilities compared to the nano-banana [21].

Sequential Input Superiority. As illustrated in Figure 6(c), sequential image input demonstrates significant performance advantages over concatenated input. This effect is particularly pronounced with the Bagel model, where concatenation results in a 10.3% performance degradation. These findings highlight the potential of UMMs as effective editing models, especially considering that traditional editing models cannot directly process multiple images and historical information as input.

###### 4.2. Train on WEAVE-100k

To demonstrate the effectiveness of our data, we conduct experiments on Bagel [22], with detailed training specifications provided in Appendix B.2. Our approach improved performance across four task categories:

(i) Vision Comprehension. Our data effectively enhanced performance on understanding tasks, particularly yielding a

Model BG Color Mat. Motion Port. Style Add Remove Replace Text Tone Avg AnyEdit [86] 4.31 4.25 2.64 0.67 1.90 1.95 3.72 3.75 3.23 0.77 4.21 2.85 MagicBrush [90] 6.17 5.41 4.75 1.55 2.90 4.10 5.53 4.13 5.10 1.33 5.07 4.19 InstructPix2Pix [9] 3.94 5.40 3.52 1.27 2.62 4.39 3.07 1.50 3.48 1.13 5.10 3.22 OmniGen [80] 5.23 5.93 5.44 3.12 3.17 4.88 6.33 6.35 5.34 4.31 4.96 5.01 Step1X-Edit [48] 7.03 6.26 6.46 3.66 5.23 7.24 7.17 6.42 7.39 7.40 6.62 6.44 OminiGen2 [77] 6.99 6.66 4.88 2.55 3.66 6.08 7.09 6.60 6.65 4.49 6.03 5.57 UltraEdit (SD3) [91] 5.83 5.51 5.86 3.55 5.00 5.73 5.06 3.15 5.79 2.24 5.45 4.83 EditMGT [3] 7.69 7.71 5.77 3.84 5.13 6.53 6.13 5.24 5.56 4.53 6.42 5.87 GoT-6B [23] 4.11 5.75 3.04 1.71 2.69 4.72 5.77 4.59 5.65 1.16 4.24 3.95 VAREdit-8B [53] 6.77 6.64 5.40 3.33 4.20 6.46 5.86 7.29 6.67 3.87 6.54 5.73 FluxKontext.dev [42] 7.06 7.03 5.52 5.62 4.68 5.55 6.95 6.76 6.13 6.10 7.48 6.26 Bagel [22] 7.44 6.99 6.26 5.09 4.82 6.04 7.94 7.37 7.31 7.16 6.17 6.52 + WEAVE-100k 7.45 7.00 7.10 4.97 4.83 6.98 7.88 7.39 7.75 7.06 6.81 6.83

Table 3. Comparison of fine-tuned Bagel and other models on GEdit-EN-full benchmark [48].

Understanding GenEval RISEBench Model MMB MMMU MMVet Single Obj. Two Obj. Count. Color Position Attri. Overall Tem. Cau. Spa. Log.

Emu3[71] 58.5 31.6 37.2 0.99 0.81 0.42 0.80 0.49 0.45 0.66 - - - Show-o [81] - 27.4 - 0.98 0.80 0.66 0.84 0.31 0.50 0.68 - - - Janus-Pro-7B [16] 75.5 36.3 39.8 0.99 0.89 0.59 0.90 0.79 0.66 0.80 - - - MetaQuery-XL [56] 83.5 58.6 66.6 - - - - - - 0.80 - - - Ovis-U1 [67] 77.8 51.1 66.7 0.98 0.98 0.90 0.92 0.79 0.75 0.89 1.2 3.3 4.0 2.4 BLIP3-o [11] 83.5 58.6 66.6 1.00 0.92 0.63 0.91 0.86 0.67 0.83 - - - EMU2 [64] - 34.1 48.5 - - - - - - - 1.2 1.1 0.0 0.0 OmniGen [80] 0.99 0.86 0.64 0.85 0.31 0.55 0.70 1.2 1.0 0.0 1.2 OmniGen2 [77] 79.1 53.1 61.8 1.00 0.95 0.64 0.88 0.55 0.76 0.80 - - - BAGEL [22] 85.0 55.3 67.2 0.99 0.94 0.81 0.88 0.64 0.63 0.82 2.4 5.6 14.0 1.2 + WEAVE-100k 85.2 60.7 67.4 1.00 0.94 0.83 0.89 0.65 0.70 0.84 4.7 6.7 21.0 2.4

Table 4. Comparison of different models on understanding tasks (MMB, MMMU, MMVet), GenEval and RISEBench.

- 9.8% improvement on MMMU [88]. (ii) Image Editing. As shown in Table 3, the fine-tuned Bagel demonstrates a 4.8% improvement in overall score on GEditBench [49]. Furthermore, the model surpasses its baseline counterpart in the majority of tasks, with particularly notable enhancements in material change and style change categories, showing improvements of 13.4% and 15.6%, respectively. (iii) Comprehension and Generation Collaboration. As evidenced in Table 4, the fine-tuned Bagel demonstrates significant improvements across RISE cognitive tasks. Particularly noteworthy are the 100% performance increases in both Spatial and Logical reasoning tasks. These results suggest that the fine-tuned Bagel more effectively leverages comprehension capabilities and world knowledge to enhance generation processes. Furthermore, these findings substantiate the high quality of the WEAVE-100k methodology. (iv) Interleaved Cross-modality Comprehension and Generation. As shown in Table 2, our fine-tuned model demonstrated a 42.5% improvement over Bagel on WEAVEBench. Notably, there was a 34.6% performance enhancement on more challenging science questions, indicating that training with our dataset significantly improved the model’s interleaved

cross-modality comprehension and generation capabilities.

###### 4.3. Quality Analysis.

As illustrated in Figure 7, our analysis of quality results yields the following conclusions: (i) Instruction-following capabilities still require further improvement. For instance, in the case on the left side of the figure, OmniGen and Ovis failed to execute the generation correctly. Similarly, in the case on the right side, the third column shows that QwenImage-Edit only generated a tower without including any human figures. (ii) Fine-tuning on the weave dataset resulted in the emergence of visual memory capabilities. The fine-tuned model correctly differentiates between protagonists wearing pink and yellow clothing in the left case, and in the right case, demonstrates the ability to first remove human figures and subsequently reintegrate them.

###### 4.4. Reliability of Judge Usage

To assess the reliability of VLM-as-a-judge scores, we conducted an expert evaluation study involving three human specialists across Nano-banana, Qwen-Image-Edit, and SeeDream models, analyzing 100 instances per model.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

no partial yes-first yes-front

vs human vs claude

seq cat

[Figure 204]

[Figure 205]

[Figure 206]

yes yes-gt

90

70

70

70

80

50

50

50

75

30

30

- 10

70

30

10

QwenVL Qwen-Edit Step1X Nano Qwen-Edit Bagel SeeDream KP VC IQ

GPT4o Nano

(a) In-context mode (b) Use GT history (c) Seq or cat (d) Evaluation reliability

- Figure 6. (a) Impact of different in-context modes on performance. (b) Reasoning performance using ground truth as in-context examples. (c) Performance variations when concatenating sequential images. (d) Evaluation reliability of GPT4.1 judger.

Ground Truth

BAGELweave

OminiGen2

Ovis-U1

UniPic2

QwenImage-Edit

BAGEL

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

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

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

- ① Mei chasing a small white and blue soot (Totoro) into the forest.
- ② Mei crawling into a tree hollow and falling asleep on the large Totoro's belly.
- ③ Satsuki holding an umbrella for Totoro, while Totoro gives her an acorn.
- ④ Satsuki and Mei riding the multi-legged cat bus to the hospital.

[Figure 272]

[Figure 273]

- ① Change the background of #1 to the most famous mountain in the country.
- ② Generate the most famous tower in the women‘s country’s capital.
- ③ Place the woman in #1 in front of the tower of #3

- Figure 7. Qualitative comparison between different methods. The left-side task requires preservation of character IDs, while the rightside task necessitates the application of world knowledge and maintenance of character removal followed by reinsertion.

We computed Pearson correlation coefficients [8] between GPT-4.1 scores and expert ratings, with a comparative analysis against Claude Opus 4.1 evaluations (Figure 6). Results demonstrate that correlations between GPT-4.1 and human ratings consistently exceed 0.8, while Claude evaluations exhibit strong cross-VLM consistency, suggesting that the specific choice of VLM evaluator has minimal impact on assessment outcomes.

#### 5. Conclusion

This work presents WEAVE, the first comprehensive suite for in-context interleaved cross-modality comprehension and generation. We introduce WEAVE-100k, a large-scale dataset comprising 100K samples that encompass 370K di-

alogue turns and 500K images, alongside WEAVEBench, a human-annotated benchmark consisting of 100 tasks with 480 images and featuring a hybrid VLM judge evaluation framework. Our experiments demonstrate that training on WEAVE-100k yields substantial improvements across established benchmarks, including 9.8% gains on MMMU and 4.8% on GEditBench, while facilitating the emergence of visual memory capabilities in UMMs. At mean while, extensive evaluations on WEAVEBench reveal that current models still struggle with multi-turn, context-aware generation, particularly as content length increases. Moreover, this challenging task proves beyond the capabilities of conventional editing models. WEAVE establish a foundation and underscore the critical need for in-context interleaved multimodal comprehension and generation.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 6, 2, 4, 11, 12, 13, 14

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 14

- [3] Anonymous. EditMGT: Unleashing potentials of masked generative transformers in image editing. In Submitted to The Fourteenth International Conference on Learning Representations, 2025. under review. 7, 2
- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 3
- [5] Jinbin Bai, Wei Chow, Ling Yang, Xiangtai Li, Juncheng Li, Hanwang Zhang, and Shuicheng Yan. Humanedit: A highquality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280, 2024. 2
- [6] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. In The Thirteenth International Conference on Learning Representations, 2024. 15
- [7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 11, 12, 13, 14
- [8] Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. Pearson correlation coefficient. In Noise reduction in speech processing, pages 1–4. Springer, 2009. 8
- [9] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 1, 7
- [10] Fu Chaoyou, Chen Peixian, Shen Yunhang, Qin Yulei, Zhang Mengdan, Lin Xu, Yang Jinrui, Zheng Xiawu, Li Ke, Sun Xing, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 3, 2023. 14
- [11] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 7

- [12] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019. 14
- [13] Sixiang Chen, Jinbin Bai, Zhuoran Zhao, Tian Ye, Qingyu Shi, Donghao Zhou, Wenhao Chai, Xin Lin, Jianzong Wu, Chao Tang, et al. An empirical study of gpt-4o image generation capabilities. arXiv preprint arXiv:2504.05979, 2025. 14
- [14] Wei Chen, Lin Li, Yongqi Yang, Bin Wen, Fan Yang, Tingting Gao, Yu Wu, and Long Chen. Comm: A coherent interleaved image-text dataset for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8073–8082, 2025. 3
- [15] Xi Chen and Xiao Wang. Pali: Scaling language-image learning in 100+ languages. In Conference on Neural Information Processing Systems (NeurIPS), 2022. 14
- [16] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2, 7

- [17] Wei Chow, Juncheng Li, Qifan Yu, Kaihang Pan, Hao Fei, Zhiqi Ge, Shuai Yang, Siliang Tang, Hanwang Zhang, and Qianru Sun. Unified generative and discriminative training for multi-modal large language models. Advances in Neural Information Processing Systems, 37:23155–23190, 2024. 15
- [18] Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, et al. Merit: Multilingual semantic retrieval with interleaved multi-condition query. arXiv preprint arXiv:2506.03144,

2025. 14

- [19] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411, 2025. 5, 3, 10
- [20] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 6, 2, 11, 12, 13, 14
- [21] Google DeepMind. Gemini 2.5 flash image. https : / / developers . googleblog . com / en / introducing-gemini-2-5-flash-image/, 2025. Accessed: 2025-10-30. 1, 6
- [22] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 2, 6, 7, 10, 11, 12, 13, 14, 15
- [23] Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Xihui Liu, and Hongsheng Li. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025. 7

- [24] Taoran Fang, Wei Zhou, Yifei Sun, Kaiqiao Han, Lvbin Ma, and Yang Yang. Exploring correlations of self-supervised tasks for graphs. arXiv preprint arXiv:2405.04245, 2024. 10
- [25] Taoran Fang, Tianhong Gao, Chunping Wang, Yihao Shang, Wei Chow, Lei Chen, and Yang Yang. Kaa: Kolmogorovarnold attention for enhancing attentive graph neural networks. arXiv preprint arXiv:2501.13456, 2025. 10
- [26] Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. Bias and fairness in large language models: A survey. Computational Linguistics, 50

(3):1097–1179, 2024. 15

- [27] Zhiqi Ge, Juncheng Li, Qifan Yu, Wei Zhou, Siliang Tang, and Yueting Zhuang. Demon24: Acm mm24 demonstrative instruction following challenge. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11426–11428, 2024. 14
- [28] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 14
- [29] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15733–15744,

2025. 13

- [30] Qingdong He, Xueqin Chen, Chaoyi Wang, Yanjie Pan, Xiaobin Hu, Zhenye Gan, Yabiao Wang, Chengjie Wang, Xiangtai Li, and Jiangning Zhang. Reasoning to edit: Hypothetical instruction-based image editing with visual reasoning. arXiv preprint arXiv:2507.01908, 2025. 3
- [31] Dan Hendrycks, Dawn Song, Christian Szegedy, Honglak Lee, Yarin Gal, Erik Brynjolfsson, Sharon Li, Andy Zou, Lionel Levine, Bo Han, et al. A definition of agi. arXiv preprint arXiv:2510.18212, 2025. 1
- [32] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 14
- [33] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025. 1
- [34] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025. 3, 14
- [35] Xuanwen Huang, Wei Chow, Yize Zhu, Yang Wang, Ziwei Chai, Chunping Wang, Lei Chen, and Yang Yang. Enhancing cross-domain link prediction via evolution process modeling. In Proceedings of the ACM on Web Conference 2025, pages 2158–2171, 2025. 3
- [36] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex

- instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362– 8371, 2024. 3
- [37] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1
- [38] Shashank Mohan Jain. Hugging face. In Introduction to transformers for NLP: With the hugging face library and models to solve problems, pages 51–67. Springer, 2022. 12
- [39] Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung. Towards mitigating llm hallucination via self reflection. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1827–1843, 2023. 15
- [40] Ying Jin, Pengyang Ling, Xiaoyi Dong, Pan Zhang, Jiaqi Wang, and Dahua Lin. Reasonpix2pix: instruction reasoning dataset for advanced image editing. arXiv preprint arXiv:2405.11190, 2024. 3
- [41] George Karypis, Eui-Hong Han, and Vipin Kumar. Chameleon: Hierarchical clustering using dynamic modeling. computer, 32(8):68–75, 1999. 2
- [42] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 6, 7, 11, 12, 13, 14

- [43] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36:71683–71702, 2023. 14
- [44] Xuanyu Lei, Zonghan Yang, Xinrui Chen, Peng Li, and Yang Liu. Scaffolding coordinates to promote vision-language coordination in large multi-modal models. arXiv preprint arXiv:2402.12058, 2024. 14
- [45] Ang Li, Charles Wang, Kaiyu Yue, Zikui Cai, Ollie Liu, Deqing Fu, Peng Guo, Wang Bill Zhu, Vatsal Sharan, Robin Jia, et al. Zebra-cot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025. 3, 6, 10, 11, 12, 13, 14, 15
- [46] Yongyuan Liang, Wei Chow, Feng Li, Ziqiao Ma, Xiyao Wang, Jiageng Mao, Jiuhai Chen, Jiatao Gu, Yue Wang, and Furong Huang. Rover: Benchmarking reciprocal crossmodal reasoning for omnimodal generation. arXiv preprint arXiv:2511.01163, 2025. 3, 14, 15
- [47] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [48] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu,

- Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 6, 7, 2, 11, 12, 13, 14
- [49] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 2, 3, 7
- [50] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 14
- [51] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 14
- [52] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751,

2025. 1, 2

- [53] Qingyang Mao, Qi Cai, Yehao Li, Yingwei Pan, Mingyue Cheng, Ting Yao, Qi Liu, and Tao Mei. Visual autoregressive modeling for instruction-guided image editing. arXiv preprint arXiv:2508.15772, 2025. 6, 7, 11, 12, 13, 14
- [54] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 14
- [55] Kaihang Pan, Siliang Tang, Juncheng Li, Zhaoyu Fan, Wei Chow, Shuicheng Yan, Tat-Seng Chua, Yueting Zhuang, and Hanwang Zhang. Auto-encoding morph-tokens for multimodal llm. arXiv preprint arXiv:2405.01926, 2024. 2
- [56] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 7
- [57] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 13
- [58] Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao Li. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv:2508.05606,

2025. 2

- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [60] Neal J Roese. Counterfactual thinking. Psychological bulletin, 121(1):133, 1997. 14
- [61] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 12
- [62] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward nextgeneration multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 1, 2, 3, 6, 11, 12, 13, 14
- [63] Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, et al. Query-kontext: An unified multimodal model for image generation and editing. arXiv preprint arXiv:2509.26641, 2025. 1, 2
- [64] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024. 7
- [65] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 14
- [66] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025. 6, 11, 12, 13, 14
- [67] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, et al. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025. 6, 7, 10, 11, 12, 13, 14
- [68] Peiyu Wang, Yi Peng, Yimeng Gan, Liang Hu, Tianyidan Xie, Xiaokun Wang, Yichen Wei, Chuanxin Tang, Bo Zhu, Changshi Li, et al. Skywork unipic: Unified autoregressive modeling for visual understanding and generation. arXiv preprint arXiv:2508.03320, 2025. 6, 10, 11, 12, 13, 14
- [69] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025. 1, 2, 3
- [70] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 6, 11, 12, 13, 14
- [71] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang,

- Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 7
- [72] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025. 1, 2
- [73] Hongyang Wei, Baixin Xu, Hongbo Liu, Cyrus Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, et al. Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model. arXiv preprint arXiv:2509.04548, 2025. 6, 11, 12, 13, 14
- [74] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 14
- [75] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025. 2
- [76] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 6, 2, 11, 12, 13, 14

- [77] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 6, 7, 10, 11, 12, 13, 14
- [78] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, MingHsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv:2505.16707, 2025. 3
- [79] Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, et al. Dreamomni2: Multimodal instruction-based editing and generation. arXiv preprint arXiv:2510.06679, 2025. 1, 2
- [80] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13294–13304,

2025. 6, 7, 10, 11, 12, 13, 14

- [81] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 1, 7

- [82] Shilin Xu, Yanwei Li, Rui Yang, Tao Zhang, Yueyi Sun, Wei Chow, Linfeng Li, Hang Song, Qi Xu, Yunhai Tong, et al. Mixed-r1: Unified reward perspective for reasoning capability in multimodal large language models. arXiv preprint arXiv:2505.24164, 2025. 3
- [83] Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vuli´c. Visual planning: Let’s think only with images. arXiv preprint arXiv:2505.11409, 2025. 14
- [84] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025. 1, 2, 3
- [85] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 1, 2, 14
- [86] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025. 1, 2, 6, 7, 11, 12, 13, 14
- [87] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 14
- [88] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 7, 14
- [89] Jieyu Zhang, Weikai Huang, Zixian Ma, Oscar Michel, Dong He, Tanmay Gupta, Wei-Chiu Ma, Ali Farhadi, Aniruddha Kembhavi, and Ranjay Krishna. Task me anything. In ThirtyEighth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. 5, 10
- [90] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024. 1, 2, 7
- [91] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. arXiv preprint arXiv:2407.05282, 2024. 6, 7, 11, 12, 13, 14
- [92] Siheng Zhao, Jiageng Mao, Wei Chow, Zeyu Shangguan, Tianheng Shi, Rong Xue, Yuxi Zheng, Yijia Weng, Yang You, Daniel Seita, et al. Robot learning from any images. In Conference on Robot Learning, pages 4226–4245. PMLR,

2025. 15

- [93] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pix-

- els: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 3
- [94] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 1, 2
- [95] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. Advances in Neural Information Processing Systems, 36:8958–8974, 2023. 14

[Figure 274]

## WEAVE: Unleashing and Benchmarking the In-context Interleaved

## Comprehension and Generation Supplementary Material

#### Contents

- 1. Introduction 1
- 2. Related Works 2
- 3. WEAVE 3

- 3.1. Data Collection . . . . . . . . . . . . . . . . . . . 3
- 3.2. Evaluation Settings and Metrics . . . . . . . . . . 3
- 3.3. Data Statics . . . . . . . . . . . . . . . . . . . . . 5

4. Experiment 5

- 4.1. WEAVEBench . . . . . . . . . . . . . . . . . . . 5

- 5. Conclusion 8

- 4.2. Train on WEAVE-100k . . . . . . . . . . . . . . . 6
- 4.3. Quality Analysis. . . . . . . . . . . . . . . . . . . 7
- 4.4. Reliability of Judge Usage . . . . . . . . . . . . . 7

- A. Weave Analysis 2

- A.1. Collection Process . . . . . . . . . . . . . . . . . 2
- A.2. Data Source for WEAVEBench . . . . . . . . . . 3
- A.3. Statistics . . . . . . . . . . . . . . . . . . . . . . 4

- B. Experiment Details 4

- B.1. Evaluation Prompts . . . . . . . . . . . . . . . . . 4
- B.2. Training Details . . . . . . . . . . . . . . . . . . . 6
- B.3. Details on Benchmarks and Metrics . . . . . . . . 9
- B.4. Baselines Details . . . . . . . . . . . . . . . . . . 10

- C. More Related Works 14
- D. Additional Examples for WEAVE 15

- D.1. Additional Examples for WEAVE-100k . . . . . . 15
- D.2. More example for WEAVEBench . . . . . . . . . 15

- E. Broader Impact 15

#### A. Weave Analysis

###### A.1. Collection Process

To ensure the quality of the generated data, we incorporated manual sampling verification into the design process of each pipeline to validate the success rate after filtering. Specifically, we utilized four pipelines, each with integrated quality assurance mechanisms.

- (i) Multi-image fusion: We achieved reference to previ-

ous iterations by fusing edited or directly generated images. For image fusion data, we utilized two primary sources. First, we leveraged the multi-image fusion dataset from Echo-4o [84], where image fusion was initially performed using GPT-Image. Due to quality inconsistencies in this dataset, we regenerated images using Seedream 4.0 [62] and refined instructions with GPT-4.1. Second, we generated single-round image fusion instructions with GPT4.1, including original image captions. We then produced original images using Qwen-Image [76], substituting suboptimal generations with Seedream 4.0 outputs, and performed multi-image fusion using Seedream 4.0. Building upon these single-round fusion data, we employed GPT-4.1 to annotate image editing instructions for the original images, categorizing them into five types: ’add’, ’remove’, ’replace’, ’color alter’, and ’background change’ following the taxonomy in [3]. We subsequently applied Step1XEdit(v1.2) [48] for single-round editing. For images failing our quality verification protocol, we utilized Nano Banana [20] for additional refinement. Finally, GPT-4.1 provided reverse instructions and captions for edited images. We used these edited images as originals and multi-fusion input images as edited results, concatenating the data to create comprehensive multi-round editing and multi-image fusion sequences.

- (ii) Remove-then-back: We employed GPT-4.1 [1] to

generate instructions for multi-round editing. Specifically, we designed the instructions such that one round would require adding back an object that had been previously removed or replaced in an earlier round. Following instruction generation, we implemented a filtering process wherein approximately 25% of instructions successfully met our criteria. The filtered instructions were subsequently utilized to generate outputs using Seedream 4.0 [62] and Nano Banana [20], after which we retained the superior generation based on qualitative assessment.

- (iii) Derivative imagination and comparison: We incor-

porated methods for deriving or imagining alternative solu-

tions or new images before fusion. Due to the inherent challenges in automating LLMs to generate associative content or editing data, we adapted chess game and visual jigsaw datasets from Zebra-CoT [45] using GPT-4.1 for both recombination and self-verification processes. Specifically, we modified the abbreviated chess notations into explicit editing instructions to mitigate potential comprehension difficulties in generative models when interpreting condensed commands.

(iv) Sequential procedures: We implemented sequential edits following narrative progressions or structured operations requiring visual memory during generation. This approach was particularly effective for scenarios where characters disappear and subsequently reappear within narratives. Multiple editing rounds on identical scenes evaluated model consistency maintenance capabilities. Our pipeline employed GPT-4.1 to generate instructions satisfying three requirements: (1) multi-step processes requiring visual representation at each stage, (2) explicit inter-step relationships, and (3) identifiable animated characters. To maximize generation diversity, we utilized the 12 categories defined in Table 6 to produce editing instructions. These constraints imposed significant demands on generative models; even state-of-the-art systems such as Seedream 4.0 [62] and Nano Banana [20] failed to produce high-quality data without human supervision. Consequently, we allocated GPT4.1-generated, human-screened story-based content to the test set, while retaining numerous multi-round editing examples identified during the filtering process for training. For data annotation, we employed SeedEdit 3.0 [69] and Nano Banana [20], while test set generation utilized Seedream 4.0 [62] and Nano Banana [20]. When using Nano Banana, we observed that providing style reference images improved generation quality. Therefore, we curated a set of style reference images, as shown in Figure 8.

Post-verification Process We identified frequent editing failures within the Nano Banana framework and implemented a supplementary verification protocol employing GPT-4.1 for processed data evaluation. Problematic samples were detected using CLIP similarity metrics [59]. Samples exhibiting abnormally high similarity scores underwent re-editing via Step1X v1.2. Unmodified samples following this secondary editing attempt—identified through joint supervision by CLIP and Qwen3-VL-4B metrics—were systematically excluded from the dataset while maintaining referential integrity of image identifiers.

Comprehension Extension To incorporate comprehension tasks into our dataset, we randomly sampled from the filtered generated data and expanded it using GPT4.1. Each data point was annotated with at most one turn. The comprehension tasks primarily consisted of captioning tasks, questions regarding quantities and relationships within images, and a small subset of knowledge-based in-

quiries [4, 35, 82].

###### A.2. Data Source for WEAVEBench

WEAVEBench primarily utilizes web-collected data, with select images refined using SeedEdit 3.0. The jigsaw and chess game images are sourced from Zebra-CoT [45], while various optical and physical phenomena images are drawn from PhysBench [19]. Additionally, the dataset incorporates synthetically generated images from three models: Seedream 4.0 [62], Nano Banana [20], and SeedEdit 3.0 [69].

Domain Type #Chats Multi-image Fusion

GPT-Image 72348 SeeDream 3648

Recall 1369 Animals 91 Architecture 74 Cartoon 135 Fashion 73 Fantasy 126 Food 116 Nature Landscapes 164 Plants 54 Products 77 Real Human 347 Sports 49 Vehicles 63 Edit 19903 None 18261 Animals 263 Architecture 114 Cartoon 96 Fashion 141 Fantasy 98 Food 234 Nature Landscapes 105 Plants 97 Products 164 Real Human 136 Sports 98 Vehicles 96

Visual Jigsaw 1286 None 1286 Chess Game 2196 None 2196

Total 100750

Table 5. The detailed statistics of the WEAVE-100k dataset.

Abstract Expressionist Acrylic Painting Anime Art Nouveau

Black And White Photography Cartoon Cel Shading

Baroque

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Charcoal Drawing Chinese Ink Wash Colored Pencil Cubist Cyberpunk Digital Painting Expressionist Fantasy Art

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Film Noir Glitch Art Graffiti Graphite Drawing Hyperrealistic Impressionist Ink Painting Isometric

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

Low Poly Manga Marker Drawing Minimalist Mosaic Neon Art Oil Painting Pastel Drawing

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Pen And Ink Pencil Sketch Photorealistic

Polaroid

Realism

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Pop Art

Pointillism

Pixel Art

[Figure 312]

[Figure 313]

[Figure 314]

Renaissance Sci-Fi Art Screen Print Sepia Tone Stained Glass Steampunk Street Art Surrealist

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Synthwave Ukiyo-E Vaporwave Vector Art Vintage Photography Watercolor Woodcut

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

###### A.3. Statistics

Figure 8. Image style examples used in Nano bana inference.

#### B. Experiment Details

While Section 3.3 presents the proportional distribution of various data types in WEAVE-100k and WEAVEBench, Table 5 in this section provides a more granular breakdown of the composition of sub-domains and domains within the complete WEAVE-100k dataset.

###### B.1. Evaluation Prompts

We employ GPT-4o [1] as our evaluation judge for the main experimental results presented in Figure 2. The evaluation prompts used to assess the four dimensions—Key Point Correctness, Visual Consistency, Image Quality, and Accuracy—are illustrated in Figures 9, 10, 11, and 12, respectively.

Table 6. Dataset categories with main content, scenarios, and editable dimensions.

Category Main Content Scenarios Editable Dimensions

Food & Drink Staples, snacks, desserts, fruits, beverages (hot/cold) Dining tables, restaurants, street stalls, picnics, festive banquets

Ingredient substitution, plating style, scene modification, style adjustment

Real Humans Portraits, full-body, half-body, group photos; Actions: standing, walking, exercising, socializing, working

Indoor/outdoor, offices, streets, event venues

Clothing change, pose adjustment, background modification, expression change

Animals & Pets Pets (cats, dogs, rabbits, birds), farm animals (cattle, sheep, horses), wildlife (lions, elephants, bears), marine life (fish, dolphins, whales), insects & reptiles (butterflies, spiders, snakes), mythical creatures (dragons, unicorns, phoenix); Actions: playing, running, sleeping, eating, flying, swimming

Homes, parks, zoos, natural habitats, aquariums

Breed change, color variation, accessory addition, scene switching, pose adjustment

Architecture & Interior

Exteriors (modern buildings, historical structures, skyscrapers, bridges, churches, castles), interiors (living rooms, bedrooms, kitchens, offices, caf´es); Styles: modern, vintage, industrial, Nordic, Japanese, Chinese

City skylines, countryside, historic districts, campus landscapes

Style change, furniture replacement, lighting adjustment, seasonal variation, decoration modification

Natural environments

Weather change, time transition, seasonal switching, color adjustment, natural element addition

Nature & Landscapes

Terrain (mountains, canyons, plains, deserts, glaciers), water bodies (oceans, lakes, rivers, waterfalls), vegetation (forests, grasslands, bamboo groves, rainforests), sky (sunrise, sunset, starry sky, aurora, sea of clouds); Seasons: spring, summer, autumn, winter; Weather: sunny, rainy, foggy, snowy, stormy

Color variation, material change, arrangement combination, background switching, lighting adjustment

White background, display stands, lifestyle scenes, desktops, outdoor settings

Products & Objects

Electronics (phones, earbuds, cameras, laptops, tablets), fashion accessories (watches, bags, jewelry, sunglasses), cosmetics (perfume, lipstick, skincare), home goods (lamps, vases, cushions, tableware), books, stationery, toys, sports equipment

Fantasy worlds, modern cities, space, magic academies

Clothing change, expression adjustment, color scheme change, scene switching, style transformation

Cartoon & Stylized Characters

Anime characters (Japanese anime, manga), Western cartoons (Disney/Pixar, American comics), 3D characters (game characters, virtual avatars), mascots & avatars (brand mascots, social media avatars), Qversion/Chibi, fantasy hybrids (robots, elves, monsters, hybrid creatures)

Species change, color variation, layout adjustment, background modification, seasonal change

Flowers & Plants Flowers (roses, tulips, cherry blossoms, sunflowers, peonies, orchids), plants (potted plants, succulents, foliage plants, trees, vines)

Gardens, vases, outdoors, greenhouses, balconies, floral arrangements

Vehicles Land (cars, motorcycles, bicycles, buses, trains), air (airplanes, helicopters, hot air balloons), water (yachts, sailboats, ferries, speedboats); Views: side, front, aerial, interior

City streets, highways, racetracks, parking lots, airports, ports, showrooms

Color change, model replacement, background modification, modification addition, lighting adjustment

Fantasy & Sci-Fi Sci-fi elements (spaceships, aliens, robots, futuristic cities, cyberpunk streets), fantasy elements (magic scenes, fantasy creatures, magic academies, elf forests, dragon lairs), surreal art (dreamscapes, geometric abstractions, spacetime distortions)

Space stations, alien planets, magic worlds, parallel universes

Creature replacement, environment change, effect addition, atmosphere adjustment, style transformation

Sports & Fitness Ball sports (basketball, soccer, tennis, volleyball, golf), fitness activities (yoga, running, weightlifting, swimming, cycling), extreme sports (rock climbing, skiing, surfing, skydiving), equipment (gym machines, sports gear)

Stadiums, gyms, outdoor fields, pools, competition venues

Action variation, equipment change, scene switching, sport type change

Fashion & Clothing

Apparel (dresses, suits, casual wear, sportswear, formal wear), accessories (shoes, hats, scarves, belts), display methods (hangers, mannequins, flat lay); Styles: streetwear, elegant, athletic, business, vintage

Runways, street photography, studios, stores, fashion exhibitions

Color/pattern variation, style adjustment, combination matching, scene switching

###### Prompt for Key point Checklist

# Image Generation Evaluation Framework

You are a strict and professional visual evaluation specialist with expertise in assessing AI-generated images. Your task is to determine how accurately

and effectively a generated image fulfills the specified instructions, using a structured evaluation methodology.

## Input Structure For each evaluation, you will receive:

- 1. **Generated Image**: The final output image created by an AI system
- 2. **Task Instructions**: A detailed set of requirements that includes:

- - One or more reference images
- - Specific modification requirements for each reference image ## Evaluation Criteria

Evaluate the generated image based on these key dimensions:

- 1. **Requirement Fulfillment** (70% weight)

- - Accuracy: How precisely each requested modification was implemented
- - Completeness: Whether all specified changes were executed
- - Fidelity: How well important elements from the reference were preserved

- 2. **Visual Quality** (30% weight)

- - Coherence: Natural integration of modifications without artifacts
- - Composition: Balanced visual arrangement maintaining artistic integrity
- - Detail: Appropriate level of detail in modified elements

## Evaluation Process

- 1. **Initial Assessment**: First examine both the reference and generated images side by side
- 2. **Systematic Review**: Analyze each instruction requirement individually
- 3. **Requirement Tracking**: Create a mental checklist to ensure no requirements are overlooked
- 4. **Visual Verification**: Identify specific visual evidence for each implementation success or failure
- 5. **Holistic Scoring**: Consider both technical execution and artistic integrity in your final score ## Scoring Guidelines

- - **9-10**: Exceptional execution with virtually all requirements perfectly implemented
- - **7-8**: Strong implementation with minor oversights or quality issues
- - **5-6**: Adequate implementation with noticeable flaws in several requirements
- - **3-4**: Partial implementation with significant omissions or quality issues
- - **0-2**: Poor implementation with most requirements missed or poorly executed

## Output Format

Provide your evaluation in strict JSON format: ```json

{

"score": 0-10,

"reasoning": "Simple explanation of your assessment"

}

## Important Notes

- - Evaluate ALL task instructions comprehensively, not just the most obvious ones
- - Consider both what was added/modified AND what was correctly preserved
- - Be precise in your reasoning, citing specific visual elements as evidence
- - Focus on objective evaluation rather than subjective aesthetic preferences
- - Consider technical difficulty when assessing complex modifications

Now:

Generated Image is <image>\n Task Instructions is:

Figure 9. Prompt for Evaluating Key Point Correctness.

###### B.2. Training Details

into multi-turn dialogues would exceed the maximum context length of the H100 GPUs. Therefore, we implemented a random sampling approach where we selected individual conversation turns for training rather than including complete dialogue sequences. Additionally, our dataset utilized the notation “Image #3” to reference specific images. Since our methodology involved randomly selecting single turns, we refined these numerical references to correctly reflect the

We trained the model on 8× NVIDIA H100 GPUs with the batch size per GPU set to 1, for a total of 30,000 training steps, requiring approximately 60 hours of compute time. Due to the token-intensive nature of images in the Bagel dataset, many of our samples contained more than three images within a single conversation turn. Concatenating these

##### Prompt for Visual Consistency

# Image Consistency Evaluation Framework You are a meticulous, uncompromising visual forensics expert with exceptional attention to detail. Your mission is to conducta pixel-level analysis of

image consistency, applying the strictest possible standards in your evaluation. You will assess with scientific precision whether non-target elements

maintain perfect visual fidelity with reference images.

## Input Structure

You will receive:

- 1. **Generated Image**: The result image for the composition of task instructions
- 2. **Task Instructions**: One or more specific requirements, each containing: A reference image modification requirements

## Your Primary Objective

Evaluate whether **non-target elements** in the generated image remain **visually consistent** with the reference images included in the task instructions. Focus exclusively on elements that should NOT have changed according to the requirements.

## Consistency Evaluation Guidelines ### Elements That Must Remain Consistent

- - **Background Elements**: Environment, scenery, setting details not mentioned in tasks
- - **Unrelated Objects**: Items not involved in the editing process
- - **Structural Elements**: Basic composition, layout, perspective (unless specified for change)
- - **Identity Preservation**: People, animals, or objects should maintain their core characteristics
- - **Style Consistency**: Overall visual style, lighting conditions, color palette ### Elements Expected to Change
- - **Target Objects**: Items explicitly mentioned in task instructions
- - **Direct Consequences**: Changes that logically result from the intended transformations
- - **Process Effects**: Visual effects directly caused by the editing process ## Evaluation Process

- 1. **Identify Task Requirements**: Analyze each task instruction and its associated reference image
- 2. **Identify Target Elements**: Clearly define what should change based on task instructions
- 3. **Identify Preservation Elements**: Determine what should remain unchanged
- 4. **Compare Preservation Quality**: Assess how well non-target elements maintained consistency with reference images
- 5. **Evaluate Impact**: Determine how any inconsistencies affect overall visual coherence ## Scoring Scale (0-10)

| Score | Description |

|-------|-------------|

| **10** | **Absolute Perfection**: Forensic analysis reveals zero detectable differences in any non-target element | | **9** | **Near-Perfect**: Microscopic deviations detectable only through pixel-level analysis |

| **8** | **Superior**: Minimal deviations visible only under intense scrutiny |

| **7** | **Highly Proficient**: Minor inconsistencies visible upon close inspection | | **6** | **Proficient**: Small but noticeable inconsistencies in non-target elements |

| **5** | **Borderline Acceptable**: Multiple clear inconsistencies affecting visual coherence |

| **4** | **Substandard**: Numerous obvious inconsistencies compromising visual integrity |

| **3** | **Deficient**: Significant inconsistencies creating visual dissonance |

| **2** | **Severely Deficient**: Major alterations rendering non-target elements barely recognizable | | **1** | **Critical Failure**: Extreme inconsistencies with fundamental breakdown of visual coherence |

| **0** | **Complete Corruption**: Non-target elements utterly transformed, bearing no resemblance to references |

## Output Format

Provide your evaluation in strict JSON format:

```json

{ "score": 0-10,

"reasoning": "Simple explanation of your assessment"

} ```

Important Notes:

- - You will be evaluating a generated image against reference images embedded within multiple task instructions
- - Each task instruction contains both a reference image and specific requirements for changes
- - You must consider all tasks comprehensively when evaluating consistency
- - Focus solely on whether elements that should NOT have changed remained consistent with their appearance in the reference images

Now

Generated Image is <image>\n

Task Instructions is

- Figure 10. Prompt for Evaluating Visual Consistency.

##### Prompt for Image Quality

You are a uncompromising and professional image quality assessor specializing in AI-generated content evaluation.

You will be given:

1. **Generated Image**: an AI-generated image to evaluate Your Objective:

Evaluate the **perceptual quality** of the AI-generated image, focusing on technical excellence, visual coherence, and absence of

generation artifacts.

## Quality Assessment Dimensions: ### Structural Coherence

- - **Anatomy/Geometry**: Correct proportions, realistic structures, proper object shapes
- - **Spatial Relationships**: Logical positioning, appropriate scale relationships
- - **Compositional Logic**: Coherent scene layout, proper perspective ### Visual Fidelity
- - **Texture Quality**: Realistic surface textures, appropriate material appearance
- - **Detail Clarity**: Sharp important details, appropriate level of detail throughout
- - **Color Accuracy**: Natural color distribution, proper lighting/shadow ### Generation Artifacts
- - **Duplication Issues**: Repeated elements, phantom objects, merged features
- - **Blending Problems**: Unnatural transitions, ghosting effects, edge artifacts
- - **Distortion Errors**: Warped features, impossible geometries, scale inconsistencies ### Overall Naturalness
- - **Photorealism**: Does the image look natural and believable?
- - **Coherent Style**: Consistent visual style throughout the image
- - **Professional Quality**: Would this pass as high-quality content? ## Evaluation Scale (0 to 10):
- - **9-10 Exceptional Quality**: **Professional-grade image** with **no noticeable artifacts or flaws**; perfect technical excellence and photorealistic quality
- - **7-8 Very Good Quality**: **High-quality image** with **minimal flaws** that don't affect overall impression
- - **5-6 Good Quality**: **Decent image** with **some noticeable flaws** but overall usable
- - **3-4 Fair Quality**: **Multiple noticeable flaws** that somewhat detract from image usability
- - **1-2 Poor Quality**: **Multiple significant flaws** that severely detract from image usability
- - **0 Unusable Quality**: **Major structural problems**, severe artifacts, completely unusable

## Important Note:

If the input is a composite of multiple images (collage, grid, multiple separate images combined) rather than a single coherent

image, the maximum possible score is 4, regardless of quality.

## Reasoning Steps:

- 1. **Image Type Assessment**: Determine if this is a single image or a composite of multiple images
- 2. **Structural Analysis**: Assess geometric and anatomical correctness
- 3. **Fidelity Evaluation**: Check texture, detail, and color quality
- 4. **Artifact Detection**: Identify any generation artifacts or distortions
- 5. **Naturalness Assessment**: Evaluate overall believability and professional quality ## Input: <image>\n

## Output Format:

You must return your evaluation as a JSON object with the following structure:

```json

{ "score": 0-10, "reasoning": "Simple explanation of your assessment" }

``` Note: The score must be an integer value between 0 and 10.

- Figure 11. Prompt for Evaluating Image Quality.

###### Prompt for Key point Checklist

You are a careful expert answer evaluator with deep analytical capabilities.

You will be given:

- 1. **Standard Answer**: The correct answer or reference solution
- 2. **Generated Answer**: An answer to evaluate against the standard

Your Objective:

Evaluate how closely the generated answer matches the standard answer in terms of correctness, completeness, and accuracy.

## Evaluation Dimensions: ### Content Accuracy

- - **Factual Correctness**: Whether facts, data, and information are correct
- - **Conceptual Alignment**: Whether key concepts and ideas match the standard
- - **Error Presence**: Absence of incorrect statements or misunderstandings ### Completeness
- - **Key Points Coverage**: Whether all essential points from standard are covered
- - **Detail Level**: Appropriate depth of information compared to standard
- - **Scope Alignment**: Whether the generated answer stays within the proper scope ## Scoring System (0, 5, or 10 only):
- - **10 - Excellent Match**: The generated answer contains all key information from the standard answer with no significant errors or omissions. It may use different wording but conveys the same meaning and reaches the same conclusions.
- - **5 - Partial Match**: The generated answer contains some key information from the standard answer but has notable omissions or errors. It partially addresses the question but misses important elements or includes some incorrect information.
- - **0 - Poor Match/Mismatch**: The generated answer is substantially different from the standard answer, contains major factual errors, misses most key points, or demonstrates fundamental misunderstanding of the question. ## Reasoning Steps:

- 1. **Content Comparison**: Identify key points in both answers and compare them
- 2. **Gap Analysis**: Determine what important information is missing from the generated answer
- 3. **Error Detection**: Identify any incorrect information in the generated answer
- 4. **Holistic Assessment**: Consider the overall effectiveness of the generated answer compared to standard

## Output Format:

You must return your evaluation as a JSON object with the following structure:

{{

"score": 0|5|10,

"reasoning": "Simple explanation of your assessment"

}}

Note: The score must be exactly 0, 5, or 10 with no other values permitted. Now:

Standard Answer is : {standard_answer}

Generated Answer is : {generated_answer}

Figure 12. Prompt for Evaluating Comprehension Accuracy.

sequential position of images in the post-processing phase. During training, we employed the following hyperparameters: maximum latent size of 64, learning rate of 2 × 10−5, maximum number of tokens set to 11,520, maximum tokens per sample limited to 10,240, vision transformer conditional dropout probability of 0, and exponential moving average (EMA) decay rate of 0.9999.

fillment of requirements—specifically the Key Points (KP) mentioned in Section 3.2—is paramount. We employ the following scoring methodology: For generation tasks exclusively, the composite score is calculated as:

Score = 0.50 · KP + 0.20 · VC + 0.30 · IQ (1)

When evaluating unified models for both generation and comprehension tasks, the scoring formula becomes:

###### B.3. Details on Benchmarks and Metrics

Score = 0.40·KP+0.10·VC+0.20·IQ+0.30·ACC (2) For comprehension tasks in isolation, we report ACC di-

Score Weights The importance across evaluation dimensions varies considerably. For instance, in editing tasks, ful-

rectly.

Detailed Results for WEAVEBench The leaderboard scores for WEAVEBench are presented in Table 2. Detailed performance metrics for each model across the four major categories—Science, Creation, Logic, and Game—are provided in Table 7, Table 8, Table 9, and Table 10, respectively.

History Usage. Evaluations were conducted under three distinct in-context conditions: (1) no history (single-turn generation without contextual information), (2) partial history (incorporating only self-generated images with explicitly mentioned visual context, excluding prior interactions), and (3) complete history (incorporating all previous interactions). For image placement, we implemented two configurations: “yes-first,” where images appear at their first mention position, and “yes-front,” where all images are consolidated at the beginning of the input (results reported in Table 2). We denote the use of ground truth images in history as “yes-gt” in Figure 6, which was implemented based on the “yes-front” configuration. In the implementation of complete history, VLMs had access to all historical dialogue, while generative models only received historical images as input, since most cannot process dialogue information (with limited exceptions such as nano-banana). Consequently, we adopted the approach of providing only images as historical context.

Image Concatenation Methodology. For models incapable of processing sequence-format inputs, we implemented a concatenation approach following established precedents [19, 24, 24, 25, 89]. Specifically, images were arranged horizontally in a single row, with sequential numerical identifiers annotated in the upper-left corner of each image. We observed that after implementing the concatenation approach, certain models such as Step1X were unable to distinguish which specific image required editing, and continued to maintain the original dimensions in their outputs. Consequently, when presenting examples in Table 7, we extracted the relevant portions and rescaled them to either their original dimensions or to dimensions consistent with other models for comparative display purposes.

###### B.4. Baselines Details

We evaluated 4 LLMs, 7 Edit models, and 11 UMMs on WEAVEBench as presented in Table 2. In this section, we provide detailed information regarding the parameter configurations for these models.

###### Unified Models

• Bagel [22] is an open-source multimodal foundation model comprising 7B active parameters (14B total) trained on large-scale interleaved multimodal data. Bagel

demonstrates superior performance relative to state-ofthe-art open-source VLMs across standard multimodal understanding benchmarks. Concurrently, it achieves text-to-image generation quality comparable to specialized models such as Stable Diffusion 3. Throughout our experimental evaluation, we adhere to the officially recommended parameters and prompting strategies. Bagel-Zebra [45] is a variant of the model that has been fine-tuned using the Zebra-Chain-of-Thought (Zebra-COT) methodology [45].

- • OmniGen2 [77] represents a unified multimodal generative framework exhibiting enhanced computational efficiency and modeling capacity. Unlike its predecessor OmniGen v1, OmniGen2 implements a dual-pathway decoding architecture with modality-specific parameters for text and image generation, coupled with a decoupled image tokenization mechanism. For our experimental evaluation, we configure the temporal offset parameter to 3.0, the text guidance scale to 5.0, and the image guidance scale to 1.5. The negative prompt is specified as "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor bar". All inference procedures employ a 50-step sampling schedule.

- • OmniGen [80] is a unified image generation model capable of producing a wide range of images from multimodal prompts. This model was open-sourced by the Beijing Academy of Artificial Intelligence (BAAI). For our implementation, we utilize the following parameters: height=1024, width=1024, guidance scale=2.5, img guidance scale=1.6, and seed=0.

- • Ovis-U1 [67] is a unified model for multimodal understanding, text-to-image generation, and image editing, open-sourced by Alibaba’s AIDC group. We employ the following parameters: steps=50, img cfg=1.5, and txt cfg=6. It should be noted that Ovis’s generation tasks only support single-image input; therefore, for data with two or more images, we implemented image concatenation. The understanding tasks, however, support multiple sequential image inputs.

- • UniPic [68] is Skywork’s unified generation and understanding model, encompassing three variants:

- – UniPic-1.0 — 1.5B parameters, employing Unified Autoregressive Modeling for joint visual understanding and generation, enabling a single transformer to handle both perception and synthesis tasks.
- – UniPic-2.0 Series — SD3.5M-Kontext and MetaQuery variants based on Efficient Architectures with Diffusion Post-Training, delivering state-of-the-art perfor-

Intern3.5-VL [70] 8B - - - 0.114 0.114 Qwen3-VL [7] 8B - - - 0.432 0.432 GPT-4o [1] - - - - 0.591 0.591 GPT-4.1 [1] - - - - 0.705 0.705

AnyEdit [86] 1B 0.376 0.563 0.481 - 0.445 UltraEdit(SD3) [91] 2B 0.45 0.558 0.528 - 0.493 VAREdit-8B [53] 8B 0.437 0.661 0.618 - 0.536 Step1X-Edit v1.1 [48] 12B 0.442 0.821 0.630 - 0.574 Step1X-Edit v1.2 [48] 12B 0.497 0.622 0.625 - 0.560 FLUX.1 Kontext [42] 12B 0.500 0.755 0.628 - 0.589 Qwen-Image-Edit [76] 20B 0.510 0.622 0.687 - 0.586

OminiGen [80] 4B 0.375 0.343 0.473 - 0.398 OminiGen2 [77] 7B * 0.455 0.501 0.612 - 0.511 Ovis-U1 [67] 3B * 0.466 0.545 0.569 0.159 0.402 UniPic [68] 1.5B 0.490 0.455 0.454 - 0.472 UniPic2-SD3.5M [73] 2B 0.422 0.558 0.513 - 0.477 UniPic2-Metaquery [73] 9B 0.442 0.542 0.546 - 0.493 NextStep-1-Large [66] 15B 0.515 0.516 0.528 - 0.519 Seedream 4.0 [62] - 0.617 0.686 0.791 - 0.683 Seedream 4.0 [62] - 0.597 0.678 0.778 - 0.667 Nano Banana [20] - 0.631 0.763 0.824 - 0.715 Nano Banana [20] - 0.633 0.739 0.818 - 0.710 Bagel [22] 14B * 0.446 0.534 0.528 0.136 0.378 Bagel-Zebra [45] 14B * 0.463 0.561 0.551 0.159 0.399 + WEAVE-100k 14B * 0.500 0.584 0.569 - 0.537

- Table 7. Main results on WEAVEBench Science Part. and denote full and partial in-context history, respectively. , , and

* indicate image-only, text-only, and combined evaluations, respectively. and represent sequential and concatenated image inputs, respectively.

mance in text-to-image generation, fine-grained image editing, and multimodal reasoning.

For UniPic-1.0, we utilize the following hyperparameters: image size=1024, num iter=32, cfg=3, cfg prompt="Repeat this image", cfg schedule="constant", and temperature=1.0. For all UniPic-2.0 variants, we employ: num inference steps=50, guidance scale=3.5, and seed=42. Notably, UniPic-2.0 tokenizes images after adjusting their height and width to the nearest downward multiple of 16.

• NextStep-1-Large-Edit [66] is a 14B autoregressive model paired with a 157M flow matching head, trained on discrete text tokens and continuous image tokens with next-token prediction objectives. NextStep-1 achieves state-of-the-art performance for autoregressive models in text-to-image generation tasks, exhibiting strong capabilities in high-fidelity image synthesis. Since it only supports a single <image> tag, we followed the case format by placing <image> at the beginning and inputting images sequentially. The hy-

perparameters used were: num images per caption=1, positive prompt=None, negative prompt="Copy original image.", cfg=7.5, cfg img=2, cfg schedule="constant", use norm=True, num sampling steps=50, timesteps shift=3.2, and seed=42.

- • Seedream 4.0 [62] is a new-generation image creation model that integrates image generation and image editing capabilities into a single, unified architecture. Some images were omitted after multiple attempts due to sensitive content flags. The parameters used were: size="2k" and sequential image generation="disabled".

- • Nano Banana [20] is a top-rated AI image generation and image editing tool from Google DeepMind that enables the transformation of a single photograph into numerous novel creations. No special parameter configurations were employed in our implementation.

Image Editing Models We establish the models listed in Table 2 as baselines, comprising six open-source models: AnyEdit, UltraEdit (SD3) with diffusion architec-

Intern3.5-VL [70] 8B - - - 0.500 0.500 Qwen3-VL [7] 8B - - - 0.000 0.000 GPT-4o [1] - - - - 0.500 0.500 GPT-4.1 [1] - - - - 0.500 0.500

AnyEdit [86] 1B 0.460 0.572 0.566 - 0.514 UltraEdit(SD3) [91] 2B 0.531 0.599 0.587 - 0.561 VAREdit-8B [53] 8B 0.645 0.662 0.603 - 0.636 Step1X-Edit v1.1 [48] 12B 0.646 0.877 0.720 - 0.714 Step1X-Edit v1.2 [48] 12B 0.643 0.680 0.622 - 0.644 FLUX.1 Kontext [42] 12B 0.705 0.879 0.759 - 0.756 Qwen-Image-Edit [76] 20B 0.706 0.739 0.715 - 0.715

OminiGen [80] 4B 0.473 0.425 0.507 - 0.474 OminiGen2 [77] 7B * 0.644 0.675 0.751 - 0.682 Ovis-U1 [67] 3B * 0.500 0.593 0.590 0.555 0.557 UniPic [68] 1.5B 0.619 0.584 0.545 - 0.590 UniPic2-SD3.5M [73] 2B 0.613 0.638 0.637 - 0.625 UniPic2-Metaquery [73] 9B 0.664 0.664 0.670 - 0.666 NextStep-1-Large [66] 15B 0.652 0.636 0.556 - 0.620 Seedream 4.0 [62] - 0.840 0.869 0.843 - 0.847 Seedream 4.0 [62] - 0.828 0.842 0.824 - 0.830 Nano Banana [20] - 0.819 0.856 0.806 - 0.823 Nano Banana [20] - 0.838 0.873 0.832 - 0.843 Bagel [22] 14B * 0.683 0.685 0.666 0.000 0.475 Bagel-Zebra [45] 14B * 0.667 0.661 0.614 0.000 0.456 + WEAVE-100k 14B * 0.734 0.743 0.635 - 0.706

- Table 8. Main results on WEAVEBench Creation Part. and denote full and partial in-context history, respectively. , , and

* indicate image-only, text-only, and combined evaluations, respectively. and represent sequential and concatenated image inputs, respectively.

ture, FLUX.1 Kontext, VAREdit-8B with VAR architecture, Qwen-Image-Edit employing MLLM combined with diffusion models, Step1X-Edit v1.1, and Step1X-Edit v1.2. We strictly adhere to the default hyperparameters provided in the official GitHub repositories or Hugging Face [38] implementations of these baseline models. The key parameter configurations are enumerated below:

• Qwen-Image-Edit [76]: An image editing variant of Qwen-Image that extends the foundational 20B QwenImage model’s text rendering capabilities to instructionbased image editing tasks, enabling precise textual modifications within images. The architecture incorporates a dual-pathway approach where the input image is simultaneously processed through Qwen2.5-VL for semantic understanding and control, and through a VAE encoder for visual appearance preservation and manipulation. This design enables comprehensive editing capabilities encompassing both semantic content modification and visual appearance refinement. Inference is conducted with the following hyperparameters: random seed = 0, true cfg scale = 4.0, negative prompt

= "", and num inference steps = 50.

- • FLUX.1-Kontext [42]: A 12 billion parameter rectified flow transformer architecture designed for instructionguided image editing. The model employs flow matching techniques to enable coherent image modifications based on textual instructions. We set guidance scale

= 2.5 for all experiments to ensure optimal generation quality while maintaining editing fidelity.

- • UltraEdit [91]: This model is trained on approximately 4 million instruction-based editing samples using the Stable Diffusion 3 [61] architecture. It supports both free-form and mask-based input modalities to enhance editing performance. For consistency across all experiments, we exclusively employ its free-form variant. We note that since UltraEdit is trained on the SD3 architecture, its performance metrics may not fully reflect the intrinsic improvements attributable to its specialized editing dataset. We utilize the BleachNick/SD3 UltraEdit w mask model variant in free-form editing mode with blank mask initialization. Evaluation is conducted

Intern3.5-VL [70] 8B - - - 0.667 0.667 Qwen3-VL [7] 8B - - - 0.000 0.000 GPT-4o [1] - - - - 0.167 0.167 GPT-4.1 [1] - - - - 0.167 0.167

AnyEdit [86] 1B 0.352 0.330 0.365 - 0.351 UltraEdit(SD3) [91] 2B 0.435 0.639 0.487 - 0.491 VAREdit-8B [53] 8B 0.630 0.591 0.504 - 0.584 Step1X-Edit v1.1 [48] 12B 0.661 0.857 0.661 - 0.700 Step1X-Edit v1.2 [48] 12B 0.543 0.587 0.470 - 0.530 FLUX.1 Kontext [42] 12B 0.557 0.861 0.626 - 0.639 Qwen-Image-Edit [76] 20B 0.587 0.630 0.565 - 0.589

OminiGen [80] 4B 0.404 0.352 0.430 - 0.401 OminiGen2 [77] 7B * 0.552 0.530 0.565 - 0.551 Ovis-U1 [67] 3B * 0.535 0.478 0.509 0.000 0.364 UniPic [68] 1.5B 0.513 0.448 0.391 - 0.463 UniPic2-SD3.5M [73] 2B 0.543 0.557 0.535 - 0.543 UniPic2-Metaquery [73] 9B 0.561 0.509 0.417 - 0.507 NextStep-1-Large [66] 15B 0.483 0.417 0.374 - 0.437 Seedream 4.0 [62] - 0.674 0.643 0.713 - 0.679 Seedream 4.0 [62] - 0.678 0.578 0.639 - 0.646 Nano Banana [20] - 0.648 0.652 0.704 - 0.666 Nano Banana [20] - 0.735 0.757 0.704 - 0.730 Bagel [22] 14B * 0.583 0.630 0.548 0.000 0.406 Bagel-Zebra [45] 14B * 0.574 0.561 0.535 0.000 0.393 + WEAVE-100k 14B * 0.582 0.612 0.512 - 0.567

- Table 9. Main results on WEAVEBench Logic Part. and denote full and partial in-context history, respectively. , , and

* indicate image-only, text-only, and combined evaluations, respectively. and represent sequential and concatenated image inputs, respectively.

with hyperparameters num inference steps = 50, image guidance scale = 1.5, guidance scale = 7.5, and negative prompt = "" to maintain consistency with our experimental protocol. Inference is performed at 512 × 512 resolution.

- • VAREdit-8B [53]: A visual autoregressive (VAR) framework for instruction-guided image editing, built upon Infinity [29]. This approach reframes image editing as a next-scale prediction problem, achieving precise image modifications through the generation of multi-scale target features. We employ the following hyperparameters: classifier-free guidance scale cfg = 3.0, temperature parameter tau = 0.1, and random seed seed = 42.
- • Step1X-Edit v1.1 [48]: Step1X-Edit leverages the image understanding capabilities of multimodal large language models (MLLMs) to parse editing instructions and generate editing tokens, which are subsequently decoded into images using a DiT-based network. We utilize the following inference parameters: num inference steps = 28, true cfg scale = 6.0, and seed = 42.

- • Step1X-Edit v1.2 [48]: An enhanced version of

Step1X-Edit featuring improved reasoning capabilities and superior performance. We employ num inference steps = 28, true cfg scale = 4.0, seed = 42, enable thinking mode = True, and enable reflection mode = False. • AnyEdit [86] is a Mixture of Experts (MoE) architecturebased image editing model, which is the result of finetuning SD-XL [57] on the AnyEdit-2.5M dataset. For our implementation, we employed the following hyperparameter configuration: utilizing the general expert, guidance scale=3, num inference steps=100, and original image guidance scale=3.

Vision-Language Models We also evaluated 2 opensource VLMs and 2 proprietary VLMs:

- • Intern3.5-VL [70] is a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series. For our implementation, we utilized max new tokens=128.

- • Qwen3-VL [7] is the most powerful vision-language

Intern3.5-VL [70] 8B - - - 0.292 0.292 Qwen3-VL [7] 8B - - - 0.250 0.250 GPT-4o [1] - - - - 0.083 0.083 GPT-4.1 [1] - - - - 0.167 0.167

AnyEdit [86] 1B 0.407 0.548 0.354 - 0.419 UltraEdit(SD3) [91] 2B 0.398 0.526 0.454 - 0.440 VAREdit-8B [53] 8B 0.581 0.698 0.498 - 0.580 Step1X-Edit v1.1 [48] 12B 0.617 0.941 0.426 - 0.625 Step1X-Edit v1.2 [48] 12B 0.567 0.681 0.476 - 0.562 FLUX.1 Kontext [42] 12B 0.578 0.907 0.465 - 0.610 Qwen-Image-Edit [76] 20B 0.667 0.802 0.446 - 0.628

OminiGen [80] 4B 0.167 0.106 0.241 - 0.177 OminiGen2 [77] 7B * 0.502 0.543 0.504 - 0.511 Ovis-U1 [67] 3B * 0.470 0.526 0.393 0.125 0.357 UniPic [68] 1.5B 0.341 0.296 0.287 - 0.316 UniPic2-SD3.5M [73] 2B 0.517 0.583 0.407 - 0.497 UniPic2-Metaquery [73] 9B 0.456 0.457 0.415 - 0.444 NextStep-1-Large [66] 15B 0.356 0.265 0.259 - 0.309 Seedream 4.0 [62] - 0.652 0.689 0.572 - 0.635 Seedream 4.0 [62] - 0.609 0.672 0.533 - 0.599 Nano Banana [20] - 0.680 0.790 0.560 - 0.666 Nano Banana [20] - 0.604 0.737 0.546 - 0.613 Bagel [22] 14B * 0.506 0.635 0.431 0.042 0.365 Bagel-Zebra [45] 14B * 0.500 0.624 0.480 0.125 0.396 + WEAVE-100k 14B * 0.503 0.754 0.430 - 0.531

- Table 10. Main results on WEAVEBench Game Part. and denote full and partial in-context history, respectively. , , and

* indicate image-only, text-only, and combined evaluations, respectively. and represent sequential and concatenated image inputs, respectively.

model in the Qwen family to date. This generation demonstrates improvements to the model across multiple areas. In our experiments, we employed max new tokens=512.

• GPT-4o [1] and GPT-4.1 [1, 13] are OpenAI’s advanced VLMs. We implemented these models with the parameter max tokens=1400.

#### C. More Related Works

Interleaved Reasoning. Large-scale corpora with interleaved text and images have become essential for pretraining VLMs with reasoning capabilities [2, 15, 18, 22, 27, 65, 95]. Inspired by human cognition, where visual counterfactuals facilitate reasoning [60], recent work has incorporated analogous interleaved reasoning mechanisms into UMMs by mapping visual inputs to symbolic representations (e.g., images or bounding boxes) [44, 74]. [83] explored pure visual reasoning relying solely on visual representations without textual modalities. Zebra-CoT [43, 45] provides an interleaved vision-language reasoning trajectory dataset to enhance UMMs’ comprehension perfor-

mance. IRG [34] generates an initial image, then iteratively refines it through reflective reasoning about quality improvements. ROVER [46] investigates the reciprocal relationship between generation and comprehension capabilities. In contrast, Weave focuses on in-context interleaved multimodal comprehension and generation.

Benchmarks for UMMs. UMM capability assessment typically encompasses three dimensions: (i) Text-to-Image: evaluated using GenEval [28] and DPGBench [32], which employ image detection methods [12] to ensure policycompliant generation, and WISE [54], which examines complex semantic understanding and world knowledge for T2I generation; (ii) Vision Comprehension: consistent with Vision-Language Model (VLM) evaluation protocols, using benchmarks including MME [10], MMBench [50], MMMU [88], MM-Vet [87], and MathVista [51]; (iii) Image Editing: assessed via GEditBench [48] and ImgEdit [85], which challenge UMMs to maintain image identity preservation while demonstrating semantic understanding. Additionally, RISEBench and KRIS-Bench evaluate reasoning with world knowledge.

These benchmarks assess generation and comprehension in isolation, whereas ROVER [46] pioneered reciprocal crossmodal reasoning for omnimodal generation, systematically evaluating intermediate processes. WEAVE represents the first benchmark to comprehensively evaluate interleaved multi-turn generation and understanding.

#### D. Additional Examples for WEAVE

###### D.1. Additional Examples for WEAVE-100k

In this appendix, we present a comprehensive collection of examples that illustrate the versatility and capabilities of our WEAVE-100k framework. Figure 17 and Figure 18 demonstrate complex editing operations that require significant reasoning capabilities. The first example showcases intricate manipulations that demand careful consideration of spatial relationships and semantic coherence, while the second example introduces human subjects into the composition.

For Multi-Image Fusion operations, we provide four illustrative examples in Figures 13–16. Figure 15 demonstrates the model’s ability to preserve footwear details during fusion operations. Figure 14 exhibits dual-task face processing capabilities. Figure 13 highlights the precise execution of specific hairstyle requirements and depicts scenarios where headphones are both held by one subject and worn by another, showcasing the model’s understanding of object interactions across multiple contexts.

The Recall capability is exemplified in Figure 19, Figure 20, and Figure 21. In the first example, the model successfully restores previously removed trousers to the subject. The second example demonstrates the model’s ability to reference a full-body model from Image #2 to reconstruct the complete body and scene in Image #4, while implementing a horizontally symmetrical background transformation. The third example shows the targeted reinsertion of a single human subject into the composition.

Additionally, we present specialized examples for Chess Game manipulation in Figure 22 and Visual JigSaw processing in Figure 23, further demonstrating the framework’s adaptability to structured visual reasoning tasks.

###### D.2. More example for WEAVEBench

This section presents the details of the examples shown in Figure 2. Figure 24 demonstrates astronomical concepts, while Figure 26 tests biological knowledge. Mathematical reasoning is evaluated in Figure 32, and physical principles are examined in Figure 35. The model’s chemistry knowledge is assessed in Figure 27, and fusion-related concepts in Figure 30. Geographic reasoning is presented in Figure 31. The model’s game understanding capabilities are tested through chess problems in Figure 28 and Minecraft scenarios in Figure 25. Optical principles are demonstrated

in Figure 34. The model’s memory and recall abilities are evaluated in Figure 36, while spatial reasoning is tested in Figure 37 and Figure 39. Finally, narrative comprehension is assessed in Figure 38, and image editing capabilities in Figure 29.

#### E. Broader Impact

The broader impact of Weave carries both potential benefits and risks upon deployment and release. Some considerations are unique due to the multimodal nature of UMMs while others reflect challenges common to image creation environments. Below, we outline risks and mitigation strategies for its release.

Hallucination. Similar to other models [6, 22, 45], our approach extends and fine-tunes text-to-image generation models to obtain unified generation capabilities, which introduces potential hallucination issues [39, 92]. Analogous to existing methods, models trained on WEAVE-100k may produce outputs that deviate from user intentions or specified input conditions. This phenomenon raises significant concerns, particularly in commercial image applications where purchasing decisions rely on accurate visual representations, given that user requirements and expression modalities exhibit inherent variability.

Biases. Despite implementing human supervision and a multi-model ensemble pipeline to mitigate biases in our synthetically generated dataset, the inherent biases from the foundation models inevitably permeate our data collection process and subsequently propagate to our fine-tuned models. This propagation can yield biased retrieval results and inequitable representations across diverse cultural contexts. Multilingual processing introduces additional bias vectors through language alignment mechanisms, as demonstrated by [17, 26].

[Figure 330]

###### Figure 13. An example of multi-image fusion in WEAVE-100k.

[Figure 331]

###### Figure 14. An example of multi-image fusion in WEAVE-100k.

[Figure 332]

###### Figure 15. An example of multi-image fusion in WEAVE-100k.

[Figure 333]

###### Figure 16. An example of multi-image fusion in WEAVE-100k.

[Figure 334]

###### Figure 17. An example of edit in WEAVE-100k.

[Figure 335]

###### Figure 18. An example of edit in WEAVE-100k.

[Figure 336]

###### Figure 19. An example of recall in WEAVE-100k.

[Figure 337]

###### Figure 20. An example of recall in WEAVE-100k.

[Figure 338]

###### Figure 21. An example of recall in WEAVE-100k.

[Figure 339]

###### Figure 22. An example of Chess Game in WEAVE-100k.

[Figure 340]

###### Figure 23. An example of Chess Game in WEAVE-100k.

[Figure 341]

###### Figure 24. An example of astronomy domain testing the model’s understanding of celestial objects and phenomena.

[Figure 342]

###### Figure 25. An example of Minecraft domain testing the model’s understanding of the game mechanics and environments.

[Figure 343]

###### Figure 26. An example of biology domain testing the model’s understanding of biological structures and processes.

[Figure 344]

###### Figure 27. An example of chemistry domain testing the model’s understanding of chemical structures and reactions.

[Figure 345]

###### Figure 28. An example of chess game analysis testing the model’s understanding of chess positions and strategies.

[Figure 346]

###### Figure 29. An example of image editing task testing the model’s ability to understand and suggest visual modifications.

[Figure 347]

###### Figure 30. An example of fusion domain testing the model’s understanding of nuclear fusion concepts and processes.

[Figure 348]

###### Figure 31. An example of geography domain testing the model’s understanding of geographical features and locations.

[Figure 349]

###### Figure 32. An example of mathematics domain testing the model’s problem-solving and reasoning abilities.

[Figure 350]

###### Figure 33. An example of maze-solving task testing the model’s pathfinding and spatial reasoning abilities.

[Figure 351]

###### Figure 34. An example of optics domain testing the model’s understanding of optical principles and phenomena.

[Figure 352]

###### Figure 35. An example of physics domain testing the model’s understanding of physical laws and principles.

[Figure 353]

###### Figure 36. An example of recall task testing the model’s memory and information retrieval capabilities.

[Figure 354]

###### Figure 37. An example of spatial reasoning task testing the model’s understanding of spatial relationships and transformations.

[Figure 355]

###### Figure 38. An example of story comprehension task testing the model’s understanding of narratives and contexts.

[Figure 356]

###### Figure 39. An example of visual jigsaw task testing the model’s ability to understand and reconstruct visual patterns.

