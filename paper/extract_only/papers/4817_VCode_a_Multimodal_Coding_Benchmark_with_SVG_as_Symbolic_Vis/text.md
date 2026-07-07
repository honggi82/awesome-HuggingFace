# arXiv:2511.02778v1[cs.CV]4Nov2025

## VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation

1Kevin Qinghong Lin∗ 2Yuhao Zheng∗ 3Hangyu Ran∗ 3Dantong Zhu 3Dongxing Mao 4Linjie Li 1Philip Torr 3Alex Jinpeng Wang

1University of Oxford 2University of Science and Technology of China 3Central South University 4Microsoft Research

Project Page: https://csu-jpg.github.io/VCode

### Abstract

Code has emerged as a precise, executable medium for reasoning and action in the agent era. Yet progress has largely focused on linguistic-centric tasks (e.g., program synthesis, debugging), leaving visual-centric coding underexplored. Conventional image representations rely on dense RGB pixels that capture appearance but provide limited symbolic abstraction. Inspired by how humans reason over sketches, we advocate SVG code as a compact, interpretable, and executable visual representation. We introduce VCode, a benchmark that reframes multimodal understanding as code generation: given an image, a model must produce SVG that preserves symbolic meaning for downstream reasoning. VCode covers three challenging domains—general commonsense (MM-Vet), professional disciplines (MMMU), and visual-centric perception (CV-Bench). To assess symbolic fidelity, we propose CodeVQA, a novel evaluation protocol in which a policy model answers questions over rendered SVG; correct answers indicate faithful symbolic preservation. Empirically, frontier VLMs struggle to generate faithful SVGs, revealing a persistent gap between language-centric and visual-centric coding. To close this gap, we introduce VCoder, an agentic framework that augments VLMs along two axes: (i) Thinking with Revision, which iteratively analyzes discrepancies and refines SVG code; and (ii) Acting with Visual Tools, where detectors and parsers supply structured cues (objects, shapes, text) beyond intrinsic model capacity. Across benchmarks, frontier VLMs with strong reasoning score well overall yet remain limited on professional knowledge and 3D reasoning; VCoder delivers a +12.3-point overall gain over the top-performing Claude-4-Opus. Human studies show that both humans and VLMs perform worse on rendered SVGs, their consistency reveals the promise of symbolic visual representation The benchmark and code is available at https://github.com/CSU-JPG/VCode.

### 1 Introduction

To advance reasoning and agentic intelligence, code has emerged as a powerful medium for interacting with digital environments [1, 2, 3]. Unlike natural language, which is free-form and descriptive, code is precise, structured, and executable—making it an effective mechanism for action. Consequently, recent benchmarks have predominantly emphasized linguistic-centric coding abilities, covering tasks such as program synthesis, debugging, and competitive programming [4, 5, 6, 7, 8], where success is measured by both correctness and executability. In the multi-modal regime, coding plays a crucial role in generating executable programs that interface with tools or environments to accomplish complex task, a paradigm that has gained particular traction in embodied agents

∗Equal contribution.

[Figure 1]

[Figure 2]

<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/sv g">

Three sheep on the farm Three sheep on the farm

<rect width="800" height="600" fill="#8B7355"/>

[Figure 3]

[Figure 4]

<rect x="0" y="0" width="800" height="300" fill="#2F4F2F"/>

[Figure 5]

[Figure 6]

<rect x="600" y="0" width="200" height="400" fill="#228B22"/>

Image (by Pixel) Coder Render Rendered Image (by Code)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

VCode

- Figure 1: Illustration of VCode. An RGB image (left, represented by pixels) is translated into symbolic SVG code (middle) via VLM as Coder and rendered back into an image (right, represented by code), aiming to preserve symbolic meaning (e.g., “three sheep on the farm”). As shown at the bottom, VCode provides a compact, interpretable, and executable representation of original images.

[2, 9, 10, 11]. A parallel line of work leverages code to generate synthetic visual assets—such as charts [12, 13], diagrams [14, 15], or websites [16, 17]—which synthesis assets, are not directly grounded in the natural visual world.

When recapping the representation for natural images, the dominant practice has been to encode them as pixels or superpixels. These representations are effective in that they densely capture visual apperance. In contrast, humans often perceive and reason through sparse symbolic sketches that emphasize spatial relationships, object counts, and structural outlines [18]. Similar to an artist drafting a rough sketch before filling in appearance details, such abstraction offers a compact yet informative scaffold for reasoning. Building on this intuition, we propose using Scalable Vector Graphics (SVG) code as an alternative visual representation, owing to its compact, interpretable, and executable nature. Thus, SVGs have long been used for icons and logos [19, 14, 20] for a general visual abstraction. This perspective motivates a fundamental question: can visual representation move beyond raw pixels and learn to represent and reason through code?

In this work, we introduce VCode, a multimodal coding benchmark that pioneers the use of SVG code as a visual representation. VCode is constructed by repurposing existing multimodal understanding benchmarks across three domains: General commonsense (MM-Vet [21]), College-level disciplines (MMMU [22]), and Visual-centric Perception (e.g., 3D depth and relationships in CVBench [23]). VCode reframes these tasks as visual coding: given an image, a model must generate SVG code that faithfully renders the image, thereby reconstructing its symbolic representation. To evaluate this transformation, we propose CodeVQA, a novel protocol in which a vision–language model must answer core questions about the original image by reasoning over the rendered SVG. This provides a principled test of whether the generated code serves as an adequate and faithful visual representation. Experiments on VCode show that existing coders remain limited in such challenging setting. We observe that coding quality improves with a model’s reasoning ability, yet models still fail to preserve fine-grained visual relations (e.g., near vs. far), exposing a persistent gap between language- and visual-centric coding.

To this end, we augment existing coders with two complementary capabilities. (i) Thinking with Revision. The model compares intermediate renderings with the original image, explicitly articulates discrepancies, and iteratively updates the SVG to improve fidelity. (ii) Acting with Visual Tools. We equip the coder with external perception toolboxes—e.g., object detectors and segmenters [24, 25]—to supply structured cues (objects, shapes, text) as coding context. Together, these strategies yield a +12.3 overall gain over the top-performing Claude-4-Opus, substantially strengthening visual-centric coding. Our contributions are threefold:

- 1. VCode: A Novel Perspective for Multimodal Coding. We recast multimodal understanding as visual-centric coding: given an image, generate SVG that preserves symbolic structure for downstream reasoning. We further present CodeVQA – a protocol that asks a VLM to answer the original-image questions using only the rendered SVG, thereby testing whether the code is an adequate and faithful visual representation.
- 2. VCoder: Augmenting VLM as Strong Multimodal Coders via (i) Thinking with Revision (iterative discrepancy analysis and SVG refinement) and (ii) Acting with Visual Tools (structured visual cues from detectors). VCoder achieves a significant overall gain over a strong baseline.

Benchmarks Domain Size Inputs Outputs Evaluation Coding HumanEval [4] Algorithm 164 Text Code Execute Pass MMCode [26] Visualization 263 Text & Img Code Execute Pass ChartMimic [12] Chart 4800 Text & Img Code Similarity Design2Code [17] Web UI 484 Text & Img Code Similarity SWE-Bench [8] GitHub 2294 Text & Code Code Execute Pass SVG-Bench [14] SVG 23K Img / Text Code Similarity Multi-modal MM-Vet [21] General 218 Img. & text Text OpenQA MMBench [27] General 3217 Img. & text Text MCQ MMMU [22] College 11.5K Img. & text Text OpenQA / MCQ MMMU-Pro [28] College 1730 Img. & text Text OpenQA / MCQ CV-Bench [23] Perception 2638 Img. & text Text MCQ VCode (Ours) G&C&P 464 Img. Code Render→VQA

- Table 1: Comparison of VCode with coding (top) and multimodal (bottom) benchmarks. VCode differs in three ways: (i) Task: models must generate code directly from natural images, without extra query guidance; (ii) Scope: focuses on natural multimodal understanding across diverse domains—General (G), College (C), and Perception (P); (iii) Evaluation: introduces CodeVQA (Render→VQA), which judges whether the rendered SVG preserves the original image’s symbolic meaning.

- 3. Evaluation and Insights. Extensive experiments expose persistent weaknesses of frontier VLMs in visual-centric coding. Human studies show consistent patterns between humans and VLMs when reasoning over rendered SVGs compared to raw images, suggesting the promise of symbolic visual coding for advancing human-like multimodal intelligence.

### 2 Related Works

#### 2.1 Coding Benchmarks

Coding in LLMs. Despite there being several coding benchmarks, most of them are initially developed for purely language coding. Representative efforts include HumanEval [4] and MBPP [5], which evaluate correctness of synthesized programs given natural language or code-level prompts. Later benchmarks such as SWE-Bench [8] extend this paradigm to real-world software engineering, requiring models to resolve issues directly in large GitHub repositories. Despite their diversity, these benchmarks are fundamentally linguistic-centric: the inputs and outputs remain in textual or code form, with success measured by pass rates or test-case execution. While effective in quantifying reasoning over program text, such settings offer little insight into multimodal capabilities.

Coding in Multi-modal. Moving beyond purely textual code, a line of work incorporates visual observations into coding tasks. Benchmarks such as Plot2Code [13],Design2Code [17], and ChartMimic [12] translate charts or UI mockups into executable plotting or layout code. MMCode [26] and SWE-Bench-MM [29] further integrate images alongside text, exploring how multimodal inputs can inform code generation. At larger scale, SVGenius [15] (generation, editing, understanding) evaluates models’ ability to produce vector-graphic code, highlighting challenges in preserving both semantics and structure. Despite this progress, most of these datasets emphasize synthetic visual assets (e.g., charts, Web UI, vector icons) as shown in Tab.1 top-half, leaving open the question of whether models can encode real-world natural images into executable visual code. This gap motivates our VCode benchmark, which repurposes multimodal understanding tasks into the visual coding with SVG.

#### 2.2 Multimodal Understanding

Various benchmarks systematically evaluate multimodal understanding. Early efforts such as MMBench [27] and MM-Vet [21] emphasize general perception and text–image reasoning. More recent benchmarks, including MMMU [22] and MMMU-Pro [28], target professional knowledge and domain-specific reasoning. However, most of these evaluations interact with models through natural

language (e.g., query or answer). In VCode, we argue that generating code to represent natural images constitutes an even more advanced form of understanding. As illustrated in Tab.1 bottom-half, unlike traditional perception tasks, this requires the model to distill an image into its core concepts and structural features by a render image, and to express them in a symbolic format that bridges perception with reasoning and action.

### 3 VCode Benchmark

#### 3.1 Task Definitions

As illustrated in Fig.1, given an input RGB image V, a vision–language model ψ is tasked with generating SVG code C that encodes the image. Rendering this code yields a rendered image V. The objective is to minimize the discrepancy between the symbolic information of the original and rendered images:

L = min I(V) − I( V) , (1)

where I(·) denotes a symbolic information representation. The central challenge, however, lies in defining an applicable measure of symbolic information, which we elaborate on below.

#### 3.2 Evaluation Metrics

The key to the evaluation prototype lies in how the correspondence between the input image V and the rendered image V is defined.

SigLIP score. To define what constitutes an ideal SVG representation, we argue that it should faithfully preserve the semantic content of the original image rather than merely matching pixel-level similarity. One way to measure this is through embedding consistency. We leverage a pretrained visual encoder f(·) such as SigLIP [30, 31] to extract embeddings for both V and V, and compute their cosine distance:

L = maxcos f(V),f( V) . (2)

CodeVQA. A more direct criterion is whether the rendered image V alone supports correct reasoning. Usually, V may even facilitate answering questions that are ambiguous or harder to resolve from the original V. Hence, the evaluation should not be constrained by the original image’s responses, but instead focus directly on the correctness of answers derived from V. We define a policy model ϕ that outputs an answer A given an image and a question Q. Then goal is formulated as

A = ϕ V,Q , L = max1[Evaluator(A)].

(3)

where 1[·] is the indicator function. Evaluator(·) is a rule-based matching in multiple-choices setting, and it can be a LLM-as-Judge in open-ending. If the answer is correct, the SVG suffices to convey the required semantics; otherwise, it reveals a gap in representational fidelity.

Code tokens length. Beyond faithful representation, we argue that an effective coder should represent an image with as few code tokens as possible, producing a concise yet faithful representation. To assess this efficiency, we evaluate the length of the generated SVG in terms of its token count |C|.

#### 3.3 Data Curation

With the evaluation prototype in place, the next step is to develop appropriate question sets Q for each associated image V. To this end, we repurpose existing multimodal visual question answering benchmarks to align with our objective. To ensure diversity in taxonomy and difficulty, we focus on three representative scenarios: (i) Commonsense perception: Assesses a model’s ability to capture everyday semantics such as spatial relationships. We adopt MM-Vet [21] as the source. (ii) Professional knowledge: Targets domain-specific, diploma-level tasks that demand both reasoning and coding skills. We incorporate the development set of MMMU [22], which spans multiple disciplines and requires deeper reasoning and expert knowledge. (iii) Visual-centric: Evaluates performance in

[Figure 14]

Question: What is the lamp on, a side table or a nightstand?

Policy model

Knowledge Generation

[Figure 15]

[Figure 16]

[Figure 17]

OCR

Spatial

Recognition

General 47.0%

Math

vs

Tech&Engineering

Vision-centric

ObjectCount

Professional 31.5%

21.6%

SpatialRelationship

Business

RelativeDistance

Health & Medicine

Humanities & Social Sci

Science

[Figure 18]

Original Image Render Image 👍 Render Image Answer: Side table Side table ✔

Art & Design

Depth Order

[Figure 19]

[Figure 20]

Nightstand ✘

[Figure 21]

Distributions of VCode.

###### Illustration of CodeVQA prototype.

- Figure 2: Left: Distributions of tasks in VCode, showing the proportions of general, professional, and vision-centric categories. Right: Illustration of the CodeVQA prototype: given an image and a question (e.g., “What is the lamp on, a side table or a nightstand?”), the policy model answers based on the rendered image. A correct answer indicates that the SVG representation preserves the semantic content of the original image, while an incorrect answer highlights room for improvement.

visually intensive settings involving counting, distance estimation, and relative spatial relationships in 2D or 3D. We draw from CV-Bench [23].

Data statistics. Following this three-pronged curation strategy, we processed each source benchmark to construct our final dataset. For (i) commonsense perception, we incorporated the entirety of MM-Vet [21], resulting in 218 image-question pairs. For (ii) professional knowledge, our curation involved filtering the MMMU [22] development set to retain only single-image VQA instances, which yielded a specialized subset of 146 pairs. Finally, for (iii) visual-centric, we created a balanced 100-pair subset from CV-Bench [23] through a stratified sampling process. This involved shuffling the data and applying interval selection to ensure a specific distribution across its sub-tasks: spatial relationship (20), object count (20), depth order (30), and relative distance (30). In total, this process yields 464 image-question pairs. The taxonomy distribution of VCode is illustrated in Fig. 2(a).

### 4 VCoder

In practice, we find that directly prompting Coders to generate SVG code from natural images remains highly challenging. This difficulty arises from three factors: (i) Long-Context Code Inputs: fully representing an image typically requires thousands of tokens; composing such long sequences demands strong code reasoning over complex elements, beyond what current Coders provide. (ii) Visually-Blind Outputs: inputs and outputs are cross-modal; because the rendered image is unseen until execution, the model must anticipate the visual consequences of code edits during generation. (iii) Weak Visual Fineness: for irregular objects (e.g., a dog’s boundary), language models struggle to capture low-level details—edges, masks, and colors—that must be encoded precisely as numeric values, even though these are fundamental to code-based representations.

To address these intertwined challenges, we propose augmenting Coders at test time with two complementary capabilities. (a) Thinking with Revision: we enhance reasoning through test-time scaling and a revision strategy that allows the model to iteratively refine its outputs, bridging the gap between long-context code generation and faithful visual rendering. (b) Acting with Vision Tools: we equip Coders with external tools that extract fine-grained visual cues—such as edges, masks, and color regions—and translate them into structured code signals, enabling models to overcome their inherent limitations in low-level perception.

#### 4.1 Thinking with Revision

Since the initial reconstruction may not always yield a satisfactory result, a natural way to enhance Coders is to let them re-examine their own outputs and iteratively refine the code. Our revision strategy follows a two-step loop: detect discrepancies between the rendered output and the target image, then update the code conditioned on these differences.

- (i) Comment the Difference. Given an intermediate rendering V(t), the coder first perceives its deviation from the original image V. Although VLMs may be limited as Coders, they are already

|[Figure 22]|
|---|

[Figure 23]

[Figure 24]

- 1. Initial Coding
- 2. Comment
- 3. Recoding

Localization with Detection

Shape with Segmentation

Text with OCR

{"Label":"man", "confidence_sco re":0.92578125, "box_xy":

[503.5, 412.1, 541.4, 465.0]],

...}

{"polygon_point s": [[516,412],[515 ,413],...], "svg_path": "M 516,412 L 515,413 ..."}

{"Studying", "points": [[94, 21], [358, 30], [356, 104], [91, 94]], ...}

Acting with Tools

<svg viewBox="0 0 602 602" xmlns="http://www.w3. org/2000/svg"> ... <text x="91" y="600" font-family="Arial, sans-serif" fontsize="60"...>Studying online</text> ... </svg>

- 1. Location-specific differences:Top-Left: -Missing: Person with money flying around;- Current: Only green rectangles floating…
- 2. Visual attribute differences:Color Issues: … ; Shape Issues: …
- 3. Specific svg revision suggestions:For Zoom Panel: - Replace green `<rect>` elements with more detailed money bill paths…

<svg viewBox="0 0 602 602" xmlns="http://www.w3. org/2000/svg"> ... <text x="91" y="600" font-family="Arial, sans-serif" fontsize="72"...>Studying online</text> ... </svg>

|[Figure 25]|
|---|

|[Figure 26]|
|---|

Access Visual Tools

Figure 3: Augmenting Coders with Test-time Revision & Visual Tools. Left: Thinking with Revision – the model performs initial coding, comments on discrepancies between original and rendered images, and iteratively refines the SVG code. Right: Acting with Vision Tools – external modules provide cues on categories, locations, shapes, colors, and text, which are translated into structured code signals to guide generation. These techniques yield more faithful and accurate renderings.

strong in visual perception. We therefore design the revision process to let them capture differences through two observations. At each iteration t, we compute a difference signal ∆(t) ← ψ V, V(t) , which quantifies the discrepancy between the reconstruction and the target.

(ii) Revise with Render Feedback. The difference signal ∆(t), together with the current code C(t) and render V(t), is provided to the coder ψ to generate revised code C(t+1). Executing this code produces an updated reconstruction V(t+1) ← V, V(t),C(t),∆(t) .

This revision loop is repeated for t = 0,1,...,T, progressively refining the reconstruction until a satisfactory visual outcome is reached. The full procedure is summarized in Algorithm 1.

- 4.2 Act with Visual Tools

Original Image Rendered Image

[Figure 27]

[Figure 28]

|[Figure 29]|
|---|

|[Figure 30]|
|---|

Original + Render Image

|[Figure 31]|
|---|

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

Commen ts

SVG Code

Original + Render Image + Old Code + Comments

Revised Image

Another limitation for Coder is capture the image fine-grained attribution such as boundary. Here we able the Coder to access additional visual tools to provide meta information to complement the generated SVG quality. We display part of tools with supple information in the right side of Fig.3.

Category. Object categories are obtained from a detector [24] and provide the Coder with essential semantic labels. For example, a detected object can be annotated in SVG with an attribute like id=’bird’. These labels serve as the basic prior for generation and are always combined with geometric cues like location or shape to describe each object more completely.

Location. A key factor in reconstruction is capturing where objects appear in the image. To provide this information, we rely on bounding boxes predicted by Florence-2 [24], expressed as absolute coordinates (x1,y1,x2,y2) together with the image width and height. These cues allow the Coder to anchor elements at the correct positions on the canvas, preserving the overall layout.

Shape. While regular geometric primitives are straightforward to express, a key challenge lies in representing irregular boundaries. To address this, we employ SAM-2 [25] to generate segmentation masks that capture detailed object contours. These masks are then downsampled into sparse coordinate points through an adaptive simplification strategy, which reduces the number of vertices while keeping the overall area nearly unchanged. The resulting polygonal paths provide the Coder with compact yet faithful shape descriptions that complement category and location cues.

Text. Text often carries critical semantic information that cannot be replaced by shapes or colors.To incorporate this, We apply OpenOCR [32] to detect and transcribe text regions, and directly encode them into SVG using the native <text> tag, which preserves both content and visual attributes without the rendering issues of pixel-based methods.

CodeVQA

Success Rate (%)

SigLIP Score

Code Token (K)

Model name

MM-Vet MMMU CV-Bench

###### Overall

Rec Ocr Know Gen Spat Math Avg. Avg. 2D 3D Avg.

Orig. Image (4o-mini) NA 100.0 NA 60.5 78.9 58.5 59.5 70.9 84.2 67.1 50.0 77.4 63.3 70.3 61.7 Claude-4.5-Sonnet 99.1 66.8 1.9 29.7 57.6 11.9 17.0 57.3 52.7 39.0 42.5 50.4 55.0 52.7 43.1 Claude-4-Opus 98.2 65.9 1.5 30.4 52.3 13.9 18.5 49.5 50.4 37.5 42.5 41.6 58.3 49.9 41.7 Claude-4-Sonnet 98.2 65.5 1.6 31.8 51.2 24.9 27.9 44.8 34.6 37.8 39.0 49.0 53.3 51.2 41.1 GPT-5 100.0 72.3 2.3 33.9 64.9 20.5 23.8 60.5 65.4 43.9 42.5 51.8 66.7 59.2 46.8 GPT-4o 100.0 60.6 0.6 23.1 58.4 12.7 17.0 51.3 60.4 35.0 44.5 29.3 50.0 39.6 39.0 GPT-o3 100.0 64.1 1.1 31.3 55.2 17.7 19.7 48.5 61.5 39.8 39.0 47.4 56.7 52.1 42.2 GPT-4.1 100.0 68.6 1.6 30.8 62.0 15.5 20.4 56.0 55.8 40.9 44.5 48.2 66.7 57.4 45.6 GPT-4o-mini 100.0 61.1 0.4 20.7 58.4 13.2 18.9 46.8 63.5 33.5 44.5 27.7 48.3 38.0 37.9 Gemini-2.5-Pro 100.0 66.5 2.4 28.9 57.8 20.0 22.9 47.9 68.5 39.1 45.2 56.1 56.7 56.4 44.7 Gemini-2.5-Flash 98.0 63.7 1.9 29.3 56.7 17.4 21.1 46.3 53.8 39.1 39.7 48.8 58.3 53.6 42.4 Seed-1.6-Thinking 100.0 62.8 1.6 18.9 46.5 8.1 11.9 44.1 38.5 28.7 43.2 45.3 51.7 48.5 37.5 Llama-4-Scout-17B-16E 100.0 55.5 0.7 18.2 44.9 12.4 15.5 32.8 46.2 26.4 42.5 35.0 53.3 44.2 35.3 Qwen3-VL-235B-A22B 95.1 58.1 1.7 19.3 54.6 8.8 14.5 45.6 53.1 31.1 41.1 22.6 58.3 40.5 36.3 Qwen2.5-VL-72B 98.7 57.9 0.3 20.6 52.9 14.0 17.3 51.3 43.1 31.8 41.1 21.9 53.3 37.6 36.0 Qwen2.5-VL-7B 70.6 22.9 0.6 4.9 6.0 3.0 4.0 7.1 3.8 4.8 19.2 17.5 41.7 29.6 14.7 InternVL3.5-241B-A28B 100.0 60.2 1.0 20.4 52.4 11.9 15.7 39.2 42.3 31.1 43.8 45.3 50.0 47.6 38.7 Intern-S1 100.0 60.0 1.0 24.7 56.8 12.1 16.0 51.2 41.9 35.2 41.1 46.8 55.0 50.9 40.4 InternVL3-78B 100.0 57.7 0.7 16.9 52.7 8.3 13.9 40.5 55.0 29.1 41.8 18.3 50.0 34.1 34.2 MiniCPM-V-4.5 78.9 45.9 0.9 11.8 31.8 4.5 10.8 23.2 26.5 17.7 36.3 23.4 45.0 34.2 27.1 GLM-4.5V 99.8 63.8 1.6 22.4 54.4 7.1 15.6 46.0 56.9 33.1 40.4 43.1 66.7 54.9 40.1 GLM-4.1V-Thinking 100.0 61.7 1.2 21.1 52.0 10.4 13.7 44.8 58.8 31.9 43.2 37.9 56.7 47.3 38.8 OmniSVG 100.0 46.2 5.3 9.2 15.3 3.7 10.4 16.9 11.5 9.4 43.8 24.8 40.0 32.4 25.2 StarVector 8.3 18.1 1.3 0.0 3.4 0.0 1.6 4.4 0.0 1.5 6.8 0.0 0.0 0.0 2.8 VCoder (Claude-4-Opus) 99.3 71.0 2.0 46.6 63.4 38.8 41.5 58.1 72.7 54.2+16.7 48.6+6.2 57.7 65.0 61.3+11.4 54.0+12.3

- Table 2: Main results on VCode across various top-performing frontier VLM coders. Top half is the proprietary models, while the bottom half is the open-source model. The best scores are in bold while the second best are in underline. The Overall score is calculated as an instance-weighted average of the three subtasks (MM-Vet, MMMU, and CV-Bench) using their respective question counts.

### 5 Experiments

#### 5.1 Baseline and Settings

To comprehensively evaluate our proposed framework, we compare it against a wide range of proprietary and open-source models that represent the current state of the art in multimodal reasoning and code generation. Proprietary models, such as Claude-4.5-Sonnet, Claude-4-Opus and Claude4-Sonnet, GPT-5, GPT-4.1, GPT-o3, GPT-4o, and GPT-4o-mini [33, 34], as well as Gemini-2.5Pro and Gemini-2.5-Flash [35], and Seed-1.6-thinking [36]. These models are widely recognized for their strong reasoning and multimodal capabilities, and thus provide competitive upper baselines for our benchmark. Open-source models: including LLaMA-4-Scout, Qwen3-VL, Qwen2.5VL-72B and Qwen2.5-VL-7B [37], InternVL3.5-241B-A28B [38], Intern-S1, InternVL3-78B [39], MiniCPM-V-4.5 [40], GLM-4.5V and GLM-4.1V-Thinking [41], OmniSVG [20] and StarVector [14]. These baselines cover a diverse spectrum of model sizes and training paradigms, enabling a comparison between proprietary and open-source approaches.

Evaluation settings. Unless otherwise noted, all models are queried under a unified prompting interface with identical inputs to ensure fairness. The primary automatic evaluator is GPT-4o-mini, which provides consistent judgments across benchmarks.

#### 5.2 Main Results

In Tab. 2, we evaluate full baselines on VCode, reporting per-domain results—general, college, and vision-centric—and the overall average. We have the following observation.

Stronger reasoning yield better visual coding scores. Closed-source models consistently outperform open-source counterparts across benchmarks. GPT-5 sets the strongest baseline with the top SigLIP score (72.3) and the highest CodeVQA overall (46.8), showing robust performance on both similarity and reasoning metrics. This pattern indicates that stronger reasoning ability translates into better VCode performance—i.e., models that reason well produce more faithful symbolic renderings. We also observe a positive correlation between semantic similarity (SigLIP) and CodeVQA.

Challenges across different dimensions. (i) Best performer still trails the original-image upper bound. Even the best SVG result—GPT-5 at 46.8—remains well below the raw-image upper bound (61.7), indicating substantial headroom. This confirms that the task is sufficiently challenging and that symbolic representation still has ample room for improvement. (ii) SVG specialist underperforms. OmniSVG and StarVector-8B ranks last due to the low success rate for long context outputs, highlighting VCode’s difficulty and the gap between neatly authored SVG corpora and SVGs de-

rived from natural images. (iii) Knowledge is hardest. In MM-Vet, the Knowledge dimension is consistently the lowest, reflecting the compounded challenge of recalling facts and then encoding them faithfully in SVG (e.g., historical entities). (iv) Professional disciplines are hard to differentiate. On MMMU, models cluster within a narrow, modest band, and most fail the more demanding disciplinary settings. (v) Vision-centric perception is tough. CV-Bench scores hover near the low (randomly by 50%), especially on 3D relations (depth or spatial). Even with VCoder, improvements are meaningful but leave substantial headroom.

Absolute gains with VCoder. Built on Claude-4-Opus, VCoder lifts Overall from 41.7 to 54.0 (+12.3) via revision and vision-tool assistance, improving all three domains—demonstrating an effective enhancement for visual-centric coding.

Code token length is highly correlated with expressiveness. Models that emit short SVGs underperform (e.g., 0.3K by Qwen-2.5-VL). By contrast, stronger models (GPT-5, Gemini-2.5-Pro) produce substantially longer sequences (often >2K tokens) and attain higher scores. Length is not sufficient on its own, but performance scales with usable context, highlighting long-context reasoning and generation as a central bottleneck for visual-centric coding.

#### 5.3 Key Ablations

Effects of Vision tools. Ablations in Tab. 3 reveal three takeaways: (i) Adding fine-grained cues (location, category, shape) yields steady gains; shape is especially helpful for spatial reasoning (Spat.), even without large changes in SigLIP, indicating structural benefits. (ii) Text cues help, with the full visual-tool ensemble provides the largest overall improvement. (iii) Together, all vision tools yield a 16.6-point improvement over Claude-4-Opus, implying the strong potential of the Coder itself to autonomously call tools and leverage contextual information for code generation.

90

Evaluator

SigLIP Score

CodeVQA–MMVet Rec Ocr Know Gen Spat Math Avg.

Human

Variant

80

75.5

72.4

GPT-4o-mini

70

67.1

Claude-4-Opus

63.3

GLM-4.5V

Claude-4-Opus 65.6 30.4 52.3 13.9 18.5 49.5 50.4 37.5 +Loc. & C. 70.8 29.7 60.3 17.5 22.9 54.9 46.2 39.7 +Loc. & C. & S. 71.5 33.4 62.7 19.3 25.1 63.1 64.2 43.3 +Text 69.9 30.4 59.5 19.2 21.5 56.8 65.4 41.5 +Full vision tools 71.6 46.0 64.4 40.8 43.0 61.6 72.7 54.1

60

55.8

54.1

CodeVQA

50.4

50

40.6

40

30

20

10

- Table 3: Effects by vision tools modules, where Loc. denotes Location, C. denotes Category, and S. denotes Shape.

0

Orig. Image SVG by VCoder

Figure 4: Effects by different policy during evaluation

Effects across Policies and Human studies. Fig. 4 shows the performance differences across policy ϕ, including humans. On the original images, all models substantially surpass human perception and reasoning (50.4 for humans vs. 75.5 for GLM-4.5V). When evaluated on SVG representations, all models exhibit a noticeable performance drop, even with the human score decreasing to 40.6. Interestingly, both humans and VLMs exhibit a form of alignment when interpreting symbolic representations. Although abstraction inevitably leads to information loss compared to original visual inputs, VLMs demonstrate a comparable ability to reason from such high-level representations, suggesting that their potential in this aspect is on par with human understanding.

Effects by Revision. In Fig. 5, we examine the impact of our revision strategy. Both Claude and GLM-4.5V benefit from the first revision, with GLM-4.5V showing the most substantial gainslikely due to its built-in “thinking mode,” which excels at difference analysis and refinement. In contrast, GPT-4o initially struggles during the first revision but continues to improve in later rounds, implying that effective revision relies on a strong reasoning foundation.

Effects by Visual v.s. Textual query. In Tab. 4, we examine the impact of input modality. Using raw images (i.e., Img2SVG) gives the weakest results, suggesting that current coders are poorly adapted to direct visual input. Notably, even with deep thinking enabled (i.e., Img2SVG-Thinking), performance remains low, underscoring the difficulty of visual-centric coding and the gap between language-driven and vision-driven code generation. By contrast, translating images into linguistic captions before coding (i.e., Img2Text2SVG) achieves the best performance, highlighting the benefit of language as an intermediate representation.

Variant SigLIP Code CodeVQA

Score Token MM-Vet MMMU CV-Bench Overall

Img2SVG 65.6 1.5K 37.5 43.2 52.3 42.5 Img2SVG-Thinking 69.8 1.6K 38.2 42.5 53.7 43.5 Img2Text2SVG 68.5 1.8K 43.0 43.2 55.6 46.4

#### Table 4: Effects by different input modes of Claude-4-Opus

- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40 Model

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

Claude-4-Opus

GLM-4.5V

GPT-4o

CodeVQA

0 1 2

Revision Rounds

#### Figure 5: Effect of revision round.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Original image Initial rendered w. visual tools w. revision

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

MMMU’s example Rendered by VCoder CV-Bench’s example Rendered by VCoder

Figure 6: Qualitative examples from VCode. Top row (a–d): an internet meme rendered progressively by initial decoding, visual-tool assistance, and revision. Bottom row: challenge samples from MMMU (ArtTheory) and CV-Bench (3D), alongside their SVG renderings by VCoder.

#### 5.4 Qualitative Analysis

Fig. 6 presents qualitative results by comparing origina image and the rendered image by VCoder. Top row. Across the four stages, the initial decoding misses layout and semantics. Adding visual tools recovers key geometry (e.g., the starfish character’s triangular body and facial features), while revision corrects fine details (character proportions, text alignment, spacing), yielding a rendering that closely matches the meme’s structure and intent. Bottom row. VCoder produces SVGs that are both more faithful to the source and more interpretable for downstream reasoning. The left example (MMMU) is knowledge-intensive: accurately depicting a multi-panel historical relief requires domain cues and fine structural abstraction, where base models often collapse detail. The right example (CV-Bench) is vision-centric: success hinges on visually grounded prompts that localize and size objects correctly (e.g., monitor in front of keyboard, receding rows of chairs), after which revision tightens residual misalignments. These examples underscore the challenges by VCode.

### 6 Conclusion

We introduced VCode, offering a new perspective on multimodal coding by benchmarking multimodal understanding with SVG as a visual representation, along with CodeVQA to assess symbolic fidelity through QA over rendered SVGs. Our study shows that frontier VLMs struggle to produce faithful SVGs despite strong linguistic reasoning, revealing a persistent gap between language- and vision-centric coding. To address this, we proposed VCoder, which integrates Test-time Revision and Acting with Visual Tools, yielding substantial improvements. Human studies underscore the potential of symbolic visual coding as a pathway toward more human-aligned multimodal intelligence. Future work can focus on developing end-to-end vision–language coders with scalable training data to enable more faithful symbolic representations.

### References

- [1] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.
- [2] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
- [3] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023.
- [4] Mark Chen and Jerry Tworek et.al. Evaluating large language models trained on code. 2021.
- [5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- [6] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- [7] Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, Shengyan Liu, Di Luo, Yutao Ma, Hao Tong, Kha Trinh, Chenyu Tian, Zihan Wang, Bohao Wu, Yanyu Xiong, Shengzhu Yin, Minhui Zhu, Kilian Lieret, Yanxin Lu, Genglin Liu, Yufeng Du, Tianhua Tao, Ofir Press, Jamie Callan, Eliu Huerta, and Hao Peng. Scicode: A research coding benchmark curated by scientists, 2024.
- [8] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.
- [9] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. arXiv preprint arXiv:2209.07753, 2022.
- [10] Kevin Qinghong Lin, Linjie Li, Difei Gao, Qinchen WU, Mingyi Yan, Zhengyuan Yang, Lijuan Wang, and Mike Zheng Shou. Videogui: A benchmark for gui automation from instructional videos. Advances in Neural Information Processing Systems, 37:69329–69360, 2024.
- [11] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Stan Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19498–19508, 2025.
- [12] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024.
- [13] Chengyue Wu, Yixiao Ge, Qiushan Guo, Jiahao Wang, Zhixuan Liang, Zeyu Lu, Ying Shan, and Ping Luo. Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots, 2024.
- [14] Juan A. Rodriguez, Abhay Puri, Shubham Agarwal, Issam H. Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16175–16186, June 2025.

- [15] Siqi Chen, Xinyu Dong, Haolei Xu, Xingyu Wu, Fei Tang, Hang Zhang, Yuchen Yan, Linjuan Wu, Wenqi Zhang, Guiyang Hou, Yongliang Shen, Weiming Lu, and Yueting Zhuang. Svgenius: Benchmarking llms in svg understanding, editing and generation, 2025.
- [16] Tony Beltramelli. pix2code: Generating code from a graphical user interface screenshot. In Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems, EICS ’18, New York, NY, USA, 2018. Association for Computing Machinery.
- [17] Chenglei Si, Yanzhe Zhang, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering?, 2024.
- [18] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348– 139379, 2024.
- [19] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Trans. Graph., 42(6), December 2023.
- [20] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Jiaxu Zhang, Liao Wang, Gang Yu, Xinjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. arXiv preprint arxiv:2504.06263, 2025.
- [21] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International conference on machine learning. PMLR, 2024.
- [22] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.
- [23] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024.
- [24] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. arXiv preprint arXiv:2311.06242, 2023.
- [25] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2024.
- [26] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multimodal code large language models with visually rich programming problems, 2024.
- [27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [28] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of ACL, 2025.
- [29] John Yang, Carlos E Jimenez, Alex L Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R Narasimhan, et al. Swe-bench multimodal: Do ai systems generalize to visual software domains? arXiv preprint arXiv:2410.03859, 2024.

- [30] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [31] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier H´enaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features, 2025.
- [32] Yongkun Du, Zhineng Chen, Hongtao Xie, Caiyan Jia, and Yu-Gang Jiang. Svtrv2: Ctc beats encoder-decoder models in scene text recognition. In ICCV, 2025.
- [33] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [34] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [35] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [36] ByteDance Seed Team. Seed-oss open-source models. https://github.com/ ByteDance-Seed/seed-oss, 2025.
- [37] Qwen Team. Qwen2.5-vl, January 2025.
- [38] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [39] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [40] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. Nat Commun 16, 5509 (2025), 2025.
- [41] V Team. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025.

- A Implement Details 13
- B Experiments Ablations 14

- B.1 Effects by SigLip v.s. DINO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- B.2 Effect by revision round on MM-Vet . . . . . . . . . . . . . . . . . . . . . . . . . 14
- B.3 Effects by different policy during evaluation on MM-Vet . . . . . . . . . . . . . . 14

- C Prompt Template 14

- C.1 Img2SVG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- C.2 Img2Text2SVG . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.3 Img2SVG-Thinking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.4 Visual Tools . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.5 Revision . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- D More Visualizations 18

- D.1 VCoder vs. Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- D.1.1 MM-VET . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- D.1.2 MMMU . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.1.3 CV-Bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.2 VCoder Individual Components . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

### A Implement Details

We implement our model using the PyTorch framework on an NVIDIA RTX 4090 GPU with 24GB of memory. The maximum output length is set to 16,384 tokens, while for the Qwen2.5-VL models we use 8,192 tokens.

For evaluation, different protocols are used depending on the benchmark. In MM-Vet, we employ gpt-4-0613 as the evaluator to score model responses. In CV-Bench and MMMU experiments, we adopt a rule-based string matching parser to determine correctness.

For SigLip2, we use the siglip2-so400m-patch14-384. The token cost reported in our tables is measured using the tiktoken library with the cl100k base encoding.

It is worth noting that in the img2svg experiments, StarVector cannot take textual prompts as input. It directly performs image-to-SVG generation.

Algorithm 1 Test-time Revision

- 1: Input: Coder ψ, an image V, initial rendering V(0), initial SVG code C(0), iteration number T
- 2: Output: Refined rendering V(T)
- 3: for t = 0 → (T − 1) do
- 4: Comment the difference: ∆(t) ← ψ V, V(t)
- 5: Generate revised SVG code: C(t+1) ← ψ(V, V(t), ∆(t), C(t))
- 6: Update reconstruction: V(t+1) ← Render(C(t+1))
- 7: end for
- 8: return V(T)

### B Experiments Ablations

- B.1 Effects by SigLip v.s. DINO

Metric Claude-4.5-Sonnet Claude-4-Opus Claude-4-Sonnet GPT-5 GPT-4o GPT-o3 GPT-4.1 GPT-4o-mini Gemini-2.5-Pro Gemini-2.5-Flash Seed-1.6-Thinking VCoder

SigLip2 66.9 67.2 66.9 70.1 60.1 64.9 66.9 58.6 66.4 64.3 62.6 72.3 DINO-V2 26.2 26.1 24.2 30.4 16.5 22.4 26.0 14.3 27.2 22.5 20.2 33.0

Table 5: Effect by different feature extractor. For each metric, the best results are highlighted in bold, and the second-best results are underlined. As shown, the DINO reach lower score compare with SigLip2, as it more focus on low-level visual representation. While SigLip2 focus on more on semantic space.

- B.2 Effect by revision round on MM-Vet

Models Round Rec Ocr Know Gen Spat Math Avg.

Claude-4-Opus

- 0 30.4 52.3 13.9 18.5 49.5 50.4 37.5

- 1 29.0 54.3 18.9 21.7 56.0 53.1 38.8

- 2 29.6 54.0 16.9 14.5 53.1 55.4 39.5

GLM-4.5V

- 0 22.4 54.4 7.1 15.6 46.0 56.9 33.1

- 1 26.5 58.3 14.5 20.0 54.4 50.0 37.4

- 2 24.5 65.7 10.4 15.6 53.3 55.8 38.3

GPT-4o

- 0 23.1 58.4 12.7 17.0 51.3 60.4 35.0

- 1 23.7 53.5 12.0 15.7 46.5 53.5 34.1

- 2 25.0 60.0 14.2 18.9 50.7 56.9 36.3

Table 6: Effect by revision (round) on MM-Vet. For each model, the best results are highlighted in bold, and the second-best results are underlined.

- B.3 Effects by different policy during evaluation on MM-Vet

Setting Evaluator Rec Ocr Know Gen Spat Math Avg.

GPT-4o-mini 60.5 78.9 58.5 59.5 70.9 84.2 67.1 Human 40.8 67.6 20.4 21.5 69.5 74.8 50.4 Claude-4-Opus 68.1 79.3 59.0 57.9 82.1 72.7 72.4 GLM-4.5V 67.4 87.1 56.5 60.0 80.0 96.2 75.5

Ori

GPT-4o-mini 46.0 64.4 40.8 43.0 61.6 72.7 54.1 Human 30.5 55.8 14.4 15.6 59.8 61.4 40.6 Claude-4-Opus 43.7 76.3 37.0 41.2 68.4 76.5 55.8 GLM-4.5V 54.3 73.6 48.2 49.0 73.6 84.2 63.3

VCoder

Table 7: Evaluation results of different evaluators on Ori vs VCoder. For each setting (Ori and VCoder), the best results are highlighted in bold, and the second-best results are underlined.

### C Prompt Template

- C.1 Img2SVG Img2SVG

Convert this image to SVG code following these rules: CRITICAL REQUIREMENTS:

- – Output only pure SVG code, no markdown blocks or explanations.
- – Start with <svg viewBox="..." xmlns="http://www.w3.org/2000/svg"> and end with </svg>.
- – Use only native SVG elements (no external images or links).

- – Include viewBox to ensure all elements are visible and auto-scale properly.
- – Calculate appropriate viewBox dimensions to contain all content with some padding. Generate the SVG now.

#### C.2 Img2Text2SVG

- Img2Text2SVG – Stage 1: Image Captioning Please provide a detailed and accurate description of this image. Focus on:

- 1. Main objects, shapes, and elements
- 2. Colors, textures, and visual properties
- 3. Spatial relationships and positioning
- 4. Style and artistic characteristics
- 5. Any text, symbols, or specific details

Be precise and comprehensive. This description will be used to recreate the image as an SVG. Include geometric details, proportions, and layout information that would be necessary for accurate reproduction.

- Img2Text2SVG – Stage 2: SVG Generation

Based on the following description, generate clean and accurate SVG code: {description} CRITICAL REQUIREMENTS:

- 1. Output ONLY complete SVG code, no explanations or other text.
- 2. Use appropriate dimensions (e.g., viewBox="0 0 400 400" or similar).
- 3. Include all elements described with accurate colors, shapes, and positioning.
- 4. Use clean, well-structured SVG syntax.
- 5. Ensure the SVG is self-contained and complete.
- 6. Start with <svg viewBox="..." xmlns="http://www.w3.org/2000/svg"> and end with </svg>.
- 7. Use precise geometric shapes and paths where appropriate.
- 8. Match colors and proportions as closely as possible to the description. Generate the SVG now.

#### C.3 Img2SVG-Thinking Img2SVG – Thinking

Let’s analyze this image and create an SVG representation through a structured thinking process. Step-by-step analysis:

- 1. Visual Decomposition

- – What are the main visual elements?
- – What geometric shapes can be identified?
- – What are the key colors and their relationships?

- 2. Structural Analysis

- – How are elements arranged and layered?
- – What are the proportions and spatial relationships?
- – Are there any repeating patterns or symmetry?

- 3. SVG Implementation Strategy

- – Which SVG elements best represent each component?
- – What’s the optimal drawing order?
- – How to handle complex shapes and gradients?

- 4. Technical Considerations

- – What viewport dimensions are appropriate?
- – How to ensure scalability and responsiveness?
- – What optimizations can be applied? After your analysis, provide:

- 1. Your complete reasoning process
- 2. The final SVG code implementation Requirements for SVG output:

- – Must be complete and self-contained.
- – Include all necessary attributes and elements.
- – Start with <svg tag and end with </svg>.
- – Use appropriate viewBox and dimensions. Please proceed with the analysis and generation.

#### C.4 Visual Tools Visual Tools – System Prompt

You are a helpful assistant that converts images into clean, complete SVG vector graphics. Your primary task is img2svg conversion for Visual Question Answering. You have access to two types of metadata to assist with precision: METADATA AVAILABLE:

- – OCR metadata: Text regions with precise 4-point quadrilaterals for accurate text placement.
- – Object detection metadata: Object boundaries with labels, confidence scores, and svg path outlines.

SPECIAL CASE HANDLING (Hint Strategy): Sometimes, an image may depict a person, character, or artwork where fine details like facial features or texture could be lost during vectorization. Examples include:

- – A recognizable public figure such as a scientist or political leader
- – A well-known fictional character from popular culture
- – A famous painting or portrait by a specific artist If the subject in the image is of this nature and important identity cues might be lost:
- – Preserve recognizability by including visual hints such as characteristic clothing, accessories, environment, or symbolic elements.
- – When confident, you may add a <text> element near the subject that provides their commonly known name, the name of the associated work or series, or the title/creator of an artwork. If the subject does not fit these examples or is not clearly recognizable:
- – Generate a clean SVG with no extra text labels.
- – Focus on accurate shapes, proportions, and composition. METADATA INTEGRATION:

- 1) Text rendering: Use OCR quadrilaterals as authoritative coordinates for text placement. Render literal text strings with appropriate transforms for rotation/skew.
- 2) Object boundaries: Use detection svg paths as authoritative contours. Infer fill/stroke colors and add internal details within these boundaries.

- 3) Background reconstruction: Fill in unlabeled regions using native SVG primitives. PROCESSING PRIORITY:

- 1. Use provided metadata for precise positioning (OCR quads, detection paths).
- 2. Apply hint strategy for recognizable subjects.
- 3. Reconstruct missing background/unlabeled areas.
- 4. Ensure proper layering and visual completeness. OUTPUT REQUIREMENTS:

- – Output only pure SVG code, no markdown blocks or explanations.
- – Start with <svg viewBox="..." xmlns="http://www.w3.org/2000/svg"> and end with </svg>.

- – Use only native SVG elements (no external images or links).
- – Include viewBox to ensure all elements are visible and auto-scale properly.
- – Do not include explanations or commentary.

This SVG will be used in a Visual Question Answering task, so ensure the output retains as much semantic identity as possible when visual details are reduced.

Visual Tools – User Prompt Image dimensions: {W}x{H} METADATA: {metadata json}

Generate the complete SVG with precise metadata integration and appropriate hint strategy for recognizable subjects.

- C.5 Revision

- Revision – Stage 1: Visual Difference Analysis

Compare the original image (first) with the SVG-rendered image (second) and identify specific differences for SVG code revision. Focus on identifying:

- 1. LOCATION-SPECIFIC DIFFERENCES:

- – Which areas or regions differ (top-left, center, bottom-right, etc.).
- – Missing or extra elements in specific positions.

- 2. VISUAL ATTRIBUTE DIFFERENCES:

- – Color mismatches (specify which elements and what colors).
- – Shape distortions (which shapes are wrong and how).
- – Size or proportion issues (which elements are too big or too small).
- – Position or alignment problems.

- 3. SPECIFIC SVG REVISION SUGGESTIONS:

- – Which SVG elements need modification (circle, path, rect, etc.).
- – What attributes to change (fill, stroke, cx, cy, width, height, d, etc.).
- – Specific color values or coordinate adjustments needed. Format your response as actionable SVG revision instructions.

- Revision – Stage 2: SVG Code Correction

You are an SVG code specialist. Based on the visual analysis and comparison between the original image and the current SVG rendering, make specific code modifications to fix identified issues.

VISUAL ANALYSIS FINDINGS: {optimization goals} CURRENT SVG CODE: {current svg code} INSTRUCTIONS:

- 1. Analyze the current SVG code structure and elements.
- 2. Based on the visual analysis findings, identify which specific SVG elements need modification.
- 3. Make precise changes to fix the identified issues:

- – Adjust colors (fill, stroke attributes).

- – Fix shapes and paths (modify d attributes, coordinates).
- – Correct sizes and positions (width, height, cx, cy, x, y).
- – Add missing elements or remove incorrect ones.

- 4. Output only the complete revised SVG code.
- 5. Ensure all modifications directly address the issues mentioned in the analysis.
- 6. Start with <svg and end with </svg>. Revised SVG code:

### D More Visualizations

#### D.1 VCoder vs. Baselines

In this section, we present qualitative comparisons between VCoder and baseline models on representative examples from three benchmarks. For each case, we display: (a) the original reference image, (b) the output generated by VCoder, and (c–d) the visual results produced by the two strongest baseline models. These comparisons clearly illustrate VCoder’s superior ability to faithfully interpret and reconstruct visual content while preserving semantic consistency with the reference images.

#### D.1.1 MM-VET

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Find the pattern of how the ”X” operator is redefined, and answer the given equation in the image. Answer: 13

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: What should we add in the third step? Answer: milk

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Can you give a short introduction to this painting? Answer: The Starry Night is an oil-on-canvas painting by the Dutch Post-Impressionist painter Vincent van Gogh. Painted in June 1889, it depicts the view from the east-facing window of his asylum room at Saint-R´emy-de-Provence, just before sunrise, with the addition of an imaginary

village.It has been in the permanent collection of the Museum of Modern Art in New York City since 1941, acquired through the Lillie P. Bliss Bequest. Widely regarded as Van Gogh’s magnum opus, The Starry Night is one of the most recognizable paintings in Western art.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1 Question: On the right desk, what is to the left of the laptop? Answer: table lamp/desk lamp

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: What is the name of this landmark? Answer: Trevi Fountain

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Can you give a short introduction to this movie? Answer: The Godfather is a 1972 American crime film[2] directed by Francis Ford Coppola, who co-wrote the screenplay with Mario Puzo, based on Puzo’s best-selling 1969 novel of the same title. The film stars Marlon Brando, Al Pacino, James Caan, Richard Castellano, Robert Duvall, Sterling Hayden, John Marley, Richard Conte, and Diane Keaton. It is the first installment in The Godfather trilogy, chronicling the Corleone family under patriarch Vito Corleone (Brando) from 1945 to 1955. It focuses on the transformation of his youngest son, Michael Corleone (Pacino), from reluctant family outsider to ruthless mafia boss.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Can you give a short introduction to this painting? Answer: Along the River During the Qingming Festival (Qingming Shanghe Tu) is a handscroll painting by the Song dynasty painter Zhang Zeduan (1085–1145) and copied many times in the following centuries. It captures the daily life of people and the landscape of the capital, Bianjing (present-day Kaifeng) during the Northern Song. The theme is often said to celebrate the festive spirit and worldly commotion at the Qingming Festival, rather than the holiday’s ceremonial

aspects, such as tomb sweeping and prayers. Read right to left, as a viewer unrolled it, successive scenes reveal the lifestyle of all levels of the society from rich to poor as well as economic activities in rural areas and the city, and offer glimpses of period clothing and architecture. The painting is considered to be the most renowned work among all Chinese paintings, and it has been called ”China’s Mona Lisa.”

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Is the man going to fall down? Answer: no

#### D.1.2 MMMU

[Figure 75]

[Figure 76]

(a) Original image (b) VCoder

[Figure 77]

[Figure 78]

(c) Gemini-2.5-Pro (d) GPT-4o

Question: For your independent research, you transferred lymphocyte populations between syngeneic mice. You irradiated recipients first to ablate (get rid of) existing lymphocytes, then transferred defined cell populations from donors of same genetic background. The result is shown in . What does this experiment tell us? (A) Both B cells and T cells can produce antibodies. (B) Both B cells and T cells have memory functions. (C) Both B cells and T cells are required for an antibody response. (D) B cells are required for an antibody response in the absence of T cells. (E) B cells and T cells are co-localized and produce synergetic effects in bone marrow and thymus. Answer with the option’s letter from the given choices directly. Answer: C

[Figure 79]

[Figure 80]

(a) Original image (b) VCoder

[Figure 81]

[Figure 82]

(c) Gemini-2.5-Pro (d) GPT-4o

Question: Calculate the area of the zero circle with the following data:Assume that the tracing arm of the planimeter was so set that one revolution of the measuring wheel measures 100 cm2 on the paper. Answer the question using a single word or phrase. Answer: 1970.6

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: For 2015, calculate the cash flow from assets(1) , cash flow to creditors(2) , and cash flow to stockholders(3) . (A) 1): -$493.02 (2):-$2,384 (3):$1,890.98 (B) 1): $1843.98 (2):-$2,384 (3):$493.02 (C) 1): -$493.02 (2):-$2,384 (3):-$1,890.98 Answer with the option’s letter from the given choices directly. Answer: C

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: Which of the following correctly describes the reception stage of this signal transduction pathway? (A) epinephrine binds to a g-protein coupled receptor protein present in the cell membrane (B) the g protein changes shape, is activated, activates adenyl cyclase, which activates cAMP, which activates protein kinases (C) protein kinases phosphoylate molecules (D) glycogen synthesis is inhibited and glycogen breakdown is promoted Answer with the option’s letter from the given choices directly. Answer: A

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: The painting on the right focuses on the (A) contribution of Native Americans to landscape preservation (B) implementation of the Homestead Act (C) impact of the gold rush on landscape development (D) idea of Manifest Destiny Answer with the option’s letter from the given choices directly. Answer: D

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: Both works come from which art-historical period? (A) Baroque (B) Renaissance (C) Rococo (D) Classical Answer with the option’s letter from the given choices directly. Answer: B

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: Refer to the figure, which term best describes the practice where students take on the role of television or newspaper reporters and interview characters from the book to retell an event from a range of perspectives? (A) News Program (B) Readers Theatre (C) Hot Seat (D) News Answer with the option’s letter from the given choices directly. Answer: A

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

(a) Original image (b) VCoder (c) Gemini-2.5-Pro (d) GPT-4o

Question: Refer to the description , which type of irony is depicted when a person says or writes one thing and means another, or uses words to convey a meaning opposite to the literal meaning? (A) verbal irony (B) situational irony (C) foreshadowing (D) dramatic irony Answer with the option’s letter from the given choices directly. Answer: A

#### D.1.3 CV-Bench

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: How many persons are in the image? Select from the following choices. (A) 2 (B) 3 (C) 0 (D) 1 Answer with the option’s letter from the given choices directly. Answer: D

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: How many dogs are in the image? Select from the following choices. (A) 1 (B) 3 (C) 2 (D) 0 Answer with the option’s letter from the given choices directly. Answer: A

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Considering the relative positions of the bottle and the wine glass in the image provided, where is the bottle located with respect to the wine glass? Select from the following choices. (A) left (B) right Answer with the option’s letter from the given choices directly. Answer: A

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1 Question: Considering the relative positions of the sheep and the horse in the image provided, where is the sheep located with respect to the horse? Select from (A) left (B) right Answer with the option’s letter from the given choices directly. Answer: B

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Which object is closer to the camera taking this photo, the table (highlighted by a red box) or the bookcase (highlighted by a blue box)? (A) table (B) bookcase Answer with the option’s letter from the given choices directly. Answer: A

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Estimate the real-world distances between objects in this image. Which object is closer to the traffic cone (highlighted by a red box), the trailer (highlighted by a blue box) or the bus (highlighted by a green box)? (A) trailer (B) bus Answer with the option’s letter from the given choices directly. Answer: A

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Which object is closer to the camera taking this photo, the towel (highlighted by a red box) or the faucet (highlighted by a blue box)? (A) towel (B) faucet Answer with the option’s letter from the given choices directly. Answer: B

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

(a) Original image (b) VCoder (c) GPT-5 (d) GPT-4.1

Question: Estimate the real-world distances between objects in this image. Which object is closer to the clothes (highlighted by a red box), the lamp (highlighted by a blue box) or the towel (highlighted by a green box)? (A) lamp (B) towel Answer with the option’s letter from the given choices directly. Answer: B

#### D.2 VCoder Individual Components

In this section, we present ablation studies visualizing the contribution of individual components in VCoder. For each example, we show: (a) the original reference image, (b) the initial rendered output without any refinement, (c) the output after applying visual tools, and (d) the final output after the revision module. These progressive visualizations demonstrate how each component incrementally improves the quality and accuracy of the generated images.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is d in the last equation? Answer: 1.25 / 54.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is the answer to the second equation on the right? Answer: 12

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: The diagram below shows how the Australian Bureau of Meteorology collects up-to-the-minute information on the weather in order to produce reliable forecasts. Write a report for a university lecturer describing the information shown below. Write at least 150 words. Answer: The figure illustrates the process used by the Australian Bureau of Meteorology to forecast the weather. There are four stages in the process, beginning with the collection of information about the weather. This information is then analysed, prepared for presentation, and finally broadcast to the public. Looking at the first and second stages of the process, there are three ways of collecting weather data and three ways of analysing it. Firstly, incoming information can be received by satellite and presented for analysis as a satellite photo. The same data can also be passed to a radar station and presented on a radar screen or synoptic chart. Secondly, incoming information may be collected directly by radar and analysed on a radar screen or synoptic chart. Finally, drifting buoys also receive data which can be shown on a synoptic chart. At the third stage of the process, the weather broadcast is prepared on computers. Finally, it is delivered to the public on television, on the radio, or as a recorded telephone announcement.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What should I do before cutting herbs, sausage, and mushrooms? Answer: milk

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What should kids do after snap fingers? Answer: hop on one foot

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is the index of the step when we need to add all purpose flour? Answer: third / 3

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is the name of this landmark? Answer: Anbariya Mosque

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is the name of this landmark? Answer: baochu pagoda

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: Can you give a short introduction to this painting?

Answer: Girl With A Pearl Earring (Dutch: Meisje met de parel) is an oil painting by Dutch Golden Age painter Johannes Vermeer, dated c. 1665. Going by various names over the centuries, it became known by its present title towards the end of the 20th century after the earring worn by the girl portrayed there. The work has been in the collection of the Mauritshuis in The Hague since 1902 and has been the subject of various literary and cinematic treatments..

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: What is located to the right of the shampoo? Answer: conditioner

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: Is the curtain on the right side or on the left of the picture? Answer: left

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

(a) Original image (b) Initial rendered (c) w. visual tools (d) w. revision

Question: what is the green logo on the car? Answer: monster.

