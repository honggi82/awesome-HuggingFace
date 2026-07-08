## CodePercept: Code-Grounded Visual STEM Perception for MLLMs

# arXiv:2603.10757v2[cs.CV]20Jun2026

Tongkun Guan1, Zhibo Yang2†, Jianqiang Wan2, Mingkun Yang2, Zhengtao Guo3, Zijian Hu1, Ruilin Luo4, Ruizhe Chen5, Songtao Jiang5, Peng Wang2†, Wei Shen1( ), Junyang Lin2, Xiaokang Yang1( ) 1 MoE Key Lab of Artificial Intelligence, AI Institute, School of Computer Science, Shanghai Jiao Tong University 2 Qwen Team 3 Beijing Institute of Technology 4 Tsinghua University 5 Zhejiang University †Project leader. Corresponding Author.

gtk0615@sjtu.edu.cn

### Abstract

When MLLMs fail at visual reasoning, a fundamental question arises: is it due to perceptual deficiencies or reasoning limitations? Through systematic scaling analysis that independently scales perception and reasoning components, we uncover a critical insight: scaling perception consistently outperforms scaling reasoning. This reveals perception as the true lever limiting current STEM visual reasoning. Motivated by this insight, our work focuses on systematically enhancing the perception capabilities of MLLMs by establishing code as a powerful perceptual medium—executable code provides precise semantics that naturally align with the structured nature of STEM visuals. Specifically, we construct ICC-1M, a large-scale dataset comprising 1M Image-Caption-Code triplets that materializes this code-as-perception paradigm through two complementary approaches: (1) Code-Grounded Caption Generation treats executable code as ground truth for image captions, eliminating the hallucinations inherent in existing knowledge distillation methods; (2) STEM Image-toCode Translation prompts models to generate reconstruction code, mitigating the ambiguity of natural language for perception enhancement. To validate this paradigm, we further introduce STEM2Code-Eval, a novel benchmark that directly evaluates visual perception in STEM domains. Unlike existing work relying on problem-solving accuracy as a proxy that only measures problem-relevant understanding, our benchmark requires comprehensive visual comprehension through executable code generation for image reconstruction, providing deterministic and verifiable assessment. Code is available at https://github.com/

†Project leader. Corresponding Author. Acknowledgements: This work was supported by the NSFC under

Grant 62322604 and 62576207.

[Figure 1]

Figure 1. The scaling analysis reveals perception as the bottleneck in STEM. We decouple visual STEM reasoning into perception (image-to-caption) and reasoning (caption-to-answer) stages, then independently scale each component while keeping the other constant. All components use Qwen3-VL-Thinking [36] models and evaluation in the representative MathVision Dataset [47]. Both experiments demonstrate that scaling perception consistently outperforms scaling reasoning. This finding motivates our focus on systematically enhancing MLLMs’ perception capabilities in STEM.

TongkunGuan/Qwen-CodePercept.

### 1. Introduction

Recent breakthroughs in reinforcement learning have triggered an “Aha moment” [19, 38] for Large Language Models (LLMs), inspiring extensive research efforts to replicate this success in multimodal domains [3, 23, 25, 42– 44, 50, 54]. This trend is particularly prominent in Science, Technology, Engineering, and Mathematics (STEM), where researchers employ staged learning and sophisticated reward mechanisms to unlock stronger cross-modal reasoning of MLLMs. However, a fundamental question remains: What is the true bottleneck limiting MLLMs in STEM, and when models fail, is it due to perception deficiencies or reasoning limitations?

To answer this question, we conduct a systematic scaling analysis by decoupling the task into two stages: visual perception (generating image descriptions) and reasoning (solving problems based solely on textual descriptions). We independently scale perception and reasoning capabilities while holding the other fixed, as illustrated in Fig. 1. The results demonstrate that expanding perception capacity consistently yields greater performance gains than expanding reasoning capacity. This empirical evidence illuminates a critical insight: perception is the true lever that unlocks current STEM visual reasoning.

Motivated by this insight, we focus this work on systematically enhancing the perception capabilities of MLLMs for STEM domains. A seemingly intuitive solution would be to enhance STEM perception through knowledge distillation, leveraging advanced MLLMs like GPT-series or Gemini-series to generate descriptive captions for training. However, this path encounters critical limitations. First, these teacher models are prone to hallucination, producing factually incorrect descriptions, particularly regarding spatial positioning, quantitative relationships, and element interactions. Second, and more critically, many STEM images exhibit what we term descriptive aphasia: their complex spatial relationships and precise numerical values cannot be fully or accurately captured by natural language alone. For instance, precisely describing auxiliary line constructions in complex polyhedral geometry remains inherently challenging for natural language.

Furthermore, the field lacks a direct paradigm for evaluating visual perception capabilities in STEM domains. Existing research [30] predominantly relies on problemsolving accuracy as a proxy for evaluating perceptual ability, yet this metric only reflects the model’s capacity to understand problem-relevant information rather than measuring true comprehensive visual comprehension.

These limitations motivate a paradigm shift: grounding perception in executable code. We argue that requiring MLLMs to generate executable Python code for image reconstruction offers the most rigorous validation of perceptual capabilities. The principle is straightforward yet powerful: accurate image reproduction is possible only when a model achieves complete visual comprehension.

Building on this principle, we introduce STEM2CodeEval: a manually annotated benchmark of 1,000 images that challenges models to generate executable Python code for image reconstruction, providing deterministic and verifiable assessment of visual perception. STEM2Code-Eval draws images from established STEM benchmarks [28, 34, 47, 53, 64, 66], encompassing diverse domains including mathematics, physics, chemistry, and electrical engineering. We employ a rigorous three-stage pipeline combining code agent generation, candidate selection, and human annotation to ensure code quality.

Beyond benchmarking, we further propose code as a powerful medium to significantly enhance MLLMs’ visual perception capabilities through two Code-Grounded tasks: 1) Code-Grounded Caption Generation, which leverages executable code as ground truth for generating image captions, effectively eliminating AI-generated description errors; and 2) STEM Image-to-Code Translation, which directly trains models to generate executable reconstruction code, removing the inherent ambiguity of natural language descriptions. To enable effective training for these CodeGrounded tasks, we construct ICC-1M: a large-scale training dataset containing over 1M high-quality STEM ImageCaption-Code pairs. Our data construction employs a synthesis strategy with three pipelines: 1) executable Python code to image reproduction, 2) principled diversification that extracts STEM concepts from seed images and reinstantiates them across diverse visual contexts while preserving conceptual validity, and 3) specialized solid geometry synthesis to address the fundamental limitations of current MLLMs in generating solid geometry codes. Through rigorous three-stage quality control, ICC-1M provides a robust foundation for training models to enhance visual perception and executable code generation in STEM domains. In summary, our contributions are threefold:

- • We identify perception is the primary bottleneck in STEM visual reasoning through rigorous scaling analysis;
- • We introduce STEM2Code-Eval, a manually curated benchmark that establishes code generation as a verifiable ground truth for evaluating visual perception in STEM;
- • We construct ICC-1M to propose two Code-Grounded training tasks that systematically enhance perceptual capabilities. Experiments demonstrate the effectiveness and prove that code is an alternative to caption.

### 2. Related Work

MLLMs for STEM Current MLLMs have predominantly focused on enhancing reasoning capabilities to address STEM-related challenges [2–5, 17, 18, 20, 24, 29, 40, 42– 44, 48–51, 54]. Recent advances in this direction can be categorized into three main methods: 1) Cold-start thinking data curation, where researchers meticulously design high-quality seed datasets that provide reasoning patterns [8, 21, 32, 46]; 2) RL-based methods, which employ reinforcement learning with carefully designed reward mechanisms [31, 58, 60, 63] to iteratively improve reasoning performance; 3) Unimodal thinking data transfer, where several studies [6, 52, 54] demonstrate that highquality text-only reasoning data can substantially enhance the reasoning capabilities of MLLMs when appropriately integrated. Despite these significant advances in reasoning, our scaling analysis reveals that visual perception remains the primary bottleneck in STEM field. However, existing research has largely overlooked the fundamental role, with

few works explicitly addressing perceptual deficiencies. To bridge this gap, we introduce code as a verifiable ground truth for comprehensive visual understanding and developing systematic training methodologies to enhance perceptual capabilities through executable code.

STEM Visual Perception Evaluation Evaluating visual perception capabilities in STEM domains presents unique challenges for MLLMs. This includes accurately perceiving element types, quantities, structures, relationships, and underlying principles [9, 11–16]. A recent work [30] adopt a two-stage evaluation paradigm where image captioning is followed by LLM-based problem solving to isolate perceptual capabilities. However, this approach only measures problem-relevant information extraction rather than comprehensive visual understanding, potentially overlooking critical visual details that are irrelevant to specific questions but essential for complete perception. More importantly, different from domain-specific (UI [10, 27, 41, 62], Chart [56], SVG [26, 35, 57]) code generation tasks that primarily target downstream applications, we develop STEM Image-Code pairs serves dual purposes: (1) establish a rigorous benchmark for evaluating comprehensive visual perception, and (2) enable the construction of high-quality image-code-caption triplets for perception enhancement.

### 3. CodePercept Methodology

As illustrated in Fig.2, we construct native STEM imagecode pairs using the matplotlib library [45] to support subsequent code-grounded caption generation and STEM imageto-code translation tasks. We then enhance the model’s perceptual capabilities by the supervised fine-tuning stage and reinforcement learning stage. For clarity, the following sections describe the complete CodePercept pipeline.

#### 3.1. Image-Code Pair Construction

Given a image I, we leverage MLLMs to generate corresponding reproduction code c and render it to produce x. While MLLMs may not achieve perfect reconstruction (i.e., x ̸= I), we observe that the generated image-code pairs (x,c), after rigorous validation, exhibit high consistency between visual output and code semantics, as well as strong STEM correctness in representing the core STEM concepts.

Building on this insight, we design a scalable data engine that expands a seed dataset X of STEM-focused public training images into a large-scale image-code dataset through three parallel pipelines (Fig.2, Part 01):

{(x,c)|c ∈ FIR(X)∪FID(X)∪FSG(Tgeo),x ∼ R(c)} (1)

where FIR, FID, and FSG represent image reproduction, image diversity, and solid geometry synthesis pipelines, Tgeo refers to a collection of solid geometric templates. R(c) denotes executing the code to render image x.

- 1) Image Reproduction FIR(·): For each I ∈ X, we decompose the task into two stages to generate the code:

c = Gcode(I,Gcaption(I)) (2) where Gcaption prompts a MLLM to extract a rich textual description from the STEM image. The code generation Gcode then prompts the MLLM to produce executable Python code c conditioned on both the image and generated caption. This allows the prompted MLLM to first understand visual content explicitly before generate code, improving both accuracy and interpretability. However, this strategy remains inherently constrained by the diversity of source images in existing datasets, motivating our diversification approach.

- 2) Image Diversity FID(·) operates on a fundamental insight: principles underlying STEM images can be systematically abstracted and re-instantiated across different contexts while preserving conceptual validity. This allows us to distill advanced MLLMs’ language-to-code capabilities to render more novel images beyond existing datasets. Formally, for each I ∈ X, we generate the diversity codes by a compositional abstraction-instantiation process:

[c1,c2,...,cK] = Gcode(I,Gprinciple(I)) (3) where Gprinciple prompts a MLLM to extract the underlying scientific principle from I. Gcode prompts the MLLM to generate K diverse code variations based on that principle. For instance, from a seed image depicting domino-based logic puzzles, our pipeline generates circular domino wheel pattern, triangular domino combination arrangement, ladybug spot matrix, grid connection graph, etc., each maintaining STEM rigor while introducing structural novelty.

- 3) Solid Geometry Synthesis FSG(Tgeo). Both the reproduction and diversity pipelines face a fundamental limitation when generating code for solid geometry, as this task requires a precise configuration of spatial relationships—a capability where current LLMs and MLLMs are notably deficient. To address this gap, we construct a code templatebased synthesis pipeline that generates solid geometry images. Formally, we define a collection of code templates1

Tgeo = {c˜i}Mi=1, where each template c˜i defines the generation logic for a specific geometric type and is parameterized

by a parameter space Θi to control attributes such as cube arrangements, viewing angles, and spatial configurations. The synthesis process is formulated:

Cgeo = {ci | ci = c˜i(θ);i ∈ 1,...,M;θ ∈ Θi} (4) where we instantiates each template c˜i with parameters θ sampled from Θi to produce code ci. This ensures geometric correctness through structured templates while achieving visual diversity through systematic parameter sampling.

1These templates cover typical solid geometry scenarios, including cube net unfolding and folding sequences, orthographic three-view projection and reconstruction, cross-sectional analysis of solids, cube stacking configurations, combinations of various geometries, polyhedral constructions, spatial curve visualization, and surface integral representations.

[Figure 2]

[Figure 3]

Solid Geometry Templates

Reinforcement Learning

[Figure 4]

[Figure 5]

STEM Public Data Native Caption

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

```python ``` match. otherwise

1.0 0.0

[Figure 12]

Please observe the image carefully and provide a detailed description.

Format Reward

rfmt

=

Your description must provide a non visual AI model with a complete, un-

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Cross-section Analysis

ambiguous set of visual facts to enable it to reason. ... Now, Overall Structure; Coordinate System; Exact Count; Detailed Layout

Cube Unfolding

Cube folding

Cube Stacking

rcode rimage+ rcode+ rexec

Execution failed.

Of course! This is a black and white image displaying three rectangular, domino... From left to right: ... This represents a "1-3" domino tile ... , This represents a "2-9" domino tile,

rcnt

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

=

Content Reward

Execution succeed.

... This represents a "0-12" tile. The three tiles are positioned next to each other ...

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

0 ≤ rcode ≤ 1; 0 ≤ rimage ≤ 1; rexec = 1

Three-View Projection

Combined Solids

Spatial Curves 3D Surfaces

Total Reward

[Figure 25]

r = rfmt + rcnt

[Figure 26]

[Figure 27]

Template selection Parameter spaces

GRPO

Output

Native Image-Code Pairs

[Figure 28]

[Figure 29]

[Figure 30]

Image Reproduce Image Diversity

Solid Geometry

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

C0

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Supervised Finetuning

Reproduce Code

[Figure 40]

X2 X3 X4

C1 X1

class DominoPuzzle: def __init__(self): ... ... ... ... ... ... def draw_domino(self, x, y, top_dots, ....):

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Image2Caption Prompt

Image2Code Prompt

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

.... dominoes = [1, 2, 0, 3, 9, 12] puzzle.draw_puzzle(dominoes)

C2

....

Rendering

[Figure 52]

Vision Encoder

Large Language Model (LLM)

[Figure 53]

X5 X6 X7 Xn

....

[Figure 54]

[Figure 55]

[Figure 56]

Rendering

[Figure 57]

Cn

[Figure 58]

[Figure 59]

[Figure 60]

....

[Figure 61]

[Figure 62]

This is black and white image depicts three rectangular domino tiles .... From left to right: ... This represents a "1-3"...."2-9"...."0-12" domino tile. These three domino tiles are closely positioned next to each other ....

X0

class DominoPuzzle: def __init__(self): def draw_domino(self, x, y,....):

[Figure 63]

.... dominoes = [1, 2, 0, 3, 9, 12] puzzle.draw_puzzle(dominoes)

Role playing

[Figure 64]

STEM Image-to-Code Translation

Code-Grounded Caption Generation

[Figure 65]

[Figure 66]

[Figure 67]

cdraft = Gcode (X4)

tdraft = Gcaption (X4) (Native Caption) Code C4

import matplotlib.pyplot as plt ... ... def draw_cells(self, x=8, y=8): ... ... def draw_dots(self, dots): ... ... def draw_arrows(self, martix): ... ... plt.subplots( figsize=(8, 8)) dots=[[2,1,4,5,6,3,6,1],...,[...,3,4,1,4]] matrix = [(1,1), (2,3),...,(6,2),(8,1)]...

Excellent! As a Python expert specializing in matplotlib,...Below is a breakdown steps to reproduce...followed by...executable code.

Of course. Here is a detailed description of the image...206 dots...68 arrows... Coordinate System: ... the cell (1,1) locates in bottom-left, (8,8)... top-right ... 4.1 Black Dots (Nodes) Row 8 (Top):

Caption

- 1. Drawing Grid: First, we set up a 6x6 grid. A function...handles...
- 2. Drawing Dots: ...A dict stores the precise numbers...the top row...
- 3. Defining the Path: ...The path Arrow_Path consists of 68 red arrows...start point at (1,2)...end point at (8,6)...top-right corner cell. The following code reproduces the complex structure: ```python ``` Arrow_Path defines the path of all red arrows...You can adjust ...

- Cell (1, 8): 1 node, located at the center.
- Cell (2, 8): 3 nodes, arranged horizontally in the upper half of the cell. ... ... ... 4.3 Red Arrows (Connections) 1: From (1, 2) to (2, 3). 2: From (2, 3) to (2, 1). 3: From (2, 1) to (5, 4). ... ... 67: From (5, 7) to (4, 2). 67: From (4, 2) to (7, 7). 68: From (7, 7) to (3, 5). ... ..

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

An Image-Code Example

cnew = Grefine(cdraft, c4)

tcode=Ganalyze(C4)

tnew = Grefine (tdraft, tcode)

[Figure 73]

Excellent! As a Python expert specializing in matplotlib,...Below is a breakdown steps to reproduce...followed by...executable code.

Of course. Here is a detailed description of the image...216 dots...63 arrows... Coordinate System: ... the cell (1,1) locates in bottom-left, (8,8)... top-right ...

Code Execution Tracer

ICC-1M

[Figure 74]

- 1. Drawing Grid: First, we set up a 8x8 grid. A function...handles...
- 2. Drawing Dots: ...A dict stores the precise numbers...the top row...
- 3. Defining the Path: ...The path consists of 63 red arrows...start point at (1,1)...end point at (8,1)...bottom-right corner cell. The following code reproduces the complex structure: ```python {Code C4}``` dots defines the path of all red arrows...matrix defines... You can adjust ... ...

[Figure 75]

logs Checklist

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

###### 4.1 Black Dots (Nodes) Row 8 (Top):

[Figure 80]

[Figure 81]

- Cell (1, 8): 2 node, located at "top-left" and "bottom-right."
- Cell (2, 8): 1 node, located at the center. ... ... ...

[Figure 82]

X4

[Figure 83]

4.3 Red Arrows (Connections) 1: From (1, 1) to (2, 3). 2: From (2, 3) to (1, 5). 3: From (1, 5) to (2, 7). ... ... 67: From (7, 3) to (5, 4). 67: From (5, 4) to (6, 2). 68: From (6, 2) to (8, 1). ... ..

[Figure 84]

[Figure 85]

Image Code

[Figure 86]

Figure 2. The overview pipeline of CodePercept that enhances MLLMs’ visual perception in STEM domains through code-grounded learning. (Part 01) Starting from public STEM data, we construct high-quality image-code pairs via three complementary pipelines: (1) Image Reproduce converts existing STEM images into executable Python codes, (2) Image Diversity extracts concepts from seed images and generates diverse instantiations while preserving semantic validity, and (3) Solid Geometry employs parametric templates to generate complex solid geometry images with corresponding codes, addressing MLLMs’ solid geometry limitations. (Part 02) The synthesized data enables two novel training tasks (Code-Grounded Caption Generation and STEM Image-to-Code Translation) that fundamentally shift how we approach visual perception. (Part 03) These processes culminate in ICC-1M, a dataset of over 1M curated image-caption-code triplets. We employ both Supervised Finetuning and Reinforcement Learning to train models that achieve robust visual perception capabilities.

4) Unified Quality Control. Despite our carefully designed engines, we cannot guarantee optimal quality for all generated pairs. We therefore develop a composite filtering strategy that retains only those pairs satisfying quality criteria:

D = {(x,c) | (x,c) ∈ QI(X) ∧ QC(C) ∧ QIC(X,C)} (5) where QI for image quality, QC for code quality, and QIC for image-code consistency. X and C represent the collection of images and code gathered from the previous three pipelines, respectively. All metrics are evaluated using a SOTA MLLM with specialized prompts.

#### 3.2. Code-Grounded Caption Generation

While the quality-controlled image-code pairs provide precise structural representations, accurate natural language descriptions remain essential for comprehensive MLLM training (both validated as complementary modalities in our experiments). Specifically, natural language captions provide semantic understanding while code serves as structured captions with precise spatial and quantitative details.

However, generating semantically accurate captions for

complex STEM images presents a fundamentally challenge. Intricate STEM relationships, precise spatial configurations, and abundant quantitative details are all prone to hallucination in direct vision-to-text generation. To solve this problem, we introduce a novel framework that leverages ground-truth code c as a reliable medium for producing both accurate and linguistically natural descriptions tnew:

tnew = Grefine (Gcaption(x),Ganalyze(c,ξ(c))) (6) where Gcaption first generates a linguistically natural but potentially inaccurate draft tdraft, Ganalyze then extracts verified visual facts from code, and Grefine finally synthesizes factually precise captions while preserving linguistic naturalness. (Fig. 2, Part 02, Left side)

- 1) Native Caption. We begin by obtaining a description draft tdraft = Gcaption(x) by prompting a MLLM to describe the image directly. tdraft exhibits natural language flow but captures factual inaccuracies regarding quantitative details, spatial relationships, and STEM properties.
- 2) Code Analysis. To extract reliable visual facts, we leverage the generation code c itself as ground truth—since it directly specifies rendered content, it naturally contains

definitive information about all visual primitives. However, directly analyzing complex code with deep recursion, nested loops, or intricate transformations remains prohibitively difficult for LLMs. To address this, we augment code analysis with execution-based verification:

tcode = Ganalyze(c,ξ(c)) = LLM(c,ξ(c),Panalyze) (7) where Panalyze denotes specialized prompts that guide the LLM to extract visual facts from both the code structure itself and its execution logs ξ(c). The execution tracer ξ(c) functions as an external instruction manual of the code, which captures a structured checklist of logs about all visual elements rendered during code execution.

Specifically, the execution tracer operates by executing the code in a controlled environment and systematically records geometric precision (exact coordinates, dimensions, spatial relationships), quantitative attributes (definitive counts, RGB specifications), rendering semantics (zorder layering, transformation matrices), and STEM mappings between parameters and visual manifestations. When confronted with complex algorithmic logic, this tracer serves as a definitive reference that resolves ambiguities about spatial orientations, exact quantities, and precise visual attributes. By combining the code structure with the tracer’s output, the LLM gains complete and verified information about the image content, thereby eliminating uncertainties inherent in analyzing code in isolation.

3) Code-Grounded Caption. A factuality-preserving rewriting that keeps the complementary strengths in 1) and 2):

tnew = Grefine(tdraft,tcode) = LLM(tdraft,tcode,Prefine) (8) where Prefine instructs an LLM to perform surgical edits that systematically correct factual errors (incorrect numbers, positions, colors, or geometric relationships), replace vague quantifiers with exact counts, and supplement omissions of perceptually salient content. Crucially, these edits maintain the original syntactic structures, language style, and descriptive flow of tdraft while substituting verified visual facts from tcode, ensuring that the final captions achieve both STEM rigor and pedagogical accessibility.

#### 3.3. STEM Image-to-Code Translation

Beyond natural language, code offers a fundamentally different modality for visual description. It defines visual content as programming constructs, capturing geometric relationships, mathematical constraints, and structural details that natural language descriptions often leave ambiguous or incomplete. However, providing ground-truth code alone is insufficient for effective learning. Models need explicit guidance to learn visual-to-code mappings: mapping observed features to code segments and understanding how parameters govern visual properties. To enable this, we construct the explanatory-style code as cnew = Grefine(Gcode(x),c), where cnew represents the final explana-

tory code with code correctness and instructional richness. (Fig. 2, Part 02, Right side)

- 1) Explanatory Draft Generation. We first prompt a MLLM

to generate code directly from the image: cdraft = Gcode(x). This draft naturally exhibits desirable pedagogical patterns including step-by-step breakdown, explicit parameter choices, and instructional commentary explaining visual-tocode mappings. However, without access to ground-truth specifications, cdraft frequently contains factual errors in coordinates, dimensions, loop logic, or algorithmic structure, particularly for complex STEM images.

- 2) Code-Grounded Refinement. To correct these errors while preserving explanatory richness, we leverage the verified ground-truth code c as a reliable reference:

cnew = Grefine(cdraft,c) = LLM(cdraft,c,Prefine) (9) where Prefine instructs an LLM to adaptively refine the explanatory content and replace erroneous code with c, while carefully maintaining the original explanatory structure. This refinement preserves the logical flow, step-by-step explanations, and contextual descriptions of cdraft while ensuring code correctness through alignment with c.

Explanatory image-code pairs provide rich training signals that teach models not only what code to generate, but why specific implementations best capture content, ensuring both technical precision and instructional effectiveness.

#### 3.4. Post-Training Strategy

Building on these strategies (Sec. 3.2 and 3.3), we formalize our training data as image-caption-code triplets:

Dtrain = {(x(i),t(newi) ,c(newi) )}Ni=1 (10) where x(i) denotes our generated STEM image, t(newi) represents the code-grounded accurate caption, and c(newi) signifies the explanatory reproduction code. These three modalities constitute semantically equivalent representations of the same underlying STEM concept, providing complementary supervision signals for enhanced perceptual training. We leverage these triplets to train our model through a twostage paradigm: supervised finetuning followed by reinforcement learning.

Stage 1: Supervised Finetuning (CodePercept-S1) Using the Qwen3-VL series [36] as our base architecture, we jointly optimize two tasks: image captioning (x(i),t(newi) ) and image-to-code translation (x(i),c(newi) ). This joint training strategy provides benefits beyond training each task separately, as natural language captions t(newi) help the model understand visual content semantically, building strong visual understanding before generating code. Simultaneously, the explanatory code c(newi) acts as a structured caption that encodes visual information through executable programming constructs, providing precise spatial and quantitative details that complement natural language descriptions.

Table 1. Evaluating the perception abilities of various MLLMs using a captioner-solver setup. Each MLLM (captioner) generates an image description, and a fixed LLM solver then performs problem-solving based solely on that description. More accurate descriptions are expected to yield better reasoning results.

Benchmark Datasets (%)

Image Captioner

Average MathVision MathVista MathVerse DynaMath WeMath LogicVista

###### LLM Solver: Qwen3-30A3-Thinking [55]

Claude-Opus 4.1-Thinking [1] 59.61 71.10 56.19 73.25 44.86 59.28 60.72 GPT5-Thinking [33] 60.03 65.20 69.56 71.00 54.57 53.02 62.23 Gemini2.5-Pro 66.80 74.80 73.47 81.42 60.29 66.44 70.53

KeyeVL1.5-8B [43] 54.11 64.90 49.95 62.37 33.62 45.19 51.69 Intern-S1-8B [3] 51.67 65.70 51.90 63.61 33.43 51.23 52.92 GLM-4.1V-9B [20] 53.75 64.60 54.47 66.17 40.76 51.00 55.13 InternVL3.5-8B [50] 53.32 67.70 53.40 68.12 41.05 51.68 55.88 MiniCPM-V-4.5 [44] 53.15 66.60 57.84 65.44 43.71 52.57 56.55 Qwen2.5-VL-72B [4] 54.14 67.50 55.40 68.28 44.86 52.34 57.09 Qwen3-VL-30A3B-Instruct [36] 53.59 68.00 66.44 71.67 46.10 53.69 59.92 Qwen3-VL-235A22B-Instruct [36] 60.43 73.80 70.08 77.39 53.05 59.73 65.75

Qwen3-VL-4B-Instruct [36] 54.21 67.30 64.59 69.40 46.10 54.14 59.29 CodePercept-4B-S1 57.63+3.4 69.60+2.3 65.59+1.0 71.38+2.0 47.81+1.7 60.40+6.3 62.07+2.8 Qwen3-VL-8B-Instruct [36] 54.37 69.60 63.75 72.19 45.43 56.82 60.36 CodePercept-8B-S1 59.31+5.0 70.20+0.6 66.52+2.8 73.20+1.0 49.14+3.7 61.52+4.7 63.32+3.0 CodePercept-8B-S1-PAMI 61.74 72.60 69.79 76.85 51.90 66.44 66.55 Qwen3-VL-32B-Instruct [36] 58.55 72.20 71.09 75.78 48.00 62.19 64.63

CodePercept-32B-S1 62.27+3.7 72.90+0.7 71.70+0.6 77.41+1.6 54.19+6.2 65.33+3.1 67.30+2.7 Caption-32B-S1-PAMI 64.57+3.7 73.50+0.7 73.10+0.6 78.90+1.6 56.10+6.2 66.89+3.1 68.84+2.7

LLM Solver: Qwen3-235A22-Thinking [55]

Qwen3-VL-4B-Instruct [36] 59.80 69.20 66.39 71.22 48.86 56.82 62.05 CodePercept-4B-S1 64.71+4.9 71.30+2.1 66.73+0.3 72.40+1.2 50.00+1.1 64.65+7.8 64.97+2.9 Qwen3-VL-8B-Instruct [36] 59.67 71.00 63.88 73.69 49.14 58.16 62.59 CodePercept-8B-S1 66.45+6.8 71.40+0.4 67.95+4.1 75.05+1.4 52.29+3.2 62.64+4.5 65.96+3.4 Qwen3-VL-32B-Instruct [36] 62.66 74.00 69.90 75.54 56.48 66.44 67.50 CodePercept-32B-S1 69.96+7.3 75.90+1.9 73.56+3.6 79.50+4.0 57.81+1.3 70.02+3.6 71.13+3.6

By interleaving these supervision signals, our S1 model learns to bridge visual perception, linguistic understanding, and symbolic code generation within a unified representation space. This allows captions to provide semantic context for code generation, while code generation improves caption accuracy through its precise and executable nature. Stage 2: Reinforcement Learning (CodePercept-R1) We strategically apply reinforcement learning exclusively to code generation, as creating accurate, executable code is more challenging than natural language. Code requires strict syntax and logic, where minor errors can cause failures. Importantly, it provides inherently verifiable reward signals through executability and similarity metrics, making it well-suited for RL optimization. Building upon the SFT-trained model, we employ GRPO [19, 38] with two defined rewards (Fig.2) to further improve code quality:

– Format Reward (rfmt): A regular expression is used to validate that the generated code is encapsulated within a ‘‘‘python ‘‘‘ block, assigning a binary reward of

1.0 for a valid format and 0.0 otherwise.

– Content Reward (rcnt): (1) Execution reward rexec assigns 1 if the code executes successfully, otherwise 0;

- (2) Code-level Reward (rcode): A formatted score from GPT-4o [22] that assesses the semantic equivalence between the generated code and the ground-truth code c;
- (3) Image-level Reward (rimage): A formatted score from GPT-4o that evaluates the visual similarity between the original image x and the rendered output (only available when code executes succeeded, otherwise 0).

The overall reward is therefore defined as: r = rfmt + rcnt. Consider the standard GRPO approach, it samples a group of generated output set {o1,o2,...,oG} for each query q ∈ Dtrain (including question and image) from policy model πθ

. Then GRPO maximizes the following objective and

old

optimizes the model πθ:

G

πθ(oi|q) πθ

1 G

J (θ) = E{o

min

Ai,

i}Gi=1∼πθold(O|q)

(oi|q)

old

i=1

πθ(oi|q) πθ

clip

,1 − ϵ,1 + ϵ Ai − βDKL(πθ||πref)

(oi|q)

old

where ϵ and β are the PPO clipping hyper-parameter and the coefficient controlling the Kullback–Leibler (KL) penalty [19, 37], respectively. Specifically, for a group of G outputs {o1,...,oG} sampled from the same input q, the advantage is calculated as:

ri − mean(r1,r2,...,rG) std(r1,r2,...,rG)

(11)

Ai =

Through this reinforcement learning stage, the model is encouraged to generate not only syntactically correct and executable code, but also semantically accurate code that faithfully reconstructs the visual content.

### 4. STEM2Code-Eval Benchmark

Why STEM2Code-Eval? Existing benchmarks [28, 47, 64] evaluate MLLMs through end-task problem-solving accuracy, which combine perceptual understanding with reasoning capabilities in STEM. When models fail, we cannot determine whether the failure stems from perceptual deficiencies or reasoning limitations. While a recent work [30] adopt a two-stage evaluation paradigm (image captioning followed by LLM solving) to isolate the perception capabilities of MLLMs, yet this metric only reflects the model’s capacity to understand problem-relevant information rather than comprehensive visual perception. To close the gap, we propose a deterministic and verifiable paradigm, which requires models generate executable Python code that faithfully reproduces the original image. Only through complete and accurate visual comprehension can a model successfully reproduce the original image with high fidelity.

##### What is STEM2Code-Eval?

- 1) Source Image Collection. We collect STEM images from test sets of six VQA-based STEM benchmarks [28, 34, 47, 53, 64, 66], which span multiple STEM domains.
- 2) Code Generation. We first prompt Gemini2.5-Pro [7] to generate detailed image captions about visual elements, spatial relationships, numerical values, etc. Claude-Opus

4.1 [1] then combines the caption and image to synthesize reproducible Python code using matplotlib libraries.

- 3) Code Agent. The generated code enters an iterative refinement loop: (i) rendering images via Python; (ii) applying LLM-based code repair upon execution failure; (iii) if successfully rendered, using Gemini2.5-Pro to score image similarity (style, content, functionality) against the original image and provide improvement suggestions, then refining the code to enhance reconstruction quality. This continues until the score exceeds 90% or reaches 10 iterations.

Table 2. Performance evaluation on our STEM2Code-Eval with 1k samples. We employ three metrics to assess their performances : (1) Image Scoring measures the visual similarity between the generated and original images; (2) Code Scoring assesses the quality, structure, and correctness of the generated Python code itself; (3) Exec Rate reports the execution success rate.

Image Score

Code Score

Exec Rate

Model

Avg.

Gemini2.5-Flash-Thinking 57.25 60.87 59.06 85.20 Claude-Opus 4.1-Thinking 55.90 56.19 56.05 97.10 GPT5-Thinking 64.97 64.98 64.98 96.60 Gemini2.5-Pro-Thinking 68.89 75.41 72.15 91.70

Intern-S1-8B [3] 6.02 15.87 10.95 26.60 InternVL3.5-8B [50] 13.50 18.16 15.83 56.50 MiniCPM-V-4.5 [44] 13.91 23.69 18.80 50.80 MiMo-VL-7B-RL [54] 14.54 22.41 18.48 60.30 Ovis2.5-9B [29] 9.76 11.26 10.51 89.40 KeyeVL1.5-8B [43] 20.33 22.47 21.40 73.40 GLM-4.1V-9B [20] 21.19 26.51 23.85 72.00 Qwen3-VL-4B-Thinking [36] 25.38 34.53 29.96 75.70 Qwen2.5-VL-72B-Instruct [4] 32.82 25.83 29.33 86.30 Qwen3-VL-8B-Thinking [36] 29.82 41.71 35.77 78.90 Qwen3-VL-30A3B-Instruct [36] 33.05 31.04 32.05 87.50 Seed1.6-Vision-nothinking [18] 31.22 38.56 34.89 85.50 Qwen3-VL-32B-Instruct [36] 36.85 39.98 38.42 81.80 Qwen3-VL-30A3B-Thinking [36] 37.47 35.53 36.50 87.10 Qwen3-VL-Plus-Instruct [36] 45.94 40.40 43.17 90.00 Qwen3-VL-Plus-Thinking [36] 45.59 40.61 43.10 89.20 Seed1.6-Vision-Thinking [18] 42.03 40.74 41.39 94.70

Qwen3-VL-4B-Instruct [36] 24.55 26.42 25.49 79.40 CodePercept-4B-S1 38.13 43.43 40.78 80.70 CodePercept-4B-R1 47.17 45.86 46.52 91.30

Qwen3-VL-8B-Instruct [36] 28.59 28.23 28.41 85.30 CodePercept-8B-S1 44.53 46.78 45.66 87.60 CodePercept-8B-R1 50.25 47.04 48.65 93.40

Qwen3-VL-32B-Instruct [36] 36.85 39.98 38.42 81.80 CodePercept-32B-S1 61.14 56.99 59.07 93.00 CodePercept-32B-R1 68.97 62.53 65.75 95.90

- 4) Candidate Selection. We rank all image-code pairs by similarity scores (reflecting reconstruction quality) and iteration counts (reflecting example difficulty), selecting the top 3k that score high on both dimensions for verification.
- 5) Visual Judge. Ten expert annotators score candidates on a 5-point scale across three dimensions (style, content, functionality). The top 1,000 examples by average score are selected and undergo minor code refinement by human annotators to achieve pixel-perfect reproduction.

### 5. Experiments

Implement Details We use the latest Qwen3-VL [36] as the base model. SFT is trained on ICC-1M dataset for 1 epoch with SWIFT [65] on 32 A100 GPUs. The subsequent RL stage uses VeRL [39] for 1 epoch on 10k samples selected from our ICC-1M with the same hardware. Additional details about dataset curation, training settings, prompts, and visualizations are provided in supplementary materials.

Table 3. Ablation study on perception ability, evaluated using the captioner-solver setup. More accurate descriptions are expected to yield better reasoning results. “IM”, “ID”, “SG” refer to Image Reproduce, Image Diversity, and Solid Geometry pipeline, respectively. “CodeCap” and “ImCode” refer to code-grounded captions and explanatory image-code pairs, respectively.

Benchmark Datasets (%)

Group Image Captioner

Average MathVision MathVista MathVerse DynaMath WeMath LogicVista

###### LLM Solver: Qwen3-30A3-Thinking [55]

- 1

Qwen3-VL-8B-Instruct [36] 54.37 69.60 63.75 72.19 45.43 56.82 60.36 + IR-CodeCap 55.86 69.50 63.50 72.59 46.95 57.05 60.91 + ID-CodeCap 58.32 71.20 64.52 71.70 46.76 60.40 62.15 + SG-CodeCap 59.93 68.81 66.54 72.73 47.05 62.64 62.75

- 2

NativeCap 56.20 68.39 63.87 70.69 46.48 59.06 60.78 CodeCap 59.93 68.81 66.54 72.73 47.05 62.64 62.75

- 3

CodeCap 59.93 68.81 66.54 72.73 47.05 62.64 62.75 CodeCap + ImCode 59.31 70.20 66.52 73.20 49.14 61.52 63.32

#### 5.1. Main Results

- 1) Problem-solving Perception Evaluation We evaluate our CodePercept model against previous SOTA MLLMs in their capacity as image captioners, subsequently employing the same LLM (Qwen3-30A3-Thinking and Qwen3235A22-Thinking) [55] as the solver to generate final answers across six widely-adopted STEM reasoning benchmarks. More accurate image descriptions should yield better reasoning results. Tab. 1 presents comprehensive comparisons across all benchmarks, where our CodePercept model demonstrate consistent and substantial improvements over baseline MLLMs. Under the Qwen3-30A3Thinking solver, CodePercept-4B-S1 outperforms Qwen3VL-4B-Instruct by 2.8%. When scaled to 8B parameters, the gains further increase to 3%. When leveraging the more powerful Qwen3-235A22-Thinking solver, CodePercept4B/8B-S1 maintains comparable improvements (2.9% and 3.4%, respectively), demonstrating remarkable robustness across different solver capabilities. Notably, CodePercept8B-S1 surpasses several substantially larger models, including Qwen2.5-VL-72B (by 6.2%), and even approaches the performance of frontier models like Claude-Opus 4.1Thinking and GPT5-Thinking. These results validate our central hypothesis: code-grounded perception serves as an effective paradigm for enhancing STEM visual perception.
- 2) Image Reproduce Perception Evaluation Problemsolving metric only reflects the model’s capacity to understand problem-relevant information rather than comprehensive visual perception. To directly assess visual perception, we introduce STEM2Code-Eval to support image reproduce perception evaluation. Tab. 2 presents comprehensive results. Through joint training two code-grounded tasks, CodePercept-S1 models achieve substantial improvements, with CodePercept-4B-S1 reaching 54.09 (+10.6 over Qwen3-VL-4B-Instruct) and CodePercept-8B-S1 achieving 59.64 (+12.3 over Qwen3-VL-8B-Instruct). These gains

validate grounding perception in executable code fundamentally enhances strong visual perception.

We further optimize code generation using RL to enforce executability and reconstruction accuracy. CodePercept4B-R1 achieves 61.44 (+7.35) and CodePercept-8B-R1 reaches 63.56 (+3.92). Both models surpass super-large models including Seed1.6-Vision and Qwen3-VL-Plus by 2.29-2.97 and 4.41-5.09, respectively.

#### 5.2. Ablation Study

- 1) Three Pipeline Comparison We evaluate three data generation strategies by training models on captions generated via Code-Grounded Caption Generation (Sec. 3.2). Tab. 3 (Group 1) shows a gradual increase in the average score (60.91 → 62.15 → 62.75), demonstrating the effectiveness of our three pipelines. In particular, ID-CodeCap, which includes diverse, procedurally-generated STEM images with verified code provide stronger training signals, yields the greatest performance gain.
- 2) Code-Grounded Caption Generation To validate our code-grounded approach (Sec. 3.2), we compare it against direct caption generation (NativeCap), where we prompt Gemini2.5-pro [7] with identical instructions to generate image captions directly without code analysis on the same ICC-1M data in Group 2. Our analysis yields two key insights. First, even NativeCap improves the average score to 60.36 over the 60.78 baseline, demonstrating the general effectiveness of our data pipeline. Second and more importantly, our code-grounded method (CodeCap) further boosts performance to 62.75, achieving a significant 2.0 gain over NativeCap. This validates our hypothesis that by extracting visual facts from executable code, CodeCap effectively reduces hallucinations in numerical values and geometric details common in direct vision-language generation.
- 3) STEM Image-to-Code Translation Adding ImCode data (Group 3) further boosts performance to 63.32, gaining 0.6 over CodeCap alone. This shows that code serves

as a complementary modality for image caption. Image-tocaption and image-to-code tasks reinforce each other: captions provide semantic context for code generation, while code overcomes the inherent limitations of natural language in describing complex mathematical visuals through its unique structured representation. Together, both tasks demonstrate that code is essential for reliable STEM image perception: whether as intermediate supervision for caption synthesis or as a direct target for structured understanding.

### 6. Conclusion

We identify visual perception as the primary bottleneck for MLLMs in STEM. To address this, we introduce CodePercept, a paradigm that leverages executable Python code to enhance perception. Our contributions include ICC-1M, a large-scale image-caption-code triplets, enabling two novel training approaches: CodeGrounded Caption Generation and STEM Image-to-Code Translation. We also develop STEM2Code-Eval, a benchmark that evaluates perception by generating executable code to reconstruct an image, beyond traditional problem-solving proxies. Experimental results prove the effectiveness of our CodePercept, establishing that executable codes, whether as intermediate supervision for caption synthesis or as a direct target for structured understanding, are essential for enhancing perception.

### References

- [1] Anthropic. Claude 4.1 opus. https : / / www . anthropic.com/claude, 2025. Accessed: 2025-08-

05. 6, 7, 19

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2
- [3] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025. 1, 6, 7, 19, 20
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 7, 19, 20
- [5] Ruizhe Chen, Tianze Luo, Zhiting Fan, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Zhang Huaijian. Datasets and recipes for video temporal grounding via reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 983–992, 2025. 2
- [6] Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From opti-

mized cold start to staged reinforcement learning. arXiv

- preprint arXiv:2506.04207, 2025. 2, 12

- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv

preprint arXiv:2507.06261, 2025. 7, 8

- [8] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv preprint arXiv:2503.17352, 2025. 2
- [9] Pei Fu, Tongkun Guan, Zining Wang, Zhentao Guo, Chen Duan, Hao Sun, Boming Chen, Jiayao Ma, Qianyi Jiang, Kai Zhou, et al. Multimodal large language models for textrich image understanding: A comprehensive review. arXiv preprint arXiv:2502.16586, 2025. 3
- [10] Tong Ge, Yashu Liu, Jieping Ye, Tianyi Li, and Chao Wang. Advancing vision-language models in front-end development via data synthesis. arXiv preprint arXiv:2503.01619,

2025. 3

- [11] Tongkun Guan, Chaochen Gu, Changsheng Lu, Jingzheng Tu, Qi Feng, Kaijie Wu, and Xinping Guan. Industrial scene text detection with refined feature-attentive network. IEEE Transactions on Circuits and Systems for Video Technology, 32(9):6073–6085, 2022. 3
- [12] Tongkun Guan, Chaochen Gu, Jingzheng Tu, Xue Yang, Qi Feng, Yudi Zhao, and Wei Shen. Self-supervised implicit glyph attention for text recognition. In CVPR, pages 15285– 15294, 2023.
- [13] Tongkun Guan, Wei Shen, Xue Yang, Qi Feng, Zekun Jiang, and Xiaokang Yang. Self-supervised character-tocharacter distillation for text recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19473–19484, 2023.
- [14] Tongkun Guan, Wei Shen, Xue Yang, Xuehui Wang, and Xiaokang Yang. Bridging synthetic and real worlds for pretraining scene text detectors. In European Conference on Computer Vision, pages 428–446. Springer, 2024.
- [15] Tongkun Guan, Chengyu Lin, Wei Shen, and Xiaokang Yang. Posformer: recognizing complex handwritten mathematical expression with position forest transformer. In European Conference on Computer Vision, pages 130–147. Springer, 2025.
- [16] Tongkun Guan, Wei Shen, and Xiaokang Yang. CCDPlus: Towards accurate character to character distillation for text recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 3
- [17] Tongkun Guan, Zining Wang, Pei Fu, Zhengtao Guo, Wei Shen, Kai Zhou, Tiezhu Yue, Chen Duan, Hao Sun, Qianyi Jiang, et al. A token-level text image foundation model for document understanding. arXiv preprint arXiv:2503.02304,

2025. 2

- [18] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 2, 7, 20

- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 6, 7
- [20] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025. 2, 6, 7, 19, 20
- [21] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749,

2025. 2

- [22] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 6
- [23] Songtao Jiang, Tuo Zheng, Yan Zhang, Yeying Jin, Li Yuan, and Zuozhu Liu. Med-moe: Mixture of domain-specific experts for lightweight medical vision-language models. In Findings of the association for computational linguistics: EMNLP 2024, pages 3843–3860, 2024. 1
- [24] Songtao Jiang, Yuan Wang, Ruizhe Chen, Yan Zhang, Ruilin Luo, Bohan Lei, Sibo Song, Yang Feng, Jimeng Sun, Jian Wu, et al. Capo: Reinforcing consistent reasoning in medical decision-making. arXiv preprint arXiv:2506.12849, 2025. 2
- [25] Songtao Jiang, Yuan Wang, Sibo Song, Tianxiang Hu, Chenyi Zhou, Bin Pu, Yan Zhang, Zhibo Yang, Yang Feng, Joey Tianyi Zhou, et al. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668, 2025. 1
- [26] Jinke Li, Jiarui Yu, Chenxing Wei, Hande Dong, Qiang Lin, Liangjing Yang, Zhicai Wang, and Yanbin Hao. Unisvg: A unified dataset for vector graphic understanding and generation with multimodal large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13156–13163, 2025. 3
- [27] Kevin Lin, Zhengyuan Yang, Linjie Li, Jianfeng Wang, and Lijuan Wang. Designbench: Exploring and benchmarking dall-e 3 for imagining visual design. arXiv preprint arXiv:2310.15144, 2023. 3
- [28] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 2, 7
- [29] Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, et al. Ovis2. 5 technical report. arXiv preprint arXiv:2508.11737, 2025. 2, 7, 20
- [30] Yiting Lu, Jiakang Yuan, Zhen Li, Shitian Zhao, Qi Qin, Xinyue Li, Le Zhuo, Licheng Wen, Dongyang Liu, Yuewen Cao, et al. Omnicaptioner: One captioner to rule them all. arXiv preprint arXiv:2504.07089, 2025. 2, 3, 7
- [31] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Rui-

- hang Chu, Jin Zeng, et al. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025. 2
- [32] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 2
- [33] OpenAI. GPT-5 Technical Report, 2023. 6, 19
- [34] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning?

- arXiv preprint arXiv:2407.01284, 2024. 2, 7

[35] Zeju Qiu, Weiyang Liu, Haiwen Feng, Zhen Liu, Tim Z Xiao, Katherine M Collins, Joshua B Tenenbaum, Adrian Weller, Michael J Black, and Bernhard Sch¨olkopf. Can large language models understand symbolic graphics programs?

- arXiv preprint arXiv:2408.08313, 2024. 3

- [36] Qwen Team. QwenLM/Qwen3-VL: The Official Repo of Qwen3-VL. https://github.com/QwenLM/ Qwen3-VL, 2025. 1, 5, 6, 7, 8, 19, 20
- [37] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 7
- [38] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 6
- [39] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025. 7
- [40] Weikang Shi, Aldrich Yu, Rongyao Fang, Houxing Ren, Ke Wang, Aojun Zhou, Changyao Tian, Xinyu Fu, Yuxuan Hu, Zimu Lu, et al. Mathcanvas: Intrinsic visual chainof-thought for multimodal mathematical reasoning. arXiv preprint arXiv:2510.14958, 2025. 2
- [41] Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: Benchmarking multimodal code generation for automated front-end engineering. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3956–3974, 2025. 3
- [42] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. 1, 2
- [43] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. 6, 7, 19, 20
- [44] MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong,

- Ganqu Cui, et al. Minicpm4: Ultra-efficient llms on end devices. arXiv preprint arXiv:2506.07900, 2025. 1, 2, 6, 7, 19, 20
- [45] Sandro Tosi. Matplotlib for Python developers. Packt Publishing Birmingham, UK, 2009. 3
- [46] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025. 2
- [47] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. NIPS, 37:95095–95169, 2024. 1, 2, 7
- [48] Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, et al. Mathcoder-vl: Bridging vision and code for enhanced multimodal mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2505–2534, 2025. 2
- [49] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [50] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 6, 7, 19, 20
- [51] Zining Wang, Tongkun Guan, Pei Fu, Chen Duan, Qianyi Jiang, Zhentao Guo, Shan Guo, Junfeng Luo, Wei Shen, and Xiaokang Yang. Marten: Visual question answering with mask generation for multi-modal document understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14460–14471, 2025. 2
- [52] Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, et al. Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255, 2025. 2
- [53] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 2, 7
- [54] LLM Xiaomi, Bingquan Xia, Bowen Shen, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, et al. Mimo: Unlocking the reasoning potential of language model–from pretraining to posttraining. arXiv preprint arXiv:2505.07608, 2025. 1, 2, 7, 20
- [55] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6, 8, 19
- [56] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024. 3

- [57] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Fukun Yin, Jiaxu Zhang, Liao Wang, Gang Yu, Xingjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [58] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025. 2
- [59] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024. 12
- [60] En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954,

2025. 2

- [61] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 13
- [62] Sukmin Yun, Rusiru Thushara, Mohammad Bhat, Yongxin Wang, Mingkai Deng, Jinhong Wang, Tianhua Tao, Junbo Li, Haonan Li, Preslav Nakov, et al. Web2code: A largescale webpage-to-code dataset and evaluation framework for multimodal llms. Advances in neural information processing systems, 37:112134–112157, 2024. 3
- [63] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025. 2
- [64] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In ECCV, pages 169–186. Springer, 2024. 2, 7
- [65] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, et al. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 29733–29735, 2025. 7
- [66] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836,

2024. 2, 7

## CodePercept: Code-Grounded Visual STEM Perception for MLLMs Supplementary Material

### A. Unified Quality Control

To ensure the reliability and consistency of our data generation pipeline, we implement a unified quality control framework with explicit prompts for each stage. Our quality control operates at three critical checkpoints: (1) verifying the correctness and quality of generated code, (2) verifying the correctness and quality of rendered image, and (3) verifying the consistency of image-code pairs. Below, we present the detailed prompts used at each stage to maintain high standards of both technical accuracy and instructional effectiveness.

##### Code Quality Prompt QC

You are a leading AI scientist tasked with curating a training dataset for an Image-to-Code model designed to understand and generate expert-level human code. Your role combines philosophical insight with engineering rigor as a quality assessor.

- 1. Core Philosophy: Maximize Semantic Value Objective: Cultivate the model’s ability to map visual patterns to high-level programming concepts by distinguishing meaningful abstractions from arbitrary procedures. 1.1 Conceptual Framework

- 1. High-Value Abstraction (samples we pursue):

- • Code implements mathematical concepts or algorithms with clear, visually recognizable structural features
- • Procedural steps are necessary components for realizing the abstraction
- • Examples: y = x2 (parabola), Fibonacci spiral algorithms, noise.pnoise2 (procedural noise)

- 2. Low-Value Procedure (samples we reject):

- • Code contains arbitrary, context-free computational steps not inferable from the final image
- • Contains “magic numbers” and “black-box operations” toxic to model learning
- • Examples: data += 100, data = data * 1.1, if unseen variable: ...

- 2. Evaluation Rules

###### 2.1 Pattern Recognition Rule

- 1. [Qualified] Image exhibits strong, non-random structural patterns:

- • Symmetry, spirals, fractals, waveforms, regular grids
- • Code clearly implements mathematical formulas or algorithms producing the pattern
- • This represents the highest-value sample type

- 2. [Qualified] Image appears random but uses domain-standard tools:

- • Examples: noise maps generated via noise.pnoise2
- • Teaches proper use of professional libraries

- 3. [Qualified] Simple graphics with hardcoded constants:

- • Bar charts, scatter plots with direct data declaration (data = [1,2,3])
- • Basic but acceptable “declarative” samples

- 4. [Disqualified] True random generation without structure:

• Generated by np.random.rand() with no recognizable macroscopic patterns

###### 2.2 Computational Intent Rule

- 1. [Qualified] Computational steps are intrinsic to high-value abstraction:

• Example: Z-value-based color/linewidth computation in torus knot code implementing “pseudo-3D depth perception” (a universal rendering technique)

- 2. [Disqualified] Arbitrary, isolated numerical operations:

- • Logic or context completely lost in the image
- • Example: aqi data[0, :] += 100

- 2.3 Dependency & Source Rule

1. [Qualified] Dependencies limited to:

- • Python standard library
- • Well-known pip-installable libraries directly related to visual generation (numpy, scipy, noise, etc.)

##### Code Quality Prompt QC (continued)

2. [Disqualified] Reliance on inaccessible external resources:

- • External files (.csv, .json), databases, or network APIs
- • Model cannot access these resources during inference

###### 3. Required Output Format

- • [Verdict]: (Qualified / Disqualified)
- • [Suitability Score]: (1-5, where 1=harmful, 3=baseline qualified, 5=exceptionally educational)
- • [Rationale]: Briefly explain the verdict. Explicitly identify whether the code represents [High-Value Abstraction] or [LowValue Procedure]. If qualified, articulate the “core concept” being taught (e.g., “recognizing Fibonacci spiral algorithms” or “learning pseudo-3D rendering techniques”). If disqualified, specify which computational step is “arbitrary and noninvertible.”

Evaluate the provided Python visualization code according to these principles.

##### Image Quality Prompt QI

You are a professional image quality analyst. Your mission is to conduct a quality review of a given code-generated image according to the strict standards defined below. Your report must not only include the final verdict but, more importantly, must demonstrate a clear, systematic analysis process to prove that your conclusion is well-considered.

- 1. Stage 1: Technical Rendering Audit Task: Assess whether the image has fatal technical rendering defects. Error Classification Definitions

- A. (Class) Fatal Rendering Errors: Technical rendering failures caused by algorithmic or data interpretation failures that severely compromise the geometry, paths, or annotations of the graphic.

- • A-1. Vector Path Catastrophe:

- – Path Distortion: Paths exhibit severe unexpected spikes, kinks, or non-smoothness.
- – Projection Artifacts: A portion of a path or fill is severely stretched to or beyond the canvas boundary.
- – Unexpected Unclosed Path: A shape that should be closed has a significant gap, causing the fill color to “leak”.

- • A-2. Geometric Annotation Error:

– Severe Anchor Misplacement: The anchor point of an annotation element (e.g., angle arc, dimension arrow)

is completely detached from the object it is supposed to annotate.

- • A-3. Empty Chart:

– The coordinate system and labels are drawn correctly, but the data series (e.g., lines, bars) is entirely missing.

- • A-4. Indiscernible Density:

– Contains a large number of elements that are indescribable, chaotic, indistinguishable, densely overlapping,

or blurry (e.g., a “point cloud soup” or a “sea of lines”).

- • A-5. Layout Anomaly:

- – Abnormal layout resulting in large areas of whitespace.
- – Crowded layout making key elements difficult to distinguish.

- B. (Class) Non-Fatal Design Flaws: The technical rendering is successful, but there are flaws in aesthetics or content logic.

• Examples: Poor color choices, normal occlusion between elements, unintuitive logical layer order, questionable content that is drawn correctly.

- 2. Core Principles & Output Format

- • Principle 1: Identify Class A Errors (Critical). If at least one significant, unintentional, technical Class A rendering error is present, the verdict is Fail.
- • Principle 2: Exempt All Minor Issues. Extremely minor technical glitches (e.g., single-pixel offsets) and all Class B design flaws (e.g., aesthetic issues), regardless of their number, must never be the basis for a “Fail” verdict.

Output Format: Strictly follow the process above, analyzing step-by-step and providing reasons. Then, generate the complete analysis report using the specified Markdown format below.

## Rendering Quality Audit

* Final Verdict: [Enter ‘‘Pass" or ‘‘Fail"]

##### Image Quality Prompt QI (continued)

### Error Analysis [Note: This section should only be generated if the final verdict is ‘‘Fail". If multiple errors exist, repeat the list item.]

- * Error Type: [Enter A-1, A-2, etc.]
- * Description: [Describe the specific error phenomenon and its location in the image.]

##### Image-Code Consistency Prompt QIC

You are a professional image quality analyst. Your mission is to conduct a quality review of the given code-image pair according to the strict standards defined below. Task: Determine if the core structure and scene presented in the image are fundamentally consistent with the design intent of the code.

- 1. Workflow

- 1. Visual Evidence Inventory: Ignore the code and visually inventory the elements in the image (quantity, color, layout, relationships).
- 2. Code Logic Deduction (Design Blueprint): Mentally compile the code, analyzing variables (e.g., n=8), loops (e.g., for i in range(8)), and control flow to precisely deduce the image’s quantitative specifications (e.g., total number of elements, exact coordinates, connection relationships, etc.).
- 3. Critical Comparison: Strictly compare the “Visual Evidence” with the “Design Blueprint”.

- 2. Core Verdict Standard: “Fundamental Mismatch”

- • “Sufficient Match”: The core elements, deduced quantities, attributes, and spatial/structural relationships described by the code are fundamentally consistent with the image. Minor rendering differences (e.g., color shades, anti-aliasing effects) are permissible.
- • “Fundamental Mismatch”: There is a structural or categorical essential difference between the code’s intent and the image’s reality. Any of the following conditions qualify:

- a. Structural/Layout/Relational Breakdown: The organization of elements or their spatial relationships have completely collapsed. Example: Code intends to generate a geometric puzzle of neatly arranged shapes, but the image shows them randomly piled up with misaligned vertices.
- b. Categorical Difference: The core scene or element type is entirely different. Example: Code draws a bar chart, but the image is a scatter plot.
- c. Severe Mismatch of Key Elements/Attributes: The quantity, type, or core attributes of key elements are severely missing or incorrect. Example: Code intends to draw a 10-pointed star, but the image shows only 3 unrelated lines.
- d. Structural Collapse or Loss of Integrity: The elements in the image are fragmented or incomplete, failing to form the complete structure intended by the code. Example: Code intends to draw a complete, closed ellipse, but the image only shows two separate, unclosed curves.
- e. Data-Level Misalignment: The data order, sorting, or mapping relationship between elements does not match the code’s intent. Example: Code intends to connect points in the order A->B->C, but the image shows A->C->B.

- 3. Instructions Now, here is the code used to generate the image: {input code this} Strictly follow the process above, analyze step-by-step, provide reasons, and then generate the complete analysis report using the specified Markdown format below.
- 4. Output Format # Image Quality Analysis Report ## Consistency Review

- * Verdict: [Enter ‘‘Sufficient Match" or ‘‘Fundamental Mismatch"]
- * Reason: [Fill this in only if the verdict is ‘‘Fundamental Mismatch", explaining

##### Image-Code Consistency Prompt QIC (continued)

the type and specific phenomenon of the mismatch.]

### B. Native Image-Code Pairs v.s. Explanatory Image-Code Pairs

#### 1 The Definition of Native Image-Code Pairs

Native image-code pairs consist of raw visual image paired with ground-truth executable code without additional explanatory context. While these pairs ensure code correctness, they lack the pedagogical scaffolding necessary for models to learn the visual-to-code mappings and the rationale for specific implementation choices.

#### 2 The Definition of Explanatory Image-Code Pairs

Explanatory image-code pairs augment ground-truth code with step-by-step descriptions, explicit parameter justifications, and instructional commentary that articulates how visual elements map to code constructs. This explanatory richness enables models to understand not merely what code to generate, but why particular implementations best capture the visual content.

To illustrate the enhanced instructional value of explanatory image-code pairs, we present a representative example below. The example demonstrates how the explanatory framework transforms bare code into comprehensive learning materials by integrating visual observation, implementation logic, and detailed commentary.

##### A Explanatory Image-Code Pair Example

[Figure 87]

Figure 3. The hexagonal grid with a spiraling path, generated by the Python code below.

An analysis of the image reveals a honeycomb structure composed of seven regular hexagons. A central, darker blue hexagon is surrounded by six lighter blue hexagons. A dashed orange path originates inside the central hexagon and spirals outwards, passing through the vertices of the central hexagon. Below is a breakdown of the steps to reproduce this image using Python’s matplotlib library, followed by the complete, executable code.

- 1. Implementation Breakdown To reproduce this, we can use the matplotlib library in Python. Here’s a breakdown of the steps:

- 1. Hexagon Grid: We define a function to calculate the vertices of a regular “flat-top” hexagon given its center and size. The central hexagon is placed at the origin (0, 0), and the six surrounding hexagons are arranged in a standard hexagonal grid pattern around it.
- 2. Drawing Hexagons: The matplotlib.patches.Polygon class is used to create the hexagon shapes. The central hexagon is colored steelblue, and the outer ones are lightblue. All have black edges, with the central one having a slightly thicker border.
- 3. Defining the Path: The path is a complex polyline. It starts near the center, moves to the first vertex (p1), then “bounces” into the adjacent hexagon to an intermediate point (mid1), and then connects to the next vertex (p2). This pattern repeats for all six vertices of the central hexagon (p1 through p6).
- 4. Drawing the Path and Arrows: The path is constructed from two types of segments:

- ◦ Simple dashed lines, drawn with ax.plot().
- ◦ Dashed lines with an arrowhead, drawn with ax.annotate(). The annotate function is used here to draw an arrow from a start point (xytext) to an end point (xy) with specific styling (dashed, orange, with a particular arrowhead shape). This is used for the first leg of each “bounce” (e.g., p1 → mid1).

- 5. Labels and Markers: The vertices 1 through 6 are marked with a small white circle with a black outline, created using ax.plot() with an ’o’ marker. The corresponding numbers are placed nearby using ax.text(). The positions of the

- labels and the intermediate path points are carefully chosen to match the original image.
- 6. Final Touches: The axes are turned off, and the plot limits are set to frame the hexagons similarly to the original image, creating a clean and focused visual.

- 2. Python Implementation Code The following code reproduces the honeycomb structure:

- 1 import matplotlib.pyplot as plt

- 2 import numpy as np

- 3 from matplotlib.patches import Polygon

- 4

- 5 def generate_hexagon_path_image():

- 6 """

- 7 Generates the code to reproduce the image of

- 8 a hexagonal grid with a path.

- 9 """

- 10 # 1. Setup figure and axes

- 11 fig, ax = plt.subplots(figsize=(10, 10))

- 12 ax.set_aspect(’equal’)

- 13 ax.axis(’off’)

- 14 fig.set_facecolor(’white’)

- 15

- 16 # 2. Parameters

- 17 s = 1.0 # Hexagon side length

- 18 dark_blue_color = ’steelblue’

- 19 light_blue_color = ’#add8e6’

- 20 orange_color = ’orange’

- 21 line_props = dict(color=orange_color,

- 22 linestyle=’--’,

- 23 linewidth=2,

- 24 zorder=2)

- 25 arrow_props = dict(

- 26 arrowstyle=’->, head_length=0.7, head_width=0.4’,

- 27 **line_props)

- 28

- 29 # 3. Hexagon generation function (flat-top)

- 30 def get_flat_top_hexagon_vertices(center, size):

- 31 x, y = center

- 32 return np.array([

- 33 [x + size, y],

- 34 [x + size / 2, y + size * np.sqrt(3) / 2],

- 35 [x - size / 2, y + size * np.sqrt(3) / 2],

- 36 [x - size, y],

- 37 [x - size / 2, y - size * np.sqrt(3) / 2],

- 38 [x + size / 2, y - size * np.sqrt(3) / 2],

- 39 ])

- 40

- 41 # 4. Draw hexagons

- 42 center_main = np.array([0, 0])

- 43 vertices_main = get_flat_top_hexagon_vertices(

- 44 center_main, s)

- 45 ax.add_patch(Polygon(vertices_main,

- 46 facecolor=dark_blue_color,

- 47 edgecolor=’black’,

- 48 linewidth=2))

- 49

- 50 centers_outer = [

- 51 center_main + [0, s * np.sqrt(3)],

- 52 center_main + [s * 1.5, s * np.sqrt(3) / 2],

- 53 center_main + [s * 1.5, -s * np.sqrt(3) / 2],

- 54 center_main + [0, -s * np.sqrt(3)],

- 55 center_main + [-s * 1.5, -s * np.sqrt(3) / 2],

- 56 center_main + [-s * 1.5, s * np.sqrt(3) / 2],

- 57 ]

- 58 for center in centers_outer:

- 59 ax.add_patch(Polygon(

- 60 get_flat_top_hexagon_vertices(center, s),

- 61 facecolor=light_blue_color,

- 62 edgecolor=’black’,

- 63 linewidth=1))

- 64

- 65 # 5. Define path vertices and intermediate points

- 66 p2, p1, p6, p5, p4, p3 = vertices_main

- 67

- 68 mid1 = np.array([1.05, 0.45])

- 69 mid2 = np.array([1.05, -0.45])

- 70 mid3 = np.array([0, -1.3])

- 71 mid4 = np.array([-1.05, -0.45])

- 72 mid5 = np.array([-1.05, 0.45])

- 73 mid6 = np.array([-0.2, 1.4])

- 74

- 75 path_start = np.array([-0.4, 0])

- 76 path_final_end = np.array([-0.7, 1.8])

- 77

- 78 # 6. Draw the path segments

- 79 ax.plot([path_start[0], p1[0]],

- 80 [path_start[1], p1[1]],

- 81 **line_props)

- 82

- 83 path_segments = [

- 84 (p1, mid1, p2),

- 85 (p2, mid2, p3),

- 86 (p3, mid3, p4),

- 87 (p4, mid4, p5),

- 88 (p5, mid5, p6)

- 89 ]

- 90

- 91 for start_node, mid_node, end_node in path_segments:

- 92 ax.annotate(’’, xy=mid_node,

- 93 xytext=start_node,

- 94 arrowprops=arrow_props)

- 95 ax.plot([mid_node[0], end_node[0]],

- 96 [mid_node[1], end_node[1]],

- 97 **line_props)

- 98

- 99 ax.annotate(’’, xy=mid6, xytext=p6,

- 100 arrowprops=arrow_props)

- 101 ax.plot([mid6[0], path_final_end[0]],

- 102 [mid6[1], path_final_end[1]],

- 103 **line_props)

- 104

- 105 # 7. Add vertex markers and labels

- 106 vertex_map = {’1’: p1, ’2’: p2, ’3’: p3,

- 107 ’4’: p4, ’5’: p5, ’6’: p6}

- 108 label_positions = {

- 109 ’1’: (p1[0] - 0.25, p1[1] - 0.1),

- 110 ’2’: (p2[0] + 0.15, p2[1] - 0.1),

- 111 ’3’: (p3[0] + 0.15, p3[1] - 0.1),

- 112 ’4’: (p4[0] - 0.25, p4[1] - 0.1),

- 113 ’5’: (p5[0] - 0.2, p5[1]),

- 114 ’6’: (p6[0] - 0.25, p6[1] - 0.1),

- 115 }

- 116

- 117 for num, pos in vertex_map.items():

- 118 ax.plot(pos[0], pos[1], ’o’,

- 119 markersize=6,

- 120 markerfacecolor=’white’,

- 121 markeredgecolor=’black’,

- 122 zorder=4)

- 123 label_pos = label_positions[num]

- 124 ax.text(label_pos[0], label_pos[1], num,

- 125 fontsize=14,

- 126 ha=’center’,

- 127 va=’center’,

- 128 zorder=5)

- 129

- 130 # 8. Finalize plot

- 131 ax.set_xlim(-2.8, 2.8)

- 132 ax.set_ylim(-2.8, 2.8)

- 133

- 134 plt.show()

- 135

- 136 if __name__ == ’__main__’:

- 137 generate_hexagon_path_image()

###### 3. Usage Instructions

- • Requires: matplotlib, numpy libraries
- • Run the script to generate the visualization
- • Adjust parameters in section 2 to customize appearance
- • Modify s variable to change hexagon size Note: This structured approach allows for faithful reproduction of hexagonal grid patterns with customizable paths.

### C. Native Caption v.s. Code-Grounded Caption

#### Comparisons

In this subsection, we present a representative example to illustrate the comparative advantages of our approach. The provided image poses a significant challenge: it contains an 8×8 grid with varying dot patterns in each cell, interconnected by red arrows that simulate the L-shaped movement of a knight in Chinese chess, traversing all cells to form a complex path.

We observe that even state-of-the-art MLLMs, such as Gemini 2.5 Pro, produce detailed descriptions riddled with factual errors when analyzing this image—misidentifying node counts (e.g., claiming 1 node instead of 2 in cell (1,8)), describing incorrect spatial arrangements, and reporting 68 arrows instead of the actual 63.

In contrast, our code-grounded captions deliver precise, verifiable information. By anchoring descriptions in executable code that can be validated against actual image data, this approach systematically eliminates perceptual hallucinations common in vision-language models, ensuring the blueprint-level accuracy essential for tasks demanding exact spatial reasoning and faithful reconstruction.

##### Comparisons on Native Caption and Code-Grounded Caption

[Figure 88]

Figure 4. A representative image, where poses a significant challenge to the perceptual abilities of current MLLMs.

Model Response: Of course. Here is a detailed description of the image, designed to empower AI models without visual capabilities to reason and provide human experts with blueprint-level reconstructions.

- 1. Overall Layout and Style This image is a square, minimalist abstract schematic diagram. Its core is a grid divided by black lines, with the grid cells filled in a light gray. Numerous solid black dots are scattered throughout the grid, a special starting point is marked by a red square, and there is a complex network of numerousseveral red arrows connecting different black dotscells. The overall image has no border, and the background is pure white.

- 2. Coordinate System To precisely describe the position of all elements in the image, we will establish a Cartesian coordinate system based on the grid. The grid has an 8x8 structure. We will number the vertical columns from 1 to 8, moving from left to right, and the horizontal rows from 1 to 8, moving from bottom to top. Therefore, the cell in the bottom-left corner has the coordinates (1, 1), and the cell in the top-right corner has the coordinates (8, 8). When describing the black dots within each cell, we will use relative positional terms such as “center,” “top-left,” “bottom-right,” “top-center,” etc.
- 3. Grid and Background The main body of the image is an 8x8 grid composed of thick black lines, containing a total of 64 perfectly identical square cells. The interior of each cell is filled with a light gray color (approximate hex code: #E0E0E0).
- 4. Detailed Element Description

- • 4.1 Black Dots (Nodes) The image contains a total of 206216 uniformly sized, solid black dots (nodes). These nodes are distributed across the 64 grid cells, with the number and arrangement of nodes varying in each cell. One of these nodes is specially marked as the starting point. Below is a detailed distribution of the nodes in each cell, ordered from top to bottom (Row 8 to Row 1) and left to right (Column 1 to Column 8): Row 8 (Top)

- – Cell (1, 8): 12 node, located at the center.arranged diagonally, located at “top-left” and “bottom-right.”.

- – Cell (2, 8): 31 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”located at the center.

- – Cell (3, 8): 34 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

- – Cell (4, 8): 35 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” “center,” “bottom-left,” and “bottom-right.”

- – Cell (5, 8): 26 nodes, arranged diagonally, located at “top-left” and “bottom-right.”arranged in two horizontal rows of three nodes each.

- – Cell (6, 8): 3 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (7, 8): 36 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged in two horizontal rows of three nodes each.

- – Cell (8, 8): 31 nodes, arranged horizontally in the upper half of the cell. We will refer to them as “left,” “middle,” and “right.”located at the center.

###### Row 7

- – Cell (1, 7): 13 node, located at the center.arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (2, 7): 4 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”
- – Cell (3, 7): 3 nodes, forming a ’¡’ shape pointing left, located at “top-right,” “middle,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (4, 7): 36 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

- – Cell (5, 7): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” “center,” “bottom-left,” and “bottom-right.”

- – Cell (6, 7): 36 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

- – Cell (7, 7): 41 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (8, 7): 34 nodes, forming a ’¡’ shape pointing left, located at “top-right,” “middle,” and “bottom-right.”arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

###### Row 6

- – Cell (1, 6): 16 node, located at the center.arranged in two horizontal rows of three nodes each.

- – Cell (2, 6): 31 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (3, 6): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (4, 6): 43 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (5, 6): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (6, 6): 35 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” “center,” “bottom-left,” and “bottom-right.”

- – Cell (7, 6): 32 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (8, 6): 31 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”located at the center.

###### Row 5

- – Cell (1, 5): 13 node, located at the center.arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (2, 5): 46 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

- – Cell (3, 5): 3 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (4, 5): 4 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”
- – Cell (5, 5): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (6, 5): 32 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (7, 5): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (8, 5): 36 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

###### Row 4

- – Cell (1, 4): 14 node, located at the center.arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

- – Cell (2, 4): 35 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (3, 4): 4 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”
- – Cell (4, 4): 41 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (5, 4): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (6, 4): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (7, 4): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (8, 4): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

###### Row 3

- – Cell (1, 3): 1 node, located at the center.
- – Cell (2, 3): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (3, 3): 35 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (4, 3): 4 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”
- – Cell (5, 3): 41 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (6, 3): 34 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

- – Cell (7, 3): 41 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (8, 3): 36 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

###### Row 2

- – Cell (1, 2): 14 node, located at the center.arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.” This is the special starting point.From (1, 1) to (2, 3)

- – Cell (2, 2): 3 nodes, arranged in a pyramid shape, located at “top-center,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (3, 2): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (4, 2): 45 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in a quincunx pattern, located at “top-left,” “top-right,” ”center,” “bottom-left,” and “bottom-right.”

- – Cell (5, 2): 42 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (6, 2): 43 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (7, 2): 46 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged in two horizontal rows of three nodes each.

- – Cell (8, 2): 43 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

Row 1 (Bottom)

- – Cell (1, 1): 1 node, located at the center. This is the special starting point.
- – Cell (2, 1): 36 nodes, arranged horizontally in the lower half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged in two horizontal rows of three nodes each.

- – Cell (3, 1): 43 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (4, 1): 32 nodes, arranged horizontally in the lower half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged diagonally, located at “top-left” and “bottom-right.”

- – Cell (5, 1): 43 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”arranged diagonally, located at “top-left”, “center”, and “bottom-right.”

- – Cell (6, 1): 34 nodes, arranged horizontally in the lower half of the cell. We will refer to them as “left,” “middle,” and “right.”arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

- – Cell (7, 1): 41 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”located at the center.

- – Cell (8, 1): 4 nodes, arranged in a 2x2 square, located at “top-left,” “top-right,” “bottom-left,” and “bottom-right.”

- • 4.2 Starting Point In cell (1, 1), the single central node is specially marked as the starting point. It is enclosed by a square slightly smaller than the cell itself. This square is filled with a bright red color (approximate hex code: #D82428) and has a thick black border..

- • 4.3 Red Arrows (Connections) There are a total of 6863 red arrows in the image. Each arrow is a straight line originating from the center of one black node and pointing to the center of another, with an arrowhead at its tip to indicate direction. These arrows crisscross to form a complex network. The following is a complete list of all 6863 arrows, describing their origin and destination points:

- 1. From (1, 1, center) to (2, 3, top-left)From (1, 1) to (2, 3)

- 2. From (1, 2, center) (Start Point) to (2, 1, left)From (2, 3) to (1, 5)

- 3. From (1, 5, center) to (2, 4, top-center)From (1, 5) to (2, 7)

- 4. From (1, 6, center) to (2, 8, left)From (2, 7) to (4, 8)

- 5. From (1, 7, center) to (2, 7, bottom-right)From (4, 8) to (6, 7)

- 6. From (1, 8, center) to (1, 7, center)From (6, 7) to (8, 8)

- 7. From (2, 1, left) to (1, 1, center)From (8, 8) to (7, 6)

- 8. From (2, 2, bottom-left) to (3, 3, top-center)From (7, 6) to (6, 8)

- 9. From (2, 2, bottom-right) to (1, 3, center)From (6, 8) to (8, 7)

- 10. From (2, 3, top-left) to (3, 4, bottom-left)From (8, 7) to (7, 5)

- 11. From (2, 4, top-center) to (2, 6, top-center)From (7, 5) to (8, 3)

- 12. From (2, 4, bottom-right) to (3, 2, top-right)From (8, 3) to (7, 1)

- 13. From (2, 5, bottom-right) to (1, 6, center)From (7, 1) to (5, 2)

- 14. From (2, 6, top-center) to (1, 5, center)From (5, 2) to (3, 1)

- 15. From (2, 7, bottom-right) to (3, 7, middle)From (3, 1) to (1, 2)

- 16. From (2, 8, left) to (1, 8, center)From (1, 2) to (2, 4)

- 17. From (3, 1, top-right) to (4, 3, top-left)From (2, 4) to (1, 6)

- 18. From (3, 2, top-right) to (4, 1, right)From (1, 6) to (2, 8)

- 19. From (3, 2, bottom-right) to (2, 2, bottom-left)From (2, 8) to (3, 6)

- 20. From (3, 3, top-center) to (4, 3, bottom-right)From (3, 6) to (1, 7)

- 21. From (3, 4, bottom-left) to (4, 5, bottom-right)From (1, 7) to (3, 8)

- 22. From (3, 5, bottom-right) to (4, 7, top-left)From (3, 8) to (5, 7)

- 23. From (3, 6, left) to (2, 5, bottom-right)From (5, 7) to (7, 8)

- 24. From (3, 7, middle) to (4, 8, left)From (7, 8) to (8, 6)

- 25. From (3, 7, top-right) to (2, 6, bottom-right)From (8, 6) to (7, 4)

- 26. From (4, 1, left) to (3, 1, top-right)From (7, 4) to (8, 2)

- 27. From (4, 2, top-left) to (3, 1, bottom-left)From (8, 2) to (6, 1)

- 28. From (4, 3, top-left) to (5, 5, top-left)From (6, 1) to (4, 2)

- 29. From (4, 3, bottom-right) to (5, 4, bottom-right)From (4, 2) to (2, 1)

- 30. From (4, 4, top-left) to (5, 3, top-left)From (2, 1) to (1, 3)

- 31. From (4, 5, bottom-right) to (3, 6, left)From (1, 3) to (3, 2)

- 32. From (4, 5, top-right) to (3, 5, top-center)From (3, 2) to (5, 1)

- 33. From (4, 6, bottom-left) to (3, 5, bottom-right)From (5, 1) to (6, 3)

- 34. From (4, 6, top-right) to (5, 8, bottom-right)From (6, 3) to (8, 4)

- 35. From (4, 7, top-left) to (5, 7, bottom-left)From (8, 4) to (7, 2)

- 36. From (4, 8, left) to (5, 8, top-left)From (7, 2) to (5, 3)

- 37. From (5, 1, middle-right) to (4, 2, top-left)From (5, 3) to (4, 1)

- 38. From (5, 2, bottom-left) to (6, 1, right)From (4, 1) to (2, 2)

- 39. From (5, 3, top-left) to (6, 3, bottom-right)From (2, 2) to (1, 4)

- 40. From (5, 4, bottom-right) to (4, 4, top-left)From (1, 4) to (3, 3)

- 41. From (5, 4, top-left) to (4, 4, top-right)From (3, 3) to (2, 5)

- 42. From (5, 5, bottom-right) to (5, 2, bottom-left)From (2, 5) to (4, 4)

- 43. From (5, 5, top-left) to (4, 6, bottom-left)From (4, 4) to (6, 5)

- 44. From (5, 6, top-right) to (4, 6, top-right)From (6, 5) to (4, 6)

- 45. From (5, 7, bottom-left) to (6, 7, top-center)From (4, 6) to (3, 4)

- 46. From (5, 8, top-left) to (6, 8, left)From (3, 4) to (5, 5)

- 47. From (6, 1, middle) to (5, 1, middle-right)From (5, 5) to (4, 7)

- 48. From (6, 2, bottom-left) to (5, 1, top-left)From (4, 7) to (2, 6)

- 49. From (6, 3, bottom-right) to (7, 3, top-right)From (2, 6) to (1, 8)

- 50. From (6, 4, top-right) to (7, 4, bottom-right)From (1, 8) to (3, 7)

- 51. From (6, 5, bottom-left) to (5, 6, top-right)From (3, 7) to (4, 5)

- 52. From (6, 6, top-right) to (5, 5, bottom-right)From (4, 5) to (6, 6)

- 53. From (6, 7, top-center) to (7, 7, bottom-right)From (6, 6) to (5, 8)

- 54. From (6, 8, left) to (7, 8, left)From (5, 8) to (7, 7)

- 55. From (7, 1, top-left) to (6, 2, bottom-left)From (7, 7) to (5, 6)

- 56. From (7, 2, bottom-right) to (8, 2, bottom-left)From (5, 6) to (3, 5)

- 57. From (7, 3, top-right) to (6, 5, bottom-left)From (3, 5) to (4, 3)

- 58. From (7, 4, bottom-right) to (8, 3, top-right)From (4, 3) to (6, 4)

- 59. From (7, 5, bottom-left) to (6, 4, top-right)From (6, 4) to (8, 5)

- 60. From (7, 6, bottom-left) to (8, 6, bottom-right)From (8, 5) to (7, 3)

- 61. From (7, 7, bottom-right) to (6, 6, top-right)From (7, 3) to (5, 4)

- 62. From (7, 8, left) to (8, 8, left)From (5, 4) to (6, 2)

- 63. From (8, 1, top-right) to (7, 1, top-left)From (6, 2) to (8, 1)

- 64. From (8, 2, bottom-left) to (8, 1, top-right)

- 65. From (8, 3, top-right) to (7, 2, bottom-right)

- 66. From (8, 5, bottom-left) to (7, 5, bottom-left)

- 67. From (8, 6, bottom-right) to (8, 5, bottom-left)

- 68. From (8, 7, bottom-right) to (7, 6, bottom-left)

The above description provides the precise number, location, shape, attributes, and interrelationships of all visual elements in the image, which is sufficient to support in-depth analysis or accurate reproduction of the image.

### D. Training Details

#### 1 The Training Details of CodePercept-S1

CodePercept-S1 is built upon the Qwen3-VL-Instruct foundation model and fine-tuned using a carefully curated mixture of dataset ICC-1M, which is derived from public STEM seed data including Mulberry-SFT [59] and GRAMMAR [6]. The model is trained for 1 epochs using AdamW optimizer with DeepSpeed Zero-2 optimization, employing a cosine learning rate schedule starting from 3 × 10−6 and decaying to 2 × 10−7 with 5% warm-up steps. Training is conducted with an effective batch size of 4 per device (2 samples per device with gradient accumulation of 2 steps), using bfloat16 mixed precision and

[Figure 89]

[Figure 90]

[Figure 91]

(a) The curves of CodePercept-S1 series (b) The curves of CodePercept-R1 series

Figure 5. The training curves of our proposed models. (a) the curves of CodePercept-S1 models. (b) the curves of CodePercept-R1 models.

Flash Attention for computational efficiency.

#### 2 The Training Details of CodePercept-R1

CodePercept-R1 advances upon CodePercept-S1 through Group Relative Policy Optimization (GRPO), a reinforcement learning approach designed to enhance code generation quality. The model is initialized from the CodePercept-S1 checkpoint and further refined using a carefully selected subset of 10,000 high-quality samples from the ICC-1M dataset, chosen based on quality filter. We first perform a one-time difficulty filtering based on Qwen3-VL-7B: under 8 rollout iterations, we select samples whose accuracy falls between 0.25 and 0.75 [61]. We utilize GRPO with a temperature of 1.0, top-p sampling of 0.85, and a repetition penalty of 1.1 to encourage diverse and coherent generations. For each training sample, the model generates 8 candidate completions with a maximum length of 8,192 tokens, which are then optimized using a composite reward function. Training is conducted for 1 epoch using the AdamW optimizer with a learning rate of 1 × 10−6 and 5% warm-up steps. The optimization uses bfloat16 mixed precision and gradient clipping with a maximum norm of 0.5 for training stability. The GRPO algorithm uses a KL penalty coefficient (beta) of 0.001 to balance between reward optimization and maintaining proximity to the reference policy.

successful

Image Reproduce (p1)

Code Filter(p4)

Image-Code Pair

CodeExecution

SoureImage

Image Filter(p5)

Code Perception

failed

LLM Repair(p3)

Ten Variations (p2)

SFT Data(p8) RL Data(p9)

2D(p6) 3D(p7)

iter+1

#### 3 The Training Curves of CodePercept-S1 series and CodePercept-R1 series

As shown in Fig. 5, this image displays three charts illustrating the training performance of different versions of a model named “CodePercept”. The charts are divided into two groups: (a) for the “CodePercept-S1 series” and (b) for the “CodePercept-R1 series”. Both series compare models of three different sizes: 4 billion (4B), 8 billion (8B), and 32 billion (32B) parameters. Across both the S1 and R1 training stages, the charts consistently demonstrate a clear scaling law: larger models in the CodePercept family exhibit better performance. Whether in the supervised fine-tuning stage (S1) or the reinforcement learning stage (R1), the CodePercept model with a larger number of parameters (32B > 8B > 4B) demonstrates superior performance, namely lower loss, higher accuracy, and higher reward value.

### E. The Construction Pipeline of STEM2Code-Eval Benchmark

In the section, we provide the specific construction pipeline of our STEM2Code-Eval benchmark, as shown in Fig E. The caption prompt is the same as Native Caption Prompt (see Sec.C), and the img+cap2code prompt is foremulated as follows:

##### Img+Cap2Code prompt

You are a top-tier programmer proficient in Python and Matplotlib.

1. Primary Objective Task: Based on an input image and its corresponding detailed image description, generate a standalone, high-quality Python code to reproduce that image. The precise description of the input image is: [description]

a. Core Requirement: Observe the image, and based on the image description you received and the extracted core principles, precisely reproduce the original image.

- (a) Orginal Images

- (b) Code Generate

- (c) Code Agent

(e) Visual Judge

Orginal Image

Criterion

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Style Score (5/5)

Content Score (4/5)

Function Score (4/5)

MathVision MathVista MathVerse WeMath Dynamath

LogicVista

[Figure 102]

STEM2Code-Eval (1k)

[Figure 103]

Repro Image Repro Code

[Figure 104]

[Figure 105]

import matplotlib.pyplot as plt def draw_net(ax):

... ... ... def draw_cuba(ax)

caption Gemini2.5-pro Img+Cap2code Claude-opus-4-1

... ... ... fig = plt.figure(figsize=(10, 6.5)) plt.show()

Top 3k image-code pairs

Ranking： score iter

(d) Visual Feedback

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

scoring

LLM repair Rendering

iter+1 iter+1

Ending condition：

[Figure 113]

- (1) score>90
- (2) iter > 10

re-coding

[Figure 114]

Figure 6. The pipeline of our proposed STEM2Code-Eval Benchmark, which includes 1000 image-code pairs that have been checked and revised by human annotators.

##### Img+Cap2Code prompt (continued)

- b. Visual Matching: Must ensure complete consistency with the following aspects in the description:

- • Coordinate system range and scales
- • Colors, line widths, and styles
- • Element positions and sizes
- • If the description includes axes, ticks, grids, etc., they must be precisely reproduced.

- c. Code Algorithm: The code should not only generate the correct image, but also be written in a clear, structured manner that is intuitively reflected in the final image. The purpose is: An observer should be able to analyze the parameters or logic in the code by merely observing the rendered image, ultimately achieving Image2Code restoration.

successful

###### 2. Output Format & Prohibitions

- • Your entire response must be a single text block with the following structure: Python Code Block: Strictly enclosed by ‘‘‘python ‘‘‘.
- • Code is self-contained with import, one main function, and one main entry point.
- • Code must include plt.show() at the end.

Next, the LLM repair prompt is define as:

##### LLM Repair Prompt

You are a Code Debug Assistant. Your task is to identify and fix issues in the user’s code based on any provided errors, ensuring it works correctly. You will be given the user’s code and a corresponding error message in the following format: ### Error code ‘‘‘python [Error code] ‘‘‘

##### LLM Repair Prompt (continued)

### Error message ‘‘‘text [Error message] ‘‘‘ Your response must strictly adhere to the following criteria:

- • Return ONLY the complete, corrected Python code.
- • The code must be enclosed within a single ‘‘‘python...‘‘‘ block.
- • You must return all the complete, working code, not just the modified part.

Next, the scoring prompt is defined as follows:

##### Image Scoring Prompt

You are an expert judge in evaluating mathematical and geometric diagrams. The first image (reference image) is a ground truth mathematical figure, and the second image (AI-generated image) is created using code generated by an AI assistant. Your task is to score how well the AI-generated image matches the ground truth image.

- 1. Scoring Methodology Task: The AI-generated image’s score is based on the following criteria, totaling a score out of 100 points. The evaluation must consider the mathematical and geometric correctness of the figure, focusing on the precise arrangement and relationships of its components.

- Criterion 1. Geometric & Structural Completeness (30 points):

- • Element Types: Does the AI-generated image include all fundamental element types from the reference image (e.g., points, lines, segments, rays, circles, polygons, curves, coordinate axes, text labels)?
- • Element Quantity: Is the exact number of each element type correct? (e.g., if the reference has 8 points and 3 triangles, does the generated image also have exactly 8 points and 3 triangles?).

- Criterion 2. Positional & Relational Accuracy (30 points):

- • Absolute & Relative Positioning: Are all elements placed at their correct locations? This assesses accuracy within the image’s implicit or explicit coordinate system.
- • Spatial Relationships: Does the image correctly represent all spatial relationships, such as adjacency, intersection, containment, collinearity, and parallelism/perpendicularity?
- • Sequential & Topological Relationships: For figures like graphs or paths, is the sequence of connections correct? Is the overall topological structure preserved?
- • Layering (Z-order): Are overlapping elements stacked in the correct order (e.g., is the shaded region correctly drawn behind the boundary line)?

- Criterion 3. Text & Annotation Fidelity (10 points):

- • Content: Does the AI-generated image include all text and symbolic annotations from the reference? Is the content of the text identical?
- • Positioning & Association: Are annotations placed correctly relative to the geometric elements they describe?
- • Style: Does the style of the text (e.g., font, size, italics for variables, use of mathematical symbols) match the reference?

- Criterion 4. Visual & Stylistic Consistency (20 points):

- • Colors & Fill: Do the colors (stroke, fill) of all elements match the reference? Are shaded regions filled correctly?
- • Line & Marker Styles: Do line styles (e.g., solid, dashed, dotted), line weights, and marker styles (e.g., dots, small circles, crosses) match?
- • Overall Aesthetics: Does the overall appearance, including background color, grid lines, and aspect ratio, match the reference image?

- Criterion 5. Clarity & Legibility (10 points):

- • Is the AI-generated image clear, sharp, and well-rendered?
- • Are there any distracting artifacts, incorrect overlaps, or elements that are difficult to distinguish? Is all text legible?

- 2. Evaluation & Response Format

• Compare the two images head to head and provide a detailed assessment based on the criteria above.

##### Image Scoring Prompt (continued)

• Use the following format for your response to ensure the evaluation is clear and comprehensive.

--Comments:

- - Geometric & Structural Completeness: ${your comment and subscore}
- - Positional & Relational Accuracy: ${your comment and subscore}
- - Text & Annotation Fidelity: ${your comment and subscore}
- - Visual & Stylistic Consistency: ${your comment and subscore}
- - Clarity & Legibility: ${your comment and subscore}

Score: ${your final score out of 100}

- ---

Please use the above format to ensure the evaluation is clear and comprehensive.

Next, the re-coding prompt is defined as:

##### Re-Scoring Prompt

You are an expert in Python data visualization, skilled at accurately recreating scientific charts through visual comparison. You will receive three inputs: [Original Image]: The first image, our target for reproduction, originating from the fields of Science, Technology, Engineering, and Mathematics (STEM). [Current Render]: The second image, a preliminary version generated by the code below. [Current Code]: The Python code that generated the “Current Render”. ‘‘‘python [Current Code] ‘‘‘. Your core task is to identify the differences between the “Current Render” and the “Original Image” in terms of core scientific information and key structures, and then help me correct the code to reproduce the original image as closely as possible.

### F. The Evaluation Implementation of STEM2Code-Eval Benchmark

In this section, we provide a detailed description of the implementation for the three metrics used in our STEM2Code-Eval benchmark: Image Score, Code Score, and Execution Rate (Exec Rate).

#### F.1. Image Score

The Image Score is designed to evaluate the visual fidelity and semantic correctness of the generated image compared to the ground-truth image. We leverage the advanced multi-modal understanding capabilities of Google’s Gemini2.5-Pro for this task. The evaluation process is as follows:

- 1. Input: For each sample, we provide Gemini2.5 Pro with both the ground-truth image and the image generated by the model-under-test’s code.
- 2. Prompting: We use a carefully crafted prompt that instructs the model to act as an expert evaluator. The prompt asks for a comprehensive comparison focusing on key visual elements, data representation accuracy, color schemes, layout, and overall similarity.
- 3. Scoring: The model is instructed to provide a final score on a scale of 0 to 100, where 0 indicates no resemblance and 100 indicates a pixel-perfect match. To ensure reliable parsing, we require the score to be formatted in a specific way (e.g., “Final Score: [score]”).
- 4. Extraction: We programmatically parse the model’s textual response to extract the numerical score for aggregation. We used the same prompt for Gemini 2.5 Pro as described in Sec.E.

#### F.2. Code Score

The Code Score assesses the quality of the generated Python code itself. This evaluation goes beyond mere executability and focuses on correctness, readability, and best practices. We utilize OpenAI’s GPT-4o for its strong code understanding and generation capabilities. The evaluation process is as follows:

- 1. Input: For each sample, we provide GPT-4o with the Python code generated by the model-under-test.
- 2. Prompting: The model is prompted to act as a senior software engineer conducting a code review. The prompt specifies multiple dimensions for evaluation: logical correctness, readability, efficiency, and robustness (e.g., handling of imports).
- 3. Scoring: The model is asked to provide a score for each dimension and then an overall quality score on a scale of 0 to

100. This multi-faceted approach provides a more nuanced assessment.

- 4. Extraction: The final overall score is extracted from the formatted response. The specific prompt used for GPT-4o is as follows:

##### Code Scoring Prompt

You will act as an expert judge, responsible for rigorous visual verification of AI-generated graphics code. Your sole task is to evaluate whether the AI code is completely consistent with the reference code written by human experts in terms of the final rendered visual result. You must ignore technical differences in the code implementation (e.g., algorithms, data structures) and focus on every pixel and geometric detail that goes into rendering the final image.

##### 1. Evaluation Mission and Core Principles

- Criterion 1. Visual Identity: Two pieces of code that render the exact same image should be considered equally valid. Elegance or clumsiness of the implementation is irrelevant to the scoring.
- Criterion 2. Pixel-Level Accuracy: Your evaluation must be accurate down to the pixel level. This includes geometric shape outlines, position, number of elements, relative relationships, and all visual attributes.
- Criterion 3. Objective and Quantitative: All comments must be supported by concrete visual evidence, strictly adhering to the following scoring criteria.
- Criterion 4. Unconditional Evaluation: The evaluation must be performed on any provided AI-generated code, regardless of whether it is empty or incomplete. You must be scored accordingly by applying the standard criteria, which will naturally result in a very low score.

###### 2. Scoring Criteria and Guidelines (100 points) You will be scored based on the following five criteria. Each item is directly related to the final visual presentation.

- Criterion 1. Overall Layout and Visual Attribute Fidelity (20 points)

- • Canvas and Coordinate System: Are the canvas attributes (e.g., aspect ratio, background color) correct? If a grid or coordinate system exists, are its range, scale, and scale consistent with the reference standard?
- • Macro Layout: Is the overall basic framework of the graphic correct? (For example, where is the main subject located on the canvas, and is the overall visual center of gravity consistent?)
- • Color and Style: Are the fill color, stroke color, and opacity of all elements consistent with the reference code? Do the line width, style (solid, dashed, dotted), and cap style (round, square) match?
- • Text and Annotations: If text labels or mathematical annotations exist, are their content, font, size, position, and alignment consistent with the reference code?

- Criterion 2. Quantitative Fidelity (20 points)

- • Element List Verification: Does the AI-generated graphic contain the exact same types and numbers of geometric elements as the reference code? (e.g., 8 polygons, 14 path nodes, 1 mesh).
- • Completeness: Are there any missing or redundant geometric components compared to the reference code?

- Criterion 3. Positioning and Layout Accuracy (30 points)

- • Absolute Coordinate Accuracy: Do the coordinates of all key elements (e.g., polygon vertices, circle centers, path anchor points) precisely match those calculated in the reference code?
- • Relative Position Relationship: Are the spatial arrangement of elements correct? (e.g., A is above and left of B, C and D are horizontally aligned, and a group of elements are arranged in a circular pattern).
- • Alignment and Distribution: Do the elements follow the same alignment (left/right/center) and

##### Code Scoring Prompt (continued)

distribution (uniform/non-uniform) pattern as the reference code?

- Criterion 4. Relationship and Stacking Completeness (20 points)

- • Connectivity and Sequence: If the graph contains paths, networks, or ordered sequences, is the order of connections between nodes **perfectly reproduced**? Are the starting and ending points of lines or paths correct?
- • Spatial Interaction: Are complex relationships between elements (such as adjacency, containment, intersection, and overlap) rendered correctly? Are the shapes and sizes of overlapping areas accurate?
- • Stacking Order (Z-index): When elements overlap, are they stacked in the correct order (i.e., which element is on top and which is on the bottom)?

- Criterion 5. Code Implementation and Quality (10 points)

- • Clarity and Readability: Is the code well-structured and clear? Does it use meaningful variable names and appropriate comments?
- • Correctness and Efficiency: Is the code free of syntactical errors, logical errors, and unnecessary redundancy? Does it effectively use appropriate functions and methods from relevant libraries?
- • Reproducibility: When executed in the correct environment, does the code run correctly and produce the expected complete graph? —

- 3. Evaluation Compare the reference code to the AI code. Provide a detailed evaluation and rating for each criterion, and then calculate a final overall score. You must strictly adhere to the following format for your response. (Highest Priority)

--Comments:

- - Geometric & Structural Completeness: ${your comment and subscore}
- - Positional & Relational Accuracy: ${your comment and subscore}
- - Text & Annotation Fidelity: ${your comment and subscore}
- - Visual & Stylistic Consistency: ${your comment and subscore}
- - Clarity & Legibility: ${your comment and subscore}

Score: ${your final score out of 100}

- ---

Now, give your reference code and AI-generated code in the following format: ### Reference Code ‘‘‘python [Reference Code] ‘‘‘ ### AI-Generated Code ‘‘‘python [AI-Generated Code] ‘‘‘ Please use the above format to ensure the evaluation is clear and comprehensive.

#### F.3. Exec Rate

The Execution Rate (Exec Rate) is a binary metric that measures the direct success rate of the generated code. It is the most objective of the three metrics. A code snippet is considered successful if and only if it executes without errors and produces the intended visual output file.

The automated execution pipeline is implemented as follows:

- 1. Sandboxed Environment: Each code snippet is executed in an isolated and clean sandboxed environment. This ensures that executions do not interfere with each other and provides a consistent environment. The environment is pre-configured with a standard set of Python data science libraries, including numpy, matplotlib, scipy, pandas, etc. To prevent rendering failures, necessary font libraries for handling Chinese characters and icons are also pre-installed.
- 2. Execution: The generated Python code is saved to a file (e.g., run.py) and executed using a Python interpreter. A timeout limit of 120 seconds is imposed to handle cases of infinite loops or excessively long computations.

###### 3. Verification: Success is determined by a two-fold check:

- • The script must complete with an exit code of 0, indicating no runtime errors.
- • The script must generate an image file (e.g., .png, .jpg) in the working directory. The presence of this artifact is checked post-execution.

###### 4. Calculation: The Exec Rate is calculated as the percentage of samples that pass both verification checks: Exec Rate = Number of Successfully Executed Samples

Total Number of Samples × 100

### G. Experiments with 32B

In this section, we continue to validate the effectiveness of our proposed CodePercept method in public six STEM benchmarks and our proposed STEM2Code-Eval benchmark. As shown in Tab.4 and Tab.5, CodePercept gets consistency improvements.

Table 4. Performance comparison of various MLLMs across six STEM reasoning benchmarks. Image Captioner

Benchmark Datasets (%)

Average MathVision MathVista MathVerse DynaMath WeMath LogicVista

###### LLM Solver: Qwen3-30A3-Thinking [55]

KeyeVL1.5-8B [43] 54.11 64.90 49.95 62.37 33.62 45.19 51.69 Intern-S1-8B [3] 51.67 65.70 51.90 63.61 33.43 51.23 52.92 GLM-4.1V-9B [20] 53.75 64.60 54.47 66.17 40.76 51.00 55.13 InternVL3.5-8B [50] 53.32 67.70 53.40 68.12 41.05 51.68 55.88 MiniCPM-V-4.5 [44] 53.15 66.60 57.84 65.44 43.71 52.57 56.55 Qwen2.5-VL-72B [4] 54.14 67.50 55.40 68.28 44.86 52.34 57.09 Qwen3-VL-30A3B-Instruct [36] 53.59 68.00 66.44 71.67 46.10 53.69 59.92 Claude-Opus 4.1-Thinking [1] 59.61 71.10 56.19 73.25 44.86 59.28 60.72 GPT5-Thinking [33] 60.03 65.20 69.56 71.00 54.57 53.02 62.23 Qwen3-VL-235A22B-Instruct 60.43 73.80 70.08 77.39 53.05 59.73 65.74 Gemini2.5-Pro 66.80 74.80 73.47 81.42 60.29 66.44 70.53

Qwen3-VL-4B-Instruct [36] 54.21 67.30 64.59 69.40 46.10 54.14 59.29 CodePercept-4B-S1 57.63+3.4 69.60+2.3 65.59+1.0 71.38+2.0 47.81+1.7 60.40+6.3 62.07+2.8 Qwen3-VL-8B-Instruct [36] 54.37 69.60 63.75 72.19 45.43 56.82 60.36 CodePercept-8B-S1 59.31+5.0 70.20+0.6 66.52+2.8 73.20+1.0 49.14+3.7 61.52+4.7 63.32+3.0 Qwen3-VL-32B-Instruct [36] 58.55 72.20 71.09 75.78 48.00 62.19 64.63 CodePercept-32B-S1 62.27+3.7 72.90+0.7 71.70+0.6 77.41+1.6 54.19+6.2 65.33+3.1 67.30+2.7

###### LLM Solver: Qwen3-235A22-Thinking [55]

Qwen3-VL-4B-Instruct [36] 59.80 69.20 66.39 71.22 48.86 56.82 62.05 CodePercept-4B-S1 64.71+4.9 71.30+2.1 66.73+0.3 72.40+1.2 50.00+1.1 64.65+7.8 64.97+2.9 Qwen3-VL-8B-Instruct [36] 59.67 71.00 63.88 73.69 49.14 58.16 62.59 CodePercept-8B-S1 66.45+6.8 71.40+0.4 67.95+4.1 75.05+1.4 52.29+3.2 62.64+4.5 65.96+3.4 Qwen3-VL-32B-Instruct [36] 62.66 74.00 69.90 75.54 56.48 66.44 67.50 CodePercept-32B-S1 69.96+7.3 75.90+1.9 73.56+3.6 79.50+4.0 57.81+1.3 70.02+3.6 71.13+3.6

Table 5. Performance evaluation on our STEM2Code-Eval with 1k samples. We employ three metrics to assess their performances : (1) Image Scoring measures the visual similarity between the generated and original images; (2) Code Scoring assesses the quality, structure, and correctness of the generated Python code itself; (3) Exec Rate reports the execution success rate.

Image Score

Code Score

Exec Rate

Model

Avg

Intern-S1-8B [3] 6.02 15.87 26.60 16.16 InternVL3.5-8B [50] 13.50 18.16 56.50 29.38 MiniCPM-V-4.5 [44] 13.91 23.69 50.80 29.47 MiMo-VL-7B-RL [54] 14.54 22.41 60.30 32.42 Ovis2.5-9B [29] 9.76 11.26 89.40 36.81 KeyeVL1.5-8B [43] 20.33 22.47 73.40 38.73 GLM-4.1V-9B [20] 21.19 26.51 72.00 39.90 Qwen3-VL-4B-Thinking [36] 25.38 34.53 75.70 45.20 Qwen2.5-VL-72B-Instruct [4] 32.82 25.83 86.30 48.32 Qwen3-VL-8B-Thinking [36] 29.82 41.71 78.90 50.14 Qwen3-VL-30A3B-Instruct [36] 33.05 31.04 87.50 50.53 Seed1.6-Vision-nothinking [18] 31.22 38.56 85.50 51.76 Qwen3-VL-30A3B-Thinking [36] 37.47 35.53 87.10 53.37 Qwen3-VL-Plus-Instruct [36] 45.94 40.40 90.00 58.78 Qwen3-VL-Plus-Thinking [36] 45.59 40.61 89.20 58.47 Seed1.6-Vision-Thinking [18] 42.03 40.74 94.70 59.15 Gemini2.5-Flash-Thinking 57.25 60.87 85.20 67.77 Claude-Opus 4.1-Thinking 55.90 56.19 97.10 69.73 GPT5-Thinking 64.97 64.98 96.60 75.52 Gemini2.5-Pro-Thinking 68.89 75.41 91.70 78.67

Qwen3-VL-4B-Instruct [36] 24.55 26.42 79.40 43.46 CodePercept-4B-S1 38.13 43.43 80.70 54.09 CodePercept-4B-R1 47.17 45.86 91.30 61.44

Qwen3-VL-8B-Instruct [36] 28.59 28.23 85.30 47.37 CodePercept-8B-S1 44.53 46.78 87.60 59.64 CodePercept-8B-R1 50.25 47.04 93.40 63.56

Qwen3-VL-32B-Instruct [36] 36.85 39.98 81.80 52.88 CodePercept-32B-S1 61.14 56.99 93.00 70.38 CodePercept-32B-R1 68.97 62.53 95.90 75.80

