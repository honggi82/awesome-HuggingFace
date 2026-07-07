# arXiv:2602.11731v2[cs.CL]29Apr2026

[Figure 1]

## Thinking with Drafting: Optical Decompression via Logical Reconstruction

###### Jingxuan Wei1,2,5,∗, Honghao He1,2,∗, Caijun Jia1,2, Siyuan Li3, Zheng Sun1,2, Yuhang Xu1,2, Yuanyuan Lin3, Linzhuang Sun1,2, Yuchen Wu3, Bihui Yu1,2, Xiangxiang Zhang3,†, Cheng Tan4,†

1Shenyang Institute of Computing Technology, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3ByteDance 4Westlake University 5Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Shandong Computer Science Center (National Supercomputer Center in Jinan), Qilu University of Technology (Shandong Academy of Sciences)

∗Equal contribution, †Corresponding authors

###### Abstract

Existing multimodal large language models have achieved high-fidelity visual perception and exploratory visual generation. However, a precision paradox persists in complex reasoning tasks: optical perception systems transcribe symbols without capturing logical topology, while pixel-based generative models produce visual artifacts lacking mathematical exactness. To bridge this gap, we propose that reasoning over visual inputs be reconceptualized as optical decompression—the process of reconstructing latent logical structures from compressed visual tokens. Guided by the axiom that Parsing is Reasoning, we introduce Thinking with Drafting (TwD), which utilizes a minimalist Domain-Specific Language (DSL) as a grounding intermediate representation. Unlike standard approaches that hallucinate answers directly, TwD forces the model to draft its mental model into executable code, rendering deterministic visual proofs for self-verification. To validate this, we present VisAlg, a visual algebra benchmark. Experiments demonstrate that TwD serve as a superior cognitive scaffold. Our work establishes a closed-loop system where visual generation acts not as a creative output but as a logical verifier, offering a generalizable path for visual reasoning.

Date: January 5, 2026 Correspondence: tancheng@pjlab.org.cn, zhangxiangxiang.zxx@bytedance.com

###### 1 Introduction

Recent advances in multimodal large language models (MLLMs) mark a decisive shift in artificial intelligence from passive perception toward active cognitive interaction [2, 18, 24, 37]. On the input side, optical character recognition (OCR) systems have undergone a dramatic evolution. Modern approaches—exemplified by largescale vision-language models trained for document understanding—are now capable of faithfully transcribing complex visual artifacts [13, 20, 32, 34], including dense text, structured layouts, tables, and mathematical formulas. This progress effectively realizes what may be termed contextual optical compression [35]: rich

visual documents are compressed into high-fidelity internal representations, enabling machines to read with unprecedented accuracy.

[Figure 2]

- Figure 1 Illustration of paradigms. (a) Existing multimodal paradigms treat image understanding, textual reasoning, and visual generation as disconnected tasks. (b) Thinking with Drafting (TwD) reframes visual reasoning as logical reconstruction into a minimalist DSL. Concurrently, progress on the output side has given rise to a complementary paradigm often referred to as Thinking with images [10, 30]. Rather than relying solely on a textual chain-of-thought, recent models increasingly generate visual artifacts like diagrams, sketches, or intermediate images as part of the reasoning process. By externalizing cognition into visual form, these methods aim to mirror a fundamental aspect of human problem solving, where drawing and visualization serve as tools for thought [27, 43]. Taken together, these trends suggest that modern systems are approaching a read–draw loop [26, 29], with perception supplying faithful inputs and generation enabling visualized intermediate states.

Despite this apparent completeness, a critical gap remains when these systems are applied to tasks requiring strict logical precision. This gap manifests as a precision paradox. On the one hand, OCR systems excel at transcription: they can reliably extract symbols, numbers, and text spans from images. However, transcription alone does not capture logical topology. A numeral such as “123” may represent a total, a difference, or a constraint, depending on context. While the perceptual signal is high-fidelity, the relational semantics remain implicit and unstructured. OCR systems are designed to recognize symbols, represent the logical relations that govern them. On the other hand, visual generation models optimize perceptual plausibility rather than logical validity [40]. They can generate images that resemble diagrams or mathematical constructions without guaranteeing that the underlying relations are exact. A generated line segment may appear longer than another, yet fail to satisfy a precise quantitative ratio.

To bridge this divide, we argue that reasoning over visual inputs must be reconceptualized as a process of optical decompression [16, 28]. If OCR compresses the visual world into perceptual tokens, then reasoning is the act of reconstructing the latent logical structure encoded within those tokens. From this perspective, understanding does not hinge on producing fluent textual explanations, but on recovering an explicit, executable representation of entities, relations, and constraints. This leads to our central axiom: Parsing is Reasoning. True comprehension arises only when a model can translate ambiguous natural language and visual cues into a structured form.

We materialize this philosophy through the Thinking with Drafting (TwD) paradigm. Taking the Singapore bar model—a canonical representation of visual algebra—as our primary testbed, we introduce a minimalist geometric DSL (Domain-Specific Language). This DSL occupies a unique strategic niche: it serves as an intermediary between the ambiguity of natural language, the syntactic noise of general-purpose code, and the rigidity of geometric axioms. This DSL is designed for interoperability; it can be compiled into GeoGebra scripts for mathematical validation or SVG code for visual rendering. The generated draft serves not merely as a visualization, but as a deterministic visual verifier, enabling the system to detect logical conflicts and self-correct. Within TwD, drafting is not treated as a final output but as a deterministic visual verifier, enabling a closed logical–visual loop in which reconstruction, verification, and correction are tightly coupled.

###### 2 Related Work

###### 2.1 Optical Perception

Recent advancements in Optical Character Recognition (OCR) [13, 23, 34] and Vision Language Models (VLMs) [6, 12, 19, 38] have fundamentally transformed the landscape of document understanding. Traditional approaches have evolved to recover high-fidelity text content while preserving complex contextual structures such as layouts, tables, and formulas [11, 41]. Notably, recent works like DeepSeek-OCR [35] demonstrate the feasibility of contexts optical compression, proving that pixels can serve as an efficient compression medium for textual information.

However, optical perception alone is insufficient for tasks that require rigorous logical consistency like mathematical problem solving [15, 25, 31]. Current unstructured outputs may capture the document’s visual syntax but neglect its underlying logic, leaving entities and quantitative relations implicit and ungrounded. We argue that reliable reasoning requires a shift from transcription accuracy to logical reconstruction. Unlike standard perception tasks, our approach transforms raw perception into a verifiable intermediate representation, thereby enabling the Thinking with Drafting paradigm to operate on grounded logical structures.

###### 2.2 Visual Reasoning

While optical perception digitizes the input, reasoning requires manipulating digitized concepts to derive solutions. The dominant paradigm relies on LLMs to perform reasoning via textual generation, exemplified by Chain-of-Thought (CoT) [21, 36] and Program-of-Thought (PoT) [7, 14]. These methods decompose complex problems into step-by-step deductions or executable code snippets. Conversely, vision-centric approaches attempt to solve reasoning tasks directly in the pixel space [8, 39, 42]. Recent works such as Vision-ARC [17] demonstrate that certain abstract reasoning tasks are more naturally formulated as image-to-image translation problems.

Despite their efficacy, these models often struggle with semantic grounding—specifically, translating complex natural language constraints into geometric artifacts. We propose Thinking with Drafting to bridge the gap between implicit semantic thought and explicit visual verification. By parsing visual text into a structured intermediate representation, the model drafts its understanding into a rule-constrained canvas. It creates an optical decompression loop: implicit logical relations are decompressed into explicit visual structures.

VLM (Qwen3-VL)

Product Ready Standard

Rendered Image

[Figure 3]

DSL Intermediate Representation

ü Information Completeness Check

SVG Plotting Code

One-pass Rendering

ü Style Consistency Check

Screenshot

HL VL HB/VB

|MOM Bunny|
|---|

ü Compliance Check ü Alignment Check

Define Objects

Reward Threshold: 1.0

|MOM Bunny|
|---|

[Figure 4]

- Block A Define object bunny_bar (label: "Bunny", length: 3) define object mom_bar (label: "MOM", length: 33)

- Block B Encode status bunny_bar = 3 6 Encode status mom_bar =9 9 9 -6 Encode status bunny_bar =

1/3 mom_bar = 9

- Block C add given value: 3; get 6; give 6; 3 times the bunny add question: MOM original num?

|Object Logic<br><br>HL "Bunny" 3 HL "Mom" 33|
|---|

###### Question:

###### Score=Threshold?

A little bunny pulled 3 carrots. Her mom said: "If I give you 6 carrots, the number of carrots I have left will be 3 times the number you have." How many carrots did her mom pull?

Yes

###### No

|MOM<br><br>Origin?<br><br>Bunny<br><br>get 6<br><br>3<br><br>3 times the bunny<br><br>give 6|
|---|

96% correlation

[Figure 5]

Add Relation

|Relations<br><br>Cacl bunny_now = 3+6 Cacl mom_now = 33-6<br><br>bunny_now = 1/3 mom_now|
|---|

[Figure 6]

[Figure 7]

|Bunny<br><br>MOM<br><br>total 33|
|---|

Add Information

|Given value Question|
|---|

Human Evaluation

High-quality Discard SFT Dataset

(a) Thinking with Drafting (Generation) (b) Data pipeline with DSL Verifier

- Figure 2 Overview of Thinking with Drafting framework. (a) Optical decompression generates a Logic Graphic DSL from visual input and OCR, comprising entity, relational, and aggregation primitives. (b) A verifier scores samples by syntactic validity, visual completeness, and logical consistency, retaining high-quality data for training and discarding the rest to ensure topological and geometric correctness.

###### 3 Method

###### 3.1 Preliminaries

We consider the problem of multimodal mathematical reasoning where a model is presented with a visual input I (containing visual text, layout, and geometry) and a natural language query Q. The objective is to derive a correct final answer a ∈ A. Unlike standard end-to-end approaches that map (I,Q) → a directly, we formalize Thinking with Drafting as a multi-stage iterative generation process involving a structured intermediate representation. Let T denotes the space of unstructured natural language and S denotes the space of DSL, which represents geometric and logical constraints. We define the reasoning process Pθ, parameterized by a MLLM, as a probabilistic mapping from perception to logical reconstruction.

To underscore the theoretical distinctiveness of TwD, we contrast our formulation with three dominant paradigms: text-only CoT, thinking with images, and traditional OCR.

Text-only CoT Standard Multimodal CoT approaches rely exclusively on the linguistic space T to bridge the input and output:

tˆcot ∼ Pθ(t,I,Q,), aˆ ∼ Pθ(a|I,Q,tˆcot), (1)

where tˆcot ∈ T is a linear sequence of tokens. The fundamental limitation of CoT is that natural language is ambiguous and lacks strict geometric constraints. In contrast, our DSL space S enforces logical rigidity; a defined entity in S must satisfy explicit geometric rules, acting as a regularizer for the reasoning process.

Thinking with Images Emerging “Thinking with Images” paradigms utilize a generative model to produce an intermediate image Iˆgen:

aˆ ∼ Pθ(a | I,Q,Iˆgen) (2)

While Iˆgen provides visual feedback, it operates in the pixel space, which suffers from stochastic imprecision. A model may generate a diagram that perceptually plausible but mathematically inaccurate. TwD, conversely, employs programmatic drafting. Our intermediate representation sˆ is symbolic code. The rendered output is mathematically exact, ensuring reliable verification.

OCR focuses on transcription fidelity, mapping the visual input to a sequence of characters: Seq ∼ Pθ(Seq | I), (3)

OCR addresses the question “What is written?”, whereas TwD addresses “What does it mean?”. OCR extracts the syntax but leaves the semantics implicit. TwD performs logical reconstruction, upgrading the task from transcription to parsing. By mapping I → S, we explicitly capture the logical topology that OCR ignores, thereby converting raw pixels into actionable reasoning primitives.

###### 3.2 The Logic Graphic DSL

To instantiate the principle that Parsing is Reasoning, we formally define the structure of our DSL space, S. A statement s ∈ S is not a sequence of natural language tokens, but a structured composition of atomic reasoning primitives. Unlike general-purpose plotting languages that prioritize pixel-level control, the grammar of S is designed to abstract away rendering redundancies and expose the bare logical topology of the problem. The DSL consists of three fundamental operator categories:

Entity Primitives (HL) These represent the physical quantities or objects from the input I as horizontal line segments. A key innovation in our design is the status-aware segmentation. We define a segment sequence vector v = [v1,v2,...,vn] where |vi| denotes length. Crucially, we utilize the sign of vi to encode existential status: vi > 0 renders a solid line (existing quantity), while vi < 0 renders a dashed line (process quantity, e.g., subtracted part or hypothetical extension). This allows the model Pθ to generate a compact representation for complex change models.

Relational Primitives (VL) In bar models, logic is primarily defined by geometric alignment. The Vertical Line (VL) operator explicitly encodes relational equality between horizontal entities. Parameterized by an explicit x-coordinate and row indices, it functions as an equality constraint, asserting that specified segments coincide at a shared value. This compels the model to perform alignment reasoning, identifying shared semantic boundaries rather than treating coordinates as independent variables.

Aggregation Primitives (HB/VB) To ground abstract arithmetic operations into geometry, we employ Horizontal (HB) and Vertical (VB) Braces. An HB operator encapsulates a part-whole relationship within a single entity, while a VB represents summation or comparison across multiple entities.

###### 3.3 Topological Abstraction and Rendering

A major bottleneck in generating visual code is the high entropy of continuous coordinate spaces. To mitigate this, we introduce a Topological Abstraction layer that decouples logical reasoning from metric rendering.

Virtual Grid System We map the continuous canvas R2 to a discrete logic space Z2. We define a virtual grid where the y-axis is discretized into logical rows and the x-axis is governed by relative offsets rather than absolute pixels. The model generates code relative to this grid. For instance, creating a new entity involves assigning it to a new row_id rather than calculating a pixel offset. It ensures layout invariance: the model focuses solely on the logical ordering and grouping of entities.

Deterministic Rendering The mapping from a syntactically correct DSL statement to a visual verification image V is executed by a deterministic rendering engine: V = Render(s). We introduce common topological patterns into semantic macros. For example, a comparison pattern macro automatically generates the difference brace and alignment lines when the model detects a more than relation. These macros ensure that correct logical parsing always yields a visually canonical diagram.

###### 3.4 Thinking with Drafting

Building upon the structured space S and the deterministic renderer, we instantiate the TwD framework in Figure 2 as a sequential generation-verification process.

Optical Decompression via Logical Parsing In the first stage, the model acts as a parser. It perceives the raw input I and attempts to decompress the implicit logical topology into an explicit structural draft. This yields a preliminary textual explanation tˆ and an initial draft sˆ:

(t,ˆ sˆ) ∼ Pθ(t,s|I,Q), (4)

Crucially, the generation of sˆ is not a single step, but a step-by-step decomposition of the problem. It embodies our axiom that Parsing is Reasoning: the generation of s forces the model to resolve ambiguities in I into discrete logical atoms.

Drafting and DSL-Conditioned Inference The generated hypothesis sˆ is passed to the rendering engine to produce the verification drafting image V. It provides an explicit visual proof of the model’s internal reasoning for human verification. The model derives the solution conditioned on the structured draft sˆ it constructed. Unlike standard Chain-of-Thought which relies on ambiguous natural language, the drafting sˆ acts as a logical context:

In the second stage, the model utilizes the output from the previous stage as a “drafting context.” The initial draft sˆ1 acts as an externalized cognitive scaffold, allowing the model to inspect its own reasoning. The model generates a refined explanation tˆ2, a completed DSL sˆ2, and the final answer aˆ:

aˆ ∼ Pθ(t,s,a|I,Q,t,ˆ sˆ). (5)

By grounding the reasoning in sˆ, the calculations are guided by the explicit topology defined in the draft. The TwD paradigm thus posits that the act of constructing the draft is the reasoning engine itself, ensuring the final answer is a derivative of a verified logical structure.

###### 4 Dataset

###### Step1: Data Draft Generation Step2: Data Refining

###### Product Ready Standard

Step3: Scoring & Filtering

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Data Draft

Question: There are two wires of the same length. The first wire was sold for 36 meters, and the second ……of the second wire is twice the remaining length of the first wire. How long were the two wires originally?

Alignment Checked

Refined Dataset

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

LLM Judge

Human Judge

Prompt

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

DSL Syntax Check HL "First" "line" 0 12

Information Completeness Verified

[Figure 25]

[Figure 26]

[Figure 27]

Analysis: Problem type: …… Key information: …… Objective: ……

[Figure 28]

[Figure 29]

Answer: Analysis DSL

Match ≤ 96%

[Figure 30]

Analysis Check

[Figure 31]

Pick 1000 piece of data. Continuously refine the prompt until the LLM's

…Objects: Single object…

[Figure 32]

[Figure 33]

[Figure 34]

Compliance Reviewed

accuracy rate matches that of human evaluators by 96%. Then use LLM to judge the entire dataset.

[Figure 35]

[Figure 36]

[Figure 37]

Graph Style Check

[Figure 38]

###### DSL:

|Retain only Full Marks Data|
|---|

HL "First" 0 12 -24 -12 HL "Second" 1 12 12 -24 …… HB "24m" N 1 24 48 HB "12m" S 0 24 36

[Figure 39]

[Figure 40]

Style Consistency Validated

[Figure 41]

[Figure 42]

Refined Dataset

[Figure 43]

[Figure 44]

[Figure 45]

Product Ready Dataset

Figure 3 The benchmark data construction pipeline of VisAlg.

We introduce VisAlg, a benchmark for evaluating logic-aware visual reasoning by assessing whether a system can recover the explicit logical topology underlying visual algebra problems through optical decompression. Each instance pairs a image of a natural language algebra problem with a structured intermediate representation, an executable bar-model DSL that defines the ground-truth logical parse. VisAlg is constructed through a multi-stage pipeline, as shown in Figure 3.

###### 4.1 Dataset Construction

Drafting data generation. We collect 15,000 bar-model word problems from public datasets and websites, covering common visual algebra patterns. For each problem, we prompt Gemini-2.5-Pro [12] to produce a synchronized draft with two components. The first component is a textual analysis that explicitly parses the problem schema, and the second component is a program written in our DSL. The detailed prompt is

- provided in Appendix A.1.

Data refining. Initial drafts frequently fail to meet verifiability requirements. We therefore introduce a checklist refinement stage in which the model revises each draft through three sequential checks: (1) Syntax check, ensuring the grammar is correct and executable; (2) Analysis check, verifying that all objects, quantities, relations, and targets identified in the analysis are consistently instantiated; (3) Style check, enforcing canonical bar-model layout conventions such as boundary placement and cross-row alignment. All corrected instances are stored. The detailed prompt is in Appendix A.2.

Scoring and filtering. We employ an LLM-based judge calibrated with expert evaluations to filter the dataset. A domain expert scores 1,000 instances using a fixed rubric, and the judge prompt is iteratively refined until achieving 96% agreement. The calibrated judge is then applied to the full dataset, retaining only full-score instances, resulting in 11,372 product-ready instances. Details are provided in Appendix A.3 and A.4.

Product ready. The final filtering enforces four criteria: geometric alignment, semantic completeness, representational compliance, and stylistic consistency. (1) Alignment: Horizontal bracket endpoints must coincide with boundary coordinates defined by cumulative segment lengths; vertical links must align with these boundaries across spanned rows. (2) Completeness: All stated quantities, relations, and targets must appear explicitly in labels; unknowns may be denoted by “?”. (3) Compliance: Vertical brackets represent only multi-object aggregates, and vertical links are allowed solely at cross-row shared partition points. (4) Consistency: Transfers follow a paired −t/ + t pattern across two rows; post-transfer equality is indicated by a shared boundary via a vertical link.

###### 4.2 Dataset Analysis

Category. VisAlg focuses on optical decompression in bar-model reasoning, emphasizing recovery of logical topology over surface symbol transcription. We analyze five canonical schemas: proportional distribution, rate & percentage, change & revert, sum & split, and difference analysis. Figure 4 shows the joint composition of difficulty (inner ring) and schema (outer ring). As summarized in Table 1, proportional distribution and rate & percentage form the two dominant schema groups. In terms of difficulty, medium accounts for 72.9% of instances, with easy (13.4%) and hard (13.7%) providing balanced coverage of both basic parsing and constraint-dense cases.

Table 1 Corpus-level statistics of VisAlg.

0.55%

0.29%

0.49%

Train Test Problem schemas

3.17%

5.36%

5.05%

2.91%

3.17% 2.21%

3.97%

Proportional Distribution 4,265 245 Rate & Percentage 2,771 265 Change & Revert 1,635 119 Difference Analysis 905 141 Sum & Split 854 172

Hard

Easy

6.44%

13.03%

14.14%

5.56%

Easy Medium Hard Proportional Distribution Rate & Percentage Change & Revert Sum & Split Difference Analysis

30.33%

Medium

Difficulty levels

11.77%

72.83%

Easy 1,400 208 Medium 7,602 680 Hard 1,428 54

18.74%

Operation length

≤ 3 operations 1,834 163

- 4 operations 2,490 279
- 5 operations 2,693 296
- 6 operations 1,612 115
- 7 operations 1,002 56 ≥ 8 operations 799 33 Total instances 10,430 942

- Figure 4 Difficulty and schema composition in VisAlg.

Reasoning depth. Logical complexity is measured by the number of bar-model operations needed for reconstruction. As shown in Table 1, only 17.6% of training instances require three or fewer operations, while most fall in the four–six range. A non-trivial 7.7% require eight or more operations, reflecting long dependency chains and multi-step reasoning.

Scale and split consistency. The benchmark comprises 10,430 training and 942 test instances, with additional curated splits for fine-tuning, preference optimization, and evaluation. The test set mirrors the training distribution in schema and difficulty, ensuring evaluation emphasizes structural generalization rather than distributional shift.

###### 4.3 Evaluation Metrics

Objective metrics. Consistency is evaluated at both code and image levels. Code similarity is measured using BLEU, ROUGE-L, and chrF, with chrF as the primary metric due to its robustness to mixed symbols, numbers, and text in the DSL. Image similarity is assessed using PSNR, SSIM, and LPIPS, with SSIM prioritized for its sensitivity to structural topology and edge continuity.

Subjective metrics via LLM-as-judge. An LLM-based verifier scores outputs on five dimensions: structural alignment, information coverage, numerical consistency, semantic compliance, and answer leakage. Each is rated in [0,1], with the final subjective score given by their mean.

Main score. Main results are reported using a composite score: Score = 13 chrF + SSIM + LLMjudge , which jointly reflects code-level consistency, image-level structural fidelity, and semantic normative correctness.

Human evaluation. We additionally conduct a human evaluation of DSL quality; the criteria and protocol are

- provided in Appendix B.

###### 5 Experiment

Table 2 Main results on VisAlg. Align: structural alignment; Cover: information coverage; Num: numerical consistency; Norm: semantic compliance; Leak: answer leakage. Detailed descriptions of them are in Appendix B.3.

Code Similarity Image Similarity Verification Scores BLEU ROUGE-L chrF LPIPS SSIM PSNR Align Cover Num Norm Leak Avg. Overall

Model

InternVL3-8B [44] 9.93 48.51 37.57 32.64 82.70 24.25 0.31 0.56 0.32 0.15 0.89 44.69 54.99 InternVL2.5-8B [9] 9.12 46.38 48.41 51.10 58.97 17.22 0.32 0.44 0.36 0.14 0.68 38.73 48.70 Intern-S1-mini [4] 8.41 36.47 22.68 59.52 49.32 14.65 0.57 0.26 0.87 0.36 0.97 60.39 44.13 Mimo-VL-7B-RL [22] 10.36 46.17 33.43 79.47 25.87 7.75 0.38 0.48 0.76 0.23 0.85 54.05 37.78 Qwen3-VL-8B [5] 6.65 39.04 23.94 83.69 20.10 0.00 0.60 0.16 0.84 0.29 1.00 57.80 33.95

Gemini-3-Pro [33] 30.18 59.06 57.53 18.23 90.36 27.32 0.97 0.95 0.94 0.78 0.96 91.98 79.96 Gemini-2.5-Pro [12] 28.94 58.25 57.43 18.12 89.97 26.92 0.95 0.76 0.99 0.73 0.32 74.97 74.12 Claude-4 [3] 28.66 58.54 57.17 18.16 89.97 26.88 0.94 0.74 0.99 0.72 0.30 73.71 73.62 GPT-5.1 [1] 22.93 56.13 51.23 25.86 86.89 25.70 0.77 0.67 0.95 0.51 0.19 61.69 66.60 GPT-4o [19] 16.24 50.64 35.73 24.49 89.15 26.59 0.50 0.42 0.83 0.31 0.72 55.44 60.11

TwD (Ours) 48.23 72.22 68.29 11.97 93.68 30.25 0.90 0.96 0.70 0.73 1.00 85.91 82.63

###### 5.1 Experimental Setup

We evaluate VisAlg against state-of-the-art MLLMs. The proprietary models include GPT-5.1 [1], GPT-4o [19], Claude-4 [3], Gemini-3 [33], and Gemini-2.5-Pro [12], representing the current upper bound of general-purpose multimodal reasoning. For open-weight baselines, we consider InternVL3-8B [44], InternVL2.5-8B [9], InternS1-mini [4], Mimo-VL-7B-RL [22], and Qwen3-VL-8B [5]. Our model is initialized from Qwen3-VL-8B and supervised fine-tuned on the training split, enabling parameter-efficient comparison with open-weight peers while treating proprietary models as upper bounds. SFT is conducted on a 8-GPU node with a visual token cap of 2,048 and a maximum sequence length of 5,128. We train for 2 epochs using a learning rate of 5 × 10−6 and a warmup ratio of 0.05.

###### 5.2 Main Results

Table 2 reports the main results on VisAlg across code similarity, image similarity, and verifier-based evaluation. Our model, initialized from Qwen3-VL-8B and supervised on VisAlg, achieves the highest overall score of 82.63, surpassing all open-weight baselines and outperforming the strongest proprietary models, including Gemini3-Pro [33] (79.96) and Gemini-2.5-Pro [12] (74.12). This highlights the importance of explicit supervision on logic reconstruction for verifiable bar-model reasoning. A clear performance gap is observed between open-weight and proprietary systems. Open-weight models such as InternVL3-8B [44], InternVL2.5-8B [9], Intern-S1-mini [4], Mimo-VL-7B-RL [22], and Qwen3-VL-8B [5] score below 55, with weaknesses in code fidelity and diagram reconstruction, indicating difficulty in generating syntactically valid and topologically consistent DSL programs without task-specific alignment.

Our gains primarily arise from improved structural fidelity. The model leads in code and diagram alignment, achieves strong information coverage, and avoids answer leakage. Relative to top proprietary models, the remaining gap is mainly in numerical consistency, while structural legality and semantic completeness are largely preserved.

###### 5.3 Results by Visual Algebra Schema

- Figure 5 reports schema-wise performance across five visual algebra types. Our TwD consistently achieve competitive performance compared to both open-weight and proprietary baselines across all schemas. The gains are most pronounced on structure-intensive schemas such as proportional distribution and difference analysis, where accurate multi-segment decomposition and boundary-aligned comparison are critical. While

- 84.08 82.15 74.18 73.97 64.32 55.79 57.74 52.73 53.27 57.23 57.71
- 85.77 82.03 76.98 76.13 71.20 65.23 58.30 54.40 53.82 60.70 57.09

Difference Analysis

86.69

Sum & Split

Change & Revert

77.50 78.47 72.38 72.74 65.37 57.22 56.22 53.77 54.40 58.54 54.76

Rate & Percentage

78.56 77.43 72.17 71.27 65.04 59.01 56.06 52.99 55.19 59.04 56.06

86.69 80.65 75.37 74.84 67.79 62.77 57.04 54.76 55.14 60.81 57.63

Proportional Distribution

52.73

TwDGemini-3 Gemini-2.5-proClaude-4

GPT-5.1 GPT-4oInternVL-3

InternVL-2.5 Intern-S1-mini MiMo-VL Qwen3-VL

Figure 5 Schema-wise performance comparison across five visual algebra problem types.

proprietary models achieve competitive results, their performance varies noticeably across schemas. In contrast, TwD remains uniformly strong across problem types, supporting the claim that optical decompression benefits from explicit, verifiable logic.

###### 5.4 Alignment with Human Expert

- Figure 6 shows a strong correlation between expert human ratings and verifier-based VisAlg scores r = 0.9575, validating the verifier as a reliable proxy for human judgment in visual algebra reasoning. Model rankings are largely preserved across the full performance range. TwD remains top-ranked under both evaluations, indicating that the reported gains reflect genuine improvements in structural correctness rather than metric artifacts.

[Figure 46]

gpt绘的图虽然在集合总量层⾯看似⾃洽 （60/50/40 都能凑出来），但在语义表达上 发⽣了严重偏移：它把题⼲中“交集数值及其 嵌套约束”错误地落到了不可能的坐标区间 （越界）和错误的区间⻓度（如把 A∩C=15A\cap C=15A∩C=15 画成 35），导致 交集关系在图中既不可读出也不可验证，属 于结构性错误。这反映了模型在“计算得到交 集数”与“⽤ HSIR 保持交集—⼦交集约束”的映 射上缺乏⼀致性校验，出现了典型的“数值能 算、结构乱画”的割裂。

LLM Score vs Human Score

Question: Given a universal set U with 100 elements, set A has 60 elements,……, and A∩B∩C has 5 elements. Find the number of elements in A∪B∪C, and the number of elements in U that do not belong to A∪B∪C.

100

r=0.9575

90

Gemini-3

TwD

80

###### Ground Truth: A∪B∪C=90, the others = 10

Claude-4 GPT-5.1

HumanScore

Gemini-2.5-pro

###### TwD

70

Oper 1~4. Represent objects

Oper 5~9. Value comments

Intern-S1-mini

60

GPT-4o

[Figure 47]

[Figure 48]

Qwen3-VL-Instruct

50

MiMo-VL

40

Oper 10. Question comments

InternVL-3

###### Answer

[Figure 49]

A∪B∪C has 90 elements. There’re 10 elements in U that do not belong to either A, B or C.

30

30 40 50 60 70 80 90 100

LLM Score

Figure 6 Correlation between verifier-based VisAlg scores and human expert ratings.

[Figure 50]

###### GPT-5.2

Oper 1~3. Represent objects

Oper 4~6. Align comments

[Figure 51]

[Figure 52]

Oper 7~10. Value comments Answer

[Figure 53]

|A∪B∪C| = 85 |"! A∪B∪C | = 15

Figure 7 Generalization to set-theoretic reasoning.

###### 5.5 Generalize to Complex Logical Topology

We extend our evaluation to advanced set-theoretic reasoning tasks involving multi-set constraints. As shown in Figure 7, these tasks require the model to manage high-order intersections and nested boolean boundaries. Frontier MLLMs like GPT-5 [1] often exhibit topological hallucination in this regime. While they may attempt to align segments visually, they fail to preserve the strict boolean logic of overlaps. The model cannot distinctively ground intersections A ∩ C and A ∩ B ∩ C, violating containment and alignment constraints and rendering the graphic unreadable and unverifiable. This calculation–construction gap highlights that correct arithmetic does not guarantee preservation of global structural invariants such as boundary legality and consistency. TwD successfully decomposes the abstract set problem into sequential geometric operations. By explicitly rendering the atomic intersections, TwD effectively visualizes the algebra of sets. Additional case studies are provided in Appendix D.

###### 6 Conclusion

In this work, we addressed the precision paradox in multimodal reasoning, where systems achieve high perceptual fidelity yet fail to preserve rigorous logical topology. We formalized this challenge through the lens of optical decompression, introducing VisAlg benchmark to evaluate whether models can reconstruct latent logical structures into verifiable artifacts. To bridge the gap between perception and reasoning, we established Thinking with Drafting paradigm that enforces structural invariants via a minimalist graphic DSL. Experiments demonstrate that that a compact 8B model, when equipped with the TwD cognitive scaffold, outperforms leading proprietary frontiers on visual algebra problems. By closing this loop, we show that explicit structural drafting acts as a necessary foundation for trustworthy multimodal intelligence.

###### Limitations

The core limitation lies in the scope of structural representation: the DSL is intentionally designed around bar-model visual algebra, emphasizing linear topological relations to enable intuitive structural supervision. Extending this DSL to support broader classes of scientific diagrams remains an important direction for future research.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

- [3] Anthropic. Claude sonnet 4.5 system card. Technical report, Anthropic PBC, 2025. Official system card describing Claude Sonnet 4.5 capabilities and safety evaluation. Available at: https://assets.anthropic.com/ m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf.
- [4] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025.

- [5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025.
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [7] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research.

- [8] Yang Chen, Yufan Shen, Wenxuan Huang, Sheng Zhou, Qunshu Lin, Xinyu Cai, Zhi Yu, Jiajun Bu, Botian Shi, and Yu Qiao. Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. arXiv preprint arXiv:2507.20766, 2025.

- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

- [10] Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, and Pengfei Liu. Thinking with generated images. arXiv preprint arXiv:2505.22525, 2025.

- [11] Kateryna Chumachenko, Amala Sanjay Deshmukh, Jarno Seppanen, Ilia Karmanov, Chia-Chih Chen, Lukas Voegtle, Philipp Fischer, Marek Wawrzos, Saeid Motiian, Roman Ageev, et al. Nvidia nemotron parse 1.1. arXiv preprint arXiv:2511.20478, 2025.

- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [13] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.

- [14] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

- [15] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14953–14962, 2023.

- [16] Joy Hsu, Jiayuan Mao, and Jiajun Wu. Ns3d: Neuro-symbolic grounding of 3d objects and relations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2614–2623, 2023.

- [17] Keya Hu, Ali Cy, Linlu Qiu, Xiaoman Delores Ding, Runqian Wang, Yeyin Eva Zhu, Jacob Andreas, and Kaiming He. Arc is a vision problem! arXiv preprint arXiv:2511.14761, 2025.

- [18] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 36:72096–72109, 2023.

- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [20] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer, 2022.

- [21] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

- [22] Jiaze Li, Jingyang Chen, Yuxun Qu, Jianzhong Ju, Zhenbo Luo, Jian Luan, Shijie Xu, Zhenru Lin, Junyou Zhu, Boshen Xu, et al. Xiaomi mimo-vl-miloco technical report. arXiv preprint arXiv:2512.17436, 2025.

- [23] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

- [25] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Intergps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.

- [26] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. Advances in Neural Information Processing Systems, 36:43447–43478, 2023.

- [27] Runqi Qiao, Qiuna Tan, Minghan Yang, Guanting Dong, Peiqing Yang, Shiqiang Lang, Enhui Wan, Xiaowan Wang, Yida Xu, Lan Yang, et al. V-thinker: Interactive thinking with images. arXiv preprint arXiv:2511.04460, 2025.

- [28] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

- [29] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.

- [30] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.

- [31] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11888–11898, 2023.

- [32] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. Unifying vision, text, and layout for universal document processing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19254–19264, 2023.

- [33] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [34] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.

- [35] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

- [36] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [37] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024.

- [38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- [39] Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218, 2025.

- [40] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

- [41] Qintong Zhang, Junyuan Zhang, Zhifei Ren, Linke Ouyang, Zichen Wen, Junbo Niu, Yuan Qu, Bin Wang, Ka-Ho Chow, Conghui He, et al. Docr-inspector: Fine-grained and automated evaluation of document parsing with vlm. arXiv preprint arXiv:2512.10619, 2025.

- [42] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

- [43] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

- [44] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## Appendix

###### A Additional Details for Dataset Construction

- A.1 Prompt for Data Draft Generation

- This subsection presents the prompt used in Step 1 (Data Draft Generation) of the VisAlg construction pipeline. The prompt elicits a synchronized draft consisting of structured problem analysis, diagram planning under strict bar-model constraints, and an initial executable DSL program. This stage establishes the logical and visual foundation for subsequent refinement and verification. The complete prompt is provided in Figure 8.

A.2 Prompt for Data Refining

- This subsection presents the prompt used in Step 2 of the VisAlg construction pipeline. Given an initial problem analysis and a draft DSL generated in the previous stage, this prompt instructs the model to perform checklist-driven verification and conditional refinement. The objective is to determine whether the draft is product-ready, and if not, to apply minimal, targeted corrections. The full checklist-driven refinement prompt is shown in Figure 9.

- A.3 Prompt for Scoring and Filtering

This subsection presents the prompt used in the final stage of the VisAlg construction pipeline. Given a refined DSL draft produced after checklist-driven revision, this prompt instructs an LLM-based verifier to perform strict, criteria-based scoring. Only instances receiving a full score are retained as product-ready samples in the final dataset. The scoring prompt used for LLM-based verification is shown in Figure 10.

- A.4 Human Expert Evaluation Criteria

In addition to automated LLM-based verification, we perform human expert screening on all refined instances. Human evaluators assess each DSL using the same zero-tolerance philosophy, serving as the final gatekeeper for dataset inclusion.

An instance is accepted into the final dataset only if all of the following conditions are satisfied:

- (1) Numerical validity: all bar-segment lengths correspond to valid quantities in the correct solution process, without arbitrary scaling or distortion.
- (2) Information sufficiency: the rendered diagram alone, based on visible annotations, is sufficient to solve the problem without consulting the original text.
- (3) Alignment accuracy: all brackets and alignment markers precisely coincide with valid segment boundaries.
- (4) Semantic fidelity: the diagram correctly encodes object relationships described in the natural language problem.
- (5) Format compliance: all constructions adhere strictly to the prescribed DSL conventions for reduction, transfer, multiplicative relations, and alignment. Only instances that satisfy all five criteria are retained as product-ready samples in VisAlg.

###### B Human Evaluation Criteria for Evaluation Metrics

###### B.1 Evaluation Target and Boundary Conditions

The human assessment is strictly limited to the given DSL output. Reviewers must evaluate whether the DSL expresses the problem’s key quantities and relationships in a norm-compliant, non-leaking, and structurally self-consistent manner, such that a reader can reliably reconstruct the intended structure and carry out a correct derivation from the diagram.

Step 1: Data Draft Generation Prompt You are an expert instructor in bar-model reasoning. You specialize in constructing bar-model diagrams using a structured Graphic Intermediate Representation (DSL) to support mathematical problem solving. Given a math word problem and its associated image, directly produce the output in the following format. Do not include any preamble, explanation, or commentary.

- Step 1: Problem Analysis Analyze the core structure of the problem using concise language. Do not restate the problem text. Output only the following four items:

- (1) Problem type: e.g., sum–difference, proportional, transfer, comparison.
- (2) Objects: specify whether the problem involves a single object or multiple independent objects.
- (3) Key information: short phrases describing all given quantities and relations.
- (4) Query: one sentence stating the required unknown.

- Step 2: Diagram Planning List the key considerations required to construct a correct and verifiable bar-model diagram:

- (1) Whether a vertical bracket (VB) is permitted (VB is forbidden for single-object problems and allowed only for multi-object aggregation).
- (2) Whether all given information from the problem is explicitly labeled in the diagram under a visible-text-only policy (numerical segment lengths do not count as textual labels).

- (3) Whether the queried quantity is explicitly marked in the diagram using interrogative naming (e.g., “total ?”, “how many ?”).
- (4) The core operation type involved (reduction, increase, transfer, comparison) and its corresponding visual encoding.
- (5) Whether vertical alignment lines (VL) are required to express cross-row shared boundaries or post-operation equality.
- (6) Any special constraints, including strict prohibition of answer leakage in all labels.

- Step 3: Initial DSL Draft Perform the necessary internal reasoning to determine correct segment lengths, then output an initial DSL program. Output only a clean DSL code block, without comments, explanations, or blank lines. “‘dsl <DSL code only> “‘ Output Format Requirements

- (1) Begin directly with “Step 1: Problem Analysis”.
- (2) Use the exact step titles shown above.
- (3) Output only the three steps; do not generate any additional content. DSL Syntax

- (1) HL “name” row l1 l2 ...: horizontal bar composed of semantic subsegments; positive values denote solid segments, negative values denote dashed segments (absolute value as length).
- (2) VL x row0 row1: vertical alignment line used only for cross-row shared boundaries.
- (3) HB “label” N|S row x0 x1: horizontal bracket spanning interval [x0, x1], placed above (N) or below (S) the bar.
- (4) VB “label” col row0 row1: vertical bracket used only for aggregating multiple independent objects. Core Construction Rules (Strict)

- (1) All horizontal bars must be decomposed into semantic subsegments; drawing a single undivided bar is not allowed.
- (2) Fractions and ratios must be represented via equal-length subsegments; multiplicative relations must be expressed by repeating equal segments, with the base quantity placed on the upper row.
- (3) Reduction must be encoded as (A - t) followed by (-t), with solid segments on the left and dashed segments on the right; negative segments denote only removal, deficit, transfer-out, or unknown placeholders.

- (4) Transfer must be represented as paired −t/ + t segments across rows; post-transfer equality must be marked with a shared VL at the corresponding boundary.
- (5) Vertical alignment lines are permitted only for cross-row shared boundaries and must align exactly with segment boundaries on all involved rows.

- (6) Bracket endpoints must coincide exactly with segment boundaries; floating or misaligned brackets are invalid.
- (7) Final numerical answers must not appear in any labels; computed values may appear only as segment lengths for rendering purposes.
- (8) All given information from the problem statement must appear explicitly as textual labels in the diagram. Silent Self-Check Before outputting, internally verify alignment correctness, semantic compliance, information completeness, and non-leakage of final answers.

- Figure 8 Prompt used in Step 1 for generating structured analysis and the initial DSL draft during VisAlg dataset construction.

To minimize subjective preference, prior belief, and post-hoc “mental correction,” the evaluation is conducted under the following boundary conditions: (i) reviewers must not introduce any extra information beyond what is explicitly encoded in the DSL; (ii) reviewers must not modify the output in any form, including adding segments, changing numeric values, renaming labels, or reformatting the program; (iii) the assessment does not consider the writing quality, fluency, or style of any accompanying natural-language solution.

Checklist-Driven Refinement Prompt

You are an expert instructor in bar-model reasoning. You specialize in auditing and refining bar-model diagrams expressed in a structured Graphic Intermediate Representation (DSL). You are given a math word problem with its image, together with an existing analysis and an initial DSL draft. Your task is to verify the draft using a checklist and revise it only if violations are found. Do not include any preamble or commentary. Checklist-Based Verification Select 2--4 verification items that are most relevant to the given problem. The following two checks are mandatory and must always be included. Each check must be marked as either [PASS] or [FAIL].

- (1) Alignment: verify that all bracket endpoints coincide with bar-segment boundaries or shared alignment positions, and that any vertical alignment markers lie on valid bar segments.
- (2) Information completeness: verify that all original quantities and relations stated in the problem are explicitly labeled in the diagram, and that the queried quantity is clearly indicated. In addition, select applicable checks from the following categories.
- (3) Norm compliance: verify that reductions follow the left-solid/right-dashed convention; comparisons use right-side dashed segments; and transfers are encoded as paired subtraction and addition across rows.
- (4) Style consistency: verify that multiplicative structures place the base quantity on the upper row; and that bracket placement prioritizes upper positioning, resolving overlaps by span length. Refinement Decision State in one sentence whether the DSL draft satisfies all selected checks. If any check is marked [FAIL], briefly describe the violations and output a corrected DSL program. If all checks pass, state that the draft can be used without modification. Final Answer Generation After the refinement decision, continue directly with the final solution generation. The solution must include, in order: (1) a one-sentence identification of the problem type; (2) a concise reasoning outline with formulas written in $...$; (3) a clean DSL code block representing the final diagram; (4) step-by-step computations; and (5) the final numerical answer. Output Constraints

- (1) Begin output directly with the checklist-based verification section.
- (2) Use bold text only for section headers; do not use numbered steps or section markers.
- (3) Do not repeat the problem statement.
- (4) Do not include any content outside the specified structure. DSL Semantics All DSL syntax and construction rules strictly follow those defined in the draft-generation stage, including semantic subsegment decomposition, VB usage restrictions, reduction and transfer encoding, alignment constraints, and the prohibition of answer leakage.

- Figure 9 Prompt used for checklist-driven verification and conditional refinement of initial DSL drafts during VisAlg dataset construction.

###### B.2 Review Protocol and Evidence-Driven Practice

A structured, evidence-driven expert review protocol is adopted to maximize objectivity and reproducibility. Three domain experts with backgrounds in mathematics education and diagram-oriented coding are recruited. Each sample is rated independently by at least two reviewers; disagreements spanning two or more score levels are resolved by a third reviewer via arbitration.

To discourage intuition-based judgments, each assigned rating must be accompanied by minimal sufficient evidence that is directly verifiable from the DSL. Typical evidence includes: an HB endpoint failing to coincide with an HL segment boundary; a VL failing to align with a cross-row critical boundary; a quoted label explicitly containing the final numeric answer to the queried quantity (answer leakage); or arithmetic constraints that cannot hold under the implied sum–difference or transfer relations. Evidence logs enable third-party auditing without reliance on reviewer-specific interpretation.

###### B.3 Evaluation Dimensions DSL quality is characterized along five dimensions that jointly determine usability:

Structural Alignment. Whether HB endpoints and VL coordinates strictly coincide with HL segment boundaries, reflecting geometric legality and representational precision.

Scoring Prompt for DSL Verification

You are an expert instructor in bar-model reasoning. You are familiar with evaluating bar-model diagrams expressed in a structured Graphic Intermediate Representation (DSL). Your task is to score the given DSL code based solely on the problem statement and the provided solution (including DSL code). You must not modify, complete, or reinterpret the DSL code. Do not introduce any information beyond what is explicitly provided. Evaluation Scope All judgments must follow the visible-text-only principle: only quoted strings in HL names and HB/VB labels are considered visible annotations. Numeric segment lengths and coordinates are not treated as textual information. Scoring Criteria Evaluate the DSL code according to the following checklist. For each item, output either [PASS] or [FAIL], together with a brief justification. Critical Criteria (Fail Any ⇒ Score = 0.0)

- (1) Alignment correctness: all bracket endpoints align with bar-segment boundaries; all vertical alignment markers lie on valid shared boundaries.
- (2) Information completeness: all given quantities and required unknowns from the problem statement are explicitly annotated using visible text.
- (3) Numerical consistency: all segment lengths are numerically self-consistent with the problem logic, without arbitrary scaling or unexplained values.
- (4) Transfer correctness: transfer operations must be represented by paired subtraction and addition across rows, with appropriate alignment markers when required.
- (5) Answer leakage: no final answer values appear in any visible annotations.
- (6) VB/VL usage: vertical brackets are used only for multi-object aggregation; vertical alignment markers appear only at shared cross-row boundaries. Non-Critical Criteria The following criteria affect the score only if all critical criteria pass:
- (7) Reduction conventions: reduction and deficit relations follow the left-solid/right-dashed convention.
- (8) Multiplicative structure: multiplicative relations are expressed using repeated equal-length segments, with the base quantity placed on the upper row.
- (9) Semantic decomposition: horizontal bars are decomposed into semantically meaningful subsegments rather than drawn as undivided totals.
- (10) Label conciseness: visible annotations are concise, non-redundant, and free of embedded calculations. Scoring Rule If any critical criterion is marked [FAIL], the final score is 0.0. Otherwise, the base score is 1.0, with a penalty of 0.1 deducted for each failed non-critical criterion. Output Format

First output a section titled [Scoring Rationale], listing each criterion with its pass/fail status and justification. Then output a single line: [Final Score]: <float between 0.0 and 1.0> Do not output JSON or any additional formatting.

Figure 10 Prompt used for strict LLM-based scoring and filtering of refined DSL drafts in VisAlg.

Information Coverage. Whether all key givens and the queried unknown are explicitly marked or clearly represented based only on visible textual labels (i.e., quoted strings). Numeric segment lengths alone do not count as textual labels. This dimension measures whether the intended problem structure is recoverable from the diagram content.

Numerical Consistency. Whether the segment lengths satisfy the intended arithmetic constraints (sum, difference, increase/decrease, transfer amount). Systematic errors such as uniform scaling artifacts or the use of numerous uninterpretable numbers are treated as violations.

Semantic Conformity. Whether the DSL follows task-specific construction conventions, including: reduction encoded as solid-left and dashed-right segments; transfer encoded as paired −t/ + t segments with equal magnitude across rows; multiplicative relations expressed via repeated equal-length subsegments with the base quantity emphasized; non-abusive use of VL/VB; and semantically motivated HL decomposition by problem type.

Answer Leakage. A hard constraint assessed solely from visible textual labels: if the final numeric answer to the queried quantity appears explicitly in quoted labels, the output is considered leaking.

- B.4 Overall Rating Scale An overall five-level score is assigned based on the five dimensions above:

- • 5 (Excellent): No leakage; strict structural alignment; complete information coverage; numerically consistent relations; and clear, norm-compliant semantic decomposition.
- • 4 (Good): Overall correct and readable; minor non-critical imperfections (e.g., slight alignment or labeling issues) that do not hinder understanding or derivation.
- • 3 (Acceptable): Still usable for problem solving but requires frequent reference to the problem statement to resolve ambiguities; partial missing labels, coarse decomposition, or localized norm violations may be present.
- • 2 (Poor): High risk of misinterpretation due to unreliable alignment, missing critical information, multiple semantic violations, or strained numerical relations, making stable derivation difficult.
- • 1 (Unacceptable): Fatal violations, including answer leakage, uniform scaling artifacts, fundamentally invalid sum–difference or transfer relations, large-scale alignment failures, or incorrect core semantic structure.

- B.5 Leakage as a Dominant Violation

Answer leakage is treated as the most destructive violation because it breaks the boundary that the diagram should encode structure rather than disclose the solution. Accordingly, once leakage is confirmed (from quoted labels), the output is rated as unacceptable and the evidence must be recorded explicitly.

###### C Case Studies on Visual Algebra Schemas

We present one representative example for each of the five visual algebra schemas in VisAlg. These cases illustrate how Thinking with Drafting (TwD) operationalizes optical decompression: it converts abstract textual constraints into an explicit and spatially aligned DSL, so that the problem can be solved directly from the rendered structure without relying on implicit, unverified reasoning.

Alignment-centric schemas. Difference Analysis (Figure 19) and Proportional Distribution (Figure 11) both require cross-object alignment to make relational constraints executable. In Figure 19, the DSL anchors one entity as a reference and encodes “more than” / “fewer than” relations as explicit offset segments, turning comparative language into a geometry-consistent subtraction layout. In Figure 11, the DSL realizes the multiplicative constraint by repeating equal-length unit segments and aligning boundaries across rows, so the “×12” relation is enforced by topology rather than inferred implicitly; the final query is then represented as a single unknown bracket on the composed total.

Decomposition-centric schemas. Sum & Split (Figure 16) and Rate & Percentage (Figure 13) emphasize part–whole partition and unit grounding. Figure 16 isolates the known remainder as a dedicated segment and marks the target as the complementary part, making the computation a direct completion on the bar. Figure 13 grounds fractional change by first fixing the base quantity as the unit reference and then attaching the fractional increment as an explicit subsegment, reducing ambiguity about the comparison base.

State-transition schema. Change & Revert (Figure 12) involves a counterfactual transfer and a post-transfer relation. The DSL externalizes the hypothetical “give” operation with paired decrease/increase segments and then imposes the after-state constraint on the aligned configuration, enabling reverse deduction while keeping all visible labels faithful to the original statement (i.e., without leaking computed answers).

###### D Additional Error Analysis

We summarize a Taxonomy of Structural Degeneration observed in baseline diagrams, where the output may remain arithmetically compatible yet loses the structural invariants required for verification.

- D.1 Semantic Erasure: Multiplicative Topology Collapsed

In Figure 21, the baseline collapses the given ×3 constraint into an additive “difference” layout. This erases the repeated-unit structure, so the multiplier is no longer visually provable even if the final arithmetic is correct.

- D.2 Label Injection: Numbers without Geometric Support

Figure 22 shows a label–structure mismatch: the model writes the computed difference as text, but does not allocate a corresponding sub-segment. The diagram therefore contains claims without geometric evidence, and downstream reasoning can mistakenly treat labels as quantities.

- D.3 Alignment Conflict: Incompatible Global Boundaries

In Figure 23, the baseline mixes incompatible alignment cues: dashed completion implies one shared endpoint, while vertical guides declare another boundary. This breaks global boundary consistency, so the “less by 35” relation is not stably encoded in the diagram.

###### E Potential Risks

We identify two primary risks that stem from the formalization of reasoning introduced by the Thinking with Drafting (TwD) paradigm, particularly in educational contexts.

First, the use of a structured DSL may amplify automation bias. Because the generated diagrams resemble formal proofs, users may conflate structural validity with semantic correctness, implicitly assuming that a well-formed intermediate representation guarantees a correct solution.

Second, TwD introduces a risk of cognitive offloading that may lead to skill atrophy in diagrammatic reasoning. By externalizing key steps of problem decomposition and visualization, the system may reduce the learner’s engagement in constructing and maintaining structural invariants. Over time, excessive reliance on automated drafting can weaken the user’s ability to independently translate textual constraints into spatial representations, undermining the development of foundational visual reasoning skills.

###### Proportional Distribution

###### Key Feature

[Figure 54]

Multiples by shared quantity

Question

The school bought 124 boxes of colored chalk. The number of white chalk boxes purchased is 12 times that of colored chalk. How many boxes of both types of chalk did the school buy in total?

Solution

Oper 1&2. Represent Objects.

[Figure 55]

[Figure 56]

- Oper 3. Alignment comment.
- Oper 4&5. Value comments.

[Figure 57]

[Figure 58]

[Figure 59]

Oper 6. Question comment.

[Figure 60]

###### Answer

A total of 1612 boxes of two types of chalk were purchased.

- Figure 11 Proportional Distribution. The diagram enforces the multiplicative relation via repeated equal-length units and boundary alignment, and marks the queried total as an explicit unknown on the composed bar. TwD does not merely calculate 12 × 124; instead, it enforces the multiplicative constraint via topological repetition. By rendering the ‘White Chalk’ bar as a composite of 12 equal-length units aligned with the ‘Color Chalk’ reference unit, the model transforms an abstract arithmetic operation into a concrete unit-repetition task, making the total sum visually deducible.

###### Change & Revert

###### Key Feature

[Figure 61]

Relationship between before-and-after states or reverse deduction/reversion.

###### Question

A little bunny pulled 3 carrots. Her mom said: "If I give you 6 carrots, the number of carrots I have left will be 3 times the number you have." How many carrots did her mom pull?

Solution

Oper 1&2. Represent Objects.

[Figure 62]

[Figure 63]

- Oper 3. Alignment comment.
- Oper 4~6. Value comments.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Oper 7. Question comment.

[Figure 68]

###### Answer

Her mon pulled out 33 radishes

- Figure 12 Change & Revert. A counterfactual transfer is rendered as paired decrease/increase segments, after which the post-transfer constraint is imposed on the aligned after-state topology. This example illustrates how TwD handles hypothetical state transitions. The model employs a dual-segment representation where transfers are rendered as paired decrease/increase segments. Crucially, the “post-transfer” multiplicative constraint ×3 is imposed not on the initial state, but on the aligned after-state topology. This proves the model’s ability to reason about dynamic temporal states within a static spatial diagram.

#### Rate & Percentage

###### Key Feature

[Figure 69]

Expressed as rates; primarily about unit "1" and percentage

###### Question

Lele drank 4/5 liters of water in the morning. In the afternoon, he drank 1/4 more than what he drank in the morning. How many liters more water did Lele drink in the afternoon than in the morning?

###### Solution

Oper 1&2. Represent Objects.

[Figure 70]

[Figure 71]

- Oper 3. Alignment comment.
- Oper 4. Value comment.

Answer

He drinks 1/5 L more

- Oper 5. Question comment.

[Figure 72]

[Figure 73]

[Figure 74]

- Figure 13 Rate & Percentage. Fractional change is grounded by fixing the base as a unit reference and attaching the fractional increment as a dedicated subsegment aligned to the shared boundary. The model fixes the morning consumption as the holistic unit “1”, and attaches the fractional increment (1/4) as a dedicated sub-segment aligned to the unit boundary. This explicit segmentation allows the model to visually isolate the ∆ from the whole, preventing unit confusion.

[Figure 75]

###### Figure 14 Rate & Percentage. A complex multi-step ratio problem involving chain dependencies. TwD manages this hierarchy through cascading alignment: each subsequent row’s length is topologically anchored to the specific fraction of the preceding row. The vertical dashed lines serve as transitive logic gates, ensuring that the final quantity is derived from a rigorously valid chain of geometric proportions, minimizing error propagation.

[Figure 76]

###### Figure 15 Rate & Percentage. Fractional relationships are represented by fixing the base quantity as a unit reference and aligning proportional segments to this shared unit. The fractional change is visualized as a dedicated subsegment, supporting reasoning based on rates and percentages.

### Sum & Split

##### Key Feature

[Figure 77]

Finding total/remaining amount; whole-part; merging multiple parts.about "sum, remainder, composition/splitting"

##### Question

Xia Ya needs to do 60 mental arithmetic problems. 18 problems are still left unfinished. How many problems has Xia Ya already finished?

Solution

- Oper 1. Represent Object.
- Oper 2&3. Value comment.

[Figure 78]

[Figure 79]

[Figure 80]

Oper 4. Question comment.

[Figure 81]

###### Answer

She has finished 42 problems.

- Figure 16 Sum & Split. The whole–part structure is made explicit by isolating the known remainder segment and marking the target as the complementary unknown, directly supporting completion-by-subtraction. The unknown remainder is highlighted as the target, supporting solution by subtraction.

[Figure 82]

###### Figure 17 Sum & Split. The whole–part structure is made explicit by isolating the known remainder segment and marking the target as the complementary unknown, directly supporting completion-by-subtraction. The unknown remainder is highlighted as the target, supporting solution by subtraction.

[Figure 83]

###### Figure 18 Sum & Split. The whole–part structure is made explicit by isolating the known remainder segment and marking the target as the complementary unknown, directly supporting completion-by-subtraction. The final total is identified as the complementary unknown, supporting solution by subtraction and addition.

###### Difference Analysis

###### Key Feature

[Figure 84]

Comparison and difference. Primarily about "difference relationship".

Oper 5~7. Value comments.

[Figure 85]

Question

Xiao Hong has 16 storybooks. This is 3 more than what Xiao Fang has, and 2 fewer than what Xiao Ming has. How many storybooks do Xiao Fang and Xiao Ming each have?

[Figure 86]

Solution

Oper 1~3. Represent Objects.

[Figure 87]

[Figure 88]

[Figure 89]

Oper 8&9. Question comments.

[Figure 90]

[Figure 91]

Oper 4. Alignment Comment.

[Figure 92]

Answer

Xiao Fang has 13 storybooks. Xiao Ming has 18 storybooks.

- Figure 19 Difference Analysis. This example illustrates the Thinking with Drafting process on a multi-entity comparison problem. The model does not hallucinate the answer directly; instead, it performs logical reconstruction in steps: (1) instantiating objects (Oper 1-3), (2) enforcing topological alignment via vertical anchors (Oper 4), and (3) encoding “more than/fewer than” relations as explicit offset segments (Oper 5-7). This step-by-step grounding ensures that the final arithmetic inference is derived from a verified geometric structure.

[Figure 93]

###### Figure 20 Difference Analysis. Application of TwD to a continuous-value scenario involving bidirectional differences (“lower than” vs. “higher than”). The system successfully decodes the textual constraints into precise spatial alignments. Note how the vertical dashed lines act as logical anchors, physically locking the relative positions of the reference entity and derived entities. This transforms an abstract arithmetic word problem into a concrete visual subtraction and addition task, mitigating logical errors in multi-step calculation.

Question

Class 1, Grade 2 has 5 red rubber balls. The number of yellow rubber balls is 3 times that of red rubber balls. How many more yellow rubber balls are there than red rubber balls?

[Figure 94]

###### TwD

Oper 1&2. Represent objects

Oper 1&2. Represent objects

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Oper 3&4. Value comments

Oper 3. Alignment comments

[Figure 99]

[Figure 100]

- Oper 5. Value comments

- Oper 6. Question comments

[Figure 101]

[Figure 102]

[Figure 103]

Oper 5. Question comment

[Figure 104]

###### Answer.

###### Answer.

There are 5 more yellow balls than red balls.

There are 10 more yellow balls than red balls.

- Figure 21 Semantic Erasure. The ×3 constraint is collapsed into an additive layout, removing repeated-unit evidence. The baseline model suffers from Semantic Erasure: it collapses the multiplicative constraint into a generic additive layout, failing to render the repeated unit segments. TwD explicitly preserves the unit topology, rendering three distinct segments for the yellow ball row. This structural fidelity enforces the correct arithmetic operation.

[Figure 105]

Question

There are 66 peach trees and 38 pear trees in the orchard.

Oper 1&2. Represent objects

[Figure 106]

- (1) How many peach trees and pear trees are there in total?
- (2) How many more peach trees are there than pear trees?

[Figure 107]

###### TwD

Oper 3. Alignment comments

Oper 1&2. Represent objects

[Figure 108]

[Figure 109]

Oper 5~8. Value comments

[Figure 110]

[Figure 111]

- Oper 3. Alignment comment
- Oper 4&5. Question comments

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

###### Answer.

###### Answer.

There are a total of 132 peach and pear trees. There are 28 more peach trees than pear trees.

There are a total of 104 peach and pear trees, with 28 more peach trees than pear trees.

- Figure 22 Label Injection. A computed value is written as text without a supporting sub-segment, yielding an ungrounded claim. The baseline model exhibits Label Injection: it hallucinates a computed value (“132”) and injects it as a text label without generating the supporting geometric sub-segments. The visual diagram thus becomes a deceptive artifact that does not physically represent the sum. TwD constructs the result bottom-up. By strictly aligning the start and end points of the ‘Peach’ and ‘Pear’ segments, it creates a valid geometric aggregation, ensuring the final answer is visually deducible.

Question

Xiao Ming practices running. He ran 480 meters on the first day, and 35 meters less on the second day than on the first day. How many meters did he run in total over the two days?

[Figure 118]

###### TwD

Oper 1&2. Represent objects

Oper 1&2. Represent objects

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- Oper 3. Alignment comments
- Oper 4&5. Value comments

Oper 3. Alignment comments

[Figure 124]

[Figure 125]

- Oper 5. Value comments

- Oper 6. Question comments

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Oper 5. Question comments

[Figure 130]

[Figure 131]

###### Answer.

###### Answer.

He ran a total of 925 meters in two days.

He ran a total of 890 meters in two days.

- Figure 23 Alignment Conflict. Conflicting global boundaries break the stability of cross-row relations. The baseline model generates an Alignment Conflict: the vertical dashed line (alignment anchor) is misplaced, visually suggesting that Day 2 is longer than Day 1 despite the label “35 less”. This topological contradiction breaks the logical chain, leading to an erroneous calculation. TwD correctly places the subtractive anchor. The dashed line precisely demarcates the difference segment, enforcing a consistent spatial logic where the length of Day 2 is physically constrained to be shorter, guiding the correct subtraction.

