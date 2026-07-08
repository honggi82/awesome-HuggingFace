## Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation

Chenghao Zhang, Guanting Dong, Yufan Liu, Tong Zhao, Xiaoxi Li, Zhicheng Dou* Gaoling School of Artificial Intelligence, Renmin University of China davidzhang@ruc.edu.cn, dou@ruc.edu.cn

### Abstract

###### Interleaved Image-Text Research Reports

[Figure 1]

[Figure 2]

###### Benefits Challenges

Large Language Models (LLMs) have advanced autonomous agents from deep search, which retrieves concise factual answers, to deep research, which synthesizes scattered evidence into long-form reports. However, verifiable multimodal deep research remains challenging due to open-ended synthesis without deterministic ground truth and the need to interleave textual arguments with visual evidence. We propose PTAH, a multi-agent harness for interleaved report generation. PTAH orchestrates the lifecycle from user query to rendered web report through planning, research, and writing stages, where specialized agents construct visual-aware plans, collect claimgrounded evidence, maintain source-aligned images in a Visual Working Memory, and compose reports through declarative multimodal tool use. A verifier agent serves as the harness’s acceptance function, enforcing factual grounding, citation fidelity, and crossmodal consistency throughout the workflow. We further introduce PTAHEval, an evaluation protocol that augments existing benchmarks with image-level and presentation-level assessments. Experiments on deep research benchmarks show that PTAH produces more reliable, visually informative, and usable humanfacing multimodal reports than strong baselines. Our code is released at https://github.com/ SnowNation101/Ptah

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2605.29861v2[cs.CL]3Jun2026

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder …

The Transformer follows this overall architecture using stacked selfattention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

[Figure 6]

[Figure 7]

Illustration images make concepts concrete, reducing cognitive load and aiding understanding

[Figure 8]

Incorrect Image References

The Transformer follows this overall architecture using stacked selfattention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

[1] What is Transformer? https://url/to/this/page

[Figure 9]

[Figure 10]

Invalid References

[Figure 11]

[Figure 12]

Evidence images substantiate claims with measurable, verifiable results

When deeper networks are able to start converging, a degradation

problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly …

[Figure 13]

Uninformative Images

[Figure 14]

[Figure 15]

Improper Image Placement

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments. Fig. 1 shows a typical example.

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments. Fig. 1 shows a typical example.

Figure 1: Illustration of how images enhance report quality, and the challenges of generating high-quality interleaved image–text reports.

hallucination remains a critical bottleneck for their deployment in knowledge-intensive tasks. To mitigate this, Retrieval-Augmented Generation (RAG) (Gao et al., 2023; Zhang et al., 2025a; Dong et al., 2025d) has emerged as a prevailing paradigm, leveraging external knowledge bases and search tools to provide factual grounding.

Building on this paradigm, Deep Search has emerged across both academia and industry as an agentic multi-step search paradigm, where autonomous agents leverage complex toolchains to tackle more demanding tasks. Benchmarks such as GAIA (Mialon et al., 2024) and HLE (Phan et al., 2025), along with complex mathematical reasoning tasks, have showcased the efficacy of multi-step search and reasoning in solving hard problems. Nevertheless, these tasks are primarily characterized by deterministic answers in closed domains, where outcomes can be rigorously verified and refined through ground-truth labels or automated scripts.

### 1 Introduction

In recent years, Large Language Models (LLMs) (Yang et al., 2025; Team, 2025; DeepSeek-AI, 2025) and Vision-Language Models (VLMs) (Bai et al., 2025; Team, 2026) have demonstrated exceptional reasoning capabilities in content understanding and generation, enabling them to tackle sophisticated, cross-domain challenges. However, the inherent issue of

In contrast, the recent emergence of Deep Re-

*Corresponding author.

search systems in industry (e.g., OpenAI Deep Research (OpenAI, 2025)) marks a paradigm shift from seeking singular, objective answers toward synthesizing comprehensive, long-form reports. Compared with closed-end deep search, deep research poses two distinctive challenges: (1) Openendedness. Deep research reports lack a deterministic ground truth, requiring agents to perform multi-round iterative searches in open domains where outputs cannot be straightforwardly verified. (2) Multimodal interleaving. A professional report characteristically interleaves text with visual evidence such as trend charts and illustrative figures (Figure 1), demanding tight integration of multimodal content rather than text-only synthesis.

Despite the rapid progress of these systems, existing approaches fall short on both fronts. For open-endedness, multi-step research pipelines lack stage-wise verification, allowing noise introduced early on to accumulate and ultimately produce factually unreliable text and misaligned visuals. For multimodal interleaving, current frameworks treat image integration as a post-hoc decorative step rather than a core component of the research process, leaving visual evidence loosely tied to textual arguments and far from the interleaved quality expected in professional reports. These shortcomings motivate a holistic agentic approach that can autonomously plan, investigate, and verify research findings within a unified multimodal loop.

To address these challenges, we propose PTAH1, an agentic harness for credible multimodal deep research. Rather than treating multimodal report generation as a monolithic generation problem, PTAH organizes specialized agents, external tools, intermediate research states, and verification signals into a controlled execution workflow. The harness orchestrates the full lifecycle from user query to rendered multimodal report through three stages: Planning, Research, and Writing. In Planning, PTAH constructs a visual-aware research plan that specifies both textual structure and intended visual evidence. In Research, parallel agents instantiate this plan with claim-grounded evidence, citations, numerical data, and source-aligned visual candidates maintained as intermediate research state. In Writing, a writer agent composes the final interleaved report through declarative multimodal tool

1Named after Ptah, the ancient Egyptian creator deity and patron of craftsmen, the name reflects the harness’s role in orchestrating the composition of structured multimodal reports from heterogeneous textual and visual materials.

use. Across all stages, verifier hooks serve as the harness’s acceptance function, checking protocol compliance, factual grounding, citation fidelity, visual relevance, and cross-modal consistency before the workflow advances.

Furthermore, to bridge the gap in evaluation metrics for interleaved image–text reports, we introduce PTAHEval, a flexible evaluation protocol that integrates seamlessly into existing deep research benchmarks. PTAHEval assesses report quality along two dimensions: Image Content Quality and Multimodal Presentation Quality. Experimental results demonstrate that PTAH generates high-quality, credible, and professionally interleaved research reports.

To summarize, we make the following contributions:

- • We propose PTAH, an agentic harness that coordinates specialized agents, external tools, research states, and verification signals for credible multimodal deep research.
- • We design a visual-aware workflow that organizes multimodal deep research into Planning, Research, and Writing, maintaining plans, evidence, citations, numerical data, and sourcealigned visual candidates as inspectable intermediate artifacts.
- • We introduce verifier hooks that implement the harness’s acceptance function, enabling stagewise checks for protocol compliance, factual grounding, citation fidelity, visual relevance, and cross-modal consistency.
- • We present PTAHEval, an evaluation protocol for interleaved image–text research reports, and show that PTAH improves multimodal report quality and readability while maintaining strong textual reliability.

### 2 Related Work

#### 2.1 Deep Search and Deep Research

Following ReAct (Yao et al., 2023), deep search augments LLMs with iterative tool use for multistep information retrieval. Early efforts extend RAG with iterative retrieval and evidence verification (Press et al., 2023; Shao et al., 2023; Asai et al., 2024), while more recent work generalizes this into agent-based frameworks with richer action spaces (Wang et al., 2024; Li et al., 2025a; Jin et al., 2025; Chen et al., 2025b; Wu et al., 2025c;

Dong et al., 2025b,c,a, 2026). However, these approaches primarily target closed-end question answering with deterministic answers (Xi et al., 2025; Wu et al., 2025d).

Recent systems extend deep search to openended, long-form report generation, including OpenAI Deep Research (OpenAI, 2025), Grok Deep Research (Grok, 2025), WebThinker (Li et al., 2025b), OWL (Hu et al., 2025), Auto Deep Research (Tang et al., 2025), and Multimodal DeepResearcher (Yang et al., 2026). Nevertheless, most systems struggle to jointly achieve deep multi-hop reasoning and broad information coverage, exposing fundamental limitations of single-agent architectures in complex research settings (Lan et al., 2025; Yen et al., 2025; Shi et al., 2025).

#### 2.2 Interleaved Image–Text Generation

While recent MLLMs such as Qwen3-VL (Bai et al., 2025), InternVL (Chen et al., 2023), GPT-

- 4V (OpenAI, 2023), and LLaVA (Liu et al., 2023) excel at understanding interleaved image–text inputs, they are primarily designed for perception and generally cannot generate interleaved outputs (Deng et al., 2025; Xie et al., 2025a).

Two paradigms have emerged for interleaved generation (Guo et al., 2025). The first builds native multimodal generative models within unified architectures, integrating diffusion-based decoders with autoregressive language models (Xie et al., 2025b; Wu et al., 2025b; Team, 2024; Wu et al., 2024; Ge et al., 2024; Caffagni et al., 2024). The second treats interleaved generation as a tool-augmented agentic process, exemplified by THYME (Zhang et al., 2025b) and WebWatcher (Geng et al., 2025). Dedicated benchmarks such as MMInterleaved (Tian et al., 2024), OpenLEAF (An

- et al., 2024), and ISG-Bench (Chen et al., 2025a) further support evaluation of interleaved generation quality. However, existing methods generally lack explicit verification and cross-modal consistency checks, often producing weakly grounded visual outputs in open-ended scenarios. 3 Task Formulation

Given a plain-text user query q, our goal is to produce a multimodal research report r and its rendered web page h. We represent r as an ordered sequence of content blocks

r = (b1,b2,...,bM), (1)

where each block bi is either a textual segment ti or a visual element vi, allowing flexible interleaved layouts such as (t1,v1,v2,t2,...) that reflect the structure of research reports.

We formulate multimodal deep research as a harnessed agentic process. At step t, the harness maintains a research state st = (q,Mt,τ<t), where Mt is the structured working state—intermediate plans, evidence, citations, numerical data, and visual candidates—and τ<t is the interaction history. The model produces a reasoning step zt and may invoke a tool at = (ut,xt) with ut ∈ U, yielding an observation ot = ut(xt) that updates Mt+1; we write τ = {(zt,at,ot)}Tt=1 for the full trajectory.

After the research state is constructed, the final report is sampled as r ∼ pθ(· | q,MT,τ) and rendered into the final web page h = Render(r), where Render(·) first serializes the interleaved blocks into HTML and then displays them as a webpage.

### 4 PTAH: Verifiable Multi-Agent Harness

PTAH is an agentic harness for credible multimodal deep research. As illustrated in Figure 2, it orchestrates the lifecycle from a user query to a rendered multimodal report through three stages: Planning, Research, and Writing. The Planner Agent constructs a visual-aware research plan, the Researcher Agents instantiate it with claim-grounded evidence and source-aligned images stored in Visual Working Memory, and the Writer Agent composes the final interleaved report through declarative multimodal tool use. Across this lifecycle, a Verifier Agent acts as the harness’s acceptance function, combining rule-based checks with LLM-based rubric verification to ensure protocol compliance, factual grounding, citation fidelity, visual relevance, and crossmodal consistency before the workflow advances.

4.1 Planning: Visual-Aware Research State Initialization

The Planner Agent initializes the research state by iteratively invoking text search tools to explore relevant domain knowledge. It produces a structured plan that contains a high-level overview, sectionlevel research goals, expected evidence types, and explicit visual specifications. These visual specifications describe where visual elements should appear, what communicative role they should serve, and which form of visual evidence, such as charts, diagrams, screenshots, or illustrative figures, best

###### Planning Stage Research Stage

###### Writing Stage

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Planner Reject Accept Researcher Reject Accept Writer

Webpage Image Extraction

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

LLM-based Rubric Verification

LLM-based Rubric Verification

Text Search

Image Reference

Text Search

Verifier

Verifier

[Figure 36]

[Figure 37]

Image Search

Visual Requirements

Rule-based Verification

Rule-based Verification

Webpage Summary

[Figure 38]

Image Generation

Webpage Summary

VLM Selection

[Figure 39]

[Figure 40]

Section 1

Raw Multimodal Report

[Figure 41]

Report Title Overview

###### Visual Working Memory

Finding 1 Research Finding Source

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Section Refine

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Section 1 Section Title

Image Refine

[Figure 50]

[Figure 51]

Overall Refine

[Figure 52]

[Figure 53]

Section Outline

Data

Tables and Data

HTML Generate

[Figure 54]

[Figure 55]

[Figure 56]

Writing Instruction Reference List

Visual Requirements

HTML Refine

Render

Figure 2: Overview of PTAH, a multi-agent harness for verifiable multimodal deep research.

supports the narrative.

ory. Here, visual evidence is broadly defined as source-aligned visual material that supports, explains, or contextualizes the report content, including charts, screenshots, diagrams, photographs, and illustrative figures. Raw image candidates first undergo rule-based filtering to remove low-resolution, duplicate, irrelevant, or non-informative images. Then, a VLM-based selector evaluates the remaining candidates according to the visual requirements specified in the planning stage. Each retained visual candidate is stored together with its source URL, surrounding webpage context, associated section, and intended visual role. By externalizing webpage images into Visual Working Memory, PTAH preserves source-aligned visual materials as structured cross-modal state rather than treating images as post-hoc decorative assets.

The plan acts as the first structured working state maintained by the harness. It constrains downstream research and writing by making the expected textual coverage and visual evidence explicit. Once produced, the plan is checked by the Verifier Agent on two levels: rule-based validation of the interaction protocol, tool-use constraints, and JSON format; and LLM-based rubric assessment of query coverage, section coherence, and visual– argument relevance. Plans that fail either check are revised, optionally with additional searches, before the workflow proceeds.

- 4.2 Research: Parallel Evidence Collection and Visual Working Memory

While the planning stage determines the breadth of the report, the research stage instantiates the plan with grounded evidence. For each planned section, a Researcher Agent performs an independent investigation through search and retrieval tools. Each researcher produces a structured research package containing key findings, claim-grounded evidence, numerical data, tables, references, and writing instructions for the downstream writer. This design allows the harness to scale the research process across sections while keeping each section’s evidence traceable and inspectable.

Each research package is then checked by the Verifier Agent for citation, including claim support, coverage of the planned goals, numerical/reference consistency, and visual relevance to the section intent. Packages that fail are returned to the corresponding researcher for revision or further evidence collection.

4.3 Writing: Declarative Multimodal Composition

The Writer Agent composes the report using the global plan, verified research packages, and Visual Working Memory. Instead of selecting and inserting images through an ad hoc post-processing step,

In parallel with textual evidence collection, each researcher extracts images from visited webpages and constructs a task-specific Visual Working Mem-

the writer follows a declarative multimodal composition strategy. It generates textual content and image directives jointly, embedding image tool tags at the positions where visual elements should appear. These specify the intended visual role and the tool operation required to realize the image.

###### PtahEval: Evaluation Protocol

Image Content Quality (ICQ)

An Existing Deep Research Benchmark

[Figure 57]

Interleaved Image-Text Content

VC CMA

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Question

IC ES

| | |
|---|---|
| | |

[Figure 64]

[Figure 65]

###### ICQ Score MPQ Score

Multimodal Research Report

Vision-Language Model as a Judge

Two Dims

The harness then arbitrates among three types of image operations. Image Reference reuses sourcealigned images from Visual Working Memory and is preferred when suitable candidates exist. Image Search retrieves additional web images when the existing Visual Working Memory does not satisfy the section requirement. Image Generation creates new visual elements when the report requires synthesized visuals, such as charts, structured diagrams, or illustrative figures. For data-driven visuals, PTAH can invoke executable code rendering to generate charts or visualizations; for illustrative content, it can invoke image generation models from textual descriptions. After all sections are composed, the writer generates a conclusion and assembles the sections into a raw interleaved report.

[Figure 66]

| | |
|---|---|
| | |

DLB IS

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

First-Screen Screenshot

Base Benchmark Score

1k*2k pixels

VE VED

Browser

Multimodal Presentation Quality (MPQ)

Figure 3: An illustration of our PTAHEval evaluation.

arguments, visual evidence, and rendered layouts. We propose PTAHEval, a flexible protocol that preserves the original questions and text-oriented metrics of existing benchmarks while adding multimodal evaluation procedures over the generated report artifact. Given a benchmark query, a system must produce a rendered multimodal report rather than a text-only answer, which is then assessed from two complementary perspectives: Image Content Quality (ICQ), measuring whether individual images are clear, relevant, informative, and aligned with the surrounding text; and Multimodal Presentation Quality (MPQ), measuring whether the rendered report presents information in a readable, well-organized, and visually coherent manner.

Test-Time Scaling After initial composition, PTAH applies verifier-guided test-time scaling through a sequence of lifecycle refinement hooks instead of directly returning the raw multimodal report. As shown in Figure 2, this process consists of six steps: (1) Section Refine revises each section for clarity, evidence coverage, citation fidelity, and local coherence; (2) Image Refine decides whether each visual element should be Keep, Delete, or Edit, and executes editing instructions for images marked as Edit; (3) Overall Refine improves global organization, cross-section consistency, and image– text alignment; (4) HTML Generate converts the refined report into an HTML document with layout and styling specifications; (5) HTML Refine further adjusts the HTML structure, style consistency, spacing, and rendered readability; and (6) Render displays the final HTML document in a browser as a user-facing multimodal report. Together, these refinement and rendering steps improve the readability and usability of the final report by presenting its layout, visual placement, and image–text organization in a form that is directly accessible to human readers.

#### 5.1 Image Content Quality Evaluation

For ICQ, we feed interleaved text–image inputs to a VLM, which judges whether each image meaningfully contributes to the report in terms of informativeness, consistency with the surrounding text, and support for textual explanations. ICQ comprises four dimensions: (1) Visual Clarity (VC): image legibility and ease of interpretation; (2) CrossModal Alignment (CMA): semantic consistency with the surrounding text and appropriateness of the placement context; (3) Information Complementarity (IC): whether the image conveys meaningful information that complements the text, especially content hard to express in words alone; (4) Evidentiary Support (ES): whether the image supports, explains, or contextualizes the claims and conclusions in the surrounding text.

### 5 PTAHEval Evaluation Protocol

#### 5.2 Multimodal Presentation Quality Evaluation

Existing evaluation protocols for deep research systems focus mainly on textual outputs and are insufficient for multimodal reports that integrate textual

MPQ targets the presentation quality of the rendered report under realistic reading conditions.

- Table 1: The overall results on DeepResearch Bench and DeepConsult. DeepResearch Bench evaluates Comprehensiveness (Comp.), Insight/Depth (Insight), Instruction-Following (Inst.), and Readability (Read.), together with the overall score. DeepConsult evaluates Instruction-Following (Inst.), Comprehensiveness (Comp.), Completeness (Compl.), and Writing Quality (Writ.), along with their average score.

DeepResearch Bench DeepConsult

Method

Comp. Insight Inst. Read. Overall Inst. Comp. Compl. Writ. AVG. Direct Generation

Qwen3-32B 40.73 39.59 45.85 44.80 42.22 0.98 0.98 0.98 0.98 0.98 QwQ-32B 40.97 40.27 46.09 45.23 42.59 0.98 0.98 0.98 1.96 1.23

###### Text-Only Generation

ReAct 42.63 40.42 47.66 46.37 43.70 0.98 0.98 0.98 6.86 2.45 Search-o1 41.57 39.70 46.82 45.96 42.91 0.98 0.98 0.98 7.84 2.69 WebThinker 44.63 43.26 46.86 46.61 45.00 2.94 17.64 2.94 5.88 7.35

Multimodal Generation LLM-I 35.14 31.77 41.14 40.07 36.36 0.98 0.98 0.98 1.96 1.23

PTAH (ours) 42.97 44.32 46.71 47.95 45.16 13.73 18.63 17.64 14.71 16.18

Since PTAH produces a user-facing web artifact, the report is first rendered as a webpage and its visible viewport (1000 × 2000 pixels) is captured as the evaluation input, reflecting what human readers see in terms of layout, spacing, visual placement, and image–text organization. The captured page image is then assessed by the VLM along four dimensions: (1) Density-Legibility Balance (DLB): balance between information density and perceptual clarity within the viewport; (2) Informational Saliency (IS): whether key insights and structural elements are effectively highlighted through visual hierarchy; (3) Visual Encoding Diversity (VED): use of diverse visual forms (e.g., tables, callouts, icons, charts, diagrams, illustrative figures) to support comprehension; (4) Visual Ergonomics (VE): spacing, visual rhythm, alignment, and perceptual clarity, evaluating whether the layout reduces reading effort while preserving clear entry points.

Following Lee et al. (2024), each ICQ and MPQ dimension is scored on a five-point Likert scale (1–

- 5). Together with the original benchmark metrics, ICQ and MPQ provide complementary signals on textual reliability, image-level quality, and reportlevel presentation.

6 Experiments

- 6.1 Experimental Setup

Implementation. We use Qwen3-32B (Yang

- et al., 2025) as the Planner, Researcher, and Verifier, and Qwen3-VL-32B-Instruct (Bai et al., 2025) as the Writer. Qwen3-32B is additionally employed for LLM-based verification, while Qwen3-VL-32BInstruct is used for image selection during the Re-

search stage. Detailed descriptions of all tools are provided in Appendix B.

Datasets and Baselines. We use the widely adopted benchmark DeepResearch Bench (Du et al., 2025). Following Han et al. (2025), we additionally include DeepConsult (you.com, 2025). We generate reports using questions from both benchmarks and evaluate the textual content using the evaluation metrics defined in each benchmark. To accommodate interleaved text–image outputs, we replace all LLM-as-judge evaluators with Qwen3-VL-235B-A22B-Instruct, a VLM capable of jointly processing textual and visual inputs.

As baselines, we include two direct report generation methods using Qwen3-32B and QwQ-32B. We also compare with three single-agent text-only search methods: ReAct (Yao et al., 2023), Searcho1 (Li et al., 2025a), and WebThinker (Li et al., 2025b). Since there is currently no readily reproducible open-source general multimodal research report generation framework, we additionally include LLM-I (Guo et al., 2025), an agent-based method for generating multimodal content, as a baseline. All these baselines use Qwen3-32B as the base model. For all methods, we uniformly retrieve the top-5 web pages for each query as external knowledge during text search.

#### 6.2 Main Results

We evaluate PTAH against baselines along three dimensions: textual content quality, visual quality, and factual credibility. PTAH consistently surpasses prior approaches in all three.

#### (1) Overall Content Quality. Table 1 reports

- Table 2: Overall PTAHEval results on DeepResearch Bench. The best results are highlighted in bold, and the second-best results are underlined. Since direct generation and text-only generation baselines do not produce reports with images, their Image Content Quality scores are not applicable and are marked as “-”.

Image Content Quality Multimodal Presentation Quality

Method

VC CMA IC ES Avg. DLB IS VED VE Avg. Direct Generation

Qwen3-32B - - - - - 3.55 3.60 2.73 3.53 3.35 QwQ-32B - - - - - 3.65 3.67 2.84 3.65 3.45

###### Text-Only Generation

ReAct - - - - - 3.13 3.21 2.41 3.55 3.08 Search-o1 - - - - - 3.49 3.14 2.62 3.24 3.12 WebThinker - - - - - 3.42 2.89 2.78 3.35 3.11

Multimodal Generation

LLM-I 2.10 2.28 1.96 1.52 1.97 3.12 2.51 3.25 3.11 3.00 PTAH (ours) 4.42 4.79 4.35 4.01 4.39 3.72 3.78 3.61 3.74 3.71

results on DeepResearch Bench and DeepConsult. On DeepResearch Bench, PTAH attains the best overall score of 45.16, leading on Insight/Depth and Readability while remaining competitive on Instruction-Following and Comprehensiveness, demonstrating that the multi-agent decomposition yields reports with deeper analysis and clearer structural organization. On DeepConsult, PTAH outperforms every baseline across all dimensions, reaching an average of 16.18—more than double the best baseline (WebThinker). The most pronounced gains, on Instruction-Following, Completeness, and Writing Quality, indicate stronger adherence to complex task specifications.

(2) Visual Quality. Table 2 reports PTAHEval results on DeepResearch Bench. For Image Content Quality, PTAH achieves the highest scores across all four dimensions. Its near-ceiling CrossModal Alignment score stems from two factors: (i) retrieved webpage images are inherently aligned with their surrounding textual context, and (ii) the test-time scaling (TTS) mechanism further refines image–text coherence. By contrast, the multimodal baseline LLM-I performs markedly worse, confirming that images produced by PTAH are clearer, more contextually relevant, and more effective as supporting evidence.

For Multimodal Presentation Quality, PTAH likewise leads on every dimension, benefiting from improved image quality and TTS-driven HTML layout optimization. Gains on Density-Legibility Balance and Visual Ergonomics reflect well-balanced spacing, stronger visual rhythm, and reduced perceptual load, while gains on Informational Saliency and Visual Encoding Diversity show that PTAH

anchors images in appropriate contexts and leverages diverse visual forms to highlight key insights and guide reader attention. Unlike text-only baselines, which rely on plain prose, or LLM-I, which lacks systematic layout refinement, PTAH integrates visual elements into a coherent global page design, yielding more professional and readable multimodal reports.

(3) Credibility. Table 3 reports FACT metrics on DeepResearch Bench. Since open-source baselines do not natively generate references, we prompt them to produce references alongside their reports. PTAH attains a Citation Accuracy of 87.53 with 9.64 effective citations per task, substantially outperforming all baselines. Case studies show that baselines frequently emit invalid or hallucinated URLs, whereas the Verifier Agent guarantees that every reference in PTAH maps to a valid, accessible source; the residual errors mainly stem from minor mismatches between cited content and the corresponding source. PTAH also issues more search tool calls than competing methods, reflecting its more thorough exploration of external knowledge, which directly translates into stronger factual grounding.

#### 6.3 Human Evaluation

To validate the VLM-based judgments used in PTAHEval, we randomly sample 25 tasks from DeepResearch Bench and collect reports from PTAH, LLM-I, and WebThinker. Four annotators (two Ph.D. and two undergraduate students in AI) perform anonymized pairwise comparisons between PTAH and each baseline under the PTAHEval criteria, with majority voting determining the final

Ptah vs. LLM-I Image Content Quality

Ptah vs. LLM-I Multimodal Presentation Quality

Ptah vs. WebThinker Multimodal Presentation Quality

| | | | | |
|---|---|---|---|---|
|88| |4|8| |
| | | | | |
|88| |8| |4|
| | | | | |
|96| | | |4|
| | | | | |
|96| | | |4|
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
|92| |4|4|
| | | | |
|96| | |4|
| | | | |
|10|0| | |
| | | | |
|92| |4|4|
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
|80| |12|8| |
| | | | | |
|92| | |4|4|
| | | | | |
|92| | |8| |
| | | | | |
|92| | |4|4|
| | | | | |
| | | | | |

VC CMA

DLB

DLB

IS VED

IS VED

IC ES

VE

VE

0% 50% 100%

0% 50% 100%

0% 50% 100%

Ptah Win

Ptah Lose

Tie

| |
|---|

| |
|---|

| |
|---|

Figure 4: Human evaluation of PTAH against LLM-I and WebThinker on DeepResearch Bench via PTAHEval.

- Table 3: FACT evaluation on DeepResearch Bench. We report Citation Accuracy (C.Acc.) and the average number of Effective Citations per task (E.Cit.), along with the average number of search tool calls per task.

Method C.Acc. E.Cit. #Search

ReAct 37.28 0.23 4.17 Search-o1 40.91 0.31 2.78 WebThinker 60.74 2.32 5.91

PTAH w/o Verifier 30.29 4.75 5.13 PTAH 87.53 9.64 12.82

Table 4: Ablation on test-time scaling (TTS), reporting the overall DeepResearch Bench (DRB) score, average Image Content Quality (ICQ) and Multimodal Presentation Quality (MPQ) scores, and the average number of generated and failed images per report.

Method DRB ICQ MPQ #Img / #Fail

LLM-I 36.36 1.97 3.00 0.74 / 0.14 PTAH w/o TTS 42.13 2.77 3.49 5.06 / 0.38 PTAH 45.16 4.39 3.71 3.76 / 0.12

preference. As shown in Figure 4, human preferences closely track the VLM-based scores. Annotators consistently favor PTAH over LLM-I on Image Content Quality, indicating clearer images that better complement and support the surrounding text, and over both LLM-I and WebThinker on Multimodal Presentation Quality, indicating that the gains stem from coherent multimodal organization rather than superficial image insertion. These findings confirm both the reliability of the PTAHEval evaluator and the consistent advantage of PTAH.

#### 6.4 Ablation Studies

Influence of the Verifier Agent. We remove the Verifier and rerun generation on DeepResearch Bench. Without it, 14 out of 100 tasks fail to proceed in the planning stage due to parsing errors, repetitive outputs, or incorrect tool calls. Among the remaining 86 tasks, another 18 fail during the research stage, leaving only 68 tasks that successfully produce final reports. This highlights the role of the Verifier in maintaining the stability of the multi-agent framework.

We further evaluate these 68 reports under the FACT metrics and record the number of text-search tool calls (Table 3). Removing the Verifier causes substantial drops in both citation validity and factual correctness. The comparison also shows that

Verifier feedback encourages the model to issue additional search calls, thereby expanding its exploration of external knowledge.

Effect of Test-Time Scaling. We further study the impact of test-time scaling in Table 4. Removing TTS reduces the overall DRB score by 3.03 points, and Image Content Quality and Multimodal Presentation Quality also decline noticeably. This indicates that TTS plays an important role in improving content quality, image quality, and the final HTML rendering of the report.

We additionally observe that without TTS the model inserts more invalid images and exhibits a higher rate of image-generation failures, further confirming that TTS is crucial for producing stable, high-quality multimodal reports.

### 7 Conclusion

We present PTAH, a multi-agent harness for verifiable multimodal deep research that addresses the lack of stage-wise verification and the post-hoc use of visual evidence in existing systems. PTAH decomposes the research lifecycle from query to rendered report into Planning, Research, and Writing, where a Verifier Agent enforces factual grounding, citation fidelity, and cross-modal consistency through rule-based and LLM-based checks across

all stages. Together with PTAHEval, which augments existing benchmarks with image content and presentation metrics, our experiments, human studies, and ablations show that PTAH consistently produces credible and professionally interleaved reports. These results advance multimodal deep research toward evidence-grounded, visually informative, and human-centric report generation.

### Limitations

Due to the constrained reasoning capabilities of existing open-source models, achieving a stable and autonomous agentic workflow for long-horizon multimodal search and generation remains a significant challenge. To ensure system controllability and reliability, we decompose the holistic framework into three distinct sequential stages rather than adopting a single-pass agentic generation process. While this modular design introduces manually defined boundaries, it facilitates more granular monitoring and rigorous validation of intermediate outputs. Furthermore, this decoupled architecture allows for independent optimization of specific modules in future iterations.

### References

Jie An, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Lijuan Wang, and Jiebo Luo. 2024. Openleaf: A novel benchmark for opendomain interleaved image-text generation. In Proceedings of the 32nd ACM International Conference on Multimedia, MM 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November 2024, pages 11137– 11145. ACM.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. CoRR, abs/2511.21631.

Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, and Rita Cucchiara. 2024. The revolution of multimodal large language models: A survey. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual

meeting, August 11-16, 2024, pages 13590–13618. Association for Computational Linguistics.

Dongping Chen, Ruoxi Chen, Shu Pu, Zhaoyi Liu, Yanru Wu, Caixi Chen, Benlin Liu, Yue Huang, Yao Wan, Pan Zhou, and Ranjay Krishna. 2025a. Interleaved scene graphs for interleaved text-andimage generation assessment. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. 2025b. Research: Learning to reason with search for llms via reinforcement learning. CoRR, abs/2503.19470.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2023. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. CoRR, abs/2312.14238.

DeepSeek-AI. 2025. Deepseek-v3.2: Pushing the frontier of open large language models. CoRR, abs/2512.02556.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Shi Guang, and Haoqi Fan. 2025. Emerging properties in unified multimodal pretraining. CoRR, abs/2505.14683.

Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025a. Agentic entropy-balanced policy optimization. CoRR, abs/2510.14545.

Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. 2025b. Toolstar: Empowering llm-brained multi-tool reasoner via reinforcement learning. CoRR, abs/2505.16410.

Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, and 1 others. 2026. Agent-world: Scaling real-world environment synthesis for evolving general agent intelligence. arXiv preprint arXiv:2604.18292.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025c. Agentic reinforced policy optimization. CoRR, abs/2507.19849.

Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and Zhicheng Dou. 2025d. Understand what LLM needs: Dual preference alignment for retrieval-augmented generation. In Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025, pages 4206–4225. ACM.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. 2025. Deepresearch bench: A comprehensive benchmark for deep research agents. CoRR, abs/2506.11763.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. CoRR, abs/2312.10997.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. 2024. SEED-X: multimodal models with unified multi-granularity comprehension and generation. CoRR, abs/2404.14396.

Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding, Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. Webwatcher: Breaking new frontier of vision-language deep research agent. Preprint, arXiv:2508.05748.

Grok. 2025. Grok 4. https://x.ai/news/grok-4. Accessed: 2026-01-18.

Zirun Guo, Feng Zhang, Kai Jia, and Tao Jin. 2025. LLM-I: llms are naturally interleaved multimodal creators. CoRR, abs/2509.13642.

Rujun Han, Yanfei Chen, Zoey CuiZhu, Lesly Miculicich, Guan Sun, Yuanjun Bi, Weiming Wen, Hui Wan, Chunfeng Wen, Solène Maître, George Lee, Vishy Tirumalashetty, Emily Xue, Zizhao Zhang, Salem Haykal, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. 2025. Deep researcher with test-time diffusion. CoRR, abs/2507.16075.

Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, and Guohao Li. 2025. OWL: optimized workforce learning for general multi-agent assistance in real-world task automation. CoRR, abs/2505.23885.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. CoRR, abs/2503.09516.

Tian Lan, Bin Zhu, Qianghuai Jia, Junyang Ren, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, and Kaifu Zhang. 2025. Deepwidesearch: Benchmarking depth and width in agentic information seeking. CoRR, abs/2510.20168.

Seongyun Lee, Seungone Kim, Sue Park, Geewook Kim, and Minjoon Seo. 2024. Prometheus-vision: Vision-language model as a judge for fine-grained evaluation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11286–11315, Bangkok, Thailand. Association for Computational Linguistics.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025a. Search-o1: Agentic search-enhanced large reasoning models. CoRR, abs/2501.05366.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. 2025b. Webthinker: Empowering large reasoning models with deep research capability. CoRR, abs/2504.21776.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: a benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

OpenAI. 2025. Introducing deep research. https://openai.com/index/ introducing-deep-research. Accessed: 202601-18.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Daron Anderson, Tung Nguyen, Mobeen Mahmood, Fiona Feng, and 81 others. 2025. Humanity’s last exam. CoRR, abs/2501.14249.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 5687–5711. Association for Computational Linguistics.

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 9248–9274. Association for Computational Linguistics.

Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, and 1 others. 2025. Deep research: A systematic survey. arXiv preprint arXiv:2512.02038.

Jiabin Tang, Tianyu Fan, and Chao Huang. 2025. Autoagent: A fully-automated and zero-code framework for LLM agents. CoRR, abs/2502.05957.

Chameleon Team. 2024. Chameleon: Mixedmodal early-fusion foundation models. CoRR, abs/2405.09818.

- Qwen Team. 2025. Qwq-32b: Embracing the power of reinforcement learning.
- Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, and Jifeng Dai. 2024. Mm-interleaved: Interleaved image-text generative modeling via multi-modal feature synchronizer. CoRR, abs/2401.10208.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024. Executable code actions elicit better LLM agents. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, and 20 others. 2025a. Qwen-image technical report. CoRR, abs/2508.02324.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. 2025b. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 12966–12977. Computer Vision Foundation / IEEE.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025c. Webdancer: Towards autonomous information seeking agency. CoRR, abs/2505.22648.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025d. Webwalker: Benchmarking llms in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 10290–10305. Association for Computational Linguistics.

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and TatSeng Chua. 2024. Next-gpt: Any-to-any multimodal LLM. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.

Yunjia Xi, Jianghao Lin, Yongzhao Xiao, Zheli Zhou, Rong Shan, Te Gao, Jiachen Zhu, Weiwen Liu, Yong Yu, and Weinan Zhang. 2025. A survey of llm-based deep search agents: Paradigm, optimization, evaluation, and challenges. CoRR, abs/2508.05668.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. 2025a. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. 2025b. Show-o2: Improved native unified multimodal models. CoRR, abs/2506.15564.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 40 others. 2025. Qwen3 technical report. CoRR, abs/2505.09388.

Zhaorui Yang, Bo Pan, Han Wang, Yiyao Wang, Xingyu Liu, Luoxuan Weng, Yingchaojie Feng, Haozhe Feng, Minfeng Zhu, Bo Zhang, and Wei Chen. 2026. Multimodal deepresearcher: Generating text-chart interleaved reports from scratch with agentic framework. In Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, pages 34368– 34377. AAAI Press.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Howard Yen, Ashwin Paranjape, Mengzhou Xia, Thejas Venkatesh, Jack Hessel, Danqi Chen, and Yuhao Zhang. 2025. Lost in the maze: Overcoming context limitations in long-horizon agentic search. CoRR, abs/2510.18939.

you.com. 2025. Deepconsult: A deep research benchmark for consulting / business queries. https://github.com/youdotcom-oss/ ydc-deep-research-evals. GitHub repository.

Chenghao Zhang, Guanting Dong, Xinyu Yang, and Zhicheng Dou. 2025a. Towards mixed-modal retrieval for universal retrieval-augmented generation. CoRR, abs/2510.17354.

Yifan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, Haonan Fan, Kaibing Chen, Jiankang Chen, Haojie Ding, Kaiyu Tang, Zhang Zhang, Liang Wang, Fan Yang, Tingting Gao, and Guorui Zhou. 2025b. Thyme: Think beyond images. CoRR, abs/2508.11630.

### A More Implementation Details

All experiments were conducted on a machine equipped with 4 × A800 80GB GPUs. On this machine, we locally deployed Qwen3-32B and Qwen3-VL-32B-Instruct using vLLM. For image generation, image editing, and evaluation, we accessed Qwen-Image, Qwen-Image-Edit, and Qwen3-VL-235B-A22B-Instruct through the APIs provided by the SiliconFlow platform2. Text search and image search APIs were provided by Serper3, and webpage parsing was supported by Jina Reader4.

External APIs were used as replaceable implementation interfaces in our experiments. For image generation, image editing, and VLM-based evaluation, the API-accessed components correspond to open-source models. These models were invoked with their default inference parameters unless otherwise specified. The use of hosted APIs was mainly intended to reduce the local GPU cost of largescale experiments; the same components can be reproduced by locally deploying the corresponding open-source models with the same settings.

For web access, Serper and Jina Reader serve as general-purpose interfaces for search result retrieval and webpage parsing, respectively. This follows the common setting of web-augmented deep research and search-based agent systems, where agents interact with external search and reading tools to obtain up-to-date evidence. These interfaces are not tied to the core design of PTAH and can be replaced by alternative search engines, browser tools, or local retrieval systems.

### B Details of the Tools

To support the distinct operational needs of our multi-agent framework, we integrate a suite of specialized tools. The PLANNER and RESEARCHER leverage text retrieval to gather information, while the WRITER utilizes visual retrieval, synthesis, and code execution for draft composition, supported

- 2SiliconFlow: https://www.siliconflow.cn
- 3Serper: https://serper.dev 4Jina Reader: https://jina.ai/reader

by image editing for subsequent refinement. The specific implementations are defined as follows:

- • Text Search: Facilitates knowledge acquisition. Accepting a text query as input, this tool returns summaries of the top-K relevant web pages. The pipeline first retrieves URLs via the Google Search API, parses the raw HTML into Markdown using the Jina Reader API5, and finally employs a Qwen3-32B model to summarize each page for key information extraction.
- • Image Search: Retrieves factual visual evidence, such as specific real-world entities. It maps a text query to the top-K matching images using the Google Image Search API (via Serper).
- • Image Generation: Synthesizes thematic illustrations or abstract concepts. It converts textual descriptions into images using the Qwen-Image diffusion model (Wu et al., 2025a), hosted on the SiliconFlow API.
- • Image Editing: Modifies visual details or adjusts styles during the refinement phase. It accepts an initial image (whether retrieved or generated) alongside a textual editing instruction to produce a modified output via the Qwen-ImageEdit-2509 model (Wu et al., 2025a).
- • Code Execution: Enables precise data visualization. It executes generated Python scripts within a secure, isolated sandbox environment to render rigorous charts and plots from structured data.

### C Dataset Details

DeepResearch Bench. DeepResearch Bench is a comprehensive benchmark for evaluating deep research agents on complex, long-form analytical tasks. It consists of 100 PhD-level research tasks spanning 22 distinct domains, with 50 tasks in English and 50 in Chinese, each carefully designed and curated by domain experts and senior practitioners to ensure high standards of complexity, clarity, and realism. Each task requires generating a comprehensive research report that involves multistep web exploration, information integration, and analytical reasoning. The benchmark provides a diverse and realistic evaluation setting for assessing models’ capabilities in deep research, covering both report generation quality and information retrieval effectiveness.

5Jina Reader: https://jina.ai/reader

DeepConsult. DeepConsult is a benchmark designed for evaluating deep research capabilities on business and consulting-oriented queries. It contains 102 queries covering a broad range of realworld consulting scenarios, including market analysis, investment opportunity assessment, industry evaluation, financial modeling, technology trend analysis, and strategic business planning. Each query is formulated to require comprehensive analysis and multi-step reasoning, reflecting the complexity of practical consulting tasks. The dataset is designed to assess whether models can produce structured, insightful, and actionable reports comparable to those generated in professional consulting settings.

### D Verifier Agent Details

The Verifier agent operates across the Planning, Research, and Writing stages, providing both rulebased and LLM-based verification to ensure the correctness, consistency, and quality of the generated outputs.

Planning Stage. During the planning stage, rulebased verification primarily checks the structural validity of the Planner’s outputs, including output formatting and the correctness of tool invocation schemas. In parallel, LLM-based rubric verification evaluates the entire reasoning trajectory of the Planner, including intermediate thoughts, tool calls, tool responses, and the final plan. It assesses aspects such as the rationality of the search strategy, the completeness of the generated outline, and the consistency between the outline and the retrieved web content. The Verifier produces a structured evaluation consisting of a scoring rubric and a review report, which summarizes strengths, weaknesses, potential improvements, and a final decision (accept or reject).

Research Stage. In the research stage, rule-based verification plays a critical role in ensuring citation fidelity. Specifically, all referenced URLs in the final output must exactly match those retrieved and accessed during the text search process, guaranteeing strict consistency between citations and evidence sources. It also validates the structural correctness of the output format. Meanwhile, LLMbased rubric verification focuses on assessing the depth and completeness of the exploration process, as well as the consistency between the synthesized findings and the retrieved web content, ensuring

both the rigor and reliability of the research.

Writing Stage. During the writing stage, the Verifier guides the Writer in refining and polishing the generated report. It ensures the correctness of image tool invocation syntax and enforces consistency between the final report and the research findings. Through iterative feedback, the Verifier helps improve both the clarity and the fidelity of the generated content.

### E Details of Webpage Image Selection

During the research stage, the Researcher actively invokes the text search tool. Through this process, Jina Reader extracts the full content of retrieved web pages, including the URLs of embedded images. We first download all accessible images from these web pages.

To ensure image quality, we apply a rule-based filtering step to remove low-quality or irrelevant images. Specifically, we discard images with low resolution, extremely small dimensions, extreme aspect ratios, as well as SVG files. This step effectively filters out non-informative visuals such as logos, icons, and banner images.

The remaining images are then processed in batches by a VLM. Conditioned on the Planner’s specified image requirements, as well as its own assessment of image quality and semantic relevance, the VLM determines whether each image should be retained or discarded.

Finally, for each section, we construct a curated Visual Working Memory, which serves as the candidate image set for the subsequent writing stage.

### F Efficiency Analysis

We further analyze the computational efficiency of PTAH on DeepResearch Bench. Table 5 reports the average wall-clock latency of each stage. The full pipeline takes 1015 seconds on average. Among all stages, Research is the most time-consuming stage, taking 459 seconds on average, because it requires open-ended evidence collection, webpage inspection, and image-pool construction for multiple report sections. Test-Time Scaling (TTS) takes 243 seconds on average, reflecting the additional cost of verifier-guided section refinement, image refinement, overall refinement, and HTML refinement. These results show that PTAH introduces additional computation compared with text-only deep-research agents, but the cost is mainly concentrated in evidence acquisition and final multi-

[Figure 71]

[Figure 72]

[Figure 73]

Figure 5: First-screen views of multimodal analytical reports generated by the PTAH framework.

Stage Avg. Time (s)

Planning Stage 192 Research Stage 459 Writing Stage 121 TTS 243

Total 1015

Table 5: Stage-wise average latency of PTAH on DeepResearch Bench. Research-stage latency is measured as the wall-clock time of parallel section-level investigations.

modal refinement, which are essential for credible multimodal report generation.

We also evaluate the efficiency benefit of parallel section-level research. As shown in Table 6, executing Researchers in parallel reduces the average Research-stage latency from 1328 seconds to 459 seconds. This corresponds to a 65.4% reduction in wall-clock time, or a 2.89× slowdown when the same Researcher agents are executed sequentially. The result indicates that the multi-agent design of PTAH is not merely an added source of computation; it also serves as an important efficiency mechanism by decomposing a long-form research task into section-level investigations that can be performed concurrently.

Finally, we examine the latency impact of verifier strength. Table 7 compares the current Verifier with a stronger reasoning model, DeepSeek-R1.

Research Execution Avg. Time (s) Relative Change

Parallel 459 1.00× Sequential 1328 2.89× slower

Table 6: Latency comparison between parallel and sequential Researcher execution. Parallel section-level research substantially reduces the wall-clock latency of the Research stage.

Replacing the current Verifier with DeepSeek-R1 increases Planning latency from 192 seconds to 853 seconds and Research latency from 459 seconds to 1408 seconds. This increase comes from both the longer reasoning time of the stronger verifier and the additional revision rounds triggered by stricter verification. Therefore, verifier selection introduces a clear quality–efficiency trade-off: stronger verifiers may provide stricter intermediate checking, but they can substantially increase the overall latency of long-form multimodal report generation. In our main experiments, we use the current Verifier setting as a balanced configuration that preserves strong factual grounding and cross-modal consistency while avoiding excessive revision overhead.

#### F.1 User-Centric Human Evaluation

We conduct a user-centric human evaluation to further examine the practical reading experience of multimodal deep-research reports. While

###### Setting Time (s)

Current Verifier – Planning 192 Current Verifier – Research 459 DeepSeek-R1 Verifier – Planning 853 DeepSeek-R1 Verifier – Research 1408

Table 7: Latency impact of verifier strength. A stronger reasoning verifier substantially increases both Planning and Research latency due to more expensive verification and additional revision rounds.

PTAHEval provides a scalable automatic protocol for evaluating image content quality and multimodal presentation quality, human evaluation offers complementary evidence on whether the generated reports are readable, usable, and helpful for information acquisition in realistic viewing scenarios.

We randomly sample 20 reports from DeepResearch Bench and compare the rendered HTML reports generated by PTAH and WebThinker. WebThinker is selected as the comparison system because it is the strongest text-only deep-research baseline in our experiments. We recruit four evaluators, including two AI PhD students as expert evaluators and two AI undergraduate students as general users. Each evaluator is asked to compare the two reports for the same query and judge whether PTAH wins, ties, or loses against WebThinker along four user-centric dimensions: readability, usability, information acquisition efficiency, and overall preference. Readability measures whether the report is easy to read and visually clear. Usability measures whether the report organization supports convenient browsing and understanding. Information acquisition efficiency measures whether users can quickly identify and absorb key information from the report. Overall preference captures the evaluator’s holistic judgment of which report better supports the research task.

- Table 8 reports the win-or-tie rate of PTAH over

WebThinker. PTAH achieves high win-or-tie rates across all four dimensions, with 88.75% on readability, 88.75% on usability, 96.25% on information acquisition efficiency, and 95.00% on overall preference. These results indicate that the multimodal reports generated by PTAH are not only preferred by automatic evaluators, but also provide a better practical reading experience for human users. In particular, the large gain in information acquisition efficiency suggests that interleaved visual evidence helps users locate and understand important

Evaluator Read. Usability Info. Overall

- Expert E1 85% 90% 95% 95%
- Expert E2 85% 80% 95% 90%

- General U1 90% 95% 100% 100%
- General U2 95% 90% 95% 95% Average 88.75% 88.75% 96.25% 95.00%

Table 8: User-centric human evaluation of PTAH against WebThinker on 20 sampled DeepResearch Bench reports. Each value denotes the win-or-tie rate of PTAH. “Info.” denotes information acquisition efficiency and “Read.” denotes readability.

information more effectively than text-only reports. This human evaluation complements PTAHEval by showing that the improvements in multimodal presentation quality correspond to meaningful gains in perceived usability and reading efficiency.

#### F.2 Ablation on Visual Elements

We conduct an additional same-framework ablation to isolate the contribution of visual elements in PTAH. The ablated variant, denoted as PTAH w/o images, uses the same Planning–Research–Writing pipeline, stage-wise verification, and test-time refinement procedure as the full PTAH system, but removes all images from the final rendered report. This setting allows us to examine whether the improvement comes from the multimodal visual elements themselves or only from the underlying agentic research framework.

Table 9 reports the results on DeepResearch Bench. Removing images only slightly changes the text-oriented DRB overall score, from 45.16 to 45.10. This is expected because DeepResearch Bench primarily evaluates textual research quality and does not directly reward whether visual evidence is relevant, evidential, or helpful for reading. Importantly, PTAH w/o images still outperforms WebThinker on DRB overall, indicating that the Planning–Research–Writing framework and stagewise verification preserve strong textual research quality even without visual output.

In contrast, removing images leads to a clear drop in multimodal presentation quality. The MPQ average score decreases from 3.71 to 3.29 after visual elements are removed. This result shows that interleaved visuals make a substantial contribution to the rendered report’s multimodal presentation quality, rather than merely changing its surface appearance. Compared with WebThinker, the full PTAH system improves both DRB overall and

###### Method DRB Overall MPQ Avg.

WebThinker 45.00 3.11 PTAH w/ images 45.16 3.71 PTAH w/o images 45.10 3.29

- Table 9: Ablation on visual elements. PTAH w/o images keeps the same framework as PTAH, but removes all images from the final report.

MPQ average, suggesting that PTAH enhances multimodal readability and presentation while maintaining competitive textual research quality.

### G Example Cases

As shown in Figure 5, we present three representative first-screen views of multimodal analytical reports generated by the PTAH framework.

### H Usage of AI Assistants

The authors used ChatGPT only for language polishing and grammar correction during the preparation of this manuscript. All research content and technical contributions were developed and verified by the authors.

