# arXiv:2601.12346v1[cs.CV]18Jan2026

[Figure 1]

[Figure 2]

## MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents

Peizhou Huang*, Zixuan Zhong*, Zhongwei Wan*, Donghao Zhou*, Samiul Alam, Xin Wang, Zexin Li, Zhihao Dou, Li Zhu, JingXiong, Chaofan Tao, Yan Xu,

Dimitrios Dimitriadis, Tuo Zhang, Mi Zhang

* Equal Contribution OSU, Amazon, UMich, UCL, CUHK, UCR, CWRU, HKU Correspondence: Tuo Zhang tuozhang@amazon.com, Mi Zhang mizhang.1@osu.edu Project Page: https://mmdeepresearch-bench.github.io

### Abstract

Deep Research Agents (DRAs) generate citation-rich reports via multi-step search and synthesis, yet existing benchmarks mainly target text-only settings or short-form multimodal QA, missing end-to-end multimodal evidence use. We introduce MMDeepResearch-Bench (MMDR-Bench), a benchmark of 140 expert-crafted tasks across 21 domains, where each task provides an image–text bundle to evaluate multimodal understanding and citation-grounded report generation. Compared to prior setups, MMDR-Bench emphasizes report-style synthesis with explicit evidence use, where models must connect visual artifacts to sourced claims and maintain consistency across narrative, citations, and visual references. We further propose a unified, interpretable evaluation pipeline: Formula–LLM Adaptive Evaluation (FLAE) for report quality, Trustworthy Retrieval-Aligned Citation Evaluation (TRACE) for citation-grounded evidence alignment, and Multimodal Support-Aligned Integrity Check (MOSAIC) for text–visual integrity, each producing fine-grained signals that support error diagnosis beyond a single overall score. Experiments across 25 state-of-the-art models reveal systematic trade-offs between generation quality, citation discipline, and multimodal grounding, highlighting that strong prose alone does not guarantee faithful evidence use and that multimodal integrity remains a key bottleneck for deep research agents.

### 1 Introduction

Recent advancements in foundation models have driven a shift from language-centric systems to large multimodal models (LMMs) that jointly process text and visual inputs [1]. Enabled by vision–language pretraining and instruction tuning [2–4], modern LMMs can reason over structured visual artifacts such as charts and documents, forming the basis of current vision–language benchmarks [5–8].

Yet static models remain limited by fixed parametric memory, motivating retrieval-augmented generation [9] and tool-using agents that browse and collect external evidence [10–12]. Building on this paradigm, Deep Research Agents (DRAs) target open-ended, long-horizon tasks by iteratively retrieving sources, reconciling hypotheses, and producing research-style reports [13–15]. Since real research is rarely text-only, DRAs must also align textual claims with figures, charts, and diagrams, motivating multimodal deep research [15–17].

60

[Figure 3]

50

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

40

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

FinalScore

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

30

20

10

0

Gemini-DRGemini-3-ProGemini-3-FlashDeepSeek-V3.2GPT-5miniGemini-2.5-FlashGemini-2.5-ProPerplexity-Sonar-DRGPT-4.1Kimi-K2Grok-4Qwen-3Qwen-3-VLGPT-4.1miniClaude-4.5-OpusClaude-4.5-HaikuClaude-4.5-SonnetGPT-5.2GPT-5.1o3-miniGrok-3ChatGPT-DRTongyi-DR(30B)GPT-4oGPT-4.1nano

Figure 1 Overall MMDR-Bench score (0–100; higher is better) on 140 tasks for representative tool-using LMMs and Deep Research systems, ranked by score.

As DRAs proliferate [16, 17], evaluation becomes crucial but difficult: intermediate reasoning and retrieval are opaque, and openended questions seldom admit a single gold answer, making the final cited report the primary evaluation interface [17]. Existing benchmarks either isolate web or retrieval competence [10, 11], focus on text-only deep research reports [15, 16], or emphasize shorthorizon multimodal perception [8, 18]. Live retrieval further complicates evaluation via issues such as search-time data contamination [19]. This leaves a gap: a unified benchmark for end-to-end deep research with multimodal sources.

Deep Research Level (Integrated)

###### Long-Form Report Synthesis

Multimodal Task

Visually-Grounded Planning

###### Citation-Grounded Reasoning

Understanding

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| | |
|---|---|
| | |

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Link claims with

Produce citation-

Use visuals to set search plan and

Jointly parse

multimodal sources, verify support

rich multimodal

and identify visual-

reports

iteratively refine

textual content

Foundational Level (Atomic)

###### Instruction Following

Long-Context

###### Web Search Tools

Visual Perception

Understanding

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Multi-step browse, retrieve, call

Digest long multi-

Respect constraints

Read visual elements,

source inputs,

and produce aligned output

extract entities,

tools，collect

maintain consistency

numbers, relations

sources

Figure 2 MMDR-Bench evaluates multimodal deep research abilities at both integrated and atomic levels.

To fill this gap, we introduce MMDeepResearch-Bench (MMDR-Bench), a 140-task benchmark that targets long-horizon research workflows with both textual and visual evidence. Each task instance is packaged as an image–text bundle, to jointly evaluate the integrated deep-research capabilities and their atomic foundations, spanning multimodal understanding, citation-grounded reasoning, and long-form multimodal report synthesis capabilities of DRAs, as illustrated in Figure 2. MMDR-Bench includes two complementary regimes, Daily and Research, reflecting lightweight everyday usage and analysis-heavy research settings. All tasks are iteratively refined by doctoral-level domain experts to ensure multimodal necessity and verifiability.

Building on MMDR-Bench, we develop an evaluation framework along three aspects: Formula–LLM Adaptive Evaluation (FLAE) for long-form report quality, Trustworthy Retrieval–Aligned Citation Evaluation (TRACE) for citation-grounded support and source quality, and Multimodal Support–Aligned Integrity Check (MOSAIC) for text–image consistency of the report. The evaluation results are shown in Figure 1.

The contributions of our work are summarized as follows:

- • MMDR-Bench: A Novel Multimodal Research Benchmark. We introduce MMDR-Bench, the first end-to-end benchmark specifically designed to evaluate Deep Research Agents (DRAs) in multimodal settings. It comprises 140 expert-crafted tasks spanning 21 diverse domains. These tasks are organized into two complementary regimes—Daily and Research—to reflect the complexity of both casual information seeking and information-dense technical analysis. Every task was iteratively refined by doctoral-level experts to ensure multimodal necessity and verifiability.
- • A Unified Multi-Stage Evaluation Framework. We propose a comprehensive evaluation pipeline consisting of three specialized modules to assess the multifaceted nature of research reports. This framework includes FLAE (Formula-LLM Adaptive Evaluation) for measuring structural and insightful report quality, TRACE (Trustworthy Retrieval-Aligned Citation Evaluation) for auditing citation-grounded reasoning, and MOSAIC (Multimodal Support-Aligned Integrity Check) for verifying the consistency between textual claims and visual artifacts.
- • Enforcement of Faithfulness via Visual Evidence Fidelity. Within the evaluation pipeline, we introduce Visual Evidence Fidelity (VEF), a rigorous metric that enforces strict alignment between an agent’s claims and the provided visual evidence. By implementing a hard PASS/FAIL constraint thresholded against a task-specific textualized visual ground truth, VEF ensures that agents are held accountable for misinterpreting critical visual data or generating hallucinations.
- • Systematic Evaluation and Open-Source Contribution. We conduct an extensive evaluation of 25 stateof-the-art LLMs and agentic systems, encompassing single-modal baselines, web-enabled models, and specialized agents. Our findings reveal persistent trade-offs between writing quality, citation discipline, and multimodal grounding. To foster reproducibility and further community development, we publicly release the full benchmark dataset, evaluation source code, and comprehensive metrics.

### 2 Related Work

Deep Search and Agentic Reasoning. Early agentic search frameworks decompose queries into sequential subtasks via chain-of-thought reasoning [14, 20]. Recent large reasoning models further introduce explicit phases of exploration and self-correction. Methods such as Search-R1 [21] and DeepDive [22] leverage reinforcement learning to improve search trajectories and query refinement. However, evaluation in this line of work still largely emphasizes final answer accuracy against ground-truth labels, which may overlook failures in the underlying research process.

Multimodal Search and Reasoning. To support web-scale, visually grounded information seeking, recent multimodal agents aim to interpret heterogeneous content beyond plain text. Benchmarks such as MMSearch [23] and BrowseComp-Plus [24] evaluate capabilities including visual re-ranking and image-conditioned reasoning [25]. Most existing settings emphasize whether an agent can locate the correct image or answer a localized question, but rarely test whether subtle visual details are correctly used to substantiate claims in a long-form research report.

Benchmarks and Evaluation. Designing deep research benchmarks requires both task realism and evaluability. BrowseComp [26] and BrowseComp-Plus [24] improve fairness via fixed corpora that reduce sensitivity to web drift, while DeepResearch Bench [15] and DeepScholar [16] examine long-form synthesis and report writing. Yet a core gap persists: text-only deep research benchmarks do not test multimodal evidence use, and multimodal benchmarks typically focus on short-form QA. Our work targets this intersection by introducing a protocol for multimodal search and citation-grounded report generation.

### 3 Dataset Collection

We define multimodal deep research as tasks that require multi-round web browsing, evidence gathering, and report synthesis while explicitly interpreting and using provided images. Following this definition, MMDRBench targets realistic long-horizon cases that require multimodal understanding and evidence-grounded

Entertainment Characters Current Events Information Updates Miscellaneous Science Advice & Recommendations Satire / Humor History Health Games Other Interdisciplinary Humanities & Culture Social & Policy Environment & Energy Economics & Business Mathematics & Engineering Biomedical Science Computer & Data Science

6%

14%

[Figure 55]

[Figure 56]

Research Task: Religious Science Domain

6%

Daily Task:

[Figure 57]

[Figure 58]

Health Domain

4%

3%

My eyes are experiencing

Images depict (a) a GPT-style architecture

-

Daily Tasks

14%

dryness and itching, along with excessive tearing. I have

diagram, (b) scaled dot-product attention, and (c) absolute positional encoding. Task:1) Extract

-

3% 2%

no history of allergies. Is the

10+ facts directly from the images (module

28.6%

2% 2%

eyedrop shown in the picture

names, dataflow, what attention computes, what positional encoding looks like). 2) Using open

suitable for my condition? Is there any other eye drops

sources, state the time and memory complexity

2% 1%

28.

71.4%

available for me to purchase

of self-attention for sequence length L and

-

12%

Research Tasks

in the London area?

hidden size d.\n Provide citations.

- 1%
- 2%

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

3%

10%

6% 8%

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

By PhD in Biological Medicine

By PhD in Computer Science

Figure 3 Task distribution of MMDR-Bench.

Figure 4 Two example tasks from MMDR-Bench.

report writing. Each task is an image–text bundle: a textual query paired with a small set of images that must be interpreted and integrated into a research report. This design jointly evaluates (i) multimodal question understanding and (ii) evidence-grounded multimodal report generation with citations.

With a large query pool (98k+ real queries) that we collected, we construct MMDR-Bench as 140 tasks across 21 domains (Figure 3), organized into two regimes: Daily (40 tasks across 10 domains) with casual visuals such as screenshots and UI captures, and Research (100 tasks across 9 domains) with structured, information-dense figures such as charts, tables, and diagrams that require deeper synthesis. Two example tasks are shown in Figure 4.

### 4 Evaluation Methodology

Our evaluation framework for multimodal Deep Research agents assesses both multimodal information retrieval through multi-round search and the quality of the generated research report. To this end, we develop an evaluation pipeline consisting of three complementary components: FLAE, TRACE, and MOSAIC. The workflow of MMDR-Bench is shown in Figure 5.

The pipeline runs sequentially: for each Deep Research report, we compute FLAE and TRACE in parallel, and trigger MOSAIC only when the text layer derives a valid non-zero score. Let SF and ST denote the FLAE and TRACE scores, and let τF and τT be the corresponding gating thresholds. The MOSAIC evaluation activates if and only if SF ≥ τF and ST ≥ τT. If MOSAIC is not activated, we set its score to zero.

#### 4.1 FLAE: Formula-LLM Adaptive Evaluation

Evaluating long-form Deep Research reports is challenging because writing requirements vary across tasks and domains. Fixed rubrics often underfit this diversity, while LLM-as-a-judge is harder to audit. We introduce FLAE (Formula–LLM Adaptive Evaluation), which combines a reproducible formula score from text features and a task-aware LLM judge score, then fuses them with an adaptive coefficient for interpretability.

FLAE evaluates each report on three task-agnostic dimensions: Readability (Read.), Insightfulness (Insh.), and Structural Completeness (Stru.). We use a Judge LLM to generate task-adaptive weights over the three dimensions, improving robustness across heterogeneous tasks (details and justification in Appendix A.2.1).

Formula-Based Channel. We extract lightweight statistics ϕ(R) such as lexical diversity, section structure, sentence-length distribution, and compliance indicators and compute per-dimension scores via fixed, auditable transforms:

sformd (R) = fd ϕ(R) , d ∈ D, (1) where D = {Read.,Insh.,Stru.}. This channel is fully reproducible and provides a stable evaluation without access to any judge model. Full feature definitions and the complete fixed formulas fd(·) are provided in Appendix A.3.2.

MOSAIC Evaluation Multimodal

Multimodal

Gate

Dynamic Weighting

Dynamic Weighting

Multimodal

Multimodal

Itemwise Score

Weight

Weight

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Judge LLM

Judge LLM

Deep Research

Deep Research

SEM: Wsem, ACC: Wacc, VQA: Wvqa

[Figure 75]

[Figure 76]

Report

Report

Route

Categorical Evaluators

[Figure 77]

[Figure 78]

Judge LLM

Judge LLM

Parse

Multimodal Items Collection

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Item 1 Item 2

Item n

...

...

...

...

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

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

Main Pipeline

Deep Research

Deep Research

Multimodal

###### Final Score

Agents

[Figure 107]

[Figure 108]

Deep Research

Deep Research

[Figure 109]

[Figure 110]

Report

Report

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Multimodal Deep Research

Multimodal Deep Research

TRACE Evaluation FLAE

[Figure 117]

[Figure 118]

Pairwise Score

Report

Report

Dynamic Weighting

Dynamic Weighting

[Figure 119]

[Figure 120]

Judge LLM Judge LLM

Judge LLM Judge LLM

[Figure 121]

[Figure 122]

CON: Wcon

VEF: Wvef

Dynamic

[Figure 123]

[Figure 124]

Weighting

COV: Wcov, FID: Wfid

Weight

Weight

Multimodal Deep Research

Multimodal Deep Research

[Figure 125]

[Figure 126]

*T

Report

Report

Claim–URL Pairs

Extract

[Figure 127]

[Figure 128]

Claim n

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Claim 2

Claim 1

... ...

... ...

[Figure 133]

[Figure 134]

Judge LLM

Judge LLM

[Figure 135]

[Figure 136]

URL n

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

URL 2

URL 1

Pair 1 Pair 2 Pair n

Figure 5 The MMDR-Bench evaluation pipeline. Reports are processed through parallel FLAE and TRACE modules, followed by a gated MOSAIC stage.

LLM Judge Channel. Given the task and report, a judge LLM with calibrated prompts outputs per-dimension scores over the three FLAE dimensions. The Judge prompts (dimension scoring, task-adaptive weighting) are in Appendix A.3.3.

Adaptive Fusion. We combine the two channels with a judge LLM calculation. To mitigate bias, the fusion weights depend only on model-agnostic, directly observable signals such as length, section presence, and formatting compliance, not on model identity. The prompt templates for adaptive fusion are detailed in Appendix A.3.

In the calculation of overall score, considering that tasks emphasize dimensions differently, we derive taskspecific weights Wd(t,R) (normalized to d Wd = 1) and compute:

Wd(t,R)sd(R), (2)

FLAE(t,R) = 100 ·

d∈D

where t is the task, sd(R) ∈ [0,1] is the fused per-dimension score after combining the formula channel and the LLM-judge channel, and the factor 100 scales the weighted average to a 0–100 score.

#### 4.2 TRACE: Trustworthy Retrieval-Aligned Citation Evaluation

To assess whether a report is verifiably grounded in cited sources and whether it faithfully addresses the intended (multimodal) task, we introduce TRACE (Trustworthy Retrieval-Aligned Citation Evaluation). TRACE measures evidence quality along with citation fidelity, which is to check whether the cited claims are supported by the referenced content, and prompt fidelity, which is to verify whether the report correctly interprets and answers the question.

Given a report, we parse citation markers section to map indices to URLs, then extract atomic claims and align them to their cited URL(s) to form claim-URL pairs with light deduplication. For each cited URL, we retrieve the referenced content and record accessibility. For accessible pages, a Judge LLM checks whether each Claim–URL pair is supported, accounting for missing evidence, contradiction, over-specificity, and causal inversion. Pair-level judgments are aggregated into three citation-fidelity metrics in [0,1]: Consistency (Con.), Coverage (Cov.), Textual Fidelity (Fid.),and Visual Evidence Fidelity (Vef.), which penalizes evidence-mismatched reasoning. Notably, We introduce Vef. as a strict prompt-faithfulness check that verifies alignment to a task-specific textualized visual requirement. The detailed prompt for judge LLM is presented in Appendix A.3.6.

TRACE uses Judge-LLM task-adaptive weights over {Con.,Cov.,Fid.}, while keeping the Vef. share fixed and treating Vef. as a strict PASS/FAIL constraint against a task-specific textualized visual ground truth: the judge returns a 0-10 score and PASS/FAIL, and we force FAIL when the score is below a threshold τVef. = 6, ensuring prompt-faithfulness is enforced consistently across task regimes (see Appendix A.1.2 and Appendix A.3.6 for more details).

Let K = {Con.,Cov.,Fid.}, and we calculate the final TRACE score as:

TRACE(t,R) = 100 λVef. · Vef(t,R) + (1 − λVef.)

k∈K

Wk(t,R)Ek(R) , (3)

where λVef. is a fixed weight and Wk(t,R) are task-adaptive weights with k Wk = 1.

#### 4.3 MOSAIC: Multimodal Support-Aligned Integrity Check

To evaluate whether a multimodal deep-research report is visually grounded, i.e., whether its image-referenced statements faithfully reflect the underlying figures, charts, diagrams, and photos, we introduce MOSAIC (Multimodal Support-Aligned Integrity Check). MOSAIC enables multimodal verification as an item-level consistency test between textual claims that reference visual artifacts and the referenced images themselves.

Multimodal Itemization and Routing. Given a generated report R, MOSAIC first constructs a grounding map from visual mentions to concrete image artifacts. We parse the report to extract multimodal items (MM-items), including inline image blocks and visually grounded paragraphs that cite an image URL. MM-items span heterogeneous visual modalities to diagrams, data charts to photos. Judging a statistical chart differs from judging a natural photo. MOSAIC therefore routes each item into a small set of visual types using a lightweight router that combines rule-based cues with optional embedding-based classification. Each routed bucket is evaluated by a type-specific multimodal judge that uses a consistent rubric but modality-appropriate checks such as numeric plausibility for charts, structural correspondence for diagrams, semantic grounding for photos.

Multimodal Support Scoring. For each MM-item i, the judge produces a vector mi of dimension scores in [0,1] across three dimensions: Visual-Semantic Alignment Sem., Visual Data Interpretation Accuracy Acc., and Complex Visual Question Answering Quality vqa. The item-level score is then computed by a weighted aggregation:

ωk = 1. (4)

si =

ωk mi,k,

k

k

MOSAIC uses LLM-based routing type-specific weighting for multimodal signals to better match different visual evidence types, with full weighting settings and justification in Appendix A.2.1.

FLAE TRACE MOSAIC

Model Overall

Read. Insh. Stru. Vef. Con. Cov. Fid. Sem. Acc. VQA Single-Modal, w/o Search

OpenAI o3-mini 31.96 53.75 52.65 37.11 13.57 28.45 33.74 48.35 15.47 90.00 12.60 DeepSeek-V3.2 43.71 75.37 87.82 58.16 19.28 33.34 45.48 18.77 42.19 83.85 12.88 Kimi K2 (Thinking) 36.91 71.34 77.27 47.34 17.14 23.54 24.62 27.20 42.00 90.00 9.50 Qwen 3 235B (A22B) 36.04 77.56 85.74 54.05 17.14 35.60 45.73 22.98 20.43 53.09 4.95

Multimodal, w/o Search Qwen 3 VL 235B (A22B) 35.08 77.01 86.48 52.21 43.57 18.34 15.25 10.68 30.58 93.52 16.98

- GPT-4o 28.62 52.52 68.41 40.90 10.04 10.94 4.61 11.89 24.10 71.43 18.72 GPT-4.1 36.95 79.34 89.04 53.00 39.29 15.90 10.06 5.61 29.66 80.56 19.92

- GPT-4.1 mini 34.23 71.25 83.62 49.60 12.86 24.20 25.44 12.33 32.62 89.91 13.21

- GPT-4.1 nano 28.07 49.77 64.82 37.28 10.79 18.99 19.86 24.42 27.02 76.30 13.04

GPT-5 mini 38.49 70.06 81.73 47.18 39.29 20.02 26.64 32.61 33.90 94.23 15.60

- GPT-5.1 32.69 79.34 89.04 53.00 35.71 15.90 2.30 13.67 22.03 84.29 14.32

- GPT-5.2 32.76 69.75 83.92 54.31 46.43 14.00 1.43 5.30 12.83 50.00 9.16

- Grok-3 29.89 75.17 86.13 52.24 20.00 12.57 5.79 2.80 22.18 68.39 13.89
- Grok-4 (Fast Reasoning) 36.10 60.62 80.49 52.99 36.43 17.30 14.62 6.12 28.46 87.45 19.34 Multimodal, w/ Search

Claude 4.5 Haiku 33.67 74.60 81.80 53.22 28.57 17.90 14.10 18.56 25.98 76.90 11.70 Claude 4.5 Sonnet 33.61 77.63 82.31 51.65 32.14 14.36 15.09 16.11 20.73 70.13 14.41 Claude 4.5 Opus 33.84 77.81 83.86 50.70 35.00 30.64 41.14 21.97 21.30 77.21 14.75 Gemini 2.5 Flash 38.40 56.22 68.58 55.44 32.86 25.35 27.77 38.30 40.67 75.96 25.49 Gemini 2.5 Pro 38.04 80.04 85.94 51.44 38.57 30.18 28.77 14.98 19.47 92.86 12.50 Gemini 3 Flash 44.43 81.22 90.22 52.00 45.71 31.95 35.07 15.42 36.61 87.31 18.99 Gemini 3 Pro 44.68 58.05 75.39 49.85 46.43 37.98 41.85 6.46 40.69 80.44 23.15

Deep Research Agent

Tongyi Deep Research (30B-A3B) 29.02 54.27 62.67 40.07 12.86 25.99 30.87 24.25 20.39 93.33 20.39 Perplexity Sonar Deep Research 37.55 62.29 64.35 47.80 27.86 33.12 41.51 16.68 50.79 87.75 21.22 ChatGPT Deep Research (o3-mini) 29.50 52.40 63.61 37.30 29.29 10.19 4.16 11.07 27.32 73.44 21.75 Gemini Deep Research (Gemini 3 Pro) 49.41 84.53 89.56 70.86 35.71 56.17 52.84 31.29 41.29 87.54 28.45

Table 1 Overall results on MMDR-Bench. Best scores in each column are highlighted.

- 5 Experiments

#### 5.1 Experimental Setup

Benchmark and Protocol. We use Gemini-2.5-Pro as the Judge LLM for all judge-in-the-loop steps, with temperature set to 0.2. The overall MMDR-Bench score is a weighted combination of the three modules: FLAE (20%), TRACE (50%), and MOSAIC (30%). We assign the largest weight to TRACE because citation-grounded evidence quality is the most central requirement for deep research, while MOSAIC evaluates the additional report-quality constraints introduced by visual evidence. FLAE is weighted lower as writing quality is less safety-critical and can be partially reflected by evidence and multimodal consistency. We further apply a gated MOSAIC stage with thresholds τF = τT = 0. The unscorable cases in the evaluation process are handled by a reason-aware validity penalty; the taxonomy is provided in Appendix A.5.2 for reference purposes and ensuring reproducibility.

Evaluated Report Models. We evaluate a diverse set of systems spanning three tiers: (i) Single-modal LLM baselines without web search function, (ii) multimodal LLM baselines without web search function, (iii) Multimodal LLM baselines with web search function and (iv) specific deep research agents. For each model, we report the overall MMDR-Bench score and metric dimensions for FLAE, TRACE and MOSAIC.

#### 5.2 Main Results and Findings

Table 1 reports overall performance and metric breakdowns. Gemini Deep Research ranks first overall, driven by strong evidence quality (TRACE consistency/coverage) while maintaining competitive multimodal alignment (MOSAIC). Among non-agent, web-enabled models, Gemini 3 Pro (Preview) is the strongest, and GPT-5.1/5.2 together with GPT-4.1 form a close cluster with complementary strengths. We observe clear cross-metric trade-offs: GPT-4.1 achieves the best multimodal extraction accuracy (Acc.) and strong evidence fidelity (Fid.), while GPT-5.2 attains the highest Vef. score, indicating better visual grounding but not uniformly better citation discipline.

- Finding 1: Vision is beneficial only when it is reliable as evidence. Comparisons within the same model family such as Qwen 3 235B (A22B) vs. Qwen 3 VL 235B (A22B) show that adding vision is not a monotonic win. Although their multimodal variants improve visual grounding, the unified error analysis for Vef. in Figure 6 reveals increased detail-level extraction failures from mis-reading fine-grained literals such as numerals, dates, labels and table cells, reflecting limitations in visual prompt understanding rather than language generation or reasoning. When images provide non-substitutable evidence, vision constrains premises and improves faithfulness; otherwise, noisy or auxiliary visual inputs can introduce spurious assumptions that propagate through retrieval and synthesis, degrading correctness, with failures correlating primarily with prompt-fidelity signals. The detailed failure case analysis can be seen at Appendix A.5.3.
- Finding 2: Multimodal alignment and citation grounding can diverge. Stronger multimodal alignment or prompt-following does not guarantee more reliable citation grounding. Contrasting single-turn models with agentic systems shows that, compared to Gemini 2.5 Pro, Gemini Deep Research improves evidence aggregation and coverage via multi-step search and cross-checking, yet the unified error analysis for Vef. in Figure 6 indicates a marked rise in entity-level failures. These failures arise when entities identified correctly early become mis-attributed during later synthesis after multiple retrieval and summarization steps, especially when consolidating the overlapping sources.

Qwen-Max (Text)

Qwen-VL (Vision)

Gemini Pro (Base)

Gemini DDR (Agent)

0

20

40

60

80

100

120

AbsoluteErrorCount

20% DTE Error Increase

4.3× EMI Increase

Model Capability Comparison Agentic System Comparison

| |
|---|

Entity Mis-id (EMI) Ref & Mapping Drift (RMD) Detail/Text Extraction (DTE)

| |
|---|

| |
|---|

| |
|---|

Log/Know Conflict (LKC) Structural/Task Omission (STO)

| |
|---|

Figure 6 Unified failure mode analysis.

- Finding 3: Tool use helps, but strong backbones and richer retrieval matter most. Within single-turn web-enabled families like Claude, scaling shows limited separation on MMDR-Bench and TRACE, suggesting that retrieval interaction patterns, rather than model size alone, are the primary bottleneck. At the system level, agents can amplify strong backbones but cannot replace them: Tongyi Deep Research (30B-A3B) underperforms substantially larger models, while Gemini Deep Research (Gemini 3 Pro) combines high evidence coverage with strong overall performance. We also observe that offline models can outperform some web-enabled models on coverage (Cov.), implying agent retrieval constraints limit surfaced evidence despite tool access.

#### 5.3 Fine-Grained Domains Analysis

Figure 7 shows clear regime-level differences. On Daily tasks, domain performance is more volatile, and the most competitive models are those that robustly handle noisy, user-style visuals like screenshots. In this regime, Gemini 2.5 Flash and GPT-5.2 are the most consistently strong, with Claude Opus remaining competitive on recommendation- and explanation-heavy categories.

On Research tasks, performance separation becomes more domain-dependent. Gemini Deep Research (Gemini 3 Pro) and Gemini 3 Flash (Preview) stay strong across most research domains, while GPT-5.2 peaks on structured technical areas like Computer and Data Science. Qwen 3 VL 235B (A22B) is particularly strong on visually dense scientific domains like Environment and Energy domain, consistent with cases where charts and diagrams provide decisive evidence.

###### RESEARCH

Gemini 3 Pro

7.0 1.0 3.0 2.0 4.010.01.010.02.0 2.0 9.0 9.0 7.0 2.0 2.0 3.0 3.0 3.0 2.0 6.0 6.010.010.08.010.08.010.03.0 6.010.010.04.010.06.0 8.0 2.010.02.0 7.0 2.0 4.0 1.0 3.010.03.0 2.0 3.0 1.0 3.010.02.0 2.0 4.0 1.0 3.0 2.0 3.0 5.0 2.0 2.0 4.0 4.0 6.0 5.0 4.0 3.0 3.0 3.0 3.0 2.0 3.0 3.0 9.0 4.0 5.0 4.0 1.0 3.0 6.0 3.0 1.0 3.0 3.0 3.0 3.010.03.0 2.0 4.0 3.0 2.0 4.010.02.0 3.0 3.0 2.0 3.0 4.0 9.0

GPT-5.2

1.0 3.0 3.0 3.0 2.010.01.0 1.0 4.0 2.0 3.0 3.0 9.0 2.0 2.0 2.0 3.0 3.0 3.0 1.0 2.0 1.0 1.010.010.05.010.01.0 3.010.06.0 7.0 3.0 3.0 9.0 3.0 3.0 2.0 3.0 7.0 3.0 1.0 3.0 6.0 1.0 1.0 3.0 1.0 6.0 9.010.04.0 2.0 1.0 9.0 9.0 2.0 3.010.04.0 2.0 9.0 6.010.09.010.03.0 4.0 2.0 2.0 1.0 1.0 9.010.010.03.010.09.0 7.0 9.0 8.0 3.0 3.0 4.0 2.0 4.0 1.0 6.0 1.010.03.0 3.010.09.0 2.0 3.0 3.0 2.0 3.0 3.0

Gemini 3 Flash

10.03.0 7.0 3.0 4.010.01.0 9.0 5.0 1.0 7.0 2.0 3.0 2.010.02.010.03.0 2.0 4.0 5.0 9.010.010.02.0 9.0 9.0 9.010.03.010.06.0 4.0 3.0 3.0 2.010.09.010.010.09.0 6.0 5.010.010.06.0 1.0 3.0 1.0 8.0 3.0 3.010.06.0 2.0 3.0 3.0 4.0 2.0 3.0 3.0 2.0 4.0 7.0 4.0 3.0 2.0 3.0 3.0 2.0 2.0 3.0 4.0 9.0 3.0 4.0 4.010.02.0 2.0 3.0 9.0 2.0 3.0 3.0 4.0 2.0 1.0 3.0 4.0 1.0 3.0 2.0 3.0 2.0 3.0 3.0 2.0 1.0 2.0

Qwen 3 VL 235B (A22B)

8.0 2.0 3.0 7.0 3.0 2.0 1.0 2.0 4.0 2.0 4.0 9.0 2.010.04.0 4.010.01.0 2.0 1.0 3.010.02.0 2.0 2.010.09.010.02.0 9.0 3.0 9.010.05.0 6.0 3.0 2.0 1.010.02.0 6.0 6.0 3.0 1.0 4.010.02.010.02.010.010.02.0 9.0 2.0 1.0 1.0 2.0 4.010.010.03.0 3.010.09.0 6.0 2.0 1.0 9.0 3.0 1.0 2.0 4.010.010.08.0 3.0 2.010.05.0 6.010.01.0 1.0 9.0 2.010.02.0 9.0 3.0 2.0 3.0 2.0 2.0 1.0 3.010.02.0 3.0 3.0 2.0

GPT-4.1

5.010.03.0 2.0 3.0 2.010.02.0 7.0 2.0 9.0 8.0 9.0 1.0 2.0 2.0 2.0 3.0 2.0 2.0 3.0 1.0 3.0 3.0 1.0 2.010.09.0 2.0 7.010.03.0 4.0 6.010.06.0 8.0 2.0 1.010.010.01.0 1.010.09.0 2.0 1.0 2.0 9.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 1.0 4.0 2.0 3.0 3.0 2.0 8.0 3.0 4.0 3.0 3.0 3.0 2.0 2.0 3.010.07.0 6.0 3.0 3.0 4.0 3.0 8.0 3.0 1.010.05.0 3.0 3.0 2.0 3.0 4.0 2.0 3.0 2.0 3.0 2.0 2.0 1.0 3.0 6.0 2.0 6.0 3.0

###### EXP XDIS HCS SPS EES EBS MES LHS CDS

Figure 7 Domain-level score breakdown on MMDR-Bench, restricted to the Research regime. Domain names and abbreviations are: Other Exploratory Topics (EXP), Interdisciplinary Studies (XDIS), Humanities & Cultural Studies (HCS), Social & Policy Studies (SPS), Environment & Energy Studies (EES), Economics & Business Studies (EBS), Mathematics & Engineering (MES), Life & Health Sciences (LHS), and Computer & Data Science (CDS). Higher values indicate stronger performance.

#### 5.4 Human Consistency Check Method PAR↑ OPC↑

We evaluate alignment between our evaluator and expert judgments on open-ended multimodal reports. We use the full 140 tasks from both Daily and Research regimes and collect reports from all evaluated models. For each task, we form balanced system pairs by stratifying on overall score and tier (to avoid trivial comparisons), and sample a fixed number of pairs per task. Twelve expert annotators independently assess report pairs: for each pair, three experts provide an overall preference and a coarse score on the same rubric, and we aggregate by majority vote. Evaluator–human agreement is measured by pairwise agreement (PAR) on preferences and score correlation (OPC), where OPC is the Pearson correlation between system-level mean scores computed by averaging over tasks and sampled pairs. To reduce confounds from the evidence pipeline, we also manually audit a subset of sampled pairs by spot-checking extracted Claim–URL units and their supporting snippets, confirming that observed disagreements are dominated by borderline evidence interpretation rather than systematic extraction drift. Finally, to validate that the judge-generated fusion coefficient α(t,R) contributes beyond deterministic observables, we replace α(t,R) with a transparent heuristic computed from compliance signals and report the change in PAR/OPC in Appendix A.3.5 (with all fixed formula coefficients and prompts released for exact reproduction).

Vanilla Prompt Judge 61.2 93.0

MMDR-Bench-Eval (Full) 73.5 96.4

w/o Vef. 68.0 95.2 w/o MOSAIC 70.1 95.8

Human Inter-Annotator Agreement 69.8 –

Table 2 Human consistency on MMDR-Bench. PAR: agreement with majority expert preferences; OPC: Pearson correlation of system-level average scores.

As shown in Table 2, the full evaluator aligns more closely with expert preferences than a vanilla prompt-based judge. Ablations show that both Vef. and MOSAIC improve human-aligned scoring, supporting our choice to keep Vef. as a fixed-share requirement in TRACE while letting the remaining citation-fidelity weights adapt by task (full details are provided in Appendix A.2.1).

#### 5.5 Robustness of Judge Model and Weights

We test robustness to judge backbones and aggregation weights via a cross-judge re-scoring experiment. We fix the report set to the 140 outputs produced by Gemini-2.5-Pro and re-evaluate them with two judge backbones under the same parsing, retrieval, and aggregation pipeline.

- Table 3 shows that judges differ in module-level scoring tendencies, which we attribute to judge-specific inductive biases (e.g., emphasis on conservative evidence attribution versus stricter prompt-faithfulness and multimodal precision), rather than evaluator fragility.

In our runs, GPT-5.2 is more stringent on FLAE and applies a stricter Vef. criterion, where borderline semantic matches to the fixed visual ground truth drive most absolute variance, while MOSAIC remains

###### highly stable across judges, as shown in Table 3.

Despite these per-stage shifts, the overall MMDR score is stable: the mean changes from 36.76 to 37.06, an absolute difference of 0.30 points (about 0.8% relative). This indicates that the three-stage design balances complementary signals of judge LLMs, without changing the conclusions.

Metric Gemini-2.5-Pro GPT-5.2

AVG FLAE 61.89 45.82 AVG TRACE (w/o Vef.) 28.39 39.87 AVG Vef. 38.57 26.42 AVG MOSAIC 29.44 29.53

AVG MMDR (w/ Vef.) 36.76 37.06 AVG MMDR (w/o Vef.) 38.04 35.23

We also perturb the aggregation weights around (wF,wT,wM) = (0.2,0.5,0.3) and sweep feasible integer triples. The top system and top tier remain unchanged, while ablating MOSAIC shifts rankings toward text-centric systems.

Table 3 Cross-judge robustness on the same report set. GPT-5.2 is stricter on Vef., while MOSAIC is stable across judges.

### 6 Conclusion

In this work we introduce MMDR-Bench, the first comprehensive benchmark for end-to-end multimodal deep research. Building on the benchmark dataset comprising 140 tasks across 21 domains under Daily and Research regimes, we further propose a unified evaluation pipeline that jointly measures report quality, citation-grounded faithfulness, and text–visual evidence consistency. Results across 25 state-of-the-art LLMs and DRAs reveal persistent trade-offs between writing quality, citation discipline, and multimodal grounding.

### References

- [1] Rishi Bommasani. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

- [2] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

- [3] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

- [4] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

- [5] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

- [6] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.

- [7] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.

- [8] Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, et al. Chartmuseum: Testing visual reasoning capabilities of large vision-language models. arXiv preprint arXiv:2505.13444, 2025.

- [9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

- [10] Ryan Wong, Jiawei Wang, Junjie Zhao, Li Chen, Yan Gao, Long Zhang, Xuan Zhou, Zuo Wang, Kai Xiang, Ge Zhang, et al. Widesearch: Benchmarking agentic broad info-seeking. arXiv preprint arXiv:2508.07999, 2025.

- [11] Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao, Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Weiwen Liu, Yasheng Wang, et al. Infodeepseek: Benchmarking agentic information seeking for retrieval-augmented generation. arXiv preprint arXiv:2505.15872, 2025.

- [12] Tian Lan, Bin Zhu, Qianghuai Jia, Junyang Ren, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, and Kaifu Zhang. Deepwidesearch: Benchmarking depth and width in agentic information seeking. arXiv preprint arXiv:2510.20168, 2025.

- [13] Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, Qiujie Xie, Xinyu Guo, Qu Yang, Jiayi Wu, Jujia Zhao, Xiaqiang Tang, Xinbei Ma, Cunxiang Wang, Jiaxin Mao, Qingyao Ai, Jen-Tse Huang, Wenxuan Wang, Yue Zhang, Yiming Yang, Zhaopeng Tu, and Zhaochun Ren. Deep research: A systematic survey, 2025. URL https://arxiv.org/abs/2512.02038.
- [14] Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam. Assisting in writing wikipedia-like articles from scratch with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6252–6278, 2024.

- [15] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A comprehensive benchmark for deep research agents, 2025. URL https://arxiv.org/abs/2506.11763.
- [16] Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Zaharia, and Carlos Guestrin. Deepscholar-bench: A live benchmark and automated evaluation for generative research synthesis. arXiv preprint

- arXiv:2508.20033, 2025.

[17] Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Kung-Hsiang Huang, Yixin Mao, and Chien-Sheng Wu. Deeptrace: Auditing deep research ai systems for tracking reliability across citations and evidence. arXiv preprint

- arXiv:2509.04499, 2025.

- [18] Kai Zou, Ziqi Huang, Yuhao Dong, Shulin Tian, Dian Zheng, Hongbo Liu, Jingwen He, Bin Liu, Yu Qiao, and Ziwei Liu. Uni-mmmu: A massive multi-discipline multimodal unified benchmark. arXiv preprint arXiv:2510.13759, 2025.

- [19] Ziwen Han, Meher Mankikar, Julian Michael, and Zifan Wang. Search-time data contamination. arXiv preprint arXiv:2508.13180, 2025.

- [20] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

- [21] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

- [22] Rui Lu, Zhenyu Hou, Zihan Wang, Hanchen Zhang, Xiao Liu, Yujiang Li, Shi Feng, Jie Tang, and Yuxiao Dong. Deepdive: Advancing deep search agents with knowledge graphs and multi-turn rl. arXiv preprint arXiv:2509.10446, 2025.

- [23] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, Peng Gao, Yu Liu, Chunyuan Li, and Hongsheng Li. Mmsearch: Benchmarking the potential of large models as multi-modal search engines, 2024. URL https://arxiv.org/abs/2409.12959.
- [24] Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, et al. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025.

- [25] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

- [26] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- [27] OpenAI. o3 / o4-mini system card. https://openai.com/index/o3-o4-mini-system-card/, 2025. Official OpenAI system card for o3 and o4-mini models.
- [28] OpenAI. Gpt-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf, 2025. System card describing the GPT-5 model family.
- [29] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [30] DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jingchang Chen, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y Xu, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yiliang Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R. J. Chen, R. L. Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dongjie Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, Shengfeng Ye, T. Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, and Zihua Qu. Deepseek-v3.2: Pushing the frontier of open large language models, 2025. URL https://arxiv.org/abs/2512.02556.
- [31] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

- [32] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [33] xAI. Grok: xai’s multimodal reasoning model. https://x.ai/blog/grok, 2024. Official description of the Grok model family.

- [34] Anthropic. Claude: Constitutional ai models from anthropic. https://www.anthropic.com/claude, 2024. Official description of the Claude model family.
- [35] Gemini Team. Gemini: A family of highly capable multimodal models, 2025. URL https://arxiv.org/abs/ 2312.11805.
- [36] Google. Gemini deep research documentation. https://ai.google.dev/gemini-api/docs/deep-research, 2025. Official documentation for Gemini Deep Research endpoints.
- [37] Tongyi DeepResearch Team. Tongyi deepresearch: A new era of open-source ai researchers. https://github. com/Alibaba-NLP/DeepResearch, 2025.
- [38] Perplexity AI. Perplexity ai: Answer engine and sonar models. https://www.perplexity.ai/, 2025. Official Perplexity AI website and Sonar model family overview.

### A Appendix

#### A.1 Additional Dataset Construction Details

Two Task Regimes. MMDR-Bench contains 140 expert-crafted tasks spanning 21 domains, which are organized into two complementary regimes: Daily (40 tasks across 11 domains) and Research (100 tasks across 10 domains). Daily tasks reflect everyday deep-research workflows driven by loosely structured visual inputs, such as screenshots, photographs, and user-facing interfaces, and require lightweight yet verifiable evidence use. In contrast, Research tasks span scientific and social-science domains and emphasize analysis-heavy settings with information-dense visuals, including charts, diagrams, and tables, where models must integrate multi-source evidence and synthesize it into coherent, citation-grounded reports under stricter fidelity constraints.

Expert-Driven Task Proposal and Refinement. Domain experts propose candidate tasks, which we iteratively refine with clarity checks, multimodal-necessity checks (tasks must be image-dependent), and evidencegrounding checks (reports must be verifiable via citations). This process yields the final 140-task benchmark.

Multimodal Packaging, Difficulty, and Multilinguality. Each task is packaged as an image–text bundle with a variable number of images. We annotate instance difficulty (easy, hard, complex) and record the task language. The benchmark is multilingual, dominated by English and Chinese, with additional languages in the long tail.

###### A.1.1 Standardized Report Generation Protocol

We prompt models to generate reports grounded in Claim–URL verifiability and inline visual support. Citations should appear immediately after factual claims, and each citation index must map to exactly one URL in a References: block. For image-dependent claims, the report should embed the referenced image inline with a short caption carrying the same citation index, and task-provided input images should be embedded before drawing image-dependent conclusions. We encourage diverse sources (Daily about 6 and Research about 10 when feasible) while avoiding social media and question-answering forums, and we note that Daily inputs can be noisy or partial.

###### A.1.2 Textualized Visual Ground Truth for Vef.

Vef. uses a task-specific textualized visual ground truth (visual GT) that records only what is directly observable from the task-provided images, without adding any external/background knowledge. Concretely, when a domain expert authors a benchmark task, they additionally write a concise visual GT by describing the key visual facts in the image bundle as-is. Depending on the visual type, the GT may include: salient entities/objects and their identities; explicit numbers, labels, and table entries; chart titles, axes, units, legends, and trend/ordering relations; and screenshot/UI states such as selected options, warnings, and on-screen text. This GT is intended to be minimal yet sufficient to distinguish correct visual interpretation from visually grounded hallucinations such as mis-identifying the central entity in an image.

Decision Rule and Threshold. Vef. is evaluated as a strict PASS/FAIL check. The judge outputs an integer score in {0,...,10} together with a PASS/FAIL verdict, and we apply a fixed threshold τVef. = 6: any case with score < τVef. is forced to FAIL even if the raw verdict is PASS. In addition, identity-critical errors are

treated as immediate FAIL, including wrong visual identity and false presence, consistent with the judge instructions in Appendix A.3.6.

Versioning, Drift Handling, and Regression Protection. Each task package stores its textualized visual ground truth (visual GT) together with the image bundle and a task-level GT version identifier. Evaluation logs record both the task ID and the GT version, ensuring that every reported score is tied to an immutable GT snapshot.

If any task image is updated or replaced, we regenerate the corresponding visual GT and bump the task-level version identifier accordingly. To prevent silent regressions across benchmark releases, we retain all prior GT snapshots and support re-scoring using the exact GT file associated with a given release, so historical results remain fully reproducible even as newer GT versions are introduced.

Before finalizing a new release, we additionally run a regression check on a fixed canary set of tasks to detect shifts introduced by GT edits. Any such changes are documented together with the corresponding GT version updates.

Vef. Score Semantics. Vef. is reported as a pass-rate (percentage) over tasks. Let y(t,R) ∈ {0,1} be the strict Vef. pass indicator for report R on task t (1 for PASS, 0 for FAIL), where PASS requires the judge score ≥ τVef. = 6 and no identity-critical violations. We compute:

1 |D| t∈D

Vef.(D) = 100 ·

y(t,R), (5)

so values like 38.57 correspond to a 38.57% strict pass-rate on the evaluated split.

Quality Control and Calibration. We enforce a strict judging instruction for Vef. and require the judge to base its decision on the provided images and the corresponding visual GT only. The judge returns a binary verdict (PASS, FAIL) together with a confidence score on a 0–10 scale. We calibrate the strict prompt and decision rule using expert spot-checking on a small development subset, iteratively refining the instruction until the LLM judgments match expert expectations on common visual failure modes such as incorrect entity identity, swapped labels, or incorrect numeric reading.

#### A.2 Evaluation Metrics

We score each generated report with three modules: FLAE (generation quality), TRACE (evidence and task faithfulness), and MOSAIC (image-grounded integrity). Figure 5 summarizes the evaluation pipeline: FLAE and TRACE are computed in parallel, and MOSAIC is evaluated only when multimodal items are meaningfully scorable.

- A.2.1 Weighting Hyperparameters

Table 4 summarizes the weighting hyperparameters used by the evaluator, including task-adaptive weights and fixed shares.

Rationale. Task-adaptive weights are used to maintain evaluator stability across heterogeneous tasks, allowing different task types and domains to emphasize distinct evaluation dimensions without introducing bias from a single fixed weighting scheme. At the same time, the fixed Vef. share serves as a consistent constraint that strictly enforces task faithfulness, ensuring that multimodal requirements specified by each task are respected regardless of domain or difficulty. This design choice also reflects the substantial annotation and engineering effort involved in maintaining reliable per-task multimodal ground truth, as well as the need for scalable and robust Claim–URL checking across a large and diverse benchmark.

A.3 FLAE Details: Fixed Formulas and Judge Prompts

- A.3.1 Text Features ϕ(R)

We compute lightweight, directly observable text features ϕ(R) from the report R to support a fully reproducible formula channel. In practice, ϕ(R) includes lexical diversity and repetition signals, section and heading

Scope What is weighted Symbol Method Meaning Overall Module aggregation (wF, wT, wM) Fixed Final score weights for FLAE,

TRACE, and MOSAIC.

FLAE Dimension importance Wd(t, R) Task-adaptive Weights over D = {Read., Insh., Stru.} with

d Wd = 1.

FLAE Fusion coefficient α(t, R) Task-adaptive Mixes reproducible text signals with judge scores when forming sd(R), with α ∈ [0, 1].

TRACE Prompt-fidelity share λVef. Fixed With wT = 0.5, setting λVef. = 0.4 yields a 0.2 overall share for Vef..

TRACE Citation-fidelity breakdown Wk(t, R) Task-adaptive Weights over K = {Con., Cov., Fid.} with k Wk = 1. MOSAIC Item weights (by type) (ωSem., ωAcc., ωvqa) Fixed Default type-specific weights in Ta-

ble 6.

- Table 4 Weighting hyperparameters used by the evaluator. Task-adaptive weights vary across tasks and reports; fixed weights encode non-negotiable priorities, notably Vef..

coverage, sentence-length statistics, and basic compliance indicators such as the presence of a references section and in-body citation usage. All features are computed directly from the report text without accessing any judge model.

###### A.3.2 Fixed Formula Channel fd(ϕ(R))

The formula channel maps ϕ(R) to per-dimension scores in [0,1] using fixed transforms. We use σ(x) = 1/(1 + e−x) and clip(x;0,1) = min(1,max(0,x)). Let ϕRead.(R), ϕInsh.(R), and ϕStru.(R) denote the feature subsets used for each dimension. We compute:

sformRead.(R) = clip σ βRead.⊤ ϕRead.(R) ; 0, 1 , sformInsh.(R) = clip σ βInsh.⊤ ϕInsh.(R) ; 0, 1 , sformStru.(R) = clip σ βStru.⊤ ϕStru.(R) ; 0, 1 .

(6)

All coefficients βRead., βInsh., and βStru. are fixed constants shared across all tasks and models to keep the formula channel auditable and stable.

###### A.3.3 LLM Judge Prompt for Dimension Scoring

The judge receives a task and the generated Deep Research report (t,R) and outputs per-dimension scores in [0,1]. We use the prompt template presented in Figure 8 (line breaks are for readability).

###### A.3.4 Task-Adaptive Weighting and Fusion Prompts

FLAE uses a Judge LLM to produce (i) task-adaptive dimension weights Wd(t,R) and (ii) a fusion coefficient α(t,R). Both are generated per task and report.

Task-adaptive Dimension Weights Wd(t, R). The Judge outputs three non-negative weights that sum to 1, reflecting how much the task should emphasize Read./Insh./Stru.. The prompt used for generating task-adaptive dimension weights is presented in Figure 9.

Adaptive Fusion Coefficient α(t, R). α(t,R) ∈ [0,1] controls the mix between formula and judge scores. We constrain α(t,R) to depend only on directly observable signals rather than model identity. The prompt used for generating task-adaptive fusion weights is presented in Figure 10.

|SYSTEM You are a careful evaluator of long-form deep research reports. Score strictly and consistently. Return only valid JSON.<br><br>USER Task:<br><br>{TASK_TEXT}<br><br>Report:<br><br>{REPORT_TEXT}<br><br>Score the report on three dimensions in [0, 1]:<br><br>(1) Read.: clarity, coherence, and ease of reading.<br>(2) Insh.: depth beyond surface summary; synthesis, comparisons, or non-trivial reasoning.<br>(3) Stru.: report completeness and organization (sections, references, and visual integration where applicable).<br><br><br>Output JSON with keys: read, insh, stru.|
|---|

- Figure 8 FLAE Judge prompt for per-dimension scoring.

|SYSTEM You set task-adaptive importance weights. Use only the task description and observable report properties. Return only valid JSON.<br><br>USER Task:<br><br>{TASK_TEXT}<br><br>Report (for observables only, do not score quality here):<br><br>{REPORT_TEXT}<br><br>Produce weights over {Read., Insh., Stru.} such that they sum to 1. Output JSON with keys: w_read, w_insh, w_stru.|
|---|

- Figure 9 FLAE prompt for task-adaptive dimension weights.

- A.3.5 Ablation on the fusion coefficient α FLAE fuses the formula channel and the judge channel at the dimension level. For each d ∈ D, we use

sd(R) = α(t,R)sformd (R) + 1 − α(t,R) sjudged (t,R), (7) where α(t,R) ∈ [0,1] is constrained to depend only on directly observable signals of the report.

To test whether the judge-based α(t,R) is necessary (beyond a transparent heuristic), we replace it with a fully deterministic coefficient αdet(R) computed from ϕ(R) and compliance features, while keeping all other components unchanged (dimension scoring prompts, task-adaptive weights Wd(t,R), and aggregation).

Deterministic Coefficient. We define four normalized observables in [0,1] from the report R: length completeness L(R), heading coverage H(R), citation compliance C(R), and reference-block validity Rref(R), where larger means better formed. We then compute

αdet(R) = clip 1 − 0.35L(R) + 0.35H(R) + 0.20C(R) + 0.10Rref(R) , (8)

with clip(x) = min(1,max(0,x)). Intuitively, αdet(R) increases when the report is short, poorly structured, citation-sparse, or missing a valid reference block, thereby putting more weight on the reproducible formula channel in Eq. (7). The results are summarized in Table 5.

Evaluation Protocol. We rerun the evaluator on the same human-study report pairs using αdet(R) in place of α(t,R), and measure agreement with experts using the same pairwise agreement and system-level score correlation metrics as in Section 5.4. We also include two extreme controls: formula-only fusion (α = 1) and

|SYSTEM<br><br>You set a fusion coefficient α ∈ [0, 1] using only observable signals (length, section presence, formatting compliance). Do not use any model identity or external metadata. Return only valid JSON.<br><br>USER Task:<br><br>{TASK_TEXT}<br><br>Report:<br><br>{REPORT_TEXT}<br><br>Choose α so that when the report is short, poorly structured, or non-compliant, the formula channel gets higher weight (larger α); when the report is complete and well-formed, rely more on judge scoring (smaller α). Output JSON: {"alpha": number}.|
|---|

Figure 10 FLAE prompt for task-adaptive fusion weights.

Method (fusion for Eq. (7)) PAR↑ OPC↑

α(t, R) from judge (default) 73.5 96.4 αdet(R) from Eq. (8) 72.8 96.1 α = 1 (formula-only) 64.5 92.5 α = 0 (judge-only) 71.6 96.0

- Table 5 Ablation on the fusion coefficient α. We replace the judge-based α(t, R) with a deterministic αdet(R) computed from observable report properties, and evaluate agreement with expert judgments using the same protocol as Section 5.4.

judge-only fusion (α = 0). This ablation isolates the contribution of the judge-based α(t,R) while keeping the rest of the pipeline fixed.

###### A.3.6 TRACE: Evidence and Task Faithfulness

TRACE evaluates citation fidelity and multimodal task faithfulness, aggregating as Eq. (3). We parse citation markers and the references: block to build a one-to-one index-to-URL map, extract atomic claims, and align each claim to its cited URL(s) to form claim-URL pairs. For accessible pages, a judge verifies whether each claim is supported and aggregates citation fidelity into K: Con., Cov., and Fid.. We additionally compute Vef. by matching the report to a textualized visual ground truth for the task, then threshold it at τVef..

Vef. Thresholding. The Vef. judge returns a discrete score in {0,...,10} and a PASS/FAIL verdict. We use a fixed threshold τVef. = 6 and force FAIL when the score is below the threshold, so the final Vef. decision is determined by a stable rule rather than judge verbosity. This design makes the Vef. contribution auditable and consistent across judge backbones.

Vef. Judge Prompt Template. Figure 11 shows the prompt used to score Vef. against the task-specific textualized visual ground truth and to output a PASS/FAIL verdict.

###### A.3.7 MOSAIC: Image-Grounded Integrity

MOSAIC checks whether image-grounded statements match the referenced visuals and aggregates by Eq. (4). It extracts figure-linked items and routes them into visual types (photo, datachart, ocrchart, diagram). Each item is scored with a formatting and integration factor fi ∈ [0,1]; for diagrams we additionally use a hallucination factor hi ∈ [0,1].

#### A.4 Evaluated Model Details

In this section, we provide detailed specifications for the 25 systems evaluated in our experiments. To support reproducibility, we report the API snapshot names (or version IDs) used during our testing window (December

|SYSTEM You are a STRICT QA Judge for Vef.. Use the task, the provided visual ground truth, and the report. Return only valid JSON.<br><br>USER Segment:<br><br>{SEGMENT}<br><br>Question:<br><br>{TASK_TEXT}<br><br>Visual ground truth (text-form requirements):<br><br>{VEF_GT}<br><br>Report:<br><br>{REPORT_TEXT}<br><br>Rules: any wrong visual identity is FAIL; any false presence is FAIL; missing details allowed only if no wrong identities; score below 6 must be FAIL. Output JSON with keys: score, reason, verdict (PASS or FAIL).|
|---|

Figure 11 TRACE prompt template for Vef..

Item score si (normalized)

Visual type

datachart / ocrchart si = fi · 0.9 Acc.i + 0.1 VQAi photo si = fi · 0.5 Sem.i + 0.5 VQAi diagram si = fi · 0.5 VQAi + 0.5 (1 − hi)

Table 6 Default MOSAIC item-score weights by routed visual type.

2025), along with the corresponding modality and brief notes on the intended usage mode (see Table 7). We group systems into three tiers to reflect progressively stronger tool access and orchestration capability. Tier 1 covers single-shot LLM/LMM baselines without external browsing, which isolates intrinsic reasoning, writing, and image understanding. Tier 2 includes web-enabled report generators with built-in browsing, representing mainstream tool-using LMM deployments. Tier 3 contains dedicated Deep Research Agents that explicitly orchestrate multi-step retrieval and synthesis, and thus reflect agentic search-and-write behavior beyond a single response.

#### A.5 Reliability and Reason-Aware N/A Handling

- A.5.1 Additional Experimental Setup and Reliability Handling We report reliability-related settings that affect score stability and effective coverage.

TRACE Weighting. With the benchmark-level module weight wT = 0.5, we set λVef. = 0.4 so that Vef.

contributes 0.5 × 0.4 = 0.2 of the overall score. The remaining 0.3 overall share from TRACE is allocated to citation-fidelity metrics (Con., Cov., Fid.) using task-adaptive weights. Vef. Score Semantics. Vef. is reported as pass-rate×100 over tasks, based on a strict PASS/FAIL check against a task-specific textualized visual ground truth. Details of visual GT construction, QC, and versioning are provided in Appendix A.1.2. MOSAIC Gate and N/A Handling. We set MOSAIC activation thresholds to τF = τT = 0, so MOSAIC is triggered whenever FLAE and TRACE produce valid (non-zero) scores. When MOSAIC is not scorable, we record it as N/A and treat it as zero in aggregation, while the reliability impact is handled by the reason-aware validity scheme below.

- A.5.2 Failure Reasons and Reason-Aware N/A Weights

N/A arises when a stage is unscorable due to model output, evaluation pipeline issues, provider instability, or limited evidence access. Each N/A case is assigned a reason r and a validity weight w(r) ∈ [0,1] (higher

Tier System (paper) Modality/Setting API snapshot / version ID Notes

###### Tier 1: Single-shot LLM/LMM baselines

T1 OpenAI o3-mini Text/Reasoning gpt-o3-mini OpenAI snapshota

- T1 DeepSeek-V3.2 (Base) Text deepseek-v3.2 DeepSeek APIb

- T1 Kimi K2 (Thinking) Text/Reasoning kimi-k2-thinking-preview Kimi APIc

- T1 Qwen 3 235B (A22B) Text qwen3-235b-a22b-2507 Qwen APId

- T1 Qwen 3 VL 235B (A22B) Multimodal qwen3-vl-235b-a22b-instruct Qwen-VLd

- T1 GPT-4o Multimodal gpt-4o OpenAI snapshota

- T1 GPT-4.1 Multimodal gpt-4.1 OpenAI snapshota

- T1 GPT-4.1 mini Multimodal gpt-4.1-mini OpenAI snapshota

- T1 GPT-4.1 nano Multimodal gpt-4.1-nano OpenAI snapshota

T1 GPT-5 mini Multimodal gpt-5-mini OpenAI snapshota

- T1 GPT-5.1 Multimodal gpt-5.1 OpenAI snapshota

- T1 GPT-5.2 Multimodal gpt-5.2 OpenAI snapshota

- T1 Grok-3 (As listed) grok-3-1212 xAI APIe
- T1 Grok-4 (Fast Reasoning) Reasoning grok-4-fast-reasoning-beta xAI APIe

Tier 2: Web-enabled report generators

- T2 Claude 4.5 Haiku Web-enabled claude-haiku-4.5-20251022 Anthropicf

T2 Claude 4.5 Sonnet Web-enabled claude-sonnet-4.5-20251022 Anthropicf T2 Claude 4.5 Opus Web-enabled claude-opus-4.5-20251115 Anthropicf T2 Gemini 2.5 Flash Web-enabled gemini-2.5-flash-002 Gemini APIg T2 Gemini 2.5 Pro Web-enabled gemini-2.5-pro-002 Gemini APIg T2 Gemini 3 Flash Web-enabled gemini-3-flash-preview-1215 Gemini APIg T2 Gemini 3 Pro Web-enabled gemini-3-pro-preview-1215 Gemini APIg

Tier 3: Deep Research Agents (DRA)

T3 Gemini Deep Research (Gemini 3 Pro) Agentic DR gemini-deep-research-1220 Google DRh T3 ChatGPT Deep Research (o3-mini) Agentic DR gpt-o3-deep-research OpenAI DRa

- T3 Tongyi Deep Research (30B-A3B) Agentic/IR tongyi-deepresearch-30b-a3b-v2 Tongyi DRi T3 Perplexity Sonar Deep Research Agentic/IR sonar-deep-research-large-1222 Perplexityj

- a OpenAI model snapshots and system documentation: [27–29].
- b DeepSeek documentation: [30].
- c Kimi documentation: [31].
- d Qwen/Qwen-VL documentation: [5, 32].
- e xAI Grok documentation: [33].
- f Anthropic Claude documentation: [34].
- g Gemini API snapshots / model docs: [35].
- h Google Gemini Deep Research: [35, 36].
- i Tongyi Deep Research: [37].
- j Perplexity Sonar Deep Research: [38]. Table 7 Evaluated systems and API snapshots used in the December 2025 testing window.

Reason bucket r w(r) Operational signature (from logs and artifacts) Model failure 0.0 Empty or unusable report without upstream errors; references or citation indices

cannot be resolved; required input-image embeds are missing while the report makes image-dependent assertions.

Pipeline failure 0.5 Model output exists, but scoring fails due to parser exceptions, schema mismatch, missing intermediate artifacts, or module crashes (router, OCR, chart reader). System or provider failure 0.8 Explicit API or infrastructure errors prevent generation or judging (rate limit, overload, timeout, connection reset, auth, misconfiguration). Data accessibility failure 0.9 Evidence assets are unreachable or non-extractable (dead links, blocks, paywalls, non-text pages, images requiring login, expired URLs).

Table 8 Reason-aware N/A buckets and default validity weights.

indicates less model attribution), preventing capability from being confounded with operational noise. Table 8 summarizes the reason categories and default weights.

Assignment Rule. When multiple signals apply, we assign the most specific non-model reason using a fixed priority order so that clear provider or accessibility issues are not misattributed to the model. The fixed

Priority Trigger (first match wins) Bucket

- 1 Any explicit API or infrastructure error in generation or judging logs System or provider failure
- 2 URLs are well-formed but blocked, paywalled, region-restricted, removed, or yield non-extractable content within budget

Data accessibility failure

- 3 Internal exceptions, schema or parsing failures, missing artifacts, or module crashes without upstream API errors

Pipeline failure

- 4 Unusable output or format failures remain after checks above Model failure

Table 9 Priority rules for assigning N/A reasons.

priority order is summarized in Table 9.

Why Reason-Aware Weights. This design preserves a throughput and reliability signal without collapsing model scores due to transient outages or inaccessible assets. It is also diagnostic: stage logs and attribution rules enable reproducible failure breakdowns. Maintaining per-task textualized visual ground truth for Vef. and scalable Claim–URL verification are major annotation and engineering components, and this handling makes their operational impact explicit.

###### A.5.3 Failure Case Analysis

To facilitate qualitative error analysis of multimodal task understanding, we define failures only at the level of Visual Evidence Fidelity (Vef.). A case is marked as FAIL if and only if it fails the Vef. check in TRACE, indicating an incorrect interpretation of the task’s visual requirements or an improper use of the provided images as evidence. Other metrics are not used to define FAIL but to diagnose why Vef. failed.

We group Vef.-level failures into five categories: EMI (entity mis-identification), RMD (reference or mapping drift), DTE (detail or symbol extraction errors), LKC (logical or knowledge conflicts), and STO (structural or task-level omissions). Figure 6 summarizes the distribution of these failure modes, comparing (left) a text-only backbone versus a vision-enabled backbone and (right) a base model versus its agentic deep-research system.

On the left, enabling vision does not monotonically reduce failures: the vision-enabled model shows a clear increase in DTE, consistent with mis-reading fine-grained literals such as small numerals, axis labels, timestamps, or table cells. These errors are often local, but can cascade by seeding an incorrect premise that retrieval and synthesis then treat as evidence-backed.

On the right, the agentic system exhibits a pronounced increase in EMI, indicating that longer pipelines amplify entity-level drift. A common pattern is that entities are identified correctly early on, but become mis-attributed after multiple retrieval, summarization, and consolidation steps, causing the final report to bind correct evidence to the wrong referent.

#### A.6 Examples of Two Scored Reports

Please refer to the next page for two scored example reports shown in Figure 12 and Figure 13 from the Computer Science domain and the Math and Engineering domain. These reports, generated by Grok-4 (Fast Reasoning) and Gemini-2.5-Pro respectively, illustrate representative scoring outcomes under our evaluation framework.

###### Example of a good report (scored 91), from Claude-4.5-opus

###### -4.5-opus

Task (Research): The Evolution of South Asian Deities in Their Eastward Transmission

Figures 00 through 02 present "visual clues of the same deity/motif across different regions and periods." Begin by proposing one to two candidate

hypotheses regarding the identity of the deity based on visual evidence

(these may be parallel assumptions). Subsequently, utilize publicly available sources—such as digital archives, museum entries, academic papers, or monographs—to systematically verify each hypothesis and converge toward the most substantiated conclusion. Finally, with citable

—

—

references, elucidate key shifts in "iconographic elements" and

Figure 00 Figure 01

"functional/role transformations" during its eastward transmission (e.g., the institutionalization process from a certain type of deity to a guardian deity

in East Asia). Requirements: Provide URLs for critical conclusions;

distinguish between primary sources and secondary research, and clearly indicate which judgments derive from visual observation and which rely on external evidence.

Figure 02

###### Excerpt Report from Claude-4.5-opus

###### -4.5-opus

Correctly recognized the task character

Report on the Evolution of Images of Deities Transferred from South Asia: The Cross-Cultural Transformation of Vaiśravaṇa,

-

the King of the South Asian Gods

Clear, well-structured and well-reasoned

- -

executive summary, strictly sticking to the rules

Executive Summary

This report analyzes three cross-regional and cross-temporal image materials to trace the visual evolution process of an important South Asian deity as it spread to East Asia. Based on the observation of image features and cross-validation with literature, this study confirms that this deity is Vaiśravaṇa/Kubera - the wealth god in Hinduism who transformed into one of

- -

-

-

the four major Buddhist guardians of the north.

Correctly and fully understands the visual and

textual task input, giving combined analysis

- I. Presentation of Image Materials and Initial Observations

- Figure 00: A South Asian-style stone statue of a deity, featuring typical Indian iconographic characteristics...
- Figure 01: A Chinese-style image of the Heavenly King with a canopy..
- Figure 02: Statue of Mahadhyaksha (Vishwakarma) in a Japanese temple, wearing armor and holding a pagoda in his hand...

- II. Candidate Hypotheses and Verification Process

- 2.1 Image-based Candidate Judgment Candidate Hypothesis One: Vaiśravaṇa / Mahāvīra (the god of many virtues)

- Basis: All three diagrams feature themes of wealth and guardianship..

Candidate Hypothesis Two: Kubera / Lord of the Demons..

- 2.2 Literature Verification Process

- Verification 1: Verification of the Indian prototype According to the collection records of the British Museum, Kubera (also known as "Jupila") is the god of wealth and the king of the naga (night demons) in Hinduism. In early Indian art, he is often depicted as a short and plump, well-fed figure...

|[1].|
|---|

- Verification 2: The transmission route in Central Asia...

- 2.3 Assumption of Convergence Based on the comprehensive analysis of the images and the evidence from the literature, the candidate hypothesis one (Vishwamitra, the King of the Asuras) received the strongest support:...

- III. Analysis of Changes in Image Elements [1] ...
- IV. Analysis of Function and Role Changes ...
- V. Description of Research Methods 5.1 Material Classification | Material Type | Content | Annotation | | Primary Material | Museum Collection Records | [1][7][8] | | Second-hand research | Academic papers / monographs | [2][6][9] | ...

-

-

Rigorous and well-grounded process of hypothesis formulation and verification

-

-

-

-

Well-structured report content, smooth and

-

scientific in writing, adhereing to executive

...

Authorized reference materials that can be accessed, consistent with the report statement

- ...

Accessible figure citation from authorized source, adhrering to the textual statement

## VI. Conclusion Through image analysis and literature verification, this study confirmed that the deities depicted in the three images are Vaiśravaṇa, the King of the Vidyas. It also revealed...

Authoritative and accessible list of references

References

- [1] https://it.wikipedia.org/wiki/File:British_Museum_Ganesha_Matrikas_Kubera.jpg
- [2] https://www.jstor.org/stable/25261189
- [3] https://www.cbeta.org/...

###### Example of a bad report (scored 85), from Gemini-2.5-Pro Task (Research): From Confusion Matrix to ROC: Calibrating a Binary Classifier Under Class Imbalance

Clear, well-structured and well-reasoned executive summary, strictly sticking to the rules

###### Excerpt Report from Gemini-2.5-Pro

Technical Memo: Classifier Evaluation and Calibration Under Class Imbalance This report addresses the challenges of evaluating binary classifiers when the underlying class distribution is imbalanced.

- 1. Analysis of Visual Evidence The provided images illustrate two fundamental tools for binary classification evaluation: the confusion matrix and the Receiver Operating Characteristic (ROC) curve.

- Figure 1: A standard 2x2 confusion matrix layout [901]
- Figure 2: An illustrative ROC curve plot [902] Based on these images, we can extract the following foundational facts: The confusion matrix tabulates performance by comparing the "Actual Class" against the "Predicted Class". The top-left quadrant, True Positives (TP), represents cases correctly predicted as "Yes"...

- 2. Impact of Class Imbalance on Performance Metrics To demonstrate the effect of class imbalance, we construct two hypothetical confusion matrices... Dataset A: Balanced (50% Positive Rate)**

Actual Positives (P): 1,000, Actual Negatives (N): 1,000

Calculations: TP = TPR × P = 0.80 × 1000 = 800, FN = P - TP = 1000 - 800 = 200...

| Metric | Formula | Dataset A (Balanced) | Dataset B (Imbalanced) | Accuracy| (TP+TN)/(P+N) | (800+900)/2000 = **85.0%** | (80+1710)/2000 = **89.5%** |...

This comparison reveals a critical insight: while metrics like Accuracy, Balanced Accuracy, Recall (TPR), and Specificity

(TNR) remain high or even improve...

- 3. Foundational Concepts: ROC and Precision-Recall Curves Authoritative definitions clarify the roles of different evaluation curves: ROC Curve & AUC: The Receiver Operating Characteristic (ROC) curve is a graph showing the performance of a classification model... Precision-Recall (PR) Curve: A PR curve plots Precision (positive predictive value) against Recall (true positive rate) at various thresholds...
- 4. Why ROC Invariance Hides Precision Degradation A fixed point on an ROC curve corresponds to a fixed TPR and FPR... We can rewrite the precision formula using rates and class counts (P = Positives, N = Negatives):Precision = (TPR * P) / (TPR * P + FPR * N)` This formula shows that precision is highly dependent on the class ratio `P/N`

|[8]...|
|---|

..

Numerical Demonstration (using data from Section 2):...

- 5. Proposed Production Monitoring Protocol To manage a classifier in a production environment where the base rate may drift, a minimal evaluation protocol should include monitoring, alerting, and recalibration.

Correctly recognized the task character

Clear, fluid and accurate mathematical

interpretation, aligning with domain facts

Accurate, professional and scientific mathematical definitions, without any hallucinations

Insightful discussions of the proposed questions

according to the calculated resullts

[Figure 154]

Accessible multimodal citation from authorized artical, adhrering to the

report statement

Based on the calculation results and discussions,

- 5.1 Continuous Monitoring:...
- 5.2 Automated Alerting:...
- 5.3 Recalibration and Thresholding Policy... References

Feasible suggestions are proposed, which are consistent with human professional insights.

Authoritative and accessible list of references

- [1] https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
- [3] https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
- [4] https://en.wikipedia.org/wiki/Receiver_operating_characteristic

...

Example of a bad report (scored 82), from Grok-4 (Fast Reasoning) Task (Research): Edge LLM Deployment: End-to-End Latency Budgeting with

###### Amdahl’s Law and Model Architecture

Executive summary sets the correct direction of implementation the report

Excerpt Report from Grok-4 (Fast Reasoning) An analysis of Large Language Model (LLM) deployment on edge devices, focusing on latency, architectural bottlenecks, and optimization strategies, is presented below. This report integrates insights from the provided architectural diagrams and Amdahl's law to form a comprehensive deployment recommendation for a mobile environment.

[Figure 155]

Correctly recognized the task character

- 1. Image-Anchored Architectural Analysis The provided images illustrate the fundamental building blocks of a GPT-style LLM and the theoretical limits of performance optimization.

- Figure 1: GPT-style LLM architecture and a detailed Transformer Block [901]
- Figure 2: Scaled Dot-product Attention mechanism [902]
- Figure 3: Amdahl's Law chart showing speedup vs. sequential fraction [903] Based on a detailed review of these figures, the following facts can be extracted: From GPT-style Architecture (Figure 1) [901]:

- 1. The overall model pipeline begins with an "Input Embedding" layer to convert input tokens into vectors [901]... From Scaled Dot-Product Attention (Figure 2) [902]:
- 2. The attention mechanism computes its output using three inputs: Query (Qi), Key (Ki), and Value (Vi) vectors [902]...

- 2. Latency Budget for On-Device Inference During autoregressive decoding (generating one token at a time), the primary bottleneck is memory bandwidth.. A plausible latency breakdown for a single token decode step on an edge device is as follows: Tokenization & Embedding (CPU-bound, ~2%):... For this analysis, a plausible budget for a single-token step is: Attention (KV Cache Read & MatMul): 50%, MLP (Linear Layers): 45%, Other (Embedding, Sampling, Norm): 5%
- 3. Application of Amdahl's Law Amdahl's Law provides a formula for calculating the theoretical speedup of a task when only a fraction of it is improved. Formula: `Speedup = 1 / ((1 - P) + (P / S))` [5], Calculation: `Speedup = 1 / (0.6 + (0.4 / 8))`.. Conclusion: Even with an 8x acceleration of the attention kernel, the total theoretical speedup is only about 1.54x...
- 4. Research on LLM Inference Acceleration Techniques Several key techniques have been developed to accelerate LLM inference on both server and edge devices:

- 1. FlashAttention: This is an I/O-aware attention algorithm that optimizes memory access patterns...
- 2. Quantization...

- 5. Decision Memo: Mobile LLM Deployment Strategy To: Edge AI Deployment Team, From: Research Assistant Subject: Recommended Strategy for On-Device LLM Deployment Recommended Deployment Strategy:

Correct extraction of facts with search according to the recognized images

Accurately identified the pain points of the problem and provided feasible solutions

Correct formula selection and mathematical calculation with no computing errors

Feasible advises proposal and systematical synthesis of mobile LLM deployment strategy according to the result of calculation

[Figure 156]

- 1. Quantization Level: INT4 with AWQ/GPTQ[5]. An aggressive 4-bit integer quantization is necessary to meet the strict memory and storage constraints...
- 2. Batching:...
- 3. KV Cache Policy:... Risk Register & Mitigations:... References

Accessible multimodal citation from authorized source, providing insight to the deployment strategy

Aaccessible list of references, with a minor subset of links from unarthourized sources, thus points are deducted

- [1] https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-gpus-part-1/
- [2] https://www.predera.com/blog/llm-inference-optimization-a-comprehensive-guide
- [3] https://deci.ai/blog/llm-inference-the-complete-guide/
- [4] https://gizasystems.com/blog-post/llm-inference-optimization/

...

###### Figure 12 Example scored report generated by Grok-4 (Fast Reasoning).

##### Example of a good report (scored 91), from Claude-4.5-opus

##### -4.5-opus

Task (Research): The Evolution of South Asian Deities in Their Eastward Transmission

Figures 00 through 02 present "visual clues of the same deity/motif across different regions and periods." Begin by proposing one to two candidate

hypotheses regarding the identity of the deity based on visual evidence

(these may be parallel assumptions). Subsequently, utilize publicly available sources—such as digital archives, museum entries, academic papers, or monographs—to systematically verify each hypothesis and converge toward the most substantiated conclusion. Finally, with citable

—

—

references, elucidate key shifts in "iconographic elements" and

Figure 00 Figure 01

"functional/role transformations" during its eastward transmission (e.g., the institutionalization process from a certain type of deity to a guardian deity

in East Asia). Requirements: Provide URLs for critical conclusions;

distinguish between primary sources and secondary research, and clearly indicate which judgments derive from visual observation and which rely on external evidence.

Figure 02

###### Excerpt Report from Claude-4.5-opus

###### -4.5-opus

Correctly recognized the task character

Report on the Evolution of Images of Deities Transferred from South Asia: The Cross-Cultural Transformation of Vaiśravaṇa,

-

the King of the South Asian Gods

Clear, well-structured and well-reasoned

- -

executive summary, strictly sticking to the rules

Executive Summary

This report analyzes three cross-regional and cross-temporal image materials to trace the visual evolution process of an important South Asian deity as it spread to East Asia. Based on the observation of image features and cross-validation with literature, this study confirms that this deity is Vaiśravaṇa/Kubera - the wealth god in Hinduism who transformed into one of

- -

-

-

the four major Buddhist guardians of the north.

Correctly and fully understands the visual and

textual task input, giving combined analysis

I. Presentation of Image Materials and Initial Observations

- Figure 00: A South Asian-style stone statue of a deity, featuring typical Indian iconographic characteristics...
- Figure 01: A Chinese-style image of the Heavenly King with a canopy.. Figure 02: Statue of Mahadhyaksha (Vishwakarma) in a Japanese temple, wearing armor and holding a pagoda in his hand...

-

-

Rigorous and well-grounded process of hypothesis formulation and verification

-

II. Candidate Hypotheses and Verification Process 2.1 Image-based Candidate Judgment Candidate Hypothesis One: Vaiśravaṇa / Mahāvīra (the god of many virtues)

-

-

- Basis: All three diagrams feature themes of wealth and guardianship..

Candidate Hypothesis Two: Kubera / Lord of the Demons.. 2.2 Literature Verification Process

Verification 1: Verification of the Indian prototype

According to the collection records of the British Museum, Kubera (also known as "Jupila") is the god of wealth and the king of the naga (night demons) in Hinduism. In early Indian art, he is often depicted as a short and plump, well-fed figure... Verification 2: The transmission route in Central Asia...

|[1].|
|---|

-

2.3 Assumption of Convergence

Based on the comprehensive analysis of the images and the evidence from the literature, the candidate hypothesis one (Vishwamitra, the King of the Asuras) received the strongest support:...

Well-structured report content, smooth and

-

scientific in writing, adhereing to executive

III. Analysis of Changes in Image Elements [1] ... IV. Analysis of Function and Role Changes ... V. Description of Research Methods 5.1 Material Classification

...

Authorized reference materials that can be accessed, consistent with the report statement

| Material Type | Content | Annotation |

| Primary Material | Museum Collection Records | [1][7][8] | | Second-hand research | Academic papers / monographs | [2][6][9] | ...

- ...

Accessible figure citation from authorized source, adhrering to the textual statement

## VI. Conclusion Through image analysis and literature verification, this study confirmed that the deities depicted in the three images are Vaiśravaṇa, the King of the Vidyas. It also revealed...

Authoritative and accessible list of references

References [1] https://it.wikipedia.org/wiki/File:British_Museum_Ganesha_Matrikas_Kubera.jpg

- [2] https://www.jstor.org/stable/25261189
- [3] https://www.cbeta.org/...

###### Example of a bad report (scored 85), from Gemini-2.5-Pro Task (Research): From Confusion Matrix to ROC: Calibrating a Binary Classifier Under Class Imbalance

Clear, well-structured and well-reasoned executive summary, strictly sticking to the rules

###### Excerpt Report from Gemini-2.5-Pro

Technical Memo: Classifier Evaluation and Calibration Under Class Imbalance This report addresses the challenges of evaluating binary classifiers when the underlying class distribution is imbalanced.

1. Analysis of Visual Evidence The provided images illustrate two fundamental tools for binary classification evaluation: the confusion matrix and the Receiver Operating Characteristic (ROC) curve.

[Figure 169]

- Figure 1: A standard 2x2 confusion matrix layout [901]
- Figure 2: An illustrative ROC curve plot [902] Based on these images, we can extract the following foundational facts:

Correctly recognized the task character

The confusion matrix tabulates performance by comparing the "Actual Class" against the "Predicted Class".

The top-left quadrant, True Positives (TP), represents cases correctly predicted as "Yes"...

Clear, fluid and accurate mathematical

interpretation, aligning with domain facts

2. Impact of Class Imbalance on Performance Metrics To demonstrate the effect of class imbalance, we construct two hypothetical confusion matrices... Dataset A: Balanced (50% Positive Rate)**

Actual Positives (P): 1,000, Actual Negatives (N): 1,000

Calculations: TP = TPR × P = 0.80 × 1000 = 800, FN = P - TP = 1000 - 800 = 200...

| Metric | Formula | Dataset A (Balanced) | Dataset B (Imbalanced) | Accuracy| (TP+TN)/(P+N) | (800+900)/2000 = **85.0%** | (80+1710)/2000 = **89.5%** |...

This comparison reveals a critical insight: while metrics like Accuracy, Balanced Accuracy, Recall (TPR), and Specificity

(TNR) remain high or even improve...

Accurate, professional and scientific mathematical definitions, without any hallucinations

3. Foundational Concepts: ROC and Precision-Recall Curves

Authoritative definitions clarify the roles of different evaluation curves:

ROC Curve & AUC: The Receiver Operating Characteristic (ROC) curve is a graph showing the performance of a classification model...

Precision-Recall (PR) Curve: A PR curve plots Precision (positive predictive value) against Recall (true positive rate) at

various thresholds...

Insightful discussions of the proposed questions

according to the calculated resullts

4. Why ROC Invariance Hides Precision Degradation

A fixed point on an ROC curve corresponds to a fixed TPR and FPR...

[Figure 170]

We can rewrite the precision formula using rates and class counts (P = Positives, N = Negatives):Precision = (TPR * P) / (TPR * P + FPR * N)`

This formula shows that precision is highly dependent on the class ratio `P/N`

..

|[8]...|
|---|

Numerical Demonstration (using data from Section 2):...

Accessible multimodal citation from authorized artical, adhrering to the

5. Proposed Production Monitoring Protocol To manage a classifier in a production environment where the base rate may drift, a minimal evaluation protocol should include monitoring, alerting, and recalibration. 5.1 Continuous Monitoring:...

report statement

Based on the calculation results and discussions,

Feasible suggestions are proposed, which are consistent with human professional insights.

- 5.2 Automated Alerting:...
- 5.3 Recalibration and Thresholding Policy...

Authoritative and accessible list of references

References

[1] https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/ [3] https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc

[4] https://en.wikipedia.org/wiki/Receiver_operating_characteristic

...

###### Figure 13 Example scored report generated by Gemini-2.5-Pro.

Example of a bad report (scored 82), from Grok-4 (Fast Reasoning) Task (Research): Edge LLM Deployment: End-to-End Latency Budgeting with

##### Amdahl’s Law and Model Architecture

Executive summary sets the correct direction of implementation the report

Excerpt Report from Grok-4 (Fast Reasoning) An analysis of Large Language Model (LLM) deployment on edge devices, focusing on latency, architectural bottlenecks, and optimization strategies, is presented below. This report integrates insights from the provided architectural diagrams and Amdahl's law to form a comprehensive deployment recommendation for a mobile environment.

Correctly recognized the task character

1. Image-Anchored Architectural Analysis The provided images illustrate the fundamental building blocks of a GPT-style LLM and the theoretical limits of performance

optimization.

Figure 1: GPT-style LLM architecture and a detailed Transformer Block [901] Figure 2: Scaled Dot-product Attention mechanism [902]

Correct extraction of facts with search according to the recognized images

Figure 3: Amdahl's Law chart showing speedup vs. sequential fraction [903]

Based on a detailed review of these figures, the following facts can be extracted: From GPT-style Architecture (Figure 1) [901]:

1. The overall model pipeline begins with an "Input Embedding" layer to convert input tokens into vectors [901]...

From Scaled Dot-Product Attention (Figure 2) [902]:

2. The attention mechanism computes its output using three inputs: Query (Qi), Key (Ki), and Value (Vi) vectors [902]...

Accurately identified the pain points of the problem and provided feasible solutions

2. Latency Budget for On-Device Inference

During autoregressive decoding (generating one token at a time), the primary bottleneck is memory bandwidth.. A plausible latency breakdown for a single token decode step on an edge device is as follows: Tokenization & Embedding (CPU-bound, ~2%):...

For this analysis, a plausible budget for a single-token step is:

Attention (KV Cache Read & MatMul): 50%, MLP (Linear Layers): 45%, Other (Embedding, Sampling, Norm): 5%

Correct formula selection and mathematical calculation with no computing errors

3. Application of Amdahl's Law

Amdahl's Law provides a formula for calculating the theoretical speedup of a task when only a fraction of it is improved. Formula: `Speedup = 1 / ((1 - P) + (P / S))` [5], Calculation: `Speedup = 1 / (0.6 + (0.4 / 8))`.. Conclusion: Even with an 8x acceleration of the attention kernel, the total theoretical speedup is only about 1.54x...

Feasible advises proposal and systematical synthesis of mobile LLM deployment strategy according to the result of calculation

4. Research on LLM Inference Acceleration Techniques Several key techniques have been developed to accelerate LLM inference on both server and edge devices:

- 1. FlashAttention: This is an I/O-aware attention algorithm that optimizes memory access patterns...
- 2. Quantization...

5. Decision Memo: Mobile LLM Deployment Strategy

To: Edge AI Deployment Team, From: Research Assistant Subject: Recommended Strategy for On-Device LLM Deployment Recommended Deployment Strategy:

1. Quantization Level: INT4 with AWQ/GPTQ[5]. An aggressive 4-bit integer quantization is necessary to meet the strict

Accessible multimodal citation from authorized source, providing insight to the deployment strategy

memory and storage constraints...

- 2. Batching:...
- 3. KV Cache Policy:...

Risk Register & Mitigations:... References

Aaccessible list of references, with a minor subset of links from unarthourized sources, thus points are deducted

- [1] https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-gpus-part-1/
- [2] https://www.predera.com/blog/llm-inference-optimization-a-comprehensive-guide [3] https://deci.ai/blog/llm-inference-the-complete-guide/

[4] https://gizasystems.com/blog-post/llm-inference-optimization/

...

